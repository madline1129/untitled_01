"""训练 DangerMaker 多智能体 MAPPO v0。"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .config import TrainConfig, config_dict, load_config
from .data import load_runtime_split
from .env import AdversarialDrivingEnv, ControlMode, load_cuda_simulator
from .model import MAPPOPolicy
from .ppo import RolloutBuffer, update_policy


def curriculum(global_world_steps: int, maximum: int) -> tuple[int, ControlMode, str]:
    if global_world_steps < 500_000:
        return min(4, maximum), ControlMode.RESIDUAL, "residual_4"
    if global_world_steps < 1_500_000:
        return min(8, maximum), ControlMode.RESIDUAL, "residual_8"
    if global_world_steps < 3_000_000:
        return maximum, ControlMode.RESIDUAL, "residual_16"
    return maximum, ControlMode.DIRECT, "direct_16"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _save_checkpoint(
    path: Path,
    policy: MAPPOPolicy,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    global_world_steps: int,
    update: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config_dict(config),
            "global_world_steps": global_world_steps,
            "update": update,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        },
        path,
    )


def _load_checkpoint(
    path: Path,
    policy: MAPPOPolicy,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, int]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    policy.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    random.setstate(checkpoint["python_rng_state"])
    return int(checkpoint["global_world_steps"]), int(checkpoint["update"])


def train(config: TrainConfig, resume: Path | None = None) -> Path:
    device = torch.device(config.device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("CUDA training requires an explicit device such as cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA on this workstation")
    _seed_everything(config.seed)
    torch.set_float32_matmul_precision("high")

    runtime_paths = load_runtime_split(Path(config.runtime_split), "train")
    simulator = load_cuda_simulator(runtime_paths, config.num_worlds, device.index)
    env = AdversarialDrivingEnv(simulator, config.max_attackers, config.reward)
    policy = MAPPOPolicy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.ppo.learning_rate, eps=1e-5)
    global_world_steps = 0
    update = 0
    if resume is not None:
        global_world_steps, update = _load_checkpoint(resume, policy, optimizer, device)

    output = Path(config.output_dir)
    checkpoint_dir = output / "checkpoints"
    metrics_path = output / "metrics.jsonl"
    if resume is None and metrics_path.exists():
        raise FileExistsError(
            f"{metrics_path} already exists; use --resume or choose a clean output_dir"
        )
    output.mkdir(parents=True, exist_ok=True)
    with (output / "resolved_config.json").open("w", encoding="utf-8") as stream:
        json.dump(config_dict(config), stream, indent=2, sort_keys=True)
        stream.write("\n")
    writer = SummaryWriter(output / "tensorboard")
    writer.add_text(
        "run/resolved_config",
        "```json\n" + json.dumps(config_dict(config), indent=2, sort_keys=True) + "\n```",
        global_world_steps,
    )
    writer.add_text("run/device", torch.cuda.get_device_name(device), global_world_steps)
    observation = env.reset()
    next_checkpoint = (
        global_world_steps // config.checkpoint_interval + 1
    ) * config.checkpoint_interval
    next_log = (global_world_steps // config.log_interval + 1) * config.log_interval
    previous_stage: str | None = None
    start_time = time.perf_counter()
    run_world_steps = 0
    running_episode_return = torch.zeros(config.num_worlds, device=device)
    running_episode_length = torch.zeros(
        config.num_worlds, dtype=torch.int32, device=device
    )
    torch.cuda.reset_peak_memory_stats(device)

    try:
        while global_world_steps < config.total_world_steps:
            active_count, mode, stage = curriculum(global_world_steps, config.max_attackers)
            if stage != previous_stage:
                env.set_curriculum(active_count, mode)
                observation = env.reset(
                    torch.zeros(config.num_worlds, dtype=torch.uint8, device=device)
                )
                previous_stage = stage

            progress = global_world_steps / max(1, config.total_world_steps)
            learning_rate = config.ppo.learning_rate * (1.0 - progress)
            for group in optimizer.param_groups:
                group["lr"] = max(learning_rate, 1e-6)

            buffer = RolloutBuffer(config.ppo.rollout_steps, observation)
            rollout_reward = torch.zeros((), device=device)
            rollout_progress = torch.zeros((), device=device)
            rollout_risk = torch.zeros((), device=device)
            rollout_penalty = torch.zeros((), device=device)
            rollout_offroad = torch.zeros((), device=device)
            rollout_road_collisions = torch.zeros((), device=device)
            rollout_non_ego = torch.zeros((), device=device)
            rollout_min_distance = torch.zeros((), device=device)
            rollout_min_ttc = torch.zeros((), device=device)
            rollout_action_acceleration = torch.zeros((), device=device)
            rollout_action_steering = torch.zeros((), device=device)
            completed_episode_return = torch.zeros((), device=device)
            completed_episode_length = torch.zeros((), device=device)
            completed_episodes = torch.zeros((), device=device)
            successful_episodes = torch.zeros((), device=device)
            rollout_started = time.perf_counter()

            policy.eval()
            for _ in range(config.ppo.rollout_steps):
                with torch.no_grad():
                    action, log_prob, _, value = policy.act(observation)
                    next_observation, reward, terminated, truncated, info = env.step(action)
                    next_value = policy.value(next_observation)
                buffer.store(
                    observation,
                    action,
                    log_prob,
                    value,
                    reward,
                    terminated,
                    truncated,
                    next_value,
                )
                rollout_reward += reward.mean()
                rollout_progress += info["progress"].mean()
                rollout_risk += info["risk"].mean()
                rollout_penalty += info["penalty"].mean()
                rollout_offroad += info["offroad_rate"].mean()
                rollout_road_collisions += info["road_collision_rate"].mean()
                rollout_non_ego += info["non_ego_collision_rate"].mean()
                rollout_min_distance += info["min_distance"].mean()
                rollout_min_ttc += info["min_ttc"].clamp(max=30.0).mean()
                active = observation["attacker_mask"].to(action.dtype)
                active_denominator = active.sum().clamp_min(1.0)
                rollout_action_acceleration += (
                    action[..., 0].abs() * active
                ).sum() / active_denominator
                rollout_action_steering += (
                    action[..., 1].abs() * active
                ).sum() / active_denominator

                running_episode_return += reward
                running_episode_length += 1
                done_bool = terminated | truncated
                done_float = done_bool.to(torch.float32)
                completed_episode_return += (running_episode_return * done_float).sum()
                completed_episode_length += (
                    running_episode_length.to(torch.float32) * done_float
                ).sum()
                completed_episodes += done_float.sum()
                successful_episodes += terminated.to(torch.float32).sum()
                running_episode_return = torch.where(
                    done_bool, torch.zeros_like(running_episode_return), running_episode_return
                )
                running_episode_length = torch.where(
                    done_bool, torch.zeros_like(running_episode_length), running_episode_length
                )

                # 始终使用设备侧 mask reset，避免每步 any().item() 触发 GPU 同步。
                done = done_bool.to(torch.uint8)
                observation = env.reset(done)
                global_world_steps += config.num_worlds
                run_world_steps += config.num_worlds

            policy.train()
            stats = update_policy(policy, optimizer, buffer, config.ppo)
            update += 1
            elapsed = max(1e-6, time.perf_counter() - start_time)
            rollout_elapsed = max(1e-6, time.perf_counter() - rollout_started)
            rollout_steps = config.ppo.rollout_steps
            episode_count = completed_episodes.clamp_min(1.0)
            values: dict[str, Any] = {
                "global_world_steps": global_world_steps,
                "update": update,
                "stage": stage,
                "curriculum/active_attackers": active_count,
                "curriculum/control_mode": int(mode),
                "train/learning_rate": learning_rate,
                "train/step_reward_mean": float((rollout_reward / rollout_steps).item()),
                "train/episode_return_mean": float(
                    (completed_episode_return / episode_count).item()
                ),
                "train/episode_length_mean": float(
                    (completed_episode_length / episode_count).item()
                ),
                "train/episode_success_rate": float(
                    (successful_episodes / episode_count).item()
                ),
                "train/episode_timeout_rate": float(
                    ((completed_episodes - successful_episodes) / episode_count).item()
                ),
                "train/episodes_completed": float(completed_episodes.item()),
                "reward/progress_mean": float((rollout_progress / rollout_steps).item()),
                "reward/risk_mean": float((rollout_risk / rollout_steps).item()),
                "reward/penalty_mean": float((rollout_penalty / rollout_steps).item()),
                "safety/min_distance_mean_m": float(
                    (rollout_min_distance / rollout_steps).item()
                ),
                "safety/min_ttc_mean_s_clipped_30": float(
                    (rollout_min_ttc / rollout_steps).item()
                ),
                "safety/offroad_rate": float((rollout_offroad / rollout_steps).item()),
                "safety/road_collision_rate": float(
                    (rollout_road_collisions / rollout_steps).item()
                ),
                "safety/non_ego_collision_rate": float(
                    (rollout_non_ego / rollout_steps).item()
                ),
                "action/normalized_acceleration_abs_mean": float(
                    (rollout_action_acceleration / rollout_steps).item()
                ),
                "action/normalized_steering_abs_mean": float(
                    (rollout_action_steering / rollout_steps).item()
                ),
                "policy/log_std_acceleration": float(policy.log_std[0].detach().item()),
                "policy/log_std_steering": float(policy.log_std[1].detach().item()),
                "system/world_steps_per_second": run_world_steps / elapsed,
                "system/last_update_seconds": rollout_elapsed,
                "system/cuda_peak_allocated_gib": torch.cuda.max_memory_allocated(device)
                / (1024**3),
                "system/cuda_peak_reserved_gib": torch.cuda.max_memory_reserved(device)
                / (1024**3),
            }
            values.update({f"ppo/{key}": value for key, value in asdict(stats).items()})
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(values, sort_keys=True) + "\n")
            for key, value in values.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(key, value, global_world_steps)
            writer.flush()
            if global_world_steps >= next_log:
                print(json.dumps(values, sort_keys=True), flush=True)
                next_log += config.log_interval

            if global_world_steps >= next_checkpoint:
                _save_checkpoint(
                    checkpoint_dir / f"step_{global_world_steps:09d}.pt",
                    policy,
                    optimizer,
                    config,
                    global_world_steps,
                    update,
                )
                next_checkpoint += config.checkpoint_interval

        final_path = checkpoint_dir / "final.pt"
        _save_checkpoint(final_path, policy, optimizer, config, global_world_steps, update)
        return final_path
    finally:
        writer.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    checkpoint = train(load_config(args.config), args.resume)
    print(f"final checkpoint: {checkpoint.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
