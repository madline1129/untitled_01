"""确定性评估 MAPPO checkpoint，并导出兼容现有 renderer 的轨迹。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import torch

from .config import load_config
from .data import load_runtime_split
from .env import AdversarialDrivingEnv, load_cuda_simulator
from .model import MAPPOPolicy
from .train import curriculum


TRACE_HEADER = [
    "world",
    "step",
    "agent_slot",
    "agent_id",
    "valid",
    "control_mode",
    "x",
    "y",
    "yaw",
    "vx",
    "vy",
    "acceleration",
    "steering",
    "actual_acceleration",
    "actual_steering",
    "longitudinal_velocity",
    "lateral_velocity",
    "yaw_rate",
    "collided_vehicle",
    "collided_ego",
    "collided_road",
    "offroad",
    "reached_goal",
    "world_done",
]


def _control_mode_name(value: int) -> str:
    return {0: "auto", 1: "residual", 2: "direct"}.get(value, "auto")


def _write_world_frame(
    writer: csv.writer,
    frame: dict[str, torch.Tensor],
    agent_ids: list[str],
) -> None:
    # 评估导出允许显式复制到 CPU；训练热路径不会调用这里。
    values = {
        key: frame[key][0].detach().cpu()
        for key in (
            "states",
            "dynamics",
            "applied_actions",
            "events",
            "valid",
            "control_modes",
        )
    }
    step = int(frame["world_step"][0].item())
    done = int(frame["world_done"][0].item())
    for slot in range(values["states"].shape[0]):
        state = values["states"][slot]
        dynamics = values["dynamics"][slot]
        action = values["applied_actions"][slot]
        event = values["events"][slot]
        writer.writerow(
            [
                0,
                step,
                slot,
                agent_ids[slot] if slot < len(agent_ids) else f"padding_{slot}",
                int(values["valid"][slot]),
                _control_mode_name(int(values["control_modes"][slot])),
                *[float(item) for item in state],
                float(action[0]),
                float(action[1]),
                float(dynamics[4]),
                float(dynamics[3]),
                float(dynamics[0]),
                float(dynamics[1]),
                float(dynamics[2]),
                int(event[0]),
                int(event[1]),
                int(event[2]),
                int(event[3]),
                int(event[4]),
                done,
            ]
        )


def evaluate(
    config_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    render: bool,
) -> Path:
    config = load_config(config_path)
    device = torch.device(config.device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("evaluation requires an explicit CUDA device")
    runtime_paths = sorted(load_runtime_split(Path(config.runtime_split), "eval"))
    worlds = min(16, len(runtime_paths))
    simulator = load_cuda_simulator(runtime_paths, worlds, device.index)
    env = AdversarialDrivingEnv(simulator, config.max_attackers, config.reward)
    policy = MAPPOPolicy().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    policy.load_state_dict(checkpoint["model"])
    policy.eval()
    active_count, mode, stage = curriculum(
        int(checkpoint.get("global_world_steps", config.total_world_steps)),
        config.max_attackers,
    )
    env.set_curriculum(active_count, mode)
    observation = env.reset()

    first_runtime = Path(runtime_paths[0])
    with (first_runtime / "manifest.json").open(encoding="utf-8") as stream:
        runtime_manifest = json.load(stream)
    agent_ids = list(runtime_manifest.get("agent_ids", []))
    episode_steps = int(round(float(env.raw_frame["self"][:, 0, 10].max().item())))
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "eval_trace.csv"
    collision_seen = torch.zeros(worlds, dtype=torch.bool, device=device)
    collision_time = torch.full((worlds,), -1, dtype=torch.int32, device=device)
    minimum_distance = torch.full((worlds,), torch.inf, device=device)
    minimum_ttc = torch.full((worlds,), torch.inf, device=device)
    offroad_total = torch.zeros(worlds, device=device)
    non_ego_total = torch.zeros(worlds, device=device)

    with trace_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(TRACE_HEADER)
        _write_world_frame(writer, env.raw_frame, agent_ids)
        for step in range(episode_steps):
            with torch.no_grad():
                actions, _, _, _ = policy.act(observation, deterministic=True)
                observation, _, terminated, _, info = env.step(actions)
            new_collision = terminated & ~collision_seen
            collision_time = torch.where(
                new_collision, torch.full_like(collision_time, step + 1), collision_time
            )
            collision_seen |= terminated
            minimum_distance = torch.minimum(minimum_distance, info["min_distance"])
            minimum_ttc = torch.minimum(minimum_ttc, info["min_ttc"])
            offroad_total += info["offroad_rate"]
            non_ego_total += info["non_ego_collision_rate"]
            _write_world_frame(writer, env.raw_frame, agent_ids)

    finite_ttc = minimum_ttc[torch.isfinite(minimum_ttc)]
    successful_times = collision_time[collision_time >= 0].to(torch.float32)
    summary = {
        "checkpoint": str(checkpoint_path.resolve()),
        "runtime_split": str(Path(config.runtime_split).resolve()),
        "worlds": worlds,
        "stage": stage,
        "active_attackers": active_count,
        "ego_collision_rate": float(collision_seen.to(torch.float32).mean().item()),
        "minimum_distance_mean": float(minimum_distance.mean().item()),
        "minimum_ttc_mean": float(finite_ttc.mean().item()) if finite_ttc.numel() else None,
        "offroad_rate": float((offroad_total / max(1, episode_steps)).mean().item()),
        "non_ego_collision_rate": float(
            (non_ego_total / max(1, episode_steps)).mean().item()
        ),
        "mean_collision_time_seconds": (
            float(successful_times.mean().item()) * 0.1 if successful_times.numel() else None
        ),
        "trace": str(trace_path.resolve()),
    }
    summary_path = output_dir / "eval_summary.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")

    if render:
        repository_root = Path(__file__).resolve().parents[2]
        command = [
            sys.executable,
            str(repository_root / "gpudrive_cuda" / "tools" / "render_trace.py"),
            "--runtime",
            str(first_runtime),
            "--trace",
            str(trace_path),
            "--output",
            str(output_dir / "rollout.gif"),
            "--mp4-output",
            str(output_dir / "rollout.mp4"),
            "--final-png",
            str(output_dir / "final_frame.png"),
            "--duration",
            "10",
            "--show-trails",
        ]
        subprocess.run(command, check=True)
    return summary_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    summary = evaluate(args.config, args.checkpoint, args.output, args.render)
    print(f"evaluation summary: {summary.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
