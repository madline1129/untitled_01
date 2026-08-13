"""训练配置及固定课程定义。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar


@dataclass(frozen=True)
class RewardConfig:
    progress: float = 0.30
    risk: float = 0.20
    ego_collision: float = 10.0
    time: float = 0.01
    offroad: float = 1.0
    road_collision: float = 0.5
    non_ego_collision: float = 2.0
    steering: float = 0.02
    acceleration: float = 0.01
    action_delta: float = 0.02


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.20
    value_clip_coef: float = 0.20
    learning_rate: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    num_minibatches: int = 4
    rollout_steps: int = 128
    target_kl: float = 0.02


@dataclass(frozen=True)
class TrainConfig:
    runtime_split: str = "dataset/nuplan/rl_runtime/runtime_split.json"
    output_dir: str = "outputs/mappo_v0"
    device: str = "cuda:0"
    num_worlds: int = 64
    max_attackers: int = 16
    total_world_steps: int = 5_000_000
    seed: int = 42
    checkpoint_interval: int = 100_000
    log_interval: int = 10_000
    reward: RewardConfig = RewardConfig()
    ppo: PPOConfig = PPOConfig()


T = TypeVar("T")


def _dataclass_from_dict(cls: type[T], values: dict[str, Any]) -> T:
    known = {field.name for field in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**values)


def load_config(path: Path) -> TrainConfig:
    """读取 JSON，并拒绝拼写错误或未知配置项。"""

    with path.open(encoding="utf-8") as stream:
        values = json.load(stream)
    reward = _dataclass_from_dict(RewardConfig, values.pop("reward", {}))
    ppo = _dataclass_from_dict(PPOConfig, values.pop("ppo", {}))
    return _dataclass_from_dict(TrainConfig, {**values, "reward": reward, "ppo": ppo})


def config_dict(config: TrainConfig) -> dict[str, Any]:
    return asdict(config)
