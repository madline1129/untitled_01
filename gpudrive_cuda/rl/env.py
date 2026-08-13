"""将 CUDA simulator 包装成多智能体对抗训练环境。"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import Any

import torch

from .config import RewardConfig


class ControlMode(IntEnum):
    AUTO = 0
    RESIDUAL = 1
    DIRECT = 2


def load_cuda_simulator(runtime_paths: list[str], num_worlds: int, device_id: int = 0) -> Any:
    """延迟导入扩展，使纯 Python 测试不要求本机安装 CUDA。"""

    try:
        from gpudrive_cuda_torch import TorchDriveSim
    except ImportError as error:
        raise RuntimeError(
            "cannot import gpudrive_cuda_torch; build with "
            "-DBUILD_TORCH_BINDINGS=ON and add build/gpudrive_cuda/python to PYTHONPATH"
        ) from error
    return TorchDriveSim(runtime_paths, num_worlds, device_id)


class AdversarialDrivingEnv:
    """最多选择 16 辆背景车，对固定轨迹 ego 进行协同攻击。"""

    def __init__(
        self,
        simulator: Any,
        max_attackers: int = 16,
        reward_config: RewardConfig | None = None,
    ) -> None:
        if max_attackers <= 0:
            raise ValueError("max_attackers must be positive")
        self.simulator = simulator
        self.num_worlds = int(simulator.num_worlds)
        self.max_agents = int(simulator.max_agents)
        self.max_attackers = max_attackers
        self.reward_config = reward_config or RewardConfig()

        frame = simulator.reset()
        self.device = frame["states"].device
        self._ego_slots, self._attacker_slots, self._candidate_mask = self._select_agents(frame)
        self._episode_steps = self._gather_agents(frame["self"], self._ego_slots[:, None])[
            :, 0, 10
        ].clamp_min(1.0)
        self._active_limit = max_attackers
        self._control_mode = ControlMode.RESIDUAL
        self._active_mask = self._candidate_mask.clone()
        self._previous_actions = torch.zeros(
            (self.num_worlds, self.max_attackers, 2),
            dtype=torch.float32,
            device=self.device,
        )
        self._apply_control_modes()
        self._previous_min_distance = self._minimum_distance(frame)
        self._last_frame = frame

    @property
    def attacker_slots(self) -> torch.Tensor:
        return self._attacker_slots

    @property
    def attacker_mask(self) -> torch.Tensor:
        return self._active_mask

    @property
    def raw_frame(self) -> dict[str, torch.Tensor]:
        return self._last_frame

    def set_curriculum(self, max_active_attackers: int, mode: ControlMode) -> None:
        if max_active_attackers <= 0 or max_active_attackers > self.max_attackers:
            raise ValueError("max_active_attackers is outside the configured range")
        self._active_limit = max_active_attackers
        self._control_mode = ControlMode(mode)
        rank = torch.arange(self.max_attackers, device=self.device)[None, :]
        self._active_mask = self._candidate_mask & (rank < max_active_attackers)
        self._apply_control_modes()

    def reset(self, reset_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if reset_mask is None:
            frame = self.simulator.reset()
            reset_bool = torch.ones(self.num_worlds, dtype=torch.bool, device=self.device)
        else:
            if reset_mask.shape != (self.num_worlds,):
                raise ValueError("reset_mask must have shape [num_worlds]")
            reset_byte = reset_mask.to(device=self.device, dtype=torch.uint8).contiguous()
            frame = self.simulator.reset_worlds(reset_byte)
            reset_bool = reset_byte.bool()

        # simulator reset 会把控制模式恢复为 AUTO，因此必须重新应用课程状态。
        self._apply_control_modes()
        current_distance = self._minimum_distance(frame)
        self._previous_min_distance = torch.where(
            reset_bool, current_distance, self._previous_min_distance
        )
        self._previous_actions = torch.where(
            reset_bool[:, None, None],
            torch.zeros_like(self._previous_actions),
            self._previous_actions,
        )
        self._last_frame = frame
        return self._build_observation(frame)

    def step(
        self, normalized_actions: torch.Tensor
    ) -> tuple[
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        expected = (self.num_worlds, self.max_attackers, 2)
        if normalized_actions.shape != expected:
            raise ValueError(f"actions must have shape {expected}")
        if normalized_actions.device != self.device:
            raise ValueError("actions must be on the simulator CUDA device")

        actions = normalized_actions.to(dtype=torch.float32).clamp(-1.0, 1.0)
        actions = actions * self._active_mask[..., None]
        scaled = torch.empty_like(actions)
        if self._control_mode == ControlMode.RESIDUAL:
            scaled[..., 0] = 2.0 * actions[..., 0]
            scaled[..., 1] = 0.15 * actions[..., 1]
        else:
            scaled[..., 0] = 5.0 * actions[..., 0] - 1.0
            scaled[..., 1] = 0.60 * actions[..., 1]

        full_actions = torch.zeros(
            (self.num_worlds, self.max_agents, 2),
            dtype=torch.float32,
            device=self.device,
        )
        world_indices = torch.arange(self.num_worlds, device=self.device)[:, None].expand_as(
            self._attacker_slots
        )
        active = self._active_mask
        full_actions[world_indices[active], self._attacker_slots[active]] = scaled[active]

        frame = self.simulator.step(full_actions.contiguous())
        reward, terminated, truncated, info = self._compute_reward(frame, actions)
        observation = self._build_observation(frame)
        self._last_frame = frame
        return observation, reward, terminated, truncated, info

    def _select_agents(
        self, frame: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ego_mask = frame["agent_is_ego"].bool() & frame["valid"].bool()
        if not bool(torch.all(ego_mask.sum(dim=1) == 1).item()):
            raise ValueError("each world must contain exactly one valid ego")
        ego_slots = ego_mask.to(torch.int64).argmax(dim=1)
        ego_states = self._gather_agents(frame["states"], ego_slots[:, None])[:, 0]
        delta = frame["states"][..., :2] - ego_states[:, None, :2]
        distance = torch.linalg.vector_norm(delta, dim=-1)
        candidate = (
            frame["valid"].bool()
            & frame["agent_controllable"].bool()
            & (frame["agent_type"] == 1)
            & ~ego_mask
        )
        distance = distance.masked_fill(~candidate, torch.inf)

        selected_count = min(self.max_attackers, self.max_agents)
        _, slots = torch.topk(distance, k=selected_count, dim=1, largest=False, sorted=True)
        selected_mask = torch.gather(candidate, 1, slots)
        if selected_count < self.max_attackers:
            padding = self.max_attackers - selected_count
            slots = torch.cat(
                [slots, torch.zeros((self.num_worlds, padding), dtype=torch.long, device=self.device)],
                dim=1,
            )
            selected_mask = torch.cat(
                [
                    selected_mask,
                    torch.zeros((self.num_worlds, padding), dtype=torch.bool, device=self.device),
                ],
                dim=1,
            )
        return ego_slots, slots, selected_mask

    def _apply_control_modes(self) -> None:
        modes = torch.zeros(
            (self.num_worlds, self.max_agents), dtype=torch.uint8, device=self.device
        )
        worlds = torch.arange(self.num_worlds, device=self.device)[:, None].expand_as(
            self._attacker_slots
        )
        active = self._active_mask
        modes[worlds[active], self._attacker_slots[active]] = int(self._control_mode)
        self.simulator.set_control_modes(modes.contiguous())

    def _gather_agents(self, values: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        worlds = torch.arange(self.num_worlds, device=self.device)[:, None].expand_as(slots)
        return values[worlds, slots]

    def _relative_metrics(
        self, frame: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attacker = self._gather_agents(frame["states"], self._attacker_slots)
        ego = self._gather_agents(frame["states"], self._ego_slots[:, None])[:, 0]
        relative_position = ego[:, None, :2] - attacker[..., :2]
        relative_velocity = ego[:, None, 3:5] - attacker[..., 3:5]
        distance = torch.linalg.vector_norm(relative_position, dim=-1).clamp_min(1e-4)
        distance_rate = (relative_position * relative_velocity).sum(dim=-1) / distance
        closing_speed = (-distance_rate).clamp_min(0.0)
        ttc = torch.where(
            closing_speed > 0.1,
            distance / closing_speed.clamp_min(0.1),
            torch.full_like(distance, torch.inf),
        )
        return distance, ttc

    def _minimum_distance(self, frame: dict[str, torch.Tensor]) -> torch.Tensor:
        distance, _ = self._relative_metrics(frame)
        masked = distance.masked_fill(~self._active_mask, torch.inf)
        minimum = masked.min(dim=1).values
        return torch.where(self._active_mask.any(dim=1), minimum, torch.zeros_like(minimum))

    def _build_observation(self, frame: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        slots = self._attacker_slots
        raw_self = self._gather_agents(frame["self"], slots)
        dynamics = self._gather_agents(frame["dynamics"], slots)
        episode_fraction = raw_self[..., 10] / self._episode_steps[:, None]
        self_features = torch.stack(
            [
                raw_self[..., 5] / 40.0,
                dynamics[..., 0] / 40.0,
                dynamics[..., 1] / 20.0,
                dynamics[..., 2] / 2.5,
                dynamics[..., 3] / 0.6,
                dynamics[..., 4] / 6.0,
                raw_self[..., 6] / 10.0,
                raw_self[..., 7] / 3.0,
                episode_fraction,
            ],
            dim=-1,
        )

        attacker = self._gather_agents(frame["states"], slots)
        ego = self._gather_agents(frame["states"], self._ego_slots[:, None])[:, 0]
        ego_self = self._gather_agents(frame["self"], self._ego_slots[:, None])[:, 0]
        dx = ego[:, None, 0] - attacker[..., 0]
        dy = ego[:, None, 1] - attacker[..., 1]
        cosine = torch.cos(attacker[..., 2])
        sine = torch.sin(attacker[..., 2])
        relative_vx_world = ego[:, None, 3] - attacker[..., 3]
        relative_vy_world = ego[:, None, 4] - attacker[..., 4]
        relative_yaw = torch.atan2(
            torch.sin(ego[:, None, 2] - attacker[..., 2]),
            torch.cos(ego[:, None, 2] - attacker[..., 2]),
        )
        local_x = cosine * dx + sine * dy
        local_y = -sine * dx + cosine * dy
        local_vx = cosine * relative_vx_world + sine * relative_vy_world
        local_vy = -sine * relative_vx_world + cosine * relative_vy_world
        distance = torch.sqrt(dx.square() + dy.square())
        ego_features = torch.stack(
            [
                local_x / 50.0,
                local_y / 50.0,
                local_vx / 20.0,
                local_vy / 20.0,
                relative_yaw / math.pi,
                ego_self[:, None, 6].expand_as(local_x) / 10.0,
                ego_self[:, None, 7].expand_as(local_x) / 3.0,
                distance / 50.0,
                torch.ones_like(distance),
            ],
            dim=-1,
        )

        partners = self._gather_agents(frame["partners"], slots).clone()
        partners[..., 0:2] /= 50.0
        partners[..., 2:4] /= 20.0
        partners[..., 4] /= math.pi
        partners[..., 5] /= 10.0
        partners[..., 6] /= 3.0
        partners[..., 8] /= 50.0
        partner_mask = self._gather_agents(frame["partner_valid"], slots).bool()
        partner_mask &= self._active_mask[..., None]

        map_features = self._gather_agents(frame["map"], slots).clone()
        map_features[..., 0:4] /= 50.0
        map_features[..., 5] /= 40.0
        map_features[..., 6] /= 50.0
        map_mask = self._gather_agents(frame["map_valid"], slots).bool()
        map_mask &= self._active_mask[..., None]

        active_float = self._active_mask[..., None].to(torch.float32)
        return {
            "self": self_features * active_float,
            "ego": ego_features * active_float,
            "partners": partners * partner_mask[..., None],
            "partner_mask": partner_mask,
            "map": map_features * map_mask[..., None],
            "map_mask": map_mask,
            "attacker_mask": self._active_mask.clone(),
        }

    def _compute_reward(
        self,
        frame: dict[str, torch.Tensor],
        normalized_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        config = self.reward_config
        distance, ttc = self._relative_metrics(frame)
        active = self._active_mask
        active_count = active.sum(dim=1).clamp_min(1).to(torch.float32)

        masked_distance = distance.masked_fill(~active, torch.inf)
        min_distance = masked_distance.min(dim=1).values
        has_attacker = active.any(dim=1)
        min_distance = torch.where(has_attacker, min_distance, torch.zeros_like(min_distance))
        progress = (self._previous_min_distance - min_distance).clamp(-1.0, 1.0)
        progress = torch.where(has_attacker, progress, torch.zeros_like(progress))
        risk_per_agent = torch.exp(-ttc.clamp(max=30.0) / 3.0) * torch.exp(-distance / 20.0)
        max_risk = risk_per_agent.masked_fill(~active, 0.0).max(dim=1).values

        events = self._gather_agents(frame["events"], self._attacker_slots)
        collided_vehicle = events[..., 0].bool()
        collided_ego = events[..., 1].bool()
        collided_road = events[..., 2].bool()
        offroad = events[..., 3].bool()
        ego_collision = (collided_ego & active).any(dim=1)
        non_ego_collision = collided_vehicle & ~collided_ego

        dynamics = self._gather_agents(frame["dynamics"], self._attacker_slots)
        action_delta = normalized_actions - self._previous_actions
        offroad_rate = (offroad & active).sum(dim=1).to(torch.float32) / active_count
        road_collision_rate = (collided_road & active).sum(dim=1).to(torch.float32) / active_count
        non_ego_collision_rate = (
            (non_ego_collision & active).sum(dim=1).to(torch.float32) / active_count
        )
        steering_cost = (
            (dynamics[..., 3] / 0.6).square() * active
        ).sum(dim=1) / active_count
        acceleration_cost = (
            (dynamics[..., 4] / 6.0).square() * active
        ).sum(dim=1) / active_count
        action_delta_cost = (
            action_delta.square().sum(dim=-1) * active
        ).sum(dim=1) / active_count
        penalty = (
            config.offroad * offroad_rate
            + config.road_collision * road_collision_rate
            + config.non_ego_collision * non_ego_collision_rate
            + config.steering * steering_cost
            + config.acceleration * acceleration_cost
            + config.action_delta * action_delta_cost
        )
        reward = (
            config.progress * progress
            + config.risk * max_risk
            + config.ego_collision * ego_collision.to(torch.float32)
            - config.time
            - penalty
        )

        terminated = ego_collision
        truncated = frame["world_done"].bool() & ~terminated
        self._previous_min_distance = min_distance.detach()
        self._previous_actions = normalized_actions.detach().clone()
        info = {
            "ego_collision": ego_collision,
            "min_distance": min_distance,
            "min_ttc": ttc.masked_fill(~active, torch.inf).min(dim=1).values,
            "progress": progress,
            "risk": max_risk,
            "penalty": penalty,
            "offroad_rate": offroad_rate,
            "road_collision_rate": road_collision_rate,
            "non_ego_collision_rate": non_ego_collision_rate,
            "steering_cost": steering_cost,
            "acceleration_cost": acceleration_cost,
            "action_delta_cost": action_delta_cost,
            "active_attackers": active.sum(dim=1),
        }
        return reward, terminated, truncated, info
