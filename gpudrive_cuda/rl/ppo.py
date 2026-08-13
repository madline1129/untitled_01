"""MAPPO rollout buffer、GAE 与 clipped PPO 更新。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import PPOConfig
from .model import MAPPOPolicy


@dataclass
class UpdateStats:
    loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    explained_variance: float
    gradient_norm: float
    optimizer_steps: int


class RolloutBuffer:
    def __init__(self, rollout_steps: int, observation: dict[str, torch.Tensor]) -> None:
        if rollout_steps <= 0:
            raise ValueError("rollout_steps must be positive")
        self.rollout_steps = rollout_steps
        self.observations = {
            key: torch.empty(
                (rollout_steps, *value.shape), dtype=value.dtype, device=value.device
            )
            for key, value in observation.items()
        }
        worlds, attackers = observation["attacker_mask"].shape
        device = observation["attacker_mask"].device
        self.actions = torch.empty(
            (rollout_steps, worlds, attackers, 2), dtype=torch.float32, device=device
        )
        self.log_probs = torch.empty(
            (rollout_steps, worlds, attackers), dtype=torch.float32, device=device
        )
        self.values = torch.empty((rollout_steps, worlds), dtype=torch.float32, device=device)
        self.next_values = torch.empty_like(self.values)
        self.rewards = torch.empty_like(self.values)
        self.terminated = torch.empty(
            (rollout_steps, worlds), dtype=torch.bool, device=device
        )
        self.truncated = torch.empty_like(self.terminated)
        self.position = 0

    def store(
        self,
        observation: dict[str, torch.Tensor],
        action: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        next_value: torch.Tensor,
    ) -> None:
        if self.position >= self.rollout_steps:
            raise RuntimeError("rollout buffer is already full")
        for key, value_tensor in observation.items():
            self.observations[key][self.position].copy_(value_tensor)
        self.actions[self.position].copy_(action)
        self.log_probs[self.position].copy_(log_prob)
        self.values[self.position].copy_(value)
        self.rewards[self.position].copy_(reward)
        self.terminated[self.position].copy_(terminated)
        self.truncated[self.position].copy_(truncated)
        self.next_values[self.position].copy_(next_value)
        self.position += 1

    def compute_advantages(self, gamma: float, gae_lambda: float) -> tuple[torch.Tensor, torch.Tensor]:
        if self.position != self.rollout_steps:
            raise RuntimeError("rollout buffer is not full")
        advantages = torch.zeros_like(self.rewards)
        next_advantage = torch.zeros_like(self.rewards[0])
        for step in reversed(range(self.rollout_steps)):
            # timeout 使用终止观测 bootstrap，但 GAE 不跨越 reset 边界传播。
            bootstrap = (~self.terminated[step]).to(torch.float32)
            continue_episode = (~(self.terminated[step] | self.truncated[step])).to(torch.float32)
            delta = (
                self.rewards[step]
                + gamma * self.next_values[step] * bootstrap
                - self.values[step]
            )
            next_advantage = delta + gamma * gae_lambda * continue_episode * next_advantage
            advantages[step] = next_advantage
        return advantages, advantages + self.values


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def update_policy(
    policy: MAPPOPolicy,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    config: PPOConfig,
) -> UpdateStats:
    advantages, returns = buffer.compute_advantages(config.gamma, config.gae_lambda)
    batch_worlds = buffer.rollout_steps * buffer.values.shape[1]
    flat_observations = {
        key: value.flatten(0, 1) for key, value in buffer.observations.items()
    }
    flat_actions = buffer.actions.flatten(0, 1)
    flat_log_probs = buffer.log_probs.flatten(0, 1)
    flat_values = buffer.values.flatten(0, 1)
    flat_advantages = advantages.flatten(0, 1)
    flat_returns = returns.flatten(0, 1)
    normalized_advantages = (flat_advantages - flat_advantages.mean()) / (
        flat_advantages.std(unbiased=False) + 1e-8
    )

    minibatch_size = max(1, batch_worlds // config.num_minibatches)
    totals = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "gradient_norm": 0.0,
    }
    with torch.no_grad():
        return_variance = flat_returns.var(unbiased=False)
        explained_variance = torch.where(
            return_variance > 1e-8,
            1.0 - (flat_returns - flat_values).var(unbiased=False) / return_variance,
            torch.zeros_like(return_variance),
        )
    updates = 0
    stop_early = False
    for _ in range(config.update_epochs):
        permutation = torch.randperm(batch_worlds, device=flat_values.device)
        for start in range(0, batch_worlds, minibatch_size):
            indices = permutation[start : start + minibatch_size]
            observation = {key: value[indices] for key, value in flat_observations.items()}
            mask = observation["attacker_mask"]
            _, new_log_prob, entropy, new_value = policy.act(
                observation, action=flat_actions[indices]
            )
            log_ratio = new_log_prob - flat_log_probs[indices]
            ratio = log_ratio.exp()
            advantage = normalized_advantages[indices, None]
            policy_loss_unclipped = -advantage * ratio
            policy_loss_clipped = -advantage * ratio.clamp(
                1.0 - config.clip_coef, 1.0 + config.clip_coef
            )
            policy_loss = _masked_mean(
                torch.maximum(policy_loss_unclipped, policy_loss_clipped), mask
            )

            old_value = flat_values[indices]
            value_unclipped = (new_value - flat_returns[indices]).square()
            value_clipped = old_value + (new_value - old_value).clamp(
                -config.value_clip_coef, config.value_clip_coef
            )
            value_loss = 0.5 * torch.maximum(
                value_unclipped, (value_clipped - flat_returns[indices]).square()
            ).mean()
            entropy_mean = _masked_mean(entropy, mask)
            loss = (
                policy_loss
                + config.value_coef * value_loss
                - config.entropy_coef * entropy_mean
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(), config.max_grad_norm
            )
            optimizer.step()

            with torch.no_grad():
                approx_kl = _masked_mean((ratio - 1.0) - log_ratio, mask)
                clip_fraction = _masked_mean(
                    ((ratio - 1.0).abs() > config.clip_coef).to(torch.float32), mask
                )
            values = {
                "loss": loss,
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy_mean,
                "approx_kl": approx_kl,
                "clip_fraction": clip_fraction,
                "gradient_norm": gradient_norm,
            }
            for key, value in values.items():
                totals[key] += float(value.detach().item())
            updates += 1
            if float(approx_kl.item()) > config.target_kl:
                stop_early = True
                break
        if stop_early:
            break

    divisor = max(1, updates)
    averaged = {key: value / divisor for key, value in totals.items()}
    return UpdateStats(
        **averaged,
        explained_variance=float(explained_variance.item()),
        optimizer_steps=updates,
    )
