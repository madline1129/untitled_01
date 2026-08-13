"""参数共享 actor 与集中式 critic。"""

from __future__ import annotations

import math

import torch
from torch import nn


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, output_dim),
        nn.Tanh(),
    )


def masked_max(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """空集合返回零，避免 padding world 产生 -inf 或 NaN。"""

    masked = values.masked_fill(~mask[..., None], -torch.inf)
    maximum = masked.max(dim=dim).values
    # mask 少了最后一个 feature 维，负索引需要向右平移一位。
    mask_dim = dim if dim >= 0 else dim + 1
    any_valid = mask.any(dim=mask_dim)
    return torch.where(any_valid[..., None], maximum, torch.zeros_like(maximum))


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weights = mask[..., None].to(values.dtype)
    total = (values * weights).sum(dim=dim)
    count = weights.sum(dim=dim).clamp_min(1.0)
    return total / count


class LocalActorEncoder(nn.Module):
    def __init__(self, latent_dim: int = 256) -> None:
        super().__init__()
        self.agent_type_embedding = nn.Embedding(5, 8)
        self.map_type_embedding = nn.Embedding(10, 8)
        self.self_encoder = _mlp(18, 64, 64)
        self.partner_encoder = _mlp(16, 64, 64)
        self.map_encoder = _mlp(14, 64, 64)
        self.fusion = _mlp(192, latent_dim, latent_dim)

    def forward(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        self_ego = torch.cat([observation["self"], observation["ego"]], dim=-1)
        self_latent = self.self_encoder(self_ego)

        partners = observation["partners"]
        partner_type = partners[..., 7].round().long().clamp(0, 4)
        partner_numeric = torch.cat([partners[..., :7], partners[..., 8:]], dim=-1)
        partner_input = torch.cat(
            [partner_numeric, self.agent_type_embedding(partner_type)], dim=-1
        )
        partner_entities = self.partner_encoder(partner_input)
        partner_latent = masked_max(
            partner_entities, observation["partner_mask"], dim=-2
        )

        map_features = observation["map"]
        map_type = map_features[..., 4].round().long().clamp(0, 9)
        map_numeric = torch.cat([map_features[..., :4], map_features[..., 5:]], dim=-1)
        map_input = torch.cat([map_numeric, self.map_type_embedding(map_type)], dim=-1)
        map_entities = self.map_encoder(map_input)
        map_latent = masked_max(map_entities, observation["map_mask"], dim=-2)

        latent = self.fusion(torch.cat([self_latent, partner_latent, map_latent], dim=-1))
        return latent * observation["attacker_mask"][..., None]


class MAPPOPolicy(nn.Module):
    """actor 按 Agent 因子化，critic 对一个 world 输出一个团队价值。"""

    def __init__(self, latent_dim: int = 256) -> None:
        super().__init__()
        self.encoder = LocalActorEncoder(latent_dim)
        self.actor_mean = nn.Linear(latent_dim, 2)
        self.log_std = nn.Parameter(torch.full((2,), -0.5))
        self.critic = nn.Sequential(
            nn.Linear(2 * latent_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, 1),
        )
        self.apply(self._initialize)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.zeros_(self.critic[-1].bias)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
            nn.init.zeros_(module.bias)

    def encode(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.encoder(observation)

    def value_from_latent(self, latent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat(
            [masked_mean(latent, mask, dim=-2), masked_max(latent, mask, dim=-2)],
            dim=-1,
        )
        return self.critic(pooled).squeeze(-1)

    def value(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        latent = self.encode(observation)
        return self.value_from_latent(latent, observation["attacker_mask"])

    def act(
        self,
        observation: dict[str, torch.Tensor],
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.encode(observation)
        mean = self.actor_mean(latent)
        std = self.log_std.clamp(-5.0, 2.0).exp().expand_as(mean)
        distribution = torch.distributions.Normal(mean, std)
        if action is None:
            pre_tanh = mean if deterministic else distribution.rsample()
            action = torch.tanh(pre_tanh)
        else:
            bounded = action.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            pre_tanh = torch.atanh(bounded)
        log_prob = (
            distribution.log_prob(pre_tanh)
            - torch.log(1.0 - action.square() + 1e-6)
        ).sum(dim=-1)
        # 正态分布熵作为 squashed distribution 的稳定近似。
        entropy = distribution.entropy().sum(dim=-1)
        value = self.value_from_latent(latent, observation["attacker_mask"])
        return action, log_prob, entropy, value
