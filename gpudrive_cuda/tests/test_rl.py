from __future__ import annotations

import math
import unittest

import torch

from gpudrive_cuda.rl.config import PPOConfig
from gpudrive_cuda.rl.env import AdversarialDrivingEnv, ControlMode
from gpudrive_cuda.rl.model import MAPPOPolicy
from gpudrive_cuda.rl.ppo import RolloutBuffer, update_policy


def observation(worlds: int, attackers: int, partners: int = 3, maps: int = 4):
    result = {
        "self": torch.zeros(worlds, attackers, 9),
        "ego": torch.zeros(worlds, attackers, 9),
        "partners": torch.zeros(worlds, attackers, partners, 9),
        "partner_mask": torch.zeros(worlds, attackers, partners, dtype=torch.bool),
        "map": torch.zeros(worlds, attackers, maps, 7),
        "map_mask": torch.zeros(worlds, attackers, maps, dtype=torch.bool),
        "attacker_mask": torch.zeros(worlds, attackers, dtype=torch.bool),
    }
    return result


class FakeSimulator:
    def __init__(self, worlds: int = 2, agents: int = 6):
        self.num_worlds = worlds
        self.max_agents = agents
        self._initial_states = torch.zeros(worlds, agents, 5)
        self._initial_states[:, 0, 3] = 6.0
        self._initial_states[:, 1, 0] = 8.0
        self._initial_states[:, 2, 0] = 18.0
        self._valid = torch.zeros(worlds, agents, dtype=torch.uint8)
        self._valid[:, :3] = 1
        self._agent_type = torch.zeros(worlds, agents, dtype=torch.int32)
        self._agent_type[:, :3] = 1
        self._is_ego = torch.zeros(worlds, agents, dtype=torch.uint8)
        self._is_ego[:, 0] = 1
        self._controllable = self._valid.clone()
        self._collision_on_step = False
        self.reset()

    def _update_observations(self):
        self._self.zero_()
        self._self[..., :5] = self._states
        self._self[..., 5] = torch.linalg.vector_norm(self._states[..., 3:5], dim=-1)
        self._self[..., 6] = 4.8
        self._self[..., 7] = 1.9
        self._self[..., 10] = (95 - self._world_step)[:, None]

    def _frame(self):
        return {
            "states": self._states,
            "dynamics": self._dynamics,
            "applied_actions": self._actions,
            "events": self._events,
            "self": self._self,
            "self_valid": self._valid,
            "partners": self._partners,
            "partner_valid": self._partner_valid,
            "map": self._map,
            "map_valid": self._map_valid,
            "valid": self._valid,
            "agent_type": self._agent_type,
            "agent_is_ego": self._is_ego,
            "agent_controllable": self._controllable,
            "control_modes": self._modes,
            "world_step": self._world_step,
            "world_done": self._world_done,
        }

    def reset(self):
        self._states = self._initial_states.clone()
        self._dynamics = torch.zeros(self.num_worlds, self.max_agents, 5)
        self._actions = torch.zeros(self.num_worlds, self.max_agents, 2)
        self._events = torch.zeros(self.num_worlds, self.max_agents, 5, dtype=torch.int32)
        self._modes = torch.zeros(self.num_worlds, self.max_agents, dtype=torch.uint8)
        self._world_step = torch.zeros(self.num_worlds, dtype=torch.int32)
        self._world_done = torch.zeros(self.num_worlds, dtype=torch.int32)
        self._self = torch.zeros(self.num_worlds, self.max_agents, 11)
        self._partners = torch.zeros(self.num_worlds, self.max_agents, 16, 9)
        self._partner_valid = torch.zeros(
            self.num_worlds, self.max_agents, 16, dtype=torch.uint8
        )
        self._map = torch.zeros(self.num_worlds, self.max_agents, 64, 7)
        self._map_valid = torch.zeros(self.num_worlds, self.max_agents, 64, dtype=torch.uint8)
        self._update_observations()
        return self._frame()

    def reset_worlds(self, reset_mask):
        mask = reset_mask.bool()
        self._states[mask] = self._initial_states[mask]
        self._dynamics[mask] = 0
        self._actions[mask] = 0
        self._events[mask] = 0
        self._modes[mask] = 0
        self._world_step[mask] = 0
        self._world_done[mask] = 0
        self._update_observations()
        return self._frame()

    def set_control_modes(self, modes):
        self._modes.copy_(modes)

    def step(self, actions):
        self._actions.copy_(actions)
        self._events.zero_()
        self._world_step += 1
        if self._collision_on_step:
            self._events[:, 1, 0] = 1
            self._events[:, 1, 1] = 1
        self._update_observations()
        return self._frame()


class EnvironmentTest(unittest.TestCase):
    def test_agent_selection_reward_and_padding(self):
        simulator = FakeSimulator()
        env = AdversarialDrivingEnv(simulator, max_attackers=16)
        env.set_curriculum(4, ControlMode.RESIDUAL)
        obs = env.reset()
        self.assertTrue(torch.equal(env.attacker_slots[:, :2], torch.tensor([[1, 2], [1, 2]])))
        self.assertEqual(env.attacker_mask.sum().item(), 4)
        for value in obs.values():
            if value.is_floating_point():
                self.assertTrue(torch.isfinite(value).all())

        simulator._collision_on_step = True
        actions = torch.zeros(2, 16, 2)
        _, reward, terminated, truncated, info = env.step(actions)
        self.assertTrue(terminated.all())
        self.assertFalse(truncated.any())
        self.assertTrue((reward > 9.0).all())
        self.assertTrue(info["ego_collision"].all())
        for key in (
            "progress",
            "risk",
            "penalty",
            "road_collision_rate",
            "steering_cost",
            "acceleration_cost",
            "action_delta_cost",
        ):
            self.assertIn(key, info)
            self.assertTrue(torch.isfinite(info[key]).all())

    def test_control_mode_is_written_only_to_active_agents(self):
        simulator = FakeSimulator(worlds=1)
        env = AdversarialDrivingEnv(simulator, max_attackers=16)
        env.set_curriculum(1, ControlMode.DIRECT)
        self.assertEqual(int(simulator._modes[0, 1]), int(ControlMode.DIRECT))
        self.assertEqual(int(simulator._modes[0, 2]), int(ControlMode.AUTO))

        actions = torch.zeros(1, 16, 2)
        actions[0, 0] = torch.tensor([-1.0, 1.0])
        env.step(actions)
        self.assertAlmostEqual(float(simulator._actions[0, 1, 0]), -6.0)
        self.assertAlmostEqual(float(simulator._actions[0, 1, 1]), 0.6)


class ModelAndPPOTest(unittest.TestCase):
    def test_all_padding_is_finite(self):
        policy = MAPPOPolicy(latent_dim=32)
        obs = observation(2, 3)
        action, log_prob, entropy, value = policy.act(obs)
        for tensor in (action, log_prob, entropy, value):
            self.assertTrue(torch.isfinite(tensor).all())

    def test_squashed_log_probability_near_limit(self):
        policy = MAPPOPolicy(latent_dim=32)
        obs = observation(1, 2)
        obs["attacker_mask"].fill_(True)
        action = torch.full((1, 2, 2), 0.999999)
        _, log_prob, _, _ = policy.act(obs, action=action)
        self.assertTrue(torch.isfinite(log_prob).all())

    def test_gae_distinguishes_timeout_and_termination(self):
        obs = observation(1, 1)
        buffer = RolloutBuffer(1, obs)
        buffer.store(
            obs,
            torch.zeros(1, 1, 2),
            torch.zeros(1, 1),
            torch.ones(1),
            torch.ones(1),
            torch.zeros(1, dtype=torch.bool),
            torch.ones(1, dtype=torch.bool),
            torch.full((1,), 2.0),
        )
        advantage, _ = buffer.compute_advantages(0.99, 0.95)
        self.assertAlmostEqual(float(advantage[0, 0]), 1.98, places=5)

        terminal = RolloutBuffer(1, obs)
        terminal.store(
            obs,
            torch.zeros(1, 1, 2),
            torch.zeros(1, 1),
            torch.ones(1),
            torch.ones(1),
            torch.ones(1, dtype=torch.bool),
            torch.zeros(1, dtype=torch.bool),
            torch.full((1,), 2.0),
        )
        advantage, _ = terminal.compute_advantages(0.99, 0.95)
        self.assertAlmostEqual(float(advantage[0, 0]), 0.0, places=5)

    def test_ppo_update_changes_parameters(self):
        torch.manual_seed(7)
        policy = MAPPOPolicy(latent_dim=32)
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        obs = observation(2, 2)
        obs["attacker_mask"].fill_(True)
        obs["partner_mask"].fill_(True)
        obs["map_mask"].fill_(True)
        obs["self"].normal_()
        obs["ego"].normal_()
        buffer = RolloutBuffer(2, obs)
        before = [parameter.detach().clone() for parameter in policy.parameters()]
        for _ in range(2):
            with torch.no_grad():
                action, log_prob, _, value = policy.act(obs)
            buffer.store(
                obs,
                action,
                log_prob,
                value,
                torch.ones(2),
                torch.zeros(2, dtype=torch.bool),
                torch.zeros(2, dtype=torch.bool),
                torch.zeros(2),
            )
        stats = update_policy(
            policy,
            optimizer,
            buffer,
            PPOConfig(update_epochs=1, num_minibatches=1, target_kl=1.0),
        )
        self.assertTrue(math.isfinite(stats.loss))
        self.assertTrue(math.isfinite(stats.explained_variance))
        self.assertTrue(math.isfinite(stats.gradient_norm))
        self.assertGreater(stats.optimizer_steps, 0)
        self.assertTrue(
            any(not torch.equal(old, new) for old, new in zip(before, policy.parameters()))
        )


if __name__ == "__main__":
    unittest.main()
