from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scenario_pipeline.adapters.nuplan import convert_nuplan_database
from scenario_pipeline.models import MapFeature
from scenario_pipeline.runtime import (
    RuntimeConfig,
    compile_runtime_scenario,
    read_runtime_tensor,
    validate_runtime_directory,
    write_runtime_scenario,
)


ROOT = Path(__file__).resolve().parents[2]
MOCK_DB = ROOT / "dataset/nuplan/nuplan-v1.1/splits/mock/mock_nuplan.db"


class RuntimeCompilerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scenario = convert_nuplan_database(MOCK_DB)[0]
        self.config = RuntimeConfig(
            max_agents=8,
            history_steps=4,
            max_future_steps=100,
            max_map_features=8,
            max_map_points=32,
            max_map_edges=16,
            max_traffic_lights=4,
            max_route_features=8,
        )

    def test_reset_history_and_reference_are_separated(self) -> None:
        runtime = compile_runtime_scenario(self.scenario, self.config)

        self.assertEqual(runtime.manifest["agent_ids"][0], "ego")
        self.assertEqual(runtime.manifest["episode_steps"], 95)
        self.assertEqual(runtime.tensors["agent_initial_state"].values[:5], [0.0, 0.0, 0.0, 6.0, 0.0])
        self.assertEqual(runtime.tensors["agent_history_valid"].values[3 * 8], 1)
        self.assertEqual(sum(runtime.tensors["agent_history_valid"].values[:3 * 8]), 0)
        self.assertFalse(runtime.tensors["reference_future"].policy_visible)
        self.assertFalse(runtime.tensors["agent_goal"].policy_visible)
        self.assertGreater(runtime.tensors["reference_future"].values[0], 0.0)
        self.assertEqual(runtime.tensors["agent_initial_valid"].values[:4], [1, 1, 1, 1])
        self.assertEqual(runtime.tensors["agent_initial_valid"].values[4:], [0, 0, 0, 0])

    def test_map_topology_and_traffic_light_link(self) -> None:
        self.scenario.map_features = [
            MapFeature(
                id="63908",
                type="lane",
                geometry_type="polyline",
                geometry=[[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
                successor_ids=["next_lane"],
            ),
            MapFeature(
                id="next_lane",
                type="lane",
                geometry_type="polyline",
                geometry=[[20.0, 0.0, 0.0], [40.0, 1.0, 0.0]],
                predecessor_ids=["63908"],
            ),
        ]
        runtime = compile_runtime_scenario(self.scenario, self.config)

        self.assertEqual(runtime.manifest["counts"]["map_features"], 2)
        self.assertEqual(runtime.manifest["counts"]["map_points"], 4)
        self.assertEqual(runtime.manifest["counts"]["map_edges"], 2)
        self.assertEqual(runtime.tensors["traffic_light_feature_index"].values[0], 0)
        self.assertEqual(runtime.tensors["traffic_light_state"].values[0], 3)

    def test_binary_round_trip_and_directory_validation(self) -> None:
        runtime = compile_runtime_scenario(self.scenario, self.config)
        with tempfile.TemporaryDirectory() as directory:
            runtime_dir = Path(directory) / "scene"
            write_runtime_scenario(runtime, runtime_dir)
            warnings = validate_runtime_directory(runtime_dir)
            spec, values = read_runtime_tensor(runtime_dir, "agent_initial_state")

            self.assertEqual(spec["shape"], [8, 5])
            self.assertEqual(values[:5], [0.0, 0.0, 0.0, 6.0, 0.0])
            self.assertTrue(any("maps_root" in warning for warning in warnings))


if __name__ == "__main__":
    unittest.main()
