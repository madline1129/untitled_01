from __future__ import annotations

import json
import math
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

from scenario_pipeline.adapters.nuplan import _resolve_map_api_arguments, convert_nuplan_database
from scenario_pipeline.io import read_scenario, write_scenario
from scenario_pipeline.validation import validate_scenario


ROOT = Path(__file__).resolve().parents[2]
MOCK_DB = ROOT / "dataset/nuplan/nuplan-v1.1/splits/mock/mock_nuplan.db"


class NuPlanAdapterTest(unittest.TestCase):
    def test_map_arguments_support_metadata_version_in_map_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            maps_root = Path(directory)
            metadata_path = maps_root / "nuplan-maps-v1.0.json"
            metadata_path.write_text(
                json.dumps({"us-nv-las-vegas-strip": {}}),
                encoding="utf-8",
            )

            arguments = _resolve_map_api_arguments(
                maps_root,
                "nuplan-maps-v1.0",
                "us-nv-las-vegas-strip",
            )

        self.assertEqual(arguments, ("nuplan-maps-v1.0", "us-nv-las-vegas-strip"))

    def test_map_arguments_support_city_slug_in_map_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            maps_root = Path(directory)
            metadata_path = maps_root / "nuplan-maps-v1.0.json"
            metadata_path.write_text(
                json.dumps({"us-nv-las-vegas-strip": {}}),
                encoding="utf-8",
            )

            arguments = _resolve_map_api_arguments(
                maps_root,
                "us-nv-las-vegas-strip",
                "las_vegas",
            )

        self.assertEqual(arguments, ("nuplan-maps-v1.0", "us-nv-las-vegas-strip"))

    def test_mock_database_conversion(self) -> None:
        scenario = convert_nuplan_database(MOCK_DB)[0]
        self.assertEqual(scenario.timing.num_steps, 96)
        self.assertEqual(len(scenario.agents), 4)
        self.assertEqual(sum(agent.type == "vehicle" for agent in scenario.agents), 3)
        ego = scenario.agents[0]
        self.assertAlmostEqual(ego.x[0], 0.0)
        self.assertAlmostEqual(ego.y[0], 0.0)
        self.assertAlmostEqual(ego.yaw[0], 0.0)
        self.assertAlmostEqual(ego.vx[0], 6.0, places=4)
        self.assertEqual(len(scenario.traffic_lights[0].states), 96)
        self.assertEqual(scenario.route.roadblock_ids, ["65581"])
        self.assertFalse(scenario.quality.map_available)
        validate_scenario(scenario)

    def test_initial_distance_is_preserved(self) -> None:
        scenario = convert_nuplan_database(MOCK_DB)[0]
        other = scenario.agents[1]
        local_distance = math.hypot(other.x[0], other.y[0])
        with sqlite3.connect(MOCK_DB) as connection:
            ego_x, ego_y = connection.execute(
                "SELECT x, y FROM ego_pose ORDER BY timestamp LIMIT 1"
            ).fetchone()
            box_x, box_y = connection.execute(
                "SELECT x, y FROM lidar_box JOIN lidar_pc ON lidar_pc.token=lidar_box.lidar_pc_token "
                "ORDER BY lidar_pc.timestamp, hex(lidar_box.track_token) LIMIT 1"
            ).fetchone()
        self.assertAlmostEqual(local_distance, math.hypot(box_x - ego_x, box_y - ego_y), places=4)

    def test_json_round_trip(self) -> None:
        scenario = convert_nuplan_database(MOCK_DB)[0]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "scenario.json"
            write_scenario(scenario, output)
            restored = read_scenario(output)
            self.assertEqual(restored.to_dict(), scenario.to_dict())
            validate_scenario(restored)

    def test_map_crop_coverage_warning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            changed_db = Path(directory) / "coverage.db"
            shutil.copyfile(MOCK_DB, changed_db)
            with sqlite3.connect(changed_db) as connection:
                connection.execute(
                    "UPDATE ego_pose SET x=x+500 WHERE timestamp=(SELECT MAX(timestamp) FROM ego_pose)"
                )
                connection.commit()
            scenario = convert_nuplan_database(changed_db)[0]
        self.assertFalse(scenario.quality.map_coverage_complete)
        self.assertTrue(any("150 meter" in warning for warning in scenario.quality.warnings))


if __name__ == "__main__":
    unittest.main()
