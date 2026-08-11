from __future__ import annotations

import csv
import math
import tempfile
import unittest
from pathlib import Path

from gpudrive_cuda.tools.render_trace import load_trace, runtime_for_world, vehicle_corners


class RenderTraceTest(unittest.TestCase):
    def test_vehicle_corners_preserve_dimensions(self) -> None:
        corners = vehicle_corners(2.0, 3.0, 0.0, 4.0, 2.0)
        self.assertEqual(corners, [(4.0, 4.0), (4.0, 2.0), (0.0, 2.0), (0.0, 4.0)])

    def test_vehicle_corners_follow_yaw(self) -> None:
        corners = vehicle_corners(0.0, 0.0, math.pi / 2.0, 4.0, 2.0)
        expected = [(-1.0, 2.0), (1.0, 2.0), (1.0, -2.0), (-1.0, -2.0)]
        for actual, target in zip(corners, expected):
            self.assertAlmostEqual(actual[0], target[0], places=6)
            self.assertAlmostEqual(actual[1], target[1], places=6)

    def test_trace_is_filtered_and_grouped_by_world(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            fields = [
                "world", "step", "agent_slot", "agent_id", "valid", "control_mode",
                "x", "y", "yaw", "vx", "vy", "acceleration", "steering",
                "actual_acceleration", "actual_steering", "longitudinal_velocity",
                "lateral_velocity", "yaw_rate",
                "collided_vehicle", "collided_road", "offroad", "reached_goal", "world_done",
            ]
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for world in (0, 1):
                    writer.writerow({
                        "world": world, "step": 0, "agent_slot": 0, "agent_id": "ego",
                        "valid": 1, "control_mode": "auto", "x": world, "y": 0,
                        "yaw": 0, "vx": 1, "vy": 0, "acceleration": 0, "steering": 0,
                        "actual_acceleration": 0, "actual_steering": 0,
                        "longitudinal_velocity": 1, "lateral_velocity": 0, "yaw_rate": 0,
                        "collided_vehicle": 0, "collided_road": 0, "offroad": 0,
                        "reached_goal": 0, "world_done": 0,
                    })
            frames = load_trace(path, 1)
            self.assertEqual(list(frames), [0])
            self.assertEqual(frames[0][0].world, 1)
            self.assertEqual(frames[0][0].x, 1.0)

    def test_runtime_directory_round_robin(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("a", "b"):
                scene = root / name
                scene.mkdir()
                (scene / "manifest.json").write_text("{}", encoding="utf-8")
            self.assertEqual(runtime_for_world(root, 3).name, "b")


if __name__ == "__main__":
    unittest.main()
