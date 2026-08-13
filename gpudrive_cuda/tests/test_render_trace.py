from __future__ import annotations

import csv
import math
import tempfile
import unittest
from pathlib import Path

from gpudrive_cuda.tools.render_trace import (
    TraceRow,
    interpolate_trace_frame,
    interpolate_yaw,
    load_trace,
    runtime_for_world,
    sample_times,
    vehicle_corners,
)


class RenderTraceTest(unittest.TestCase):
    @staticmethod
    def trace_row(step: int, x: float, yaw: float) -> TraceRow:
        return TraceRow(
            world=0,
            step=step,
            agent_slot=0,
            agent_id="ego",
            valid=True,
            control_mode="auto",
            x=x,
            y=0.0,
            yaw=yaw,
            vx=1.0,
            vy=0.0,
            collided_vehicle=False,
            collided_road=False,
            offroad=False,
            reached_goal=False,
        )

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

    def test_yaw_interpolation_uses_shortest_arc(self) -> None:
        value = interpolate_yaw(math.radians(179.0), math.radians(-179.0), 0.5)
        self.assertAlmostEqual(abs(value), math.pi, places=6)

    def test_trace_interpolation_blends_state_and_uses_nearest_event(self) -> None:
        left = self.trace_row(0, 0.0, 0.0)
        right = TraceRow(
            **{
                **self.trace_row(1, 2.0, math.pi / 2.0).__dict__,
                "collided_vehicle": True,
            }
        )
        frames = {0: [left], 1: [right]}
        result = interpolate_trace_frame(frames, 0.75)[0]
        self.assertAlmostEqual(result.x, 1.5)
        self.assertAlmostEqual(result.yaw, 3.0 * math.pi / 8.0)
        self.assertTrue(result.collided_vehicle)

    def test_ten_second_frame_schedule_has_exact_counts(self) -> None:
        gif_times = sample_times(10.0, 10)
        mp4_times = sample_times(10.0, 20)
        self.assertEqual(len(gif_times), 100)
        self.assertEqual(len(mp4_times), 200)
        self.assertEqual(gif_times[0], 0.0)
        self.assertAlmostEqual(gif_times[-1], 9.9)
        self.assertAlmostEqual(mp4_times[-1], 9.95)


if __name__ == "__main__":
    unittest.main()
