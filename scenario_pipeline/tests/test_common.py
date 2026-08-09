from __future__ import annotations

import math
import unittest

from scenario_pipeline.resample import Sample, build_timeline, resample_samples


class ResampleTest(unittest.TestCase):
    def test_shortest_angle_interpolation(self) -> None:
        timeline = build_timeline(0.0, 0.2)
        series = resample_samples(
            [
                Sample(0.0, 0, 0, math.pi - 0.01, 0, 0),
                Sample(0.2, 2, 0, -math.pi + 0.01, 2, 0),
            ],
            timeline,
        )
        self.assertAlmostEqual(series["x"][1], 1.0)
        self.assertAlmostEqual(abs(series["yaw"][1]), math.pi)

    def test_invalid_samples_are_not_interpolated(self) -> None:
        timeline = build_timeline(0.0, 0.2)
        series = resample_samples(
            [
                Sample(0.0, 0, 0, 0, 0, 0, True),
                Sample(0.1, 0, 0, 0, 0, 0, False),
                Sample(0.2, 2, 0, 0, 0, 0, True),
            ],
            timeline,
        )
        self.assertEqual(series["valid"], [True, False, True])
        self.assertIsNone(series["x"][1])


if __name__ == "__main__":
    unittest.main()
