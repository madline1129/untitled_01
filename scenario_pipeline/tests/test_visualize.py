from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from scenario_pipeline.visualize import (
    animate_nuplan_comparison,
    animate_nuplan_conversion,
    visualize_nuplan_conversion,
)


ROOT = Path(__file__).resolve().parents[2]
MOCK_DB = ROOT / "dataset/nuplan/nuplan-v1.1/splits/mock/mock_nuplan.db"


@unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is not installed")
class NuPlanVisualizationTest(unittest.TestCase):
    def test_before_after_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "before_after.png"
            image, summary = visualize_nuplan_conversion(MOCK_DB, image_path)

            self.assertEqual(image, image_path)
            self.assertGreater(image.stat().st_size, 10_000)
            self.assertIn("转换后帧数：96", summary.read_text(encoding="utf-8"))

    def test_trajectory_animation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "trajectory.gif"
            animation = animate_nuplan_conversion(MOCK_DB, output, fps=2, stride=24)

            self.assertEqual(animation, output)
            self.assertGreater(animation.stat().st_size, 10_000)

    def test_synchronized_comparison_animation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison.gif"
            before, after, comparison = animate_nuplan_comparison(
                MOCK_DB, output, fps=2, stride=48
            )

            for artifact in (before, after, comparison):
                self.assertTrue(artifact.exists())
                self.assertGreater(artifact.stat().st_size, 10_000)


if __name__ == "__main__":
    unittest.main()
