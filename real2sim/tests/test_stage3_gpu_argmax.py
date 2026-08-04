import importlib
from pathlib import Path
import sys
import types

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_stage3():
    if "carla" not in sys.modules:
        sys.modules["carla"] = types.ModuleType("carla")
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return importlib.import_module("real2sim.stage3_local_map")


def test_select_best_angles_from_similarity_matrix():
    stage3 = _load_stage3()
    sim = torch.tensor([
        [0.1, 0.7, 0.3],
        [0.8, 0.2, 0.6],
    ], dtype=torch.float32)
    angles = [0, 90, 180]

    best_scores, best_angles = stage3._select_best_angles_from_similarity(sim, angles)

    assert best_scores[0] == pytest.approx(0.7)
    assert best_scores[1] == pytest.approx(0.8)
    assert best_angles == [90, 0]
