import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "stage3_selection.py"
SPEC = importlib.util.spec_from_file_location("stage3_selection", MODULE_PATH)
stage3_selection = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage3_selection)

compute_mean_ssim = stage3_selection.compute_mean_ssim
compute_spawn_success_rate = stage3_selection.compute_spawn_success_rate
select_best_by_ssim = stage3_selection.select_best_by_ssim
select_top_q_by_spawn = stage3_selection.select_top_q_by_spawn


def test_compute_spawn_success_rate_uses_effective_annotations():
    spawn_result = {
        "total_annotations": 20,
        "skipped": 5,
        "spawned": 12,
    }
    assert compute_spawn_success_rate(spawn_result) == pytest.approx(12 / 15)


def test_compute_spawn_success_rate_handles_zero_effective():
    spawn_result = {
        "total_annotations": 5,
        "skipped": 5,
        "spawned": 0,
    }
    assert compute_spawn_success_rate(spawn_result) == 0.0


def test_compute_mean_ssim_ignores_missing_fields():
    sim_results = {
        "FRONT": {"ssim": 0.8},
        "BACK": {"ssim": 0.6},
        "LEFT": {"lpips": 0.2},
        "_summary": {"ssim_mean": 0.1},
    }
    assert compute_mean_ssim(sim_results) == pytest.approx(0.7)


def test_select_top_q_by_spawn_orders_descending():
    candidates = [
        {"id": "a", "spawn_rate": 0.5},
        {"id": "b", "spawn_rate": 0.9},
        {"id": "c", "spawn_rate": 0.7},
    ]
    top = select_top_q_by_spawn(candidates, q=2)
    assert [entry["id"] for entry in top] == ["b", "c"]


def test_select_best_by_ssim_returns_highest_candidate():
    candidates = [
        {"id": "a", "ssim_score": 0.61},
        {"id": "b", "ssim_score": 0.87},
        {"id": "c", "ssim_score": 0.81},
    ]
    best = select_best_by_ssim(candidates)
    assert best["id"] == "b"
