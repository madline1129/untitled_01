import importlib
import logging
from pathlib import Path
import sys
import types

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_stage3():
    if "carla" not in sys.modules:
        sys.modules["carla"] = types.ModuleType("carla")
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return importlib.import_module("real2sim.stage3_local_map")


class _FakeWaypoint:
    def __init__(self, x, y, yaw=0.0):
        self.transform = types.SimpleNamespace(
            location=types.SimpleNamespace(x=x, y=y, z=0.0),
            rotation=types.SimpleNamespace(yaw=yaw),
        )
        self.road_id = 1
        self.lane_id = 1


class _FakeMap:
    def __init__(self, waypoints):
        self._waypoints = waypoints

    def generate_waypoints(self, _spacing):
        return self._waypoints


class _FakeWorld:
    def __init__(self, waypoints):
        self._map = _FakeMap(waypoints)

    def wait_for_tick(self):
        return None

    def tick(self):
        return None

    def get_settings(self):
        import types as _types
        return _types.SimpleNamespace(synchronous_mode=False, fixed_delta_seconds=0.0)

    def apply_settings(self, _settings):
        return None

    def get_map(self):
        return self._map


class _FakeClient:
    def __init__(self, waypoints):
        self._world = _FakeWorld(waypoints)

    def load_world(self, _town):
        return self._world


def test_match_carla_town_logs_refine_improvement(caplog, monkeypatch, tmp_path):
    stage3 = _load_stage3()
    caplog.set_level(logging.INFO, logger="real2sim")

    wp_seed = _FakeWaypoint(0.0, 0.0)
    wp_neighbor = _FakeWaypoint(1.0, 0.0)
    client = _FakeClient([wp_seed, wp_neighbor])

    def _fake_render(waypoint, *_args, **_kwargs):
        marker = 1 if waypoint is wp_seed else 2
        mask = np.zeros((3, 5, 5), dtype=np.uint8)
        mask[0] = marker
        return mask

    score_by_marker = {1: [0.9], 2: [0.8, 0.95]}

    def _fake_similarity(_rotated, m_prime, use_lpips=True):
        marker = int(m_prime[0, 0, 0])
        return score_by_marker[marker].pop(0)

    monkeypatch.setattr(stage3, "render_carla_local_map", _fake_render)
    monkeypatch.setattr(stage3, "compute_map_similarity", _fake_similarity)

    rotated = {0: np.zeros((3, 5, 5), dtype=np.uint8)}
    stage3.match_carla_town(
        rotated,
        "Town01",
        client,
        tmp_path,
        top_k=1,
        use_lpips=False,
        refine_radius=5.0,
    )

    assert "refine improved" in caplog.text
