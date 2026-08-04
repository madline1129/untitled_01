import importlib.util
import importlib
from pathlib import Path
import sys
import types


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(path: Path, name: str):
    if "carla" not in sys.modules:
        sys.modules["carla"] = types.ModuleType("carla")
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return importlib.import_module(f"real2sim.{name}")


def test_stage4_module_exists_and_exports_spawn():
    module = _load_module(ROOT / "stage4_spawn.py", "stage4_spawn")
    assert hasattr(module, "stage4_spawn_actors")


def test_stage5_module_exists_and_exports_capture():
    module = _load_module(ROOT / "stage5_capture.py", "stage5_capture")
    assert hasattr(module, "stage5_capture")
    assert hasattr(module, "capture")
