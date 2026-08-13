"""训练场景集合文件读取。"""

from __future__ import annotations

import json
from pathlib import Path


def load_runtime_split(path: Path, split: str) -> list[str]:
    with path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != "rl-runtime-split-1.0":
        raise ValueError(f"unsupported runtime split schema: {manifest.get('schema_version')}")
    if split not in {"train", "eval"}:
        raise ValueError("split must be train or eval")
    root_value = Path(manifest["runtime_root"])
    root = root_value if root_value.is_absolute() else (path.parent / root_value).resolve()
    values = manifest.get(split, [])
    if not values:
        raise ValueError(f"runtime split contains no {split} scenes")
    paths = [(root / value).resolve() for value in values]
    missing = [value for value in paths if not (value / "manifest.json").is_file()]
    if missing:
        raise ValueError(f"runtime split references missing scenes: {missing[:3]}")
    return [str(value) for value in paths]
