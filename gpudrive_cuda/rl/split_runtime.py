"""从已编译 RuntimeScenario 中创建可复现的 train/eval 划分。"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_bytes(directory: Path, manifest: dict[str, Any], tensor: str) -> bytes:
    descriptor = manifest["tensors"][tensor]
    return (directory / descriptor["file"]).read_bytes()


def _candidate_vehicle_count(directory: Path, manifest: dict[str, Any]) -> int:
    valid = _read_bytes(directory, manifest, "agent_initial_valid")
    controllable = _read_bytes(directory, manifest, "agent_controllable")
    is_ego = _read_bytes(directory, manifest, "agent_is_ego")
    type_bytes = _read_bytes(directory, manifest, "agent_type")
    count = 0
    for slot in range(len(valid)):
        offset = slot * 4
        agent_type = int.from_bytes(type_bytes[offset : offset + 4], "little", signed=True)
        if valid[slot] and controllable[slot] and not is_ego[slot] and agent_type == 1:
            count += 1
    return count


def create_split(
    runtime_root: Path,
    output: Path,
    train_count: int,
    eval_count: int,
    seed: int,
    min_attackers: int,
    require_map: bool,
) -> dict[str, Any]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for manifest_path in sorted(runtime_root.rglob("manifest.json")):
        directory = manifest_path.parent
        with manifest_path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        if manifest.get("schema_version") != "rl-runtime-1.0":
            continue
        if require_map and int(manifest["counts"]["map_features"]) <= 0:
            continue
        if _candidate_vehicle_count(directory, manifest) < min_attackers:
            continue
        capacity_key = json.dumps(manifest["capacities"], sort_keys=True)
        groups[capacity_key].append(directory)

    required = train_count + eval_count
    compatible = [values for values in groups.values() if len(values) >= required]
    if not compatible:
        sizes = sorted((len(values) for values in groups.values()), reverse=True)
        raise ValueError(
            f"need {required} compatible scenes, largest capacity group has "
            f"{sizes[0] if sizes else 0}"
        )
    candidates = max(compatible, key=len)
    random.Random(seed).shuffle(candidates)
    selected = candidates[:required]
    train = selected[:train_count]
    evaluation = selected[train_count:]
    relative = lambda value: str(value.relative_to(runtime_root))
    result = {
        "schema_version": "rl-runtime-split-1.0",
        "runtime_root": str(runtime_root.resolve()),
        "seed": seed,
        "min_attackers": min_attackers,
        "require_map": require_map,
        "train": [relative(value) for value in train],
        "eval": [relative(value) for value in evaluation],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train", type=int, default=64)
    parser.add_argument("--eval", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-attackers", type=int, default=2)
    parser.add_argument("--allow-empty-map", action="store_true")
    args = parser.parse_args()
    result = create_split(
        args.runtime_root.resolve(),
        args.output,
        args.train,
        args.eval,
        args.seed,
        args.min_attackers,
        not args.allow_empty_map,
    )
    print(f"train scenes: {len(result['train'])}")
    print(f"eval scenes: {len(result['eval'])}")
    print(f"split: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
