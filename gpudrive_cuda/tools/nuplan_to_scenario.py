#!/usr/bin/env python3

"""Compatibility entry point for the canonical nuPlan converter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scenario_pipeline.cli import main as pipeline_main  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("db", type=Path, help="nuPlan .db file")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "dataset/canonical/nuplan",
    )
    parser.add_argument("--maps-root", type=Path)
    parser.add_argument("--scene-id")
    args = parser.parse_args()

    command = [
        "convert-nuplan",
        "--input", str(args.db),
        "--output", str(args.output),
    ]
    if args.maps_root is not None:
        command.extend(["--maps-root", str(args.maps_root)])
    if args.scene_id is not None:
        command.extend(["--scene-id", args.scene_id])
    return pipeline_main(command)


if __name__ == "__main__":
    raise SystemExit(main())
