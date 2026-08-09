#!/usr/bin/env python3

"""Compile CanonicalScenario JSON files into fixed-capacity RL tensors."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scenario_pipeline.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(["compile-rl", *sys.argv[1:]]))
