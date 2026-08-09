"""Canonical JSON input and output."""

from __future__ import annotations

import json
from pathlib import Path

from .models import CanonicalScenario


def write_scenario(scenario: CanonicalScenario, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(scenario.to_dict(), output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")


def read_scenario(path: Path) -> CanonicalScenario:
    with path.open(encoding="utf-8") as source:
        return CanonicalScenario.from_dict(json.load(source))
