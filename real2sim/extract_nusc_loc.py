#!/usr/bin/env python3
"""Extract location-specific scene index from nuScenes trainval metadata.

Output: real2sim/nusc-loc.json — maps location prefix ("boston", "singapore")
to lists of scene names and tokens. Also prints summary to stdout.

Usage:
  python real2sim/extract_nusc_loc.py [--meta-dir PATH]
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_META_DIR = REPO_ROOT / "data" / "nuscenes" / "v1.0-trainval"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta-dir", type=Path, default=DEFAULT_META_DIR)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "real2sim" / "nusc-loc.json")
    args = parser.parse_args()

    with open(args.meta_dir / "scene.json") as f:
        scenes = json.load(f)
    with open(args.meta_dir / "log.json") as f:
        logs = json.load(f)

    log_by_token = {l["token"]: l for l in logs}

    loc_groups = {}
    for s in scenes:
        loc = log_by_token[s["log_token"]]["location"]
        prefix = loc.split("-")[0]
        loc_groups.setdefault(prefix, {"scene_names": [], "scene_tokens": []})
        loc_groups[prefix]["scene_names"].append(s["name"])
        loc_groups[prefix]["scene_tokens"].append(s["token"])

    with open(args.output, "w") as f:
        json.dump(loc_groups, f, indent=2)
    print(f"Saved → {args.output}")
    for prefix, data in sorted(loc_groups.items()):
        print(f"  {prefix}: {len(data['scene_names'])} scenes")

    return loc_groups


if __name__ == "__main__":
    main()
