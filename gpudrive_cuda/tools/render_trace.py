#!/usr/bin/env python3

"""Render a drive_sim_cli CSV trace as a top-down GIF and final PNG."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from array import array
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class TraceRow:
    world: int
    step: int
    agent_slot: int
    agent_id: str
    valid: bool
    control_mode: str
    x: float
    y: float
    yaw: float
    vx: float
    vy: float
    collided_vehicle: bool
    collided_road: bool
    offroad: bool
    reached_goal: bool


def runtime_directories(path: Path) -> list[Path]:
    if (path / "manifest.json").is_file():
        return [path]
    if not path.is_dir():
        raise ValueError(f"runtime input does not exist: {path}")
    directories = sorted({manifest.parent for manifest in path.rglob("manifest.json")})
    if not directories:
        raise ValueError(f"no runtime manifests found under: {path}")
    return directories


def runtime_for_world(path: Path, world: int) -> Path:
    directories = runtime_directories(path)
    return directories[world % len(directories)]


def read_tensor(runtime_dir: Path, manifest: dict, name: str) -> list[int | float]:
    spec = manifest["tensors"][name]
    typecode = {"float32": "f", "int32": "i", "uint8": "B"}[spec["dtype"]]
    count = math.prod(spec["shape"])
    values = array(typecode)
    with (runtime_dir / spec["file"]).open("rb") as stream:
        values.fromfile(stream, count)
        if stream.read(1):
            raise ValueError(f"tensor {name} has trailing bytes")
    if sys.byteorder != "little" and values.itemsize > 1:
        values.byteswap()
    return values.tolist()


def load_trace(path: Path, world: int) -> dict[int, list[TraceRow]]:
    frames: dict[int, list[TraceRow]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {
            "world", "step", "agent_slot", "agent_id", "valid", "control_mode",
            "x", "y", "yaw", "vx", "vy", "collided_vehicle", "collided_road",
            "offroad", "reached_goal",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("trace CSV is missing required columns")
        for value in reader:
            if int(value["world"]) != world:
                continue
            row = TraceRow(
                world=world,
                step=int(value["step"]),
                agent_slot=int(value["agent_slot"]),
                agent_id=value["agent_id"],
                valid=bool(int(value["valid"])),
                control_mode=value["control_mode"],
                x=float(value["x"]),
                y=float(value["y"]),
                yaw=float(value["yaw"]),
                vx=float(value["vx"]),
                vy=float(value["vy"]),
                collided_vehicle=bool(int(value["collided_vehicle"])),
                collided_road=bool(int(value["collided_road"])),
                offroad=bool(int(value["offroad"])),
                reached_goal=bool(int(value["reached_goal"])),
            )
            frames[row.step].append(row)
    if not frames:
        raise ValueError(f"trace contains no rows for world {world}")
    return dict(sorted(frames.items()))


def vehicle_corners(x: float, y: float, yaw: float, length: float, width: float) -> list[tuple[float, float]]:
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    result = []
    for forward, left in (
        (length / 2, width / 2),
        (length / 2, -width / 2),
        (-length / 2, -width / 2),
        (-length / 2, width / 2),
    ):
        result.append((x + cosine * forward - sine * left, y + sine * forward + cosine * left))
    return result


def _feature_segments(
    points: list[int | float],
    feature_start: list[int | float],
    feature_count: list[int | float],
    feature_type: list[int | float],
    feature_valid: list[int | float],
    count: int,
) -> Iterable[tuple[int, list[tuple[float, float]]]]:
    for feature in range(count):
        if not feature_valid[feature]:
            continue
        start = int(feature_start[feature])
        length = int(feature_count[feature])
        geometry = [
            (float(points[(start + index) * 3]), float(points[(start + index) * 3 + 1]))
            for index in range(length)
        ]
        if len(geometry) >= 2:
            yield int(feature_type[feature]), geometry


def render(runtime: Path, trace: Path, output: Path, world: int, fps: int, final_png: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
        from matplotlib.patches import Polygon
    except ImportError as error:
        raise RuntimeError("rendering requires matplotlib and Pillow") from error

    runtime_dir = runtime_for_world(runtime, world)
    manifest = json.loads((runtime_dir / "manifest.json").read_text(encoding="utf-8"))
    frames = load_trace(trace, world)
    dimensions = read_tensor(runtime_dir, manifest, "agent_dimensions")
    goals = read_tensor(runtime_dir, manifest, "agent_goal")
    goal_valid = read_tensor(runtime_dir, manifest, "agent_goal_valid")
    points = read_tensor(runtime_dir, manifest, "map_points")
    feature_start = read_tensor(runtime_dir, manifest, "map_feature_point_start")
    feature_count = read_tensor(runtime_dir, manifest, "map_feature_point_count")
    feature_type = read_tensor(runtime_dir, manifest, "map_feature_type")
    feature_valid = read_tensor(runtime_dir, manifest, "map_feature_valid")
    map_features = list(_feature_segments(
        points,
        feature_start,
        feature_count,
        feature_type,
        feature_valid,
        int(manifest["counts"]["map_features"]),
    ))

    valid_rows = [row for rows in frames.values() for row in rows if row.valid]
    xs = [row.x for row in valid_rows]
    ys = [row.y for row in valid_rows]
    for _, geometry in map_features:
        xs.extend(point[0] for point in geometry)
        ys.extend(point[1] for point in geometry)
    if not xs or not ys:
        raise ValueError("trace and runtime contain no visible geometry")
    padding = 10.0
    x_min, x_max = min(xs) - padding, max(xs) + padding
    y_min, y_max = min(ys) - padding, max(ys) + padding
    if x_max - x_min < 30.0:
        center = 0.5 * (x_min + x_max)
        x_min, x_max = center - 15.0, center + 15.0
    if y_max - y_min < 30.0:
        center = 0.5 * (y_min + y_max)
        y_min, y_max = center - 15.0, center + 15.0

    fig, axis = plt.subplots(figsize=(10, 8))
    step_values = list(frames)
    map_colors = {
        1: "#86a789",
        2: "#6f9273",
        3: "#d0d4d8",
        4: "#343a40",
        5: "#55a7a0",
        6: "#d14d3f",
        8: "#c8d8c2",
        9: "#b8cdb1",
    }

    def draw(frame_index: int) -> None:
        axis.clear()
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(y_min, y_max)
        axis.set_facecolor("#f5f6f4")
        axis.grid(color="#d9ddda", linewidth=0.5, alpha=0.5)
        step = step_values[frame_index]
        axis.set_title(f"CUDA nuPlan simulator | world {world} | step {step}")
        axis.set_xlabel("local x (m)")
        axis.set_ylabel("local y (m)")

        for map_type, geometry in map_features:
            axis.plot(
                [point[0] for point in geometry],
                [point[1] for point in geometry],
                color=map_colors.get(map_type, "#a7ada9"),
                linewidth=1.6 if map_type == 4 else 1.0,
                alpha=0.8,
                zorder=1,
            )

        current_rows = sorted(frames[step], key=lambda row: row.agent_slot)
        for row in current_rows:
            if not row.valid:
                continue
            trail = [
                (history_row.x, history_row.y)
                for history_step in step_values[: frame_index + 1]
                for history_row in frames[history_step]
                if history_row.valid and history_row.agent_slot == row.agent_slot
            ]
            axis.plot(
                [point[0] for point in trail],
                [point[1] for point in trail],
                color="#4f6d7a" if row.agent_slot == 0 else "#c97835",
                linewidth=1.0,
                alpha=0.45,
                zorder=2,
            )
            length = max(0.4, float(dimensions[row.agent_slot * 3]))
            width = max(0.4, float(dimensions[row.agent_slot * 3 + 1]))
            if row.collided_vehicle or row.collided_road:
                color = "#d62828"
            elif row.offroad:
                color = "#8f3bb8"
            elif row.control_mode == "external":
                color = "#ef476f"
            elif row.agent_slot == 0:
                color = "#277da1"
            else:
                color = "#f8961e"
            polygon = Polygon(
                vehicle_corners(row.x, row.y, row.yaw, length, width),
                closed=True,
                facecolor=color,
                edgecolor="#202326",
                linewidth=0.8,
                zorder=4,
            )
            axis.add_patch(polygon)
            axis.text(row.x, row.y, str(row.agent_slot), ha="center", va="center", fontsize=7, zorder=5)
            if row.agent_slot < len(goal_valid) and goal_valid[row.agent_slot]:
                axis.scatter(
                    [float(goals[row.agent_slot * 2])],
                    [float(goals[row.agent_slot * 2 + 1])],
                    marker="x",
                    color=color,
                    s=28,
                    linewidths=1.2,
                    zorder=3,
                )

    draw(0)
    animation = FuncAnimation(fig, draw, frames=len(step_values), interval=1000 / fps, repeat=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output, writer=PillowWriter(fps=fps))
    draw(len(step_values) - 1)
    final_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(final_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="output GIF")
    parser.add_argument("--final-png", type=Path, help="final frame PNG")
    parser.add_argument("--world", type=int, default=0)
    parser.add_argument("--fps", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.world < 0 or args.fps <= 0:
        raise ValueError("--world must be non-negative and --fps must be positive")
    final_png = args.final_png or args.output.with_suffix(".png")
    render(args.runtime, args.trace, args.output, args.world, args.fps, final_png)
    print(f"animation: {args.output}")
    print(f"final frame: {final_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
