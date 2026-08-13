#!/usr/bin/env python3

"""Render a CUDA simulator trace as a TerraZero-style top-down demo."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import sys
from array import array
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path


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


@dataclass(frozen=True)
class MapFeature:
    feature_type: int
    geometry_type: int
    geometry: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class RenderConfig:
    world: int = 0
    duration: float = 10.0
    gif_fps: int = 10
    mp4_fps: int = 20
    view: str = "follow"
    show_goals: bool = False
    show_trails: bool = False
    viewport_width: float = 160.0
    viewport_height: float = 90.0


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
    return {
        step: sorted(rows, key=lambda row: row.agent_slot)
        for step, rows in sorted(frames.items())
    }


def vehicle_corners(
    x: float,
    y: float,
    yaw: float,
    length: float,
    width: float,
) -> list[tuple[float, float]]:
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


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def interpolate_yaw(start: float, end: float, alpha: float) -> float:
    return wrap_angle(start + wrap_angle(end - start) * alpha)


def interpolate_trace_frame(
    frames: dict[int, list[TraceRow]],
    step_position: float,
) -> list[TraceRow]:
    steps = list(frames)
    if step_position < steps[0] - 1e-6 or step_position > steps[-1] + 1e-6:
        raise ValueError(
            f"requested trace step {step_position:.3f} is outside [{steps[0]}, {steps[-1]}]"
        )
    upper_index = bisect.bisect_left(steps, step_position)
    if upper_index == 0:
        return frames[steps[0]]
    if upper_index == len(steps):
        return frames[steps[-1]]
    upper_step = steps[upper_index]
    if math.isclose(step_position, upper_step, abs_tol=1e-6):
        return frames[upper_step]
    lower_step = steps[upper_index - 1]
    alpha = (step_position - lower_step) / (upper_step - lower_step)
    lower = {row.agent_slot: row for row in frames[lower_step]}
    upper = {row.agent_slot: row for row in frames[upper_step]}
    nearest = upper if alpha >= 0.5 else lower
    result: list[TraceRow] = []
    for slot in sorted(lower.keys() | upper.keys()):
        left = lower.get(slot)
        right = upper.get(slot)
        state = nearest.get(slot) or left or right
        if left is None or right is None or state is None:
            if state is not None:
                result.append(state)
            continue
        result.append(TraceRow(
            world=state.world,
            step=round(step_position),
            agent_slot=slot,
            agent_id=state.agent_id,
            valid=state.valid,
            control_mode=state.control_mode,
            x=left.x + (right.x - left.x) * alpha,
            y=left.y + (right.y - left.y) * alpha,
            yaw=interpolate_yaw(left.yaw, right.yaw, alpha),
            vx=left.vx + (right.vx - left.vx) * alpha,
            vy=left.vy + (right.vy - left.vy) * alpha,
            collided_vehicle=state.collided_vehicle,
            collided_road=state.collided_road,
            offroad=state.offroad,
            reached_goal=state.reached_goal,
        ))
    return result


def sample_times(duration: float, fps: int) -> list[float]:
    if duration <= 0.0 or fps <= 0:
        raise ValueError("duration and fps must be positive")
    frame_count = max(1, round(duration * fps))
    return [index / fps for index in range(frame_count)]


def _feature_segments(
    points: list[int | float],
    feature_start: list[int | float],
    feature_count: list[int | float],
    feature_type: list[int | float],
    geometry_type: list[int | float],
    feature_valid: list[int | float],
    count: int,
) -> Iterable[MapFeature]:
    for feature in range(count):
        if not feature_valid[feature]:
            continue
        start = int(feature_start[feature])
        length = int(feature_count[feature])
        geometry = tuple(
            (float(points[(start + index) * 3]), float(points[(start + index) * 3 + 1]))
            for index in range(length)
        )
        if len(geometry) >= 2:
            yield MapFeature(
                int(feature_type[feature]),
                int(geometry_type[feature]),
                geometry,
            )


def _enum_value(manifest: dict, group: str, name: str, fallback: int) -> int:
    return int(manifest.get("enums", {}).get(group, {}).get(name, fallback))


def _event_edge(row: TraceRow) -> tuple[str, float]:
    if row.collided_vehicle or row.collided_road:
        return "#ff334f", 2.4
    if row.offroad:
        return "#bd6cff", 2.2
    if row.reached_goal:
        return "#55e68a", 2.2
    if row.control_mode == "external":
        return "#f451c5", 1.8
    return "#e8edf2", 0.8


def _agent_fill(agent_type: int, is_ego: bool, types: dict[str, int]) -> str:
    if is_ego:
        return "#087cf0"
    if agent_type == types["vehicle"]:
        return "#ff7a12"
    if agent_type == types["pedestrian"]:
        return "#a56de2"
    if agent_type == types["cyclist"]:
        return "#32d583"
    return "#7f8794"


def _full_map_bounds(
    map_features: Sequence[MapFeature],
    frames: dict[int, list[TraceRow]],
) -> tuple[float, float, float, float]:
    xs = [point[0] for feature in map_features for point in feature.geometry]
    ys = [point[1] for feature in map_features for point in feature.geometry]
    xs.extend(row.x for rows in frames.values() for row in rows if row.valid)
    ys.extend(row.y for rows in frames.values() for row in rows if row.valid)
    if not xs or not ys:
        raise ValueError("trace and runtime contain no visible geometry")
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    width = max(30.0, x_max - x_min)
    height = max(30.0, y_max - y_min)
    center_x = 0.5 * (x_min + x_max)
    center_y = 0.5 * (y_min + y_max)
    # 保持16:9画幅，避免保存时出现额外留白。
    target_ratio = 16.0 / 9.0
    if width / height < target_ratio:
        width = height * target_ratio
    else:
        height = width / target_ratio
    return (
        center_x - width * 0.55,
        center_x + width * 0.55,
        center_y - height * 0.55,
        center_y + height * 0.55,
    )


def render(
    runtime: Path,
    trace: Path,
    output: Path,
    world: int,
    fps: int,
    final_png: Path,
    *,
    mp4_output: Path | None = None,
    mp4_fps: int = 20,
    duration: float = 10.0,
    view: str = "follow",
    show_goals: bool = False,
    show_trails: bool = False,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle, Polygon
    except ImportError as error:
        raise RuntimeError("rendering requires matplotlib and Pillow") from error

    config = RenderConfig(
        world=world,
        duration=duration,
        gif_fps=fps,
        mp4_fps=mp4_fps,
        view=view,
        show_goals=show_goals,
        show_trails=show_trails,
    )
    runtime_dir = runtime_for_world(runtime, world)
    manifest = json.loads((runtime_dir / "manifest.json").read_text(encoding="utf-8"))
    frames = load_trace(trace, world)
    dt = float(manifest["dt"])
    required_step = config.duration / dt
    if required_step > max(frames) + 1e-6:
        available = max(frames) * dt
        raise ValueError(
            f"trace covers {available:.3f}s but --duration requests {config.duration:.3f}s"
        )

    dimensions = read_tensor(runtime_dir, manifest, "agent_dimensions")
    agent_types = read_tensor(runtime_dir, manifest, "agent_type")
    ego_mask = read_tensor(runtime_dir, manifest, "agent_is_ego")
    goals = read_tensor(runtime_dir, manifest, "agent_goal")
    goal_valid = read_tensor(runtime_dir, manifest, "agent_goal_valid")
    points = read_tensor(runtime_dir, manifest, "map_points")
    feature_start = read_tensor(runtime_dir, manifest, "map_feature_point_start")
    feature_count = read_tensor(runtime_dir, manifest, "map_feature_point_count")
    feature_type = read_tensor(runtime_dir, manifest, "map_feature_type")
    geometry_type = read_tensor(runtime_dir, manifest, "map_geometry_type")
    feature_valid = read_tensor(runtime_dir, manifest, "map_feature_valid")
    light_feature_index = read_tensor(runtime_dir, manifest, "traffic_light_feature_index")
    light_states = read_tensor(runtime_dir, manifest, "traffic_light_state")
    light_valid = read_tensor(runtime_dir, manifest, "traffic_light_valid")
    map_features = list(_feature_segments(
        points,
        feature_start,
        feature_count,
        feature_type,
        geometry_type,
        feature_valid,
        int(manifest["counts"]["map_features"]),
    ))
    max_agents = int(manifest["capacities"]["max_agents"])
    max_lights = int(manifest["capacities"]["max_traffic_lights"])
    max_future = int(manifest["capacities"]["max_future_steps"])
    ego_slots = [index for index, value in enumerate(ego_mask) if value]
    ego_slot = ego_slots[0] if ego_slots else 0
    agent_type_values = {
        name: _enum_value(manifest, "agent_type", name, fallback)
        for name, fallback in (("vehicle", 1), ("pedestrian", 2), ("cyclist", 3))
    }
    map_type_values = {
        name: _enum_value(manifest, "map_type", name, fallback)
        for name, fallback in (
            ("lane", 1), ("lane_connector", 2), ("road_line", 3),
            ("road_edge", 4), ("crosswalk", 5), ("stop_line", 6),
            ("walkway", 7), ("roadblock", 8), ("roadblock_connector", 9),
        )
    }

    fig, axis = plt.subplots(figsize=(16, 9), dpi=100)
    fig.patch.set_facecolor("#17181b")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    axis.set_facecolor("#17181b")
    axis.set_aspect("equal", adjustable="box")
    axis.set_axis_off()

    # 静态地图只创建一次，动画阶段仅更新车辆和镜头。
    polygon_types = {
        map_type_values["roadblock"],
        map_type_values["roadblock_connector"],
        map_type_values["crosswalk"],
        map_type_values["walkway"],
    }
    for feature in map_features:
        if feature.feature_type in polygon_types and len(feature.geometry) >= 3:
            if feature.feature_type in {
                map_type_values["roadblock"], map_type_values["roadblock_connector"]
            }:
                face, zorder, alpha = "#303236", 0, 1.0
            elif feature.feature_type == map_type_values["crosswalk"]:
                face, zorder, alpha = "#aeb2b7", 1, 0.5
            else:
                face, zorder, alpha = "#4b4e52", 1, 0.8
            axis.add_patch(Polygon(
                feature.geometry,
                closed=True,
                facecolor=face,
                edgecolor="none",
                alpha=alpha,
                zorder=zorder,
            ))

    line_styles = {
        map_type_values["lane"]: ("#44474c", 0.7, "-", 2),
        map_type_values["lane_connector"]: ("#4a4d52", 0.7, "-", 2),
        map_type_values["road_line"]: ("#e2e4e7", 1.6, (0, (7, 7)), 3),
        map_type_values["road_edge"]: ("#b5a53a", 1.0, "-", 3),
        map_type_values["stop_line"]: ("#f0f1f2", 3.0, "-", 4),
        map_type_values["crosswalk"]: ("#d7dade", 1.0, "-", 3),
        map_type_values["walkway"]: ("#777c82", 0.8, "-", 2),
    }
    for feature in map_features:
        style = line_styles.get(feature.feature_type)
        if style is None:
            continue
        color, linewidth, linestyle, zorder = style
        axis.plot(
            [point[0] for point in feature.geometry],
            [point[1] for point in feature.geometry],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=0.9,
            solid_capstyle="round",
            zorder=zorder,
        )

    body_artists: list[Polygon | Circle] = []
    heading_artists: list[Line2D] = []
    trail_artists: list[Line2D] = []
    goal_artists: list[Line2D] = []
    for slot in range(max_agents):
        agent_type = int(agent_types[slot])
        fill = _agent_fill(agent_type, bool(ego_mask[slot]), agent_type_values)
        if agent_type == agent_type_values["pedestrian"]:
            body = Circle((0.0, 0.0), radius=0.55, facecolor=fill, edgecolor="#e8edf2")
        else:
            body = Polygon(
                [(0.0, 0.0)] * 4,
                closed=True,
                facecolor=fill,
                edgecolor="#e8edf2",
            )
        body.set_visible(False)
        body.set_zorder(10 if ego_mask[slot] else 8)
        axis.add_patch(body)
        body_artists.append(body)
        heading, = axis.plot([], [], color="#f7f9fb", linewidth=1.1, zorder=11)
        heading.set_visible(False)
        heading_artists.append(heading)
        trail, = axis.plot([], [], color=fill, linewidth=1.3, alpha=0.45, zorder=6)
        trail.set_visible(False)
        trail_artists.append(trail)
        goal, = axis.plot(
            [], [], marker="*", markersize=9, markerfacecolor=fill,
            markeredgecolor="#f7f9fb", markeredgewidth=0.7, linestyle="none", zorder=7,
        )
        goal.set_visible(False)
        goal_artists.append(goal)

    signal_artists: list[Circle] = []
    signal_positions: list[tuple[float, float] | None] = []
    for light in range(max_lights):
        feature_index = int(light_feature_index[light])
        position = None
        if 0 <= feature_index < len(map_features) and map_features[feature_index].geometry:
            position = map_features[feature_index].geometry[-1]
        signal_positions.append(position)
        marker = Circle((0.0, 0.0), radius=1.0, facecolor="#6f7680", edgecolor="#f4f5f6")
        marker.set_visible(False)
        marker.set_linewidth(0.7)
        marker.set_zorder(12)
        axis.add_patch(marker)
        signal_artists.append(marker)

    source = manifest.get("source", {})
    scenario_name = str(source.get("source_scenario_id", runtime_dir.name))
    dataset_name = str(source.get("dataset", "runtime")).upper()
    hud = axis.text(
        0.018,
        0.965,
        "",
        transform=axis.transAxes,
        ha="left",
        va="top",
        color="#f4f5f6",
        family="monospace",
        fontsize=11,
        linespacing=1.35,
        bbox={"facecolor": "#0b0c0e", "edgecolor": "none", "alpha": 0.78, "pad": 5},
        zorder=20,
    )
    full_bounds = _full_map_bounds(map_features, frames)
    row_history: dict[int, list[TraceRow]] = defaultdict(list)
    for step_rows in frames.values():
        for row in step_rows:
            if row.valid:
                row_history[row.agent_slot].append(row)

    def update(time_seconds: float) -> list[object]:
        step_position = time_seconds / dt
        current_rows = interpolate_trace_frame(frames, step_position)
        row_by_slot = {row.agent_slot: row for row in current_rows}
        ego = row_by_slot.get(ego_slot)
        if config.view == "follow" and ego is not None and ego.valid:
            axis.set_xlim(
                ego.x - config.viewport_width / 2.0,
                ego.x + config.viewport_width / 2.0,
            )
            axis.set_ylim(
                ego.y - config.viewport_height / 2.0,
                ego.y + config.viewport_height / 2.0,
            )
        else:
            axis.set_xlim(full_bounds[0], full_bounds[1])
            axis.set_ylim(full_bounds[2], full_bounds[3])

        for slot in range(max_agents):
            row = row_by_slot.get(slot)
            body = body_artists[slot]
            heading = heading_artists[slot]
            trail = trail_artists[slot]
            goal = goal_artists[slot]
            if row is None or not row.valid:
                body.set_visible(False)
                heading.set_visible(False)
                trail.set_visible(False)
                goal.set_visible(False)
                continue
            length = max(0.8, float(dimensions[slot * 3]))
            width = max(0.6, float(dimensions[slot * 3 + 1]))
            if isinstance(body, Circle):
                body.center = (row.x, row.y)
                body.set_radius(max(0.45, min(0.8, 0.35 * width)))
            else:
                body.set_xy(vehicle_corners(row.x, row.y, row.yaw, length, width))
            edge_color, edge_width = _event_edge(row)
            body.set_edgecolor(edge_color)
            body.set_linewidth(edge_width)
            body.set_visible(True)
            front = 0.35 * length
            heading.set_data(
                [row.x, row.x + math.cos(row.yaw) * front],
                [row.y, row.y + math.sin(row.yaw) * front],
            )
            heading.set_visible(int(agent_types[slot]) != agent_type_values["pedestrian"])
            if config.show_trails:
                history = [item for item in row_history[slot] if item.step <= step_position]
                trail.set_data([item.x for item in history], [item.y for item in history])
                trail.set_visible(bool(history))
            else:
                trail.set_visible(False)
            if config.show_goals and slot < len(goal_valid) and goal_valid[slot]:
                goal.set_data([float(goals[slot * 2])], [float(goals[slot * 2 + 1])])
                goal.set_visible(True)
            else:
                goal.set_visible(False)

        signal_step = min(max_future, max(0, round(step_position)))
        light_colors = {0: "#77808b", 1: "#ff3b4e", 2: "#ffca3a", 3: "#2bdb70"}
        for light, marker in enumerate(signal_artists):
            position = signal_positions[light]
            index = signal_step * max_lights + light
            if position is None or index >= len(light_valid) or not light_valid[index]:
                marker.set_visible(False)
                continue
            marker.center = position
            marker.set_facecolor(light_colors.get(int(light_states[index]), "#77808b"))
            marker.set_visible(True)

        speed = math.hypot(ego.vx, ego.vy) if ego is not None and ego.valid else 0.0
        hud.set_text(
            f"{dataset_name}  |  scene {scenario_name}\n"
            f"t={time_seconds:04.1f}s  ego={speed:4.1f} m/s  world={world}"
        )
        return [
            *body_artists,
            *heading_artists,
            *trail_artists,
            *goal_artists,
            *signal_artists,
            hud,
        ]

    output.parent.mkdir(parents=True, exist_ok=True)
    gif_animation = FuncAnimation(
        fig,
        update,
        frames=sample_times(config.duration, config.gif_fps),
        interval=1000 / config.gif_fps,
        repeat=False,
        blit=False,
    )
    gif_animation.save(output, writer=PillowWriter(fps=config.gif_fps), dpi=100)

    if mp4_output is not None:
        try:
            import imageio_ffmpeg
        except ImportError as error:
            raise RuntimeError("MP4 rendering requires imageio-ffmpeg") from error
        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        mp4_output.parent.mkdir(parents=True, exist_ok=True)
        mp4_animation = FuncAnimation(
            fig,
            update,
            frames=sample_times(config.duration, config.mp4_fps),
            interval=1000 / config.mp4_fps,
            repeat=False,
            blit=False,
        )
        writer = FFMpegWriter(
            fps=config.mp4_fps,
            codec="h264",
            bitrate=5000,
            extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )
        mp4_animation.save(mp4_output, writer=writer, dpi=100)

    update(config.duration)
    final_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(final_png, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="output GIF")
    parser.add_argument("--mp4-output", type=Path, help="optional H.264 MP4 output")
    parser.add_argument("--final-png", type=Path, help="final frame PNG")
    parser.add_argument("--world", type=int, default=0)
    parser.add_argument("--fps", type=int, default=10, help="GIF frames per second")
    parser.add_argument("--mp4-fps", type=int, default=20)
    parser.add_argument("--duration", type=float, default=10.0, help="demo duration in seconds")
    parser.add_argument("--view", choices=("follow", "full-map"), default="follow")
    parser.add_argument("--show-goals", action="store_true")
    parser.add_argument("--show-trails", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.world < 0 or args.fps <= 0 or args.mp4_fps <= 0 or args.duration <= 0.0:
        raise ValueError("world must be non-negative; fps and duration must be positive")
    final_png = args.final_png or args.output.with_suffix(".png")
    render(
        args.runtime,
        args.trace,
        args.output,
        args.world,
        args.fps,
        final_png,
        mp4_output=args.mp4_output,
        mp4_fps=args.mp4_fps,
        duration=args.duration,
        view=args.view,
        show_goals=args.show_goals,
        show_trails=args.show_trails,
    )
    print(f"animation GIF: {args.output}")
    if args.mp4_output is not None:
        print(f"animation MP4: {args.mp4_output}")
    print(f"final frame: {final_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
