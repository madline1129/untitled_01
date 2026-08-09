"""Before/after visualization for nuPlan canonical conversion."""

from __future__ import annotations

import math
import sqlite3
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .adapters.nuplan import MAP_RADIUS_METERS, convert_nuplan_database
from .models import AgentTrajectory, CanonicalScenario


@dataclass
class RawAgent:
    id: str
    type: str
    is_ego: bool
    x: list[float]
    y: list[float]
    yaw: list[float]
    timestamps: list[float]
    length: float
    width: float


def _raw_type(category: str) -> str:
    value = category.lower()
    if "vehicle" in value:
        return "vehicle"
    if "pedestrian" in value:
        return "pedestrian"
    if "bicycle" in value or "cycl" in value:
        return "cyclist"
    return "other"


def _load_raw_agents(db_path: Path, scene_id: Optional[str]) -> tuple[str, list[RawAgent]]:
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        scenes = connection.execute(
            "SELECT token, lower(hex(token)) AS id, name FROM scene ORDER BY name, id"
        ).fetchall()
        if scene_id is not None:
            scenes = [row for row in scenes if row["id"] == scene_id or row["name"] == scene_id]
        if not scenes:
            raise ValueError(f"no matching scene in {db_path}")
        scene = scenes[0]
        ego_rows = connection.execute(
            """
            SELECT pc.timestamp, ego.x, ego.y, ego.qw, ego.qx, ego.qy, ego.qz
            FROM lidar_pc AS pc
            JOIN ego_pose AS ego ON ego.token=pc.ego_pose_token
            WHERE pc.scene_token=? ORDER BY pc.timestamp
            """,
            (scene["token"],),
        ).fetchall()
        from .geometry import quaternion_to_yaw

        scene_start = ego_rows[0]["timestamp"] / 1_000_000.0

        agents = [
            RawAgent(
                "ego", "vehicle", True,
                [row["x"] for row in ego_rows],
                [row["y"] for row in ego_rows],
                [quaternion_to_yaw(row["qw"], row["qx"], row["qy"], row["qz"]) for row in ego_rows],
                [row["timestamp"] / 1_000_000.0 - scene_start for row in ego_rows],
                5.18, 2.30,
            )
        ]
        rows = connection.execute(
            """
            SELECT lower(hex(track.token)) AS id, category.name AS category,
                   pc.timestamp, box.x, box.y, box.yaw, box.length, box.width
            FROM lidar_box AS box
            JOIN lidar_pc AS pc ON pc.token=box.lidar_pc_token
            JOIN track ON track.token=box.track_token
            JOIN category ON category.token=track.category_token
            WHERE pc.scene_token=? ORDER BY id, pc.timestamp
            """,
            (scene["token"],),
        ).fetchall()
        grouped: dict[str, list[sqlite3.Row]] = defaultdict(list)
        for row in rows:
            grouped[row["id"]].append(row)
        for track_id, states in sorted(grouped.items()):
            agents.append(
                RawAgent(
                    track_id, _raw_type(states[0]["category"]), False,
                    [row["x"] for row in states],
                    [row["y"] for row in states],
                    [row["yaw"] for row in states],
                    [row["timestamp"] / 1_000_000.0 - scene_start for row in states],
                    float(states[-1]["length"]), float(states[-1]["width"]),
                )
            )
        return scene["id"], agents
    finally:
        connection.close()


def _color(agent_type: str, is_ego: bool) -> str:
    if is_ego:
        return "#1261A0"
    return {
        "vehicle": "#D1495B",
        "pedestrian": "#6A994E",
        "cyclist": "#F28E2B",
        "other": "#777777",
    }.get(agent_type, "#777777")


def _draw_box(axis: object, x: float, y: float, yaw: float, length: float, width: float, color: str) -> None:
    from matplotlib.patches import Polygon

    corners = []
    c = math.cos(yaw)
    s = math.sin(yaw)
    for forward, left in (
        (length / 2, width / 2), (length / 2, -width / 2),
        (-length / 2, -width / 2), (-length / 2, width / 2),
    ):
        corners.append((x + c * forward - s * left, y + s * forward + c * left))
    axis.add_patch(Polygon(corners, closed=True, fill=False, edgecolor=color, linewidth=1.5))


def _focus_bounds(axis: object, x_values: list[float], y_values: list[float]) -> None:
    """Keep trajectories readable while preserving equal x/y distance scale."""

    min_x, max_x = min(x_values), max(x_values)
    min_y, max_y = min(y_values), max(y_values)
    span = max(max_x - min_x, max_y - min_y, 20.0)
    padding = max(8.0, span * 0.15)
    half_extent = span / 2 + padding
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    axis.set_xlim(center_x - half_extent, center_x + half_extent)
    axis.set_ylim(center_y - half_extent, center_y + half_extent)


def _plot_raw(axis: object, agents: list[RawAgent]) -> None:
    all_x: list[float] = []
    all_y: list[float] = []
    for agent in agents:
        all_x.extend(agent.x)
        all_y.extend(agent.y)
        color = _color(agent.type, agent.is_ego)
        axis.plot(agent.x, agent.y, color=color, linewidth=2.6 if agent.is_ego else 1.8, alpha=0.9)
        axis.scatter(agent.x[0], agent.y[0], color=color, marker="o", s=35, zorder=4)
        axis.scatter(agent.x[-1], agent.y[-1], color=color, marker="x", s=42, zorder=4)
        _draw_box(axis, agent.x[0], agent.y[0], agent.yaw[0], agent.length, agent.width, color)
    _focus_bounds(axis, all_x, all_y)
    axis.text(
        0.98, 0.02, f"地图读取范围：起点周围 {MAP_RADIUS_METERS:.0f} m",
        transform=axis.transAxes, ha="right", va="bottom", color="#666666", fontsize=9,
    )
    axis.set_title("转换前：nuPlan UTM 全局坐标", fontsize=14)
    axis.set_xlabel("UTM Easting x (m)")
    axis.set_ylabel("UTM Northing y (m)")
    axis.ticklabel_format(style="plain", useOffset=False)


def _valid_values(agent: AgentTrajectory, field: str) -> list[float]:
    values = getattr(agent, field)
    return [float(value) if valid else float("nan") for value, valid in zip(values, agent.valid)]


def _plot_canonical(axis: object, scenario: CanonicalScenario) -> None:
    all_x: list[float] = []
    all_y: list[float] = []
    map_colors = {
        "lane": "#A7A7A7",
        "lane_connector": "#B8B8B8",
        "road_edge": "#404040",
        "crosswalk": "#7A9E9F",
        "stop_line": "#C44E52",
    }
    for feature in scenario.map_features:
        if len(feature.geometry) < 2:
            continue
        axis.plot(
            [point[0] for point in feature.geometry],
            [point[1] for point in feature.geometry],
            color=map_colors.get(feature.type, "#C0C0C0"),
            linewidth=1.2 if feature.type == "road_edge" else 0.8,
            alpha=0.65,
            zorder=1,
        )
    for agent in scenario.agents:
        x = _valid_values(agent, "x")
        y = _valid_values(agent, "y")
        all_x.extend(value for value in x if not math.isnan(value))
        all_y.extend(value for value in y if not math.isnan(value))
        color = _color(agent.type, agent.is_ego)
        axis.plot(x, y, color=color, linewidth=2.6 if agent.is_ego else 1.8, alpha=0.9)
        first = next(index for index, valid in enumerate(agent.valid) if valid)
        last = max(index for index, valid in enumerate(agent.valid) if valid)
        axis.scatter(x[first], y[first], color=color, marker="o", s=35, zorder=4)
        axis.scatter(x[last], y[last], color=color, marker="x", s=42, zorder=4)
        _draw_box(
            axis, x[first], y[first], float(agent.yaw[first]),
            agent.length, agent.width, color,
        )
    _focus_bounds(axis, all_x, all_y)
    axis.text(
        0.98, 0.02, "+X 为初始 ego 车头方向",
        transform=axis.transAxes, ha="right", va="bottom", color="#1261A0", fontsize=9,
    )
    axis.set_title("转换后：Canonical ego 局部坐标", fontsize=14)
    axis.set_xlabel("Local x / forward (m)")
    axis.set_ylabel("Local y / left (m)")


def _summary_text(scenario: CanonicalScenario, raw_agents: list[RawAgent]) -> str:
    counts = Counter(agent.type for agent in scenario.agents)
    return "\n".join([
        f"场景: {scenario.source.source_scenario_id}",
        f"原始帧数: {len(raw_agents[0].x)}    转换后: {scenario.timing.num_steps} 帧 @ 10 Hz",
        f"Agent: {len(scenario.agents)}    类型: {dict(counts)}",
        f"全局原点: ({scenario.coordinate_frame.origin_x:.2f}, {scenario.coordinate_frame.origin_y:.2f})",
        f"初始 yaw: {scenario.coordinate_frame.origin_yaw:.3f} rad    地图裁剪半径: {MAP_RADIUS_METERS:.0f} m",
        "标记: ○ 起点    × 终点    地图读取范围: 起点周围 150 m",
    ])


def _write_summary(path: Path, scenario: CanonicalScenario, raw_agents: list[RawAgent]) -> None:
    counts = Counter(agent.type for agent in scenario.agents)
    warnings = scenario.quality.warnings or ["无"]
    content = f"""# nuPlan 转换前后可视化摘要

- 场景 ID：`{scenario.source.source_scenario_id}`
- 原始帧数：{len(raw_agents[0].x)}
- 转换后帧数：{scenario.timing.num_steps}（10 Hz）
- Agent 数量：{len(scenario.agents)}
- Agent 类型：`{dict(counts)}`
- 初始全局位置：`({scenario.coordinate_frame.origin_x:.2f}, {scenario.coordinate_frame.origin_y:.2f})`
- 初始全局 yaw：`{scenario.coordinate_frame.origin_yaw:.3f} rad`
- 转换后 ego 初始状态：`(x=0, y=0, yaw=0)`
- 地图是否载入：`{scenario.quality.map_available}`
- 轨迹是否位于 150 米裁剪范围：`{scenario.quality.map_coverage_complete}`

## 警告

""" + "\n".join(f"- {warning}" for warning in warnings) + "\n"
    path.write_text(content, encoding="utf-8")


def visualize_nuplan_conversion(
    db_path: Path,
    output_path: Path,
    maps_root: Optional[Path] = None,
    scene_id: Optional[str] = None,
) -> tuple[Path, Path]:
    """Render raw and canonical trajectories into one comparison figure."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    raw_scene_id, raw_agents = _load_raw_agents(db_path, scene_id)
    scenarios = convert_nuplan_database(db_path, maps_root, raw_scene_id)
    scenario = scenarios[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(16, 9), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=[8.5, 1.5])
    axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])]
    summary_axis = figure.add_subplot(grid[1, :])
    _plot_raw(axes[0], raw_agents)
    _plot_canonical(axes[1], scenario)
    for axis in axes:
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, color="#DDDDDD", linewidth=0.7)
    summary_axis.axis("off")
    summary_axis.text(
        0.5, 0.5, _summary_text(scenario, raw_agents),
        ha="center", va="center", fontsize=10, linespacing=1.35,
    )
    figure.suptitle("nuPlan → CanonicalScenario 转换检查", fontsize=17)
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)

    summary_path = output_path.with_suffix(".md")
    _write_summary(summary_path, scenario, raw_agents)
    return output_path, summary_path


def _agent_polygon(agent: AgentTrajectory, step: int) -> list[tuple[float, float]]:
    x = float(agent.x[step])
    y = float(agent.y[step])
    yaw = float(agent.yaw[step])
    c = math.cos(yaw)
    s = math.sin(yaw)
    return [
        (x + c * forward - s * left, y + s * forward + c * left)
        for forward, left in (
            (agent.length / 2, agent.width / 2),
            (agent.length / 2, -agent.width / 2),
            (-agent.length / 2, -agent.width / 2),
            (-agent.length / 2, agent.width / 2),
        )
    ]


def _animation_bounds(scenario: CanonicalScenario) -> tuple[float, float, float, float]:
    x_values = [
        float(value)
        for agent in scenario.agents
        for value, valid in zip(agent.x, agent.valid)
        if valid and value is not None
    ]
    y_values = [
        float(value)
        for agent in scenario.agents
        for value, valid in zip(agent.y, agent.valid)
        if valid and value is not None
    ]
    center_x = (min(x_values) + max(x_values)) / 2
    center_y = (min(y_values) + max(y_values)) / 2
    width = max(max(x_values) - min(x_values) + 24, 50)
    height = max(max(y_values) - min(y_values) + 24, width * 9 / 16)
    return (
        center_x - width / 2,
        center_x + width / 2,
        center_y - height / 2,
        center_y + height / 2,
    )


def animate_nuplan_conversion(
    db_path: Path,
    output_path: Path,
    maps_root: Optional[Path] = None,
    scene_id: Optional[str] = None,
    fps: int = 10,
    stride: int = 1,
) -> Path:
    """Create a TerraZero-style top-down animated trajectory replay."""

    if fps <= 0 or stride <= 0:
        raise ValueError("fps and stride must be positive")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Polygon

    scenarios = convert_nuplan_database(db_path, maps_root, scene_id)
    scenario = scenarios[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(12.8, 7.2), facecolor="#17191D")
    axis.set_facecolor("#202226")
    figure.subplots_adjust(left=0.04, right=0.98, bottom=0.06, top=0.90)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(*_animation_bounds(scenario)[:2])
    axis.set_ylim(*_animation_bounds(scenario)[2:])
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#44474F")

    map_colors = {
        "lane": "#D6D8DC",
        "lane_connector": "#D6D8DC",
        "road_edge": "#6A6E76",
        "crosswalk": "#A8A8A8",
        "stop_line": "#E6C84F",
    }
    for feature in scenario.map_features:
        if len(feature.geometry) < 2:
            continue
        axis.plot(
            [point[0] for point in feature.geometry],
            [point[1] for point in feature.geometry],
            color=map_colors.get(feature.type, "#565A62"),
            linewidth=1.25 if feature.type == "road_edge" else 0.8,
            alpha=0.8,
            zorder=1,
        )

    body_artists: dict[str, object] = {}
    heading_artists: dict[str, object] = {}
    trail_artists: dict[str, object] = {}
    for agent in scenario.agents:
        valid_steps = [index for index, valid in enumerate(agent.valid) if valid]
        if not valid_steps:
            continue
        path_x = [agent.x[index] for index in valid_steps]
        path_y = [agent.y[index] for index in valid_steps]
        color = "#1677FF" if agent.is_ego else {
            "vehicle": "#FF7A12",
            "pedestrian": "#9B6AD6",
            "cyclist": "#43AA8B",
        }.get(agent.type, "#9AA0A8")
        axis.plot(path_x, path_y, color=color, linewidth=1.0, alpha=0.20, linestyle="--", zorder=2)
        first = valid_steps[0]
        if agent.type == "pedestrian":
            body = Circle((float(agent.x[first]), float(agent.y[first])), 0.65, color=color, zorder=6)
        else:
            body = Polygon(
                _agent_polygon(agent, first), closed=True, facecolor=color,
                edgecolor="#F5F7FA" if agent.is_ego else "#FFD2A8",
                linewidth=1.8 if agent.is_ego else 0.8, zorder=6,
            )
        axis.add_patch(body)
        heading, = axis.plot([], [], color="#F5F7FA", linewidth=1.1, zorder=7)
        trail, = axis.plot([], [], color=color, linewidth=2.4, alpha=0.75, zorder=4)
        body_artists[agent.id] = body
        heading_artists[agent.id] = heading
        trail_artists[agent.id] = trail

    title = axis.text(
        0.02, 1.045, "nuPlan trajectory replay", transform=axis.transAxes,
        color="#F4F5F7", fontsize=17, fontweight="bold", ha="left", va="center",
    )
    status = axis.text(
        0.98, 1.045, "", transform=axis.transAxes,
        color="#C8CBD0", fontsize=11, ha="right", va="center",
    )
    if not scenario.map_features:
        axis.text(
            0.02, 0.02, "HD map unavailable · trajectory layer only",
            transform=axis.transAxes, color="#8D929A", fontsize=9, ha="left",
        )
    axis.legend(
        handles=[
            Line2D([0], [0], color="#1677FF", linewidth=6, label="Ego"),
            Line2D([0], [0], color="#FF7A12", linewidth=6, label="Vehicle"),
            Line2D([0], [0], color="#9AA0A8", linewidth=6, label="Other"),
        ],
        loc="upper right", frameon=False, ncol=3, bbox_to_anchor=(1, 1.01),
        labelcolor="#D8DADE", fontsize=9,
    )

    frame_steps = list(range(0, scenario.timing.num_steps, stride))
    if frame_steps[-1] != scenario.timing.num_steps - 1:
        frame_steps.append(scenario.timing.num_steps - 1)

    def update(frame_number: int) -> list[object]:
        step = frame_steps[frame_number]
        changed: list[object] = [title, status]
        for agent in scenario.agents:
            body = body_artists.get(agent.id)
            if body is None:
                continue
            heading = heading_artists[agent.id]
            trail = trail_artists[agent.id]
            if not agent.valid[step]:
                body.set_visible(False)
                heading.set_data([], [])
                trail.set_data([], [])
            else:
                body.set_visible(True)
                x = float(agent.x[step])
                y = float(agent.y[step])
                yaw = float(agent.yaw[step])
                if agent.type == "pedestrian":
                    body.center = (x, y)
                else:
                    body.set_xy(_agent_polygon(agent, step))
                nose = min(agent.length * 0.38, 2.0)
                heading.set_data([x, x + math.cos(yaw) * nose], [y, y + math.sin(yaw) * nose])
                history = [
                    index for index in range(max(0, step - 24), step + 1)
                    if agent.valid[index]
                ]
                trail.set_data([agent.x[index] for index in history], [agent.y[index] for index in history])
            changed.extend([body, heading, trail])
        status.set_text(
            f"t = {scenario.timing.timestamps[step] - scenario.timing.timestamps[0]:4.1f} s"
            f"   ·   frame {step + 1}/{scenario.timing.num_steps}"
        )
        return changed

    animation = FuncAnimation(
        figure, update, frames=len(frame_steps), interval=1000 / fps, blit=False,
    )
    animation.save(output_path, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(figure)
    return output_path


def _raw_polygon(agent: RawAgent, step: int) -> list[tuple[float, float]]:
    x = agent.x[step]
    y = agent.y[step]
    yaw = agent.yaw[step]
    c = math.cos(yaw)
    s = math.sin(yaw)
    return [
        (x + c * forward - s * left, y + s * forward + c * left)
        for forward, left in (
            (agent.length / 2, agent.width / 2),
            (agent.length / 2, -agent.width / 2),
            (-agent.length / 2, -agent.width / 2),
            (-agent.length / 2, agent.width / 2),
        )
    ]


def _raw_step_at_time(agent: RawAgent, timestamp: float) -> Optional[int]:
    index = bisect_right(agent.timestamps, timestamp + 1e-8) - 1
    return index if index >= 0 else None


def _comparison_error(scenario: CanonicalScenario, raw_agents: list[RawAgent]) -> float:
    """Measure pairwise-distance preservation at original sample timestamps."""

    canonical = {agent.id: agent for agent in scenario.agents}
    raw_ego = next(agent for agent in raw_agents if agent.is_ego)
    errors: list[float] = []
    for raw_agent in raw_agents:
        target = canonical.get(raw_agent.id)
        if raw_agent.is_ego or target is None:
            continue
        for source_index, timestamp in enumerate(raw_agent.timestamps):
            ego_index = _raw_step_at_time(raw_ego, timestamp)
            canonical_index = round(timestamp / scenario.timing.dt)
            if ego_index is None or canonical_index >= scenario.timing.num_steps:
                continue
            if not target.valid[canonical_index]:
                continue
            raw_distance = math.hypot(
                raw_agent.x[source_index] - raw_ego.x[ego_index],
                raw_agent.y[source_index] - raw_ego.y[ego_index],
            )
            ego = next(agent for agent in scenario.agents if agent.is_ego)
            local_distance = math.hypot(
                float(target.x[canonical_index]) - float(ego.x[canonical_index]),
                float(target.y[canonical_index]) - float(ego.y[canonical_index]),
            )
            errors.append(abs(raw_distance - local_distance))
    return max(errors, default=0.0)


def _save_gif_halves(source: Path, before: Path, after: Path) -> None:
    """Split the synchronized comparison into independently playable GIFs."""

    from PIL import Image

    with Image.open(source) as image:
        midpoint = image.width // 2
        before_frames = []
        after_frames = []
        for index in range(image.n_frames):
            image.seek(index)
            frame = image.convert("RGB")
            before_frames.append(frame.crop((0, 0, midpoint, image.height)))
            after_frames.append(frame.crop((midpoint, 0, image.width, image.height)))
        duration = image.info.get("duration", 100)
    before_frames[0].save(
        before, save_all=True, append_images=before_frames[1:],
        duration=duration, loop=0, optimize=True,
    )
    after_frames[0].save(
        after, save_all=True, append_images=after_frames[1:],
        duration=duration, loop=0, optimize=True,
    )


def animate_nuplan_comparison(
    db_path: Path,
    output_path: Path,
    maps_root: Optional[Path] = None,
    scene_id: Optional[str] = None,
    fps: int = 10,
    stride: int = 1,
) -> tuple[Path, Path, Path]:
    """Render synchronized raw-global and canonical-local trajectory GIFs."""

    if fps <= 0 or stride <= 0:
        raise ValueError("fps and stride must be positive")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Circle, Polygon

    raw_scene_id, raw_agents = _load_raw_agents(db_path, scene_id)
    scenario = convert_nuplan_database(db_path, maps_root, raw_scene_id)[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    before_path = output_path.with_name(f"{output_path.stem}_before.gif")
    after_path = output_path.with_name(f"{output_path.stem}_after.gif")

    figure, axes = plt.subplots(1, 2, figsize=(16, 7.2), facecolor="#17191D")
    figure.subplots_adjust(left=0.035, right=0.985, bottom=0.08, top=0.88, wspace=0.10)
    raw_axis, local_axis = axes
    for axis in axes:
        axis.set_facecolor("#202226")
        axis.set_aspect("equal", adjustable="box")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#484C54")

    raw_x = [x for agent in raw_agents for x in agent.x]
    raw_y = [y for agent in raw_agents for y in agent.y]
    raw_span = max(max(raw_x) - min(raw_x), max(raw_y) - min(raw_y), 30.0)
    raw_center_x = (min(raw_x) + max(raw_x)) / 2
    raw_center_y = (min(raw_y) + max(raw_y)) / 2
    raw_axis.set_xlim(raw_center_x - raw_span * 0.68, raw_center_x + raw_span * 0.68)
    raw_axis.set_ylim(raw_center_y - raw_span * 0.58, raw_center_y + raw_span * 0.58)
    local_bounds = _animation_bounds(scenario)
    local_axis.set_xlim(*local_bounds[:2])
    local_axis.set_ylim(*local_bounds[2:])

    map_colors = {
        "lane": "#D6D8DC", "lane_connector": "#D6D8DC",
        "road_edge": "#6A6E76", "crosswalk": "#A8A8A8", "stop_line": "#E6C84F",
    }
    origin = scenario.coordinate_frame
    c = math.cos(origin.origin_yaw)
    s = math.sin(origin.origin_yaw)
    for feature in scenario.map_features:
        if len(feature.geometry) < 2:
            continue
        local_x = [point[0] for point in feature.geometry]
        local_y = [point[1] for point in feature.geometry]
        global_x = [origin.origin_x + c * x - s * y for x, y in zip(local_x, local_y)]
        global_y = [origin.origin_y + s * x + c * y for x, y in zip(local_x, local_y)]
        color = map_colors.get(feature.type, "#565A62")
        width = 1.25 if feature.type == "road_edge" else 0.8
        raw_axis.plot(global_x, global_y, color=color, linewidth=width, alpha=0.75, zorder=1)
        local_axis.plot(local_x, local_y, color=color, linewidth=width, alpha=0.75, zorder=1)

    raw_bodies: dict[str, object] = {}
    raw_trails: dict[str, object] = {}
    for agent in raw_agents:
        color = "#1677FF" if agent.is_ego else "#FF7A12" if agent.type == "vehicle" else "#9AA0A8"
        raw_axis.plot(agent.x, agent.y, color=color, linewidth=1.0, alpha=0.22, linestyle="--")
        if agent.type == "pedestrian":
            body = Circle((agent.x[0], agent.y[0]), 0.65, color=color, zorder=6)
        else:
            body = Polygon(
                _raw_polygon(agent, 0), closed=True, facecolor=color,
                edgecolor="#F5F7FA" if agent.is_ego else "#FFD2A8",
                linewidth=1.8 if agent.is_ego else 0.8, zorder=6,
            )
        raw_axis.add_patch(body)
        trail, = raw_axis.plot([], [], color=color, linewidth=2.5, alpha=0.8, zorder=4)
        raw_bodies[agent.id] = body
        raw_trails[agent.id] = trail

    local_bodies: dict[str, object] = {}
    local_trails: dict[str, object] = {}
    for agent in scenario.agents:
        valid_steps = [index for index, valid in enumerate(agent.valid) if valid]
        if not valid_steps:
            continue
        color = "#1677FF" if agent.is_ego else "#FF7A12" if agent.type == "vehicle" else "#9AA0A8"
        local_axis.plot(
            [agent.x[index] for index in valid_steps], [agent.y[index] for index in valid_steps],
            color=color, linewidth=1.0, alpha=0.22, linestyle="--",
        )
        first = valid_steps[0]
        if agent.type == "pedestrian":
            body = Circle((float(agent.x[first]), float(agent.y[first])), 0.65, color=color, zorder=6)
        else:
            body = Polygon(
                _agent_polygon(agent, first), closed=True, facecolor=color,
                edgecolor="#F5F7FA" if agent.is_ego else "#FFD2A8",
                linewidth=1.8 if agent.is_ego else 0.8, zorder=6,
            )
        local_axis.add_patch(body)
        trail, = local_axis.plot([], [], color=color, linewidth=2.5, alpha=0.8, zorder=4)
        local_bodies[agent.id] = body
        local_trails[agent.id] = trail

    raw_axis.set_title("BEFORE · nuPlan UTM global coordinates", color="#F4F5F7", fontsize=14, pad=14)
    local_axis.set_title("AFTER · Canonical ego-local coordinates", color="#F4F5F7", fontsize=14, pad=14)
    raw_status = raw_axis.text(0.02, 0.02, "", transform=raw_axis.transAxes, color="#C8CBD0", fontsize=9)
    local_status = local_axis.text(0.02, 0.02, "", transform=local_axis.transAxes, color="#C8CBD0", fontsize=9)
    max_error = _comparison_error(scenario, raw_agents)
    figure.suptitle(
        f"Synchronized conversion check · pairwise-distance max error {max_error:.6f} m",
        color="#F4F5F7", fontsize=16, fontweight="bold", y=0.97,
    )

    frame_steps = list(range(0, scenario.timing.num_steps, stride))
    if frame_steps[-1] != scenario.timing.num_steps - 1:
        frame_steps.append(scenario.timing.num_steps - 1)

    def update(frame_number: int) -> list[object]:
        canonical_step = frame_steps[frame_number]
        timestamp = scenario.timing.timestamps[canonical_step] - scenario.timing.timestamps[0]
        changed: list[object] = [raw_status, local_status]
        raw_frame_number = 0
        for agent in raw_agents:
            source_step = _raw_step_at_time(agent, timestamp)
            body = raw_bodies[agent.id]
            trail = raw_trails[agent.id]
            if source_step is None:
                body.set_visible(False)
                trail.set_data([], [])
            else:
                raw_frame_number = max(raw_frame_number, source_step)
                body.set_visible(True)
                if agent.type == "pedestrian":
                    body.center = (agent.x[source_step], agent.y[source_step])
                else:
                    body.set_xy(_raw_polygon(agent, source_step))
                start = max(0, source_step - 5)
                trail.set_data(agent.x[start:source_step + 1], agent.y[start:source_step + 1])
            changed.extend([body, trail])

        for agent in scenario.agents:
            body = local_bodies.get(agent.id)
            if body is None:
                continue
            trail = local_trails[agent.id]
            if not agent.valid[canonical_step]:
                body.set_visible(False)
                trail.set_data([], [])
            else:
                body.set_visible(True)
                if agent.type == "pedestrian":
                    body.center = (float(agent.x[canonical_step]), float(agent.y[canonical_step]))
                else:
                    body.set_xy(_agent_polygon(agent, canonical_step))
                history = [
                    index for index in range(max(0, canonical_step - 24), canonical_step + 1)
                    if agent.valid[index]
                ]
                trail.set_data([agent.x[index] for index in history], [agent.y[index] for index in history])
            changed.extend([body, trail])

        raw_status.set_text(
            f"t={timestamp:4.1f}s  ·  source frame {raw_frame_number + 1}/{len(raw_agents[0].timestamps)}"
            "  ·  sample-and-hold"
        )
        local_status.set_text(
            f"t={timestamp:4.1f}s  ·  canonical frame {canonical_step + 1}/{scenario.timing.num_steps}"
            "  ·  10 Hz resampled"
        )
        return changed

    animation = FuncAnimation(
        figure, update, frames=len(frame_steps), interval=1000 / fps, blit=False,
    )
    animation.save(output_path, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(figure)
    _save_gif_halves(output_path, before_path, after_path)
    return before_path, after_path, output_path
