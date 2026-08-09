"""Convert nuPlan SQLite scenes and optional HD maps to CanonicalScenario v1."""

from __future__ import annotations

import math
import re
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

from ..geometry import LocalFrame, quaternion_to_yaw
from ..models import (
    AgentTrajectory,
    CanonicalScenario,
    CoordinateFrame,
    Goal,
    MapFeature,
    QualityInfo,
    RouteInfo,
    SourceInfo,
    TimingInfo,
    TrafficLight,
)
from ..resample import Sample, TARGET_DT, build_timeline, resample_discrete_states, resample_samples
from ..validation import validate_scenario


MAP_RADIUS_METERS = 150.0
EGO_DIMENSIONS = (5.18, 2.30, 1.78)


def _token_text(value: bytes) -> str:
    return value.hex().lower()


def _split_ids(value: Optional[str]) -> list[str]:
    if not value:
        return []
    return [item for item in re.split(r"[\s,]+", value.strip()) if item]


def _agent_type(category: str) -> str:
    value = category.lower()
    if "vehicle" in value:
        return "vehicle"
    if "pedestrian" in value:
        return "pedestrian"
    if "bicycle" in value or "cycl" in value:
        return "cyclist"
    return "other"


def _light_state(value: str) -> str:
    normalized = value.lower()
    if normalized in {"green", "go", "arrow_go"}:
        return "go"
    if normalized in {"yellow", "caution", "arrow_caution"}:
        return "caution"
    if normalized in {"red", "stop", "arrow_stop"}:
        return "stop"
    return "unknown"


def _median_gap(times: list[float]) -> Optional[float]:
    gaps = [right - left for left, right in zip(times, times[1:]) if right > left]
    return statistics.median(gaps) if gaps else None


def _last_valid_goal(series: dict[str, list[object]], source: str) -> Goal:
    for index in range(len(series["valid"]) - 1, -1, -1):
        if series["valid"][index]:
            return Goal(
                available=True,
                x=float(series["x"][index]),
                y=float(series["y"][index]),
                source=source,
            )
    return Goal(available=False)


def _safe_attr(value: object, name: str, default: Any = None) -> Any:
    try:
        return getattr(value, name)
    except (AttributeError, RuntimeError, ValueError, NotImplementedError, KeyError, AssertionError):
        return default


def _edge_ids(edges: Optional[Iterable[object]]) -> list[str]:
    if edges is None:
        return []
    return [str(_safe_attr(edge, "id")) for edge in edges if _safe_attr(edge, "id") is not None]


def _geometry_from_map_object(map_object: object) -> tuple[str, list[list[float]]]:
    baseline = _safe_attr(map_object, "baseline_path")
    path = _safe_attr(baseline, "discrete_path") if baseline is not None else None
    if path is None:
        path = _safe_attr(map_object, "discrete_path")
    if path:
        return "polyline", [[float(point.x), float(point.y), 0.0] for point in path]

    linestring = _safe_attr(map_object, "linestring")
    if linestring is not None:
        return "polyline", [[float(x), float(y), 0.0] for x, y, *_ in linestring.coords]

    polygon = _safe_attr(map_object, "polygon")
    if polygon is not None:
        return "polygon", [[float(x), float(y), 0.0] for x, y, *_ in polygon.exterior.coords]
    return "point", []


def _map_subtype(map_object: object) -> Optional[str]:
    for name in ("turn_type", "stop_line_type", "intersection_type"):
        value = _safe_attr(map_object, name)
        if callable(value):
            try:
                value = value()
            except (RuntimeError, ValueError, NotImplementedError):
                value = None
        if value is not None:
            return getattr(value, "name", str(value)).lower()
    return None


def _load_map_features(
    maps_root: Path,
    map_version: str,
    map_name: str,
    origin_x: float,
    origin_y: float,
    local_frame: LocalFrame,
) -> list[MapFeature]:
    """Load nuPlan vector map objects through the official map API."""

    try:
        from nuplan.common.actor_state.state_representation import Point2D
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer
        from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
    except ImportError as error:
        raise RuntimeError("nuPlan devkit is required for map conversion") from error

    map_api = get_maps_api(str(maps_root), map_version, map_name)
    layer_types = {
        SemanticMapLayer.LANE: "lane",
        SemanticMapLayer.LANE_CONNECTOR: "lane_connector",
        SemanticMapLayer.BOUNDARIES: "road_edge",
        SemanticMapLayer.CROSSWALK: "crosswalk",
        SemanticMapLayer.STOP_LINE: "stop_line",
        SemanticMapLayer.WALKWAYS: "walkway",
        SemanticMapLayer.ROADBLOCK: "roadblock",
        SemanticMapLayer.ROADBLOCK_CONNECTOR: "roadblock_connector",
    }
    available = set(map_api.get_available_map_objects())
    layers = [layer for layer in layer_types if layer in available]
    proximal = map_api.get_proximal_map_objects(
        Point2D(origin_x, origin_y),
        MAP_RADIUS_METERS,
        layers,
    )

    features: list[MapFeature] = []
    seen: set[tuple[str, str]] = set()
    for layer in layers:
        for map_object in proximal.get(layer, []):
            feature_id = str(map_object.id)
            feature_type = layer_types[layer]
            identity = (feature_type, feature_id)
            if identity in seen:
                continue
            seen.add(identity)
            geometry_type, global_geometry = _geometry_from_map_object(map_object)
            geometry = [list(local_frame.point(*point)) for point in global_geometry]
            incoming = _safe_attr(map_object, "incoming_edges")
            outgoing = _safe_attr(map_object, "outgoing_edges")
            adjacent = _safe_attr(map_object, "adjacent_edges")
            left_neighbors: list[str] = []
            right_neighbors: list[str] = []
            if adjacent:
                left, right = adjacent
                left_neighbors = _edge_ids([left] if left is not None else [])
                right_neighbors = _edge_ids([right] if right is not None else [])
            speed_limit = _safe_attr(map_object, "speed_limit_mps")
            features.append(
                MapFeature(
                    id=feature_id,
                    type=feature_type,
                    geometry_type=geometry_type,
                    geometry=geometry,
                    subtype=_map_subtype(map_object),
                    speed_limit_mps=float(speed_limit) if speed_limit is not None else None,
                    predecessor_ids=_edge_ids(incoming),
                    successor_ids=_edge_ids(outgoing),
                    left_neighbor_ids=left_neighbors,
                    right_neighbor_ids=right_neighbors,
                    source_type=layer.name.lower(),
                )
            )

            # nuPlan exposes lane boundaries through lane objects rather than
            # get_available_map_objects(). Preserve them as road-line geometry.
            if feature_type in {"lane", "lane_connector"}:
                for side in ("left_boundary", "right_boundary"):
                    boundary = _safe_attr(map_object, side)
                    boundary_id = _safe_attr(boundary, "id") if boundary is not None else None
                    boundary_identity = ("road_line", str(boundary_id))
                    if boundary_id is None or boundary_identity in seen:
                        continue
                    boundary_geometry_type, boundary_global = _geometry_from_map_object(boundary)
                    if not boundary_global:
                        continue
                    seen.add(boundary_identity)
                    features.append(
                        MapFeature(
                            id=str(boundary_id),
                            type="road_line",
                            geometry_type=boundary_geometry_type,
                            geometry=[list(local_frame.point(*point)) for point in boundary_global],
                            subtype=side,
                            source_type="boundary",
                        )
                    )

            # A roadblock polygon is the best available nuPlan representation
            # of the local drivable road boundary.
            if feature_type in {"roadblock", "roadblock_connector"} and global_geometry:
                edge_id = f"{feature_type}:{feature_id}:edge"
                edge_identity = ("road_edge", edge_id)
                if edge_identity not in seen:
                    seen.add(edge_identity)
                    features.append(
                        MapFeature(
                            id=edge_id,
                            type="road_edge",
                            geometry_type="polyline",
                            geometry=[list(local_frame.point(*point)) for point in global_geometry],
                            subtype="drivable_boundary",
                            source_type=f"{layer.name.lower()}_exterior",
                        )
                    )
    return features


def _scene_rows(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    return connection.execute(
        """
        SELECT
            scene.token AS scene_token,
            lower(hex(scene.token)) AS scene_id,
            scene.name,
            scene.roadblock_ids,
            scene.goal_ego_pose_token,
            log.location AS map_name,
            log.map_version
        FROM scene
        JOIN log ON log.token = scene.log_token
        ORDER BY scene.name, scene_id
        """
    ).fetchall()


def _convert_scene(
    connection: sqlite3.Connection,
    scene: sqlite3.Row,
    source_file: Path,
    maps_root: Optional[Path],
) -> CanonicalScenario:
    frame_rows = connection.execute(
        """
        SELECT
            pc.token AS frame_token,
            pc.timestamp,
            ego.x, ego.y, ego.z,
            ego.qw, ego.qx, ego.qy, ego.qz,
            ego.vx, ego.vy,
            ego.epsg
        FROM lidar_pc AS pc
        JOIN ego_pose AS ego ON ego.token = pc.ego_pose_token
        WHERE pc.scene_token = ?
        ORDER BY pc.timestamp
        """,
        (scene["scene_token"],),
    ).fetchall()
    if not frame_rows:
        raise ValueError(f"nuPlan scene {scene['scene_id']} has no frames")

    source_times = [row["timestamp"] / 1_000_000.0 for row in frame_rows]
    start_time = source_times[0]
    relative_times = [timestamp - start_time for timestamp in source_times]
    timeline = build_timeline(0.0, relative_times[-1], TARGET_DT)
    source_gap = _median_gap(relative_times)
    max_gap = source_gap * 1.5 if source_gap is not None else None

    first = frame_rows[0]
    origin_yaw = quaternion_to_yaw(first["qw"], first["qx"], first["qy"], first["qz"])
    local_frame = LocalFrame(first["x"], first["y"], first["z"], origin_yaw)

    ego_samples: list[Sample] = []
    frame_times: dict[bytes, float] = {}
    for row, timestamp in zip(frame_rows, relative_times):
        frame_times[row["frame_token"]] = timestamp
        x, y, _ = local_frame.point(row["x"], row["y"], row["z"])
        vx, vy = local_frame.vector(row["vx"], row["vy"])
        ego_samples.append(Sample(timestamp, x, y, local_frame.yaw(
            quaternion_to_yaw(row["qw"], row["qx"], row["qy"], row["qz"])
        ), vx, vy))
    ego_series = resample_samples(ego_samples, timeline, max_gap)
    ego_length, ego_width, ego_height = EGO_DIMENSIONS
    agents = [
        AgentTrajectory(
            id="ego",
            type="vehicle",
            is_ego=True,
            length=ego_length,
            width=ego_width,
            height=ego_height,
            x=ego_series["x"],
            y=ego_series["y"],
            yaw=ego_series["yaw"],
            vx=ego_series["vx"],
            vy=ego_series["vy"],
            valid=ego_series["valid"],
            goal=_last_valid_goal(ego_series, "logged_terminal_position"),
            roles={"is_sdc": True},
        )
    ]

    object_rows = connection.execute(
        """
        SELECT
            box.lidar_pc_token,
            lower(hex(track.token)) AS track_id,
            category.name AS category,
            box.x, box.y, box.z, box.yaw, box.vx, box.vy,
            box.length, box.width, box.height
        FROM lidar_box AS box
        JOIN lidar_pc AS pc ON pc.token = box.lidar_pc_token
        JOIN track ON track.token = box.track_token
        JOIN category ON category.token = track.category_token
        WHERE pc.scene_token = ?
        ORDER BY track_id, pc.timestamp
        """,
        (scene["scene_token"],),
    ).fetchall()
    by_track: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in object_rows:
        by_track[row["track_id"]].append(row)

    for track_id, rows in sorted(by_track.items()):
        samples: list[Sample] = []
        for row in rows:
            timestamp = frame_times[row["lidar_pc_token"]]
            x, y, _ = local_frame.point(row["x"], row["y"], row["z"])
            vx, vy = local_frame.vector(row["vx"], row["vy"])
            samples.append(Sample(timestamp, x, y, local_frame.yaw(row["yaw"]), vx, vy))
        series = resample_samples(samples, timeline, max_gap)
        dimensions = next(
            ((row["length"], row["width"], row["height"]) for row in reversed(rows)
             if all(value is not None and value > 0 for value in (row["length"], row["width"], row["height"]))),
            (0.1, 0.1, 0.1),
        )
        agents.append(
            AgentTrajectory(
                id=track_id,
                type=_agent_type(rows[0]["category"]),
                is_ego=False,
                length=float(dimensions[0]),
                width=float(dimensions[1]),
                height=float(dimensions[2]),
                x=series["x"],
                y=series["y"],
                yaw=series["yaw"],
                vx=series["vx"],
                vy=series["vy"],
                valid=series["valid"],
                goal=_last_valid_goal(series, "logged_terminal_position"),
            )
        )

    raw_lights: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for row in connection.execute(
        """
        SELECT tls.lidar_pc_token, tls.lane_connector_id, tls.status
        FROM traffic_light_status AS tls
        JOIN lidar_pc AS pc ON pc.token = tls.lidar_pc_token
        WHERE pc.scene_token = ?
        ORDER BY pc.timestamp
        """,
        (scene["scene_token"],),
    ):
        raw_lights[str(row["lane_connector_id"])].append(
            (frame_times[row["lidar_pc_token"]], row["status"].lower())
        )
    traffic_lights: list[TrafficLight] = []
    for lane_id, samples in sorted(raw_lights.items()):
        states, valid, _ = resample_discrete_states(
            [(timestamp, _light_state(state)) for timestamp, state in samples], timeline
        )
        _, _, source_states = resample_discrete_states(samples, timeline)
        traffic_lights.append(TrafficLight(lane_id, states, valid, source_states=source_states))

    tags: list[dict[str, Any]] = []
    for row in connection.execute(
        """
        SELECT tag.type, tag.agent_track_token, pc.timestamp
        FROM scenario_tag AS tag
        JOIN lidar_pc AS pc ON pc.token = tag.lidar_pc_token
        WHERE pc.scene_token = ?
        ORDER BY pc.timestamp, tag.type
        """,
        (scene["scene_token"],),
    ):
        tags.append({
            "type": row["type"],
            "timestamp": row["timestamp"] / 1_000_000.0 - start_time,
            "agent_id": _token_text(row["agent_track_token"]) if row["agent_track_token"] else None,
        })

    route_ids = _split_ids(scene["roadblock_ids"])
    route_goal = Goal(available=False)
    if scene["goal_ego_pose_token"] is not None:
        goal_row = connection.execute(
            "SELECT x, y, z FROM ego_pose WHERE token = ?",
            (scene["goal_ego_pose_token"],),
        ).fetchone()
        if goal_row is not None:
            goal_x, goal_y, _ = local_frame.point(goal_row["x"], goal_row["y"], goal_row["z"])
            route_goal = Goal(True, goal_x, goal_y, "dataset_mission_goal")

    warnings: list[str] = []
    map_features: list[MapFeature] = []
    map_available = False
    if maps_root is not None:
        try:
            map_features = _load_map_features(
                maps_root, scene["map_version"], scene["map_name"],
                first["x"], first["y"], local_frame,
            )
            map_available = True
        except (RuntimeError, OSError, AssertionError, ValueError) as error:
            warnings.append(f"map conversion unavailable: {error}")
    else:
        warnings.append("map conversion skipped because maps_root was not provided")

    coverage_complete = all(
        math.hypot(row["x"] - first["x"], row["y"] - first["y"]) <= MAP_RADIUS_METERS
        for row in frame_rows
    )
    if not coverage_complete:
        warnings.append("ego trajectory leaves the 150 meter map crop")

    feature_ids = {feature.id for feature in map_features}
    unresolved: list[str] = []
    if map_available:
        for light in traffic_lights:
            if light.lane_id not in feature_ids:
                unresolved.append(f"traffic_light_lane:{light.lane_id}")
        for roadblock_id in route_ids:
            if roadblock_id not in feature_ids:
                unresolved.append(f"route_roadblock:{roadblock_id}")

    scenario = CanonicalScenario(
        source=SourceInfo("nuplan", str(source_file), scene["scene_id"]),
        timing=TimingInfo(TARGET_DT, timeline, len(timeline)),
        coordinate_frame=CoordinateFrame(
            "local_cartesian",
            "x_forward_y_left_z_up",
            first["x"], first["y"], first["z"], origin_yaw,
            int(first["epsg"]) if first["epsg"] is not None else None,
        ),
        agents=agents,
        map_features=map_features,
        traffic_lights=traffic_lights,
        route=RouteInfo(bool(route_ids or route_goal.available), route_ids, [], route_goal),
        tags=tags,
        quality=QualityInfo(map_available, coverage_complete, unresolved, warnings),
    )
    validate_scenario(scenario)
    return scenario


def convert_nuplan_database(
    db_path: Path,
    maps_root: Optional[Path] = None,
    scene_id: Optional[str] = None,
) -> list[CanonicalScenario]:
    """Convert every scene in one nuPlan SQLite database."""

    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        scenes = _scene_rows(connection)
        if scene_id is not None:
            scenes = [scene for scene in scenes if scene["scene_id"] == scene_id or scene["name"] == scene_id]
        if not scenes:
            raise ValueError(f"no matching nuPlan scenes in {db_path}")
        return [_convert_scene(connection, scene, db_path, maps_root) for scene in scenes]
    finally:
        connection.close()
