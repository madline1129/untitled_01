"""Convert Waymo Open Motion Scenario protos to CanonicalScenario v1."""

from __future__ import annotations

import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from ..geometry import LocalFrame
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


_OBJECT_TYPES = {0: "other", 1: "vehicle", 2: "pedestrian", 3: "cyclist", 4: "other"}
_LANE_TYPES = {0: "undefined", 1: "freeway", 2: "surface_street", 3: "bike_lane"}
_ROAD_LINE_TYPES = {
    0: "unknown", 1: "broken_single_white", 2: "solid_single_white",
    3: "solid_double_white", 4: "broken_single_yellow", 5: "broken_double_yellow",
    6: "solid_single_yellow", 7: "solid_double_yellow", 8: "passing_double_yellow",
}
_ROAD_EDGE_TYPES = {0: "unknown", 1: "boundary", 2: "median"}
_LIGHT_STATES = {
    0: ("unknown", "unknown"),
    1: ("stop", "arrow_stop"),
    2: ("caution", "arrow_caution"),
    3: ("go", "arrow_go"),
    4: ("stop", "stop"),
    5: ("caution", "caution"),
    6: ("go", "go"),
    7: ("stop", "flashing_stop"),
    8: ("caution", "flashing_caution"),
}


def _median_gap(times: list[float]) -> float | None:
    gaps = [right - left for left, right in zip(times, times[1:]) if right > left]
    return statistics.median(gaps) if gaps else None


def _last_goal(series: dict[str, list[object]]) -> Goal:
    for index in range(len(series["valid"]) - 1, -1, -1):
        if series["valid"][index]:
            return Goal(True, float(series["x"][index]), float(series["y"][index]), "logged_terminal_position")
    return Goal(False)


def _point(point: object, local_frame: LocalFrame) -> list[float]:
    return list(local_frame.point(float(point.x), float(point.y), float(point.z)))


def _feature_data(feature: object) -> tuple[str | None, object | None]:
    feature_type = feature.WhichOneof("feature_data")
    return feature_type, getattr(feature, feature_type) if feature_type else None


def _map_feature(feature: object, local_frame: LocalFrame) -> MapFeature | None:
    source_type, data = _feature_data(feature)
    if source_type is None or data is None:
        return None

    feature_type = source_type
    geometry_type = "polyline"
    points: Iterable[object]
    subtype = None
    speed_limit = None
    predecessors: list[str] = []
    successors: list[str] = []
    left_neighbors: list[str] = []
    right_neighbors: list[str] = []

    if source_type == "lane":
        feature_type = "lane"
        points = data.polyline
        subtype = _LANE_TYPES.get(int(data.type), "unknown")
        speed_limit = float(data.speed_limit_mph) * 0.44704 if data.speed_limit_mph > 0 else None
        predecessors = [str(value) for value in data.entry_lanes]
        successors = [str(value) for value in data.exit_lanes]
        left_neighbors = [str(value.feature_id) for value in data.left_neighbors]
        right_neighbors = [str(value.feature_id) for value in data.right_neighbors]
    elif source_type == "road_line":
        points = data.polyline
        subtype = _ROAD_LINE_TYPES.get(int(data.type), "unknown")
    elif source_type == "road_edge":
        points = data.polyline
        subtype = _ROAD_EDGE_TYPES.get(int(data.type), "unknown")
    elif source_type == "stop_sign":
        points = [data.position]
        geometry_type = "point"
        successors = [str(value) for value in data.lane]
    else:
        points = data.polygon
        geometry_type = "polygon"

    return MapFeature(
        id=str(feature.id),
        type=feature_type,
        geometry_type=geometry_type,
        geometry=[_point(point, local_frame) for point in points],
        subtype=subtype,
        speed_limit_mps=speed_limit,
        predecessor_ids=predecessors,
        successor_ids=successors,
        left_neighbor_ids=left_neighbors,
        right_neighbor_ids=right_neighbors,
        source_type=source_type,
    )


def convert_waymo_scenario(
    scenario_proto: object,
    source_file: str = "<memory>",
) -> CanonicalScenario:
    """Convert one official Waymo Scenario protobuf."""

    source_timestamps = [float(value) for value in scenario_proto.timestamps_seconds]
    if not source_timestamps:
        raise ValueError("Waymo scenario contains no timestamps")
    if not 0 <= int(scenario_proto.sdc_track_index) < len(scenario_proto.tracks):
        raise ValueError("Waymo scenario has an invalid sdc_track_index")

    sdc_track = scenario_proto.tracks[int(scenario_proto.sdc_track_index)]
    first_valid_index = next((index for index, state in enumerate(sdc_track.states) if state.valid), None)
    if first_valid_index is None:
        raise ValueError("Waymo SDC track contains no valid state")

    source_timestamps = source_timestamps[first_valid_index:]
    start_time = source_timestamps[0]
    relative_times = [timestamp - start_time for timestamp in source_timestamps]
    timeline = build_timeline(0.0, relative_times[-1], TARGET_DT)
    source_gap = _median_gap(relative_times)
    max_gap = source_gap * 1.5 if source_gap is not None else None
    first_sdc = sdc_track.states[first_valid_index]
    local_frame = LocalFrame(first_sdc.center_x, first_sdc.center_y, first_sdc.center_z, first_sdc.heading)

    object_interest = {int(value) for value in scenario_proto.objects_of_interest}
    prediction_roles = {
        int(value.track_index): int(value.difficulty)
        for value in scenario_proto.tracks_to_predict
    }
    agents: list[AgentTrajectory] = []
    for track_index, track in enumerate(scenario_proto.tracks):
        samples: list[Sample] = []
        valid_dimensions: list[tuple[float, float, float]] = []
        for timestamp, state in zip(relative_times, track.states[first_valid_index:]):
            if state.valid:
                x, y, _ = local_frame.point(state.center_x, state.center_y, state.center_z)
                vx, vy = local_frame.vector(state.velocity_x, state.velocity_y)
                valid_dimensions.append((state.length, state.width, state.height))
                samples.append(Sample(timestamp, x, y, local_frame.yaw(state.heading), vx, vy, True))
            else:
                samples.append(Sample(timestamp, 0.0, 0.0, 0.0, 0.0, 0.0, False))
        series = resample_samples(samples, timeline, max_gap)
        if valid_dimensions:
            length, width, height = valid_dimensions[-1]
        else:
            length, width, height = (0.1, 0.1, 0.1)
        roles = {
            "is_sdc": track_index == int(scenario_proto.sdc_track_index),
            "is_object_of_interest": int(track.id) in object_interest,
            "is_track_to_predict": track_index in prediction_roles,
            "prediction_difficulty": prediction_roles.get(track_index),
        }
        agents.append(
            AgentTrajectory(
                id=str(track.id),
                type=_OBJECT_TYPES.get(int(track.object_type), "other"),
                is_ego=roles["is_sdc"],
                length=max(float(length), 0.1),
                width=max(float(width), 0.1),
                height=max(float(height), 0.1),
                x=series["x"], y=series["y"], yaw=series["yaw"],
                vx=series["vx"], vy=series["vy"], valid=series["valid"],
                goal=_last_goal(series),
                roles=roles,
            )
        )

    map_features = [
        converted
        for feature in scenario_proto.map_features
        if (converted := _map_feature(feature, local_frame)) is not None
    ]

    canonical_lights: dict[str, list[tuple[float, str]]] = defaultdict(list)
    raw_lights: dict[str, list[tuple[float, str]]] = defaultdict(list)
    stop_points: dict[str, list[float]] = {}
    dynamic_states = list(scenario_proto.dynamic_map_states)[first_valid_index:]
    for timestamp, dynamic_state in zip(relative_times, dynamic_states):
        for lane_state in dynamic_state.lane_states:
            lane_id = str(lane_state.lane)
            canonical, raw = _LIGHT_STATES.get(int(lane_state.state), ("unknown", "unknown"))
            canonical_lights[lane_id].append((timestamp, canonical))
            raw_lights[lane_id].append((timestamp, raw))
            stop_points[lane_id] = _point(lane_state.stop_point, local_frame)

    traffic_lights: list[TrafficLight] = []
    for lane_id in sorted(canonical_lights):
        states, valid, _ = resample_discrete_states(canonical_lights[lane_id], timeline)
        _, _, source_states = resample_discrete_states(raw_lights[lane_id], timeline)
        traffic_lights.append(TrafficLight(lane_id, states, valid, stop_points[lane_id], source_states))

    feature_ids = {feature.id for feature in map_features}
    unresolved = [
        f"traffic_light_lane:{light.lane_id}"
        for light in traffic_lights
        if light.lane_id not in feature_ids
    ]
    anchor_time = float(scenario_proto.timestamps_seconds[int(scenario_proto.current_time_index)])
    anchor_index = round((anchor_time - start_time) / TARGET_DT)
    anchor_index = min(max(anchor_index, 0), len(timeline) - 1)
    warnings = [] if map_features else ["Waymo scenario contains no map features"]
    scenario = CanonicalScenario(
        source=SourceInfo("waymo", source_file, str(scenario_proto.scenario_id)),
        timing=TimingInfo(TARGET_DT, timeline, len(timeline), anchor_index),
        coordinate_frame=CoordinateFrame(
            "local_cartesian", "x_forward_y_left_z_up",
            float(first_sdc.center_x), float(first_sdc.center_y), float(first_sdc.center_z),
            float(first_sdc.heading), None,
        ),
        agents=agents,
        map_features=map_features,
        traffic_lights=traffic_lights,
        route=RouteInfo(False),
        tags=[
            {"type": "objects_of_interest", "agent_ids": sorted(str(value) for value in object_interest)},
            {"type": "tracks_to_predict", "track_indices": sorted(prediction_roles)},
        ],
        quality=QualityInfo(bool(map_features), bool(map_features), unresolved, warnings),
    )
    validate_scenario(scenario)
    return scenario


def iter_waymo_tfrecord(path: Path) -> Iterable[object]:
    """Yield official Scenario protos from an uncompressed WOMD TFRecord."""

    try:
        import tensorflow as tf
        from waymo_open_dataset.protos import scenario_pb2
    except ImportError as error:
        raise RuntimeError(
            "Waymo conversion requires TensorFlow and waymo-open-dataset in the Python 3.10 environment"
        ) from error

    for record in tf.data.TFRecordDataset(str(path), compression_type=""):
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(bytes(record.numpy()))
        yield scenario


def convert_waymo_tfrecord(path: Path) -> list[CanonicalScenario]:
    return [convert_waymo_scenario(proto, str(path)) for proto in iter_waymo_tfrecord(path)]
