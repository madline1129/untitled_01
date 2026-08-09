"""Compile CanonicalScenario into fixed-capacity RL runtime tensors."""

from __future__ import annotations

import json
import math
import sys
from array import array
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from .models import AgentTrajectory, CanonicalScenario, MapFeature
from .validation import validate_scenario


RUNTIME_SCHEMA_VERSION = "rl-runtime-1.0"
STATE_FIELDS = ["x", "y", "yaw", "vx", "vy"]
AGENT_TYPE_ENUM = {"unknown": 0, "vehicle": 1, "pedestrian": 2, "cyclist": 3, "other": 4}
MAP_TYPE_ENUM = {
    "unknown": 0,
    "lane": 1,
    "lane_connector": 2,
    "road_line": 3,
    "road_edge": 4,
    "crosswalk": 5,
    "stop_line": 6,
    "walkway": 7,
    "roadblock": 8,
    "roadblock_connector": 9,
}
GEOMETRY_TYPE_ENUM = {"unknown": 0, "point": 1, "polyline": 2, "polygon": 3}
MAP_EDGE_RELATION_ENUM = {"predecessor": 1, "successor": 2, "left_neighbor": 3, "right_neighbor": 4}
TRAFFIC_LIGHT_STATE_ENUM = {"unknown": 0, "stop": 1, "caution": 2, "go": 3}


@dataclass(frozen=True)
class RuntimeConfig:
    max_agents: int = 64
    history_steps: int = 11
    max_future_steps: int = 128
    max_map_features: int = 2048
    max_map_points: int = 32768
    max_map_edges: int = 8192
    max_traffic_lights: int = 128
    max_route_features: int = 512

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if value <= 0:
                raise ValueError(f"runtime capacity {name} must be positive")


@dataclass
class TensorData:
    dtype: str
    shape: list[int]
    values: list[int | float]
    semantic: str
    policy_visible: bool

    def validate(self, name: str) -> None:
        expected = math.prod(self.shape)
        if expected != len(self.values):
            raise ValueError(f"tensor {name} expects {expected} values, got {len(self.values)}")
        if self.dtype not in {"float32", "int32", "uint8"}:
            raise ValueError(f"tensor {name} has unsupported dtype {self.dtype}")


@dataclass
class RuntimeScenario:
    manifest: dict[str, Any]
    tensors: dict[str, TensorData]

    def validate(self) -> None:
        if self.manifest["schema_version"] != RUNTIME_SCHEMA_VERSION:
            raise ValueError("unsupported runtime schema version")
        for name, tensor in self.tensors.items():
            tensor.validate(name)


def _zeros(shape: Iterable[int], value: int | float = 0) -> list[int | float]:
    return [value] * math.prod(shape)


def _offset(shape: list[int], *indices: int) -> int:
    if len(shape) != len(indices):
        raise ValueError("tensor index rank mismatch")
    result = 0
    stride = 1
    for size, index in zip(reversed(shape), reversed(indices)):
        result += index * stride
        stride *= size
    return result


def _set(values: list[int | float], shape: list[int], indices: tuple[int, ...], value: int | float) -> None:
    values[_offset(shape, *indices)] = value


def _state(agent: AgentTrajectory, step: int) -> list[float]:
    return [float(getattr(agent, field)[step]) for field in STATE_FIELDS]


def _initial_distance(agent: AgentTrajectory, anchor: int) -> float:
    if not agent.valid[anchor]:
        return math.inf
    return math.hypot(float(agent.x[anchor]), float(agent.y[anchor]))


def _ordered_agents(scenario: CanonicalScenario) -> list[AgentTrajectory]:
    anchor = scenario.timing.anchor_index
    return sorted(
        scenario.agents,
        key=lambda agent: (
            not agent.is_ego,
            _initial_distance(agent, anchor),
            agent.type,
            agent.id,
        ),
    )


def _feature_distance(feature: MapFeature) -> float:
    if not feature.geometry:
        return math.inf
    return min(math.hypot(point[0], point[1]) for point in feature.geometry)


def _selected_map_features(
    scenario: CanonicalScenario,
    config: RuntimeConfig,
    warnings: list[str],
) -> list[MapFeature]:
    route_ids = set(scenario.route.lane_ids) | set(scenario.route.roadblock_ids)
    candidates = sorted(
        scenario.map_features,
        key=lambda feature: (feature.id not in route_ids, _feature_distance(feature), feature.type, feature.id),
    )
    selected: list[MapFeature] = []
    point_count = 0
    for feature in candidates:
        if len(selected) >= config.max_map_features:
            break
        if point_count + len(feature.geometry) > config.max_map_points:
            continue
        selected.append(feature)
        point_count += len(feature.geometry)
    if len(selected) < len(candidates):
        warnings.append(
            f"map truncated: kept {len(selected)}/{len(candidates)} features and "
            f"{point_count}/{sum(len(feature.geometry) for feature in candidates)} points"
        )
    return selected


def _tensor(
    dtype: str,
    shape: list[int],
    values: list[int | float],
    semantic: str,
    policy_visible: bool,
) -> TensorData:
    return TensorData(dtype, shape, values, semantic, policy_visible)


def compile_runtime_scenario(
    scenario: CanonicalScenario,
    config: Optional[RuntimeConfig] = None,
) -> RuntimeScenario:
    """Compile one canonical scene into padded tensors for reset and rollout."""

    validate_scenario(scenario)
    config = config or RuntimeConfig()
    config.validate()
    warnings = list(scenario.quality.warnings)
    anchor = scenario.timing.anchor_index

    ordered = _ordered_agents(scenario)
    agents = ordered[:config.max_agents]
    if len(agents) < len(ordered):
        warnings.append(f"agents truncated: kept {len(agents)}/{len(ordered)}")
    if not agents or not agents[0].is_ego:
        raise ValueError("ego agent must be retained at runtime slot 0")

    agent_shape = [config.max_agents, len(STATE_FIELDS)]
    initial_state = _zeros(agent_shape, 0.0)
    initial_valid = _zeros([config.max_agents], 0)
    agent_type = _zeros([config.max_agents], AGENT_TYPE_ENUM["unknown"])
    agent_is_ego = _zeros([config.max_agents], 0)
    agent_controllable = _zeros([config.max_agents], 0)
    agent_dimensions = _zeros([config.max_agents, 3], 0.0)
    agent_goal = _zeros([config.max_agents, 2], 0.0)
    agent_goal_valid = _zeros([config.max_agents], 0)
    history_shape = [config.history_steps, config.max_agents, len(STATE_FIELDS)]
    history = _zeros(history_shape, 0.0)
    history_valid = _zeros([config.history_steps, config.max_agents], 0)
    future_shape = [config.max_future_steps, config.max_agents, len(STATE_FIELDS)]
    reference_future = _zeros(future_shape, 0.0)
    reference_future_valid = _zeros([config.max_future_steps, config.max_agents], 0)

    for agent_index, agent in enumerate(agents):
        agent_type[agent_index] = AGENT_TYPE_ENUM.get(agent.type, AGENT_TYPE_ENUM["unknown"])
        agent_is_ego[agent_index] = int(agent.is_ego)
        agent_controllable[agent_index] = int(agent.type in {"vehicle", "pedestrian", "cyclist"})
        for dim_index, value in enumerate((agent.length, agent.width, agent.height)):
            _set(agent_dimensions, [config.max_agents, 3], (agent_index, dim_index), float(value))
        if agent.goal.available:
            agent_goal_valid[agent_index] = 1
            agent_goal[_offset([config.max_agents, 2], agent_index, 0)] = float(agent.goal.x)
            agent_goal[_offset([config.max_agents, 2], agent_index, 1)] = float(agent.goal.y)
        if agent.valid[anchor]:
            initial_valid[agent_index] = 1
            for state_index, value in enumerate(_state(agent, anchor)):
                _set(initial_state, agent_shape, (agent_index, state_index), value)

        first_history_step = max(0, anchor - config.history_steps + 1)
        for source_step in range(first_history_step, anchor + 1):
            history_step = config.history_steps - 1 - (anchor - source_step)
            if not agent.valid[source_step]:
                continue
            history_valid[_offset([config.history_steps, config.max_agents], history_step, agent_index)] = 1
            for state_index, value in enumerate(_state(agent, source_step)):
                _set(history, history_shape, (history_step, agent_index, state_index), value)

        available_future = min(config.max_future_steps, scenario.timing.num_steps - anchor - 1)
        for future_step in range(available_future):
            source_step = anchor + future_step + 1
            if not agent.valid[source_step]:
                continue
            reference_future_valid[
                _offset([config.max_future_steps, config.max_agents], future_step, agent_index)
            ] = 1
            for state_index, value in enumerate(_state(agent, source_step)):
                _set(reference_future, future_shape, (future_step, agent_index, state_index), value)

    available_future_steps = max(0, scenario.timing.num_steps - anchor - 1)
    future_steps = min(config.max_future_steps, available_future_steps)
    if future_steps < available_future_steps:
        warnings.append(f"reference future truncated: kept {future_steps}/{available_future_steps} steps")

    features = _selected_map_features(scenario, config, warnings)
    feature_index = {feature.id: index for index, feature in enumerate(features)}
    map_points = _zeros([config.max_map_points, 3], 0.0)
    map_point_valid = _zeros([config.max_map_points], 0)
    map_feature_type = _zeros([config.max_map_features], MAP_TYPE_ENUM["unknown"])
    map_geometry_type = _zeros([config.max_map_features], GEOMETRY_TYPE_ENUM["unknown"])
    map_feature_point_start = _zeros([config.max_map_features], 0)
    map_feature_point_count = _zeros([config.max_map_features], 0)
    map_feature_valid = _zeros([config.max_map_features], 0)
    map_speed_limit = _zeros([config.max_map_features], 0.0)
    map_speed_limit_valid = _zeros([config.max_map_features], 0)
    point_cursor = 0
    for index, feature in enumerate(features):
        map_feature_valid[index] = 1
        map_feature_type[index] = MAP_TYPE_ENUM.get(feature.type, MAP_TYPE_ENUM["unknown"])
        map_geometry_type[index] = GEOMETRY_TYPE_ENUM.get(
            feature.geometry_type, GEOMETRY_TYPE_ENUM["unknown"]
        )
        map_feature_point_start[index] = point_cursor
        map_feature_point_count[index] = len(feature.geometry)
        if feature.speed_limit_mps is not None:
            map_speed_limit[index] = float(feature.speed_limit_mps)
            map_speed_limit_valid[index] = 1
        for point in feature.geometry:
            map_point_valid[point_cursor] = 1
            for coordinate in range(min(3, len(point))):
                _set(map_points, [config.max_map_points, 3], (point_cursor, coordinate), float(point[coordinate]))
            point_cursor += 1

    map_edges = _zeros([config.max_map_edges, 3], 0)
    map_edge_valid = _zeros([config.max_map_edges], 0)
    edge_cursor = 0
    relations = (
        ("predecessor", "predecessor_ids"),
        ("successor", "successor_ids"),
        ("left_neighbor", "left_neighbor_ids"),
        ("right_neighbor", "right_neighbor_ids"),
    )
    for source_index, feature in enumerate(features):
        for relation_name, field_name in relations:
            for target_id in getattr(feature, field_name):
                target_index = feature_index.get(target_id)
                if target_index is None:
                    continue
                if edge_cursor >= config.max_map_edges:
                    break
                for value_index, value in enumerate(
                    (source_index, target_index, MAP_EDGE_RELATION_ENUM[relation_name])
                ):
                    _set(map_edges, [config.max_map_edges, 3], (edge_cursor, value_index), value)
                map_edge_valid[edge_cursor] = 1
                edge_cursor += 1
    if edge_cursor >= config.max_map_edges:
        warnings.append(f"map topology edges reached capacity {config.max_map_edges}")

    lights = sorted(scenario.traffic_lights, key=lambda light: light.lane_id)[:config.max_traffic_lights]
    if len(lights) < len(scenario.traffic_lights):
        warnings.append(f"traffic lights truncated: kept {len(lights)}/{len(scenario.traffic_lights)}")
    light_feature_index = _zeros([config.max_traffic_lights], -1)
    light_valid = _zeros([config.max_future_steps + 1, config.max_traffic_lights], 0)
    light_state = _zeros([config.max_future_steps + 1, config.max_traffic_lights], 0)
    for light_index, light in enumerate(lights):
        light_feature_index[light_index] = feature_index.get(light.lane_id, -1)
        for runtime_step in range(future_steps + 1):
            source_step = anchor + runtime_step
            if source_step >= len(light.states) or not light.valid[source_step]:
                continue
            _set(
                light_valid,
                [config.max_future_steps + 1, config.max_traffic_lights],
                (runtime_step, light_index),
                1,
            )
            _set(
                light_state,
                [config.max_future_steps + 1, config.max_traffic_lights],
                (runtime_step, light_index),
                TRAFFIC_LIGHT_STATE_ENUM.get(light.states[source_step], 0),
            )

    route_ids = scenario.route.lane_ids + scenario.route.roadblock_ids
    selected_route = [feature_index[feature_id] for feature_id in route_ids if feature_id in feature_index]
    route_feature_index = _zeros([config.max_route_features], -1)
    route_feature_valid = _zeros([config.max_route_features], 0)
    for index, value in enumerate(selected_route[:config.max_route_features]):
        route_feature_index[index] = value
        route_feature_valid[index] = 1
    route_goal = _zeros([2], 0.0)
    route_goal_valid = [0]
    if scenario.route.goal.available:
        route_goal = [float(scenario.route.goal.x), float(scenario.route.goal.y)]
        route_goal_valid = [1]

    tensors = {
        "agent_initial_state": _tensor("float32", agent_shape, initial_state, "reset state", True),
        "agent_initial_valid": _tensor("uint8", [config.max_agents], initial_valid, "valid reset slots", True),
        "agent_type": _tensor("int32", [config.max_agents], agent_type, "agent type enum", True),
        "agent_is_ego": _tensor("uint8", [config.max_agents], agent_is_ego, "ego slot mask", True),
        "agent_controllable": _tensor("uint8", [config.max_agents], agent_controllable, "policy candidate mask", True),
        "agent_dimensions": _tensor("float32", [config.max_agents, 3], agent_dimensions, "length width height", True),
        "agent_goal": _tensor("float32", [config.max_agents, 2], agent_goal, "simulator-private local goal xy", False),
        "agent_goal_valid": _tensor("uint8", [config.max_agents], agent_goal_valid, "simulator-private goal mask", False),
        "agent_history": _tensor("float32", history_shape, history, "history ending at reset", True),
        "agent_history_valid": _tensor("uint8", [config.history_steps, config.max_agents], history_valid, "history mask", True),
        "reference_future": _tensor("float32", future_shape, reference_future, "logged evaluation target after reset", False),
        "reference_future_valid": _tensor("uint8", [config.max_future_steps, config.max_agents], reference_future_valid, "reference mask", False),
        "map_points": _tensor("float32", [config.max_map_points, 3], map_points, "flattened local map geometry", True),
        "map_point_valid": _tensor("uint8", [config.max_map_points], map_point_valid, "map point mask", True),
        "map_feature_type": _tensor("int32", [config.max_map_features], map_feature_type, "map type enum", True),
        "map_geometry_type": _tensor("int32", [config.max_map_features], map_geometry_type, "geometry type enum", True),
        "map_feature_point_start": _tensor("int32", [config.max_map_features], map_feature_point_start, "map point offsets", True),
        "map_feature_point_count": _tensor("int32", [config.max_map_features], map_feature_point_count, "map point counts", True),
        "map_feature_valid": _tensor("uint8", [config.max_map_features], map_feature_valid, "map feature mask", True),
        "map_speed_limit": _tensor("float32", [config.max_map_features], map_speed_limit, "speed limit mps", True),
        "map_speed_limit_valid": _tensor("uint8", [config.max_map_features], map_speed_limit_valid, "speed limit mask", True),
        "map_edges": _tensor("int32", [config.max_map_edges, 3], map_edges, "source target relation", True),
        "map_edge_valid": _tensor("uint8", [config.max_map_edges], map_edge_valid, "topology edge mask", True),
        "traffic_light_feature_index": _tensor("int32", [config.max_traffic_lights], light_feature_index, "controlled lane feature", False),
        "traffic_light_state": _tensor("uint8", [config.max_future_steps + 1, config.max_traffic_lights], light_state, "simulator signal schedule", False),
        "traffic_light_valid": _tensor("uint8", [config.max_future_steps + 1, config.max_traffic_lights], light_valid, "signal schedule mask", False),
        "route_feature_index": _tensor("int32", [config.max_route_features], route_feature_index, "ordered route map indices", True),
        "route_feature_valid": _tensor("uint8", [config.max_route_features], route_feature_valid, "route index mask", True),
        "route_goal": _tensor("float32", [2], route_goal, "local route goal xy", True),
        "route_goal_valid": _tensor("uint8", [1], route_goal_valid, "route goal mask", True),
    }
    manifest = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "source": asdict(scenario.source),
        "canonical_schema_version": scenario.schema_version,
        "coordinate_frame": asdict(scenario.coordinate_frame),
        "dt": scenario.timing.dt,
        "anchor_index": anchor,
        "episode_steps": future_steps,
        "state_fields": STATE_FIELDS,
        "tensor_layout": "C-contiguous little-endian",
        "capacities": asdict(config),
        "counts": {
            "agents": len(agents),
            "map_features": len(features),
            "map_points": point_cursor,
            "map_edges": edge_cursor,
            "traffic_lights": len(lights),
            "route_features": min(len(selected_route), config.max_route_features),
        },
        "agent_ids": [agent.id for agent in agents],
        "map_feature_ids": [feature.id for feature in features],
        "traffic_light_lane_ids": [light.lane_id for light in lights],
        "enums": {
            "agent_type": AGENT_TYPE_ENUM,
            "map_type": MAP_TYPE_ENUM,
            "geometry_type": GEOMETRY_TYPE_ENUM,
            "map_edge_relation": MAP_EDGE_RELATION_ENUM,
            "traffic_light_state": TRAFFIC_LIGHT_STATE_ENUM,
        },
        "warnings": warnings,
    }
    runtime = RuntimeScenario(manifest, tensors)
    runtime.validate()
    return runtime


def _write_values(path: Path, dtype: str, values: list[int | float]) -> None:
    typecode = {"float32": "f", "int32": "i", "uint8": "B"}[dtype]
    data = array(typecode, values)
    expected_size = {"float32": 4, "int32": 4, "uint8": 1}[dtype]
    if data.itemsize != expected_size:
        raise RuntimeError(f"host array type {typecode} is not {expected_size} bytes")
    if sys.byteorder != "little" and expected_size > 1:
        data.byteswap()
    with path.open("wb") as stream:
        data.tofile(stream)


def write_runtime_scenario(runtime: RuntimeScenario, output_dir: Path) -> Path:
    """Write one runtime scene directory, publishing manifest last."""

    runtime.validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_specs: dict[str, dict[str, Any]] = {}
    for name, tensor in runtime.tensors.items():
        filename = f"{name}.bin"
        _write_values(output_dir / filename, tensor.dtype, tensor.values)
        tensor_specs[name] = {
            "dtype": tensor.dtype,
            "shape": tensor.shape,
            "file": filename,
            "semantic": tensor.semantic,
            "policy_visible": tensor.policy_visible,
        }
    manifest = {**runtime.manifest, "tensors": tensor_specs}
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def read_runtime_tensor(runtime_dir: Path, name: str) -> tuple[dict[str, Any], list[int | float]]:
    """Read a runtime tensor without requiring NumPy."""

    manifest = json.loads((runtime_dir / "manifest.json").read_text(encoding="utf-8"))
    spec = manifest["tensors"][name]
    typecode = {"float32": "f", "int32": "i", "uint8": "B"}[spec["dtype"]]
    values = array(typecode)
    with (runtime_dir / spec["file"]).open("rb") as stream:
        values.fromfile(stream, math.prod(spec["shape"]))
    if sys.byteorder != "little" and values.itemsize > 1:
        values.byteswap()
    return spec, values.tolist()


def validate_runtime_directory(runtime_dir: Path) -> list[str]:
    """Validate manifest, tensor byte sizes, masks, and reset invariants."""

    manifest_path = runtime_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if manifest.get("schema_version") != RUNTIME_SCHEMA_VERSION:
        errors.append("unsupported runtime schema version")
    byte_sizes = {"float32": 4, "int32": 4, "uint8": 1}
    for name, spec in manifest.get("tensors", {}).items():
        path = runtime_dir / spec["file"]
        expected = math.prod(spec["shape"]) * byte_sizes[spec["dtype"]]
        if not path.is_file() or path.stat().st_size != expected:
            errors.append(f"tensor {name} byte size mismatch")
    try:
        _, initial = read_runtime_tensor(runtime_dir, "agent_initial_state")
        _, valid = read_runtime_tensor(runtime_dir, "agent_initial_valid")
        if not valid or valid[0] != 1:
            errors.append("runtime slot 0 ego is not valid")
        elif any(abs(float(value)) > 1e-4 for value in initial[:3]):
            errors.append("runtime ego reset pose is not (0, 0, 0)")
        if manifest["tensors"]["reference_future"]["policy_visible"]:
            errors.append("reference future must not be policy visible")
    except (KeyError, OSError, EOFError, ValueError) as error:
        errors.append(f"cannot validate reset tensors: {error}")
    if errors:
        raise ValueError("; ".join(errors))
    return list(manifest.get("warnings", []))
