"""Strict validation for CanonicalScenario v1."""

from __future__ import annotations

import math

from .models import CanonicalScenario, SCHEMA_VERSION


class ScenarioValidationError(ValueError):
    pass


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def validate_scenario(scenario: CanonicalScenario) -> list[str]:
    """Validate invariants and return non-fatal warnings."""

    errors: list[str] = []
    warnings = list(scenario.quality.warnings)
    if scenario.schema_version != SCHEMA_VERSION:
        errors.append(f"unsupported schema_version: {scenario.schema_version}")
    timing = scenario.timing
    if timing.dt <= 0 or timing.num_steps <= 0:
        errors.append("timing dt and num_steps must be positive")
    if len(timing.timestamps) != timing.num_steps:
        errors.append("timestamps length does not match num_steps")
    if timing.timestamps != sorted(timing.timestamps):
        errors.append("timestamps are not ordered")
    if not all(_finite(value) for value in timing.timestamps):
        errors.append("timestamps contain non-finite values")
    frame = scenario.coordinate_frame
    if not all(_finite(value) for value in (frame.origin_x, frame.origin_y, frame.origin_z, frame.origin_yaw)):
        errors.append("coordinate frame contains non-finite values")

    ids = [agent.id for agent in scenario.agents]
    if len(ids) != len(set(ids)):
        errors.append("agent IDs are not unique")
    if sum(agent.is_ego for agent in scenario.agents) != 1:
        errors.append("scenario must contain exactly one ego agent")

    trajectory_fields = ("x", "y", "yaw", "vx", "vy", "valid")
    for agent in scenario.agents:
        for field_name in trajectory_fields:
            if len(getattr(agent, field_name)) != timing.num_steps:
                errors.append(f"agent {agent.id} {field_name} length does not match num_steps")
        if not all(_finite(value) and value > 0 for value in (agent.length, agent.width, agent.height)):
            errors.append(f"agent {agent.id} has non-positive dimensions")
        for index, is_valid in enumerate(agent.valid):
            values = (agent.x[index], agent.y[index], agent.yaw[index], agent.vx[index], agent.vy[index])
            if is_valid and not all(_finite(value) for value in values):
                errors.append(f"agent {agent.id} has invalid numeric state at step {index}")
            if not is_valid and any(value is not None for value in values):
                errors.append(f"agent {agent.id} invalid step {index} must contain null states")
        if agent.goal.available and not all(_finite(value) for value in (agent.goal.x, agent.goal.y)):
            errors.append(f"agent {agent.id} has an invalid available goal")

    for feature in scenario.map_features:
        if feature.speed_limit_mps is not None and not _finite(feature.speed_limit_mps):
            errors.append(f"map feature {feature.id} has invalid speed limit")
        for point in feature.geometry:
            if len(point) not in (2, 3) or not all(_finite(value) for value in point):
                errors.append(f"map feature {feature.id} has invalid geometry")
                break

    for light in scenario.traffic_lights:
        if len(light.states) != timing.num_steps or len(light.valid) != timing.num_steps:
            errors.append(f"traffic light {light.lane_id} length does not match num_steps")
        if light.source_states and len(light.source_states) != timing.num_steps:
            errors.append(f"traffic light {light.lane_id} source_states length does not match num_steps")
        if light.stop_point is not None and not all(_finite(value) for value in light.stop_point):
            errors.append(f"traffic light {light.lane_id} has invalid stop point")
        if any(state not in {"unknown", "stop", "caution", "go"} for state in light.states):
            errors.append(f"traffic light {light.lane_id} has an unsupported state")

    if scenario.route.goal.available and not all(
        _finite(value) for value in (scenario.route.goal.x, scenario.route.goal.y)
    ):
        errors.append("route has an invalid available goal")

    ego = next((agent for agent in scenario.agents if agent.is_ego), None)
    if ego is not None and ego.valid and ego.valid[0]:
        if abs(float(ego.x[0])) > 1e-4 or abs(float(ego.y[0])) > 1e-4 or abs(float(ego.yaw[0])) > 1e-4:
            errors.append("initial ego pose is not the local origin")

    if errors:
        raise ScenarioValidationError("; ".join(errors))
    return warnings
