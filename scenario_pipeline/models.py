"""CanonicalScenario v1 data model.

The model deliberately uses JSON-compatible primitives so it can be shared by
the separate nuPlan and Waymo extraction environments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


SCHEMA_VERSION = "1.0"


@dataclass
class SourceInfo:
    dataset: str
    source_file: str
    source_scenario_id: str


@dataclass
class TimingInfo:
    dt: float
    timestamps: list[float]
    num_steps: int
    anchor_index: int = 0


@dataclass
class CoordinateFrame:
    type: str
    axis_convention: str
    origin_x: float
    origin_y: float
    origin_z: float
    origin_yaw: float
    epsg: Optional[int] = None


@dataclass
class Goal:
    available: bool
    x: Optional[float] = None
    y: Optional[float] = None
    source: Optional[str] = None


@dataclass
class AgentTrajectory:
    id: str
    type: str
    is_ego: bool
    length: float
    width: float
    height: float
    x: list[Optional[float]]
    y: list[Optional[float]]
    yaw: list[Optional[float]]
    vx: list[Optional[float]]
    vy: list[Optional[float]]
    valid: list[bool]
    goal: Goal
    roles: dict[str, Any] = field(default_factory=dict)


@dataclass
class MapFeature:
    id: str
    type: str
    geometry_type: str
    geometry: list[list[float]]
    subtype: Optional[str] = None
    speed_limit_mps: Optional[float] = None
    predecessor_ids: list[str] = field(default_factory=list)
    successor_ids: list[str] = field(default_factory=list)
    left_neighbor_ids: list[str] = field(default_factory=list)
    right_neighbor_ids: list[str] = field(default_factory=list)
    source_type: Optional[str] = None


@dataclass
class TrafficLight:
    lane_id: str
    states: list[str]
    valid: list[bool]
    stop_point: Optional[list[float]] = None
    source_states: list[Optional[str]] = field(default_factory=list)


@dataclass
class RouteInfo:
    available: bool
    roadblock_ids: list[str] = field(default_factory=list)
    lane_ids: list[str] = field(default_factory=list)
    goal: Goal = field(default_factory=lambda: Goal(available=False))


@dataclass
class QualityInfo:
    map_available: bool
    map_coverage_complete: bool
    unresolved_references: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class CanonicalScenario:
    source: SourceInfo
    timing: TimingInfo
    coordinate_frame: CoordinateFrame
    agents: list[AgentTrajectory]
    map_features: list[MapFeature]
    traffic_lights: list[TrafficLight]
    route: RouteInfo
    tags: list[dict[str, Any]]
    quality: QualityInfo
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CanonicalScenario":
        """Reconstruct the typed model from decoded JSON."""

        agents = [
            AgentTrajectory(
                **{
                    **agent,
                    "goal": Goal(**agent["goal"]),
                }
            )
            for agent in value["agents"]
        ]
        features = [MapFeature(**feature) for feature in value["map_features"]]
        lights = [TrafficLight(**light) for light in value["traffic_lights"]]
        route_value = value["route"]
        route = RouteInfo(
            **{
                **route_value,
                "goal": Goal(**route_value["goal"]),
            }
        )
        return cls(
            schema_version=value["schema_version"],
            source=SourceInfo(**value["source"]),
            timing=TimingInfo(**value["timing"]),
            coordinate_frame=CoordinateFrame(**value["coordinate_frame"]),
            agents=agents,
            map_features=features,
            traffic_lights=lights,
            route=route,
            tags=value["tags"],
            quality=QualityInfo(**value["quality"]),
        )
