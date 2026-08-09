"""Small protobuf-shaped objects used without the heavy Waymo dependency."""

from __future__ import annotations

from types import SimpleNamespace


class Feature(SimpleNamespace):
    def WhichOneof(self, _: str) -> str:
        return self.feature_type


def point(x: float, y: float, z: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=z)


def state(index: int, lateral: float = 0.0, valid: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        center_x=float(index), center_y=lateral, center_z=0.0,
        length=4.8, width=2.0, height=1.6,
        heading=0.0, velocity_x=10.0, velocity_y=0.0, valid=valid,
    )


def build_scenario() -> SimpleNamespace:
    timestamps = [index * 0.1 for index in range(91)]
    ego = SimpleNamespace(id=10, object_type=1, states=[state(index) for index in range(91)])
    other = SimpleNamespace(id=20, object_type=99, states=[state(index, 4.0) for index in range(91)])
    lane = SimpleNamespace(
        type=2,
        speed_limit_mph=30.0,
        polyline=[point(0, 0), point(100, 0)],
        entry_lanes=[], exit_lanes=[101],
        left_neighbors=[], right_neighbors=[],
    )
    crosswalk = SimpleNamespace(polygon=[point(20, -3), point(20, 3), point(25, 3), point(25, -3)])
    features = [
        Feature(id=100, feature_type="lane", lane=lane),
        Feature(id=200, feature_type="crosswalk", crosswalk=crosswalk),
    ]
    dynamic_states = []
    for index in range(91):
        lane_state = SimpleNamespace(
            lane=100,
            state=4 if index < 45 else 6,
            stop_point=point(45, 0),
        )
        dynamic_states.append(SimpleNamespace(lane_states=[lane_state]))
    return SimpleNamespace(
        scenario_id="fake-waymo-001",
        timestamps_seconds=timestamps,
        current_time_index=10,
        tracks=[ego, other],
        sdc_track_index=0,
        objects_of_interest=[20],
        tracks_to_predict=[SimpleNamespace(track_index=1, difficulty=1)],
        map_features=features,
        dynamic_map_states=dynamic_states,
    )
