"""生成无需 nuPlan devkit 的 10 秒 MAPPO synthetic RuntimeScenario。"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from scenario_pipeline.models import (
    AgentTrajectory,
    CanonicalScenario,
    CoordinateFrame,
    Goal,
    MapFeature,
    QualityInfo,
    RouteInfo,
    SourceInfo,
    TimingInfo,
)
from scenario_pipeline.runtime import RuntimeConfig, compile_runtime_scenario, write_runtime_scenario


def _agent(identifier: str, is_ego: bool, start_x: float, y: float, speed: float) -> AgentTrajectory:
    steps = 101
    timestamps = [0.1 * index for index in range(steps)]
    x = [start_x + speed * timestamp for timestamp in timestamps]
    return AgentTrajectory(
        id=identifier,
        type="vehicle",
        is_ego=is_ego,
        length=4.8,
        width=1.9,
        height=1.6,
        x=x,
        y=[y] * steps,
        yaw=[0.0] * steps,
        vx=[speed] * steps,
        vy=[0.0] * steps,
        valid=[True] * steps,
        goal=Goal(True, x[-1], y, "synthetic_terminal_position"),
        roles={"is_sdc": True} if is_ego else {},
    )


def synthetic_scenario() -> CanonicalScenario:
    steps = 101
    timestamps = [0.1 * index for index in range(steps)]
    agents = [
        _agent("ego", True, 0.0, 0.0, 6.0),
        _agent("attacker_1", False, 8.0, -3.5, 6.0),
        _agent("attacker_2", False, 18.0, 0.1, 5.7),
        _agent("attacker_3", False, 12.0, 3.5, 5.8),
        _agent("attacker_4", False, 28.0, -2.0, 5.5),
    ]
    map_features = [
        MapFeature(
            id="road_surface",
            type="roadblock",
            geometry_type="polygon",
            geometry=[[-20.0, -7.0, 0.0], [90.0, -7.0, 0.0], [90.0, 7.0, 0.0], [-20.0, 7.0, 0.0]],
            speed_limit_mps=15.0,
        ),
        MapFeature(
            id="left_edge",
            type="road_edge",
            geometry_type="polyline",
            geometry=[[-20.0, 7.0, 0.0], [90.0, 7.0, 0.0]],
        ),
        MapFeature(
            id="right_edge",
            type="road_edge",
            geometry_type="polyline",
            geometry=[[-20.0, -7.0, 0.0], [90.0, -7.0, 0.0]],
        ),
        MapFeature(
            id="center_lane",
            type="lane",
            geometry_type="polyline",
            geometry=[[-20.0, 0.0, 0.0], [90.0, 0.0, 0.0]],
            speed_limit_mps=15.0,
        ),
    ]
    return CanonicalScenario(
        source=SourceInfo("nuplan", "synthetic", "mock_nuplan_00100000"),
        timing=TimingInfo(0.1, timestamps, steps, anchor_index=0),
        coordinate_frame=CoordinateFrame(
            "local_cartesian", "x_forward_y_left_z_up", 0.0, 0.0, 0.0, 0.0
        ),
        agents=agents,
        map_features=map_features,
        traffic_lights=[],
        route=RouteInfo(
            True,
            roadblock_ids=["road_surface"],
            lane_ids=["center_lane"],
            goal=Goal(True, 60.0, 0.0, "synthetic_route_goal"),
        ),
        tags=[{"type": "synthetic_mappo_smoke", "timestamp": 0.0, "agent_id": None}],
        quality=QualityInfo(True, True),
    )


def create_runtime(output: Path, force: bool = False) -> Path:
    if (output / "manifest.json").is_file() and not force:
        return output
    if output.exists() and force:
        shutil.rmtree(output)
    runtime = compile_runtime_scenario(synthetic_scenario(), RuntimeConfig())
    write_runtime_scenario(runtime, output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/nuplan/rl_runtime/mock_nuplan_00100000"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = create_runtime(args.output, args.force)
    print(f"mock runtime: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
