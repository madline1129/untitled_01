# nuPlan 到 Simulator 的数据格式

```text
nuPlan DB + Map
    -> CanonicalScenario
    -> RuntimeScenario (`rl-runtime-1.0`)
    -> C++ `RuntimeScene`
    -> CUDA `DriveSim`
```

每个场景对应一个目录：

```text
scene_id/
├── manifest.json
├── agent_initial_state.bin
├── reference_future.bin
├── map_points.bin
├── traffic_light_state.bin
└── ...
```

`manifest.json` 保存场景来源、`dt`、episode 长度、容量、有效数量，以及每个
`.bin` 的文件名、dtype 和 shape。`.bin` 没有文件头，统一使用
`C-contiguous little-endian` 排列。

## 通用约定

- 坐标系：以 reset 时 ego 为局部原点，`x` 向前、`y` 向左、`z` 向上。
- Agent 状态：`[x, y, yaw, vx, vy]`。
- Agent 尺寸：`[length, width, height]`，单位为米。
- `float32` 和 `int32` 占 4 字节，`uint8` 占 1 字节。
- slot 0 固定为有效 ego；其余 Agent 按 reset 时与 ego 的距离排序。
- 所有张量使用固定容量并进行 padding，必须结合 `*_valid` mask 使用。

下面用这些符号表示容量：

```text
A = max_agents             F = max_future_steps
M = max_map_features       P = max_map_points
E = max_map_edges          L = max_traffic_lights
R = max_route_features
```

## Simulator 输入张量

| 张量 | dtype / shape | 含义 |
| --- | --- | --- |
| `agent_initial_state` | `float32 [A,5]` | reset 时状态 |
| `agent_initial_valid` | `uint8 [A]` | 有效 Agent mask |
| `agent_type` | `int32 [A]` | Agent 类型 |
| `agent_is_ego` | `uint8 [A]` | ego mask |
| `agent_controllable` | `uint8 [A]` | 可控制 Agent mask |
| `agent_dimensions` | `float32 [A,3]` | 长、宽、高 |
| `agent_goal` | `float32 [A,2]` | simulator 私有目标点 |
| `agent_goal_valid` | `uint8 [A]` | 目标有效 mask |
| `reference_future` | `float32 [F,A,5]` | 自动控制器私有参考轨迹 |
| `reference_future_valid` | `uint8 [F,A]` | 参考轨迹 mask |
| `map_points` | `float32 [P,3]` | 展平后的局部地图点 |
| `map_point_valid` | `uint8 [P]` | 地图点 mask |
| `map_feature_type` | `int32 [M]` | lane、road edge 等类型 |
| `map_geometry_type` | `int32 [M]` | point、polyline 或 polygon |
| `map_feature_point_start` | `int32 [M]` | feature 在 `map_points` 中的起点 |
| `map_feature_point_count` | `int32 [M]` | feature 的点数量 |
| `map_feature_valid` | `uint8 [M]` | 地图 feature mask |
| `map_speed_limit` | `float32 [M]` | 速度限制，单位 m/s |
| `map_speed_limit_valid` | `uint8 [M]` | 速度限制 mask |
| `map_edges` | `int32 [E,3]` | `[source, target, relation]` |
| `map_edge_valid` | `uint8 [E]` | 地图拓扑边 mask |
| `traffic_light_feature_index` | `int32 [L]` | 信号灯关联的地图 feature |
| `traffic_light_state` | `uint8 [F+1,L]` | 从 reset 开始的信号状态表 |
| `traffic_light_valid` | `uint8 [F+1,L]` | 信号状态 mask |
| `route_feature_index` | `int32 [R]` | 有序路线 feature 索引 |
| `route_feature_valid` | `uint8 [R]` | 路线索引 mask |
| `route_goal` | `float32 [2]` | ego 路线终点 `[x,y]` |
| `route_goal_valid` | `uint8 [1]` | 路线终点 mask |

`agent_history [H,A,5]` 和 `agent_history_valid [H,A]` 仍保存在
RuntimeScenario 中并接受文件完整性检查，但当前 `DriveSim` 从 reset 状态开始，
不会把历史轨迹上传到 CUDA。

## 枚举值

```text
agent_type: 0 unknown, 1 vehicle, 2 pedestrian, 3 cyclist, 4 other
geometry:   0 unknown, 1 point, 2 polyline, 3 polygon
map_type:   0 unknown, 1 lane, 2 lane_connector, 3 road_line,
            4 road_edge, 5 crosswalk, 6 stop_line, 7 walkway,
            8 roadblock, 9 roadblock_connector
edge:       1 predecessor, 2 successor, 3 left_neighbor, 4 right_neighbor
signal:     0 unknown, 1 stop, 2 caution, 3 go
```

`reference_future`、完整交通灯时间表和其他 Agent 的 `agent_goal` 只供 simulator
内部使用，不允许直接进入策略 observation。
