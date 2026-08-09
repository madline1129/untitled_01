# RuntimeScenario v1

`RuntimeScenario` 是 `CanonicalScenario` 与 C++/CUDA RL 环境之间的编译格式。
每个场景是一个目录，包含一个 `manifest.json` 和若干 little-endian 连续
二进制张量。二进制文件没有文件头，dtype 和 shape 以 manifest 为准。

## 数据边界

```text
CanonicalScenario JSON
        │ compile-rl
        ▼
RuntimeScenario directory
        │ load / mmap / cudaMemcpy
        ▼
C++/CUDA environment reset
```

`agent_initial_state` 是环境 reset 输入。`agent_history` 可以进入 policy
observation。`reference_future` 只用于回放、reward 或评估，manifest 中固定
标记 `policy_visible=false`，不允许作为 policy observation。

## Agent 张量

状态字段顺序固定为：

```text
[x, y, yaw, vx, vy]
```

主要张量：

| 名称 | Shape | 用途 |
| --- | --- | --- |
| `agent_initial_state` | `[A, 5]` | reset 状态 |
| `agent_initial_valid` | `[A]` | 有效 agent slot |
| `agent_history` | `[H, A, 5]` | 截止 reset 的历史 |
| `agent_history_valid` | `[H, A]` | 历史 mask |
| `reference_future` | `[F, A, 5]` | reset 后真实参考轨迹 |
| `reference_future_valid` | `[F, A]` | 未来轨迹 mask |
| `agent_dimensions` | `[A, 3]` | length、width、height |
| `agent_goal` | `[A, 2]` | simulator-private 局部目标点 |

slot 0 固定为 ego。其他 agent 先按 reset 时与 ego 的距离排序，再做 padding
或截断。无效 slot 的数值为 0，必须结合对应 valid mask 使用。

完整的 `agent_goal` 不直接向 policy 暴露，避免 ego 读取其他车辆由真实终点
推导出的意图。observation builder 只能按当前受控 agent 提取它自己的目标；
ego 的公开任务路线使用 `route_goal`。

## 地图和交通灯

地图点使用 `map_points[M, 3]` 连续保存。每个 feature 通过
`map_feature_point_start` 和 `map_feature_point_count` 引用自己的点区间。
`map_edges[E, 3]` 保存：

```text
[source_feature_index, target_feature_index, relation_type]
```

交通灯使用 `[F + 1, L]` 的状态时间表供 simulator 推进。完整时间表不向
policy 暴露；policy 应只观察 simulator 当前 step 公开的信号状态。

## 默认容量

```text
agents=64
history=11
future=128
map_features=2048
map_points=32768
map_edges=8192
traffic_lights=128
route_features=512
```

容量可以通过 `compile-rl` 参数覆盖。超出容量时按确定性规则截断，并写入
manifest 的 `warnings`，不会静默丢弃。

## 生成与验证

```bash
python -m scenario_pipeline compile-rl \
  --input dataset/nuplan/canonical \
  --output dataset/nuplan/rl_runtime

python -m scenario_pipeline validate-rl \
  --input dataset/nuplan/rl_runtime
```
