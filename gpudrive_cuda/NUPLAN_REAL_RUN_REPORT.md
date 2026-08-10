# 真实 nuPlan 到 CUDA Simulator 运行报告

## 1. 结论

本次在 NVIDIA Linux 工作站上使用真实 nuPlan mini 数据库和 Las Vegas HD Map，完成了以下端到端流水线：

```text
nuPlan SQLite + HD Map
  -> CanonicalScenario
  -> RuntimeScenario
  -> CUDA simulator
  -> trace.csv / summary.json
  -> rollout.gif / final_frame.png
```

运行使用的场景 token 为 `165060762e765a5a`，不是 mock 数据。Canonical 和 Runtime 验证、C++ runtime loader、CUDA integration test、118 步 CUDA rollout、CSV/JSON 检查以及 GIF/PNG 渲染均通过。

运行日期：2026-08-10（Asia/Shanghai）

## 2. 实际数据

### 2.1 nuPlan 数据库

```text
/home/user/whz/agent/data/dataset/nuPlan/raw/cache/mini/2021.05.12.22.00.38_veh-35_01008_01518.db
```

数据库已确认包含以下核心表：

```text
scene, log, lidar_pc, ego_pose, lidar_box, track, category,
traffic_light_status, scenario_tag
```

### 2.2 真实场景

```text
scene name:          scene-0001
scene token:         165060762e765a5a
location:            las_vegas
log.map_version:     us-nv-las-vegas-strip
raw lidar_pc frames: 239
non-ego tracks:      171
```

### 2.3 地图

地图根目录：

```text
/home/user/whz/agent/data/dataset/nuPlan/maps
```

实际使用的地图文件：

```text
/home/user/whz/agent/data/dataset/nuPlan/maps/nuplan-maps-v1.0.json
/home/user/whz/agent/data/dataset/nuPlan/maps/us-nv-las-vegas-strip/9.15.1915/map.gpkg
```

数据组织与 nuPlan devkit API 存在一个命名差异：数据库的 `log.map_version` 是城市地图 slug，而 `get_maps_api()` 的 `map_version` 参数要求地图元数据 JSON 的文件名 stem。本次增加了本地地图元数据解析，将：

```text
log.location = las_vegas
log.map_version = us-nv-las-vegas-strip
```

转换为 devkit 所需的：

```text
map_version = nuplan-maps-v1.0
map_name = us-nv-las-vegas-strip
```

## 3. 环境

### 3.1 数据转换环境

按工作站要求，原始 nuPlan 转换使用 `gpudrive` Conda 环境：

```text
Python:          3.11.8
nuplan-devkit:   1.1.0
GeoPandas:       1.1.4
Rasterio:        1.4.4
Pyogrio:         0.13.0
nuPlan source:   /data/fc/hugsim/atg/nuplan-devkit
```

nuPlan 源码通过以下环境变量加入 Python 搜索路径：

```bash
export PYTHONPATH=/data/fc/hugsim/atg/nuplan-devkit
```

### 3.2 Runtime、构建和渲染环境

RuntimeScenario、CMake 和渲染使用根目录 `uv run --frozen` 环境：

```text
uv:              0.12.3
uv Python:       3.10.20
CMake:           3.31.10
Ninja:           1.13.0
G++:             9.4.0
CUDA Toolkit:    11.8
NVIDIA driver:   570.133.07
GPU:             4 x NVIDIA GeForce RTX 4090
GPU memory:      24564 MiB per GPU
```

`gpudrive` 环境中另有 CUDA 12.9 `nvcc`。若将整个 Conda 环境 `bin` 目录放在 PATH 最前面，CMake 会选中 CUDA 12.9，并在当前驱动上产生 `unsupported toolchain` PTX 错误。最终通过单独指定 `UV_BIN`、保留系统 PATH 的方式使用 `/usr/local/cuda/bin/nvcc`（CUDA 11.8）。

## 4. 完整复现命令

在仓库根目录执行：

```bash
cd /home/user/whz/agent

export NUPLAN_DB="/home/user/whz/agent/data/dataset/nuPlan/raw/cache/mini/2021.05.12.22.00.38_veh-35_01008_01518.db"
export SCENE_ID="165060762e765a5a"
export NUPLAN_MAPS_ROOT="/home/user/whz/agent/data/dataset/nuPlan/maps"

export REAL_RUN_ROOT="/data/user/agent_runs/nuplan_real_smoke"
export CANONICAL_DIR="${REAL_RUN_ROOT}/canonical"
export RUNTIME_DIR="${REAL_RUN_ROOT}/runtime"
export SIM_OUTPUT_DIR="${REAL_RUN_ROOT}/simulator"

export GPUDRIVE_PYTHON="/home/user/anaconda3/envs/gpudrive/bin/python"
export UV_BIN="/home/user/anaconda3/envs/gpudrive/bin/uv"
export PYTHONPATH="/data/fc/hugsim/atg/nuplan-devkit"

mkdir -p "${CANONICAL_DIR}" "${RUNTIME_DIR}" "${SIM_OUTPUT_DIR}"
```

转换并验证 CanonicalScenario：

```bash
"${GPUDRIVE_PYTHON}" -m scenario_pipeline convert-nuplan \
  --input "${NUPLAN_DB}" \
  --scene-id "${SCENE_ID}" \
  --maps-root "${NUPLAN_MAPS_ROOT}" \
  --output "${CANONICAL_DIR}"

"${GPUDRIVE_PYTHON}" -m scenario_pipeline validate \
  --input "${CANONICAL_DIR}"
```

编译并验证 RuntimeScenario：

```bash
"${UV_BIN}" sync --frozen

"${UV_BIN}" run --frozen python -m scenario_pipeline compile-rl \
  --input "${CANONICAL_DIR}" \
  --output "${RUNTIME_DIR}" \
  --max-agents 192

"${UV_BIN}" run --frozen python -m scenario_pipeline validate-rl \
  --input "${RUNTIME_DIR}"
```

运行 CUDA simulator 和渲染：

```bash
export RUNTIME_SCENE_DIR="${RUNTIME_DIR}/2021_05_12_22_00_38_veh-35_01008_01518_165060762e765a5a"

UV_BIN="${UV_BIN}" \
GPU_SIM_BUILD_DIR="/home/user/whz/agent/gpudrive_cuda/build.cuda11" \
bash gpudrive_cuda/scripts/run_nuplan_demo.sh \
  "${RUNTIME_SCENE_DIR}" \
  "${SIM_OUTPUT_DIR}" \
  1
```

## 5. CanonicalScenario 结果

```text
schema_version:                 1.0
source.dataset:                 nuplan
source_scenario_id:             165060762e765a5a
dt:                             0.1 s
num_steps:                      119
agents:                         172
map_features:                   648
traffic_lights:                 11
route roadblocks:               21
quality.map_available:          true
quality.map_coverage_complete:  true
warnings:                       0
maximum logged displacement:    121.381 m
```

Agent 类型：

```text
vehicle:    38
pedestrian: 62
other:      72
total:      172
```

地图元素：

```text
road_line:              157
lane_connector:         145
lane:                   119
road_edge:               90
roadblock_connector:     52
roadblock:               38
stop_line:               27
walkway:                 11
crosswalk:                9
total:                  648
```

Canonical 验证结果：

```text
summary: succeeded=1 warned=0 failed=0
```

质量检查确认 ego 恰好一个、地图非空、有效轨迹存在位移，所有有效数值均无 NaN/Inf。

## 6. RuntimeScenario 结果

默认 `max_agents=64` 会截断 172 个真实 Agent。最终使用 `--max-agents 192` 重新编译，保留全部 Agent。

```text
schema_version:          rl-runtime-1.0
canonical_schema:        1.0
source_scenario_id:      165060762e765a5a
dt:                      0.1 s
episode_steps:           118
max_agents:              192
agents:                  172
map_features:            648
map_points:              26781
map_edges:               358
traffic_lights:          11
route_features:          8
tensor files:            30
total tensor bytes:      1183305
tensor size mismatches:  0
```

Runtime 验证结果：

```text
summary: succeeded=1 warned=0 failed=0
```

## 7. CUDA Simulator 结果

构建和集成测试：

```text
runtime loader ok: 165060762e765a5a agents=172 map_features=648
CUDA simulator integration ok
```

Rollout：

```text
worlds:                  1
executed_steps:          118
runtime agents:          172
runtime max_agents:      192
initially valid agents:   91
moving valid agents:      91
maximum displacement:     93.4875 m
CUDA errors:               0
```

最大位移 Agent：

```text
world:       0
agent_slot:  31
start:       (25.1128864, 19.6358814)
end:         (-59.80019, -19.4752808)
distance:    93.4875 m
```

CSV 验收：

```text
rows:                              22848
(world, step, agent_slot) sorted:  true
keys unique:                       true
state/action NaN or Inf:           0
```

Simulator 事件累计值：

```text
vehicle collision agent-steps: 1055
road collision agent-steps:     731
offroad agent-steps:           5969
reached goal agent-steps:      6628
```

碰撞和 offroad 是当前 simulator 的诊断事件；按设计只记录，不改变状态，也不提前终止 world。本次 world 正常完成全部 118 步。

## 8. 输出文件验收

工作站输出目录：

```text
/data/user/agent_runs/nuplan_real_smoke/simulator
```

产物：

```text
trace.csv:       2006070 bytes
summary.json:        727 bytes
rollout.gif:     8031498 bytes, GIF, 1000x800, 119 frames
final_frame.png:  324674 bytes, PNG, 1158x1048
```

绝对路径：

```text
/data/user/agent_runs/nuplan_real_smoke/simulator/trace.csv
/data/user/agent_runs/nuplan_real_smoke/simulator/summary.json
/data/user/agent_runs/nuplan_real_smoke/simulator/rollout.gif
/data/user/agent_runs/nuplan_real_smoke/simulator/final_frame.png
```

`summary.json` 可正常解析；GIF 和 PNG 均通过 Pillow 打开验证。最终 PNG 中能看到 lane、lane connector、road line、road edge、Agent、轨迹和 goal，确认真实道路几何进入了渲染结果。

## 9. 本次修复

### 9.1 nuPlan 地图参数解析

文件：`scenario_pipeline/adapters/nuplan.py`

新增 `_resolve_map_api_arguments()`，根据工作站已有的 `nuplan-maps-*.json` 将数据库地图 slug 转换为 devkit API 所需的 metadata version 和 map name。

### 9.2 CUDA 设备指针声明

文件：`gpudrive_cuda/src/drive_sim.cuh`

`drive_sim.cu` 已上传、释放并使用 `d_map_geometry_type_`，但 `DriveSim` 类原本缺少成员声明，导致 CUDA 编译失败。已补充：

```cpp
std::int32_t *d_map_geometry_type_ = nullptr;
```

### 9.3 显式 uv 可执行文件

文件：`gpudrive_cuda/scripts/run_nuplan_demo.sh`

新增 `UV_BIN` 参数，允许单独指定 uv，而不需要将整个 Conda `bin` 目录提前到 PATH，从而避免错误选择 CUDA 12.9 `nvcc`。

## 10. 测试

```text
Canonical validation:           passed
Runtime validation:             passed
runtime_loader_test:            passed
simulator_integration_test:     passed
CUDA rollout:                   passed
CSV finite-value validation:    passed
CSV ordering/uniqueness:        passed
summary.json validation:        passed
GIF Pillow validation:          passed
PNG Pillow validation:          passed
road geometry visual check:     passed
source_scenario_id real check:  passed
git diff --check:               passed
```

Renderer 单元测试：

```text
test_runtime_directory_round_robin ... ok
test_trace_is_filtered_and_grouped_by_world ... ok
test_vehicle_corners_preserve_dimensions ... ok

Ran 3 tests
OK
```

## 11. 已知问题

1. 工作站已有 nuPlan devkit 为 v1.1.0，而复现指南优先建议 v1.2.x。当前真实数据库、地图 API 和端到端流水线已经在 v1.1.0 下通过，因此没有为版本号强制升级。

2. Canonical 中有 13 个 route roadblock 位于起点 150 m 地图裁剪范围外，被记录为 unresolved references；`quality.map_available=true` 且 `quality.map_coverage_complete=true`，不阻塞当前 Runtime 和 CUDA rollout。

3. Runtime 中保留 172 条 Agent 轨迹，其中 91 个 Agent 在 anchor/reset step 有有效状态并参与本次完整 rollout，其余 Agent 不具备有效初始状态，没有被人为生成状态。
