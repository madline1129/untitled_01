# 真实 nuPlan 工作站复现指南

本文档用于让工作站上的 Codex 自动识别已有 nuPlan 数据目录，并将一个真实场景完整运行到 CUDA simulator。它适用于工作站上的数据组织方式与本仓库示例路径不一致的情况。

## 使用方式

1. 在 NVIDIA Linux 工作站拉取仓库最新 `main` 分支。
2. 进入仓库根目录并启动 Codex。
3. 将下方“Codex 执行提示”完整发送给 Codex。
4. Codex 完成后，按文末验收清单核对结果。

## Codex 执行提示

```text
你现在位于一台 NVIDIA Linux 工作站上的本仓库根目录。仓库包含：

- scenario_pipeline：nuPlan SQLite -> CanonicalScenario -> RuntimeScenario
- gpudrive_cuda：RuntimeScenario -> CUDA simulator -> CSV/JSON -> GIF/PNG

请不要只给方案，直接在当前工作站把“真实 nuPlan 数据 -> CUDA simulator”完整复现并运行成功。

目标流水线：

真实 nuPlan SQLite + HD Map
  -> CanonicalScenario
  -> RuntimeScenario
  -> CUDA simulator
  -> trace.csv / summary.json
  -> rollout.gif / final_frame.png

约束：

1. 先阅读仓库代码和 README，再判断数据组织形式，不要假设工作站目录与示例完全一致。
2. 不要重新下载 nuPlan；工作站上已经有数据。
3. 不要删除、移动或修改原始数据集。
4. 不要执行 git clean -fdx、rm -rf 数据目录或其他可能删除数据的命令。
5. 优先通过命令行参数、环境变量或软链接适配现有目录。
6. 先用一个数据库中的一个真实 scene 跑通，不要一开始转换完整数据集。
7. 不要退回 mock 数据，也不要重新实现 toy simulator。
8. 可以小范围修改仓库代码和脚本来解决真实兼容问题，但要解释原因并完成验证。
9. 不要自动提交或推送 Git，完成后先报告 diff。
10. raw nuPlan 转换和 CUDA simulator 可以使用两个独立 Python 环境。如果 nuplan-devkit 与根目录 Python 3.10 环境不兼容，不要强行混入同一个环境。

请持续工作到真实场景完整跑通。只有缺少数据、权限或硬件时才停止，并给出已验证的阻塞证据。

第一阶段：检查仓库

执行：

git status --short
git branch --show-current
git log -1 --oneline
find . -maxdepth 3 \( -name pyproject.toml -o -name uv.lock -o -name CMakeLists.txt \) -print
sed -n '1,220p' scenario_pipeline/README.md
sed -n '1,240p' gpudrive_cuda/README.md

确认这些入口存在：

python -m scenario_pipeline convert-nuplan
python -m scenario_pipeline compile-rl
python -m scenario_pipeline validate-rl
gpudrive_cuda/scripts/setup_uv_env.sh
gpudrive_cuda/scripts/run_nuplan_demo.sh

如果根目录缺少 pyproject.toml、uv.lock 或 UV 环境脚本，根据当前代码的真实 import 补充最小环境。当前 simulator 阶段至少需要 Python 3.10、CMake、Ninja、Matplotlib 和 Pillow。不要进行无关依赖升级。

第二阶段：定位真实 nuPlan 数据

优先检查 /data、/datasets、/mnt 和 $HOME，避免扫描 /proc、/sys 等虚拟目录：

find /data /datasets /mnt "$HOME" \
  -type f -name "*.db" 2>/dev/null | head -n 200

find /data /datasets /mnt "$HOME" \
  -type f -name "map.gpkg" 2>/dev/null | head -n 100

find /data /datasets /mnt "$HOME" \
  -type f -name "nuplan-maps-*.json" 2>/dev/null | head -n 100

不要把任意 SQLite 文件当作 nuPlan 数据库。使用 sqlite3 或 Python sqlite3 检查候选文件，确认至少存在以下表：

scene
log
lidar_pc
ego_pose
lidar_box
track
category
traffic_light_status
scenario_tag

查询候选数据库：

SELECT location, map_version, COUNT(*) FROM log GROUP BY location, map_version;
SELECT lower(hex(token)), name FROM scene LIMIT 10;

最终确定：

- NUPLAN_DB：一个真实 nuPlan .db 文件
- SCENE_ID：该数据库中一个真实 scene token 或 scene name
- NUPLAN_MAPS_ROOT：包含地图元数据 JSON 和城市目录的地图根目录

地图根目录通常类似：

maps/
├── nuplan-maps-v1.0.json
├── us-ma-boston/<version>/map.gpkg
├── us-nv-las-vegas-strip/<version>/map.gpkg
├── us-pa-pittsburgh-hazelwood/<version>/map.gpkg
└── sg-one-north/<version>/map.gpkg

必须以工作站实际结构为准。确认数据库中的 log.location 和 log.map_version 能对应地图目录。如果不能对应，判断是根目录层级、地图版本还是 devkit API 参数问题。

第三阶段：检查环境

执行：

nvidia-smi
nvcc --version
g++ --version
cmake --version || true
uv --version || true
python3 --version
df -h

如果 uv 不存在，按 Astral 官方方式安装。CUDA Toolkit 和 NVIDIA 驱动必须使用系统环境，不要尝试用 Python 包代替 nvcc。

检查是否已有 nuplan-devkit 环境：

conda env list 2>/dev/null || true
find "$HOME" /opt /data -maxdepth 4 -type d -name "nuplan-devkit" 2>/dev/null

优先复用已有 nuplan-devkit v1.2.x 环境。若必须新建，为 raw-data conversion 建立独立环境；根目录 UV 环境继续用于 RuntimeScenario、CMake、CUDA CLI 和渲染。

第四阶段：转换一个真实场景

输出放在仓库外的数据盘，或确保目录已被 .gitignore 忽略：

export REAL_RUN_ROOT="/data/$USER/agent_runs/nuplan_real_smoke"
export CANONICAL_DIR="$REAL_RUN_ROOT/canonical"
export RUNTIME_DIR="$REAL_RUN_ROOT/runtime"
export SIM_OUTPUT_DIR="$REAL_RUN_ROOT/simulator"

mkdir -p "$CANONICAL_DIR" "$RUNTIME_DIR" "$SIM_OUTPUT_DIR"

设置第二阶段找到的真实路径：

export NUPLAN_DB="/actual/path/to/one.db"
export SCENE_ID="actual_scene_token_or_name"
export NUPLAN_MAPS_ROOT="/actual/path/to/maps"

在能够 import nuplan-devkit 的环境中，从仓库根目录运行：

python -m scenario_pipeline convert-nuplan \
  --input "$NUPLAN_DB" \
  --scene-id "$SCENE_ID" \
  --maps-root "$NUPLAN_MAPS_ROOT" \
  --output "$CANONICAL_DIR"

python -m scenario_pipeline validate --input "$CANONICAL_DIR"

如果使用独立 nuPlan 环境，确保仓库根目录位于 PYTHONPATH，或从仓库根目录运行。转换后读取生成的 JSON，并明确验证：

- agents 数量大于 1
- ego 恰好一个
- timing.num_steps 大于 1
- quality.map_available 为 true
- map_features 非空
- 至少一个有效 Agent 轨迹存在位移
- 所有有效数值没有 NaN/Inf
- traffic_lights 可以为空；非空时 lane 关联和时间长度必须合理

如果 quality.map_available=false，不要直接忽略警告。定位地图加载失败原因。只有确认工作站缺少对应地图时，才允许先做无地图诊断运行，并在最终报告中明确说明。

第五阶段：生成 RuntimeScenario

回到仓库根目录 UV 环境：

uv sync --frozen

uv run --frozen python -m scenario_pipeline compile-rl \
  --input "$CANONICAL_DIR" \
  --output "$RUNTIME_DIR"

uv run --frozen python -m scenario_pipeline validate-rl \
  --input "$RUNTIME_DIR"

找到实际生成的场景目录：

find "$RUNTIME_DIR" -name manifest.json -print

将包含 manifest.json 的目录设置为：

export RUNTIME_SCENE_DIR="/actual/generated/runtime/scene_directory"

检查 manifest 中的 schema_version、dt、num_steps、source_scenario_id 和各项容量。确认所有 .bin 的文件大小与 manifest dtype/shape 一致。

第六阶段：运行 CUDA simulator

优先运行一键脚本：

bash gpudrive_cuda/scripts/run_nuplan_demo.sh \
  "$RUNTIME_SCENE_DIR" \
  "$SIM_OUTPUT_DIR" \
  1

如果脚本失败，直接定位并修复 CMake、CUDA、路径、runtime loader 或真实容量兼容问题，不要改回 mock 数据。

必要时手动编译和测试：

uv run --frozen cmake \
  -S gpudrive_cuda \
  -B gpudrive_cuda/build \
  -G Ninja \
  -DBUILD_TESTING=ON

uv run --frozen cmake --build gpudrive_cuda/build -j

gpudrive_cuda/build/runtime_loader_test "$RUNTIME_SCENE_DIR"
gpudrive_cuda/build/simulator_integration_test "$RUNTIME_SCENE_DIR"

然后运行 drive_sim_cli 和 gpudrive_cuda/tools/render_trace.py，生成 CSV、JSON、GIF 和 PNG。

第七阶段：验收

必须检查：

- runtime_loader_test 通过
- simulator_integration_test 通过
- CUDA rollout 完成且没有 CUDA error
- trace.csv 存在且非空
- summary.json 是合法 JSON
- rollout.gif 和 final_frame.png 能被 Pillow 打开
- 至少一辆有效车辆的终点位置不同于起点
- 状态和动作没有 NaN/Inf
- CSV 的 world、step、agent_slot 排序合理
- 地图存在时 final_frame.png 中包含道路几何
- source_scenario_id 对应真实 scene，不是 mock_nuplan

使用 Python 读取 trace.csv，按 world 和 agent_slot 比较第一帧与最后一帧位置，并打印最大位移。

如果修改了代码，最后执行：

git diff --check
git status --short
git diff --stat

最终报告必须包含：

1. 实际 nuPlan 数据库绝对路径
2. 实际地图根目录
3. 数据库文件、scene ID、城市和地图版本
4. Python、nuplan-devkit、CUDA、CMake 版本
5. 从转换到渲染的完整可复现命令
6. CanonicalScenario 和 RuntimeScenario 的关键统计
7. simulator 运行步数、Agent 数量和最大位移
8. trace.csv、summary.json、GIF、PNG 的绝对路径
9. 实际数据组织与仓库示例的差异及适配方式
10. 修改文件列表和 git diff 概要
11. 尚未解决的问题
```

## 人工验收清单

工作站 Codex 完成后，至少应提供以下四个文件：

```text
<SIM_OUTPUT_DIR>/trace.csv
<SIM_OUTPUT_DIR>/summary.json
<SIM_OUTPUT_DIR>/rollout.gif
<SIM_OUTPUT_DIR>/final_frame.png
```

重点确认 `summary.json` 和 `RuntimeScenario/manifest.json` 中的 `source_scenario_id` 来自真实数据库，而不是 `mock_nuplan`。若地图加载成功，CanonicalScenario 中应满足：

```text
quality.map_available = true
len(map_features) > 0
```

首次复现只要求一个真实场景和一个 CUDA world。跑通后再扩大到多个场景和多个 world，以便将环境问题、数据兼容问题和批处理容量问题分开定位。
