# CUDA nuPlan Simulator

这是基于仓库原有 `gpudrive_cuda` 底座扩展的独立 C++17/CUDA
simulator。它直接读取 `scenario_pipeline compile-rl` 生成的
`RuntimeScenario v1`，不依赖 Madrona，也不读取原始 nuPlan SQLite。

动态自行车模型的完整服务器复现步骤见
[动态自行车模型复现指南](DYNAMIC_BICYCLE_REPRODUCTION.md)。
真实场景的正式可视化流程见
[TerraZero 风格 10 秒 Demo 工作站复现指南](TERRAZERO_STYLE_DEMO_REPRODUCTION.md)。

## 流水线

```text
CanonicalScenario JSON
        │ scenario_pipeline compile-rl
        ▼
RuntimeScenario v1
        │ C++ runtime loader
        ▼
CUDA reset / controller / dynamics / events / observations
        │ drive_sim_cli
        ▼
trace.csv + summary.json
        │ render_trace.py
        ▼
rollout.gif + rollout.mp4 + final_frame.png
```

## 已实现的 Simulator 语义

- 每个 world 独立加载场景、`dt`、episode 长度和 reset step。
- 状态为 `[x, y, yaw, vx, vy]`，动作为 `[acceleration, steering]`。
- 默认控制器优先跟踪 simulator-private `reference_future`，未来缺失时跟踪
  `agent_goal`；未来轨迹不会出现在 policy observation 中。
- `external_control_mask` 可以让任意可控 Agent 改用外部动作。
- 车辆使用带侧偏力、轮胎力饱和和横摆响应的动态自行车模型；低速区与
  运动学模型平滑混合，行人和骑行者使用稳定的运动学回退。
- 动作为目标加速度和目标前轮转角，执行器包含一阶响应、jerk 限制和转向
  速率限制；实际加速度、实际转角、纵横向速度和横摆角速度写入 trace。
- nuPlan 不提供质量和轮胎标定。simulator 根据 `agent_type` 选择模型，并根据
  `length/width` 为车辆选择可配置的乘用车或大型车参数预设。
- 车辆碰撞使用二维 OBB SAT；有地图时检测 road edge 和 drivable polygon
  越界；碰撞只记录，不改变车辆状态，也不结束 world。
- 输出 self、最近 16 个 partner、最近 64 个 map segment 和当前交通灯观测。
- `world_done` 只由 episode timeout 产生，不在 simulator 内计算 reward。

## 服务器环境

已有真实 nuPlan 数据的工作站可直接使用
[真实 nuPlan 工作站复现指南](WORKSTATION_NUPLAN_REPRODUCTION.md)，让 Codex 自动定位数据库和地图并完成端到端验收。

系统需要：

```text
NVIDIA GPU + CUDA toolkit（需要 nvcc）
C++17 compiler
```

Python 3.10、CMake、Ninja、Matplotlib、Pillow 和 imageio-ffmpeg 由仓库根目录的 uv
环境统一管理。新服务器先安装 uv：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

cd /path/to/agent
bash gpudrive_cuda/scripts/setup_uv_env.sh
```

`uv sync --frozen` 会严格使用仓库提交的 `uv.lock` 创建 `.venv`。
`nlohmann/json` 已在仓库 `gpudrive/external/json` 中，不需要额外下载。

这个 uv 环境覆盖当前 `RuntimeScenario -> CUDA simulator -> GIF/MP4` 阶段。
原始 nuPlan HD map 提取所需的 `nuplan-devkit`，以及 Waymo TFRecord 转换所需的
TensorFlow/Waymo SDK 暂未放入该环境；当前 mock runtime 和 CUDA simulator
运行不需要它们，后续处理真实原始数据时再建立独立数据转换环境。

## 一键生成 10 秒真实 nuPlan Demo

正式展示需要地图非空且至少包含10秒future的真实RuntimeScenario。在仓库根目录执行：

```bash
export RUNTIME_SCENE_DIR="/data/$USER/runtime/real_nuplan_scene"

bash gpudrive_cuda/scripts/run_nuplan_demo.sh \
  "$RUNTIME_SCENE_DIR" \
  outputs/nuplan_cuda_demo \
  1
```

输出：

```text
outputs/nuplan_cuda_demo/trace.csv
outputs/nuplan_cuda_demo/summary.json
outputs/nuplan_cuda_demo/rollout.gif
outputs/nuplan_cuda_demo/rollout.mp4
outputs/nuplan_cuda_demo/final_frame.png
```

`trace.csv` 中的 `acceleration/steering` 是限幅后的控制命令，
`actual_acceleration/actual_steering` 是经过执行器响应后真正进入动力学模型的
状态。`longitudinal_velocity/lateral_velocity/yaw_rate` 用于动力学验收；renderer
采用16:9深色局部跟随视图，并忽略这些附加列。

当前 mock runtime 的 `map_features=0` 且未来只有9.5秒，只适合短时功能测试，
不适合正式可视化。用带 `--maps-root` 重新转换并编译的runtime会自动显示道路面、
车道线、道路边界、交通灯和多类型Agent，不需要修改simulator。

## 分步运行

构建：

```bash
uv run --frozen cmake -S gpudrive_cuda -B gpudrive_cuda/build -G Ninja -DBUILD_TESTING=ON
uv run --frozen cmake --build gpudrive_cuda/build -j
```

运行全部自动控制 Agent：

```bash
gpudrive_cuda/build/drive_sim_cli \
  --runtime dataset/nuplan/rl_runtime/mock_nuplan_00100000 \
  --worlds 2 \
  --output outputs/nuplan_cuda_demo
```

让 slot 1 使用恒定外部动作：

```bash
gpudrive_cuda/build/drive_sim_cli \
  --runtime dataset/nuplan/rl_runtime/mock_nuplan_00100000 \
  --worlds 1 \
  --external-agent 1 \
  --acceleration 1.0 \
  --steering 0.15 \
  --output outputs/nuplan_cuda_external
```

渲染：

```bash
uv run --frozen python gpudrive_cuda/tools/render_trace.py \
  --runtime "$RUNTIME_SCENE_DIR" \
  --trace outputs/nuplan_cuda_demo/trace.csv \
  --output outputs/nuplan_cuda_demo/rollout.gif \
  --mp4-output outputs/nuplan_cuda_demo/rollout.mp4 \
  --final-png outputs/nuplan_cuda_demo/final_frame.png \
  --duration 10 \
  --view follow
```

## 测试

CUDA 服务器端集成测试：

```bash
bash gpudrive_cuda/scripts/test_current.sh \
  dataset/nuplan/rl_runtime/mock_nuplan_00100000
```

主机侧 loader 和 Python renderer 单测：

```bash
gpudrive_cuda/build/runtime_loader_test \
  dataset/nuplan/rl_runtime/mock_nuplan_00100000
uv run --frozen python -m unittest gpudrive_cuda.tests.test_render_trace -v
```

## 后续接 ADRL

训练层只需要调用 `set_external_control_mask()` 选择攻击 Agent，按
`[world, agent]` 写入 `AgentAction`，然后读取 observation 和 event buffer。
目前没有 Python binding；后续可以在保持 `DriveSim` C++ API 不变的前提下
增加 pybind11 或 PyTorch CUDA tensor 适配层。
