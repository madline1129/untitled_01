# CUDA nuPlan Simulator

这是基于仓库原有 `gpudrive_cuda` 底座扩展的独立 C++17/CUDA
simulator。它直接读取 `scenario_pipeline compile-rl` 生成的
`RuntimeScenario v1`，不依赖 Madrona，也不读取原始 nuPlan SQLite。

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
rollout.gif + final_frame.png
```

## 已实现的 Simulator 语义

- 每个 world 独立加载场景、`dt`、episode 长度和 reset step。
- 状态为 `[x, y, yaw, vx, vy]`，动作为 `[acceleration, steering]`。
- 默认控制器优先跟踪 simulator-private `reference_future`，未来缺失时跟踪
  `agent_goal`；未来轨迹不会出现在 policy observation 中。
- `external_control_mask` 可以让任意可控 Agent 改用外部动作。
- 运动学自行车模型包含加速度、转角和速度限制。
- 车辆碰撞使用二维 OBB SAT；有地图时检测 road edge 和 drivable polygon
  越界；碰撞只记录，不改变车辆状态，也不结束 world。
- 输出 self、最近 16 个 partner、最近 64 个 map segment 和当前交通灯观测。
- `world_done` 只由 episode timeout 产生，不在 simulator 内计算 reward。

## 服务器环境

需要：

```text
NVIDIA GPU + CUDA toolkit（需要 nvcc）
CMake >= 3.24
C++17 compiler
Python >= 3.10
matplotlib
Pillow
```

Python 渲染依赖可以安装为：

```bash
python3 -m pip install matplotlib Pillow
```

`nlohmann/json` 已在仓库 `gpudrive/external/json` 中，不需要额外下载。

## 一键运行 mock nuPlan

在仓库根目录执行：

```bash
bash gpudrive_cuda/scripts/run_nuplan_demo.sh \
  dataset/nuplan/rl_runtime/mock_nuplan_00100000 \
  outputs/nuplan_cuda_demo \
  1
```

输出：

```text
outputs/nuplan_cuda_demo/trace.csv
outputs/nuplan_cuda_demo/summary.json
outputs/nuplan_cuda_demo/rollout.gif
outputs/nuplan_cuda_demo/final_frame.png
```

当前 mock runtime 的 `map_features=0`，因此动画会显示车辆、目标和轨迹，
但没有道路底图。用带 `--maps-root` 重新转换并编译的 runtime 会自动显示地图，
不需要修改 simulator。

## 分步运行

构建：

```bash
cmake -S gpudrive_cuda -B gpudrive_cuda/build -DBUILD_TESTING=ON
cmake --build gpudrive_cuda/build -j
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
python3 gpudrive_cuda/tools/render_trace.py \
  --runtime dataset/nuplan/rl_runtime/mock_nuplan_00100000 \
  --trace outputs/nuplan_cuda_demo/trace.csv \
  --output outputs/nuplan_cuda_demo/rollout.gif
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
python3 -m unittest gpudrive_cuda.tests.test_render_trace -v
```

## 后续接 ADRL

训练层只需要调用 `set_external_control_mask()` 选择攻击 Agent，按
`[world, agent]` 写入 `AgentAction`，然后读取 observation 和 event buffer。
目前没有 Python binding；后续可以在保持 `DriveSim` C++ API 不变的前提下
增加 pybind11 或 PyTorch CUDA tensor 适配层。
