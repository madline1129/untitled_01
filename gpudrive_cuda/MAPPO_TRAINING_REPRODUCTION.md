# 多智能体 MAPPO v0 工作站复现指南

单张 RTX 4090 24GB 请优先使用
[RTX 4090 MAPPO 训练指南](RTX4090_MAPPO_TRAINING_GUIDE.md)，其中包含专用配置、
显存策略、完整 TensorBoard 指标和 SSH 访问方法。
真实训练场景不足时，先使用
[nuPlan 批量场景转换到 RTX 4090 训练指南](NUPLAN_BATCH_CONVERSION_AND_TRAINING.md)。

本文覆盖当前独立 CUDA simulator 的强化学习训练流水线：

```text
RuntimeScenario
  -> Torch CUDA bridge
  -> 最多 16 辆攻击车
  -> 参数共享 actor + 集中式 critic
  -> MAPPO checkpoint
  -> CSV / JSON / GIF / MP4
```

第一版 ego 使用 simulator 内部参考轨迹控制器，不主动避障。训练结果用于验证攻击训练闭环，不等价于攻破真实自动驾驶系统。

## 1. 工作站要求

- Linux x86_64、NVIDIA GPU 和可用的 `nvidia-smi`
- 包含 `nvcc` 的 CUDA Toolkit
- CMake 3.24 以上、Ninja、C++17 编译器
- Python 3.10、uv 0.12 以上

```bash
nvidia-smi
nvcc --version
cmake --version
uv --version
```

进入仓库并同步锁定环境：

```bash
cd /path/to/agent
uv sync --frozen --group train
```

`train` group 包含 PyTorch、NumPy、TensorBoard 和 pybind11。不要再用系统 Python 安装另一份 PyTorch，否则 CMake 可能链接到错误的 Torch。
Linux 锁文件沿用仓库原有的官方 PyTorch `cu124` 索引，建议工作站使用 CUDA Toolkit 12.4；至少要保证 `torch.version.cuda` 与本机 `nvcc` 主版本一致。

## 2. 编译 Torch CUDA 扩展

```bash
./gpudrive_cuda/scripts/build_torch_extension.sh
```

脚本会自动读取 uv 环境中的 Torch 与 pybind11 CMake 路径。自动识别 GPU 架构失败时可以显式指定：

```bash
CMAKE_CUDA_ARCHITECTURES=89 \
  ./gpudrive_cuda/scripts/build_torch_extension.sh
```

编译后在当前终端设置：

```bash
export PYTHONPATH="$PWD/build/gpudrive_cuda_rl/python:$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

检查扩展：

```bash
uv run --frozen --group train python - <<'PY'
import torch
from gpudrive_cuda_torch import TorchDriveSim

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0))
print("extension:", TorchDriveSim)
PY
```

## 3. 测试

全新 clone 不包含被 `.gitignore` 忽略的 runtime 二进制，先生成 synthetic mock：

```bash
uv run --frozen python -m gpudrive_cuda.rl.create_mock_runtime
```

先运行纯 Python 测试：

```bash
uv run --frozen --group train python -m unittest \
  gpudrive_cuda.tests.test_rl -v
```

再运行 CUDA simulator 测试：

```bash
build/gpudrive_cuda_rl/runtime_loader_test \
  dataset/nuplan/rl_runtime/mock_nuplan_00100000

build/gpudrive_cuda_rl/simulator_integration_test \
  dataset/nuplan/rl_runtime/mock_nuplan_00100000
```

## 4. Mock 过拟合实验

一键运行：

```bash
./gpudrive_cuda/scripts/run_mappo_mock.sh
```

该脚本会在缺失时自动生成带道路和 4 辆背景车的 10 秒 synthetic runtime，
再执行 32 个重复 mock world、约 20 万 world-step 的训练，最后确定性评估并渲染动画。输出：

```text
outputs/mappo_mock/
  checkpoints/final.pt
  metrics.jsonl
  tensorboard/
  evaluation/eval_trace.csv
  evaluation/eval_summary.json
  evaluation/rollout.gif
  evaluation/rollout.mp4
  evaluation/final_frame.png
```

查看 TensorBoard：

```bash
uv run --frozen --group train tensorboard \
  --logdir outputs/mappo_mock/tensorboard \
  --bind_all
```

Mock 的重点是确认 loss 有限、checkpoint 可恢复、攻击碰撞率高于随机策略，以及 GIF 中车辆确实受策略控制。

## 5. 准备 64/16 真实场景

如果已经有 canonical JSON，可以直接编译：

```bash
uv run --frozen python -m scenario_pipeline.cli compile-rl \
  --input /data/nuplan/canonical \
  --output /data/nuplan/rl_runtime
```

如果只有 nuPlan DB 与地图：

```bash
uv run --frozen python -m scenario_pipeline.cli convert-nuplan \
  --input /data/nuplan/nuplan-v1.1/splits/mini \
  --maps-root /data/nuplan/maps \
  --output /data/nuplan/canonical

uv run --frozen python -m scenario_pipeline.cli compile-rl \
  --input /data/nuplan/canonical \
  --output /data/nuplan/rl_runtime
```

从容量一致、有地图、至少两辆可控背景车的场景中固定划分 64/16：

```bash
uv run --frozen --group train python -m gpudrive_cuda.rl.split_runtime \
  --runtime-root /data/nuplan/rl_runtime \
  --output dataset/nuplan/rl_runtime/runtime_split.json \
  --train 64 \
  --eval 16 \
  --seed 42 \
  --min-attackers 2
```

`runtime_split.json` 只记录场景路径，不复制真实数据。训练和评估场景不会重叠。由于仓库 `.gitignore` 会忽略 `dataset/`，真实路径文件也不会被误提交。

## 6. 正式训练与恢复

默认配置为 `gpudrive_cuda/configs/mappo_v0.json`。运行完整 500 万 world-step：

```bash
./gpudrive_cuda/scripts/run_mappo_v0.sh
```

只启动训练：

```bash
export PYTHONPATH="$PWD/build/gpudrive_cuda_rl/python:$PWD${PYTHONPATH:+:$PYTHONPATH}"
uv run --frozen --group train python -m gpudrive_cuda.rl.train \
  --config gpudrive_cuda/configs/mappo_v0.json
```

从 checkpoint 恢复：

```bash
uv run --frozen --group train python -m gpudrive_cuda.rl.train \
  --config gpudrive_cuda/configs/mappo_v0.json \
  --resume outputs/mappo_v0/checkpoints/step_001000000.pt
```

固定课程：

```text
0-0.5M world-step:  4 辆，RESIDUAL
0.5M-1.5M:          8 辆，RESIDUAL
1.5M-3.0M:         16 辆，RESIDUAL
3.0M-5.0M:         16 辆，DIRECT
```

## 7. 评估和渲染

```bash
uv run --frozen --group train python -m gpudrive_cuda.rl.evaluate \
  --config gpudrive_cuda/configs/mappo_v0.json \
  --checkpoint outputs/mappo_v0/checkpoints/final.pt \
  --output outputs/mappo_v0/evaluation \
  --render
```

重点检查：

- `ego_collision_rate`
- `minimum_distance_mean`
- `minimum_ttc_mean`
- `offroad_rate`
- `non_ego_collision_rate`
- `mean_collision_time_seconds`

不能只看 ego 碰撞率。若 offroad 或无关车辆碰撞同步上升，策略可能只学到了不合理的硬撞行为。

## 8. 常见问题

找不到扩展时检查：

```bash
find build/gpudrive_cuda_rl/python -iname 'gpudrive_cuda_torch*'
echo "$PYTHONPATH"
```

检查 Torch CUDA：

```bash
uv run --frozen --group train python -c \
  'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

显存不足时，先把配置中的 `num_worlds` 从 64 降到 32，再把 `ppo.rollout_steps` 从 128 降到 64。不要先减少地图和伙伴观测容量，否则会改变模型接口与 checkpoint 结构。

如果 loss 正常但碰撞率不升，先确认 mock 能过拟合，再检查真实场景是否至少有两辆背景车、攻击车 mask 是否有效，以及 `collided_ego` 是否在 CUDA integration test 中触发。
