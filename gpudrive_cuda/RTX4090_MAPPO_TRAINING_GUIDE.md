# 单卡 RTX 4090 MAPPO 训练指南

本文用于在一张 RTX 4090 24GB 上训练 DangerMaker MAPPO，并通过 TensorBoard
读取训练日志。第一版固定 ego 的参考轨迹控制器，只训练最多 16 辆背景攻击车。

## 1. 默认训练规格

专用配置为 `gpudrive_cuda/configs/mappo_rtx4090.json`：

```text
GPU                    1 x RTX 4090 24GB
并行 world             64
rollout 长度           128 simulator step
每次更新样本           8192 world-step
PPO minibatch          8
PPO epoch              4
总训练量               5,000,000 world-step
攻击车                 4 -> 8 -> 16，最后切换 DIRECT
```

`num_minibatches=8` 比通用配置更保守，用于降低 actor/critic 反向传播时的峰值显存。
训练开始后以 TensorBoard 中的实际吞吐和显存为准，再决定是否增加并行度。

## 2. 拉取代码和安装环境

```bash
cd /path/to/agent
git pull --ff-only origin main
uv sync --frozen --group train
```

检查单卡环境：

```bash
nvidia-smi
nvcc --version

uv run --frozen --group train python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM GiB:", torch.cuda.get_device_properties(0).total_memory / 1024**3)
PY
```

锁定环境使用 PyTorch 2.6 + CUDA 12.4。系统 NVIDIA Driver 必须支持 CUDA 12.4；
编译扩展时，本机 `nvcc` 最好也是 12.4，同一 CUDA 主版本是最低要求。

## 3. 准备 RuntimeScenario split

训练配置读取：

```text
dataset/nuplan/rl_runtime/runtime_split.json
```

如果真实 runtime 已在 `/data/nuplan/rl_runtime`，生成固定的 64/16 划分：

```bash
uv run --frozen --group train python -m gpudrive_cuda.rl.split_runtime \
  --runtime-root /data/nuplan/rl_runtime \
  --output dataset/nuplan/rl_runtime/runtime_split.json \
  --train 64 \
  --eval 16 \
  --seed 42 \
  --min-attackers 2
```

split 文件只保存路径，不复制数据。所有场景必须具有相同 RuntimeScenario 容量，
每个场景必须有一个 ego 和至少两辆有效、可控的背景车辆。

正式训练前先检查 split：

```bash
uv run --frozen --group train python - <<'PY'
from pathlib import Path
from gpudrive_cuda.rl.data import load_runtime_split

path = Path("dataset/nuplan/rl_runtime/runtime_split.json")
print("train scenes:", len(load_runtime_split(path, "train")))
print("eval scenes:", len(load_runtime_split(path, "eval")))
PY
```

## 4. 编译并做 smoke test

RTX 4090 的 CUDA compute capability 是 8.9：

```bash
CMAKE_CUDA_ARCHITECTURES=89 \
  ./gpudrive_cuda/scripts/build_torch_extension.sh

export PYTHONPATH="$PWD/build/gpudrive_cuda_rl/python:$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

运行测试：

```bash
uv run --frozen --group train python -m unittest \
  gpudrive_cuda.tests.test_rl \
  gpudrive_cuda.tests.test_torch_bridge -v
```

建议正式训练前完成一次 mock 闭环：

```bash
./gpudrive_cuda/scripts/run_mappo_mock.sh
```

## 5. 启动正式训练

建议放在 `tmux` 中，防止 SSH 断开导致训练退出：

```bash
tmux new -s dangermaker
cd /path/to/agent
mkdir -p outputs/mappo_rtx4090
set -o pipefail
./gpudrive_cuda/scripts/run_mappo_rtx4090.sh 2>&1 | tee outputs/mappo_rtx4090/train.log
```

脚本固定使用一张可见 GPU，并设置：

```text
CUDA_VISIBLE_DEVICES=0
CMAKE_CUDA_ARCHITECTURES=89
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

需要使用物理编号为 1 的 GPU 时：

```bash
CUDA_VISIBLE_DEVICES=1 ./gpudrive_cuda/scripts/run_mappo_rtx4090.sh
```

训练输出不会提交到 Git：

```text
outputs/mappo_rtx4090/
  resolved_config.json
  metrics.jsonl
  tensorboard/events.out.tfevents.*
  checkpoints/step_*.pt
  checkpoints/final.pt
  evaluation/eval_trace.csv
  evaluation/eval_summary.json
  evaluation/rollout.gif
  evaluation/rollout.mp4
```

## 6. 打开 TensorBoard

在工作站启动服务，默认只监听本机，避免直接暴露到公网：

```bash
./gpudrive_cuda/scripts/run_tensorboard.sh
```

从自己的电脑建立 SSH 转发：

```bash
ssh -N -L 6006:127.0.0.1:6006 user@workstation
```

浏览器打开：

```text
http://127.0.0.1:6006
```

端口被占用时，工作站和本地两端同时改成 6007：

```bash
TENSORBOARD_PORT=6007 ./gpudrive_cuda/scripts/run_tensorboard.sh
ssh -N -L 6007:127.0.0.1:6007 user@workstation
```

## 7. 必看 TensorBoard 曲线

### 训练目标

- `train/episode_success_rate`：完成 episode 中撞到 ego 的比例，是首要训练指标。
- `train/episode_timeout_rate`：完成 episode 中运行到超时的比例。
- `train/episode_return_mean`：完整 episode 团队回报，应该总体上升。
- `safety/min_distance_mean_m`：攻击车与 ego 的最小距离，通常应下降。
- `safety/min_ttc_mean_s_clipped_30`：平均最小 TTC，通常应下降。

### 行为合理性

- `safety/offroad_rate`：不能随着成功率同步大幅上升。
- `safety/road_collision_rate`：检查车辆是否持续撞道路边界。
- `safety/non_ego_collision_rate`：检查策略是否只学会无差别撞车。
- `action/normalized_*_abs_mean`：长期接近 1 表示动作持续饱和。

### PPO 健康度

- `ppo/approx_kl`：应大部分低于 `target_kl=0.02`。
- `ppo/clip_fraction`：持续过高通常说明更新幅度太大。
- `ppo/entropy` 和 `policy/log_std_*`：观察探索是否过早坍缩。
- `ppo/explained_variance`：越接近 1，critic 对 return 的解释越好；长期小于 0
  表示 value 学习存在问题。
- `ppo/gradient_norm`：记录裁剪前梯度范数，可用于发现梯度爆炸。

### 单卡性能

- `system/world_steps_per_second`：实际训练吞吐。
- `system/last_update_seconds`：一次 rollout 加 PPO update 的时间。
- `system/cuda_peak_allocated_gib`：PyTorch 实际分配峰值。
- `system/cuda_peak_reserved_gib`：PyTorch 缓存分配器保留峰值。

这两个显存指标不包含 C++ simulator 直接调用 `cudaMalloc` 的部分。判断整张卡是否接近
24GB 上限时，必须同时观察 `nvidia-smi`。

课程在 0.5M、1.5M 和 3.0M world-step 切换。切换时 reward 或成功率短暂下降是正常的，
判断趋势时应在 TensorBoard 中按 `curriculum/active_attackers` 和
`curriculum/control_mode` 分阶段查看。

## 8. 中断和恢复

正常停止时使用 `Ctrl+C`。训练会保留最近一次周期 checkpoint；当前 rollout
中尚未写入 checkpoint 的数据不会保留。

恢复训练时把 checkpoint 传给一键脚本：

```bash
./gpudrive_cuda/scripts/run_mappo_rtx4090.sh \
  outputs/mappo_rtx4090/checkpoints/step_001007616.pt
```

也可以只运行训练入口：

```bash
export PYTHONPATH="$PWD/build/gpudrive_cuda_rl/python:$PWD${PYTHONPATH:+:$PYTHONPATH}"
uv run --frozen --group train python -m gpudrive_cuda.rl.train \
  --config gpudrive_cuda/configs/mappo_rtx4090.json \
  --resume outputs/mappo_rtx4090/checkpoints/step_001007616.pt
```

恢复会加载模型、优化器、global world-step 和随机数状态，并继续写入同一个
TensorBoard 目录。不要在未传 `--resume` 时复用已有输出目录；训练入口会拒绝覆盖
已有 `metrics.jsonl`。

## 9. 24GB 显存不足时

先运行另一个终端观察：

```bash
watch -n 1 nvidia-smi
```

按以下顺序调整 `mappo_rtx4090.json`：

1. 把 `ppo.num_minibatches` 从 8 增加到 16，减小反向传播 minibatch。
2. 把 `num_worlds` 从 64 降到 48 或 32。
3. 把 `ppo.rollout_steps` 从 128 降到 64。

改变 `num_worlds` 或 rollout 长度会改变每次 PPO 更新的数据量和吞吐，但不改变模型
结构。不要优先减少 partner/map observation 容量，否则会改变策略输入与 checkpoint
兼容性。

## 10. 训练完成后的验收

一键脚本会自动对固定 eval split 做确定性评估并渲染 GIF。检查：

```bash
cat outputs/mappo_rtx4090/evaluation/eval_summary.json
```

必须同时报告 ego 碰撞率、最小距离、最小 TTC、offroad、无关车辆碰撞率和平均碰撞
时间。当前 ego 不进行闭环避障，因此结果只证明 simulator-MAPPO 攻击闭环有效，
不能表述为攻破真实自动驾驶规划系统。
