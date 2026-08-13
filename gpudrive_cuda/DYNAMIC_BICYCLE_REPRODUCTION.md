# CUDA 动态自行车模型复现指南

本文档用于在 NVIDIA Linux 服务器复现以下完整流水线：

```text
RuntimeScenario v1
    -> CUDA 自动控制或外部动作
    -> 混合动态自行车模型
    -> trace.csv + summary.json
    -> rollout.gif + rollout.mp4 + final_frame.png
```

当前实现保持 `AgentState=[x,y,yaw,vx,vy]` 和
`AgentAction=[target_acceleration,target_front_wheel_angle]` 不变。车辆在高速区
使用动态自行车模型，低速区平滑切换到运动学模型；行人、骑行者和其他类型始终
使用运动学回退。

## 1. 环境要求

服务器需要：

```text
NVIDIA GPU 和可用驱动
CUDA Toolkit（必须包含 nvcc）
Git
uv
Python 3.10
```

检查系统环境：

```bash
nvidia-smi
nvcc --version
git --version
uv --version
python3 --version
```

如果没有 uv：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

## 2. 获取代码与 Python 环境

```bash
git clone git@github.com:madline1129/untitled_01.git agent
cd agent
git switch main
git pull --ff-only origin main

uv sync --frozen
```

如果服务器已有仓库且允许丢弃本地代码修改，可使用：

```bash
git fetch origin
git reset --hard origin/main
```

不要对包含真实数据的目录执行 `git clean -fdx`。

## 3. 准备 RuntimeScenario

如果当前机器已经有此前生成的 mock 场景，可使用：

```bash
export RUNTIME_PATH="$PWD/dataset/nuplan/rl_runtime/mock_nuplan_00100000"
```

`dataset/` 被 `.gitignore` 排除，fresh clone 不包含该场景。新服务器必须从已有
工作站复制 RuntimeScenario，或者从真实 nuPlan 数据重新生成。例如从本地执行：

```bash
rsync -av --progress \
  dataset/nuplan/rl_runtime/mock_nuplan_00100000/ \
  user@server:/data/$USER/runtime/mock_nuplan_00100000/
```

然后在服务器设置：

```bash
export RUNTIME_PATH="/data/$USER/runtime/mock_nuplan_00100000"
```

使用真实 nuPlan 数据时，将变量改为实际生成的 RuntimeScenario 场景目录。该目录
必须直接包含：

```text
manifest.json
agent_initial_state.bin
agent_dimensions.bin
agent_type.bin
reference_future.bin
map_points.bin
traffic_light_state.bin
...
```

验证数据：

```bash
uv run --frozen python -m scenario_pipeline validate-rl \
  --input "$RUNTIME_PATH"
```

从原始 nuPlan SQLite 和地图生成真实 RuntimeScenario 时，先按
[真实 nuPlan 工作站复现指南](WORKSTATION_NUPLAN_REPRODUCTION.md)完成转换。

## 4. 一键编译与验收

```bash
bash gpudrive_cuda/scripts/test_current.sh "$RUNTIME_PATH"
```

脚本会自动完成：

```text
uv 环境同步
CMake/Ninja 配置与 CUDA 编译
RuntimeScenario loader 测试
CUDA 动力学集成测试
10 步双 world rollout
CSV 动力学字段检查
GIF、MP4 和 PNG 渲染
GIF/MP4 帧数及播放时长检查
```

默认输出：

```text
outputs/gpudrive_cuda_test/trace.csv
outputs/gpudrive_cuda_test/summary.json
outputs/gpudrive_cuda_test/rollout.gif
outputs/gpudrive_cuda_test/rollout.mp4
outputs/gpudrive_cuda_test/final_frame.png
```

## 5. 手动运行外部转向

先编译：

```bash
uv run --frozen cmake \
  -S gpudrive_cuda \
  -B gpudrive_cuda/build \
  -G Ninja \
  -DBUILD_TESTING=ON

uv run --frozen cmake --build gpudrive_cuda/build -j
```

让 ego（slot 0）使用恒定目标加速度和目标前轮转角：

```bash
export OUTPUT_DIR="$PWD/outputs/dynamic_bicycle_external"

gpudrive_cuda/build/drive_sim_cli \
  --runtime "$RUNTIME_PATH" \
  --worlds 1 \
  --steps 50 \
  --external-agent 0 \
  --acceleration 1.0 \
  --steering 0.25 \
  --output "$OUTPUT_DIR"
```

生成动画：

```bash
uv run --frozen python gpudrive_cuda/tools/render_trace.py \
  --runtime "$RUNTIME_PATH" \
  --trace "$OUTPUT_DIR/trace.csv" \
  --output "$OUTPUT_DIR/rollout.gif" \
  --mp4-output "$OUTPUT_DIR/rollout.mp4" \
  --final-png "$OUTPUT_DIR/final_frame.png" \
  --world 0 \
  --fps 10 \
  --mp4-fps 20 \
  --duration 5
```

`acceleration/steering` 是限幅后的目标命令，实际执行器状态位于：

```text
actual_acceleration
actual_steering
longitudinal_velocity
lateral_velocity
yaw_rate
```

实际值不会瞬间跳到目标值，因为模型包含 `0.25s` 加速度时间常数、`0.15s`
转向时间常数、`8m/s^3` jerk 上限和 `0.8rad/s` 转向速率上限。

## 6. 自动检查输出

```bash
uv run --frozen python - "$OUTPUT_DIR" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

import imageio_ffmpeg
from PIL import Image

root = Path(sys.argv[1])
trace_path = root / "trace.csv"
summary_path = root / "summary.json"
gif_path = root / "rollout.gif"
mp4_path = root / "rollout.mp4"
png_path = root / "final_frame.png"

summary = json.loads(summary_path.read_text(encoding="utf-8"))
assert summary["dynamics_model"] == "hybrid_dynamic_bicycle"
assert summary["dynamics_substeps"] == 5

with trace_path.open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))

required = {
    "x", "y", "yaw", "vx", "vy", "acceleration", "steering",
    "actual_acceleration", "actual_steering", "longitudinal_velocity",
    "lateral_velocity", "yaw_rate",
}
assert rows and required.issubset(rows[0])

numeric_fields = required
for row in rows:
    assert all(math.isfinite(float(row[name])) for name in numeric_fields)
    assert float(row["longitudinal_velocity"]) >= -1e-5
    assert abs(float(row["actual_steering"])) <= 0.60001

ego = [row for row in rows if row["world"] == "0" and row["agent_slot"] == "0"]
assert len(ego) >= 2
displacement = math.hypot(
    float(ego[-1]["x"]) - float(ego[0]["x"]),
    float(ego[-1]["y"]) - float(ego[0]["y"]),
)
assert displacement > 0.01

with Image.open(gif_path) as animation:
    assert animation.size == (1600, 900)
    assert animation.n_frames == 50
mp4_frames, mp4_seconds = imageio_ffmpeg.count_frames_and_secs(str(mp4_path))
assert mp4_frames == 100
assert abs(mp4_seconds - 5.0) <= 0.1
with Image.open(png_path) as image:
    assert image.size == (1600, 900)

print(f"model={summary['dynamics_model']}")
print(f"steps={len({int(row['step']) for row in ego})} displacement={displacement:.3f}m")
print(f"gif={gif_path} mp4={mp4_path} png={png_path}")
PY
```

## 7. 验收标准

- `runtime_loader_test` 和 `simulator_integration_test` 全部通过。
- CUDA 运行没有 kernel launch、非法内存访问或 NaN/Inf。
- 正转角产生正 yaw，负转角产生负 yaw。
- 制动后纵向速度不会变成负数。
- 乘用车与大型车在相同动作下具有不同横摆响应。
- `summary.json` 中 `dynamics_model` 为 `hybrid_dynamic_bicycle`。
- GIF 帧数等于 `duration * 10`，MP4 帧数等于 `duration * 20`。
- `rollout.gif`、`rollout.mp4` 和 `final_frame.png` 均非空且可读取。

mock runtime 的 `map_features=0`，因此 GIF 没有道路底图属于正常现象。真实
RuntimeScenario 包含地图 feature 时，renderer 会自动绘制道路几何。

## 8. 常见问题

`nvcc not found`：服务器只有 NVIDIA 驱动，没有 CUDA Toolkit。安装或加载包含
`nvcc` 的 CUDA module，并确保其 `bin` 目录位于 `PATH`。

`no runtime manifests found`：`RUNTIME_PATH` 必须指向包含 `manifest.json` 的
场景目录，或者包含多个该类场景目录的父目录。

GIF 中车辆不转向：确认 `--external-agent` 对应有效且可控制的 Agent，并检查
CSV 中 `control_mode=external`、`steering` 和 `actual_steering` 是否非零。

GIF 没有地图：检查 `manifest.json` 的 `counts.map_features`。如果为 0，需要用
nuPlan 地图重新转换 RuntimeScenario，而不是修改 renderer。
