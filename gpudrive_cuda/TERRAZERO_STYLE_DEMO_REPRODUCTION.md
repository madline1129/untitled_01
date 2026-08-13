# TerraZero 风格 10 秒 Demo 工作站复现指南

本文档用于在 NVIDIA Linux 工作站上复现真实 nuPlan 场景的 CUDA 推演，并生成
TerraZero 风格的 16:9 俯视动画：

视觉层级参考 [TerraZero 官方 Top-down Demo](https://terra-applied.github.io/TerraZero/)；
本仓库只复现其清晰的对象级俯视表达，不使用或分发对方的代码和媒体资产。

```text
真实 nuPlan SQLite + HD Map
  -> CanonicalScenario
  -> RuntimeScenario
  -> 10 秒 CUDA rollout
  -> trace.csv / summary.json
  -> 1600x900 GIF / H.264 MP4 / PNG
```

标准场景使用已经完成端到端验证的 nuPlan scene：

```text
scene token: 165060762e765a5a
location:    las_vegas
map:         us-nv-las-vegas-strip
dt:          0.1 s
future:      118 steps / 11.8 s
```

正式 Demo 执行 100 个 simulator step，GIF 为 10 FPS、100 帧，MP4 为
20 FPS、200 帧，两者播放时长均为 10 秒。

## 1. 拉取代码

工作站已有仓库时：

```bash
cd /home/user/whz/agent
git fetch origin
git switch main
git pull --ff-only origin main
```

全新工作站：

```bash
git clone git@github.com:madline1129/untitled_01.git agent
cd agent
```

检查代码版本和工作区，不要清理真实数据目录：

```bash
git log -1 --oneline
git status --short
```

## 2. 检查系统环境

```bash
nvidia-smi
nvcc --version
g++ --version
uv --version
```

如果没有 uv：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

同步仓库环境：

```bash
uv sync --frozen
```

Matplotlib、Pillow、CMake、Ninja 和 `imageio-ffmpeg` 均由 uv 管理。
MP4 编码使用 `imageio-ffmpeg` 自带的 ffmpeg，不要求系统额外安装 ffmpeg。

## 3. 设置真实 nuPlan 路径

此前成功运行的工作站使用：

```bash
export NUPLAN_DB="/home/user/whz/agent/data/dataset/nuPlan/raw/cache/mini/2021.05.12.22.00.38_veh-35_01008_01518.db"
export SCENE_ID="165060762e765a5a"
export NUPLAN_MAPS_ROOT="/home/user/whz/agent/data/dataset/nuPlan/maps"

export GPUDRIVE_PYTHON="/home/user/anaconda3/envs/gpudrive/bin/python"
export UV_BIN="/home/user/anaconda3/envs/gpudrive/bin/uv"
export PYTHONPATH="/data/fc/hugsim/atg/nuplan-devkit"
```

如果工作站路径不同，只替换上述绝对路径。数据库必须包含 `scene`、`log`、
`lidar_pc`、`ego_pose`、`lidar_box` 和 `track` 等 nuPlan 表；地图根目录必须包含：

```text
nuplan-maps-v1.0.json
us-nv-las-vegas-strip/<version>/map.gpkg
```

不要使用 mock runtime 制作正式展示。mock 场景没有地图，而且只有 9.5 秒未来数据。

## 4. 生成真实 RuntimeScenario

输出放在数据盘，不写入 Git 仓库：

```bash
export REAL_RUN_ROOT="/data/$USER/agent_runs/nuplan_terrazero_demo"
export CANONICAL_DIR="$REAL_RUN_ROOT/canonical"
export RUNTIME_DIR="$REAL_RUN_ROOT/runtime"
export SIM_OUTPUT_DIR="$REAL_RUN_ROOT/simulator"

mkdir -p "$CANONICAL_DIR" "$RUNTIME_DIR" "$SIM_OUTPUT_DIR"
```

在能够 import nuPlan devkit 的 Python 环境中转换：

```bash
"$GPUDRIVE_PYTHON" -m scenario_pipeline convert-nuplan \
  --input "$NUPLAN_DB" \
  --scene-id "$SCENE_ID" \
  --maps-root "$NUPLAN_MAPS_ROOT" \
  --output "$CANONICAL_DIR"

"$GPUDRIVE_PYTHON" -m scenario_pipeline validate \
  --input "$CANONICAL_DIR"
```

使用 `max_agents=192` 保留该场景的全部 172 个 Agent：

```bash
"$UV_BIN" run --frozen python -m scenario_pipeline compile-rl \
  --input "$CANONICAL_DIR" \
  --output "$RUNTIME_DIR" \
  --max-agents 192

"$UV_BIN" run --frozen python -m scenario_pipeline validate-rl \
  --input "$RUNTIME_DIR"
```

找到直接包含 `manifest.json` 的场景目录：

```bash
find "$RUNTIME_DIR" -name manifest.json -print

export RUNTIME_SCENE_DIR="$RUNTIME_DIR/2021_05_12_22_00_38_veh-35_01008_01518_165060762e765a5a"
```

检查正式 Demo 的三个前置条件：

```bash
"$UV_BIN" run --frozen python - "$RUNTIME_SCENE_DIR/manifest.json" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
assert manifest["source"]["source_scenario_id"] == "165060762e765a5a"
assert manifest["counts"]["map_features"] > 0
assert manifest["episode_steps"] * manifest["dt"] >= 10.0
print({
    "scene": manifest["source"]["source_scenario_id"],
    "dt": manifest["dt"],
    "episode_steps": manifest["episode_steps"],
    "agents": manifest["counts"]["agents"],
    "map_features": manifest["counts"]["map_features"],
})
PY
```

## 5. 一键生成 10 秒 Demo

为避免工作站 Conda 环境中的另一套 CUDA 抢占系统 `nvcc`，只显式传递
`UV_BIN`，不要把整个 Conda `bin` 放到 `PATH` 最前面：

```bash
UV_BIN="$UV_BIN" \
GPU_SIM_BUILD_DIR="$PWD/gpudrive_cuda/build.cuda11" \
DEMO_DURATION_SECONDS=10 \
bash gpudrive_cuda/scripts/run_nuplan_demo.sh \
  "$RUNTIME_SCENE_DIR" \
  "$SIM_OUTPUT_DIR" \
  1
```

脚本会验证 runtime、编译 CUDA、运行 loader/integration test、执行 100 步推演，
然后生成：

```text
$SIM_OUTPUT_DIR/trace.csv
$SIM_OUTPUT_DIR/summary.json
$SIM_OUTPUT_DIR/rollout.gif
$SIM_OUTPUT_DIR/rollout.mp4
$SIM_OUTPUT_DIR/final_frame.png
```

## 6. 手动重新渲染

渲染不需要重新运行 CUDA。默认是以 ego 为中心的局部跟随视图：

```bash
"$UV_BIN" run --frozen python gpudrive_cuda/tools/render_trace.py \
  --runtime "$RUNTIME_SCENE_DIR" \
  --trace "$SIM_OUTPUT_DIR/trace.csv" \
  --output "$SIM_OUTPUT_DIR/rollout.gif" \
  --mp4-output "$SIM_OUTPUT_DIR/rollout.mp4" \
  --final-png "$SIM_OUTPUT_DIR/final_frame.png" \
  --duration 10 \
  --fps 10 \
  --mp4-fps 20 \
  --view follow
```

用于排查地图时可切换到全图，并临时显示目标和历史轨迹：

```bash
"$UV_BIN" run --frozen python gpudrive_cuda/tools/render_trace.py \
  --runtime "$RUNTIME_SCENE_DIR" \
  --trace "$SIM_OUTPUT_DIR/trace.csv" \
  --output "$SIM_OUTPUT_DIR/debug_full_map.gif" \
  --final-png "$SIM_OUTPUT_DIR/debug_full_map.png" \
  --duration 10 \
  --view full-map \
  --show-goals \
  --show-trails
```

## 7. 自动验收

```bash
"$UV_BIN" run --frozen python - "$SIM_OUTPUT_DIR" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

import imageio_ffmpeg
from PIL import Image

root = Path(sys.argv[1])
summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
assert summary["executed_steps"] == 100
assert summary["dynamics_model"] == "hybrid_dynamic_bicycle"

with (root / "trace.csv").open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
ego = [row for row in rows if row["world"] == "0" and row["agent_slot"] == "0"]
assert len({int(row["step"]) for row in ego}) == 101
for row in rows:
    assert all(math.isfinite(float(row[key])) for key in ("x", "y", "yaw", "vx", "vy"))

with Image.open(root / "rollout.gif") as gif:
    assert gif.size == (1600, 900)
    assert gif.n_frames == 100
with Image.open(root / "final_frame.png") as png:
    assert png.size == (1600, 900)
mp4_frames, mp4_seconds = imageio_ffmpeg.count_frames_and_secs(str(root / "rollout.mp4"))
assert mp4_frames == 200
assert abs(mp4_seconds - 10.0) <= 0.1

displacement = math.hypot(
    float(ego[-1]["x"]) - float(ego[0]["x"]),
    float(ego[-1]["y"]) - float(ego[0]["y"]),
)
assert displacement > 0.01
print(f"GIF=100 frames, MP4={mp4_frames} frames/{mp4_seconds:.2f}s")
print(f"ego displacement={displacement:.3f}m")
PY
```

人工查看最终帧和动画，必须满足：

- 深色道路面、白色道路标线和黄色道路边界可辨认。
- ego 为蓝色，普通车辆为橙色，行人和骑行者使用独立颜色。
- 画面跟随 ego，不再把整张地图压缩到一帧。
- 车辆框方向与 yaw 一致，运动连续，无突然跳变。
- 碰撞、越界或到达目标时只改变轮廓，不掩盖 Agent 类型颜色。

## 8. 常见问题

`runtime only covers ... the demo requires 10s`：场景未来长度不足。换用本文标准
scene，或重新编译具有至少100个future step的RuntimeScenario；不要复制最后一帧凑时长。

`map_features == 0`：转换阶段没有成功加载HD Map。检查`--maps-root`、地图元数据JSON
和`map.gpkg`，不要用无地图mock制作正式Demo。

`unsupported toolchain`或PTX错误：CMake选到了与驱动不兼容的`nvcc`。保持系统PATH，
通过`UV_BIN=/path/to/uv`单独指定uv。

MP4生成失败：先执行`uv sync --frozen`，确认`uv run python -c
"import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"`能够返回可执行文件。
