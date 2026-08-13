# nuPlan 批量场景转换到 RTX 4090 训练指南

本文从工作站已有的 nuPlan SQLite 数据开始，批量生成真实 RuntimeScenario，
固定划分 64 个训练场景和 16 个不重复评估场景，然后启动单卡 RTX 4090 MAPPO。

完整流水线：

```text
nuPlan SQLite + HD Map
  -> 128 个候选 scene
  -> CanonicalScenario JSON
  -> 固定容量 RuntimeScenario
  -> 64 train + 16 eval
  -> RTX 4090 MAPPO
  -> TensorBoard + checkpoint + evaluation GIF
```

## 1. 为什么现在只有一个场景

此前真实数据复现是 smoke test，命令同时指定了：

```text
一个 NUPLAN_DB
一个 --scene-id=165060762e765a5a
```

`--scene-id` 会明确过滤掉数据库中的其他 scene。一个 Canonical JSON 只会编译成
一个 RuntimeScenario，因此工作站最终只有一个真实 runtime。这不代表 nuPlan 数据集
本身只有一个场景，只代表之前只转换了一个。

正式划分工具要求至少 80 个同时满足以下条件的不同 runtime：

- 使用同一组固定容量；
- HD Map 非空；
- 初始时刻至少有两辆有效、可控、非 ego 的车辆；
- 64 个用于训练，另外 16 个只用于评估。

不能把一个 scene 复制 80 次冒充正式数据。复制只适合 smoke test，会导致严重过拟合，
也无法得到独立评估结果。建议先转换 128 个候选 scene，为地图或车辆数量筛选留余量。

## 2. 设置工作站路径

在仓库根目录执行，并替换为工作站上的真实路径：

```bash
cd /path/to/agent

export REPO_ROOT="$PWD"
export NUPLAN_DB_ROOT="/actual/path/to/nuplan/raw/cache"
export NUPLAN_MAPS_ROOT="/actual/path/to/nuplan/maps"
export NUPLAN_DEVKIT_ROOT="/actual/path/to/nuplan-devkit"

# 该 Python 必须能够 import nuplan-devkit。
export NUPLAN_PYTHON="/actual/path/to/conda/env/bin/python"
export UV_BIN="$(command -v uv)"

# 使用新的目录，避免与之前单场景 smoke 输出混在一起。
export PREP_ROOT="/data/$USER/agent_runs/nuplan_mappo_128"
export SELECTION_FILE="$PREP_ROOT/selected_scenes.tsv"
export CANONICAL_DIR="$PREP_ROOT/canonical"
export RUNTIME_DIR="$PREP_ROOT/runtime"
export SPLIT_FILE="$REPO_ROOT/dataset/nuplan/rl_runtime/runtime_split.json"

mkdir -p "$PREP_ROOT" "$CANONICAL_DIR" "$RUNTIME_DIR" "$(dirname "$SPLIT_FILE")"
export PYTHONPATH="$REPO_ROOT:$NUPLAN_DEVKIT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
```

根据上一次工作站真实复现报告，对应路径曾经是：

```bash
export REPO_ROOT="/home/user/whz/agent"
export NUPLAN_DB_ROOT="/home/user/whz/agent/data/dataset/nuPlan/raw/cache"
export NUPLAN_MAPS_ROOT="/home/user/whz/agent/data/dataset/nuPlan/maps"
export NUPLAN_DEVKIT_ROOT="/data/fc/hugsim/atg/nuplan-devkit"
export NUPLAN_PYTHON="/home/user/anaconda3/envs/gpudrive/bin/python"
export UV_BIN="/home/user/anaconda3/envs/gpudrive/bin/uv"
```

先确认这些路径在当前工作站仍然存在，不要机械照抄。特别是
`NUPLAN_DB_ROOT` 如果只包含 `mini` 数据，原始 scene 数量可能仍然不足 80。

不要把 `CANONICAL_DIR` 或 `RUNTIME_DIR` 设置到原始 nuPlan 数据目录中。转换过程只读
SQLite 和地图，但输出应独立放在数据盘。

检查两个 Python 环境：

```bash
"$NUPLAN_PYTHON" - <<'PY'
import nuplan
print("nuplan import: OK", nuplan.__path__)
PY

"$UV_BIN" sync --frozen --group train
"$UV_BIN" run --frozen --group train python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
PY
```

## 3. 统计原始数据库里有多少 scene

下面的检查只读取 SQLite 的 `scene` 表，不加载地图，也不执行转换：

```bash
export NUPLAN_DB_ROOT
"$NUPLAN_PYTHON" - <<'PY'
import os
import sqlite3
from pathlib import Path

root = Path(os.environ["NUPLAN_DB_ROOT"]).resolve()
databases = sorted(root.rglob("*.db"))
total = 0
valid_databases = 0
for path in databases:
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        count = int(connection.execute("SELECT COUNT(*) FROM scene").fetchone()[0])
        connection.close()
    except (sqlite3.Error, TypeError):
        continue
    if count:
        valid_databases += 1
        total += count
        print(f"{count:4d}  {path}")

print(f"valid nuPlan databases: {valid_databases}")
print(f"raw scenes: {total}")
if total < 80:
    raise SystemExit("fewer than 80 raw scenes: install more official nuPlan DB files first")
PY
```

结果解释：

- `raw scenes >= 128`：按后面的命令固定抽取 128 个候选场景。
- `80 <= raw scenes < 128`：可以把候选数量改成实际总数，但筛选后可能不足 80。
- `raw scenes < 80`：无法建立 64/16 独立划分，需要先补充更多官方 nuPlan DB 文件。
- 找到很多 `.db` 但 `valid nuPlan databases=0`：`NUPLAN_DB_ROOT` 指错了目录。

## 4. 固定抽取 128 个候选 scene

以下脚本使用 seed 42，从所有 SQLite scene 中按 token 去重并确定性抽取最多 128 个，
再保存数据库绝对路径和 scene token。它不会复制数据库：

```bash
export CANDIDATE_SCENES=128
export SELECTION_SEED=42
export NUPLAN_DB_ROOT SELECTION_FILE CANDIDATE_SCENES SELECTION_SEED

"$NUPLAN_PYTHON" - <<'PY'
import os
import random
import sqlite3
from pathlib import Path

root = Path(os.environ["NUPLAN_DB_ROOT"]).resolve()
output = Path(os.environ["SELECTION_FILE"]).resolve()
limit = int(os.environ["CANDIDATE_SCENES"])
seed = int(os.environ["SELECTION_SEED"])

scenes_by_id = {}
for database in sorted(root.rglob("*.db")):
    try:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        rows = connection.execute(
            "SELECT lower(hex(token)) FROM scene ORDER BY name, token"
        ).fetchall()
        connection.close()
    except sqlite3.Error:
        continue
    for row in rows:
        scene_id = str(row[0])
        scenes_by_id.setdefault(scene_id, str(database.resolve()))

scenes = [(database, scene_id) for scene_id, database in sorted(scenes_by_id.items())]
if len(scenes) < 80:
    raise SystemExit(f"only {len(scenes)} unique raw scenes; need at least 80")
random.Random(seed).shuffle(scenes)
selected = scenes[: min(limit, len(scenes))]
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", encoding="utf-8") as stream:
    for database, scene_id in selected:
        stream.write(f"{database}\t{scene_id}\n")

print(f"available raw scenes: {len(scenes)}")
print(f"selected candidates: {len(selected)}")
print(f"selection file: {output}")
PY

wc -l "$SELECTION_FILE"
sed -n '1,5p' "$SELECTION_FILE"
```

保留 `selected_scenes.tsv`。它记录了本次实验的数据选择，是复现实验的重要元数据。

## 5. 逐场景转换为 CanonicalScenario

这里必须使用能够 import nuPlan devkit 的 `NUPLAN_PYTHON`。逐 scene 调用的好处是某个
scene 转换失败时不会让同一数据库里的其他 scene 一起失败。

命令中故意不使用 `set -e`，失败场景会被记录并继续；`pipefail` 确保转换失败不会被
后面的 `tee` 隐藏：

```bash
export CONVERSION_LOG="$PREP_ROOT/convert.log"
export FAILED_SCENES="$PREP_ROOT/failed_scenes.tsv"
set -o pipefail
: > "$CONVERSION_LOG"
: > "$FAILED_SCENES"

succeeded=0
failed=0
while IFS=$'\t' read -r database scene_id; do
  echo "[convert] $scene_id from $database" | tee -a "$CONVERSION_LOG"
  if "$NUPLAN_PYTHON" -m scenario_pipeline convert-nuplan \
      --input "$database" \
      --scene-id "$scene_id" \
      --maps-root "$NUPLAN_MAPS_ROOT" \
      --output "$CANONICAL_DIR" 2>&1 | tee -a "$CONVERSION_LOG"; then
    succeeded=$((succeeded + 1))
  else
    printf '%s\t%s\n' "$database" "$scene_id" >> "$FAILED_SCENES"
    failed=$((failed + 1))
  fi
done < "$SELECTION_FILE"

echo "canonical conversion: succeeded=$succeeded failed=$failed"
find "$CANONICAL_DIR" -maxdepth 1 -type f -name '*.json' | wc -l
```

验证全部 CanonicalScenario：

```bash
"$NUPLAN_PYTHON" -m scenario_pipeline validate --input "$CANONICAL_DIR" \
  2>&1 | tee "$PREP_ROOT/validate_canonical.log"
```

统计地图、ego 和初始攻击车候选数量。这里复现 Runtime 编译器的 Agent 排序，只检查
最终会被 `max_agents=64` 保留的 slots：

```bash
export CANONICAL_DIR
"$NUPLAN_PYTHON" - <<'PY'
import os
from pathlib import Path
from scenario_pipeline.io import read_scenario

root = Path(os.environ["CANONICAL_DIR"])
total = map_ready = preliminarily_eligible = 0
source_ids = set()
for path in sorted(root.glob("*.json")):
    scenario = read_scenario(path)
    total += 1
    source_ids.add(scenario.source.source_scenario_id)
    anchor = scenario.timing.anchor_index

    def order_key(agent):
        if agent.valid[anchor]:
            distance = (float(agent.x[anchor]) ** 2 + float(agent.y[anchor]) ** 2) ** 0.5
        else:
            distance = float("inf")
        return (not agent.is_ego, distance, agent.type, agent.id)

    retained_agents = sorted(scenario.agents, key=order_key)[:64]
    attackers = sum(
        not agent.is_ego and agent.type == "vehicle" and bool(agent.valid[anchor])
        for agent in retained_agents
    )
    has_one_ego = sum(agent.is_ego for agent in scenario.agents) == 1
    has_map = scenario.quality.map_available and bool(scenario.map_features)
    map_ready += int(has_map)
    preliminarily_eligible += int(has_one_ego and has_map and attackers >= 2)

print("canonical files:", total)
print("unique source scene IDs:", len(source_ids))
print("map-ready scenes:", map_ready)
print("preliminarily eligible scenes:", preliminarily_eligible)
if preliminarily_eligible < 80:
    raise SystemExit("fewer than 80 preliminary scenes; convert more candidates")
PY
```

如果这里不足 80，不要复制现有 JSON。提高 `CANDIDATE_SCENES`，使用更多原始 scene
重新执行选择和转换；或者先修复 `convert.log` 中反复出现的地图兼容错误。

## 6. 统一编译为固定容量 RuntimeScenario

训练不需要保留场景中的全部远处 Agent。建议固定 `max_agents=64`：编译器会保留 ego，
再按初始距离保留最近 Agent。这比单场景展示使用的 192 slots 更适合 64-world 训练，
也显著降低 OBB 两两碰撞检测开销。

所有场景必须在同一次命令中使用完全相同的容量参数：

```bash
set -o pipefail
"$UV_BIN" run --frozen --group train python -m scenario_pipeline compile-rl \
  --input "$CANONICAL_DIR" \
  --output "$RUNTIME_DIR" \
  --max-agents 64 \
  --history-steps 11 \
  --max-future-steps 128 \
  --max-map-features 2048 \
  --max-map-points 32768 \
  --max-map-edges 8192 \
  --max-traffic-lights 128 \
  --max-route-features 512 \
  2>&1 | tee "$PREP_ROOT/compile_runtime.log"

"$UV_BIN" run --frozen --group train python -m scenario_pipeline validate-rl \
  --input "$RUNTIME_DIR" \
  2>&1 | tee "$PREP_ROOT/validate_runtime.log"

find "$RUNTIME_DIR" -mindepth 2 -maxdepth 2 -name manifest.json | wc -l
```

`agents truncated` 或 `map truncated` 是容量裁剪警告，不代表文件损坏。训练会使用固定
容量和 valid mask。若转换过程中断后重新执行，建议使用一个新的 `PREP_ROOT`，避免把
不同容量的旧 runtime 混到同一目录。

## 7. 筛选并创建 64/16 划分

该工具会执行最终筛选：schema、容量分组、地图非空，以及至少两辆有效攻击候选车。

```bash
"$UV_BIN" run --frozen --group train python -m gpudrive_cuda.rl.split_runtime \
  --runtime-root "$RUNTIME_DIR" \
  --output "$SPLIT_FILE" \
  --train 64 \
  --eval 16 \
  --seed 42 \
  --min-attackers 2
```

成功输出必须是：

```text
train scenes: 64
eval scenes: 16
```

再次检查 scene ID 不重复：

```bash
export SPLIT_FILE
"$UV_BIN" run --frozen --group train python - <<'PY'
import json
import os
from pathlib import Path

split_path = Path(os.environ["SPLIT_FILE"])
split = json.loads(split_path.read_text(encoding="utf-8"))
root = Path(split["runtime_root"])

def source_ids(names):
    result = []
    for name in names:
        manifest = json.loads((root / name / "manifest.json").read_text(encoding="utf-8"))
        result.append(manifest["source"]["source_scenario_id"])
    return result

train = source_ids(split["train"])
evaluation = source_ids(split["eval"])
assert len(train) == len(set(train)) == 64
assert len(evaluation) == len(set(evaluation)) == 16
assert set(train).isdisjoint(evaluation)
print("unique 64/16 split: OK")
print("runtime root:", root)
PY
```

当前 `split_runtime` 保证 train/eval 的 **scene token 不重叠**，但没有按原始 log/数据库
分组。同一个驾驶 log 中的不同 scene 仍可能分别进入 train 和 eval。第一版闭环训练可以
先采用这个划分；需要报告跨日志泛化能力时，应再增加按 `source_file` 分组的 log-disjoint
划分，不能把当前结果称为严格的跨日志泛化评估。

如果提示 `largest capacity group has 1`，说明同一 runtime 根目录混入了不同容量的旧
输出；在新的空 `RUNTIME_DIR` 中用同一条 `compile-rl` 命令重新编译。若最大组容量一致
但不足 80，说明通过地图和攻击车筛选的场景不够，需要转换更多候选 scene。

## 8. 编译 CUDA Torch 扩展并测试

```bash
export CMAKE_CUDA_ARCHITECTURES=89
./gpudrive_cuda/scripts/build_torch_extension.sh

export PYTHONPATH="$REPO_ROOT/build/gpudrive_cuda_rl/python:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

"$UV_BIN" run --frozen --group train python -m unittest \
  gpudrive_cuda.tests.test_rl \
  gpudrive_cuda.tests.test_torch_bridge -v
```

必须确认 `test_torch_bridge` 实际执行并通过，而不是因为找不到 CUDA 扩展而 `skipped`。

## 9. 启动单卡 RTX 4090 正式训练

训练配置会读取刚生成的固定路径：

```text
dataset/nuplan/rl_runtime/runtime_split.json
```

建议使用 `tmux`：

```bash
tmux new -s dangermaker
cd "$REPO_ROOT"
mkdir -p outputs/mappo_rtx4090
set -o pipefail
./gpudrive_cuda/scripts/run_mappo_rtx4090.sh \
  2>&1 | tee outputs/mappo_rtx4090/train.log
```

不要先执行 `run_mappo_mock.sh` 来代替正式训练。mock 只用于验证工程闭环；上述 64/16
真实 split 成功后，应直接使用 `run_mappo_rtx4090.sh`。

恢复 checkpoint：

```bash
./gpudrive_cuda/scripts/run_mappo_rtx4090.sh \
  outputs/mappo_rtx4090/checkpoints/step_001007616.pt
```

## 10. 打开 TensorBoard

在工作站运行：

```bash
./gpudrive_cuda/scripts/run_tensorboard.sh
```

在自己的电脑建立 SSH 隧道：

```bash
ssh -N -L 6006:127.0.0.1:6006 user@workstation
```

浏览器打开 `http://127.0.0.1:6006`。重点观察：

- `train/episode_success_rate`
- `train/episode_return_mean`
- `safety/min_distance_mean_m`
- `safety/min_ttc_mean_s_clipped_30`
- `safety/offroad_rate`
- `safety/non_ego_collision_rate`
- `ppo/approx_kl`
- `ppo/explained_variance`
- `system/world_steps_per_second`

更详细的训练指标、显存和断点恢复说明见
[RTX4090_MAPPO_TRAINING_GUIDE.md](RTX4090_MAPPO_TRAINING_GUIDE.md)。

## 11. 最终验收清单

数据准备：

```text
selected_scenes.tsv 记录固定候选 scene
CanonicalScenario 至少 80 个初步合格场景
RuntimeScenario 验证通过
train=64，eval=16，scene token 不重叠
训练与评估 source_scenario_id 不重复
所有 runtime capacities 完全一致
```

训练产物：

```text
outputs/mappo_rtx4090/metrics.jsonl
outputs/mappo_rtx4090/tensorboard/events.out.tfevents.*
outputs/mappo_rtx4090/checkpoints/final.pt
outputs/mappo_rtx4090/evaluation/eval_summary.json
outputs/mappo_rtx4090/evaluation/rollout.gif
outputs/mappo_rtx4090/evaluation/rollout.mp4
```

当前 ego 仍是固定参考轨迹控制器，不主动避障。该实验验证的是多场景
simulator-MAPPO 对抗训练闭环，不应表述为已经攻破真实自动驾驶规划系统。
