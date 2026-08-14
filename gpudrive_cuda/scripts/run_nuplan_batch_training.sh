#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
NUPLAN_DB_ROOT="${NUPLAN_DB_ROOT:-${REPO_ROOT}/data/dataset/nuPlan/raw/cache}"
NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${REPO_ROOT}/data/dataset/nuPlan/maps}"
NUPLAN_DEVKIT_ROOT="${NUPLAN_DEVKIT_ROOT:-/data/fc/hugsim/atg/nuplan-devkit}"
NUPLAN_PYTHON="${NUPLAN_PYTHON:-/home/user/anaconda3/envs/gpudrive/bin/python}"
UV_BIN="${UV_BIN:-/home/user/anaconda3/envs/gpudrive/bin/uv}"
PREP_ROOT="${PREP_ROOT:-/data/user/agent_runs/nuplan_mappo_128}"
CANDIDATE_SCENES="${CANDIDATE_SCENES:-128}"
SELECTION_SEED="${SELECTION_SEED:-42}"
SELECTION_FILE="${PREP_ROOT}/selected_scenes.tsv"
CANONICAL_DIR="${PREP_ROOT}/canonical"
RUNTIME_DIR="${PREP_ROOT}/runtime"
SPLIT_FILE="${REPO_ROOT}/dataset/nuplan/rl_runtime/runtime_split.json"
CONVERSION_LOG="${PREP_ROOT}/convert.log"
FAILED_SCENES="${PREP_ROOT}/failed_scenes.tsv"

export REPO_ROOT NUPLAN_DB_ROOT NUPLAN_MAPS_ROOT NUPLAN_DEVKIT_ROOT
export NUPLAN_PYTHON UV_BIN PREP_ROOT CANDIDATE_SCENES SELECTION_SEED
export SELECTION_FILE CANONICAL_DIR RUNTIME_DIR SPLIT_FILE
export PYTHONPATH="${REPO_ROOT}:${NUPLAN_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-89}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${PREP_ROOT}" "${CANONICAL_DIR}" "${RUNTIME_DIR}" "$(dirname "${SPLIT_FILE}")"
cd "${REPO_ROOT}"

"${NUPLAN_PYTHON}" - <<'PY'
import nuplan
print("[ok] nuplan import:", list(nuplan.__path__))
PY
"${UV_BIN}" sync --frozen --group train

"${NUPLAN_PYTHON}" - <<'PY'
import os
import random
import sqlite3
from pathlib import Path

root = Path(os.environ["NUPLAN_DB_ROOT"]).resolve()
output = Path(os.environ["SELECTION_FILE"]).resolve()
limit = int(os.environ["CANDIDATE_SCENES"])
seed = int(os.environ["SELECTION_SEED"])
scenes_by_id = {}
valid_databases = 0
raw_scenes = 0
for database in sorted(root.rglob("*.db")):
    try:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        rows = connection.execute(
            "SELECT lower(hex(token)) FROM scene ORDER BY name, token"
        ).fetchall()
        connection.close()
    except sqlite3.Error:
        continue
    if rows:
        valid_databases += 1
        raw_scenes += len(rows)
    for row in rows:
        scenes_by_id.setdefault(str(row[0]), str(database.resolve()))
scenes = [(database, scene_id) for scene_id, database in sorted(scenes_by_id.items())]
if len(scenes) < 80:
    raise SystemExit(f"only {len(scenes)} unique raw scenes; need at least 80")
random.Random(seed).shuffle(scenes)
selected = scenes[: min(limit, len(scenes))]
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", encoding="utf-8") as stream:
    for database, scene_id in selected:
        stream.write(f"{database}\t{scene_id}\n")
print(f"[ok] databases={valid_databases} raw_scenes={raw_scenes} unique={len(scenes)}")
print(f"[ok] selected candidates={len(selected)} file={output}")
PY

: > "${CONVERSION_LOG}"
: > "${FAILED_SCENES}"
succeeded=0
failed=0
while IFS=$'\t' read -r database scene_id; do
    echo "[convert] ${scene_id} from ${database}" | tee -a "${CONVERSION_LOG}"
    if "${NUPLAN_PYTHON}" -m scenario_pipeline convert-nuplan \
        --input "${database}" \
        --scene-id "${scene_id}" \
        --maps-root "${NUPLAN_MAPS_ROOT}" \
        --output "${CANONICAL_DIR}" 2>&1 | tee -a "${CONVERSION_LOG}"; then
        succeeded=$((succeeded + 1))
    else
        printf '%s\t%s\n' "${database}" "${scene_id}" >> "${FAILED_SCENES}"
        failed=$((failed + 1))
    fi
done < "${SELECTION_FILE}"
echo "[convert] succeeded=${succeeded} failed=${failed}"

"${NUPLAN_PYTHON}" -m scenario_pipeline validate --input "${CANONICAL_DIR}" \
    2>&1 | tee "${PREP_ROOT}/validate_canonical.log"

"${NUPLAN_PYTHON}" - <<'PY'
import os
from pathlib import Path
from scenario_pipeline.io import read_scenario

root = Path(os.environ["CANONICAL_DIR"])
total = map_ready = eligible = 0
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
    retained = sorted(scenario.agents, key=order_key)[:64]
    attackers = sum(
        not agent.is_ego and agent.type == "vehicle" and bool(agent.valid[anchor])
        for agent in retained
    )
    has_ego = sum(agent.is_ego for agent in scenario.agents) == 1
    has_map = scenario.quality.map_available and bool(scenario.map_features)
    map_ready += int(has_map)
    eligible += int(has_ego and has_map and attackers >= 2)
print(f"[canonical] files={total} unique={len(source_ids)} map_ready={map_ready} eligible={eligible}")
if eligible < 80:
    raise SystemExit("fewer than 80 preliminary scenes; convert more candidates")
PY

"${UV_BIN}" run --frozen --group train python -m scenario_pipeline compile-rl \
    --input "${CANONICAL_DIR}" \
    --output "${RUNTIME_DIR}" \
    --max-agents 64 \
    --history-steps 11 \
    --max-future-steps 128 \
    --max-map-features 2048 \
    --max-map-points 32768 \
    --max-map-edges 8192 \
    --max-traffic-lights 128 \
    --max-route-features 512 \
    2>&1 | tee "${PREP_ROOT}/compile_runtime.log"

"${UV_BIN}" run --frozen --group train python -m scenario_pipeline validate-rl \
    --input "${RUNTIME_DIR}" 2>&1 | tee "${PREP_ROOT}/validate_runtime.log"

"${UV_BIN}" run --frozen --group train python -m gpudrive_cuda.rl.split_runtime \
    --runtime-root "${RUNTIME_DIR}" \
    --output "${SPLIT_FILE}" \
    --train 64 \
    --eval 16 \
    --seed 42 \
    --min-attackers 2

"${UV_BIN}" run --frozen --group train python - <<'PY'
import json
import os
from pathlib import Path

split_path = Path(os.environ["SPLIT_FILE"])
split_data = json.loads(split_path.read_text(encoding="utf-8"))
root = Path(split_data["runtime_root"])
def source_ids(names):
    return [
        json.loads((root / name / "manifest.json").read_text(encoding="utf-8"))["source"]["source_scenario_id"]
        for name in names
    ]
train = source_ids(split_data["train"])
evaluation = source_ids(split_data["eval"])
assert len(train) == len(set(train)) == 64
assert len(evaluation) == len(set(evaluation)) == 16
assert set(train).isdisjoint(evaluation)
print("[ok] unique 64/16 split:", split_path)
PY

./gpudrive_cuda/scripts/build_torch_extension.sh
export PYTHONPATH="${REPO_ROOT}/build/gpudrive_cuda_rl/python:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
"${UV_BIN}" run --frozen --group train python -m unittest \
    gpudrive_cuda.tests.test_rl gpudrive_cuda.tests.test_torch_bridge -v

mkdir -p "${REPO_ROOT}/outputs/mappo_rtx4090"
./gpudrive_cuda/scripts/run_mappo_rtx4090.sh
