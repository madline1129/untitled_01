#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
RUNTIME_PATH="${1:-${REPO_ROOT}/dataset/nuplan/rl_runtime/mock_nuplan_00100000}"
GPU_SIM_BUILD_DIR="${GPU_SIM_BUILD_DIR:-${PROJECT_ROOT}/build_test}"
TEST_OUTPUT_DIR="${TEST_OUTPUT_DIR:-${REPO_ROOT}/outputs/gpudrive_cuda_test}"
BIN_PATH="${GPU_SIM_BUILD_DIR}/drive_sim_cli"
LOG_PATH="${TEST_OUTPUT_DIR}/drive_sim_cli.out"

command -v cmake >/dev/null 2>&1 || {
    echo "[error] cmake not found" >&2
    exit 1
}
command -v nvcc >/dev/null 2>&1 || {
    echo "[error] nvcc not found; run this test on the CUDA server" >&2
    exit 1
}
mkdir -p "${GPU_SIM_BUILD_DIR}" "${TEST_OUTPUT_DIR}"
cmake -S "${PROJECT_ROOT}" -B "${GPU_SIM_BUILD_DIR}" -DBUILD_TESTING=ON
cmake --build "${GPU_SIM_BUILD_DIR}" -j
"${GPU_SIM_BUILD_DIR}/runtime_loader_test" "${RUNTIME_PATH}"
"${GPU_SIM_BUILD_DIR}/simulator_integration_test" "${RUNTIME_PATH}"
"${BIN_PATH}" \
    --runtime "${RUNTIME_PATH}" \
    --worlds 2 \
    --steps 10 \
    --output "${TEST_OUTPUT_DIR}" | tee "${LOG_PATH}"

grep -q "simulation complete" "${LOG_PATH}"
grep -q "world,step,agent_slot" "${TEST_OUTPUT_DIR}/trace.csv"
python3 - "${TEST_OUTPUT_DIR}/trace.csv" <<'PY'
import csv
import math
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as stream:
    rows = [row for row in csv.DictReader(stream) if row["world"] == "0" and row["agent_slot"] == "0"]
if len(rows) < 2:
    raise SystemExit("ego trace has fewer than two frames")
distance = math.hypot(float(rows[-1]["x"]) - float(rows[0]["x"]), float(rows[-1]["y"]) - float(rows[0]["y"]))
if not math.isfinite(distance) or distance <= 0.01:
    raise SystemExit(f"ego did not move: displacement={distance}")
print(f"[test] ego displacement: {distance:.3f} m")
PY

echo "[test] CUDA simulator integration ok"
