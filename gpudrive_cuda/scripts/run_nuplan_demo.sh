#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
RUNTIME_PATH="${1:-${REPO_ROOT}/dataset/nuplan/rl_runtime/mock_nuplan_00100000}"
DEMO_OUTPUT_DIR="${2:-${REPO_ROOT}/outputs/nuplan_cuda_demo}"
NUM_WORLDS="${3:-1}"
GPU_SIM_BUILD_DIR="${GPU_SIM_BUILD_DIR:-${PROJECT_ROOT}/build}"

command -v cmake >/dev/null 2>&1 || {
    echo "[error] cmake not found" >&2
    exit 1
}
command -v nvcc >/dev/null 2>&1 || {
    echo "[error] nvcc not found; activate the CUDA environment" >&2
    exit 1
}
command -v python3 >/dev/null 2>&1 || {
    echo "[error] python3 not found" >&2
    exit 1
}
python3 - <<'PY'
try:
    import matplotlib
    from PIL import Image
except ImportError as error:
    raise SystemExit(f"[error] rendering dependency missing: {error}; install matplotlib and Pillow")
PY

mkdir -p "${GPU_SIM_BUILD_DIR}" "${DEMO_OUTPUT_DIR}"
cd "${REPO_ROOT}"
python3 -m scenario_pipeline validate-rl --input "${RUNTIME_PATH}"
cmake -S "${PROJECT_ROOT}" -B "${GPU_SIM_BUILD_DIR}" -DBUILD_TESTING=ON
cmake --build "${GPU_SIM_BUILD_DIR}" -j
"${GPU_SIM_BUILD_DIR}/runtime_loader_test" "${RUNTIME_PATH}"
"${GPU_SIM_BUILD_DIR}/simulator_integration_test" "${RUNTIME_PATH}"
"${GPU_SIM_BUILD_DIR}/drive_sim_cli" \
    --runtime "${RUNTIME_PATH}" \
    --worlds "${NUM_WORLDS}" \
    --output "${DEMO_OUTPUT_DIR}"
python3 "${PROJECT_ROOT}/tools/render_trace.py" \
    --runtime "${RUNTIME_PATH}" \
    --trace "${DEMO_OUTPUT_DIR}/trace.csv" \
    --output "${DEMO_OUTPUT_DIR}/rollout.gif" \
    --final-png "${DEMO_OUTPUT_DIR}/final_frame.png"

echo "[ok] trace: ${DEMO_OUTPUT_DIR}/trace.csv"
echo "[ok] summary: ${DEMO_OUTPUT_DIR}/summary.json"
echo "[ok] animation: ${DEMO_OUTPUT_DIR}/rollout.gif"
echo "[ok] final frame: ${DEMO_OUTPUT_DIR}/final_frame.png"
