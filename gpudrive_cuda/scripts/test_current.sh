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

command -v uv >/dev/null 2>&1 || {
    echo "[error] uv not found; run gpudrive_cuda/scripts/setup_uv_env.sh first" >&2
    exit 1
}
command -v nvcc >/dev/null 2>&1 || {
    echo "[error] nvcc not found; run this test on the CUDA server" >&2
    exit 1
}
mkdir -p "${GPU_SIM_BUILD_DIR}" "${TEST_OUTPUT_DIR}"
cd "${REPO_ROOT}"
uv sync --frozen
uv run --frozen cmake -S "${PROJECT_ROOT}" -B "${GPU_SIM_BUILD_DIR}" -G Ninja -DBUILD_TESTING=ON
uv run --frozen cmake --build "${GPU_SIM_BUILD_DIR}" -j
"${GPU_SIM_BUILD_DIR}/runtime_loader_test" "${RUNTIME_PATH}"
"${GPU_SIM_BUILD_DIR}/simulator_integration_test" "${RUNTIME_PATH}"
"${BIN_PATH}" \
    --runtime "${RUNTIME_PATH}" \
    --worlds 2 \
    --steps 10 \
    --output "${TEST_OUTPUT_DIR}" | tee "${LOG_PATH}"

grep -q "simulation complete" "${LOG_PATH}"
grep -q "world,step,agent_slot" "${TEST_OUTPUT_DIR}/trace.csv"
grep -q "actual_acceleration,actual_steering" "${TEST_OUTPUT_DIR}/trace.csv"
uv run --frozen python - "${TEST_OUTPUT_DIR}/trace.csv" <<'PY'
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

uv run --frozen python "${PROJECT_ROOT}/tools/render_trace.py" \
    --runtime "${RUNTIME_PATH}" \
    --trace "${TEST_OUTPUT_DIR}/trace.csv" \
    --output "${TEST_OUTPUT_DIR}/rollout.gif" \
    --mp4-output "${TEST_OUTPUT_DIR}/rollout.mp4" \
    --duration 1 \
    --final-png "${TEST_OUTPUT_DIR}/final_frame.png"
test -s "${TEST_OUTPUT_DIR}/rollout.gif"
test -s "${TEST_OUTPUT_DIR}/rollout.mp4"
test -s "${TEST_OUTPUT_DIR}/final_frame.png"
uv run --frozen python - "${TEST_OUTPUT_DIR}/rollout.gif" "${TEST_OUTPUT_DIR}/rollout.mp4" <<'PY'
import sys
from collections import Counter

import imageio_ffmpeg
from PIL import Image

with Image.open(sys.argv[1]) as animation:
    if animation.size != (1600, 900) or animation.n_frames != 10:
        raise SystemExit(
            f"unexpected GIF: size={animation.size}, frames={animation.n_frames}"
        )
    animation.seek(animation.n_frames - 1)
    rgb = animation.convert("RGB")
    pixels = rgb.get_flattened_data() if hasattr(rgb, "get_flattened_data") else rgb.getdata()
    colors = Counter(pixels)
    if colors[(8, 124, 240)] < 5 or colors[(255, 122, 18)] < 5:
        raise SystemExit("GIF does not contain visible ego and traffic agents")
frames, seconds = imageio_ffmpeg.count_frames_and_secs(sys.argv[2])
if frames != 20 or abs(seconds - 1.0) > 0.1:
    raise SystemExit(f"unexpected MP4: frames={frames}, duration={seconds}")
print(f"[test] GIF frames: 10; MP4 frames: {frames}; duration: {seconds:.3f}s")
PY

echo "[test] CUDA simulator integration ok"
