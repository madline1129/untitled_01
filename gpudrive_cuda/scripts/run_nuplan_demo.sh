#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
RUNTIME_PATH="${1:-${REPO_ROOT}/dataset/nuplan/rl_runtime/mock_nuplan_00100000}"
DEMO_OUTPUT_DIR="${2:-${REPO_ROOT}/outputs/nuplan_cuda_demo}"
NUM_WORLDS="${3:-1}"
GPU_SIM_BUILD_DIR="${GPU_SIM_BUILD_DIR:-${PROJECT_ROOT}/build}"
UV_BIN="${UV_BIN:-uv}"
DEMO_DURATION_SECONDS="${DEMO_DURATION_SECONDS:-10}"

command -v "${UV_BIN}" >/dev/null 2>&1 || {
    echo "[error] uv not found at ${UV_BIN}; run gpudrive_cuda/scripts/setup_uv_env.sh first" >&2
    exit 1
}
command -v nvcc >/dev/null 2>&1 || {
    echo "[error] nvcc not found; activate the CUDA environment" >&2
    exit 1
}

mkdir -p "${GPU_SIM_BUILD_DIR}" "${DEMO_OUTPUT_DIR}"
cd "${REPO_ROOT}"
"${UV_BIN}" sync --frozen
"${UV_BIN}" run --frozen python -m scenario_pipeline validate-rl --input "${RUNTIME_PATH}"
"${UV_BIN}" run --frozen cmake -S "${PROJECT_ROOT}" -B "${GPU_SIM_BUILD_DIR}" -G Ninja -DBUILD_TESTING=ON
"${UV_BIN}" run --frozen cmake --build "${GPU_SIM_BUILD_DIR}" -j
"${GPU_SIM_BUILD_DIR}/runtime_loader_test" "${RUNTIME_PATH}"
"${GPU_SIM_BUILD_DIR}/simulator_integration_test" "${RUNTIME_PATH}"

DEMO_STEPS="$("${UV_BIN}" run --frozen python - "${RUNTIME_PATH}" "${NUM_WORLDS}" "${DEMO_DURATION_SECONDS}" <<'PY'
import json
import math
import sys
from pathlib import Path

runtime = Path(sys.argv[1])
worlds = int(sys.argv[2])
duration = float(sys.argv[3])
if duration <= 0:
    raise SystemExit("DEMO_DURATION_SECONDS must be positive")
if (runtime / "manifest.json").is_file():
    directories = [runtime]
else:
    directories = sorted({path.parent for path in runtime.rglob("manifest.json")})
if not directories:
    raise SystemExit(f"no runtime manifests found under: {runtime}")
selected = [directories[index % len(directories)] for index in range(worlds)]
manifests = [json.loads((directory / "manifest.json").read_text()) for directory in selected]
dt = float(manifests[0]["dt"])
if any(not math.isclose(float(manifest["dt"]), dt, rel_tol=0.0, abs_tol=1e-6) for manifest in manifests):
    raise SystemExit("all demo worlds must use the same dt")
steps = int(round(duration / dt))
if not math.isclose(steps * dt, duration, rel_tol=0.0, abs_tol=1e-5):
    raise SystemExit(f"duration {duration}s is not an integer multiple of runtime dt {dt}s")
for directory, manifest in zip(selected, manifests):
    if int(manifest["episode_steps"]) < steps:
        raise SystemExit(
            f"runtime {directory} only covers {int(manifest['episode_steps']) * dt:.3f}s; "
            f"the demo requires {duration:.3f}s"
        )
print(steps)
PY
)"

"${GPU_SIM_BUILD_DIR}/drive_sim_cli" \
    --runtime "${RUNTIME_PATH}" \
    --worlds "${NUM_WORLDS}" \
    --steps "${DEMO_STEPS}" \
    --output "${DEMO_OUTPUT_DIR}"
"${UV_BIN}" run --frozen python "${PROJECT_ROOT}/tools/render_trace.py" \
    --runtime "${RUNTIME_PATH}" \
    --trace "${DEMO_OUTPUT_DIR}/trace.csv" \
    --output "${DEMO_OUTPUT_DIR}/rollout.gif" \
    --mp4-output "${DEMO_OUTPUT_DIR}/rollout.mp4" \
    --duration "${DEMO_DURATION_SECONDS}" \
    --final-png "${DEMO_OUTPUT_DIR}/final_frame.png"

"${UV_BIN}" run --frozen python - "${DEMO_OUTPUT_DIR}" "${DEMO_DURATION_SECONDS}" "${DEMO_STEPS}" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

import imageio_ffmpeg
from PIL import Image

root = Path(sys.argv[1])
duration = float(sys.argv[2])
expected_steps = int(sys.argv[3])
summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
if int(summary["executed_steps"]) != expected_steps:
    raise SystemExit(
        f"simulator step mismatch: expected={expected_steps}, actual={summary['executed_steps']}"
    )
with Image.open(root / "rollout.gif") as animation:
    expected_gif_frames = int(round(duration * 10))
    if animation.size != (1600, 900) or animation.n_frames != expected_gif_frames:
        raise SystemExit(
            f"unexpected GIF: size={animation.size}, frames={animation.n_frames}, "
            f"expected_frames={expected_gif_frames}"
        )
with Image.open(root / "final_frame.png") as image:
    if image.size != (1600, 900):
        raise SystemExit(f"unexpected final PNG size: {image.size}")
    rgb = image.convert("RGB")
    pixels = rgb.get_flattened_data() if hasattr(rgb, "get_flattened_data") else rgb.getdata()
    colors = Counter(pixels)
    if colors[(8, 124, 240)] < 5 or colors[(255, 122, 18)] < 5:
        raise SystemExit("final PNG does not contain visible ego and traffic agents")
    if summary["scenes"][0]["map_features"] > 0 and colors[(48, 50, 54)] < 50:
        raise SystemExit("final PNG does not contain a visible drivable map surface")
mp4_frames, mp4_seconds = imageio_ffmpeg.count_frames_and_secs(str(root / "rollout.mp4"))
expected_mp4_frames = int(round(duration * 20))
if mp4_frames != expected_mp4_frames or abs(mp4_seconds - duration) > 0.1:
    raise SystemExit(
        f"unexpected MP4: frames={mp4_frames}, duration={mp4_seconds}, "
        f"expected_frames={expected_mp4_frames}, expected_duration={duration}"
    )
print(
    f"[ok] media: 1600x900, GIF={expected_gif_frames} frames, "
    f"MP4={mp4_frames} frames/{mp4_seconds:.2f}s"
)
PY

echo "[ok] trace: ${DEMO_OUTPUT_DIR}/trace.csv"
echo "[ok] summary: ${DEMO_OUTPUT_DIR}/summary.json"
echo "[ok] animation: ${DEMO_OUTPUT_DIR}/rollout.gif"
echo "[ok] video: ${DEMO_OUTPUT_DIR}/rollout.mp4"
echo "[ok] final frame: ${DEMO_OUTPUT_DIR}/final_frame.png"
