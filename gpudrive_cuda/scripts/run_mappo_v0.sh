#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${GPUDRIVE_RL_BUILD_DIR:-${ROOT_DIR}/build/gpudrive_cuda_rl}"
OUTPUT_DIR="${ROOT_DIR}/outputs/mappo_v0"
CONFIG="${ROOT_DIR}/gpudrive_cuda/configs/mappo_v0.json"

"${ROOT_DIR}/gpudrive_cuda/scripts/build_torch_extension.sh"
export PYTHONPATH="${BUILD_DIR}/python:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT_DIR}"

uv run --frozen --group train python -m gpudrive_cuda.rl.train --config "${CONFIG}"
uv run --frozen --group train python -m gpudrive_cuda.rl.evaluate \
    --config "${CONFIG}" \
    --checkpoint "${OUTPUT_DIR}/checkpoints/final.pt" \
    --output "${OUTPUT_DIR}/evaluation" \
    --render

echo "[ok] run directory: ${OUTPUT_DIR}"
