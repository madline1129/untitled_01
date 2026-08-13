#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${GPUDRIVE_RL_BUILD_DIR:-${ROOT_DIR}/build/gpudrive_cuda_rl}"
OUTPUT_DIR="${ROOT_DIR}/outputs/mappo_mock"

cd "${ROOT_DIR}"
uv run --frozen python -m gpudrive_cuda.rl.create_mock_runtime
"${ROOT_DIR}/gpudrive_cuda/scripts/build_torch_extension.sh"
export PYTHONPATH="${BUILD_DIR}/python:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

uv run --frozen --group train python -m gpudrive_cuda.rl.train \
    --config gpudrive_cuda/configs/mappo_mock.json

uv run --frozen --group train python -m gpudrive_cuda.rl.evaluate \
    --config gpudrive_cuda/configs/mappo_mock.json \
    --checkpoint "${OUTPUT_DIR}/checkpoints/final.pt" \
    --output "${OUTPUT_DIR}/evaluation" \
    --render

echo "[ok] metrics: ${OUTPUT_DIR}/metrics.jsonl"
echo "[ok] checkpoint: ${OUTPUT_DIR}/checkpoints/final.pt"
echo "[ok] GIF: ${OUTPUT_DIR}/evaluation/rollout.gif"
