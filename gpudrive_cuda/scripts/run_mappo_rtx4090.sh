#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${GPUDRIVE_RL_BUILD_DIR:-${ROOT_DIR}/build/gpudrive_cuda_rl}"
CONFIG="${ROOT_DIR}/gpudrive_cuda/configs/mappo_rtx4090.json"
OUTPUT_DIR="${ROOT_DIR}/outputs/mappo_rtx4090"
RESUME_CHECKPOINT="${1:-${RESUME_CHECKPOINT:-}}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-89}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${ROOT_DIR}"
uv run --frozen --group train python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA")
properties = torch.cuda.get_device_properties(0)
print(f"[gpu] {properties.name}")
print(f"[gpu] memory: {properties.total_memory / 1024**3:.1f} GiB")
print(f"[torch] {torch.__version__}, CUDA {torch.version.cuda}")
if properties.major != 8 or properties.minor != 9:
    print("[warning] expected RTX 4090 compute capability 8.9")
PY

"${ROOT_DIR}/gpudrive_cuda/scripts/build_torch_extension.sh"
export PYTHONPATH="${BUILD_DIR}/python:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

TRAIN_ARGS=(
    --config "${CONFIG}"
)
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
    TRAIN_ARGS+=(--resume "${RESUME_CHECKPOINT}")
fi

uv run --frozen --group train python -m gpudrive_cuda.rl.train "${TRAIN_ARGS[@]}"
uv run --frozen --group train python -m gpudrive_cuda.rl.evaluate \
    --config "${CONFIG}" \
    --checkpoint "${OUTPUT_DIR}/checkpoints/final.pt" \
    --output "${OUTPUT_DIR}/evaluation" \
    --render

echo "[ok] checkpoint: ${OUTPUT_DIR}/checkpoints/final.pt"
echo "[ok] TensorBoard: ${OUTPUT_DIR}/tensorboard"
echo "[ok] evaluation: ${OUTPUT_DIR}/evaluation"
