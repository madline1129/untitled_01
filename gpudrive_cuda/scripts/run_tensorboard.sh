#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${1:-${ROOT_DIR}/outputs/mappo_rtx4090/tensorboard}"
HOST="${TENSORBOARD_HOST:-127.0.0.1}"
PORT="${TENSORBOARD_PORT:-6006}"

if [[ ! -d "${LOG_DIR}" ]]; then
    echo "TensorBoard log directory does not exist: ${LOG_DIR}" >&2
    exit 1
fi

cd "${ROOT_DIR}"
exec uv run --frozen --group train tensorboard \
    --logdir "${LOG_DIR}" \
    --host "${HOST}" \
    --port "${PORT}"
