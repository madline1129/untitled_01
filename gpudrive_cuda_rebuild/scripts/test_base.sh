#!/usr/bin/env bash
set -euo pipefail

echo "[error] gpudrive_cuda_rebuild is a paused WIP and is not buildable yet" >&2
echo "[error] use gpudrive_cuda/scripts/test_current.sh for the production simulator" >&2
exit 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${PROJECT_ROOT}/build}"
OUTPUT_FILE="${BUILD_DIR}/drive_sim_base.out"
UV_BIN="${UV_BIN:-uv}"

command -v "${UV_BIN}" >/dev/null 2>&1 || {
    echo "[error] uv not found at ${UV_BIN}" >&2
    exit 1
}
command -v nvcc >/dev/null 2>&1 || {
    echo "[error] nvcc not found" >&2
    exit 1
}

cd "${REPO_ROOT}"
"${UV_BIN}" sync --frozen
"${UV_BIN}" run --frozen cmake \
    -S "${PROJECT_ROOT}" \
    -B "${BUILD_DIR}" \
    -G Ninja
"${UV_BIN}" run --frozen cmake --build "${BUILD_DIR}" -j
"${BUILD_DIR}/drive_sim_base" | tee "${OUTPUT_FILE}"

grep -q "finished step 10" "${OUTPUT_FILE}"
grep -q "reset world 1" "${OUTPUT_FILE}"
grep -q "world 0" "${OUTPUT_FILE}"
grep -q "self observations world 0" "${OUTPUT_FILE}"
grep -q "partner observations world 0" "${OUTPUT_FILE}"

echo "[test] base simulator ok"
