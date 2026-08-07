#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${PROJECT_ROOT}/build_test}"
BIN_PATH="${BUILD_DIR}/drive_sim"
LOG_PATH="${BUILD_DIR}/drive_sim.out"

echo "[test] project root: ${PROJECT_ROOT}"
echo "[test] build dir:    ${BUILD_DIR}"

command -v cmake >/dev/null 2>&1 || {
    echo "[error] cmake not found. Install it in your active environment first." >&2
    exit 1
}

command -v nvcc >/dev/null 2>&1 || {
    echo "[error] nvcc not found. Activate your CUDA environment first." >&2
    exit 1
}

mkdir -p "${BUILD_DIR}"

echo "[test] configure"
cmake -S "${PROJECT_ROOT}" -B "${BUILD_DIR}"

echo "[test] build"
cmake --build "${BUILD_DIR}" -j

echo "[test] run"
"${BIN_PATH}" | tee "${LOG_PATH}"

echo "[test] validate output"
grep -q "finished step 10" "${LOG_PATH}"
grep -q "reset world 1" "${LOG_PATH}"
grep -q "world 0" "${LOG_PATH}"
grep -q "observations world 0" "${LOG_PATH}"
grep -q "partner observations world 0" "${LOG_PATH}"

echo "[test] ok"
