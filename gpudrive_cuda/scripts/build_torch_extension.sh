#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${GPUDRIVE_RL_BUILD_DIR:-${ROOT_DIR}/build/gpudrive_cuda_rl}"

cd "${ROOT_DIR}"
uv sync --frozen --group train

TORCH_CMAKE_PREFIX="$(uv run --frozen --group train python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
PYBIND11_CMAKE_DIR="$(uv run --frozen --group train python -c 'import pybind11; print(pybind11.get_cmake_dir())')"

CMAKE_ARGS=(
    -S "${ROOT_DIR}/gpudrive_cuda"
    -B "${BUILD_DIR}"
    -G Ninja
    -DCMAKE_BUILD_TYPE=Release
    -DBUILD_TESTING=ON
    -DBUILD_TORCH_BINDINGS=ON
    "-DCMAKE_PREFIX_PATH=${TORCH_CMAKE_PREFIX};${PYBIND11_CMAKE_DIR}"
)
if [[ -n "${CMAKE_CUDA_ARCHITECTURES:-}" ]]; then
    CMAKE_ARGS+=("-DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
fi

cmake "${CMAKE_ARGS[@]}"
cmake --build "${BUILD_DIR}" --parallel

echo "[ok] Torch extension: ${BUILD_DIR}/python/gpudrive_cuda_torch"
echo "export PYTHONPATH=${BUILD_DIR}/python:${ROOT_DIR}"
