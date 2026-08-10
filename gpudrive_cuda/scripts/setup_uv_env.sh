#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

command -v uv >/dev/null 2>&1 || {
    echo "[error] uv not found" >&2
    echo "Install it first: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    echo "Then open a new shell or add \$HOME/.local/bin to PATH." >&2
    exit 1
}

cd "${REPO_ROOT}"
echo "[setup] syncing locked Python 3.10 environment"
uv sync --frozen

echo "[setup] checking managed tools"
uv run --frozen python --version
uv run --frozen cmake --version | head -n 1
uv run --frozen ninja --version
uv run --frozen python - <<'PY'
import matplotlib
from PIL import Image

print(f"matplotlib {matplotlib.__version__}")
print(f"Pillow {Image.__version__}")
PY

if command -v nvcc >/dev/null 2>&1; then
    nvcc --version | tail -n 1
    echo "[setup] CUDA compiler found"
else
    echo "[warning] nvcc not found" >&2
    echo "The uv environment is ready, but CUDA toolkit must be installed system-wide before building." >&2
fi

echo "[setup] environment ready: ${REPO_ROOT}/.venv"
