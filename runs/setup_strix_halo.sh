#!/usr/bin/env bash
# Install AMD's Python 3.12 / Linux x86_64 wheels, then exercise the GPU.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    echo "Strix Halo setup requires native Linux x86_64. See docs/strix-halo.md." >&2
    exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "Install uv from https://docs.astral.sh/uv/getting-started/installation/ and rerun." >&2
    exit 1
fi

# Use the CPU dependency graph to avoid downloading NVIDIA libraries, but
# skip torch itself: AMD's wheel is installed explicitly in the next step.
uv sync --frozen --python 3.12 --extra cpu --no-install-package torch "$@"
uv pip install --python .venv/bin/python --no-deps --require-hashes -r requirements-rocm.txt
uv pip check --python .venv/bin/python
.venv/bin/python -m scripts.smoke_test --device-type cuda --require-rocm
echo "Setup complete. Run: bash runs/strix_halo.sh"
