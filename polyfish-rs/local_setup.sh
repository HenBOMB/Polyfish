#!/usr/bin/env bash
# Local dev box: macOS/Apple silicon, or a plain Linux CPU box.
# Shared core comes from requirements.txt; the per-target extra here is the
# torch wheel — PyPI on macOS (MPS, and the libtorch tch-eval links against),
# the CPU index on Linux so a laptop does not pull ~3GB of CUDA wheels.
set -euo pipefail

cd "$(dirname "$0")"

TORCH_VERSION="$(sed -n 's/^# *POLYFISH_TORCH_VERSION=//p' requirements.txt | head -n 1)"
if [ -z "$TORCH_VERSION" ]; then
    echo "requirements.txt is missing the POLYFISH_TORCH_VERSION pin" >&2
    exit 1
fi

if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
.venv/bin/pip install --upgrade pip

INSTALLED="$(.venv/bin/python3 -c 'import torch,sys; sys.stdout.write(torch.__version__)' 2>/dev/null || true)"
if [ -z "$INSTALLED" ] || [ "${POLYFISH_FORCE_TORCH:-0}" = 1 ]; then
    if [ "$(uname -s)" = "Darwin" ]; then
        .venv/bin/pip install "torch==$TORCH_VERSION"
    else
        .venv/bin/pip install "torch==$TORCH_VERSION" --index-url https://download.pytorch.org/whl/cpu
    fi
elif [ "${INSTALLED%%+*}" != "$TORCH_VERSION" ]; then
    echo "WARNING: torch $INSTALLED is installed but the pin is $TORCH_VERSION." >&2
    echo "         tch-eval requires 2.12.x; re-run with POLYFISH_FORCE_TORCH=1 to install the pin." >&2
fi

.venv/bin/pip install -r requirements.txt

echo "=== Environment ==="
.venv/bin/python3 -c 'import torch, numpy, safetensors; print("torch", torch.__version__, "| numpy", numpy.__version__)'
if [ "$(uname -s)" = "Darwin" ]; then
    cat <<EOF

For a tch-eval / metal-eval build, export first:
  export LIBTORCH_USE_PYTORCH=1 LIBTORCH_BYPASS_VERSION_CHECK=1
  export PATH="$PWD/.venv/bin:\$PATH"
EOF
fi
