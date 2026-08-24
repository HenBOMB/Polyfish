#!/usr/bin/env bash
# Vast.ai box (PyTorch template, x86_64, CUDA 12.8+ GPU).
# Differences vs remote_setup.sh: cu128 first (Blackwell sm_120 kernels land
# there), and it builds BOTH eval backends (candle CUDA + tch-eval) so they can
# be compared with --eval-backend at runtime. Containers run as root with no
# systemd; run the loop inside tmux.
set -eo pipefail

cd "$(dirname "$0")"

TORCH_CUDA_INDEXES="${TORCH_CUDA_INDEXES:-cu128 cu129}"

apt-get update && apt-get install -y build-essential curl pkg-config libssl-dev python3-venv tmux git htop

if ! command -v cargo &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
export PATH="$HOME/.cargo/bin:$PATH"

TORCH_VERSION="$(sed -n 's/^# *POLYFISH_TORCH_VERSION=//p' requirements.txt | head -n 1)"
if [ -z "$TORCH_VERSION" ]; then
    echo "requirements.txt is missing the POLYFISH_TORCH_VERSION pin" >&2
    exit 1
fi

if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
.venv/bin/pip install --upgrade pip

# tch-eval links against this exact torch, so an off-pin wheel is fatal here
# rather than a warning: torch-sys' version check is bypassed at build time.
INSTALLED="$(.venv/bin/python3 -c 'import torch,sys; sys.stdout.write(torch.__version__)' 2>/dev/null || true)"
if [ -z "$INSTALLED" ] || [ "${POLYFISH_FORCE_TORCH:-0}" = 1 ]; then
    OK=0
    for IDX in $TORCH_CUDA_INDEXES; do
        if .venv/bin/pip install "torch==$TORCH_VERSION" --index-url "https://download.pytorch.org/whl/$IDX"; then
            OK=1
            break
        fi
    done
    if [ "$OK" -ne 1 ]; then
        echo "No CUDA wheel for torch==$TORCH_VERSION in: $TORCH_CUDA_INDEXES" >&2
        exit 1
    fi
elif [ "${INSTALLED%%+*}" != "$TORCH_VERSION" ]; then
    echo "torch $INSTALLED is installed but tch-eval needs the pin $TORCH_VERSION." >&2
    echo "Re-run with POLYFISH_FORCE_TORCH=1, or build without --features tch-eval." >&2
    exit 1
fi

.venv/bin/pip install -r requirements.txt

# tch-eval links against the venv's libtorch (no separate 2GB download).
# See Cargo.toml for why the bypass flag is safe within 2.12.x.
TORCH_LIB="$PWD/.venv/lib/$(.venv/bin/python3 -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')/site-packages/torch/lib"
export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1
export LD_LIBRARY_PATH="$TORCH_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PATH="$PWD/.venv/bin:$PATH" cargo build --release --no-default-features --features cuda,tch-eval \
    --bin self_play --bin polyfish --bin benchmark --bin arena

# Persist the env for interactive shells and tmux panes.
if ! grep -q "LIBTORCH_USE_PYTORCH" ~/.bashrc 2>/dev/null; then
    {
        echo "export PATH=\"\$HOME/.cargo/bin:$PWD/.venv/bin:\$PATH\""
        echo "export LIBTORCH_USE_PYTORCH=1"
        echo "export LIBTORCH_BYPASS_VERSION_CHECK=1"
        echo "export LD_LIBRARY_PATH=\"$TORCH_LIB\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}\""
    } >> ~/.bashrc
fi

echo "=== Sanity checks ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
.venv/bin/python3 -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
nproc
echo "=== Setup complete. Next: get model.safetensors onto the box (scp or init_model.py), then benchmark in tmux. ==="
