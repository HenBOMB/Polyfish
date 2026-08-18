#!/usr/bin/env bash
# Remote CUDA box (generic cloud GPU, Ubuntu-ish, apt available).
# Shared core comes from requirements.txt; the per-target extras here are the
# system packages, the Rust toolchain, a CUDA torch wheel, and a candle-CUDA
# release build. For a Vast.ai box use vast_setup.sh (adds the tch-eval build).
set -eo pipefail

cd "$(dirname "$0")"

# CUDA wheel index to try, in order. Override for a different driver/arch.
TORCH_CUDA_INDEXES="${TORCH_CUDA_INDEXES:-cu128 cu126}"

apt-get update && apt-get install -y build-essential curl pkg-config libssl-dev python3-venv tmux

if ! command -v cargo &> /dev/null; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    . "$HOME/.cargo/env"
else
    echo "Rust is already installed."
fi

if ! grep -q "export PATH=.*cargo/bin" ~/.bashrc 2>/dev/null; then
    echo 'export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"' >> ~/.bashrc
    echo "Added Rust and CUDA to PATH in .bashrc"
fi
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"

TORCH_VERSION="$(sed -n 's/^# *POLYFISH_TORCH_VERSION=//p' requirements.txt | head -n 1)"
if [ -z "$TORCH_VERSION" ]; then
    echo "requirements.txt is missing the POLYFISH_TORCH_VERSION pin" >&2
    exit 1
fi

if [ ! -d .venv ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi
.venv/bin/pip install --upgrade pip

INSTALLED="$(.venv/bin/python3 -c 'import torch,sys; sys.stdout.write(torch.__version__)' 2>/dev/null || true)"
if [ -z "$INSTALLED" ] || [ "${POLYFISH_FORCE_TORCH:-0}" = 1 ]; then
    OK=0
    for IDX in $TORCH_CUDA_INDEXES; do
        echo "Installing torch==$TORCH_VERSION ($IDX)..."
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
    echo "WARNING: torch $INSTALLED is installed but the pin is $TORCH_VERSION." >&2
    echo "         Re-run with POLYFISH_FORCE_TORCH=1 to install the pin." >&2
else
    echo "torch $INSTALLED already matches the pin. Skipping..."
fi

.venv/bin/pip install -r requirements.txt

# --no-default-features: opt out of the macOS `metal` default, which does not
# compile on Linux.
echo "Building PolyFish (Release)..."
cargo build --release --no-default-features --features cuda \
    --bin self_play --bin polyfish --bin benchmark --bin arena

echo "=== Sanity checks ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
.venv/bin/python3 -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
echo "Setup complete. Run the training loop inside tmux."
