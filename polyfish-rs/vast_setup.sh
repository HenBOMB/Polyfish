#!/bin/bash
# Setup for a Vast.ai box (PyTorch template, x86_64, CUDA 12.8+ GPU).
# Differences vs remote_setup.sh: torch 2.12.1 cu128 (matches the tch 0.25
# pin; Blackwell needs cu128+), and builds BOTH eval backends (candle CUDA +
# tch-eval) so they can be compared with --eval-backend at runtime.
# Vast containers run as root with no systemd; run the loop inside tmux.
set -e

if [[ "$(basename "$PWD")" != "polyfish-rs" ]]; then
    cd polyfish-rs || exit 1
fi

apt-get update && apt-get install -y build-essential curl pkg-config libssl-dev python3-venv tmux git htop

if ! command -v cargo &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
export PATH="$HOME/.cargo/bin:$PATH"

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
.venv/bin/pip install --upgrade pip
# Torch pinned to match tch 0.25 (targets PyTorch 2.12). cu128 is the first
# CUDA build with Blackwell (sm_120) kernels. Fall back to cu129 if missing.
if ! .venv/bin/python3 -c "import torch" &> /dev/null; then
    .venv/bin/pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu128 \
      || .venv/bin/pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu129
fi
.venv/bin/pip install -r requirements.txt

# tch-eval links against the venv's libtorch (no separate 2GB download).
# See Cargo.toml lines 33-43 for why the bypass flag is safe.
TORCH_LIB="$PWD/.venv/lib/$(.venv/bin/python3 -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')/site-packages/torch/lib"
export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1
export LD_LIBRARY_PATH="$TORCH_LIB:$LD_LIBRARY_PATH"
PATH="$PWD/.venv/bin:$PATH" cargo build --release --no-default-features --features cuda,tch-eval \
    --bin self_play --bin polyfish --bin benchmark --bin arena

# Persist the env for interactive shells and tmux panes.
if ! grep -q "LIBTORCH_USE_PYTORCH" ~/.bashrc; then
    {
        echo "export PATH=\"$HOME/.cargo/bin:$PWD/.venv/bin:\$PATH\""
        echo "export LIBTORCH_USE_PYTORCH=1"
        echo "export LIBTORCH_BYPASS_VERSION_CHECK=1"
        echo "export LD_LIBRARY_PATH=\"$TORCH_LIB:\$LD_LIBRARY_PATH\""
    } >> ~/.bashrc
fi

echo "=== Sanity checks ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
.venv/bin/python3 -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
nproc
echo "=== Setup complete. Next: get model.safetensors onto the box (scp or init_model.py), then benchmark in tmux. ==="
