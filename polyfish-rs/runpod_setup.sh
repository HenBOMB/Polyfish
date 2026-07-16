#!/bin/bash
set -e

# ============================================================================
# RunPod one-time setup for all-local GPU training.
#
# Idempotent: safe to re-run. Every expensive step is skipped if already done,
# so a pod restart on a PERSISTENT VOLUME re-runs this in seconds.
#
# ── The one thing that saves you real money ────────────────────────────────
# Put this repo on a RunPod *persistent network volume* (mounted at /workspace)
# and run everything from there. The Rust target/ dir, the Python .venv, and
# the cargo registry then survive pod stop/start — so you compile ONCE (the
# slow part) and every future pod boots straight into training. Storage is
# ~$0.10/GB/month while the pod is stopped; a rebuild on GPU time is not.
#
# ── Pod template ───────────────────────────────────────────────────────────
# Pick a template that ships the CUDA *toolkit* (nvcc), not just the runtime —
# candle's CUDA build compiles kernels. RunPod "PyTorch 2.x" / CUDA 12.x devel
# images work. Verify with:  nvcc --version   and   nvidia-smi
#
# Usage:
#   ./runpod_setup.sh              # full setup (fat-LTO build; slow once)
#   FAST_BUILD=1 ./runpod_setup.sh # thin-LTO build (compiles ~2-3x faster)
# ============================================================================

echo "== Polyfish RunPod setup =="

# 0. Sanity: GPU + CUDA toolkit present
if ! command -v nvidia-smi &>/dev/null; then
    echo "❌ nvidia-smi not found — this is not a GPU pod. Aborting." >&2
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
if ! command -v nvcc &>/dev/null && [ ! -x /usr/local/cuda/bin/nvcc ]; then
    echo "⚠️  nvcc (CUDA toolkit) not found on PATH. candle's --features cuda build"
    echo "    needs it. Use a CUDA *devel* pod template, or: apt-get install -y cuda-toolkit"
fi

# 1. System deps
export DEBIAN_FRONTEND=noninteractive
apt-get update && apt-get install -y build-essential curl pkg-config libssl-dev python3-venv jq tmux

# 2. Rust
if ! command -v cargo &>/dev/null && [ ! -x "$HOME/.cargo/bin/cargo" ]; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
if ! grep -q "cargo/bin" ~/.bashrc 2>/dev/null; then
    echo 'export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"' >> ~/.bashrc
fi
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"

# 3. Python venv + PyTorch (CUDA) + deps
if [ ! -d ".venv" ]; then
    echo "Creating .venv..."
    python3 -m venv .venv
fi
.venv/bin/pip install --upgrade pip
if ! .venv/bin/python3 -c "import torch" &>/dev/null; then
    echo "Installing PyTorch (CUDA 12.4 nightly)..."
    .venv/bin/pip install --pre --upgrade torch --index-url https://download.pytorch.org/whl/nightly/cu124
else
    echo "PyTorch already installed — skipping."
fi
.venv/bin/pip install -r requirements.txt

# Confirm torch actually sees the GPU (the whole point).
.venv/bin/python3 - <<'PY'
import torch
print(f"torch {torch.__version__} | CUDA build {torch.version.cuda} | cuda.is_available()={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: torch does NOT see a GPU — training would fall back to CPU.")
PY

# 4. Build CUDA binaries (release). This is the slow, compile-once step.
if [ -x ./target/release/self_play ] && [ -x ./target/release/polyfish ] && [ -x ./target/release/arena ]; then
    echo "Binaries already built — skipping cargo build (delete target/release to force)."
else
    echo "Building CUDA binaries (release)... this is the one-time slow part."
    BUILD_ENV=()
    if [ "${FAST_BUILD:-0}" = "1" ]; then
        echo "⚡ FAST_BUILD: thin LTO + 16 codegen-units"
        BUILD_ENV=(CARGO_PROFILE_RELEASE_LTO=thin CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16)
    fi
    env "${BUILD_ENV[@]}" cargo build --bin polyfish --bin self_play --bin arena --release \
        --no-default-features --features cuda
fi

echo ""
echo "✅ Setup complete. Start training with:"
echo "     ./run_training_runpod.sh --reset          # fresh model from scratch"
echo "     ./run_training_runpod.sh --resume         # continue latest run"
echo "  (Inside tmux so it survives disconnects:  tmux new -s train)"
