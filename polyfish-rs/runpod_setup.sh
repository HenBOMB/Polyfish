#!/usr/bin/env bash
# ==============================================================================
# RunPod One-Click Environment Setup Script
# ==============================================================================
# Run this inside your RunPod pod terminal:
#   cd /workspace/PolyAI/polyfish-rs && ./runpod_setup.sh
# ==============================================================================
set -eo pipefail

cd "$(dirname "$0")"

echo "=== [1/6] Checking GPU Status ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
else
    echo "WARNING: nvidia-smi not detected. Ensure pod was launched with a GPU attached."
fi

echo "=== [2/6] Installing Essential System Packages ==="
if command -v apt-get &> /dev/null; then
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl pkg-config libssl-dev git tmux htop
fi

echo "=== [3/6] Setting Up Rust Toolchain ==="
if ! command -v cargo &> /dev/null; then
    echo "Installing Rust toolchain..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
    source "$HOME/.cargo/env"
else
    echo "Rust is already installed: $(cargo --version)"
fi
export PATH="$HOME/.cargo/bin:$PATH"

echo "=== [4/6] Setting Up Python Virtual Environment ==="
if [ ! -d ".venv" ]; then
    echo "Creating .venv..."
    python3 -m venv .venv
fi
.venv/bin/pip install --upgrade pip

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/cuda_arch_helper.sh" ]; then
    source "$SCRIPT_DIR/cuda_arch_helper.sh"
fi

# Detect host GPU or resolve user-defined CUDA_COMPUTE_CAP (e.g. "5090", "3090 ti", "120")
DETECTED_TARGET=""
if [ -n "${CUDA_COMPUTE_CAP:-}" ]; then
    DETECTED_TARGET="$CUDA_COMPUTE_CAP"
elif command -v nvidia-smi &> /dev/null; then
    DETECTED_TARGET=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1)
    if [ -z "$DETECTED_TARGET" ]; then
        DETECTED_TARGET=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d '.' | head -n 1)
    fi
fi

RESOLVED_CAP="86"
if command -v resolve_cuda_cap &>/dev/null && [ -n "$DETECTED_TARGET" ]; then
    RESOLVED_CAP=$(resolve_cuda_cap "$DETECTED_TARGET" 2>/dev/null || echo "86")
elif [ -n "$DETECTED_TARGET" ]; then
    RESOLVED_CAP="$DETECTED_TARGET"
fi
TARGET_CAP="${RESOLVED_CAP:-86}"
export CUDA_COMPUTE_CAP="$TARGET_CAP"
echo "Targeting CUDA Compute Capability: sm_${TARGET_CAP} (Input/GPU: '${DETECTED_TARGET:-default}')"

# PyTorch installation: cu128 covers Blackwell (sm_120) down to Ampere (sm_86) seamlessly.
TORCH_CUDA_INDEXES="${TORCH_CUDA_INDEXES:-cu128 cu126 cu124 cu121}"
TORCH_VERSION="$(sed -n 's/^# *POLYFISH_TORCH_VERSION=//p' requirements.txt | head -n 1)"
INSTALLED_TORCH="$(.venv/bin/python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || true)"

if [ -z "$INSTALLED_TORCH" ]; then
    echo "Installing PyTorch with CUDA support..."
    OK=0
    for IDX in $TORCH_CUDA_INDEXES; do
        echo "Trying CUDA wheel index $IDX..."
        if [ -n "$TORCH_VERSION" ]; then
            if .venv/bin/pip install "torch==$TORCH_VERSION" --index-url "https://download.pytorch.org/whl/$IDX"; then
                OK=1; break
            fi
        else
            if .venv/bin/pip install torch --index-url "https://download.pytorch.org/whl/$IDX"; then
                OK=1; break
            fi
        fi
    done
    if [ "$OK" -ne 1 ]; then
        echo "Failed to install CUDA PyTorch from $TORCH_CUDA_INDEXES, falling back to default PyPI wheel..."
        .venv/bin/pip install torch
    fi
else
    echo "PyTorch $INSTALLED_TORCH is already installed."
fi

.venv/bin/pip install -r requirements.txt

echo "=== [5/6] Restoring DAgger Foundation Model ==="
if [ -f "model_dagger_foundation.safetensors" ] && [ ! -f "model.safetensors" ]; then
    cp model_dagger_foundation.safetensors model.safetensors
    echo "Restored model.safetensors from model_dagger_foundation.safetensors"
elif [ -f "model.safetensors" ]; then
    echo "model.safetensors is already present."
else
    echo "Initializing fresh weights..."
    .venv/bin/python3 init_model.py
fi

echo "=== [6/6] Building CUDA-Accelerated Binaries (Native CPU Vectorization) ==="
ARCH_BIN_DIR="target/bin_sm${TARGET_CAP}"
mkdir -p "$ARCH_BIN_DIR" target/release

RUSTFLAGS="-C target-cpu=native ${RUSTFLAGS:-}" cargo build --release --no-default-features --features cuda \
    --bin self_play --bin arena --bin polyfish
echo "$TARGET_CAP" > ./target/release/.cuda_cap
cp ./target/release/self_play ./target/release/polyfish ./target/release/arena "$ARCH_BIN_DIR/" 2>/dev/null || true
echo "Cached sm_${TARGET_CAP} binaries to $ARCH_BIN_DIR"

chmod +x *.sh 2>/dev/null || true

echo ""
echo "=============================================================================="
echo " Setup complete! Ready for GPU-accelerated training."
echo "=============================================================================="
echo " To launch adversarial self-play reinforcement learning, run:"
echo "   ./start_adversarial_training.sh"
echo " Or inside a persistent tmux session:"
echo "   tmux new -s train './start_adversarial_training.sh'"
echo "=============================================================================="
