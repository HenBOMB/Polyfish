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

# Detect CUDA version or default to cu121
CUDA_INDEX="cu121"
if command -v nvcc &> /dev/null; then
    NVCC_VER=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
    if [[ "$NVCC_VER" =~ ^12\. ]]; then
        CUDA_INDEX="cu121"
    elif [[ "$NVCC_VER" =~ ^11\. ]]; then
        CUDA_INDEX="cu118"
    fi
fi

echo "Installing PyTorch ($CUDA_INDEX)..."
.venv/bin/pip install torch==2.1.2 --index-url "https://download.pytorch.org/whl/$CUDA_INDEX" || \
.venv/bin/pip install torch --index-url "https://download.pytorch.org/whl/$CUDA_INDEX"

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

echo "=== [6/6] Building CUDA-Accelerated Binaries ==="
cargo build --release --no-default-features --features cuda \
    --bin self_play --bin arena --bin polyfish

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
