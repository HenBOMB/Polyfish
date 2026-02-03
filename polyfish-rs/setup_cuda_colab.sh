#!/bin/bash
# Colab CUDA Setup Script
# Run this in Colab BEFORE building to force CUDA kernel compilation

set -e

echo "=== Configuring CUDA Build for T4 GPU ==="
echo ""

# 1. Check CUDA version
echo "1. CUDA Version:"
nvcc --version | grep "release"
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
echo ""

# 2. Set environment variables to force kernel compilation
echo "2. Setting build environment..."
export CUDA_COMPUTE_CAP=75  # T4 GPU compute capability
export RUSTFLAGS="-C target-cpu=native -C opt-level=3"
export CUDA_LAUNCH_BLOCKING=1

# Force rebuild of CUDA kernels
export CUDARC_FORCE_BUILD=1

# Set CUDA paths (usually automatic on Colab, but just in case)
export CUDA_PATH=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

echo "✅ Environment configured for CUDA 12.4 + T4"
echo ""

# 3. Clean previous builds
echo "3. Cleaning previous builds..."
cargo clean
echo "✅ Clean complete"
echo ""

# 4. Build with CUDA
echo "4. Building with CUDA support..."
echo "   This will take ~5-10 minutes on first build..."
cargo build --release --features cuda --bin self_play

echo ""
echo "=== Build Complete ==="
echo ""

# 5. Test
echo "Testing GPU..."
cargo run --release --features cuda --bin benchmark

echo ""
echo "✅ Setup complete!"
