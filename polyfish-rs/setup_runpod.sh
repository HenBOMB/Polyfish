#!/bin/bash
set -e

# Polyfish RunPod Setup Script
# Usage: ./setup_runpod.sh

echo "=== Polyfish RunPod Setup ==="

# 1. Install Dependencies
echo "1. Installing system dependencies..."
apt-get update
apt-get install -y build-essential libssl-dev pkg-config git curl

# 2. Install Rust
echo "2. Installing Rust..."
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# 3. Configure CUDA Environment
# RunPod usually has CUDA at /usr/local/cuda
echo "3. Configuring CUDA..."
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

# Add to .bashrc for persistence
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc

# 4. Verify GPU
echo "4. Verifying GPU..."
nvidia-smi
nvcc --version

echo "=== Setup Complete! ==="
echo "You can now build the project with:"
echo "cargo build --release --features cuda --bin train"
