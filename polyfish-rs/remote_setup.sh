#!/bin/bash
set -e

# 1. Update System & Install Deps
apt-get update && apt-get install -y build-essential curl pkg-config libssl-dev python3-venv

# 2. Install Rust (if not present)
if ! command -v cargo &> /dev/null; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
else
    echo "Rust is already installed."
fi

# Add paths to .bashrc for persistence across sessions
if ! grep -q "export PATH=.*cargo/bin" ~/.bashrc; then
    echo 'export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"' >> ~/.bashrc
    echo "Added Rust and CUDA to PATH in .bashrc"
fi

# Source for current session
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"

# 3. Setup Python Virtual Environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

echo "Installing Python dependencies..."
.venv/bin/pip install --upgrade pip
# Install specific pytorch with CUDA support (Nightly for RTX 5090 / CUDA 12.x)
# Split installation to avoid dependency resolution errors with nightly builds
.venv/bin/pip install --pre --upgrade --no-cache-dir torch --index-url https://download.pytorch.org/whl/nightly/cu124

.venv/bin/pip install -r requirements.txt

# 4. Build Project
echo "Building PolyFish (Release)..."
$HOME/.cargo/bin/cargo build --release --bin self_play --features cuda
$HOME/.cargo/bin/cargo build --release --bin polyfish --features cuda
$HOME/.cargo/bin/cargo build --release --bin benchmark --features cuda

echo "Setup Complete! You can now run the training loop."

sudo apt-get update && sudo apt-get install -y tmux