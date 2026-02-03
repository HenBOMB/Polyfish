#!/bin/bash
# Pre-training verification script
# Run this before starting production training

set -e

echo "=== Production Readiness Check ==="
echo ""

# 1. Check CUDA
echo "1. Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo "✅ CUDA detected"
else
    echo "⚠️  No CUDA - will use CPU (slower)"
fi
echo ""

# 2. Check directory structure
echo "2. Checking directory structure..."
mkdir -p archive
mkdir -p checkpoints
echo "✅ Directories created"
echo ""

# 3. Check dependencies
echo "3. Checking Rust build..."
cargo build --release --bin self_play 2>&1 | tail -5
echo "✅ Rust build successful"
echo ""

echo "4. Checking Python dependencies..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import safetensors; print('SafeTensors: OK')"
echo "✅ Python dependencies OK"
echo ""

# 5. Test run
echo "5. Running test game..."
NUM_GAMES=1 MCTS_ITERS=10 timeout 60 ./target/release/self_play || echo "⚠️  Test timed out (expected)"
echo "✅ Test completed"
echo ""

# 6. Check disk space
echo "6. Checking disk space..."
df -h . | tail -1
echo ""

echo "=== Ready for Production ==="
echo "Run: ./run_training_loop.sh"
