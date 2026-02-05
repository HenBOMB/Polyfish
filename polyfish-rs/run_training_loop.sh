#!/bin/bash
set -e

# Configuration
ITERATIONS=100
GAMES_PER_ITER=25
export MCTS_ITERS=200 # Optimized for RunPod GPU (~0.8s per move)
export RAYON_NUM_THREADS=8
export OMP_NUM_THREADS=8

echo "Building simulator..."
cargo build --bin polyfish --release --features cuda

echo "Building self play..."
cargo build --release --bin self_play --features cuda

# Parse arguments
FORCE_TRAIN=false
while getopts "f" opt; do
  case $opt in
    f)
      FORCE_TRAIN=true
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

if [ "$FORCE_TRAIN" = true ]; then
    echo "Force training flag detected! Running training immediately..."
    echo "[Training] Training model..."
    TRAIN_OUTPUT=$(.venv/bin/python3 train.py)
    echo "$TRAIN_OUTPUT"
    
    # Extract Loss (optional logging, but good to see)
    LOSS=$(echo "$TRAIN_OUTPUT" | grep "METRICS:" | grep -o 'loss": [0-9.]*' | awk '{print $2}')
    echo "Immediate training complete. Loss: $LOSS"
fi

# 0. Initialize Model (if needed)
echo "Initializing/Checking model..."
.venv/bin/python3 init_model.py

for ((i=1; i<=ITERATIONS; i++))
do
    echo "=================================================="
    echo "Starting Iteration $i / $ITERATIONS"
    echo "=================================================="
    
    # 1. Self Play
    echo "[Self-Play] Generating games..."
    # Capture output to extract metrics
    SP_OUTPUT=$(NUM_GAMES=$GAMES_PER_ITER ./target/release/self_play)
    echo "$SP_OUTPUT"
    
    # Extract Avg Score and Max Score using grep and sed or awk
    AVG_SCORE=$(echo "$SP_OUTPUT" | grep "METRICS:" | grep -o '"avg_score": [0-9.]*' | awk -F': ' '{print $2}')
    MAX_SCORE=$(echo "$SP_OUTPUT" | grep "METRICS:" | grep -o '"max_score": [0-9]*' | awk -F': ' '{print $2}')
    P1_AVG=$(echo "$SP_OUTPUT" | grep "METRICS:" | grep -o '"p1_avg": [0-9.]*' | awk -F': ' '{print $2}')
    P2_AVG=$(echo "$SP_OUTPUT" | grep "METRICS:" | grep -o '"p2_avg": [0-9.]*' | awk -F': ' '{print $2}')
    
    # 2. Training
    echo "[Training] Training model..."
    TRAIN_OUTPUT=$(.venv/bin/python3 train.py)
    echo "$TRAIN_OUTPUT"
    
    # Extract Loss
    LOSS=$(echo "$TRAIN_OUTPUT" | grep "METRICS:" | grep -o 'loss": [0-9.]*' | awk '{print $2}')
    
    # 3. Log
    TIMESTAMP=$(date +%s)
    echo "$i,$TIMESTAMP,$AVG_SCORE,$MAX_SCORE,$P1_AVG,$P2_AVG,$LOSS" >> training_log.csv
    echo "Iteration $i complete. Avg: $AVG_SCORE | Max: $MAX_SCORE | P1: $P1_AVG | P2: $P2_AVG | Loss: $LOSS"
    
    # 4. Checkpoint (Every 5 iterations)
    if (( i % 5 == 0 )); then
        echo "Creating checkpoint for iteration $i..."
        mkdir -p checkpoints
        cp model.safetensors checkpoints/model_checkpoint_$i.safetensors
    fi
    
    # 4. Cleanup (Fresh Games Only)
    # Move played games to archive so train.py only sees new ones next time
    mkdir -p archive
    mv games_*.safetensors archive/
    
done
