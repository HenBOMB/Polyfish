#!/bin/bash
set -e

# Configuration
ITERATIONS=50
GAMES_PER_ITER=20
export MCTS_ITERS=25

echo "Building simulator..."
cargo build --bin polyfish --release

echo "Building self play..."
cargo build --release --bin self_play

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
    AVG_SCORE=$(echo "$SP_OUTPUT" | grep "METRICS:" | grep -o 'avg_score": [0-9.]*' | awk '{print $2}')
    MAX_SCORE=$(echo "$SP_OUTPUT" | grep "METRICS:" | grep -o 'max_score": [0-9]*' | awk '{print $2}')
    
    # 2. Training
    echo "[Training] Updating model..."
    TRAIN_OUTPUT=$(.venv/bin/python3 train.py)
    echo "$TRAIN_OUTPUT"
    
    # Extract Loss
    LOSS=$(echo "$TRAIN_OUTPUT" | grep "METRICS:" | grep -o 'loss": [0-9.]*' | awk '{print $2}')
    
    # 3. Log
    TIMESTAMP=$(date +%s)
    echo "$i,$TIMESTAMP,$AVG_SCORE,$MAX_SCORE,$LOSS" >> training_log.csv
    echo "Iteration $i complete. Avg: $AVG_SCORE | Max: $MAX_SCORE | Loss: $LOSS"
    
    # 4. Cleanup (Fresh Games Only)
    # Move played games to archive so train.py only sees new ones next time
    mv games_*.safetensors archive/
done
