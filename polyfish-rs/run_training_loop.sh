#!/bin/bash
set -e

# Configuration
ITERATIONS=100
GAMES_PER_ITER=7
export MCTS_ITERS=100 # 200 = Optimized for RunPod GPU (~0.8s per move)
export RAYON_NUM_THREADS=12
export OMP_NUM_THREADS=12
export RUST_BACKTRACE=1 # Enable full panic backtraces

# Log all output to session.log while still showing on console
LOG_FILE="session.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"

# Background System Monitor (Logs RAM/GPU every 10s)
start_system_monitor() {
   echo "Starting system monitor logging to system_stats.log..."
   while true; do
       TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
       echo "--- $TIMESTAMP ---" >> system_stats.log
       echo "[RAM]" >> system_stats.log
       free -h >> system_stats.log
       echo "[GPU]" >> system_stats.log
       # Check if nvidia-smi exists (for local testing vs runpod)
       if command -v nvidia-smi &> /dev/null; then
           nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv,noheader >> system_stats.log
       else
           echo "No GPU detected" >> system_stats.log
       fi
       sleep 10
   done &
   MONITOR_PID=$!
   trap "kill $MONITOR_PID" EXIT
}
start_system_monitor

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
    
    # 1. League Training Logic (20% chance)
    # Check if we have checkpoints to play against
    OPPONENT_FLAG=""
    MATCH_TYPE="Self-Play"
    
    # Simple random check (1-100 <= 20)
    RAND_VAL=$((1 + RANDOM % 100))
    
    if [ "$RAND_VAL" -le 20 ] && [ -d "checkpoints" ] && [ "$(ls -A checkpoints)" ]; then
        # Pick a random checkpoint
        RANDOM_CHECKPOINT=$(ls checkpoints/*.safetensors | shuf -n 1)
        if [ -n "$RANDOM_CHECKPOINT" ]; then
             OPPONENT_FLAG="--opponent $RANDOM_CHECKPOINT"
             MATCH_TYPE="League Match vs $(basename $RANDOM_CHECKPOINT)"
        fi
    fi

    echo "[$MATCH_TYPE] Generative games..."
    
    # Capture output to extract metrics
    # We pass args via CLI now, not env vars alone
    SP_OUTPUT=$(./target/release/self_play --num-games $GAMES_PER_ITER --mcts-iters $MCTS_ITERS $OPPONENT_FLAG)
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
    # Add match type column if needed, or just log to console
    echo "$i,$TIMESTAMP,$AVG_SCORE,$MAX_SCORE,$P1_AVG,$P2_AVG,$LOSS" >> training_log.csv
    echo "Iteration $i complete. Type: $MATCH_TYPE | Avg: $AVG_SCORE | Max: $MAX_SCORE | P1: $P1_AVG | P2: $P2_AVG | Loss: $LOSS"
    
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
