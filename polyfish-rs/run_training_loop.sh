#!/bin/bash
set -e

# Configuration
# NUM_GAMES/MCTS_ITERS/ACTORS/EVAL_SERVERS are named to match self_play's own
# --num-games/--mcts-iters/--actors/--eval-servers flags 1:1 — see
# `self_play --help` (or src/bin/self_play.rs) for what each actually does.
ITERATIONS=1000
NUM_GAMES=10
export MCTS_ITERS=64
# self_play's actor pool is plain std::thread (not rayon), and actors block
# (park, no CPU) while awaiting eval-server replies — oversubscribing past
# core count is fine, RAM is the real ceiling. 32 measured ~2.5x the
# throughput of the --actors 0 (auto = core count) default on an M3 Max.
ACTORS=32
# 0 = defer to self_play's own auto (2 shards on tch backend, 1 on candle).
EVAL_SERVERS=1
export RUST_BACKTRACE=1

# Log all output to session.log while still showing on console
LOG_FILE="session.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"

echo "Building binaries..."
# Detect platform and use appropriate GPU features
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS: Metal/Accelerate + tch-eval so self_play inference runs on libtorch/MPS
    echo "Building with Metal + Accelerate + tch-eval (libtorch/MPS) for macOS..."
    export LIBTORCH_USE_PYTORCH=1
    export LIBTORCH_BYPASS_VERSION_CHECK=1
    PATH="$(pwd)/.venv/bin:$PATH" cargo build --bin polyfish --bin self_play --release --features metal,accelerate,tch-eval
    # The tch-linked binary has no rpath for libtorch; point dyld at the venv's torch dylibs
    export DYLD_LIBRARY_PATH="$(.venv/bin/python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
elif command -v nvidia-smi &> /dev/null; then
    # CUDA available (Linux/Windows with NVIDIA GPU)
    echo "Building with CUDA support..."
    # --no-default-features: opt out of the macOS `metal` default, which does
    # not compile on Linux.
    cargo build --bin polyfish --bin self_play --release --no-default-features --features cuda
else
    # CPU-only fallback
    echo "Building CPU-only version..."
    cargo build --bin polyfish --bin self_play --release --no-default-features
fi

# Parse arguments
FORCE_TRAIN=false
BOOST=false
CHILL=false
REWARD_SHAPING=false
while getopts "fbcri:g:n:a:e:" opt; do
  case $opt in
    f)
      FORCE_TRAIN=true
      ;;
    b)
      BOOST=true
      ;;
    c)
      CHILL=true
      ;;
    r)
      REWARD_SHAPING=true
      ;;
    i)
      ITERATIONS=$OPTARG
      ;;
    g)
      NUM_GAMES=$OPTARG
      ;;
    n)
      MCTS_ITERS=$OPTARG
      ;;
    a)
      ACTORS=$OPTARG
      ;;
    e)
      EVAL_SERVERS=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

REWARD_FLAG=""
if [ "$REWARD_SHAPING" = true ]; then
    REWARD_FLAG="--reward-shaping"
    echo "🎯 Reward shaping enabled!"
fi

if [ "$BOOST" = true ]; then
    ACTORS=$((ACTORS * 2))
    echo "🚀 Boost mode enabled! Using $ACTORS actors"
fi

if [ "$CHILL" = true ]; then
    ACTORS=8
    echo "❄️ Chill mode! Using $ACTORS actors"
fi

if [ "$FORCE_TRAIN" = true ]; then
    echo "Force training flag detected! Running training immediately..."
    echo "[Training] Training model..."
    .venv/bin/python3 train.py
fi

# Determine starting iteration from log
START_ITER=1
if [ -f "training_log.csv" ]; then
    LAST_ITER=$(tail -n 1 training_log.csv | cut -d',' -f1)
    if [[ "$LAST_ITER" =~ ^[0-9]+$ ]]; then
        START_ITER=$((LAST_ITER + 1))
        echo "Resuming from iteration $START_ITER"
    fi
fi

# Portable replacement for GNU `shuf` (not present on stock macOS)
portable_shuf() {
    local n=$1
    local lines=()
    while IFS= read -r line; do
        [ -n "$line" ] && lines+=("$line")
    done
    local count=${#lines[@]}
    for ((idx = count - 1; idx > 0; idx--)); do
        local j=$((RANDOM % (idx + 1)))
        local tmp="${lines[idx]}"
        lines[idx]="${lines[j]}"
        lines[j]="$tmp"
    done
    for ((idx = 0; idx < n && idx < count; idx++)); do
        echo "${lines[idx]}"
    done
}

# 0. Initialize & Auto-Restore Model
echo "Initializing/Checking model..."
# If resuming but model.safetensors is missing, restore latest checkpoint
if [ "$START_ITER" -gt 1 ] && [ ! -f "model.safetensors" ]; then
    LATEST_CP=$(ls checkpoints/model_checkpoint_iter*.safetensors 2>/dev/null | sort -V | tail -n 1 || true)
    if [ -n "$LATEST_CP" ]; then
        echo "🔄 Resuming: Restoring latest checkpoint $(basename $LATEST_CP) to model.safetensors"
        cp "$LATEST_CP" model.safetensors
    fi
fi
.venv/bin/python3 init_model.py

for ((i=START_ITER; i<START_ITER+ITERATIONS; i++))
do
    echo "=================================================="
    echo "Starting Iteration $i"
    echo "=================================================="
    
    # 1. League Training Logic (20% chance)
    # Check if we have checkpoints to play against
    mkdir -p checkpoints
    OPPONENT_FLAG=""
    MATCH_TYPE="Self-Play"
    
    # Simple random check (1-100 <= 20)
    RAND_VAL=$((1 + RANDOM % 100))
    
    if [ "$RAND_VAL" -le 20 ] && [ -d "checkpoints" ] && [ "$(ls -A checkpoints)" ]; then
        # SMART LEAGUE SELECTION: 50% chance 'Fresh' (latest), 50% chance 'Historical' (diverse)
        ALL_CPS=$(ls -t checkpoints/model_checkpoint_iter*.safetensors 2>/dev/null || true)
        FRESH_CPS=$(echo "$ALL_CPS" | head -n 5)
        HIST_CPS=$(echo "$ALL_CPS" | tail -n +6)
        
        if [ -n "$HIST_CPS" ] && [ $((RANDOM % 2)) -eq 0 ]; then
             SELECTED_CP=$(echo "$HIST_CPS" | portable_shuf 1)
        else
             SELECTED_CP=$(echo "$FRESH_CPS" | portable_shuf 1)
        fi

        if [ -n "$SELECTED_CP" ]; then
             OPPONENT_FLAG="--opponent $SELECTED_CP"
             MATCH_TYPE="League Match vs $(basename $SELECTED_CP)"
        fi
    fi

    # Pick 2 random tribes for this iteration
    TRIBE_LIST=("Imperius" "Imperius")
    # TRIBE_LIST=("Imperius" "Bardur" "Oumaji" "Kickoo" "XinXi" "Zebasi" "AiMo" "Vengir" "Quetzali" "Hoodrick" "Yadakk")
    # Shuffle and pick top 2 (portable, no external shuf dependency)
    SELECTED_TRIBES=($(printf "%s\n" "${TRIBE_LIST[@]}" | portable_shuf 2))
    TRIBE1=${SELECTED_TRIBES[0]}
    TRIBE2=${SELECTED_TRIBES[1]}
    
    echo "[$MATCH_TYPE] Generative games... Tribes: $TRIBE1 vs $TRIBE2"
    
    # Capture output to extract metrics
    # We pass args via CLI now, not env vars alone
    SP_OUTPUT=$(./target/release/self_play --num-games $NUM_GAMES --mcts-iters $MCTS_ITERS --actors $ACTORS --eval-servers $EVAL_SERVERS $REWARD_FLAG $OPPONENT_FLAG --tribe1 "$TRIBE1" --tribe2 "$TRIBE2" --iteration "$i")
    echo "$SP_OUTPUT"
    
    # Extract Avg Score and Max Score using grep and sed or awk
    AVG_SCORE=$(echo "$SP_OUTPUT" | grep "METRICS:" | grep -o '"avg_score": [0-9.]*' | awk -F': ' '{print $2}')
    MAX_SCORE=$(echo "$SP_OUTPUT" | grep "METRICS:" | grep -o '"max_score": [0-9]*' | awk -F': ' '{print $2}')
    P1_AVG=$(echo "$SP_OUTPUT" | grep "METRICS:" | grep -o '"p1_avg": [0-9.]*' | awk -F': ' '{print $2}')
    P2_AVG=$(echo "$SP_OUTPUT" | grep "METRICS:" | grep -o '"p2_avg": [0-9.]*' | awk -F': ' '{print $2}')
    
    AVG_CAPTURES=$(echo "$SP_OUTPUT" | grep "METRICS:" | grep -o '"avg_captures": [0-9.]*' | awk -F': ' '{print $2}')
    AVG_HARVESTS=$(echo "$SP_OUTPUT" | grep "METRICS:" | grep -o '"avg_harvests": [0-9.]*' | awk -F': ' '{print $2}')
    AVG_BUILDS=$(echo "$SP_OUTPUT" | grep "METRICS:" | grep -o '"avg_builds": [0-9.]*' | awk -F': ' '{print $2}')
    AVG_RESEARCH=$(echo "$SP_OUTPUT" | grep "METRICS:" | grep -o '"avg_research": [0-9.]*' | awk -F': ' '{print $2}')
    AVG_ATTACKS=$(echo "$SP_OUTPUT" | grep "METRICS:" | grep -o '"avg_attacks": [0-9.]*' | awk -F': ' '{print $2}')
    
    # 2. Training
    echo "[Training] Training model..."
    TRAIN_OUTPUT=$(.venv/bin/python3 train.py)
    echo "$TRAIN_OUTPUT"
    LOSS=$(echo "$TRAIN_OUTPUT" | grep "METRICS:" | grep -o '"loss": [0-9.]*' | awk -F': ' '{print $2}')
    
    # 3. Log
    TIMESTAMP=$(date +%s)
    echo "$i,$TIMESTAMP,$AVG_SCORE,$MAX_SCORE,$P1_AVG,$P2_AVG,$LOSS,$AVG_CAPTURES,$AVG_HARVESTS,$AVG_BUILDS,$AVG_RESEARCH,$AVG_ATTACKS" >> training_log.csv
    echo "Iteration $i complete. Type: $MATCH_TYPE | Avg: $AVG_SCORE | Loss: $LOSS"
    echo "  -> STATS/GAME: Captures: $AVG_CAPTURES | Harvests: $AVG_HARVESTS | Builds: $AVG_BUILDS | Tech: $AVG_RESEARCH | Attacks: $AVG_ATTACKS"
    
    # 4. Checkpoint (Every 50 iterations)
    if [ $((i % 50)) -eq 0 ]; then
        TS=$(date +%Y%m%d_%H%M%S)
        echo "Creating checkpoint for iteration $i (Timestamp: $TS)..."
        cp model.safetensors "checkpoints/model_checkpoint_iter${i}_${TS}.safetensors"
    fi
    
    # Smart Pruning: Keep recent density and historical milestones
    # This keeps:
    # - Last 50 checkpoints (for fine-tuned self-play)
    # - Every 100th checkpoint forever (for long-term diversity)
    # - Iteration 1 (baseline)
    ALL_FILES=$(ls -t checkpoints/model_checkpoint_iter*.safetensors 2>/dev/null || true)
    if [ -n "$ALL_FILES" ]; then
        idx=0
        echo "$ALL_FILES" | while read -r FILE; do
            idx=$((idx + 1))
            # Extract iteration number from filename
            ITER_VAL=$(echo "$FILE" | sed -n 's/.*iter\([0-9]\+\)_.*/\1/p')
            
            KEEP=false
            if [ $idx -le 50 ]; then
                # Keep the last 50 most recent
                KEEP=true
            elif [ -n "$ITER_VAL" ]; then
                # Keep historical milestones
                if [ $((ITER_VAL % 100)) -eq 0 ] || [ "$ITER_VAL" -eq 1 ]; then
                    KEEP=true
                fi
            fi
            
            if [ "$KEEP" = false ]; then
                rm "$FILE"
            fi
        done
    fi

    # 4. Cleanup (Fresh Games Only)
    # Move played games to archive so train.py only sees new ones next time
    mkdir -p archive
    # Use || true to avoid script exit if no games were generated
    mv games_*.safetensors archive/ 2>/dev/null || true
    
    # Keep only the last 10 game files to save space and match train.py replay buffer
    ls -t archive/games_*.safetensors 2>/dev/null | tail -n +31 | xargs -r rm
    
done
