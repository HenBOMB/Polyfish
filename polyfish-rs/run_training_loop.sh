#!/bin/bash
set -e

# Write PID file for server detection, clean up on exit
echo $$ > .training.pid
trap 'rm -f .training.pid' EXIT

# Configuration
ITERATIONS=1000
GAMES_PER_ITER=10
GAMEMODE=2
USE_THREADS=12
export MCTS_ITERS=200
export RUST_BACKTRACE=1

# Log all output to session.log while still showing on console
LOG_FILE="session.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"

echo "Building binaries..."
cargo build --bin polyfish --bin self_play --release

# Note: The server start logic has been moved after argument parsing so we know if -n was passed.

# Parse arguments
FORCE_TRAIN=false
BOOST=false
CHILL=false
REWARD_SHAPING=false
START_SERVER=true
while getopts "fbcrn" opt; do
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
    n)
      START_SERVER=false
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

if [ "$START_SERVER" = true ]; then
    echo "Starting backend server in background..."
    ./target/release/polyfish &
    SERVER_PID=$!
    trap "echo 'Shutting down server...'; kill $SERVER_PID 2>/dev/null; rm -f .training.pid" EXIT
else
    echo "Skipping backend server startup (-n flag provided)..."
fi

if [ "$BOOST" = true ]; then
    USE_THREADS=$((USE_THREADS * 2))
    echo "🚀 Boost mode enabled! Using $USE_THREADS threads"
fi

if [ "$CHILL" = true ]; then
    USE_THREADS=4
    echo "❄️ Chill mode! Using 4 threads"
fi

export RAYON_NUM_THREADS=$USE_THREADS
export OMP_NUM_THREADS=$USE_THREADS

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

for ((i=START_ITER; i<=ITERATIONS+START_ITER; i++))
do
    if [ -f "config.json" ]; then
        DYNAMIC_CORES=$(jq -r '.cores // empty' config.json)
        if [[ -n "$DYNAMIC_CORES" && "$DYNAMIC_CORES" =~ ^[0-9]+$ ]]; then
            export RAYON_NUM_THREADS=$DYNAMIC_CORES
            export OMP_NUM_THREADS=$DYNAMIC_CORES
        fi
        
        DYNAMIC_ITERS=$(jq -r '.iterations // empty' config.json)
        if [[ -n "$DYNAMIC_ITERS" && "$DYNAMIC_ITERS" =~ ^[0-9]+$ ]]; then
            ITERATIONS=$DYNAMIC_ITERS
        fi
        
        DYNAMIC_MCTS=$(jq -r '.mctsIters // empty' config.json)
        if [[ -n "$DYNAMIC_MCTS" && "$DYNAMIC_MCTS" =~ ^[0-9]+$ ]]; then
            export MCTS_ITERS=$DYNAMIC_MCTS
        fi
        
        DYNAMIC_GAMEMODE=$(jq -r '.gamemode // empty' config.json)
        if [[ -n "$DYNAMIC_GAMEMODE" && "$DYNAMIC_GAMEMODE" =~ ^[0-9]+$ ]]; then
            GAMEMODE=$DYNAMIC_GAMEMODE
        fi
    fi

    echo "=================================================="
    echo "Starting Iteration $i (Threads: $RAYON_NUM_THREADS)"
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
             SELECTED_CP=$(echo "$HIST_CPS" | shuf -n 1)
        else
             SELECTED_CP=$(echo "$FRESH_CPS" | shuf -n 1)
        fi

        if [ -n "$SELECTED_CP" ]; then
             OPPONENT_FLAG="--opponent $SELECTED_CP"
             MATCH_TYPE="League Match vs $(basename $SELECTED_CP)"
        fi
    fi

    # Read active tribes from config if it exists, otherwise use default
    if [ -f "config.json" ] && [ "$(jq -r '.tribes' config.json)" != "null" ]; then
        mapfile -t TRIBE_LIST < <(jq -r '.tribes[]' config.json)
    fi
    if [ ${#TRIBE_LIST[@]} -eq 0 ]; then
        TRIBE_LIST=("Imperius" "Imperius")
    fi
    
    # Shuffle and pick top 2 (using shuf)
    SELECTED_TRIBES=($(printf "%s\n" "${TRIBE_LIST[@]}" | shuf -n 2))
    TRIBE1=${SELECTED_TRIBES[0]}
    TRIBE2=${SELECTED_TRIBES[1]}
    
    echo "[$MATCH_TYPE] Generative games... Tribes: $TRIBE1 vs $TRIBE2"
    
    # Capture output to extract metrics
    # We pass args via CLI now, not env vars alone
    SP_OUTPUT=$(./target/release/self_play --num-games $GAMES_PER_ITER --mcts-iters $MCTS_ITERS $REWARD_FLAG $OPPONENT_FLAG --tribe1 "$TRIBE1" --tribe2 "$TRIBE2" --iteration "$i" --gamemode "$GAMEMODE")
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
    AVG_ABILITY=$(echo "$SP_OUTPUT" | grep "METRICS:" | grep -o '"avg_ability": [0-9.]*' | awk -F': ' '{print $2}')
    AVG_STEPS=$(echo "$SP_OUTPUT" | grep "METRICS:" | grep -o '"avg_steps": [0-9.]*' | awk -F': ' '{print $2}')
    
    # 2. Training
    echo "[Training] Training model..."
    TRAIN_OUTPUT=$(.venv/bin/python3 train.py)
    echo "$TRAIN_OUTPUT"
    LOSS=$(echo "$TRAIN_OUTPUT" | grep "METRICS:" | grep -o '"loss": [0-9.]*' | awk -F': ' '{print $2}')
    
    # 3. Log
    TIMESTAMP=$(date +%s)
    echo "$i,$TIMESTAMP,$AVG_SCORE,$MAX_SCORE,$P1_AVG,$P2_AVG,$LOSS,$AVG_CAPTURES,$AVG_HARVESTS,$AVG_BUILDS,$AVG_RESEARCH,$AVG_ATTACKS,$AVG_ABILITY,$AVG_STEPS" >> training_log.csv
    echo "Iteration $i complete. Type: $MATCH_TYPE | Avg: $AVG_SCORE | Loss: $LOSS"
    echo "  -> STATS/GAME: Captures: $AVG_CAPTURES | Harvests: $AVG_HARVESTS | Builds: $AVG_BUILDS | Tech: $AVG_RESEARCH | Attacks: $AVG_ATTACKS | Ability: $AVG_ABILITY | Steps: $AVG_STEPS"
    
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
    
    # 5. Cleanup High Scores
    # Sort high score replays by score (ascending) and keep only the top 10
    mkdir -p replays/high_scores
    ls -v replays/high_scores/best_game_score_*.json 2>/dev/null | head -n -10 | xargs -r rm
    
done
