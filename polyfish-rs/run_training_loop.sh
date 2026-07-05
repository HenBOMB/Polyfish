#!/bin/bash
set -e

# NUM_GAMES/MCTS_ITERS/ACTORS/EVAL_SERVERS match self_play CLI flags.
# Games/iteration doubled, iterations halved: same total games, faster wall time.
# Schedules and milestone cadence recalibrated for new window.
# See self_play --help and expert_boost_throughput.md for details.
ITERATIONS=500
NUM_GAMES=64
export MCTS_ITERS=64
# 128 actors measured best on an M3 Max with metal (~578 moves/s @ 128 games+).
# Throughput scales with concurrent games; small NUM_GAMES (-g) is a real limiter, not this knob.
# See expert_boost_throughput.md for details.
ACTORS=128
# 0 = auto: 2 on metal, 1 on tch/candle. Don't force >1 on tch —
# MPS queue halves performance with 2 tch shards.
EVAL_SERVERS=0
# self_play picks fastest backend: metal, tch, or candle.
# Override with --eval-backend if needed.
export RUST_BACKTRACE=1

# Log all output to session.log while still showing on console
LOG_FILE="session.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"

echo "Building binaries..."
# Detect platform and use appropriate GPU features
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS: Metal/Accelerate + metal-eval (MPSGraph, auto-preferred) with
    # tch-eval kept as an explicit --eval-backend tch fallback
    echo "Building with Metal + Accelerate + metal-eval (MPSGraph) + tch-eval for macOS..."
    export LIBTORCH_USE_PYTORCH=1
    export LIBTORCH_BYPASS_VERSION_CHECK=1
    PATH="$(pwd)/.venv/bin:$PATH" cargo build --bin polyfish --bin self_play --release --features metal,accelerate,tch-eval,metal-eval
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

# Parse long options first, then short options via getopts
RESUME_RUN=""
RESET=false
PASSTHROUGH=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume)
      shift
      if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
        RESUME_RUN="$1"
        shift
      else
        RESUME_RUN="latest"
      fi
      ;;
    --new-run|-N)
      RESUME_RUN=""
      shift
      ;;
    --reset)
      RESET=true
      shift
      ;;
    *)
      PASSTHROUGH+=("$1")
      shift
      ;;
  esac
done
set -- "${PASSTHROUGH[@]}"

# Parse arguments
FORCE_TRAIN=false
BOOST=false
CHILL=false
REWARD_SHAPING=false
# Early-exit if policy loss stalls across iterations (see -p/-d below). 0
# patience disables the check entirely.
EARLY_EXIT_PATIENCE=50
EARLY_EXIT_MIN_DELTA=0.01
while getopts "fbcri:g:n:a:e:p:d:" opt; do
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
    p)
      EARLY_EXIT_PATIENCE=$OPTARG
      ;;
    d)
      EARLY_EXIT_MIN_DELTA=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

# Policy-loss stall tracking (across iterations of the loop below). Scoped to
# this script invocation only — a fresh run gets a fresh patience budget.
BEST_POLICY_LOSS=""
STALL_ITERS=0

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

if [ "$RESET" = true ]; then
    echo "🗑️  Reset flag detected! Deleting model.safetensors and self-play game data to seed a fresh model..."
    rm -f model.safetensors
    rm -f games_*.safetensors
    rm -f archive/games_*.safetensors
    if [ -n "$RESUME_RUN" ]; then
        echo "   (ignoring --resume since --reset always starts a fresh run)"
        RESUME_RUN=""
    fi
fi

# Migrate legacy CSV and resolve run (new run by default; --resume to continue)
.venv/bin/python3 training_log.py migrate
if [ -n "$RESUME_RUN" ]; then
    RUN_INFO=$(.venv/bin/python3 training_log.py resolve-run --resume "$RESUME_RUN")
else
    RUN_INFO=$(.venv/bin/python3 training_log.py resolve-run)
fi
RUN_ID=$(echo "$RUN_INFO" | .venv/bin/python3 -c "import sys,json; print(json.load(sys.stdin)['run_id'])")
RUN_STARTED_AT=$(echo "$RUN_INFO" | .venv/bin/python3 -c "import sys,json; print(json.load(sys.stdin)['run_started_at'])")
START_ITER=$(echo "$RUN_INFO" | .venv/bin/python3 -c "import sys,json; print(json.load(sys.stdin)['start_iter'])")
echo "Training run_id=$RUN_ID started_at=$RUN_STARTED_AT starting at iteration $START_ITER"

trap '.venv/bin/python3 training_log.py finish-run 2>/dev/null || true' EXIT

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
    MATCH_TYPE="selfplay"
    
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
             MATCH_TYPE="league"
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
    
    SP_LOG=$(mktemp)
    ./target/release/self_play --num-games $NUM_GAMES --mcts-iters $MCTS_ITERS --actors $ACTORS --eval-servers $EVAL_SERVERS $REWARD_FLAG $OPPONENT_FLAG --tribe1 "$TRIBE1" --tribe2 "$TRIBE2" --iteration "$i" | tee "$SP_LOG"
    SP_STATUS=${PIPESTATUS[0]}
    SP_OUTPUT=$(cat "$SP_LOG")
    rm -f "$SP_LOG"
    if [ "$SP_STATUS" -ne 0 ]; then
        echo "Self-play failed with exit code $SP_STATUS" >&2
        exit "$SP_STATUS"
    fi
    
    GAME_JSON=$(echo "$SP_OUTPUT" | .venv/bin/python3 training_log.py parse-self-play --input -)
    GAMES_FILE=$(echo "$GAME_JSON" | .venv/bin/python3 -c "import sys,json; print(json.load(sys.stdin).get('games_file',''))")
    
    # 2. Training
    echo "[Training] Training model..."
    # Stream train.py's output live (batch/epoch progress) instead of buffering
    # it silently until the process exits, while still capturing it to parse
    # METRICS. A plain `TRAIN_OUTPUT=$(...)` would swallow all output until
    # train.py finished, so pipe through `tee` and check PIPESTATUS instead
    # (command substitution here would otherwise hide `set -e` failures too).
    TRAIN_LOG=$(mktemp)
    .venv/bin/python3 train.py | tee "$TRAIN_LOG"
    TRAIN_STATUS=${PIPESTATUS[0]}
    TRAIN_JSON=$(.venv/bin/python3 training_log.py parse-train --input "$TRAIN_LOG")
    rm -f "$TRAIN_LOG"
    if [ "$TRAIN_STATUS" -ne 0 ]; then
        echo "Training failed with exit code $TRAIN_STATUS" >&2
        exit "$TRAIN_STATUS"
    fi
    LOSS=$(echo "$TRAIN_JSON" | .venv/bin/python3 -c "import sys,json; print(json.load(sys.stdin).get('loss',''))")
    POLICY_LOSS=$(echo "$TRAIN_JSON" | .venv/bin/python3 -c "import sys,json; print(json.load(sys.stdin).get('policy_loss',''))")

    # 3. Log
    .venv/bin/python3 training_log.py append-row \
        --run-id "$RUN_ID" \
        --run-started-at "$RUN_STARTED_AT" \
        --iteration "$i" \
        --games-file "$GAMES_FILE" \
        --game-json "$GAME_JSON" \
        --train-json "$TRAIN_JSON" \
        --match-type "$MATCH_TYPE"
    AVG_SCORE=$(echo "$GAME_JSON" | .venv/bin/python3 -c "import sys,json; print(json.load(sys.stdin).get('avg_score',''))")
    AVG_CAPTURES=$(echo "$GAME_JSON" | .venv/bin/python3 -c "import sys,json; print(json.load(sys.stdin).get('avg_captures',''))")
    AVG_HARVESTS=$(echo "$GAME_JSON" | .venv/bin/python3 -c "import sys,json; print(json.load(sys.stdin).get('avg_harvests',''))")
    AVG_BUILDS=$(echo "$GAME_JSON" | .venv/bin/python3 -c "import sys,json; print(json.load(sys.stdin).get('avg_builds',''))")
    AVG_RESEARCH=$(echo "$GAME_JSON" | .venv/bin/python3 -c "import sys,json; print(json.load(sys.stdin).get('avg_research',''))")
    AVG_ATTACKS=$(echo "$GAME_JSON" | .venv/bin/python3 -c "import sys,json; print(json.load(sys.stdin).get('avg_attacks',''))")
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
    
    # Keep only the last 10 game files (matches train.py's replay_buffer_size;
    # this comment used to claim 10 while the code kept 30 — now actually true)
    ls -t archive/games_*.safetensors 2>/dev/null | tail -n +11 | xargs -r rm

    # 5. Early-exit if policy loss has stalled across iterations
    if [ "$EARLY_EXIT_PATIENCE" -gt 0 ] && [ -n "$POLICY_LOSS" ]; then
        if [ -z "$BEST_POLICY_LOSS" ]; then
            BEST_POLICY_LOSS=$POLICY_LOSS
            STALL_ITERS=0
        else
            IMPROVED=$(awk -v cur="$POLICY_LOSS" -v best="$BEST_POLICY_LOSS" -v delta="$EARLY_EXIT_MIN_DELTA" 'BEGIN { print (cur <= best - delta) ? 1 : 0 }')
            if [ "$IMPROVED" -eq 1 ]; then
                BEST_POLICY_LOSS=$POLICY_LOSS
                STALL_ITERS=0
            else
                STALL_ITERS=$((STALL_ITERS + 1))
            fi
        fi
        echo "  -> Policy loss: $POLICY_LOSS (best: $BEST_POLICY_LOSS, stalled $STALL_ITERS/$EARLY_EXIT_PATIENCE iterations)"
        if [ "$STALL_ITERS" -ge "$EARLY_EXIT_PATIENCE" ]; then
            echo "Policy loss hasn't improved by >= $EARLY_EXIT_MIN_DELTA in $EARLY_EXIT_PATIENCE iterations (best: $BEST_POLICY_LOSS). Stopping training loop early."
            break
        fi
    fi

done
