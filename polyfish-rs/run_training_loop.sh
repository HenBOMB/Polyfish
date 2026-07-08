#!/bin/bash
set -e

# Write PID file for server detection, clean up on exit
echo $$ > .training.pid

# NUM_GAMES/MCTS_ITERS/ACTORS/EVAL_SERVERS match self_play CLI flags.
# All iteration-keyed schedules (total iterations, curriculum pacing,
# checkpoint cadence, milestone spacing, replay-buffer retention) are tuned
# in GAMES at BASELINE_GAMES games/iteration and derived from -g below —
# changing -g keeps the training regime per game identical.
# See self_play --help and expert_boost_throughput.md for details.
BASELINE_GAMES=64
ITERATIONS=500
NUM_GAMES=64
export MCTS_ITERS=64
# 128 actors measured best on an M3 Max with metal (~578 moves/s @ 128 games+).
# Throughput scales with concurrent games; small NUM_GAMES (-g) is a real limiter, not this knob.
# See expert_boost_throughput.md for details.
ACTORS=128
# 3 servers × 2 workers measured best on metal after buffer pooling
# (~610-650 moves/s — see expert_boost_throughput.md). 0 = auto (2 on metal,
# 1 on tch/candle). Don't force >1 on tch — MPS serializes across shards.
EVAL_SERVERS=3
# self_play picks fastest backend: metal, tch, or candle.
# Override with --eval-backend if needed.
export RUST_BACKTRACE=1
# stdout is a pipe (tee below), so Python would block-buffer and train.py's
# progress would appear frozen for the whole training phase without this.
export PYTHONUNBUFFERED=1

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
elif false; then
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
# Play a league match every LEAGUE_INTERVAL iterations (iteration 10, 20, 30,
# ... by default). 0 disables league play entirely. Override with -l.
LEAGUE_INTERVAL=10
while getopts "fbcri:g:n:a:e:l:" opt; do
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
      ITERATIONS_SET=true
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
    l)
      LEAGUE_INTERVAL=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

# Derive iteration-keyed schedules from -g so the regime is constant in
# GAMES: scaled(x) = max(1, round(x * BASELINE_GAMES / NUM_GAMES)).
scaled() {
    awk -v x="$1" -v b="$BASELINE_GAMES" -v g="$NUM_GAMES" \
        'BEGIN { v = int(x * b / g + 0.5); print (v < 1 ? 1 : v) }'
}
if [ "${ITERATIONS_SET:-false}" != true ]; then
    ITERATIONS=$(scaled "$ITERATIONS")
fi
CHECKPOINT_EVERY=$(scaled 50)
MILESTONE_EVERY=$(scaled 100)
# Replay window: constant ~10*BASELINE_GAMES games regardless of -g.
# train.py reads REPLAY_BUFFER_FILES; archive pruning keeps window + 1 in sync.
ARCHIVE_KEEP=$(scaled 10)
export REPLAY_BUFFER_FILES=$ARCHIVE_KEEP
echo "Schedule (games-based, -g $NUM_GAMES vs baseline $BASELINE_GAMES): $ITERATIONS iterations, checkpoint every $CHECKPOINT_EVERY, milestone every $MILESTONE_EVERY, league every $LEAGUE_INTERVAL iterations, replay window $ARCHIVE_KEEP files"

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

# Set up config.json sync if not present
if [ ! -f "config.json" ]; then
    echo "{\"gamemode\": 2, \"iterations\": $MCTS_ITERS, \"cores\": 2, \"tribes\": [\"Imperius\", \"Bardur\", \"Oumaji\", \"Kickoo\", \"XinXi\"]}" > config.json
fi

echo "Starting backend server in background..."
./target/release/polyfish &
SERVER_PID=$!

trap '.venv/bin/python3 training_log.py finish-run 2>/dev/null || true; kill $SERVER_PID 2>/dev/null; rm -f .training.pid' EXIT

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
    ITER_STARTED_AT=$(.venv/bin/python3 training_log.py now-iso)
    echo "=================================================="
    echo "Starting Iteration $i"
    echo "=================================================="
    
    # 1. League Training Logic (every LEAGUE_INTERVAL iterations, deterministic)
    # Check if we have checkpoints to play against
    mkdir -p checkpoints
    OPPONENT_FLAG=""
    MATCH_TYPE="selfplay"

    if [ "$LEAGUE_INTERVAL" -gt 0 ] && [ $((i % LEAGUE_INTERVAL)) -eq 0 ] && [ -d "checkpoints" ] && [ "$(ls -A checkpoints)" ]; then
        # HISTORICAL-ONLY league selection: the latest checkpoint is ~the
        # current net, so playing it is mirror play with extra steps and
        # breaks no symmetry. Prefer genuinely old checkpoints; fall back to
        # anything that isn't the newest one.
        ALL_CPS=$(ls -t checkpoints/model_checkpoint_iter*.safetensors 2>/dev/null || true)
        HIST_CPS=$(echo "$ALL_CPS" | tail -n +6)
        NON_LATEST_CPS=$(echo "$ALL_CPS" | tail -n +2)

        if [ -n "$HIST_CPS" ]; then
             SELECTED_CP=$(echo "$HIST_CPS" | portable_shuf 1)
        elif [ -n "$NON_LATEST_CPS" ]; then
             SELECTED_CP=$(echo "$NON_LATEST_CPS" | portable_shuf 1)
        else
             SELECTED_CP=""
        fi

        if [ -n "$SELECTED_CP" ]; then
             OPPONENT_FLAG="--opponent $SELECTED_CP"
             MATCH_TYPE="league"
        fi
    fi

    # Heuristic-anchor games (selfplay iterations only; league already has an
    # asymmetric opponent). ANCHOR_FRAC of each iteration's games are played
    # vs the network-free heuristic backend so passivity actually loses and
    # the relative value label carries signal. ANCHOR_FRAC=0 disables.
    ANCHOR_FLAG=""
    if [ "$MATCH_TYPE" = "selfplay" ]; then
        ANCHOR_FLAG="--anchor-frac ${ANCHOR_FRAC:-0.25}"
    fi

    # Value-head trust ramp, RUN-relative (loop iteration i, not EFF_ITER —
    # ITER_OFFSET-shifted runs would saturate the in-binary iteration ramp
    # immediately). Gates sigma(Q) in-tree and in exported policy targets.
    # VALUE_TRUST_CAP env caps the ramp's destination (e.g. from calibration).
    VALUE_TRUST=$(awk -v i="$i" -v r="${VALUE_TRUST_RAMP_ITERS:-30}" -v cap="${VALUE_TRUST_CAP:-1.0}" \
        'BEGIN { t = i / r; if (t > 1) t = 1; t = t * cap; printf "%.3f", t }')

    # Dynamically fetch parameters from config.json (set by dashboard UI)
    if [ -f "config.json" ]; then
        GAMEMODE=$(jq -r '.gamemode // 2' config.json)
        MCTS_ITERS=$(jq -r '.mctsIters // 64' config.json)
        # Parse tribes array into bash array safely
        TRIBE_LIST=()
        while IFS= read -r line; do
            if [ -n "$line" ]; then
                TRIBE_LIST+=("$line")
            fi
        done < <(jq -r '.tribes[]? // empty' config.json)
    else
        GAMEMODE=2
    fi

    # Fallback to defaults if parsing failed or file missing
    if [ ${#TRIBE_LIST[@]} -eq 0 ]; then
        TRIBE_LIST=("Imperius" "Imperius")
    fi

    # Shuffle and pick top 2
    SELECTED_TRIBES=($(printf "%s\n" "${TRIBE_LIST[@]}" | portable_shuf 2))
    TRIBE1=${SELECTED_TRIBES[0]}
    TRIBE2=${SELECTED_TRIBES[1]}
    
    # Curriculum pacing is keyed to GAMES seen, not loop count: self_play's
    # iteration thresholds (50/100/150) were tuned at BASELINE_GAMES/iter.
    # ITER_OFFSET (env, default 0) shifts the schedule forward — e.g.
    # ITER_OFFSET=76 starts at the 30-turn curriculum stage with the heuristic
    # prior mostly annealed, for resuming from a behavior-cloned model.
    EFF_ITER=$(awk -v i="$i" -v b="$BASELINE_GAMES" -v g="$NUM_GAMES" \
        'BEGIN { print int((i - 1) * g / b) + 1 }')
    EFF_ITER=$((EFF_ITER + ${ITER_OFFSET:-0}))

    SP_LOG=$(mktemp)
    ./target/release/self_play --num-games $NUM_GAMES --mcts-iters $MCTS_ITERS --actors $ACTORS --eval-servers $EVAL_SERVERS $REWARD_FLAG $OPPONENT_FLAG $ANCHOR_FLAG --value-trust "$VALUE_TRUST" --tribe1 "$TRIBE1" --tribe2 "$TRIBE2" --iteration "$EFF_ITER" --gamemode "$GAMEMODE" | tee "$SP_LOG"
    SP_STATUS=${PIPESTATUS[0]}
    rm -f "$SP_LOG"
    if [ "$SP_STATUS" -ne 0 ]; then
        echo "Self-play failed with exit code $SP_STATUS" >&2
        exit "$SP_STATUS"
    fi
    
    GAME_JSON=$(.venv/bin/python3 training_log.py parse-self-play)
    GAMES_FILE=$(echo "$GAME_JSON" | .venv/bin/python3 -c "import sys,json; print(json.load(sys.stdin).get('games_file',''))")
    
    # 2. Training
    # Stream train.py's output live (batch/epoch progress) instead of buffering
    # it silently until the process exits. Metrics are read from the sidecar
    # JSON file after train.py finishes.
    .venv/bin/python3 train.py
    TRAIN_STATUS=$?
    TRAIN_JSON=$(.venv/bin/python3 training_log.py parse-train)
    if [ "$TRAIN_STATUS" -ne 0 ]; then
        echo "Training failed with exit code $TRAIN_STATUS" >&2
        exit "$TRAIN_STATUS"
    fi
    LOSS=$(echo "$TRAIN_JSON" | .venv/bin/python3 -c "import sys,json; print(json.load(sys.stdin).get('loss',''))")

    # 3. Log
    .venv/bin/python3 training_log.py append-row \
        --run-id "$RUN_ID" \
        --iter-started-at "$ITER_STARTED_AT" \
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
    AVG_REVEALED_TILES=$(echo "$GAME_JSON" | .venv/bin/python3 -c "import sys,json; print(json.load(sys.stdin).get('avg_revealed_tiles',''))")
    AVG_CAPTURED_TILES=$(echo "$GAME_JSON" | .venv/bin/python3 -c "import sys,json; print(json.load(sys.stdin).get('avg_captured_tiles',''))")
    echo "Iteration $i complete. Type: $MATCH_TYPE | Avg: $AVG_SCORE | Loss: $LOSS"
    echo "  -> STATS/GAME: Captures: $AVG_CAPTURES | Harvests: $AVG_HARVESTS | Builds: $AVG_BUILDS | Tech: $AVG_RESEARCH | Attacks: $AVG_ATTACKS | Revealed: $AVG_REVEALED_TILES | Owned: $AVG_CAPTURED_TILES"
    
    # 4. Checkpoint (every CHECKPOINT_EVERY iterations ≈ every 50*BASELINE_GAMES games)
    if [ $((i % CHECKPOINT_EVERY)) -eq 0 ]; then
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
                # Keep historical milestones (games-based spacing; checkpoints
                # from runs with a different -g prune on this run's spacing)
                if [ $((ITER_VAL % MILESTONE_EVERY)) -eq 0 ] || [ "$ITER_VAL" -eq 1 ]; then
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
    
    # Keep only ARCHIVE_KEEP game files — a constant ~10*BASELINE_GAMES-game
    # replay window regardless of -g (train.py reads the same value via
    # REPLAY_BUFFER_FILES)
    ls -t archive/games_*.safetensors 2>/dev/null | tail -n +$((ARCHIVE_KEEP + 1)) | xargs -r rm

done
