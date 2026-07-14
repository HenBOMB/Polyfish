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
export MCTS_ITERS=128
export DETACH_VALUE_TRUNK=1
# Actor count is platform-resolved in the build block below: metal parks
# actors on the GPU (128 best on M3 Max, ~578 moves/s), but on CPU the actors
# share cores with inference, so ~2x physical cores beats an oversubscribed
# 128. Empty = take the platform default; -a overrides, -b doubles, -c pins 8.
# See expert_boost_throughput.md for details.
ACTORS=""
# Extra self_play flags chosen per platform in the build block (e.g. the CPU
# path forces --eval-backend tch so a forward fans across all cores via OpenMP).
EVAL_BACKEND_FLAG=""
# 3 servers × 2 workers measured best on metal after buffer pooling
# (~610-650 moves/s — see expert_boost_throughput.md). 0 = auto (3 on metal,
# 1 on tch/candle). Don't force >1 on tch — MPS serializes across shards.
EVAL_SERVERS=0
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
    PLATFORM_DEFAULT_ACTORS=128   # actors park on the GPU; oversubscription is free
elif false; then
    # CUDA available (Linux/Windows with NVIDIA GPU)
    echo "Building with CUDA support..."
    # --no-default-features: opt out of the macOS `metal` default, which does
    # not compile on Linux.
    cargo build --bin polyfish --bin self_play --release --no-default-features --features cuda
    PLATFORM_DEFAULT_ACTORS=128
else
    # CPU-only: route inference through libtorch/tch (MKL + OpenMP), which fans
    # a single forward across all cores — candle's BLAS-less gemm can't, and one
    # candle eval-server thread starves the actors (~60% CPU, the low-throughput
    # symptom). tch auto-selects Device::Cpu when no MPS is present. Reuses the
    # venv's torch libs, same as the macOS branch.
    echo "Building CPU-only version (optimized for native CPU)..."
    #echo "Building CPU-only version (tch/libtorch inference, native CPU)..."
    #export LIBTORCH_USE_PYTORCH=1
    #export LIBTORCH_BYPASS_VERSION_CHECK=1
    RUSTFLAGS="-C target-cpu=native" cargo build --bin polyfish --bin self_play --release --no-default-features
    # No rpath for libtorch; point the loader at the venv's torch .so files.
    #export LD_LIBRARY_PATH="$(.venv/bin/python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    #EVAL_BACKEND_FLAG="--eval-backend tch"
    # ~2x physical cores: enough parked actors to hide inference latency without
    # thrashing the cores the OpenMP forward needs. Tune with -a.
    PLATFORM_DEFAULT_ACTORS=32
    # libtorch intra-op threads default to all cores, which is what we want for a
    # fast forward (actors are blocked waiting on it anyway). Cap here if actor
    # tree-search and the forward start fighting: export OMP_NUM_THREADS=<n>.
fi

# Parse long options first, then short options via getopts
RESUME_RUN=""
RESET=false
SNAPSHOT_ONLY=false
SNAPSHOT_RESET=false
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
    --snapshot)
      SNAPSHOT_ONLY=true
      shift
      ;;
    --snapshot-reset)
      SNAPSHOT_RESET=true
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
# Rate new checkpoints on the anchored Elo ladder (rate_checkpoints.sh) in a
# niced background job that mostly runs during the idle Kaggle window.
# -E 0 disables. ELO_SEEDS / ELO_WORKERS env override the arena defaults.
ELO_TRACK=1
# Run a Kaggle training round every TRAIN_EVERY iterations (-K). Between
# rounds, fresh games accumulate in kaggle_pending/ — batching amortizes the
# fixed per-round Kaggle overhead (dataset processing + kernel queue + polls).
TRAIN_EVERY=1
# Greedy-anchor gate knobs (-A/-H/-W/-P), passed through to self_play.
# Empty = use the binary's defaults (hold 10, graduate 0.55, probe 0.05).
ANCHOR_HOLD_ITERS="${ANCHOR_HOLD_ITERS:-}"
ANCHOR_GRADUATE_WINRATE="${ANCHOR_GRADUATE_WINRATE:-}"
ANCHOR_PROBE_FRAC="${ANCHOR_PROBE_FRAC:-}"
while getopts "fbcri:g:n:a:e:l:K:A:H:W:P:E:" opt; do
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
    K)
      TRAIN_EVERY=$OPTARG
      ;;
    A)
      ANCHOR_FRAC=$OPTARG
      ;;
    H)
      ANCHOR_HOLD_ITERS=$OPTARG
      ;;
    W)
      ANCHOR_GRADUATE_WINRATE=$OPTARG
      ;;
    P)
      ANCHOR_PROBE_FRAC=$OPTARG
      ;;
    E)
      ELO_TRACK=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

# Resolve actor count: -a wins (set in getopts); otherwise take the per-platform
# default chosen in the build block. -b/-c below still apply on top.
if [ -z "$ACTORS" ]; then
    ACTORS="${PLATFORM_DEFAULT_ACTORS:-128}"
fi

# Derive iteration-keyed schedules from -g so the regime is constant in
# GAMES: scaled(x) = max(1, round(x * BASELINE_GAMES / NUM_GAMES)).
scaled() {
    awk -v x="$1" -v b="$BASELINE_GAMES" -v g="$NUM_GAMES" \
        'BEGIN { v = int(x * b / g + 0.5); print (v < 1 ? 1 : v) }'
}
if [ "${ITERATIONS_SET:-false}" != true ]; then
    ITERATIONS=$(scaled "$ITERATIONS")
fi
CHECKPOINT_EVERY=$(scaled 5)
MILESTONE_EVERY=$(scaled 100)
# Replay window: constant ~10*BASELINE_GAMES games regardless of -g.
# train.py reads REPLAY_BUFFER_FILES; archive pruning keeps window + 1 in sync.
ARCHIVE_KEEP=$(scaled 10)
export REPLAY_BUFFER_FILES=$ARCHIVE_KEEP
echo "Schedule (games-based, -g $NUM_GAMES vs baseline $BASELINE_GAMES): $ITERATIONS iterations, checkpoint every $CHECKPOINT_EVERY, milestone every $MILESTONE_EVERY, league every $LEAGUE_INTERVAL iterations, replay window $ARCHIVE_KEEP files, Kaggle training every $TRAIN_EVERY iteration(s), Elo tracking $([ "$ELO_TRACK" != "0" ] && echo on || echo off)"

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
    .venv/bin/python3 kaggle_manager.py all
fi

# --- Snapshot: save current model + run state into models/ ---
# Used by --snapshot (save only) and --snapshot-reset (save then reset).
make_snapshot() {
    local ts
    ts=$(date +%Y%m%d_%H%M%S)
    local iter=0
    if [ -f training_log.csv ]; then
        iter=$(awk -F, 'NR>1 && $1 ~ /^[0-9]+$/ && $3 ~ /^[0-9]+$/ {v=$3} END{print v+0}' training_log.csv)
    fi
    local snap="models/snapshot_iter${iter}_${ts}"
    mkdir -p "$snap"
    echo "📸 Making snapshot at $snap"
    if [ -f model.safetensors ]; then
        cp -v model.safetensors "$snap/model.safetensors"
    fi
    # Run continuity + analytics state
    for f in training_log.csv config.json .last_train_metrics.json \
             .last_self_play_metrics.json .anchor_state.json \
             value_distribution.json moves_by_turn.json elo.log elo_ratings.json; do
        if [ -f "$f" ]; then
            cp -v "$f" "$snap/$f"
        fi
    done
    # Latest checkpoint (loop restores model.safetensors from this on resume)
    local latest_cp
    latest_cp=$(ls -t checkpoints/model_checkpoint_iter*.safetensors 2>/dev/null | head -n 1)
    if [ -n "$latest_cp" ]; then
        cp -v "$latest_cp" "$snap/$(basename "$latest_cp")"
    fi
    echo "✅ Snapshot complete: $snap"
}

if [ "$SNAPSHOT_ONLY" = true ] || [ "$SNAPSHOT_RESET" = true ]; then
    make_snapshot
fi
if [ "$SNAPSHOT_ONLY" = true ]; then
    echo "📸 Snapshot-only requested; exiting without training."
    exit 0
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

SERVER_PID=""
trap '.venv/bin/python3 training_log.py finish-run 2>/dev/null || true; [ -n "$SERVER_PID" ] && kill $SERVER_PID 2>/dev/null; rm -f .training.pid' EXIT

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
.venv/bin/python3 supabase_sync.py download model.safetensors
# If resuming but model.safetensors is missing, restore latest checkpoint
if [ "$START_ITER" -gt 1 ] && [ ! -f "model.safetensors" ]; then
    LATEST_CP=$(ls checkpoints/model_checkpoint_iter*.safetensors 2>/dev/null | sort -V | tail -n 1 || true)
    if [ -n "$LATEST_CP" ]; then
        echo "🔄 Resuming: Restoring latest checkpoint $(basename $LATEST_CP) to model.safetensors"
        cp "$LATEST_CP" model.safetensors
    fi
fi
.venv/bin/python3 init_model.py

# Start the server only after the model exists — it panics on a missing
# model.safetensors (e.g. right after --reset), killing the live dashboard.
echo "Starting backend server in background..."
./target/release/polyfish &
SERVER_PID=$!

TRAIN_FAILS=0
for ((i=START_ITER; i<START_ITER+ITERATIONS; i++))
do
    ITER_STARTED_AT=$(.venv/bin/python3 training_log.py now-iso)
    echo ""
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

    # Greedy-anchor games (selfplay iterations only; league already has an
    # asymmetric opponent). Up to ANCHOR_FRAC (-A) of each iteration's games
    # are played vs the network-free greedy backend, which actually ELIMINATES
    # passive play; after -H hold iterations self_play gates the fraction on
    # the measured win rate vs greedy (graduate at -W, residual probe -P).
    # ANCHOR_FRAC=0 / -A 0 disables.
    ANCHOR_FLAG=""
    if [ "$MATCH_TYPE" = "selfplay" ]; then
        ANCHOR_FLAG="--anchor-frac ${ANCHOR_FRAC:-0.25}"
        [ -n "$ANCHOR_HOLD_ITERS" ] && ANCHOR_FLAG="$ANCHOR_FLAG --anchor-hold-iters $ANCHOR_HOLD_ITERS"
        [ -n "$ANCHOR_GRADUATE_WINRATE" ] && ANCHOR_FLAG="$ANCHOR_FLAG --anchor-graduate-winrate $ANCHOR_GRADUATE_WINRATE"
        [ -n "$ANCHOR_PROBE_FRAC" ] && ANCHOR_FLAG="$ANCHOR_FLAG --anchor-probe-frac $ANCHOR_PROBE_FRAC"
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
    #./target/release/self_play --num-games $NUM_GAMES --mcts-iters $MCTS_ITERS --gumbel-k 16 --actors $ACTORS --eval-servers $EVAL_SERVERS $REWARD_FLAG $OPPONENT_FLAG $ANCHOR_FLAG --value-trust "$VALUE_TRUST" --tribe1 "$TRIBE1" --tribe2 "$TRIBE2" --iteration "$EFF_ITER" --gamemode "$GAMEMODE" | tee "$SP_LOG"
    ./target/release/self_play --num-games $NUM_GAMES --mcts-iters $MCTS_ITERS --gumbel-k 16 --actors $ACTORS --eval-servers $EVAL_SERVERS $EVAL_BACKEND_FLAG $REWARD_FLAG $OPPONENT_FLAG $ANCHOR_FLAG --value-trust "$VALUE_TRUST" --tribe1 "$TRIBE1" --tribe2 "$TRIBE2" --iteration "$EFF_ITER" --gamemode "$GAMEMODE" | tee "$SP_LOG"
    SP_STATUS=${PIPESTATUS[0]}
    rm -f "$SP_LOG"
    if [ "$SP_STATUS" -ne 0 ]; then
        echo "Self-play failed with exit code $SP_STATUS" >&2
        exit "$SP_STATUS"
    fi
    
    GAME_JSON=$(.venv/bin/python3 training_log.py parse-self-play)
    GAMES_FILE=$(echo "$GAME_JSON" | .venv/bin/python3 -c "import sys,json; print(json.load(sys.stdin).get('games_file',''))")

    # Stage fresh games for the next Kaggle round: kaggle_manager uploads
    # ONLY what's in kaggle_pending/ (cleared by it after a successful round).
    mkdir -p kaggle_pending
    cp games_*.safetensors kaggle_pending/ 2>/dev/null || true

    # Elo: rate any not-yet-rated checkpoint in the background. Niced + capped
    # workers so the arena games mostly fill the idle Kaggle window below.
    # rate_checkpoints.sh skips players already in the ledger, so relaunching
    # is cheap; one rater at a time (stale pid files fail kill -0 and pass).
    if [ "$ELO_TRACK" != "0" ]; then
        NEWEST_CP=$(ls -t checkpoints/model_checkpoint_iter*.safetensors 2>/dev/null | head -n 1 || true)
        RATER_RUNNING=false
        if [ -f .elo_rating.pid ] && kill -0 "$(cat .elo_rating.pid 2>/dev/null)" 2>/dev/null; then
            RATER_RUNNING=true
        fi
        if [ -n "$NEWEST_CP" ] && [ "$RATER_RUNNING" = false ] \
           && { [ ! -f .elo_rating.stamp ] || [ "$NEWEST_CP" -nt .elo_rating.stamp ]; }; then
            touch .elo_rating.stamp
            SEEDS="${ELO_SEEDS:-8}" WORKERS="${ELO_WORKERS:-2}" \
                nice -n 10 bash ./rate_checkpoints.sh >> elo.log 2>&1 &
            echo $! > .elo_rating.pid
            echo "📈 Elo: rating $(basename "$NEWEST_CP") in background (log: elo.log)"
        fi
    fi

    # 2. Training (Kaggle round every TRAIN_EVERY iterations). A failed round
    # is tolerated up to 3 consecutive times: pending games stay staged, the
    # model stays at its last pulled version, and the next round retries with
    # more accumulated data.
    TRAIN_JSON="{}"
    LOSS=""
    if [ $((i % TRAIN_EVERY)) -eq 0 ]; then
        if .venv/bin/python3 kaggle_manager.py all; then
            TRAIN_FAILS=0
            TRAIN_JSON=$(.venv/bin/python3 training_log.py parse-train)
            LOSS=$(echo "$TRAIN_JSON" | .venv/bin/python3 -c "import sys,json; print(json.load(sys.stdin).get('loss',''))")
        else
            TRAIN_FAILS=$((TRAIN_FAILS + 1))
            echo "⚠️  Kaggle training round failed ($TRAIN_FAILS consecutive); games remain staged in kaggle_pending/" >&2
            if [ "$TRAIN_FAILS" -ge 3 ]; then
                echo "3 consecutive Kaggle training failures — aborting" >&2
                exit 1
            fi
        fi
    fi

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
    AVG_CAP_CAPITALS=$(echo "$GAME_JSON" | .venv/bin/python3 -c "import sys,json; print(json.load(sys.stdin).get('avg_cap_capitals',''))")
    echo "Iteration $i complete. Type: $MATCH_TYPE | Avg: $AVG_SCORE | Loss: $LOSS"
    echo "  -> STATS/GAME: Captures: $AVG_CAPTURES (Capitals: $AVG_CAP_CAPITALS) | Harvests: $AVG_HARVESTS | Builds: $AVG_BUILDS | Tech: $AVG_RESEARCH | Attacks: $AVG_ATTACKS | Revealed: $AVG_REVEALED_TILES | Owned: $AVG_CAPTURED_TILES"

    # Print the ladder whenever the background rater has produced a new fit.
    if [ "$ELO_TRACK" != "0" ] && [ -f elo_ratings.json ] \
       && { [ ! -f .elo_reported.stamp ] || [ elo_ratings.json -nt .elo_reported.stamp ]; }; then
        touch .elo_reported.stamp
        echo "📊 Elo ladder (anchored: random = 0):"
        .venv/bin/python3 elo.py report 2>/dev/null || true
    fi

    # 4. Checkpoint (every CHECKPOINT_EVERY iterations ≈ every 50*BASELINE_GAMES games)
    if [ $((i % CHECKPOINT_EVERY)) -eq 0 ]; then
        TS=$(date +%Y%m%d_%H%M%S)
        echo "Creating checkpoint for iteration $i (Timestamp: $TS)..."
        cp model.safetensors "checkpoints/model_checkpoint_iter${i}_${TS}.safetensors"
    fi
    
    # Supabase: Backup the new model weights
    .venv/bin/python3 supabase_sync.py upload model.safetensors
    if [ -f training_log.csv ]; then .venv/bin/python3 supabase_sync.py upload training_log.csv; fi
    if [ -f elo_ratings.json ]; then .venv/bin/python3 supabase_sync.py upload elo_ratings.json; fi
    
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
