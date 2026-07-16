#!/bin/bash
set -e

# ============================================================================
# RunPod all-local training loop.
#
# Same UX as run_training_loop.sh, but purpose-built for a single NVIDIA GPU
# box (RunPod): self_play runs on the GPU (candle CUDA) AND the gradient step
# runs locally via train.py on the SAME GPU. No Kaggle, no upload/download.
#
# This script never modifies run_training_loop.sh — it is a standalone parallel
# entrypoint. All the training-regime knobs (schedules, league, anchor gate,
# curriculum, checkpoint cadence, Elo) are kept identical to the main loop so
# the regime is unchanged; only the *where* changes (GPU-local instead of
# Kaggle) and the build is CUDA-only.
#
# One-time setup (compile once, ideally onto a persistent /workspace volume):
#   ./runpod_setup.sh
#
# Then run exactly like the main loop:
#   ./run_training_runpod.sh                 # new run, 500 iters (games-scaled)
#   ./run_training_runpod.sh --resume        # continue latest run
#   ./run_training_runpod.sh --reset         # wipe model + games, fresh model
#   ./run_training_runpod.sh -g 128 -n 256   # more games / deeper MCTS
#
# Build-time env (all optional):
#   SKIP_BUILD=1   skip the cargo rebuild if binaries already exist (persistent
#                  volume restarts: near-instant startup)
#   FAST_BUILD=1   thin-LTO / 16 codegen-units release build — much faster to
#                  compile (~2-3x), marginally slower binaries. Good for the
#                  first impatient build; the fat-LTO default is worth it once.
#   TRAIN_EPOCHS=N epochs per local train.py round (default 2, same as Kaggle)
# ============================================================================

echo $$ > .training.pid

BASELINE_GAMES=64
ITERATIONS=500
NUM_GAMES=64
export MCTS_ITERS=128
export DETACH_VALUE_TRUNK=1
# Actors park on the GPU (inference is off the actor cores), so oversubscription
# is effectively free — 128 is the GPU default, same as the metal/CUDA path in
# the main loop. Override with -a; -b doubles; -c pins 8.
ACTORS=""
# candle CUDA is auto-selected by the cuda-featured build; no --eval-backend
# flag needed (that flag is only for the CPU/tch path in the main loop).
EVAL_BACKEND_FLAG=""
EVAL_SERVERS=0
export RUST_BACKTRACE=1
export PYTHONUNBUFFERED=1
export POLYFISH_DEVICE=cuda
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"

LOG_FILE="session.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"

# --- Build (CUDA-only) ------------------------------------------------------
# In the Docker image, binaries are precompiled and cargo is not installed.
# Only attempt a build if cargo is available AND binaries are missing or stale.
if [ -x ./target/release/self_play ] && [ -x ./target/release/polyfish ] && [ -x ./target/release/arena ]; then
    echo "Precompiled binaries found — skipping build."
elif command -v cargo >/dev/null 2>&1; then
    echo "Building CUDA binaries (release)..."
    # --no-default-features: opt out of the macOS `metal` default, which does
    # not compile on Linux. --features cuda routes candle inference onto the GPU.
    BUILD_ENV=()
    if [ "${FAST_BUILD:-0}" = "1" ]; then
        echo "⚡ FAST_BUILD: thin LTO + 16 codegen-units (faster compile, slightly slower binary)"
        BUILD_ENV=(CARGO_PROFILE_RELEASE_LTO=thin CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16)
    fi
    env "${BUILD_ENV[@]}" cargo build --bin polyfish --bin self_play --bin benchmark --bin arena --release \
        --no-default-features --features cuda
else
    echo "Error: binaries not found and cargo not available to build them." >&2
    exit 1
fi
PLATFORM_DEFAULT_ACTORS=128

# --- Long-option parsing (mirror of run_training_loop.sh) -------------------
RESUME_RUN=""
RESET=false
IDLE=false
BENCHMARK=false
RESUME_FROM_ITER=""
PASSTHROUGH=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume)
      shift
      if [[ "${1:-}" =~ ^[0-9]+$ ]]; then RESUME_RUN="$1"; shift; else RESUME_RUN="latest"; fi
      ;;
    --resume-from)
      shift
      if [[ "${1:-}" =~ ^[0-9]+$ ]]; then RESUME_FROM_ITER="$1"; shift
      else echo "Error: --resume-from requires an iteration number (e.g. --resume-from 84)" >&2; exit 1; fi
      ;;
    --new-run|-N) RESUME_RUN=""; shift ;;
    --reset) RESET=true; shift ;;
    --idle) IDLE=true; shift ;;
    --benchmark) BENCHMARK=true; shift ;;
    --supabase) SYNC_SUPABASE=true; shift ;;
    *) PASSTHROUGH+=("$1"); shift ;;
  esac
done
set -- "${PASSTHROUGH[@]}"

if [ "$IDLE" = true ]; then
    echo "Idle mode requested (--idle). Sleeping indefinitely to keep the pod alive without training."
    sleep infinity
fi

if [ "$BENCHMARK" = true ]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║            📊 GPU THROUGHPUT BENCHMARK SWEEP                ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""

    # Ensure model exists for benchmark runs
    if [ ! -f "model.safetensors" ]; then
        .venv/bin/python3 supabase_sync.py download model.safetensors
        .venv/bin/python3 init_model.py
    fi

    BENCH_GAMES=${BENCH_GAMES:-8}
    BENCH_MCTS=${MCTS_ITERS:-128}
    BENCH_RESULTS=""
    BENCH_BEST_MPS=0
    BENCH_BEST_ACTORS=128
    BENCH_BEST_ESERVERS=0

    # Configurations to sweep: (actors, eval-servers)
    # eval-servers=0 means auto (the binary picks its default)
    BENCH_CONFIGS=(
        "64,0"
        "96,0"
        "128,0"
        "160,0"
        "192,0"
        "256,0"
        "128,1"
        "128,2"
        "128,3"
        "192,2"
        "192,3"
        "256,2"
        "256,3"
    )

    echo "Testing ${#BENCH_CONFIGS[@]} configurations (${BENCH_GAMES} games × ${BENCH_MCTS} MCTS iters each)..."
    echo ""

    for cfg in "${BENCH_CONFIGS[@]}"; do
        IFS=',' read -r b_actors b_eservers <<< "$cfg"
        printf "  ⏱  actors=%-4s eval-servers=%-2s ... " "$b_actors" "$b_eservers"

        BENCH_OUT=$(./target/release/self_play \
            --num-games "$BENCH_GAMES" \
            --mcts-iters "$BENCH_MCTS" \
            --gumbel-k 16 \
            --actors "$b_actors" \
            --eval-servers "$b_eservers" \
            --iteration 1 \
            --gamemode 2 \
            --tribe1 Imperius --tribe2 Bardur \
            $EVAL_BACKEND_FLAG 2>&1 || true)

        # Parse the throughput line: "  Throughput: 123.45 moves/sec (N moves over Xs)"
        MPS=$(echo "$BENCH_OUT" | grep -oP 'Throughput:\s+\K[0-9]+(\.[0-9]+)?' | tail -1)
        if [ -z "$MPS" ]; then MPS="0"; fi

        printf "%8s moves/sec\n" "$MPS"

        BENCH_RESULTS="${BENCH_RESULTS}${MPS},${b_actors},${b_eservers}\n"

        # Track best
        IS_BETTER=$(awk -v new="$MPS" -v best="$BENCH_BEST_MPS" 'BEGIN { print (new+0 > best+0) ? 1 : 0 }')
        if [ "$IS_BETTER" -eq 1 ]; then
            BENCH_BEST_MPS="$MPS"
            BENCH_BEST_ACTORS="$b_actors"
            BENCH_BEST_ESERVERS="$b_eservers"
        fi
    done

    echo ""
    echo "┌─────────────────────────────────────────────────┐"
    echo "│              RESULTS (ranked)                   │"
    echo "├──────────┬──────────────┬────────────────────── ┤"
    printf "│ %-8s │ %-12s │ %-22s│\n" "actors" "eval-servers" "moves/sec"
    echo "├──────────┼──────────────┼────────────────────── ┤"
    echo -e "$BENCH_RESULTS" | sort -t',' -k1 -rn | while IFS=',' read -r mps actors eservers; do
        [ -z "$mps" ] && continue
        if [ "$actors" = "$BENCH_BEST_ACTORS" ] && [ "$eservers" = "$BENCH_BEST_ESERVERS" ]; then
            printf "│ %-8s │ %-12s │ %-18s 🏆 │\n" "$actors" "$eservers" "$mps"
        else
            printf "│ %-8s │ %-12s │ %-22s│\n" "$actors" "$eservers" "$mps"
        fi
    done
    echo "└──────────┴──────────────┴────────────────────── ┘"

    echo ""
    echo "🏆 Best config: actors=$BENCH_BEST_ACTORS eval-servers=$BENCH_BEST_ESERVERS → $BENCH_BEST_MPS moves/sec"
    echo "   Applying to training loop."
    echo ""

    ACTORS="$BENCH_BEST_ACTORS"
    EVAL_SERVERS="$BENCH_BEST_ESERVERS"
    ACTORS_SET=true
    ESERVERS_SET=true
fi

FORCE_TRAIN=false
BOOST=false
CHILL=false
REWARD_SHAPING=false
LEAGUE_INTERVAL=10
ELO_TRACK=1
TRAIN_EVERY=1
ANCHOR_HOLD_ITERS="${ANCHOR_HOLD_ITERS:-}"
ANCHOR_GRADUATE_WINRATE="${ANCHOR_GRADUATE_WINRATE:-}"
ANCHOR_PROBE_FRAC="${ANCHOR_PROBE_FRAC:-}"
while getopts "fbcri:g:n:a:e:l:K:A:H:W:P:E:" opt; do
  case $opt in
    f) FORCE_TRAIN=true ;;
    b) BOOST=true ;;
    c) CHILL=true ;;
    r) REWARD_SHAPING=true ;;
    i) ITERATIONS=$OPTARG; ITERATIONS_SET=true ;;
    g) NUM_GAMES=$OPTARG ;;
    n) MCTS_ITERS=$OPTARG; MCTS_ITERS_SET=true ;;
    a) ACTORS=$OPTARG; ACTORS_SET=true ;;
    e) EVAL_SERVERS=$OPTARG; ESERVERS_SET=true ;;
    l) LEAGUE_INTERVAL=$OPTARG ;;
    K) TRAIN_EVERY=$OPTARG ;;
    A) ANCHOR_FRAC=$OPTARG ;;
    H) ANCHOR_HOLD_ITERS=$OPTARG ;;
    W) ANCHOR_GRADUATE_WINRATE=$OPTARG ;;
    P) ANCHOR_PROBE_FRAC=$OPTARG ;;
    E) ELO_TRACK=$OPTARG ;;
    \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
  esac
done

if [ -z "$ACTORS" ]; then ACTORS="${PLATFORM_DEFAULT_ACTORS:-128}"; fi

scaled() {
    awk -v x="$1" -v b="$BASELINE_GAMES" -v g="$NUM_GAMES" \
        'BEGIN { v = int(x * b / g + 0.5); print (v < 1 ? 1 : v) }'
}
if [ "${ITERATIONS_SET:-false}" != true ]; then ITERATIONS=$(scaled "$ITERATIONS"); fi
CHECKPOINT_EVERY=$(scaled 10)
MILESTONE_EVERY=$(scaled 100)
ARCHIVE_KEEP=$(scaled 10)
export REPLAY_BUFFER_FILES=$ARCHIVE_KEEP
echo "Schedule (games-based, -g $NUM_GAMES vs baseline $BASELINE_GAMES): $ITERATIONS iterations, checkpoint every $CHECKPOINT_EVERY, milestone every $MILESTONE_EVERY, league every $LEAGUE_INTERVAL iterations, replay window $ARCHIVE_KEEP files, local train every $TRAIN_EVERY iteration(s), Elo tracking $([ \"$ELO_TRACK\" != \"0\" ] && echo on || echo off)"

REWARD_FLAG=""
if [ "$REWARD_SHAPING" = true ]; then REWARD_FLAG="--reward-shaping"; echo "🎯 Reward shaping enabled!"; fi
if [ "$BOOST" = true ]; then ACTORS=$((ACTORS * 2)); echo "🚀 Boost mode! Using $ACTORS actors"; fi
if [ "$CHILL" = true ]; then ACTORS=8; echo "❄️ Chill mode! Using $ACTORS actors"; fi

if [ "$FORCE_TRAIN" = true ]; then
    echo "Force training flag detected! Running a local train.py round now..."
    TRAIN_EPOCHS=${TRAIN_EPOCHS:-2} .venv/bin/python3 train.py
fi

# --resume-from N: download checkpoint iter N from Supabase, start a fresh run
# with that checkpoint as the initial model (like --reset but seeded from a
# specific historical checkpoint instead of random init).
if [ -n "$RESUME_FROM_ITER" ]; then
    echo "🔄 --resume-from $RESUME_FROM_ITER: downloading checkpoint iter $RESUME_FROM_ITER from Supabase..."
    rm -f model.safetensors
    if ! .venv/bin/python3 supabase_sync.py download-checkpoint-iter "$RESUME_FROM_ITER"; then
        echo "❌ Failed to download checkpoint for iteration $RESUME_FROM_ITER from Supabase. Aborting." >&2
        exit 1
    fi
    # Clean stale game data — fresh run, old games are meaningless
    rm -f games_*.safetensors archive/games_*.safetensors
    # Force new run (ignore any --resume that was also passed)
    RESUME_RUN=""
    echo "✅ Loaded checkpoint iter $RESUME_FROM_ITER as initial model. Starting fresh run."
fi

if [ "$RESET" = true ]; then
    echo "🗑️  Reset: deleting model.safetensors and self-play game data..."
    rm -f model.safetensors games_*.safetensors archive/games_*.safetensors
    if [ -n "$RESUME_RUN" ]; then echo "   (ignoring --resume since --reset starts fresh)"; RESUME_RUN=""; fi
fi

if [ "$SYNC_SUPABASE" = true ] && [ -n "$RESUME_RUN" ]; then
    echo "☁️ Restoring full pod state from Supabase..."
    .venv/bin/python3 supabase_sync.py restore-pod
fi

.venv/bin/python3 training_log.py migrate
if [ -n "$RESUME_RUN" ]; then
    RUN_INFO=$(.venv/bin/python3 training_log.py resolve-run --resume "$RESUME_RUN")
else
    RUN_INFO=$(.venv/bin/python3 training_log.py resolve-run)
fi
RUN_ID=$(echo "$RUN_INFO" | .venv/bin/python3 -c "import sys,json; print(json.load(sys.stdin)['run_id'])")
RUN_STARTED_AT=$(echo "$RUN_INFO" | .venv/bin/python3 -c "import sys,json; print(json.load(sys.stdin)['run_started_at'])")
START_ITER=$(echo "$RUN_INFO" | .venv/bin/python3 -c "import sys,json; print(json.load(sys.stdin)['start_iter'])")
if [ -n "$RESUME_FROM_ITER" ]; then
    START_ITER="$RESUME_FROM_ITER"
fi
if [ -n "$RESUME_FROM_ITER" ]; then
    echo "Training run_id=$RUN_ID started_at=$RUN_STARTED_AT starting at iteration $START_ITER (seeded from checkpoint iter $RESUME_FROM_ITER)"
else
    echo "Training run_id=$RUN_ID started_at=$RUN_STARTED_AT starting at iteration $START_ITER"
fi

if [ ! -f "config.json" ]; then
    echo "{\"gamemode\": 2, \"mctsIters\": $MCTS_ITERS, \"cores\": 2, \"tribes\": [\"Imperius\", \"Bardur\", \"Oumaji\", \"Kickoo\", \"XinXi\"]}" > config.json
fi

SERVER_PID=""
trap '.venv/bin/python3 training_log.py finish-run 2>/dev/null || true; [ -n "$SERVER_PID" ] && kill $SERVER_PID 2>/dev/null; rm -f .training.pid' EXIT

portable_shuf() {
    local n=$1
    local lines=()
    while IFS= read -r line; do [ -n "$line" ] && lines+=("$line"); done
    local count=${#lines[@]}
    for ((idx = count - 1; idx > 0; idx--)); do
        local j=$((RANDOM % (idx + 1)))
        local tmp="${lines[idx]}"; lines[idx]="${lines[j]}"; lines[j]="$tmp"
    done
    for ((idx = 0; idx < n && idx < count; idx++)); do echo "${lines[idx]}"; done
}

# 0. Initialize & Auto-Restore Model
# --resume-from already placed model.safetensors; skip the normal download path
# so we don't overwrite it with the generic latest model from Supabase.
if [ -n "$RESUME_FROM_ITER" ]; then
    echo "Initializing model from checkpoint iter $RESUME_FROM_ITER (already downloaded)..."
else
    echo "Initializing/Checking model..."
    .venv/bin/python3 supabase_sync.py download model.safetensors
    if [ "$START_ITER" -gt 1 ] && [ ! -f "model.safetensors" ]; then
        LATEST_CP=$(ls checkpoints/model_checkpoint_iter*.safetensors 2>/dev/null | sort -V | tail -n 1 || true)
        if [ -n "$LATEST_CP" ]; then
            echo "🔄 Resuming: Restoring $(basename $LATEST_CP) to model.safetensors"
            cp "$LATEST_CP" model.safetensors
        fi
    fi
fi
.venv/bin/python3 init_model.py

echo "Starting backend server in background (dashboard on :3000)..."
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

    mkdir -p checkpoints
    OPPONENT_FLAG=""
    MATCH_TYPE="selfplay"

    if [ "$LEAGUE_INTERVAL" -gt 0 ] && [ $((i % LEAGUE_INTERVAL)) -eq 0 ] && [ -d "checkpoints" ] && [ "$(ls -A checkpoints)" ]; then
        ALL_CPS=$(ls -t checkpoints/model_checkpoint_iter*.safetensors 2>/dev/null || true)
        HIST_CPS=$(echo "$ALL_CPS" | tail -n +6)
        NON_LATEST_CPS=$(echo "$ALL_CPS" | tail -n +2)
        if [ -n "$HIST_CPS" ]; then SELECTED_CP=$(echo "$HIST_CPS" | portable_shuf 1)
        elif [ -n "$NON_LATEST_CPS" ]; then SELECTED_CP=$(echo "$NON_LATEST_CPS" | portable_shuf 1)
        else SELECTED_CP=""; fi
        if [ -n "$SELECTED_CP" ]; then OPPONENT_FLAG="--opponent $SELECTED_CP"; MATCH_TYPE="league"; fi
    fi

    ANCHOR_FLAG=""
    if [ "$MATCH_TYPE" = "selfplay" ]; then
        ANCHOR_FLAG="--anchor-frac ${ANCHOR_FRAC:-0.25}"
        [ -n "$ANCHOR_HOLD_ITERS" ] && ANCHOR_FLAG="$ANCHOR_FLAG --anchor-hold-iters $ANCHOR_HOLD_ITERS"
        [ -n "$ANCHOR_GRADUATE_WINRATE" ] && ANCHOR_FLAG="$ANCHOR_FLAG --anchor-graduate-winrate $ANCHOR_GRADUATE_WINRATE"
        [ -n "$ANCHOR_PROBE_FRAC" ] && ANCHOR_FLAG="$ANCHOR_FLAG --anchor-probe-frac $ANCHOR_PROBE_FRAC"
    fi

    VALUE_TRUST=$(awk -v i="$i" -v r="${VALUE_TRUST_RAMP_ITERS:-30}" -v cap="${VALUE_TRUST_CAP:-1.0}" \
        'BEGIN { t = i / r; if (t > 1) t = 1; t = t * cap; printf "%.3f", t }')

    if [ -f "config.json" ]; then
        GAMEMODE=$(jq -r '.gamemode // 2' config.json)
        if [ "${MCTS_ITERS_SET:-false}" != true ]; then
            MCTS_ITERS=$(jq -r '.mctsIters // 64' config.json)
        fi
        TRIBE_LIST=()
        while IFS= read -r line; do [ -n "$line" ] && TRIBE_LIST+=("$line"); done < <(jq -r '.tribes[]? // empty' config.json)
    else
        GAMEMODE=2
    fi
    if [ ${#TRIBE_LIST[@]} -eq 0 ]; then TRIBE_LIST=("Imperius" "Imperius"); fi
    SELECTED_TRIBES=($(printf "%s\n" "${TRIBE_LIST[@]}" | portable_shuf 2))
    TRIBE1=${SELECTED_TRIBES[0]}
    TRIBE2=${SELECTED_TRIBES[1]}

    EFF_ITER=$(awk -v i="$i" -v b="$BASELINE_GAMES" -v g="$NUM_GAMES" \
        'BEGIN { print int((i - 1) * g / b) + 1 }')
    EFF_ITER=$((EFF_ITER + ${ITER_OFFSET:-0}))

    SP_LOG=$(mktemp)
    ./target/release/self_play --num-games $NUM_GAMES --mcts-iters $MCTS_ITERS --gumbel-k 16 --actors $ACTORS --eval-servers $EVAL_SERVERS $EVAL_BACKEND_FLAG $REWARD_FLAG $OPPONENT_FLAG $ANCHOR_FLAG --value-trust "$VALUE_TRUST" --tribe1 "$TRIBE1" --tribe2 "$TRIBE2" --iteration "$EFF_ITER" --gamemode "$GAMEMODE" | tee "$SP_LOG"
    SP_STATUS=${PIPESTATUS[0]}
    rm -f "$SP_LOG"
    if [ "$SP_STATUS" -ne 0 ]; then echo "Self-play failed with exit code $SP_STATUS" >&2; exit "$SP_STATUS"; fi

    GAME_JSON=$(.venv/bin/python3 training_log.py parse-self-play)
    GAMES_FILE=$(echo "$GAME_JSON" | .venv/bin/python3 -c "import sys,json; print(json.load(sys.stdin).get('games_file',''))")

    # Elo tracking moved to run synchronously after checkpoint creation.

    # 2. Training — LOCAL train.py on the GPU (replaces kaggle_manager.py all).
    # train.py reads fresh games_*.safetensors (still in cwd — cleanup below runs
    # AFTER) + archive/ replay window + teachers/, auto-selects CUDA, and writes
    # model.safetensors + .last_train_metrics.json (the exact sidecar parse-train
    # reads). Failure tolerated up to 3 consecutive rounds, same as the main loop.
    TRAIN_JSON="{}"
    LOSS=""
    if [ $((i % TRAIN_EVERY)) -eq 0 ]; then
        if TRAIN_EPOCHS=${TRAIN_EPOCHS:-2} .venv/bin/python3 train.py; then
            TRAIN_FAILS=0
            TRAIN_JSON=$(.venv/bin/python3 training_log.py parse-train)
            LOSS=$(echo "$TRAIN_JSON" | .venv/bin/python3 -c "import sys,json; print(json.load(sys.stdin).get('loss',''))")
        else
            TRAIN_FAILS=$((TRAIN_FAILS + 1))
            echo "⚠️  Local train round failed ($TRAIN_FAILS consecutive)" >&2
            if [ "$TRAIN_FAILS" -ge 3 ]; then echo "3 consecutive training failures — aborting" >&2; exit 1; fi
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
    DECISIVE_FRAC=$(echo "$GAME_JSON" | .venv/bin/python3 -c "import sys,json; print(json.load(sys.stdin).get('decisive_frac',''))")
    VLAB_WL=$(echo "$GAME_JSON" | .venv/bin/python3 -c "import sys,json; print(json.load(sys.stdin).get('vlab_wl_share',''))")
    echo "Iteration $i complete. Type: $MATCH_TYPE | Avg: $AVG_SCORE | Loss: $LOSS"
    echo "  -> STATS/GAME: Captures: $AVG_CAPTURES (Capitals: $AVG_CAP_CAPITALS) | Harvests: $AVG_HARVESTS | Builds: $AVG_BUILDS | Tech: $AVG_RESEARCH | Attacks: $AVG_ATTACKS | Revealed: $AVG_REVEALED_TILES | Owned: $AVG_CAPTURED_TILES"
    echo "  -> META: Decisive Frac: $DECISIVE_FRAC | VLab WL Share: $VLAB_WL"

    # 4. Checkpoint
    if [ $((i % CHECKPOINT_EVERY)) -eq 0 ]; then
        TS=$(date +%Y%m%d_%H%M%S)
        CP_NAME="checkpoints/model_checkpoint_iter${i}_${TS}.safetensors"
        echo "Creating checkpoint for iteration $i (Timestamp: $TS)..."
        cp model.safetensors "$CP_NAME"
        .venv/bin/python3 supabase_sync.py upload "$CP_NAME"
    fi

    # Supabase: Backup the new model weights, training logs, and elo ratings
    if [ "$SYNC_SUPABASE" = true ]; then
        .venv/bin/python3 supabase_sync.py backup-pod
        .venv/bin/python3 supabase_sync.py upload model.safetensors
    else
        .venv/bin/python3 supabase_sync.py upload model.safetensors
        if [ -f training_log.csv ]; then .venv/bin/python3 supabase_sync.py upload training_log.csv; fi
        if [ -f elo_ratings.json ]; then .venv/bin/python3 supabase_sync.py upload elo_ratings.json; fi
    fi

    # Smart pruning: last 10 + every MILESTONE_EVERY-th + iter 1
    ALL_FILES=$(ls -t checkpoints/model_checkpoint_iter*.safetensors 2>/dev/null || true)
    if [ -n "$ALL_FILES" ]; then
        idx=0
        echo "$ALL_FILES" | while read -r FILE; do
            idx=$((idx + 1))
            ITER_VAL=$(echo "$FILE" | sed -n 's/.*iter\([0-9]\+\)_.*/\1/p')
            KEEP=false
            if [ $idx -le 10 ]; then KEEP=true
            elif [ -n "$ITER_VAL" ]; then
                if [ $((ITER_VAL % MILESTONE_EVERY)) -eq 0 ] || [ "$ITER_VAL" -eq 1 ]; then KEEP=true; fi
            fi
            if [ "$KEEP" = false ]; then rm "$FILE"; fi
        done
    fi

    # 4b. Cleanup: move played games to archive so train.py sees only fresh
    # ones next iteration; keep a constant ~10*BASELINE_GAMES replay window.
    mkdir -p archive
    mv games_*.safetensors archive/ 2>/dev/null || true
    ls -t archive/games_*.safetensors 2>/dev/null | tail -n +$((ARCHIVE_KEEP + 1)) | xargs -r rm
done

# ============================================================================
# Final Elo Rating
# ============================================================================
if [ "$ELO_TRACK" != "0" ]; then
    echo "📈 Final Elo: rating checkpoints (log: elo.log)..."
    SEEDS="${ELO_SEEDS:-8}" WORKERS="${ELO_WORKERS:-2}" ./rate_checkpoints.sh >> elo.log 2>&1
    echo "📊 Final Elo ladder (anchored: random = 0):"
    .venv/bin/python3 elo.py report 2>/dev/null || true
    if [ -f elo_ratings.json ]; then .venv/bin/python3 supabase_sync.py upload elo_ratings.json; fi
fi
