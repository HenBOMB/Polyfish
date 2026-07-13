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

LOG_FILE="session.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"

# --- Build (CUDA-only) ------------------------------------------------------
if [ ! -x ./target/release/self_play ] || [ ! -x ./target/release/polyfish ]; then
    NEED_BUILD=1
fi
if [ "${SKIP_BUILD:-0}" = "1" ] && [ -x ./target/release/self_play ] && [ -x ./target/release/polyfish ]; then
    echo "SKIP_BUILD=1 and binaries present — skipping cargo build."
else
    echo "Building CUDA binaries (release)..."
    # --no-default-features: opt out of the macOS `metal` default, which does
    # not compile on Linux. --features cuda routes candle inference onto the GPU.
    BUILD_ENV=()
    if [ "${FAST_BUILD:-0}" = "1" ]; then
        echo "⚡ FAST_BUILD: thin LTO + 16 codegen-units (faster compile, slightly slower binary)"
        BUILD_ENV=(CARGO_PROFILE_RELEASE_LTO=thin CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16)
    fi
    env "${BUILD_ENV[@]}" cargo build --bin polyfish --bin self_play --release \
        --no-default-features --features cuda
fi
PLATFORM_DEFAULT_ACTORS=128

# --- Long-option parsing (mirror of run_training_loop.sh) -------------------
RESUME_RUN=""
RESET=false
PASSTHROUGH=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume)
      shift
      if [[ "${1:-}" =~ ^[0-9]+$ ]]; then RESUME_RUN="$1"; shift; else RESUME_RUN="latest"; fi
      ;;
    --new-run|-N) RESUME_RUN=""; shift ;;
    --reset) RESET=true; shift ;;
    *) PASSTHROUGH+=("$1"); shift ;;
  esac
done
set -- "${PASSTHROUGH[@]}"

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
    n) MCTS_ITERS=$OPTARG ;;
    a) ACTORS=$OPTARG ;;
    e) EVAL_SERVERS=$OPTARG ;;
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
echo "Schedule (games-based, -g $NUM_GAMES vs baseline $BASELINE_GAMES): $ITERATIONS iterations, checkpoint every $CHECKPOINT_EVERY, milestone every $MILESTONE_EVERY, league every $LEAGUE_INTERVAL iterations, replay window $ARCHIVE_KEEP files, local train every $TRAIN_EVERY iteration(s), Elo tracking $([ "$ELO_TRACK" != "0" ] && echo on || echo off)"

REWARD_FLAG=""
if [ "$REWARD_SHAPING" = true ]; then REWARD_FLAG="--reward-shaping"; echo "🎯 Reward shaping enabled!"; fi
if [ "$BOOST" = true ]; then ACTORS=$((ACTORS * 2)); echo "🚀 Boost mode! Using $ACTORS actors"; fi
if [ "$CHILL" = true ]; then ACTORS=8; echo "❄️ Chill mode! Using $ACTORS actors"; fi

if [ "$FORCE_TRAIN" = true ]; then
    echo "Force training flag detected! Running a local train.py round now..."
    TRAIN_EPOCHS=${TRAIN_EPOCHS:-2} .venv/bin/python3 train.py
fi

if [ "$RESET" = true ]; then
    echo "🗑️  Reset: deleting model.safetensors and self-play game data..."
    rm -f model.safetensors games_*.safetensors archive/games_*.safetensors
    if [ -n "$RESUME_RUN" ]; then echo "   (ignoring --resume since --reset starts fresh)"; RESUME_RUN=""; fi
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
echo "Training run_id=$RUN_ID started_at=$RUN_STARTED_AT starting at iteration $START_ITER"

if [ ! -f "config.json" ]; then
    echo "{\"gamemode\": 2, \"iterations\": $MCTS_ITERS, \"cores\": 2, \"tribes\": [\"Imperius\", \"Bardur\", \"Oumaji\", \"Kickoo\", \"XinXi\"]}" > config.json
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
echo "Initializing/Checking model..."
if [ "$START_ITER" -gt 1 ] && [ ! -f "model.safetensors" ]; then
    LATEST_CP=$(ls checkpoints/model_checkpoint_iter*.safetensors 2>/dev/null | sort -V | tail -n 1 || true)
    if [ -n "$LATEST_CP" ]; then
        echo "🔄 Resuming: Restoring $(basename $LATEST_CP) to model.safetensors"
        cp "$LATEST_CP" model.safetensors
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
        MCTS_ITERS=$(jq -r '.mctsIters // 64' config.json)
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

    # Elo: rate any not-yet-rated checkpoint in the background (niced). Uses the
    # anchored ladder; rate_checkpoints.sh builds its own arena binary on first
    # use. -E 0 disables (skips a first-run arena compile if you want).
    if [ "$ELO_TRACK" != "0" ]; then
        NEWEST_CP=$(ls -t checkpoints/model_checkpoint_iter*.safetensors 2>/dev/null | head -n 1 || true)
        RATER_RUNNING=false
        if [ -f .elo_rating.pid ] && kill -0 "$(cat .elo_rating.pid 2>/dev/null)" 2>/dev/null; then RATER_RUNNING=true; fi
        if [ -n "$NEWEST_CP" ] && [ "$RATER_RUNNING" = false ] \
           && { [ ! -f .elo_rating.stamp ] || [ "$NEWEST_CP" -nt .elo_rating.stamp ]; }; then
            touch .elo_rating.stamp
            SEEDS="${ELO_SEEDS:-8}" WORKERS="${ELO_WORKERS:-2}" \
                nice -n 10 ./rate_checkpoints.sh >> elo.log 2>&1 &
            echo $! > .elo_rating.pid
            echo "📈 Elo: rating $(basename "$NEWEST_CP") in background (log: elo.log)"
        fi
    fi

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
    echo "Iteration $i complete. Type: $MATCH_TYPE | Avg: $AVG_SCORE | Loss: $LOSS"
    echo "  -> STATS/GAME: Captures: $AVG_CAPTURES (Capitals: $AVG_CAP_CAPITALS) | Harvests: $AVG_HARVESTS | Builds: $AVG_BUILDS | Tech: $AVG_RESEARCH | Attacks: $AVG_ATTACKS | Revealed: $AVG_REVEALED_TILES | Owned: $AVG_CAPTURED_TILES"

    if [ "$ELO_TRACK" != "0" ] && [ -f elo_ratings.json ] \
       && { [ ! -f .elo_reported.stamp ] || [ elo_ratings.json -nt .elo_reported.stamp ]; }; then
        touch .elo_reported.stamp
        echo "📊 Elo ladder (anchored: random = 0):"
        .venv/bin/python3 elo.py report 2>/dev/null || true
    fi

    # 4. Checkpoint
    if [ $((i % CHECKPOINT_EVERY)) -eq 0 ]; then
        TS=$(date +%Y%m%d_%H%M%S)
        echo "Creating checkpoint for iteration $i (Timestamp: $TS)..."
        cp model.safetensors "checkpoints/model_checkpoint_iter${i}_${TS}.safetensors"
    fi

    # Smart pruning: last 50 + every MILESTONE_EVERY-th + iter 1
    ALL_FILES=$(ls -t checkpoints/model_checkpoint_iter*.safetensors 2>/dev/null || true)
    if [ -n "$ALL_FILES" ]; then
        idx=0
        echo "$ALL_FILES" | while read -r FILE; do
            idx=$((idx + 1))
            ITER_VAL=$(echo "$FILE" | sed -n 's/.*iter\([0-9]\+\)_.*/\1/p')
            KEEP=false
            if [ $idx -le 50 ]; then KEEP=true
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
