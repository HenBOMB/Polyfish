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
# Open variable, NOT a known-good setting: audit A1 records no verdict for it
# anywhere, and the recommendation is to default it off only AFTER the A2b label
# fix lands, then run the arm both ways and write the verdict. Left on until
# then so the change is measured, not flipped blind. Registered as EXP_TRUNK_001
# in hypothesis_driven_improvements.md; run the off arm without editing this
# file: `DETACH_VALUE_TRUNK=0 ./run_training_loop.sh --new-run ...`.
export DETACH_VALUE_TRUNK="${DETACH_VALUE_TRUNK:-1}"
# 128 actors measured best on an M3 Max with metal (~578 moves/s @ 128 games+).
# Throughput scales with concurrent games; small NUM_GAMES (-g) is a real limiter, not this knob.
# See expert_boost_throughput.md for details.
ACTORS=128
# Seeds per gauge reading; each is played twice with sides swapped, so the
# reading's n is 2x this. Deliberately a declared number rather than an inline
# default: at 32 (64 games, p~0.33) a reading resolves to about +/-11pp, and
# calling the +8pp effect the registered experiments are written against needs
# ~571 games — `.venv/bin/python3 ladder.py power --baseline 0.33 --games 64`.
# It is left at 32 because raising it costs gauge wall-clock linearly and the
# plateau gate already pools eight readings; what changed is that the shortfall
# is now recorded on every reading (`resolves_pp`) and echoed in the log rather
# than being rediscovered from the interval later. See audit M3.
GAUGE_GAMES="${GAUGE_GAMES:-32}"
# Seeds for the link match that ties a newly frozen anchor to the outgoing one;
# it sets the Elo scale of the whole chain, so leave it at 64 for a real run.
# Declared as a knob only so the smoke can exercise the freeze branch, which no
# reading it can afford would otherwise ever reach (#35).
GAUGE_LINK_GAMES="${GAUGE_LINK_GAMES:-64}"
# Audit cross-checks (greedy + a rotating retired anchor, plus the training-pair
# row) run every this-many gauge readings; 0 disables them, as 0 does for -l.
GAUGE_AUDIT_EVERY="${GAUGE_AUDIT_EVERY:-5}"
# Auto-select 3 servers on the dedicated Metal backend and 1 on tch/candle.
# This preserves the measured Metal optimum without making CPU/Candle runs
# fail at startup. An explicit -e still overrides the automatic selection.
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
    PATH="$(pwd)/.venv/bin:$PATH" cargo build --bin polyfish --bin self_play --bin arena --release --features metal,accelerate,metal-eval,tch-eval
    # The tch-linked binary has no rpath for libtorch; point dyld at the venv's torch dylibs
    export DYLD_LIBRARY_PATH="$(.venv/bin/python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
elif command -v nvcc >/dev/null 2>&1; then
    # CUDA toolkit available; compile CUDA support into the binary.
    # Runtime selection will verify that a usable CUDA device exists.
    echo "Building with CUDA support..."
    echo "CUDA compiler: $(nvcc --version | tail -n1)"
    # --no-default-features: opt out of the macOS `metal` default, which does
    # not compile on Linux.
    cargo build --bin polyfish --bin self_play --bin arena --release --no-default-features --features cuda
else
    # CPU-only fallback
    echo "Building CPU-only version..."
    cargo build --bin polyfish --bin self_play --bin arena --release --no-default-features
fi

# Snapshot the binaries this run will use. Any concurrent `cargo build`/`cargo
# test` (e.g. a dev session) silently replaces target/release/* — possibly
# with different features — mid-run; executing from a private copy makes the
# run immune to that.
RUN_BIN_DIR=".run_bin"
mkdir -p "$RUN_BIN_DIR"
cp -f target/release/self_play target/release/arena target/release/polyfish "$RUN_BIN_DIR/"
SELF_PLAY_BIN="$RUN_BIN_DIR/self_play"
ARENA_BIN="$RUN_BIN_DIR/arena"
SERVER_BIN="$RUN_BIN_DIR/polyfish"

# Parse long options first, then short options via getopts
RESUME_RUN=""
NEW_RUN_EXPLICIT=false
RESET=false
START_SERVER=true
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
      NEW_RUN_EXPLICIT=true
      shift
      ;;
    --reset)
      RESET=true
      shift
      ;;
    --no-server)
      START_SERVER=false
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
GUMBEL_K=16
while getopts "fbcri:g:n:a:e:l:k:" opt; do
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
      MCTS_ITERS_SET=true
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
    k)
      GUMBEL_K=$OPTARG
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

# Minimal JSON helpers. Python is already required by the training pipeline,
# so the loop does not need the optional jq system package.
json_get() {
    local key="$1"
    local default_value="${2:-}"
    .venv/bin/python3 -c 'import json,sys; value=json.load(sys.stdin).get(sys.argv[1], sys.argv[2]); print(value if value is not None else sys.argv[2])' "$key" "$default_value"
}

json_array_values() {
    local key="$1"
    .venv/bin/python3 -c 'import json,sys; [print(value) for value in json.load(sys.stdin).get(sys.argv[1], []) if value is not None]' "$key"
}

json_array_items() {
    .venv/bin/python3 -c 'import json,sys; [print(json.dumps(value)) for value in json.load(sys.stdin)]'
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

if [ "$RESET" = true ]; then
    echo "🗑️  Reset flag detected! Deleting model.safetensors and self-play game data to seed a fresh model..."
    rm -f model.safetensors
    rm -f games_*.safetensors
    rm -f archive/games_*.safetensors
    # Adam moments + cosine position belong to the model they were fit on (#30);
    # keeping them would hand a from-scratch run a floored LR.
    rm -f optimizer_state.pt
    # EXP_ELO_002: the anchor decay clock belongs to the model it graduated.
    rm -f .anchor_decay_start
    if [ -n "$RESUME_RUN" ]; then
        echo "   (ignoring --resume since --reset always starts a fresh run)"
        RESUME_RUN=""
    fi
fi

# A bare launch is a NEW run, which keeps model.safetensors but rewinds every
# iteration-keyed mechanism to iteration 1: curriculum back to 10-turn Tiny
# maps, heuristic prior back to 0.5, value-trust to ~0, anchor-frac to 0.25.
# On a mature model that is ~30 iterations of degenerate data plus 10-turn
# gauge readings joining the ladder series, all looking like a fresh experiment
# in the CSV. Refuse to guess which was meant (#37).
if [ "$RESET" != true ] && [ "$NEW_RUN_EXPLICIT" != true ] && [ -z "$RESUME_RUN" ] \
   && [ -f model.safetensors ] && [ -f training_log.csv ] \
   && [ "$(wc -l < training_log.csv)" -gt 1 ]; then
    echo "" >&2
    echo "================================================================" >&2
    echo " REFUSING TO START: this would be a NEW run on an EXISTING model." >&2
    echo "" >&2
    echo " model.safetensors and training_log.csv history are both present," >&2
    echo " but no --resume was given. A new run rewinds the curriculum, the" >&2
    echo " heuristic prior, value-trust and anchor-frac to iteration 1 while" >&2
    echo " keeping the trained weights — ~30 iterations of degenerate data," >&2
    echo " and short-cap gauge readings joining the ladder series." >&2
    echo "" >&2
    echo " Continue the campaign:   ./run_training_loop.sh --resume [run_id]" >&2
    echo " Deliberately start over: ./run_training_loop.sh --new-run" >&2
    echo " From scratch (no model): ./run_training_loop.sh --reset" >&2
    echo "================================================================" >&2
    exit 1
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

# Run-scope the trainer (#30): train.py spans its cosine LR schedule and Adam
# moments across per-iteration invocations keyed by these. On resume the
# sidecar keeps the schedule position, so total = prior iterations + this launch.
export TRAIN_RUN_ID="$RUN_ID"
export TRAIN_TOTAL_ITERS=$((START_ITER - 1 + ITERATIONS))

if [ "$FORCE_TRAIN" = true ]; then
    echo "Force training flag detected! Running training immediately..."
    .venv/bin/python3 train.py
fi

# Restore point: snapshot the model at every launch (new run or resume), so
# no experiment can ever start without a recoverable "before" state.
mkdir -p checkpoints
if [ -f model.safetensors ]; then
    LAUNCH_CP="checkpoints/run_${RUN_ID}_iter${START_ITER}_start.safetensors"
    if [ ! -f "$LAUNCH_CP" ]; then
        cp model.safetensors "$LAUNCH_CP"
        echo "Launch checkpoint: $LAUNCH_CP"
    fi
fi

# Set up config.json sync if not present
if [ ! -f "config.json" ]; then
    echo "{\"gamemode\": 2, \"mctsIters\": $MCTS_ITERS, \"cores\": 2, \"tribes\": [\"Imperius\", \"Bardur\", \"Oumaji\", \"Kickoo\", \"XinXi\"]}" > config.json
fi

SERVER_PID=""
if [ "$START_SERVER" = true ]; then
    echo "Starting backend server in background..."
    "$SERVER_BIN" &
    SERVER_PID=$!
else
    echo "Using existing backend server."
fi

cleanup() {
    .venv/bin/python3 training_log.py finish-run 2>/dev/null || true
    if [ -n "${SERVER_PID:-}" ]; then
        kill "$SERVER_PID" 2>/dev/null || true
    fi
    rm -f .training.pid
}
trap cleanup EXIT

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
    # Restore only from THIS run. Taking the global latest by iteration silently
    # adopts another run's weights, which is worse than failing to resume.
    LATEST_CP=$(ls checkpoints/model_checkpoint_iter*_run${RUN_ID}_*.safetensors 2>/dev/null | sort -V | tail -n 1 || true)
    if [ -z "$LATEST_CP" ]; then
        LATEST_CP=$(ls checkpoints/run_${RUN_ID}_iter*_start.safetensors 2>/dev/null | sort -V | tail -n 1 || true)
    fi
    if [ -n "$LATEST_CP" ]; then
        echo "🔄 Resuming run $RUN_ID: restoring $(basename "$LATEST_CP") to model.safetensors"
        cp "$LATEST_CP" model.safetensors
    else
        UNTAGGED=$(ls checkpoints/model_checkpoint_iter*.safetensors 2>/dev/null | sort -V | tail -n 1 || true)
        if [ -n "$UNTAGGED" ]; then
            echo "ERROR: resuming run $RUN_ID at iteration $START_ITER but no checkpoint belongs to it." >&2
            echo "       Newest untagged checkpoint is $(basename "$UNTAGGED") (written before checkpoints" >&2
            echo "       carried a run id, or by a different run). Copy it to model.safetensors by hand if" >&2
            echo "       that is really the run you mean to continue." >&2
        else
            echo "ERROR: resuming run $RUN_ID at iteration $START_ITER with no model.safetensors and no checkpoint." >&2
        fi
        exit 1
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
    # asymmetric opponent). ANCHOR_FRAC is the STARTING rate of each
    # iteration's games played vs the network-free heuristic backend so
    # passivity actually loses and the relative value label carries signal;
    # it decays in-binary (see decay_crutch in self_play.rs) alongside
    # prior_heuristic_weight. ANCHOR_FRAC=0 disables.
    ANCHOR_FLAG=""
    if [ "$MATCH_TYPE" = "selfplay" ]; then
        ANCHOR_FLAG="--anchor-frac ${ANCHOR_FRAC:-0.25}"
    fi

    # Final phase-out of both heuristic crutches (search-prior blend +
    # anchor games): both decay to a 10% floor and hold there until this
    # EFF_ITER-relative cutoff, then hard-cut to 0. 150 is a starting point
    # (past curriculum maturity at iter 75 and the floor point at ~53, well
    # inside a default 500-iteration run) — validate/adjust via the
    # hypothesis-driven loop, not a measured value. Ideally this would
    # instead be gated on the model consistently beating the heuristic-only
    # backend (see EXP 10's arena ladder); until that gate is wired up,
    # DECAY_LAST_ITER is the fallback trigger.
    DECAY_LAST_ITER_FLAG="--decay-last-iter ${DECAY_LAST_ITER:-150}"

    # Value-head trust ramp, RUN-relative (loop iteration i, not EFF_ITER —
    # ITER_OFFSET-shifted runs would saturate the in-binary iteration ramp
    # immediately). Gates sigma(Q) in-tree and in exported policy targets.
    # VALUE_TRUST_CAP env caps the ramp's destination (e.g. from calibration).
    VALUE_TRUST=$(awk -v i="$i" -v r="${VALUE_TRUST_RAMP_ITERS:-30}" -v cap="${VALUE_TRUST_CAP:-1.0}" \
        'BEGIN { t = i / r; if (t > 1) t = 1; t = t * cap; printf "%.3f", t }')

    # Dynamically fetch parameters from config.json (set by dashboard UI).
    # -n on the command line is an explicit override and must survive the
    # whole run — it must not be re-clobbered by config.json each iteration.
    if [ -f "config.json" ]; then
        GAMEMODE=$(json_get gamemode 2 < config.json)
        if [ "${MCTS_ITERS_SET:-false}" != true ]; then
            MCTS_ITERS=$(json_get mctsIters 64 < config.json)
        fi
        # Parse tribes array into bash array safely
        TRIBE_LIST=()
        while IFS= read -r line; do
            if [ -n "$line" ]; then
                TRIBE_LIST+=("$line")
            fi
        done < <(json_array_values tribes < config.json)
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

    # EXP_ELO_002: hold anchor_frac at its starting rate until the model has
    # crossed 50% vs the greedy anchor. The gauge block persists the crossing
    # EFF_ITER into .anchor_decay_start; until then, passing the current
    # EFF_ITER keeps the anchor decay exponent at 0 (no decay, no cutover).
    if [ -n "$ANCHOR_FLAG" ]; then
        if [ -f .anchor_decay_start ]; then
            ANCHOR_DECAY_START=$(cat .anchor_decay_start)
        else
            ANCHOR_DECAY_START=$EFF_ITER
        fi
        ANCHOR_FLAG="$ANCHOR_FLAG --anchor-decay-start $ANCHOR_DECAY_START"
    fi

    SP_LOG=$(mktemp)
    "$SELF_PLAY_BIN" --num-games $NUM_GAMES --mcts-iters $MCTS_ITERS --gumbel-k $GUMBEL_K --actors $ACTORS --eval-servers $EVAL_SERVERS $REWARD_FLAG $OPPONENT_FLAG $ANCHOR_FLAG $DECAY_LAST_ITER_FLAG --value-trust "$VALUE_TRUST" --tribe1 "$TRIBE1" --tribe2 "$TRIBE2" --iteration "$EFF_ITER" --gamemode "$GAMEMODE" | tee "$SP_LOG"
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
    # Record the configuration this iteration actually ran at. config.json is
    # re-read inside this loop, so a dashboard edit shifts a run mid-flight;
    # without this the CSV cannot say which iterations ran under which settings
    # (audit M5). The tribe pair matters most: it is reshuffled every iteration
    # and its block effect on the behaviour metrics rivals the whole campaign's
    # measured improvement.
    ITER_CONFIG=$(TRIBE1="$TRIBE1" TRIBE2="$TRIBE2" MCTS_ITERS="$MCTS_ITERS" \
        GUMBEL_K="$GUMBEL_K" NUM_GAMES="$NUM_GAMES" GAMEMODE="$GAMEMODE" \
        ANCHOR_FRAC_EFF="$([ -n "$ANCHOR_FLAG" ] && echo "${ANCHOR_FRAC:-0.25}" || echo 0)" \
        VALUE_TRUST="$VALUE_TRUST" DETACH="${DETACH_VALUE_TRUNK:-}" \
        .venv/bin/python3 -c 'import json, os; print(json.dumps({
            "tribe1": os.environ["TRIBE1"], "tribe2": os.environ["TRIBE2"],
            "mcts_iters": os.environ["MCTS_ITERS"], "gumbel_k": os.environ["GUMBEL_K"],
            "num_games": os.environ["NUM_GAMES"], "gamemode": os.environ["GAMEMODE"],
            "anchor_frac": os.environ["ANCHOR_FRAC_EFF"], "value_trust": os.environ["VALUE_TRUST"],
            "detach_value_trunk": os.environ["DETACH"],
        }))')
    .venv/bin/python3 training_log.py append-row \
        --run-id "$RUN_ID" \
        --iter-started-at "$ITER_STARTED_AT" \
        --iteration "$i" \
        --games-file "$GAMES_FILE" \
        --game-json "$GAME_JSON" \
        --train-json "$TRAIN_JSON" \
        --config-json "$ITER_CONFIG" \
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
        cp model.safetensors "checkpoints/model_checkpoint_iter${i}_run${RUN_ID}_${TS}.safetensors"
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
            # [0-9][0-9]* not [0-9]\+: BSD sed's BRE has no \+, so on the macOS
            # training box ITER_VAL parsed empty and every milestone past the
            # newest 50 was pruned (#37).
            ITER_VAL=$(echo "$FILE" | sed -n 's/.*iter\([0-9][0-9]*\)_.*/\1/p')
            
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

    # 5. Cleanup (Fresh Games Only)
    # Move played games to archive so train.py only sees new ones next time.
    # Before the gauge, not after: a failed reading aborts the run (below), and
    # a trained-on file left in root is re-trained as fresh next launch (#37).
    mkdir -p archive
    # Use || true to avoid script exit if no games were generated
    mv games_*.safetensors archive/ 2>/dev/null || true
    
    # Keep only ARCHIVE_KEEP game files — a constant ~10*BASELINE_GAMES-game
    # replay window regardless of -g (train.py reads the same value via
    # REPLAY_BUFFER_FILES)
    ls -t archive/games_*.safetensors 2>/dev/null | tail -n +$((ARCHIVE_KEEP + 1)) | xargs -r rm

    # 6. Strength gauge (EXP 10/11): paired arena reading vs the ladder's
    # active anchor. ladder.py owns ladder.json (anchors, readings, verdicts):
    # >=80% freezes the model as the next anchor (n=64 link match); two
    # consecutive 8-reading windows that are flat-or-down with slope <= 0 stop
    # the run (plateau, EXP 11's registered rule).
    if [ "$LEAGUE_INTERVAL" -gt 0 ] && [ $((i % LEAGUE_INTERVAL)) -eq 0 ]; then
        ACTIVE_JSON=$(.venv/bin/python3 ladder.py active)
        ANCHOR_PATH=$(json_get path "" <<< "$ACTIVE_JSON")
        ANCHOR_NAME=$(json_get name "" <<< "$ACTIVE_JSON")
        GAUGE_LOG=$(mktemp)

        # M4/#32: match what self-play is currently generating AND the searcher
        # it is generating it with. self_play's schedules are the source of
        # truth; ask it rather than mirroring them here. --decay-last-iter and
        # --value-trust are the same flags the generation call above passes, so
        # the reported knobs are the ones self-play actually searched with.
        GAUGE_CURRICULUM=$("$SELF_PLAY_BIN" --print-curriculum --iteration "$EFF_ITER" \
            $DECAY_LAST_ITER_FLAG --value-trust "$VALUE_TRUST")
        GAUGE_MAX_TURNS=$(json_get max_turns 30 <<< "$GAUGE_CURRICULUM")
        GAUGE_PRIOR_W=$(json_get prior_heuristic_w 0.1 <<< "$GAUGE_CURRICULUM")
        GAUGE_Q_W=$(json_get policy_target_q_w 1.0 <<< "$GAUGE_CURRICULUM")

        # The gauge's tribe pair is pinned, unlike self-play's: the tribe block
        # effect rivals a whole campaign's measured improvement, so a reshuffled
        # pair would move readings for a reason that is not strength. That fixes
        # the instrument's scope to an Imperius mirror while training optimizes
        # the 5-tribe pool -- recorded on every reading and in ladder.json's
        # `scope` note, not assumed (#34).
        GAUGE_TRIBE1="${GAUGE_TRIBE1:-Imperius}"
        GAUGE_TRIBE2="${GAUGE_TRIBE2:-Imperius}"

        # $1 = opponent model path ("" = greedy backend), $2 = seeds (games x2),
        # $3 = per-turn stats dump dir (optional; summarized into the reading),
        # $4/$5 = tribe pair (optional; defaults to the pinned gauge pair).
        # Returns non-zero if arena failed OR its output did not parse — callers
        # must not swallow that (a swallowed error meant the ladder recorded no
        # readings at all for the whole campaign).
        run_gauge_match () {
            GAUGE_STATS_DIR="$3"
            local tribe1="${4:-$GAUGE_TRIBE1}" tribe2="${5:-$GAUGE_TRIBE2}"
            # Fixed seed set: arena otherwise seeds from the wall clock, so
            # readings would not share a map set and could not be compared
            # across iterations.
            local -a cmd=("$ARENA_BIN" --model1 model.safetensors
                --mcts "$MCTS_ITERS" --gumbel-k "$GUMBEL_K"
                --games "$2" --gamemode "$GAMEMODE"
                --max-turns "$GAUGE_MAX_TURNS" --seed "${GAUGE_SEED:-20260811}"
                --symmetric "${GAUGE_SYMMETRIC:-true}"
                --tribe1 "$tribe1" --tribe2 "$tribe2"
                --prior-heuristic-weight "$GAUGE_PRIOR_W"
                --policy-target-q-weight "$GAUGE_Q_W"
                --tree-q-weight "$GAUGE_Q_W")
            if [ -z "$1" ]; then
                cmd+=(--model2 model.safetensors --backend1 gumbel --backend2 greedy)
            else
                cmd+=(--model2 "$1" --backend1 gumbel --backend2 gumbel)
            fi
            if [ -n "$GAUGE_STATS_DIR" ]; then
                cmd+=(--dump-stats-dir "$GAUGE_STATS_DIR")
            fi
            "${cmd[@]}" | tee "$GAUGE_LOG"
            local arena_status=${PIPESTATUS[0]}
            if [ "$arena_status" -ne 0 ]; then
                echo "GAUGE: arena exited $arena_status (opponent '${1:-greedy}', $2 seeds)" >&2
                return "$arena_status"
            fi
            GAUGE_W=$(sed -n 's/^Config 1 Wins: \([0-9][0-9]*\).*/\1/p' "$GAUGE_LOG")
            GAUGE_L=$(sed -n 's/^Config 2 Wins: \([0-9][0-9]*\).*/\1/p' "$GAUGE_LOG")
            GAUGE_D=$(sed -n 's/^Draws: *\([0-9][0-9]*\).*/\1/p' "$GAUGE_LOG")
            GAUGE_S1=$(sed -n 's/^Avg Score Config 1: \([0-9.][0-9.]*\).*/\1/p' "$GAUGE_LOG")
            GAUGE_S2=$(sed -n 's/^Avg Score Config 2: \([0-9.][0-9.]*\).*/\1/p' "$GAUGE_LOG")
            GAUGE_WP1=$(sed -n 's/^Config 1 Wins as P1: \([0-9][0-9]*\).*/\1/p' "$GAUGE_LOG")
            GAUGE_WP2=$(sed -n 's/^Config 1 Wins as P2: \([0-9][0-9]*\).*/\1/p' "$GAUGE_LOG")
            GAUGE_BACKEND=$(sed -n 's/.*| eval \([a-z]*\).*/\1/p' "$GAUGE_LOG" | head -1)
            # From arena's own output, never from self-play's shuffled pair: the
            # ladder used to be handed the training tribes for a match arena had
            # hardcoded to an Imperius mirror, so the permanent record carried
            # metadata about a variable the gauge never varied (#34).
            GAUGE_TRIBES=$(sed -n 's/^Tribes: \(.*\)$/\1/p' "$GAUGE_LOG" | head -1)
            # A panicked game is dropped by arena, which silently shrinks n and
            # unbalances the side-swap pairing the seeded design depends on.
            # Carry both into the reading instead of recording a clean-looking
            # count (audit M5).
            GAUGE_ATTEMPTED=$(sed -n 's/^Total Games: [0-9]* completed \/ \([0-9][0-9]*\) attempted.*/\1/p' "$GAUGE_LOG")
            GAUGE_DROPPED=$(sed -n 's/^Total Games:.*, \([0-9][0-9]*\) seed(s) dropped.*/\1/p' "$GAUGE_LOG")
            GAUGE_UNPAIRED=$(sed -n 's/^Unpaired Seeds: \([0-9][0-9]*\).*/\1/p' "$GAUGE_LOG")
            if [ -z "$GAUGE_W" ] || [ -z "$GAUGE_L" ] || [ -z "$GAUGE_TRIBES" ]; then
                echo "GAUGE: arena exited 0 but its win counts or tribe pair did not parse (opponent '${1:-greedy}') — output format changed?" >&2
                return 1
            fi
        }

        # A failed reading is fatal: the ladder is the instrument every
        # experiment is judged on, and continuing past a broken gauge is what
        # left the whole campaign without a single recorded reading.
        if ! run_gauge_match "$ANCHOR_PATH" "$GAUGE_GAMES" "replays/gauge_stats/${RUN_ID}_iter${i}"; then
            rm -f "$GAUGE_LOG"
            echo "GAUGE: strength reading failed at iteration $i — aborting instead of continuing blind." >&2
            exit 1
        fi
        VERDICT=$(.venv/bin/python3 ladder.py record --kind gauge \
            --run-id "$RUN_ID" --iteration "$i" \
            --wins "$GAUGE_W" --losses "$GAUGE_L" --draws "${GAUGE_D:-0}" \
            --avg-score-model "${GAUGE_S1:-0}" --avg-score-opponent "${GAUGE_S2:-0}" \
            --mcts "$MCTS_ITERS" --gumbel-k "$GUMBEL_K" --eval-backend "${GAUGE_BACKEND:-}" \
            --wins-p1 "${GAUGE_WP1:-0}" --wins-p2 "${GAUGE_WP2:-0}" \
            --games-attempted "${GAUGE_ATTEMPTED:-0}" --games-dropped "${GAUGE_DROPPED:-0}" \
            --unpaired-seeds "${GAUGE_UNPAIRED:-0}" \
            --tribes "$GAUGE_TRIBES" \
            --max-turns "$GAUGE_MAX_TURNS" \
            --prior-heuristic-w "$GAUGE_PRIOR_W" --q-weight "$GAUGE_Q_W" \
            --stats-dir "$GAUGE_STATS_DIR")
        echo "GAUGE: $VERDICT"
        GAUGE_ACTION=$(json_get action "" <<< "$VERDICT")

        # Audit M3: a reading cannot adjudicate a difference smaller than its
        # own resolution. ladder.py sets these when it cannot, so the log says
        # so at the time the number is taken rather than leaving the next reader
        # to work it out from the interval months later.
        GAUGE_UNDERPOWERED=$(json_get underpowered_for_pp "" <<< "$VERDICT")
        if [ -n "$GAUGE_UNDERPOWERED" ]; then
            echo "GAUGE: this reading resolves to +/-$(json_get resolves_pp "?" <<< "$VERDICT")pp;" \
                 "calling a ${GAUGE_UNDERPOWERED}pp effect needs ~$(json_get games_needed "?" <<< "$VERDICT")" \
                 "games (this reading: $(( GAUGE_GAMES * 2 ))). Trend across readings, not one reading."
        fi

        # EXP_ELO_002: first >=50% reading vs the greedy anchor starts
        # the anchor-frac decay clock (EFF_ITER units, matching
        # --anchor-decay-start above).
        if [ ! -f .anchor_decay_start ] && [ "$ANCHOR_NAME" = "greedy" ]; then
            CROSSED=$(awk -v w="$GAUGE_W" -v l="$GAUGE_L" -v d="${GAUGE_D:-0}" \
                'BEGIN { n = w + l + d; if (n > 0 && (w + d / 2) / n >= 0.5) print 1; else print 0 }')
            if [ "$CROSSED" = "1" ]; then
                echo "$EFF_ITER" > .anchor_decay_start
                echo "GAUGE: crossed 50% vs greedy at EFF_ITER $EFF_ITER — anchor-frac decay clock started (EXP_ELO_002)"
            fi
        fi

        if [ "$GAUGE_ACTION" = "freeze" ]; then
            TS=$(date +%Y%m%d_%H%M%S)
            NEW_ANCHOR="checkpoints/anchor_iter${i}_${TS}.safetensors"
            if ! cp model.safetensors "$NEW_ANCHOR"; then
                rm -f "$GAUGE_LOG"
                echo "GAUGE: failed to snapshot $NEW_ANCHOR — aborting instead of skipping the anchor freeze." >&2
                exit 1
            fi
            echo "GAUGE: cleared the freeze bar vs active anchor — freezing $NEW_ANCHOR, link match (n=$((GAUGE_LINK_GAMES * 2)))..."
            if ! run_gauge_match "$ANCHOR_PATH" "$GAUGE_LINK_GAMES" "replays/gauge_stats/${RUN_ID}_iter${i}_link"; then
                rm -f "$GAUGE_LOG"
                echo "GAUGE: link match failed at iteration $i — aborting instead of freezing an unlinked anchor." >&2
                exit 1
            fi
            .venv/bin/python3 ladder.py freeze --run-id "$RUN_ID" --iteration "$i" \
                --path "$NEW_ANCHOR" \
                --wins "${GAUGE_W:-0}" --losses "${GAUGE_L:-0}" --draws "${GAUGE_D:-0}" \
                --avg-score-model "${GAUGE_S1:-0}" --avg-score-opponent "${GAUGE_S2:-0}" \
                --tribes "$GAUGE_TRIBES"
        elif [ "$GAUGE_ACTION" = "stop" ]; then
            rm -f "$GAUGE_LOG"
            echo "=================================================="
            echo "PLATEAU STOP at iteration $i: two consecutive 8-reading"
            echo "windows flat-or-down with slope <= 0 vs the active anchor"
            echo "(see ladder.json)."
            echo "=================================================="
            break
        fi

        # Audit block every GAUGE_AUDIT_EVERY-th gauge: greedy + one retired
        # anchor, rotating — observed vs chain-predicted win rate flags cycles.
        if [ "$GAUGE_AUDIT_EVERY" -gt 0 ] && [ $((i % (LEAGUE_INTERVAL * GAUGE_AUDIT_EVERY))) -eq 0 ]; then
            while read -r AUD; do
                AUD_NAME=$(json_get name "" <<< "$AUD")
                AUD_PATH=$(json_get path "" <<< "$AUD")
                # Cross-check rows, not the reading the run steers on: report a
                # failure and keep going rather than aborting the whole run.
                if ! run_gauge_match "$AUD_PATH" "$GAUGE_GAMES" "replays/gauge_stats/${RUN_ID}_iter${i}_audit_${AUD_NAME}"; then
                    echo "GAUGE: audit match vs $AUD_NAME failed — audit row skipped (non-fatal)" >&2
                    continue
                fi
                .venv/bin/python3 ladder.py record --kind audit --opponent "$AUD_NAME" \
                    --run-id "$RUN_ID" --iteration "$i" \
                    --wins "$GAUGE_W" --losses "$GAUGE_L" --draws "${GAUGE_D:-0}" \
                    --avg-score-model "${GAUGE_S1:-0}" --avg-score-opponent "${GAUGE_S2:-0}" \
                    --mcts "$MCTS_ITERS" --gumbel-k "$GUMBEL_K" --eval-backend "${GAUGE_BACKEND:-}" \
                    --wins-p1 "${GAUGE_WP1:-0}" --wins-p2 "${GAUGE_WP2:-0}" \
                    --tribes "$GAUGE_TRIBES" \
                    --prior-heuristic-w "$GAUGE_PRIOR_W" --q-weight "$GAUGE_Q_W" \
                    --stats-dir "$GAUGE_STATS_DIR"
            done < <(.venv/bin/python3 ladder.py audit-opponents | json_array_items)

            # Same match, same anchor, this iteration's *training* pair: the one
            # row that says whether the pinned Imperius reading generalizes to
            # the pool training actually optimizes. Cross-check only — recorded
            # under its own kind so it never enters the gauge's window (#34).
            if [ "$GAUGE_TRIBE1,$GAUGE_TRIBE2" != "$TRIBE1,$TRIBE2" ]; then
                if run_gauge_match "$ANCHOR_PATH" "$GAUGE_GAMES" \
                        "replays/gauge_stats/${RUN_ID}_iter${i}_tribes" "$TRIBE1" "$TRIBE2"; then
                    # --opponent, not the active anchor: a freeze earlier in
                    # this same iteration has already retired the anchor this
                    # match was actually played against.
                    .venv/bin/python3 ladder.py record --kind tribe_audit \
                        --opponent "$ANCHOR_NAME" \
                        --run-id "$RUN_ID" --iteration "$i" \
                        --wins "$GAUGE_W" --losses "$GAUGE_L" --draws "${GAUGE_D:-0}" \
                        --avg-score-model "${GAUGE_S1:-0}" --avg-score-opponent "${GAUGE_S2:-0}" \
                        --mcts "$MCTS_ITERS" --gumbel-k "$GUMBEL_K" --eval-backend "${GAUGE_BACKEND:-}" \
                        --wins-p1 "${GAUGE_WP1:-0}" --wins-p2 "${GAUGE_WP2:-0}" \
                        --tribes "$GAUGE_TRIBES" --max-turns "$GAUGE_MAX_TURNS" \
                        --prior-heuristic-w "$GAUGE_PRIOR_W" --q-weight "$GAUGE_Q_W" \
                        --stats-dir "$GAUGE_STATS_DIR"
                else
                    echo "GAUGE: per-tribe audit ($TRIBE1,$TRIBE2) failed — row skipped (non-fatal)" >&2
                fi
            fi
        fi
        rm -f "$GAUGE_LOG"
    fi

done
