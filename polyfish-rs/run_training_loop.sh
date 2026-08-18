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
export DETACH_VALUE_TRUNK=1
# 128 actors measured best on an M3 Max with metal (~578 moves/s @ 128 games+).
# Throughput scales with concurrent games; small NUM_GAMES (-g) is a real limiter, not this knob.
# See expert_boost_throughput.md for details.
ACTORS=128
# 3 servers × 2 workers measured best on metal after buffer pooling
# (~610-650 moves/s — see expert_boost_throughput.md). 0 = auto (3 on metal,
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
    PATH="$(pwd)/.venv/bin:$PATH" cargo build --bin polyfish --bin self_play --bin arena --release --features apple
    # The tch-linked binary has no rpath for libtorch; point dyld at the venv's torch dylibs
    export DYLD_LIBRARY_PATH="$(.venv/bin/python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
elif command -v nvidia-smi >/dev/null 2>&1; then
    # CUDA available (Linux/Windows with NVIDIA GPU)
    # `cudnn` matters as much as `cuda`: without it candle uses its own conv
    # kernels, which is exactly what made the Metal backend ~19x slower than
    # PyTorch for PolyZeroNet's conv shapes. Set POLYFISH_NO_CUDNN=1 to drop it
    # if the box's cuDNN is missing/mismatched and the build fails.
    CUDA_FEATURES="cuda,cudnn"
    if [ -n "${POLYFISH_NO_CUDNN:-}" ]; then CUDA_FEATURES="cuda"; fi
    echo "Building with CUDA support (features: $CUDA_FEATURES)..."
    # --no-default-features: opt out of the macOS `metal` default, which does
    # not compile on Linux.
    cargo build --bin polyfish --bin self_play --bin arena --release --no-default-features --features "$CUDA_FEATURES"
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
# On by default since EXP_ELO_004 (Jul 13, 2026): a clean matched-baseline
# comparison showed reward-shaping ON (TD(lambda) + final-outcome blend)
# clearly beats the flat final-outcome-only fallback at matched budget (best
# 16-sim vs-Greedy reading to date, 48.4%, vs 35.9% with shaping off). -r now
# OPTS OUT (flips from its pre-Jul-13 meaning of opting in) — pass it to
# reproduce old runs or isolate a regression.
REWARD_SHAPING=true
# Play a league match every LEAGUE_INTERVAL iterations (iteration 10, 20, 30,
# ... by default). 0 disables league play entirely. Override with -l.
LEAGUE_INTERVAL=10
# Strength gauge cadence, DECOUPLED from the league cadence (it used to share
# -l, so reading more often also changed the data diet). Deterministic gauging
# is ~2.5x cheaper per move, so every 5 iterations costs about what the old
# every-10 noisy gauge did. 0 falls back to LEAGUE_INTERVAL.
# NB: ladder.py's plateau rule counts READINGS, not iterations, so halving this
# also halves the iteration span of its 8-reading windows.
GAUGE_INTERVAL="${GAUGE_INTERVAL:-5}"
# Exported (like MCTS_ITERS) so arena inherits the run's search budget rather
# than carrying its own default — a reading at a different (mcts, k) is not
# comparable to the ladder.
export GUMBEL_K=16
while getopts "fbcri:g:n:a:e:l:k:p:" opt; do
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
      REWARD_SHAPING=false
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
    p)
      SELF_PLAY_LOOPS=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

# Derive iteration-keyed schedules from -g so the regime is constant in
# GAMES: scaled(x) = max(1, round(x * BASELINE_GAMES / NUM_GAMES)).
# -p runs self_play SELF_PLAY_LOOPS times per iteration (memory stays bounded
# at -g scale per pass); schedules key on the EFFECTIVE games an iteration
# produces.
SELF_PLAY_LOOPS="${SELF_PLAY_LOOPS:-1}"
EFFECTIVE_GAMES=$((NUM_GAMES * SELF_PLAY_LOOPS))
scaled() {
    awk -v x="$1" -v b="$BASELINE_GAMES" -v g="$EFFECTIVE_GAMES" \
        'BEGIN { v = int(x * b / g + 0.5); print (v < 1 ? 1 : v) }'
}
if [ "${ITERATIONS_SET:-false}" != true ]; then
    ITERATIONS=$(scaled "$ITERATIONS")
fi
CHECKPOINT_EVERY=$(scaled 50)
MILESTONE_EVERY=$(scaled 100)
# Replay window: constant ~10*BASELINE_GAMES games regardless of -g.
# train.py reads REPLAY_BUFFER_FILES; archive pruning keeps window + 1 in sync.
# self_play shards at 64 games/file (Jul 28), so every file is baseline-sized
# and a fixed file count IS a fixed game window (10 files = 640 games).
ARCHIVE_KEEP="${ARCHIVE_KEEP:-10}"
export REPLAY_BUFFER_FILES=$ARCHIVE_KEEP
echo "Schedule (games-based, -g $NUM_GAMES vs baseline $BASELINE_GAMES): $ITERATIONS iterations, checkpoint every $CHECKPOINT_EVERY, milestone every $MILESTONE_EVERY, league every $LEAGUE_INTERVAL iterations, gauge every ${GAUGE_INTERVAL:-$LEAGUE_INTERVAL} (scale ${GAUGE_GUMBEL_SCALE:-0}, ${GAUGE_GAMES:-32}x2 games), replay window $ARCHIVE_KEEP files"

REWARD_FLAG=""
if [ "$REWARD_SHAPING" = true ]; then
    echo "🎯 Reward shaping enabled (default)."
else
    REWARD_FLAG="--no-reward-shaping"
    echo "⚠️  Reward shaping disabled (-r): flat final-outcome value target only."
fi

# EXP_ELO_025 outcome-space labels (±1 z flat arm + outcome-space TD/q-target
# arm). Default OFF — the experiment was ABANDONED unrun (Jul 28, Verdi:
# shaped labels preferred per EXP_ELO_004); production keeps the legacy
# reward-shaped score labels. WL_LABELS=1 opts in for comparison runs.
WL_FLAG=""
if [ "${WL_LABELS:-0}" = "1" ]; then
    WL_FLAG="--wl-labels"
    echo "⚖️  WL_LABELS=1: win/loss value labels (z=±1 flat arm + outcome-space TD/q-target arm)."
fi

# EXP_ELO_028 Phase 1: GOAL_CHANNELS=1 drives the appended goal channels with
# the Stage-1 scripted goal-setter — self_play net seats get painted orders +
# stance + star gate, and gauges/league probe the model WITH the script on so
# the deployed policy is measured under the same conditioning it trains with.
GOAL_FLAG=""
GOAL_ARENA_FLAG=""
if [ "${GOAL_CHANNELS:-0}" = "1" ]; then
    # Phase 1c: GOAL_W_TREE prices the painted goal in the in-tree rewards
    # (GROW→SPT, ARM→army value, EXPAND→approach progress). Default ON at the
    # designed magnitude; GOAL_W_TREE=0 reverts to channels-only conditioning.
    GOAL_FLAG="--goal-channels --goal-w-tree ${GOAL_W_TREE:-1}"
    GOAL_ARENA_FLAG="--goal-script --goal-w-tree ${GOAL_W_TREE:-1}"
    echo "🎯 GOAL_CHANNELS=1 (EXP_ELO_028): scripted orders/stance drive the goal channels (goal_w_tree=${GOAL_W_TREE:-1})."
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
    rm -f optimizer_state.pt
    rm -f games_*.safetensors
    rm -f archive/games_*.safetensors
    # EXP_ELO_002: the anchor decay clock belongs to the model it graduated.
    rm -f .anchor_decay_start
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

# Restore point: snapshot the model at every launch (new run or resume), so
# no experiment can ever start without a recoverable "before" state.
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

echo "Starting backend server in background..."
"$SERVER_BIN" &
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

# Keep only checkpoints whose conv1 input-channel count matches the current
# model.safetensors. The Rust opponent loader (self_play) is strict — it does
# NOT zero-pad like train.py — so a stale-arch checkpoint would hard-crash a
# league iteration. Reads candidate paths on stdin (one per line), prints the
# arch-compatible subset. Migrate stale checkpoints with migrate_pursuit.py.
compatible_checkpoints() {
    .venv/bin/python3 -c '
import sys, json, struct
def conv1_in(path):
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        return hdr["conv1.weight"]["shape"][1]
    except Exception:
        return None
ref = conv1_in("model.safetensors")
for line in sys.stdin:
    p = line.strip()
    if p and (ref is None or conv1_in(p) == ref):
        print(p)
'
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

    # EXP_ELO_007: BOOTSTRAP=1 skips league matches — every training iteration
    # is pure Greedy demonstration data; the net is only measured by the gauge.
    if [ -z "${BOOTSTRAP:-}" ] && [ "$LEAGUE_INTERVAL" -gt 0 ] && [ $((i % LEAGUE_INTERVAL)) -eq 0 ] && [ -d "checkpoints" ] && [ "$(ls -A checkpoints)" ]; then
        # HISTORICAL-ONLY league selection: the latest checkpoint is ~the
        # current net, so playing it is mirror play with extra steps and
        # breaks no symmetry. Prefer genuinely old checkpoints; fall back to
        # anything that isn't the newest one.
        ALL_CPS_RAW=$(ls -t checkpoints/model_checkpoint_iter*.safetensors 2>/dev/null || true)
        # Drop arch-incompatible checkpoints so a league iteration degrades to
        # plain self-play instead of hard-crashing self_play's strict loader.
        ALL_CPS=$(echo "$ALL_CPS_RAW" | compatible_checkpoints)
        RAW_N=$(echo "$ALL_CPS_RAW" | grep -c . || true)
        CMP_N=$(echo "$ALL_CPS" | grep -c . || true)
        if [ "$RAW_N" -gt "$CMP_N" ]; then
            echo "⚠️  League: skipped $((RAW_N - CMP_N)) arch-incompatible checkpoint(s) (conv1 channels != current model.safetensors); migrate them with migrate_pursuit.py"
        fi
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
    # Under MACRO_GEN the anchor's stated rationale is void: Greedy anchors
    # exist because Greedy is exactly what blend_heuristic_prior injects into
    # the net's root — and that blending is Gumbel-only. Anchor games there
    # are just macro-vs-Greedy blowouts diluting the cloning data, so the
    # default drops to 0 (override with ANCHOR_FRAC to opt back in).
    ANCHOR_FLAG=""
    if [ "$MATCH_TYPE" = "selfplay" ] && [ -z "${BOOTSTRAP:-}" ]; then
        if [ -n "${MACRO_GEN:-}" ]; then
            ANCHOR_FLAG="--anchor-frac ${ANCHOR_FRAC:-0}"
        else
            ANCHOR_FLAG="--anchor-frac ${ANCHOR_FRAC:-0.25}"
        fi
    fi

    # EXP_ELO_007: bootstrap mode — every game is Greedy-vs-Greedy demonstration
    # data (soft policy targets + TD labels recorded for BOTH seats); anchor
    # games are redundant while cloning, so ANCHOR_FLAG stays empty above.
    # Stage 3 (Aug 2026): MACRO_GEN=1 — macro-mcts (the repo's strongest
    # agent, NN-free) generates the games instead: one-hot behavior-cloning
    # policy targets for the executed ply + outcome/TD value labels, with the
    # tree's COMMITTED directive painted into the goal channels (not the
    # script's — they disagree on ~40-55% of turns). The trained value head
    # is the Stage-3 macro-leaf candidate; gate it with the registered paired
    # A/B (net leaf vs heuristic leaf) before it touches the tree.
    BACKEND_FLAG=""
    if [ -n "${MACRO_GEN:-}" ]; then
        # MACRO_LEAF=net makes the tree consult the network at its leaves (and
        # supplies a turn-level root value for the TD bootstrap). Until these
        # flags existed the backend silently ran MacroParams::default(), i.e.
        # the heuristic leaf, no matter what the run intended.
        # sims=64/k=6 (Aug 18, Verdi): EXP_ELO_033b found sims=64 alone at
        # k=4 didn't help (depth saturated at 32 with only ~4 live
        # candidates); raising k too means more candidates competing for the
        # same sim budget, which is a different regime -- see EXP_ELO_061.
        BACKEND_FLAG="--search-backend macro-mcts --macro-leaf ${MACRO_LEAF:-heuristic} --macro-sims ${MACRO_SIMS:-64} --macro-k ${MACRO_K:-6}"
        echo "🌲 MACRO_GEN=1 (Stage 3): macro-mcts generates self-play games (behavior cloning + on-distribution value labels), leaf=${MACRO_LEAF:-heuristic} sims=${MACRO_SIMS:-64} k=${MACRO_K:-6}."
    elif [ -n "${BOOTSTRAP:-}" ]; then
        BACKEND_FLAG="--search-backend greedy"
    fi

    # A heuristic-leaf macro run reports no root value, so every n-step return
    # bootstraps with 0.0 and the labels are systematically truncated. `mc`
    # carries that weight to the terminal return instead; under a net leaf the
    # root value exists and the flag never fires, so the default is safe there.
    if [ -n "${TD_MISSING_BOOTSTRAP:-}" ]; then
        TD_MISSING="$TD_MISSING_BOOTSTRAP"
    elif [ -n "${MACRO_GEN:-}" ]; then
        TD_MISSING="mc"
    else
        TD_MISSING="zero"
    fi
    TD_MISSING_FLAG="--td-missing-bootstrap $TD_MISSING"

    # EXP_ELO_006: label-only relative weight for TD labels; unset = binary
    # default (reward::REL_W), in-tree backup unaffected either way.
    LABEL_REL_W_FLAG=""
    if [ -n "${LABEL_REL_W:-}" ]; then
        LABEL_REL_W_FLAG="--label-rel-w $LABEL_REL_W"
    fi

    # EXP_ELO_016: development-potential shaping weights — label snapshots
    # (SHAPE_W_LABEL) and the in-tree Gumbel backup (SHAPE_W_TREE) are
    # threaded separately. Unset = 0 = raw score deltas (legacy).
    SHAPE_FLAGS=""
    if [ -n "${SHAPE_W_LABEL:-}" ]; then
        SHAPE_FLAGS="--shape-w-label $SHAPE_W_LABEL"
    fi
    if [ -n "${SHAPE_W_TREE:-}" ]; then
        SHAPE_FLAGS="$SHAPE_FLAGS --shape-w-tree $SHAPE_W_TREE"
    fi

    # EXP_ELO_018: isolated pursuit-progress shaping weights — label
    # (PURSUIT_W_LABEL) and in-tree backup (PURSUIT_W_TREE), independent of
    # the SHAPE_W_* development weights. Unset = 0 = off.
    if [ -n "${PURSUIT_W_LABEL:-}" ]; then
        SHAPE_FLAGS="$SHAPE_FLAGS --pursuit-w-label $PURSUIT_W_LABEL"
    fi
    if [ -n "${PURSUIT_W_TREE:-}" ]; then
        SHAPE_FLAGS="$SHAPE_FLAGS --pursuit-w-tree $PURSUIT_W_TREE"
    fi

    # EXP_ELO_020: DAgger expert dose — blend Greedy's ranking at the model's
    # own states into the policy target. Unset = 0 = off.
    if [ -n "${DAGGER_ALPHA:-}" ]; then
        SHAPE_FLAGS="$SHAPE_FLAGS --dagger-alpha $DAGGER_ALPHA"
    fi

    # EXP_ELO_017: unfreeze the opponent during in-tree EndTurn crossings —
    # training-data generation only. Gauges/arena always search frozen (see
    # arena.rs's hardcoded Some(false)), so no prior strength reading is
    # ever confounded by this flag. UNFREEZE_OPPONENT=1 to enable.
    UNFREEZE_FLAG=""
    if [ "${UNFREEZE_OPPONENT:-0}" = "1" ]; then
        UNFREEZE_FLAG="--unfreeze-opponent"
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

    # beta on sigma(Q), in-tree and in the exported policy targets. sigma's own
    # scale grows with visits, so this is the dial that makes a bigger --mcts-iters
    # pay off; damping it cancels the budget (EXP_ELO_023 ladder was measured at
    # 1.0). Ramp is RUN-relative (loop iteration i, not EFF_ITER), so on a resumed
    # or ITER_OFFSET-shifted run it re-imposes a warmup the model already earned —
    # hence default off. Set VALUE_TRUST_RAMP_ITERS=30 for a from-scratch model.
    # VALUE_TRUST_CAP env caps the ramp's destination (e.g. from calibration).
    VALUE_TRUST=$(awk -v i="$i" -v r="${VALUE_TRUST_RAMP_ITERS:-1}" -v cap="${VALUE_TRUST_CAP:-1.0}" \
        'BEGIN { t = i / r; if (t > 1) t = 1; t = t * cap; printf "%.3f", t }')

    # Dynamically fetch parameters from config.json (set by dashboard UI).
    # -n on the command line is an explicit override and must survive the
    # whole run — it must not be re-clobbered by config.json each iteration.
    if [ -f "config.json" ]; then
        GAMEMODE=$(jq -r '.gamemode // 2' config.json)
        if [ "${MCTS_ITERS_SET:-false}" != true ]; then
            MCTS_ITERS=$(jq -r '.mctsIters // 64' config.json)
        fi
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
    
    # EFF_ITER pacing is keyed to GAMES seen, not loop count, and still drives
    # the heuristic-prior/anchor-frac decay ramps (self_play's turn-count
    # curriculum was dropped — max_turns is now a flat 50, see self_play.rs).
    # ITER_OFFSET (env, default 0) shifts the schedule forward — e.g.
    # ITER_OFFSET=76 starts with the heuristic prior mostly annealed, for
    # resuming from a behavior-cloned model.
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

    # One-line config echo so silent env misconfigurations (ITER_OFFSET,
    # LABEL_REL_W, BOOTSTRAP) are visible in the log — EXP_ELO_006 post-mortem:
    # two runs voided by a missing ITER_OFFSET that nothing surfaced.
    echo "CONFIG iter=$i eff_iter=$EFF_ITER iter_offset=${ITER_OFFSET:-0} match=$MATCH_TYPE backend=${BACKEND_FLAG:---search-backend gumbel} anchor='${ANCHOR_FLAG}' td_w=${TD_W:-0.7} td_lambda=${TD_LAMBDA:-0.8} label_rel_w=${LABEL_REL_W:-default} wl_labels=${WL_LABELS:-0} td_missing=${TD_MISSING} goal_channels=${GOAL_CHANNELS:-0} goal_w_tree=${GOAL_W_TREE:-1} shape_w_label=${SHAPE_W_LABEL:-0} shape_w_tree=${SHAPE_W_TREE:-0} pursuit_w_label=${PURSUIT_W_LABEL:-0} pursuit_w_tree=${PURSUIT_W_TREE:-0} unfreeze_opponent=${UNFREEZE_OPPONENT:-0} dagger_alpha=${DAGGER_ALPHA:-0} value_trust=$VALUE_TRUST games=${NUM_GAMES}x${SELF_PLAY_LOOPS} mcts=$MCTS_ITERS gauge_mcts=${GAUGE_MCTS:-$MCTS_ITERS} gauge_gumbel_scale=${GAUGE_GUMBEL_SCALE:-0} k=$GUMBEL_K kl_ref_model=${KL_REF_MODEL:-none} kl_ref_weight=${KL_REF_WEIGHT:-0}"

    SP_LOG=$(mktemp)
    for ((sp=1; sp<=SELF_PLAY_LOOPS; sp++)); do
        if [ "$SELF_PLAY_LOOPS" -gt 1 ]; then
            echo "🎲 Self-play pass $sp/$SELF_PLAY_LOOPS (-g $NUM_GAMES each)"
        fi
        "$SELF_PLAY_BIN" --num-games $NUM_GAMES --mcts-iters $MCTS_ITERS --gumbel-k $GUMBEL_K --actors $ACTORS --eval-servers $EVAL_SERVERS $REWARD_FLAG $WL_FLAG $GOAL_FLAG $OPPONENT_FLAG $ANCHOR_FLAG $BACKEND_FLAG $DECAY_LAST_ITER_FLAG --td-w "${TD_W:-0.7}" --td-lambda "${TD_LAMBDA:-0.8}" $TD_MISSING_FLAG --outcome-scale "${OUTCOME_SCALE:-3.0}" $LABEL_REL_W_FLAG $SHAPE_FLAGS $UNFREEZE_FLAG --value-trust "$VALUE_TRUST" --tribe1 "$TRIBE1" --tribe2 "$TRIBE2" --iteration "$EFF_ITER" --gamemode "$GAMEMODE" | tee "$SP_LOG"
        SP_STATUS=${PIPESTATUS[0]}
        if [ "$SP_STATUS" -ne 0 ]; then
            echo "Self-play failed with exit code $SP_STATUS" >&2
            rm -f "$SP_LOG"
            exit "$SP_STATUS"
        fi
    done
    rm -f "$SP_LOG"
    
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

    # 5. Strength gauge (EXP 10/11): paired arena reading vs the ladder's
    # active anchor. ladder.py owns ladder.json (anchors, readings, verdicts):
    # >=80% freezes the model as the next anchor (n=64 link match); two
    # consecutive 8-reading windows with no gain stop the run (plateau).
    GAUGE_EVERY=$([ "$GAUGE_INTERVAL" -gt 0 ] && echo "$GAUGE_INTERVAL" || echo "$LEAGUE_INTERVAL")
    if [ "$GAUGE_EVERY" -gt 0 ] && [ $((i % GAUGE_EVERY)) -eq 0 ]; then
        ACTIVE_JSON=$(.venv/bin/python3 ladder.py active)
        ANCHOR_PATH=$(echo "$ACTIVE_JSON" | jq -r '.path')
        ANCHOR_NAME=$(echo "$ACTIVE_JSON" | jq -r '.name')
        GAUGE_LOG=$(mktemp)

        # Readings are only comparable at a FIXED budget — sigma(Q)'s scale grows
        # with visits, so a reading at a new --mcts-iters moves for free (the
        # EXP_ELO_023 ladder spans 44.3% at n=64 to 58.0% at n=2048). Pin
        # GAUGE_MCTS to keep the ladder on one rung while training budget varies.
        GAUGE_MCTS_EFF="${GAUGE_MCTS:-$MCTS_ITERS}"

        # $1 = opponent model path ("" = greedy backend), $2 = seeds (games x2),
        # $3 = per-turn stats dump dir (optional; summarized into the reading)
        # Gauge the policy we would DEPLOY, not the one that generates training
        # data. GUMBEL_SCALE defaults to 1.0 = self-play exploration, and
        # `argmax(logit + Gumbel)` is exactly a SAMPLE from softmax(prior) — so
        # a noisy gauge measures a sampled policy, worth a measured -9..-13pp
        # (60.6% vs 51.6% at n=256, same weights, Jul 27). It also halves
        # ms/move and cuts reading variance, since only map sampling remains.
        # ⚠️ Readings before Jul 27 2026 are noisy-policy and NOT comparable.
        GAUGE_SCALE="${GAUGE_GUMBEL_SCALE:-0}"

        run_gauge_match () {
            GAUGE_STATS_DIR="$3"
            DUMP_FLAG=""
            if [ -n "$GAUGE_STATS_DIR" ]; then
                DUMP_FLAG="--dump-stats-dir $GAUGE_STATS_DIR"
            fi
            if [ -z "$1" ]; then
                GUMBEL_SCALE="$GAUGE_SCALE" "$ARENA_BIN" --model1 model.safetensors --model2 model.safetensors \
                    --backend1 gumbel --backend2 greedy $GOAL_ARENA_FLAG \
                    --mcts "$GAUGE_MCTS_EFF" --gumbel-k "$GUMBEL_K" $DUMP_FLAG \
                    --games "$2" --gamemode "$GAMEMODE" | tee "$GAUGE_LOG"
            else
                GUMBEL_SCALE="$GAUGE_SCALE" "$ARENA_BIN" --model1 model.safetensors --model2 "$1" \
                    --backend1 gumbel --backend2 gumbel $GOAL_ARENA_FLAG \
                    --mcts "$GAUGE_MCTS_EFF" --gumbel-k "$GUMBEL_K" $DUMP_FLAG \
                    --games "$2" --gamemode "$GAMEMODE" | tee "$GAUGE_LOG"
            fi
            GAUGE_W=$(sed -n 's/^Config 1 Wins: \([0-9][0-9]*\).*/\1/p' "$GAUGE_LOG")
            GAUGE_L=$(sed -n 's/^Config 2 Wins: \([0-9][0-9]*\).*/\1/p' "$GAUGE_LOG")
            GAUGE_D=$(sed -n 's/^Draws: *\([0-9][0-9]*\).*/\1/p' "$GAUGE_LOG")
            GAUGE_S1=$(sed -n 's/^Avg Score Config 1: \([0-9.][0-9.]*\).*/\1/p' "$GAUGE_LOG")
            GAUGE_S2=$(sed -n 's/^Avg Score Config 2: \([0-9.][0-9.]*\).*/\1/p' "$GAUGE_LOG")
            GAUGE_WP1=$(sed -n 's/^Config 1 Wins as P1: \([0-9][0-9]*\).*/\1/p' "$GAUGE_LOG")
            GAUGE_WP2=$(sed -n 's/^Config 1 Wins as P2: \([0-9][0-9]*\).*/\1/p' "$GAUGE_LOG")
            GAUGE_BACKEND=$(sed -n 's/.*| eval \([a-z]*\).*/\1/p' "$GAUGE_LOG" | head -1)
        }

        # Snapshot the exact model each reading measures — without this, peaks
        # between run-start snapshots are unrecoverable (EXP_ELO_007 lost its
        # 45.3% iter-25 clone to a 28% iter-30 overwrite).
        GAUGE_SNAPSHOT="checkpoints/gauge_${RUN_ID}_iter${i}.safetensors"
        cp model.safetensors "$GAUGE_SNAPSHOT"

        run_gauge_match "$ANCHOR_PATH" "${GAUGE_GAMES:-32}" "replays/gauge_stats/${RUN_ID}_iter${i}"
        if [ -n "$GAUGE_W" ] && [ -n "$GAUGE_L" ]; then
            VERDICT=$(.venv/bin/python3 ladder.py record --kind gauge \
                --run-id "$RUN_ID" --iteration "$i" \
                --wins "$GAUGE_W" --losses "$GAUGE_L" --draws "${GAUGE_D:-0}" \
                --avg-score-model "${GAUGE_S1:-0}" --avg-score-opponent "${GAUGE_S2:-0}" \
                --mcts "$GAUGE_MCTS_EFF" --gumbel-k "$GUMBEL_K" --eval-backend "${GAUGE_BACKEND:-}" \
                --wins-p1 "${GAUGE_WP1:-0}" --wins-p2 "${GAUGE_WP2:-0}" \
                --model-path "$GAUGE_SNAPSHOT" \
                --stats-dir "$GAUGE_STATS_DIR")
            echo "GAUGE: $VERDICT"
            GAUGE_ACTION=$(echo "$VERDICT" | jq -r '.action')

            # EXP_ELO_002: first >=50% reading vs the greedy anchor starts
            # the anchor-frac decay clock (EFF_ITER units, matching
            # --anchor-decay-start above). Not during BOOTSTRAP (EXP_ELO_007):
            # crossing 50% while cloning doesn't graduate the RL model — the
            # decay clock belongs to the released phase-2 run.
            # NO_ANCHOR_DECAY=1 gates the write entirely (EXP_ELO_009): runs
            # that must hold anchor at 1.0 even while scoring >=50%.
            if [ -z "${BOOTSTRAP:-}" ] && [ -z "${NO_ANCHOR_DECAY:-}" ] && [ ! -f .anchor_decay_start ] && [ "$ANCHOR_NAME" = "greedy" ]; then
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
                cp model.safetensors "$NEW_ANCHOR"
                echo "GAUGE: >=80% vs active anchor — freezing $NEW_ANCHOR, link match (n=64)..."
                run_gauge_match "$ANCHOR_PATH" 64 "replays/gauge_stats/${RUN_ID}_iter${i}_link"
                .venv/bin/python3 ladder.py freeze --run-id "$RUN_ID" --iteration "$i" \
                    --path "$NEW_ANCHOR" --model-path "$NEW_ANCHOR" \
                    --wins "${GAUGE_W:-0}" --losses "${GAUGE_L:-0}" --draws "${GAUGE_D:-0}" \
                    --avg-score-model "${GAUGE_S1:-0}" --avg-score-opponent "${GAUGE_S2:-0}"
            elif [ "$GAUGE_ACTION" = "stop" ]; then
                rm -f "$GAUGE_LOG"
                echo "=================================================="
                echo "PLATEAU STOP at iteration $i: two consecutive 8-reading"
                echo "windows with no gain vs the active anchor (see ladder.json)."
                echo "=================================================="
                break
            fi

            # Audit block every 5th gauge: greedy + one retired anchor,
            # rotating — observed vs chain-predicted win rate flags cycles.
            if [ $((i % (LEAGUE_INTERVAL * 5))) -eq 0 ]; then
                .venv/bin/python3 ladder.py audit-opponents | jq -c '.[]' | while read -r AUD; do
                    AUD_NAME=$(echo "$AUD" | jq -r '.name')
                    AUD_PATH=$(echo "$AUD" | jq -r '.path')
                    run_gauge_match "$AUD_PATH" "${GAUGE_GAMES:-32}" "replays/gauge_stats/${RUN_ID}_iter${i}_audit_${AUD_NAME}"
                    if [ -n "$GAUGE_W" ] && [ -n "$GAUGE_L" ]; then
                        .venv/bin/python3 ladder.py record --kind audit --opponent "$AUD_NAME" \
                            --run-id "$RUN_ID" --iteration "$i" \
                            --wins "$GAUGE_W" --losses "$GAUGE_L" --draws "${GAUGE_D:-0}" \
                            --avg-score-model "${GAUGE_S1:-0}" --avg-score-opponent "${GAUGE_S2:-0}" \
                            --mcts "$GAUGE_MCTS_EFF" --gumbel-k "$GUMBEL_K" --eval-backend "${GAUGE_BACKEND:-}" \
                            --wins-p1 "${GAUGE_WP1:-0}" --wins-p2 "${GAUGE_WP2:-0}" \
                            --model-path "$GAUGE_SNAPSHOT" \
                            --stats-dir "$GAUGE_STATS_DIR"
                    fi
                done
            fi
        else
            echo "GAUGE: arena reading failed to parse — skipping this reading" >&2
        fi
        rm -f "$GAUGE_LOG"
    fi

    # 6. Cleanup (Fresh Games Only)
    # Move played games to archive so train.py only sees new ones next time
    mkdir -p archive
    # Use || true to avoid script exit if no games were generated
    mv games_*.safetensors archive/ 2>/dev/null || true
    
    # Keep only ARCHIVE_KEEP game files — a constant ~10*BASELINE_GAMES-game
    # replay window regardless of -g (train.py reads the same value via
    # REPLAY_BUFFER_FILES)
    ls -t archive/games_*.safetensors 2>/dev/null | tail -n +$((ARCHIVE_KEEP + 1)) | xargs -r rm

done
