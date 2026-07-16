#!/bin/bash
# Rate checkpoints on the anchored Elo ladder.
#
# For every checkpoint in checkpoints/ (or explicit model files passed as
# args) that has no games in the ledger yet, plays paired side-swapped arena
# matches against: the 2 nearest-strength fixed anchors, the previous
# checkpoint, and one random earlier checkpoint. Then refits all ratings
# (elo.py) and prints the ladder.
#
# Near-strength opponents matter: a >400-Elo mismatch carries almost no rating
# information and score-adjudication noise biases it, so anchors are chosen by
# the previous checkpoint's fitted rating when one exists.
#
# Env overrides:
#   SEEDS=8        seeds per pairing (x2 sides = 16 games)
#   MCTS=128       eval search budget for network players
#   BACKEND=gumbel eval search backend for network players
#   MAX_TURNS=30   arena game length (score adjudication at cap)
#   WORKERS=0      arena worker cap (0 = all cores)
#   MATCHES=matches.jsonl  RATINGS=elo_ratings.json
#   FORCE=1        re-rate even if the ledger already has games for a player

set -e
cd "$(dirname "$0")"

SEEDS=${SEEDS:-8}
MCTS=${MCTS:-128}
BACKEND=${BACKEND:-gumbel}
MAX_TURNS=${MAX_TURNS:-30}
WORKERS=${WORKERS:-0}
MATCHES=${MATCHES:-matches.jsonl}
RATINGS=${RATINGS:-elo_ratings.json}
ARENA=./target/release/arena

if [ -x .venv/bin/python3 ]; then PY=.venv/bin/python3
elif [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe
else PY=python3
fi

# Anchors: name / backend / mcts. Names must match arena's player_name().
ANCHORS=("random:random:1" "greedy:greedy:1")
# Rough ladder positions used only for nearest-anchor selection before a fit
# exists for the anchor itself; real ratings take over once fitted.
ANCHOR_GUESS=("random:0" "greedy:250")

if command -v cargo >/dev/null 2>&1; then
    cargo build --release --bin arena
elif [ ! -x "$ARENA" ]; then
    echo "Error: $ARENA not found and cargo not available to build it." >&2
    exit 1
fi

is_rated() { # player name has any ledger games
    [ "${FORCE:-0}" = "1" ] && return 1
    [ -f "$MATCHES" ] && grep -qF "\"$1\"" "$MATCHES"
}

rating_of() { # fitted elo of player $1, else empty
    [ -f "$RATINGS" ] || return 0
    "$PY" -c "import json,sys; r=json.load(open('$RATINGS')); p=r.get('$1'); print(r'' if p is None else p['elo'])" 2>/dev/null || true
}

play() { # play $SEEDS side-swapped seeds: model1 name1 backend1 mcts1 model2 name2 backend2 mcts2
    echo ""
    echo "--- $2 vs $6 ($SEEDS seeds x 2 sides) ---"
    "$ARENA" --model1 "$1" --model2 "$5" \
        --backend1 "$3" --backend2 "$7" --mcts1 "$4" --mcts2 "$8" \
        --games "$SEEDS" --max-turns "$MAX_TURNS" --workers "$WORKERS" \
        --json-out "$MATCHES" --name1 "$2" --name2 "$6"
}

# Two anchors nearest to an expected rating (all four when no expectation).
nearest_anchors() {
    local expected=$1
    if [ -z "$expected" ]; then
        printf '%s\n' "${ANCHORS[@]:0:3}"
        return
    fi
    for spec in "${ANCHORS[@]}"; do
        local name=${spec%%:*}
        local elo
        elo=$(rating_of "$name")
        if [ -z "$elo" ]; then
            for g in "${ANCHOR_GUESS[@]}"; do
                [ "${g%%:*}" = "$name" ] && elo=${g##*:}
            done
        fi
        echo "$spec $elo"
    done | awk -v e="$expected" '{d=$2-e; if(d<0)d=-d; print d, $1}' | sort -n | head -2 | awk '{print $2}'
}

# Collect models to rate: explicit args, else checkpoints/ sorted by iteration.
SYNC_SUPABASE=false
LOAD_PROGRESS=false
MIN_ITER=${MIN_ITER:-0}
MODELS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --supabase|--save)
            SYNC_SUPABASE=true
            shift
            ;;
        --load)
            LOAD_PROGRESS=true
            shift
            ;;
        --min-iter)
            MIN_ITER="$2"
            shift 2
            ;;
        --min-iter=*)
            MIN_ITER="${1#*=}"
            shift
            ;;
        *)
            MODELS+=("$1")
            shift
            ;;
    esac
done

if [ "$LOAD_PROGRESS" = true ] && [ -f "supabase_sync.py" ]; then
    echo "Downloading previous Elo progress from Supabase..."
    "$PY" supabase_sync.py download "$MATCHES" || true
    "$PY" supabase_sync.py download "$RATINGS" || true
fi

if [ ${#MODELS[@]} -eq 0 ]; then
    # Auto-sync missing checkpoints from supabase before rating if requested
    if [ "$SYNC_SUPABASE" = true ] && [ -f "supabase_sync.py" ]; then
        "$PY" supabase_sync.py download-all-checkpoints "$MIN_ITER" || true
    fi

    while IFS= read -r f; do
        if [ "$MIN_ITER" -gt 0 ]; then
            base=$(basename "$f")
            if [[ "$base" =~ model_checkpoint_iter([0-9]+)(_.*)?\.safetensors ]]; then
                iter="${BASH_REMATCH[1]}"
                if [ "$iter" -lt "$MIN_ITER" ]; then
                    continue
                fi
            fi
        fi
        MODELS+=("$f")
    done \
        < <(ls checkpoints/model_checkpoint_iter*.safetensors 2>/dev/null | sort -V)
fi
if [ ${#MODELS[@]} -eq 0 ]; then
    echo "No models to rate (no args, nothing in checkpoints/)." >&2
    exit 1
fi
DUMMY_MODEL=${MODELS[0]} # network-free backends still need a loadable model path

# First run: cross-link the anchors so the fit is connected to random=0.
if ! is_rated "greedy"; then
    play "$DUMMY_MODEL" "greedy" "greedy" 1 "$DUMMY_MODEL" "random" "random" 1
fi

PREV_MODEL=""
PREV_NAME=""
RATED_HIST=() # "model|name" of checkpoints already in the ledger this session
NEW_RATED=0

for MODEL in "${MODELS[@]}"; do
    NAME="$(basename "$MODEL" .safetensors)@${BACKEND}${MCTS}"
    if is_rated "$NAME"; then
        RATED_HIST+=("$MODEL|$NAME")
        PREV_MODEL=$MODEL; PREV_NAME=$NAME
        continue
    fi

    echo ""
    echo "=================================================="
    echo "Rating $NAME"
    echo "=================================================="

    EXPECTED=""
    [ -n "$PREV_NAME" ] && EXPECTED=$(rating_of "$PREV_NAME")

    while IFS= read -r spec; do
        A_NAME=${spec%%:*}; rest=${spec#*:}; A_BACKEND=${rest%%:*}; A_MCTS=${rest##*:}
        play "$MODEL" "$NAME" "$BACKEND" "$MCTS" "$MODEL" "$A_NAME" "$A_BACKEND" "$A_MCTS" &
    done < <(nearest_anchors "$EXPECTED")

    if [ -n "$PREV_MODEL" ]; then
        play "$MODEL" "$NAME" "$BACKEND" "$MCTS" "$PREV_MODEL" "$PREV_NAME" "$BACKEND" "$MCTS" &
    fi

    # One random earlier checkpoint (not prev) for long-range graph links.
    if [ ${#RATED_HIST[@]} -gt 1 ]; then
        HIST=${RATED_HIST[$((RANDOM % (${#RATED_HIST[@]} - 1)))]}
        H_MODEL=${HIST%%|*}; H_NAME=${HIST##*|}
        play "$MODEL" "$NAME" "$BACKEND" "$MCTS" "$H_MODEL" "$H_NAME" "$BACKEND" "$MCTS" &
    fi
    wait

    RATED_HIST+=("$MODEL|$NAME")
    PREV_MODEL=$MODEL; PREV_NAME=$NAME
    NEW_RATED=$((NEW_RATED + 1))

    # Refit after each checkpoint so nearest_anchors sees fresh ratings.
    "$PY" elo.py fit --matches "$MATCHES" --out "$RATINGS" --bootstrap 0 >/dev/null
    
    if [ "$SYNC_SUPABASE" = true ] && [ -f "supabase_sync.py" ]; then
        echo "Uploading mid-run progress to Supabase..."
        "$PY" supabase_sync.py upload "$MATCHES" >/dev/null 2>&1 || true
        "$PY" supabase_sync.py upload "$RATINGS" >/dev/null 2>&1 || true
    fi
done

echo ""
if [ "$NEW_RATED" -eq 0 ] && ! [ -f "$MATCHES" ]; then
    echo "Nothing rated and no ledger; done." >&2
    exit 0
fi
"$PY" elo.py fit --matches "$MATCHES" --out "$RATINGS"

if [ "$SYNC_SUPABASE" = true ] && [ -f "supabase_sync.py" ]; then
    echo "Uploading results to Supabase..."
    "$PY" supabase_sync.py upload "$MATCHES" || true
    "$PY" supabase_sync.py upload "$RATINGS" || true
fi
