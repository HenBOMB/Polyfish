#!/bin/bash
# Runs on the WORKER Mac. Generates self-play games against whatever model and
# config the main box last published, and drops finished games in outbox/ for
# the main box to collect.
#
# Mirrors the main loop's flags verbatim (SELF_PLAY_ARGS from the manifest) so
# --iteration, --value-trust, --anchor-* and the td/outcome weights match. Only
# hardware knobs are set locally.
#
#   GAMES=32 ./worker/worker_loop.sh
set -euo pipefail

cd "$(dirname "$0")/.."

GAMES=${GAMES:-32}
ACTORS=${ACTORS:-128}
EVAL_SERVERS=${EVAL_SERVERS:-3}
TAG=${TAG:-$(hostname -s | tr -cd '[:alnum:]')}
PY=${PY:-python3}

STAGING=worker/staging
OUTBOX=worker/outbox
mkdir -p "$OUTBOX" "$STAGING" worker/rejected

BIN=./target/release/self_play
[ -x "$BIN" ] || { echo "missing $BIN — run worker/setup_worker_mac.sh first" >&2; exit 1; }

# Guards against the one silent-corruption path: a worker binary whose feature
# width no longer matches the model's conv1. Everything else fails loudly at
# model load; a width mismatch would ship mislabeled tensors.
check_width() {
    "$PY" - "$1" "$2" <<'PY'
import json, struct, sys
with open(sys.argv[1], "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(n))
cols = header["spatial_maps"]["shape"][1]
got, want = cols // 121, int(sys.argv[2])
if got != want:
    sys.exit(f"feature width {got} != model conv1 {want}")
PY
}

echo "worker '$TAG': $GAMES games/batch, actors=$ACTORS, eval-servers=$EVAL_SERVERS"

while true; do
    if [ ! -f "$STAGING/manifest.env" ] || [ ! -f "$STAGING/model.safetensors" ]; then
        echo "waiting for main box to publish model + manifest..."
        sleep 20
        continue
    fi

    # shellcheck disable=SC1091
    source "$STAGING/manifest.env"
    if [ -z "${SELF_PLAY_ARGS:-}" ]; then
        echo "manifest has no SELF_PLAY_ARGS (main box mid-train?); waiting..."
        sleep 20
        continue
    fi

    LOCAL_SHA=$(git rev-parse HEAD 2>/dev/null || echo unknown)
    if [ -n "${GIT_SHA:-}" ] && [ "$LOCAL_SHA" != "$GIT_SHA" ]; then
        echo "⚠️  source drift: worker $LOCAL_SHA vs main ${GIT_SHA:0:12} — rebuild if features changed"
    fi

    cp "$STAGING/model.safetensors" model.safetensors

    echo "── iter ${ITERATION:-?} · $GAMES games · $(date +%H:%M:%S)"
    # shellcheck disable=SC2086
    $BIN $SELF_PLAY_ARGS \
        --num-games "$GAMES" --actors "$ACTORS" --eval-servers "$EVAL_SERVERS" \
        || { echo "self_play failed; retrying in 30s" >&2; sleep 30; continue; }

    shopt -s nullglob
    for f in games_*.safetensors; do
        DEST="$OUTBOX/${f%.safetensors}_${TAG}.safetensors"
        if [ -n "${CONV1_IN:-}" ] && ! check_width "$f" "$CONV1_IN"; then
            echo "⚠️  quarantining $f (width mismatch)" >&2
            mv "$f" worker/rejected/
            continue
        fi
        mv "$f" "$DEST"     # atomic: main's rsync only ever sees whole files
        echo "→ $(basename "$DEST")"
    done
    shopt -u nullglob
done
