#!/bin/bash
# Runs on the MAIN box, alongside run_training_loop.sh. Pushes the current
# model + self-play config to the worker Mac and pulls finished games back.
#
# Only this box initiates SSH, so the worker is the only machine that needs
# Remote Login enabled.
#
#   WORKER_HOST=verdi@m3b.local ./worker/sync_worker.sh
set -euo pipefail

cd "$(dirname "$0")/.."

WORKER_HOST=${WORKER_HOST:?set WORKER_HOST, e.g. verdi@m3b.local}
WORKER_DIR=${WORKER_DIR:-Development/Polyfish/polyfish-rs}
SSH_PORT=${SSH_PORT:-22}
INTERVAL=${INTERVAL:-30}

SSH="ssh -p $SSH_PORT -o BatchMode=yes -o ConnectTimeout=10"
INBOX=worker/inbox
SNAP=worker/staging
mkdir -p "$INBOX" "$SNAP"

# train.py's save_file() is not atomic, so a naive rsync can ship a
# half-written model. Snapshot, then parse the safetensors header to prove the
# copy is whole before it leaves the box.
snapshot_model() {
    # -p keeps mtime so rsync skips the transfer while the model is unchanged;
    # a plain cp would restamp it and re-ship the whole file every cycle.
    cp -p model.safetensors "$SNAP/model.safetensors.tmp"
    .venv/bin/python3 - "$SNAP/model.safetensors.tmp" <<'PY' || return 1
import json, struct, sys
with open(sys.argv[1], "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(n))
size = __import__("os").path.getsize(sys.argv[1])
end = max(v["data_offsets"][1] for k, v in header.items() if k != "__metadata__")
assert 8 + n + end == size, f"truncated: expected {8 + n + end}, got {size}"
PY
    mv "$SNAP/model.safetensors.tmp" "$SNAP/model.safetensors"
}

collect() {
    shopt -s nullglob
    local moved=0
    for f in "$INBOX"/games_*.safetensors; do
        mv "$f" .            # atomic rename: train.py sees whole file or none
        moved=$((moved + 1))
    done
    shopt -u nullglob
    echo "$moved"
}

echo "sync → $WORKER_HOST:$WORKER_DIR every ${INTERVAL}s (Ctrl-C to stop)"

while true; do
    TS=$(date +%H:%M:%S)

    if .venv/bin/python3 worker/publish_manifest.py worker/manifest.env >/dev/null 2>&1 \
       && snapshot_model; then
        $SSH "$WORKER_HOST" "mkdir -p $WORKER_DIR/worker/staging $WORKER_DIR/worker/outbox" 2>/dev/null || true
        rsync -a -e "$SSH" \
            "$SNAP/model.safetensors" worker/manifest.env \
            "$WORKER_HOST:$WORKER_DIR/worker/staging/" 2>/dev/null \
            && PUSH=ok || PUSH=FAIL
    else
        PUSH=skip
    fi

    rsync -a --remove-source-files -e "$SSH" \
        "$WORKER_HOST:$WORKER_DIR/worker/outbox/" "$INBOX/" 2>/dev/null \
        && PULL=ok || PULL=FAIL

    GOT=$(collect)
    ITER=$(sed -n 's/^ITERATION=//p' worker/manifest.env 2>/dev/null || echo "?")
    echo "[$TS] push=$PUSH pull=$PULL iter=${ITER:-?} games_collected=$GOT"

    sleep "$INTERVAL"
done
