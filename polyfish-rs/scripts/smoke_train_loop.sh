#!/usr/bin/env bash
# End-to-end smoke test of the training seam:
#   run_training_loop.sh -> self_play -> games_*.safetensors -> train.py -> model.safetensors
#
# Runs the real driver at toy settings inside an isolated copy under
# target/smoke, so it never touches the checked-out model, training_log.csv,
# checkpoints or archive. Env knobs: SMOKE_DIR, SMOKE_VENV, SMOKE_GAMES,
# SMOKE_MCTS, SMOKE_ACTORS, SMOKE_GUMBEL_K, SMOKE_LEAGUE, SMOKE_TIMEOUT.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SMOKE_DIR="${SMOKE_DIR:-$REPO/target/smoke}"
# Kept outside SMOKE_DIR (and symlinked in) so re-staging does not throw away
# the release build, and so rust-cache still sees it under polyfish-rs/target.
SMOKE_CARGO_DIR="${SMOKE_CARGO_DIR:-$REPO/target/smoke-cargo}"
SMOKE_VENV="${SMOKE_VENV:-$REPO/.venv}"
GAMES="${SMOKE_GAMES:-2}"
MCTS="${SMOKE_MCTS:-4}"
ACTORS="${SMOKE_ACTORS:-2}"
GUMBEL_K="${SMOKE_GUMBEL_K:-2}"
LEAGUE="${SMOKE_LEAGUE:-1}"
TIMEOUT="${SMOKE_TIMEOUT:-5400}"

case "$SMOKE_DIR" in
    /*/*smoke*) ;;
    *) echo "smoke: refusing to wipe SMOKE_DIR=$SMOKE_DIR (want an absolute path naming 'smoke')" >&2
       exit 2 ;;
esac
if [ ! -x "$SMOKE_VENV/bin/python3" ]; then
    echo "smoke: no python venv at $SMOKE_VENV (run ./local_setup.sh or set SMOKE_VENV)" >&2
    exit 2
fi

echo "smoke: staging $REPO -> $SMOKE_DIR"
rm -rf "$SMOKE_DIR"
mkdir -p "$SMOKE_DIR" "$SMOKE_CARGO_DIR"
tar -C "$REPO" -cf - \
    --exclude=./target --exclude=./.venv --exclude=./.git \
    --exclude=./archive --exclude=./checkpoints --exclude=./replays \
    --exclude=./.run_bin --exclude='./games_*.safetensors' \
    --exclude=./model.safetensors --exclude=./optimizer_state.pt \
    --exclude=./training_log.csv --exclude=./ladder.json \
    --exclude=./moves_by_turn.json --exclude='./*.log' \
    --exclude='./.last_*' --exclude='./.anchor_*' --exclude=./.training.pid \
    . | tar -C "$SMOKE_DIR" -xf -
ln -s "$SMOKE_VENV" "$SMOKE_DIR/.venv"
ln -s "$SMOKE_CARGO_DIR" "$SMOKE_DIR/target"

# The release profile is lto = "fat" + codegen-units = 1; a smoke test only
# needs the binaries to run, not to be fast.
export CARGO_PROFILE_RELEASE_LTO=false
export CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16
export CARGO_PROFILE_RELEASE_DEBUG=false
export CARGO_PROFILE_RELEASE_OPT_LEVEL=1
# Keep the gauge match to a single seed pair; the default 32 is a real reading.
export GAUGE_GAMES="${GAUGE_GAMES:-1}"

echo "smoke: run_training_loop.sh -i 1 -g $GAMES -n $MCTS -a $ACTORS -k $GUMBEL_K -l $LEAGUE"
set +e
( cd "$SMOKE_DIR" && timeout "$TIMEOUT" bash ./run_training_loop.sh --no-server \
    -i 1 -g "$GAMES" -n "$MCTS" -a "$ACTORS" -k "$GUMBEL_K" -l "$LEAGUE" )
status=$?
set -e
if [ "$status" -ne 0 ]; then
    echo "smoke: run_training_loop.sh exited $status" >&2
    tail -n 40 "$SMOKE_DIR/session.log" 2>/dev/null >&2 || true
    exit "$status"
fi

fail() { echo "smoke: $1" >&2; exit 1; }

[ -f "$SMOKE_DIR/model.safetensors" ] || fail "train.py produced no model.safetensors"
compgen -G "$SMOKE_DIR/archive/games_*.safetensors" > /dev/null \
    || compgen -G "$SMOKE_DIR/games_*.safetensors" > /dev/null \
    || fail "self_play produced no games_*.safetensors"
[ "$(wc -l < "$SMOKE_DIR/training_log.csv")" -ge 2 ] || fail "training_log.csv has no data row"
# self_play and train.py hand their metrics to training_log.py through these
# sidecars; a METRICS: stdout line is the older path.
[ -s "$SMOKE_DIR/.last_self_play_metrics.json" ] \
    || grep -q "METRICS:" "$SMOKE_DIR/session.log" \
    || fail "self_play recorded no metrics"
[ -s "$SMOKE_DIR/.last_train_metrics.json" ] || fail "train.py recorded no metrics"
if [ "$LEAGUE" -gt 0 ]; then
    [ -s "$SMOKE_DIR/ladder.json" ] || fail "the strength gauge recorded no ladder reading"
fi

echo "smoke: OK (artifacts in $SMOKE_DIR)"
