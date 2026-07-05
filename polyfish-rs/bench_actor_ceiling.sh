#!/usr/bin/env bash
# Focused actor_ceiling sweep — not a full latency×lanes grid.
#
# Prior partial run (64 games, 128 actors) showed moves/s scales ~linearly with
# lanes and ~1/latency. Midpoints (1500/2500µs, 6 lanes) add little; lane=1 is
# only the serial-GPU floor. This script fills gaps + sweeps actors at the
# deployment-shaped point (1000µs, 4 lanes ≈ metal's 4 workers @ ~1ms).
#
# Usage:
#   ./bench_actor_ceiling.sh          # focused follow-up (~6 runs, ~5 min)
#   ./bench_actor_ceiling.sh --full   # old 36-run grid (slow)
set -euo pipefail
cd "$(dirname "$0")"

LOCKDIR=bench_actor_ceiling.lock.d
if ! mkdir "$LOCKDIR" 2>/dev/null; then
    echo "Another actor_ceiling sweep is already running." >&2
    exit 1
fi
trap 'rmdir "$LOCKDIR"' EXIT

BIN=./target/release/actor_ceiling
LOG=actor_ceiling_sweep.log
NUM_GAMES=${NUM_GAMES:-32}
MCTS=64
ACTORS=128
MODE="${1:-focused}"

log() { echo "$@" | tee -a "$LOG"; }

run_one() {
    local latency="$1" lanes="$2" actors="$3"
    local out
    out=$("$BIN" --actors "$actors" --num-games "$NUM_GAMES" --mcts-iters "$MCTS" \
        --eval-latency-us "$latency" --eval-lanes "$lanes" 2>&1 \
        | awk '/=== Actor-Ceiling Results ===/,0')
    local moves_s leaves_s leaves_move wall moves
    moves_s=$(echo "$out" | sed -n 's/.*Moves \/ s (ceiling): //p')
    leaves_s=$(echo "$out" | sed -n 's/.*Leaves \/ s: *//p')
    leaves_move=$(echo "$out" | sed -n 's/.*Leaves \/ move: *//p')
    wall=$(echo "$out" | sed -n 's/.*Wall time: *//p' | sed 's/s//')
    moves=$(echo "$out" | sed -n 's/^Moves: *//p' | head -1 | tr -d ' ')
    log "$(date +%H:%M:%S),$latency,$lanes,$actors,$moves_s,$leaves_s,$leaves_move,$wall,$moves"
}

run_focused() {
    log ""
    log "== focused sweep: num_games=$NUM_GAMES mcts=$MCTS =="
    log "timestamp,latency_us,lanes,actors,moves_per_s,leaves_per_s,leaves_per_move,wall_s,moves"

    # Gaps from partial grid + deployment-relevant points
    log "-- latency gaps (128 actors) --"
    run_one 3000 4 "$ACTORS"   # user's original query point
    run_one 3000 8 "$ACTORS"   # 4-lane scaling extrapolation check

    log "-- actors @ 1000us / 4 lanes (metal-shaped) --"
    for actors in 64 96 128 160; do
        run_one 1000 4 "$actors"
    done

    log "-- pure ceiling reference (32 games) --"
    run_one 0 0 128
}

run_full() {
    : > "$LOG"
    log "actor_ceiling sweep (FULL): num_games=$NUM_GAMES mcts=$MCTS actors=$ACTORS"
    log "timestamp,latency_us,lanes,actors,moves_per_s,leaves_per_s,leaves_per_move,wall_s,moves"
    log "---"

    log "== pure ceiling (0 latency) =="
    run_one 0 0 "$ACTORS"

    log "== latency × lanes (actors=$ACTORS) =="
    for latency in 1000 1500 2000 2500 3000; do
        for lanes in 1 2 4 6 8; do
            run_one "$latency" "$lanes" "$ACTORS"
        done
    done

    log "== actors sweep =="
    for actors in 64 96 128 160 192; do
        run_one 1000 8 "$actors"
        run_one 1000 4 "$actors"
    done
}

case "$MODE" in
    --full) run_full ;;
    *) run_focused ;;
esac

log ""
log "== done =="
