#!/usr/bin/env bash
# Sweep actor_ceiling configs; parse moves/s from output.
set -euo pipefail
cd "$(dirname "$0")"
BIN=./target/release/actor_ceiling
NUM_GAMES=128
MCTS=64
ACTORS=128

echo "actor_ceiling sweep: num_games=$NUM_GAMES mcts=$MCTS actors=$ACTORS"
echo "timestamp,latency_us,lanes,actors,moves_per_s,leaves_per_s,leaves_per_move,wall_s,moves"
echo "---"

run_one() {
    local latency="$1" lanes="$2" actors="$3"
    local out
    out=$("$BIN" --actors "$actors" --num-games "$NUM_GAMES" --mcts-iters "$MCTS" \
        --eval-latency-us "$latency" --eval-lanes "$lanes" 2>&1 \
        | grep -E "Moves / s|Leaves / s|Leaves / move|Wall time|Moves:" || true)
    local moves_s leaves_s leaves_move wall moves
    moves_s=$(echo "$out" | sed -n 's/.*Moves \/ s (ceiling): //p')
    leaves_s=$(echo "$out" | sed -n 's/.*Leaves \/ s: *//p')
    leaves_move=$(echo "$out" | sed -n 's/.*Leaves \/ move: *//p')
    wall=$(echo "$out" | sed -n 's/.*Wall time: *//p' | sed 's/s//')
    moves=$(echo "$out" | sed -n 's/.*Moves: *//p')
    echo "$(date +%H:%M:%S),$latency,$lanes,$actors,$moves_s,$leaves_s,$leaves_move,$wall,$moves"
}

# Pure actor ceiling (no eval sim)
echo "== pure ceiling (0 latency) =="
run_one 0 0 "$ACTORS"

echo "== latency × lanes (actors=$ACTORS) =="
for latency in 1000 1500 2000 2500 3000; do
    for lanes in 1 2 4 6 8; do
        run_one "$latency" "$lanes" "$ACTORS"
    done
done

echo "== actors sweep at best-looking configs =="
for actors in 64 96 128 160 192; do
    run_one 1000 8 "$actors"
done
for actors in 64 96 128 160 192; do
    run_one 1000 4 "$actors"
done
