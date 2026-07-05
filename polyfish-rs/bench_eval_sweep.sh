#!/bin/bash
# Sparse self-play throughput sweep over the eval server × worker grid.
# Usage: ./bench_eval_sweep.sh ["S W" ...]     e.g. ./bench_eval_sweep.sh "2 2" "4 2"
# Env overrides: GAMES, ACTORS, MCTS_ITERS, ITERATION.
# Note: each run emits a games_*.safetensors like any self_play invocation.
set -e
cd "$(dirname "$0")"

export LIBTORCH_USE_PYTORCH=1 LIBTORCH_BYPASS_VERSION_CHECK=1
export DYLD_LIBRARY_PATH="$(.venv/bin/python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"

GAMES=${GAMES:-128}
ACTORS=${ACTORS:-128}
MCTS_ITERS=${MCTS_ITERS:-64}
ITERATION=${ITERATION:-40}

CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then
  CONFIGS=("1 3" "2 2" "2 3" "3 2" "3 3" "4 2" "5 3")
fi

echo "games=$GAMES actors=$ACTORS mcts=$MCTS_ITERS iteration=$ITERATION"
printf "%-8s %-9s %-10s %-22s %s\n" "config" "moves/s" "avg_batch" "prep/wait/post (s)" "busy_frac"
for cfg in "${CONFIGS[@]}"; do
  read -r S W <<< "$cfg"
  out=$(./target/release/self_play --num-games "$GAMES" --actors "$ACTORS" --mcts-iters "$MCTS_ITERS" \
        --eval-servers "$S" --eval-workers "$W" --iteration "$ITERATION" 2>/dev/null \
        | grep -E "Throughput|EVAL_SERVER_STATS_AGG")
  mps=$(sed -n 's/.*Throughput: \([0-9.]*\) moves\/sec.*/\1/p' <<< "$out" | head -1)
  agg=$(grep EVAL_SERVER_STATS_AGG <<< "$out" | head -1)
  ab=$(sed -n 's/.*"avg_batch": \([0-9.]*\).*/\1/p' <<< "$agg")
  prep=$(sed -n 's/.*"prep_s": \([0-9.]*\).*/\1/p' <<< "$agg")
  waits=$(sed -n 's/.*"wait_s": \([0-9.]*\).*/\1/p' <<< "$agg")
  post=$(sed -n 's/.*"post_s": \([0-9.]*\).*/\1/p' <<< "$agg")
  bf=$(sed -n 's/.*"busy_frac": \([0-9.]*\).*/\1/p' <<< "$agg")
  printf "%-8s %-9s %-10s %-22s %s\n" "${S}x${W}" "$mps" "$ab" "${prep}/${waits}/${post}" "$bf"
done
