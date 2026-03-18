#!/bin/bash
# Polyfish Evolutionary Optimization Launcher

echo "Building Polyfish Arena in release mode..."
cargo build --release --bin arena

export RAYON_NUM_THREADS=10
export OMP_NUM_THREADS=10

# Configuration
POP_SIZE=16
GENS=1000
MCTS_ITERS=200
MATCHES_PER_PAIR=2
MUTATION_RATE=0.05
ELITES=4
OUTPUT_DIR="evolution_results"

# Run the evolution
echo "Starting Evolution: Pop=$POP_SIZE, Gens=$GENS, MCTS=$MCTS_ITERS"

# Find latest best candidate to resume if it exists
LOAD_ARGS=""
if [ -d "$OUTPUT_DIR" ]; then
    # Look for both "gen_X_best.json" and "gen_X_fit_Y.json"
    LATEST_BEST=$(ls -v "$OUTPUT_DIR"/gen_*.json 2>/dev/null | tail -n 1)
    if [ -n "$LATEST_BEST" ]; then
        echo "Found existing evolution data. Resuming from $LATEST_BEST"
        LOAD_ARGS="--load $LATEST_BEST"
    fi
fi

./target/release/arena \
    --pop-size $POP_SIZE \
    --gens $GENS \
    --mcts $MCTS_ITERS \
    --matches $MATCHES_PER_PAIR \
    --mutation-rate $MUTATION_RATE \
    --elites $ELITES \
    --output $OUTPUT_DIR \
    $LOAD_ARGS
