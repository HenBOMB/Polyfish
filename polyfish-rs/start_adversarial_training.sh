#!/usr/bin/env bash
# ==============================================================================
# GPU-Accelerated Adversarial Self-Play Reinforcement Learning
# ==============================================================================
# This script launches the AlphaZero self-play training loop with:
#   1. POLYFISH_ADVERSARIAL_SEARCH=1 (Explicit minimax tree handover between P1 and P2)
#   2. ITER_OFFSET=76 (Warm start curriculum at mature 30-turn games)
#   3. DAgger Foundation Model (Pre-trained on 50,777 expert transitions)
#   4. Gumbel MCTS with N=128 rollouts
# ==============================================================================
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================================================="
echo " Starting PolyZero Adversarial Self-Play Training (Path A)"
echo "=============================================================================="

# 1. Ensure Foundation Model Weights Exist
if [ ! -f "model.safetensors" ]; then
    if [ -f "model_dagger_foundation.safetensors" ]; then
        echo "[+] Initializing model.safetensors from DAgger Foundation Model (model_dagger_foundation.safetensors)..."
        cp model_dagger_foundation.safetensors model.safetensors
    else
        echo "[!] WARNING: Neither model.safetensors nor model_dagger_foundation.safetensors found!"
        echo "    Initializing fresh random weights..."
        python3 init_model.py
    fi
else
    echo "[+] Using existing model.safetensors"
fi

# 2. Set Adversarial MCTS and Curriculum Environment Variables
export POLYFISH_ADVERSARIAL_SEARCH=1
export ITER_OFFSET="${ITER_OFFSET:-76}"
export RUST_BACKTRACE=1
export PYTHONUNBUFFERED=1

echo "[+] Adversarial MCTS Search: ENABLED (POLYFISH_ADVERSARIAL_SEARCH=1)"
echo "[+] Iteration Curriculum Offset: ${ITER_OFFSET} (skips solitaire warmup)"
echo "[+] Search Rollouts per Move: 128"
echo "[+] Games per Iteration: 64"
echo "[+] Parallel Actors: 64"
echo "=============================================================================="

# 3. Check for Python venv
if [ -d ".venv" ]; then
    export PATH="$SCRIPT_DIR/.venv/bin:$PATH"
fi

# 4. Launch Training Loop
# Accepts pass-through flags (e.g. ./start_adversarial_training.sh --resume)
if [ $# -eq 0 ]; then
    exec ./run_training_loop.sh --new-run -g 64 -n 128 -a 64 --no-server
else
    exec ./run_training_loop.sh "$@"
fi
