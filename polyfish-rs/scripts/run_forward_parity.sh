#!/bin/bash
# Rust<->Python forward parity (audit T1): load the same model.safetensors into
# network.rs (candle, CPU) and train.py (PyTorch, CPU) and compare raw outputs.
#
# The two are separate implementations of one architecture that read and write
# the same file, and nothing enforced that they agree. This is the check that
# found the strided cross-attention bug — see the note in
# hypothesis_driven_improvements.md.
#
#   scripts/run_forward_parity.sh [model.safetensors]
#
# With no argument it uses model.safetensors if present, otherwise it builds a
# fresh one with init_model.py in a scratch dir (so a clean checkout can run it).
set -euo pipefail

cd "$(dirname "$0")/.."
REPO="$PWD"

if [ -x .venv/bin/python ]; then
    PY="$PWD/.venv/bin/python"
else
    PY=$(command -v python3)
fi

if ! "$PY" -c 'import torch' 2>/dev/null; then
    echo "forward parity needs torch (polyfish-rs/.venv, or see local_setup.sh)" >&2
    exit 1
fi

MODEL="${1:-}"
SCRATCH=""
if [ -z "$MODEL" ]; then
    if [ -f model.safetensors ]; then
        MODEL=model.safetensors
    else
        SCRATCH=$(mktemp -d)
        echo "no model.safetensors — initialising one in $SCRATCH"
        # init_model.py imports PolyZeroNet from train.py and writes into cwd,
        # so run it from the scratch dir with this tree on PYTHONPATH.
        ( cd "$SCRATCH" && PYTHONPATH="$REPO" "$PY" "$REPO/init_model.py" >/dev/null )
        MODEL="$SCRATCH/model.safetensors"
    fi
fi

RUST_JSON=$(mktemp)
trap 'rm -f "$RUST_JSON"; [ -n "$SCRATCH" ] && rm -rf "$SCRATCH"' EXIT

cargo run --quiet --no-default-features --example py_parity -- "$MODEL" > "$RUST_JSON"
"$PY" scripts/py_parity.py "$RUST_JSON" "$MODEL"
