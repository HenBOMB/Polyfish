#!/bin/bash
# One-time bootstrap for a macOS self-play worker. Run ON THE WORKER.
#
# Builds self_play WITHOUT tch-eval: a pure generation worker never touches
# libtorch, so this skips PyTorch, the 2GB libtorch download, the version pin
# and DYLD_LIBRARY_PATH entirely. MPSGraph (metal-eval) is the fast path.
set -euo pipefail

cd "$(dirname "$0")/.."

xcode-select -p >/dev/null 2>&1 || { echo "run: xcode-select --install" >&2; exit 1; }

if ! command -v cargo >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
fi

echo "building self_play (metal + accelerate + metal-eval, no tch)..."
cargo build --release --bin self_play \
    --no-default-features --features "metal,accelerate,metal-eval"

mkdir -p worker/staging worker/outbox worker/rejected

cat <<'EOF'

✅ worker ready.

Enable Remote Login so the main box can reach this machine:
    System Settings → General → Sharing → Remote Login  (on)

Then on the MAIN box, add your key once:
    ssh-copy-id verdi@<this-machine>.local

Start generating (waits until the main box publishes a model):
    ./worker/worker_loop.sh
EOF
