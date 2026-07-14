#!/usr/bin/env python3
"""Python half of the candle<->PyTorch network parity test.

Usage:
  cargo run --release --bin parity_dump [model.safetensors] [parity_dump.safetensors]
  .venv/bin/python3 parity_check.py [model.safetensors] [parity_dump.safetensors]

Loads the same weights into the PyTorch PolyZeroNet, replays the exact input
the Rust side used, and compares every head. Weights are stored fp16, compute
is f32 on both sides, so agreement should be ~1e-3; a real architecture
mismatch shows up orders of magnitude larger (or as a shape error).
"""

import sys

import torch
from safetensors.torch import load_file

from train import PolyZeroNet

TOLERANCE = 5e-3

HEADS = [
    ("action_type", "policy", "action_type"),
    ("source_spatial", "policy", "source_spatial"),
    ("target_spatial", "policy", "target_spatial"),
    ("move_option", "policy", "move_option"),
    ("win_value", "value", "win"),
    ("progress_value", "value", "progress"),
    ("ownership_value", "value", "ownership"),
]


def main() -> int:
    model_path = sys.argv[1] if len(sys.argv) > 1 else "model.safetensors"
    dump_path = sys.argv[2] if len(sys.argv) > 2 else "parity_dump.safetensors"

    dump = load_file(dump_path)
    spatial = dump["input_spatial"].float()
    player = dump["input_player"].float()
    _, channels, map_h, map_w = spatial.shape
    player_dim = player.shape[1]

    model = PolyZeroNet(channels, player_dim, map_h, map_w)
    state = {k: v.float() for k, v in load_file(model_path).items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    hard_missing = [k for k in missing if "num_batches_tracked" not in k]
    if hard_missing or unexpected:
        print(f"FAIL: state_dict mismatch: missing={hard_missing} unexpected={list(unexpected)}")
        return 1
    model.eval()

    with torch.no_grad():
        policy, values = model(spatial, player)

    failed = False
    for dump_key, group, py_key in HEADS:
        if dump_key not in dump:
            print(f"{dump_key:16s} SKIP (not in Rust dump)")
            continue
        rust = dump[dump_key].float()
        py = (policy if group == "policy" else values)[py_key]
        if rust.shape != py.shape:
            print(f"{dump_key:16s} FAIL shape rust={tuple(rust.shape)} torch={tuple(py.shape)}")
            failed = True
            continue
        diff = (rust - py).abs().max().item()
        status = "ok" if diff <= TOLERANCE else "FAIL"
        failed |= diff > TOLERANCE
        print(f"{dump_key:16s} {status}  max|diff|={diff:.2e}  rust[0,0]={rust.flatten()[0]:+.5f}  torch[0,0]={py.flatten()[0]:+.5f}")

    print("\nPARITY " + ("FAIL — the two implementations diverge" if failed else f"OK (tolerance {TOLERANCE})"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
