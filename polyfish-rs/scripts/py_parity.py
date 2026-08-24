#!/usr/bin/env python3
"""Rust↔Python forward parity — the check audit T1 calls the highest-value
missing test in the repo.

`network.rs` (candle) and `train.py` (PyTorch) are two implementations of one
architecture that read and write the same `model.safetensors`. Nothing enforced
that they agree: a mismatch surfaces as a load error if you are lucky and as
silent garbage if you are not. `tch_parity.rs` and `metal_parity.rs` cover
candle-vs-tch and tch-vs-MPS, both of which need libtorch and Apple hardware,
and neither covers this split.

Usage:
    cargo run --no-default-features --example py_parity -- MODEL > /tmp/rust.json
    .venv/bin/python3 scripts/py_parity.py /tmp/rust.json MODEL

Or just `scripts/run_forward_parity.sh`, which does both halves.

Compares RAW logits, not softmaxed ones: softmax is shift-invariant and would
hide a constant logit offset — the same reasoning `tch_parity.rs` records for
its own tolerance.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

import train  # noqa: E402

# Max |Δ| between two CPU implementations of the same graph. Copied from
# tch_parity.rs's TOL_CPU: this is a port-correctness check, not a
# numerical-stability one, so it should be tight.
TOL = 1e-3

MAP_SIZE = 11
SPATIAL_CHANNELS = 142
PLAYER_STATE_DIM = 16

# Mirrors examples/py_parity.rs. Keep the two in step: if they drift apart the
# comparison stops meaning anything rather than failing loudly.
# The index is wrapped before scaling so sin/cos take a small argument; see the
# comment in examples/py_parity.rs for why an unwrapped index makes the harness
# measure its own input.
def spatial_value(i):
    return torch.sin((i % 1009).float() * 0.017)


def player_value(i):
    return torch.cos((i % 251).float() * 0.31)


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: py_parity.py RUST_JSON [MODEL]")
    with open(sys.argv[1]) as f:
        rust = json.load(f)
    model_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "model.safetensors")

    batch = rust["batch"]
    for name, expected, got in (
        ("spatial_channels", SPATIAL_CHANNELS, rust["spatial_channels"]),
        ("player_state_dim", PLAYER_STATE_DIM, rust["player_state_dim"]),
        ("map_size", MAP_SIZE, rust["map_size"]),
    ):
        if expected != got:
            sys.exit(f"FAIL: {name} — train.py says {expected}, Rust says {got}")

    hw = MAP_SIZE * MAP_SIZE
    spatial = spatial_value(torch.arange(batch * SPATIAL_CHANNELS * hw)).reshape(
        batch, SPATIAL_CHANNELS, MAP_SIZE, MAP_SIZE
    )
    player = player_value(torch.arange(batch * PLAYER_STATE_DIM)).reshape(batch, PLAYER_STATE_DIM)

    for name, mine, theirs in (
        ("spatial", float(spatial.double().sum()), rust["spatial_check"]),
        ("player", float(player.double().sum()), rust["player_check"]),
    ):
        if abs(mine - theirs) > 1e-3:
            sys.exit(
                f"FAIL: the two sides built different {name} input "
                f"(sum {mine:.6f} vs {theirs:.6f}) — fix the generator before "
                f"reading anything into an output difference"
            )

    model = train.PolyZeroNet(SPATIAL_CHANNELS, PLAYER_STATE_DIM, MAP_SIZE, MAP_SIZE)
    ckpt = load_file(model_path)
    ckpt = {k: v.float() for k, v in ckpt.items()}  # checkpoints are stored f16
    ckpt, migrations = train._migrate_checkpoint(ckpt, model, PLAYER_STATE_DIM)
    missing, unexpected = model.load_state_dict(ckpt, strict=False)

    # The aux_* heads are training-only and deliberately absent from every Rust
    # backend, so a Rust-written checkpoint legitimately lacks them. Anything
    # else missing means the two definitions have drifted.
    unexplained = [k for k in missing if not k.startswith(("aux_", "v_ownership"))]
    if unexplained:
        sys.exit(f"FAIL: checkpoint is missing non-aux tensors the model needs: {unexplained}")
    if unexpected:
        print(f"note: checkpoint carries {len(unexpected)} tensor(s) the model does not use")
    if migrations:
        print(f"note: migrated checkpoint ({len(migrations)} change(s))")

    model.eval()
    with torch.no_grad():
        out = model(spatial, player)

    # train.py's forward returns (policy_dict, values_dict); accept either that
    # or a flat tuple so this keeps working if the signature is tidied.
    if isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], dict):
        policy, values = out
    else:
        sys.exit(f"FAIL: unrecognised forward() return shape: {type(out)}")

    def flat(t):
        return t.reshape(-1).tolist()

    pairs = [
        ("win", flat(values["win"])),
        ("action_type", flat(policy["action_type"])),
        ("source_spatial", flat(policy["source_spatial"])),
        ("target_spatial", flat(policy["target_spatial"])),
        ("move_option", flat(policy["move_option"])),
    ]
    if "progress" in values:
        pairs.append(("progress", flat(values["progress"])))

    worst_name, worst = None, 0.0
    failures = []
    for name, py in pairs:
        rs = rust.get(name)
        if rs is None:
            failures.append(f"{name}: Rust emitted nothing")
            continue
        if len(rs) != len(py):
            failures.append(f"{name}: length {len(rs)} (Rust) vs {len(py)} (Python)")
            continue
        d = max(abs(a - b) for a, b in zip(rs, py))
        if d > worst:
            worst_name, worst = name, d
        status = "ok " if d <= TOL else "FAIL"
        print(f"  {status} {name:<16} n={len(py):<5} max|delta| = {d:.3e}")
        if d > TOL:
            failures.append(f"{name}: max|delta| {d:.3e} > {TOL:.0e}")

    if failures:
        print("\nFORWARD PARITY FAILED — network.rs and train.py have drifted:")
        for f in failures:
            print(f"  - {f}")
        print("\nA mismatch here means model.safetensors does not round-trip between")
        print("self_play/arena and the trainer. See CLAUDE.md, 'multi-implementation")
        print("sync constraint'.")
        return 1

    print(f"\nforward parity OK (worst: {worst_name} at {worst:.3e}, tol {TOL:.0e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
