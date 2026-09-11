"""Compare a candle PolyZeroNet forward against the PyTorch definition.

Rust side (writes inputs + outputs for a fixed pseudo-random batch):
    POLYFISH_PARITY_MODEL=<ckpt.safetensors> POLYFISH_PARITY_OUT=<dump.safetensors> \
        cargo test --lib dump_forward_for_parity -- --ignored --nocapture
    (add POLYFISH_PARITY_DEVICE=cuda on a CUDA build to test that path)

Python side:
    python parity_check.py <train.py> <ckpt.safetensors> <dump.safetensors>

For a July-2026 BatchNorm checkpoint pass the July trainer:
    git show e73713f5:polyfish-rs/train.py > /tmp/train_july.py
"""
import importlib.util
import sys

import torch
from safetensors.torch import load_file

TOL = 1e-4


def main():
    trainpy, ckpt, dump = sys.argv[1:4]
    spec = importlib.util.spec_from_file_location("train_ref", trainpy)
    mod = importlib.util.module_from_spec(spec)
    sys.argv = [trainpy]
    spec.loader.exec_module(mod)

    net = mod.PolyZeroNet(142, 16, 11, 11)
    sd = {k: v.float() for k, v in load_file(ckpt).items()}
    res = net.load_state_dict(sd, strict=False)
    if res.missing_keys or res.unexpected_keys:
        print("missing:", list(res.missing_keys))
        print("unexpected:", list(res.unexpected_keys))
    net.eval()

    d = load_file(dump)
    with torch.no_grad():
        policy, values = net(d["spatial"].float(), d["player"].float())
    pairs = [
        ("action_type", policy["action_type"]),
        ("source_spatial", policy["source_spatial"]),
        ("target_spatial", policy["target_spatial"]),
        ("move_option", policy["move_option"]),
        ("win_value", values["win"]),
        ("progress_value", values["progress"]),
    ]
    worst_abs = worst_rel = 0.0
    for name, ref in pairs:
        got = d[name].float()
        diff = (got - ref).abs().max().item()
        scale = max(ref.abs().max().item(), 1e-6)
        worst_abs = max(worst_abs, diff)
        worst_rel = max(worst_rel, diff / scale)
        print(f"{name:16s} max|diff|={diff:.3e}  max|ref|={scale:.3e}  shape={tuple(ref.shape)}")
    ok = worst_abs < TOL or worst_rel < 1e-5
    print(("PASS" if ok else "FAIL"), f"worst_abs={worst_abs:.3e} worst_rel={worst_rel:.3e}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
