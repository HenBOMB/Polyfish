"""One-off, behavior-preserving migration for the pursuit-representation change.

Two shape changes require migrating model.safetensors (a shape mismatch on a
present key makes train.py silently reinit the whole net, and candle/train.rs
hard-error):

  1. conv1.weight  [64,161,3,3] -> [64,162,3,3]   (new pursuit input channel)
     Appended input channel is ZERO -> no-op at load.

  2. value pool widened 1 -> 8 channels:
       v_pool_conv.weight [1,64,1,1] -> [8,64,1,1],  bias [1] -> [8]
       v_fc_shared.weight [64,121]   -> [64,968]      (8 * 11 * 11)
     Warm-start: channel 0 keeps the old value function exactly; channels 1..7
     get tiny random pool weights but ZERO v_fc_shared columns, so the value
     head output is IDENTICAL at init, then learns to use the extra channels.

Run once from polyfish-rs/ BEFORE the training loop (init_model.py leaves an
existing file untouched). Backs up to model.safetensors.bak.
"""
import os
import sys
import torch
from safetensors.torch import load_file, save_file

NEW_SPATIAL = 162
V_POOL_K = 8
HW = 11 * 11


def migrate(path="model.safetensors"):
    if not os.path.exists(path):
        print(f"{path} not found — nothing to migrate.")
        return
    sd = load_file(path)

    already_conv1 = sd["conv1.weight"].shape[1] == NEW_SPATIAL
    already_vpool = sd["v_pool_conv.weight"].shape[0] == V_POOL_K
    if already_conv1 and already_vpool:
        print("Model already migrated (conv1 162-wide, value pool 8-ch). Skipping.")
        return

    # 1. conv1: append one zero input channel (dim=1).
    if not already_conv1:
        w = sd["conv1.weight"]
        assert w.shape[1] == NEW_SPATIAL - 1, f"unexpected conv1 in-channels: {w.shape}"
        sd["conv1.weight"] = torch.nn.functional.pad(w, (0, 0, 0, 0, 0, 1))
        print(f"conv1.weight {tuple(w.shape)} -> {tuple(sd['conv1.weight'].shape)}")

    # 2. value pool 1 -> 8, warm-started so init output is unchanged.
    if not already_vpool:
        pw = sd["v_pool_conv.weight"]          # [1,64,1,1]
        pb = sd["v_pool_conv.bias"]            # [1]
        fw = sd["v_fc_shared.weight"]          # [64,121]
        assert pw.shape[0] == 1 and fw.shape[1] == HW, f"unexpected value head: {pw.shape}, {fw.shape}"
        filters = pw.shape[1]
        dt = pw.dtype

        extra_pool = (torch.randn(V_POOL_K - 1, filters, 1, 1) * 0.01).to(dt)
        sd["v_pool_conv.weight"] = torch.cat([pw, extra_pool], dim=0)              # [8,64,1,1]
        sd["v_pool_conv.bias"] = torch.cat([pb, torch.zeros(V_POOL_K - 1, dtype=dt)])  # [8]

        zeros_cols = torch.zeros(fw.shape[0], (V_POOL_K - 1) * HW, dtype=fw.dtype)
        sd["v_fc_shared.weight"] = torch.cat([fw, zeros_cols], dim=1)              # [64,968]
        print(f"v_pool_conv.weight -> {tuple(sd['v_pool_conv.weight'].shape)}, "
              f"v_fc_shared.weight -> {tuple(sd['v_fc_shared.weight'].shape)} "
              f"(cols {HW}..{V_POOL_K * HW} zeroed -> identical value output at init)")

    backup = path + ".bak"
    if os.path.exists(backup):
        print(f"Backup {backup} already exists — leaving it as-is.")
    else:
        os.rename(path, backup)
        print(f"Backed up original to {backup}")
    save_file(sd, path)
    print(f"Saved migrated model to {path}")

    # Adam moments no longer match; train.py resets on hash mismatch, but clear
    # it explicitly to be safe. Resolve next to the model file (never cwd) so
    # migrating a copy can't touch an unrelated optimizer_state.pt.
    opt = os.path.join(os.path.dirname(os.path.abspath(path)), "optimizer_state.pt")
    if os.path.exists(opt):
        os.remove(opt)
        print(f"Removed {opt} (Adam state no longer matches migrated shapes)")


if __name__ == "__main__":
    migrate(sys.argv[1] if len(sys.argv) > 1 else "model.safetensors")
