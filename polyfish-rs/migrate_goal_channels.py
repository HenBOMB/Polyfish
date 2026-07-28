"""One-off migration for the EXP_ELO_028 goal channels.

conv1.weight [64, <162, 3, 3] -> [64, 168, 3, 3]: appended input channels are
ZERO, so the migrated net is bit-identical at load ("no goal set" semantics).
All Rust backends build conv1 at NUM_CHANNELS=168 and load strictly by shape,
so every model file the new binaries touch must be migrated — model.safetensors
plus checkpoints/ (league opponents, gauge snapshots).

Usage (from polyfish-rs/):
    python migrate_goal_channels.py                 # model.safetensors + checkpoints/
    python migrate_goal_channels.py path [path ...] # specific files or dirs

`checkpoints/bn_era/` is skipped (pre-GN quarantine, unusable anyway).
model.safetensors gets a .bak; checkpoints are padded in place (reversible by
slicing the zero planes back off).
"""
import os
import sys
import torch
from safetensors.torch import load_file, save_file

NEW_SPATIAL = 168


def migrate_file(path, backup=False):
    sd = load_file(path)
    if "conv1.weight" not in sd:
        print(f"  SKIP {path}: no conv1.weight")
        return
    w = sd["conv1.weight"]
    have = w.shape[1]
    if have == NEW_SPATIAL:
        print(f"  ok   {path}: already {NEW_SPATIAL}")
        return
    if have > NEW_SPATIAL:
        raise SystemExit(f"{path}: conv1 in-channels {have} > {NEW_SPATIAL} — refusing")
    sd["conv1.weight"] = torch.nn.functional.pad(w, (0, 0, 0, 0, 0, NEW_SPATIAL - have))
    if backup and not os.path.exists(path + ".bak"):
        os.rename(path, path + ".bak")
    save_file(sd, path)
    print(f"  pad  {path}: {have} -> {NEW_SPATIAL}")


def targets_from(args):
    if args:
        return args
    return ["model.safetensors", "checkpoints"]


def main():
    for t in targets_from(sys.argv[1:]):
        if os.path.isdir(t):
            for name in sorted(os.listdir(t)):
                if name.endswith(".safetensors"):
                    migrate_file(os.path.join(t, name))
        elif os.path.exists(t):
            migrate_file(t, backup=(os.path.basename(t) == "model.safetensors"))
        else:
            print(f"  MISSING {t}")
    opt = "optimizer_state.pt"
    if not sys.argv[1:] and os.path.exists(opt):
        os.remove(opt)
        print(f"Removed {opt} (Adam state no longer matches migrated shapes)")


if __name__ == "__main__":
    main()
