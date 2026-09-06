#!/usr/bin/env python3
"""Convert train_ply_ranker.py's --golden-vectors .npz into a flat binary
file a Rust test can read with no new crate dependency: for each of the N
examples, in order --
  [n_channels u32][spatial f32 * n_channels*121][player f32 * 10]
  [action_logits f32 * 11][source_logits f32 * 121][target_logits f32 * 121]
  [option_logits f32 * 192]
little-endian throughout. Verifies PlyRanker::forward_raw's Rust port
against the exact logits the Python TrunkNet produced from the same
checkpoint -- the only thing that catches a silent shape/transposition bug
before it reaches self_play.
"""
import struct
import sys

import numpy as np


def main():
    npz_path, out_path = sys.argv[1], sys.argv[2]
    data = np.load(npz_path)
    spatial, player = data["spatial"], data["player"]
    action_logits, source_logits = data["action_logits"], data["source_logits"]
    target_logits, option_logits = data["target_logits"], data["option_logits"]
    n = spatial.shape[0]
    n_channels = spatial.shape[1]
    with open(out_path, "wb") as f:
        for i in range(n):
            f.write(struct.pack("<I", n_channels))
            f.write(spatial[i].astype("<f4").tobytes())
            f.write(player[i].astype("<f4").tobytes())
            f.write(action_logits[i].astype("<f4").tobytes())
            f.write(source_logits[i].astype("<f4").tobytes())
            f.write(target_logits[i].astype("<f4").tobytes())
            f.write(option_logits[i].astype("<f4").tobytes())
    print(f"wrote {n} golden examples ({n_channels} channels) to {out_path}")


if __name__ == "__main__":
    main()
