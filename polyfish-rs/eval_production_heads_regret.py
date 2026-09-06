#!/usr/bin/env python3
"""EXP_ELO_131 follow-up: before porting a NEW distilled head into network.rs
(Phase 1, multi-session), check whether the EXISTING production policy heads
already do the job for free.

`micro_mcts.rs`'s own comment says these heads are "already behavior-cloned
on macro-mcts's own committed picks" -- i.e. argmax(rank_plies), the same
target family EXP_ELO_131 trained against (just listwise instead of one-hot).
So this is the cheapest possible way to close the loop: reuse the harvest
and the regret metric from train_ply_ranker.py, swap in model.safetensors's
real action/source/target/option logits instead of a freshly trained trunk,
and see where it lands relative to 54.3 (EXP_ELO_131) and 181.3 (score_move-
alone, the `--macro-rollout-lambda 0.0` bar).

Usage: python3 eval_production_heads_regret.py --jsonl <probe.jsonl> \
    --features <probe.jsonl.features.bin> [--model model.safetensors]
"""
import argparse
import sys

import torch
import torch.nn as nn

from train_ply_ranker import build_dataset, evaluate, load_features_bin, load_jsonl

sys.path.insert(0, ".")
from train import PolyZeroNet  # noqa: E402


class ProductionHeadsAdapter(nn.Module):
    """Presents PolyZeroNet's policy heads through TrunkNet's 4-tuple
    interface so `evaluate()`/`batched_forward()` need no changes."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, spatial, player):
        policy, _values, _aux = self.net(spatial, player)
        return (
            policy["action_type"],
            policy["source_spatial"],
            policy["target_spatial"],
            policy["move_option"],
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--model", default="model.safetensors")
    ap.add_argument("--temperature", type=float, default=50.0)
    ap.add_argument("--max-calls", type=int, default=0, help="0 = no cap")
    args = ap.parse_args()

    sys.stdout.reconfigure(line_buffering=True)

    calls = load_jsonl(args.jsonl)
    feats = load_features_bin(args.features)
    examples = build_dataset(calls, feats)
    if args.max_calls > 0 and len(examples) > args.max_calls:
        import numpy as np

        idx = np.random.permutation(len(examples))[: args.max_calls]
        examples = [examples[i] for i in idx]
        print(f"[main] capped to {len(examples)} calls", file=sys.stderr)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[main] device={device}", file=sys.stderr)

    from safetensors.torch import load_file

    MAP_SIZE, SPATIAL_CHANNELS, PLAYER_STATE_DIM = 11, 169, 10
    net = PolyZeroNet(SPATIAL_CHANNELS, PLAYER_STATE_DIM, MAP_SIZE, MAP_SIZE).to(device)
    state = load_file(args.model)
    missing, unexpected = net.load_state_dict(state, strict=False)
    print(f"[main] loaded {args.model}: missing={len(missing)} unexpected={len(unexpected)}", file=sys.stderr)
    model = ProductionHeadsAdapter(net).to(device)

    print(f"[main] {len(examples)} calls, evaluating production heads (no training)")
    evaluate(model, examples, device, args.temperature)


if __name__ == "__main__":
    main()
