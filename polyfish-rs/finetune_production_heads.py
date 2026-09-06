#!/usr/bin/env python3
"""EXP_ELO_131 follow-up, rung A2: before building brand-new heads (Phase 1
proper), check whether fine-tuning the EXISTING production policy heads
(pi_action/pi_option/pi_source/pi_target + their shared pool) against the
same true_score = score_move + lambda*dphi_full target closes the gap to
EXP_ELO_131's dedicated TrunkNet (54.3 held-out regret).

Rationale: the existing heads are already behavior-cloned on argmax(rank_plies)
(micro_mcts.rs's own comment) -- the same target family as this listwise loss,
just one-hot instead of full-distribution. Aligned objectives, not competing
ones (unlike the macro_stance/value DETACH_MACRO_HEADS split). If this closes
the gap, Phase 1's Rust work is consumer-only: thread &Evaluator into
rank_plies and read the EXISTING RawPolicyOutput fields via
compute_move_priors_raw -- no new network.rs heads, no dual-network-sync
shape change, no new tch/Metal plumbing.

Trunk stays FROZEN (requires_grad=False) on the first pass -- only
p_pool_conv/p_fc_shared/pi_action/pi_option/pi_source/pi_target train. Same
train/val split (seed 0) as train_ply_ranker.py's default, so the reported
val regret is directly comparable to EXP_ELO_131's 54.3.
"""
import argparse
import sys
import time

import numpy as np
import torch

from train_ply_ranker import (
    build_dataset,
    composed_log_probs,
    listwise_loss,
    load_features_bin,
    load_jsonl,
    make_batches,
    regret,
)

sys.path.insert(0, ".")
from train import PolyZeroNet  # noqa: E402

FINE_TUNE_MODULES = ["p_pool_conv", "p_fc_shared", "pi_action", "pi_option", "pi_source", "pi_target"]


def set_trainable(net, trunk_lr):
    train_params, frozen_params = [], []
    for name, p in net.named_parameters():
        top = name.split(".")[0]
        if top in FINE_TUNE_MODULES:
            p.requires_grad_(True)
            train_params.append(p)
        elif trunk_lr > 0.0:
            p.requires_grad_(True)
            frozen_params.append(p)
        else:
            p.requires_grad_(False)
    return train_params, frozen_params


def batched_forward_pz(net, batch, device):
    sp = torch.from_numpy(np.stack([ex.spatial for ex in batch])).float().to(device)
    pl = torch.from_numpy(np.stack([ex.player for ex in batch])).float().to(device)
    policy, _values, _aux = net(sp, pl)
    return policy["action_type"], policy["source_spatial"], policy["target_spatial"], policy["move_option"]


def evaluate_pz(net, examples, device, temperature, batch_size=64):
    net.eval()
    from collections import defaultdict

    total_loss = 0.0
    model_regrets, baseline_regrets = [], []
    by_type_model, by_type_baseline = defaultdict(list), defaultdict(list)
    with torch.no_grad():
        for batch in make_batches(examples, batch_size):
            action_logits, source_logits, target_logits, option_logits = batched_forward_pz(net, batch, device)
            for i, ex in enumerate(batch):
                scores = composed_log_probs(
                    action_logits[i], source_logits[i], target_logits[i], option_logits[i], ex, device
                )
                true_scores = torch.from_numpy(ex.true_score).to(device)
                loss = listwise_loss(scores, true_scores, temperature)
                total_loss += loss.item()
                model_top = int(scores.argmax().item())
                baseline_top = int(np.argmax(ex.score_move))
                true_top_idx = int(true_scores.argmax().item())
                true_top_type = ex.move_types[true_top_idx]
                r_model = regret(true_scores, model_top)
                r_base = regret(true_scores, baseline_top)
                model_regrets.append(r_model)
                baseline_regrets.append(r_base)
                by_type_model[true_top_type].append(r_model)
                by_type_baseline[true_top_type].append(r_base)
    n = len(examples)
    print(f"\n  held-out loss: {total_loss / n:.4f}")
    print(f"  held-out regret -- model: {np.mean(model_regrets):.4f}  baseline(score_move-alone): {np.mean(baseline_regrets):.4f}")
    print(f"  {'move_type':<10} {'n':>6} {'model regret':>14} {'baseline regret':>16}")
    for mt in sorted(by_type_model, key=lambda k: -len(by_type_model[k])):
        n_mt = len(by_type_model[mt])
        print(f"  {mt:<10} {n_mt:>6} {np.mean(by_type_model[mt]):>14.4f} {np.mean(by_type_baseline[mt]):>16.4f}")
    return np.mean(model_regrets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--model", default="model.safetensors")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--trunk-lr", type=float, default=0.0, help="0 = trunk fully frozen")
    ap.add_argument("--temperature", type=float, default=50.0)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    calls = load_jsonl(args.jsonl)
    feats = load_features_bin(args.features)
    examples = build_dataset(calls, feats)

    idx = np.random.permutation(len(examples))
    n_val = int(len(examples) * args.val_frac)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    train_examples = [examples[i] for i in train_idx]
    val_examples = [examples[i] for i in val_idx]
    print(f"[main] train={len(train_examples)} val={len(val_examples)}", file=sys.stderr)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[main] device={device}", file=sys.stderr)

    from safetensors.torch import load_file

    net = PolyZeroNet(169, 10, 11, 11).to(device)
    missing, unexpected = net.load_state_dict(load_file(args.model), strict=False)
    print(f"[main] loaded {args.model}: missing={missing} unexpected={unexpected}", file=sys.stderr)

    train_params, frozen_params = set_trainable(net, args.trunk_lr)
    print(f"[main] fine-tuning {len(train_params)} param tensors"
          f"{f' + {len(frozen_params)} trunk tensors @ lr={args.trunk_lr}' if frozen_params else ' (trunk frozen)'}",
          file=sys.stderr)
    param_groups = [{"params": train_params, "lr": args.lr}]
    if frozen_params:
        param_groups.append({"params": frozen_params, "lr": args.trunk_lr})
    opt = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    print("\n=== BEFORE fine-tuning (frozen production heads) ===")
    evaluate_pz(net, val_examples, device, args.temperature)

    best_val_regret = float("inf")
    best_state = None
    for epoch in range(args.epochs):
        net.train()
        np.random.shuffle(train_examples)
        total_loss = 0.0
        t0 = time.time()
        for batch in make_batches(train_examples, args.batch_size):
            action_logits, source_logits, target_logits, option_logits = batched_forward_pz(net, batch, device)
            losses = []
            for i, ex in enumerate(batch):
                scores = composed_log_probs(
                    action_logits[i], source_logits[i], target_logits[i], option_logits[i], ex, device
                )
                true_scores = torch.from_numpy(ex.true_score).to(device)
                losses.append(listwise_loss(scores, true_scores, args.temperature))
            loss = torch.stack(losses).sum() / len(batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(batch)
        train_loss = total_loss / len(train_examples)
        dt = time.time() - t0
        print(f"epoch {epoch:3d}  train_loss={train_loss:.4f}  ({dt:.1f}s)")

        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            model_regret = evaluate_pz(net, val_examples, device, args.temperature)
            if model_regret < best_val_regret:
                best_val_regret = model_regret
                best_state = {k: v.clone() for k, v in net.state_dict().items()}

    print("\n=== FINAL (best checkpoint by val regret) ===")
    net.load_state_dict(best_state)
    evaluate_pz(net, val_examples, device, args.temperature)


if __name__ == "__main__":
    main()
