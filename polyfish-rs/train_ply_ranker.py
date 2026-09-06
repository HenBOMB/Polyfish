#!/usr/bin/env python3
"""EXP_ELO_131 offline gate: train a decomposed-head ply ranker to replace
rank_plies's per-candidate Δφ scoring (simulate_move + goal_potential_with_belief,
70-80% of macro-mcts actor CPU per EXP_ELO_062/065).

Fixes the two flaws EXP_ELO_065 named in its own postmortem:
  1. Trained as regression (Huber on raw Δφ) then judged by argmax agreement.
     Here: trained AS ranking (listwise softmax cross-entropy over the real
     candidate set of each rank_plies call), never regressed to a scalar.
  2. Undertrained + wrong metric (raw top-1 hit/miss).
     Here: trained to a held-out-loss plateau, evaluated by REGRET
     (true_score(true_top) - true_score(model_top)), not hit/miss.

Reconstructed label per candidate: score_move + lambda*dphi_full (the same
sum rank_plies ranks on; probe logs both terms separately at commit
59b6672 specifically so this reconstruction is possible offline). This is
the pre-EndTurn-revival candidate set -- revival logic stays in Rust either
way (it's a cheap flat-floor override, not something worth learning).

Bar to clear: mean held-out regret must be clearly BELOW the regret of
score_move-alone (dphi==0), because that's what the already-shipped,
already-measured `--macro-rollout-lambda 0.0` flag actually plays (4.52x
throughput, EXP_ELO_065's behavioral A/B). If this head doesn't beat that
bar on Step (74% of volume) and Research (the one type Δφ demonstrably
matters for), it loses to a flag that requires zero new code.

Usage: generate a harvest first (POLYFISH_DPHI_PROBE=<path> self_play ...),
then: python3 train_ply_ranker.py --jsonl <path> --features <path>.features.bin
"""
import argparse
import json
import struct
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MAP_SIZE = 11
SPATIAL = MAP_SIZE * MAP_SIZE  # 121
N_ACTION = 11
N_OPTION = 192


def load_jsonl(path):
    """Group probe rows by call_id. Returns dict[call_id] -> list[row]."""
    calls = defaultdict(list)
    n_rows = 0
    n_bad = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A hard-killed harvest can leave one partial trailing line.
                n_bad += 1
                continue
            calls[row["call_id"]].append(row)
            n_rows += 1
    if n_bad:
        print(f"[load_jsonl] skipped {n_bad} truncated/malformed line(s)", file=sys.stderr)
    print(f"[load_jsonl] {n_rows} rows across {len(calls)} calls", file=sys.stderr)
    return calls


def load_features_bin(path):
    """Parse the binary sidecar: [call_id u64][n_spatial u32][n_player u32]
    [spatial f32 * n_spatial][player f32 * n_player], repeated. Returns
    dict[call_id] -> (spatial: np.ndarray[C,11,11], player: np.ndarray[10])."""
    feats = {}
    with open(path, "rb") as f:
        data = f.read()
    off = 0
    n = len(data)
    truncated = False
    while off < n:
        # A hard-killed harvest process can leave a partial trailing record
        # (SIGKILL mid-write, no fsync/flush chance) -- stop cleanly there
        # instead of crashing; every prior record is still complete and usable.
        if off + 16 > n:
            truncated = True
            break
        call_id = struct.unpack_from("<Q", data, off)[0]
        n_spatial = struct.unpack_from("<I", data, off + 8)[0]
        n_player = struct.unpack_from("<I", data, off + 12)[0]
        record_len = 16 + 4 * n_spatial + 4 * n_player
        if off + record_len > n:
            truncated = True
            break
        spatial_off = off + 16
        player_off = spatial_off + 4 * n_spatial
        spatial = np.frombuffer(data, dtype="<f4", count=n_spatial, offset=spatial_off)
        player = np.frombuffer(data, dtype="<f4", count=n_player, offset=player_off)
        off += record_len
        n_channels = n_spatial // SPATIAL
        # Last write wins if a call_id repeats (shouldn't, but be safe).
        feats[call_id] = (spatial.reshape(n_channels, MAP_SIZE, MAP_SIZE).copy(), player.copy())
    if truncated:
        print(f"[load_features_bin] stopped at a truncated trailing record ({n - off} bytes unread)", file=sys.stderr)
    print(f"[load_features_bin] {len(feats)} feature records", file=sys.stderr)
    return feats


class CallExample:
    """One rank_plies call's candidates, pre-vectorized into numpy arrays so
    scoring is a handful of batched gathers, not a Python loop per candidate
    (the latter was the actual bottleneck: ~50 candidates/call x individual
    MPS-dispatched tensor indexing ops was orders of magnitude slower than
    the trunk forward pass itself)."""

    __slots__ = (
        "spatial", "player", "move_types",
        "action_idx", "source_idx", "has_source", "target_idx", "has_target",
        "option_idx", "has_option", "true_score", "score_move",
    )

    def __init__(self, spatial, player, rows):
        n = len(rows)
        self.spatial = spatial
        self.player = player
        self.move_types = [r["move_type"] for r in rows]
        self.action_idx = np.array([r["action_type"] for r in rows], dtype=np.int64)
        # The Rust probe writes a bare JSON `null` (not the string "null")
        # for an inapplicable head; json.loads parses that as None.
        source_present = np.array([r["source"] is not None for r in rows])
        target_present = np.array([r["target"] is not None for r in rows])
        option_present = np.array([r["option"] is not None for r in rows])
        self.has_source = source_present
        self.has_target = target_present
        self.has_option = option_present
        self.source_idx = np.array(
            [int(r["source"]) if r["source"] is not None else 0 for r in rows], dtype=np.int64
        )
        self.target_idx = np.array(
            [int(r["target"]) if r["target"] is not None else 0 for r in rows], dtype=np.int64
        )
        self.option_idx = np.array(
            [int(r["option"]) if r["option"] is not None else 0 for r in rows], dtype=np.int64
        )
        score_move = np.array([r["score_move"] for r in rows], dtype=np.float32)
        dphi_full = np.array([r["dphi_full"] for r in rows], dtype=np.float32)
        lam = np.array([r["lambda"] for r in rows], dtype=np.float32)
        self.score_move = score_move
        self.true_score = score_move + lam * dphi_full
        assert n == len(self.true_score)


def build_dataset(calls, feats):
    """Returns a list of CallExample, one per call_id present in both the
    JSONL and the feature sidecar."""
    examples = []
    skipped = 0
    for call_id, rows in calls.items():
        if call_id not in feats:
            skipped += 1
            continue
        spatial, player = feats[call_id]
        if len(rows) < 2:
            continue  # nothing to rank
        examples.append(CallExample(spatial, player, rows))
    print(f"[build_dataset] {len(examples)} usable calls ({skipped} missing features)", file=sys.stderr)
    return examples


class TrunkNet(nn.Module):
    """Small conv trunk + 4 decomposed heads, mirroring network.rs's own
    head shapes (pi_action: Linear(filters,11), pi_option: Linear(filters,192),
    pi_source/pi_target: 1x1 conv pooled to spatial). Not the full PolyZeroNet
    (no cross-attention/ResBlocks) -- this is an offline go/no-go gate, not
    the production head; if it clears the bar, Phase 1 ports into network.rs
    proper.
    """

    def __init__(self, n_channels, player_dim, filters=64):
        super().__init__()
        self.player_proj = nn.Linear(player_dim, filters)
        self.stem = nn.Conv2d(n_channels, filters, 3, padding=1)
        self.stem_gn = nn.GroupNorm(8, filters)
        blocks = []
        for _ in range(4):
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(filters, filters, 3, padding=1),
                    nn.GroupNorm(8, filters),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(filters, filters, 3, padding=1),
                    nn.GroupNorm(8, filters),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.action_head = nn.Linear(filters, N_ACTION)
        self.option_head = nn.Linear(filters, N_OPTION)
        self.source_head = nn.Conv2d(filters, 1, 1)
        self.target_head = nn.Conv2d(filters, 1, 1)

    def forward(self, spatial, player):
        # spatial: [B, C, 11, 11], player: [B, player_dim]
        x = self.stem(spatial)
        x = F.relu(self.stem_gn(x))
        p = self.player_proj(player).unsqueeze(-1).unsqueeze(-1)  # [B, filters, 1, 1]
        x = x + p
        for block in self.blocks:
            x = F.relu(x + block(x))
        pooled = x.mean(dim=(2, 3))  # [B, filters]
        action_logits = self.action_head(pooled)  # [B, 11]
        option_logits = self.option_head(pooled)  # [B, 192]
        source_logits = self.source_head(x).flatten(1)  # [B, 121]
        target_logits = self.target_head(x).flatten(1)  # [B, 121]
        return action_logits, source_logits, target_logits, option_logits


def composed_log_probs(action_logits, source_logits, target_logits, option_logits, ex, device):
    """Per-candidate composed log-prob = sum of log_softmax(head)[idx] over
    applicable heads -- the log-domain equivalent of compute_move_priors's
    multiplicative composition (product of independent head probabilities).

    Fully vectorized over the call's whole candidate set: a handful of
    batched gathers, not a Python loop with one GPU-indexed op per candidate
    (measured: the latter made a single epoch over ~17K calls x ~50
    candidates each too slow to finish in a reasonable session)."""
    action_lp = F.log_softmax(action_logits, dim=-1)
    source_lp = F.log_softmax(source_logits, dim=-1)
    target_lp = F.log_softmax(target_logits, dim=-1)
    option_lp = F.log_softmax(option_logits, dim=-1)

    action_idx = torch.from_numpy(ex.action_idx).to(device)
    scores = action_lp[action_idx]

    source_idx = torch.from_numpy(ex.source_idx).to(device)
    has_source = torch.from_numpy(ex.has_source.astype(np.float32)).to(device)
    scores = scores + source_lp[source_idx] * has_source

    target_idx = torch.from_numpy(ex.target_idx).to(device)
    has_target = torch.from_numpy(ex.has_target.astype(np.float32)).to(device)
    scores = scores + target_lp[target_idx] * has_target

    option_idx = torch.from_numpy(ex.option_idx).to(device)
    has_option = torch.from_numpy(ex.has_option.astype(np.float32)).to(device)
    scores = scores + option_lp[option_idx] * has_option

    return scores


def listwise_loss(composed_scores, true_scores, temperature):
    """Softmax cross-entropy: target = softmax(centered true_scores / T),
    prediction = softmax(composed_scores). Per-call centering (subtract the
    call's own mean) is EXP_ELO_065's own flagged-but-never-run next step --
    it removes cross-call scale variance so one temperature works everywhere."""
    centered = true_scores - true_scores.mean()
    target = F.softmax(centered / temperature, dim=-1)
    log_pred = F.log_softmax(composed_scores, dim=-1)
    return -(target * log_pred).sum()


def regret(true_scores, argmax_idx):
    true_top = true_scores.max().item()
    return true_top - true_scores[argmax_idx].item()


def make_batches(examples, batch_size):
    for i in range(0, len(examples), batch_size):
        yield examples[i : i + batch_size]


def batched_forward(model, batch, device):
    """One trunk forward pass for a whole batch of game states -- the
    expensive GPU part. Per-candidate composed scores are then sliced out
    per-example below (cheap indexing, not another GPU dispatch)."""
    sp = torch.from_numpy(np.stack([ex.spatial for ex in batch])).float().to(device)
    pl = torch.from_numpy(np.stack([ex.player for ex in batch])).float().to(device)
    return model(sp, pl)


def evaluate(model, examples, device, temperature, batch_size=64):
    model.eval()
    total_loss = 0.0
    model_regrets = []
    baseline_regrets = []
    by_type_model = defaultdict(list)
    by_type_baseline = defaultdict(list)
    with torch.no_grad():
        for batch in make_batches(examples, batch_size):
            action_logits, source_logits, target_logits, option_logits = batched_forward(model, batch, device)
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
    return np.mean(model_regrets), np.mean(baseline_regrets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=50.0)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-calls", type=int, default=0, help="0 = no cap")
    args = ap.parse_args()

    # Line-buffer stdout even when redirected to a file, so progress is
    # visible while the run is still going instead of only at exit.
    sys.stdout.reconfigure(line_buffering=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    calls = load_jsonl(args.jsonl)
    feats = load_features_bin(args.features)
    examples = build_dataset(calls, feats)
    if args.max_calls > 0 and len(examples) > args.max_calls:
        idx = np.random.permutation(len(examples))[: args.max_calls]
        examples = [examples[i] for i in idx]
        print(f"[main] capped to {len(examples)} calls", file=sys.stderr)

    n_channels = examples[0].spatial.shape[0]
    player_dim = examples[0].player.shape[0]
    print(f"[main] n_channels={n_channels} player_dim={player_dim}", file=sys.stderr)

    idx = np.random.permutation(len(examples))
    n_val = int(len(examples) * args.val_frac)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    train_examples = [examples[i] for i in train_idx]
    val_examples = [examples[i] for i in val_idx]
    print(f"[main] train={len(train_examples)} val={len(val_examples)}", file=sys.stderr)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[main] device={device}", file=sys.stderr)

    model = TrunkNet(n_channels, player_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_regret = float("inf")
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        np.random.shuffle(train_examples)
        total_loss = 0.0
        t0 = time.time()
        for batch in make_batches(train_examples, args.batch_size):
            action_logits, source_logits, target_logits, option_logits = batched_forward(model, batch, device)
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
            model_regret, baseline_regret = evaluate(model, val_examples, device, args.temperature)
            if model_regret < best_val_regret:
                best_val_regret = model_regret
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    print("\n=== FINAL (best checkpoint by val regret) ===")
    model.load_state_dict(best_state)
    evaluate(model, val_examples, device, args.temperature)


if __name__ == "__main__":
    main()
