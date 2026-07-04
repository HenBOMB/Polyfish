#!/usr/bin/env python3
"""
Analyze a self_play generation: value-target distribution + game METRICS,
compared against a historical baseline, to surface stalemate / collapse signals
(the exact check the ML advisor flagged: avg game length + combat counts).

Usage:
  python3 analyze_generation.py <games.safetensors> \\
      --metrics-log <logfile-with-METRICS-line> \\
      [--baseline-log session.log] \\
      [--label "Gumbel-50"]

Outputs go to journal/<timestamp>_*.{png,md} so they group with the
visualize_values.py charts for the same generation.
"""

from safetensors import safe_open
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import re
import os
import sys
from datetime import datetime

JOURNAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "journal")
DEFAULT_BASELINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "session.log")

# METRICS fields that describe a *game* run (as opposed to a training loss line).
GAME_METRIC_FIELDS = [
    "avg_score", "max_score", "avg_moves",
    "p1_avg", "p2_avg",
    "avg_captures", "avg_harvests", "avg_builds", "avg_research", "avg_attacks",
]


def _group_prefix(filename_base):
    parts = filename_base.split("_")
    for part in reversed(parts):
        if part.isdigit():
            return part
    return filename_base


def parse_metrics_lines(path):
    """Return a list of dicts parsed from every 'METRICS: {...}' line in `path`
    that carries game-level fields (i.e. not pure training-loss lines)."""
    if not path or not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("METRICS:"):
                continue
            try:
                data = json.loads(line.replace("METRICS:", "", 1).strip())
            except Exception:
                continue
            if any(k in data for k in ("avg_score", "avg_moves")):
                out.append(data)
    return out


def latest_game_metrics(path):
    rows = parse_metrics_lines(path)
    return rows[-1] if rows else None


def value_bucket_stats(values_np):
    abs_v = np.abs(values_np)
    n = len(values_np)
    buckets = {
        "weak (<0.1)": int(np.sum(abs_v < 0.1)),
        "moderate (0.1-0.3)": int(np.sum((abs_v >= 0.1) & (abs_v < 0.3))),
        "strong (0.3-0.5)": int(np.sum((abs_v >= 0.3) & (abs_v < 0.5))),
        "saturated (>=0.5)": int(np.sum(abs_v >= 0.5)),
    }
    pct = {k: 100.0 * v / n for k, v in buckets.items()}
    return {
        "n": n,
        "mean": float(np.mean(values_np)),
        "std": float(np.std(values_np)),
        "min": float(np.min(values_np)),
        "max": float(np.max(values_np)),
        "near_zero_pct": pct["weak (<0.1)"],
        "saturated_pct": pct["saturated (>=0.5)"],
        "buckets": buckets,
        "pct": pct,
    }


def baseline_stats(path):
    rows = parse_metrics_lines(path)
    stats = {}
    for field in GAME_METRIC_FIELDS:
        vals = [r[field] for r in rows if field in r]
        if not vals:
            continue
        arr = np.array(vals, dtype=float)
        stats[field] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "n": len(vals),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    return stats


def zscore(value, b):
    if not b or b["std"] == 0:
        return None
    return (value - b["mean"]) / b["std"]


def render_report(group_prefix, label, vs, metrics, base):
    os.makedirs(JOURNAL_DIR, exist_ok=True)
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.3)

    # Panel A: value buckets
    ax = fig.add_subplot(gs[0, 0])
    if vs:
        labels = list(vs["pct"].keys())
        pcts = [vs["pct"][k] for k in labels]
        colors = ['#e74c3c', '#2ecc71', '#f39c12', '#e74c3c']
        ax.bar(range(len(labels)), pcts, color=colors, alpha=0.75,
               edgecolor='black', linewidth=1.2)
        for i, p in enumerate(pcts):
            ax.text(i, p, f'{p:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([l.split(' (')[0] for l in labels], fontsize=8, rotation=20, ha='right')
        ax.set_ylabel('% of samples')
        ax.set_title(f'Value targets (n={vs["n"]})', fontweight='bold')
        ax.set_ylim(0, max(pcts) * 1.18)
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'no safetensors', ha='center', transform=ax.transAxes)
        ax.set_axis_off()

    # Panel B: economy metrics (run vs baseline mean)
    ax = fig.add_subplot(gs[0, 1])
    econ = ["avg_score", "avg_research", "avg_harvests", "avg_moves"]
    run_vals = [metrics.get(k, 0) for k in econ]
    base_vals = [base.get(k, {}).get("mean", 0) for k in econ]
    x = np.arange(len(econ))
    ax.bar(x - 0.2, run_vals, 0.4, label=f'{label} (this run)', color='#3498db', alpha=0.8)
    ax.bar(x + 0.2, base_vals, 0.4, label='baseline mean', color='#95a5a6', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(econ, fontsize=8, rotation=20, ha='right')
    ax.set_title('Economy: run vs baseline', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel C: combat metrics (run vs baseline mean) — the stalemate signal
    ax = fig.add_subplot(gs[0, 2])
    combat = ["avg_attacks", "avg_captures", "avg_builds"]
    run_vals = [metrics.get(k, 0) for k in combat]
    base_vals = [base.get(k, {}).get("mean", 0) for k in combat]
    x = np.arange(len(combat))
    ax.bar(x - 0.2, run_vals, 0.4, label=f'{label} (this run)', color='#e74c3c', alpha=0.8)
    ax.bar(x + 0.2, base_vals, 0.4, label='baseline mean', color='#95a5a6', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(combat, fontsize=8, rotation=20, ha='right')
    ax.set_title('Combat: run vs baseline (stalemate check)', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel D: z-scores across all metrics
    ax = fig.add_subplot(gs[1, :2])
    fields = [k for k in GAME_METRIC_FIELDS if k in base and k in metrics]
    zs = [zscore(metrics[k], base[k]) for k in fields]
    colors = ['#e74c3c' if (z is not None and abs(z) >= 2) else '#3498db' for z in zs]
    bars = ax.bar(range(len(fields)), [z if z is not None else 0 for z in zs], color=colors, alpha=0.8)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axhline(2, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.axhline(-2, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_xticks(range(len(fields)))
    ax.set_xticklabels(fields, fontsize=8, rotation=25, ha='right')
    ax.set_ylabel('z-score vs baseline')
    ax.set_title('Per-metric z-score (|z|>=2 flagged red)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel E: P1 vs P2 balance
    ax = fig.add_subplot(gs[1, 2])
    p1 = metrics.get("p1_avg", 0)
    p2 = metrics.get("p2_avg", 0)
    ax.bar(["P1", "P2"], [p1, p2], color=['#3498db', '#e67e22'], alpha=0.8)
    for i, v in enumerate([p1, p2]):
        ax.text(i, v, f'{v:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_title('Player score balance', fontweight='bold')
    ax.set_ylabel('avg score')
    ax.grid(True, alpha=0.3, axis='y')

    title = f'Generation analysis: {label} (ts {group_prefix})'
    fig.suptitle(title, fontsize=15, fontweight='bold')
    png = os.path.join(JOURNAL_DIR, f"{group_prefix}_analysis.png")
    plt.savefig(png, dpi=160, bbox_inches='tight')
    plt.close()
    return png


def build_flags(metrics, base, vs):
    flags = []
    def cmp(metric, ratio, label):
        b = base.get(metric)
        if not b:
            return
        run = metrics.get(metric, 0)
        if b["mean"] > 0 and run < ratio * b["mean"]:
            flags.append(f"⚠ {label}: run={run:.2f} vs baseline={b['mean']:.2f} (<{int(ratio*100)}%)")
    cmp("avg_moves", 0.7, "early termination / short games")
    cmp("avg_attacks", 0.5, "combat collapse (attacks)")
    cmp("avg_captures", 0.5, "combat collapse (captures)")
    cmp("avg_score", 0.7, "economy collapse (score)")
    cmp("avg_research", 0.5, "economy collapse (research)")
    cmp("avg_harvests", 0.5, "economy collapse (harvests)")

    p1 = metrics.get("p1_avg", 0)
    p2 = metrics.get("p2_avg", 0)
    if max(p1, p2) > 0 and abs(p1 - p2) / max(p1, p2) > 0.15:
        flags.append(f"⚠ P1/P2 imbalance: P1={p1:.0f} P2={p2:.0f}")

    if vs and vs["near_zero_pct"] > 30:
        flags.append(f"⚠ high near-zero value targets: {vs['near_zero_pct']:.1f}%")
    return flags


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("safetensors", nargs="?", help="games_*.safetensors (optional with --metrics-log)")
    ap.add_argument("--metrics-log", required=True,
                    help="file containing the METRICS: line for this run")
    ap.add_argument("--baseline-log", default=DEFAULT_BASELINE,
                    help="log with historical METRICS lines (default: session.log)")
    ap.add_argument("--label", default="run")
    args = ap.parse_args()

    metrics = latest_game_metrics(args.metrics_log)
    if not metrics:
        print(f"ERROR: no game METRICS line found in {args.metrics_log}")
        sys.exit(1)

    base = baseline_stats(args.baseline_log)
    n_base = base.get('avg_attacks', {}).get('n', 0)
    print(f"Baseline: {n_base} game-METRICS rows parsed from {args.baseline_log}")

    vs = None
    group_prefix = args.label.replace(" ", "_")
    if args.safetensors:
        if not os.path.exists(args.safetensors):
            print(f"ERROR: {args.safetensors} not found")
            sys.exit(1)
        with safe_open(args.safetensors, framework="pt") as f:
            if "values" not in f.keys():
                print("ERROR: no 'values' tensor")
                sys.exit(1)
            values_np = f.get_tensor("values").cpu().numpy().flatten()
        vs = value_bucket_stats(values_np)
        group_prefix = _group_prefix(os.path.splitext(os.path.basename(args.safetensors))[0])

    print("\n=== Value targets ===")
    if vs:
        print(f"n={vs['n']}  mean={vs['mean']:.3f}  std={vs['std']:.3f}")
        print(f"near-zero (<0.1): {vs['near_zero_pct']:.1f}%   saturated (>=0.5): {vs['saturated_pct']:.1f}%")
        for k, p in vs["pct"].items():
            print(f"  {k}: {p:.1f}%")
    else:
        print("(no safetensors provided)")

    print(f"\n=== Run METRICS ({args.label}) ===")
    for k in GAME_METRIC_FIELDS:
        if k in metrics:
            b = base.get(k)
            zs = zscore(metrics[k], b) if b else None
            zs_s = f"  z={zs:+.2f}" if zs is not None else ""
            bm = f"  baseline={b['mean']:.2f}" if b else ""
            print(f"  {k:14s} {metrics[k]:.2f}{bm}{zs_s}")

    flags = build_flags(metrics, base, vs)
    print("\n=== Flags ===")
    if flags:
        for fl in flags:
            print(fl)
    else:
        print("no flags — run looks within baseline range.")

    png = render_report(group_prefix, args.label, vs, metrics, base)
    print(f"\n✅ Report image: {png}")

    md = os.path.join(JOURNAL_DIR, f"{group_prefix}_analysis.md")
    with open(md, "w") as f:
        f.write(f"# Generation analysis: {args.label} (ts {group_prefix})\n\n")
        f.write(f"generated: {datetime.now().isoformat()}\n\n")
        f.write(f"## Value targets\n\n")
        if vs:
            f.write(f"- n={vs['n']}, mean={vs['mean']:.3f}, std={vs['std']:.3f}\n")
            f.write(f"- near-zero (<0.1): {vs['near_zero_pct']:.1f}%\n")
            f.write(f"- saturated (>=0.5): {vs['saturated_pct']:.1f}%\n\n")
            f.write("| bucket | % |\n|---|---|\n")
            for k, p in vs["pct"].items():
                f.write(f"| {k} | {p:.1f}% |\n")
        else:
            f.write("_(no safetensors)_\n")
        f.write(f"\n## Run METRICS vs baseline (n={base.get('avg_attacks', {}).get('n', 0)})\n\n")
        f.write("| metric | run | baseline mean | z |\n|---|---|---|---|\n")
        for k in GAME_METRIC_FIELDS:
            if k in metrics:
                b = base.get(k)
                zs = zscore(metrics[k], b) if b else None
                f.write(f"| {k} | {metrics[k]:.2f} | {b['mean']:.2f} | "
                        f"{zs:+.2f} |\n" if b else f"| {k} | {metrics[k]:.2f} | — | — |\n")
        f.write("\n## Flags\n\n")
        f.write("\n".join(flags) if flags else "_none_\n")
        f.write(f"\n\n## Image\n\n![analysis]({os.path.basename(png)})\n")
    print(f"✅ Report markdown: {md}")


if __name__ == "__main__":
    main()
