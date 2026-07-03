#!/usr/bin/env python3
"""
Chart training_log.csv: loss, score, player balance, and per-game move stats.

Usage:
    .venv/bin/python3 plot_training.py [--csv training_log.csv] [--out training_progress.png] [--show]
"""

import argparse
import csv as csv_module
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLUMNS = [
    "iteration",
    "timestamp",
    "avg_score",
    "max_score",
    "p1_avg",
    "p2_avg",
    "loss",
    "avg_captures",
    "avg_harvests",
    "avg_builds",
    "avg_research",
    "avg_attacks",
]


def load_rows(path):
    rows = []
    with open(path, newline="") as f:
        for line in csv_module.reader(f):
            if not line:
                continue
            rows.append([float(x) for x in line])
    if not rows:
        print(f"No rows found in {path}", file=sys.stderr)
        sys.exit(1)
    return {name: [row[i] for row in rows] for i, name in enumerate(COLUMNS)}


def plot(data, out_path, show):
    it = data["iteration"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Training progress")

    for ax_row in axes:
        for ax in ax_row:
            ax.grid(True, linestyle="-", color="#cccccc", alpha=0.5)

    axes[0][0].plot(it, data["loss"], label="loss")
    axes[0][0].set_title("Loss")
    axes[0][0].set_xlabel("iteration")

    axes[0][1].plot(it, data["avg_score"], label="avg score")
    axes[0][1].plot(it, data["max_score"], label="max score")
    axes[0][1].set_title("Score")
    axes[0][1].set_xlabel("iteration")
    axes[0][1].legend()

    axes[1][0].plot(it, data["p1_avg"], label="P1 avg")
    axes[1][0].plot(it, data["p2_avg"], label="P2 avg")
    axes[1][0].set_title("Player balance")
    axes[1][0].set_xlabel("iteration")
    axes[1][0].legend()

    for key, label in [
        ("avg_captures", "captures"),
        ("avg_harvests", "harvests"),
        ("avg_builds", "builds"),
        ("avg_research", "research"),
        ("avg_attacks", "attacks"),
    ]:
        axes[1][1].plot(it, data[key], label=label)
    axes[1][1].set_title("Moves per game (by type)")
    axes[1][1].set_xlabel("iteration")
    axes[1][1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    print(f"Saved {out_path}")

    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="training_log.csv")
    parser.add_argument("--out", default="training_progress.png")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    data = load_rows(args.csv)
    plot(data, args.out, args.show)


if __name__ == "__main__":
    main()
