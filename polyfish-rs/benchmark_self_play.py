#!/usr/bin/env python3
"""
Benchmark the self_play loop and log results to journal/self_play_benchmarks.csv
so each optimization step (native arm64 toolchain, eval server, eval cache,
arena-allocated tree, allocation cleanup, ...) can be measured against the
same baseline workload.

Usage:
  python3 benchmark_self_play.py --label baseline_rosetta_x86_64 --build
  python3 benchmark_self_play.py --label native_arm64 --target aarch64-apple-darwin --build
  python3 benchmark_self_play.py --label eval_server --build --notes "after eval server landed"
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(ROOT, "journal", "self_play_benchmarks.csv")
FIELDS = [
    "timestamp", "label", "arch", "target_triple", "backend", "mcts_iters",
    "gumbel_k", "num_games", "games_duration_s", "avg_s_per_game", "avg_moves",
    "total_moves", "moves_per_sec", "cores", "notes",
]


def binary_arch(path):
    out = subprocess.run(["file", path], capture_output=True, text=True).stdout
    if "arm64" in out:
        return "arm64"
    if "x86_64" in out:
        return "x86_64"
    return "unknown"


def clean_env():
    env = os.environ.copy()
    env.pop("CARGO_TARGET_DIR", None)
    return env


def build(target, features):
    cmd = ["cargo", "build", "--release", "--bin", "self_play"]
    if target:
        cmd += ["--target", target]
    if features:
        cmd += ["--features", features]
    print("Building:", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True, env=clean_env())


def binary_path(target):
    if target:
        return os.path.join(ROOT, "target", target, "release", "self_play")
    return os.path.join(ROOT, "target", "release", "self_play")


def run_self_play(binpath, games, mcts, backend, gumbel_k, iteration):
    cmd = [
        binpath, "--num-games", str(games), "--mcts-iters", str(mcts),
        "--search-backend", backend, "--iteration", str(iteration),
    ]
    if backend == "gumbel":
        cmd += ["--gumbel-k", str(gumbel_k)]
    print("Running:", " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=clean_env())
    wall = time.time() - t0
    return proc, wall


def parse_output(text):
    data = {}
    m = re.search(r"Game generation completed in: ([\d.]+)s \((\d+) games\)", text)
    if m:
        data["games_duration_s"] = float(m.group(1))
        data["completed_games"] = int(m.group(2))
    m = re.search(r"Average: ([\d.]+)s per game", text)
    if m:
        data["avg_s_per_game"] = float(m.group(1))
    m = re.search(r"METRICS: (\{.*?\"avg_moves\".*?\})", text)
    if m:
        metrics = json.loads(m.group(1))
        data["avg_moves"] = metrics.get("avg_moves")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="short tag for this optimization step")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--mcts", type=int, default=200)
    ap.add_argument("--backend", choices=["zero", "gumbel"], default="zero")
    ap.add_argument("--gumbel-k", type=int, default=16)
    ap.add_argument("--target", default=None, help="cargo --target triple, e.g. aarch64-apple-darwin")
    ap.add_argument("--features", default=None, help="cargo --features, e.g. metal,accelerate")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    if args.build:
        build(args.target, args.features)

    binpath = binary_path(args.target)
    if not os.path.exists(binpath):
        print(f"ERROR: binary not found at {binpath}. Pass --build or build it first.")
        sys.exit(1)

    arch = binary_arch(binpath)
    cores = os.cpu_count()
    print(f"Binary arch: {arch}  |  cores: {cores}")

    rows = []
    for i in range(args.repeats):
        proc, wall = run_self_play(binpath, args.games, args.mcts, args.backend, args.gumbel_k, iteration=1)
        if proc.returncode != 0:
            print(proc.stdout[-4000:])
            print(proc.stderr[-4000:])
            print(f"ERROR: self_play exited {proc.returncode}")
            sys.exit(1)
        data = parse_output(proc.stdout)
        if "games_duration_s" not in data or "avg_moves" not in data:
            print("ERROR: could not parse expected output from self_play. Raw stdout tail:")
            print(proc.stdout[-2000:])
            sys.exit(1)
        total_moves = data["avg_moves"] * args.games
        moves_per_sec = total_moves / data["games_duration_s"] if data["games_duration_s"] else 0
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "label": args.label,
            "arch": arch,
            "target_triple": args.target or "host",
            "backend": args.backend,
            "mcts_iters": args.mcts,
            "gumbel_k": args.gumbel_k if args.backend == "gumbel" else "",
            "num_games": args.games,
            "games_duration_s": round(data["games_duration_s"], 2),
            "avg_s_per_game": round(data.get("avg_s_per_game", 0), 3),
            "avg_moves": data["avg_moves"],
            "total_moves": round(total_moves, 1),
            "moves_per_sec": round(moves_per_sec, 2),
            "cores": cores,
            "notes": args.notes,
        }
        rows.append(row)
        print(f"Run {i + 1}/{args.repeats}: moves/sec={row['moves_per_sec']}  "
              f"avg_s/game={row['avg_s_per_game']}  games_duration_s={row['games_duration_s']}")

    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    write_header = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"\n✅ Logged {len(rows)} run(s) to {CSV_PATH}")

    with open(CSV_PATH) as f:
        all_rows = list(csv.DictReader(f))
    print("\n=== Benchmark history ===")
    for r in all_rows:
        print(
            f"  {r['timestamp']}  {r['label']:24s} arch={r['arch']:7s} backend={r['backend']:6s} "
            f"mcts={r['mcts_iters']:>4s} games={r['num_games']:>3s}  "
            f"moves/sec={r['moves_per_sec']:>8s}  avg_s/game={r['avg_s_per_game']:>7s}"
        )


if __name__ == "__main__":
    main()
