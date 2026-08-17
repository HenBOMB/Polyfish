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

  # Eval-server knobs (forwarded to self_play). max_turns is a flat 50
  # regardless of --iteration (self_play's turn-count curriculum was
  # dropped); --iteration still drives the heuristic-prior/anchor decay.
  python3 benchmark_self_play.py --label eval_server_gumbel_64 \
      --backend gumbel --gumbel-k 16 --mcts 64 --games 128 --iteration 200 \
      --actors 96 --max-batch 256 --coalesce-timeout-us 1000 --leaf-batch 4 \
      --build --repeats 3 --features metal,accelerate
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
    # Eval-server config (blank for pre-eval-server historical rows).
    "actors", "max_batch", "coalesce_timeout_us", "leaf_batch", "iteration",
    # Measured eval-server behavior, parsed from EVAL_SERVER_STATS_AGG output:
    # average coalesced batch size and fraction of wall time the inference
    # thread spent inside tensorize/forward/readback.
    "avg_eval_batch", "eval_busy_frac",
    # Eval-cache hit rate (fraction of leaf rows served from the LRU without
    # hitting the GPU). Blank for pre-cache historical rows.
    "cache_hit_rate",
]


def migrate_csv_header():
    """Reorder/extend an existing benchmark CSV to the current FIELDS schema.

    Rewrites the file in FIELDS order: columns absent from the old header get
    blank values for every existing row, and any unknown old columns are
    dropped. Lets new eval-server columns be added without invalidating the
    historical baseline rows already on disk.
    """
    if not os.path.exists(CSV_PATH):
        return
    with open(CSV_PATH, newline="") as f:
        existing = list(csv.reader(f))
    if not existing:
        return
    header = existing[0]
    if header == FIELDS:
        return
    idx = {name: i for i, name in enumerate(header)}
    rows = existing[1:]
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        for r in rows:
            w.writerow([r[idx[name]] if name in idx and idx[name] < len(r) else ""
                        for name in FIELDS])
    print(f"Migrated CSV header to {len(FIELDS)} columns (was {len(header)}).")


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


def run_self_play(binpath, games, mcts, backend, gumbel_k, iteration,
                  actors=None, max_batch=None, coalesce_timeout_us=None,
                  leaf_batch=None, cache_cap=None):
    cmd = [
        binpath, "--num-games", str(games), "--mcts-iters", str(mcts),
        "--search-backend", backend, "--iteration", str(iteration),
    ]
    if backend == "gumbel":
        cmd += ["--gumbel-k", str(gumbel_k)]
    # Eval-server knobs: forwarded only when set, so a bare invocation keeps
    # self_play's compiled defaults and stays comparable to historical rows.
    if actors is not None:
        cmd += ["--actors", str(actors)]
    if max_batch is not None:
        cmd += ["--max-batch", str(max_batch)]
    if coalesce_timeout_us is not None:
        cmd += ["--coalesce-timeout-us", str(coalesce_timeout_us)]
    if leaf_batch is not None:
        cmd += ["--leaf-batch", str(leaf_batch)]
    if cache_cap is not None:
        cmd += ["--cache-cap", str(cache_cap)]
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
    metrics_path = os.path.join(ROOT, ".last_self_play_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
        data["avg_moves"] = metrics.get("avg_moves")
    else:
        m = re.search(r"METRICS: (\{.*?\"avg_moves\".*?\})", text)
        if m:
            metrics = json.loads(m.group(1))
            data["avg_moves"] = metrics.get("avg_moves")
    # One line per eval server; benchmark runs are same-weights self-play, so
    # there is exactly one. If an --opponent run ever emits two, take server 1.
    m = re.search(r"EVAL_SERVER_STATS_AGG: (\{.*?\})", text)
    if not m:
        m = re.search(r"EVAL_SERVER_STATS: (\{.*?\})", text)
    if m:
        stats = json.loads(m.group(1))
        data["avg_eval_batch"] = stats.get("avg_batch")
        data["eval_busy_frac"] = stats.get("busy_frac")
        data["cache_hit_rate"] = stats.get("cache_hit_rate")
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
    # Eval-server knobs. Forwarded to self_play only when set; when omitted,
    # self_play's compiled defaults apply so old baselines stay reproducible.
    ap.add_argument("--actors", type=int, default=None,
                    help="concurrent game actors; forwarded to self_play --actors")
    ap.add_argument("--max-batch", type=int, default=None,
                    help="eval-server batch cap; forwarded to self_play --max-batch")
    ap.add_argument("--coalesce-timeout-us", type=int, default=None,
                    help="eval-server flush timeout in microseconds")
    ap.add_argument("--leaf-batch", type=int, default=None,
                    help="per-game virtual-loss mini-batch size")
    ap.add_argument("--cache-cap", type=int, default=None,
                    help="eval-cache LRU capacity forwarded to self_play --cache-cap "
                         "(0 disables; default keeps self_play's compiled 524288)")
    ap.add_argument("--iteration", type=int, default=1,
                    help="training iteration passed to self_play; drives the "
                         "heuristic-prior/anchor decay ramps (max_turns is a flat 50)")
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
        proc, wall = run_self_play(
            binpath, args.games, args.mcts, args.backend, args.gumbel_k,
            iteration=args.iteration,
            actors=args.actors, max_batch=args.max_batch,
            coalesce_timeout_us=args.coalesce_timeout_us,
            leaf_batch=args.leaf_batch,
            cache_cap=args.cache_cap,
        )
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
            "actors": args.actors if args.actors is not None else "",
            "max_batch": args.max_batch if args.max_batch is not None else "",
            "coalesce_timeout_us": args.coalesce_timeout_us if args.coalesce_timeout_us is not None else "",
            "leaf_batch": args.leaf_batch if args.leaf_batch is not None else "",
            "iteration": args.iteration,
            "avg_eval_batch": data.get("avg_eval_batch", ""),
            "eval_busy_frac": data.get("eval_busy_frac", ""),
            "cache_hit_rate": data.get("cache_hit_rate", ""),
        }
        rows.append(row)
        print(f"Run {i + 1}/{args.repeats}: moves/sec={row['moves_per_sec']}  "
              f"avg_s/game={row['avg_s_per_game']}  games_duration_s={row['games_duration_s']}  "
              f"avg_eval_batch={row['avg_eval_batch']}  eval_busy_frac={row['eval_busy_frac']}  "
              f"cache_hit_rate={row['cache_hit_rate']}")

    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    migrate_csv_header()
    write_header = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"\n✅ Logged {len(rows)} run(s) to {CSV_PATH}")

    with open(CSV_PATH) as f:
        all_rows = list(csv.DictReader(f))
    print("\n=== Benchmark history ===")
    for r in all_rows:
        eval_cfg = []
        if r.get("actors"):
            eval_cfg.append(f"actors={r['actors']}")
        if r.get("max_batch"):
            eval_cfg.append(f"maxb={r['max_batch']}")
        if r.get("leaf_batch"):
            eval_cfg.append(f"lb={r['leaf_batch']}")
        if r.get("iteration") and r["iteration"] != "1":
            eval_cfg.append(f"iter={r['iteration']}")
        eval_str = ("  " + " ".join(eval_cfg)) if eval_cfg else ""
        print(
            f"  {r['timestamp']}  {r['label']:24s} arch={r['arch']:7s} backend={r['backend']:6s} "
            f"mcts={r['mcts_iters']:>4s} games={r['num_games']:>3s}  "
            f"moves/sec={r['moves_per_sec']:>8s}  avg_s/game={r['avg_s_per_game']:>7s}"
            f"{eval_str}"
        )


if __name__ == "__main__":
    main()
