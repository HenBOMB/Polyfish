#!/usr/bin/env python3
"""
Self-play GENERATION throughput & cost estimator (latency-hiding model).

moves/s is an OUTPUT, not an input. It is derived from fundamentals:

Self-play is a latency-hiding pipeline. Each actor loops:
    traverse the tree to a leaf  (CPU: ~t_visit per node-visit)
    submit the leaf, BLOCK on the NN round-trip  (~eval_us, amortized over batching)
    back up, repeat
You run many actors so that while most are parked waiting on the GPU, a few keep
the cores busy producing leaves. The GPU is tiny (<2MB model) and mostly idle
(~2.5% on the M3 Max trace) — the wall is producing leaf requests fast enough.

Two ceilings on node-visits/sec (= simulations/sec); the smaller wins:

    CPU ceiling      = cores * nv_per_core                    (raw tree-walk speed)
    latency-hiding   = actors / (t_visit + eval_us)           (in-flight parallelism)
                       where t_visit = 1e6 / nv_per_core  (us per node-visit)

    node_visits_s = min(CPU ceiling, latency-hiding)
    moves_s       = node_visits_s / budget

On a 14-core laptop you are LATENCY-HIDING bound: cores can't run enough un-parked
actors to fill big batches, so node_visits_s (~36K) sits far below the CPU ceiling
(~280K). Horizontal scaling raises BOTH terms (more physical cores -> more actors
runnable), which is the whole point. Raising the budget divides moves_s directly.

MEASURE the two fundamentals per box:
  nv_per_core : actor_ceiling --actors 1 --mcts-iters <budget>  ->  moves/s * budget
  eval_us     : from a real self_play run, ~ (eval-thread busy_s / forwards) in us,
                or the amortized round-trip an actor waits per leaf. Bigger batches
                (more actors) lower it; it's why the model is tiny on purpose.

Examples:
  python3 throughput_calc.py --budget 128                          # laptop anchor -> ~280 moves/s
  python3 throughput_calc.py --cores 48 --budget 640 --boxes 20 --cost-per-hr 0.30
  python3 throughput_calc.py --cores 48 --actors 400 --eval-us 2500 --budget 640
"""

import argparse

# Avg moves per game, fixed for the 30-turn curriculum we train/measure on.
# From training_log.csv avg_moves: full 30-turn games run ~400-630; 450 is the
# representative constant. Change this one line if the curriculum changes.
MOVES_PER_GAME = 450.0


def fmt(x):
    return f"{x:,.0f}" if abs(x) >= 100 else f"{x:,.2f}"


def main():
    p = argparse.ArgumentParser(
        description="Derive self-play moves/s, games/hr and cost from CPU tree speed + NN latency.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cores", type=float, default=14.0,
                   help="CPU cores doing actor/tree work on the box")
    p.add_argument("--nv-per-core", type=float, default=20000.0,
                   help="node-visits/sec per core = raw tree-walk speed (dummy eval). "
                        "Measure: actor_ceiling --actors 1; ~20K on M3 Max perf cores")
    p.add_argument("--actors", type=float, default=None,
                   help="concurrent actors (in-flight games). Default ~9x cores "
                        "(14 cores -> 128). Capped by oversubscription; scales with cores")
    p.add_argument("--eval-us", type=float, default=3500.0,
                   help="amortized wall-time an actor waits per NN call, microseconds "
                        "(round-trip: coalesce + dispatch + GPU + return). Smaller batches "
                        "-> bigger number. Laptop ~3500us (small, starved batches)")
    p.add_argument("--budget", type=float, default=128.0,
                   help="MCTS iterations (node-visits) per move")
    p.add_argument("--cache", type=float, default=0.15,
                   help="MCTS eval-cache hit rate (visits that skip the NN)")
    p.add_argument("--cost-per-hr", type=float, default=0.0,
                   help="rental cost per box, $/hour (0 = skip cost)")
    p.add_argument("--boxes", type=int, default=1, help="boxes in the fleet")
    args = p.parse_args()

    actors = args.actors if args.actors is not None else round(args.cores * 9)
    if min(args.cores, args.nv_per_core, actors, args.eval_us, args.budget) <= 0:
        p.error("all rate/count inputs must be > 0")
    keep = 1.0 - args.cache

    t_visit_us = 1e6 / args.nv_per_core                       # us per node-visit (one core)
    cpu_ceiling = args.cores * args.nv_per_core                # node-visits/s
    lat_hiding = actors / ((t_visit_us + args.eval_us) * 1e-6) # node-visits/s

    node_visits_s = min(cpu_ceiling, lat_hiding)
    lat_bound = lat_hiding <= cpu_ceiling
    bound = "LATENCY-HIDING (eval wait vs actor count)" if lat_bound else "CPU cores (raw tree speed)"

    moves_s = node_visits_s / args.budget
    leaf_evals_s = node_visits_s * keep
    games_hr = (moves_s / MOVES_PER_GAME) * 3600.0
    n = max(1, args.boxes)

    print("=" * 66)
    print("SELF-PLAY GENERATION ESTIMATE  (latency-hiding, moves/s derived)")
    print("=" * 66)
    print(f"  cores={fmt(args.cores)}  nv/core={fmt(args.nv_per_core)}  actors={fmt(actors)}  "
          f"eval_us={fmt(args.eval_us)}")
    print(f"  budget={fmt(args.budget)}  cache={args.cache:.0%}  moves/game={fmt(MOVES_PER_GAME)} (30-turn const)")
    print("-" * 66)
    print("  node-visits/s ceilings (smaller wins):")
    print(f"    CPU cores      : {fmt(cpu_ceiling):>12}  = {fmt(args.cores)} x {fmt(args.nv_per_core)}")
    print(f"    latency-hiding : {fmt(lat_hiding):>12}  = {fmt(actors)} / ({fmt(t_visit_us)}us + {fmt(args.eval_us)}us)")
    print(f"  >>> BOUND BY: {bound} <<<")
    print("-" * 66)
    print("  PER BOX (derived):")
    print(f"    node-visits/s : {fmt(node_visits_s):>12}")
    print(f"    leaf-evals/s  : {fmt(leaf_evals_s):>12}   (NN demand; GPU can do more -> idle)")
    print(f"    moves/s       : {fmt(moves_s):>12}   = node-visits/s / budget")
    print(f"    games/min     : {fmt(games_hr/60):>12}")
    print(f"    games/hour    : {fmt(games_hr):>12}")
    print(f"    games/day     : {fmt(games_hr*24):>12}")
    if args.cost_per_hr > 0:
        per_1k = args.cost_per_hr / (games_hr / 1000.0) if games_hr > 0 else float("inf")
        print(f"    $/1000 games  : {'$'+format(per_1k, ',.3f'):>12}   (@ ${args.cost_per_hr:.2f}/hr/box)")
    if n > 1:
        print("-" * 66)
        print(f"  FLEET x{n}:")
        print(f"    games/min     : {fmt(games_hr*n/60):>12}")
        print(f"    games/hour    : {fmt(games_hr*n):>12}")
        print(f"    games/day     : {fmt(games_hr*n*24):>12}")
        if args.cost_per_hr > 0:
            print(f"    $/day         : {'$'+format(args.cost_per_hr*n*24, ',.2f'):>12}")
            per_1k = (args.cost_per_hr * n) / (games_hr * n / 1000.0) if games_hr > 0 else float("inf")
            print(f"    $/1000 games  : {'$'+format(per_1k, ',.3f'):>12}")
    print("-" * 66)
    if lat_bound:
        print(f"  note: latency-hiding bound — the CPU ceiling ({fmt(cpu_ceiling)}) is "
              f"{fmt(cpu_ceiling/node_visits_s)}x higher and idle.")
        print("        levers: more actors, lower eval_us (bigger batches), or more boxes.")
    print("  NOTE: this is GENERATION. Your train-only laptop must consume games")
    print("        this fast or they age out — check train.py GPU util.")
    print("=" * 66)


if __name__ == "__main__":
    main()
