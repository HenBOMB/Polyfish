#!/usr/bin/env python3
"""Anchored Elo ratings from the arena match ledger or the strength ladder.

Batch Bradley-Terry maximum-likelihood fit over ALL recorded games, refit from
scratch on every run — no order-dependent K-factor drift. The scale is pinned
by one player that never moves, so ratings stay comparable across runs,
checkpoints, and architecture changes. Draws count as half a win.

Two sources, same fit:
  --source matches  arena --json-out ledger (matches.jsonl), anchor `random`.
  --source ladder   ladder.json's readings, anchor `greedy` (its Elo-0 floor).
The ladder source replaces run_training_loop.sh's chained per-reading win rates
with one joint fit over every gauge, audit and link match ever recorded, with
bootstrap intervals. `tribe_audit` readings are excluded (EXCLUDED_KINDS): they
replay the gauge match on another tribe pair, so pooling them would fold the
block effect the pinned pair exists to remove back into the rating. Ladder rows
carry no elimination flag, so finish% is 0 there.

Usage:
  python3 elo.py fit    [--source ladder] [--ladder ladder.json]
  python3 elo.py fit    [--matches matches.jsonl] [--out elo_ratings.json]
  python3 elo.py report [--ratings elo_ratings.json]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict

MATCHES_PATH = "matches.jsonl"
LADDER_PATH = "ladder.json"
RATINGS_PATH = "elo_ratings.json"

ANCHOR = "random"
LADDER_ANCHOR = "greedy"
ANCHOR_ELO = 0.0
LN10_400 = math.log(10.0) / 400.0
# Virtual draws per played pair (BayesElo-style shrinkage): bounds the MLE for
# sweep results like 16-0 instead of letting the gap run to infinity. Costs a
# downward bias on large gaps (~10% per 350-Elo link at 16 games/pair), which
# shrinks as games accumulate.
VIRTUAL_DRAWS = 0.5
MAX_NEWTON_STEP = 150.0
BOOTSTRAP_REPS = 200


def load_games(path: str) -> list[tuple[str, str, float, bool]]:
    """Ledger rows -> (player1, player2, score-for-player1 in {1, 0.5, 0},
    decisive: ended by elimination rather than score at the turn cap)."""
    games: list[tuple[str, str, float, bool]] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"warning: skipping malformed line {lineno}", file=sys.stderr)
                continue
            a, b = row.get("player1"), row.get("player2")
            result = row.get("result")
            if not a or not b or a == b:
                continue
            res_str = str(result)
            if res_str == "dropped":
                raise ValueError(f"ledger contains dropped/incomplete game on line {lineno}")
            score = {"1": 1.0, "2": 0.0, "draw": 0.5}.get(res_str)
            if score is None:
                continue
            games.append((a, b, score, bool(row.get("decisive", False))))
    return games


def _ladder_node(reading: dict) -> str:
    """run_id-qualified player name; bare `model@iterN` collides across runs."""
    run, model = reading.get("run_id") or "", reading.get("model") or ""
    return f"{run}/{model}" if run else model


# Cross-check readings, not rating evidence: a tribe_audit replays the gauge
# match on another tribe pair, and its games share the (model, anchor) node pair
# with the pinned reading. Pooling them would fold the tribe block effect the pin
# exists to remove straight back into the ladder Elo (#34).
EXCLUDED_KINDS = {"tribe_audit"}


def load_ladder_games(path: str) -> list[tuple[str, str, float, bool]]:
    """ladder.json readings -> the same rows as the arena ledger, expanded from
    each reading's W/D/L. Each anchor is aliased back to the model it was frozen
    from (matched on its link match's iteration) or the graph splits in two at
    every freeze. Ladder rows carry no elimination flag, so finish% is 0."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    readings = [r for r in data.get("readings", []) if r.get("kind") not in EXCLUDED_KINDS]
    links = {
        r.get("iteration"): _ladder_node(r) for r in readings if r.get("kind") == "link"
    }
    alias = {
        a["name"]: links[a.get("frozen_iteration")]
        for a in data.get("anchors", [])
        if a.get("frozen_iteration") in links
    }

    games: list[tuple[str, str, float, bool]] = []
    for r in readings:
        model = _ladder_node(r)
        opponent = alias.get(r.get("opponent"), r.get("opponent"))
        if not model or not opponent or model == opponent:
            continue
        for count, score in (
            (r.get("wins", 0), 1.0),
            (r.get("losses", 0), 0.0),
            (r.get("draws", 0), 0.5),
        ):
            games.extend((model, opponent, score, False) for _ in range(int(count)))
    return games


def _pair_stats(
    games: list[tuple[str, str, float]],
    virtual_draws: float = VIRTUAL_DRAWS,
) -> dict[tuple[str, str], tuple[float, float]]:
    """Collapse games to per-ordered-pair (total score, game count), with
    virtual draws added once per unordered pair."""
    stats: dict[tuple[str, str], tuple[float, float]] = defaultdict(lambda: (0.0, 0.0))
    pairs: set[tuple[str, str]] = set()
    for a, b, s in games:
        sc, n = stats[(a, b)]
        stats[(a, b)] = (sc + s, n + 1.0)
        pairs.add((a, b) if a < b else (b, a))
    for a, b in pairs:
        sc, n = stats[(a, b)]
        stats[(a, b)] = (sc + virtual_draws / 2.0, n + virtual_draws)
    return dict(stats)


def fit_ratings(
    games: list[tuple[str, str, float]],
    warm_start: dict[str, float] | None = None,
    max_sweeps: int = 500,
    tol: float = 1e-3,
    virtual_draws: float = VIRTUAL_DRAWS,
    anchor: str = ANCHOR,
) -> dict[str, float]:
    """Per-player Newton coordinate sweeps on the BT log-likelihood.
    The anchor is held fixed; everything else moves."""
    stats = _pair_stats(games, virtual_draws)
    players: set[str] = set()
    for a, b in stats:
        players.update((a, b))

    r = {p: ANCHOR_ELO for p in players}
    if warm_start:
        for p in players:
            r[p] = warm_start.get(p, ANCHOR_ELO)
    r[anchor] = ANCHOR_ELO

    # opponents[p] = list of (opponent, score for p, games) merging both orders
    opponents: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    for (a, b), (sc, n) in stats.items():
        opponents[a].append((b, sc, n))
        opponents[b].append((a, n - sc, n))

    order = sorted(players)
    for _ in range(max_sweeps):
        max_delta = 0.0
        for p in order:
            if p == anchor:
                continue
            grad = 0.0  # in units of expected-score
            hess = 0.0
            for q, sc, n in opponents[p]:
                e = 1.0 / (1.0 + 10.0 ** ((r[q] - r[p]) / 400.0))
                grad += sc - n * e
                hess += n * e * (1.0 - e)
            if hess <= 0.0:
                continue
            step = grad / (hess * LN10_400)
            step = max(-MAX_NEWTON_STEP, min(MAX_NEWTON_STEP, step))
            r[p] += step
            max_delta = max(max_delta, abs(step))
        if max_delta < tol:
            break
    return r


def connected_to_anchor(games: list[tuple[str, str, float]], anchor: str = ANCHOR) -> set[str]:
    adj: dict[str, set[str]] = defaultdict(set)
    for a, b, _ in games:
        adj[a].add(b)
        adj[b].add(a)
    seen = {anchor}
    stack = [anchor]
    while stack:
        for q in adj[stack.pop()]:
            if q not in seen:
                seen.add(q)
                stack.append(q)
    return seen


def bootstrap_ci(
    games: list[tuple[str, str, float]],
    point: dict[str, float],
    reps: int = BOOTSTRAP_REPS,
    virtual_draws: float = VIRTUAL_DRAWS,
    anchor: str = ANCHOR,
) -> dict[str, tuple[float, float]]:
    samples: dict[str, list[float]] = defaultdict(list)
    rng = random.Random(0)
    n = len(games)
    for _ in range(reps):
        resample = [games[rng.randrange(n)] for _ in range(n)]
        r = fit_ratings(
            resample,
            warm_start=point,
            max_sweeps=100,
            virtual_draws=virtual_draws,
            anchor=anchor,
        )
        for p, v in r.items():
            samples[p].append(v)
    ci: dict[str, tuple[float, float]] = {}
    for p, vals in samples.items():
        vals.sort()
        lo = vals[max(0, int(0.025 * len(vals)) - 1)]
        hi = vals[min(len(vals) - 1, int(0.975 * len(vals)))]
        ci[p] = (lo, hi)
    return ci


def record(games: list[tuple[str, str, float, bool]]) -> dict[str, dict[str, int]]:
    rec: dict[str, dict[str, int]] = defaultdict(
        lambda: {"wins": 0, "draws": 0, "losses": 0, "decisive_wins": 0}
    )
    for a, b, s, decisive in games:
        if s == 1.0:
            rec[a]["wins"] += 1
            rec[b]["losses"] += 1
            if decisive:
                rec[a]["decisive_wins"] += 1
        elif s == 0.0:
            rec[a]["losses"] += 1
            rec[b]["wins"] += 1
            if decisive:
                rec[b]["decisive_wins"] += 1
        else:
            rec[a]["draws"] += 1
            rec[b]["draws"] += 1
    return rec


def print_table(ratings: dict[str, dict], anchor: str = ANCHOR) -> None:
    rows = sorted(ratings.items(), key=lambda kv: -kv[1]["elo"])
    width = max((len(p) for p in ratings), default=6)
    print(
        f"{'player':<{width}}  {'elo':>7}  {'95% CI':>16}  {'games':>5}  "
        f"{'W-D-L':>11}  {'finish%':>7}"
    )
    for p, info in rows:
        lo, hi = info["ci95"]
        wdl = f"{info['wins']}-{info['draws']}-{info['losses']}"
        # % of the player's games won by actually eliminating the opponent —
        # the "can it close out a Domination game" number.
        finish = 100.0 * info.get("decisive_wins", 0) / max(info["games"], 1)
        anchor_mark = "  (anchor)" if p == anchor else ""
        print(
            f"{p:<{width}}  {info['elo']:>7.0f}  [{lo:>6.0f}, {hi:>6.0f}]  "
            f"{info['games']:>5}  {wdl:>11}  {finish:>6.1f}%{anchor_mark}"
        )


def cmd_fit(args: argparse.Namespace) -> None:
    if args.source == "ladder":
        anchor = args.anchor or LADDER_ANCHOR
        if not os.path.exists(args.ladder):
            sys.exit(f"no ladder at {args.ladder} — run a gauge match first")
        games = load_ladder_games(args.ladder)
        source = args.ladder
    else:
        anchor = args.anchor or ANCHOR
        if not os.path.exists(args.matches):
            sys.exit(f"no ledger at {args.matches} — run arena with --json-out first")
        games = load_games(args.matches)
        source = args.matches
    if not games:
        sys.exit(f"{source} contains no usable games")
    bt_games = [(a, b, s) for a, b, s, _ in games]

    players = {p for a, b, _ in bt_games for p in (a, b)}
    if anchor not in players:
        print(
            f"warning: anchor '{anchor}' has no games — the scale floats "
            "(ratings are relative-only until the anchor plays)",
            file=sys.stderr,
        )
    else:
        stranded = players - connected_to_anchor(bt_games, anchor)
        if stranded:
            print(
                f"warning: not connected to the anchor (ratings unreliable): "
                f"{', '.join(sorted(stranded))}",
                file=sys.stderr,
            )

    point = fit_ratings(bt_games, virtual_draws=args.virtual_draws, anchor=anchor)
    ci = (
        bootstrap_ci(
            bt_games,
            point,
            reps=args.bootstrap,
            virtual_draws=args.virtual_draws,
            anchor=anchor,
        )
        if args.bootstrap > 0
        else {}
    )
    rec = record(games)

    out: dict[str, dict] = {}
    for p in sorted(players):
        w = rec[p]
        out[p] = {
            "elo": round(point[p], 1),
            "ci95": [round(v, 1) for v in ci.get(p, (point[p], point[p]))],
            "games": w["wins"] + w["draws"] + w["losses"],
            "wins": w["wins"],
            "draws": w["draws"],
            "losses": w["losses"],
            "decisive_wins": w["decisive_wins"],
        }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"{len(games)} games, {len(players)} players -> {args.out}\n")
    print_table(out, anchor)


def cmd_report(args: argparse.Namespace) -> None:
    if not os.path.exists(args.ratings):
        sys.exit(f"no ratings at {args.ratings} — run `elo.py fit` first")
    with open(args.ratings, encoding="utf-8") as f:
        print_table(json.load(f), args.anchor)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_fit = sub.add_parser("fit", help="refit all ratings from the ledger")
    p_fit.add_argument("--source", choices=["matches", "ladder"], default="matches")
    p_fit.add_argument("--matches", default=MATCHES_PATH)
    p_fit.add_argument("--ladder", default=LADDER_PATH)
    p_fit.add_argument(
        "--anchor",
        default=None,
        help=f"player pinned at 0 Elo (default: {ANCHOR} / {LADDER_ANCHOR} for --source ladder)",
    )
    p_fit.add_argument("--out", default=RATINGS_PATH)
    p_fit.add_argument(
        "--bootstrap",
        type=int,
        default=BOOTSTRAP_REPS,
        help="bootstrap resamples for the 95%% CI (0 disables)",
    )
    p_fit.add_argument(
        "--virtual-draws",
        type=float,
        default=VIRTUAL_DRAWS,
        help="shrinkage: virtual draws added per played pair",
    )
    p_fit.set_defaults(func=cmd_fit)

    p_rep = sub.add_parser("report", help="print the last fitted table")
    p_rep.add_argument("--ratings", default=RATINGS_PATH)
    p_rep.add_argument("--anchor", default=ANCHOR, help="player to mark as the 0-Elo anchor")
    p_rep.set_defaults(func=cmd_report)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
