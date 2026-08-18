#!/usr/bin/env python3
"""Strength-ladder store (ladder.json) and gauge verdicts — EXP 10/11.

Anchors are frozen model files (greedy = Elo 0 floor); the last anchor is
"active". `record --kind gauge` appends an arena reading and answers:
continue / freeze (>=80% vs active) / stop (plateau, see _plateau).
Win/loss counts are always from the current model's side. Every reading
carries a Wilson interval and both verdicts are drawn from it, not from the
point estimate a ~64-game reading resolves to only +/-12pp.
"""
import argparse
import json
import math
import os
from datetime import datetime, timezone

LADDER_FILE = os.environ.get("LADDER_FILE", "ladder.json")
FREEZE_WR = 0.80
PLATEAU_WINDOW = 8  # gauge readings vs the same anchor (= 80 iters at interval 10)
PLATEAU_STRIKES = 2  # consecutive flagged readings before the loop stops
CI_Z = 1.96  # 95%
# The effect size the registered experiment bars are written against (EXP_ELO_002
# used +8pp). Readings whose own resolution is coarser than this get flagged:
# a 64-game reading resolves to ~+/-11pp and cannot adjudicate +8pp on its own.
MIN_DETECTABLE_EFFECT = float(os.environ.get("GAUGE_MIN_EFFECT", "0.08"))


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load():
    if os.path.exists(LADDER_FILE):
        with open(LADDER_FILE) as f:
            return json.load(f)
    return {
        "anchors": [
            {"name": "greedy", "path": "", "elo": 0.0, "frozen_iteration": None, "frozen_at": None}
        ],
        "readings": [],
        "plateau_strikes": 0,
    }


def _save(data):
    tmp = LADDER_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.replace(tmp, LADDER_FILE)


def _win_rate(wins, losses, draws):
    games = wins + losses + draws
    return (wins + 0.5 * draws) / games if games else 0.0


def _wilson(win_rate, games, z=CI_Z):
    """Wilson score interval for a win rate. Unlike the normal approximation it
    stays inside [0, 1] and keeps its coverage near 0 and 1, which is where the
    freeze bar (0.80) and the greedy-anchor readings sit."""
    if games <= 0:
        return [0.0, 1.0]
    p = min(max(win_rate, 0.0), 1.0)
    d = 1.0 + z * z / games
    center = (p + z * z / (2.0 * games)) / d
    half = z * math.sqrt(p * (1.0 - p) / games + z * z / (4.0 * games * games)) / d
    return [round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)]


def _half_width(win_rate, games, z=CI_Z):
    """Half-width of the Wilson interval, in percentage points. This is the
    resolution of a reading: the smallest difference it can adjudicate."""
    lo, hi = _wilson(win_rate, games, z)
    return round(100.0 * (hi - lo) / 2.0, 2)


def _z_from_tail(tail):
    """Inverse standard normal at an upper-tail probability, via the Beasley-
    Springer-Moro rational approximation. Avoids a scipy dependency — this file
    is called from the training loop, which pins no scientific stack."""
    p = 1.0 - tail
    if not 0.0 < p < 1.0:
        raise ValueError("tail must be in (0, 1)")
    a = [-39.69683028665376, 220.9460984245205, -275.9285104469687,
         138.3577518672690, -30.66479806614716, 2.506628277459239]
    b = [-54.47609879822406, 161.5858368580409, -155.6989798598866,
         66.80131188771972, -13.28068155288572]
    c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.938163982698783]
    d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996,
         3.754408661907416]
    lo, hi = 0.02425, 1.0 - 0.02425
    if p < lo:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > hi:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def required_games(baseline, delta, power=0.80, alpha=0.05):
    """Games per reading needed to call a `delta` change from `baseline` at
    `power`, two-sided `alpha`, comparing two independent readings.

    This is the number M3 asks for: size the budget against the effect you
    actually want to detect, instead of reading a verdict off an interval that
    was never wide enough to carry one. Unpaired, so it is the conservative
    figure — a paired analysis over the seeded map set (both readings now run
    the same seeds) needs fewer, by a factor that depends on how correlated the
    per-seed outcomes are, which nothing measures yet.
    """
    p0 = min(max(baseline, 1e-6), 1.0 - 1e-6)
    p1 = min(max(baseline + delta, 1e-6), 1.0 - 1e-6)
    if p0 == p1:
        return None
    z_a = _z_from_tail(alpha / 2.0)
    z_b = _z_from_tail(1.0 - power)
    pbar = (p0 + p1) / 2.0
    num = z_a * math.sqrt(2.0 * pbar * (1.0 - pbar)) + \
        z_b * math.sqrt(p0 * (1.0 - p0) + p1 * (1.0 - p1))
    return math.ceil(num * num / ((p1 - p0) ** 2))


def _counts(reading):
    """(score, games) for a reading, draws counted as half a win. Readings
    written before the counts existed fall back to win_rate x games."""
    games = reading.get("games")
    if games is None:
        games = reading.get("wins", 0) + reading.get("losses", 0) + reading.get("draws", 0)
    if "wins" in reading:
        return reading["wins"] + 0.5 * reading.get("draws", 0), games
    return reading.get("win_rate", 0.0) * games, games


def _pool(readings):
    """(win_rate, games) over a group of readings, as one combined sample."""
    score = sum(_counts(r)[0] for r in readings)
    games = sum(_counts(r)[1] for r in readings)
    return (score / games if games else 0.0), games


def _overlap(a, b):
    return a[0] <= b[1] and b[0] <= a[1]


def _elo(win_rate, base):
    p = min(max(win_rate, 0.005), 0.995)
    return round(base + 400.0 * math.log10(p / (1.0 - p)), 1)


def _anchor_by_name(data, name):
    for a in data["anchors"]:
        if a["name"] == name:
            return a
    raise SystemExit(f"unknown anchor: {name}")


def _gauge_series(data):
    active = data["anchors"][-1]["name"]
    return [r for r in data["readings"] if r["kind"] == "gauge" and r["opponent"] == active]


def _plateau(series):
    """True when the last PLATEAU_WINDOW readings vs the same anchor show no
    measurable gain: pooling each half and comparing intervals, the second
    half's interval still overlaps the first's, so any apparent movement is
    inside the noise a single ~64-game reading cannot resolve."""
    if len(series) < PLATEAU_WINDOW:
        return False
    window = series[-PLATEAU_WINDOW:]
    half = PLATEAU_WINDOW // 2
    first, second = _pool(window[:half]), _pool(window[half:])
    return _overlap(_wilson(*first), _wilson(*second))


def cmd_active(_args):
    data = _load()
    print(json.dumps(data["anchors"][-1]))


def cmd_audit_opponents(_args):
    """Greedy (when not active) + one retired net anchor, rotated per audit."""
    data = _load()
    active = data["anchors"][-1]
    opponents = []
    if active["name"] != "greedy":
        opponents.append(_anchor_by_name(data, "greedy"))
    retired_nets = [a for a in data["anchors"][:-1] if a["name"] != "greedy"]
    if retired_nets:
        retired_names = {a["name"] for a in retired_nets}
        n_audits = sum(
            1 for r in data["readings"]
            if r["kind"] == "audit" and r["opponent"] in retired_names
        )
        opponents.append(retired_nets[n_audits % len(retired_nets)])
    print(json.dumps(opponents))


def _sample_at(samples, turn):
    """Last per-turn sample with sample.turn <= turn (None if none)."""
    best = None
    for s in samples:
        if s["turn"] <= turn:
            best = s
        else:
            break
    return best


# Turn milestones for behavior curves (matches the CSV's SPT milestones).
BEHAVIOR_TURNS = [5, 10, 15, 20, 25]
BEHAVIOR_METRICS = ["score", "spt", "cities", "units", "unit_cost", "techs"]


def _summarize_stats(stats_dir):
    """Mean per-metric curves at BEHAVIOR_TURNS from an arena --dump-stats-dir
    directory (config 1 = the model). Deliberately threshold-free so it stays
    meaningful across map sizes; threshold questions (Nth city by turn T) are
    analysis-time queries over the raw dumps, which are retained. Returns None
    when the dir is missing/empty so dump-less calls stay unchanged."""
    import glob

    files = sorted(glob.glob(os.path.join(stats_dir, "game_*.json")))
    if not files:
        return None
    acc = {
        m: {side: [[] for _ in BEHAVIOR_TURNS] for side in ("model", "opp")}
        for m in BEHAVIOR_METRICS
    }
    for path in files:
        with open(path) as f:
            samples = json.load(f)["samples"]
        for ti, turn in enumerate(BEHAVIOR_TURNS):
            s = _sample_at(samples, turn)
            if s is None:
                continue
            for m in BEHAVIOR_METRICS:
                acc[m]["model"][ti].append(s[m][0])
                acc[m]["opp"][ti].append(s[m][1])

    mean = lambda xs: round(sum(xs) / len(xs), 2) if xs else None
    out = {"games": len(files), "turns": BEHAVIOR_TURNS}
    for m in BEHAVIOR_METRICS:
        out[m] = {side: [mean(v) for v in acc[m][side]] for side in ("model", "opp")}
    return out


def _append_reading(data, args, kind, opponent):
    win_rate = round(_win_rate(args.wins, args.losses, args.draws), 4)
    games = args.wins + args.losses + args.draws
    ci = _wilson(win_rate, games)
    reading = {
        "at": _now(),
        "run_id": args.run_id,
        "iteration": args.iteration,
        "kind": kind,
        "model": f"model@iter{args.iteration}",
        "opponent": opponent["name"],
        "games": games,
        "wins": args.wins,
        "losses": args.losses,
        "draws": args.draws,
        "win_rate": win_rate,
        "win_rate_ci": ci,
        "ci_level": 0.95,
        # Half-width of that interval in pp: the smallest difference this
        # reading can adjudicate. Recorded per reading so a verdict drawn from
        # a smaller difference is visibly unsupported (audit M3).
        "resolves_pp": _half_width(win_rate, games),
        "elo_est": _elo(win_rate, opponent["elo"]),
        "elo_ci": [_elo(ci[0], opponent["elo"]), _elo(ci[1], opponent["elo"])],
        "avg_score_model": args.avg_score_model,
        "avg_score_opponent": args.avg_score_opponent,
    }
    if getattr(args, "mcts", None) is not None:
        reading["budget"] = {
            "mcts": args.mcts,
            "gumbel_k": args.gumbel_k,
            "eval_backend": args.eval_backend,
        }
    if getattr(args, "wins_p1", None) is not None:
        reading["wins_as_p1"] = args.wins_p1
        reading["wins_as_p2"] = args.wins_p2
    if getattr(args, "stats_dir", None):
        behavior = _summarize_stats(args.stats_dir)
        if behavior is not None:
            reading["behavior"] = behavior
    data["readings"].append(reading)
    return reading


def cmd_record(args):
    data = _load()
    if args.kind == "gauge":
        opponent = data["anchors"][-1]
    else:
        opponent = _anchor_by_name(data, args.opponent)
    reading = _append_reading(data, args, args.kind, opponent)

    action = "continue"
    if args.kind == "gauge":
        # The freeze bar is on the lower bound: a point estimate at 0.80 with a
        # +/-0.12 interval is not evidence the model beats the anchor 4:1.
        if reading["win_rate_ci"][0] >= FREEZE_WR:
            action = "freeze"
            data["plateau_strikes"] = 0
        elif _plateau(_gauge_series(data)):
            data["plateau_strikes"] += 1
            if data["plateau_strikes"] >= PLATEAU_STRIKES:
                action = "stop"
        else:
            data["plateau_strikes"] = 0
    _save(data)
    verdict = {
        "action": action,
        "opponent": opponent["name"],
        "win_rate": reading["win_rate"],
        "win_rate_ci": reading["win_rate_ci"],
        "resolves_pp": reading["resolves_pp"],
        "elo_est": reading["elo_est"],
        "elo_ci": reading["elo_ci"],
        "plateau_strikes": data["plateau_strikes"],
    }
    # A single reading this size cannot carry a verdict about a difference
    # smaller than its own resolution. Say so on every reading rather than
    # leaving the next reader to rediscover it from the interval.
    if reading["resolves_pp"] > 100.0 * MIN_DETECTABLE_EFFECT:
        verdict["underpowered_for_pp"] = round(100.0 * MIN_DETECTABLE_EFFECT, 1)
        verdict["games_needed"] = required_games(reading["win_rate"], MIN_DETECTABLE_EFFECT)
    if "behavior" in reading:
        b = reading["behavior"]
        verdict["cities_curve"] = {
            "turns": b["turns"], "model": b["cities"]["model"], "opp": b["cities"]["opp"]
        }
    print(json.dumps(verdict))


def cmd_freeze(args):
    """Register a new anchor from the link-match result vs the outgoing one."""
    data = _load()
    outgoing = data["anchors"][-1]
    link_wr = _win_rate(args.wins, args.losses, args.draws)
    link_ci = _wilson(link_wr, args.wins + args.losses + args.draws)
    new_anchor = {
        "name": os.path.splitext(os.path.basename(args.path))[0],
        "path": args.path,
        "elo": _elo(link_wr, outgoing["elo"]),
        # Link-match uncertainty, so the chain's accumulated error stays visible.
        "elo_ci": [_elo(link_ci[0], outgoing["elo"]), _elo(link_ci[1], outgoing["elo"])],
        "frozen_iteration": args.iteration,
        "frozen_at": _now(),
    }
    _append_reading(data, args, "link", outgoing)
    data["anchors"].append(new_anchor)
    data["plateau_strikes"] = 0
    _save(data)
    print(json.dumps(new_anchor))


def cmd_power(args):
    """Answer 'how many games do I need' before spending the compute, and
    'what could this reading have detected' after."""
    out = {
        "baseline": args.baseline,
        "effect_pp": round(100.0 * args.effect, 2),
        "power": args.power,
        "alpha": args.alpha,
        "games_per_reading": required_games(args.baseline, args.effect, args.power, args.alpha),
        "paired": False,
    }
    if args.games:
        out["at_games"] = args.games
        out["resolves_pp"] = _half_width(args.baseline, args.games)
        out["ci_at_games"] = _wilson(args.baseline, args.games)
    print(json.dumps(out, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("active").set_defaults(func=cmd_active)
    sub.add_parser("audit-opponents").set_defaults(func=cmd_audit_opponents)

    pw = sub.add_parser("power", help="sample size for a target effect (audit M3)")
    pw.add_argument("--baseline", type=float, default=0.33, help="assumed win rate")
    pw.add_argument("--effect", type=float, default=MIN_DETECTABLE_EFFECT,
                    help="difference to detect, as a fraction (0.08 = 8pp)")
    pw.add_argument("--power", type=float, default=0.80)
    pw.add_argument("--alpha", type=float, default=0.05)
    pw.add_argument("--games", type=int, help="also report what this many games resolves to")
    pw.set_defaults(func=cmd_power)

    def match_args(p):
        p.add_argument("--run-id", default="")
        p.add_argument("--iteration", type=int, required=True)
        p.add_argument("--wins", type=int, required=True)
        p.add_argument("--losses", type=int, required=True)
        p.add_argument("--draws", type=int, default=0)
        p.add_argument("--avg-score-model", type=float, default=0.0)
        p.add_argument("--avg-score-opponent", type=float, default=0.0)
        # Reading conditions + granularity (all optional, EXP_ELO observability)
        p.add_argument("--mcts", type=int, help="search sims used for this reading")
        p.add_argument("--gumbel-k", type=int, default=16)
        p.add_argument("--eval-backend", default="")
        p.add_argument("--wins-p1", type=int, help="model wins seated as P1")
        p.add_argument("--wins-p2", type=int, help="model wins seated as P2")
        p.add_argument("--stats-dir", help="arena --dump-stats-dir to summarize into the reading")

    rec = sub.add_parser("record")
    match_args(rec)
    rec.add_argument("--kind", choices=["gauge", "audit"], default="gauge")
    rec.add_argument("--opponent", help="anchor name (required for --kind audit)")
    rec.set_defaults(func=cmd_record)

    frz = sub.add_parser("freeze")
    match_args(frz)
    frz.add_argument("--path", required=True, help="frozen anchor model file")
    frz.set_defaults(func=cmd_freeze)

    args = parser.parse_args()
    if getattr(args, "cmd", None) == "record" and args.kind == "audit" and not args.opponent:
        parser.error("--kind audit requires --opponent")
    args.func(args)


if __name__ == "__main__":
    main()
