#!/usr/bin/env python3
"""Strength-ladder store (ladder.json) and gauge verdicts — EXP 10/11.

Anchors are frozen model files (greedy = Elo 0 floor); the last anchor is
"active". `record --kind gauge` appends an arena reading and answers:
continue / freeze (>=80% vs active) / stop (plateau, see _plateau).
Win/loss counts are always from the current model's side. Every reading
carries a Wilson interval and both verdicts are drawn from it, not from the
point estimate a ~64-game reading resolves to only +/-12pp.

Scope: the gauge is a fixed Imperius-mirror instrument while self-play trains
on a 5-tribe pool. Pinning the pair is deliberate variance control — the tribe
block effect rivals a campaign's whole measured improvement — but it makes
every ladder Elo a statement about Imperius-vs-Imperius play, not about the
distribution training optimizes. `--kind tribe_audit` rows are the cross-check
on that gap; they carry the pair they were played on and never enter a verdict.
`--tribes` is the pair the *match* used, read off arena, never the pair
self-play trained on that iteration (#34).
"""
import argparse
import json
import math
import os
from datetime import datetime, timezone

LADDER_FILE = os.environ.get("LADDER_FILE", "ladder.json")
# Recorded into ladder.json itself: a reader of the experiment record should not
# have to find this file to learn what the numbers in it are a measurement of.
# Rewritten from here on every save, so the stored note cannot drift from the code.
SCOPE_NOTE = (
    "Readings are taken on the fixed tribe pair in each reading's `tribes` "
    "field (the gauge pins an Imperius mirror) while self-play trains on the "
    "5-tribe pool in config.json. The pin is variance control: the tribe block "
    "effect rivals a campaign's measured improvement. Ladder Elo is therefore "
    "Imperius-mirror strength, not pool strength; `kind: tribe_audit` readings "
    "are the periodic cross-check and take no part in any verdict."
)
DEFAULT_FREEZE_WR = 0.80
# The bar a reading's Wilson lower bound must clear to freeze a new anchor.
# Overridable only so the branch can be exercised: at 0.80 no cheap reading can
# reach it, which is why the freeze path had never run anywhere (#35). A reading
# judged against a non-default bar records the bar it was judged against.
FREEZE_WR = float(os.environ.get("GAUGE_FREEZE_WR", DEFAULT_FREEZE_WR))
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
            data = json.load(f)
        data["scope"] = SCOPE_NOTE
        return data
    return {
        "scope": SCOPE_NOTE,
        "anchors": [
            {"name": "greedy", "path": "", "elo": 0.0, "frozen_iteration": None, "frozen_at": None}
        ],
        "readings": [],
        "plateau_strikes": 0,
        "plateau_run_id": None,
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


def _elo(win_rate, base):
    p = min(max(win_rate, 0.005), 0.995)
    return round(base + 400.0 * math.log10(p / (1.0 - p)), 1)


def _anchor_by_name(data, name):
    for a in data["anchors"]:
        if a["name"] == name:
            return a
    raise SystemExit(f"unknown anchor: {name}")


def _budget_key(reading):
    """What a reading was taken at. Ladder Elo is a function of (weights x sims
    x turn cap); chaining readings across budgets attributes a search or
    curriculum change to the weights (audit M5). EXP_ELO_002 had to
    hand-quarantine a 16-sim stint for exactly this, and the loop varies
    `max_turns` with the curriculum, so a 10-turn and a 45-turn reading are not
    the same instrument either."""
    b = reading.get("budget")
    if not b:
        return None
    return (b.get("mcts"), b.get("gumbel_k"), b.get("max_turns"))


def _gauge_series(data):
    """Gauge readings vs the active anchor, restricted to the run and the search
    budget the most recent one used. A previous campaign's readings are a
    different model's, so pooling them into this run's window judged a trend
    that never happened. Readings from before `run_id`/`budget` were recorded
    carry no key, so those ladders keep the old pool-everything behaviour rather
    than silently emptying the window."""
    active = data["anchors"][-1]["name"]
    series = [r for r in data["readings"] if r["kind"] == "gauge" and r["opponent"] == active]
    if not series:
        return series
    latest_run = series[-1].get("run_id")
    if latest_run:
        series = [r for r in series if r.get("run_id") == latest_run]
    latest = _budget_key(series[-1])
    if latest is None:
        return series
    return [r for r in series if _budget_key(r) == latest]


def _slope(readings):
    """Least-squares slope of win rate over reading index, in win-rate units per
    reading. The trend half of the EXP 11 plateau rule."""
    n = len(readings)
    if n < 2:
        return 0.0
    ys = []
    for r in readings:
        score, games = _counts(r)
        ys.append(score / games if games else 0.0)
    x_mean = (n - 1) / 2.0
    y_mean = sum(ys) / n
    denom = sum((i - x_mean) ** 2 for i in range(n))
    if denom == 0:
        return 0.0
    return sum((i - x_mean) * (y - y_mean) for i, y in enumerate(ys)) / denom


def _plateau(series):
    """True when the last PLATEAU_WINDOW readings vs the same anchor show no
    gain, by the rule EXP 11 registered: pooled window halves flat-or-down AND
    least-squares slope <= 0.

    Both conditions are directional, and that is the point. The interval-overlap
    test this replaces struck whenever a climb could not be *proven*, and at
    64 games a reading resolves to only ~+/-12pp — so a steady +1pp/reading
    climb (+8pp across the window, exactly the effect EXP_ELO_002 was registered
    against) struck every time, and two gauge cycles later stopped the run
    mid-improvement. A false stop costs a campaign; a late stop costs compute.
    """
    if len(series) < PLATEAU_WINDOW:
        return False
    window = series[-PLATEAU_WINDOW:]
    half = PLATEAU_WINDOW // 2
    first, second = _pool(window[:half]), _pool(window[half:])
    return second[0] <= first[0] and _slope(window) <= 0.0


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
            "max_turns": getattr(args, "max_turns", None),
            # Deliberately not in _budget_key: these ramp every iteration to
            # track the searcher self-play is generating with, so keying on
            # them would give every reading its own budget and no window would
            # ever accumulate. Recorded so the drift is visible in the series.
            "prior_heuristic_w": getattr(args, "prior_heuristic_w", None),
            "q_weight": getattr(args, "q_weight", None),
        }
    if getattr(args, "wins_p1", None) is not None:
        reading["wins_as_p1"] = args.wins_p1
        reading["wins_as_p2"] = args.wins_p2
    # A panicked game is dropped by arena. Recording only the surviving count
    # makes a damaged reading indistinguishable from a clean one, and a drop
    # also unbalances the side-swap pairing the seeded map set buys (audit M5).
    attempted = getattr(args, "games_attempted", 0) or 0
    dropped = getattr(args, "games_dropped", 0) or 0
    unpaired = getattr(args, "unpaired_seeds", 0) or 0
    if attempted or dropped or unpaired:
        reading["games_attempted"] = attempted
        reading["games_dropped"] = dropped
        reading["unpaired_seeds"] = unpaired
    # The pair this match was played on, read off arena's own output. It used to
    # be handed self-play's shuffled training pair for a match arena hardcoded to
    # an Imperius mirror, which made the permanent record disagree with both the
    # instrument and its own per-game JSONs (#34, audit M5).
    if getattr(args, "tribes", None):
        reading["tribes"] = args.tribes
    if getattr(args, "stats_dir", None):
        behavior = _summarize_stats(args.stats_dir)
        if behavior is not None:
            reading["behavior"] = behavior
    data["readings"].append(reading)
    return reading


def cmd_record(args):
    data = _load()
    # A tribe_audit is the gauge match on a different tribe pair, so it reads
    # against the same anchor the gauge did -- the difference between the two
    # rows is the block effect the pinned pair buys away. It names that anchor
    # when the caller has one: an audit cadence landing on a freeze iteration
    # plays the outgoing anchor, which by then is no longer anchors[-1].
    if args.kind == "gauge" or (args.kind == "tribe_audit" and not args.opponent):
        opponent = data["anchors"][-1]
    else:
        opponent = _anchor_by_name(data, args.opponent)
    reading = _append_reading(data, args, args.kind, opponent)

    action = "continue"
    if args.kind == "gauge":
        # A strike is evidence about one campaign. Strikes persisted in
        # ladder.json across runs, so a fresh run could inherit one and stop
        # two readings into its own series.
        if data.get("plateau_run_id") != args.run_id:
            data["plateau_strikes"] = 0
            data["plateau_run_id"] = args.run_id
        if FREEZE_WR != DEFAULT_FREEZE_WR:
            # The bar was moved for this reading, so the record has to say so:
            # a forced freeze must not be indistinguishable from an earned one.
            reading["freeze_wr"] = FREEZE_WR
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
    if "freeze_wr" in reading:
        verdict["freeze_wr"] = reading["freeze_wr"]
    # A single reading this size cannot carry a verdict about a difference
    # smaller than its own resolution. Say so on every reading rather than
    # leaving the next reader to rediscover it from the interval.
    if reading.get("games_dropped") or reading.get("unpaired_seeds"):
        verdict["games_dropped"] = reading.get("games_dropped", 0)
        verdict["unpaired_seeds"] = reading.get("unpaired_seeds", 0)
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
    data["plateau_run_id"] = args.run_id
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


def build_parser():
    """The CLI itself, separate from main() so the shell<->argparse contract can
    be checked without running a command (#35)."""
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
        p.add_argument("--games-attempted", type=int, default=0,
                       help="games arena started, before panicked ones were dropped")
        p.add_argument("--games-dropped", type=int, default=0,
                       help="seeds arena dropped after an in-game panic")
        p.add_argument("--unpaired-seeds", type=int, default=0,
                       help="seeds that lost one half of their side swap")
        p.add_argument("--tribes", default="",
                       help="tribe pair the match was played on, from arena's `Tribes:` "
                            "line -- not the pair self-play trained on this iteration")
        p.add_argument("--max-turns", type=int,
                       help="turn cap this reading was played at (the loop varies it "
                            "with the curriculum, so it is part of the budget key)")
        p.add_argument("--prior-heuristic-w", type=float,
                       help="heuristic/net prior blend the gauge searched with; ramps "
                            "with the iteration to match self-play")
        p.add_argument("--q-weight", type=float,
                       help="sigma(Q) weight in-tree and in the root policy target, "
                            "matching self-play's value-trust ramp")

    rec = sub.add_parser("record")
    match_args(rec)
    rec.add_argument("--kind", choices=["gauge", "audit", "tribe_audit"], default="gauge",
                     help="gauge steers the run; audit cross-checks the anchor chain; "
                          "tribe_audit re-reads the gauge's anchor on another tribe pair. "
                          "Only gauge carries a verdict or enters the plateau window.")
    rec.add_argument("--opponent",
                     help="anchor name; required for --kind audit, and for a "
                          "tribe_audit whose match was played against an anchor "
                          "a freeze in the same iteration has since retired")
    rec.set_defaults(func=cmd_record)

    frz = sub.add_parser("freeze")
    match_args(frz)
    frz.add_argument("--path", required=True, help="frozen anchor model file")
    frz.set_defaults(func=cmd_freeze)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "cmd", None) == "record" and args.kind == "audit" and not args.opponent:
        parser.error("--kind audit requires --opponent")
    args.func(args)


if __name__ == "__main__":
    main()
