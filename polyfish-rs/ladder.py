#!/usr/bin/env python3
"""Strength-ladder store (ladder.json) and gauge verdicts — EXP 10/11.

Anchors are frozen model files (greedy = Elo 0 floor); the last anchor is
"active". `record --kind gauge` appends an arena reading and answers:
continue / freeze (>=80% vs active) / stop (plateau, see _plateau).
Win/loss counts are always from the current model's side.
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
    gain: second-half mean <= first-half mean AND regression slope <= 0."""
    if len(series) < PLATEAU_WINDOW:
        return False
    w = [r["win_rate"] for r in series[-PLATEAU_WINDOW:]]
    half = PLATEAU_WINDOW // 2
    if sum(w[half:]) / half > sum(w[:half]) / half:
        return False
    n = len(w)
    xbar, ybar = (n - 1) / 2, sum(w) / n
    slope = sum((i - xbar) * (y - ybar) for i, y in enumerate(w)) / sum(
        (i - xbar) ** 2 for i in range(n)
    )
    return slope <= 0


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
        n_audits = sum(1 for r in data["readings"] if r["kind"] == "audit")
        opponents.append(retired_nets[n_audits % len(retired_nets)])
    print(json.dumps(opponents))


def _append_reading(data, args, kind, opponent):
    win_rate = round(_win_rate(args.wins, args.losses, args.draws), 4)
    reading = {
        "at": _now(),
        "run_id": args.run_id,
        "iteration": args.iteration,
        "kind": kind,
        "model": f"model@iter{args.iteration}",
        "opponent": opponent["name"],
        "games": args.wins + args.losses + args.draws,
        "wins": args.wins,
        "losses": args.losses,
        "draws": args.draws,
        "win_rate": win_rate,
        "elo_est": _elo(win_rate, opponent["elo"]),
        "avg_score_model": args.avg_score_model,
        "avg_score_opponent": args.avg_score_opponent,
    }
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
        if reading["win_rate"] >= FREEZE_WR:
            action = "freeze"
            data["plateau_strikes"] = 0
        elif _plateau(_gauge_series(data)):
            data["plateau_strikes"] += 1
            if data["plateau_strikes"] >= PLATEAU_STRIKES:
                action = "stop"
        else:
            data["plateau_strikes"] = 0
    _save(data)
    print(
        json.dumps(
            {
                "action": action,
                "opponent": opponent["name"],
                "win_rate": reading["win_rate"],
                "elo_est": reading["elo_est"],
                "plateau_strikes": data["plateau_strikes"],
            }
        )
    )


def cmd_freeze(args):
    """Register a new anchor from the link-match result vs the outgoing one."""
    data = _load()
    outgoing = data["anchors"][-1]
    link_wr = _win_rate(args.wins, args.losses, args.draws)
    new_anchor = {
        "name": os.path.splitext(os.path.basename(args.path))[0],
        "path": args.path,
        "elo": _elo(link_wr, outgoing["elo"]),
        "frozen_iteration": args.iteration,
        "frozen_at": _now(),
    }
    _append_reading(data, args, "link", outgoing)
    data["anchors"].append(new_anchor)
    data["plateau_strikes"] = 0
    _save(data)
    print(json.dumps(new_anchor))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("active").set_defaults(func=cmd_active)
    sub.add_parser("audit-opponents").set_defaults(func=cmd_audit_opponents)

    def match_args(p):
        p.add_argument("--run-id", default="")
        p.add_argument("--iteration", type=int, required=True)
        p.add_argument("--wins", type=int, required=True)
        p.add_argument("--losses", type=int, required=True)
        p.add_argument("--draws", type=int, default=0)
        p.add_argument("--avg-score-model", type=float, default=0.0)
        p.add_argument("--avg-score-opponent", type=float, default=0.0)

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
