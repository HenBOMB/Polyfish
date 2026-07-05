#!/usr/bin/env python3
"""Canonical training_log.csv helpers for run_training_loop.sh."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime, timezone
from typing import Any

CSV_PATH = "training_log.csv"
MOVES_PATH = "moves_by_turn.json"
CURRENT_RUN_PATH = ".current_run"

HEADER = [
    "run_id",
    "iter_started_at",
    "iteration",
    "games_file",
    "avg_score",
    "max_score",
    "p1_avg",
    "p2_avg",
    "loss",
    "policy_loss",
    "value_loss",
    "avg_captures",
    "avg_harvests",
    "avg_builds",
    "avg_research",
    "avg_attacks",
    "avg_revealed_tiles",
    "avg_captured_tiles",
    "avg_moves",
    "match_type",
]

OLD_13 = [
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
    "policy_loss",
]


def _iso_from_unix(ts: int | float) -> str:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).astimezone().isoformat()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def _read_rows(path: str = CSV_PATH) -> list[dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    if not lines:
        return []
    first = lines[0].split(",")
    if first[0] == "run_id":
        reader = csv.DictReader(lines)
        return list(reader)
    rows: list[dict[str, str]] = []
    for line in lines:
        cols = line.split(",")
        if len(cols) == len(OLD_13):
            row = dict(zip(OLD_13, cols))
        elif len(cols) == len(HEADER):
            row = dict(zip(HEADER, cols))
        else:
            continue
        rows.append(row)
    return rows


def _write_rows(rows: list[dict[str, Any]], path: str = CSV_PATH) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in HEADER})


def migrate_csv(path: str = CSV_PATH) -> None:
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        first_line = f.readline().strip()
    if first_line.startswith("run_id,"):
        headers = first_line.split(",")
        if "iter_started_at" in headers:
            return
        if "run_started_at" in headers:
            rows = _read_rows(path)
            for row in rows:
                row["iter_started_at"] = row.pop("run_started_at", "")
            _write_rows(rows, path)
            print(f"Migrated {len(rows)} rows: run_started_at -> iter_started_at")
            return
        return
    rows = _read_rows(path)
    if not rows:
        return
    if rows and "run_id" in rows[0] and rows[0]["run_id"]:
        _write_rows(rows, path)
        return
    first_ts = int(float(rows[0].get("timestamp", "0") or "0"))
    run_id = str(first_ts if first_ts > 0 else int(datetime.now().timestamp()))
    run_started_at = _iso_from_unix(first_ts if first_ts > 0 else int(run_id))
    migrated: list[dict[str, Any]] = []
    for row in rows:
        migrated.append(
            {
                "run_id": run_id,
                "iter_started_at": run_started_at,
                "iteration": row.get("iteration", ""),
                "games_file": "",
                "avg_score": row.get("avg_score", ""),
                "max_score": row.get("max_score", ""),
                "p1_avg": row.get("p1_avg", ""),
                "p2_avg": row.get("p2_avg", ""),
                "loss": row.get("loss", ""),
                "policy_loss": row.get("policy_loss", ""),
                "value_loss": "",
                "avg_captures": row.get("avg_captures", ""),
                "avg_harvests": row.get("avg_harvests", ""),
                "avg_builds": row.get("avg_builds", ""),
                "avg_research": row.get("avg_research", ""),
                "avg_attacks": row.get("avg_attacks", ""),
                "avg_moves": "",
                "match_type": "selfplay",
            }
        )
    _write_rows(migrated, path)
    print(f"Migrated {len(migrated)} rows to new schema (run_id={run_id})")


def _latest_run_id(rows: list[dict[str, str]]) -> str | None:
    if not rows:
        return None
    run_ids = [r.get("run_id", "") for r in rows if r.get("run_id")]
    if not run_ids:
        return None
    return max(run_ids, key=lambda x: int(x))


def _max_iteration_for_run(rows: list[dict[str, str]], run_id: str) -> int:
    iters = [
        int(r["iteration"])
        for r in rows
        if r.get("run_id") == run_id and str(r.get("iteration", "")).isdigit()
    ]
    return max(iters) if iters else 0


def resolve_run(resume: str | None) -> dict[str, Any]:
    migrate_csv()
    rows = _read_rows()
    now = int(datetime.now().timestamp())
    now_iso_val = now_iso()

    if resume is not None:
        target = resume if resume != "latest" else _latest_run_id(rows)
        if not target:
            print("No runs to resume; starting new run.", file=sys.stderr)
            target = None
        else:
            run_rows = [r for r in rows if r.get("run_id") == target]
            if not run_rows:
                print(f"Run {target} not found; starting new run.", file=sys.stderr)
                target = None
            else:
                started = (
                    run_rows[0].get("iter_started_at")
                    or run_rows[0].get("run_started_at")
                    or _iso_from_unix(target)
                )
                start_iter = _max_iteration_for_run(rows, target) + 1
                info = {
                    "run_id": target,
                    "run_started_at": started,
                    "start_iter": start_iter,
                    "mode": "resume",
                }
                with open(CURRENT_RUN_PATH, "w", encoding="utf-8") as f:
                    json.dump(info, f)
                print(json.dumps(info))
                return info

    info = {
        "run_id": str(now),
        "run_started_at": now_iso_val,
        "start_iter": 1,
        "mode": "new",
    }
    with open(CURRENT_RUN_PATH, "w", encoding="utf-8") as f:
        json.dump(info, f)
    print(json.dumps(info))
    return info


def _parse_metrics_line(text: str, game: bool) -> dict[str, Any]:
    for line in text.splitlines():
        if not line.startswith("METRICS:"):
            continue
        payload = line.replace("METRICS:", "", 1).strip()
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if game and ("avg_score" in data or "avg_moves" in data):
            return data
        if not game and "loss" in data:
            return data
    return {}


def parse_self_play_output(text: str) -> dict[str, Any]:
    data = _parse_metrics_line(text, game=True)
    if not data.get("games_file"):
        m = re.search(r"Saved to (games_\d+\.safetensors)", text)
        if m:
            data["games_file"] = m.group(1)
    return data


def parse_train_output(text: str) -> dict[str, Any]:
    return _parse_metrics_line(text, game=False)


def normalize_match_type(match_type: str) -> str:
    if match_type.lower().startswith("league"):
        return "league"
    return "selfplay"


def append_row(
    run_id: str,
    iter_started_at: str,
    iteration: int,
    games_file: str,
    game_metrics: dict[str, Any],
    train_metrics: dict[str, Any],
    match_type: str,
) -> None:
    migrate_csv()
    archived = games_file
    if archived and not archived.startswith("archive/"):
        archived = f"archive/{archived}"

    row = {
        "run_id": run_id,
        "iter_started_at": iter_started_at,
        "iteration": str(iteration),
        "games_file": archived,
        "avg_score": game_metrics.get("avg_score", ""),
        "max_score": game_metrics.get("max_score", ""),
        "p1_avg": game_metrics.get("p1_avg", ""),
        "p2_avg": game_metrics.get("p2_avg", ""),
        "loss": train_metrics.get("loss", ""),
        "policy_loss": train_metrics.get("policy_loss", ""),
        "value_loss": train_metrics.get("value_loss", ""),
        "avg_captures": game_metrics.get("avg_captures", ""),
        "avg_harvests": game_metrics.get("avg_harvests", ""),
        "avg_builds": game_metrics.get("avg_builds", ""),
        "avg_research": game_metrics.get("avg_research", ""),
        "avg_attacks": game_metrics.get("avg_attacks", ""),
        "avg_revealed_tiles": game_metrics.get("avg_revealed_tiles", ""),
        "avg_captured_tiles": game_metrics.get("avg_captured_tiles", ""),
        "avg_moves": game_metrics.get("avg_moves", ""),
        "match_type": normalize_match_type(match_type),
    }

    file_exists = os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 0
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    moves = game_metrics.get("moves_by_turn")
    if moves is not None:
        update_moves_by_turn(run_id, iteration, moves)


def update_moves_by_turn(run_id: str, iteration: int, moves_by_turn: Any) -> None:
    store: dict[str, Any] = {}
    if os.path.exists(MOVES_PATH):
        with open(MOVES_PATH, encoding="utf-8") as f:
            try:
                store = json.load(f)
            except json.JSONDecodeError:
                store = {}
    store.setdefault(str(run_id), {})[str(iteration)] = moves_by_turn
    with open(MOVES_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f)


def finish_run() -> None:
    if os.path.exists(CURRENT_RUN_PATH):
        os.remove(CURRENT_RUN_PATH)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_migrate = sub.add_parser("migrate")
    p_migrate.set_defaults(func=lambda a: migrate_csv())

    p_resolve = sub.add_parser("resolve-run")
    p_resolve.add_argument("--resume", default=None, help="run_id or 'latest'")
    p_resolve.set_defaults(func=lambda a: resolve_run(a.resume))

    p_sp = sub.add_parser("parse-self-play")
    p_sp.add_argument("--input", required=True, help="file path or '-' for stdin")
    p_sp.set_defaults(
        func=lambda a: print(
            json.dumps(
                parse_self_play_output(
                    sys.stdin.read() if a.input == "-" else open(a.input, encoding="utf-8").read()
                )
            )
        )
    )

    p_tr = sub.add_parser("parse-train")
    p_tr.add_argument("--input", required=True, help="file path or '-' for stdin")
    p_tr.set_defaults(
        func=lambda a: print(
            json.dumps(
                parse_train_output(
                    sys.stdin.read() if a.input == "-" else open(a.input, encoding="utf-8").read()
                )
            )
        )
    )

    p_append = sub.add_parser("append-row")
    p_append.add_argument("--run-id", required=True)
    p_append.add_argument("--iter-started-at", required=True)
    p_append.add_argument("--iteration", type=int, required=True)
    p_append.add_argument("--games-file", default="")
    p_append.add_argument("--game-json", required=True)
    p_append.add_argument("--train-json", required=True)
    p_append.add_argument("--match-type", default="selfplay")
    p_append.set_defaults(
        func=lambda a: append_row(
            a.run_id,
            a.iter_started_at,
            a.iteration,
            a.games_file,
            json.loads(a.game_json),
            json.loads(a.train_json),
            a.match_type,
        )
    )

    p_now = sub.add_parser("now-iso")
    p_now.set_defaults(func=lambda a: print(now_iso()))

    p_finish = sub.add_parser("finish-run")
    p_finish.set_defaults(func=lambda a: finish_run())

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
