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
TEMPO_PATH = "tempo_by_turn.json"
VALUE_DIST_PATH = "value_distribution.json"
CURRENT_RUN_PATH = ".current_run"
HIST_BINS = 80
MAX_VALUE_SAMPLES = 8000
SELF_PLAY_METRICS_PATH = ".last_self_play_metrics.json"
TRAIN_METRICS_PATH = ".last_train_metrics.json"

SPT_MILESTONES = [0, 5, 10, 15, 20, 25, 30]
SPT_COLUMNS = [f"avg_spt_t{t}" for t in SPT_MILESTONES]

# Absolute army-composition ratios (net seats only, no opponent term), so they
# can move in mirror self-play where contested counts cannot. Greedy reference
# at t15: unit_worth ~3.7, army_stars_per_city ~10.0 — the model sits at ~2.2
# and ~5.4, i.e. ~2x headroom on a metric with ~1.5%/iteration noise.
# -1.0 means the milestone turn was never reached (0.0 is a legal value).
ARMY_RATIO_COLUMNS = [
    "unit_worth_t15",
    "unit_worth_t25",
    "army_stars_per_city_t15",
    "army_stars_per_city_t25",
]

HEADER = [
    "run_id",
    "iter_started_at",
    "iteration",
    "games_file",
    "num_games",
    "tribes",
    "avg_score",
    "max_score",
    "p1_avg",
    "p2_avg",
    "loss",
    "policy_loss",
    "value_loss",
    "value_r2",
    "aux_own_loss",
    "aux_fog_loss",
    "aux_spt_loss",
    "aux_tech_loss",
    "kl_ref_loss",
    "avg_captures",
    "avg_cap_ruins",
    "avg_cap_villages",
    "avg_cap_cities",
    "avg_cap_capitals",
    "avg_harvests",
    "avg_builds",
    "avg_research",
    "avg_attacks",
    "avg_abilities",
    "avg_kills",
    "avg_revealed_tiles",
    "avg_captured_tiles",
    *SPT_COLUMNS,
    *ARMY_RATIO_COLUMNS,
    "villages_t2c_first",
    "villages_first_rate",
    "villages_t2c_first_cond",
    "villages_t2c_p50",
    "villages_t2c_p80",
    "villages_t2c_all",
    "ruins_t2c_p50",
    "ruins_t2c_p80",
    "ruins_t2c_all",
    "avg_units_spawned",
    "avg_units_granted",
    "avg_units_lost",
    "avg_giants_made",
    "t2c_2nd_rate",
    "t2c_2nd_turn",
    "t2c_3rd_rate",
    "t2c_3rd_turn",
    "t2c_4th_rate",
    "t2c_4th_turn",
    "anchor_games",
    "anchor_net_wr",
    "avg_moves",
    "match_type",
]

TEMPO_COLUMNS = [
    "avg_units_spawned",
    "avg_units_granted",
    "avg_units_lost",
    "avg_giants_made",
    "t2c_2nd_rate",
    "t2c_2nd_turn",
    "t2c_3rd_rate",
    "t2c_3rd_turn",
    "t2c_4th_rate",
    "t2c_4th_turn",
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
        if headers == HEADER:
            return
        if "run_started_at" in headers:
            rows = _read_rows(path)
            for row in rows:
                row["iter_started_at"] = row.pop("run_started_at", "")
            _write_rows(rows, path)
            print(f"Migrated {len(rows)} rows: run_started_at -> iter_started_at")
            return
        if "iter_started_at" in headers:
            # Current-era file missing newly added columns: rewrite under the
            # current HEADER; old rows keep blanks for the new columns.
            rows = _read_rows(path)
            _write_rows(rows, path)
            print(f"Migrated {len(rows)} rows to updated header ({len(HEADER)} columns)")
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
                "avg_cap_ruins": "",
                "avg_cap_villages": "",
                "avg_cap_cities": "",
                "avg_cap_capitals": "",
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


def _load_json_file(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def parse_self_play_output(text: str | None = None) -> dict[str, Any]:
    if os.path.exists(SELF_PLAY_METRICS_PATH):
        return _load_json_file(SELF_PLAY_METRICS_PATH)
    data = _parse_metrics_line(text or "", game=True)
    if not data.get("games_file"):
        m = re.search(r"Saved to (games_\d+\.safetensors)", text or "")
        if m:
            data["games_file"] = m.group(1)
    return data


def parse_train_output(text: str | None = None) -> dict[str, Any]:
    if os.path.exists(TRAIN_METRICS_PATH):
        return _load_json_file(TRAIN_METRICS_PATH)
    return _parse_metrics_line(text or "", game=False)


def normalize_match_type(match_type: str) -> str:
    if match_type.lower().startswith("league"):
        return "league"
    return "selfplay"


def resolve_games_file(games_file: str) -> str | None:
    if not games_file:
        return None
    bare = games_file.removeprefix("archive/").removeprefix("./")
    for path in (games_file, bare, f"archive/{bare}"):
        if path and os.path.exists(path):
            return path
    return None


def compute_value_distribution(
    values: list[float], games_file: str, num_games: int | None = None
) -> dict[str, Any]:
    n = len(values)
    if n == 0:
        return {
            "file": games_file,
            "n": 0,
            "num_games": num_games or 0,
            "stats": {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "weak_pct": 0.0,
                "moderate_pct": 0.0,
                "strong_pct": 0.0,
                "saturation_pct": 0.0,
                "in_target_range_pct": 0.0,
            },
            "hist": {"bins": [], "counts": []},
            "abs_hist": {"bins": [], "counts": []},
            "buckets": {
                "weak": 0.0,
                "moderate": 0.0,
                "strong": 0.0,
                "saturation": 0.0,
            },
            "samples": [],
        }

    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std = variance**0.5
    vmin = min(values)
    vmax = max(values)

    weak = moderate = strong = saturation = in_target = 0
    for v in values:
        av = abs(v)
        if av < 0.1:
            weak += 1
        elif av < 0.3:
            moderate += 1
        elif av < 0.5:
            strong += 1
        else:
            saturation += 1
        if 0.1 <= av <= 0.5:
            in_target += 1

    def pct(count: int) -> float:
        return 100.0 * count / n

    hist_counts = [0] * HIST_BINS
    abs_hist_counts = [0] * HIST_BINS
    for v in values:
        idx = min(int(((v + 1.0) / 2.0) * HIST_BINS), HIST_BINS - 1)
        if idx < 0:
            idx = 0
        hist_counts[idx] += 1
        av = min(max(abs(v), 0.0), 1.0)
        aidx = min(int(av * HIST_BINS), HIST_BINS - 1)
        abs_hist_counts[aidx] += 1

    hist_bins = [-1.0 + (2.0 * (i + 0.5) / HIST_BINS) for i in range(HIST_BINS)]
    abs_bins = [(i + 0.5) / HIST_BINS for i in range(HIST_BINS)]

    if n <= MAX_VALUE_SAMPLES:
        samples = values
    else:
        step = max(n // MAX_VALUE_SAMPLES, 1)
        samples = values[::step]

    return {
        "file": games_file,
        "n": n,
        "num_games": num_games or 0,
        "stats": {
            "mean": mean,
            "std": std,
            "min": vmin,
            "max": vmax,
            "weak_pct": pct(weak),
            "moderate_pct": pct(moderate),
            "strong_pct": pct(strong),
            "saturation_pct": pct(saturation),
            "in_target_range_pct": pct(in_target),
        },
        "hist": {"bins": hist_bins, "counts": hist_counts},
        "abs_hist": {"bins": abs_bins, "counts": abs_hist_counts},
        "buckets": {
            "weak": pct(weak),
            "moderate": pct(moderate),
            "strong": pct(strong),
            "saturation": pct(saturation),
        },
        "samples": samples,
    }


def load_values_from_games(path: str) -> list[float]:
    from safetensors.numpy import load_file

    data = load_file(path)
    if "values" not in data:
        return []
    arr = data["values"]
    return [float(x) for x in arr.reshape(-1)]


def update_value_distribution(
    run_id: str,
    iteration: int,
    games_file: str,
    path: str | None = None,
    num_games: int | None = None,
) -> None:
    resolved = path or resolve_games_file(games_file)
    if not resolved:
        return
    try:
        values = load_values_from_games(resolved)
    except Exception as e:
        print(f"value_distribution: skip {resolved}: {e}", file=sys.stderr)
        return
    if not values:
        return

    store: dict[str, Any] = {}
    if os.path.exists(VALUE_DIST_PATH):
        with open(VALUE_DIST_PATH, encoding="utf-8") as f:
            try:
                store = json.load(f)
            except json.JSONDecodeError:
                store = {}
    store.setdefault(str(run_id), {})[str(iteration)] = compute_value_distribution(
        values, games_file, num_games
    )
    with open(VALUE_DIST_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f)


def backfill_value_distribution() -> int:
    """Populate value_distribution.json from CSV rows whose games files still exist."""
    if not os.path.exists(CSV_PATH):
        return 0
    filled = 0
    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_id = row.get("run_id", "")
            iteration = row.get("iteration", "")
            games_file = row.get("games_file", "")
            if not run_id or not iteration or not games_file:
                continue
            store: dict[str, Any] = {}
            if os.path.exists(VALUE_DIST_PATH):
                with open(VALUE_DIST_PATH, encoding="utf-8") as sf:
                    try:
                        store = json.load(sf)
                    except json.JSONDecodeError:
                        store = {}
            if store.get(run_id, {}).get(iteration):
                continue
            path = resolve_games_file(games_file)
            if not path:
                continue
            ng = row.get("num_games", "")
            num_games = int(ng) if str(ng).isdigit() else None
            update_value_distribution(run_id, int(iteration), games_file, path, num_games)
            filled += 1
    return filled


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
        "num_games": game_metrics.get("num_games", ""),
        "tribes": game_metrics.get("tribes", ""),
        "avg_score": game_metrics.get("avg_score", ""),
        "max_score": game_metrics.get("max_score", ""),
        "p1_avg": game_metrics.get("p1_avg", ""),
        "p2_avg": game_metrics.get("p2_avg", ""),
        "loss": train_metrics.get("loss", ""),
        "policy_loss": train_metrics.get("policy_loss", ""),
        "value_loss": train_metrics.get("value_loss", ""),
        "value_r2": train_metrics.get("value_r2", ""),
        "aux_own_loss": train_metrics.get("aux_own_loss", ""),
        "aux_fog_loss": train_metrics.get("aux_fog_loss", ""),
        "aux_spt_loss": train_metrics.get("aux_spt_loss", ""),
        "aux_tech_loss": train_metrics.get("aux_tech_loss", ""),
        "kl_ref_loss": train_metrics.get("kl_ref_loss", ""),
        "avg_captures": game_metrics.get("avg_captures", ""),
        "avg_cap_ruins": game_metrics.get("avg_cap_ruins", ""),
        "avg_cap_villages": game_metrics.get("avg_cap_villages", ""),
        "avg_cap_cities": game_metrics.get("avg_cap_cities", ""),
        "avg_cap_capitals": game_metrics.get("avg_cap_capitals", ""),
        "avg_harvests": game_metrics.get("avg_harvests", ""),
        "avg_builds": game_metrics.get("avg_builds", ""),
        "avg_research": game_metrics.get("avg_research", ""),
        "avg_attacks": game_metrics.get("avg_attacks", ""),
        "avg_abilities": game_metrics.get("avg_abilities", ""),
        "avg_kills": game_metrics.get("avg_kills", ""),
        "avg_revealed_tiles": game_metrics.get("avg_revealed_tiles", ""),
        "avg_captured_tiles": game_metrics.get("avg_captured_tiles", ""),
        **{col: game_metrics.get(col, "") for col in SPT_COLUMNS},
        **{col: game_metrics.get(col, "") for col in ARMY_RATIO_COLUMNS},
        "villages_t2c_first": game_metrics.get("villages_t2c_first", ""),
        "villages_first_rate": game_metrics.get("villages_first_rate", ""),
        "villages_t2c_first_cond": game_metrics.get("villages_t2c_first_cond", ""),
        "villages_t2c_p50": game_metrics.get("villages_t2c_p50", ""),
        "villages_t2c_p80": game_metrics.get("villages_t2c_p80", ""),
        "villages_t2c_all": game_metrics.get("villages_t2c_all", ""),
        "ruins_t2c_p50": game_metrics.get("ruins_t2c_p50", ""),
        "ruins_t2c_p80": game_metrics.get("ruins_t2c_p80", ""),
        "ruins_t2c_all": game_metrics.get("ruins_t2c_all", ""),
        **{col: game_metrics.get(col, "") for col in TEMPO_COLUMNS},
        "anchor_games": game_metrics.get("anchor_games", ""),
        "anchor_net_wr": game_metrics.get("anchor_net_wr", ""),
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

    tempo = game_metrics.get("tempo_by_turn")
    if tempo is not None:
        update_tempo_by_turn(run_id, iteration, tempo)

    if archived:
        ng = game_metrics.get("num_games")
        num_games = int(ng) if ng not in (None, "") else None
        update_value_distribution(run_id, iteration, archived, num_games=num_games)


def update_tempo_by_turn(run_id: str, iteration: int, tempo_by_turn: Any) -> None:
    """Sidecar for the per-role development-tempo curves (moves_by_turn pattern)."""
    store: dict[str, Any] = {}
    if os.path.exists(TEMPO_PATH):
        with open(TEMPO_PATH, encoding="utf-8") as f:
            try:
                store = json.load(f)
            except json.JSONDecodeError:
                store = {}
    store.setdefault(str(run_id), {})[str(iteration)] = tempo_by_turn
    with open(TEMPO_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f)


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
    p_sp.add_argument(
        "--input",
        default=None,
        help="optional legacy stdout capture (file path or '-'); default reads sidecar JSON",
    )
    p_sp.set_defaults(
        func=lambda a: print(
            json.dumps(
                parse_self_play_output(
                    sys.stdin.read()
                    if a.input == "-"
                    else open(a.input, encoding="utf-8").read()
                    if a.input
                    else None
                )
            )
        )
    )

    p_tr = sub.add_parser("parse-train")
    p_tr.add_argument(
        "--input",
        default=None,
        help="optional legacy stdout capture (file path or '-'); default reads sidecar JSON",
    )
    p_tr.set_defaults(
        func=lambda a: print(
            json.dumps(
                parse_train_output(
                    sys.stdin.read()
                    if a.input == "-"
                    else open(a.input, encoding="utf-8").read()
                    if a.input
                    else None
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

    p_backfill = sub.add_parser("backfill-value-distribution")
    p_backfill.set_defaults(
        func=lambda a: print(json.dumps({"filled": backfill_value_distribution()}))
    )

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
