#!/usr/bin/env python3
"""Value-head calibration/discrimination analysis over a --dump-value-calib
JSONL dump. Fills a real gap: every past reading of this kind (EXP_ELO_021,
022, 024, 046, 060, 067, 069, 072, 073, 123, 124) was computed with uncommitted
scratch Python, re-derived from scratch each time. This is that script,
committed once.

Per turn-band, per signal column present in the dump (raw_value, root_value,
heur_value, macro_root_q, micro_root_q), reports:
  - n (rows where both the signal and final_outcome are non-null)
  - Pearson r and r^2 vs final_outcome
  - OLS slope/intercept of final_outcome ~ signal -- the direct
    over-confidence number (slope << 1 means "says 0.9 when 0.3 is
    warranted", which r^2 alone can hide)
  - saturation rate |signal| > 0.8

No sklearn in the project venv -- r^2/OLS computed by hand with numpy.

Usage:
    analyze_value_calib.py <dump.jsonl> [--turn-band-width 5]
"""
import argparse
import json
import sys

import numpy as np

SIGNAL_COLUMNS = ["raw_value", "root_value", "heur_value", "macro_root_q", "micro_root_q"]


def load_rows(path):
    rows = []
    skipped = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                skipped = skipped + 1
    if skipped:
        print(f"WARNING: skipped {skipped} malformed line(s) in {path}", file=sys.stderr)
    return rows


def turn_band(turn, width):
    lo = (turn // width) * width
    return lo, lo + width


def paired(rows, col, target="final_outcome"):
    """(signal, target) arrays for rows where both are non-null."""
    xs, ys = [], []
    for r in rows:
        v = r.get(col)
        o = r.get(target)
        if v is None or o is None:
            continue
        xs.append(v)
        ys.append(o)
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def ols_r2_slope(x, y):
    """OLS fit y ~ a*x + b; returns (r2, slope, intercept). None if degenerate."""
    n = len(x)
    if n < 2 or np.std(x) == 0:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return r2, slope, intercept


def pearson_r(x, y):
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def saturation_rate(x, threshold=0.8):
    if len(x) == 0:
        return float("nan")
    return float(np.mean(np.abs(x) > threshold))


def analyze_column(rows, col, band_width):
    x_all, y_all = paired(rows, col)
    if len(x_all) == 0:
        return None
    bands = {}
    for r in rows:
        v = r.get(col)
        o = r.get("final_outcome")
        t = r.get("turn")
        if v is None or o is None or t is None:
            continue
        lo, hi = turn_band(t, band_width)
        bands.setdefault((lo, hi), ([], []))
        bands[(lo, hi)][0].append(v)
        bands[(lo, hi)][1].append(o)

    out = {"overall": summarize(x_all, y_all), "bands": {}}
    for (lo, hi), (xs, ys) in sorted(bands.items()):
        out["bands"][f"[{lo},{hi})"] = summarize(np.array(xs), np.array(ys))
    return out


def summarize(x, y):
    n = len(x)
    r = pearson_r(x, y)
    fit = ols_r2_slope(x, y)
    sat = saturation_rate(x)
    if fit is None:
        return {"n": n, "r": r, "r2": float("nan"), "slope": float("nan"),
                "intercept": float("nan"), "saturation": sat}
    r2, slope, intercept = fit
    return {"n": n, "r": r, "r2": r2, "slope": slope, "intercept": intercept, "saturation": sat}


def print_table(col, result):
    print(f"\n=== {col} ===")
    print(f"{'band':<12} {'n':>6} {'r':>7} {'r2':>7} {'slope':>7} {'intercept':>9} {'sat>0.8':>8}")
    ov = result["overall"]
    print(f"{'overall':<12} {ov['n']:>6} {ov['r']:>7.3f} {ov['r2']:>7.3f} "
          f"{ov['slope']:>7.3f} {ov['intercept']:>9.3f} {ov['saturation']:>8.3f}")
    for band, s in result["bands"].items():
        print(f"{band:<12} {s['n']:>6} {s['r']:>7.3f} {s['r2']:>7.3f} "
              f"{s['slope']:>7.3f} {s['intercept']:>9.3f} {s['saturation']:>8.3f}")


def print_bias_breakdown(rows, col, band_width):
    """Signed mean, saturation split by sign, and per-seat mean -- separates
    "undifferentiated optimism" (both seats told they're winning) from
    genuine discrimination failure, which OLS r2/slope alone can't tell
    apart (mean(final_outcome) ~= 0 in mirror play collapses both into the
    same slope). See EXP_ELO_135 ACTUAL."""
    print(f"\n=== {col}: signed bias / seat split ===")
    print(f"{'band':<12} {'n':>6} {'mean':>7} {'sat>+0.8':>9} {'sat<-0.8':>9} "
          f"{'mean(p1)':>9} {'mean(p2)':>9}")
    bands = {}
    for r in rows:
        v = r.get(col)
        t = r.get("turn")
        if v is None or t is None:
            continue
        lo, hi = turn_band(t, band_width)
        bands.setdefault((lo, hi), []).append(r)
    for (lo, hi), rs in sorted(bands.items()):
        x = np.array([r[col] for r in rs])
        p1 = np.array([r[col] for r in rs if r.get("p") == 1])
        p2 = np.array([r[col] for r in rs if r.get("p") == 2])
        mean_p1 = np.mean(p1) if len(p1) else float("nan")
        mean_p2 = np.mean(p2) if len(p2) else float("nan")
        print(f"[{lo},{hi}){'':<4} {len(rs):>6} {np.mean(x):>7.3f} "
              f"{np.mean(x > 0.8):>9.3f} {np.mean(x < -0.8):>9.3f} "
              f"{mean_p1:>9.3f} {mean_p2:>9.3f}")


def print_target_comparison(rows, col, band_width):
    """r2/slope of `col` against final_outcome vs against value_target,
    inner-joined so both use the same rows -- discriminates whether
    turn-3 over-confidence is a network fit failure (low r2 against BOTH)
    or a target-choice mismatch (fits value_target fine, decoupled from
    final_outcome because value_target itself is ~70% a td_w-weighted
    ~5-turn-ahead bootstrap off the search's own future root_value, not
    the game's actual outcome -- see labels.rs LAMBDA_RETURN=0.8)."""
    joined = [r for r in rows
              if r.get(col) is not None and r.get("final_outcome") is not None
              and r.get("value_target") is not None]
    if not joined:
        print(f"\n=== {col}: final_outcome vs value_target (no overlapping rows) ===")
        return
    print(f"\n=== {col}: fit against final_outcome vs against value_target (n={len(joined)} total) ===")
    print(f"{'band':<12} {'n':>6} {'fo_r2':>8} {'fo_slope':>9} {'vt_r2':>8} {'vt_slope':>9}")
    bands = {}
    for r in joined:
        lo, hi = turn_band(r["turn"], band_width)
        bands.setdefault((lo, hi), []).append(r)
    for (lo, hi), rs in sorted(bands.items()):
        x = np.array([r[col] for r in rs])
        fo = np.array([r["final_outcome"] for r in rs])
        vt = np.array([r["value_target"] for r in rs])
        fit_fo = ols_r2_slope(x, fo)
        fit_vt = ols_r2_slope(x, vt)
        fo_r2, fo_slope = (fit_fo[0], fit_fo[1]) if fit_fo else (float("nan"), float("nan"))
        vt_r2, vt_slope = (fit_vt[0], fit_vt[1]) if fit_vt else (float("nan"), float("nan"))
        print(f"[{lo},{hi}){'':<4} {len(rs):>6} {fo_r2:>8.3f} {fo_slope:>9.3f} "
              f"{vt_r2:>8.3f} {vt_slope:>9.3f}")


def print_pair_comparison(rows, col_a, col_b, band_width):
    """Inner-joined comparison: rows where BOTH columns are non-null."""
    joined = [r for r in rows
              if r.get(col_a) is not None and r.get(col_b) is not None
              and r.get("final_outcome") is not None]
    if not joined:
        print(f"\n=== {col_a} vs {col_b} (inner join): no overlapping rows ===")
        return
    print(f"\n=== {col_a} vs {col_b} (inner join, n={len(joined)} total) ===")
    print(f"{'band':<12} {'n':>6} {col_a+'_r2':>12} {col_b+'_r2':>12} "
          f"{col_a+'_sat':>10} {col_b+'_sat':>10}")
    bands = {}
    for r in joined:
        lo, hi = turn_band(r["turn"], band_width)
        bands.setdefault((lo, hi), []).append(r)
    for (lo, hi), rs in sorted(bands.items()):
        xa = np.array([r[col_a] for r in rs])
        xb = np.array([r[col_b] for r in rs])
        y = np.array([r["final_outcome"] for r in rs])
        fit_a = ols_r2_slope(xa, y)
        fit_b = ols_r2_slope(xb, y)
        r2a = fit_a[0] if fit_a else float("nan")
        r2b = fit_b[0] if fit_b else float("nan")
        print(f"[{lo},{hi}){'':<4} {len(rs):>6} {r2a:>12.3f} {r2b:>12.3f} "
              f"{saturation_rate(xa):>10.3f} {saturation_rate(xb):>10.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dump", help="--dump-value-calib JSONL file")
    ap.add_argument("--turn-band-width", type=int, default=5)
    args = ap.parse_args()

    rows = load_rows(args.dump)
    print(f"Loaded {len(rows)} rows from {args.dump}")

    present = [c for c in SIGNAL_COLUMNS if any(r.get(c) is not None for r in rows)]
    print(f"Signal columns present (non-null somewhere): {present}")

    for col in SIGNAL_COLUMNS:
        result = analyze_column(rows, col, args.turn_band_width)
        if result is None:
            print(f"\n=== {col} ===\n(no non-null rows)")
            continue
        print_table(col, result)

    # Primary diagnostic pair: does the RAW pre-search value already show the
    # over-confidence, or is it search-amplified on top of a more modest
    # signal? See plan_net_root_candidates.md-adjacent EXP_ELO_13x entry.
    print_pair_comparison(rows, "raw_value", "micro_root_q", args.turn_band_width)
    print_bias_breakdown(rows, "raw_value", args.turn_band_width)
    print_bias_breakdown(rows, "micro_root_q", args.turn_band_width)

    # Discriminating check: is turn-3 over-confidence a network fit failure,
    # or a target-choice mismatch (value_target is dominated by a ~5-turn
    # bootstrap, not final_outcome)? See conversation notes / EXP_ELO_135 addendum.
    print_target_comparison(rows, "raw_value", args.turn_band_width)
    print_target_comparison(rows, "micro_root_q", args.turn_band_width)


if __name__ == "__main__":
    sys.exit(main())
