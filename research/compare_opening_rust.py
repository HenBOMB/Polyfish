"""Interleave old and new Rust opening traces on the same host."""

import argparse
import json
from pathlib import Path
import random

from run_engine_benchmark import ROOT, SIDES, measure, paired_summary


def rate(row):
    return row["iterations"] * row["actionsPerTrace"] * 1e9 / row["traceNs"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("baseline_commit")
    parser.add_argument("candidate_commit")
    args = parser.parse_args()
    binaries = {key: path.resolve(strict=True) for key, path in
                (("before", args.baseline), ("after", args.candidate))}
    rng = random.Random(423)
    raw = []
    for repeat in range(7):
        sizes = list(SIDES)
        rng.shuffle(sizes)
        for side in sizes:
            engines = list(binaries.items())
            rng.shuffle(engines)
            for order, (revision, binary) in enumerate(engines):
                row = measure([str(binary), str(side**2), "250", "trace", "500"], ROOT / "polyfish-rs")
                assert row["actionsPerTrace"] == 40 and row["size"] == side
                raw.append(dict(row, revision=revision, repeat=repeat, order=order))
    summary = []
    for side in SIDES:
        before, after = ([rate(row) for row in raw if row["size"] == side and row["revision"] == build]
                         for build in ("before", "after"))
        pair = paired_summary(before, after, True, rng)
        summary.append({
            "tiles": side**2, "beforeMedian": pair["javaMedian"], "afterMedian": pair["rustMedian"],
            "afterVsBefore": pair["rustAdvantageRatio"], "paired95": pair["pairedBootstrap95"],
        })
    print(json.dumps({"baselineCommit": args.baseline_commit, "candidateCommit": args.candidate_commit,
                      "raw": raw, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
