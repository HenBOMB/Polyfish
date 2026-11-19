"""Summarize before/after throughput and retained-state memory from paired runs."""

import argparse
import json
from pathlib import Path
import statistics

from run_midgame_benchmark import RETAIN_COUNTS


def lookup(report):
    return {(row["tiles"], row["mode"], row["metric"], row["count"] if row["mode"] == "retain" else None): row
            for row in report["summary"]}


def summarize(baseline, candidate):
    for field in ("tribesCommit", "javaFlags", "machine", "java", "rust", "sha256"):
        if baseline[field] != candidate[field]:
            raise ValueError(f"Benchmark inputs differ: {field}")
    before, after = lookup(baseline), lookup(candidate)
    for key, row in after.items():
        if row["count"] != before[key]["count"]:
            raise ValueError(f"Measurement budget differs: {key}")
    for tiles in sorted({row["tiles"] for row in candidate["summary"]}):
        results = {"tiles": tiles}
        for mode, metric in (("matched", "expansionsPerSecond"), ("native", "decisionsPerSecond")):
            old, new = (data[tiles, mode, metric, None] for data in (before, after))
            results[mode] = {
                "rustBefore": old["rustMedian"], "rustAfter": new["rustMedian"],
                "java": new["javaMedian"], "afterVsBefore": new["rustMedian"] / old["rustMedian"],
                "rustVsJava": new["rustAdvantageRatio"], "paired95": new["pairedBootstrap95"],
                "supports20PercentLead": new["pairedBootstrap95"][0] >= 1.20,
            }
        memory = {
            key: [data[tiles, "retain", "peakRssMiB", count][column] for count in RETAIN_COUNTS]
            for key, data, column in (("rustBefore", before, "rustMedian"),
                                      ("rustAfter", after, "rustMedian"),
                                      ("java", after, "javaMedian"))
        }
        slopes = {key: 1024 * statistics.linear_regression(RETAIN_COUNTS, values).slope
                  for key, values in memory.items()}
        results["memory"] = {
            "retainedStates": RETAIN_COUNTS, "peakMiB": memory, "estimatedKiBPerState": slopes,
            "rustSlopeReduction": 1 - slopes["rustAfter"] / slopes["rustBefore"],
        }
        results["componentsNsPerExpansion"] = {
            key: {metric: data[tiles, "components", metric, None][column]
                  for metric in ("copyNs", "legalNs", "lookupNs", "applyNs")}
            for key, data, column in (("rustBefore", before, "rustMedian"),
                                      ("rustAfter", after, "rustMedian"),
                                      ("java", after, "javaMedian"))
        }
        yield results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    before, after = (json.loads(path.read_text()) for path in (args.baseline, args.candidate))
    print(json.dumps({"rustBefore": before["rustCommit"], "rustAfter": after["rustCommit"],
                      "boards": list(summarize(before, after))}, indent=2))


if __name__ == "__main__":
    main()
