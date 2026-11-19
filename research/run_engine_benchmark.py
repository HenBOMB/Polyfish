"""Compare Java Tribes and Rust Polyfish on fixed model-free simulator workloads."""

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import re
import statistics
import subprocess
import tempfile
import time

from run_engine_pilot import REVISION, ROOT, checked


SIDES = (11, 14, 16, 18, 20, 30)
OPS = ("cloneNs", "viewNs", "legalNs", "stepNs", "endTurnNs")
JAVA_FLAGS = ("-Xms32m", "-Xmx512m", "-Djava.awt.headless=true")
HARNESS = (
    "research/PolyfishBench.java",
    "polyfish-rs/examples/research_bench.rs",
    "research/run_engine_benchmark.py",
    "research/run_engine_pilot.py",
)


def measure(command, cwd):
    if platform.system() == "Darwin":
        timer = ["/usr/bin/time", "-l"]
        rss_pattern = r"(\d+)\s+maximum resident set size"
        rss_multiplier = 1
    else:
        timer = ["/usr/bin/time", "-f", "peak_rss_kib %M"]
        rss_pattern = r"peak_rss_kib (\d+)"
        rss_multiplier = 1024
    start = time.perf_counter_ns()
    result = subprocess.run([*timer, *command], cwd=cwd, text=True, capture_output=True,
                            env={**os.environ, "LC_ALL": "C"})
    elapsed = time.perf_counter_ns() - start
    if result.returncode:
        raise RuntimeError(f"{command} exited with {result.returncode}\n{result.stdout}\n{result.stderr}")
    rss = int(re.search(rss_pattern, result.stderr).group(1)) * rss_multiplier
    return dict(json.loads(result.stdout), peakRssBytes=rss, wallElapsedNs=elapsed)


def paired_summary(java, rust, higher_is_better, rng):
    median = statistics.median
    numerator, denominator = (rust, java) if higher_is_better else (java, rust)
    draws = []
    for _ in range(4000):
        indices = rng.choices(range(len(java)), k=len(java))
        draws.append(median(numerator[i] for i in indices) / median(denominator[i] for i in indices))
    draws.sort()
    return {
        "javaMedian": median(java),
        "rustMedian": median(rust),
        "rustAdvantageRatio": median(numerator) / median(denominator),
        "pairedBootstrap95": [draws[100], draws[3900]],
    }


def summarize(records):
    pairs = defaultdict(dict)
    for row in records:
        pairs[row["size"], row["mode"], row["repeat"]][row["engine"]] = row
    grouped = defaultdict(list)
    for (size, mode, _), pair in sorted(pairs.items()):
        java, rust = pair["tribes"], pair["polyfish"]
        stats = {"peakRssMiB": (java["peakRssBytes"] / 2**20, rust["peakRssBytes"] / 2**20)}
        if mode == "ops":
            for metric in OPS:
                stats[metric] = java[metric] / java["iterations"], rust[metric] / rust["iterations"]
        if mode == "trace":
            assert java["actionsPerTrace"] == rust["actionsPerTrace"] == 40
            stats["actionsPerSecond"] = tuple(
                row["actionsPerTrace"] * row["iterations"] * 1e9 / row["traceNs"]
                for row in (java, rust)
            )
        for metric, values in stats.items():
            grouped[size, mode, metric].append(values)
    rng = random.Random(8585)
    return [
        dict(side=size, tiles=size**2, mode=mode, metric=metric,
             **paired_summary(*zip(*samples), metric == "actionsPerSecond", rng))
        for (size, mode, metric), samples in sorted(grouped.items())
    ]


def source_metadata(tribes):
    engine_paths = ["polyfish-rs/src", "polyfish-rs/Cargo.lock", "polyfish-rs/Cargo.toml"]
    source = ROOT / "SOURCE_COMMIT"
    polyfish_revision = source.read_text().strip() if source.exists() else checked(["git", "rev-parse", "HEAD"], ROOT)
    if not source.exists() and checked(["git", "status", "--porcelain", "--", *engine_paths], ROOT):
        raise RuntimeError("Use an unmodified Rust engine source tree")
    if checked(["git", "status", "--porcelain", "--", "src", "lib"], tribes):
        raise RuntimeError("Use an unmodified Tribes source tree")
    machine_cmd = ["sysctl", "-n", "machdep.cpu.brand_string"] if platform.system() == "Darwin" else ["uname", "-m"]
    info = {
        "polyfishCommit": polyfish_revision,
        "tribesCommit": REVISION,
        "machine": checked(machine_cmd, ROOT),
        "platform": platform.platform(),
        "java": checked(["java", "--version"], ROOT),
        "rust": checked(["rustc", "--version", "--verbose"], ROOT),
        "javaFlags": JAVA_FLAGS,
        "harnessSha256": {file: hashlib.sha256((ROOT / file).read_bytes()).hexdigest() for file in HARNESS},
    }
    if platform.system() == "Darwin":
        info["hostMemoryBytes"] = int(checked(["sysctl", "-n", "hw.memsize"], ROOT))
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tribes", type=Path)
    parser.add_argument("javafx_base", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--target-dir", type=Path, default=ROOT / "polyfish-rs" / "target")
    parser.add_argument("--sides", nargs="+", type=int, choices=SIDES, default=SIDES)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--op-iterations", type=int, default=20000)
    parser.add_argument("--traces", type=int, default=250)
    parser.add_argument("--copies", type=int, default=512)
    args = parser.parse_args()
    if min(args.op_iterations, args.traces, args.copies) < 1 or args.repeats < 3:
        parser.error("Use positive workloads and at least three process pairs")
    if len(set(args.sides)) != len(args.sides):
        parser.error("Each side length may appear once")
    csv_output = args.output.with_suffix(".csv")
    if (args.output.suffix.lower() == ".csv" or args.output.resolve() == csv_output.resolve()
            or (args.output.exists() and csv_output.exists() and args.output.samefile(csv_output))):
        parser.error("JSON and CSV reports must have distinct output paths")

    tribes = args.tribes.resolve()
    if checked(["git", "rev-parse", "HEAD"], tribes) != REVISION:
        parser.error(f"Check out Tribes at {REVISION}")
    metadata = source_metadata(tribes)
    rust_dir = ROOT / "polyfish-rs"
    target = args.target_dir.resolve()
    subprocess.run(["cargo", "build", "--locked", "--release", "--no-default-features",
                    "--example", "research_bench", "--target-dir", str(target)], cwd=rust_dir, check=True)

    started = datetime.now(timezone.utc).isoformat()
    report = []
    rng = random.Random(420)
    with tempfile.TemporaryDirectory(prefix="polyfish-java-bench-") as classes:
        deps = os.pathsep.join([str(tribes / "lib" / "*"), str(args.javafx_base.resolve())])
        java_src = sorted(str(file) for file in (tribes / "src").rglob("*.java"))
        subprocess.run(["javac", "-cp", deps, "-d", classes, *java_src,
                        str(ROOT / "research" / "PolyfishBench.java")], cwd=tribes, check=True)
        classpath = os.pathsep.join([classes, deps])
        java = ["java", *JAVA_FLAGS, "-cp", classpath, "core.game.PolyfishBench"]
        rust = [str(target / "release" / "examples" / "research_bench")]
        for repeat in range(args.repeats):
            tests = [(side, mode, count) for side in args.sides for mode, count in
                     [("ops", args.op_iterations), ("trace", args.traces), ("retain", args.copies)]]
            rng.shuffle(tests)
            for side, mode, count in tests:
                warmup = 0 if mode == "retain" else count * 2
                engines = [(java, side, tribes), (rust, side**2, rust_dir)]
                rng.shuffle(engines)
                for order, (executable, size, cwd) in enumerate(engines):
                    command = [*executable, str(size), str(count), mode, str(warmup)]
                    row = measure(command, cwd)
                    assert row["size"] == side
                    report.append(dict(row, repeat=repeat, order=order, mode=mode, warmup=warmup))
            print(f"Completed {repeat + 1}/{args.repeats} process pairs per workload", flush=True)

    results = summarize(report)
    args.output.write_text(json.dumps({
        **metadata,
        "startedUtc": started,
        "completedUtc": datetime.now(timezone.utc).isoformat(),
        "fixture": "Flat two-capital opening with one warrior per seat and fog of war",
        "trace": "Ten rounds, each seat alternates a one-tile diagonal step and EndTurn, enumerating native legal actions before every advance",
        "memory": "Maximum resident set size of each process, including runtime, warm-up where applicable, and harness",
        "ratio": "Values above one favor Rust. Java/Rust for timings and RSS, Rust/Java for actions/s",
        "computations": "Median of process means for nanoseconds per operation, median of process rates for trace throughput, 4000 resampled process pairs",
        "raw": report,
        "summary": results,
    }, indent=2) + "\n")
    with csv_output.open("w", newline="") as file:
        fieldnames = ("side", "tiles", "mode", "metric", "javaMedian", "rustMedian", "rustAdvantageRatio", "pairedBootstrap95")
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
