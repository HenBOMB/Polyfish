"""Repeat the fixed-board Java and Rust simulator measurements."""

import argparse
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import tempfile
import time


REVISION = "015a551d1a6be69b5931788fcc418e4c51eb8ea1"
ROOT = Path(__file__).resolve().parents[1]
OPERATIONS = ("cloneNs", "viewNs", "legalNs", "endTurnNs")


def checked(command, cwd):
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def ratio_interval(java, rust, rng):
    ratio = statistics.median(java) / statistics.median(rust)
    draws = sorted(statistics.median(rng.choices(java, k=len(java))) /
                   statistics.median(rng.choices(rust, k=len(rust))) for _ in range(2000))
    return {"ratio": ratio, "bootstrap95": [draws[50], draws[1950]]}


def native_runs(tribes, classes, java, rust_dir):
    config = json.loads((ROOT / "research" / "tribes-pilot.json").read_text())
    config["Players"] = ["Rule Based", "Rule Based"]
    path = Path(classes) / "tribes-native.json"
    path.write_text(json.dumps(config))
    start = time.perf_counter_ns()
    results = checked([*java, "Tournament", str(path)], tribes)
    java_elapsed = time.perf_counter_ns() - start
    assert results.count("Game Results:") == 4, results[-1200:]
    records = [{"engine": "tribes", "workload": "native 11×11 Capitals", "games": 4,
                "elapsedNs": java_elapsed, "rawLog": results}]

    binary = str(rust_dir / "target" / "release" / "research_match")
    for tiles in (121, 196, 256):
        start = time.perf_counter_ns()
        matches = [
            json.loads(checked([binary, "--map-size", str(tiles), "--seed", str(seed),
                                "--preset1", "normal", "--preset2", "normal"], rust_dir))
            for seed in (42, 43, 44, 45)
        ]
        records.append({"engine": "polyfish", "workload": f"native {tiles}-tile Drylands Might",
                        "games": len(matches), "elapsedNs": time.perf_counter_ns() - start,
                        "rawMatches": matches})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tribes", type=Path, help="Pinned Tribes reference checkout")
    parser.add_argument("javafx_base", type=Path, help="JavaFX Base jar for this host")
    parser.add_argument("output", type=Path)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.iterations < 1 or args.repeats < 3:
        parser.error("Use at least one iteration and three independent process repeats")
    tribes = args.tribes.resolve()
    if checked(["git", "rev-parse", "HEAD"], tribes) != REVISION:
        parser.error(f"Check out Tribes at {REVISION}")

    rust_dir = ROOT / "polyfish-rs"
    subprocess.run(["cargo", "build", "--locked", "--release", "--no-default-features",
                    "--example", "research_bench", "--bin", "research_match"], cwd=rust_dir, check=True)
    with tempfile.TemporaryDirectory(prefix="polyfish-tribes-") as classes:
        dependencies = os.pathsep.join([str(tribes / "lib" / "*"), str(args.javafx_base.resolve())])
        sources = sorted(str(source) for source in (tribes / "src").rglob("*.java"))
        subprocess.run(["javac", "-cp", dependencies, "-d", classes, *sources,
                        str(ROOT / "research" / "PolyfishBench.java")], cwd=tribes, check=True)
        classpath = os.pathsep.join([classes, dependencies])
        java = ["java", "-Xms512m", "-Xmx2g", "-Djava.awt.headless=true", "-cp", classpath]
        rust = [str(rust_dir / "target" / "release" / "examples" / "research_bench")]
        records = []
        for size in (11, 14, 16):
            for repeat in range(args.repeats):
                for command, value, cwd in [([*java, "core.game.PolyfishBench"], size, tribes),
                                            (rust, size * size, rust_dir)]:
                    row = json.loads(checked([*command, str(value), str(args.iterations)], cwd))
                    records.append(dict(row, repeat=repeat))
        native = native_runs(tribes, classes, java, rust_dir)

    rng = random.Random(42)
    summaries = {}
    for size in (11, 14, 16):
        rows = [row for row in records if row["size"] == size]
        summaries[size] = {
            operation: ratio_interval(
                [row[operation] for row in rows if row["engine"] == "tribes"],
                [row[operation] for row in rows if row["engine"] == "polyfish"],
                rng,
            )
            for operation in OPERATIONS
        }
    machine = checked(["sysctl", "-n", "machdep.cpu.brand_string"], ROOT) if platform.system() == "Darwin" else platform.processor()
    source = ROOT / "SOURCE_COMMIT"
    revision = source.read_text().strip() if source.exists() else checked(["git", "rev-parse", "HEAD"], ROOT)
    tracked_dirty = not source.exists() and bool(checked(["git", "status", "--short", "--untracked-files=no"], ROOT))
    report = {
        "fixture": "flat two-capital opening, same board size and end-turn action",
        "ratios": "descriptive Java/Rust ratios; differing candidate sets are not a matched legal workload",
        "tribesCommit": REVISION,
        "polyfishCommit": revision,
        "polyfishTrackedSourceDirty": tracked_dirty,
        "machine": machine,
        "platform": platform.platform(),
        "java": checked(["java", "--version"], ROOT),
        "rust": checked(["rustc", "--version"], ROOT),
        "measurements": records,
        "summaries": summaries,
        "nativeRuns": native,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
