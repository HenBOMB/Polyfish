"""Validate matched midgame branches and measure Polyfish and pinned Java Tribes."""

import argparse
import collections
from contextlib import contextmanager, ExitStack
import hashlib
from itertools import zip_longest
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import tempfile

from run_engine_benchmark import JAVA_FLAGS, REVISION, ROOT, SIDES, checked, measure, paired_summary


FIXTURES = ROOT / "research" / "fixtures" / "midgame"
JAVA_SOURCE = ROOT / "research" / "PolyfishMidgameBench.java"
RUST_SOURCE = ROOT / "polyfish-rs" / "examples" / "midgame_bench.rs"
KINDS = ("move", "attack", "harvest", "endTurn")
BENCH_STATES = 512
RETAIN_COUNTS = (0, 128, 512, 1024)


def write_json(path, value, *, compact=False):
    if compact:
        output = json.dumps(value, separators=(",", ":"))
    else:
        output = json.dumps(value, indent=2)
    path.write_text(output + "\n")


@contextmanager
def java_bench(tribes, javafx):
    if checked(["git", "rev-parse", "HEAD"], tribes) != REVISION:
        raise RuntimeError(f"Check out Tribes at {REVISION}")
    deps = os.pathsep.join([str(tribes / "lib" / "*"), str(javafx)])
    sources = sorted(str(path) for path in (tribes / "src").rglob("*.java"))
    with tempfile.TemporaryDirectory(prefix="polyfish-midgame-java-") as classes:
        subprocess.run(["javac", "-cp", deps, "-d", classes, *sources, str(JAVA_SOURCE)],
                       cwd=tribes, capture_output=True, check=True)
        yield ["java", *JAVA_FLAGS, "-cp", os.pathsep.join([classes, deps]), "core.game.PolyfishMidgameBench"]


@contextmanager
def oracle(executable, fixture, cwd):
    with tempfile.TemporaryFile(mode="w+") as errors:
        proc = subprocess.Popen([*executable, str(fixture), "oracle"], cwd=cwd,
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors, text=True)

        def query(op, node, **params):
            proc.stdin.write(json.dumps(dict(op=op, node=node, **params)) + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
            try:
                return json.loads(line)
            except ValueError as exc:
                errors.seek(0)
                raise RuntimeError(f"{executable[0]} returned {line!r}: {errors.read()}") from exc

        try:
            yield query
        finally:
            proc.stdin.close()
            proc.stdout.close()
            if proc.poll() is None:
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    proc.wait()


def first_divergence(a, b):
    return next((key for key in a if a[key] != b[key]), None)


def check_branches(java, rust, schedule=None):
    roots = [engine("inspect", 0)["projection"] for engine in (java, rust)]
    if roots[0] != roots[1]:
        key = first_divergence(*roots)
        for java_value, rust_value in zip(roots[0][key], roots[1][key]) if isinstance(roots[0][key], list) else []:
            if java_value != rust_value:
                raise RuntimeError(f"Root fixture differs: {key}, Java {java_value}, Rust {rust_value}")
        raise RuntimeError(f"Root fixture differs: {key}, Java {roots[0][key]}, Rust {roots[1][key]}")

    totals = collections.Counter()
    rejected = collections.Counter()
    action_mix = collections.Counter()
    accepted = []
    depths = [0]
    digest = hashlib.sha256()

    def inspect(node):
        left, right = (engine("inspect", node) for engine in (java, rust))
        j, r = (set(map(tuple, report["actions"])) for report in (left, right))
        common = j & r
        totals.update(inspectedNodes=1, javaLegal=left["legal"], rustLegal=right["legal"],
                      javaOnlyMapped=len(j - r), rustOnlyMapped=len(r - j),
                      sharedMapped=len(common), javaOutsideSubset=left["legal"] - len(j),
                      rustOutsideSubset=right["legal"] - len(r))
        return common

    def attempt(node, action):
        slot = len(accepted) + 1
        a, b = (engine("expand", node, action=action, slot=slot) for engine in (java, rust))
        difference = first_divergence(a, b)
        kind = KINDS[action[0]]
        if difference is not None:
            rejected[f"{kind}/{difference}"] += 1
            return False
        digest.update(json.dumps(a, sort_keys=True, separators=(",", ":")).encode())
        accepted.append({"parent": node, "action": action})
        depths.append(depths[node] + 1)
        action_mix[kind] += 1
        return True

    if schedule is None:
        rng = random.Random(420)
        for node in range(max(RETAIN_COUNTS)):
            if node > len(accepted) or len(accepted) >= max(RETAIN_COUNTS):
                break
            choices = sorted(inspect(node))
            rng.shuffle(choices)
            # Give each supported action family a branch before filling remaining slots.
            preference = [3, 1, 2, 0]
            groups = [[action for action in choices if action[0] == kind] for kind in preference]
            balanced = (action for column in zip_longest(*groups) for action in column if action is not None)
            target = min(len(accepted) + 4, max(RETAIN_COUNTS))
            for action in balanced:
                attempt(node, action)
                if len(accepted) == target:
                    break
        if len(accepted) != max(RETAIN_COUNTS):
            raise RuntimeError(f"Produced {len(accepted)} branches, exclusions {dict(rejected)}")
    else:
        current_node = None
        for branch in schedule:
            node, action = branch["parent"], tuple(branch["action"])
            if node != current_node:
                common = inspect(node)
                current_node = node
            if action not in common or not attempt(node, action):
                raise RuntimeError(f"Stored branch {branch} diverged: {dict(rejected)}")

    primary = collections.Counter(KINDS[branch["action"][0]] for branch in accepted[:BENCH_STATES])
    if set(primary) != set(KINDS):
        raise RuntimeError(f"Primary schedule must cover all action families, got {dict(primary)}")
    return accepted, {
        "actionCounts": dict(primary), "fullActionCounts": dict(action_mix),
        "maxDepth": max(depths), "projectionSha256": digest.hexdigest(),
        "legalCounts": dict(totals), "rejectedTransitions": dict(rejected),
    }


def percentile_summary(raw):
    groups = collections.defaultdict(list)
    for row in raw:
        groups[row["size"], row["mode"], row["count"]].append(row)
    rng = random.Random(421)
    summary = []
    for (side, mode, count), rows in sorted(groups.items()):
        by_engine = {engine: sorted(
            (row for row in rows if row["engine"] == engine), key=lambda row: row["repeat"]
        ) for engine in ("tribes", "polyfish")}
        fields = {"peakRssMiB": (lambda row: row["peakRssBytes"] / 2**20, False)}
        if mode == "native":
            fields["decisionsPerSecond"] = (lambda row: row["decisions"] * 1e9 / row["nanos"], True)
        elif mode == "matched":
            fields["expansionsPerSecond"] = (lambda row: row["expansions"] * 1e9 / row["nanos"], True)
        elif mode == "components":
            for field in ("copyNs", "legalNs", "lookupNs", "applyNs"):
                fields[field] = (lambda row, field=field: row[field] / row["expansions"], False)
        for metric, (value, higher) in fields.items():
            java = [value(row) for row in by_engine["tribes"]]
            rust = [value(row) for row in by_engine["polyfish"]]
            ratio, interval = paired_summary(java, rust, higher, rng)
            summary.append(dict(tiles=side**2, mode=mode, count=count, metric=metric,
                                javaMedian=statistics.median(java), rustMedian=statistics.median(rust),
                                rustAdvantageRatio=ratio, pairedBootstrap95=interval))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tribes", type=Path)
    parser.add_argument("javafx", type=Path)
    parser.add_argument("rust_binary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sides", type=int, nargs="+", choices=SIDES, default=SIDES)
    parser.add_argument("--preflight", choices=("create", "verify"))
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--decisions", type=int, default=4)
    parser.add_argument("--modes", nargs="+", choices=("matched", "components", "native", "retain"),
                        default=("matched", "components", "native", "retain"))
    parser.add_argument("--rust-revision", help="SHA from which the supplied Rust executable was built")
    args = parser.parse_args()
    if args.repeats < 3 or args.batches < 1 or args.decisions < 2 or args.decisions % 2:
        parser.error("Use at least three pairs, a positive batch count and an even decision count")
    tribes = args.tribes.resolve()
    rust = [str(args.rust_binary.resolve(strict=True))]
    rust_dir = ROOT / "polyfish-rs"
    paths = {side: (FIXTURES / f"midgame-{side**2}.json", FIXTURES / f"branches-{side**2}.json")
             for side in args.sides}
    with java_bench(tribes, args.javafx.resolve()) as java:
        if args.preflight:
            checks = {}
            for side, (fixture, schedule_path) in paths.items():
                with ExitStack() as stack:
                    j = stack.enter_context(oracle(java, fixture, tribes))
                    r = stack.enter_context(oracle(rust, fixture, rust_dir))
                    schedule = None if args.preflight == "create" else json.loads(schedule_path.read_text())
                    branches, checks[side**2] = check_branches(j, r, schedule)
                    if args.preflight == "create":
                        write_json(schedule_path, branches, compact=True)
                    print(side**2, checks[side**2]["actionCounts"], flush=True)
            write_json(args.output, checks)
            return

        if not args.rust_revision:
            parser.error("Pass --rust-revision for reproducible measurements")
        rng = random.Random(422)
        raw = []
        modes = {
            "matched": [(args.batches, args.batches * 2)],
            "components": [(args.batches, args.batches * 2)],
            "native": [(args.decisions, args.decisions * 2)],
            "retain": [(count, 0) for count in RETAIN_COUNTS],
        }
        for repeat in range(args.repeats):
            workloads = [(side, mode, count, warm) for side in args.sides
                         for mode in args.modes for count, warm in modes[mode]]
            rng.shuffle(workloads)
            for side, mode, count, warm in workloads:
                fixture, schedule = paths[side]
                engines = [(java, tribes), (rust, rust_dir)]
                rng.shuffle(engines)
                for order, (executable, cwd) in enumerate(engines):
                    row = measure([*executable, str(fixture), mode, str(count), str(warm), str(schedule)], cwd)
                    assert row["size"] == side
                    raw.append(dict(row, mode=mode, repeat=repeat, order=order, count=count, warmup=warm))
            print(f"Completed {repeat + 1}/{args.repeats} pairs per workload", flush=True)
        inputs = [Path(__file__), JAVA_SOURCE, RUST_SOURCE, ROOT / "research/run_engine_benchmark.py",
                  *(file for pair in paths.values() for file in pair)]
        write_json(args.output, {
            "rustCommit": args.rust_revision, "tribesCommit": REVISION, "javaFlags": JAVA_FLAGS,
            "machine": checked(["sysctl", "-n", "machdep.cpu.brand_string"], ROOT) if os.uname().sysname == "Darwin" else os.uname().machine,
            "java": checked(["java", "-version"], tribes), "rust": checked(["rustc", "--version", "--verbose"], ROOT),
            "sha256": {str(file.relative_to(ROOT)): hashlib.sha256(file.read_bytes()).hexdigest() for file in inputs},
            "raw": raw, "summary": percentile_summary(raw),
        })


if __name__ == "__main__":
    main()
