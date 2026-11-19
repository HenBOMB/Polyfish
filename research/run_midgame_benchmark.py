"""Validate matched midgame branches and measure Polyfish and pinned Java Tribes."""

import argparse
import collections
from contextlib import contextmanager, ExitStack, suppress
import hashlib
from itertools import zip_longest
import json
import os
from pathlib import Path
import platform
import plistlib
import random
import selectors
import subprocess
import tempfile
import time

from run_engine_benchmark import JAVA_FLAGS, REVISION, ROOT, SIDES, checked, measure, paired_summary


FIXTURES = ROOT / "research" / "fixtures" / "midgame"
JAVA_SOURCE = ROOT / "research" / "PolyfishMidgameBench.java"
RUST_SOURCE = ROOT / "polyfish-rs" / "examples" / "midgame_bench.rs"
KINDS = ("move", "attack", "harvest", "endTurn")
BENCH_STATES = 512
RETAIN_COUNTS = (0, 128, 512, 1024)
ORACLE_REPLY_SECONDS = 30


def write_json(path, value, *, compact=False):
    if compact:
        output = json.dumps(value, separators=(",", ":"))
    else:
        output = json.dumps(value, indent=2)
    path.write_text(output + "\n")


@contextmanager
def java_bench(tribes, javafx, source):
    if checked(["git", "rev-parse", "HEAD"], tribes) != REVISION:
        raise RuntimeError(f"Check out Tribes at {REVISION}")
    if checked(["git", "status", "--porcelain", "--untracked-files=all", "--", "src", "lib"], tribes):
        raise RuntimeError("Use an unmodified Tribes source tree")
    deps = os.pathsep.join([str(tribes / "lib" / "*"), str(javafx)])
    sources = sorted(str(path) for path in (tribes / "src").rglob("*.java"))
    with tempfile.TemporaryDirectory(prefix="polyfish-midgame-java-") as classes:
        subprocess.run(["javac", "-cp", deps, "-d", classes, *sources, str(source)],
                       cwd=tribes, capture_output=True, check=True)
        yield ["java", *JAVA_FLAGS, "-cp", os.pathsep.join([classes, deps]), "core.game.PolyfishMidgameBench"]


@contextmanager
def oracle(executable, fixture, cwd):
    with tempfile.TemporaryFile(mode="w+") as errors, selectors.DefaultSelector() as ready:
        proc = subprocess.Popen([*executable, str(fixture), "oracle"], cwd=cwd,
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors)
        ready.register(proc.stdout, selectors.EVENT_READ)
        buffer = bytearray()

        def query(op, node, **params):
            deadline = time.monotonic() + ORACLE_REPLY_SECONDS
            try:
                proc.stdin.write((json.dumps(dict(op=op, node=node, **params)) + "\n").encode())
                proc.stdin.flush()
                while buffer.find(b"\n") == -1:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not ready.select(remaining):
                        raise TimeoutError(f"{executable[0]} did not reply to {op} for node {node}")
                    chunk = os.read(proc.stdout.fileno(), 65536)
                    if not chunk:
                        raise EOFError("Oracle closed stdout")
                    buffer.extend(chunk)
                end = buffer.index(b"\n")
                line = bytes(buffer[:end])
                del buffer[:end + 1]
                return json.loads(line)
            except (BrokenPipeError, EOFError, ValueError) as exc:
                errors.seek(0)
                raise RuntimeError(f"{executable[0]} failed at {op} for node {node}: {errors.read()}") from exc

        try:
            yield query
        finally:
            with suppress(BrokenPipeError):
                proc.stdin.close()
            proc.stdout.close()
            if proc.poll() is None:
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


def check_schedule(schedule):
    if not isinstance(schedule, list) or len(schedule) != max(RETAIN_COUNTS):
        raise ValueError(f"Expected exactly {max(RETAIN_COUNTS)} scheduled branches")
    for slot, branch in enumerate(schedule):
        if not isinstance(branch, dict):
            raise ValueError(f"Invalid branch at slot {slot + 1}")
        node, action = branch.get("parent"), branch.get("action")
        if (type(node) is not int or not 0 <= node <= slot or not isinstance(action, list)
                or len(action) != 3 or any(type(value) is not int for value in action)
                or not 0 <= action[0] < len(KINDS)):
            raise ValueError(f"Invalid parent or action at branch {slot + 1}")


def first_divergence(a, b):
    return next((key for key in a if a[key] != b[key]), None)


def check_branches(java, rust, schedule=None):
    if schedule is not None:
        check_schedule(schedule)
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
            summary.append(dict(tiles=side**2, mode=mode, count=count, metric=metric,
                                **paired_summary(java, rust, higher, rng)))
    return summary


def benchmark_host():
    if platform.system() == "Darwin":
        model = checked(["sysctl", "-n", "machdep.cpu.brand_string"], ROOT)
        devices = plistlib.loads(subprocess.check_output(["ioreg", "-rd1", "-c", "IOPlatformExpertDevice", "-a"]))
        machine_id = next((device["IOPlatformUUID"] for device in devices if "IOPlatformUUID" in device), None)
    elif platform.system() == "Linux":
        model = next((line.partition(":")[2].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                      if line.split(":", 1)[0].strip() == "model name"), platform.machine())
        machine_id = Path("/etc/machine-id").read_text().strip()
    else:
        raise RuntimeError("Benchmarks require macOS or Linux")
    if not machine_id:
        raise RuntimeError("Cannot identify this benchmark host")
    fingerprint = hashlib.sha256(f"polyfish-midgame-v1\n{machine_id}".encode()).hexdigest()
    return {"machine": model, "platform": platform.platform(), "hostFingerprint": fingerprint}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tribes", type=Path)
    parser.add_argument("javafx", type=Path)
    parser.add_argument("rust_binary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sides", type=int, nargs="+", choices=SIDES, default=SIDES)
    parser.add_argument("--preflight", choices=("create", "verify"))
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--batches", type=int, default=6)
    parser.add_argument("--decisions", type=int, default=2)
    parser.add_argument("--modes", nargs="+", choices=("matched", "components", "native", "retain"),
                        default=("matched", "components", "native", "retain"))
    parser.add_argument("--rust-revision", help="SHA from which the supplied Rust executable was built")
    args = parser.parse_args()
    if args.repeats < 3 or args.batches < 1 or args.decisions < 2 or args.decisions % 2:
        parser.error("Use at least three pairs, a positive batch count and an even decision count")
    if len(set(args.sides)) != len(args.sides) or len(set(args.modes)) != len(args.modes):
        parser.error("Each board side and workload mode may appear once")
    if not args.preflight and not args.rust_revision:
        parser.error("Pass --rust-revision for reproducible measurements")
    tribes = args.tribes.resolve()
    rust = [str(args.rust_binary.resolve(strict=True))]
    rust_dir = ROOT / "polyfish-rs"
    paths = {side: (FIXTURES / f"midgame-{side**2}.json", FIXTURES / f"branches-{side**2}.json")
             for side in args.sides}
    with tempfile.TemporaryDirectory(prefix="polyfish-midgame-inputs-") as directory, ExitStack() as stack:
        snapshots = Path(directory)
        hashes = {str(file.relative_to(ROOT)): hashlib.sha256(file.read_bytes()).hexdigest()
                  for file in (Path(__file__).resolve(), RUST_SOURCE, ROOT / "research/run_engine_benchmark.py")}

        def snapshot(file):
            content = file.read_bytes()
            hashes[str(file.relative_to(ROOT))] = hashlib.sha256(content).hexdigest()
            copy = snapshots / file.name
            copy.write_bytes(content)
            return copy

        java_source = snapshot(JAVA_SOURCE)
        if args.preflight != "create":
            paths = {side: (snapshot(fixture), snapshot(schedule)) for side, (fixture, schedule) in paths.items()}
            try:
                for _, schedule in paths.values():
                    check_schedule(json.loads(schedule.read_text()))
            except ValueError as exc:
                parser.error(f"{schedule.name}: {exc}")
        host = benchmark_host() if not args.preflight else {}
        java = stack.enter_context(java_bench(tribes, args.javafx.resolve(), java_source))
        if args.preflight:
            checks = {}
            for side, (fixture, schedule_path) in paths.items():
                with ExitStack() as oracles:
                    j = oracles.enter_context(oracle(java, fixture, tribes))
                    r = oracles.enter_context(oracle(rust, fixture, rust_dir))
                    schedule = None if args.preflight == "create" else json.loads(schedule_path.read_text())
                    branches, checks[side**2] = check_branches(j, r, schedule)
                    if args.preflight == "create":
                        write_json(schedule_path, branches, compact=True)
                    print(side**2, checks[side**2]["actionCounts"], flush=True)
            write_json(args.output, checks)
            return

        rng = random.Random(422)
        raw = []
        modes = {
            "matched": [(args.batches, args.batches * 2)],
            "components": [(args.batches, args.batches * 2)],
            "native": [(args.decisions, args.decisions * 2)],
            "retain": [(count, 0) for count in RETAIN_COUNTS],
        }
        checkpoint = args.output.with_suffix(".checkpoint.jsonl")
        with checkpoint.open("x") as journal:
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
                        row = dict(row, mode=mode, repeat=repeat, order=order, count=count, warmup=warm)
                        raw.append(row)
                        journal.write(json.dumps(row) + "\n")
                        journal.flush()
                print(f"Completed {repeat + 1}/{args.repeats} pairs per workload", flush=True)
        write_json(args.output, {
            "rustCommit": args.rust_revision, "tribesCommit": REVISION, "javaFlags": JAVA_FLAGS,
            **host,
            "java": checked(["java", "--version"], tribes), "rust": checked(["rustc", "--version", "--verbose"], ROOT),
            "sha256": hashes,
            "raw": raw, "summary": percentile_summary(raw),
        })
        checkpoint.unlink()


if __name__ == "__main__":
    main()
