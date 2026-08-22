#!/usr/bin/env python3
"""Verify that every long flag the shell drivers pass to a CLI actually parses.

The shell scripts form an unchecked CLI contract with the binaries in src/bin/
and with the python CLIs beside them: renaming a clap field or an argparse
option breaks training at runtime and nothing else notices. This extracts the
flags each script passes to each target and diffs them against that target's
--help. Python targets are keyed per subcommand ("ladder.py freeze"), since
each subparser accepts a different set — and `freeze` / `audit-opponents` are
reached from run_training_loop.sh alone, on a branch no test had ever entered
(#35).

Fails closed: a script it cannot parse, a target whose --help it cannot read,
or a flag-bearing shell variable it cannot resolve is an error, not a pass.
"""

import argparse
import os
import re
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE_ROOT = os.path.dirname(REPO_ROOT)

# Shell drivers to check, with the binaries each one must be seen invoking.
# A required binary that goes missing fails loudly, so a restructured script
# cannot pass by yielding an empty flag set.
CONTRACTS = {
    "polyfish-rs/run_training_loop.sh": {
        "self_play", "arena", "polyfish",
        "ladder.py active", "ladder.py record", "ladder.py freeze",
        "ladder.py audit-opponents", "training_log.py append-row",
    },
    "polyfish-rs/bisect_arm.sh": {"self_play"},
    "polyfish-rs/bench_actor_ceiling.sh": {"actor_ceiling"},
    "polyfish-rs/bench_eval_sweep.sh": {"self_play"},
    "run-server.sh": {"polyfish"},
    "auto_train.sh": set(),
}

# Shell words allowed to precede a binary inside one command segment.
PREFIX_WORDS = {
    "if", "then", "do", "!", "exec", "command", "time", "env", "sudo", "nohup",
    "local", "declare", "typeset", "export", "readonly", "-a",
}

# Python CLIs the shell drivers drive, by basename. `python3 -c '...'` never
# matches: the interpreter must be followed by a .py path.
PY_SCRIPTS = {"ladder.py", "training_log.py"}
PY_INVOKE_RE = re.compile(
    r"(?:^|[\s(<`\"'])[\w./-]*python3?\s+([\w./-]+\.py)(?:\s+([a-z][a-z0-9-]*))?"
)

FLAG_RE = re.compile(r"(?<![\w-])--[a-z][a-z0-9]*(?:-[a-z0-9]+)*")
VAR_RE = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?")
# Variables whose name says they carry CLI arguments; an unresolved one is fatal.
FLAG_VAR_NAME_RE = re.compile(r"_(FLAG|FLAGS|ARGS|OPTS)$")
ASSIGN_RE = re.compile(r"^(?:local\s+(?:-a\s+)?|export\s+)?([A-Za-z_][A-Za-z0-9_]*)\+?=(.*)$", re.S)


class ContractError(Exception):
    pass


def known_binaries():
    bins = {"polyfish"}
    bin_dir = os.path.join(REPO_ROOT, "src", "bin")
    if not os.path.isdir(bin_dir):
        raise ContractError(f"cannot list binaries: {bin_dir} does not exist")
    for entry in os.listdir(bin_dir):
        if entry.endswith(".rs"):
            bins.add(entry[:-3])
    return bins


def strip_comments(text):
    """Drop shell comments, tracking quote state across lines."""
    lines = []
    quote = None
    for line in text.splitlines():
        out = []
        i = 0
        while i < len(line):
            ch = line[i]
            if quote:
                out.append(ch)
                if ch == quote:
                    quote = None
                elif ch == "\\" and quote == '"' and i + 1 < len(line):
                    i += 1
                    out.append(line[i])
                i += 1
                continue
            if ch in "\"'":
                quote = ch
            elif ch == "\\" and i + 1 < len(line):
                out.append(ch)
                i += 1
                out.append(line[i])
                i += 1
                continue
            elif ch == "#" and (not out or out[-1].isspace()):
                break
            out.append(ch)
            i += 1
        lines.append("".join(out))
    if quote:
        raise ContractError("unterminated quote at end of file")
    return lines


def paren_balance(line):
    """Net ( ... ) depth outside quotes."""
    depth = 0
    quote = None
    for ch in line:
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
    return depth


ARRAY_OPEN_RE = re.compile(r"^(?:local\s+(?:-a\s+)?|declare\s+-a\s+)?[A-Za-z_][A-Za-z0-9_]*\+?=\(")


def logical_lines(text):
    """Join backslash continuations and multi-line array literals."""
    joined = []
    buf = ""
    for line in strip_comments(text):
        buf = f"{buf} {line.strip()}" if buf else line.rstrip()
        if buf.endswith("\\"):
            buf = buf[:-1]
            continue
        joined.append(buf.strip())
        buf = ""
    if buf.strip():
        joined.append(buf.strip())

    out = []
    pending = None
    for line in joined:
        if pending is None:
            if ARRAY_OPEN_RE.match(line) and paren_balance(line) > 0:
                pending = line
                continue
            out.append(line)
            continue
        pending = f"{pending} {line}"
        if paren_balance(pending) <= 0:
            out.append(pending)
            pending = None
    if pending is not None:
        raise ContractError(f"unterminated array literal at: {pending[:80]!r}")
    return out


def split_segments(line):
    segments = []
    quote = None
    cur = ""
    i = 0
    while i < len(line):
        ch = line[i]
        if quote:
            cur += ch
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in "\"'":
            quote = ch
            cur += ch
            i += 1
            continue
        if ch in "|;&":
            run = 1
            while i + run < len(line) and line[i + run] == ch:
                run += 1
            segments.append(cur)
            cur = ""
            i += run
            continue
        cur += ch
        i += 1
    segments.append(cur)
    return [s.strip() for s in segments if s.strip()]


def token_core(token):
    """Strip shell decoration so `VAR=$("$ARENA_BIN"` resolves to `$ARENA_BIN`."""
    core = token
    changed = True
    while changed:
        changed = False
        m = re.match(r"^[A-Za-z_][A-Za-z0-9_]*\+?=", core)
        if m:
            core = core[m.end():]
            changed = True
        for prefix in ("$(", "(", "`"):
            if core.startswith(prefix):
                core = core[len(prefix):]
                changed = True
        if core[:1] in ("\"", "'"):
            core = core[1:]
            changed = True
    return core.rstrip("\"'`)")


def resolve_binary(token, aliases, bins):
    core = token_core(token)
    m = re.fullmatch(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?", core)
    if m:
        return aliases.get(m.group(1))
    if not core:
        return None
    name = os.path.basename(core)
    if name in bins:
        return name
    return None


def strip_command_subs(value):
    """Drop $( ... ) and ` ... ` regions: their flags belong to the inner command."""
    out = []
    depth = 0
    i = 0
    while i < len(value):
        if value.startswith("$(", i):
            depth += 1
            i += 2
            continue
        if depth:
            if value[i] == "(":
                depth += 1
            elif value[i] == ")":
                depth -= 1
            i += 1
            continue
        if value[i] == "`":
            end = value.find("`", i + 1)
            i = len(value) if end < 0 else end + 1
            continue
        out.append(value[i])
        i += 1
    return "".join(out)


def collect_assignments(lines, bins):
    """Map variables to the binary or the flag list they hold."""
    aliases, flag_vars = {}, {}
    for line in lines:
        for segment in split_segments(line):
            m = ASSIGN_RE.match(segment)
            if not m:
                continue
            name, value = m.group(1), m.group(2).strip()
            if not value or " " not in value.strip("\"'"):
                target = resolve_binary(value, aliases, bins)
                if target:
                    aliases[name] = target
                    continue
            flags = FLAG_RE.findall(strip_command_subs(value))
            if flags:
                flag_vars.setdefault(name, set()).update(flags)
    return aliases, flag_vars


def cargo_run_target(tokens, bins):
    """Resolve `cargo run ... --bin NAME [-- flags]` to (binary, flag tokens)."""
    if not tokens or token_core(tokens[0]) != "cargo" or "run" not in tokens:
        return None, None
    try:
        name = tokens[tokens.index("--bin") + 1]
    except (ValueError, IndexError):
        raise ContractError(f"cannot read --bin from: {' '.join(tokens)[:80]!r}")
    if name not in bins:
        raise ContractError(f"cargo run --bin {name} is not a known binary")
    rest_at = tokens.index("--") + 1 if "--" in tokens else len(tokens)
    return name, rest_at


def resolve_flag_vars(rest, flag_vars, path, target):
    """Long flags in one argument region, with flag-bearing variables expanded."""
    found = set(FLAG_RE.findall(rest))
    for var in VAR_RE.findall(rest):
        if var in flag_vars:
            found |= flag_vars[var]
        elif FLAG_VAR_NAME_RE.search(var):
            raise ContractError(
                f"{path}: cannot resolve ${var} passed to {target} — "
                "the extractor would silently skip its flags"
            )
    return found


def collect_python_usage(lines, flag_vars, path):
    """Return {"script.py subcommand": {flags}} for the python CLIs a script drives."""
    usage = {}
    for line in lines:
        for segment in split_segments(line):
            calls = list(PY_INVOKE_RE.finditer(segment))
            for idx, call in enumerate(calls):
                if os.path.basename(call.group(1)) not in PY_SCRIPTS:
                    continue
                target = os.path.basename(call.group(1))
                if call.group(2):
                    target = f"{target} {call.group(2)}"
                # Stop at the next invocation in the same segment, so one
                # command's flags are never charged to the one before it.
                end = calls[idx + 1].start() if idx + 1 < len(calls) else len(segment)
                rest = strip_command_subs(segment[call.end():end])
                usage.setdefault(target, set()).update(
                    resolve_flag_vars(rest, flag_vars, path, target)
                )
    return usage


def collect_usage(path, bins):
    """Return {target: {flags}} used by one shell script."""
    with open(path, encoding="utf-8") as fh:
        lines = logical_lines(fh.read())
    aliases, flag_vars = collect_assignments(lines, bins)
    usage = collect_python_usage(lines, flag_vars, path)
    array_owner = {}

    for line in lines:
        for segment in split_segments(line):
            m = ASSIGN_RE.match(segment)
            array_name = None
            if m and m.group(2).strip().startswith("("):
                array_name = m.group(1)
            tokens = segment.split()
            target, rest_at = cargo_run_target(tokens, bins)
            for idx, token in enumerate(tokens if target is None else []):
                cand = resolve_binary(token, aliases, bins)
                if cand:
                    target, rest_at = cand, idx + 1
                    break
                if not (
                    token in PREFIX_WORDS
                    or re.match(r"^[A-Za-z_][A-Za-z0-9_]*\+?=", token)
                    or token in ("(", "$(")
                ):
                    break
            if target is None and array_name in array_owner:
                target, rest_at = array_owner[array_name], 0
            if target is None:
                continue
            if array_name:
                array_owner[array_name] = target
            rest = " ".join(tokens[rest_at:])
            usage.setdefault(target, set()).update(
                resolve_flag_vars(rest, flag_vars, path, target)
            )
    return usage


def python_help_flags(target):
    """Long flags one `script.py [subcommand] --help` accepts."""
    script, _, subcommand = target.partition(" ")
    cmd = [sys.executable, os.path.join(REPO_ROOT, script)]
    if subcommand:
        cmd.append(subcommand)
    cmd.append("--help")
    try:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired as err:
        raise ContractError(f"{target} --help did not exit within 60s") from err
    if proc.returncode != 0:
        raise ContractError(
            f"{target} --help exited {proc.returncode}: {(proc.stderr or proc.stdout)[:200]!r}"
        )
    flags = set(FLAG_RE.findall(proc.stdout))
    if not flags:
        raise ContractError(f"{target} --help produced no long flags: {proc.stdout[:200]!r}")
    return flags


def help_flags(binary, bin_dir, build):
    exe = os.path.join(bin_dir, binary)
    if not os.path.exists(exe):
        if not build:
            raise ContractError(
                f"{exe} not found; build it first "
                f"(cargo build --no-default-features --bin {binary})"
            )
        subprocess.run(
            ["cargo", "build", "--no-default-features", "--bin", binary],
            cwd=REPO_ROOT,
            check=True,
        )
    if not os.path.exists(exe):
        raise ContractError(f"{exe} still missing after build")
    try:
        proc = subprocess.run(
            [exe, "--help"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=60
        )
    except subprocess.TimeoutExpired as err:
        raise ContractError(f"{binary} --help did not exit within 60s") from err
    text = proc.stdout + proc.stderr
    flags = set(FLAG_RE.findall(text))
    if not flags:
        raise ContractError(
            f"{binary} --help produced no long flags (exit {proc.returncode}): {text[:200]!r}"
        )
    return flags


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scripts", nargs="*", help="extra shell scripts to check")
    parser.add_argument("--bin-dir", default=os.path.join(REPO_ROOT, "target", "debug"))
    parser.add_argument("--no-build", action="store_true", help="never invoke cargo build")
    args = parser.parse_args()

    bins = known_binaries()
    targets = dict(CONTRACTS)
    for extra in args.scripts:
        targets.setdefault(os.path.relpath(os.path.abspath(extra), WORKSPACE_ROOT), set())

    failures, total_flags = [], 0
    for rel, required in sorted(targets.items()):
        path = os.path.join(WORKSPACE_ROOT, rel)
        if not os.path.exists(path):
            failures.append(f"{rel}: script not found")
            continue
        try:
            usage = collect_usage(path, bins)
        except ContractError as err:
            failures.append(str(err))
            continue

        missing = required - set(usage)
        if missing:
            failures.append(
                f"{rel}: expected invocations of {sorted(missing)} but found none — "
                "extraction is stale or the script was restructured"
            )

        for target in sorted(usage):
            used = usage[target]
            total_flags += len(used)
            if not used:
                print(f"[ok] {rel} -> {target}: invoked with no long flags")
                continue
            try:
                if target.split(" ")[0].endswith(".py"):
                    known = python_help_flags(target)
                else:
                    known = help_flags(target, args.bin_dir, not args.no_build)
            except (ContractError, subprocess.CalledProcessError) as err:
                failures.append(f"{rel}: {target}: {err}")
                continue
            unknown = sorted(used - known)
            status = "FAIL" if unknown else "ok"
            print(f"[{status}] {rel} -> {target}: {len(used)} flags")
            for flag in sorted(used):
                mark = "!!" if flag in unknown else "  "
                print(f"    {mark} {flag}")
            for flag in unknown:
                failures.append(f"{rel}: {target} does not accept {flag}")

    if total_flags == 0:
        failures.append("no flags extracted from any contract script — extractor is broken")

    if failures:
        print("\nCLI CONTRACT BROKEN:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print("\nCLI contract OK")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ContractError as err:
        print(f"CLI CONTRACT CHECK FAILED: {err}", file=sys.stderr)
        sys.exit(2)
