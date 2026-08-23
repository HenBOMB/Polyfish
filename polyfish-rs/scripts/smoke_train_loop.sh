#!/usr/bin/env bash
# End-to-end smoke test of the training seam:
#   run_training_loop.sh -> self_play -> games_*.safetensors -> train.py -> model.safetensors
#
# Runs the real driver at toy settings inside an isolated copy under
# target/smoke, so it never touches the checked-out model, training_log.csv,
# checkpoints or archive. Env knobs: SMOKE_DIR, SMOKE_VENV, SMOKE_GAMES,
# SMOKE_MCTS, SMOKE_ACTORS, SMOKE_GUMBEL_K, SMOKE_LEAGUE, SMOKE_TIMEOUT,
# SMOKE_FORCE_FREEZE.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SMOKE_DIR="${SMOKE_DIR:-$REPO/target/smoke}"
# Kept outside SMOKE_DIR (and symlinked in) so re-staging does not throw away
# the release build, and so rust-cache still sees it under polyfish-rs/target.
SMOKE_CARGO_DIR="${SMOKE_CARGO_DIR:-$REPO/target/smoke-cargo}"
SMOKE_VENV="${SMOKE_VENV:-$REPO/.venv}"
GAMES="${SMOKE_GAMES:-2}"
MCTS="${SMOKE_MCTS:-4}"
ACTORS="${SMOKE_ACTORS:-2}"
GUMBEL_K="${SMOKE_GUMBEL_K:-2}"
LEAGUE="${SMOKE_LEAGUE:-1}"
TIMEOUT="${SMOKE_TIMEOUT:-5400}"
FORCE_FREEZE="${SMOKE_FORCE_FREEZE:-1}"

case "$SMOKE_DIR" in
    /*/*smoke*) ;;
    *) echo "smoke: refusing to wipe SMOKE_DIR=$SMOKE_DIR (want an absolute path naming 'smoke')" >&2
       exit 2 ;;
esac
if [ ! -x "$SMOKE_VENV/bin/python3" ]; then
    echo "smoke: no python venv at $SMOKE_VENV (run ./local_setup.sh or set SMOKE_VENV)" >&2
    exit 2
fi

echo "smoke: staging $REPO -> $SMOKE_DIR"
rm -rf "$SMOKE_DIR"
mkdir -p "$SMOKE_DIR" "$SMOKE_CARGO_DIR"
tar -C "$REPO" -cf - \
    --exclude=./target --exclude=./.venv --exclude=./.git \
    --exclude=./archive --exclude=./checkpoints --exclude=./replays \
    --exclude=./.run_bin --exclude='./games_*.safetensors' \
    --exclude=./model.safetensors --exclude=./optimizer_state.pt \
    --exclude=./training_log.csv --exclude=./ladder.json \
    --exclude=./moves_by_turn.json --exclude='./*.log' \
    --exclude='./.last_*' --exclude='./.anchor_*' --exclude=./.training.pid \
    . | tar -C "$SMOKE_DIR" -xf -
ln -s "$SMOKE_VENV" "$SMOKE_DIR/.venv"
ln -s "$SMOKE_CARGO_DIR" "$SMOKE_DIR/target"

# Pin the training tribe pool the loop would otherwise write for itself. The
# gauge's own pair is pinned to an Imperius mirror, and its training-pair audit
# row only fires when the two differ — an Imperius-free pool makes that
# deterministic instead of leaving it to the loop's per-iteration shuffle.
cat > "$SMOKE_DIR/config.json" <<JSON
{"gamemode": 2, "mctsIters": $MCTS, "cores": 2, "tribes": ["Bardur", "Kickoo"]}
JSON

# The release profile is lto = "fat" + codegen-units = 1; a smoke test only
# needs the binaries to run, not to be fast.
export CARGO_PROFILE_RELEASE_LTO=false
export CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16
export CARGO_PROFILE_RELEASE_DEBUG=false
export CARGO_PROFILE_RELEASE_OPT_LEVEL=1
# Keep the gauge match to a single seed pair; the default 32 is a real reading.
export GAUGE_GAMES="${GAUGE_GAMES:-1}"
# #35: at the production freeze bar (Wilson lower bound >= 0.80) no reading this
# smoke can afford could ever reach the anchor-freeze branch, and the audit block
# only fires every 5th gauge — so `ladder.py freeze` and `audit-opponents` had
# never executed anywhere, and their first run would have been mid-campaign in a
# fail-fatal loop. Force both here, one seed each. The lowered bar is recorded on
# the reading itself, so a forced freeze is never mistaken for an earned one.
if [ "$FORCE_FREEZE" = 1 ]; then
    export GAUGE_FREEZE_WR=0
    export GAUGE_LINK_GAMES="${GAUGE_LINK_GAMES:-1}"
    export GAUGE_AUDIT_EVERY=1
fi

echo "smoke: run_training_loop.sh -i 1 -g $GAMES -n $MCTS -a $ACTORS -k $GUMBEL_K -l $LEAGUE"
set +e
( cd "$SMOKE_DIR" && timeout "$TIMEOUT" bash ./run_training_loop.sh --no-server \
    -i 1 -g "$GAMES" -n "$MCTS" -a "$ACTORS" -k "$GUMBEL_K" -l "$LEAGUE" )
status=$?
set -e
if [ "$status" -ne 0 ]; then
    echo "smoke: run_training_loop.sh exited $status" >&2
    tail -n 40 "$SMOKE_DIR/session.log" 2>/dev/null >&2 || true
    exit "$status"
fi

fail() { echo "smoke: $1" >&2; exit 1; }

[ -f "$SMOKE_DIR/model.safetensors" ] || fail "train.py produced no model.safetensors"
compgen -G "$SMOKE_DIR/archive/games_*.safetensors" > /dev/null \
    || compgen -G "$SMOKE_DIR/games_*.safetensors" > /dev/null \
    || fail "self_play produced no games_*.safetensors"
[ "$(wc -l < "$SMOKE_DIR/training_log.csv")" -ge 2 ] || fail "training_log.csv has no data row"
# self_play and train.py hand their metrics to training_log.py through
# .last_*_metrics.json sidecars, which the parse now consumes (#37) -- so assert
# what their presence was standing in for: the numbers reached the canonical
# record. A sidecar surviving the run means a parse did not consume it, and a
# later iteration would log it again as its own.
"$SMOKE_VENV/bin/python3" - "$SMOKE_DIR/training_log.csv" <<'CSV_ASSERTS' \
    || fail "self_play/train.py metrics did not reach training_log.csv"
import csv
import sys

rows = list(csv.DictReader(open(sys.argv[1])))
problems = []
if not rows:
    problems.append("no data row")
else:
    row = rows[-1]
    for col in ("num_games", "avg_score", "loss", "policy_loss", "value_loss"):
        if not (row.get(col) or "").strip():
            problems.append(f"{col} is empty")
if problems:
    print("; ".join(problems), file=sys.stderr)
    sys.exit(1)
CSV_ASSERTS
for sidecar in .last_self_play_metrics.json .last_train_metrics.json; do
    if [ -e "$SMOKE_DIR/$sidecar" ]; then
        fail "$sidecar outlived the parse that read it"
    fi
done
if [ "$LEAGUE" -gt 0 ]; then
    [ -s "$SMOKE_DIR/ladder.json" ] || fail "the strength gauge recorded no ladder reading"
    # #34: the reading's metadata must describe the match arena actually played.
    # It used to be handed self-play's shuffled training pair for a match arena
    # hardcoded to an Imperius mirror, and nothing downstream could notice.
    PLAYED=$(sed -n 's/^Tribes: \(.*\)$/\1/p' "$SMOKE_DIR/session.log" | head -1)
    RECORDED=$("$SMOKE_VENV/bin/python3" -c 'import json,sys
print(json.load(open(sys.argv[1]))["readings"][0].get("tribes",""))' "$SMOKE_DIR/ladder.json")
    [ -n "$PLAYED" ] || fail "arena printed no tribe pair for the gauge match"
    [ "$PLAYED" = "$RECORDED" ] \
        || fail "ladder recorded tribes '$RECORDED' for a match arena played on '$PLAYED'"
fi

# The branch this smoke exists to reach at all (#35): a freeze snapshots an
# anchor, links it to the outgoing one, and the audit block cross-checks the
# chain. Nothing else in the repo executes any of it.
if [ "$LEAGUE" -gt 0 ] && [ "$FORCE_FREEZE" = 1 ]; then
    compgen -G "$SMOKE_DIR/checkpoints/anchor_iter*.safetensors" > /dev/null \
        || fail "the freeze branch snapshotted no anchor into checkpoints/"
    "$SMOKE_VENV/bin/python3" - "$SMOKE_DIR/ladder.json" <<'LADDER_ASSERTS' \
        || fail "the freeze/audit branch did not leave the rows the ladder needs"
import json
import sys

data = json.load(open(sys.argv[1]))
by_kind = {}
for r in data["readings"]:
    by_kind.setdefault(r["kind"], []).append(r)
problems = [f"no {k} reading" for k in ("gauge", "link", "audit", "tribe_audit")
            if k not in by_kind]

if len(data["anchors"]) < 2:
    problems.append("no anchor was frozen")
else:
    frozen = data["anchors"][-1]
    if not frozen.get("path", "").startswith("checkpoints/anchor_iter"):
        problems.append(f"frozen anchor path is {frozen.get('path')!r}")
    if frozen.get("frozen_iteration") != 1:
        problems.append(f"anchor frozen_iteration is {frozen.get('frozen_iteration')!r}, want 1")

gauge = (by_kind.get("gauge") or [{}])[0]
if gauge.get("freeze_wr") != 0.0:
    problems.append("the lowered freeze bar is not recorded on the reading it decided")

# Every row here was played against greedy, which the freeze in this same
# iteration has already retired -- so each must name the anchor its own match
# used, not whichever one is active by the time the row is written.
for kind in ("link", "audit", "tribe_audit"):
    for r in by_kind.get(kind, []):
        if r.get("opponent") != "greedy":
            problems.append(f"{kind} row names opponent {r.get('opponent')!r}, want greedy")

for r in by_kind.get("tribe_audit", []):
    if r.get("tribes") == gauge.get("tribes"):
        problems.append(f"tribe_audit was played on the pinned pair {gauge.get('tribes')!r}")

if problems:
    print("smoke: ladder freeze/audit branch: " + "; ".join(problems), file=sys.stderr)
    sys.exit(1)
LADDER_ASSERTS
fi

echo "smoke: OK (artifacts in $SMOKE_DIR)"
