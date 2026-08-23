# Pipeline Audit — Aug 2026

Twelve-dimension audit of the training pipeline, engine, and measurement layer,
run at commit `2bc3160` (branch `main`). Companion to `expert_review.md` (search
and learning-signal review) and `expert_boost_throughput.md` (throughput).
Rendered report: https://claude.ai/code/artifact/fdc8836d-b4ac-4ff0-bef3-642c3e88c588

**Headline (as written, at `2bc3160`):** the training loop cannot execute at
HEAD, and the strength gauge has never produced a reading. Three shell-to-binary
contract breaks account for both. Every conclusion in
`hypothesis_driven_improvements.md` that depends on a gauge reading — the plateau
verdict, EXP_ELO_002's "success bar not met" — was drawn from an instrument that
was returning parse failures.

## Status — Aug 18, 2026

Two repair waves have landed since the audit was written: commit `73dafb9`
(blockers + gauge + head widths) and the wave that follows it in the working
tree (engine correctness, adversarial search, self-play data, ops, dashboard).
`cargo test --no-default-features --lib --tests --bin self_play` is green and an
end-to-end `self_play → train.py → self_play` run completes.

**The three blockers are fixed and the gauge is repaired. No gauge reading has
been taken on the repaired instrument yet.** Every verdict in this file and in
`hypothesis_driven_improvements.md` that rests on a reading is therefore still
provisional, and step 5 of the order of operations (re-baseline) is the next
thing that should happen.

Per-item statuses below are updated in place; the original finding text is left
intact so the history of what was found is not erased. `**What landed:**` blocks
carry the file:line citations and the re-verify command as it stands today. The
original `# Verify` snippets are kept as written — for a FIXED item they now
report the opposite of what their trailing comment predicts.

| Item | Status | Item | Status |
|---|---|---|---|
| P1 blocker flags | FIXED | A3 optimizer reset | FIXED |
| P2 gauge swallows errors | FIXED | A4 D4 caveat | FIXED |
| P3 action-head width | FIXED (Resign still ungenerated) | A5 misc signal | PARTLY FIXED |
| M1 seed control | FIXED | R1 policy rank-1 bottleneck | OPEN (verified) |
| M2 gauge/self-play mismatch | FIXED | R2 player state | OPEN |
| M3 reading resolution | PARTLY FIXED | R3 product-of-marginals | OPEN (verified; dedup fixed) |
| M4 gauge game length | FIXED | R4 receptive field | REFINED — no action |
| M5 misc measurement | PARTLY FIXED | E1 metal GN keys | FIXED; compiles in CI, runtime unverified |
| A1 `DETACH_VALUE_TRUNK` | REGISTERED, arm not run | E2 engine correctness | VERIFIED; 6 of 7 fixed |
| A2 two reward conventions | FIXED | E3 hot-path allocation | OPEN (unchanged) |
| A2b label vs win condition | OPEN (measured, not acted on) | T1 forward parity | FIXED (found a live candle bug) |
| | | T2 CI coverage | FIXED |
| | | T3 misc testing/ops | MOSTLY FIXED |

## How to use this file

Each item has an ID, a status, and a **Verify** command that re-checks it in
seconds. Update `Status:` as items are fixed. Confidence tiers:

- **CONFIRMED** — verified by reading the cited lines at `2bc3160`.
- **FLAGGED** — from the audit sweep, survived an adversarial verification pass,
  but not independently reproduced. Treat citations as leads.

**Scope:** the **Polaris** tribe is out of scope — skip Polaris-specific
mechanics and any finding that only affects them. Items marked
"Resolved (owner)" carry a decision from the repo owner and should not be
re-litigated. Note this repo is a **fork**; upstream development happens under
`HenBOMB/Polyfish`, so the provenance of a setting is often unrecoverable from
here — prefer settling such questions by measurement rather than archaeology.

---

## Correction to the first draft of this audit

The initial write-up recommended enabling `AUGMENT_D4=1`, on the grounds that D4
augmentation is implemented but never switched on. **That recommendation was
wrong**, and the reason is instructive.

The current comment in `train.py` presents D4 as unconditionally valid. An
earlier revision (`git show 5ecdb5d~1:polyfish-rs/train.py`) carried a measured
caveat that was deleted when the comment was rewritten:

> Geometrically valid (no feature plane, player scalar, or rule is
> orientation-dependent) but OFF by default: enabling it MID-RUN on the
> 586K-param net collapsed play for ~8 iterations (run 1783556259 — policy lost
> its orientation-specific fit, degraded games then fed back through self-play).
> Opt in only for from-scratch runs, where the net never learns orientation
> shortcuts to begin with.

So the switch is off deliberately, for a reason that was measured and then lost.
The real finding is **the deletion**, not the setting — the current comment will
lead the next reader to flip it on mid-run and repeat run 1783556259. See A4.

---

## P — Blockers (pipeline does not run)

### P1 · `self_play` rejects `--decay-last-iter`; the loop exits on iteration 1
**Status:** FIXED (`73dafb9`) · **CONFIRMED** · Effort: hours

**What landed:** both arguments are back on `self_play` (`--decay-last-iter`
`src/bin/self_play.rs:1291`, `--anchor-decay-start` `:1300`) and both now drive a
real phase-out: `decay_crutch` (`:41`) takes the search prior blend and the
greedy-anchor game rate to zero instead of asymptoting at a floor, and the two
gates are combined tightest-wins at `:1716-1720`.

```bash
# Re-verify
grep -n 'decay_last_iter\|anchor_decay_start' polyfish-rs/src/bin/self_play.rs
# → hits at :1291, :1300, :1716-1720 = wired
```

`run_training_loop.sh:364` builds the flag unconditionally; `:425` passes it.
`self_play` parses with a single strict clap `Args::parse()` (`self_play.rs:1340`,
declared inside `main()` at `:1167`, no `ignore_errors`), so an unknown argument
is exit 2 — which `:429` propagates via `exit "$SP_STATUS"`.

`--anchor-decay-start` (`:421`) has the same problem; it is appended whenever
`ANCHOR_FLAG` is non-empty, i.e. whenever anchor games are on.

```bash
# Verify
grep -rn 'decay_last_iter\|decay-last-iter\|anchor_decay_start\|anchor-decay-start' polyfish-rs/src/bin/
# → no output = still broken
```

Both flags entered together in `46e9a15`. The squashed re-import `3893daf`
restored an older `self_play.rs` while keeping the newer script — the Rust half
is gone, the shell half survives. Fix by restoring the arguments or dropping
them from the script; decide which by whether the EXP_ELO_002 decay machinery is
still wanted.

### P2 · `arena` rejects `--dump-stats-dir`; every gauge reading is a swallowed error
**Status:** FIXED (`73dafb9`) · **CONFIRMED** · Effort: hours

**What landed:** `arena --dump-stats-dir` writes one JSON per game (seed, seat
assignment, tribes, winner, turns, and whether the game was dropped), and the
failure is no longer swallowed — `run_gauge_match` checks arena's exit status
(`run_training_loop.sh:549-552`), and a failed reading now aborts the run rather
than printing "failed to parse" and continuing (`:573-577`).

```bash
# Re-verify
grep -n 'dump_stats_dir' polyfish-rs/src/bin/arena.rs
grep -n 'arena_status' polyfish-rs/run_training_loop.sh
```

`run_gauge_match` is always called with a stats directory (`:547`), so
`DUMP_FLAG` (`:523`) is always set. `arena` has no `dump` argument at all.

```bash
# Verify
grep -n 'dump' polyfish-rs/src/bin/arena.rs      # → no output = still broken
```

The failure is not propagated: the loop tests whether the win count parsed
(`:547`) rather than checking the exit code, then prints
`GAUGE: arena reading failed to parse — skipping this reading` (`:615`) and
continues. Consequence chain: `ladder.py record` never runs → `ladder.json`
gains no readings → no plateau early-stop, no ≥80% anchor freeze, and
`.anchor_decay_start` is never written, which pins the anchor-decay exponent at 0
for the whole run.

Fix: restore the flag **and** make the gauge fail loudly on a non-zero exit.

### P3 · `action_type` targets are 12 wide; the head is 11
**Status:** FIXED (`73dafb9`) · **CONFIRMED** · Effort: hours

**What landed:** `NUM_ACTION_TYPES = 12` (`src/ai/network.rs:10`) is the single
source of truth — the candle layer is built from it (`:227`), `train.py` derives
its width from a named constant (`train.py:96`, layer at `:165`), a const
assertion keeps `MoveType::Resign` inside the head (`network.rs:20`,
`mapper.rs:97-102`), and `tests/parity_widths.rs` fails the build if the Rust and
Python widths ever disagree again. `migrate_model.py` / `train.py:391-406` pad
existing 11-row checkpoints.

**Still open — the follow-on, unchanged:** `ResignMove` is still never generated.
`generate_legal_moves` does not emit it; the only construction site remains the
web move-by-index path (`src/main.rs:574`). Slot 11 therefore still receives zero
self-play gradient. Widening the head was necessary, not sufficient.

```bash
# Re-verify the width
cd polyfish-rs && cargo test --no-default-features --test parity_widths
# Re-verify Resign is still unreachable from search
grep -rn 'ResignMove' polyfish-rs/src/ | grep -v 'struct\|impl Move\|use \|test'
# → only main.rs:574 and mapper.rs tests
```

`network.rs` contradicts itself in one file:

```rust
// network.rs:7   — consumed by the data writers (self_play.rs:768, :2269)
pub const NUM_ACTION_TYPES: usize = 12;
// network.rs:178 — the actual layer
let num_action_types = 11;
```

`train.py:145` also builds `nn.Linear(self.filters, 11)`. The 12th slot exists
because `mapper.rs:95` maps `MoveType::Resign → 11`, one past the head. Every
`games_*.safetensors` written by current `self_play` carries a target the loss
cannot broadcast against.

```bash
# Verify
grep -n 'NUM_ACTION_TYPES\|num_action_types = ' polyfish-rs/src/ai/network.rs
grep -n 'pi_action = nn.Linear' polyfish-rs/train.py
```

**Resolved (owner):** Resign stays — resigning a clearly lost game is legitimate.
So the fix is to **widen both heads to 12**, not to drop the mapping. Make
`network.rs:178` read `NUM_ACTION_TYPES` instead of a literal, mirror the width
in `train.py:145`, migrate checkpoints, and add a producer/consumer width
assertion so the two cannot drift again.

Follow-on that this exposes: `ResignMove` is **never generated as a legal move**.
It is a bare struct at `moves/mod.rs:139` and is only ever constructed by hand at
`main.rs:575` (the web/API move-by-index path). `generate_legal_moves` never
emits it, so slot 11 would receive zero self-play gradient and the net could
never learn to use it. Wiring the width is therefore necessary but not
sufficient — resignation has to become a generated move (or be injected by a
value threshold) before the head slot means anything.

```bash
# Verify Resign is unreachable from search
grep -rn 'ResignMove' polyfish-rs/src/ | grep -v 'struct\|impl Move\|use '
# → only main.rs:575
```

---

## M — Measurement (readings are not comparable)

### M1 · No seed control anywhere; every reading uses different maps
**Status:** FIXED (`73dafb9`) · **CONFIRMED** · Effort: hours

**What landed:** `arena --seed` pins the evaluation map set (`src/bin/arena.rs:99-102`,
consumed at `:536-545`) and composes with the existing side-swap, so seed N is
played by both seats. The gauge passes a fixed default
(`run_training_loop.sh:539`, `GAUGE_SEED:-20260811`), making readings paired
across iterations.

```bash
# Re-verify
grep -n 'seed' polyfish-rs/src/bin/arena.rs | head
grep -n 'GAUGE_SEED' polyfish-rs/run_training_loop.sh
```

`arena.rs:348` derives `base_seed` from `SystemTime::now()` and exposes no
`--seed`. Side-swapping is already implemented (`:465–467`), so seat bias is
handled — but map difficulty is re-rolled every reading. The gauge series
`31.2 → 37.5 → 23.4 → 35.9 → 40.6 → 33.6%` carries full map variance on top of
the model delta, and that is what the plateau detector reads.

Fix: add `--seed`, pin a fixed evaluation map set for the ladder. Converts
between-reading comparisons into paired ones for zero extra compute. This is
also a prerequisite for reproducing any past experiment.

### M2 · The gauge grades a different agent than self-play trains
**Status:** FIXED (`73dafb9` + this wave) · **CONFIRMED** · Effort: hours

**What landed (two halves).** `arena` now threads all three search knobs and
defaults them to the self-play values (`src/bin/arena.rs:111-119`, `:515-519`),
so the ladder grades the agent training produces. The second half was found while
verifying E2: `arena` was calling `select_move` on the REAL, un-obscured game
while self-play searches a fog-obscured clone, so the graded agent could see
through fog the trained agent never gets. `arena` now searches
`game.clone_for_mcts(current_pid)` like self-play does (`arena.rs:306-313`).

**A gap opened by the same wave, and closed in it.** `self_play --symmetric`
defaults to true (`src/bin/self_play.rs:1342`) while `arena --symmetric`
defaulted to false, and `run_gauge_match` did not pass it — training would have
run on point-symmetric maps while the gauge read asymmetric ones. `arena` now
defaults `--symmetric` to true with the same `ArgAction::Set` shape self_play
uses (`arena.rs:90-94`), and `run_gauge_match` passes it explicitly as
`--symmetric "${GAUGE_SYMMETRIC:-true}"` so the gauge's map family is visible at
the call site rather than inherited from a default.

```bash
# Re-verify the knobs
grep -n 'prior_heuristic\|policy_target_q\|tree_q\|clone_for_mcts' polyfish-rs/src/bin/arena.rs
# Re-verify the symmetry gap
grep -n 'symmetric' polyfish-rs/src/bin/arena.rs polyfish-rs/src/bin/self_play.rs polyfish-rs/run_training_loop.sh
```

`self_play.rs:644–650` configures three search knobs; `arena.rs:173` passes
`None` for all four parameters:

```rust
// self_play.rs
Brain::with_backend(eval1, mcts_iters, backend1)
    .with_prior_heuristic_weight(prior_w)      // permanent 0.1 floor
    .with_policy_target_q_weight(q_target_w)
    .with_tree_q_weight(q_target_w)
// arena.rs:173
make_search_agent(backend1, eval1, mcts1, None, None, None, None)
```

The heuristic prior blend (`HEURISTIC_PRIOR_W_FLOOR = 0.1`, `self_play.rs:34`) is
present in training and absent at evaluation. Beyond the measurement mismatch
this is a plausible strength ceiling: the net is trained toward targets produced
by a net+heuristic blend, so it never has to learn the 10% the heuristic supplies.

### M3 · 64 games resolves ~±12pp; verdicts are drawn from 1–6pp
**Status:** PARTLY FIXED (`73dafb9`) · **FLAGGED** · Effort: days

**What landed:** M1 removed the map-variance component for free, and every
reading now stores its Wilson interval (`ladder.py:194`, `:210`) plus the search
budget it was taken at (`:214-218`). Both gates are interval-aware: an anchor
freeze requires the interval's LOWER bound to clear 80% (`:275-281`), and the
plateau test compares pooled halves by interval overlap rather than by mean
(`:106-116`). The dashboard draws the band (`src/public/training.html`, elo
chart).

**Superseded (#31):** the interval-overlap plateau test described above was
itself the defect — it made failure-to-*prove*-improvement count as a plateau,
which at this budget is every climb below ~12pp, including the +8pp EXP_ELO_002
was registered against. It has been replaced by EXP 11's registered rule (halves
flat-or-down AND slope ≤ 0). The freeze bar is unaffected; a lower bound is the
right test for "beats the anchor 4:1", it was only wrong as a test for "is still
improving".

**Also landed:** the budget is now sized rather than assumed. `ladder.py`
computes the games a target effect needs (`required_games`, exposed as
`python3 ladder.py power --baseline 0.33 --games 64`), stores each reading's own
resolution as `resolves_pp`, and flags a reading that cannot adjudicate the
effect the registered bars are written against. The loop echoes that flag at the
moment the reading is taken (`run_training_loop.sh`, gauge block), so the caveat
travels with the number instead of being rediscovered from the interval later.
`tests/test_ladder.py` pins the statistics, including that a lucky 17-of-20 does
not clear the freeze bar.

The number that calculation returns is the finding: **detecting +8pp at a ≈33%
baseline, 80% power, α=0.05, needs ~571 games per reading.** The gauge spends 64.
It is ~9× too small for the bar EXP_ELO_002 registered against it — not
marginally underpowered, an order of magnitude. The observed +1pp and the
30.7→36.7% within-run drift were never separable from noise by a single reading.

**Still open:** actually spending them. Three ways out, in cost order:

1. **Trend, not readings.** The plateau gate already pools eight readings (~512
   games), which is why it is meaningful at this budget. Any verdict drawn from
   *one* reading is not.
2. **A paired analysis.** M1 means both sides now play the same seeded map set,
   so the per-seed outcomes are correlated and a paired test needs materially
   fewer games than the unpaired ~571. Nothing measures that correlation yet;
   `arena --dump-stats-dir` already writes per-game records, so it is
   computable without new match compute.
3. **Raise `GAUGE_GAMES`.** Honest and linear in gauge wall-clock. Decide it
   against the effect size, not against how long it feels acceptable to wait.

### M4 · Gauge plays a shorter game than training generates
**Status:** FIXED (`73dafb9`) · **CONFIRMED** · Effort: hours

**What landed:** the gauge asks `self_play` for the curriculum rather than
mirroring its thresholds — `self_play --print-curriculum --iteration N` emits the
stage, and the loop passes its `max_turns` to arena
(`run_training_loop.sh:524`, `:539`).

```bash
# Re-verify
grep -n 'print-curriculum\|GAUGE_MAX_TURNS' polyfish-rs/run_training_loop.sh
```

`curriculum()` (`self_play.rs:201–211`) runs `max_turns = 45` past iteration 30.
The gauge never passes `--max-turns`, so arena uses its default of 30
(`arena.rs:57`). Late-game strength is outside the measured window.

### M5 · Other measurement gaps
**Status:** PARTLY FIXED · **FLAGGED**

FIXED: `elo.py` now fits ratings from `ladder.json`'s own readings against its
Elo-0 greedy floor (`elo.py:10-11`, `:84-98`), so it grades the same matches the
ladder records. `value_r2` gained a holdout split **by game file**, not by
position (`train.py:475-494`), reported next to the in-sample number so
underfitting and overfitting can be told apart; the dashboard plots both. `arena`
records every game it drops in its per-game JSON and the loop no longer ignores
its exit code (P2).

ALSO FIXED: the record now says what it was taken under.

- **The holdout number reached nothing.** `train.py` has emitted
  `value_r2_holdout` since the split landed and `training.html` has had the chart
  code to draw it, but `training_log.py`'s `HEADER` had no such column, so the
  trainer computed the one diagnostic the plateau question turns on and the CSV
  dropped it on the floor. `value_r2_insample`, `value_r2_holdout`,
  `holdout_samples` and `ownership_loss` are columns now, and the endpoint serves
  them (verified against a running server, not by inspection).
- **The holdout is self-play only (#36).** The split ran over the combined file
  list, teachers included, and membership is a deliberately stable function of
  the basename — so a teacher file that hashed into the bucket was withheld from
  fitting *permanently* (teachers never rotate out of the buffer the way
  self-play files do at ~`REPLAY_BUFFER_FILES` iterations), and its static
  known-good positions contaminated `value_r2_holdout`, the one series that is
  supposed to say how the net generalizes on fresh self-play. `partition_buffer`
  (`train.py`) now splits fresh + archive only and appends the teachers to the
  training side; an out-of-sample teacher number, if ever wanted, is its own
  series. The small-buffer case went with it: the guard against an empty
  training set now sees the self-play buffer alone, so an iteration can no
  longer end up fitting teachers only while withholding every self-play file it
  had.
- **Per-iteration configuration is recorded.** `config.json` is still re-read
  inside the loop — that is the dashboard's live-control feature, not an accident
  — but every row now carries the settings it actually ran at: `tribe1`,
  `tribe2`, `cfg_mcts_iters`, `cfg_gumbel_k`, `cfg_num_games`, `cfg_gamemode`,
  `cfg_anchor_frac`, `cfg_value_trust`, `cfg_detach_value_trunk`. A mid-flight
  edit is now visible in the record instead of invisible, and the tribe-pair
  block effect can finally be conditioned on rather than merely deplored.
- **The plateau detector no longer mixes search budgets.** `_gauge_series`
  restricts the window to the budget the latest reading used, so a 16-sim stint
  cannot be chained with 64-sim readings as if it measured the weights. Ladders
  whose readings predate the `budget` field keep the old pool-everything
  behaviour rather than silently emptying the window. *(#31 extended this: the
  window is also scoped to the current `run_id`, `max_turns` joined the budget
  key since the loop varies it with the curriculum, and strikes reset on a run
  change instead of carrying into the next campaign.)*
- **A dropped game no longer hides.** `arena` reports `Unpaired Seeds:` — seeds
  that lost one half of their side swap, which is the pairing the seeded design
  buys — and the loop carries `games_attempted`, `games_dropped` and
  `unpaired_seeds` into the reading, surfaced in the verdict.
- **The tribe pair on a reading is the pair the match was played on (#34).** The
  fix above landed correctly in the CSV and incorrectly in the ladder: the loop
  passed self-play's shuffled *training* pair to `ladder.py record` for a match
  `arena` had hardcoded to an Imperius mirror, so the permanent record carried
  metadata about a variable the gauge never varied, and disagreed with its own
  `--dump-stats-dir` JSONs. `arena` now takes `--tribe1/--tribe2` (rejecting an
  unknown name rather than defaulting it) and prints the pair it played; the loop
  reads that line back and records *that*, failing the reading if it does not
  parse. The pin itself stays — the block effect is exactly why the gauge should
  not vary it — but the scope limit is now written into `ladder.json`'s `scope`
  field instead of being tacit, and a `tribe_audit` row at the audit cadence
  re-reads the anchor on the training pair as a cross-check (kept out of both
  `_gauge_series` and the `elo.py` fit). `scripts/smoke_train_loop.sh` now fails
  if a recorded pair disagrees with the one `arena` printed.
- **The freeze/audit branch now runs somewhere other than a live campaign
  (#35).** `ladder.py freeze` and `audit-opponents` are invoked from
  `run_training_loop.sh` and nowhere else, and nothing could reach them: the
  smoke's 2-game reading cannot clear the 0.80 Wilson bar, and the audit block
  additionally needs `i % (LEAGUE_INTERVAL * 5) == 0`. In a loop that now aborts
  on a failed reading, the first execution of that shell↔argparse contract would
  have been the first good reading of the re-baseline campaign. The smoke forces
  both branches (`GAUGE_FREEZE_WR`, `GAUGE_LINK_GAMES`, `GAUGE_AUDIT_EVERY`) and
  asserts the anchor snapshot, the link reading and the audit rows appear;
  `tests/test_ladder.py` runs the same command lines as subprocesses, with the
  flags extracted from the loop script itself; and the CLI contract check now
  covers python CLIs per subcommand, not just Rust binaries. A moved freeze bar
  is recorded on the reading it decided, so a forced freeze can never pass for
  an earned one. The interaction the branch surfaced is fixed too: an audit
  cadence landing on a freeze iteration plays its cross-checks against the
  outgoing anchor, which by then is no longer `anchors[-1]`, so the loop names
  that anchor explicitly on the `tribe_audit` row.

STILL OPEN: the search-budget confound is contained, not resolved — restricting
the window is correct but it means a budget change silently shortens the plateau
series rather than flagging it. And no reading has yet been taken with any of
this in place.

- `elo.py` is orphaned and anchored to a player that never plays; the ratings
  actually used are un-intervalled chained win rates.
- `value_r2` is computed in-sample on the buffer the net just fit — there is no
  holdout split anywhere, so underfitting vs overfitting cannot be distinguished.
  This is the question the whole plateau turns on.
- `arena` silently drops panicked games and the loop ignores its exit code, so a
  reading's `n` and its pairing can differ from what is recorded.
- Nothing records per-run configuration; `config.json` is re-read *inside* the
  iteration loop (`:379`), so dashboard edits change a run mid-flight.
- Per-iteration behaviour metrics are confounded by an unlogged tribe pair that
  is reshuffled every iteration (block effect ~2.5 turns on t2c, comparable to
  the entire campaign's measured improvement).
- The plateau detector mixes search budgets, so ladder Elo is a function of
  (weights × sims) but is chained as if it measured weights alone.

---

## A — Learning signal

### A1 · `DETACH_VALUE_TRUNK=1` is exported in the production loop
**Status:** REGISTERED, arm not yet run · **CONFIRMED** · Effort: hours

**What landed:** the switch is registered as `EXP_TRUNK_001` in
`hypothesis_driven_improvements.md` — hypothesis, the A2b sequencing below,
expected results and a falsifier — so it is an open variable on the record rather
than an unexplained export. `run_training_loop.sh:23` now reads
`export DETACH_VALUE_TRUNK="${DETACH_VALUE_TRUNK:-1}"`, so the off arm runs
without editing the driver (`DETACH_VALUE_TRUNK=0 ./run_training_loop.sh`).
Production behaviour is unchanged: the default is still 1.

**What remains:** running the arm. That is the verdict, and it is sequenced after
A2b — see below. Its other prerequisites (P1/P2/M1/M2/M4) are now closed.

`run_training_loop.sh:17`. `train.py:35–39` documents it as "bisect Arm D", and
`bisect_arm.sh:14` treats it as a diagnostic. With it on, no value-loss gradient
reaches `conv1`, the ResBlocks, or the cross-attention — the trunk is shaped only
by the four policy heads plus a 0.15-weight ownership aux, and the value head is
a linear probe on features selected for something else.

Nuance: the export predates the `3893daf` re-import (present at `5ecdb5d~1` too),
so it is a long-standing setting rather than a fresh accident. But there is **no
recorded verdict for it** in `hypothesis_driven_improvements.md`.

**Resolved (owner):** provenance is unrecoverable — this repo is a fork and the
switch was set upstream. So stop trying to establish intent and settle it
empirically instead: run the arm both ways once the gauge works (P1/P2/M1) and
record the result as the missing verdict. Until then treat it as an open
variable, not as a known-good setting.

```bash
# Verify
grep -n 'DETACH_VALUE_TRUNK' polyfish-rs/run_training_loop.sh
grep -n -i 'detach' hypothesis_driven_improvements.md   # → no verdict recorded
```

#### What the switch actually does

The net is a shared trunk with two heads. The trunk reads the board; the policy
head picks moves from it; the value head answers "am I winning" from it. Normally
both heads push gradient back into the trunk, so the trunk learns to represent
what both need. `DETACH_VALUE_TRUNK=1` cuts the value head's gradient path: the
head still trains, but only on whatever representation the trunk built for
move-picking. It cannot cause the trunk to represent winning-ness at all.

This matters because MCTS queries the value head at every leaf of every search.
A value head that is a passenger on policy features makes the search weak no
matter how good the policy is.

The defensible reason to have set it: two heads sharing a trunk can genuinely
fight, and a noisy value signal can degrade the policy. Detaching isolates that.

#### Recommendation — default it off, but only after A2b

Remove `export DETACH_VALUE_TRUNK=1` from `run_training_loop.sh` and let the
value gradient reach the trunk. **Sequence it after the A2b label fix, not
before.**

The reasoning is that A2b changes the prior on why the switch exists. Detaching
only pays if value gradient was actively harming the policy — and A2b now shows
the value label is built on a quantity that is ~8pp worse than an available
alternative at every turn, and that actively degrades in the late game. A
plausible history is: the value signal genuinely was harmful, someone correctly
observed the policy suffering, and detaching treated the symptom rather than the
cause. If that is what happened, removing the detach *before* fixing the label
would reproduce the original harm and look like a failed experiment.

Fix the label first, then the value gradient is far more likely to help than
hurt, and the arm becomes worth running:

1. Land A2b (reweight the label toward army value).
2. Land P1/P2/M1 so the gauge produces comparable readings.
3. Run the arm both ways at equal budget on a fixed seed set.
4. Record the result in `hypothesis_driven_improvements.md` as the verdict that
   was never written.

Do not remove the export as a standalone change before those steps — with the
current label there is a real chance it measures worse and gets wrongly
re-litigated as "value gradient hurts the trunk".

### A2 · Two reward definitions disagree about zero-sum
**Status:** FIXED (`73dafb9`) · **CONFIRMED** · Effort: hours

**What landed:** one constant, one convention. `reward::REL_W = 1.0`
(`src/ai/reward.rs:26`) is read by both the TD body (`:47`) and the final-outcome
tail (`src/bin/self_play.rs:560`); `FINAL_OUTCOME_REL_W` is gone. The label is
antisymmetric again, which is what the negamax backup in `mcts_common.rs`
requires. `GOOD_BOT_FINAL_SCORE` (`self_play.rs:49`) is live again as the
absolute yardstick, reachable only if `REL_W` is lowered.

This changed ~70% of the value label's weighting and is pre-registered as
**EXP_LABEL_001** in `hypothesis_driven_improvements.md`. It has NOT been
measured — see the status note at the top of this file.

```bash
# Re-verify there is only one constant
grep -rn 'REL_W' polyfish-rs/src/ai/reward.rs polyfish-rs/src/bin/self_play.rs
```

```rust
// self_play.rs:47 — "an absolute own-progress component is NOT antisymmetric"
const FINAL_OUTCOME_REL_W: f32 = 1.0;   // pure relative
// ai/reward.rs:19 — "Abs-dominant: … rewards it regardless of the opponent"
pub const REL_W: f32 = 0.4;             // 60% absolute
```

`reward::REL_W` feeds the TD(λ) body of the label, which carries `TD_W = 0.7`.
So 70% of the label is 60% non-antisymmetric while the search negates across
every turn boundary (`mcts_common.rs`). Both comments are internally reasoned and
mutually exclusive. Pick one convention, make it one constant.

Also: `GOOD_BOT_FINAL_SCORE` (`self_play.rs:37`) is dead while `REL_W` is 1.0.

### A2b · The value label is built from `score`, but training plays Domination
**Status:** OPEN — measured, not acted on · **CONFIRMED** · Effort: days · *Raised by the owner*

**Aug 18:** the recommended reweight has NOT landed. `src/ai/reward.rs` still
reads only `t.score` (`score_snapshot`, `:54`); there is no army-value term
anywhere in it. A2's
constant was settled ahead of A2b (see A2), which the note at the end of this
item advises against — so if the re-baseline reads worse, this is the first
confound to check.

This is the deeper version of A2, and it may be the single best explanation for
why the net cannot beat the teacher it was distilled from.

Every TD label term routes through the raw scoreboard:

```rust
// ai/reward.rs:46 — the only quantity the reward ever reads
let my = state.tribes.get(&player).map(|t| t.score).unwrap_or(0);
```

But training runs **Domination** (`run_training_loop.sh:377,389` default
`GAMEMODE=2`; `ModeType::Domination = 2` at `types.rs:25`), where the win
condition is elimination — `self_play.rs:939` resolves the winner as the sole
surviving tribe, falling back to score only on timeout. In Domination the
scoreboard is not the objective; it is a loosely correlated side-channel.

The codebase already knows this. The heuristic scorer branches on exactly this
distinction, using the owner's own example:

```rust
// ai/scoring.rs:713-730
CityRewardType::Park => {
    if is_perfection {
        base + 20.0   // "Always choose Park in Perfection — +250 score is massive"
    } else {
        base + 5.0    // "In Domination, Park is +1 SPT but no tactical advantage"
    }
}
CityRewardType::SuperUnit => {
    if is_perfection { base + 8.0 }
    else { base + 18.0 }   // "In Domination, super unit is game-changing"
}
```

So at a level-5 city in Domination the greedy teacher correctly prefers the
Giant, while the TD value label — reading `t.score` — credits the Park's +250 and
teaches the net that the teacher's move was the worse one. The heuristic is mode-
aware; the learning signal is not. Anywhere score and winning diverge (parks,
monuments, tech tier bought for points, score-dense but tactically idle play) the
label actively pulls against the teacher it is meant to distil.

That also predicts the specific failure EXP_ELO_001 recorded — over-investment in
research (17.3 techs vs Greedy's 12.1 by turn 24) — since tech tier pays score
directly.

#### Measured (option c, run Aug 2026)

`src/bin/score_predictiveness.rs` plays greedy-vs-greedy Domination games and
asks, at each turn, how often the leader on a given quantity goes on to win.
Games decided by the turn cap are excluded — score trivially predicts a winner
it defined.

```bash
cargo build --release --no-default-features --bin score_predictiveness
./target/release/score_predictiveness --games 400 --max-turns 45
```

400 games, 235 decided by elimination, 165 (41%) by turn cap:

| turn | score | cities | units | n |
|-----:|------:|-------:|------:|---:|
|  6 | 0.592 | 0.534 | **0.658** | 234 |
|  9 | 0.693 | 0.612 | **0.741** | 228 |
| 12 | 0.766 | 0.745 | **0.846** | 218 |
| 15 | 0.851 | 0.824 | **0.934** | 188 |
| 18 | 0.875 | 0.885 | **0.938** | 152 |
| 21 | 0.870 | 0.908 | **0.978** | 92 |
| 24 | 0.846 | 0.942 | **0.990** | 52 |

**This refines the claim above rather than confirming it.** Score is *not*
uninformative — 0.85 by turn 15 is far from a coin flip, so the hypothesis as
originally written ("the label optimizes a proxy that isn't the win condition")
was too strong. Three real results stand:

1. **Unit count beats score at every turn measured**, by ~8pp through the
   decision-relevant window (turn 12: 0.846 vs 0.766; turn 15: 0.934 vs 0.851;
   turn 21: 0.978 vs 0.870). At n≈190–220 the standard error is ≈0.026, so an
   8pp gap is ~3 SE — and it is consistent across every row, which is stronger
   evidence than any single row.
2. **Score degrades late while the others sharpen.** Score peaks around turn 18
   (0.875) and then *falls* — 0.870, 0.846, 0.812 — while cities and units climb
   monotonically toward 1.0. As a Domination game approaches its decision, the
   quantity the label is built from gets worse at predicting who wins. That is
   the park-versus-giant effect showing up in aggregate.
3. **41% of games never reach the win condition.** For those the outcome label
   is a score comparison, so score predicts it by construction — the label is
   circular in nearly half the training corpus.

Caveats: greedy-vs-greedy, not NN self-play, so the state distribution differs
from real training games. Unit *count* is a crude stand-in for army value
(ignores type, HP, veteran status) — a proper army-value metric would likely do
better still. And this measures aggregate predictiveness, not per-decision
correctness; it does not by itself prove the park/giant case, it shows the
signal quality the label inherits.

#### Recommended next step

Reweight rather than replace. Keep a score term, add an army-value term, and
weight toward the latter in Domination — the data says that strictly dominates
raw score at every point in the game. Then re-measure with this same tool using
a proper army-value function instead of unit count, and re-run once the gauge
works (P1/P2/M1) to confirm the label change moves strength, not just the proxy.

Note this cuts against A2's framing: making the label zero-sum in `score` does
not help if `score` is the wrong quantity. Resolve A2b before spending effort on
A2's constant.

### A3 · Optimizer and LR schedule reset every iteration
**Status:** FIXED (`73dafb9`) · **CONFIRMED** · Effort: days

**What landed:** Adam moments and the scheduler step are persisted across
invocations in `optimizer_state.pt` (`train.py:412`, load `:423-457`, save
`:459-466`, wired at `:651` and `:943`), keyed by `run_id` so a new run starts
clean. The cosine schedule now spans the run instead of restarting at top LR on
every call.

```bash
# Re-verify
grep -n 'OPTIMIZER_STATE_PATH\|load_optimizer_state\|save_optimizer_state' polyfish-rs/train.py
```

`train.py:426` constructs a fresh `Adam` per invocation and `:429` a
`CosineAnnealingWarmRestarts` that restarts at the top LR on every call — a
sawtooth, not a schedule, and Adam's moments are discarded each time.
`expert_review.md` listed "persistent optimizer" as a cleanup; not landed.

### A4 · The measured rationale for `AUGMENT_D4` was deleted from its comment
**Status:** FIXED (`73dafb9`) · **CONFIRMED** · Effort: minutes

**What landed:** the measured caveat is back at `train.py:57-64`, naming run
1783556259 and restricting D4 to from-scratch runs.

```bash
# Re-verify
sed -n '56,65p' polyfish-rs/train.py   # → mentions run 1783556259
```

See the correction section above. The current comment (`train.py:46–50`) reads as
an unconditional endorsement; the measured mid-run collapse (run 1783556259) is
gone. Restore the caveat. D4 remains a legitimate multiplier **for from-scratch
runs only**.

### A5 · Other learning-signal items
**Status:** PARTLY FIXED · **FLAGGED**

FIXED: move-selection temperature is back, as opening-move sampling —
`self_play --opening-temp-moves` (default 8 plies, `src/bin/self_play.rs:1349`)
plays a draw from the search's improved policy π′ instead of its argmax
(`sample_opening_move`, `:580`; applied at `:866-873`), while the policy target
stays π′. Registered as **EXP_DATA_002**, unmeasured. The training glob no longer
swallows `games_human_*` / `games_pro_*` (`train.py:575`, `:589`), and
`GameRecorder` now refuses to write steps that have no outcome at all rather than
labelling them `0.0` (`src/recorder.rs:79-108`).

FIXED: the `move_option` target, on both sides. The consumer normalizes each
policy target row by its own sum (`train.py:264-270`), and the producer now
divides `p_option` by `total_visits` alongside the other three heads — the
`// ... (others)` placeholder is gone (`src/bin/self_play.rs:842-857`). This
matters for consumers that do not renormalize, notably `src/bin/train.rs:168`,
whose option-head loss was scaled by N. The consumer-side normalize is
idempotent (it clamps the divisor at 1.0), so it is safe on both old and new
data files.

STILL OPEN, unchanged: no playout-cap randomization, no resignation, equal
per-step weighting; the `progress` head trains on a per-game-constant label at
full weight while two of three inference backends stub it to 0; weights still
round-trip through f16 every iteration (`train.py:942`).

- Move-selection temperature is disabled (`TEMPERATURE_MOVE_THRESHOLD = 0`), so
  there is no opening diversity within an iteration.
- No playout-cap randomization, no resignation, equal per-step weighting — search
  is spent evenly on decided and undecided positions.
- The `move_option` policy target is never normalized (a literal
  `// ... (others)` placeholder) — benign for Gumbel/Greedy, N×-scaled otherwise.
- The training glob still swallows `games_human_*` (whose win label is hardcoded
  `0.0` at `recorder.rs:53`) and `games_pro_*`. `expert_review.md` flagged this;
  it was fixed only in the Rust trainer, which is not the one in use.
- The `progress` head trains on a per-game-constant label at full weight and is
  added into the search's Q, while two of three inference backends stub it to 0.
- Weights round-trip through f16 every iteration.

---

## R — Representation and architecture

### R1 · Rank-1 bottleneck on `pi_action` and `pi_option`
**Status:** OPEN — independently verified, no code change · **CONFIRMED** · Effort: days

**Aug 18 verification, with a scope correction.** The bottleneck is real and
still present: `p_pool_conv` is `Conv2d(64 → 1, k=1)` (`src/ai/network.rs:223`,
`train.py:163`) feeding `p_fc_shared` `Linear(121 → 64)` (`network.rs:224`,
`train.py:164`), which feeds `pi_action` (`network.rs:227`) and `pi_option`
(`:230`) only. Correction to the text below: `pi_source` / `pi_target` are
per-tile `Conv2d(64 → 1)` and read all 64 channels, so they are NOT
channel-bottlenecked — the pathology is confined to the two heads that choose
*what to do*. The value head was already fixed by EXP_ARCH_001 (mean+max pool
over all 64 channels, `network.rs:234`, forward at `:316-319`).
Unchanged recommendation, still blocked on a working gauge.

```python
# train.py:142-148, mirrored at network.rs:205-216
self.p_pool_conv = nn.Conv2d(self.filters, 1, 1)     # 64ch → 1ch, no norm/act
self.p_fc_shared = nn.Linear(map_h * map_w, self.filters)
self.pi_action   = nn.Linear(self.filters, 11)
self.pi_option   = nn.Linear(self.filters, 192)
```

The heads that choose *what to do* — action type, and which unit/tech/structure
across 192 slots — read a single scalar per tile. This is exactly the pathology
EXP_ARCH_001 diagnosed and fixed for the value head ("collapsed the trunk to ONE
channel … a near-linear probe that cannot represent 'am I winning'"); the fix was
never applied to the policy side. EXP_ELO_001 named research over-investment and
army composition as the behavioural bottlenecks — these are those heads.

Fix: reuse the shape that already worked — global mean+max pool over the full
64-channel trunk → 2-layer MLP → `pi_action` / `pi_option`. Leave the spatial
heads as 1×1 convs. Mirror in `network.rs` in the same commit.

### R2 · The player-state vector has no opponent information and no tech identity
**Status:** OPEN · **FLAGGED** · Effort: weeks (checkpoint migration)

The value target is relative/zero-sum but the 16-dim player state describes only
the agent's own side. Tech is reportedly a count, not a set, so the net cannot
represent "I have Riding, so Roads is next". Adding opponent scalars and a tech
bitmask changes `PLAYER_STATE_DIM` — a coordinated `features.rs` + `network.rs` +
`train.py` change plus migration. Schedule deliberately.

### R3 · Product-of-marginals policy
**Status:** OPEN — now CONFIRMED; the duplicate-generation half is FIXED · **FLAGGED** · Effort: days

**Aug 18 verification.** `src/ai/policy_composer.rs` composes a pure product of
four independently-softmaxed marginals, in probability space
(`compute_move_priors_from_probs`, `:69`, used by `mcts_zero`) and in log space
(`compute_move_log_probs_from_logs`, `:206`, used by the Gumbel backend training
actually runs). Move types consume 1 to 4 factors — `EndTurn` 1;
`Capture`/`Harvest`/`Research` 2; `Step`/`Attack`/`Build`/`Summon`/`Reward` and
single-spatial abilities 3; two-spatial abilities (convert, diplomacy) 4. With
uniform heads the resulting priors span 9.1e-2 down to 3.2e-8: a ~2.8-million-fold
arity skew that renormalizing over the legal set does not remove.

FIXED — the "unit-ability moves are generated twice" sub-claim was true and is
gone. `generate_legal_moves` called `generate_unit_action_moves` on top of the
per-unit emission inside `generate_unit_moves`; the redundant call and the
now-unused function are deleted (`src/moves/mod.rs:210-214`, per-unit emission at
`:242`). Two identical children at one policy coordinate can no longer happen
that way.

Search forms `P(move) ∝ P(action)·P(source)·P(target)·P(option)`; training fits
each marginal independently. Move types using fewer heads are multiplied by fewer
sub-1 factors — a structural prior bias unrelated to anything learned. Two
independent remedies: a "not applicable" slot per head so every move consumes the
same factor count, and conditioning source/target/option on action type.
Related: unit-ability moves are reportedly generated twice, putting two identical
children at one policy coordinate.

### R4 · Receptive field and cross-attention
**Status:** REFINED — measured, no action recommended · **FLAGGED**

Audit reports an effective receptive field of ~±3 tiles with no global spatial
mixing, and that cross-attention is the terminal layer — no feed-forward
sublayer, no post-injection nonlinearity. Worth checking against the 6-block
trunk before acting.

**Aug 18 measurement (three parts, two of them corrections).**

1. *Receptive field* — "±3 tiles" is not a support limit. The trunk is 13 conv
   layers, all 3×3 / stride 1 / pad 1 / dilation 1 (stem + 6 ResBlocks × 2), so
   the theoretical radius is 13, larger than the 11×11 map's diameter of 10. It
   is a fair description of the EFFECTIVE field: measuring d(logit)/d(input) on
   the real PyTorch net over 5 seeds gives a per-axis gradient-weighted std of
   2.47 tiles for `pi_source`/`pi_target`, with 68.5% of sensitivity mass inside
   Chebyshev radius 3 — and a non-trivial 31% outside it.
2. *"No global spatial mixing"* — true for `pi_source`/`pi_target` only, refuted
   for the rest of the net. Cross-attention adds no spatial mixing (spatial
   tokens are queries; K/V are the 16 player tokens). `p_fc_shared` IS fully
   connected over all 121 tiles, but feeds only `pi_action`/`pi_option`. The
   value head's global mean+max pool (`network.rs:316-319`) is a sibling branch
   off `shared` and contributes nothing to the spatial heads at inference; the
   only coupling is training-time gradient through the trunk, which is exactly
   what `DETACH_VALUE_TRUNK` switches off.
3. *Cross-attention as terminal layer* — CONFIRMED in both implementations
   (`network.rs:290` → reshape → heads; `train.py:203` → heads). No feed-forward
   sublayer (q/k/v/o + one LayerNorm, `network.rs:114` / `train.py:119`), and no
   nonlinearity between the attention output and `pi_source`/`pi_target`. One
   ReLU does exist downstream on the action/option path (`network.rs:300` /
   `train.py:209`), so the claim is exactly true for the spatial heads and
   slightly overstated for action/option. `train.py:120` declares an unused
   `self.relu` inside `CrossAttention`.

No architecture change recommended on this item; like R1 it is blocked on a
working gauge.

---

## E — Engine and backends

### E1 · `metal_network.rs` looks up BatchNorm-era tensor names
**Status:** FIXED; COMPILES IN CI, RUNTIME UNVERIFIED (`73dafb9`, `ce35b31`, `f605deb`) · **CONFIRMED** · Effort: hours

**What landed:** the `bn*` prefixes are renamed to the `gn*` names checkpoints
actually store (`src/ai/metal_network.rs`), and `examples/metal_parity.rs` /
`examples/tch_parity.rs` now assert instead of printing, ignoring the
deliberately Python-only `aux_*` heads.

**Now covered automatically, in two ways the original fix was not.**

1. **It compiles.** The CI feature matrix added for T2 builds `metal-eval` and
   `tch-eval` on `macos-latest`. On its first run all three GPU backend jobs
   failed, and two of the breaks predated this work entirely: `eval_backend.rs`
   called `utils::cuda_is_available()` / `utils::metal_is_available()` without
   importing `utils` (so neither the `cuda` nor the `metal` path had compiled
   since that helper was written), and `metal_network.rs` called
   `graph.concatenate(...)`, which apple-mpsgraph does not have — that one sits
   on the value head's mean+max pool, so **`metal-eval` had not compiled since
   EXP_ARCH_001 landed**. Fixed in `ce35b31`; all four backends now build.
   Consequence for `expert_boost_throughput.md`: its ~610–650 moves/s was
   measured on a backend the tree could not build, so those numbers need
   re-taking before they are quoted again.
2. **The key contract is checked without Apple hardware.**
   `tests/test_backend_weight_keys.py` reads the weight keys each backend looks
   up straight out of the Rust source and diffs them against the state_dict
   `init_model.py` writes — including an explicit "no backend asks for a
   BatchNorm-era key" case, which is this finding. Both backends panic on a
   missing key at graph-build time, so a name-level check catches the whole
   failure mode statically.

**STILL UNVERIFIED — runtime.** Neither backend has been *executed* against a
current checkpoint; compiling and asking for the right key names is not the same
as producing correct numbers. The first person on Apple hardware should run:

```bash
cd polyfish-rs && cargo run --release --features metal,metal-eval --example metal_parity
cd polyfish-rs && cargo run --release --features tch-eval --example tch_parity
```

```rust
// metal_network.rs:554
let x_gn = self.group_norm(&graph, &x, "bn1", b);
// :365 — prefix is used directly as a weight key
let (wshape, _) = self.get(&format!("{prefix}.weight"));   // → "bn1.weight"
```

Checkpoints store `gn1.weight`. The op itself is a correct GroupNorm (`:360–382`);
only the key names were missed in the migration. This is the backend the entire
`expert_boost_throughput.md` campaign was built around — confirm whether
`metal-eval` can load a post-migration model at all. `examples/metal_parity.rs`
exists for exactly this.

### E2 · Engine correctness items
**Status:** VERIFIED (was FLAGGED) — 6 of 7 fixed, 1 documented as deliberate

All seven claims were checked against the source and reproduced with throwaway
integration probes; six CONFIRMED, `Research` PARTIALLY_CONFIRMED. The
per-claim outcome is below; the original list follows unchanged.

- **In-tree `EndTurn` deletes the opponent's turn** — CONFIRMED, and it was the
  largest behaviour defect in the repo. Proven: `simulate_move(EndTurn)` left
  player 1 to move while the opponent collected income (stars 5→7) and had its
  units refreshed; a 60-ply descent across 15 turn boundaries visited only player
  1; production `self_play` reported `Sim EndTurn edges: 2076 total (5.06 per
  move decision)` at only 16 MCTS iterations. FIXED, **opt-in**:
  `game::adversarial_search()` (`src/game.rs:34`, override
  `set_adversarial_search` `:51`, env `POLYFISH_ADVERSARIAL_SEARCH=1`) makes the
  `EndTurn` branch of `simulate_move` (`:390`) hand over once instead of looping
  back to the mover. The blocker shipped with it: `clone_for_mcts` (`:258`) now
  confines the in-tree opponent to the root player's vision, so it plays a
  belief-state army rather than moving units the search state has erased. Sign
  handling was audited across all backends — `mcts_zero` negates a handover
  child before comparing siblings (`src/ai/mcts_zero.rs:57`,
  `mcts_common::edge_hands_over` `:159`), the plain `mcts.rs` minimises at
  opponent nodes (`:72`), and Gumbel's tree reuse refuses to re-root across a
  handover (`src/ai/gumbel_mcts.rs:1290`). **Default is OFF and nothing in
  `run_training_loop.sh` sets it** — registered as EXP_SEARCH_001, unmeasured.
  `arena --adversarial` grades it.
- **`max_turns_ahead` ignores its `max_turns` argument** — CONFIRMED. FIXED:
  `(max_turns - current_turn).clamp(MIN_TURNS_AHEAD, MAX_TURNS_AHEAD)`
  (`src/ai/brain.rs:430`, bounds at `:421`/`:426`). Registered as EXP_SEARCH_002.
- **`freeze_area` never freezes, and its undo permanently turns water into Ice** —
  CONFIRMED. Marked out of scope as Polaris-only, but it was fixed anyway because
  a non-round-tripping undo corrupts the MCTS tree for everyone: the mutation is
  applied forward and the OLD terrain is captured in the undo
  (`src/actions/mod.rs:430`, tile branch at `:444-450`), `AutoFreeze` is applied
  on unit movement
  (`src/actions/units.rs:568`), and `tests/freeze_undo.rs` pins the round-trip.
- **`Research` is close to a no-op inside the search** — PARTIALLY_CONFIRMED
  (`discovered` was gated on `_are_you_sure`, which the sim path never sets).
  FIXED: `unlock_tech` always marks the tech discovered
  (`src/actions/tech.rs:37`), so the search can plan "research X, then use X".
  The dead sibling helper `technology::is_tech_unlocked` — documented as the
  simulation-aware check and never called from anywhere — was removed.
- **The search cannot reveal fog** — CONFIRMED as behaviour, but it is the
  deliberate anti-cheating design (`_are_you_sure` is intentionally not set on
  the sim path, `src/game.rs:385-395`). **Documented, not "fixed"** — do not
  schedule it.
- **`self_play` records a sample for a move, then silently drops the move if
  `execute` fails** — CONFIRMED. FIXED: nothing is recorded until the move
  actually lands, and a rejected legal move discards the whole game and counts
  into a new `aborted_games` metric instead of vanishing
  (`src/bin/self_play.rs:933`, `:962`, metric at `:2506-2514`).
- **Self-play maps are asymmetric (Drylands seat imbalance)** — CONFIRMED, and
  worse than stated. Over 500 seeds of the exact Tiny/Drylands config self_play
  generates, player 2 started on an island with ≤2 land neighbours in 166/500
  games (33%) versus 0/500 for player 1; mean land in the 8 tiles around the
  capital was P1 8.00 vs P2 6.01; 469/500 seeds gave materially different seats.
  With `symmetric: true` every metric is identical and island starts drop to
  0/500. FIXED for training: `self_play --symmetric` now defaults to true
  (`src/bin/self_play.rs:1342`), and `symmetric_maps_give_both_seats_the_same_start`
  (`src/mapgen.rs:1712`) pins seat equality over 120 seeds; the measurement itself
  is kept as an `--ignored` diagnostic at `:1729`. **NOT fixed for the gauge** —
  see M2.

- ~~`freeze_area` never freezes, and its undo permanently turns water into Ice.~~
  **OUT OF SCOPE (owner):** Polaris is out of scope. Do not spend effort on
  Polaris-specific mechanics; the same applies to any other Polaris finding.
- `max_turns_ahead` ignores its `max_turns` argument and hard-codes a 20-turn
  game, collapsing the search horizon to 2 turns from turn 18 onward.
- The search cannot reveal fog, so multi-turn expansion into unexplored ground is
  unplannable.
- `Research` is close to a no-op inside the search: it costs stars and grants
  score, but the `discovered` flag is false in simulation so nothing unlocks.
- In-tree `EndTurn` reportedly deletes the opponent's turn — the opponent
  collects income and is refreshed, then passes without moving. If true this
  means the search is not adversarial. **Verify this first**; it is the largest
  claim in the sweep and I did not reproduce it.
- `self_play` records a training sample for a move, then silently drops the move
  if `execute` fails.
- Self-play maps are asymmetric (Drylands seat imbalance); symmetric mapgen
  exists but is unused for training.

### E3 · Hot-path allocation
**Status:** OPEN — unchanged · **FLAGGED** · Effort: days

Audit estimates, not profiles — measure before changing, per
`expert_boost_throughput.md`'s own rule:

- `settings/units.rs` and `settings/structures.rs` rebuild a `HashSet` per lookup
  inside movegen and Dijkstra (~15% of actor instructions).
- `predict_explorer` rebuilds a whole-map field 12× per call with a container
  clone each pass (~9%).
- Pathfinding uses `HashMap` + `BinaryHeap` on a ≤121-node graph, run twice per
  step move.
- `IndexMap` + SipHash for tile/structure/resource lookups (~7%).
- Leaf feature hashing costs ~26µs/leaf, comparable to generating all legal moves.
- Each leaf's 67 KB feature buffer is copied twice, once on the single-threaded
  coalescer.

Release profile is already tuned (`lto = "fat"`, `codegen-units = 1`); only
`target-cpu` is unset.

---

## T — Testing, CI, and ops

### T1 · No Rust↔Python forward-parity test
**Status:** FIXED · **CONFIRMED** · Effort: days

**What landed earlier (`73dafb9`):** `tests/parity_widths.rs` runs in CI and
fails if any Rust head width, channel count or player-state dim disagrees with
the constant `train.py` declares — that half would have caught P3.
`examples/tch_parity.rs` and `examples/metal_parity.rs` assert on output
agreement instead of printing, but both need macOS/libtorch.

**What landed now:** the numerical half, on Linux CPU, in CI.
`examples/py_parity.rs` loads a `model.safetensors` into the candle network and
emits its raw outputs; `scripts/py_parity.py` builds `train.py`'s PyTorch
definition on the same file and the same closed-form input and compares raw
logits at 1e-3. `scripts/run_forward_parity.sh` runs both halves and
initialises a checkpoint if the tree has none, so a clean checkout can run it.
CI job: `forward-parity`.

**It found a real bug on its first run, and the bug was in the default
backend.** `network.rs` built its cross-attention query tokens as
`x.flatten_from(2)?.transpose(1, 2)?` — a strided view — and fed that straight
into the attention's `q_proj`. What was measured:

- without a `.contiguous()`, candle disagrees with PyTorch by O(10) on every
  head at batch 4, and agrees at batch 1;
- with it, candle matches PyTorch to ~1e-4 at both batch sizes, on a freshly
  initialised checkpoint and on the tree's own `model.safetensors`;
- in isolation, `candle_nn::Linear` on that exact strided layout returns
  different values from its contiguous equivalent for every row after the
  first — including when all rows carry identical data, so it is the row's
  position, not its contents.

`tch_network.rs` builds the same strided view (`:192`) but hands it to
libtorch, and `metal_network.rs` composes an MPSGraph; neither has the problem.
Candle is the default backend and the only one on non-Apple hardware, so this
was live for every Linux and CUDA run, on every batched evaluation — which is
what the eval server does by design.

**Note for whoever extends this.** A batch-invariance test — batched row *k*
against row *k* evaluated alone — does **not** catch it, and was tried. candle's
batched matmul is bitwise row-independent, so both paths return the same wrong
answer and the test passes cleanly with the bug present. An oracle outside
candle is required, which is precisely why this test is worth its cost.

### T2 · CI cannot catch the failure modes that actually occur
**Status:** FIXED (`73dafb9`) · **CONFIRMED** · Effort: days

**What landed:** `.github/workflows/rust.yml` gained the width-parity test, a
shell→binary CLI contract check (`scripts/check_cli_contract.py --no-build`, run
against freshly built binaries), correctness clippy as a gate plus a full clippy
and `cargo fmt` advisory pass, a Python syntax compile pass, and a feature-flag
compile matrix. `.github/workflows/smoke.yml` is a nightly (and
`workflow_dispatch`) end-to-end run of `scripts/smoke_train_loop.sh` — self_play
→ `games_*.safetensors` → train.py → model.safetensors plus one arena gauge
reading — which is the seam all three blockers hid in. The smoke also forces the
anchor-freeze and audit branches, and the contract check covers the shell's
python CLIs per subcommand (#35).

```bash
# Re-verify locally
cd polyfish-rs && python3 scripts/check_cli_contract.py
cd polyfish-rs && bash scripts/smoke_train_loop.sh
```

`.github/workflows/rust.yml` builds and tests with `--no-default-features` only.
No clippy, no fmt, no release build, no feature-flag builds, no Python tests, and
nothing exercises the shell → binary → safetensors → train.py seam. P1, P2, and
P3 are all invisible to it. A one-iteration end-to-end smoke run would catch all
three.

### T3 · Other testing and ops items
**Status:** MOSTLY FIXED · **FLAGGED**

FIXED: the decomposed mapper has tests, and its option block now carries a
compile-time assertion per family (`src/ai/mapper.rs:86-92`) so a 23rd
`AbilityType` fails the build instead of silently aliasing onto
`CityRewardType::CityWall`. `model.safetensors` is written atomically
(`train.py:416-420`, `:942`) and a present-but-unloadable checkpoint is fatal
rather than a silent restart from random weights. The experiment record is no
longer single-machine: `training_log.csv` and `ladder.json` are tracked in git
(`.gitignore`) and `scripts/backup_experiment_record.sh` snapshots the record
(plus `checkpoints/`, which stays gitignored) to another disk or a remote, with a
MANIFEST and SHA256SUMS. The Python env is pinned and consistent
(`requirements.txt`, single `POLYFISH_TORCH_VERSION` pin read by all three setup
scripts, each installing the torch wheel its target needs — `local_setup.sh` no
longer installs none). The dashboard now emits every column `training_log.csv`
records, header-driven rather than a fixed struct (`src/main.rs:1839`,
`src/bin/dashboard.rs:55`), and `training.html` drops the charts nothing produced
in favour of value-label composition, decisive-game rate and policy KL.

ALSO FIXED: `train.py` has a test suite, and search is reproducible.

**The mapper's ability-block guard was vacuous.** The audit claimed a 23rd
`AbilityType` would fail the build. `ability_slot`'s match is exhaustive, so a
new variant does force you to write an arm — but writing `=> 21` compiled fine
and landed the ability on `OFFSET_REWARDS`, because `ABILITY_SLOTS` is a
hardcoded `21` and the assertion was against that count, not against the
mapping. A const block now walks `AbilityType::from_repr` and checks each slot
lands inside the block and that no two share one; both arms of it were verified
to fail the build.

`tests/test_train.py` and `tests/test_ladder.py` (stdlib `unittest`, no new
pinned dependency; `scripts/run_python_tests.sh`, CI job `python-tests`) cover
the helpers whose failure mode is silent — the holdout split's partition and
stability invariants, its exclusion of the teacher anchor files (#36),
`pad_spatial`'s append-don't-prepend contract, D4 as a group action, the
shell↔`ladder.py` command lines the training loop builds (#35), and the
Rust↔Python width contract read from the Python side, which runs without torch.

**Search reproducibility took two fixes, and the second was the real one.**
Every search agent now owns a seeded `SmallRng` — `GumbelMctsAgent`,
`ZeroMctsAgent`, `HeuristicMctsAgent`, `GreedyHeuristicAgent` and `RandomAgent`,
via `mcts_common::next_search_rng` and a per-agent `with_search_seed`, or
`POLYFISH_SEARCH_SEED` for a pinned base stream that still differs per agent (a
shared stream across actors would make every actor play the same game). The last
three matter beyond tests: the heuristic agents are the greedy teacher, so their
randomness reaches training data, and `RandomAgent` is the ladder's Elo-0 floor.
That
alone did not make a search replayable: `generate_legal_moves` returned the same
moves in a **different order** on every run. Two containers in movegen were
iterated to emit moves — `compute_reachable_tiles`'s `HashMap` for step targets
(`src/moves/mod.rs:377`) and `generate_research_moves`'s `HashSet`
(`src/moves/research.rs:113`) — and Rust seeds each map instance separately.
Order decides which move receives which Gumbel draw, so a permuted list is a
different search. Both are ordered now (`BTreeMap`, and a sort before emission),
and `tests/search_determinism.rs` holds it. Note this is invisible to a
move-*type* comparison: the permutation is within a type, which is why an
earlier order check passed while the search still diverged.

**Correction — the crash-recovery item above was stale.** Re-verified at
`ce35b31`: the automatic path is already run-scoped
(`run_training_loop.sh:306-328`). It restores only
`model_checkpoint_iter*_run${RUN_ID}_*`, falls back to this run's per-launch
snapshot, and when neither exists it **exits 1** naming the newest untagged
checkpoint rather than adopting it. Nothing left to do here.

STILL OPEN: nothing in this item. The remaining T3-adjacent gap is that
determinism now holds for the search and for movegen order, but a full run is
still not reproducible end to end — mapgen seeds, actor scheduling and the
eval-server batching order are all outside what these tests pin.

- `train.py`, the primary trainer, has no test infrastructure at all.
- The decomposed mapper has no tests; its ability block has zero headroom — a
  23rd `AbilityType` silently aliases onto `CityRewardType::CityWall`.
- Search agents draw from the unseeded global RNG, so no test can pin search
  behaviour.
- `model.safetensors` is written non-atomically, and a failed load falls back
  silently to "starting from scratch". `ladder.py` already does this correctly
  (`.tmp` + `os.replace`) — copy that pattern.
- ~~Crash recovery restores a checkpoint from the wrong run.~~ FIXED: periodic
  checkpoints now carry the writing run's id
  (`model_checkpoint_iter${i}_run${RUN_ID}_${TS}.safetensors`, still matching the
  glob the pruning logic uses), and the resume path restores only from this run —
  its own periodic checkpoints first, then its launch snapshot. If neither
  exists it aborts and names the newest untagged checkpoint rather than silently
  adopting another run's weights (`run_training_loop.sh:303-325`).
- `training_log.csv`, `ladder.json` and `checkpoints/` are gitignored with no
  off-box durability — every experiment record lives on one machine.
- Python env is unpinned and inconsistent across the three setup scripts;
  `local_setup.sh` installs no torch.
- The dashboard plots four metric families nothing produces, and the API drops
  five the CSV does record.

---

## Refuted — do not re-report

| Claim | Why it fails |
|---|---|
| "D4 aug is off by oversight; just enable it" | Off deliberately; mid-run enable measured to collapse play for ~8 iterations (run 1783556259). See A4. |
| "A chunk OOM is silently swallowed, dropping the chunk" | `train.py:554–558` only continues on a genuine OOM signature; the finding's own evidence contradicts its headline. |

Also note `expert_boost_throughput.md` has a "What NOT to do" section, and the
arena-tree / O(D²) path-walk refactor from `expert_review.md` was re-examined and
judged not worth doing.

---

## Order of operations

This is a dependency chain, not a ranking. Nothing below step 5 can be evaluated
until 1–4 are done, because until then there is no working instrument.

1. **P1 + P2** — restore the flag contract, make the gauge fail loudly.
2. **P3** — reconcile the action-head width, add a width assertion.
3. **M1 + M2 + M4** — seed control, aligned search knobs, matched `max_turns`.
4. **A4** — restore the deleted D4 caveat (minutes, prevents a repeat of run
   1783556259).
5. **Re-baseline.** With a working, seeded, aligned gauge, take a fresh reading.
   The "cannot beat its own greedy anchor" premise may not survive it.
6. **A2b, then A1, in that order.** Reweight the value label toward army value —
   measured at ~8pp better than score at every turn of a Domination game. Settle
   that before tuning A2's relative/absolute constant, and before removing the
   detach export: with the current label, removing it early risks reproducing the
   harm it was probably introduced to mask. Once the label is fixed and a gauge
   exists, run the detach arm both ways and record the verdict.
7. **T1 + T2** — parity test and an end-to-end smoke run, so this class of break
   cannot recur silently.
8. **R1**, then R2/R3 — architecture work, scheduled against the new baseline.

Steps 1–4 are all hours of work. A2b is the first item that is genuinely a
design change rather than a repair, which is why it sits after the re-baseline —
it needs a working instrument to be judged against.

### Where the chain actually stands (Aug 18, 2026)

Steps 1, 2, 3, 4 and 7 are **done**. Step 5 — the re-baseline — is the next
action and nothing downstream of it should be started first.

Two things must be settled *before* the re-baseline reading is taken, or it will
measure the wrong thing:

- **What the baseline is a baseline OF.** This wave landed several behaviour
  changes that are pre-registered but unmeasured (`EXP_LABEL_001`,
  `EXP_SEARCH_002`, `EXP_DATA_001`, `EXP_DATA_002`, `EXP_TEACH_001`,
  `EXP_TRAIN_001` in `hypothesis_driven_improvements.md`). They are all ON by
  default and cannot be separated by a single reading. `EXP_SEARCH_001`
  (adversarial in-tree search) is the exception — it is OFF by default and is the
  one arm that can be A/B'd against the same baseline.

Step 6 is unchanged and its ordering warning now matters more, not less: **A2 was
settled ahead of A2b** (one zero-sum constant landed; the label still reads raw
`score`). If the re-baseline reads worse than the old band, check that before
concluding anything about the search changes.

---

## Method and coverage

Twelve parallel dimension audits (search, targets, trainer, net-sync, features,
architecture, throughput, engine, testing, measurement, ops, hygiene), each
followed by an adversarial verification pass instructed to refute. The verifiers
upheld an unusually high fraction of findings, which is itself a reason to treat
FLAGGED items as leads rather than conclusions — one of the two refutations
(D4) overturned a claim I had already written up as high-severity.

Not covered in depth: the `polyfish-mod` C# side, `polyfish-scraper`, the replay
subsystem's correctness, and the `polyfish-ui` fork. The engine dimension (E2)
returned more than is captured here and deserves its own pass.

**Aug 18:** E2 got that pass — all seven claims reproduced with throwaway
integration probes, six of them fixed (see E2). R1/R3/R4 were independently
re-verified against the source and the live PyTorch net; R4 came back materially
different from how it was written. The `polyfish-mod` C# side, `polyfish-scraper`
and the replay subsystem are still unaudited.

CLAUDE.md was corrected in the same session (commit `459a32b`) — its
dual-network sync section had the wrong channel count, player-state dim, and
legacy pad width, and referenced a `replayer.rs` and a `verdi` branch that no
longer exist.
