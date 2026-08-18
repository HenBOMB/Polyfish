# Pipeline Audit — Aug 2026

Twelve-dimension audit of the training pipeline, engine, and measurement layer,
run at commit `2bc3160` (branch `main`). Companion to `expert_review.md` (search
and learning-signal review) and `expert_boost_throughput.md` (throughput).
Rendered report: https://claude.ai/code/artifact/fdc8836d-b4ac-4ff0-bef3-642c3e88c588

**Headline:** the training loop cannot execute at HEAD, and the strength gauge
has never produced a reading. Three shell-to-binary contract breaks account for
both. Every conclusion in `hypothesis_driven_improvements.md` that depends on a
gauge reading — the plateau verdict, EXP_ELO_002's "success bar not met" — was
drawn from an instrument that was returning parse failures.

## How to use this file

Each item has an ID, a status, and a **Verify** command that re-checks it in
seconds. Update `Status:` as items are fixed. Confidence tiers:

- **CONFIRMED** — verified by reading the cited lines at `2bc3160`.
- **FLAGGED** — from the audit sweep, survived an adversarial verification pass,
  but not independently reproduced. Treat citations as leads.

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
**Status:** OPEN · **CONFIRMED** · Effort: hours

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
**Status:** OPEN · **CONFIRMED** · Effort: hours

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
**Status:** OPEN · **CONFIRMED** · Effort: hours

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

Fix: decide whether Resign is a real action, make both sides read one constant
(`network.rs:178` should use `NUM_ACTION_TYPES`), and add a producer/consumer
width assertion so they cannot drift again.

---

## M — Measurement (readings are not comparable)

### M1 · No seed control anywhere; every reading uses different maps
**Status:** OPEN · **CONFIRMED** · Effort: hours

`arena.rs:348` derives `base_seed` from `SystemTime::now()` and exposes no
`--seed`. Side-swapping is already implemented (`:465–467`), so seat bias is
handled — but map difficulty is re-rolled every reading. The gauge series
`31.2 → 37.5 → 23.4 → 35.9 → 40.6 → 33.6%` carries full map variance on top of
the model delta, and that is what the plateau detector reads.

Fix: add `--seed`, pin a fixed evaluation map set for the ladder. Converts
between-reading comparisons into paired ones for zero extra compute. This is
also a prerequisite for reproducing any past experiment.

### M2 · The gauge grades a different agent than self-play trains
**Status:** OPEN · **CONFIRMED** · Effort: hours

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
**Status:** OPEN · **FLAGGED** · Effort: days

At `GAUGE_GAMES=32` (64 games after swapping) and p≈0.33, binomial SE ≈5.9pp →
95% interval ≈±11.5pp. EXP_ELO_002's registered bar was +8pp, observed +1pp; the
30.7→36.7% within-run drift was called "suggestive". Both sit inside the noise.

Fix M1 first (free variance reduction), then size the remaining budget against
the effect you want to detect, and store the interval in `ladder.json` alongside
every reading.

### M4 · Gauge plays a shorter game than training generates
**Status:** OPEN · **CONFIRMED** · Effort: hours

`curriculum()` (`self_play.rs:201–211`) runs `max_turns = 45` past iteration 30.
The gauge never passes `--max-turns`, so arena uses its default of 30
(`arena.rs:57`). Late-game strength is outside the measured window.

### M5 · Other measurement gaps
**Status:** OPEN · **FLAGGED**

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
**Status:** OPEN · **CONFIRMED** · Effort: hours

`run_training_loop.sh:17`. `train.py:35–39` documents it as "bisect Arm D", and
`bisect_arm.sh:14` treats it as a diagnostic. With it on, no value-loss gradient
reaches `conv1`, the ResBlocks, or the cross-attention — the trunk is shaped only
by the four policy heads plus a 0.15-weight ownership aux, and the value head is
a linear probe on features selected for something else.

Nuance: the export predates the `3893daf` re-import (present at `5ecdb5d~1` too),
so it is a long-standing setting rather than a fresh accident. But there is **no
recorded verdict for it** in `hypothesis_driven_improvements.md`. Establish
whether it is deliberate before removing it — and if it is, record why.

```bash
# Verify
grep -n 'DETACH_VALUE_TRUNK' polyfish-rs/run_training_loop.sh
grep -n -i 'detach' hypothesis_driven_improvements.md   # → no verdict recorded
```

### A2 · Two reward definitions disagree about zero-sum
**Status:** OPEN · **CONFIRMED** · Effort: hours

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

### A3 · Optimizer and LR schedule reset every iteration
**Status:** OPEN · **CONFIRMED** · Effort: days

`train.py:426` constructs a fresh `Adam` per invocation and `:429` a
`CosineAnnealingWarmRestarts` that restarts at the top LR on every call — a
sawtooth, not a schedule, and Adam's moments are discarded each time.
`expert_review.md` listed "persistent optimizer" as a cleanup; not landed.

### A4 · The measured rationale for `AUGMENT_D4` was deleted from its comment
**Status:** OPEN · **CONFIRMED** · Effort: minutes

See the correction section above. The current comment (`train.py:46–50`) reads as
an unconditional endorsement; the measured mid-run collapse (run 1783556259) is
gone. Restore the caveat. D4 remains a legitimate multiplier **for from-scratch
runs only**.

### A5 · Other learning-signal items
**Status:** OPEN · **FLAGGED**

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
**Status:** OPEN · **CONFIRMED** · Effort: days

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
**Status:** OPEN · **FLAGGED** · Effort: days

Search forms `P(move) ∝ P(action)·P(source)·P(target)·P(option)`; training fits
each marginal independently. Move types using fewer heads are multiplied by fewer
sub-1 factors — a structural prior bias unrelated to anything learned. Two
independent remedies: a "not applicable" slot per head so every move consumes the
same factor count, and conditioning source/target/option on action type.
Related: unit-ability moves are reportedly generated twice, putting two identical
children at one policy coordinate.

### R4 · Receptive field and cross-attention
**Status:** OPEN · **FLAGGED**

Audit reports an effective receptive field of ~±3 tiles with no global spatial
mixing, and that cross-attention is the terminal layer — no feed-forward
sublayer, no post-injection nonlinearity. Worth checking against the 6-block
trunk before acting.

---

## E — Engine and backends

### E1 · `metal_network.rs` looks up BatchNorm-era tensor names
**Status:** OPEN · **CONFIRMED** · Effort: hours

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
**Status:** OPEN · **FLAGGED**

- `freeze_area` never freezes, and its undo permanently turns water into Ice.
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
**Status:** OPEN · **FLAGGED** · Effort: days

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
**Status:** OPEN · **FLAGGED** · Effort: days

Nothing loads a Python-produced `model.safetensors` into the Rust network and
compares outputs. The parity examples never assert. Given four backends must
agree byte-for-byte, this is the highest-value missing test in the repo — and it
would have caught P3 and E1.

### T2 · CI cannot catch the failure modes that actually occur
**Status:** OPEN · **CONFIRMED** · Effort: days

`.github/workflows/rust.yml` builds and tests with `--no-default-features` only.
No clippy, no fmt, no release build, no feature-flag builds, no Python tests, and
nothing exercises the shell → binary → safetensors → train.py seam. P1, P2, and
P3 are all invisible to it. A one-iteration end-to-end smoke run would catch all
three.

### T3 · Other testing and ops items
**Status:** OPEN · **FLAGGED**

- `train.py`, the primary trainer, has no test infrastructure at all.
- The decomposed mapper has no tests; its ability block has zero headroom — a
  23rd `AbilityType` silently aliases onto `CityRewardType::CityWall`.
- Search agents draw from the unseeded global RNG, so no test can pin search
  behaviour.
- `model.safetensors` is written non-atomically, and a failed load falls back
  silently to "starting from scratch". `ladder.py` already does this correctly
  (`.tmp` + `os.replace`) — copy that pattern.
- Crash recovery restores a checkpoint from the wrong run.
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
4. **A1 + A2 + A4** — resolve the detach switch, unify `REL_W`, restore the D4 caveat.
5. **Re-baseline.** With a working, seeded, aligned gauge, take a fresh reading.
   The "cannot beat its own greedy anchor" premise may not survive it.
6. **T1 + T2** — parity test and an end-to-end smoke run, so this class of break
   cannot recur silently.
7. **R1**, then R2/R3 — architecture work, scheduled against the new baseline.

Steps 1–4 are all hours of work.

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

CLAUDE.md was corrected in the same session (commit `459a32b`) — its
dual-network sync section had the wrong channel count, player-state dim, and
legacy pad width, and referenced a `replayer.rs` and a `verdi` branch that no
longer exist.
