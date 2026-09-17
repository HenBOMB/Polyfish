# PolyStar v3 - Next Move & Issue Ledger

Status snapshot: 2026-09-12. Spec is [`POLYSTAR-V3-ARCH.md`](POLYSTAR-V3-ARCH.md).
Nothing in the v3 pipeline (G0, Step 2, G1) is runnable today. Every claim below
cites a file and line.

---

## NEXT MOVE

Build the pipeline, then gate it honestly.

1. Fix the three blockers below (B1-B3). B1 is decided (re-enable `EndTurn`
   in the teacher; see BLOCKING) and needs the adversarial-search companion flag.
2. Run roadmap Steps 1-4 (spec section 8) to produce `train_polystar.py`,
   `policy.pt`, `src/ai/tch_policy.rs`, and a `policy` backend.
3. Run Gate G1: `arena --backend1 policy --backend2 greedy --symmetric --games 100`
   (`POLYSTAR-V3-ARCH.md:414`).

**G1 reading.** The spec's bar is >= 50%. That bar is too soft to be informative:
the *legacy* PolyZeroNet already sat at ~75-81% vs Greedy
(`notes-runs-2026-07-14-16.md` run 2: iter72 14-2, iter84 13-3). Treat 50% as
"not catastrophically broken" only. The number that actually justifies v3 is
**beating the legacy's ~75-81%**, measured over enough games that the 16-game
noise band no longer decides the verdict.

---

## BLOCKING

### B1. Teacher target and candidate set disagree on `EndTurn` - APPLIED
**Status: CODE CHANGE APPLIED 2026-09-12, compiling, tests green.**
Gated behind `POLYFISH_TEACHER_KEEP_END_TURN=1` / `set_teacher_keep_end_turn(true)`;
default OFF, so existing runs are byte-identical. Requires adversarial search
(asserted in debug builds). Remaining: spec text edits and the recorder wiring.
**Decision (2026-09-12): option (b) - add `EndTurn` back to the teacher's root
set for corpus generation, so pi-prime carries real stop-decision signal.**

Why: PolyStar dropped MCTS (D1), so the policy makes the stop decision at every
ply. `EndTurn` is the single most consequential move - the spec calls it "the
most important stop decision" (`POLYSTAR-V3-ARCH.md:272`, `:304`). Suppression
was a safety rail bolted onto a *search* process (`gumbel_mcts.rs:533-534`:
"Suppress EndTurn at the root when any other move exists to prevent passive
play"); with the tree gone it removes supervision of the one decision the policy
cannot learn from anywhere else. `score_move` already scores `EndTurn` at 0.0
(`polyfish-rs/src/ai/scoring.rs:736`), so Greedy has a sane, conservative prior
for it.

**REQUIRED COMPANION CHANGE (do not skip).** Re-enabling `EndTurn` at the root
is *unsafe* unless the teacher also runs with adversarial search ON:
- `simulate_move` with `adversarial_search() == false` DELETES every opponent
  turn in between and hands control straight back to the mover
  (`polyfish-rs/src/game.rs:433-445`, "keep ending turns until control is back
  with the mover, i.e. delete every opponent turn in between").
- So a non-adversarial search would value `EndTurn` as "I get another turn
  immediately" - a degenerate, falsely high target, and the policy would learn
  to chain EndTurns forever.
- `self_play` never enables it today: `args.adversarial` -> `set_adversarial_search`
  exists only in `arena` (`polyfish-rs/src/bin/arena.rs:506-507`); `grep -c
  adversarial src/bin/self_play.rs` = 0, and the env default is off
  (`game.rs:38-44`).
- Therefore corpus generation must run with `POLYFISH_ADVERSARIAL_SEARCH=1` (or
  call `set_adversarial_search(true)`) whenever the EndTurn-inclusive teacher is
  used. This is a behavioral change from every prior self-play run and must be
  logged in the METRICS line.

**Change sites** (all root-level suppressions; interior expansion already keeps
`EndTurn`, `gumbel_mcts.rs:475-476`):
- Gumbel fresh root: `polyfish-rs/src/ai/gumbel_mcts.rs:533-539`
- Gumbel reused root: `polyfish-rs/src/ai/gumbel_mcts.rs:477-486`
- Gumbel reuse-consistency check: `polyfish-rs/src/ai/gumbel_mcts.rs:1250-1266`
  (must be changed in lockstep or the multiset compare breaks and every reuse
  falls back to a fresh root)
- Greedy: `polyfish-rs/src/ai/heuristic_mcts.rs:142-145`
- Random: `polyfish-rs/src/ai/heuristic_mcts.rs:247-250`
- Heuristic MCTS: `polyfish-rs/src/ai/heuristic_mcts.rs:310-313`
- StateDiffGreedy: `polyfish-rs/src/ai/heuristic_mcts.rs:644-648`
- Spec text to fix: `POLYSTAR-V3-ARCH.md:380` and `:479` claim pi-prime is
  exported "over the full legal set" - true only once this lands; and any
  suppression assumption in the recorder (`POLYSTAR-V3-ARCH.md:238`) must match.

Gate it: add a `flag` so the legacy training path keeps suppression (protecting
the existing 75-81% checkpoints) and only the entity-corpus path re-enables it.
Verify with a METRICS counter (`endturn_cand_frac`) that `EndTurn` appears in the
candidate set and carries non-zero pi-prime mass in states where other moves
exist.

### B2. `tch-eval` build contract contradicts the repo's own torch pin
- Spec: build `LIBTORCH_USE_PYTORCH=1` against the CUDA pip torch in `.venv`
  (`POLYSTAR-V3-ARCH.md:441`).
- Repo pin is 2.12.x: `polyfish-rs/requirements.txt:13`
  (`# POLYFISH_TORCH_VERSION=2.12.1`), with `:15-16` stating 2.12 is required
  because tch 0.25 targets torch 2.12 and the version-check bypass is only
  justified across 2.12 patch releases.
- Installed venv is **2.13.0+cu130** - off-pin, outside that patch window.
  tch-rs is pinned to the rev "that merged 2.12 support"
  (`polyfish-rs/Cargo.toml:35-45`).
- Setup treats it as fatal: `polyfish-rs/vast_setup.sh:36-37` ("links against
  this exact torch, so an off-pin wheel is fatal here rather than a warning").
- Needed: re-pin torch to 2.12.x **or** re-validate tch-rs on 2.13. Step 4 is
  gated on this.

### B3. Step 2's "existing path" cannot load a checkpoint
- Roadmap Step 2 records the BC corpus via `self_play --record-schema entity`
  on the existing path, "No new network is required" (`POLYSTAR-V3-ARCH.md:446`).
- Code expects a 12-wide action head (`polyfish-rs/src/ai/network.rs:10`).
- Every checkpoint is 11-wide `pi_action.weight` (all 13 files in
  `polyfish-rs/checkpoints/`, plus `polyfish-rs/model.safetensors`).
  `self_play` and `arena` both fail:
  `shape mismatch for pi_action.weight, expected: [12, 64], got: [11, 64]`
  (`polyfish-rs/src/bin/arena.rs:521` load path).
- The migration exists but has never been run:
  `polyfish-rs/train.py:391-406` pads `pi_action` to `NUM_ACTION_TYPES`.
- Needed: run one checkpoint through that migration, or the branch cannot
  self-play at all.

---

## NOT BLOCKING (fix later, one-liners)

- **G1 bar too soft** - spec floor 50% (`POLYSTAR-V3-ARCH.md:414`) vs legacy
  ~75-81%; see NEXT MOVE above. Tighten or reframe G1.
- **`ABILITY_MAP` does not exist** - cited at `POLYSTAR-V3-ARCH.md:267`, `:556`;
  the real table is `ability_slot` (`polyfish-rs/src/ai/mapper.rs:167-194`).
- **Stale tie/adjudication citations** - spec says `self_play.rs:931-957` /
  `:950-956` (`POLYSTAR-V3-ARCH.md:327`); the actual `max_by_key` is
  `polyfish-rs/src/bin/self_play.rs:1059`.
- **Terrain count mismatch** - `TerrainType` has 9 variants incl. `Wetland`/
  `Mangrove` (`polyfish-rs/src/types.rs:56-57`) but `TERRAIN_COUNT = 8`
  (`polyfish-rs/src/ai/features.rs:26`). Unreachable from `mapgen.rs` today, so
  the 142/145 arithmetic still holds; the constant and its "Algae" comment are
  stale.
- **RunPod patch referenced but absent** - spec claims the scripts live only in
  an unapplied `runpod-training.patch` (`POLYSTAR-V3-ARCH.md:441`); no such file
  in the tree.
- **`forward_is` vs `forward_ts` unverified** - Appendix C item 1 and
  `POLYSTAR-V3-ARCH.md:436` cannot be checked offline (no vendored tch-rs source,
  no cargo git cache). Unverified, not disproven.
- **Throughput target likely unreachable by construction** - spec section 2
  targets "2,000+ moves/s" (`POLYSTAR-V3-ARCH.md:60`) while the actor ceiling its
  own Appendix A.3 sources measured was ~1,650 (`expert_boost_throughput.md:39`).
  Re-derive the target.
- **`matches.jsonl` does not exist** - G2 says results are logged to
  `elo_ratings.json` / `matches.jsonl` (`POLYSTAR-V3-ARCH.md:426`); only
  `polyfish-rs/elo_ratings.json` is present.

---

## APPLIED 2026-09-12 (B1 fix)

Gate added in `polyfish-rs/src/game.rs:59-105`:
- `TEACHER_KEEP_END_TURN` atomic, tri-state like `ADVERSARIAL_SEARCH`; env
  `POLYFISH_TEACHER_KEEP_END_TURN`, default off.
- `set_teacher_keep_end_turn(bool)` override.
- `end_turn_retained()` - single gate every backend consults; `debug_assert!`s
  the pairing with `adversarial_search()`.

Suppression sites now conditional (all no-ops by default):
- `polyfish-rs/src/ai/gumbel_mcts.rs:535-540` (fresh root)
- `polyfish-rs/src/ai/gumbel_mcts.rs:477-486` (reused root)
- `polyfish-rs/src/ai/gumbel_mcts.rs:1250-1262` (reuse-consistency check, kept in
  lockstep so reuse does not silently degrade to a fresh root)
- `polyfish-rs/src/ai/heuristic_mcts.rs:144` (Greedy)
- `polyfish-rs/src/ai/heuristic_mcts.rs:249` (Random)
- `polyfish-rs/src/ai/heuristic_mcts.rs:312` (Heuristic MCTS)
- `polyfish-rs/src/ai/heuristic_mcts.rs:648` (StateDiffGreedy)

Test: `polyfish-rs/src/ai/heuristic_mcts.rs` `teacher_end_turn_flag_controls_candidate_set`
- legacy mode: no `EndTurn` in the exported candidate set.
- teacher mode: `EndTurn` present with non-zero target mass, and `score_move`
  leaves it at 0.0 (`scoring.rs:736`) so it does not dominate.
- one test (not two) because the flags are process-global and parallel tests
  would race the atomics.

Verification: `cargo check --lib` clean; `cargo test --lib` 102 passed, 0 failed,
3 ignored.

STILL TODO for B1:
1. Wire `--record-schema entity` in `self_play` to call
   `set_teacher_keep_end_turn(true)` + `set_adversarial_search(true)` and log
   both in the METRICS line.
2. Add the `endturn_cand_frac` counter.
3. Edit `POLYSTAR-V3-ARCH.md:380`, `:479` ("full legal set" is now true under
   the flag) and reconcile `:238` / the `:304` wording.

## FALSIFIER (from the spec's own gates)

Run Gate G1's 100-game symmetric argmax arena vs deterministic Greedy
(`POLYSTAR-V3-ARCH.md:414`). It is machine-independent: Greedy is a
deterministic argmax (`polyfish-rs/src/ai/heuristic_mcts.rs:125-127`, with
`TEMPERATURE_MOVE_THRESHOLD = 0` at `polyfish-rs/src/ai/mcts_zero.rs:289`), and
the arena swaps seats (`polyfish-rs/src/bin/arena.rs:91-95`). Hardware only
changes how long the 100 games take.

- **< 50% wins -> design is falsified.** The entity-transformer policy failed to
  reproduce the Greedy teacher it was cloned from.
- **50-75% -> not broken, but weaker than legacy.** Does not justify v3.
- **> ~81% -> clears the legacy bar.** The only reading that argues for the new
  architecture.
