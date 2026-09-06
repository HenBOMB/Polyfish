# NN-driven root candidates: cut `rank_plies` out of micro-mcts and macro-mcts's rollouts

## Context

Yesterday we built a fast NN replacement (`PlyRanker`, `polyfish-rs/src/ai/ply_ranker.rs`) for `rank_plies`'s expensive CPU work (simulate-every-candidate-then-undo to price a goal-potential term), wired it in as a branch *inside* `rank_plies`, and measured it on the paired seed-770425 gauge: it beat the current full-Δφ approach on throughput (1.6-1.7x) but was strictly dominated by the already-shipped `--macro-rollout-lambda 0.0` lever on both throughput *and* quality (29.0% win rate vs. 38.7%/41.9%). Root cause: the network was trained to imitate `rank_plies`'s own hand-coded formula, which caps its ceiling at that formula's own (mediocre, apparently) quality — a copy can't beat the thing it copies.

Verdi's read, which this plan acts on: the *plumbing* is backward regardless of training target. Micro-mcts should get its root candidates directly from one NN forward pass over the cheap legal-move list, not from a scored heuristic list it then has to trim/patch. The same mechanism should drive how macro-mcts's ~6 root-candidate turns get played out during its own internal search. Separately, we should stop trying to imitate the fixed Δφ formula and instead train against what the AI's own search actually decides — a different, likely higher-ceiling target.

This plan has two parts: **Part A** (architecture rework) is being implemented in this session. **Part B** (training-target research) is written in full executable detail for a separate agent/session to pick up in parallel — nothing in Part B gets built by whoever executes Part A.

A directly relevant precedent already exists and ships today: **Gumbel MCTS** (`polyfish-rs/src/ai/search/gumbel_mcts/`), a separate, older search backend in this same codebase, already generates its root candidates exactly this way — cheap legal-move enumeration, one net forward pass, blend in a heuristic prior at a *decaying* weight, THEN cut to top-K. Part A is substantially a port of that pattern, not a new invention.

**Explicit scope note**: yesterday's checkpoint (`ply_ranker.safetensors`, currently sitting in `polyfish-rs/`) measured worse than production. This rework is a plumbing/throughput change, not a quality fix — Part B is where quality gets addressed. Don't expect the rework alone to beat the bar.

All file paths below are relative to `polyfish-rs/` unless stated otherwise.

---

## Part A — Architecture rework (implement now)

### A0. Current call graph

```
MacroMctsAgent::select_move (macro_mcts.rs:1449)
  → macro_agent::rank_view (macro_agent.rs:237)
      → macro_exec::rank_plies (macro_exec.rs:375)
          → [env-gated] net_rank_plies (macro_exec.rs:336) — being removed (A5)
          → CPU Δφ loop (simulate_move/undo per candidate)
  → micro_mcts::micro_search_pick (micro_mcts.rs:382)
      → [net_prior_w>0] second forward pass + union-widen (micro_mcts.rs:416-450) — simplified away for the net-derived case (A3)

expand() in MacroMctsSearch's tree (macro_mcts.rs:858)
  → macro_exec::execute_turn (macro_exec.rs:689), called at line 904
      → loops macro_exec::rank_plies once per ply
```

**Blast radius, verified directly** (do not assume these functions are private to the paths above):
- `execute_turn`/`execute_turn_recorded` are called from 13 sites: `belief/mod.rs:1091,1193,1316` (fog-of-war particle-filter rollouts), `macro_exec.rs` itself (the `execute_turn`/`execute_turn_recorded` wrapper relationship, plus 3 test sites), `macro_mcts.rs:904,1111,1665,1736,1927` (the rollout call plus a tier-probe and 3 tests), `macro_agent.rs:610` (`MacroLookaheadAgent::replan`).
- `rank_view` is called from 4 sites: `macro_mcts.rs:209` (`run_micro_probe`, a diagnostic that explicitly documents itself as mirroring "byte-for-byte the same per-ply sequence the real decision uses" — see A2b), `macro_mcts.rs:1452` (the one we're changing), `macro_agent.rs:218` (`MacroScriptAgent::select_move`, a frozen comparison arm whose doc comment demands byte-identical pricing), `macro_agent.rs:570` (`MacroLookaheadAgent::select_move`).

**Design consequence**: do not modify `execute_turn`, `execute_turn_recorded`, or `rank_view` themselves. Add new sibling functions that only the two call sites we actually want to change (`MacroMctsAgent::select_move`, `expand()`) opt into. Every other caller keeps calling the untouched originals and is unaffected.

### A1. New module: `src/ai/search/net_root.rs`

Register as `pub mod net_root;` in `src/ai/search/mod.rs` (alphabetical, after `micro_mcts`). New file, ~250-300 lines — keeps this net-new logic out of `macro_exec.rs` (1259 lines) and `macro_mcts.rs` (2158 lines), both already over the project's ~1000-line convention.

**Move `blend_heuristic_into_logits` out of `gumbel_mcts`.** It currently lives at `src/ai/search/gumbel_mcts/reuse.rs:41-56` as `pub(super) fn` (visible only inside `gumbel_mcts`). Move the function body + its `HEURISTIC_TEMP: f32 = 20.0` constant to `src/ai/search/policy_composer.rs` as `pub(crate) fn blend_heuristic_into_logits(logits: &mut [f32], heur_scores: &[f32], weight: f32)`. Update `gumbel_mcts/reuse.rs::blend_heuristic_prior` (line 119) to call the moved version via `crate::ai::search::policy_composer::blend_heuristic_into_logits`. One canonical implementation, shared by the old backend and the new one — not two copies of the same blend math.

**Core candidate builder** (this supersedes `net_rank_plies`, which is deleted in A5):

```rust
pub enum RootRankerSource<'a> {
    Ply(&'a crate::ai::ply_ranker::PlyRanker),
    MainNet(&'a crate::ai::eval_server::Evaluator),
}
```
(`MainNet` variant exists now so Part B's B1 experiment can plug in without touching this function again — not used by anything in Part A itself.)

```rust
pub fn net_rank_root_candidates(
    game: &Game,
    player: PlayerId,
    goal: &MacroGoal,
    aux: &GoalAux,
    star_gate: bool,
    source: RootRankerSource,
    unit_goals: Option<&crate::ai::search::unit_goals::UnitGoalStore>,
    eco_plan: Option<&crate::ai::eco_plan_commit::EcoPlanCommit>,
) -> Option<Vec<(f32, Box<dyn Move>)>>
```

Body (mirrors `gumbel_mcts::root::build_fresh_root`, `root.rs:136-198` — the validated shipping pattern — not `net_rank_plies`'s ad-hoc formula):

1. `let mut moves = game.legal_moves(); moves.retain(|m| macro_exec::gate_ok(&game.state, m.as_ref(), star_gate, Some(goal.stance), Some(aux)));` — reuse the existing `pub fn gate_ok` (`macro_exec.rs:216`, already public, no visibility change needed). Same EndTurn-suppression convention `rank_plies` already uses (`macro_exec.rs:392-398`): if any non-EndTurn move survives, drop EndTurn from the set; if empty, return `Some(vec![(0.0, Box::new(EndTurnMove))])`.
2. `let feats = crate::ai::features::state_to_cpu_features_goal(&game.state, player, None, Some(goal)).ok()?;` — same goal-painting convention every other net-prior call site in this codebase uses.
3. `let raw = source.forward(feats)?;` where `RootRankerSource::forward` dispatches to `PlyRanker::forward_raw` or `Evaluator::evaluate(vec![feats]).into_iter().next().map(|r| r.2)`.
4. `let mut logits = crate::ai::search::policy_composer::compute_move_log_probs_raw(&raw, &moves, map_size);` — the **log-domain** function (`policy_composer.rs:163-174`), not the probability-domain `compute_move_priors_raw` + manual `.ln()` that `net_rank_plies` used. This is the numerically cleaner building block Gumbel already trusts.
5. `let heur_scores: Vec<f32> = moves.iter().map(|m| scoring::score_move_with_unit_goals(game, m.as_ref(), unit_goals, eco_plan)).collect();`
6. `policy_composer::blend_heuristic_into_logits(&mut logits, &heur_scores, net_root_heuristic_blend_w());` — this is the steering knob. `score_move_with_unit_goals` is the same heuristic Gumbel already blends this way, on the same numeric scale, so no new rescaling is needed.
7. Sort `(moves, logits)` descending by the blended logit; that blended value **is** the returned score. (Verified: downstream consumers — `micro_search_pick`'s `softmax_priors`, `first_true_legal` — only care about relative order or the move object, never the score's absolute units.)
8. Apply only the type-based revival: `macro_exec::revive_endturn_for_lone_doomed_unit(scored, has_other, true, &game.state)` — **not** `revive_endturn_if_worse_than_floor`, whose `-700` floor is calibrated to `score_move + λΔφ`'s scale and was already correctly excluded from the (now-removed) net path for exactly this reason.
   - Bump `revive_endturn_for_lone_doomed_unit` (`macro_exec.rs:650`, currently private) to `pub(super)`.
   - Rename its third parameter from `lambda: f32` to `pricing_active: bool` (its only use is a `lambda == 0.0 → return early` guard at `macro_exec.rs:656` — it never scales anything). Update the two existing call sites (`macro_exec.rs`'s own `rank_plies` tail, and the new one here) to pass `lambda != 0.0` / `true` respectively. Small, self-contained cleanup; do it as part of this change rather than passing a bare `1.0`/`true` with no explanation.

**Steering-knob constant**:
```rust
const NET_ROOT_HEURISTIC_BLEND_W_DEFAULT: f32 = 0.5; // matches Gumbel's own HEURISTIC_PRIOR_W0
fn net_root_heuristic_blend_w() -> f32 {
    static W: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *W.get_or_init(|| std::env::var("POLYFISH_NET_ROOT_HEURISTIC_W").ok().and_then(|s| s.parse().ok()).unwrap_or(NET_ROOT_HEURISTIC_BLEND_W_DEFAULT))
}
```
Same idiom as `endturn_revive_price()` (`macro_exec.rs`). No iteration-based decay for now (unlike Gumbel's `decay_crutch`) — that schedule only means something inside a training loop with an `iteration` counter, and there's no retraining loop yet for whichever source Part B ends up validating. Revisit once Part B has one.

**Diagnostics**: add `NET_ROOT_CALLS`/`NET_ROOT_CANDIDATES: AtomicU64` in `net_root.rs`, mirroring `RANK_PLIES_CALLS`/`RANK_PLIES_CANDIDATES` (`macro_exec.rs:47-49`) exactly. Necessary, not cosmetic: that atomic pair's own doc comment says it feeds "the input to the ply-distillation throughput envelope" — once the net path bypasses `rank_plies` for real plies, `RANK_PLIES_CALLS` alone would silently under-count real decisions.

### A2. Wire into `MacroMctsAgent::select_move` only

New function in `net_root.rs`:
```rust
pub fn rank_view_net_or_cpu(
    view: &mut Game, pov: PlayerId, goal: &MacroGoal,
    lane_state: &mut LaneState, counters: &mut TurnCounters, lambda: f32,
    unit_goals: Option<&UnitGoalStore>, eco_plan: Option<&EcoPlanCommit>,
) -> (Vec<(f32, Box<dyn Move>)>, bool) // bool = true iff net-derived
```
Body: the same three calls `rank_view` makes internally (`observe_lane_state`, `compute_goal_aux`, `tech_discipline_active` — `macro_agent.rs:255-264`), then: if `crate::ai::ply_ranker::ply_ranker().is_some()`, try `net_rank_root_candidates(...)`; on `Some(v)`, return `(v, true)`. On `None` (ranker absent, or this specific call failed — feature extraction hiccup, shape mismatch) fall through to a **verbatim** `(macro_exec::rank_plies(view, pov, goal, &aux, gate, lambda, unit_goals, eco_plan), false)`.

**Activation condition drops `lambda != 0.0`.** The old gate made sense when the net path replaced only the Δφ *term* inside `rank_plies`. This builder replaces `rank_plies` in full — there's no Δφ term to gate on, and keeping the old gate would mean `--macro-rollout-lambda 0.0` (the currently-best lever) silently disables the net path too. Activation is purely `ply_ranker().is_some()`.

**Call site change**: in `MacroMctsAgent::select_move` (`macro_mcts.rs:1449-1461`), replace the `rank_view(...)` call with `net_root::rank_view_net_or_cpu(...)`, capturing `(mut ranked, net_derived)`. Thread `net_derived` into the `micro_search_pick` call (A3).

`MacroScriptAgent`/`MacroLookaheadAgent` (`macro_agent.rs:218,570`) are untouched — they keep calling the original `rank_view` unconditionally, so their documented byte-identical-pricing guarantee holds regardless of whether a ranker is loaded.

### A2b. `run_micro_probe`'s stale-doc-comment risk (small, don't skip)

`macro_mcts.rs:209`'s `run_micro_probe` calls plain `rank_view` internally and its comment says this is "byte-for-byte the same per-ply sequence the real decision uses, for free." Once `select_move` can use the net path (A2), that claim becomes conditionally false. This probe is diagnostic-only (measures eval-cache hit-rate behavior, gated by `micro_probe_sims()`, off by default) — not real game quality — so leaving it on the CPU path is fine. **Required**: update its doc comment to note the caveat ("...when a `PlyRanker` is not loaded; with one loaded, `select_move`'s real decision uses `net_root::rank_view_net_or_cpu` instead and this probe no longer mirrors it exactly"). Do not silently leave the stale claim in place.

### A3. `micro_search_pick` simplification (`src/ai/search/micro_mcts.rs`)

Add one parameter to `micro_search_pick`'s signature (`micro_mcts.rs:382-392`): `root_already_net_ranked: bool`. Gate the existing second-forward-pass/union-widening block (lines 416-450, the `if params.net_prior_w > 0.0 { ... }`) behind `!root_already_net_ranked && params.net_prior_w > 0.0`. When `root_already_net_ranked` is true, execution falls straight through to the pre-existing `idxs = (0..heur_top).collect()` / `softmax_priors` path — no other code changes needed in that function; it degrades to exactly the pre-EXP_ELO_126 code path, which is already correct for this case (there's nothing left for a second forward pass to add once `ranked` already reflects the net+heuristic blend from A1).

Prefer this minimal gate over physically splitting `micro_search_pick` into two functions — it's provably behavior-preserving for the CPU-fallback case (unchanged code path) at much lower diff risk than extracting a shared PUCT-loop core.

**Document, don't silently absorb**: `MicroParams::net_prior_w` (line 66) becomes a no-op whenever `root_already_net_ranked` is true. Update its doc comment to say so explicitly. Leave the field itself alone — repurposing it (e.g., as an interior-node blend weight) is a separate future decision.

**Call site**: `macro_mcts.rs:1477-1487` passes `net_derived` (from A2) as the new argument.

**Test updates** (all currently exercise the CPU-widening path via directly-constructed `rank_plies` output, so all pass `false` for the new param, unchanged behavior):
- `union_widening_is_fully_gated_off_at_net_prior_w_zero` (`micro_mcts.rs:645`)
- `union_pick_maps_back_to_the_original_ranked_index` (`micro_mcts.rs:692`)
- `measures_own_emergent_depth_at_production_params` (`micro_mcts.rs:570`)

**New test**: construct a synthetic `ranked` list, call with `root_already_net_ranked: true` and `params.net_prior_w > 0.0`, assert the pick never falls outside `heur_top` and `MICRO_MCTS_UNION_WIDENED` never fires — pins that the flag actually disables the block.

**New diagnostic**: `MICRO_MCTS_NET_DERIVED_ROOTS: AtomicU64`, incremented once per call when `root_already_net_ranked` is true — cheap live confirmation during a gauge run that the net path is actually being exercised.

### A4. Macro-mcts rollout greedy-walk

Do not modify `execute_turn`/`execute_turn_recorded` (preserves `MacroLookaheadAgent::replan` and all three `belief/mod.rs` sites exactly). New sibling in `net_root.rs`:
```rust
pub fn execute_turn_net_greedy(
    game: &mut Game, player: PlayerId, goal: &MacroGoal,
    lane_state: &mut LaneState, counters: &mut TurnCounters,
    ranker: &crate::ai::ply_ranker::PlyRanker,
) -> bool
```
Loop structure mirrors `execute_turn_recorded` (`macro_exec.rs:725-783`): same `MAX_EXEC_PLIES` cap (reuse the existing `pub const MAX_EXEC_PLIES`), same per-ply `observe_lane_state`/`compute_goal_aux`/`tech_discipline_active` preamble. Replace the `rank_plies` call with `net_rank_root_candidates(game, player, goal, &aux, gate, RootRankerSource::Ply(ranker), None, None)` (rollouts pass `None, None` for `unit_goals`/`eco_plan`, matching `execute_turn_recorded`'s own existing comment about rollouts never seeing the real trajectory's `UnitGoalStore`), and take `.into_iter().next()` — the single best move, no search — instead of the CPU ranking. On a per-ply `None` (feature/forward-pass failure), fall back to one `macro_exec::rank_plies` call for that ply only and continue — a hiccup must never crash a rollout, matching the existing "never panic" philosophy this codebase already applies to net paths.

**Activation — no new `MacroParams` field, no CLI flag, no entry in the `--macro-*` flag-drift guard** (`self_play/main.rs:123-135`). `expand()`'s call site (`macro_mcts.rs:904`) becomes:
```rust
let ok = if let Some(ranker) = crate::ai::ply_ranker::ply_ranker().filter(|_| net_rollouts_enabled()) {
    net_root::execute_turn_net_greedy(&mut game, player, &goal, &mut lane_states[s], &mut counters[s], ranker)
} else {
    macro_exec::execute_turn(&mut game, player, &goal, &mut lane_states[s], &mut counters[s], params.rollout_lambda)
};
```
`net_rollouts_enabled()`: `OnceLock<bool>` reading `POLYFISH_PLY_RANKER_ROLLOUTS` (default on; `=0` disables), mirroring `endturn_hard_gate()`'s exact idiom (`macro_exec.rs:616-619`). This gives independent on/off control over "real-ply decision" (A2) vs. "rollout playout" (A4) for gauge isolation, at zero cost to `MacroParams`/CLI surface.

### A5. Remove the superseded fast path from `rank_plies`

Delete `net_rank_plies` (`macro_exec.rs:302-368`, doc comment and `NET_RANK_LOG_PRIOR_SCALE` constant included) and its call site inside `rank_plies` (`macro_exec.rs:401-424`). A2 and A4 supersede its only two production consumers. Leaving it live would mean two differently-formulated net-blend mechanisms coexist (flat-additive-uncentered vs. the new Gumbel-style logit blend) — and because it lived as a branch *inside* the shared `rank_plies`, it still carried the full blast-radius risk (MacroScriptAgent, MacroLookaheadAgent, belief propagation, `reward_lab.rs`) this rework exists to eliminate. `PlyRanker`/`ply_ranker()` themselves are kept — just consumed from `net_root.rs` now. Net effect: `macro_exec.rs` shrinks by ~90 lines, moving it in the right direction relative to the file-size convention.

**Low-priority follow-up**: `reward_lab.rs`'s doc comment claims it "mirrors the same ranking function the real search uses" — becomes conditionally stale once `select_move` can diverge onto the net path. Flag with a one-line comment update; not blocking.

### A6. Default-enablement — auto-load if present (per direction)

`ply_ranker()` (`src/ai/ply_ranker.rs:190-212`) changes from strictly requiring `POLYFISH_PLY_RANKER=<path>` to also checking for `ply_ranker.safetensors` in the working directory by default, mirroring `model.safetensors`'s own implicit-load convention (`self_play/main.rs:226-227`):
```rust
pub fn ply_ranker() -> Option<&'static PlyRanker> {
    static RANKER: std::sync::OnceLock<Option<PlyRanker>> = std::sync::OnceLock::new();
    RANKER.get_or_init(|| {
        let path = match std::env::var("POLYFISH_PLY_RANKER") {
            Ok(v) if v == "0" => return None, // explicit disable, matches POLYFISH_ENDTURN_HARD_GATE's convention
            Ok(v) => v,
            Err(_) => "ply_ranker.safetensors".to_string(),
        };
        // ... existing load_optional call, unchanged ...
    }).as_ref()
}
```
`load_optional`'s existing "file doesn't exist ⇒ `Ok(None)`, not an error" behavior (already implemented) makes this safe when no checkpoint is present at all — behavior is then identical to today.

**Required companion step, not optional**: `ply_ranker.safetensors` currently sitting in `polyfish-rs/` is yesterday's REJECTED checkpoint (29.0% win rate). Auto-load-if-present means it would activate on the very next `self_play`/`arena`/`polyfish` run the moment this change lands, silently. As part of landing this change: **move it aside** (e.g. `mv ply_ranker.safetensors ply_ranker.safetensors.rejected_20260906`) so auto-load starts inactive, and only reactivate once Part B produces a checkpoint that's actually cleared the paired-gauge bar. This is a required step of A6, not a footnote.

### A7. Verification/measurement plan

1. Before touching anything, confirm current baseline: `cargo test --lib --tests --bin self_play` (per `CLAUDE.md`'s own CI-equivalent command) — record pass/fail counts.
2. Unit tests: (a) every existing `macro_exec.rs`/`micro_mcts.rs` test must still pass with no `ply_ranker.safetensors` present and `POLYFISH_PLY_RANKER` unset (byte-identical CPU path — the load-bearing invariant); (b) new tests for `net_rank_root_candidates` — EndTurn degenerate case, gate filtering, blend-weight monotonicity (higher `w` moves ranking closer to pure heuristic order), shape-mismatch → `None`; (c) `micro_search_pick`'s three updated tests plus the one new test (A3).
3. One determinism regression test: `net_rank_root_candidates`'s top pick is identical across two runs on the same input (mirrors the existing `executor_is_deterministic` test, `macro_exec.rs:859-875`).
4. Re-run the full test suite after: `cargo test --lib --tests --bin self_play --features apple` (needs `LIBTORCH_USE_PYTORCH=1 LIBTORCH_BYPASS_VERSION_CHECK=1 DYLD_LIBRARY_PATH=<venv torch lib>`, and a scratch `CARGO_TARGET_DIR` if any training loop might be live).
5. Re-run the seed-770425 paired gauge (same production macro-mcts recipe as the EXP_ELO_131 measurement — `--search-backend macro-mcts --macro-leaf net-asym --macro-sims 64 --macro-k 6 --macro-root-prior-w 0.05 --macro-rollout-nn-w 1.0 --macro-rollout-nn-min-depth 1 --goal-channels --goal-w-tree 1 --macro-lambda 1.0`, `--base-seed 770425 --anchor-frac 1.0`, fixed Imperius/Imperius), now with a **new** arm for this rework (dedicated `PlyRanker` via `net_root`) alongside the original three (full-Δφ / `--macro-rollout-lambda 0.0` / yesterday's rejected direct-in-`rank_plies` NN arm, for reference). Expected result to confirm: throughput-positive relative to yesterday's rejected arm (removes the redundant second forward pass in `micro_search_pick`, removes `rank_plies` cost from the real-ply path unconditionally rather than only when `lambda != 0.0`). Quality is **not** expected to clear the bar with this checkpoint — that's what Part B is for. Record the result in `hypothesis_driven_improvements.md` as its own entry (proposed `EXP_ELO_132`) regardless of outcome.
6. Pre-registration discipline: write the hypothesis/method/bar for step 5 into `hypothesis_driven_improvements.md` *before* running it, per this repo's standing convention.

### Part A — small decisions made (not re-litigated, override if you disagree during review)

- Function/constant names (`net_rank_root_candidates`, `rank_view_net_or_cpu`, `execute_turn_net_greedy`, `POLYFISH_PLY_RANKER_ROLLOUTS`, `POLYFISH_NET_ROOT_HEURISTIC_W`) are working names, not sacred.
- Per-ply (not whole-rollout) CPU fallback on a mid-rollout net failure (A4) — cheaper to reason about, consistent with existing "never crash a real turn" precedent.
- `revive_endturn_for_lone_doomed_unit`'s `lambda: f32` param renamed to `pricing_active: bool` — pure hygiene, its only use was already a boolean guard.
- No CLI flags added for any of this (env-var-only, matching `POLYFISH_MICRO_MCTS_*`/`POLYFISH_ENDTURN_HARD_GATE` precedent) — keeps the `--macro-*` flag-drift guard untouched.

---

## Part B — Training-target research plan (fully specified, for a separate agent/session)

Everything below is written to hypothesis-driven-loop standard (pre-register hypothesis/method/bar in `hypothesis_driven_improvements.md` before running, record ACTUAL after) so it's ready to execute independently, in parallel with Part A. **Dependency note**: B1 needs `RootRankerSource` (Part A, A1) to exist before it can run — check with the Part-A session on progress/merge status before starting B1's implementation, or build just that one enum locally if you need to start sooner. B2/B3 have no hard Part-A dependency beyond that same enum.

Proposed ledger numbering: `EXP_ELO_133` (B1), `EXP_ELO_134` (B2), `EXP_ELO_135` (B3) — adjust to whatever's actually next in `hypothesis_driven_improvements.md` when filing.

### B1 — Cheapest first check: existing production heads, zero new training

**Dependency**: needs `RootRankerSource` (Part A, A1) to exist — it already includes a `MainNet` variant for exactly this purpose.

**CONTEXT**: an earlier pass this session ("Rung B") checked the existing `PolyZeroNet` policy heads against the Δφ-*regret* metric and found them ~2x worse than the dedicated Δφ-replica, with a specific regression on Attack-type moves. That comparison judged the heads against a target (`rank_plies`'s formula) this whole effort is now abandoning as ground truth — it says nothing about how well those heads actually play.

**HYPOTHESIS**: the existing policy heads, used as the real-per-ply `RootRankerSource` only (not for rollouts — see confound note below), will measure competitively on the seed-770425 paired win-rate gauge, because micro-mcts's bounded search is the actual decision-maker downstream, not raw fidelity to a heuristic.

**METHOD**: zero training. Add a selector (e.g. `POLYFISH_NET_ROOT_SOURCE=main_net|ply_ranker`, default `ply_ranker` for A6 back-compat) at the `MacroMctsAgent::select_move` call site only — **deliberately exclude the rollout call site (A4)** from this arm, since `ply_ranker.rs`'s own module doc explains the dedicated small net exists specifically because the main net is ~20x larger and unsuited to `rank_plies`'s call frequency; mixing it into rollouts would conflate a throughput confound with the quality question this experiment asks. Run the paired gauge with arms: full-Δφ, `lambda=0.0`, dedicated-`PlyRanker` (Part A rework), main-net-heads (real-ply only).

**BAR**: proceed only if the main-net-heads arm beats `lambda=0.0` on win rate AND beats full-Δφ on throughput — same two-sided bar as EXP_ELO_131.

**Known confound to record up front**: expect a smaller throughput win than the dedicated `PlyRanker` gave — the real per-ply commit happens every ply regardless, so the ~20x-larger forward-pass cost is fully paid here. It's plausible this arm clears quality but fails throughput; that's still a real, useful, recordable result (it would mean the *target* is right and only the *inference cost* needs solving, e.g. by later distilling the main net's demonstrated behavior into the small dedicated net).

### B2 — New training label: real committed picks (only if B1 is insufficient)

**HYPOTHESIS**: training a ranker on "which move was actually chosen at this ply, after the full pipeline (root-candidate builder → micro-mcts's bounded PUCT search) resolved it, among the other candidates it competed against" beats the rejected EXP_ELO_131 checkpoint on the paired gauge, because it's no longer capped by `rank_plies`'s formula as ground truth.

**METHOD — new instrumentation**: mirror `POLYFISH_DPHI_PROBE`'s shape exactly (`macro_exec.rs:60-173` — sampled 1-in-N, capped total rows, one persistent buffered writer) but hook it at `MacroMctsAgent::select_move`'s post-`micro_search_pick` point (`macro_mcts.rs:1488-1496`, right after `ranked.swap(0, idx)`), since that's the smartest available decision — heuristic/net ranking topped by a bounded search. Capture per sampled real ply: turn, player, goal (for feature painting), the full candidate list as fed into `micro_search_pick` with each move's decomposed coordinates (reuse `DecomposedMapper::move_to_targets`, the same encoding `dphi_probe_row` already uses), and `picked_orig_idx`.

**Required signature change to get a soft label, not just a hard one**: `micro_search_pick` currently returns only `(Option<usize>, Option<MicroTreeCarry>, Option<f32>)` — the winner and its Q, not the per-child visit-count vector. Extend the return (or add a companion out-parameter) to also expose `root.children`'s visit counts (already computed, just discarded after the winner is picked, `micro_mcts.rs:512-520`), so the probe can build a **visit-weighted** listwise label instead of one-hot — falls back to one-hot naturally whenever `micro_mcts_params()` is `None` (micro-mcts disabled).

**Loss**: same listwise softmax cross-entropy shape as `train_ply_ranker.py` used — only the label source changes (visit-weighted committed-pick target instead of the reconstructed `score_move + λΔφ` label).

**BAR**: same two-sided paired gauge, run against both Part A's dedicated-`PlyRanker` arm and the original rejected checkpoint, to isolate "did the *target* change help" from "did the *architecture rework alone* help."

### B3 — Named gap: rollout turns have no supervision under B1/B2

Neither B1 nor B2 ever observes a rollout branch (A4's rollouts are a flat greedy walk, no search inside them) — a net trained purely on real-ply labels has no direct supervision for that specific use case.

**(a) Measure the gap first (cheap, do this before (b))**: use `POLYFISH_PLY_RANKER_ROLLOUTS` (A4's kill switch) to run a gauge variant isolating rollout quality specifically — real-ply commit forced onto whichever arm wins B1/B2, rollouts toggled net-greedy vs. CPU-Δφ-greedy independently. Attributes any win-rate delta specifically to rollout quality at effectively zero implementation cost (the switch already exists from Part A).

**(b) Only if (a) shows a real gap**: extend `execute_turn_recorded`'s existing `rec: Option<&mut Vec<PlyRec>>` pattern (`macro_exec.rs:705-785`, which already runs two extra `rank_plies` calls per ply for its `no_phi`/`no_goal` diagnostic arms when recording is on) with a new recording arm that additionally runs a bounded search (reuse `micro_search_pick`, or a cheaper variant) at sampled rollout plies purely to harvest richer labels — a one-time offline harvest cost, not a live inference cost; the live rollout mechanism (A4) stays the flat single greedy call regardless.

### B4 — Bar and discipline (every sub-experiment, no exceptions)

Pre-register hypothesis/method/bar in `hypothesis_driven_improvements.md` before running each of B1/B2/B3(b). Two-sided bar every time: paired win rate vs. full-Δφ AND vs. `lambda=0.0`, throughput vs. both, seed 770425 (or explicitly note and justify any deviation, the way EXP_ELO_131's own entry notes its n≈31 sample-size caveat).

### B5 — Sequencing recommendation

1. Part A lands first (B1 needs its `RootRankerSource` plumbing) — coordinate with whoever's running Part A rather than assuming it's done.
2. B1 next — zero training cost, could resolve the whole quality question in an afternoon, and directly re-tests Rung B's finding against the right yardstick.
3. B3(a) can run in parallel with B1 (independent, cheap, uses the same kill switch) — have the rollout-gap size in hand regardless of B1's outcome.
4. B2 only if B1 fails — multi-day (new instrumentation, harvest, retrain), and its value depends on B1 having actually failed the *right* bar, not merely underperformed the dedicated `PlyRanker` on regret (already known going in).
5. B3(b) only after B2 is underway.

### Part B — open bookkeeping (decide when each item is actually greenlit, not now)

- Exact `EXP_ELO_1XX` numbers (proposed 133/134/135 above) — check `hypothesis_driven_improvements.md` for the actual next-free number before filing, in case Part A's own gauge re-run (EXP_ELO_132) has landed first.
- Whether B1's `POLYFISH_NET_ROOT_SOURCE` selector becomes a permanent knob or a throwaway removed after B1 concludes.
- Whether B2's visit-count exposure is a breaking signature change to `micro_search_pick` or an additive out-parameter.

---

## Verification summary (Part A)

- `cargo test --lib --tests --bin self_play` clean before and after, on a scratch `CARGO_TARGET_DIR` if any training loop might be live (see project convention on shared target-dir clobbering).
- New/updated unit tests per A1/A3 above.
- Seed-770425 paired gauge re-run with the new arm, logged to `hypothesis_driven_improvements.md` as `EXP_ELO_132` regardless of outcome, pre-registered before running.
- Confirm `ply_ranker.safetensors` (the rejected checkpoint) has been moved aside as part of landing A6, before merging — auto-load-if-present must not go live pointed at a known-bad model.
