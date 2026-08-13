# Current Understanding

**The single source of *current* truth for how the Polyfish AI plays, where it's weak, and why.**
Last synthesized: **Aug 12, 2026** (through EXP_ELO_032 — the macro-search bootstrap GO result — on top of the EXP_ELO_028 Phase 1c goal-conditioned macro stack, EXP_ELO_026's causal result and the EXP_ELO_M1 measurement audit).

> **The one rule that keeps this doc useful:** current truth lives *here*. The research logs
> (`notes.md`, `hypothesis_driven_improvements.md`) are append-only audit trails — they contain
> earlier interpretations that were later corrected, written in the present tense right next to
> their corrections. **Do not cite a claim from those logs as current** unless it also appears here.
> When a finding changes, update this file first.

---

## How to read the docs

| Doc | Role | Trust for "what's true now?" |
|---|---|---|
| **`current_understanding.md`** (this file) | Single source of current truth. Read first. | ✅ Authoritative |
| `hypothesis_driven_improvements.md` | Experiment ledger (audit trail). One current Verdict per EXP. | ⚠️ Only the final Verdict of each entry |
| `notes.md` | Chronological research journal. | ⚠️ Historical — lines tagged `(SUPERSEDED)` are dead |
| `notes-heuristics.md` | Evaluator/heuristic design spec (`[IMPLEMENTED]/[TODO]` checklist). | ✅ For design *intent*, not current strength |
| `expert_boost_throughput.md` | Self-play throughput tuning (referenced by code). | ✅ For its narrow topic |

---

## Current understanding

### ⚠️ Read this first: the measurement floor

**Most of the instruments are coarser than the effects being chased.** Established Jul 27 (EXP_ELO_M1). Until this is fixed, a chart moving is not evidence.

- **Do NOT read `model − anchor` difference curves** (the dashboard's model-vs-Greedy behavior charts). At `ANCHOR_FRAC 0.25` the model gets 112 seats per 64-game iteration and Greedy gets 16, decaying to **65 vs 7.6 alive by t25**. **93–99% of the difference curve's iteration-to-iteration variance is opponent-side sampling noise** (t25: cities 97.8%, army_stars 99.1%, spt 97.3%). The anchor wobble's implied per-game sd lands exactly on the Poisson prediction — **there is no residual left for real drift.** Meanwhile the model's own curve is *very* stable (t25 cities sd **0.081** across 13 iterations).
  - **Fix:** Greedy is a fixed policy — pool its curve once across iterations/runs into a static reference line and plot the model curve alone. Do not add error bands (Verdi, Jul 26). Raising `ANCHOR_FRAC` is the alternative and costs mirror data.
- **The 64-game gauge is a ±12pp instrument.** The last 8 readings vs Greedy (43.75 … 51.56 … 40.62) have mean 43.16% and observed sd **4.94pp — *below* the 6.19pp of binomial sampling alone.** Not one recent run has separated from the pack. Resolving 6pp at 2σ needs **~555 games/side**.
- **Consequence:** every "the model finally passed Greedy for several iterations" and every plateau/regression call made from these charts between **Jul 22 and Jul 26 is unsupported** — including reads on unfreeze-opponent, `value_trust`, and λ. Conversely, the model's behavior has been *remarkably stable* across all of those changes, which is itself the evidence that none of them touched the binding constraint.
- **Metric discontinuity (Jul 25):** `villages_first_rate` / `villages_t2c_first*` were re-based from *per game* to *per net seat* in `self_play.rs`. Rows before run `1785087189` are not comparable to rows after. Validated against production: the old 0.9243 implies a per-seat 0.8079, the new metric measures 0.7932 — **the definitions agree**; the drop is the 2-seat OR being removed. Same discontinuity class as the Jul 24 `avg_score` change (mean-winner → mean-net-seat).
- **Metric discontinuity (Jul 27) — the action family, `max_score`, and T2C.** Same re-basing, finally applied to the metrics the Jul 25 fix missed:
  - **`avg_captures`, `avg_cap_*`, `avg_harvests`, `avg_builds`, `avg_research`, `avg_attacks`, `avg_abilities`, `avg_moves`, `avg_revealed_tiles`, `avg_captured_tiles` moved from ÷`num_games` to ÷net player-games.** Their numerators were always net-seat-only (the `net_move` gate), so a mirror game contributed two seats' worth against an anchor game's one — and since `anchor_frac` decays over a run, **the whole family drifted downward as training progressed, with no behavioral cause.** Expect a one-time step down (~35% at `anchor_frac 0.5`) at the switchover. Pre-Jul-27 rows are not comparable to later ones.
  - **`max_score` is now the best *net seat* score**, not `winner_score` — an anchor or league opponent winning used to set the line (and got its replay saved as the "best game"). Same population as `avg_score` now.
  - **T2C split into speed and reliability.** `villages_t2c_first` charged `max_turns` to seats that never captured, so "slow" and "never" were one number. The dashboard now plots `villages_t2c_first_cond` (mean turn among seats that *did* capture) and `villages_first_rate` separately. The censored column is still emitted, just not charted.
  - **New: `avg_kills`** (+ per-role `_totals.kills` and a per-turn kills curve in `tempo_by_turn`), read from the engine's `TribeState::kills`. Verified exact: over 24 games, total kills == total unit losses, and per-role attribution balances (anchor kills == `model_vs_anchor` losses, and vice versa). Conversions are not kills. Historical rows emit `null` and render as a gap.
  - `/metrics` now returns **`null` rather than `0.0`** for a blank/absent CSV cell, so rows predating a column show a gap instead of a fake zero reading.

### 🆕 Macro-level search works: the EXP_ELO_032 bootstrap (Aug 12 — GO on the hierarchical redesign)

The planned redesign — **MCTS over per-turn strategic directives with a deterministic executor for the plies** — cleared its gate experiment. Two new inference-only arena backends (`macro-script` = scripted `MacroGoal` + goal-conditioned whole-turn executor in `src/ai/macro_exec.rs`/`macro_agent.rs`; `macro-lookahead` = one turn-level lookahead step over K=4 candidate directives, H=2 own turns per rollout on FOW-honest `clone_for_mcts` views, ghost-Greedy opponent). At 2500 games/arm (base 1786500000):

- **Lookahead beats its own executor 55.2% (z=+5.2)** — replicated at 55.1% with net-value leaves. One shallow directive-selection step is worth +5.2pp; this is the existence proof for macro search.
- **Leaf choice is a wash (heuristic 55.2 vs net 55.1)** — the net value head adds nothing over `evaluate_state` as a rollout leaf at H=2 (consistent with its over-confidence). Do not assume net leaves in the redesign.
- **Stage0 (script+executor, no lookahead) is Greedy-parity in wins (50.9%) despite reaching 3 cities 74.6% vs 62.4%** — execution without directive *selection* converts a +12pp reach edge into nothing; selection is where wins come from.
- **Lookahead beats production Gumbel n=64 head-to-head: 62.5% over 2500 games (z=+12.5) at 16 vs 173 ms/move** — the registered "exploits-the-script's-determinism" risk tell came back negative (the strength transfers to a completely different opponent). The NN-free lookahead agent is plausibly the strongest agent in the repo at ~1/10 production compute.
- ⚠️ **The engine is not run-to-run deterministic**: identical NN-free arms flip the winner in **16% of games** (movegen-order ties suspected). Seed-paired cross-run McNemar is therefore invalid; use within-arm head-to-head z (flip noise is inside its binomial variance). This retroactively weakens any cross-run paired read, including parts of the 026 protocol.

**Stage 2 landed the same day (EXP_ELO_033, CONFIRMED): the adversarial turn-level MCTS (`src/ai/macro_mcts.rs`, backend `macro-mcts`, UCT c=0.05 dialed to the measured directive-q spread, negamax over the antisymmetry-tested heuristic, opponent turns searched not scripted) beats the Stage-1 lookahead 58.9% (z=+8.9) and production Gumbel n=64 66.0% (z=+15.2) at sims=32, 95 ms/move.** The strength ladder compounds: script (Greedy-parity) < one-ply lookahead < adversarial tree — all NN-free. **Macro depth saturates by ~32 sims at k=4 (EXP_ELO_033b: sims=64 read 48.4%, flat, at 2× cost — a registered, useful null).** The levers are now candidate-set richness, leaf quality, and belief-state fidelity — not sims. Next: Stage 3 = AlphaZero-ify (macro policy head on tree visits, value at turn boundaries; heuristic leaf stays until beaten in a paired A/B), and/or a richer directive generator. Ledger: EXP_ELO_032, EXP_ELO_033, EXP_ELO_033b.

**Belief state over hidden information — design locked Aug 13, build not started (EXP_ELO_034 candidate).** The macro tree's known bias is *optimism*: `obscure_fog` deletes unseen enemy units/cities, so every rollout simulates an opponent weaker than reality. The fix is a persistent per-seat `Belief` maintained from legal observables only, materialized into the fogged clone before each plan so the simulated opponent has approximately the right strength. Decisions:

- **Score counting runs on an event stream of per-action deltas, not turn totals.** A turn-boundary jump (+175) is ambiguous; the per-move sequence (+10, +25, +100…) is the clue, exactly as human score-counters use it. The engine applies opponent moves one at a time, so a hook after each true move application records the observer's delta log. `calculate_detailed_tribe_score` (functions.rs:1031) is deterministic and invertible: units cost×5, tech 100×tier, capture +100 base +20/territory tile, exploration +5/tile — subtract witnessed events, the residual constrains hidden production. Score is public info in the real game (leaderboard), so the side-channel is FOW-honest.
- **Every guessed item carries a tanh-bounded confidence** (staleness decays it, corroborating observations sharpen it). Confidence gates materialization and weights risk discounting at leaves; at Stage 3 it is a candidate feature plane alongside the existing ghost channels.
- Other raw material already in the repo: capital-location prior from mapgen (uniform placement + min-capital-distance constraint), `enemy_ghosts` survive `obscure_fog` (states.rs:717 strips only the opponent's dossier), `_prediction` terrain guesses for unexplored tiles.
- **Calibration before search**: the arena holds ground truth, so belief accuracy is measured offline first (Brier on capital cell, error on hidden army value per turn) — only a calibrated belief enters the materialization A/B (belief-fed opponent vs today's empty-fog opponent).
- **Search skew = root determinization ensemble (spec'd Aug 13).** The belief proposes the top-M *joint* worlds (capital hypothesis + unit placements sampled together, never a cross-product); M is adaptive — smallest M whose cumulative posterior mass clears a coverage target (top confidence ≥0.95 → M=1), hard-capped by a sims-effort knob. One macro tree runs per world (independent → parallelizable; the 033b saturation null frees exactly this budget axis — extra sims in ONE world were worthless, extra WORLDS are the untested dimension). Directive committed by confidence-weighted value across worlds, Q(d)=Σ_m w_m·Q_m(d). This is PIMC/IS-MCTS-style determinization applied at the directive level, where classic strategy fusion mostly evaporates (the root action set is identical and semantically stable across worlds). After each real enemy turn, worlds update particle-filter style: reweight against observations (score-delta log, fog-edge appearances, attacks received), kill inconsistent worlds, resample from the posterior. Staged: EXP_ELO_034 belief+calibration → 035 MAP materialization (M=1) vs empty fog → 036 ensemble (M>1 + effort knob) vs M=1; each stage must beat the previous or the next doesn't run.

### The goal-conditioned macro stack (EXP_ELO_028 Phase 1c, live since Jul 29 — current production default)

The model now trains and gauges under a scripted macro layer: **goal channels** (EXPAND/ATTACK/DEFEND order planes + GROW/ARM/UNLOCK stance, painted into the input), **goal-priced in-tree shaping** (GROW pays 150/SPT, ARM 50/army-star, EXPAND 200/tile of approach progress with achieved-holds-cap semantics, + environment-fit tech bonus and a path-aware Rider bonus), a **granular stance-aware research gate** (GROW gates combat-unit tech, ARM gates pure-eco tech, 5-star reserve), and **tech discipline caps** (≤8 own-star techs/game, ≤1 tier-3). `GOAL_CHANNELS=1` enables all of it; `tech_tree.json` + `settings/technology.rs::get_tech_effects` is the tech-annotation lookup it runs on.

- **The core finding (10 iterations, run 1785279937/15–24): pricing the goal in the search objective steers behavior durably, where hard masks alone were re-optimized around within 3 iterations** (the 026 corollary, now demonstrated both ways). Same-pairing consecutive A/B (iter 14→15, XinXi+Oumaji): research 10.7→7.1, SPT_t15 9.1→**12.6**, units 13→**18.6** — and the shift held through iter 24 with no decay.
- **Doctrine flipped TECH-modal → ECO-modal** (the Phase-0 target delta): research allocation is now stance-conditional (GROW:ARM research-mass ratio went 1.5× → 6–15×), tech mix is eco-dominant (Farming presence 62%→92%), and **the net leads Greedy on SPT in contested games** (15–18.7 vs 6.6–10.6 at t15) — it had never led on SPT before.
- **Strength: embedded-anchor win rate pooled ~71% (74/104) vs 61% pre-stack; the deterministic gauge reads 56→59% (elo 566), still below its 62.5%/589 peak — ladder confirmation pending.** `avg_score` is no longer comparable to pre-stack rows (8 fewer techs ≈ −900 score mechanically; the model plays better while scoring less).
- ⚠️ **The behavior is currently RENTED from search-time shaping, not owned by the net**: policy_loss rose 1.39→1.53 across the run (value_loss at record-low 0.22) — the prior is still chasing the crutch-shifted targets. Stage-2 (macro heads + crutch decay) must wait until it closes. Known gaps: prepare-ARM never became summon-conditional (units rose globally instead; the ARM response expresses in Step/Build mass), ATTACK orders still lit on 30% of plies (target <25%), and the 3rd city runs ~1 turn later on Kickoo pairings (eco-first opening trade — first suspect if the gauge stalls, since third-city reach is causal).

### The bottleneck and the current lever

- **Bottleneck metric = the third-city *reach rate*** (not pace). Confirmed and sharpened by the 352-game autopsy above: reaching 3 cities is worth **74.1% vs 16.4%**, the model gets there in only 55% of games, and *when* it gets there is the same in wins and losses. ⚠️ **The older "out-raced, not out-fought / attrition ≈ 0, the model's units survive" reading is retracted** — in games it loses the model is militarily erased (units peak 5.77 → 0.85 final, cities 2.06 → 0.50). It is out-raced *and* then out-fought. (ledger §6, superseded by EXP_ELO_M2)
- **The third-city link is CAUSAL, and the mechanism is star allocation (EXP_ELO_026, Jul 28).** An inference-only "oracle macro" A/B (250 paired games/arm, deterministic n=64, same seeds both arms): a **star gate** — drop root tech purchases that would leave <5 stars while a capturable village is FOW-visible and <3 cities — raised reach **+7.6pp alone** (McNemar z=+2.32; +10.0pp z=+3.20 with the pursuit-focus rule added), bent the research-inelasticity signature exactly as predicted (techs-per-city@t10 5.14 → 4.17), and cut *Greedy's* reach −9.6pp (the model wins races it used to lose). The causal read: in the 43 paired games where the macro flipped reach on, wins went **27.9% → 81.4%**; in the 18 it broke, **78% → 11%**; conditional win rates are identical across arms (win|reach ≈ 75–76%) — reach converts at the full margin in both directions, killing the selection-effect alternative. The **commitment/pursuit-focus rule alone is inert-to-negative** (+1.2pp reach, −4.4pp win — focusing `CH_PURSUIT` hides non-target villages), reconfirming that pursuit *representation* is not the constraint. Overall win% moved only +2.0pp because the crude always-on gate also breaks ~18–24 games baseline handled; the **gross flip effect ≈ +9pp win at this budget is the headroom for a selective/learned macro**. Baseline reach here is 64.8% (deterministic n=64), vs M2's 55% (noisy n=256) — condition difference, not a contradiction.
- **The current lever is search depth / throughput (EXP_ELO_023, Jul 25).** Search *does* beat the prior: prior-only argmax **36.7%** vs normal search **45.3%** = **+8.6pp (2.4σ)**. Depth grows ~`sims^0.5`. Polytopia needs ~8 plies to finish **one** game turn, so **production self-play at n=64 (~4 plies) is depth-starved — it cannot see a single turn ahead.**
  - **Ladder, pooled across both Jul 25 sweeps** (supersedes the single-sweep rungs): n=64 **44.3%** (255/576, depth ~4), n=256 **53.1%** (204/384, depth ~9), n=1024 51.3% (82/160, depth ~19), n=2048 58.0% (65/112, depth ~26). **The one statistically established step is 64→256: +8.9pp, 2.70σ, p≈0.007.** 256→2048 is only 0.9σ — real but unproven. Do **not** state the ladder as "+5pp per 4× forever", and do **not** state it saturates past 256; both readings were made and retracted the same day.
  - **Recommended budgets: generate/train at n=256** (proven gain, only **2.7× per game** — 1.4s vs 0.4s — and the first rung past the one-turn threshold); **deploy at n=64** (166 ms/move ≈ 3.3s for a 20-action turn). n=2048 is 3428 ms/move ≈ 68s/turn — training-only at best.
  - **Distillation headroom is large and measurable.** New `PRIOR OVERRIDE` metric (root decisions where search's pick ≠ `argmax(prior)`; `GumbelMctsAgent::agree_count/decision_count`, printed by arena): **24.2% (n=64) → 24.1% (256) → 29.4% (1024) → 34.0% (2048)**. It *rises with depth* — deep search finds moves the prior does not rank first — and notably only **starts** rising past ~9 plies, i.e. once the tree clears one full game turn. Gap from bare prior to n=256 search ≈ **16pp**. Caveat: the metric counts decisions, not importance; some share are near-ties.
  - **Do NOT swap Gumbel→PUCT.** At fixed n=64, concentrating (k=4, k=2) is *worse* than k=16 despite deeper PVs — root breadth beats principal-variation depth. Gumbel's anti-concentration is a feature.
  - **Fix the horizon ceiling before pushing sims:** `brain.rs:435` hardcodes `max_turns_ahead = 20 − current_turn` while games run to turn 30, so from ~turn 18 the horizon pins to its 2-turn floor (horizon-capped descents hit 9.8% at n=1024).

### ⭐ Why games are won and lost: the third city (352-game autopsy, EXP_ELO_M2)

**One binary explains almost the whole result.** From 352 arena games vs Greedy at n=256 (Imperius v Imperius — arena hardcodes both tribes, so **no tribe confound**), split by outcome.

> ⚠️ **Measured at the default `GUMBEL_SCALE=1.0` = "normal self-play exploration"** (`gumbel_mcts.rs:260`; the loop never sets it). So these are arena games played with *training-style root noise*, not maximum-strength play — part of the ≤2-city population may be self-inflicted exploration noise. **The same caveat applies to every reading in `ladder.json`.** Deterministic (`GUMBEL_SCALE=0`) replication pending.

| condition | n | model win% |
|---|---|---|
| **model reached 3+ cities** | 193 | **74.1%** |
| **model stalled at ≤2 cities** | 159 | **16.4%** |
| reached city #3 before Greedy | 143 | 73.4% |
| Greedy reached city #3 first | 194 | 27.3% |

A ~58pp swing on one variable. The model reaches 3 cities in only **55%** of games; Greedy manages it in **97.8%** of the games it wins.

- **It is NOT slower — it is binary.** Timing is nearly identical in wins and losses; only the *reach rate* differs: city #2 **95.3% @t7.84** vs **70.5% @t8.11**; city #3 **84.6% @t12.33** vs **27.3% @t11.20**; city #4 61.5% vs 7.1%. When it expands, it expands on schedule. Losses are *absent* expansion, not late expansion. **This supersedes the "out-raced / tempo gap" framing** — the deficit is a binary reach failure, not a pace deficit.
- **Bimodal, no middle.** Peak city count in losses: **29.5% never get a 2nd city, 43.2% peak at exactly 2** → 72.7% of losses top out at ≤2. In wins the mode is 4–5 cities. Peak-city *turn*: **t15.17 in wins vs t7.16 in losses** — so the "universal turn-15 collapse" seen in aggregate was two populations averaged.
- **Losses are annihilations.** In lost games the model goes from peak **2.06 cities → 0.50 final** (−75.6%) and **5.77 units → 0.85** (−85.2%): it loses its capital in most losses. Asymmetric — when *Greedy* loses it still ends with 1.34 cities. ⚠️ **This kills the old "attrition ≈ 0; the model's units survive" claim.**
- **Decided in turns 8–12.** AUC of the score gap: 0.489 @t5 → 0.635 @t8 → **0.718 @t10** → 0.829 @t12 → **0.900 @t15** → 0.966 @t20. (The model leads on score at t5 in most games it wins *and* most it loses — an early score lead is uninformative.)
- **Research is inelastic, and that is the tower's real signature.** Absolute techs at t10 is **flat** (8.59 won vs 8.77 lost, AUC 0.463) while cities differ 2.09 vs 1.62. So it buys the same tech with 1 city as with 3 → techs-per-city **4.98 vs 6.40 (t = −4.74)**, widening to 5.10 vs 8.21 by t15 (t = −7.94). The defect is **failure to redirect stars when expansion stalls**, not over-researching per se. (Caveat: the ratio's denominator is itself the top discriminator; the load-bearing fact is that *absolute* tech does not adapt.)
- **Not the map.** Each seed is played in both orientations; both give the same winner only **39.2%** of the time vs 50.1% expected from an independent coin — terrain explains ~nothing. No seat effect either (P1 51.1% vs P2 44.9%, z≈1.2; a 160-game held-out run split 40/80 vs 39/80).

**Army composition is real but NOT the margin.** `$/unit` is **2.16 in wins vs 2.08 in losses at t10 (AUC 0.536)** and 2.20 vs 2.10 at t15 — essentially zero discrimination. The model makes *more* units when winning (8.15 vs 4.25 at t15), not better ones, funded by the cities. The aggregate 2× gap vs Greedy (model ~2.2, Greedy ~4.4) is genuine and *may* be a uniform handicap — a constant cannot discriminate — but **it cannot be shown to decide anything, and much of the apparent gap is Greedy's own $/unit sagging when it loses (3.19 vs 4.42).** ⚠️ An earlier revision of this file called it "the one behavioral gap that survives" and the place to aim structural change; that over-claimed. Priority belongs to third-city reach.

### The value head

- **Mis-calibrated, not blind — and not the binding lever.** It is a *worse* overall outcome predictor than the raw score ratio late-game (where score≈outcome), but **beats the scoreboard early-game** (real foresight where score is blindest). Its defect is **over-confidence when ahead** (~2×: predicts +0.6/+0.9 where actual is +0.28/+0.53). (EXP_ELO_021, memory `value-head-calibration-diagnostic`)
- **The over-confidence is robust to target re-weighting** — EXP_ELO_021 (de-saturate the outcome label), EXP_ELO_022 (down-weight the biased TD arm) and EXP_ELO_024 (widen the TD window, λ 0.8→0.875) were **all REJECTED**: they moved the target but not the head's outcome-*discrimination* (corr(raw,outcome) ≈ **0.40**, unchanged) and not behavior. It's not a cheap target-design bug. **The value-target parameterization is a closed family — three independent re-weightings, three nulls. Stop re-weighting the value target.**
- **λ specifically is not a lever at this game length** (EXP_ELO_024): every model-side behavior metric came back |t| < 1.2 while `value_loss` rose 15.6% (t = +6.31). In a 30-turn game λ=0.8's `λ^n` tail already reaches terminal, so widening the window buys label variance, not horizon. `--td-lambda` is shipped as a knob; leave it at 0.8.
- The value head does **not** global-average-pool the trunk (a claim repeated in earlier notes): `network.rs:315` applies an 8-channel `v_pool_conv` then `flatten_from(1)`, so it sees the full 8×11×11 map. **Do not motivate an architecture change on value-head spatial capacity.**
- At a **pursuit fork** the value head is **indifferent, not opposed** (own_value(toward) − own_value(away) ≈ 0), because step-toward and step-home produce near-identical boards. (memory `no-mirror-excuse-for-tempo`)

### The policy prior

- **Healthy, not suppressed.** The toward-village prior rose **0.14 → 0.43** under training. Earlier "Build/Harvest suppressed 2–4 orders of magnitude" and "won't seek" reads were **same-ply popularity-contest artifacts** — one ply bundles ~25 competing Step candidates, so a per-candidate prior comparison measures the wrong thing. (ledger §5, notes.md third-city arc, memory `wander-search-rejection-value-blind`)
- **"Wandering" with no village in view is mostly benign.** In eval (Gumbel off) the net picks **0.80** frontier-ward (1.0 = most frontier-ward), top-2 direction 72%. The scary "41%" was a strict argmax metric plus Gumbel self-play exploration jitter.

### Pursuit and representation

- **Pursuit progress was a search-time reward potential, never a network input feature** — that gap is now closed (`CH_PURSUIT`, Jul 2026). ⚠️ The original diagnosis also claimed the value head global-average-pools the trunk; **that part was wrong** (see above) and is not a reason the head was pursuit-blind. (memory `pursuit-failure-representation-gap`)
- **The representation fix (CH_PURSUIT channel + `aux_pursuit` head, Jul 23) is a mechanistic success with NO strength gain.** The fork value gap opened from **3.2e-5 → ~+0.03** and the prior flipped toward the village, but arena stayed 50/50 vs Greedy. **Pursuit-perception was NOT the binding constraint. Do not re-attempt expecting strength.**
- Wasted-pursuer-turn rate is **~50–60%** (identity-tracked whole-turn metric), **but whether that is a defect is unresolved** — in mirror self-play the opponent doesn't contest uncontested villages, so there is no urgency to "rush."

### Reward, labels, and score-pricing

- **Reward shaping is on by default** (`--no-reward-shaping` opts out); EXP_ELO_004 showed dense shaped labels beat sparse. **The label family — delivery, content, in-tree shaping — is fully eliminated as the tower's driver** (EXP_ELO_008 → 011-EXT closed it; recalibrated by EXP_ELO_S0).
- **Score-pricing (ledger §7):** the TD label prices **tech ≈ 15–25 pts/star (instant, riskless, unclawbackable)** vs **army units 5 pts/star (clawed back on death)** — so tech-towering is ≈ the greedy-optimal policy under the label. **This is tech vs army *units*, NOT tech vs village *capture*.** A village capture (+0.42 normalized) is worth **~2.5× a tier-1 tech** (+0.17) even after travel discount — capture is **not underpriced**; its problem is that the payoff is **contingent on winning a race and holding the city** (achieved only ~half the time).
- **Why tempo isn't learned — the four mechanisms (NOT mirror-cancellation):**
  1. **Credit assignment / horizon** — the loss arrives 25+ turns later, discounted ×0.9^Δ ≈ 0.07, smeared over every ply.
  2. **V(state), not Q(state,action)** — step-toward vs step-home → near-identical boards → value can't distinguish them (own_value gap ~0).
  3. **Frozen-opponent search** — the in-tree MCTS auto-skips opponents, so it cannot represent the race.
  4. **Instant-unconditional (tech) vs contingent-multi-ply (capture)** reward realization.
  - **Explicitly NOT** "mirror self-play makes the relative reward net to ~0" — rejected forcefully by Verdi (Jul 22, 2026); do not resurrect it. **Supporting evidence, correctly scoped:** the EXP_ELO_007–012 fine-tuning campaign ran at `ANCHOR_FRAC=1.0` (every game NN-vs-Greedy) and the tempo deficit persisted anyway — so mirror symmetry is not the driver. ⚠️ **Earlier revisions of this line claimed "training uses `ANCHOR_FRAC=1.0`" as a general fact. That was wrong** — 1.0 was an env override specific to that campaign. See the data diet below for what training actually does. (memory `no-mirror-excuse-for-tempo`)

### The data diet (what training actually runs)

- **Default `ANCHOR_FRAC = 0.25`** (`run_training_loop.sh:364`) → per 64-game iteration: **16 anchor games (net vs Greedy) + 48 mirror self-play (net vs net)**. So normal training is **75% mirror**, not majority-anchor.
- Anchor games are **evenly spread** (every 4th game, `self_play.rs:2542`), and the **anchor seat alternates** by anchor ordinal so the net plays both seats vs Greedy.
- The anchor opponent is **`Greedy`** (zero-search `score_move` argmax), *not* the rollout Heuristic MCTS — measured: first-village capture 1.00/t6.5 vs 0.94/t8.9 (rollout noise drowned the ordering gradient). It is also the same distribution `blend_heuristic_prior` injects into the net's root, so anchor data and search priors agree.
- **`anchor_frac` does not decay until the model graduates.** `decay_crutch` uses exponent `iteration − anchor_decay_start`, and `.anchor_decay_start` is only written once the model clears ≥80% vs its active anchor (EXP_ELO_002: "the decay clock belongs to the model it graduated"). Absent that file the loop sets `decay_start = eff_iter` each iteration → exponent 0 → the anchor rate stays at its starting value for the whole run. Constants: `ANCHOR_FRAC_DECAY = 0.97`, `CRUTCH_FLOOR = 0.1`.

### Aux heads

- **Causally disconnected from the decision the search consumes.** MCTS backs up `win_value` only; the Rust backends define no aux heads; zeroing `aux_ownership` leaves the value output **bit-identical** across 256 states. Aux heads shape only the shared *trunk* during training. Changing/decomposing them changes nothing unless wired into `v_win` (a dual-network change + retrain). (ledger §2)

### The plateau, localized

- Across Jul 23–25 the real plateau resolved to: the model only **matches a zero-search heuristic** (50/50 vs Greedy at n=64), and the binding lever is **search depth / throughput** (EXP_ELO_023) — not the value target (021/022/024 rejected), not the prior (healthy), not pursuit representation (no strength gain), not the reward labels (family closed).
- **Jul 27 amendment:** the plateau is *also partly an instrument artifact.* The gauge has sat at 43.2% ± 4.9pp for 8 runs with a per-reading binomial sd of 6.2pp — statistically a **constant**. So "20+ flat experiments" is partly "20+ experiments measured with a ±12pp ruler." Before concluding anything further about levers, either (a) raise the gauge to ≥384 games/reading, or (b) judge changes on the model's own behavior curves (sd 0.08 on t25 cities — 8× better resolved than the gauge). The **army-composition gap** above is currently the only behavioral target that the existing instruments can actually track.

---

## Retracted claims — do NOT cite these as current

You will encounter these in `notes.md` / the ledger stated in the present tense. Every one is dead.

| Claim you may still see | Status | Current truth | Where it died |
|---|---|---|---|
| Value head is "blind" / "barely learning" / "it sucks, THE thing dragging us down" | ❌ Retracted | Mis-calibrated, not blind; contributes +8.6pp to search; not the binding lever | notes.md Jul 7–8; EXP_ELO_021–023 |
| "Value-target miscalibration is the tower" / "the label is empty" | ❌ Retracted | Value head *correctly* devalues late tech; target re-weighting rejected (021/022) | ledger §3; EXP_ELO_009–011-EXT, S0 |
| "57% vs Greedy — best model on record" | ❌ Retracted | Winner's curse; true strength **45.9%** [41.6–50.2]; the peak never existed | EXP_ELO_S0 |
| "Model wanders / is lost ~41% of the time" | ❌ Retracted | Strict-argmax + Gumbel artifact; eval is 0.80 frontier-ward, mostly benign | memory `wander-…` |
| "41.8% of plies" **and** "81.4% of turns" pursuit is wasted | ❌ Both retracted | Identity-tracked ~50–60%; and whether it's a defect at all is unresolved | notes.md ~L1081; EXP_ELO_018 |
| "Model over-researches / prefers tech to economy" | ❌ Retracted | Economy gets **more** stars (47.3% vs 38.3%); deficit is expansion *pace*. A mild tech lean survives | notes.md ~L946/983 |
| "Score favors tech over village capture" | ❌ Retracted | Capture +0.42 > tier-1 tech +0.17 (~2.5×). The §7 pricing is tech vs *units* | memory `no-mirror-…` Jul 22 |
| "Search is 2–3 plies deep / within-turn sequencing only" | ⚠️ Corrected | Mean ~4 plies at n=64 (grows ~sims^0.5), crosses ~5 of the searcher's *own* future turns; "within-turn only" wrong, "zero *adversarial* content" right (opponent frozen) | ledger §1; EXP_ELO_023 |
| "MCTS + prior is no better than the prior / the plateau is a dead loop" | ❌ Retracted | Search beats the prior **+8.6pp (2.4σ)** | EXP_ELO_023 |
| "Just raising MCTS budget won't help" | ⚠️ Nuanced | True for the *narrow pursuit-wander* micro-behavior (value indifferent at the fork); but overall strength **is** monotonic in depth (n=1024 beats Greedy) — n=64 is depth-starved | memory `no-mirror-…` (Jul 22) vs EXP_ELO_023 (Jul 25) |
| "Pursuit is proposal-side suppressed (prior crushed 2–4 OOM)" / "valuation-side confirmed" | ❌ Retracted | Same-ply popularity-contest artifact; prior is healthy (→0.43); value is *indifferent*, not opposed | notes.md ~L783/810; ledger §5 |
| "Production collapse (Summon=0.000)" / "chose not-to-seek ~70%" / "loses the fight" | ❌ Retracted | Army-ply & single-ply parser artifacts; produces in ~80% of pursuits, pursues 86–98%; loses the **race**, not the fight | ledger §4/§5/§6 |
| "Fixing pursuit representation will raise strength" | ❌ Retracted | Implemented → mechanistic success, no strength gain; not the binding constraint | memory `pursuit-…` Jul 23 |
| "~100 ms per CPU NN eval" | ❌ Retracted | That was candle-Metal at batch-128, not CPU; a single CPU eval is **~2.65 ms** (~40× off) | notes.md ~L1183 |
| "Aux heads make the net understand the game (help the decision)" | ❌ Retracted | Causally disconnected from the value/policy the search consumes | ledger §2 |
| "The value head global-average-pools the trunk" (hence pursuit-blind / capacity-limited) | ❌ Retracted | `network.rs:315` uses an 8-channel `v_pool_conv` + `flatten_from(1)` — it sees the full 8×11×11 map | Jul 26, 2026 |
| "Model city count passed Greedy for several iterations — expansion is improving" | ❌ Retracted | 93–99% of the difference curve is 7.6-game anchor sampling noise; model side sd 0.081 did not move | EXP_ELO_M1 |
| "First-village rate is trending down below 0.9 — concerning" | ❌ Retracted | Denominator artifact; per-seat rate is 0.79 with slope **+0.0037/iter** (rising), and old/new definitions agree | EXP_ELO_M1 §4 |
| "λ=0.875 widened the army gap / moved the crossover earlier" | ❌ Retracted | Both were anchor-side artifacts; every model-side metric came back \|t\| < 1.2 | EXP_ELO_024 |
| "Unfreeze-opponent / value_trust produced a real gauge gain" | ⚠️ Unsupported | 8 consecutive readings are indistinguishable from a constant 43.2%; the gauge cannot resolve <12pp | EXP_ELO_M1 §3 |
| "Attrition ≈ 0 — the model's units survive; it is out-raced, not out-fought" | ❌ Retracted | In lost games units go peak 5.77 → **0.85** final and cities 2.06 → **0.50** (loses its capital). Out-raced *and* out-fought | EXP_ELO_M2 |
| "Army composition ($/unit) is the gap to aim structural change at" | ⚠️ Demoted | Real in aggregate but **AUC 0.536** on win/loss — cannot be shown to decide games. Third-city reach swings 58pp | EXP_ELO_M2 (corrects a Jul 27 revision of this file) |
| "The model's city count universally peaks at t15 then declines" | ⚠️ Corrected | Two populations averaged: wins peak **t15.17** and hold; losses peak **t7.16** and collapse | EXP_ELO_M2 |

---

## Open / unresolved (as of Jul 27, 2026)

- **Fix the instruments before running more experiments.** (a) Pool the Greedy reference curve instead of re-estimating it from 7.6 games/iteration; (b) raise the gauge to ≥384 games/reading, or accept that it cannot adjudicate anything under ~12pp. Note `--dump-stats-dir` already gives per-game outcome-split data and **`replays/gauge_stats/` holds 94 historical dumps** — the win/loss autopsy is available for free on any past reading.
- **⭐ Why does the model fail to reach a 3rd city — ANSWERED for the dominant mechanism (EXP_ELO_026, Jul 28): (b) stars diverted to tech at the decision point.** An inference-only star gate raised reach +7.6pp with the full causal signature (see the bottleneck section). Residual open parts: the gate still leaves ~28% of games short of city 3 — candidates (a) genuine map/FOW village scarcity and (c) capturer units dead/mis-positioned remain for that tail (`--dump-turn-states` dumps from `replays/exp026/` can separate them). New follow-ups, in value order: (i) a *selective* gate (fire only when a capturable village is genuinely fundable/reachable — the always-on rule breaks ~18–24 games/250); (ii) generate training data with the gate on and distill the allocation policy into the net; (iii) budget interaction at n=256.
- **Why does the model stay on the cheapest unit tier?** Real (2.2 vs 4.4 $/unit) but **demoted** — it does not discriminate wins from losses (AUC 0.536), so it is at best a uniform handicap. Do not prioritize it over third-city reach.
- **Is pursuit's ~50% "wasted" rate a real defect** or benign no-rush play in an uncontested mirror? Unresolved — needs a contested-opponent measurement, not a self-play one.
- **How far does depth scale before saturating?** No saturation is *established* through n=2048, but neither is continued growth — pooled, only 64→256 clears significance (2.70σ); 256→2048 is 0.9σ. A true 4× step to **n=4096** is the registered test (trend predicts ~60%, flat ~56% = real saturation). Confirm rungs at ≥384 games before committing training budget.

### The distillation gap (raised Jul 25, 2026 — the expert-iteration question)

Context: the policy is trained on search's own output (Gumbel π′ = softmax(logit + β·σ(completed-Q))), so in principle the prior should absorb what search knows. It measurably does not, fully. Three open questions:

- **Q1 — Is the ~16pp prior-vs-search gap the *real* gap between what the distilled policy learned and what search did?** Measured: prior-only **36.7%** vs search@256 **53.1%**, with search overriding `argmax(prior)` on **24%** of root decisions. But **the override *rate* says nothing about how different the moves are** — a 24% disagreement could be near-ties between equivalent steps (costing ~0) or genuine blunders. Needs a **magnitude-weighted** metric, not a count: the completed-Q / value gap between the prior's pick and search's pick, and how much of it is concentrated in a few decisive decisions. Until then, treat 16pp as an upper bound on recoverable headroom and 24% as an unweighted incidence rate.
- **Q2 — Can the gap be closed, given the policy learns literally from search's outcomes?** If a gap persists despite training directly on π′, the cause is one of: (a) **capacity/representation** — the state→action mapping search implies isn't expressible by this trunk; (b) **target sharpness** — π′ is a soft distribution and `policy_target_q_weight` controls how much σ(Q) sharpens it; (c) **position-specific tactics that don't compress** (irreducible, no fix); (d) **distribution mismatch** — the prior is evaluated on states its own play reaches, not search's. (a)/(b)/(d) are testable; (c) is the floor. Favorable prior: the known failure (efficient walking to an uncontested village) is simple positional habit, exactly the kind of thing that *should* compress.
- **Q3 — Can MCTS budget serve as a reliable difficulty knob for production?** The measured span is wide and useful — prior-only 36.7% → n=64 44.3% → n=256 53.1% → n=2048 58.0% vs Greedy — and `GUMBEL_SCALE`/`TREE_Q_WEIGHT` add further dials (noise-off is worth ~+7–9pp; `TREE_Q_WEIGHT=0` degenerates to argmax-prior). To ship it as difficulty tiers, needs: **strict monotonicity verified at tight CIs** (the n=1024 rung currently reads *below* n=256, likely noise), a **latency budget per tier** (n=2048 ≈ 68s per 20-action turn is unshippable), and a check that low tiers fail *plausibly* rather than blundering in ways that read as broken.
- **Value-head capacity** (vs. inherent net-vs-net self-play noise on unstable mid-game leads) is the untested residual explanation for the 0.40 discrimination ceiling.
