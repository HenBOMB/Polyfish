# Current Understanding

**The single source of *current* truth for how the Polyfish AI plays, where it's weak, and why.**
Last synthesized: **Jul 25, 2026** (through EXP_ELO_023 + its Jul 25 addenda: n=2048 rung, pooled ladder, prior-override metric).

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

### The bottleneck and the current lever

- **Bottleneck metric = expansion tempo (the "third-city rate").** vs the Greedy anchor the model is **out-*raced*, not out-fought**: Greedy's earlier unit production and earlier map presence put its units on contested villages first (attrition ≈ 0; the model's units survive). It is a *tempo/race* gap. (ledger §6)
- **The current lever is search depth / throughput (EXP_ELO_023, Jul 25).** Search *does* beat the prior: prior-only argmax **36.7%** vs normal search **45.3%** = **+8.6pp (2.4σ)**. Depth grows ~`sims^0.5`. Polytopia needs ~8 plies to finish **one** game turn, so **production self-play at n=64 (~4 plies) is depth-starved — it cannot see a single turn ahead.**
  - **Ladder, pooled across both Jul 25 sweeps** (supersedes the single-sweep rungs): n=64 **44.3%** (255/576, depth ~4), n=256 **53.1%** (204/384, depth ~9), n=1024 51.3% (82/160, depth ~19), n=2048 58.0% (65/112, depth ~26). **The one statistically established step is 64→256: +8.9pp, 2.70σ, p≈0.007.** 256→2048 is only 0.9σ — real but unproven. Do **not** state the ladder as "+5pp per 4× forever", and do **not** state it saturates past 256; both readings were made and retracted the same day.
  - **Recommended budgets: generate/train at n=256** (proven gain, only **2.7× per game** — 1.4s vs 0.4s — and the first rung past the one-turn threshold); **deploy at n=64** (166 ms/move ≈ 3.3s for a 20-action turn). n=2048 is 3428 ms/move ≈ 68s/turn — training-only at best.
  - **Distillation headroom is large and measurable.** New `PRIOR OVERRIDE` metric (root decisions where search's pick ≠ `argmax(prior)`; `GumbelMctsAgent::agree_count/decision_count`, printed by arena): **24.2% (n=64) → 24.1% (256) → 29.4% (1024) → 34.0% (2048)**. It *rises with depth* — deep search finds moves the prior does not rank first — and notably only **starts** rising past ~9 plies, i.e. once the tree clears one full game turn. Gap from bare prior to n=256 search ≈ **16pp**. Caveat: the metric counts decisions, not importance; some share are near-ties.
  - **Do NOT swap Gumbel→PUCT.** At fixed n=64, concentrating (k=4, k=2) is *worse* than k=16 despite deeper PVs — root breadth beats principal-variation depth. Gumbel's anti-concentration is a feature.
  - **Fix the horizon ceiling before pushing sims:** `brain.rs:435` hardcodes `max_turns_ahead = 20 − current_turn` while games run to turn 30, so from ~turn 18 the horizon pins to its 2-turn floor (horizon-capped descents hit 9.8% at n=1024).

### The value head

- **Mis-calibrated, not blind — and not the binding lever.** It is a *worse* overall outcome predictor than the raw score ratio late-game (where score≈outcome), but **beats the scoreboard early-game** (real foresight where score is blindest). Its defect is **over-confidence when ahead** (~2×: predicts +0.6/+0.9 where actual is +0.28/+0.53). (EXP_ELO_021, memory `value-head-calibration-diagnostic`)
- **The over-confidence is robust to target re-weighting** — both EXP_ELO_021 (de-saturate the outcome label) and EXP_ELO_022 (down-weight the biased TD arm) were **REJECTED**: they moved the target mean but not the head's outcome-*discrimination* (corr(raw,outcome) ≈ **0.40**, unchanged). It's not a cheap target-design bug. **Stop re-weighting the value target.**
- At a **pursuit fork** the value head is **indifferent, not opposed** (own_value(toward) − own_value(away) ≈ 0), because step-toward and step-home produce near-identical boards. (memory `no-mirror-excuse-for-tempo`)

### The policy prior

- **Healthy, not suppressed.** The toward-village prior rose **0.14 → 0.43** under training. Earlier "Build/Harvest suppressed 2–4 orders of magnitude" and "won't seek" reads were **same-ply popularity-contest artifacts** — one ply bundles ~25 competing Step candidates, so a per-candidate prior comparison measures the wrong thing. (ledger §5, notes.md third-city arc, memory `wander-search-rejection-value-blind`)
- **"Wandering" with no village in view is mostly benign.** In eval (Gumbel off) the net picks **0.80** frontier-ward (1.0 = most frontier-ward), top-2 direction 72%. The scary "41%" was a strict argmax metric plus Gumbel self-play exploration jitter.

### Pursuit and representation

- **Pursuit progress is a search-time reward potential, never a network input feature.** `features.rs` has no unit→capturable-village channel, and the value head global-average-pools the trunk, so it is ~blind to pursuit progress at input. (memory `pursuit-failure-representation-gap`)
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

- Across Jul 23–25 the real plateau resolved to: the model only **matches a zero-search heuristic** (50/50 vs Greedy at n=64), and the binding lever is **search depth / throughput** (EXP_ELO_023) — not the value target (021/022 rejected), not the prior (healthy), not pursuit representation (no strength gain), not the reward labels (family closed).

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

---

## Open / unresolved (as of Jul 25, 2026)

- **Is pursuit's ~50% "wasted" rate a real defect** or benign no-rush play in an uncontested mirror? Unresolved — needs a contested-opponent measurement, not a self-play one.
- **How far does depth scale before saturating?** No saturation is *established* through n=2048, but neither is continued growth — pooled, only 64→256 clears significance (2.70σ); 256→2048 is 0.9σ. A true 4× step to **n=4096** is the registered test (trend predicts ~60%, flat ~56% = real saturation). Confirm rungs at ≥384 games before committing training budget.

### The distillation gap (raised Jul 25, 2026 — the expert-iteration question)

Context: the policy is trained on search's own output (Gumbel π′ = softmax(logit + β·σ(completed-Q))), so in principle the prior should absorb what search knows. It measurably does not, fully. Three open questions:

- **Q1 — Is the ~16pp prior-vs-search gap the *real* gap between what the distilled policy learned and what search did?** Measured: prior-only **36.7%** vs search@256 **53.1%**, with search overriding `argmax(prior)` on **24%** of root decisions. But **the override *rate* says nothing about how different the moves are** — a 24% disagreement could be near-ties between equivalent steps (costing ~0) or genuine blunders. Needs a **magnitude-weighted** metric, not a count: the completed-Q / value gap between the prior's pick and search's pick, and how much of it is concentrated in a few decisive decisions. Until then, treat 16pp as an upper bound on recoverable headroom and 24% as an unweighted incidence rate.
- **Q2 — Can the gap be closed, given the policy learns literally from search's outcomes?** If a gap persists despite training directly on π′, the cause is one of: (a) **capacity/representation** — the state→action mapping search implies isn't expressible by this trunk; (b) **target sharpness** — π′ is a soft distribution and `policy_target_q_weight` controls how much σ(Q) sharpens it; (c) **position-specific tactics that don't compress** (irreducible, no fix); (d) **distribution mismatch** — the prior is evaluated on states its own play reaches, not search's. (a)/(b)/(d) are testable; (c) is the floor. Favorable prior: the known failure (efficient walking to an uncontested village) is simple positional habit, exactly the kind of thing that *should* compress.
- **Q3 — Can MCTS budget serve as a reliable difficulty knob for production?** The measured span is wide and useful — prior-only 36.7% → n=64 44.3% → n=256 53.1% → n=2048 58.0% vs Greedy — and `GUMBEL_SCALE`/`TREE_Q_WEIGHT` add further dials (noise-off is worth ~+7–9pp; `TREE_Q_WEIGHT=0` degenerates to argmax-prior). To ship it as difficulty tiers, needs: **strict monotonicity verified at tight CIs** (the n=1024 rung currently reads *below* n=256, likely noise), a **latency budget per tier** (n=2048 ≈ 68s per 20-action turn is unshippable), and a check that low tiers fail *plausibly* rather than blundering in ways that read as broken.
- **Value-head capacity** (vs. inherent net-vs-net self-play noise on unstable mid-game leads) is the untested residual explanation for the 0.40 discrimination ceiling.
