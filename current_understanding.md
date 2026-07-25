# Current Understanding

**The single source of *current* truth for how the Polyfish AI plays, where it's weak, and why.**
Last synthesized: **Jul 25, 2026** (through EXP_ELO_023).

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
- **The current lever is search depth / throughput (EXP_ELO_023, Jul 25).** Search *does* beat the prior: prior-only argmax **36.7%** vs normal search **45.3%** = **+8.6pp (2.4σ)**. Win rate is **monotonic in mean tree depth** — n=64→45.3% (depth ~4 plies), n=256→50.5%, n=1024→**56.2% (beats Greedy)**. Depth grows ~`sims^0.5`. Polytopia needs ~8 plies to finish **one** game turn, so **production self-play at n=64 (~4 plies) is depth-starved — it cannot see a single turn ahead.**
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
  - **Explicitly NOT** "mirror self-play makes the relative reward net to ~0" — this was rejected forcefully; training uses `ANCHOR_FRAC=1.0` (Greedy-anchor games are the majority) and still fails, so mirror symmetry cannot be the cause. (memory `no-mirror-excuse-for-tempo`)

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
- **How far does depth scale before saturating?** EXP_ELO_023 shows no saturation through n=2048; a true 4× step to **n=4096** is the registered test (trend predicts ~60%, flat ~56% = real saturation). Ladder win rates are noisy (±12pp at 64 games) — confirm at ≥384 games before committing training budget.
- **Value-head capacity** (vs. inherent net-vs-net self-play noise on unstable mid-game leads) is the untested residual explanation for the 0.40 discrimination ceiling.
