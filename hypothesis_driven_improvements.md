# Hypothesis Driven Improvements

The idea is that we should get more systematic about coming up with a hypothesis for bottleneck on a performance metric, come up with experiments that move it, and either "commit" or "reject" it.

This will be the loop we will run continuously to ensure the Polybot continues to improve and get better, to eventually reach human-level capabilities.

Our #1 objective is to figure out how to get into a smooth learning curve regiment. Once we figure that out and can see more training time leads systematically to better playing from the AI, then we can deploy training regiment on the Cloud and let it run over 5M self-play games to reach human-level performance. We only have one shot at a $1M training run and we cannot waste it.

## Protocol

1. Name the bottleneck metric (currently: `villages_t2c_first_cond` at `villages_first_rate` ~1.0).
2. Write the EXP entry **before** running anything: the hypothesis (why we believe it), the exact change, and the expected result as a number with a sample size.
3. Measure. Fast loop: n=32 benchmark on a fixed worst-case tribe pair (Bardur+Imperius) with a fixed model snapshot, run in a scratchpad dir. Slow loop: a live training run read through `training_log.csv` / the dashboard. A training run is not an experiment by itself — it's the slow-loop measurement that closes out whichever EXPs shipped into it, recorded as a validation note, not a numbered EXP.
4. Verdict: **COMMITTED** or **REJECTED** (rejected changes get reverted the same day). We only keep what the data pays for.
5. Ambiguous outcomes become **WATCH** items with an explicit trigger that promotes them back to an EXP.

Shorthand used below: results quoted as `rate/cond` = fraction of games that capture a first village at all / mean capture turn over the games that did. Benchmarks are greedy-teacher-seat unless marked "production mix" (75% net + 25% anchor, the real self-play blend).

*EXP 1–9 are backfilled on Jul 10, 2026 — they were run before this document existed. From EXP 10 on, entries are written before the experiment runs.*

## EXP 1: Auxiliary training heads (ownership / fog / SPT+5 / opponent tech)
*Jul 9, 2026 · COMMITTED, watching*

The value head learns from one number per game: who won. Four new training-only heads make the net also predict final territory, hidden enemy units, income five turns ahead, and opponent tech — free supervision that forces it to understand the game, not just guess winners.

### Expected Results
Aux losses trend down over ~10 iters while policy CE stays at its floor and win-MSE doesn't degrade >10% for 5+ consecutive iters. Long-term: better value generalization from a richer trunk.

### Actual Results
Over run 1783687051 (51 iters): fog ~0.039–0.043, spt 0.026→0.032, ownership 0.22→0.26, tech 0.27→0.29 — flat to slightly **rising**, not falling. But games got longer and richer in the same window (avg moves 495→525, captures/SPT up), so the targets themselves got harder; policy CE fell 2.20→2.08 throughout, no corrosion signature. **Verdict: COMMITTED.** WATCH: shares a trigger with the value-head decline in EXP 10 — if aux stays high while policy/value both fit, that's the trunk-saturation signal (capacity trigger).

## EXP 2: De-censor the first-village metric + log the tribe pair
*Jul 9, 2026 · COMMITTED*

Instrumentation, not behavior. The first-village metric counted a no-capture game as "captured on the last turn", mixing slow with never; tribe pairs also shift the baseline ~2 turns. Split it into capture rate plus average turn when captured, and log the tribe pair.

### Expected Results
Separate capture *rate* from capture *speed*; make the turn-4.5–5.5 bar directly measurable per tribe pair.

### Actual Results
Revealed the old model's rate was ~0.8 — censoring alone inflated t2c by ~1.5–2 turns, and Oumaji/rider pairs run ~5.5 while warrior pairs ran 7–8. New CSV columns (`villages_first_rate`, `villages_t2c_first_cond`, `tribes`) + dashboard lines. **Verdict: COMMITTED.**

## EXP 3: Deeper search — Gumbel 64 → 256 sims
*Jul 10, 2026 · REJECTED*

Early game turns are shallow — only 4–5 plies each — so maybe the bot captures late simply because 64 search sims can't look far enough ahead. We quadrupled the budget to 256 sims to test whether first-village speed is search-bound.

### Expected Results
cond drops toward ~5 if first-capture speed is search-depth-bound.

### Actual Results
n=32, Bardur+Imperius: rate 0.81→1.00, cond 7.9→**7.3**, throughput 129.7→56.9 moves/s (2.3× wall-clock). Depth fixed the never-capture *tail*, not speed: a first capture is a multi-**turn** walk, beyond any single-turn search horizon — priors and the value net have to carry direction between turns. **Verdict: REJECTED — stay at 64 sims.**

## EXP 4: Approach gradient in the expansion evaluator
*Jul 10, 2026 · COMMITTED (part of the EXP 5–7 stack)*

Nothing rewarded getting closer to a village — only the capture itself paid out. We added a small evaluator bonus that grows as a unit closes in on the nearest visible village, so self-play credits progress along the walk, not just the payoff. Standing still earns nothing.

### Expected Results
Shaped self-play credits partial approach before the capture lands, giving the value net a gradient across the multi-turn walk.

### Actual Results
No isolated benchmark — landed as part of the EXP 5–7 stack (stack results under EXP 7). **Verdict: COMMITTED as part of the stack.**

## EXP 5: Doorstep flight — Chebyshev distances + curiosity damping
*Jul 10, 2026 · COMMITTED*

Replays showed a unit two tiles from a village walked away half the time. Two bugs: distance used Manhattan math though units move diagonally, and exploration bonuses outbid the last approach steps. Fixed the math and damped curiosity when a capture is within two tiles.

### Expected Results
Large cond drop — this looked like *the* bug.

### Actual Results
Alone: cond ~8.9→8.93, no change. The fix was real but invisible, because the anchor "teacher" wasn't the greedy scorer at all (see EXP 7) — rollout noise drowned the ordering gradient. Included in the post-swap stack measurement. **Verdict: COMMITTED — necessary, not sufficient. Lesson: verify the change is actually on the measured path before benchmarking it.**

## EXP 6: Capture must outrank attack
*Jul 10, 2026 · COMMITTED*

In the move-ordering scores, the best attack (110) outbid capturing a village (99.8) — so a unit standing on a village would swing at an enemy instead of taking it. We raised capture scores above every possible attack score, so taking the village always wins.

### Expected Results
d=0 always converts to a capture.

### Actual Results
d=0 capture rate 100%; benchmark cond 6.47→**6.22** at rate 1.00 (n=32, Bardur+Imperius). **Verdict: COMMITTED.**

## EXP 7: Greedy anchor — replace the rollout-MCTS teacher
*Jul 10, 2026 · COMMITTED*

A quarter of self-play games use a teacher seat meant to demonstrate good habits. That seat ran a noisy rollout search that drowned out our tuned move ordering — the teacher never taught it. We swapped in the plain greedy scorer, the same scores the net's search priors use.

### Expected Results
Teacher demonstrates 4–6-turn first captures; training data quality jumps.

### Actual Results
Anchor seat: 0.94/8.9 → **1.00/6.47** (n=32, worst-case pair), with EXP 4–5 riding along. Largest single gain of the campaign. **Verdict: COMMITTED.**

## EXP 8: Frontier-resource beacon
*Jul 10, 2026 · COMMITTED*

Resources only spawn next to villages, so a fruit at the edge of the fog hints a hidden village sits nearby — the cue a human steers by at spawn (Verdi's screenshot). We added a pull toward resources that still have unexplored tiles around them.

### Expected Results
Blind-phase exploration steers toward hidden villages instead of random fog; cond drops on maps where no village is visible at spawn.

### Actual Results
Took three tries. v1 (pull toward any resource not explained by a known village) regressed ~1.5 turns: units parked on the fruit when their sight couldn't reach the hidden village behind it. v2 scaled the pull by how open the surrounding area is, which fixed the parking. v3 fixed the deeper bug: the capital's own structure "explained away" every fruit inside spawn vision — the exact evidence a human uses — so the rule became "resource still touching fog within 2 tiles". Final benchmark: 0.97/**5.97** (best single result of the campaign) vs 1.00/6.22 for the veto version — a statistical wash at n=32; kept v3 because it encodes the real signal instead of filtering it out. Remaining gap: greedy walks to the nearest single fruit and can't read the two-fruits-same-side direction cue; that inference is the net's job. **Verdict: COMMITTED.**

## EXP 9: Stronger center pull (×2)
*Jul 10, 2026 · REJECTED*

Villages are denser toward the middle of the map, so when a unit sees no village and no fruit, sweeping harder toward the center might find one faster. We doubled the center-pull weight in the move ordering to test it.

### Expected Results
Faster blind discovery, cond drop.

### Actual Results
cond 6.22→6.39 with a rate dip — the center pull overrode useful local evidence. Reverted same day. **Verdict: REJECTED (center weight stays ×1).**

## Training validation — run `1783687051` (slow-loop readout for EXP 4–8)
*Jul 10, 2026 · COMPLETE — 60 iterations*

Not a numbered experiment — nothing new changed here. This run is the slow-loop measurement that closes out the committed stack above: train on the new teacher and shaping for 60 iterations and watch whether the net absorbs it into its own play.

### Expected Results
Pre-registered: capture rate pins ~1.0; cond grinds from ~7 toward the low 5s, ending below the static teacher's own benchmark (6.2 on the worst pair, 6.58 in production mix).

### Actual Results
Cond fell 6.02 (first 10 iters) → 5.40 (iters 11–30) → **5.24** (last 10), at rate ~0.97 throughout; the censored metric went 6.54→6.02 (it hovered ~7.5 before the stack). Tribe-controlled, same trend: Imperius↔Kickoo ~6.6→~5.8, Oumaji↔XinXi ~5.0→**~4.5**. The net ends *faster than the teacher that bootstrapped it* — the direction-reading it adds over greedy nearest-fruit is real. Economy grows too: SPT@10 6.10→7.01, SPT@5 3.96→4.18; policy CE 2.20→2.08. First strength reading: the iter-60 league match vs the pre-fix checkpoint scored 5310 to 4907 for the current net (+8% — one match, average score, not a win rate). **Outcome: pre-registration met on speed and on beating the teacher; rate landed 0.97 vs the 1.0 target — residual resolved Jul 11, see the rate-residual WATCH below.**

WATCH items from this run:
- **Value head**: value_r2 slid 0.701→0.661 over iters 1–51 while games lengthened (avg moves 495→525), then held flat ~0.66 for the last 10. Plausibly the data just got harder. Trigger: r2 < 0.60 or the slide resumes → run the fixed-holdout probe (candidate below).
- **Rate residual — RESOLVED (Jul 11)**: dumped every zero-capture game from a 128-game Kickoo+Bardur probe (new `self_play --dump-failed-dir`: watcher replay + full per-decision search traces). All 7 were **Domination wins** — one side captured the enemy capital on turn 6–10 and the game ended before anyone banked a neutral village. Not lost units: the winner rushed (the greedy anchor does it too, in 2 of 7), the loser pulled units home to defend and died. The metric counts these wins as capture failures — third censoring artifact of this campaign. Artifacts: `polyfish-rs/replays/failed_games/`.
- **League cadence**: the six-run drought is explained — the GN migration quarantined every old checkpoint into `checkpoints/bn_era/`, and the selector needs ≥2 eligible `model_checkpoint_iter*` files before it fires. It self-healed at iter 60, but checkpoint-every-50 means one league reading per ~7h of training. Candidate fix: denser checkpoints or a standing arena benchmark vs the frozen `model_checkpoint_iter50_20260710_015335` (pre-fix reference).

## EXP 10: Strength gauge — the frozen-anchor Elo ladder
*Jul 11, 2026 · COMMITTED — and it immediately caught our biggest blind spot*

All our metrics so far measure behavior — capture speed, SPT, policy loss — not strength. This adds the missing y-axis: paired arena matches against frozen reference models, chained into one Elo curve. It's the line that must keep rising before we commit real money to the long cloud run.

### Design (instrumentation, no behavior change)

- **Reading**: n=32 seeds, sides swapped (64 games), `arena` at gumbel 64/k=16, `--gamemode 2`. Win rate = wins + draws/2.
- **Ladder rules**: a reading every 10th iteration vs the *active anchor* (a frozen checkpoint that never changes). ≥80% → freeze the current model as the next anchor and measure the link vs the outgoing one at n=64. Audit every 50 iters vs Greedy + one retired anchor — observed vs chain-predicted win rate flags Elo inflation/cycles.
- **Permanent floor anchor**: the Greedy backend (the production teacher seat), Elo 0 by definition — a non-net agent that can't join net-vs-net strategy cycles.
- **Backfill today**: `gn_v2` → `iter50_015335` → `iter50_220138` → current, each vs Greedy plus the informative net-vs-net pairs.

### Expected Results (pre-registered before any match ran)

1. Current beats Greedy at **≥60%**; if ≥80%, Greedy retires to audit duty on day one.
2. Monotonic ordering vs Greedy: `gn_v2` < `iter50_015335` < current.
3. Current vs `iter50_015335` ≥55%.
4. Transitivity: current vs the chain prediction within ~±10pp (no cycle).

### Actual Results

Backfill (n=32 paired, Domination, gumbel 64/k=16; Elo vs Greedy = 0; reading CI ≈ ±9pp):

| model | vs Greedy | ≈ Elo |
|---|---|---|
| `gn_v2` (era start) | 3.1% | −600 |
| `iter50_015335` (pre-fix run) | 23.4% | −206 |
| `iter50_220138` (latest run, iter 50) | 43.8% | −43 |
| current `model.safetensors` (iter ~60) | 25–34% | −110 to −190 |

Net-vs-net links: current beats `iter50_220138` at 53.1% (final-10-iters regression scare was sampling noise) and `iter50_015335` at 73.4%, inside the 63–74% chain prediction — **transitivity holds, the ladder chains**.

Pre-registrations #2–#4 met. **#1 failed: the net still loses to its greedy teacher ~2:1** while every behavioral metric said "improving" — village speed measured an opening skill, not strength. The good news is the trend: **~+500 Elo across the era, monotonic at every rung**. Graduation target for the next stint: >50% vs Greedy.

Method notes: vs-Greedy readings scatter more than net-vs-net, so the curve rides on net anchors (Greedy is for audits); the gauge is pinned to `--gamemode 2` (mode is a net input feature *and* a greedy-evaluator branch — Perfection under-read the net); arena now runs on the self_play eval-server stack (net-vs-net reading: 20 min → 83 s). Also found and fixed: arena let MCTS search mutate the real game (production was safe — `Brain` searches a clone); the corruption it caused proves some undo callbacks don't roundtrip exactly — WATCH if search ever goes clone-free.

## EXP 11: Gauge in the loop — auto-ladder + plateau early-stop
*Jul 11, 2026 · pre-registered, shipping*

Wire the EXP 10 reading into `run_training_loop.sh`: every `LEAGUE_INTERVAL` iters, arena vs the active anchor, appended to `ladder.json` (anchors + readings, human-readable, via `ladder.py`). ≥80% freezes the model as the next anchor (n=64 link match). Audit every 50 iters vs Greedy + a rotating retired anchor. Early stop: over the last 8 readings vs the same anchor, window means flat-or-down AND slope ≤ 0 counts one strike; two consecutive strikes ends the run — ~80+ iterations of evidence of non-improvement, robust to single-reading noise (±9pp).

### Expected Results
Next stint: readings every 10 iters climb from ~25–34% vs Greedy toward the >50% crossing; no false plateau stop on the way; first anchor freeze at ≥80%.

### Actual Results
It worked but we see it actually plateauing and the trained NN unable to beat the teacher enough to be made an anchor. It wins ~25% of the time against greedy-only.

---

*From here on, experiments are named by track: `EXP_ELO_*` targets the strength gauge (win rate vs the Greedy anchor / Elo curve). Other tracks get their own prefixes as they open.*

## EXP_ELO_001: Loss autopsy vs Greedy — name the mid-game bottleneck
*Jul 11, 2026 · pre-registered*

The net now opens faster than its teacher (t2c 5.24 vs 6.2) yet loses to it 2:1, and the ladder's vs-Greedy readings show a ~1,600-point average score gap (net ~3,800, Greedy ~5,400). Every metric we've optimized so far is an opening metric; the losses are being decided somewhere we don't measure. Hypothesis: Greedy pulls away in a specific mid-game window, and the first diverging sub-metric (SPT cadence, city count, army value, or units lost) is nameable and becomes the successor to `villages_t2c_first_cond` at the top of the protocol.

**Change (instrumentation only):** arena learns `--dump-stats-dir`: per-turn samples (score, SPT, city count, unit count, total unit cost, tech count — both sides) written as one JSON per game. Reading: the standard gauge setup — n=32 seeds sides-swapped (64 games), gumbel 64/k=16, `--gamemode 2`, metal eval — vs the Greedy backend, then plot the per-turn curves split by win/loss.

### Expected Results
A divergence window: Greedy's score curve breaks away from the net's between roughly turn 8 and turn 20, led by one identifiable sub-metric. Falsifiers: if the gap is uniform from turn 0, the opening work never mattered for strength; if it only appears at endgame, the bottleneck is closing, not economy. Either way the output is the new #1 bottleneck metric.

### Actual Results
n=32 seeds (64 games), model 37.5% — reading consistent with the ladder band. The score crossover lands in the predicted window (turn 8–9, gap peaking ~turn 16), but the causal chain starts earlier and has a clear shape:

1. **Units first (turn 3–4):** Greedy trains units immediately (3.0 vs 1.8 by turn 4) and never stops — by turn 16 its army value is 30 vs 13 (in its wins: 41 vs 10, then it kills us by ~turn 20).
2. **Expansion stalls after the first village:** first-capture speed is fine (the EXP 2–9 skill is real), but the model reaches a 3rd city in only **39% of games vs Greedy's 81%** — in Greedy's wins it's **20% vs 100%**. The model grabs one village and stops; Greedy runs an expand-forever engine.
3. **SPT follows cities (turn 6–8 on):** 8.4 vs 15.9 by turn 16 — a direct consequence of the city gap, amplified by harvests.
4. **Tech is anti-correlated:** the model out-researches Greedy in every split, including its losses (t24: 17.3 vs 12.1 techs). It converts stars into research (early score!) while Greedy converts them into units and cities. The model's early score *lead* (turns 1–7) is exactly this — buying scoreboard points that don't compound.

**Verdict: COMMITTED (instrument + diagnosis).** The opening-village campaign taught a skill the model has; the game is decided by expansion *continuation* and army production, where it under-invests — plausibly a research-shaped local optimum (tech = immediate score = shaped reward). New #1 bottleneck metric: **third-city rate** (target: ≥0.8 by turn 13, Greedy's level), with army value @ turn 12 as the co-metric. Caveat: per-turn means past ~turn 18 are survivorship-biased (Greedy's wins end ~turn 20, the model's ~turn 24).
*Jul 11, 2026 · pre-registered*

The plateau's timing matches the crutch schedule, not a capacity wall: `anchor_frac` starts at 0.25 and decays 0.97^iter to its 0.1 floor by ~iter 30, and the heuristic prior weight decays 0.5→0.1 on the same clock — so from mid-run onward ~90% of games are weak-net-vs-weak-net. Value targets from those games teach "who beats a weak net", not "who beats Greedy". EXP 7 showed the teacher seat was the largest single gain of the campaign; we then removed it on a schedule instead of on a condition, while the model was still below the teacher.

**Change:** the loop holds `anchor_frac` at its starting 0.25 (no decay) while the latest ladder reading vs Greedy is <50%; once a reading crosses 50%, the decay clock starts from that iteration. Heuristic prior weight keeps its existing schedule — one variable at a time.

### Expected Results
Vs-Greedy gauge readings (n=32 every `LEAGUE_INTERVAL` iters) resume climbing: mean of the first 3 post-change readings ≥ the last pre-change window mean + 8pp, and no plateau strikes fire in the first 30 iters. Secondary (from EXP_ELO_001): third-city rate climbs toward Greedy's 0.81. Falsifier: 3 consecutive readings flat within ±5pp of the old mean → REJECT (teacher starvation isn't the plateau; escalate to the capacity trigger from EXP 1/10 or the shaping candidate from EXP_ELO_001's findings).

**Method amendment (Jul 11, pre-readout):** the first stint runs at a REDUCED budget (`-n 16 -k 4`, 20 iters, gauge every 5) — the new fast-experiment tier. Greedy uses no search, so a smaller budget weakens only the net's side: these readings sit at a lower level than the 64/k=16 ladder history and MUST NOT be compared to the 25–34% pre-change band or chained into the canonical Elo curve. Judge this stint within-budget only: the slope across its own readings plus the third-city-rate trend in the training log. A climb at 16 sims = mechanism confirmed (extend/rerun at full budget for the registered +8pp criterion); flat at 16 sims is *weak* evidence against — the search-improvement operator is also degraded at 16 sims — so a null here gets one re-test at 64 before REJECT.

### Actual Results
Run `1783809008`, 82 iters total: 20 at 16/k=4 (readings 30/33/23/27% — flat, as covered by the method amendment), then 60 overnight at the registered 64/k=16. The six 64-sim readings vs Greedy: **31.2, 37.5, 23.4, 35.9, 40.6, 33.6%** (Elo −137 → best −66, ending −118).

Against the registered criteria: first-3 mean 30.7% vs the ~29.5% pre-change window = **+1pp — the +8pp success bar was NOT met**. The falsifier also did not fire (37.5% and 40.6% both broke the ±5pp flat band; plateau strikes 0). Within the run, first-3 → last-3 means rose 30.7% → 36.7% (+6pp, ~1.2σ — suggestive, not conclusive alone).

The behavior curves carry the real signal. Across readings 30→80: the post-t15 city collapse shrank (t15→t25 bleed −0.67 → −0.32/−0.41), SPT@t25 rose 6.3 → ~7.2–8.1, army value@t25 8.0 → ~9.7–10.7, and the t25 score gap roughly halved (−1471 → −547/−878) — with Greedy's own curves *pulled down* at the good readings (the model interfering with a fixed opponent). Value R² dipped 0.72→0.67 while the first-ever 30-turn training data arrived (iters 22–50), then recovered to 0.74 — the late-game distribution was absorbed. Confound to note: the curriculum crossed into the 30-turn stage at iter 16, so "restored anchor signal" and "first late-game training data" are entangled in this window. Also: the 16-sim P2-seat skew did not replicate at 64 sims (P1 77 wins vs P2 52) — artifact/noise.

**Verdict: WATCH — mechanism engaged (value head learning, late-game behavior healing, no plateau), but the strength conversion is a slow climb, below the registered bar.** The anchor hold stays in place (it's condition-gated and the data shows no harm). This outcome is precisely the promotion trigger for EXP_ELO_003 below.

### Queued follow-up — EXP_ELO_003: anchor dose-response (0.25 → 0.4–0.5)
Promoted to a live EXP only after 002 reads out. Trigger: 002 shows a real but slow climb (readings rising but <8pp over 3) → test whether more anchor games speed value-head relabeling. Run with `ANCHOR_FRAC=0.4`–`0.5`, watch vs-Greedy win rate + third-city rate, and watch policy CE for imitation-regression (anchor games record the greedy seat as teacher targets — too high a dose re-anchors the policy to the teacher, whose ceiling we're trying to pass; it also risks overfitting an exploit lane against a deterministic opponent instead of general strength). If 002 outright fails its falsifier, skip 003 — dose was never the variable.

### EXP_ELO_003 result (Jul 12) — REJECTED at the cheap tier
Two sequential 16/k=4 arms (0.4 then 0.8, each a fresh 20-iter run; NOT parallel — the 0.8 arm started from the 0.4 arm's final weights, and the pre-smoke model was not backed up, so level comparisons carry an upward bias for later arms). 0.4 arm readings: 25%, 22% vs the 0.25 baseline's 30/33/23/27; 0.8 arm stopped early by hand. Despite the training-progress tailwind, higher doses read flat-to-lower, and behavior curves showed the same city-stall shape. **Verdict: REJECTED — anchor dose is saturated at 0.25; the bottleneck is elsewhere.** Caveat: both arms ran with the run-relative value-trust ramp throttling σ(Q) to ≤0.67 (see the hygiene fix below), so the null is weaker than a clean test — but the same throttle applied to the 0.25 baseline, and its direction matched EXP_ELO_002's slow-climb at full trust. Fallout fixed: run-start model checkpoints are now automatic (`checkpoints/run_<id>_iter<start>_start.safetensors`).

## EXP_ELO_004: The TD term subsidizes the tech tower
*Jul 12, 2026 · pre-registered — probe (004a) first, stint (004b) on a positive probe*

The value target is 70% TD(λ) whose per-step reward is the score delta between turn checkpoints (`td_lambda_labels`, self_play.rs), plus 30% final outcome. Research pays 100×tier score points the same turn it's bought (actions/tech.rs); army and expansion are multi-turn investments whose score arrives late or via compounding — mostly outside λ=0.8's ~5-turn credit window. Hypothesis: the TD term systematically over-credits tech relative to compounding assets, and is the dense gradient holding the model in the tech-tower local optimum diagnosed in EXP_ELO_001. The final-outcome tail (which correctly ranks Greedy's expansion strategy above tech-towering — Greedy outscores us at turn 30) is drowned at 0.3 weight. This also explains why 30-turn training alone never fixed it (Verdi's observation): the mis-pricing is per-step and horizon-independent. History note: TD's ancestor (4-turn forward delta) was introduced for credit *speed*, not anti-passivity — the anti-passivity guard is the anchor games (notes.md Jul 8), which stay held at 0.25 throughout this EXP.

**Change:** `--td-w` flag (new; default 0.7 = production). Probe arm: `TD_W=0.2`. Unmutings required for a short probe to be readable: `VALUE_TRUST_RAMP_ITERS=1` (full σ(Q) trust from iter 1 — justified at value R² ~0.70) and `REPLAY_BUFFER_FILES=0` (archived games carry old-blend labels that would dilute the new signal).

### Probe design (004a)
10 iters, 16/k=4, `-l 5`, `ITER_OFFSET=80` (30-turn stage), from a launch checkpoint. Judged on leading indicators ONLY (win rate excluded — ±9pp on 2 readings): (1) `value_loss` spike-then-refit at iters 1–2 proving the label changed; (2) `avg_research`/game declining vs control; (3) gauge `techs` curve bending toward Greedy's with `cities` flat-or-up. **Passivity tripwires** (the 1783285900 collapse signature): sustained decline across moves/game, captures/game, attacks/game → abort and roll back to the launch checkpoint; if tripped, TD had quietly become a second anti-passivity crutch — record and rethink.

### Execution notes (Jul 12)
1. **First probe attempt was an accidental control**: it ran before `--td-w` existed, so `TD_W=0.2` (env) silently did nothing — the run (1783855452) executed at TD 0.7 with the two unmutings active. Confirmed by the absence of any value_loss shift (flat ~0.62 — a correct negative control for the label-change instrument). It is now the matched control arm: identical conditions except TD_W.
2. Control results were unexpectedly strong: iter-5 reading **37.5% — the best 16-sim reading ever** (prior 16-sim: 30/33/23/27, 25/22), shallowest post-t15 city bleed yet (2.23→1.89), opponent cities suppressed to 3.05, model army value 12–13.7. Suggests full value-trust from iter 1 and/or fresh-only training helps by itself; single reading, unconfirmed.
3. The run was killed at iter 5 by a **plateau-stop artifact**: the 8-reading window mixed readings across runs/budgets/experiment arms. Fixed: `ladder.py` now scopes the plateau series to the current `run_id`.

### Expected Results (probe)
TD_W=0.2 arm vs the accidental control: value_loss spikes at iter 1–2 then refits; avg_research trends below the control's ~16–18; gauge techs curve at t20–25 drops toward the opponent's while cities hold ≥ control. Falsifier: with full trust and fresh-only training, research/tech metrics indistinguishable from control after 10 iters → the TD-subsidy theory takes major damage; next suspects are search depth and capacity.

### Execution history (Jul 12–13) — three attempts before a valid test
1. **Attempt 1** (run 1783861463, "probe"): `-r` omitted — `--td-w` is only read inside the `reward_shaping` branch (self_play.rs:2013), which defaults false. Void; TD_W had zero effect. Retroactively repurposed as a second replicate of the pure-final-outcome regime.
2. **Attempt 2** (runs 1783865321 TD_W=0.2, 1783869341 TD_W=0.7): `-r` omitted again (confirmed with Verdi) — same bug, both runs void for this question.
3. **Attempt 3** (runs 1783877772 TD_W=0.2, 1783885395 TD_W=0.7): `-r` present this time, but the two runs were sequential, not parallel — 1783885395 continued from 1783877772's ending weights, so any difference was confounded with 10 extra iterations of training (same defect as EXP_ELO_003). Inconclusive; deltas were within noise.
4. **Clean rerun** (Jul 13): used the auto-saved launch checkpoint (`checkpoints/run_1783877772_iter1_start.safetensors`) to reset `model.safetensors` to the exact pre-arm-3 weights, then ran a fresh TD_W=0.7 arm (run **1783894201**) from that identical baseline as arm 3's TD_W=0.2 run (**1783877772**). Verified bit-identical starting weights via sha256. This is the valid test.

### Actual Results
Clean, bit-identical-baseline comparison (both `-r`, 10 iters, 16/k=4, `ITER_OFFSET=80`, `VALUE_TRUST_RAMP_ITERS=1`, `REPLAY_BUFFER_FILES=0`):

| metric (iter 5 reading) | TD_W=0.2 (1783877772) | TD_W=0.7 (1783894201) |
|---|---|---|
| win rate vs Greedy | 35.9% | **48.4%** (best 16-sim reading to date) |
| Elo est | −100.4 | **−10.8** |
| cities @t15 / @t25 | 2.05 / 1.70 | **2.58 / 2.19** |
| SPT @t25 | 7.75 | **9.36** |
| score @t25 | 3548 | **3761** |

Iter-10 reading: both arms cooled (0.2→25.0%, 0.7→35.9%) but TD_W=0.7 stayed ahead on every metric at both readings — win rate, Elo, the full cities curve, SPT, and score all move together in the same direction, both readings. This is a large, internally consistent effect, not a noise-band result like everything upstream of it.

**Verdict: REJECTED — in the opposite direction than hypothesized.** More TD weight (dense per-step credit), not less, produced the best expansion/strength numbers recorded at this budget tier. Revised model: TD's per-step reward isn't selectively crediting tech — it rewards *any* score-generating event each turn, cities included (founding = +100, plus every population/harvest tick after). At TD_W=0.2, training signal is dominated by one sparse, high-variance final-outcome label per game; at 0.7, the value head gets dense low-variance feedback every turn, and at a 10-iteration horizon that sample-efficiency advantage dominates any steady-state tech-crediting bias. The tech-tower symptom from EXP_ELO_001 more likely traces to the *absolute pricing* of research (100×tier, paid instantly, actions/tech.rs) than to the TD/final-outcome blend weight.

**Action taken:** `model.safetensors` now holds the TD_W=0.7 (1783894201) ending weights — the better-performing line — left in place as the production tip. The `tip_1783885395` backup (the older, no-reward-shaping-history line) was NOT restored.

**Follow-up implications:**
- Reopens the idea we set aside earlier in this thread: if the tech bias is a pricing problem, the direct lever is `actions/tech.rs`'s `100 * tier` score grant (or an equivalent discount applied where value targets are computed) — not scoring.rs (policy-side, already correctly under-prices research) and not the TD blend weight (now tested, wrong lever).
- Open question for a future EXP: is today's whole session's win-rate climb (EXP_ELO_002 mechanism engaging, etc.) partly attributable to reward-shaping having been OFF for ~90 iterations (pure final-outcome, high-variance, slow) rather than to the anchor-frac hold alone? The `-r` omission was present across all of EXP_ELO_002/003's runs. Worth a dedicated reward-shaping-on/off ablation from a shared baseline if the anchor story needs re-litigating.
- Lesson banked for the loop script: `run_training_loop.sh` should log which flags (`-r`, `--td-w`, etc.) were actually passed into a per-run manifest file, so this three-attempts-to-get-one-valid-test problem can't recur silently.
