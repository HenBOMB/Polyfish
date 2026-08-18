# Hypothesis Driven Improvements

The idea is that we should get more systematic about coming up with a hypothesis for bottleneck on a performance metric, come up with experiments that move it, and either "commit" or "reject" it.

This will be the loop we will run continuously to ensure the Polybot continues to improve and get better, to eventually reach human-level capabilities.

Our #1 objective is to figure out how to get into a smooth learning curve regiment. Once we figure that out and can see more training time leads systematically to better playing from the AI, then we can deploy training regimen on the Cloud and let it run over 5M self-play games to reach human-level performance. We only have one shot at a $1M training run and we cannot waste it.

## Protocol

1. Name the bottleneck metric (currently: `villages_t2c_first_cond` at `villages_first_rate` ~1.0).
2. Write the EXP entry **before** running anything: the hypothesis (why we believe it), the exact change, and the expected result as a number with a sample size.
3. Measure. Fast loop: n=32 benchmark on a fixed worst-case tribe pair (Bardur+Imperius) with a fixed model snapshot, run in a scratchpad dir. Slow loop: a live training run read through `training_log.csv` / the dashboard. A training run is not an experiment by itself — it's the slow-loop measurement that closes out whichever EXPs shipped into it, recorded as a validation note, not a numbered EXP.
4. Verdict: **COMMITTED** or **REJECTED** (rejected changes get reverted the same day). We only keep what the data pays for.
5. Ambiguous outcomes become **WATCH** items with an explicit trigger that promotes them back to an EXP.

Shorthand used below: results quoted as `rate/cond` = fraction of games that capture a first village at all / mean capture turn over the games that did. Benchmarks are greedy-teacher-seat unless marked "production mix" (75% net + 25% anchor, the real self-play blend).

*EXP 1–9 are backfilled on Jul 10, 2026 — they were run before this document existed. From EXP 10 on, entries are written before the experiment runs.*

## Standing note (Aug 18, 2026): every gauge-derived verdict below is provisional

The strength gauge — the instrument EXP 10 built and EXP 11 wired into the loop —
**never recorded a reading**. `arena` rejected `--dump-stats-dir`, which
`run_gauge_match` always passed, and the loop tested whether the win count parsed
rather than checking arena's exit code: it printed `GAUGE: arena reading failed
to parse — skipping this reading` and continued. `ladder.py record` therefore
never ran, so `ladder.json` gained no readings, no plateau strike could fire, no
anchor could freeze at ≥80%, and `.anchor_decay_start` was never written. See P2
in `expert_pipeline_audit.md`.

Consequently **EXP 11's plateau observation and EXP_ELO_002's "+1pp, the +8pp
success bar was NOT met" were drawn from an instrument that was returning parse
failures, and both are provisional pending a re-baseline.** Any reading taken by
hand in that period is a real match but is not comparable across iterations for
four further reasons, each since fixed: maps were re-rolled every reading (no
`--seed`), `arena` searched the un-obscured game while self-play searched a
fog-obscured clone, the gauge played 30-turn games against 45-turn training
games, and the map generator gave seat 2 an island start in ~1/3 of Tiny/Drylands
seeds.

This does not overturn any verdict — it withdraws the evidence for the
gauge-derived ones. Behaviour-metric verdicts (EXP 1–9, EXP_ELO_001's per-turn
diagnosis) are unaffected; they were measured on the training log, not the
ladder. The re-baseline is the first thing that should happen on the repaired
instrument, and until it exists no EXP registered on Aug 18 can be closed out.

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

### EXP_ELO_001 Expected Results
A divergence window: Greedy's score curve breaks away from the net's between roughly turn 8 and turn 20, led by one identifiable sub-metric. Falsifiers: if the gap is uniform from turn 0, the opening work never mattered for strength; if it only appears at endgame, the bottleneck is closing, not economy. Either way the output is the new #1 bottleneck metric.

### EXP_ELO_001 Actual Results
n=32 seeds (64 games), model 37.5% — reading consistent with the ladder band. The score crossover lands in the predicted window (turn 8–9, gap peaking ~turn 16), but the causal chain starts earlier and has a clear shape:

1. **Units first (turn 3–4):** Greedy trains units immediately (3.0 vs 1.8 by turn 4) and never stops — by turn 16 its army value is 30 vs 13 (in its wins: 41 vs 10, then it kills us by ~turn 20).
2. **Expansion stalls after the first village:** first-capture speed is fine (the EXP 2–9 skill is real), but the model reaches a 3rd city in only **39% of games vs Greedy's 81%** — in Greedy's wins it's **20% vs 100%**. The model grabs one village and stops; Greedy runs an expand-forever engine.
3. **SPT follows cities (turn 6–8 on):** 8.4 vs 15.9 by turn 16 — a direct consequence of the city gap, amplified by harvests.
4. **Tech is anti-correlated:** the model out-researches Greedy in every split, including its losses (t24: 17.3 vs 12.1 techs). It converts stars into research (early score!) while Greedy converts them into units and cities. The model's early score *lead* (turns 1–7) is exactly this — buying scoreboard points that don't compound.

**Verdict: COMMITTED (instrument + diagnosis).** The opening-village campaign taught a skill the model has; the game is decided by expansion *continuation* and army production, where it under-invests — plausibly a research-shaped local optimum (tech = immediate score = shaped reward). New #1 bottleneck metric: **third-city rate** (target: ≥0.8 by turn 13, Greedy's level), with army value @ turn 12 as the co-metric. Caveat: per-turn means past ~turn 18 are survivorship-biased (Greedy's wins end ~turn 20, the model's ~turn 24).

## EXP_ELO_002: Hold the greedy anchor until the gauge crosses 50%
*Jul 11, 2026 · pre-registered*
<!-- heading restored Aug 18, 2026: it was lost in an edit, leaving this entry
     running on from EXP_ELO_001's results. Text below is unchanged. -->

The plateau's timing matches the crutch schedule, not a capacity wall: `anchor_frac` starts at 0.25 and decays 0.97^iter to its 0.1 floor by ~iter 30, and the heuristic prior weight decays 0.5→0.1 on the same clock — so from mid-run onward ~90% of games are weak-net-vs-weak-net. Value targets from those games teach "who beats a weak net", not "who beats Greedy". EXP 7 showed the teacher seat was the largest single gain of the campaign; we then removed it on a schedule instead of on a condition, while the model was still below the teacher.

**Change:** the loop holds `anchor_frac` at its starting 0.25 (no decay) while the latest ladder reading vs Greedy is <50%; once a reading crosses 50%, the decay clock starts from that iteration. Heuristic prior weight keeps its existing schedule — one variable at a time.

### Expected Results
Vs-Greedy gauge readings (n=32 every `LEAGUE_INTERVAL` iters) resume climbing: mean of the first 3 post-change readings ≥ the last pre-change window mean + 8pp, and no plateau strikes fire in the first 30 iters. Secondary (from EXP_ELO_001): third-city rate climbs toward Greedy's 0.81. Falsifier: 3 consecutive readings flat within ±5pp of the old mean → REJECT (teacher starvation isn't the plateau; escalate to the capacity trigger from EXP 1/10 or the shaping candidate from EXP_ELO_001's findings).

**Method amendment (Jul 11, pre-readout):** the first stint runs at a REDUCED budget (`-n 16 -k 4`, 20 iters, gauge every 5) — the new fast-experiment tier. Greedy uses no search, so a smaller budget weakens only the net's side: these readings sit at a lower level than the 64/k=16 ladder history and MUST NOT be compared to the 25–34% pre-change band or chained into the canonical Elo curve. Judge this stint within-budget only: the slope across its own readings plus the third-city-rate trend in the training log. A climb at 16 sims = mechanism confirmed (extend/rerun at full budget for the registered +8pp criterion); flat at 16 sims is *weak* evidence against — the search-improvement operator is also degraded at 16 sims — so a null here gets one re-test at 64 before REJECT.

### Actual Results
Run `1783809008`, 80 iters total: 20 at 16/k=4 (readings 30/33/23/27% — flat, as covered by the method amendment), then 60 overnight at the registered 64/k=16. The six 64-sim readings vs Greedy: **31.2, 37.5, 23.4, 35.9, 40.6, 33.6%** (Elo −137 → best −66, ending −118).

Against the registered criteria: first-3 mean 30.7% vs the ~29.5% pre-change window = **+1pp — the +8pp success bar was NOT met**. The falsifier also did not fire (37.5% and 40.6% both broke the ±5pp flat band; plateau strikes 0). Within the run, first-3 → last-3 means rose 30.7% → 36.7% (+6pp, ~1.2σ — suggestive, not conclusive alone).

The behavior curves carry the real signal. Across readings 30→80: the post-t15 city collapse shrank (t15→t25 bleed −0.67 → −0.32/−0.41), SPT@t25 rose 6.3 → ~7.2–8.1, army value@t25 8.0 → ~9.7–10.7, and the t25 score gap roughly halved (−1471 → −547/−878) — with Greedy's own curves *pulled down* at the good readings (the model interfering with a fixed opponent). Value R² dipped 0.72→0.67 while the first-ever 30-turn training data arrived (iters 22–50), then recovered to 0.74 — the late-game distribution was absorbed. Confound to note: the curriculum crossed into the 30-turn stage at iter 16, so "restored anchor signal" and "first late-game training data" are entangled in this window. Also: the 16-sim P2-seat skew did not replicate at 64 sims (P1 77 wins vs P2 52) — artifact/noise.

**Verdict: WATCH — mechanism engaged (value head learning, late-game behavior healing, no plateau), but the strength conversion is a slow climb, below the registered bar.** The anchor hold stays in place (it's condition-gated and the data shows no harm). This outcome is precisely the promotion trigger for EXP_ELO_003 below.

### Queued follow-up — EXP_ELO_003: anchor dose-response (0.25 → 0.4–0.5)
Promoted to a live EXP only after 002 reads out. Trigger: 002 shows a real but slow climb (readings rising but <8pp over 3) → test whether more anchor games speed value-head relabeling. Run with `ANCHOR_FRAC=0.4`–`0.5`, watch vs-Greedy win rate + third-city rate, and watch policy CE for imitation-regression (anchor games record the greedy seat as teacher targets — too high a dose re-anchors the policy to the teacher, whose ceiling we're trying to pass; it also risks overfitting an exploit lane against a deterministic opponent instead of general strength). If 002 outright fails its falsifier, skip 003 — dose was never the variable.

---

*Aug 18, 2026 — the entries below were written after the changes landed, which is
backwards for this protocol and is called out deliberately. Two repair waves fixed
the pipeline and the gauge (see `expert_pipeline_audit.md`) and carried several
behaviour changes with them. Each is registered here as its own EXP so it can be
judged, but **none has been run: no gauge reading exists on the repaired
instrument yet, so none of these has a verdict.** They are all ON by default
except EXP_SEARCH_001, which means one re-baseline reading cannot separate them —
plan the re-baseline accordingly.*

## EXP_LABEL_001: One zero-sum value label (`REL_W` 0.4 → 1.0)
*Aug 18, 2026 · REGISTERED, NOT YET RUN*

Two constants disagreed about whether the value label is zero-sum:
`FINAL_OUTCOME_REL_W = 1.0` ("an absolute own-progress component is NOT
antisymmetric") governed the final-outcome tail, while `reward::REL_W = 0.4`
("abs-dominant: rewards it regardless of the opponent") governed the TD body —
which carries `TD_W = 0.7` (`src/bin/self_play.rs:52`), so 70% of the label was
60% non-antisymmetric. The MCTS backup negates across every player-turn boundary
(`mcts_common.rs`), which is only valid for antisymmetric v, so the absolute
share was being corrupted through every EndTurn-crossing line — mildly in a
2–3 ply tree, worse as search deepens (and EXP_SEARCH_001 deepens it).

This is a **restoration, not a new idea**: notes.md's "Phase-1 training-signal
fixes" (Jul 8) already set both weights to 1.0 for exactly this reason, and 0.4
came back in a later re-import. The measured problem that motivated an absolute
share is real — in mirror play the relative swing nets to ~0 and the label goes
empty (Jul 7–8 decision traces) — and is attacked in the DATA instead, via
greedy-anchor games that make passivity actually lose.

**Change:** one constant, `reward::REL_W = 1.0` (`src/ai/reward.rs:26`), read by
both the TD body (`:47`) and the final-outcome tail (`self_play.rs:560`).
`GOOD_BOT_FINAL_SCORE` (`self_play.rs:49`) is live again as the absolute
yardstick, reachable only by lowering `REL_W`.

### Expected Results
Vs-greedy gauge on the fixed seed set (n=32 seeds sides-swapped) at or above the
pre-change 25–34% band, and `value_r2_holdout` no worse than −0.05 versus the
in-sample series it replaces. Direct co-metric: `vlab_wl_share` (the win/loss
share of the label's magnitude, plotted on the dashboard) rises.

### Falsifier
Three consecutive readings below the lower bound of the pre-change band →
REJECT and restore the split constants. **Check A2b first:** the label still
reads raw `score` (`reward.rs:54`) while training plays Domination, and score was
measured ~8pp worse than unit count at predicting the winner at every turn. A
zero-sum label built on the wrong quantity is not evidence against zero-sum.

### Actual Results
NOT YET RUN.

## EXP_SEARCH_001: Adversarial in-tree search — give the opponent its turn
*Aug 18, 2026 · REGISTERED, NOT YET RUN · off by default*

The single largest behaviour defect found in this repo. `Game::simulate_move`'s
`EndTurn` branch looped `end_turn()` until control came back to the mover,
deleting every opponent turn in between — the comment read "Single-player MCTS:
skip enemy turns". The opponent still collected income and had its units
refreshed; it simply never acted. Measured: `simulate_move(EndTurn)` left player
1 to move while the opponent's stars went 5→7 and its units' `moved` flags reset;
a 60-ply descent across 15 turn boundaries visited only player 1; and production
`self_play` reported 5.06 in-tree `EndTurn` edges per move decision at only 16
MCTS iterations. So the search has always been optimising against an opponent
that banks resources and passes.

The two-player machinery was already built and correct (player-aware sign-flipping
backup, POV-correct leaf features, per-mover edge rewards); only `game.rs`
short-circuited it.

**Change:** `game::adversarial_search()` (`src/game.rs:34`, override
`set_adversarial_search` `:51`, env `POLYFISH_ADVERSARIAL_SEARCH=1`,
`arena --adversarial`) makes the `EndTurn` branch hand over exactly once
(`:390`). Shipped with it, because a naive flip hands control to an opponent with
no army: `clone_for_mcts` (`:258`) confines the in-tree opponent to the root
player's vision, so it plays a **belief-state** opponent — only the units, cities
and tiles the root player can currently see. Sign handling was audited in every
backend (`mcts_zero.rs:57` negates a handover child before comparing siblings via
`mcts_common::edge_hands_over` `:159`; `mcts.rs:72` minimises at opponent nodes;
`gumbel_mcts.rs:1290` refuses to re-root tree reuse across a handover).
`tests/adversarial_search.rs` pins the switch and the handover.

**Default OFF.** Nothing in `run_training_loop.sh` sets it.

### Expected Results
Head-to-head at equal sims, 32 seeds sides-swapped, same weights both sides:
adversarial ≥60% vs non-adversarial. Behaviour co-metrics unchanged or better
(third-city rate by t13, army value @ t12). Cost must be measured on the same
run: a turn of horizon now costs ~2× the plies (`brain.rs:423-426`), so also
compare at equal **wall-clock**, not only at equal sims.

### Falsifier
<55% at equal sims AND no better at equal wall-clock → keep it off, and record
that the null opponent was not the binding constraint. Note the honest weakness
of the fix before running it: a belief-state opponent that can only move what we
can see is a *weak* opponent model, not a correct one — a null result may be
about the belief state rather than about adversarial search.

### Actual Results
NOT YET RUN.

## EXP_SEARCH_002: `max_turns_ahead` honours its `max_turns` argument
*Aug 18, 2026 · REGISTERED, NOT YET RUN*

The in-tree horizon function ignored the argument it was given and hard-coded a
20-turn game: `if turn < 8 { 5 } else { (20 - turn).max(2).min(20) }`. From turn
18 onward it returned the floor of 2 — and the curriculum has been generating
30-turn and 45-turn games since well before that, so for most of every training
game the search was running at its minimum horizon for no stated reason.

**Change:** `(max_turns - current_turn).clamp(MIN_TURNS_AHEAD, MAX_TURNS_AHEAD)`
(`src/ai/brain.rs:430`, bounds at `:421` and `:426`). Monotonically
non-increasing in `current_turn`, never looks past the game's own end.

### Expected Results
At the 45-turn curriculum stage the horizon is 5 for every turn ≤40 instead of 2
from turn 18. The late-game metrics EXP_ELO_002 read as the real signal should
move first: SPT @ t25, army value @ t25, and the t25 score gap. Throughput falls
— quantify it, this buys depth with compute.

### Falsifier
No movement in the late-game metrics over 20 iterations while moves/sec falls
>15% → revert to a cheap constant horizon (the old function's *behaviour* was a
constant 2 in the late game; that is the arm to compare against).

### Actual Results
NOT YET RUN.

## EXP_SEARCH_003: Research actually unlocks inside the search
*Aug 18, 2026 · REGISTERED, NOT YET RUN*

`unlock_tech` set `discovered: state.settings._are_you_sure`, and the simulation
path deliberately never sets `_are_you_sure`. So inside the search, researching a
technology spent stars and granted score but unlocked nothing: no unit, no
structure, no harvest. The agent could never plan "research X, then use X" — only
"research X, then notice the scoreboard went up".

This is a plausible mechanism behind EXP_ELO_001's sharpest finding, that the
model out-researches Greedy in every split including its losses (17.3 vs 12.1
techs by t24) while converting fewer stars into units and cities.

**Change:** `discovered: true` unconditionally (`src/actions/tech.rs:37`). Real
states only ever carry discovered techs, so a simulated research now unlocks what
a real one does.

### Expected Results
Techs @ t24 in self-play falls toward Greedy's ~12, **or** stays high while
units/cities in the same window rise (research that now pays for itself). Either
is a pass; the failure mode is high tech with nothing bought with it.

### Falsifier
Techs @ t24 unchanged and army value / city count unchanged over 20 iterations →
the sim was not what made research look free, and the label is the suspect
instead (points at A2b: tech tier pays raw `score` directly).

### Actual Results
NOT YET RUN.

## EXP_DATA_001: Symmetric training maps
*Aug 18, 2026 · REGISTERED, NOT YET RUN*

Over 500 seeds of the exact Tiny/Drylands configuration `self_play` generates,
player 2 starts on an island with ≤2 land neighbours in 166/500 games (33%) while
player 1 does so in 0/500. Mean land in the 8 tiles around the capital is P1 8.00
vs P2 6.01, and 469/500 seeds give materially different seats. Symmetric mapgen
already existed and was simply never used for training. With `symmetric: true`
every metric is identical and island starts drop to 0/500.

An uncompensated seat advantage puts a seat term in every value label, which the
network can only fit as noise — and notes.md already recorded the symptom without
naming the cause ("p1 vs p2 score gap (~4256 vs 3291) is seat advantage, both
sides were the same model").

**Change:** `self_play --symmetric` defaults to true
(`src/bin/self_play.rs:1342`); `src/mapgen.rs:1712` asserts seat equality over
120 seeds, with the measurement kept as an `--ignored` diagnostic at `:1729`.

### Expected Results
The p1/p2 win-rate and score gap in the training log collapses toward even within
5 iterations. `value_r2_holdout` improves, because the seat term is no longer in
the label.

### Falsifier
p1/p2 gap unchanged over 10 iterations → the imbalance was not the source of it;
revert, since symmetric maps cost map diversity for nothing.

### Watch
Two interactions this does not settle. (1) The **gauge still plays asymmetric
maps** — `arena --symmetric` defaults to false and `run_gauge_match` does not
pass it, so training and evaluation now disagree about the map distribution; fix
before the re-baseline (audit M2). (2) The interaction with `AUGMENT_D4` is
unexamined: every training position now has a 180°-rotational relationship
between the two seats, and what that does to rotation augmentation has not been
measured. D4 is off by default and should stay off here.

### Actual Results
NOT YET RUN.

## EXP_DATA_002: Opening-move temperature — sample π′ for the first 8 plies
*Aug 18, 2026 · REGISTERED, NOT YET RUN*

Move-selection temperature was disabled (`TEMPERATURE_MOVE_THRESHOLD = 0`), so
every game in an iteration opened from the same state with the same deterministic
argmax. That is a duplicated-data problem before it is an exploration problem:
the buffer contains many near-identical opening trajectories, and the policy head
fits them repeatedly.

**Change:** `self_play --opening-temp-moves`, default 8 plies
(`src/bin/self_play.rs:1349`). For those plies the played move is a draw from the
search's improved policy π′ (`sample_opening_move`, `:580`; applied at `:868`);
the argmax resumes afterwards. **The policy target stays π′ either
way** — this changes which state the game visits, not what is learned at it,
which is the AlphaZero/Gumbel convention. A draw that is not legal in the
un-obscured state (the search ran on a fog-obscured clone) falls back to the
argmax.

8 plies is roughly one Polytopia turn (notes.md: ~8 plies to complete one game
turn), so this randomizes about the first turn only.

### Expected Results
Opening diversity within an iteration rises — measured as the fraction of games
sharing an identical first-8-ply command sequence, from ~1.0 to <0.5. No
degradation in the behaviour metrics over 10 iterations: `villages_t2c_first_cond`
within +0.5 turns, captures/game flat or up.

### Falsifier
t2c worsens by >0.5 turns or captures/game falls over 10 iterations → halve to 4
plies, and if that still costs, set 0 and record that the opening argmax was
load-bearing.

### Actual Results
NOT YET RUN.

## EXP_TEACH_001: Army-composition scoring in the greedy teacher
*Aug 18, 2026 · REGISTERED, NOT YET RUN*

EXP_ELO_001 named army production and composition as one of the two behavioural
bottlenecks. Until now every affordable unit scored identically at summon time
apart from a flat +15 for Giant, so the teacher — whose games are 25% of every
self-play iteration and whose ordering also feeds the search priors — had no
notion of an army that fits together.

**Change (`src/ai/scoring.rs`):** summon score gains the unit's meta value
(`UNIT_VALUES × SUMMON_QUALITY_W`, `:331`) plus three composition bonuses keyed
off a `UnitRole` classification (`:21`): a Frontline screen when ranged units
outnumber frontline, Ranged when frontline outnumbers ranged, and Mobile once
Roads is researched (constants `:57-60`, applied at `:349-352`).

This is a **teacher change, therefore a training-data change** — it must be
judged like one, not shipped as a tidy-up.

### Expected Results
Measured on the teacher itself, before any training: `greedy_teacher_behaviour_probe`
(`src/ai/scoring.rs`, `#[ignore]`d, 64 fixed seeds, reports the EXP_ELO_001
metrics) shows army value @ t12 up, and the frontline/ranged/mobile mix at t12
moves off a single role. Third-city rate by t13 unchanged — nothing here targets
expansion, and a drop means the composition bonuses are stealing stars from it.

```bash
cd polyfish-rs && cargo test --release --no-default-features \
    --lib greedy_teacher_behaviour_probe -- --ignored --nocapture
```

### Falsifier
Army value @ t12 flat, or third-city rate down → revert the bonuses. The teacher
is the training data; a worse teacher is strictly worse than no change (EXP 7 is
the precedent: swapping the teacher was the single largest gain of that
campaign).

### Note on the other evaluator edit
`evaluator::economy::penalty_partial_cities` no longer hits a `todo!()` on the
win-condition-less modes (Custom/Sandbox/Tutorial/None) and returns 0.0 instead
(`src/ai/evaluator/economy.rs:129`). That is a crash fix, not a behaviour change:
training runs Domination, whose arm is untouched.

### Actual Results
NOT YET RUN.

## EXP_TRAIN_001: Persistent optimizer and LR schedule across iterations
*Aug 18, 2026 · REGISTERED, NOT YET RUN*

`train.py` built a fresh Adam and a fresh `CosineAnnealingWarmRestarts` on every
invocation, and the loop invokes it once per iteration — so the LR sawtoothed
back to maximum every iteration and Adam's moments were thrown away each time.
notes.md recorded the symptom during the behaviour-cloning work without fixing
it: "each new train.py invocation restarts the cosine LR at max and undoes
fine-tuning (use one long run, or TRAIN_LR lower for continuations)".

**Change:** Adam state and the scheduler step persist in `optimizer_state.pt`,
keyed by `run_id` so a new run starts clean (`train.py:412`, load `:423-457`,
save `:459-466`, wired at `:651` and `:943`). The cosine schedule now spans the
run.

### Expected Results
Policy CE at equal iteration lower and visibly smoother — the per-iteration
sawtooth in the loss series disappears. `value_r2_holdout` up. No change to
throughput.

### Falsifier
Loss-curve shape unchanged over 20 iterations → revert, and note the real risk
being traded away: a stale optimizer state carried across a curriculum stage
change (10→15→30→45 turns) is a distribution shift Adam's moments were fitted
before.

### Actual Results
NOT YET RUN.

## EXP_TRUNK_001: `DETACH_VALUE_TRUNK` — let value gradient reach the trunk
*Aug 18, 2026 · REGISTERED, NOT YET RUN · the verdict audit A1 says was never written*

`run_training_loop.sh:23` exports `DETACH_VALUE_TRUNK=1`. `train.py:44-48`
documents it as bisect Arm D and `bisect_arm.sh:14` treats it as a diagnostic;
CLAUDE.md's own rule is that anything exported unconditionally from the loop is
a production setting. It has been on for the whole recorded history of this fork
with **no entry in this file** — which is precisely the state #12 documents for
`AUGMENT_D4`, where a measured rationale was lost and the setting then read as an
oversight.

Provenance is unrecoverable: this repo is a fork and the switch was set upstream.
So it is settled empirically rather than archaeologically, and this entry exists
so the next reader inherits a registered arm instead of an unexplained export.

**What the switch does.** With it on, no value-loss gradient reaches `conv1`, the
ResBlocks, or the cross-attention. The value head still trains at full strength,
but only on whatever representation the trunk built for move-picking — it cannot
cause the trunk to represent winning-ness at all. MCTS queries that head at every
leaf of every search, so a value head riding as a passenger on policy features
caps search quality no matter how good the policy is. The defensible reason to
have set it: two heads on one trunk can genuinely fight, and a noisy value signal
can degrade the policy. Detaching isolates that.

**Change:** run the arm both ways at equal budget on a fixed seed set. The switch
is now overridable rather than hardcoded (`run_training_loop.sh:23`), so the off
arm needs no edit to the driver:

```bash
DETACH_VALUE_TRUNK=0 ./run_training_loop.sh --new-run -i 20 -l 5
```

### Prerequisite — sequencing, not optional
Run this **after** the A2b label fix, not before. A2b changes the prior on why
the switch exists: detaching only pays if value gradient was actively harming the
policy, and A2b shows the value label is built on a quantity ~8pp worse than an
available alternative at every turn, degrading in the late game. A plausible
history is that the value signal genuinely was harmful, someone correctly saw the
policy suffering, and detaching treated the symptom rather than the cause. Remove
the detach *before* fixing the label and you reproduce the original harm and read
it as "value gradient hurts the trunk".

It also needs a gauge baseline (#1, #2, #4, #5, #7 — all now closed) so the two
arms are read on a common map set.

### Expected Results
With `DETACH_VALUE_TRUNK=0`: `value_r2_holdout` up materially — the trunk can
finally build features for the value question. vs-Greedy win rate up, because
the search's leaf evaluations improve. Policy CE flat or slightly worse early
(the trunk is now serving two objectives) and recovering.

### Falsifier
Policy CE materially worse and vs-Greedy win rate not up after 20 iterations →
detaching was load-bearing. Record that as the verdict and stop treating the
export as an accident.

Read `value_r2_holdout` rather than `value_r2`: in-sample R² rises under either
arm and cannot distinguish them.

### Actual Results
NOT YET RUN.


## Also landed Aug 18 — behaviour-affecting, not separately registered

Correctness and integrity fixes with no free parameter to tune, listed so a
behaviour change is never mistaken for noise:

- **Self-play no longer records samples for moves that never happened.** A
  training sample and a replay entry were pushed *before* `play_move`, so a move
  the engine rejected left a sample for a transition the game cannot reach.
  Nothing is recorded until the move lands, and a rejected legal move discards
  the whole game and increments a new `aborted_games` metric
  (`src/bin/self_play.rs:933`, `:962`, `:2506-2514`).
- **`GameRecorder` refuses to write steps with no outcome** instead of labelling
  them `win = 0.0` (`src/recorder.rs:79-108`). The old constant-0 label trained
  the value head toward a draw on every human/imitation state it covered.
- **`freeze_area` actually freezes, and its undo round-trips**
  (`src/actions/mod.rs:430`; `tests/freeze_undo.rs`). Polaris-only in effect, but
  a non-round-tripping undo corrupts the MCTS tree for everyone.
- **`Infiltrate` no longer exempts a unit from road/terrain movement rules**
  (`src/moves/mod.rs:691`) — it is an attack-targeting skill; the exemptions its
  carriers enjoy come from Creep or Fly. This narrows legal moves for those
  units.
- **Unimplemented structures cannot be built.** `Outpost` is skipped in movegen
  and rejected by `BuildMove::execute`
  (`src/settings/structures.rs:44`), so it can no longer silently burn 5 stars.
- **`arena` searches a fog-obscured clone**, as self-play does
  (`src/bin/arena.rs:306-313`) — a measurement-alignment fix (audit M2), but it
  changes what the graded agent can see, so old and new arena readings are not
  directly comparable.
- **`generate_legal_moves` returns a deterministic order.** Two containers in
  movegen were iterated to emit moves — `compute_reachable_tiles`'s `HashMap` of
  step targets (`src/moves/mod.rs:377`) and `generate_research_moves`'s
  `HashSet` (`src/moves/research.rs:113`) — and Rust seeds each map instance
  separately, so the same position produced the same moves in a different order
  on every run. The *policy* was insulated (`mapper.rs` maps moves to stable
  semantic coordinates precisely because raw ordering was not stable), but the
  **search** was not: root children are zipped with Gumbel draws by index, so a
  permuted list assigns different noise to different moves. This is why no
  search experiment was ever replayable. Ordered now (`BTreeMap`, and a sort
  before emission) and held by `tests/search_determinism.rs`. Behaviour-
  affecting: it changes which move gets which draw and how ties break, so
  self-play trajectories differ from any previous run even at a fixed seed.
- **Cross-attention was wrong for every batch row after the first, on the
  default backend.** `network.rs` fed the attention's `q_proj` a strided view
  (`x.flatten_from(2)?.transpose(1, 2)?`), and `candle_nn::Linear` on that
  layout returns different values from its contiguous equivalent for every row
  but row 0 — position, not contents: it reproduces with all rows identical.
  The eval server batches leaf evaluations by design, so on Linux and CUDA runs
  (candle is the default and the only non-Apple backend) most leaf evaluations
  in every search returned corrupted policy and value. `tch`/`metal` were
  unaffected — libtorch and MPSGraph handle the stride — so an Apple run was
  reading a different network from a Linux run of the same weights.
  **Behaviour-affecting in the strongest sense**: search quality on every
  non-Apple run changes, and any behaviour metric taken from one is not
  comparable with one taken after this fix. Found by the new
  `scripts/run_forward_parity.sh` on its first run; see audit T1 for the
  measurements, including why a batch-invariance test does not catch it.
- **The Gumbel agent owns its RNG.** `GumbelMctsAgent::with_search_seed(u64)`
  pins the stream; `POLYFISH_SEARCH_SEED` pins a base that still differs per
  agent, because a stream shared across actors would make every actor play the
  same game. Previously all three draw sites used the thread-local generator,
  so no test could pin search behaviour (audit T3). Not behaviour-affecting on
  its own — the default path is still seeded from OS entropy.
