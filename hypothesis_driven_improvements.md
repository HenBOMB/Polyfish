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
*Jul 11, 2026 · pre-registered, running*

All our metrics so far measure behavior — capture speed, SPT, policy loss — not strength. This adds the missing y-axis: paired arena matches against frozen reference models, chained into one Elo curve. It's the line that must keep rising before we commit real money to the long cloud run.

### Design (instrumentation, no behavior change)

- **Reading**: n=32 seeds, each played twice with sides swapped (64 games), via the existing `arena` binary — gumbel k=16 at 64 sims (the self-play search config). Win rate = wins + draws/2 over completed games.
- **Ladder rules**: a reading every 10th training iteration against the *active anchor* (a frozen checkpoint file that never changes). Measured ≥80% → freeze the current model as the next anchor, measure the link vs the outgoing anchor at n=64, and switch. Audit block every 50 iters: n=32 vs Greedy plus n=32 vs one retired anchor (rotating) — observed vs chain-predicted win rate flags Elo inflation/cycles.
- **Permanent floor anchor**: the Greedy backend (`--backend2 greedy`), the exact production anchor seat (self_play.rs:1682) — a non-net agent that can't participate in net-vs-net strategy cycles. Elo 0 by definition.
- **Backfill today (no training needed)**: `model_gn_v2` (era start) → `model_checkpoint_iter50_20260710_015335` (pre-fix reference) → current `model.safetensors` (iter 60 of run 1783687051), each vs Greedy plus the informative pairs vs each other.
- Known caveats, accepted for a *relative* gauge: arena plays Perfection scoring at 30 turns on mirror Imperius (Tiny Drylands), while training runs Domination with a tribe mix. If the gauge ever disagrees with dashboard trends, this is the first suspect.

### Expected Results (pre-registered before any match ran)

1. Current model beats Greedy at **≥60%** (it out-benchmarks its teacher on t2c and won the league read +8% on score). If ≥80%, Greedy retires to audit duty on day one.
2. Monotonic ordering vs Greedy: `gn_v2` < `iter50_015335` < current.
3. Current vs `iter50_015335` ≥55% (that checkpoint predates the full EXP 4–8 stack absorption).
4. Transitivity spot-check: current vs `gn_v2` lands within ~±10pp of the chain-predicted win rate (no cycle).

### Actual Results
*(pending)*

**Found while setting up (Jul 11):** arena searched MCTS directly on the real game state instead of a clone — search-time execute/undo leaked into the scored game, corrupting it over a match (ghost harvests, impossible star counts, duplicate monuments; 161 execute errors in the first aborted run). Production self-play was never affected because `Brain::think_decomposed` searches `game.clone()`. Fixed arena to clone the same way. WATCH (engine): the corruption proves some move undo callbacks don't restore state exactly — invisible under clone-based search, but worth an undo-roundtrip fuzz test if MCTS ever goes clone-free for speed.

## Next candidates (write the EXP entry before running)

- **De-censor the rate metric (instrumentation)**: exclude games that end by domination before any village capture from the `villages_first_rate` denominator, and log a `domination_wins` rate column. With the ~3–5% rush games counted honestly, rate should read ~1.0 and the 1.0 pre-registration becomes meetable.
- **Value-head probe**: fixed-holdout convergence probe to attribute the r2 slide (harder data vs worse head vs aux competition). Cheap, decides whether anything needs fixing.
- **Movement ground truth**: dump ordinary games (replay + full decision traces) from the latest model; Verdi labels wasted vs purposeful moments; the labels seed both a blended movement metric and a fixed eval suite scored per checkpoint.
- **Map geometry floor** (user-gated, explicitly not approved yet): 31% of Drylands spawns have no village within Chebyshev 3 (400-map measurement; mean nearest 3.44) → omniscient floor ≈4.4, competent-FOW play ≈5.0–5.5. If cond plateaus ~5.0–5.3 at rate ~1.0, the remaining gap to 4.5 is the map, not the policy — reopen the suburb-guarantee conversation with that data.
