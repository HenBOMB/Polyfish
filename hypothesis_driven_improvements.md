# Hypothesis Driven Improvements

The idea is that we should get more systematic about coming up with a hypothesis for bottleneck on a performance metric, come up with experiments that move it, and either "commit" or "reject" it.

This will be the loop we will run continuously to ensure the Polybot continues to improve and get better, to eventually reach human-level capabilities.

Our #1 objective is to figure out how to get into a smooth learning curve regiment. Once we figure that out and can see more training time leads systematically to better playing from the AI, then we can deploy training regiment on the Cloud and let it run over 5M self-play games to reach human-level performance. We only have one shot at a $1M training run and we cannot waste it.

## Protocol

1. Name the bottleneck metric (currently: **third-city rate** — see EXP_ELO_001).
2. Write the EXP entry **before** running anything: **Description** (hypothesis + exact change), **Expected Results** (a number with a sample size).
3. Measure. Fast loop: n=32 benchmark on a fixed worst-case tribe pair (Bardur+Imperius) with a fixed model snapshot, run in a scratchpad dir. Slow loop: a live training run read through `training_log.csv` / the dashboard. A training run is not an experiment by itself — it's the slow-loop measurement that closes out whichever EXPs shipped into it, recorded as a validation note, not a numbered EXP.
4. Record **Actual Results**, then a **Verdict**: **COMMITTED** or **REJECTED** (rejected changes get reverted the same day). We only keep what the data pays for.
5. Ambiguous outcomes get a **WATCH** verdict with an explicit trigger that promotes them back to a new EXP.

Every EXP entry follows the same four parts: **Description → Expected Results → Actual Results → Verdict.** Keep each part to a few lines; move process/execution detail into Actual Results as a single clause rather than its own section.

Shorthand used below: results quoted as `rate/cond` = fraction of games that capture a first village at all / mean capture turn over the games that did. Benchmarks are greedy-teacher-seat unless marked "production mix" (75% net + 25% anchor, the real self-play blend).

*EXP 1–9 are backfilled on Jul 10, 2026 — they were run before this document existed. From EXP 10 on, entries are written before the experiment runs.*

## EXP 1: Auxiliary training heads (ownership / fog / SPT+5 / opponent tech)
*Jul 9, 2026*

The value head learns from one number per game: who won. Four new training-only heads make the net also predict final territory, hidden enemy units, income five turns ahead, and opponent tech — free supervision that forces it to understand the game, not just guess winners.

### Expected Results
Aux losses trend down over ~10 iters while policy CE stays at its floor and win-MSE doesn't degrade >10% for 5+ consecutive iters. Long-term: better value generalization from a richer trunk.

### Actual Results
Over run 1783687051 (51 iters): fog ~0.039–0.043, spt 0.026→0.032, ownership 0.22→0.26, tech 0.27→0.29 — flat to slightly rising, not falling. But games got longer and richer in the same window (avg moves 495→525, captures/SPT up), so the targets themselves got harder; policy CE fell 2.20→2.08 throughout, no corrosion signature.

### Verdict
COMMITTED. WATCH: shares a trigger with the value-head decline in EXP 10 — if aux stays high while policy/value both fit, that's the trunk-saturation signal (capacity trigger).

## EXP 2: De-censor the first-village metric + log the tribe pair
*Jul 9, 2026*

Instrumentation, not behavior. The first-village metric counted a no-capture game as "captured on the last turn", mixing slow with never; tribe pairs also shift the baseline ~2 turns. Split it into capture rate plus average turn when captured, and log the tribe pair.

### Expected Results
Separate capture *rate* from capture *speed*; make the turn-4.5–5.5 bar directly measurable per tribe pair.

### Actual Results
Revealed the old model's rate was ~0.8 — censoring alone inflated t2c by ~1.5–2 turns, and Oumaji/rider pairs run ~5.5 while warrior pairs ran 7–8. New CSV columns (`villages_first_rate`, `villages_t2c_first_cond`, `tribes`) + dashboard lines.

### Verdict
COMMITTED.

## EXP 3: Deeper search — Gumbel 64 → 256 sims
*Jul 10, 2026*

Early game turns are shallow — only 4–5 plies each — so maybe the bot captures late simply because 64 search sims can't look far enough ahead. We quadrupled the budget to 256 sims to test whether first-village speed is search-bound.

### Expected Results
cond drops toward ~5 if first-capture speed is search-depth-bound.

### Actual Results
n=32, Bardur+Imperius: rate 0.81→1.00, cond 7.9→7.3, throughput 129.7→56.9 moves/s (2.3× wall-clock). Depth fixed the never-capture tail, not speed: a first capture is a multi-turn walk, beyond any single-turn search horizon — priors and the value net have to carry direction between turns.

### Verdict
REJECTED — stay at 64 sims.

## EXP 4: Approach gradient in the expansion evaluator
*Jul 10, 2026*

Nothing rewarded getting closer to a village — only the capture itself paid out. We added a small evaluator bonus that grows as a unit closes in on the nearest visible village, so self-play credits progress along the walk, not just the payoff.

### Expected Results
Shaped self-play credits partial approach before the capture lands, giving the value net a gradient across the multi-turn walk.

### Actual Results
No isolated benchmark — landed as part of the EXP 5–7 stack (results under EXP 7).

### Verdict
COMMITTED as part of the stack.

## EXP 5: Doorstep flight — Chebyshev distances + curiosity damping
*Jul 10, 2026*

Replays showed a unit two tiles from a village walked away half the time. Two bugs: distance used Manhattan math though units move diagonally, and exploration bonuses outbid the last approach steps. Fixed the math and damped curiosity when a capture is within two tiles.

### Expected Results
Large cond drop — this looked like *the* bug.

### Actual Results
Alone: cond ~8.9→8.93, no change. The fix was real but invisible, because the anchor "teacher" wasn't the greedy scorer at all (see EXP 7) — rollout noise drowned the ordering gradient. Included in the post-swap stack measurement.

### Verdict
COMMITTED — necessary, not sufficient. Lesson: verify the change is actually on the measured path before benchmarking it.

## EXP 6: Capture must outrank attack
*Jul 10, 2026*

In the move-ordering scores, the best attack (110) outbid capturing a village (99.8) — so a unit standing on a village would swing at an enemy instead of taking it. We raised capture scores above every possible attack score.

### Expected Results
d=0 always converts to a capture.

### Actual Results
d=0 capture rate 100%; benchmark cond 6.47→6.22 at rate 1.00 (n=32, Bardur+Imperius).

### Verdict
COMMITTED.

## EXP 7: Greedy anchor — replace the rollout-MCTS teacher
*Jul 10, 2026*

A quarter of self-play games use a teacher seat meant to demonstrate good habits. That seat ran a noisy rollout search that drowned out our tuned move ordering — the teacher never taught it. We swapped in the plain greedy scorer, the same scores the net's search priors use.

### Expected Results
Teacher demonstrates 4–6-turn first captures; training data quality jumps.

### Actual Results
Anchor seat: 0.94/8.9 → 1.00/6.47 (n=32, worst-case pair), with EXP 4–5 riding along. Largest single gain of the campaign.

### Verdict
COMMITTED.

## EXP 8: Frontier-resource beacon
*Jul 10, 2026*

Resources only spawn next to villages, so a fruit at the edge of the fog hints a hidden village sits nearby — the cue a human steers by at spawn. We added a pull toward resources that still have unexplored tiles around them.

### Expected Results
Blind-phase exploration steers toward hidden villages instead of random fog; cond drops on maps where no village is visible at spawn.

### Actual Results
Took three tries. v1 (pull toward any resource not explained by a known village) regressed ~1.5 turns: units parked on the fruit when their sight couldn't reach the hidden village behind it. v2 scaled the pull by how open the surrounding area is, fixing the parking. v3 fixed the deeper bug: the capital's own structure "explained away" every fruit inside spawn vision, so the rule became "resource still touching fog within 2 tiles". Final benchmark: 0.97/5.97 (best single result of the campaign) vs 1.00/6.22 for the veto version — a statistical wash at n=32; kept v3 because it encodes the real signal instead of filtering it out.

### Verdict
COMMITTED.

## EXP 9: Stronger center pull (×2)
*Jul 10, 2026*

Villages are denser toward the middle of the map, so when a unit sees no village and no fruit, sweeping harder toward the center might find one faster. We doubled the center-pull weight in the move ordering.

### Expected Results
Faster blind discovery, cond drop.

### Actual Results
cond 6.22→6.39 with a rate dip — the center pull overrode useful local evidence. Reverted same day.

### Verdict
REJECTED — center weight stays ×1.

## Training validation — run `1783687051` (slow-loop readout for EXP 4–8)
*Jul 10, 2026 · not a numbered experiment — the slow-loop measurement that closes out the committed stack above*

Train on the new teacher and shaping for 60 iterations and watch whether the net absorbs it into its own play.

### Expected Results
Capture rate pins ~1.0; cond grinds from ~7 toward the low 5s, ending below the static teacher's own benchmark (6.2 worst pair, 6.58 production mix).

### Actual Results
Cond fell 6.02 (iters 1–10) → 5.40 (11–30) → 5.24 (last 10), at rate ~0.97 throughout. Tribe-controlled, same trend: Imperius↔Kickoo ~6.6→~5.8, Oumaji↔XinXi ~5.0→~4.5. Economy grows too: SPT@10 6.10→7.01; policy CE 2.20→2.08. First strength reading: iter-60 league match vs the pre-fix checkpoint, 5310 to 4907 (+8%, one match, not a win rate).

WATCH items opened here: value_r2 slid 0.701→0.661 over iters 1–51 (games lengthening) then held flat — trigger r2<0.60 or the slide resuming. Rate landed 0.97 vs the 1.0 target — resolved Jul 11 (all misses were Domination wins ending the game before a neutral village could be banked, a censoring artifact, not a real gap). League cadence drought explained by the GN-migration checkpoint quarantine, self-healed at iter 60.

### Verdict
Pre-registration met on speed and on beating the teacher; rate residual resolved same week.

## EXP 10: Strength gauge — the frozen-anchor Elo ladder
*Jul 11, 2026*

All our metrics so far measure behavior — capture speed, SPT, policy loss — not strength. This adds the missing y-axis: paired arena matches against frozen reference models, chained into one Elo curve, with Greedy as a permanent Elo-0 floor anchor (a non-net agent that can't join net-vs-net strategy cycles). *(Rebased Jul 13, 2026: Greedy = Elo 300 — matching Greedy is far above a random-player ~0 floor; all existing `ladder.json` elo values shifted +300. Relative gaps unchanged.)* Reading: n=32 seeds, sides swapped (64 games), `arena` at gumbel 64/k=16, `--gamemode 2`. Ladder rule: a reading every 10th iteration vs the active anchor; ≥80% freezes the model as the next anchor (n=64 link match); audit every 50 iters vs Greedy + a retired anchor to catch Elo inflation/cycles. Backfill today: `gn_v2` → `iter50_015335` → `iter50_220138` → current.

### Expected Results
1. Current beats Greedy at ≥60% (≥80% retires Greedy to audit duty). 2. Monotonic ordering vs Greedy across the backfill chain. 3. Current vs `iter50_015335` ≥55%. 4. Transitivity: current vs the chain-predicted win rate within ~±10pp.

### Actual Results
Backfill (n=32 paired, reading CI ≈ ±9pp): `gn_v2` 3.1% (−600 Elo) → `iter50_015335` 23.4% (−206) → `iter50_220138` 43.8% (−43) → current 25–34% (−110 to −190). Net-vs-net: current beats `iter50_220138` at 53.1% and `iter50_015335` at 73.4%, inside the 63–74% chain prediction — transitivity holds. Pre-registrations #2–#4 met; #1 failed — the net still loses to Greedy ~2:1 while every behavioral metric said "improving." Trend: ~+500 Elo across the era, monotonic at every rung.

### Verdict
COMMITTED. New bottleneck: beat Greedy, not behavior metrics. Graduation target for the next stint: >50% vs Greedy.

## EXP 11: Gauge in the loop — auto-ladder + plateau early-stop
*Jul 11, 2026*

Wire the EXP 10 reading into `run_training_loop.sh`: every `LEAGUE_INTERVAL` iters, arena vs the active anchor, appended to `ladder.json` via `ladder.py`. Early stop: over the last 8 readings vs the same anchor, window means flat-or-down AND slope ≤0 counts one strike; two consecutive strikes ends the run.

### Expected Results
Next stint: readings every 10 iters climb from ~25–34% vs Greedy toward the >50% crossing; no false plateau stop on the way; first anchor freeze at ≥80%.

### Actual Results
It worked, but the model plateaued — unable to beat Greedy enough to earn an anchor freeze, holding around ~25% win rate.

### Verdict
WATCH — succeeded by EXP_ELO_002 (plateau diagnosis + fix).

---

*From here on, experiments are named by track: `EXP_ELO_*` targets the strength gauge (win rate vs the Greedy anchor / Elo curve). Other tracks get their own prefixes as they open.*

## EXP_ELO_001: Loss autopsy vs Greedy — name the mid-game bottleneck
*Jul 11, 2026*

The net opens faster than its teacher (t2c 5.24 vs 6.2) yet loses to it 2:1, with a ~1,600-point average score gap. Hypothesis: Greedy pulls away in a specific mid-game window, and the first diverging sub-metric becomes the new bottleneck metric. Change (instrumentation only): arena learns `--dump-stats-dir` (per-turn score/SPT/cities/units/unit-cost/techs, both sides), read from the standard gauge setup vs Greedy.

### Expected Results
A divergence window between roughly turn 8 and 20, led by one identifiable sub-metric.

### Actual Results
n=32 seeds, model 37.5%. Score crossover lands turn 8–9 as predicted, but the causal chain starts earlier: (1) Greedy trains units from turn 3–4 and never stops — army value 30 vs 13 by turn 16; (2) expansion stalls after the first village — model reaches a 3rd city in only 39% of games vs Greedy's 81% (20% vs 100% in Greedy's wins); (3) SPT follows the city gap (8.4 vs 15.9 by turn 16); (4) the model out-researches Greedy in every split, including losses — it converts stars into research (immediate score) while Greedy converts them into units and cities.

### Verdict
COMMITTED (instrument + diagnosis). New bottleneck metric: **third-city rate** (target ≥0.8 by turn 13, Greedy's level), with army value @ turn 12 as co-metric.

## EXP_ELO_002: Hold the teacher signal until graduation
*Jul 11, 2026*

The plateau's timing matches the anchor-frac decay schedule (0.25→0.1 floor by ~iter 30), not a capacity wall — value targets from mostly weak-net-vs-weak-net games teach "who beats a weak net," not "who beats Greedy." Change: hold `anchor_frac` at 0.25 (no decay) while the latest reading vs Greedy is <50%; decay clock starts once a reading crosses 50%.

### Expected Results
Mean of the first 3 post-change readings ≥ the pre-change window mean + 8pp; third-city rate climbs toward 0.81. Falsifier: 3 consecutive readings flat within ±5pp → REJECT.

### Actual Results
Run `1783809008`, 82 iters: 20 at the cheap 16/k=4 tier (not comparable to the 64-sim history), then 60 at the registered 64/k=16. Six 64-sim readings: 31.2, 37.5, 23.4, 35.9, 40.6, 33.6% (Elo −137→−66, ending −118). First-3 mean 30.7% vs ~29.5% pre-change = +1pp, short of the +8pp bar; falsifier also didn't fire. Behavior curves carried the real signal: post-t15 city collapse shrank (bleed −0.67→−0.32/−0.41), SPT@t25 and army value@t25 both rose, score gap roughly halved. Value R² dipped 0.72→0.67 while the first-ever 30-turn training data arrived, then recovered to 0.74.

### Verdict
WATCH — mechanism engaged (value head learning, late-game behavior healing) but strength conversion is a slow climb, below the bar. Anchor hold stays in place → promoted EXP_ELO_003.

## EXP_ELO_003: Anchor dose-response (0.25 → 0.4–0.8)
*Jul 12, 2026*

EXP_ELO_002 showed a real but slow climb. Hypothesis: more anchor games speed value-head relabeling. Change: `ANCHOR_FRAC=0.4` then `0.8`, each a fresh 20-iter run at the cheap tier.

### Expected Results
Vs-Greedy win rate and third-city rate climb faster than EXP_ELO_002's baseline; watch policy CE for imitation-regression (anchor games are teacher-labeled — too high a dose re-anchors the policy to Greedy's ceiling).

### Actual Results
0.4 arm: 25%, 22% vs the 0.25 baseline's 30/33/23/27. 0.8 arm stopped early by hand. Both arms ran sequentially (0.8 continued from 0.4's ending weights, no shared baseline — a training-progress tailwind that should have favored later arms) yet read flat-to-lower, with the same city-stall shape as the baseline.

### Verdict
REJECTED — anchor dose is saturated at 0.25; the bottleneck is elsewhere. (Fallout fixed: run-start model checkpoints are now automatic.)

## EXP_ELO_004: Does the TD(λ) blend weight subsidize the tech tower?
*Jul 12–13, 2026*

The value target blends 70% TD(λ) (per-step score-delta credit, ~5-turn window) with 30% final outcome. Research pays 100×tier score the instant it's bought; army/expansion pay off over many turns, mostly outside that window. Hypothesis: TD's dense credit over-subsidizes tech relative to compounding assets, holding the model in the tech-tower local optimum from EXP_ELO_001. Change: new `--td-w` flag; compare TD_W=0.2 against the TD_W=0.7 default from a shared starting checkpoint.

### Expected Results
Lower TD_W → lower avg_research, gauge tech curve bending toward Greedy's while cities hold or improve. Falsifier: no difference between arms → TD isn't the lever.

### Actual Results
Two attempts were void (missing `-r` reward-shaping flag, then a sequential non-shared-baseline pair) before a clean test: run `1783877772` (TD_W=0.2) vs run `1783894201` (TD_W=0.7), sha256-verified identical starting weights. Iter-5 reading: TD_W=0.7 scored **48.4%** vs Greedy (best 16-sim reading to date, Elo −11) against **35.9%** for TD_W=0.2, with higher cities, SPT, and score across the board. TD_W=0.7 stayed ahead on every metric at the iter-10 reading too, though both cooled (35.9% vs 25.0%).

### Verdict
REJECTED — opposite direction from hypothesized. More TD weight produced the strongest results yet: its dense per-step credit rewards any scoring event each turn (cities included), so a low weight mostly just starves the value head of training signal rather than fixing a tech bias. The tech-tower symptom likely traces to the absolute score pricing of research in `actions/tech.rs`, not the TD blend weight — candidate lever for a future EXP. `model.safetensors` now carries the TD_W=0.7 (winning) weights forward.

## EXP_ELO_005: Make the label listen — REL_W 0.4 → 0.7, anchor dose back to 0.5
*Jul 13, 2026*

The per-window TD reward algebraically reduces to `Δmy − REL_W·Δopp`, so at REL_W=0.4 a mid-game stretch of falling behind a faster-scoring opponent still labels *positive* (my +350 vs Greedy +600 → +110): the dense signal praises the tech tower even inside the anchor games built to punish it — the punishment arrives on the rel channel and gets outvoted 60/40 at the blend. Change: `REL_W` 0.4→0.7 in `ai/reward.rs` (shared by TD labels and in-tree backup; the same stretch now labels −70, and the label's sign flips whenever `Δopp/Δmy > 1/REL_W`). Stacked change: default `ANCHOR_FRAC` 0.25→0.5 — EXP_ELO_003's dose saturation was measured under REL_W=0.4, when the label mostly ignored the rel channel anchor games feed; a heard channel reopens the dose question, and the EXP_ELO_002 hold-until-50% gate still applies. Unwind order if the falsifier fires: anchor back to 0.25 first (ELO_003 read dose-alone as slightly negative), REL_W 0.7→0.5 second.

### Expected Results
10 iters at the cheap tier (16/k=4, n=64 gauge games), new run from the EXP_ELO_004 winner weights (`model.safetensors` sha256 `48059971fac8…`). Mean of the iter-5/iter-10 gauge readings ≥45% vs Greedy (EXP_ELO_004's winning arm: 42.2%); avg_research flat-to-down while avg_cap_cities and SPT hold or rise; third-city rate in the gauge stats climbing from ~0.4 toward ≥0.55. Falsifier (mirror-noise damage): policy CE stalls/rises or capture rate/t2c regress vs the ELO_004 winner arm → unwind in the pre-registered order.

### Actual Results
Run `1783928687` (start sha-verified). Readings 37.5%/32.8%, mean 35.2% vs the ≥45% bar. avg_research did halve (17–27 → 8–13) — but inside a general activity collapse, not a reallocation: self-play games fell ~500 → ~125 moves **from iteration 1, before any training on the new labels** — builds 55→5, attacks 30→4.5, capital captures ~0.6→~0.03, winner scores halved; league (net-vs-net) games equally short. Root-EndTurn suppression held and sim-EndTurn edges/decision were unchanged (~3.0 vs ~3.3): the net didn't pass more, it stopped selecting builds/attacks/expansion until each turn exhausted its options. Iteration-1 onset pins the cause on the shared **in-tree backup** (REL_W is one const for labels AND `gumbel_mcts`), not the label change this EXP was designed to test; training on the degenerate games then corroded the heads (policy CE 1.77→1.88, value R² 0.67→0.61 — falsifier fired, via the search side). Gauge games stayed non-degenerate (25+ turns) but Greedy's city curve vs this model rose 3.0→3.4 @t25.

### Verdict
REJECTED as executed — and the label-side hypothesis was never isolated: the one-const change broke search behavior first, and that wreckage is what trained in. Same-day unwind: REL_W→0.4, ANCHOR_FRAC default→0.25 (untested rider), `model.safetensors` restored from `run_1783928687_iter1_start` (final weights parked as `tip_1783928687_iter10`). Successor candidate: thread a **label-only** rel weight through `td_lambda_labels` (in-tree stays 0.4) to test "make the label listen" cleanly. Standing finding: a rel-dominant in-tree reward induces immediate hoarding/passivity in self-play — the negamax-asymmetry warning on `FINAL_OUTCOME_REL_W` cuts both ways.

**Correction (Jul 13, evening):** the "standing finding" above is RETRACTED, and the mechanism story in Actual Results is wrong. EXP_ELO_006's confound check exposed the real cause: this launch (and 006's) omitted `ITER_OFFSET=76`, so EFF_ITER 1–10 selected the `iteration ≤ 25` curriculum stage — **10-turn games** (`max_turns` curriculum in self_play.rs). The "instant hoarding" was ordinary 10-turn games misread as passivity: SPT milestones t15+ are all zero (truncation), per-turn activity was normal (~6 moves/player-turn). The in-tree effect of REL_W=0.7 is therefore UNTESTED, not convicted. The reverts stand anyway (bar missed; no evidence in favor), but on those grounds alone.

## EXP_ELO_006: Label-only rel weight — the de-confounded ELO_005
*Jul 13, 2026*

ELO_005's hypothesis, minus its accident: new `--label-rel-w` flag threaded through `td_lambda_labels` (window and terminal returns) so value labels price windows at `Δmy − 0.7·Δopp`, while the in-tree backup keeps the shared `REL_W=0.4` — search untouched, so games keep the ELO_004 behavioral repertoire and the label change can only act through training. Anchor stays at production 0.25 (the ELO_005 dose rider is dropped). Run: `LABEL_REL_W=0.7`, 10 iters at the cheap tier (16/k=4), same starting weights as ELO_004/005 (sha `48059971…`) — three-way comparable.

### Expected Results
Confound check first: iteration-1 self-play must be non-degenerate (avg_moves in the 400–590 band, builds/attacks/capitals at ELO_004-arm levels); degeneration at iter 1 would falsify the "search side broke ELO_005" attribution and aborts the run. Primary: iter-5/iter-10 gauge mean ≥45% vs Greedy (ELO_004 winner arm: 42.2%). Mechanism: avg_research bends down as *reallocation* — avg_cap_cities and SPT hold or rise — with third-city rate toward ≥0.55. Falsifier: value R² sustained <0.60 or policy CE above the ELO_004-arm band while games stay normal → REJECT 0.7; 0.5 becomes the follow-up arm.

### Actual Results
Run `1783947940` (start sha-verified `48059971…`). The confound check fired at iteration 1 — and what it caught was a **launch error**, shared with ELO_005: both Jul-13 launches omitted `ITER_OFFSET=76`, so EFF_ITER 1–10 ran the `iteration ≤ 25` curriculum stage = **10-turn games** (avg_moves ~119; SPT milestones t15+ all zero; per-turn activity normal). ELO_004's arms ran offset into the 30-turn stage — none of today's numbers are comparable to it. All TD labels were computed over 10-turn horizons; readings 18.8%/26.6% mostly measure a 30-turn gauge scoring a net fine-tuned on 10-turn games. Whether `--label-rel-w 0.7` was even active is unverifiable post-hoc (env-driven, not echoed in any log).

### Verdict
VOID — hypothesis untested. Weights restored to `48059971…` (run tip parked as `tip_1783947940_iter10`). Relaunch as ELO_006b: `ITER_OFFSET=76 LABEL_REL_W=0.7 ./run_training_loop.sh -i 10 -n 16 -k 4 -l 5`. Ops fix to land first: echo EFF_ITER, max_turns, and label-rel-w at each iteration start so curriculum stage and label config are verifiable from session.log — two silent-config voids in one day is the process telling us something.

### ELO_006b (clean relaunch): Actual Results
Run `1783956384` (`ITER_OFFSET=76 LABEL_REL_W=0.7`, start `48059971…`), 30 iterations — 10 registered plus an extension to 30 (pre-registered mid-run when the 10-iter window proved confounded by the value-target transition itself). Confound check passed: 30-turn games throughout. Label activation confirmed by the transition signature: value R² opened at 0.545 (head mispricing the re-scaled labels), re-fit to ~0.62 by iter 8 — then stayed pinned at 0.61–0.63 through iter 30, below the ~0.66 pre-change level. Gauge series vs Greedy: 37.5 / 34.4 / 37.5 / 29.7 / 23.4 / 34.4% (iters 5→30, mean ~33%) — flat; the primary (≥45% mean) missed and the extension falsifier (≤40% at iters 15 AND 20) fired. The mechanism never appeared: avg_research 16–26 with no bend, cities ~1.0 flat, and the gauge expansion gap *widened* (model 1.3–1.8 cities @t25 vs Greedy's 3.4–3.8, worse than the 004 era). Policy CE fell 2.05→1.82 but never reached the 004 band.

### Final Verdict
REJECTED — definitively, on 20 post-transition iterations (2× the registered window). The label listened (the transition signature proves the re-pricing reached the head), and what it heard made the value function worse: the R² ceiling dropped ~0.04, confirming the original REL_W comment's mirror-noise warning empirically at the label level — in the ~75% of games that are mirrors, the rel channel is mostly noise, and up-weighting it starves the head of learnable signal. Weights restored to `48059971…` (tip parked as `tip_1783956384_iter30`); the `--label-rel-w` plumbing stays (default = production). Per registration the 0.5 arm is not run. If the direction is ever revisited: the surgical variant — rel-dominant labels in anchor games only, where the channel carries real signal — is the design that dodges the mirror-noise cost. Direction pivots to the greedy-bootstrap plan (→ EXP_ELO_007).
