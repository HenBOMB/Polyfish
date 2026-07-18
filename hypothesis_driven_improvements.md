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

## EXP_ELO_007: Greedy bootstrap — clone the teacher, then release
*Jul 13, 2026*

Label-side levers are exhausted (004 blend, 003 dose, 006 rel weight) and the bottleneck stands: self-play never generates compounding-expansion trajectories, so no label scheme can price them. Change the DATA instead. **Phase 1 clones the teacher**: `BOOTSTRAP=1` makes every training game Greedy-vs-Greedy via the existing `--search-backend greedy` (no Rust changes — Greedy already emits soft policy targets, both seats recorded; TD labels unchanged, now computed over trajectories where expansion wins by conquest). Smoke test confirms the corpus is the missing behavior: 3.5 villages + 1.6 cities + 1.1 capitals/game, conquest endings, ~0.1s/game. **Phase 2 releases**: relaunch without BOOTSTRAP from the cloned tip, anchor 0.25 guardrail. Diagnostic either way: hold → initialization was the blocker; drift back to the tower → the reward is convicted, with a clean artifact to preserve. Infra: BOOTSTRAP skips league/anchor games and the `.anchor_decay_start` write; per-iteration `CONFIG` echo landed (the ELO_006 ops fix).

### Expected Results
**Phase 1**: `BOOTSTRAP=1 ITER_OFFSET=76 GAUGE_GAMES=64 ./run_training_loop.sh -i 20 -n 16 -k 4 -l 5` from `48059971…`. During bootstrap the CSV behavior columns describe the TEACHER's games (iteration-1 snap to the greedy profile = validity check); net-side evidence is only gauge + CE + R², trends only (target distribution shifted, not comparable to the 004 band). Gate to phase 2: any reading ≥50%, or two consecutive ≥45%, by iter 20. Falsifier: all four readings <45% with no upward trend → covariate shift convicted; successor is DAgger relabeling, not more cloning. **Phase 2** (≥10 iters): HOLD within ~5 pts of the exit level → ACCEPT, bootstrap becomes standard; DRIFT ≥10 pts (or self-play reverts to the tower profile) → reward convicted; next EXP targets the objective (anchor-only rel labels first).

### Actual Results
**Phase 1** run `1783979192`, 20 iters. Corpus validity passed at iter 1 (teacher profile: research ~20, builds ~69, capitals ~0.9, 30-turn games — CONFIG echo confirms offset/backend). Gauge: **29.7 → 32.0 → 35.2 → 43.4%** (n=128, accelerating: +2.3/+3.2/+8.2; the iter-5 dip below the 42.2% baseline is the value-transition, R² 0.54→0.58 re-fit). Mechanism confirmed in the net's own gauge play (iter 20 vs the ELO_006b end state): techs @t20 13.0→11.2, SPT 8.2→11.3, cities 1.94→2.37, units @t25 4.9→**7.7** (now above Greedy's 5.6), city gap vs Greedy halved (−1.33→−0.48), avg score gap 1016→560. First run in the campaign where research fell as *reallocation* (cities/units/SPT up) rather than inside an activity collapse. Gate (≥50% or two consecutive ≥45%) not met; falsifier (all <45% with no upward trend) not fired — still climbing at cutoff.

**Extension (pre-registered Jul 14):** resume +10 iters (`BOOTSTRAP=1 ITER_OFFSET=76 GAUGE_GAMES=64 ./run_training_loop.sh --resume -i 10 -n 16 -k 4 -l 5`), readings at 25/30, gate unchanged. If both readings <45% → phase 1 scored PARTIAL (BC ceiling below teacher parity) and we proceed to phase 2 from the best tip anyway — the hold-vs-drift diagnostic needs a stable exit level, not parity.

**Extension actuals (Jul 14, ran to iter 40):** readings 45.3 (it25) → 28.1 → 39.1 → 25.4 (it40). Peak 45.3% at iter 25, then real degradation (post-peak pooled 30.7% vs 44.1% for iters 20+25, ~3.4σ) while on-corpus fit stayed flat (CE 2.38→2.34, R² 0.58–0.60) and Greedy's gauge score rose monotonically (4482→5050): continued BC training past ~iter 20–25 no longer changed teacher-distribution fit — it random-walked the off-distribution states where gauge games live. **BC has an early-stopping point; we passed it.** Ops cost: per-iter checkpoints don't exist, so the 45.3% snapshot is unrecoverable; best surviving clone = iter-20 tip (43.4%, within noise of the peak), restored to `model.safetensors` (sha `a3996362…`); degraded iter-40 tip parked as `tip_1783979192_iter40`. Ops fix landed: the loop now snapshots the model at every gauge reading (`checkpoints/gauge_<run>_iter<i>`).

### Verdict
**Phase 1: PARTIAL PASS.** Cloning works and transfers — mechanism confirmed (research→expansion reallocation in the net's own gauge play), gauge 29.7→45.3 — but plateaus just under teacher parity (the covariate-shift ceiling) and degrades if overtrained. Exit tip = iter-20 clone, exit level **43.4%**. **Phase 2 amended before launch (staged, per discussion): 2a = `ANCHOR_FRAC=1.0`** (every game NN-vs-Greedy; the Greedy seat doubles as a standing ~50% demo/rehearsal stream), HOLD/DRIFT rules unchanged, judged against exit level 43.4% at matched budget (16/k=4). 2b (mirror self-play + anchor 0.25) only after 2a holds. Phase-2 verdict: (pending)

**Phase 2a actuals (Jul 14):** run `1784017302` (start sha-verified = the clone `a3996362…`), 20 iters, ANCHOR_FRAC=1.0. Readings **36.7 / 29.7 / 30.5 / 33.6%** vs exit level 43.4 — HOLD broken at the first reading (−6.7), **DRIFT fired at iter 10 (−13.7)**, then stabilized at the historic ~30–34% attractor. Mechanism is *directed* reversion, not erosion: net's gauge profile @t20 went techs 11.2→**13.1** while cities 2.37→1.94, SPT 11.3→8.1, units 6.8→5.6. Control: the phase-1 BC-overtraining degradation (iter-40, 25.4%) had NO tower signature (techs flat 11.2→11.1, everything else decayed) — so RL drift is re-pricing toward research, distinct from generic clone brittleness. Two more findings: (1) value-head fit *improved* all run (R² 0.50→0.65, vloss 0.75→0.51) while strength fell — the head faithfully learns what the labels say; pricing, not fitting, is the problem. (2) Rehearsal did not protect — the Greedy seat supplied a ~50% demo stream and drift happened anyway.

### Phase-2 Verdict
**DRIFT — the objective is convicted; initialization exonerated** (we handed RL a working expansion policy in the friendliest possible setting and it traded it back for techs). One confound blocks final sentencing of the *labels* specifically: the run-relative VALUE_TRUST ramp (i/30) muzzled the cloned value head (β 0.03–0.33) during exactly the iterations drift set in — and prior+edge-reward search with the value head gated out prefers precisely the in-horizon tech points we observed. The drift may be driven by the in-tree/policy-target channel, not the value labels. → EXP_ELO_008 separates them.

**Correction (Jul 14, hand-run arena verification) — re-baseline everything measured at 16/k=4; win-rate DRIFT retracted, behavioral reversion stands.** Manual arena on the sha-verified clone at the gauge budget (16/k=4): 46.5/128 + 46/128 + 181/512 → **35.6%** [32.2–39.0]. The in-loop 43.4% (same model, same budget, n=128) was a +1.8σ high draw we then *selected on* — winner's curse: the extension and the exit tip were both chosen on max single readings, so 45.3% is equally suspect. Same clone at **128/k=16**: **44.7%** (286/640, [40.9–48.6]), score gap halved (−999 → −419) — a +9.1 pt budget dose-response (z≈3.5). Eval strength scales with search; the numeric match between 43.4@16/4 and 43.0@128/16 is coincidence across budgets. Knock-ons: (1) phase-1 win-rate gain vs the ~32.8% pre-clone attractor is +2.8 pts, n.s. — the solid phase-1 result is the behavioral transfer, not the gauge climb; (2) BC "post-peak degradation" downgraded to plateau-by-~iter-15 (post-peak pooled 30.9% vs 35.6%, 1.6σ); (3) phase-2a **DRIFT on win rate retracted** — pooled 32.6% [28.6–36.7] vs 35.6%, z=1.1 — while the directed behavioral reversion stands on tight-SE per-turn stats (techs@t20 11.2→13.1, cities/SPT/units down). Revised phase-2 verdict: the objective demonstrably re-prices *behavior* back toward tech; whether that costs strength is UNDETERMINED at gauge power. Protocol from here: a single n=128 reading carries a ±8.6 pt 95% CI — verdicts require pooled readings + behavioral endpoints. Pending sentencing measurements at 128/16, n=256 seeds: drifted 2a tip (`tip_1784017302_iter20`) vs the clone's 44.7%, and pre-clone baseline (`run_1783979192_iter1_start`, sha `48059971…`) vs same — the true price of drift and the true gain of cloning.

## EXP_ELO_008: Phase 2a with the value head in charge
*Jul 14, 2026*

Rerun of phase 2a from the same clone with BOTH muzzles off the value head: full trust from iteration 1 (`VALUE_TRUST_RAMP_ITERS=1`, vs 2a's β 0.03–0.33) and real search budget (`-n 128 -k 16`, vs 2a's 16/4 — the Jul 14 dose-response showed 16 sims can't cash in what the head knows: same clone 35.6% @16/4 → 44.1% @128/16). If behavior still reverts with the head fully in charge, the labels are convicted cleanly; on success both changes standardize anyway, so trust-vs-budget attribution is deferred. Launch: `ANCHOR_FRAC=1.0 ITER_OFFSET=76 GAUGE_GAMES=64 VALUE_TRUST_RAMP_ITERS=1 ./run_training_loop.sh -i 20 -n 128 -k 16 -l 5` from `a3996362…` (arena-measured ~1.1 games/s at this budget → ~4 min/iter; in-loop gauge readings land at 128/16, directly comparable to the 44.1% [40.6–47.7] n=768 benchmark).

### Expected Results
Behavior-primary, judged against the clone@128/16 baseline (n=128 stats dump, `replays/gauge_stats/manual_clone_128_16`): @t20 cities 2.33, units 6.9, SPT 12.7, techs 12.0 (note: at real budget the clone plays MORE techs and more SPT than at 16/4 — richer line, score gap −236). PROTECTED: pooled iter-10/15/20 dumps hold techs@t20 ≤12.5 with cities ≥2.2 and SPT ≥12 → 2a's reversion was the muzzled-value-head channel (trust ramp + shallow search); labels exonerated; standardize both changes and proceed to 2b (mirror self-play). REVERTED: techs@t20 ≥13.0 with cities ≤2.1 (tower signature) → label pricing convicted beyond remaining doubt — drift survived full trust AND real search → label surgery next (TD_W toward outcome and/or anchor-only rel), clone preserved. Secondary (win rate): pooled iters 10–20 (n=384) vs 44.1% [40.6–47.7]; only pooled ≤38% (~2σ) counts as DRIFT — single n=128 readings carry ±8.6 pt CIs. Judge on iters 10–20, not iter 5 (head opens mispriced on the new label scale, R² 0.50).

### Actual Results
**Interim (Jul 14, first 20 iters; resume in flight):** run `1784027745`. **Iter 5: 57.0% — first reading ever above 50% vs Greedy** (73W-55L, Elo 549; z≈2.7 vs the 44.1% clone benchmark — real, not noise; snapshot: `checkpoints/gauge_1784027745_iter5.safetensors`). Peak behavior @t20: techs **14.9** AND cities 2.47 / SPT 12.3 — it out-teched Greedy while matching its expansion; not a tower. The ≥50% crossing auto-wrote `.anchor_decay_start` (eff 81, the ELO_002 rule), so anchor frac decayed 0.97^n from iter 6 — mirror share 14% / 26% / 37% at iters 10/15/20: **the run self-transitioned from 2a into an early 2b**. Readings slid 49.2 / 43.8 / 44.5% — back TO the clone benchmark (pooled 10–20: 45.8% vs 44.1 [40.6–47.7]), not below — while behavior re-drifted in step with the mirror mix-in (cities 2.47→2.07, SPT 12.3→10.5, techs held ~14.4; Greedy's own SPT vs the net rose 13.4→17.0). Scoring: the registered 2a dichotomy applies only to the pure window (iters 1–5) → **with the value head in charge the model improved past the clone; labels exonerated in the vs-Greedy regime**. Iters 10–20 are 2b evidence, not 2a: mirror data correlates with behavioral reversion; strength cost so far = only the delta above the clone. Resume (iters 21+, anchor ~0.61 and falling) extends the 2b test — mirror drift is confirmed if wr drops below 40.6% (benchmark CI floor) or cities keep sliding with techs ≥13. Ops note: `.anchor_decay_start` is global and write-once — delete it before any future launch that needs anchor pinned at 1.0.

**Update (iter ~25 of the resume):** `.anchor_decay_start` (content: 81) deleted mid-run at user request — anchor returns to 1.0 from the next selfplay iteration. Data regimes for run `1784027745`: iters 1–5 pure 2a (anchor 1.0), 6–24 decaying mirror mix (0.97^n, reaching ~44% mirror), ~26+ pure 2a again. Caveat: any future gauge reading ≥50% re-arms the clock (rewrites the file; the write-gate can't be edited while the run is live) — delete the file again after a crossing, or gate the write with an env flag post-run.

**Post-deletion (pure 2a restored, iters ~26+):** readings 43.0 (it25) / 43.8 (it30) / **38.3 (it35)** — iter 35 breaks the registered 40.6% floor, and both post-restore models trained on a 100% anchor diet (CONFIG confirms decay clock never re-armed). Behavior slid monotonically across all three regimes: @t20 cities 2.47→1.81, SPT 12.3→8.7, units 6.9→4.4, techs pinned 14.0–14.6; Greedy vs the net now develops unopposed (3.28 cities, SPT 19.0). The within-run A/B is decisive — mirror share went 0% → 44% → 0% and the slide never inflected: **mirror data refuted as the driver**.

### Verdict
**REVERTED — the labels are convicted.** The behavioral rule (techs ≥13 with cities ≤2.1) fired at iters 20/25/30/35 — two of those in the restored pure-2a regime — and the win-rate floor broke at iter 35. Drift survived full value trust, 128/16 search, and a 100% anchor diet: every non-label suspect (initialization, trust ramp, search budget, mirror data) is now individually eliminated by direct experiment, while the same pipeline produced improvement only while the head still carried the clone's pricing (iters 1–5, 57%). Mechanism consistent with 2a's faithful-student pattern: dense TD credit prices in-window tech points, every won-with-tech game re-prices tech upward, expansion decays, and the feedback only stops when the model is weak enough to lose. Registered consequence → label surgery (EXP_ELO_009). Artifact: `gauge_1784027745_iter5.safetensors` (57.0%, Elo 549 — best model on record), to be restored to `model.safetensors` once the run stops.

## EXP_ELO_009: Outcome-only value labels from the 57% artifact
*Jul 14, 2026*

008 convicted the dense TD channel: at TD_W=0.7 the value target pays credit to in-window tech points regardless of what wins the game (the +110 falling-behind algebra). Surgery: **`TD_W=0`** — value target = final outcome only — the maximal-contrast test of the dense channel. EXP_ELO_004's shaping-beats-flat result doesn't govern here: that was from-scratch training at 16/4, this is fine-tuning an already-strong model whose dense credit is the identified poison. Start: `gauge_1784027745_iter5.safetensors`. Ops prereqs — DONE (Jul 14, post-run): 008 ran to iter 40 (tip = `gauge_1784027745_iter40.safetensors`), `model.safetensors` restored from the iter-5 artifact (sha `9e4a7b6b…`, byte-verified), and the `.anchor_decay_start` write is now gated behind `NO_ANCHOR_DECAY=1` in `run_training_loop.sh` (file remains deleted; without the gate the start model's first ≥50% gauge would re-arm mirror decay and re-contaminate). League iterations (~every 5th) remain as constant background in all arms.

### Expected Results
Launch: `NO_ANCHOR_DECAY=1 ANCHOR_FRAC=1.0 ITER_OFFSET=76 GAUGE_GAMES=64 VALUE_TRUST_RAMP_ITERS=1 TD_W=0 ./run_training_loop.sh -i 20 -n 128 -k 16 -l 5` from the 57% artifact. Dense channel confirmed as driver: behavior holds or re-expands (cities ≥2.3 / SPT ≥12 @t20; high techs alone are fine — the iter-5 model teched 14.9 and won) and pooled readings 10–20 hold ≥ the 44.1% benchmark, ideally ≥50% → ACCEPT outcome-only labels for post-clone fine-tuning. Dense channel exonerated: cities ≤2.1 with SPT ≤10 recurs even on pure-outcome labels → the remaining suspect is the policy-target/in-tree channel (visit distributions shaped by per-edge instant rewards) → next lever is in-tree reward scaling, not labels. Reading discipline: pooled + behavioral endpoints; single readings carry ±8.6 pts.

### Actual Results
Run `1784049388`, 20 iters at 128/16, TD_W=0, anchor 1.0 throughout (every CONFIG line iters 1–20 confirms `td_w=0`, `--anchor-frac 1.0`, decay exponent 0). Interrupted after iter 10 and resumed at iter 11 (`run_1784049388_iter11_start`); the only casualty was the iter-10 gauge (empty stats dir; the AddrInUse panic in the log is just the backend server, harmless). NO_ANCHOR_DECAY gate untested (no reading hit 50%) but `.anchor_decay_start` stayed absent.

Gauges: iter 5 = 61W-67L **47.7%**, iter 15 = 50W-78L **39.1%**, iter 20 = 49W-79L **38.3%**. Pooled 15–20: 99/256 = **38.7%** [32.7–44.6] — below the 44.1% clone benchmark, nowhere near 50%. Behavioral @t20 (net vs greedy): iter 5 cities 2.33 / SPT 10.5 / techs 16.0-vs-11.7; iter 15 cities 2.13 / SPT 10.3; iter 20 cities **1.95** / SPT **8.7** / units 5.0-vs-9.5 / techs 14.4-vs-11.8. The exoneration condition (cities ≤2.1 with SPT ≤10) fired at iters 15 and 20. Trajectory is 008's slide replayed on outcome-only labels — both runs bottom at exactly 38.3%. Value loss rose 0.733→0.84 with R² 0.53→0.63 (noisier outcome labels, learned faithfully) — the head is again a healthy student of whatever it's fed while strength falls.

### Verdict
**EXONERATED — outcome-only labels did not stop the drift.** The tower signature (tech lead held while cities/SPT/units collapse) recurred at the same speed and to the same floor as 008 with the dense TD term surgically removed from the value target. The value-label channel is eliminated as the driver. Last remaining dense-credit channel: **in-tree per-edge shaped rewards** — they steer MCTS visits, visits are the policy target, so the policy can be taught to tech with no help from the value labels at all. Registered consequence → EXP_ELO_010. Secondary note: the 57% reading now looks partly winner's-curse — two fine-tunes from that artifact each read ≤49% within 5 iters — but the slide to 38–39% with a collapsing expansion profile is real regardless of where the artifact truly sits.

## EXP_ELO_010: Kill the in-tree reward channel
*Jul 14, 2026 — registered, not yet run*

009 leaves exactly one dense-credit path standing: reward shaping inside the search (per-edge instant rewards biasing MCTS visit counts → policy targets). Surgery: **`-r`** (disable reward shaping entirely) — no in-tree shaping, and value labels fall back to flat final-outcome (TD_W moot). Same start artifact, same budget, shorter horizon since both prior runs slid within 5 iters.

Ops prereq — DONE (Jul 15): `model.safetensors` restored from `checkpoints/gauge_1784027745_iter5.safetensors` (sha `9e4a7b6b…`, byte-verified); the 009 iter-20 tip it replaced was hash-identical to the parked `gauge_1784049388_iter20.safetensors`, so nothing was lost.

### Expected Results
Launch: `NO_ANCHOR_DECAY=1 ANCHOR_FRAC=1.0 ITER_OFFSET=96 GAUGE_GAMES=64 VALUE_TRUST_RAMP_ITERS=1 ./run_training_loop.sh -r -i 10 -n 128 -k 16 -l 5`. In-tree channel convicted: cities ≥2.3 / SPT ≥12 @t20 hold and pooled gauges 5–10 ≥44.1% → the fix for post-clone fine-tuning is shaping-off (or heavily scaled-down) search. In-tree channel exonerated too: cities ≤2.1 with SPT ≤10 recurs → no dense-credit channel is left and the dense-credit hypothesis is dead; the suspect becomes the data regime itself (64 games/iter of 100%-vs-greedy fine-tuning degrading a peak model regardless of labels — test via bigger -g, or accept the artifact as the campaign result). Reading discipline: pooled + behavioral endpoints; single readings ±8.6 pts.

### Actual Results
Run `1784069067`, 10 iters at 128/16, `-r` confirmed live (banner: "⚠️ Reward shaping disabled (-r): flat final-outcome value target only"), anchor 1.0 throughout, eff_iters 97–106. Gauges: iter 5 = 42W-86L **32.8%**, iter 10 = 48W-80L **37.5%**; pooled 90/256 = **35.2%** [29.3–41.0] — below the 44.1% benchmark. Behavioral @t20: iter 5 cities 1.95 / SPT 8.9 / units 5.3-vs-9.4 / techs 14.1-vs-11.7; iter 10 cities 1.97 / SPT 9.2 / units 5.4-vs-9.9 / techs 14.5-vs-11.7 — the exoneration condition (cities ≤2.1 with SPT ≤10) fired at both gauges. The iter-5 reading (32.8%) is the lowest of any fine-tune at that point (009 read 47.7% at iter 5): shaping-off didn't slow the slide and may have steepened it, directionally consistent with EXP_ELO_004.

### Verdict
**EXONERATED — the dense-credit hypothesis is dead in its entirety.** The tower recurred with no shaping anywhere: no in-tree per-edge rewards, no TD labels, flat final-outcome value targets only. Campaign tally of individually eliminated suspects: initialization, trust ramp, search budget, mirror data, value labels (009), in-tree shaping (010). What all three slides share is the **regime itself**: fine-tuning a peaked model on 64 games/iter of 100%-vs-greedy data with train.py's from-scratch hyperparameters — Adam at `TRAIN_LR=0.002` with **fresh optimizer state every iteration**, 2 epochs over a ~10-file replay window. That optimizer shock moves weights hard regardless of what the labels say. Second confound now unavoidable: the artifact's 57.0% was a single n=128 reading (±8.6) and its true strength was never measured — if it sits at ~45–48%, part of every "slide from 57" is baseline error, not degradation.

## Discovery (Jul 17): the value label was never win/loss — it's a score ratio

Code audit of `self_play.rs` prompted by "did we search the wrong family?": the label computation (`self_play.rs` ~2048, `relative_outcome`) is `clamp(3·(my−opp)/(my+opp), −1, 1)` from final scores, for **every game, in every experiment run to date** — the comment above it claiming "winner +1.0 / loser −1.0, score differential only on timeout" describes a branch that does not exist. `-r` (010) only removed the TD term; its "flat final outcome" fallback is still 100% score-derived (`FINAL_OUTCOME_REL_W = 1.0`).

Consequences: (1) tech = +100×tier instant risk-free score (`functions.rs:1095`) while a city's larger payoff is delayed past what a 30-turn cap credits; (2) a typical timeout loss (4257 vs 4676) labels **−0.14**, near-draw — the objective literally rewards "stay close in score, don't risk winning," which tech buying achieves most cheaply; (3) this resolves the standing paradox of R² climbing (0.50→0.63) while win rate slid — the model was getting better at its actual (wrong) objective; (4) 008/009/010 verdicts stand (dense *credit* isn't the driver) but every experiment varied how score was delivered, never that the signal *is* score. The policy gets expansion demonstrations from anchor teacher data; the score-fit value head vetoes them at the Gumbel root — match-but-never-beat is the equilibrium of this objective.

## EXP_ELO_011 (revised): win/loss value labels
*Jul 17, 2026 — registered; supersedes the fine-tune-LR version of 011 (LR test deferred as hygiene, candidate 012). Artifact audit (Step 1 below) remains registered and worth running any time.*

**Hypothesis:** the tower and the fine-tune slides are driven by the score-ratio value target; with a true ±1 win/loss label a close loss finally costs −1.0, the value head learns "expansion → win," and the tower fades.

**Surgery:** new `--wl-labels` flag in `self_play.rs` — `final_outcome = ±1` from the adjudicated winner (`GameResult.winner_id`: sole survivor, else higher score at timeout — same adjudication the gauge already uses), replacing the score ratio. Wired through `run_training_loop.sh` via `WL_LABELS=1` (banner: "⚖️ Win/loss value labels"; `wl_labels=` echoed in CONFIG lines). Everything else identical to 010, including `-r` — one variable vs 010. Ops prereq — DONE (Jul 17): 010 tip verified parked (`model.safetensors` was hash-identical to `gauge_1784069067_iter10.safetensors`, sha `1e340a73…`), then artifact restored to `model.safetensors` (sha `9e4a7b6b…`, byte-verified); `self_play` rebuilt clean with `--features apple`; `.anchor_decay_start` still absent.

**Launch:** `WL_LABELS=1 NO_ANCHOR_DECAY=1 ANCHOR_FRAC=1.0 ITER_OFFSET=96 GAUGE_GAMES=64 VALUE_TRUST_RAMP_ITERS=1 ./run_training_loop.sh -r -i 10 -n 128 -k 16 -l 5` (ITER_OFFSET=96 as in 010, same curriculum/eff_iters 97–106).

**Expected if right:** tower signature fades (cities ≥2.3 / SPT ≥12 @t20) and gauges hold or climb vs 010's 32.8/37.5 (pooled 35.2). Note: vloss/R² will re-baseline (±1 targets are a different distribution) — do not read early vloss jumps as regression. **Falsifier:** tower recurs and pooled gauges ≤ 010's level → the labels family is fully dead (delivery *and* content), suspects narrow to search budget/horizon or the data diet; the LR/optimizer-shock test then runs as 012.

**Step 1 (unchanged) — AUDIT the artifact (manual arena, no training).** 256 seeds (=512 games) at 128/16, artifact vs greedy, record as ladder audit (`--kind audit`). Re-baselines every endpoint: the "hold ≥44.1%" bar was the *clone's* pooled level, not the artifact's. Truth ≈44–48% → the 57 was winner's curse; ≥52% → fine-tuning genuinely destroys points.

### Actual Results
Run `1784251112`, 10 iters at 128/16. Both banners live (`-r` + "⚖️ Win/loss value labels"); every CONFIG line `wl_labels=1`, anchor 1.0, eff_iters 97–106 — identical to 010 except label content. Gauges: iter 5 = 50W-78L **39.1%**, iter 10 = 54W-74L **42.2%**; pooled 104/256 = **40.6%** [34.6–46.6]. First fine-tune that *climbed* between gauges (009 slid 47.7→38.3, 010 sat at 32.8/37.5); both readings beat 010's matched-iter readings (+6.3/+4.7; pooled +5.4, z≈1.3, not individually significant). Behavior @t20: iter 5 cities **2.56** / SPT **12.2** / units 6.6 vs 8.8 (passes the fade bar ≥2.3/≥12); iter 10 cities **2.22** / SPT **10.7** (just under the fade bar, above the tower-fire condition ≤2.1/≤10). **Tower signature fired at neither gauge — first time in the campaign.** Tech lead persists (14.8 vs 12.0 @t20) and greedy still out-expands, so improved-not-cured. vloss ~1.02–1.09 / R² 0.51→0.67 re-baselined as predicted (±1 targets) — not comparable to score-label runs.

### Verdict
**SUPPORTED (directionally).** Neither falsifier condition fired: gauges climbed vs 010 and the tower signature was absent at both gauges. This is the first of four fine-tunes (008/009/010/011) that didn't slide, and the only variable was the label content — consistent with the score-ratio label being the tower's driver. Caveats: pooled win-rate gain over 010 is within noise on its own (the behavioral non-firing is the stronger independent evidence); iter-10 behavior dipped below the fade bar (2.22/10.7) — watch for slow drift; still ≤ the 44.1% clone benchmark (within CI) and the artifact audit remains unrun. Consequence: **win/loss labels adopted for fine-tuning** (`WL_LABELS=1` standard). Registered next → EXP_ELO_011-EXT: resume the same run (`--resume 1784251112`, same env incl. `WL_LABELS=1`, +10 iters, ITER_OFFSET unchanged; gauges at 15/20) to test whether the climb continues through the 44.1% benchmark or stalls; falsifier: gauges regress to ≤010 pooled level or tower fires → the climb was noise and the label family closes as "necessary but not sufficient."

## EXP_ELO_011-EXT
*Jul 17–18, 2026 — ran as new run `1784299487` (the `--resume` didn't take), but functionally the registered extension: start checkpoint hash-identical to the 011 iter-10 tip (`ff332607…`), `wl_labels=1` in every CONFIG line, anchor 1.0, ITER_OFFSET=96 (eff_iters restart at 97 — no mechanical effect: curriculum flat above 75, anchor decay gated off). 20 iters instead of 10 → cumulative 30 WL fine-tune iters from the artifact.*

### Actual Results
Gauges: iter 5 = **32.8%**, iter 10 = **41.4%**, iter 15 = **36.7%**, iter 20 = **26.6%** (34W-94L — worst reading of the campaign); pooled 176/512 = **34.4%** [30.3–38.5]. Tower @t20: iter 5 = 1.97/9.1 (fired), iter 10 = 2.18/10.0 (grazed), iter 15 = 1.91/9.1 (fired), iter 20 = **1.77/8.1 vs greedy 3.71/21.5** — fired, widest expansion gap ever recorded (clone baseline: 2.33/12.7). Tech lead intact throughout (14.5 vs 11.5 @iter 20). Training metrics flat: CE ~2.26, vloss ~1.02, R² 0.67–0.69 all 20 iters — the value head fits the WL labels fine while behavior degrades.

### Verdict
**FALSIFIER FIRED on both prongs** (pooled 34.4 ≤ 010's 35.2; tower fired at 3 of 4 gauges). The 011 climb was noise; 011's "SUPPORTED (directionally)" is **overturned**. The label family is now closed in its entirety — delivery (009), in-tree shaping (010), and content (011+EXT) all eliminated: the tower re-emerges with *zero* score signal anywhere in the training target. Campaign synthesis: all 11 fine-tune gauge readings (009/010/011/EXT) oscillate around **~37.6%** with swings consistent with n=128 noise plus instability — no fine-tune regime has produced net progress, and the reproducible signal is *behavioral* (convergence to the tower under every label regime), not the win-rate wiggles. Surviving suspects, sharpened: (a) **optimizer shock** — fresh Adam @ 2e-3 every iteration is consistent with the observed volatility (41.4→26.6 in 10 iters); (b) **the data diet** — ANCHOR_FRAC=1.0 vs the heuristic anchor with the model losing ~60%+ means model-side samples are mostly labeled −1 and the policy distills its own losing play, a loop with no self-correcting pressure (AlphaZero's mirror self-play pins the label distribution at 50/50 by construction; anchor games don't); (c) **the unaudited baseline**. **The artifact audit is now gating**: truth ≈44 → fine-tuning is actively destructive (−7 pts + tower) and 012 = LR/optimizer test; truth ≈37–40 → the "57% peak" never existed, nothing was ever lost, the campaign closes as "this loop cannot improve the artifact at this data scale," and the frontier moves to loop design (data diet/scale), not fine-tune forensics.

## First-principles session (Jul 18): why anchor-only data can't teach the fork

Mechanism worked out from code + campaign data, motivating EXP_ELO_012. (1) The net learns V(state), not Q(state, action): anchor games pin V along two disjoint trails — (model states, mostly −1) and (greedy states, mostly +1) — but the tech-vs-expand decision needs a *ranking of one-move-apart successors of the model's own states*, which no anchor game ever varies: greedy always expands from greedy states, the model always towers from its own, so the contrast at the model's forks is never generated. (2) The win/loss signal becomes state-visible only ~50+ plies after the fork (early states near-identical with opposite labels; late states distinct — that's what R² certifies), while search reaches ~5–10 own-plies. Confirmed structural facts: the MCTS is **single-player** — `simulate_move` skips all enemy turns ([game.rs:263-296]), so search contributes within-turn sequencing only and zero adversarial lookahead at any budget (consistent with the 16/4→128/16 dose-response raising strength 35.6→44.1 while never touching the drift); the policy loss is plain soft-CE with **no outcome weighting** (train.py:241-247) — outcome reaches behavior only via value→search→visits, so when search can't cash the value signal, distillation photocopies the prior and any residual tilt compounds (the observed ratchet: clone 2.33 cities @t20 → 1.77 after 30 anchor iters, same drift sign in every regime). (3) 008 does NOT cover mirror-as-repair: it refuted mirror as the *cause* of drift (slide identical at 0%→44%→0% mirror share) under **score-ratio labels**, where two similar mirror games label ≈0 for both seats — no contrast. Mirror + WL labels (every game a hard ±1 pair on the model's own state distribution) is untested → EXP_ELO_012.

## EXP_ELO_012: Mirror self-play with win/loss labels
*Jul 18, 2026 — registered before running*

**Hypothesis:** the missing ingredient is outcome contrast at the model's own decision points. Mirror games under WL labels manufacture it: both seats on the model's state distribution, deviations from Gumbel root noise + per-game map/tribe variety (no temperature knob exists), labels ±1 and ~50/50 by construction (`--wl-labels` adjudication — sole survivor, else higher score at timeout — applies to mirror games unchanged). **Single change vs 011/EXT: ANCHOR_FRAC 1.0 → 0.0** (all selfplay games net-vs-net, both seats recorded; league every 5th iter remains as background in all arms; the in-tree heuristic prior blend stays at its starting rate, same clock as all arms — it keeps greedy-preferred moves in mirror candidate sets; DETACH_VALUE_TRUNK=1 stays ON for series comparability, noted as a known limiter on value-side feature carving). Deliberate deviation: the gating artifact audit is still unrun — user chose to test the mechanism first; strength endpoints below inherit the baseline uncertainty.

Ops prereq — DONE (Jul 18): `model.safetensors` restored from `checkpoints/gauge_1784027745_iter5.safetensors` (sha `9e4a7b6b…`, byte-verified); the EXT iter-20 tip it replaced was hash-identical to parked `gauge_1784299487_iter20.safetensors` (`b9e6e3aa…`), nothing lost.

### Expected Results
Launch: `NO_ANCHOR_DECAY=1 ANCHOR_FRAC=0.0 ITER_OFFSET=96 GAUGE_GAMES=64 VALUE_TRUST_RAMP_ITERS=1 WL_LABELS=1 ./run_training_loop.sh -r -i 10 -n 128 -k 16 -l 5` from the artifact. Mirror self-play ≈2× NN eval load per game (both seats) — expect a slower self-play phase than anchor runs. **Primary endpoint is drift direction (behavioral), not win rate**: the carrier model predicts the sign flips. SUPPORTED: cities ≥2.3 AND SPT ≥12 @t20 at BOTH the iter-5 and iter-10 gauges (references: artifact iter-5 peak 2.47/12.3, clone 2.33/12.7) — no prior regime ever held twice. FALSIFIED: tower fires (cities ≤2.1 with SPT ≤10 @t20) at any gauge by iter 10 → mirror+WL is insufficient at this scale; before fully convicting the mechanism, check mirror-game deviation (if twin games are near-deterministic, exploration was inadequate — weakened test); surviving suspects revert to optimizer shock (LR test) and data scale. Secondary (strength): gauges vs greedy at 128/16 as usual; removing teacher data may cost a few points short-term — hold = pooled iters 5–10 above 010's 35.2 floor, with 44.1 [40.6–47.7] as the bar for genuine progress. Sanity checks in-run: value-label mean per games file should sit ≈0 (mirror 50/50) vs the anchor era's ~−0.2; vloss/R² re-baseline yet again under balanced ±1 labels — not comparable to any prior arm.

### Actual Results
Run `1784369717` (Jul 18, 12:15–13:36; a few mid-run restarts, same run_id resumed — data continuity intact). Config verified in every CONFIG line: `--anchor-frac 0.0`, `wl_labels=1`, 128/16, eff_iters 97–106; start = `run_1784369717_iter1_start` ≡ artifact (`9e4a7b6b…`, hash-verified). Gauges: iter 5 = 55/128 **43.0%**, iter 10 = 48/128 **37.5%**; pooled 103/256 = **40.2%** [34.2–46.2] — above 010's 35.2 floor, below the 44.1 benchmark, inside the campaign's ~37.6 oscillation band. Behavioral @t20 vs the 011 control (anchor+WL, same artifact start — the single-variable contrast):

| @t20 | 012 mirror iter5 | 012 mirror iter10 | 011 anchor iter5 | 011 anchor iter10 |
|---|---|---|---|---|
| cities | 2.30 | 2.13 | 2.56 | 2.22 |
| SPT | 11.5 | 10.3 | 12.2 | 10.7 |
| units | 6.7 | 5.6 | 6.6 | 6.1 |
| techs (vs greedy) | 16.1 / 11.2 | 14.8 / 11.9 | 15.0 / 11.6 | 14.8 / 12.0 |

Same slide, same slope, same profile — statistically indistinguishable from the anchor arm at matched iterations. **Label sanity check surfaced a real finding**: labels are pure ±1 (WL confirmed active in mirror games) but the per-sample split is **~64/36 toward winners** (mean +0.26 to +0.30 across three selfplay files), not 50/50 — winners build bigger economies → more moves → more samples. Per-game it's 50/50 by construction; per-sample the policy CE is already passively winner-weighted ~1.8:1 — and the slide continued anyway. Deviation adequacy: decisive games with real variance throughout — exploration was not the weak link. Training metrics: policy CE fell 2.50→2.02 in 10 iters (anchor arms sat flat ~2.25 — mirror self-distillation fits fast) while behavior slid; vloss re-baselined ~1.15, R² 0.47→0.62.

### Verdict
**WATCH — neither registered endpoint met by the letter, but the drift sign did NOT flip.** Not SUPPORTED (iter-5 SPT 11.5 < 12; iter-10 well short). Not strictly FALSIFIED (iter-10 = 2.13/10.3, a hair above the ≤2.1/≤10 fire bar). The informative result is the 011 comparison: switching the entire data diet from 100% anchor to 100% mirror changed neither the slide's slope nor the win-rate band — a **fifth invariance** (after label content, delivery, in-tree shaping, and search budget). The contrast-at-the-fork mechanism, as theorized, is not sufficient at this scale — and the 64/36 winner-sample tilt means a soft winner-weighting was already in effect without helping, which also weakens the advantage-weighting fix as a standalone bet. Suspect list collapses to: **(a) optimizer/training dynamics** (fresh Adam @ 2e-3 × 2 epochs erodes the artifact at ~the same rate regardless of what the data says) and **(b) the unaudited baseline**. WATCH trigger: extend to iter 20 (`--resume 1784369717`, same env, `-i 20`) — tower fires at iter 15 or 20 → FALSIFIED and the data-diet family closes alongside labels; recovery to ≥2.3/≥12 → re-opened. Either way the artifact audit and the optimizer test (TRAIN_LR≤2e-4 or persistent optimizer state) are the two live levers.
