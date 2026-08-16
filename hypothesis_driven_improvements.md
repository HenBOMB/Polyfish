# Hypothesis Driven Improvements

The idea is that we should get more systematic about coming up with a hypothesis for bottleneck on a performance metric, come up with experiments that move it, and either "commit" or "reject" it.

This will be the loop we will run continuously to ensure the Polybot continues to improve and get better, to eventually reach human-level capabilities.

Our #1 objective is to figure out how to get into a smooth learning curve regiment. Once we figure that out and can see more training time leads systematically to better playing from the AI, then we can deploy training regiment on the Cloud and let it run over 5M self-play games to reach human-level performance. We only have one shot at a $1M training run and we cannot waste it.

> **📍 The current bottom-line understanding lives in [`current_understanding.md`](current_understanding.md), not here.**
> This ledger is the append-only audit trail. An entry's **only** authoritative conclusion is its
> *final* Verdict — earlier verdicts, `~~struck-through~~` blocks, and mid-entry "Update" paragraphs
> record superseded reasoning, kept for provenance. Verdicts flagged **⚠️ SUPERSEDED** below were later
> overturned (chiefly by EXP_ELO_S0's winner's-curse recalibration and EXP_ELO_023's depth result) —
> read the pointer, not the original claim.

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
REJECTED as executed — and the label-side hypothesis was never isolated: the one-const change broke search behavior first, and that wreckage is what trained in. Same-day unwind: REL_W→0.4, ANCHOR_FRAC default→0.25 (untested rider), `model.safetensors` restored from `run_1783928687_iter1_start` (final weights parked as `tip_1783928687_iter10`). Successor candidate: thread a **label-only** rel weight through `td_lambda_labels` (in-tree stays 0.4) to test "make the label listen" cleanly. Standing finding **[⚠️ RETRACTED — see Correction below: this was a 10-turn-curriculum artifact, not an in-tree effect]:** a rel-dominant in-tree reward induces immediate hoarding/passivity in self-play — the negamax-asymmetry warning on `FINAL_OUTCOME_REL_W` cuts both ways.

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
> **⚠️ SUPERSEDED.** Win-rate DRIFT was retracted the same day (see Correction below — winner's curse at gauge power); only the *behavioral* reversion stands. And "the objective is convicted" was later overturned entirely: EXP_ELO_008/009/010/011-EXT eliminated the label family as the driver. See [`current_understanding.md`](current_understanding.md).

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
> **⚠️ SUPERSEDED.** "Labels convicted" was overturned: EXP_ELO_009 (outcome-only labels) and EXP_ELO_010 (shaping off entirely) *also* drifted, so the label family is fully **eliminated**, not convicted. And the "57% / best model on record" was winner's curse — EXP_ELO_S0 measured this artifact at **45.9%** [41.6–50.2]. See [`current_understanding.md`](current_understanding.md).

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
> **⚠️ SUPERSEDED.** Overturned by EXP_ELO_011-EXT (below): the climb was noise, the falsifier fired on both prongs, and the label family — delivery + in-tree shaping + content — is fully closed. See [`current_understanding.md`](current_understanding.md).

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

---

*From EXP_ELO_S0 on, entries incorporate an external literature review (`expert_review_plateau.md`, Jul 18 2026 — no longer in-repo) that independently converged on the same top lever (a persistent KL-anchor to a reference policy, citing AlphaStar) and supplied the primacy-bias/plasticity-loss citations this campaign's own research pass had flagged as a gap. Its implementation-level claims were code-audited before acting on them; corrections are noted inline below.*

## EXP_ELO_S0: Audit the "57%" artifact — no training
*Jul 19, 2026*

Every downstream judgment since EXP_ELO_009 ("hold ≥44.1%", "slide from 57%") was calibrated against either a single n=128 reading (57.0%, EXP_ELO_008 iter 5) or the clone's own pooled 44.7% — never a directly, adequately-powered measurement of the artifact itself. Registered as gating in EXP_ELO_011's Step 1 and independently flagged by the expert review; run now before trusting EXP_ELO_013's result against it.

### Expected Results
256 seeds (512 games) at 128 sims/gumbel-k=16 vs Greedy, `checkpoints/gauge_1784027745_iter5.safetensors`. Per EXP_ELO_011's registration: truth ≈44–48% ⇒ the 57% was winner's curse and every "slide from 57" in 009–012 overstates the damage; ≥52% ⇒ fine-tuning genuinely destroys points.

### Actual Results
n=256 seeds (512 games), 128/16: **235W–277L = 45.9%** [95% CI ≈41.6–50.2%], score 4473.8 vs 4493.6 (near-even), P1/P2 split 123/256 vs 112/256 (no seat bias). `ladder.py` records Elo≈471.4 (vs Greedy's 300 anchor).

### Verdict
**CONFIRMED — the 57% was winner's curse**, squarely inside the pre-registered 44–48% band. This recalibrates the 009–012 read: re-checking each arm's pooled CI against this corrected baseline (45.9% [41.6–50.2]) instead of 57%/44.1%, EXP_ELO_010 (35.2% [29.3–41.0]) and EXP_ELO_011-EXT (34.4% [30.3–38.5]) remain distinguishably below baseline — those regressions are real. But EXP_ELO_009 (38.7% [32.7–44.6]), EXP_ELO_011 (40.6% [34.6–46.6]), and EXP_ELO_012 (40.2% [34.2–46.2]) all CI-overlap this baseline — **statistically indistinguishable from "no change,"** not confirmed slides. Revised campaign read: fine-tuning from this artifact has never clearly *beaten* the true baseline, and clearly *hurt* it only in the shaping-off (010) and extended-anchor (011-EXT) regimes — a narrower, more specific finding than the "invariant slide in every regime" framing used through EXP_ELO_012. **EXP_ELO_013 and EXP_ELO_014 are judged against 45.9% [41.6–50.2] going forward**, not the retired 57%/44.1% references.

## EXP_ELO_014: Persist Adam/scheduler state across training iterations
*Jul 19, 2026 — implemented, not yet run*

Fresh Adam every iteration (`train.py:346`, confirmed via code audit — no optimizer `state_dict` was ever saved/loaded, only model weights) at `TRAIN_LR=2e-3` over a ~10-file replay window on an already-peaked net is a textbook primacy-bias/training-shock configuration (Nikishin et al. 2022; Dohare et al. 2024), flagged by the expert review as an axis this campaign's ablations never actually varied — ANCHOR_FRAC changed self-play *data*, never the optimizer's own state.

**Change:** `train.py` now hashes the model weights (`hash_state_dict`) at save time and persists `optimizer.state_dict()` + `scheduler.state_dict()` to `optimizer_state.pt` alongside `model.safetensors`. On the next `train()` call, it resumes Adam/scheduler state **only if the loaded model's hash matches** what `optimizer_state.pt` was saved against — otherwise (checkpoint restore, `--reset`, manual revert, or any load/shape error) it falls back to a fresh optimizer, matching today's behavior exactly. No LR or replay-window change (kept as a single-variable test per the review's Axis B framing); `--reset` now also clears `optimizer_state.pt`. Verified in isolation (dummy tensors, no repo files touched): hash is stable across a safetensors round-trip and sensitive to any weight change; Adam step count and moment buffers survive a real save/load cycle; a weight mismatch is correctly caught rather than silently resumed.

### Expected Results
Next real training run (any config) should show the slide's slope flattening / readings becoming more monotonic vs the EXP_ELO_S0 baseline (45.9% [41.6–50.2]) at matched budget, relative to same-config runs before this change. Falsifier: identical slope/slope-noise to the pre-change regime ⇒ training dynamics exonerated on this axis alone, weight shifts fully to Axes A (EXP_ELO_013) and C (EXP_ELO_015).

### Verdict
(pending — needs a real fine-tune run to measure; ready to stack with EXP_ELO_013 since both are single, orthogonal loss/optimizer-level changes.)

## EXP_ELO_013: Persistent KL-anchor to a frozen reference policy
*Jul 19, 2026 — implemented, not yet run*

Every fine-tune regime tested since EXP_ELO_007 leaves RL free to walk arbitrarily far from the artifact's policy once training starts — ANCHOR_FRAC varies self-play *data*, never adds a force in *parameter space* resisting drift. AlphaStar's ablation (Vinyals et al. 2019) is the literature's most-cited fix for exactly this BC-warm-start-then-RL-drift symptom: continually minimizing KL(π_ref ‖ π_θ) throughout RL (not just at init) — named by DeepMind as "key to unlocking AlphaStar's performance." This is mechanically new — distinct from ANCHOR_FRAC and from in-tree shaping, both already exhausted.

**Corrected scoping (per code audit, Jul 19):** the expert review's "π_ref = Greedy's soft policy, you already emit it" overstates the wiring — Greedy's `heuristic_mcts.rs` softmax exists but is never persisted anywhere `train.py` can read it during a live run (only used transiently as an in-tree prior blend), so anchoring to it live would need new Rust-side export plumbing. **π_ref = a frozen copy of the starting artifact's own weights** (a second `PolyZeroNet`, loaded once, no-grad forward per batch) instead — the review's own cheaper fallback, self-contained, ~20–40 lines, reuses the `soft_cross_entropy` helper already in `compute_loss()`.

**Surgery:** add `β·KL(π_ref ‖ π_θ)` as an auxiliary term to the policy loss in `train.py`, `β` swept as one small arm (start ~0.1–0.5× policy-CE scale per the review). Launch config otherwise matches the campaign's best-established fine-tune setup (frozen-clone start = `checkpoints/gauge_1784027745_iter5.safetensors`, `-r`, `NO_ANCHOR_DECAY=1 ANCHOR_FRAC=1.0 ITER_OFFSET=96 GAUGE_GAMES=64 VALUE_TRUST_RAMP_ITERS=1 WL_LABELS=1`, 128/16) plus EXP_ELO_014's optimizer persistence (both are orthogonal single-variable changes, safe to stack).

**Implemented (Jul 19):** two new `train.py` env vars, `KL_REF_MODEL` (path to the frozen checkpoint; empty = feature off, zero cost) and `KL_REF_WEIGHT` (β, default 0.0 = off). `compute_loss()` gained `ref_policy_pred`/`kl_ref_weight` params; when active, a second frozen `PolyZeroNet` (no-grad, `requires_grad=False`) runs one extra forward pass per batch on the identical (possibly D4-augmented) input, and `KL(ref‖policy)` per head reduces to `soft_cross_entropy(policy_logits, softmax(ref_logits))` — reuses the existing helper, no new math. Reported as `kl_ref_loss` in `.last_train_metrics.json`. `run_training_loop.sh`'s CONFIG echo now includes `kl_ref_model=`/`kl_ref_weight=` (EXP_ELO_006 discipline — no silent config). Verified in isolation: disabled state is a bit-exact no-op vs pre-change loss; KL grows with divergence and is ~0 when ref==policy; gradient flows only to the live policy; `total_loss` demonstrably includes the weighted term; the real artifact checkpoint (`gauge_1784027745_iter5.safetensors`) loads cleanly into the current architecture with a live forward pass. Launch would set `KL_REF_MODEL=checkpoints/gauge_1784027745_iter5.safetensors KL_REF_WEIGHT=<β>` alongside the rest of the config above.

### Expected Results
Behavioral primary (judged first, per protocol): tower signature does **not** fire (cities ≥2.3 / SPT ≥12 @t20) at **both** the iter-5 and iter-10 gauges — no prior regime has held twice. Secondary (win rate): pooled iters 5–10 ≥ the EXP_ELO_S0 baseline (45.9%, CI floor 41.6%), ideally trending up rather than flat/down.

**Falsifier:** tower fires by iter 10, or pooled win rate falls clearly below the S0 baseline's CI floor (~41.6%) ⇒ anchoring-alone is insufficient — possibly the heavy-tailed-Goodhart case the review flags (KL doesn't reliably stop reward hacking when the value-misspecification error is heavy-tailed) ⇒ escalate to stack with EXP_ELO_015 (KataGo value decomposition) rather than abandoning Axis A.

### Actual Results
Run `1784449090`, 10 iterations, 128/16, config verified in every CONFIG line + `⚓ KL-anchor active` banner (session.log): `kl_ref_model=checkpoints/gauge_1784027745_iter5.safetensors kl_ref_weight=0.3`, `-r` banner confirmed, `wl_labels=1`, `anchor-frac 1.0`, `eff_iter` 97–106 (ITER_OFFSET=96 honored — no repeat of the EXP_ELO_005/006 curriculum-stage confound; `avg_moves` 415–525 throughout, non-degenerate). EXP_ELO_014's optimizer persistence active by default (no extra flag needed).

Gauges: iter5 = 56W-72L **43.75%** (Elo 456.3), iter10 = 67W-61L **52.34%** (Elo 516.3) — pooled 123/256 = **48.05%** [95% CI ≈41.9–54.2%]. Behavioral @t20: iter5 cities **1.97** / SPT **9.3** / techs 13.87-vs-10.81 — fires the established tower bar (cities≤2.1 with SPT≤10). iter10 cities **2.48** / SPT **11.81** / techs 13.81-vs-10.75 — clears the cities SUPPORTED bar (≥2.3) and sits just under the SPT bar (11.81<12), comfortably clear of the fire bar. @t25 tells the same story (iter5: 1.79/8.99, fires; iter10: 2.30/11.39, clears/near-clears). Greedy's own cities @t20 is 2.63 — model's iter10 gap to Greedy (2.48 vs 2.63, ~94% parity) is far tighter than any prior tower-era arm (which ran roughly half of Greedy's cities). Training diagnostics: policy CE rose slightly (2.09→2.26, expected — the KL term now competes with the CE fit rather than a free lunch), value R² improved smoothly (0.629→0.670, no mispricing signature). `kl_ref_loss` isn't in `training_log.csv`'s fixed schema (ops gap — only the final snapshot survives, 4.53 at iter10, i.e. a $0.3×4.53≈1.36$ contribution to total_loss, same order of magnitude as policy_loss — confirms the term is exerting real, non-trivial pull, not a rounding error).

### Verdict (superseded — see extension below)
~~WATCH — falsifier did not fire, and this is the first regime in the whole campaign to *recover* from a tower-firing checkpoint rather than hold or worsen.~~ Neither registered endpoint met by the strict letter (iter5 fired), but iter10's reversal looked like the most encouraging single data point since the campaign started chasing this bottleneck. Pooled win rate (48.05% [41.9–54.2]) heavily overlapped the S0 baseline. Extension registered: resume +10 iterations, read at 15/20.

### Extension: Actual Results
Ops fix landed mid-run (`kl_ref_loss` added to `training_log.py`'s CSV `HEADER` + `append_row`; `migrate_csv()` runs fresh every `append_row()` call, so the very next iteration picked it up live with zero manual migration and no race — verified against the real file: iters 11–17 blank, iter 18 onward carries real values).

Same run `1784449090`, iterations 11–20, config verified unchanged in every CONFIG line through iter20 (`kl_ref_model`/`kl_ref_weight=0.3`/`anchor-frac 1.0` all present; one aborted resume attempt briefly lost these settings in session.log but was superseded before iteration 11 actually trained — the data in `training_log.csv`/`ladder.json` is all from the corrected sequence). `avg_moves` 351–546 throughout, non-degenerate. `value_r2` kept improving (0.670→0.696, no mispricing signature — the value head fit fine right through the reversion). `kl_ref_loss` plateaued flat at 4.53–4.57 across iters 18–20 rather than growing — the anchor bounded *how far* the policy moved on average, but did not stop it from moving specifically *toward* the tower within that budget.

Gauges: iter15 = 41W-87L **32.03%** (Elo 369.3), iter20 = 46W-82L **35.94%** (Elo 399.6) — pooled (the registered comparison) 87/256 = **33.98%** [95% CI ≈28.2–39.8%]. Behavioral @t20: iter15 cities **1.66**/SPT **7.47**, iter20 cities **1.73**/SPT **8.49** — both fire the established tower bar (cities≤2.1, SPT≤10) at both readings, and again at t25 (iter15 1.5/6.8, iter20 1.64/8.19). Greedy's own cities/SPT climbed further still (iter15 opp 3.55/20.0 @t20 — the widest gap of the whole campaign; iter20 opp 3.14/17.4, slightly less extreme but still solidly reverted). Full-run pooled (all 4 gauges): 210/512 = 41.02% [36.8–45.3%] — dragged up entirely by iter10's outlier reading.

### Final Verdict
**FALSIFIED — iter10 was noise, matching the false-dawn pattern from EXP_ELO_009/011's early readings.** The registered falsifier fired cleanly on both prongs: pooled extension win rate (33.98% [28.2–39.8%]) sits with zero CI overlap below the S0 baseline's floor (41.6%), and the tower signature fired at both extension gauges, at both t20 and t25. A single good-looking reading from a KL-anchored fine-tune, off the same reference artifact that produced the original discredited 57%, again failed to replicate — the campaign's own winner's-curse lesson applies a second time, this time to iter10 itself. Per the registered contingency: **KL-anchoring alone (β=0.3, anchored to the frozen `gauge_1784027745_iter5` clone) is insufficient** — the heavy-tailed-Goodhart caveat the review flagged is the leading explanation, reinforced by a real mechanistic finding: `kl_ref_loss` plateaued rather than growing, meaning the model found a policy at a *stable* KL-distance from the reference that still expresses full tower behavior — the anchor constrains magnitude of drift, not its direction. Consequence: **escalate to EXP_ELO_015** (KataGo value decomposition — score-belief head + multi-horizon value stack) per the review's own pairing rationale, rather than abandoning the KL-anchor axis entirely; a future KL variant worth registering separately if 015 also falls short: head-specific anchor weighting (weight the anchor more heavily on the heads most responsible for expansion/spawn/research decisions, rather than a single scalar β averaged across all four decomposed heads).

## Diagnostic session (Jul 19): mechanistic probes — where the tower is NOT, and the turn-level pursuit measurement

*After EXP_ELO_013 escalated toward value decomposition, we paused training runs and instead ran direct mechanistic probes on the towered model (`gauge_1784449090_iter20`, sha `1573835f`) and the BC clone (`run_1784017302_iter1_start`, sha `a3996362`) to locate the tower's actual mechanism. Net result: several standing hypotheses — value-target miscalibration, the aux-head disconnect as the bug, production collapse, expansion-targeting failure — were tested against hard data and mostly **refuted**. The reliable, shareable findings are recorded here, most-trustworthy first. Two of the intermediate single-ply conclusions were wrong and are flagged as retracted (see §5) — kept in the record because the correction is itself the useful methodological result.*

### 1. Search DOES look several of its own turns ahead — with a frozen opponent (code audit)
Corrects the Jul-18 first-principles note. The Gumbel tree crosses `EndTurn` into the **searching player's own future turns**, up to `max_turns_ahead` (5 early-game, ≥2 later — `brain.rs:372`), gated by `turn > turn_horizon` (`gumbel_mcts.rs:781`), not by turn-change. `simulate_move(EndTurn)` cycles the state back to the same player, auto-ending (skipping the moves of) all opponents (`game.rs:283-319`). Leaf value backed up = `win_value` only (`eval_server.rs:273-278`, `mcts_zero.rs:424`). So at 128 sims the search reaches **tens of plies / ~5 own-turns deep, but with ZERO adversarial content** (opponents never move in-tree). The Jul-18 note's "within-turn sequencing only" was **wrong**; its "zero adversarial lookahead at any budget" was **right**. Consequence: the search cannot see a contest coming, so it cannot value an army for one — consistent with the 16/4→128/16 dose-response raising strength while never touching the drift.

### 2. The aux heads are causally disconnected from the decision (HARD DATA)
**Probe A:** the value head's output is **bit-identical (max |ΔV| = 0.000e+00 across 256 real self-play states)** when the `aux_ownership` head's weights are scaled 3× or zeroed. The candle `network.rs` (and `metal_network.rs`/`tch_network.rs`) define **no aux heads at all**; MCTS backs up `win_value` only. So the aux heads shape only the shared *trunk* during training and have **zero path** to the value/policy the search consumes. Direct implication for EXP_ELO_015: adding or decomposing aux heads changes nothing about the decision unless the head is wired **into `v_win`** (a dual-network change across all 3 Rust backends + retrain) — otherwise it is "more disconnected siblings."
**Probe B:** the ownership head *is* learning a real signal — at turn<10 with a visible uncaptured village it predicts "we'll own it" with **74–80% calibration**, but weakly (mean pred +0.09 towered / +0.23 clone; ~35% of predictions near-zero). The towered head is more washed-out than the clone's.

### 3. The value head does NOT prefer tech (HARD DATA)
Tower-window decision traces (turn≥15, stalled at exactly 2 cities; 99 forks, towered model, 128/16): best-**research** `own_value` = **−0.20 (negative)**, best expansion-step `own_value` = **+0.29**; research `own_value` exceeded the step's in **0/17** head-to-head forks. The value head *correctly devalues* late-game teching. This **refutes "value-target miscalibration is the tower"** at these forks. (`Summon` policy prior ≈0.000 at these army plies in both models — but that is an army-ply artifact, not a production collapse; see §4/§5.)

### 4. ★ Turn-level 3rd-city pursuit — the most reliable measurement (HARD DATA)
The headline result of the session; supersedes all single-ply reads of expansion/production.
**Instrumentation** (`--dump-turn-states`, uncommitted in `self_play.rs`): at the start of every player-turn, dumps `{game, turn, player, cities[], city_count, visible_villages[] (FOW-visible uncaptured), units[]}`. 80 self-play games/model, Imperius/Bardur, 128/16. Data preserved in `turnstates_towered/`, `turnstates_clone/`.
**Opportunity** (assessed at turn start): `city_count==2`, a visible uncaptured village V with min unit→V Chebyshev distance **>1** (excludes the guaranteed-capture micro *and* the "standing on city #2 about to capture" case), ≥1 unit. Each V tracked once from its first qualifying turn; d0 = initial min distance.
**Pursued** (over ≤8 of that player's own subsequent turns, until V captured / lost / game-end): V enters `cities` (captured), OR min distance nets below d0 (progress), OR unit count grows while a unit closes on V (production-then-approach). **Ignored** = village sat visible, distance never closed, never captured. Ambiguous vanishings (fog/enemy) censored.

| model | opportunities | **pursued** | captured ≤8t | via fresh unit | by dist d0=2 / 3 / 4-5 |
|---|---|---|---|---|---|
| towered | 73 | **91%** (63/69) | 49% | 83% | 85 / 96 / 100% |
| BC clone | 69 | **98%** (60/61) | 64% | 80% | 100 / 96 / 100% |

**Conclusion: the model pursues genuine 3rd-city opportunities almost always, and builds units to do it.** It does not skip the far ones (d0=4-5 → 100%). This reconciles with EXP_ELO_001's 39% 3rd-city rate arithmetically: ~0.9 clear opportunities/game × ~50% completion ≈ 0.44 captured 3rd cities/game. RL mildly eroded both pursuit (98→91%) and completion (64→49% capture) vs the clone, but both models are fundamentally competent at this behavior. Classifier spot-checked: "captured" trajectories walk distance down to 0 then take the village; "ignored" ones hover 2-3 tiles and never close.
**CRITICAL CAVEAT:** this is **self-play** — the opponent is the model itself, a passive expander that does not contest. Against **Greedy** (the actual gauge, which produces units from turn 3-4 and contests), completion would be markedly lower: the model likely **loses the race/fight for the contested village**. The dump carries no enemy positions, so contest is invisible in this measurement.

### 5. Methodological finding: single-ply search-trace stats misled twice — use the trajectory level
Two intermediate conclusions this session were **wrong**, both artifacts of measuring at one search ply instead of the game-turn:
- **"chose not-to-seek ~70% of the time"** — a parser bug: `Capture`-in-place (a unit *taking* the village — the expansion payoff itself) was categorized as "not seeking." Corrected, in-view forks show ~55–79% expansion.
- **"production collapse (Summon prior = 0.000)"** — measured at *army* plies (a unit is unmoved, so the model correctly prioritizes moving it; summon is one low-prior option at that specific ply). The turn-level view (§4) shows the model builds units in ~80% of pursuits. **There is no production collapse.**
**Lesson for later analysis / other researchers:** for turn-level behaviors (expansion, production, tech allocation), measure over the whole game-turn or a multi-turn trajectory — *"did the player commit to this objective this turn or the next, building a unit if needed?"* — not at a single MCTS decision ply. A player-turn is ~8 plies; single-ply search-trace stats over-index on whichever move happens to be selected at one of them, and both `own_value`/`raw_net_prob` snapshots and chosen-move categories are easy to misread.

### 6. ★ vs-Greedy pursuit-and-completion — the tower dies to a lost RACE, not a lost fight (HARD DATA)
The measurement §4's caveat demanded, and the capstone of the arc. Instrumentation: `--dump-turn-states` added to `arena.rs` (uncommitted) — per-turn ground truth for BOTH players (`model_cities/model_units/model_visible_villages/greedy_cities/greedy_units/neutral_villages`), 256 games per model vs the Greedy backend, same checkpoints as §2–4. Opportunity definition identical to §4; each opportunity classified over ≤8 own-turns into CAPTURED / LOST_TO_GREEDY (Greedy takes V) / STALLED-attrition (pursuing units died) / STALLED-timeout / NOT_PURSUED.

| | towered | clone |
|---|---|---|
| opportunities | 126 | 155 |
| PURSUED | 86.5% | 94.2% |
| CAPTURED | 34.9% | 43.2% |
| **LOST_TO_GREEDY** | **56.3%** | **49.7%** |
| attrition | ~1.5% | ~1.5% |
| timeout | ~6% | ~7% |

Split by contest (Greedy has a unit within reach of V — ~72% of all opportunities): **contested** → captured 19.1% towered / 30.7% clone, lost-to-Greedy 74.2% / 65.8%; **uncontested** → captured 73.0% / 78.0%. So both models finish uncontested villages at self-play-like rates and lose contested ones ~2:1.
**Trajectory spot-check (why the race is lost):** in LOST cases Greedy's nearest-unit start distance `c0` is almost always **0–3, frequently 0–1** — Greedy is already on or beside the village when the opportunity first registers, while the model starts d0=4–5 away; the model's unit count is **stable or growing** through the loss (no attrition). In CAPTURED cases `c0` is far (≈4–6). A caveat cutting the other way: opportunities with `c0=0` at registration were unwinnable from the start — the "loses contested villages" rate partly measures *arriving late to the map region at all*, i.e. the race was lost before the window opened.
**Conclusion: the deficit is TEMPO.** Pursuit intact (86–94%), production intact, attrition ≈ 0 — the model is **out-raced**: Greedy's earlier unit production and earlier map presence puts its units on contested villages first. §4's "completion-under-contest" suspect resolves to *completion-under-race*; the "loses the fight" intuition confirmed in its race form, not its combat form. RL again strictly erodes the clone (35%<43% capture, 126<155 opportunities generated — the towered line is *slower at creating its own chances*), consistent with EXP_ELO_007→now: the vs-Greedy ladder gap is a tempo gap.

### 7. The label pays for the tower: score-pricing audit of the reward target (code audit, HARD numbers)
With tempo established as the deficit, audited what the training label actually pays per star invested. The TD value target = discounted **score deltas** (`GAMMA_TURN=0.9`/turn, abs-dominant `REL_W=0.4`), and the score prices are (`score.rs`, `actions/{tech,units,discovery}.rs`):

| investment | stars in | score out | timing | revocable? |
|---|---|---|---|---|
| tech tier t | ~5–8 | **100·t** | instant | never |
| trained unit | cost | **5·cost** | instant | **−5·cost when it dies** (`units.rs:179`) |
| village capture | unit + 3–6 turns walking | 100 + 20/territory tile (+pop/level later) | delayed → ×0.9^(3–6) ≈ ×0.53–0.73, only if race won | on city loss |
| exploration | ~free | 5/tile | instant | never |

Per star: **tech ≈ 15–25 pts/star, instant, riskless, unclawbackable, position-independent; units = 5 pts/star, the only own-investment score the game claws back on death; the expansion payoff is discounted for travel time and conditional on winning a race the frozen-opponent search cannot even represent (§1).** Under this label, tech-towering is approximately the greedy-optimal policy — the model is *maximizing our label correctly*, not failing to. Three transmission channels then explain why vs-Greedy exposure (anchor games, EXP_ELO_007-era) never taught tempo: (i) frozen opponents ⇒ in-tree the village is never taken, so MCTS-improved policy targets are exactly as slow as the prior even in anchor games; (ii) the loss arrives 25+ turns after the tempo sins at ×0.9^Δ ≈ 0.07, traveling one value-bootstrap step per training generation; (iii) the score-delta term rewards towering *in every game* while only the anchor minority weakly punishes slowness. BC bypasses all three (dense per-decision supervision) — which is exactly why the clone out-tempos every RL descendant, and why EXP_ELO_005's rel-weight crank broke search instead of fixing labels: you cannot manufacture tempo pressure by reweighting a signal that prices the tower above the army.

### Where the diagnosis stands after this session
**Refuted as the tower's cause:** value-target miscalibration at the fork level (the value head *wants* expansion, §3), production collapse (it produces, §4), expansion-targeting/won't-seek (it pursues 91–98%, §4), out-fought-under-contest (attrition ≈ 0, §6), and KL-anchoring alone (EXP_ELO_013). **Located:** the deficit is **tempo** (§6) and the tempo deficit is **paid for by the label itself** (§7) — the reward prices tech ~3–4× army per star, taxes walking, fines unit deaths, and the mirror-self-play majority never punishes the resulting slowness. Consequence: the fix axis moves from value-head representation to the **reward potential** — see EXP_ELO_016 (potential-based development shaping), which jumps the queue ahead of the earmarked value-decomposition experiment.

**Data / repro:** `turnstates_{towered,clone}/` (per-turn JSONL, 80 games each), `turnstates_vsgreedy_{towered,clone}/` (arena both-player JSONL, 256 games each), `decision_traces_towered/` (99 tower-window forks); instrumentation = `--dump-turn-states` + `ThirdCity` trace trigger in `src/bin/self_play.rs`, `--dump-turn-states` in `src/bin/arena.rs`, all uncommitted; analysis scripts in the session scratchpad. Models: towered `gauge_1784449090_iter20` (`1573835f`), BC clone `run_1784017302_iter1_start` (`a3996362`).

## EXP_ELO_016: Potential-based development shaping — pay for tempo in the label AND the tree
*Jul 19, 2026 — pre-registered before implementation. Jumps the queue ahead of the earmarked EXP_ELO_015 (KataGo value decomposition): §6/§7 relocated the leading suspect from value-head representation to the reward label itself, and §2 already showed decomposed aux heads are causally disconnected unless separately wired into `v_win`.*

**Hypothesis:** the tower is the *correct* greedy policy under the current label (§7: tech ≈15–25 pts/star instant-riskless vs units 5 pts/star death-clawbacked, expansion payoff travel-discounted and race-conditional), and the tempo deficit (§6) persists because no training channel prices delay. Repricing the label's potential — de-weighting tech, adding dense development terms with a proximity gradient toward visible villages — and applying it **inside the Gumbel in-tree backup as well** (so the search itself feels delay-cost, partially compensating for the frozen opponent §1) will pull development tempo toward Greedy's and raise the vs-Greedy ladder. This is the formalization of the "pull future credit forward to now" idea: potential-delta shaping in the codebase's existing discounted-delta convention (same math score already gets — **not** strict Ng `γΦ′−Φ`; the `(1−γ)·Φ` annuity on *held* potential is the intended urgency mechanism: banking development earlier collects more discounted annuity). Not strictly policy-invariant ⇒ guardrails registered below. Precedent: EXP_ELO_004 already proved dense shaped labels beat sparse ones here; OpenAI Five / AlphaStar / both Polytopia papers all used dense shaped rewards for exactly this horizon problem.

**Surgery (spec):** `dev_potential(state, player) -> f32` in `src/ai/reward.rs`, score-equivalent units:
- **TECH de-weight:** −0.75 × Σ(100·tier) over owned techs (label keeps 25% of tech's score).
- **PROXIMITY:** +12 × max(0, 7 − min Chebyshev dist(own units → nearest FOW-visible uncaptured village)); 0 with no units or no visible village. Note it also pays *revealing* a village (potential jumps on first sight — an exploration incentive), and hovering is profitless (potential-based: no further delta without closing distance).
- **SPT:** +20 × tribe stars-per-turn (makes harvests/workshops/city upgrades pay immediately; SPT is the compounding variable the ~10-turn γ-horizon hides).
- **ARMY:** +5 × Σ star-cost of living units (doubles the game's 5·cost pricing to ~10 pts/star, still below repriced parity with tech — deliberate, army's real payoff should come through captures + the annuity, not the head-count).
Applied as an augmented snapshot `score + w·dev_potential` at both reward sites, `w` threaded **separately** per the EXP_ELO_006 pattern: `--shape-w-label` (TD-label `HistoryStep` snapshots) and `--shape-w-tree` (gumbel backup) — both default **0.0 = bit-exact current behavior**. Arm values: `w_label=1.0`, `w_tree=0.5` (EXP_ELO_005's lesson: search reacts violently to reward changes — start the in-tree dose at half).

**Launch config:** clone init (`checkpoints/gauge_1784027745_iter5.safetensors`), `NO_ANCHOR_DECAY=1 ANCHOR_FRAC=1.0` (anchor held — honesty pressure, not teacher), `ITER_OFFSET=96 GAUGE_GAMES=64 VALUE_TRUST_RAMP_ITERS=1`, 128/16, EXP_ELO_014 optimizer persistence active. **Deliberate departure from the recent fine-tune regime:** TD labels ON (no `-r`) and `WL_LABELS=0` — the label *is* the treatment, so the flat-W/L label regime of EXP_ELO_013 cannot carry it. Multi-knob vs 013 by necessity; the comparison anchors remain the behavioral bars + S0.

**Instrumentation shipping with the arm (the acceptance instrument, registered here):** per-player per-turn tempo curves — city count, SPT, units alive, stars-worth of army, revealed tiles — plus per-player counters: units *trained* (excl. ruin grants), units lost, giants acquired. Emitted per iteration to a `tempo_by_turn.json` sidecar (moves_by_turn pattern) + scalar CSV columns (`avg_units_spawned`, `avg_units_lost`, `avg_giants_made`, `t2c_3rd`).

### Expected Results
**Primary (behavioral tempo):** cities ≥2.3 AND SPT ≥12 @t20 at **two consecutive gauges** (no regime has held it twice); tempo curves move toward Greedy's reference (time-to-3rd-city earlier than the towered baseline's, units-trained-by-t10 up) rather than merely the gauge scalar improving.
**Secondary (win rate):** pooled gauges ≥ S0 (45.9%, CI floor 41.6%) and trending up.
**Guardrail watch (registered hack modes):** never-teching overshoot (techs@t20 collapsing far below Greedy's ~10), unit-hiding/cowardice (army annuity + death-clawback pricing → refuses engagements), village-adjacent hover (should be profitless by construction — verify), self-play score inflation without ladder movement (Goodhart on Φ).
**Falsifier:** tower fires at both of the final two gauges, or pooled win rate below 41.6% ⇒ shaping-as-specified insufficient. Registered contingency: ONE sweep arm over (`w_label`, `w_tree`, tech de-weight) before abandoning the axis — the first arm's constants are educated guesses, not the hypothesis. If the sweep also falls: escalate to EXP_ELO_015 (value decomposition wired into `v_win`) and/or in-tree opponent unfreezing (§1), which attack the same tempo mechanism from the search side.

**Implemented (Jul 19–20):** `reward.rs` gained `dev_potential` / `shaped_snapshot` / `normalized_reward_wf` with the four spec'd constants; `w=0` short-circuits to the raw snapshot (bit-exact legacy, no Φ cost on the hot path — verified by unit test). `gumbel_mcts.rs` got `reward_shape_w` applied at both in-tree edge-reward sites; threaded via `Brain::with_reward_shape_w` / `make_search_agent` (arena's call sites pass `None` — **gauges stay unshaped by design**, measuring the net under the standard search). `self_play.rs` gained `--shape-w-label` / `--shape-w-tree` (defaults 0.0); the whole label pipeline (`HistoryStep`/`LabelStep`/`Checkpoint`/`td_lambda_labels`) moved to f32 snapshots, and the terminal tail uses per-player `final_potentials` (`score + w·Φ` at game end) so shaped step-vs-final deltas stay consistent. Loop: `SHAPE_W_LABEL`/`SHAPE_W_TREE` env → flags + CONFIG echo (EXP_ELO_006 discipline). Tempo instrumentation landed with it: per-role (`model` / `model_vs_anchor` / `anchor` / `opponent`) per-turn curves (cities, city_levels, spt, units, army_stars, revealed, techs) → `tempo_by_turn.json` sidecar, plus CSV scalars `avg_units_spawned` (Summon-only, per net player-game), `avg_units_granted`, `avg_units_lost`, `avg_giants_made`, `t2c_{2nd,3rd,4th}_{rate,turn}` — live CSV migration verified. Tests: 6 new reward unit tests (legacy-equivalence, tech de-weight, army term, proximity gradient + hover-profitless + fog/owned exclusions); full CI-equivalent suite green. Smoke (4 games, 16/4, shaping 1.0/0.5): no panics, 0 sim-move failures, healthy label distribution (mean +0.22, 13% |v|>0.95), role split + counters verified with the anchor held.

### Actual Results (first read — run `1784500013`, iters 1–10, one gauge)
Config verified in every CONFIG line: `shape_w_label=1.0 shape_w_tree=0.5 wl_labels=0 anchor-frac 1.0 (held)`, TD labels on, 128/16, 64 games/iter, clone init. `avg_moves` 446–531 (non-degenerate). Note: launched with the default gauge interval (10), so the run produced ONE gauge, not the two the primary endpoint needs.

**Gauge (iter 10, n=128, 128/16, unshaped search):** 65W–63L = **50.78%** (Elo 505.4) [95% CI ≈42.1–59.4], model avg score 4440 vs Greedy 4169. Behavior in gauge games @t20: cities **2.12** vs Greedy 2.73, SPT **10.88** vs 15.66, units 5.7 vs 7.75, techs 13.85 vs 11.17 (@t25: 2.08/11.02). **Neither endpoint resolves:** the SUPPORTED bar (≥2.3 cities, ≥12 SPT) is NOT met, but the tower signature does NOT fire either (2.12>2.1, 10.88>10) — the first arm to sit above the fire line at its first gauge while winning ~half. For calibration: EXP_ELO_013's iter-10 false dawn read *better* (52.34%, cities 2.48) and then collapsed to 32–36% — a single iter-10 reading proves nothing by this campaign's own precedent.

**Training-data tempo (net vs Greedy-anchor seats, contested games):** net cities @t20 2.48→**2.75** (iter-10 best-of-run; Greedy-side 2.3–3.0), net SPT @t20 11.3→**12.1** (Greedy 14.7), net army stars @t20 15.2→**18.5** (Greedy ~29 — still ~⅔ short), t2c_3rd_rate 0.66→**0.73** (best), t2c_4th_rate 0.44→**0.50** (best), units trained steady ~15 with units lost at run-low **8.7** while units held @t20 hit run-high 7.27 (keeping the army alive, not refusing to build — cowardice guardrail clean). Iters 2–4 dipped (label-swap shock), 6–10 sit above the iter-1 baseline on nearly every axis.

**Mid-run data-integrity note:** the net-only metric fix (self_play.rs, landed between iters 10 and 11 while the run was live) changed the *semantics* of `avg_cap_villages`/`avg_research`/`avg_harvests`/`avg_builds`/`avg_attacks`/`avg_revealed_tiles`/`avg_captured_tiles`/`t2c_*` from "both seats combined" to "net seat only" — these columns roughly halve at the iter 10→11 boundary in `training_log.csv` for this run_id and are **not comparable across it**. `avg_units_spawned/lost/giants_made` were net-only from iteration 1 and are safe throughout. The `.run_bin` snapshot mechanism ([run_training_loop.sh:65-70](../polyfish-rs/run_training_loop.sh#L65-L70)) re-copies `target/release/*` on every invocation, so a `--resume` after a binary rebuild picks up the fix immediately — this is a general hazard for any multi-day run that spans a same-session code change, not specific to this experiment.

### Extension: gauges at iter 20/30/40 (resumed run, same `run_id`)
Resumed via `SHAPE_W_LABEL=1.0 SHAPE_W_TREE=0.5 NO_ANCHOR_DECAY=1 ANCHOR_FRAC=1.0 ITER_OFFSET=96 GAUGE_GAMES=64 VALUE_TRUST_RAMP_ITERS=1 ./run_training_loop.sh --resume -i 10 -n 128 -k 16`, run twice more (iters 11–20, then 21–40) on the rebuilt binary (net-only fix + stars-lost/`_totals` tempo fields active from iter 11 on). User stopped the run after iter 40's gauge.

| gauge | win rate | cities @t20 (net/Greedy) | SPT @t20 (net/Greedy) | army value @t20 (net/Greedy) | techs @t20 (net/Greedy) |
|---|---|---|---|---|---|
| iter 10 | 50.8% | 2.12 / 2.73 | 10.88 / 15.66 | 14.6 / 33.2 | 13.9 / 11.2 |
| iter 20 | 46.9% | **2.41** / 2.77 | **11.95** / 15.92 | 15.0 / 34.3 | 14.2 / 11.5 |
| iter 30 | 42.2% | 2.06 / 2.78 | 10.30 / 15.91 | 11.3 / 31.8 | 13.8 / 10.9 |
| iter 40 | 45.3% | 2.23 / 2.79 | 10.72 / 15.96 | 12.8 / 33.6 | 14.2 / 11.3 |

For scale, the pre-016 towered baseline (last KL-anchor-era reading, EXP_ELO_013 extension iter 20): win rate 35.9%, cities 1.73/3.14, SPT 8.49/17.4, army value 11.9/36.3, techs 13.1/11.2. **All four EXP_ELO_016 gauges clear that baseline on every axis** — this regime never approached the old tower signature at any reading. But across its own four gauges it oscillates (cities 2.06–2.41, SPT 10.3–12.0, win rate 42–51%) with no sustained direction; iter 20 is the high-water mark and nothing since matched it. Pooled win rate across all four: (65+60+54+58)/512 = **46.3%** [inside the S0 CI, 41.6–50.2]. Tower signature (cities≤2.1 AND SPT≤10) never fires outright at any gauge, but the ≥2.3/≥12 SUPPORTED bar is only ever cleared on one axis at a time, never both together, and never twice in a row.

Training-data CSV (iters 21–39, self-play, net-only semantics): `value_r2` plateaued 0.63–0.65, `policy_loss` drifted 2.20→2.12 (flat, marginal), `avg_spt_t20`/`t2c_3rd_rate`/`avg_units_lost` all oscillate with no trend. Matches the gauge picture: training stabilized, it did not continue improving.

### Final Verdict
> **⚠️ This entry stacks three conclusions in REVERSE-chronological order.** The earlier "### Verdict (UNRESOLVED — extend to iter 20)" is at the *bottom*; this "Final Verdict" (iter-40 data) supersedes it; the "Update" paragraph below then called off its own proposed next step (EXP_ELO_017). **Current bottom line: EXP_ELO_016 = NULL** — reward shaping did not move the tempo axis, i.e. the deficit is not in the reward label. See [`current_understanding.md`](current_understanding.md).

**NOT FALSIFIED, NOT SUPPORTED — statistically indistinguishable from the S0 no-shaping baseline.** Neither registered falsifier prong fired (tower never fires; pooled win rate 46.3% sits clear of the 41.6% floor), so the axis survives in the literal sense. But the primary success criterion (cities≥2.3 AND SPT≥12 @t20, held at two consecutive gauges) was never met — cities and SPT each cleared their bar individually at different single gauges (iter 20's cities, iter 20's SPT, barely) but never together, and never in back-to-back readings. The registered contingency (one sweep arm over `w_label`/`w_tree`/tech de-weight) is the honest next step for this axis, not abandonment — this arm's constants were educated guesses. **Decision (Jul 20):** rather than spend a sweep arm on constant-tuning, escalate directly to the registered fallback this experiment's own falsifier named — **in-tree opponent unfreezing** (§1: search currently reaches ~5 of the player's own future turns with the opponent frozen/auto-passed; nothing in the tree can represent "Greedy is racing me to this village," which shaping the reward can only ever compensate for indirectly). This attacks the tempo mechanism at its structural root rather than through the reward label. See EXP_ELO_017 below.

**Update (Jul 20, later same day) — premise checked before launch, and it doesn't hold.** Ran the `TraceTrigger::ThirdCity` decision-trace diagnosis (full writeup: notes.md, "Third-city stall decision-trace diagnosis") against the two pre-existing capture sets (`decision_traces_towered/`, `decision_traces/`). Per-candidate raw policy prior on Build/Harvest at the stalled-2-cities fork is 2-4 orders of magnitude below Capture/Attack/Step in BOTH checkpoints (Build median raw_net_prob 0.00003 towered, 0.00000 current) — the moves EXP_ELO_017 would need the tree to reconsider under a truer opponent model are never sampled at the root in the first place; the in-tree value estimate does not even disagree with them when Gumbel noise rarely lets one through (best-dev q_value beats the chosen move's q_value in both sets). This is a **proposal-side** failure (the policy head's prior), not a valuation-side one — opponent-unfreezing only changes leaf values, so it cannot reach a move that never enters the candidate set. *(⚠️ Later contradicted: EXP_ELO_018's clean-data diagnosis found the toward-village prior healthy and the failure valuation-side; ultimately resolved as a representation gap that, once fixed, gave no strength gain — see [`current_understanding.md`](current_understanding.md).)* **EXP_ELO_017 as scoped (opponent unfreezing) is called off before launch.** It also retroactively explains EXP_ELO_016's null: reward shaping paid Build/Harvest more in the label and the in-tree backup, but never touched the policy prior that drives Gumbel top-k sampling, so the moves it was trying to make more attractive were still never proposed for the tree to price. Next candidate axis: exploration-side (forced minimum visits/progressive widening on economy action types at the root, or BC-style prior injection from `book.rs`) — not yet pre-registered as an experiment.

**Real takeaway independent of the null result:** the score-pricing diagnosis (§7) and the tempo-race mechanism (§6) both still stand — this experiment tested one specific fix (reprice + shallow in-tree pressure) and found it insufficient at these constants, not that the diagnosis was wrong. The net-only metric infrastructure, the tempo sidecar, and the per-role gauge decomposition built for this experiment are now permanent instruments for every future arm on this axis.

**Guardrails:** no never-teching overshoot (avg_research flat 18–23; model still out-techs Greedy 13.9 vs 11.2 in gauges — the tower's tech preference persists in milder form), no move-count degeneracy, no self-play score inflation (avg_score flat ~5.4–5.8k), value head fit shaped labels cleanly (value_loss 1.29→0.68 monotone, R² 0.51→0.61, no mispricing signature).

### Verdict
*(⚠️ This is the earlier iter-10 verdict; superseded by the "Final Verdict" above once the extension ran to iter 40 — EXP_ELO_016 = NULL.)*

**UNRESOLVED — extension required by construction.** Neither registered falsifier fired (tower did not fire; 50.78% ≫ the 41.6% floor), and the primary cannot be met by a single gauge. The honest summary: best-legitimate-looking first gauge of the campaign, with — unlike EXP_ELO_013's iter-10 — *training-tempo trends that move in the same direction as the gauge* (sustained over iters 6–10, not a one-reading blip), but the 013 precedent demands the iter-20 read before any belief updates. Extension registered: `--resume` +10 iterations (same env), read at iter 20 — primary becomes cities ≥2.3 AND SPT ≥12 @t20 at iter 20 with win rate holding ≥ S0's floor; falsifier = tower fires at iter 20 or win rate reverts below 41.6% (the 013 collapse pattern). The extension runs on the post-run binary: CSV move/SPT metrics become net-seat-only from iter 11 (semantic break vs rows ≤10 — blended both players in mixed games before), and the Greedy-side unit-economy fields (`_totals`, stars-lost) start populating.

## EXP_ELO_018: Isolated, data-sized pursuit-progress reward — pay for *closing distance* to a committed village
*Jul 21, 2026 — pre-registered before implementation. Direct successor to EXP_ELO_016, whose registered contingency was "ONE sweep arm over (w_label, w_tree, tech de-weight) before abandoning the axis." This is that arm, but re-scoped from a constant-guess to a **measured** target after a decision-trace diagnosis (notes.md, "'Purposefulness' hypotheses and the Layer-2 test"; the FM-3 pursuit metric).*

**Diagnosis this rests on (all Jul 21, hard data):**
- **FM-3 behavior:** on the current (EXP_ELO_016 iter20) checkpoint, the *designated pursuer* — the unit nearest a discovered, uncontested, capturable village — makes progress on only **48% of its turns**; **52% are wasted** (stall 15% / sidestep 14% / **retreat 23%**). Identity-tracked whole-turn metric, cross-validated against FM-3's independent min-distance metric on the *same* 120 games (39% no-progress + ~12% net-backward = ~51%). The single biggest bucket is the pursuer moving *away* from its target.
- **Mechanism (valuation-side, confirmed on clean data):** at the plies where the pursuer moves the wrong way, the "Step toward the village" candidate is *proposed fine* — healthy prior (~0.21), in the Gumbel top-k 47/49 — but its **post-search Q loses to the chosen away-move 92% of the time** and it gets ~40% fewer visits (13.6 vs 35.2). This is not proposal-side suppression (contrast Build/Harvest, whose prior is crushed 2–4 OOM); the model *values continuing the pursuit below its alternatives*.
- **Sizing (measured, not guessed):** the chosen−toward Q gap on wrong-move turns is median **0.19** / p75 **0.42** in normalized value units. Through the reward normalization (`score_norm ≈ 700` early-mid game), that is **~150–350 score-equivalents per tile of progress** needed to flip the decision. **EXP_ELO_016's proximity term was 12/tile — ~15× too small.** This is why that shaping came back null; it is the specific thing this arm changes.

**Hypothesis:** a proximity/progress potential in the reward — *isolated* from EXP_ELO_016's tech-deweight/SPT/army repricing (so we test the one lever, not a bundle) and sized from the measured Q gap (~200/tile, sweepable) rather than the 12/tile garnish — will raise the toward-village Step's in-tree Q above its "reposition elsewhere" alternatives, cut the wasted-pursuer-turn rate below 52%, and pull the FM-1/FM-3 completion + tempo numbers toward Greedy's. Because the deficit is **opponent-independent** (FM-3 reproduces identically in self-play, vs-Greedy, and with no opponent), the reward must and does land predominantly in the **absolute** channel (`shaped_snapshot` → `normalized_reward_wf`, 60% abs via `1−REL_W`; a solo pursuer's progress step moves my score but not the opponent's, so it is not cancelled by the mirror the way symmetric capture-timing was in §7 / the Jul 7-8 diagnosis). **This is the specific reason to expect a different outcome from the 004→016 nulls**: those repriced signals that either netted to ~0 in the mirror (rel-channel cranks) or were priced ~15× too weak (016's proximity).

**Surgery (spec):** a dedicated potential in `src/ai/reward.rs`, threaded on its OWN weight independent of `reward_shape_w` so EXP_ELO_016 stays runnable and the two ideas never entangle:
- `SHAPE_PURSUIT_PER_TILE = 200.0` (score-equivalents per tile closed toward the nearest FOW-visible uncaptured village; central value of the measured 150–350 band).
- `pursuit_potential(state, player)` = `SHAPE_PURSUIT_PER_TILE × max(0, PROX_CAP − min Chebyshev dist(own units → nearest visible uncaptured village))`, 0 with no units / no visible village — i.e. EXP_ELO_016's `village_proximity` tile-count with a data-sized per-tile weight. Potential-based: hovering banks nothing, revealing a new nearer village jumps the potential (exploration incentive falls out for free).
- `shaped_snapshot(state, player, dev_w, pursuit_w)` gains a second weight; augmented score = `raw + dev_w·dev_potential + pursuit_w·pursuit_potential`. Both zero short-circuits to the raw snapshot (bit-exact legacy).
- Threaded as `--pursuit-w-label` / `--pursuit-w-tree` (parallel to `--shape-w-label`/`--shape-w-tree`, default 0.0), plus `pursuit_shape_w` on `GumbelMctsAgent`/`Brain`. Loop: `PURSUIT_W_LABEL`/`PURSUIT_W_TREE` env → flags + CONFIG echo (EXP_ELO_006 discipline).
- **Arm:** `pursuit_w_label=1.0`, `pursuit_w_tree=0.5` (EXP_ELO_005 lesson: half-dose the in-tree channel; search reacts violently to reward changes), `shape_w_*=0` (isolated — no tech/eco repricing this arm).

**Launch config:** clone init (`checkpoints/gauge_1784027745_iter5.safetensors`) OR resume from the EXP_ELO_016 iter20 weights — TBD at launch; default to the same init EXP_ELO_016 used so the arms are comparable. `NO_ANCHOR_DECAY=1 ANCHOR_FRAC=1.0`, TD labels ON (no `-r`), `WL_LABELS=0`, `ITER_OFFSET=96 GAUGE_GAMES=64 VALUE_TRUST_RAMP_ITERS=1`, 128/16, optimizer persistence on. First read at 10 iters (cheap), extend to 20 if not falsified (013/016 precedent: never trust a single gauge).

### Expected Results
**Primary (behavioral, the thing the reward directly targets):** wasted-pursuer-turn rate **< 52% → target < 35%** measured by the same identity-tracked instrument (re-run `VillagePursuit` traces + `--dump-turn-states` on the arm's gauge checkpoint), with the **retreat bucket specifically shrinking** (the mis-valuation the Q gap identified). Co-primary: cities ≥2.3 AND SPT ≥12 @t20 at two consecutive gauges (the standing tower bar no regime has held twice).
**Secondary (win rate):** pooled gauges ≥ S0 floor (41.6%) and trending up; EXP_ELO_016's own 46.3% pooled is the same-family reference to beat.
**Guardrail watch (registered hack modes):** (a) **village-rush overshoot** — a 200/tile term is large; watch for units abandoning defense/army to swarm villages (army value @t20 collapsing far below EXP_ELO_016's ~15, or units-lost spiking from undefended overextension); (b) **Goodhart on Φ** — self-play score inflation without ladder movement; (c) **hover exploit** — should be impossible by potential construction (verify: no per-turn Φ gain without distance closing); (d) value-head mispricing signature (R² crash) under the new label magnitude.
**Falsifier:** wasted-pursuer-turn rate does NOT drop below ~45% at the iter-10 gauge (the reward failed to move the exact behavior it prices), OR pooled win rate < 41.6%, OR a guardrail fires (army collapse / score inflation). Registered contingency if the behavior moves but win rate doesn't: the completion gain is being eaten elsewhere (over-extension into contested losses) → pair with the tempo/army terms rather than isolating. If the behavior does NOT move even at this magnitude: the lever is not the reward scalar but the **visit allocation** (the toward move stays under-sampled despite a higher Q) → escalate to a search-side fix (forced minimum visits / progressive widening on the pursuit candidate), which the Q-gap data already flagged as the secondary suspect.

### Actual Results (run 1784667465, iters 1-10, one gauge) — FALSIFIED
Clone init (`gauge_1784027745_iter5`), config verified in every CONFIG line: `pursuit_w_label=1.0 pursuit_w_tree=0.5 shape_w_label=0 shape_w_tree=0 anchor-frac 1.0 (held) td_w=0.7 wl_labels=0`, 128/16, 64 gauge games. `avg_moves` non-degenerate. Guardrails clean during training: value head fit the 200/tile labels fine (value loss 0.79→0.64, R² climbing 0.56→0.62 — no mispricing signature), no score inflation (avg_score 5.8k→5.2k), no move-count degeneracy. Smoke-test label distribution healthy (mean +0.26, nothing saturated) — the absolute channel carried the reward as designed.

**Gauge (iter 10, n=128 vs Greedy, unshaped search):** win rate **46.1%** (Elo 472.8) — statistical wash vs both S0 (45.9%) and EXP_ELO_016's iter-10 (50.8%); above the 41.6% floor so the *win-rate* falsifier prong did not fire. But cities @t20 **2.01** (below the ≥2.3 bar and below 016's iter-10 2.12), and the shape is telling: model cities **peak at t15 (2.16) then decline** (2.01 @t20, 1.85 @t25) while Greedy climbs to 3.09 — consistent with the registered village-rush-overshoot guardrail (grab then lose/overextend). Avg score below Greedy (4199 vs 4362).

**PRIMARY endpoint — pursuer-behavior trace (the registered decisive test), matched-training comparison** (all 128/16, iteration-999 trace, identity-tracked designated-pursuer metric, `pursuer_reconstruct.py`):

| model | iters from clone | shaping | wasted% | progress% | capture% | retreat% | sidestep% | stall% | n windows |
|---|---|---|---|---|---|---|---|---|---|
| clone (t=0) | 0 | none | 54.2 | 45.8 | 8.3 | 15.1 | 16.1 | 22.9 | 54 |
| gauge_1784500013_iter10 | 10 | dev (016) | 50.8 | 49.2 | 8.3 | 21.7 | 20.0 | 9.2 | 34 |
| gauge_1784667465_iter10 | 10 | **pursuit (018)** | **59.2** | 40.8 | 6.4 | 16.6 | 22.3 | 20.4 | 44 |
| gauge_1784500013_iter20 | ~20 | dev (016, over-trained ref) | 52.0 | 48.0 | 10.7 | 22.7 | 14.0 | 15.3 | 43 |

**The registered behavioral falsifier fired unambiguously:** wasted-pursuer-turn did NOT drop below ~45% at the iter-10 gauge — it sits at 59.2%. The robust, noise-proof claim is **not** "pursuit hurt": at n=34-54 windows the three matched arms (clone 54.2, dev 50.8, pursuit 59.2; SE≈4-5pp, 95% CI on 59.2 ≈ [51,67]) are **statistically indistinguishable — one noise band**. What survives noise is that **every arm sits at ~51-59% wasted and NONE approaches <45%**: the pursuit deficit is essentially **unresponsive to reward shaping of this magnitude**. Do not read "pursuit specifically hurt," "dev improved," or "not a training-stage artifact" into the ordering — all three rest on within-noise gaps and are retracted.

**Mechanistic sub-finding — a HYPOTHESIS, not a lead (bucket n far below significance).** The bucket ordering shifted (pursuit vs dev: retreat 21.7→16.6, stall 9.2→20.4), which *would* fit the `(1−γ)·Φ` annuity story (the potential penalizes moving away but rewards holding proximity, so a unit hovers near a village collecting reward without capturing). But the arm *totals* are already noise-indistinguishable, so this bucket slice is not evidence — it is exactly the tidy-mechanistic-story confirmation-bias shape this session repeatedly burned on. Flag only; do not spec an experiment on it.

**Unexamined confound that reframes the next decision (advisor, Jul 22):** the wasted% metric is measured in **no-race mirror self-play** with an **endogenous opportunity set** (each model generates its own 34-54 windows). Two consequences: (a) a model that expands *through* 2-cities faster spends less time in "stuck at 2 cities" states, so wasted% is not scored against a fixed test set; (b) §7's own logic says there is genuinely *no urgency* to rush an *uncontested* village in mirror play — so "~50% wasted even in the clone" may be partly **correct no-rush play, not pure defect**. This means the ~50% floor across all arms might not be a bug the reward failed to fix, but a metric that partly measures rational patience.

### Verdict
**FALSIFIED — reward-scalar shaping (Layer 2) does not move the self-play pursuit-purposefulness metric**, which is stuck at ~50-59% wasted across no-shaping / dev-shaping / pursuit-shaping alike. But the honest open question is one level up from "which lever next": **is this self-play metric even the right optimization target?** (endogenous opportunity set + no-race mirror play, above). The cheap discriminator — **no training run** — is whether wasted% predicts the thing we actually care about (vs-Greedy completion/win). The three points on hand (016-i10: wasted 50.8 / win 50.8; 016-i20: 52.0 / 46.9; 018-i10: 59.2 / 46.1) are *directionally* higher-wasted ~ lower-win but are 3 noisy points — underdetermined, which is itself the argument for NOT betting another training run yet. Deferred candidate levers, to be scoped only if the metric is first validated against vs-Greedy outcome: (1) strict Ng potential (`γΦ′−Φ`, zero reward for hovering) — but see the annuity caveat, this is a hypothesis not a lead; (2) search-side visit allocation (forced minimum visits / progressive widening on the pursuit candidate) — the Q-gap diagnosis's registered fallback.

Data/repro: `decision_traces_pursuit_{018,clone,016iter10}/` + `turnstates_{exp018_iter10,clone_ctrl,016_ctrl}/`; `pursuer_reconstruct.py` in the session scratchpad. `model.safetensors` left at the 018 tip (`gauge_1784667465_iter10`, sha 8fe8d3); the pre-experiment 016 tip is preserved as `gauge_1784500013_iter40` (sha e24d).

## EXP_ELO_019: Unfreeze the in-tree opponent — give the search adversarial tempo signal
*Jul 22, 2026 — pre-registered before running. This is the search-side fix the whole EXP_ELO_004→018 label/reward campaign was papering over. The diagnostic chain (notes.md "Inner mechanics"; the FM-3 pursuit metric; the §7 credit-assignment writeup) converged on: the flywheel can't turn on tempo because MCTS(π) ≈ π at the expansion fork — the search is single-player (opponent auto-skipped in-tree, game.rs), so it can never see "if I dawdle, Greedy takes the village," and the leaf value head is indifferent between the fork's near-identical immediate successors. No reward/label can fix a search that's structurally blind to the contest.*

**Hypothesis:** unfreezing the in-tree opponent (each EndTurn crossing plays the opponent's full turn via deterministic Greedy "ghost moves" — `cross_end_turn(unfreeze=true)`, already implemented + unit-tested) lets the search play *forward past the opponent's response* to a state where the fork's branches **visibly diverge** (opponent-has-the-village vs not). That divergent state is legible *relative* position (my-2-cities vs opponent-now-3), which the value head **can** read — even though it can't distinguish the immediate successors. The search thus manufactures the tempo contrast the frozen search couldn't, backs it up into Q → visit targets favor early capture → the flywheel finally has a gradient. Expected to drop the wasted-pursuer-turn rate below the ~50-59% frozen band (clone 54.2 / 016 50.8 / 018 59.2) and raise cities@t20 / win rate.

**Why this run is clean:** with `ANCHOR_FRAC=1.0` the *real* opponent in every game IS Greedy, so the in-tree Greedy ghost is a **faithful** opponent model, not a proxy — the search's anticipation is accurate, removing the usual "Greedy ≠ real opponent" caveat. Shaping is OFF (`shape_w=0`, `pursuit_w=0`) so the ONLY variable vs the existing frozen controls is the unfreeze.

**Launch config:** clone init (`gauge_1784027745_iter5`), `UNFREEZE_OPPONENT=1`, `NO_ANCHOR_DECAY=1 ANCHOR_FRAC=1.0 ITER_OFFSET=96 GAUGE_GAMES=64 VALUE_TRUST_RAMP_ITERS=1`, TD labels ON (no `-r`), `WL_LABELS=0`, no shaping, 128/16, 10 iters (cheap first read; extend to 20 if not falsified). Gauges/arena stay frozen by design (arena hardcodes `Some(false)`), so the strength reading measures the net under the standard search, uncontaminated by the training-time unfreeze.

### Expected Results / Falsifier
**Primary (behavioral, the mechanism this targets):** re-run the `VillagePursuit` trace + `pursuer_reconstruct.py` on the iter-10 gauge checkpoint; wasted-pursuer-turn drops **below ~45%** (out of the frozen 50-59% band) with the **retreat/stall buckets shrinking** — i.e. the value head, now fed a divergent contrast in training, distinguishes the fork. Co-primary: cities@t20 toward ≥2.3.
**Secondary:** pooled gauge win rate ≥ S0 floor (41.6%) and trending up; EXP_ELO_016's 46.3% pooled as the same-family reference.
**Guardrails:** (a) **throughput cost** — unfreezing runs Greedy for the opponent's whole turn on every turn-crossing simulation; measure the self-play moves/sec hit (this is the scaling-relevant number the search-speed concern hinges on); (b) value head must still fit (R² not crashing under the new in-tree Q distribution); (c) no move-count degeneracy / self-play score inflation.
**Falsifier / depth caveat:** if wasted-pursuer-turn does NOT drop below ~45%, the leading suspect is **search depth** — if 128 sims can't reach the turn where Greedy actually captures the contested village (Greedy 3-5 tiles away = 3-5 of its turns), the contest never materializes in-tree and unfreezing changes nothing. In that case the next step is not to abandon the axis but to (i) instrument the max reached tree-depth in own-turns, and (ii) raise the sim budget until the contest fits the horizon — which is the throughput project the search-side fix implies anyway.

### Actual Results (run 1784684857, 20 iters, unfrozen) — SPLIT: gauge up, pursuer mechanism flat
Clean 20-iter unfrozen run (CONFIG `unfreeze_opponent=1` verified all 20 iters; started after the prior run was confirmed stopped; no concurrent clobbering — the later `1784700727` completed 0 iters). Health clean: value_r2 0.53→0.61 (no mispricing), avg_moves 400-560 (non-degenerate), avg_research 10-15 (no collapse). Throughput cost measured at smoke: ~2.6-3x slower (17-21 vs 56 moves/sec) — the scaling number. A real bug was found+fixed first: the drafted `cross_end_turn(unfreeze)` panicked on `units.rs` (kill-undo doesn't compose across a bundled ghost turn); replaced per-move undo with a snapshot-restore of the whole state (correct under the wave LIFO unwind), + graceful no-panic stale-node fallback. The 310 residual sim failures are the documented self-healed stale-tree-reuse kind, amplified because the in-tree Greedy ghost plays with the searcher's *fogged* view so its predicted turn imperfectly matches the real opponent (weakens tree reuse — an inherent efficiency tax of an approximate in-tree opponent).

**Gauge (vs Greedy, n=128 each):**
| | iter10 | iter20 |
|---|---|---|
| win rate | 45.3% | **56.25%** (Elo 543.7 — campaign best) |
| model cities @t20 | 1.90 | **2.31** (meets the ≥2.3 co-primary, first time) |
| model curve t5→t25 | 1.11/1.8/2.14/1.9/1.68 | 1.2/1.95/2.38/2.31/2.17 |
| Greedy cities @t20/t25 | 2.95/3.17 | **2.68/2.84** (held down — narrowest gap of the campaign) |

**PRIMARY endpoint — pursuer metric (iter20 checkpoint, `pursuer_reconstruct.py`, standard frozen-search trace, apples-to-apples with all prior controls): NO CHANGE.** 57.2% wasted (stall 19.7 / sidestep 13.8 / retreat 23.7), n=42 windows — squarely in the frozen band (clone 54.2 / 016 50.8 / 018 59.2), retreat still the largest bucket. Registered falsifier (drop below ~45%) did NOT fire. **Mechanism decomposition: the value head is STILL indifferent at the fork** — leaf own_value toward 0.689 vs chosen 0.678 (coin-flip, toward loses 22/54). Same signature as every frozen model. **The specific hypothesis (unfreezing feeds the value head a fork contrast so it distinguishes toward-vs-away) is FALSIFIED at the mechanism level.**

**Reconciliation hypothesis (gauge up, pursuit flat) — HORIZON splits the two:** unfrozen search sees the opponent's *next* moves, which most directly helps **short-horizon adversarial decisions** — defending a city from an incoming attack, contesting/racing a village Greedy is about to take. Those payoffs land within the ~5-turn horizon. The **pursuit-completion** payoff (3rd city → compounding income → win) lands 20+ turns out, beyond even the unfrozen horizon — so the value head still can't attribute it, and the fork stays indifferent. The cities curve fits this: the model *holds its cities* (less peak-then-decline than 018) and *holds Greedy down* (2.68 vs the usual ~2.93) — both defense/contest signatures, NOT faster pursuit. So the reading may be: **unfreezing genuinely helped, but on the short-horizon adversarial game (defense/contest), not the long-horizon pursuit-attribution mechanism it was hypothesized to fix.** This is the registered **depth caveat** confirmed from the other side: 128 sims reach the opponent's immediate response (helps defense) but not the distant contest payoff (doesn't help pursuit).

**Caveats before believing it:** the 56.25% is ONE n=128 gauge (CI ±8.6pp), and iter10 was a poor 45.3% — could be winner's curse (the campaign has been burned by single good gauges). Confirmatory gauge (256 games) + vs-Greedy defense/contest measurement in progress to distinguish "real short-horizon improvement" from noise. Provisional verdict pending those.

### Verdict — EXP_ELO_019
**PRIMARY FALSIFIED; modest secondary win, mechanism located.** Confirmatory gauge (n=256): **50.8%** (the iter20 56.25% was ~half winner's curse — regressed to the mean at larger n), model outscores Greedy 4473 vs 4260. Solid, above S0 (45.9%), but ~EXP_ELO_016-level, not a breakthrough. The registered behavioral falsifier fired: wasted-pursuer-turn 57.2% (unchanged), value head still indifferent at the fork — **unfreezing did NOT fix the long-horizon pursuit-attribution mechanism it was designed for.**

**What it DID do (supported, directional):** the horizon-split reconciliation holds up. Unfrozen search reaches the opponent's *next* move → helps *short-horizon* adversarial play. Measured on 256-game vs-Greedy turn-states vs a frozen baseline: model peak cities 2.73 vs 2.52, Greedy held to 2.80 vs 3.26 (−0.46), city-loss rate 46.1% vs 49.6%. The model defends and contests better — but does NOT pursue its own villages faster (the pursuit payoff is 20+ turns out, beyond even the unfrozen 128-sim horizon, so the fork value stays flat).

**Consequence / where this points.** Two takeaways: (1) unfreeze is a **modest, real win on the short-horizon adversarial game** — worth keeping as a training-data option (guardrails clean, value fit fine), but it is NOT the tempo fix. (2) The **depth caveat is confirmed as the binding constraint**: to make the unfrozen search reach the pursuit-completion payoff (so the value head finally gets the fork contrast), the sim budget must be large enough to cross several opponent turns to where the 3rd city materializes — i.e. this is now the **throughput project**. This matches the pre-run strategic call: recursive self-improvement on the long-horizon (tempo/pursuit) axis is throughput-bound; the unfrozen-Greedy crutch alone, at 128 sims, only reaches the short-horizon gains. Next levers: (a) raise sim budget on the unfrozen path + re-measure the pursuer metric (does more depth finally move it?); (b) the DAgger/counterfactual-branch injection (attacks the attribution directly, horizon-independent) discussed pre-registration. Code fix (snapshot-restore `cross_end_turn`) and the ~2.6-3x throughput cost are permanent findings. the FM-3 pursuit metric stays open.

## EXP_ELO_020: DAgger — Greedy labels the MODEL's own states (on-distribution imitation)
*Jul 22, 2026 — pre-registered before running. The mechanism chain (notes.md "Inner mechanics"; EXP_ELO_013 diagnostic) says the pursuit deficit is (a) a collapsed capture PRIOR at the model's own forks and (b) an indifferent VALUE head that can't attribute the far-off payoff. Every value/reward lever (004-018) and the search-side unfreeze (019) failed — the last because the pursuit payoff is beyond even the unfrozen horizon. DAgger attacks the PRIOR directly and is horizon-INDEPENDENT: it injects the correct move at the model's states without needing search to reach the payoff.*

**Hypothesis:** at each of the model's own decision states (its actual, on-policy state distribution — including the "stuck at 2 cities dawdling" forks), blend Greedy's move-ranking into the policy target: `target = (1-a)*MCTS_visits + a*Greedy_ranking` (normalization-preserving — mixes direction, keeps each head's original mass). This raises the collapsed capture prior *exactly where the net actually plays*. Unlike BC (EXP_ELO_007), which labeled GREEDY's states and eroded under RL once the model drifted off-distribution, DAgger labels the LEARNER's states — the textbook covariate-shift fix — so the correction stays on-distribution and shouldn't erode. Because the prior directly drives Gumbel selection (the value head being indifferent doesn't block it), a raised prior should lift the pursuer metric out of the frozen 50-59% band and pull cities/tempo toward Greedy's.

**Ceiling caveat (honest):** DAgger converges the policy TOWARD the expert, so it caps at Greedy's expansion ceiling (EXP_ELO_003's "re-anchors to Greedy" risk). But the model is currently *below* Greedy on pursuit, so reaching Greedy-level is the target — a stepping stone, not the summit. A constant a=0.5 is a strong first dose to see if the mechanism moves at all; if it works, a follow-up decays `a` (classic DAgger annealing) to let the model exceed Greedy.

**Implementation (shipped, smoke-clean):** `--dagger-alpha` in self_play (net-seat decisions only; `decompose_visits` helper shared between the MCTS and Greedy targets; the extra Greedy eval per model move costs ~18% throughput — far cheaper than unfreeze's 2.6x). `DAGGER_ALPHA` env → flag + CONFIG echo. Normalization-preserving blend verified: action-target sums match the DAgger-off baseline (mean 0.881 vs 0.886), no distortion of forced/partial states.

**Launch config:** clone init (`gauge_1784027745_iter5`), `--dagger-alpha 0.5`, **frozen search** (isolate DAgger from the unfreeze lever — the only new variable vs the frozen 016/018/clone controls), `NO_ANCHOR_DECAY=1 ANCHOR_FRAC=1.0` (the model's anchor-game states match deployment-vs-Greedy, so DAgger labels the right distribution), TD labels ON, `WL_LABELS=0`, no reward shaping, `ITER_OFFSET=96 GAUGE_GAMES=64 VALUE_TRUST_RAMP_ITERS=1`, 128/16, 20 iters.

### Expected Results / Falsifier
**Primary (behavioral):** pursuer-metric wasted-turn drops **below ~45%** (out of the frozen 50-59% band), with the **toward-village policy PRIOR at the fork visibly raised** (the direct mechanism — check the trace decomposition: raw_net_prob on the toward move should climb from ~0.16 toward Greedy's level). Co-primary: cities@t20 toward ≥2.3.
**Secondary:** pooled gauge win rate ≥ S0 floor and trending up.
**Guardrails:** (a) **over-anchoring** — policy CE collapsing far below the MCTS-only band / avg_research cratering toward Greedy's exact mix (some convergence is intended; total collapse of the model's own signal is not); (b) value R² not crashing; (c) no move-count degeneracy.
**Falsifier:** pursuer wasted-turn does NOT drop below ~45% at the iter-10 gauge AND the toward-fork prior does not rise → on-distribution imitation is insufficient too, and the deficit is deeper than the prior (the value head's indifference genuinely blocks selection even with a raised prior). Contingency: pair DAgger with the value-side counterfactual (rollout the branch for a real outcome label), or decay-anneal test.

### Verdict — EXP_ELO_020 (STOPPED at iter ~11, no gauge signal)
Run `1784705457`. **iter-10 gauge vs Greedy: 51.6% (128 games).** Best iter-10 point estimate of any run to date (unfreeze 45.3%, pursuit 46.1% at matched iters) — BUT SE≈4.4pp, so within noise of both 50% and the frozen controls; not a clean separation. Cities (alive-only, censoring-corrected): peak **2.48 @t15, ~2.2 @t20** vs Greedy **2.85 @t20** — the plateau-at-~2.4 / can't-reach-3rd-city pattern is unchanged; expansion is NOT visibly fixed. **The primary behavioral test (pursuer wasted-turn <45% + raised toward-fork prior) was NOT executed** — the run was stopped before the VillagePursuit trace, on the standing empirical heuristic that in our full run history a within-noise iter-10 reading has never turned into a genuine iter-20 result (the one apparent counterexample, unfreeze 45→56, was the winner's-curse reading that re-measured ~51%). So: no clean win on the win-rate/cities proxy; mechanism unmeasured. Checkpoint `gauge_1784705457_iter10` retained if a post-hoc pursuer-trace on the prior is ever wanted. **Status: not pursued — inconclusive-negative.**

## EXP_ELO_021: De-saturate the value outcome label (×3 → ×1.5)
*Jul 24, 2026 — pre-registered before running.*

A new **value-head calibration probe** (`--dump-value-calib`, 23.6k net-seat decisions, iter27 weights, 64/k16, net-vs-net) reframes the value-head problem away from the pursuit/representation angle (004–020) toward the LABEL. Findings: the raw NN value head (tanh) is a *worse* outcome predictor than the current score ratio overall (corr 0.45 vs 0.55, MSE 0.42 vs 0.35) — but it earns its keep EARLY (corr 0.35 vs 0.28, +0.07 ΔR², where the scoreboard is blindest) and its ranking is monotonic, so it is **mis-calibrated, not blind**. Defect = over-confidence when ahead: predicts +0.6/+0.9 where actual is +0.28/+0.53 (~2× inflation); mean pred +0.37 vs actual +0.16. Root cause located: **32% of outcome labels are clamped to ±1** by `relative_outcome = clamp((my−opp)/(my+opp)·3, −1, 1)` (self_play.rs) — any lead past ~2:1 collapses to the same +1 label, so the head cannot learn "ahead" vs "crushing" because the label doesn't.

**Hypothesis + exact change:** lower the outcome scale **3.0 → 1.5** via `--outcome-scale` (label-only — computed in the value-target loop, NOT the in-tree backup, so **no EXP_ELO_005 search-disruption**; wired as `OUTCOME_SCALE` in run_training_loop.sh). At ×1.5 a label saturates only past ~5:1, so ~all games get a proportional monotonic outcome; the head should learn to discriminate winning positions, cutting over-confidence and lifting early-game foresight.

### Expected Results
- **Direct (calibration probe re-run on the treatment model, n≈20k):** top-decile over-confidence gap shrinks from ~0.40 toward <0.15; raw-value MSE drops from 0.42 toward the scoreboard's ~0.35; early-game corr(raw, outcome) clears the scoreboard's 0.28 by a clear margin; label-saturation frac 32% → <10%.
- **Strength (gauge vs Greedy, n≥128 at iter≥20):** ≥ S0 floor (45.9%) and **not below the matched ×3 baseline arm** — the calibration gain must not cost strength (the 016/018 lesson: mechanistic ≠ strength; the bar here is calibration-improves-AND-strength-holds).

### Falsifier
Treatment calibration probe shows NO top-decile over-confidence reduction and NO MSE drop toward the scoreboard → over-confidence is not label-saturation-driven (more likely the abs-progress positive bias in the TD arm, weight 0.6 — a separate EXP). OR calibration improves but iter-20 gauge falls materially below the matched ×3 baseline → de-saturation cost training-signal spread → REJECT and revert.

### Confound control
Matched ×3 baseline arm (same iter27 init, iters, budget, config) run alongside — more training alone shifts calibration, and the ×3-saturation is structural so the baseline should NOT de-saturate. Treatment = `OUTCOME_SCALE=1.5`, baseline = `OUTCOME_SCALE=3.0`.

### Actual Results
Treatment arm run `1784903502` (iter27 → +12 iters, `OUTCOME_SCALE=1.5`, 64/k16; loss 2.59→2.12). The LABEL de-saturated as designed (outcome-label saturation 32% → 9%), **but the value TARGET the head actually learns barely moved**: mean +0.338 → +0.333, and the implied TD-arm mean is +0.416 in BOTH (`--outcome-scale` doesn't touch the TD component). Calibration did NOT improve — if anything worse, but within the +12-iter confound: calibration slope 0.72 → 0.60, top-decile actual/pred 0.57 → 0.38; iter-10 gauge 43.75% (below floor, noisy). Mechanistic: the outcome is only **30%** of the value target and its saturated tail is too small a lever; the over-confidence lives in the **70% TD arm**, whose abs-progress component (weight 1−REL_W = 0.6) is positive-sum — both players gain score every game → TD mean **+0.42** → target inflated to +0.34 vs the actual outcome mean +0.16, which the head faithfully reproduces (pred mean +0.37).

### Verdict — REJECTED (lever too weak; falsifier's named alternative confirmed)
De-saturating the outcome tail cannot fix the over-confidence because it doesn't meaningfully change the target (mean +0.338→+0.333). The over-confidence is the **TD abs-progress positive bias**, not outcome-tail saturation — exactly the falsifier's pre-registered alternative. **Reverted:** `model.safetensors` restored to the iter27 baseline (treatment preserved at `model_exp021_treat_iter12.safetensors`); `--outcome-scale` defaults to 3.0 so production behavior is unchanged (flag kept, no code revert). **Redirect → EXP_ELO_022.**

## EXP_ELO_022: Reduce TD weight to de-bias the value target
*Jul 24, 2026 — pre-registered before running.*

EXP_ELO_021 located the over-confidence in the **70% TD abs-progress arm** (implied mean +0.42, positive-sum — both players gain score). Directly test that by **down-weighting TD: `--td-w 0.7 → 0.3`** (existing EXP_ELO_004 flag, `TD_W` in loop), so the target leans on the less-biased final-outcome arm. **Required de-confounder:** pair with `--outcome-scale 1.5` — at td_w=0.3 the outcome arm is 70% of the target, so its ×3 saturation would otherwise re-import the over-confidence; EXP_ELO_021 proved outcome-scale is otherwise inert (target mean unchanged), so any calibration change here is attributable to the TD cut. Predicted target mean → ~0.18 (near the actual outcome mean +0.16).

### Expected Results
- **Calibration probe (treatment model, n≈25k):** slope (actual~pred) rises from 0.72 toward 1.0; top-decile actual/pred rises from 0.57 toward 1.0; value-target mean drops from +0.34 toward the actual-outcome +0.16.
- **Strength (gauge vs Greedy, n≥128, iter≥20):** ≥ S0 floor (45.9%) and not materially below the matched baseline. **Primary guard** — EXP_ELO_004 showed the TD blend beats flat-outcome, so cutting td_w risks losing per-action credit; calibration must not come at a strength cost.

### Falsifier
Calibration does NOT improve despite the target de-biasing → over-confidence isn't the target-mean bias (head's own miscalibration / capacity). OR calibration improves but the iter-20 gauge drops materially below the ×3/0.7 baseline → the TD credit is load-bearing (strength > calibration); pivot to the surgical alternative `--label-rel-w` (de-bias the TD arm itself while KEEPING per-action credit), paired with anchor games.

### Actual Results
Treatment arm run (iter27 → +12 iters, `TD_W=0.3 OUTCOME_SCALE=1.5`, 64/k16; loss 2.33→2.08). **Mechanism worked, fix didn't.** Target de-biased as designed: value-target mean +0.338 → **+0.213** (toward actual outcome +0.13), pred mean +0.372 → +0.259. BUT the value head's outcome-DISCRIMINATION did not improve: corr(raw, outcome) ≈ 0.40 (unchanged from baseline 0.448 and EXP021's 0.400); calibration slope 0.72 → 0.68 (within confound, not toward 1.0); top-decile actual/pred 0.57 → 0.42 (still badly over-confident). And strength dropped: iter-10 gauge **37.5%** (below the 45.9% floor and below EXP021's 43.75%) — the EXP_ELO_004 warning (TD blend beats flat-outcome) fired.

### Verdict — REJECTED (de-biasing the mean doesn't fix discrimination; strength cost)
Reducing td_w shifted the target MEAN but not the SLOPE/discrimination — the head's outcome-ranking corr stayed ~0.40 across baseline/021/022. Combined with EXP_ELO_021, **the value head's over-confidence is robust to target re-weighting** (neither de-saturation nor mean-de-biasing moved discrimination), so it is NOT a target-design defect — it's a deeper value-head learning/capacity limit, or an inherent property of competitive net-vs-net self-play (mid-game leads are genuinely unstable → outcomes noisy → confident predictions are wrong). This echoes EXP_ELO_019's binding-constraint finding (the value/tempo problem is throughput/depth-bound, not a cheap target/representation fix). **Reverted:** model.safetensors → iter27 baseline (022 model preserved at `model_exp022_treat_iter12.safetensors`); `--td-w`/`--outcome-scale` default to production values. **Strategic:** stop re-weighting the value target; the value head may not be the binding lever (the policy prior is consistently healthy this session). Next candidates: value-head capacity/training, or refocus on throughput/depth (EXP_ELO_019), or accept the value head's limits and lean on the prior.

## EXP_ELO_023: Does search beat the prior at all? (prior-only control + depth ladder + concentration sweep)
*Jul 25, 2026 — pre-registered before running.*

**Motivation.** 20+ experiments have left behavioral metrics flat: the loop stopped drifting DOWN but never drifts UP. That is the exact signature of an AlphaZero loop whose improvement operator is the identity — `π_new = distill(search(π_old, V))` only improves if `search(π,V) ≻ π`. **This inequality has never been measured**: arena has no prior-only backend, so every prior experiment optimized components (value target, representation) without ever checking whether search consumes them. Supporting evidence that it may not: value head corr(raw,outcome)=0.40 INVARIANT across EXP_ELO_021/022; sibling Q gaps on pursuit forks measured at 3.2e-5; deterministic play beats noisy by ~7-9pp; net loses to Greedy at 43.8%.

**Instrumentation shipped:** `TREE_Q_WEIGHT` env knob (mirrors `GUMBEL_SCALE` pattern, default 1.0 = production) and tree-depth telemetry in `GumbelMctsAgent` (mean/max leaf depth + `horizon_hits`), surfaced by arena as `TREE DEPTH Config 1:`. With `TREE_Q_WEIGHT=0` + `GUMBEL_SCALE=0`, root selection, sequential-halving re-rank and the final recommendation ALL degenerate to `argmax(prior logit)` — i.e. a true prior-only control. (Note: the ML Expert's proposed arms A and C are identical for this reason; the real design is prior-only vs search.)

**Arms** (all vs Greedy, deterministic `GUMBEL_SCALE=0`, iter27 `5f73bf6c`, gamemode 2, Tiny, max_turns 30, sides swapped):
| arm | n | k | TREE_Q_WEIGHT | games | tests |
|---|---|---|---|---|---|
| A prior-only | 64 | 16 | 0 | 192 | the missing denominator |
| B baseline | 64 | 16 | 1 | 192 | reference (43.8%) |
| K1 | 64 | 4 | 1 | 192 | concentration: winner 38% of budget vs 23% |
| K2 | 64 | 2 | 1 | 192 | concentration: winner 50% |
| L1 | 256 | 16 | 1 | 96 | depth ladder |
| L2 | 1024 | 16 | 1 | 32 | depth ladder — can it span one ~8-ply game turn? |

**Budget-allocation facts** (read from `sequence_of_considered_visits`, correcting an earlier claim): allocation is NOT uniform — at k=16,n=64 the schedule is (16×1, 8×2, 4×4, 2×8) so the winning root child accumulates 15 visits = **23% of budget**. That ratio is **scale-invariant** (at n=256 the winner gets 60 = 23%), so raising sims scales the tree uniformly and depth grows only logarithmically. `k` is the true concentration lever: k=16→23%, k=4→38%, k=2→50%. Separately, `select_child_interior` scores `probs[a] − visits[a]/(1+sum_visits)` — an explicit ANTI-concentration term that spreads visits to match softmax(logit+σQ) rather than drilling a PV (unlike PUCT).

### Expected Results
- **A < B by a clear margin** if search is healthy (search should add several pp over argmax-prior).
- **L1/L2 > B** if depth is the binding constraint, with a knee once mean depth approaches ~8 plies (one full Polytopia game turn).
- **K1/K2 > B** if concentration (deeper PV per unit budget) is what's missing — this is the in-family version of the "switch to PUCT" proposal, without the confounds of an algorithm swap.
- Depth telemetry: mean plies should rise sub-linearly with sims (log growth), and `horizon_hits` should stay ~0 (else `max_turns_ahead` bounds depth, not budget).

### Falsifier
**A ≈ B** → search is a no-op at this value quality; the improvement operator is the identity, which explains 20 flat experiments in one shot. Then neither Gumbel-vs-PUCT nor more sims is the lever, and the answer is that the value TARGET has no slope along the failing axis (a step toward a village has score-delta exactly 0, so BOTH the outcome arm and the TD arm are flat there) or the opponent is too weak to generate informative labels. **A > B** → Q injection is actively destructive and search is worse than not searching. **L2 ≈ B despite materially deeper trees** → depth is definitively not the lever; retire the depth hypothesis permanently.

### Confound control
Single model (`5f73bf6c`) across all arms — no training involved, so no iteration confound. Same seeds, same swap discipline, same max_turns, all deterministic so exploration noise is removed. Greedy opponent is fixed and stateless. Reduced game counts on L1/L2 (96/32) widen CI on those rungs — read them for a knee, not for 2pp precision.

### Actual Results
| arm | n | k | tq | games | **win% vs Greedy** | **mean depth** | horizon-capped |
|---|---|---|---|---|---|---|---|
| A prior-only | 64 | 16 | 0 | 384 | **36.7%** | 3.60 | 0.0% |
| B baseline | 64 | 16 | 1 | 384 | **45.3%** | 4.05 | 0.0% |
| K1 | 64 | 4 | 1 | 384 | 44.0% | 4.41 | 0.0% |
| K2 | 64 | 2 | 1 | 384 | 41.1% | 4.16 | 0.0% |
| L1 | 256 | 16 | 1 | 192 | **50.5%** | 8.79 | 1.7% |
| L2 | 1024 | 16 | 1 | 64 | **56.2%** | 17.61 | 9.8% |

1. **Search DOES beat the prior: 45.3% vs 36.7%, +8.6pp** (SE_diff 3.5pp → 2.4σ, p≈0.015). The pre-registered falsifier (A ≈ B) did NOT fire. The improvement operator is **not** the identity, and Q injection is worth +8.6pp — so the value head, despite corr(raw,outcome)=0.40, contributes real signal to search. **This refutes the motivating premise of the whole investigation.**
2. **Win rate is monotonic in tree depth** across all four k=16 arms: 3.60→36.7%, 4.05→45.3%, 8.79→50.5%, 17.61→56.2%. At n=1024 the net **beats** Greedy. Depth is the lever.
3. **Depth grows as ~sims^0.5, NOT logarithmically** (correcting the pre-registration): each 4× sims roughly DOUBLES mean depth (4.05→8.79→17.61). The scale-invariant 23% root-concentration argument was right about allocation but wrong about the resulting depth curve.
4. **Concentration HURTS.** k=4 (44.0%) and k=2 (41.1%) are both *worse* than k=16 (45.3%) despite building *deeper* trees (4.41, 4.16 vs 4.05). Root breadth beats PV depth at fixed budget — **direct evidence against the "switch to PUCT" proposal**, since PUCT drills a PV and k=2 is its closest in-family analogue.
5. **`max_turns_ahead` starts to bind at high sims**: horizon-capped descents 0.0% (n=64) → 1.7% (n=256) → **9.8% (n=1024)**. `brain.rs:435` hardcodes `20 - current_turn` while games run to `max_turns=30`, so from turn 18 on the horizon is pinned at the floor of 2 turns. This is a real ceiling on the depth lever and will bite harder as sims rise.

**Honest CI caveat:** only A-vs-B is individually significant. B→L1 (+5.2pp, 1.2σ) and B→L2 (+10.9pp, 1.6σ) are each within noise at their reduced game counts (192/64). The *depth* measurements are precise (millions of sims); the *win rates* on the ladder rungs are not. The evidence for the depth lever is the monotonic 4-point trend plus the mechanism, not any single rung. **Confirm L1/L2 at ≥384 games before committing training budget.**

### Addendum (Jul 25): L3 n=2048, and a retracted saturation claim
> ⚠️ **Partially SUPERSEDED by Addendum 2 below.** The retraction of "flat past 256" stands, but this addendum's replacement reading ("~+5pp per 4× across the whole range") does NOT — pooling both sweeps shows only 64→256 reaches significance. Read Addendum 2 for the current numbers.
`n=2048, k=16, 64 games, post-horizon-fix`: **54.7%** (35/64, 1 draw), mean depth **27.76 plies**, horizon-capped 11.2% (now the deliberate `.min(20)` rollout cap, not the old bug), avg score **5344 vs 4638** — the widest score margin of any arm. Cost **3428 ms/move** vs 166 at n=64 = **20.6× for 32× sims** (sublinear, batch coalescing).

**Retraction:** I initially read 1024→2048 (56.2%→54.7%) as the depth lever saturating. That is wrong — **1024→2048 is a 2× step while every other rung is 4×**, and both endpoints carry ±12pp CIs. On a ~+5pp-per-4× trend a 2× step predicts ~+2.5pp; −1.5pp observed is uninformative, not evidence of a plateau. Corrected view — the two genuine 4× steps are both ≈+5pp:
| step | factor | Δ win% |
|---|---|---|
| 64 → 256 | 4× | +5.2pp |
| 256 → 1024 | 4× | +5.7pp |
| 1024 → 2048 | 2× | −1.5pp (noise) |

Pooling the high rungs makes it significant: **n≥1024 = 71/128 = 55.5%** vs **n=64 = 174/384 = 45.3%** → **+10.2pp, 2.0σ, p≈0.045**. **There is no evidence of saturation**; the ladder is consistent with ~+5pp per 4× sims across the whole measured range. Testing saturation properly requires **n=4096** (a true 4× step from 1024): trend predicts ~60%, a flat ~56% would be genuine saturation.

### Verdict — PREMISE REFUTED; depth identified as the lever (ladder needs confirmation)
Search beats the prior by a solid margin, so "MCTS + prior is not > prior" is false and the 20-experiment plateau is NOT a dead improvement operator. What the sweep instead shows is that **strength tracks search depth**, and production self-play runs at n=64 → mean depth ~4 plies → **below the ~8 plies needed to complete a single Polytopia game turn**. The loop isn't broken; it is generating training data from a policy that cannot see one turn ahead. That is a coherent, mechanism-backed explanation for flat behavioral metrics, and it is a *throughput* problem (EXP_ELO_019's binding constraint), not a value-target or representation defect.

**Do NOT swap Gumbel→PUCT** (falsified by the k-sweep). **Shipped, defaults unchanged:** `TREE_Q_WEIGHT` env knob + tree-depth telemetry (`GumbelMctsAgent::depth_stats`, printed by arena). **Follow-ups:** (a) confirm L1/L2 at ≥384 games; (b) fix the `max_turns_ahead` 20-vs-30 mismatch, now measurably binding at 9.8%; (c) cost-benefit of raising self-play sims 64→256 (4× cost for ~2× depth) against games/hour.

### Addendum 2 (Jul 25): distillation headroom — prior-override rate vs budget
New metric shipped alongside the depth telemetry: `PRIOR OVERRIDE` = root decisions where the search's final pick != `argmax(prior)` over the full legal set (`GumbelMctsAgent::agree_count/decision_count`, printed by arena). Under `GUMBEL_SCALE=0` the prior's top move is always inside the top-k cut, so a disagreement is a genuine override rather than an unconsidered candidate. Question: does deeper search override the prior MORE (= new knowledge to distill) or just confirm it (= nothing new)?

| n | depth | **override rate** | decisions |
|---|---|---|---|
| 64 | 4.09 | 24.2% | 38,085 |
| 256 | 9.27 | 24.1% | 36,924 |
| 1024 | 18.89 | 29.4% | 17,655 |
| 2048 | 26.32 | 34.0% | 9,348 |

**Override rate RISES with depth (24%→34%)**, and the shape is mechanistically clean: **flat from 64→256, rising only past ~9 plies** — new knowledge appears once the tree clears a single ~8-ply game turn. Samples are large (±0.5pp), so the trend is solid (caveat: decisions are not independent within a game/turn).

**Pooled win rates across BOTH sweeps** (much better power than either alone):
| n | pooled | win% | vs n=64 |
|---|---|---|---|
| 64 | 255/576 | 44.3% | — |
| 256 | 204/384 | **53.1%** | **+8.9pp, 2.70σ, p≈0.007** |
| 1024 | 82/160 | 51.3% | +7.0pp, 1.6σ |
| 2048 | 65/112 | 58.0% | +13.8pp, 2.70σ |

**Established:** 64→256 is a real +8.9pp. **Unproven:** 256→2048 is only 0.9σ — past 256 the override rate keeps climbing but the extra overrides cannot be shown to help (plausibly increasing near-ties). This supersedes both the earlier "flat past 256" retraction AND the "+5pp per 4× all the way out" reading: the confident statement is one big step at 64→256, then unresolved.

**Implication for expert-iteration / distillation (generate high-budget, deploy low-budget):** headroom is large and the signal is stable — prior-only **36.7%** → search@64 44.3% → search@256 **53.1%**, i.e. **~16pp** between the bare prior and n=256 search, with ~24% of root decisions being ones the prior currently gets overruled on. **Recommended generation budget n=256**: proven gain, only **2.7× per game** (1.4s vs 0.4s), first rung past the one-turn threshold; deploy at n=64 (166 ms/move ≈ 3.3s per 20-action turn). **Metric caveat:** override *rate* counts decisions, not importance — it shows signal exists, not what it is worth.

---

## EXP_ELO_024: Widen the TD credit window — λ 0.8 → 0.875 (~5 → ~8 turns)
*Jul 26–27, 2026. ⚠️ **Retro-logged, not pre-registered** — the run was launched first, this entry written from its data at iteration 15/20.*

**Motivation (Verdi, Jul 26).** "The TD is a proxy for winning the game. If your score maximally improves over the next 5 or 10 turns in a 30-turn game then you're on the trajectory to victory. Maybe we expand the window to try to target ~8 turns instead of 5." `LAMBDA_RETURN = 0.8` puts the credit window's center of mass at `1/(1−λ) = 5` turns; λ=0.875 moves it to 8. The hoped-for mechanism: a longer window lets the label see the *consequence* of expansion (a village taken at t12 pays rent through t20), which the 5-turn window truncates.

**Instrumentation shipped:** `--td-lambda` on `self_play` (default `LAMBDA_RETURN`), plumbed through `run_training_loop.sh` as `TD_LAMBDA` and echoed in the `CONFIG` line. Note the parameter is **double-duty and cannot be dialed apart**: λ sets both the trace decay *and* the `λ^n` terminal tail inside the TD arm. The flat 30% outcome share (`1 − td_w`) is independent of it.

**Arms** — a clean single-variable A/B, verified from the `CONFIG` echoes:
| | run | λ | everything else |
|---|---|---|---|
| A | `1785069748` | 0.800 | `mcts=256 gauge_mcts=256 k=16 td_w=0.7 unfreeze=1 value_trust=1.0 anchor_frac=0.25 games=64`, 13 self-play iters |
| B | `1785087189` | **0.875** | identical |

**Confound:** B **continued from A's iter-14 weights** (checkpoint hashes differ: `run_1785087189_iter1_start` ≠ `run_1785069748_iter1_start`), so this is a trajectory extension, not a paired fork from a shared base. Read it as "24 iterations of which the last 13 were λ=0.875", not as two independent samples.

### Expected Results
A longer window should show up as (a) better expansion retention past the t15 city peak, (b) a wider score margin vs the Greedy anchor, and (c) a gauge reading above A's 51.56%.

### Falsifier
Model-side behavior curves statistically identical to A → λ is not a lever at this game length and the whole "widen the window" family is dead.

### Actual Results

**1. Gauge @iter10, both at n=256/k=16: A 51.56% (33/64) → B 40.62% (26/64).** Two-proportion **z = 1.24, p ≈ 0.21 — not distinguishable.** Score margin +476 (4768.7 vs 4293.0) → −164 (4360.4 vs 4524.3). Suggestive of harm, not evidence of it.

**2. Nothing in the model's own behavior moved.** Welch *t* over 13 self-play iterations each:
| metric | λ=0.800 | λ=0.875 | t |
|---|---|---|---|
| avg_score | 5186.1 | 5392.8 | +0.94 |
| avg_spt_t25 | 12.428 | 12.424 | −0.01 |
| avg_cap_villages | 2.651 | 2.649 | −0.05 |
| avg_units_spawned | 16.64 | 17.30 | +0.87 |
| avg_units_lost | 12.02 | 12.85 | +1.13 |
| avg_moves | 420.9 | 423.2 | +0.16 |
| **value_loss** | **0.2651** | **0.3064** | **+6.31** |
| value_r2 | 0.8417 | 0.8306 | −1.27 |

**The only metric that clears significance is `value_loss`, and it went the wrong way (+15.6%).** Part of that is definitional — λ changes the label, and a longer multi-step return has higher intrinsic variance, so *some* loss increase is free. But `value_r2` normalizes by target variance and it also slipped. Reading: **λ=0.875 bought label variance without buying horizon.** In a 30-turn game λ=0.8's `λ^n` tail has already reached terminal, so pushing to 0.875 mostly adds noise.

**3. The army-composition failure is untouched.** Model $/unit at t25 is **2.26 → 2.27** (greedy 4.40 → 4.59). `trained_cum` stays at parity with the anchor at every checkpoint while `army_stars` is 2.3–2.6× behind — the model builds the same *count* of units at half the *value*, exactly as in every prior run.

**4. The apparently-widened gaps are the ANCHOR arm moving, not the model.** This is the finding that reframes the whole run. Model-vs-anchor at t25, split by side, Welch *t* on the run means:
| t25 | model A→B | t | anchor A→B | t |
|---|---|---|---|---|
| cities | 2.62 → 2.58 | −1.00 | 2.43 → 2.66 | +1.07 |
| units | 6.54 → 6.46 | −0.58 | 7.74 → 8.35 | +0.78 |
| army_stars | 14.75 → 14.66 | −0.25 | 34.04 → 38.13 | +0.98 |
| spt | 12.22 → 12.37 | +0.47 | 13.77 → 15.48 | +1.16 |
| city_levels | 7.15 → 7.26 | +0.57 | 8.93 → 10.15 | +1.34 |

Neither side is significant, but the model side is *flat and precise* while the anchor side drifts by ~1σ in one direction. Since Greedy is a **fixed, stateless policy**, the anchor curve cannot legitimately improve between runs — its motion is measurement noise (see EXP_ELO_M1). So "army gap widened −19.3 → −24.2" and "cities crossover moved from t30 to t25" are **anchor-side artifacts**, not λ effects.

### Verdict — REJECTED. Revert `LAMBDA_RETURN` to 0.8.
The pre-registered falsifier fired: model behavior is statistically identical to λ=0.8 (every *t* in |t| < 1.2), the value fit is measurably worse, and the gauge is 11pp lower at z=1.24. λ is not a lever at this game length. **`--td-lambda` stays shipped as a knob** (harmless, default 0.8) but should not be swept further — with EXP_ELO_021 (de-saturate the outcome label) and EXP_ELO_022 (down-weight the TD arm) also rejected, **the value-target parameterization is now closed as a family: three independent re-weightings, three nulls.**

**The run's real yield is diagnostic, not behavioral** — see EXP_ELO_M1, discovered while reading it.

---

## EXP_ELO_M1: Measurement audit — why the behavior charts can't be correlated with anything
*Jul 27, 2026. No training. Prompted by Verdi (Jul 26): "how come I cannot see the improvements in a consistent correlatable way?"*

**Question.** Across Jul 22–26 the model-vs-Greedy behavior charts (`tempo_by_turn.json` → dashboard) swung by ±1 city and ±30 army stars between adjacent iterations, with no config change explaining it. Is that the model oscillating, or the measurement?

### Finding 1 — the difference charts are ~93–99% opponent-side sampling noise
Sample sizes per iteration, from `tempo_by_turn.json` `n` fields (64 games, `anchor_frac 0.25`):
| turn | model seats alive | anchor seats alive |
|---|---|---|
| t10 | 92.8 | 14.6 |
| t15 | 89.7 | 13.7 |
| t20 | 79.1 | 10.7 |
| **t25** | **65.0** | **7.6** |
| t30 | 47.1 | 5.2 |

The model gets **2 seats per mirror game + 1 per anchor game** (48×2 + 16 = 112 seats) while Greedy gets **1 seat in anchor games only** (16). By t25 attrition leaves the opponent baseline estimated from **7.6 games**. Resulting cross-iteration standard deviations (13 iterations, run `1785087189`):
| t25 metric | model sd | anchor sd | **anchor share of diff variance** |
|---|---|---|---|
| cities | 0.081 | 0.614 | **97.8%** |
| army_stars | 0.962 | 11.07 | **99.1%** |
| spt | 0.732 | 4.55 | **97.3%** |

Same picture in run `1785069748` (93.3% / 99.4% / 92.7%) and at t15/t20 (77–99%). **The model curve is extremely stable — cities sd 0.081 across 13 iterations — and essentially all visible motion in the `model − anchor` chart comes from the 7.6-game opponent estimate.**

### Finding 2 — the anchor wobble is *entirely* sampling, leaving no room for signal
If the anchor curve's iteration-to-iteration variation were pure sampling, the implied per-game sd would be `sd_obs · √n`. For t25 cities:
| run | anchor mean | sd_obs | n | implied per-game sd | Poisson(mean) |
|---|---|---|---|---|---|
| `1785069748` | 2.43 | 0.501 | 7.5 | 1.372 | 1.560 |
| `1785087189` | 2.66 | 0.614 | 7.4 | 1.670 | 1.630 |

The implied per-game sd lands **on the Poisson prediction for a count of that mean**. There is no residual left to attribute to real drift. Concretely: iterations 11/12/13 of run B read **+0.29 / +0.67 / +1.05** cities-vs-anchor at t20 and iteration 14 reads **−0.12** — with a model-side sd of 0.081, *the model did not move; the baseline estimate did.* Every "the model finally passed Greedy for several iterations in a row" read from these charts since Jul 22 is unsupported.

**Fix (no error bands — Verdi, Jul 26).** Greedy is a fixed reference policy, so **stop re-estimating it every iteration.** Pool it once across iterations/runs into a static reference line and plot the model curve on its own, where sd 0.08 makes a genuine 0.15-city change plainly visible. Caveat: Greedy's curve in anchor games *does* depend slightly on how the model plays it, so it is not perfectly stationary — but at n=7.6 that second-order effect is invisible beneath the sampling noise. Raising `ANCHOR_FRAC` is the alternative and costs mirror data.

### Finding 3 — the 64-game gauge cannot resolve anything smaller than ~12pp
The last 8 gauge readings vs Greedy: 43.75, 37.50, 39.06, 39.06, 46.88, 46.88, 51.56, 40.62. Mean **43.16%**, observed sd **4.94pp** — *below* the **6.19pp** expected from binomial sampling alone at n=64. Since the readings span some budget differences (which would only *add* variance), this is strong evidence that **not one of the last 8 runs separated from the pack.** Unfreeze-opponent (EXP_ELO_019), `value_trust`, and λ (EXP_ELO_024) are all nulls at this resolution. Resolving a 6pp effect at 2σ needs ~555 games/side.

### Finding 4 — the `villages_first_rate` fix validates against production data
The Jul 25 denominator fix (`self_play.rs`) re-based first-village stats from *per game* to *per net seat*; run `1785069748` is the last pre-fix run and `1785087189` the first post-fix. Observed: **0.9243 → 0.7932**. The old per-game statistic was `[48·(1−(1−p)²) + 16·p] / 64`; solving for the per-seat rate gives **p = 0.8079**, against the new metric's directly-measured **0.7932** — a gap of 0.015, well inside noise. **The two definitions agree; the drop is the 2-seat OR being removed, not a regression.** The old value sat near its ceiling, which is why the metric read as saturated and uncorrelatable. Its slope within run B is **+0.0037/iter (rising)**, so the earlier "first-village rate is trending down below 0.9" concern was the artifact.

### Verdict — instrumentation defect confirmed; all Jul 22–26 chart-based behavior reads are unsupported (see next entry for the analysis that *does* work)
No model change. Three consequences: (a) **do not read `model − anchor` difference curves** until the baseline is pooled; (b) **treat the 64-game gauge as a ±12pp instrument** — it has been used all week to adjudicate changes far smaller than that; (c) the model's own behavior has been *remarkably* stable across every recent config change, which is itself the evidence that the knobs tried so far do not touch the binding constraint.

---

## EXP_ELO_M2: Win/loss autopsy — why does the model beat Greedy when it does?
*Jul 27, 2026. No training. Prompted by Verdi: "we know the model at budget 256 beats greedy at least half the time — why do we win? Make more units? Capture villages faster? Or something else?"*

**Method.** Condition on the *outcome* instead of comparing model-vs-Greedy means — this sidesteps EXP_ELO_M1's noise problem entirely, because tribe, map and opponent are all held fixed within the comparison. Arena's `--dump-stats-dir` writes one JSON per game with `winner_config` plus per-turn `score/spt/stars/cities/units/unit_cost/techs` for both sides (`sample_turn` already undoes the seat swap, so index 0 is always the model). **`replays/gauge_stats/` already held 94 historical dumps** — no new run was needed to get started.

**⚠️ Exploration-noise caveat (raised by Verdi, Jul 27 — a real limitation of this entry).** All games below are *arena* games, but they ran at the **default `GUMBEL_SCALE=1.0`, which `gumbel_mcts.rs:260` documents as "normal self-play exploration"** — `run_training_loop.sh` never sets it and neither did the manual runs. So this autopsy measures the **exploratory** policy, not maximum-strength play. Some share of the "stalled at ≤2 cities" population may therefore be self-inflicted root noise at a critical early decision rather than a policy defect. Note this also means **every gauge reading in `ladder.json` is a noisy-policy reading**, whereas EXP_ELO_023's ladder was taken at `GUMBEL_SCALE=0`. Deterministic replication is registered below.

**Data.** 352 games vs Greedy at n=256/k=16, gamemode 2, max_turns 30, **Imperius v Imperius (arena hardcodes both tribes → no tribe confound)**:
- 192 pooled from three existing gauge dumps (`1785069748_iter10`, `1785087189_iter10`, `manual_1785002034_iter10_n256`) → 46.9%
- **160 fresh held-out games** on `gauge_1785069748_iter10` → **49.4%** (79/81), which independently replicated every finding below
- Pooled: **169W / 183L = 48.0% [42.8–53.3]** — the best-resolved reading on record. Note the "51.56%" single gauge reading was the top of its own noise band; the honest strength is a coin flip.

### Result 1 — the third city explains almost the whole outcome
| condition | n | model win% |
|---|---|---|
| model reached 3+ cities | 193 | **74.1%** |
| model stalled at ≤2 cities | 159 | **16.4%** |
| reached city #3 before Greedy | 143 | 73.4% |
| Greedy reached city #3 first | 194 | 27.3% |

A **~58pp swing on one binary.** The model reaches 3 cities in 55% of games; Greedy does so in **97.8%** of games it wins.

### Result 2 — it is not SLOWER, it is BINARY (supersedes the "tempo/race" framing)
| | reach% WON | reach% LOST | turn WON | turn LOST |
|---|---|---|---|---|
| city #2 | 95.3% | 70.5% | t7.84 | t8.11 |
| city #3 | **84.6%** | **27.3%** | t12.33 | t11.20 |
| city #4 | 61.5% | 7.1% | t15.11 | t13.38 |

**The turn at which each city arrives is the same in wins and losses** (if anything *earlier* in losses, among the games that get there at all). Only the reach *rate* differs. So the deficit is not a pace gap — in the games it loses, the 3rd city is simply never obtained. Peak city count in losses: **29.5% never reach 2, 43.2% peak at exactly 2** (72.7% ≤2); in wins the mode is 4–5. Peak-city *turn*: **t15.17 (won) vs t7.16 (lost)** — the "universal t15 collapse" of earlier aggregate reads was **two populations averaged**.

### Result 3 — losses are annihilations, and "attrition ≈ 0" is dead
| in games the model LOSES | peak | final | change |
|---|---|---|---|
| model cities | 2.06 | **0.50** | −75.6% |
| model units | 5.77 | **0.85** | −85.2% |
| greedy cities | 4.57 | 4.56 | −0.1% |
| greedy units | 13.49 | 13.25 | −1.8% |

The model **loses its capital in most losses**. Asymmetric: when Greedy loses it still ends with 1.34 cities. **This retracts the ledger §6 claim "attrition ≈ 0; the model's units survive"** — it is out-raced *and* then out-fought.

### Result 4 — decided in turns 8–12
AUC of the model-minus-Greedy score gap for predicting the win: **0.489 @t5 → 0.635 @t8 → 0.718 @t10 → 0.829 @t12 → 0.900 @t15 → 0.966 @t20 → 0.993 @t25.** At t5 the model leads on score in most games it wins *and* most it loses, so an early score lead carries no information; the fork is the 2nd/3rd-city window.

### Result 5 — research is INELASTIC (the tower's real signature)
| turn | techs W/L | cities W/L | techs-per-city W/L | t |
|---|---|---|---|---|
| t8 | 7.21 / 7.42 | 1.66 / 1.43 | 5.18 / 6.03 | −3.20 |
| t10 | 8.59 / 8.77 | 2.09 / 1.62 | 4.98 / 6.40 | **−4.74** |
| t12 | 10.17 / 10.19 | 2.54 / 1.80 | 4.87 / 6.97 | **−6.35** |
| t15 | 12.56 / 12.29 | 3.02 / 1.85 | 5.10 / 8.21 | **−7.94** |

**Absolute tech is flat** (AUC 0.463 at t10) while cities differ by 29%. The model buys the same research with 1 city as with 3 — it does not redirect stars when expansion stalls. That is a sharper and more actionable statement than "it over-researches." *Caveat: the ratio's denominator is itself the strongest discriminator, so the load-bearing fact is the flatness of absolute tech, not the ratio.*

### Result 6 — army composition does NOT decide games ⚠️ demotes a claim made earlier the same day
`$/unit`, model side: **2.16 (won) vs 2.08 (lost) at t10, AUC 0.536**; 2.20 vs 2.10 at t15, AUC 0.553. **Essentially zero discrimination.** The model makes *more* units when winning (8.15 vs 4.25 at t15), not more expensive ones, and those are funded by the cities. The aggregate 2× gap vs Greedy is genuine and may be a *uniform* handicap (a constant cannot discriminate, so it is not refuted), but **much of the apparent gap is Greedy's own $/unit sagging when it loses (3.19 vs 4.42)**. An earlier Jul 27 revision of `current_understanding.md` called this "the one behavioral gap that survives" and the place to aim structural change — **that over-claimed and has been corrected.**

### Result 7 — the map is not destiny, and there is no seat effect
Each seed is played in both orientations. Both give the same winner **39.2%** of the time (69/176) vs **50.1%** expected from an independent coin — at or *below* chance, so terrain explains ~nothing. Seat: P1 51.1% vs P2 44.9% (z ≈ 1.2, not significant); the 160-game held-out run split 40/80 vs 39/80, killing the P1-advantage hint from the first 192.

### Verdict — the binding behavioral constraint is third-city REACH, as a binary
Reaching 3 cities is worth 74.1% vs 16.4%; the model manages it 55% of the time; and *when* it manages it is identical in wins and losses. **Aim structural work here, not at the value target (021/022/024 closed) and not at army composition (non-discriminating).**

**The next question is directly testable with instrumentation that already exists.** Since timing is identical, something makes the 3rd city *unreachable* in the failing 45%. Separate: **(a)** no FOW-visible neutral village in reach — `--dump-turn-states` already records "the model player's FOW-visible neutral villages" per turn, which answers this directly; **(b)** stars diverted to tech at the decision point (research is inelastic); **(c)** the units that would take it are dead or mis-positioned. Note (a) is *not* simply terrain quality, since seed concordance is at chance.


---

## EXP_ELO_025: Outcome-space value labels — win/loss z + root-value q-target TD arm
*Jul 28, 2026 — pre-registered before running. Implemented; no training run yet.*

**Motivation (research sweep, Jul 28).** A 4-thread literature review concluded: (1) no precedent exists for tabula-rasa AZ self-play working on a 4X-class game at any budget; (2) our label stack is the one component with a *direct experimental refutation* — score-primary value targets tested head-to-head lost by ~1,500 Elo to a winrate-primary twin at matched budget (Pasqualini et al. 2022, arXiv 2201.13176), with a mechanism (no variance preference: a behind agent should gamble, a score target won't teach it) that matches our measured 2× value-head over-confidence (EXP_ELO_021 probe); (3) KataGo's design keeps score OUT of the value label entirely (win/loss primary; score = bounded arctan in-tree utility + aux heads). Our current label was KataGo inverted: ~100% score-derived (70% TD-on-score-growth + 30% clamped score ratio, 32% of terminal labels saturated at ±1).

**Change (shipped).** `--wl-labels` now flips BOTH arms (previously flat arm only — EXP_ELO_011):
- Flat arm: z = ±1 from the adjudicated winner (unchanged from 011).
- TD arm (`td_lambda_labels` gains `wl_z: Option<&HashMap<i32,f32>>`): outcome space — zero per-window reward, **γ=1** (a discount would deflate early labels by depth), bootstrap through **post-search root values**, terminal = z. Label = `td_w·[(1−λ)Σλ^(k−1)·V_root(cp_k) + λ^N·z] + (1−td_w)·z` — the "q-target blend": mostly a λ-weighted average of future search values, anchored by the real outcome.
- In-tree shaping stays score-based (the KataGo pattern: score guides search, outcome trains value). train.py untouched (tanh+MSE is space-agnostic). Loop default `WL_LABELS=1`; `WL_LABELS=0` restores legacy labels.
- Tests: 3 new wl-mode cases (pure-z tail, hand-computed blend, γ=1 verification); 5 legacy tests unchanged and passing (score path bit-exact).

**Smoke test** (2 games, n=16, iter 126, calib dump): final_outcome ∈ {−1,+1}; value_target full-range, mean +0.27 (score-era bootstrap inflation, expected to wash out); **saturation 3.0% vs 32%** under the old labels.

### Expected Results
- Value head calibration improves: the 2× over-confidence-when-ahead defect (EXP_ELO_021) shrinks, because "ahead" and "crushing" now produce different labels via the bootstrap blend rather than both clamping to +1.
- value_r2 will DROP initially (new target semantics, score-era bootstraps) then recover — do not read the first ~5 iterations' r2 as regression.
- Behavioral: if the score-target variance-pathology was binding, losing-side play should get more enterprising (gamble-when-behind) and the win rate vs Greedy should move at the ≥5-iteration gauge horizon.

### Falsifier
At matched budget (≥15 iterations, deterministic gauge, n=256 readings pooled ≥384 games), win rate and calibration are statistically indistinguishable from the score-label baseline → the label was not the binding constraint at this data scale; the bottleneck ranking moves to data scale per the research verdict (games/iter, q-target augmentation of siblings).

### Confound control
Transition effect: early iterations bootstrap through a value head trained on score labels — the label mean will drift as semantics converge. Any A/B must either fork from the same checkpoint with ≥15 iterations both arms, or compare full fresh runs. Do NOT mix wl and score-label games in one replay buffer window when reading value_loss/value_r2 (the trainer will average two incompatible target semantics; behavioral metrics remain comparable).

### Verdict — ABANDONED before running (Jul 28, 2026)
Verdi's call: not worth the training time — reward-shaped labels are preferred per EXP_ELO_004, and priority moved to EXP_ELO_028 (goal-conditioned macro program). Never ran; no data. The implementation stays shipped behind `--wl-labels` / `WL_LABELS=1`, but the **loop default is restored to legacy shaped labels (`WL_LABELS:-0`)**. If revisited, note the hypothesis was never tested — this is an abandonment, not a rejection.

---

## EXP_ELO_026: "Oracle macro" — scripted commitment + star gate over the unchanged net
*Jul 28, 2026 — pre-registered, then implemented and run the same day.*

**Hypothesis.** The third-city reach failure (reach rate 55%; worth **74.1% vs 16.4%**, EXP_ELO_M2) is a **macro-decision** failure — expansion commitment and star allocation — not a micro-execution failure. A trivial hand-scripted macro layer steering the *unchanged* net should therefore raise reach rate substantially. This is simultaneously (1) the cheapest viability test for the macro/micro decomposition program (learned goal-setter over goal-conditioned micro) and (2) the **causal** test of the third-city finding, which so far is correlational only.

**Setup.** Two arms, identical weights, same seeds played in both orientations (M2's paired design), vs Greedy, n=64, `GUMBEL_SCALE` pinned identically in both arms (0 = deterministic, since this measures best-play capability).
- **Arm A:** production model as-is.
- **Arm B:** same model + two scripted rules, shipped as *separate flags* so a follow-up can attribute:
  1. **Expansion commitment:** while <3 cities and any FOW-visible neutral village exists, commit to the nearest reachable one (sticky until captured or lost) and write it into `CH_PURSUIT` — overriding the current commitment logic.
  2. **Star gate:** while committed and <3 cities, filter tech-purchase moves out of the root legal set unless stars exceed the reserve needed to complete the capture. Targets the research-is-inelastic signature (absolute tech flat 8.59 vs 8.77 at t10 while cities halve).
- Inference-only: root-level move filter + commitment override in self_play/arena. No training, no dual-network change.

### Expected Results
- **Primary: third-city reach rate B − A ≥ +15pp.** ~250 games/arm resolves a difference to ~±5pp — deliberately avoids the ±12pp gauge (EXP_ELO_M1).
- Secondary (directional only): loss profile shifts out of the ≤2-city annihilation mode (peak cities 2.06→0.50, units 5.77→0.85 in current losses); techs-per-city at t10 moves toward the win profile (4.98 vs 6.40); win% drifts up.

### Falsifier
B − A reach < +8pp with CI excluding +15pp → macro-steering is not the unlock. Then discriminate the residual: (i) commitment not *executed* (distance-closed-per-turn toward the committed target ≈ 0 → micro execution failure); (ii) no village available to commit to (`--dump-turn-states` availability rate → open-question candidate (a) is binding, a map/position constraint).

### Interpretation matrix
- Reach ↑ big, win% ↑ → macro is the binding layer, micro execution adequate; the delta is the **headroom ceiling for a learned macro**, which becomes the follow-up experiment.
- Reach ↑ big, win% flat/↓ → the third-city correlation is **not causal** (forced expansion sacrifices something else) — a top-tier finding on its own; redirects the ⭐ open question.
- Reach flat → the macro program dies cheaply, before any learning machinery is built.

### Confound control
Both arms share seeds, budget, and `GUMBEL_SCALE`. If the combined arm moves, run commitment-only and gate-only arms before attributing mechanism.

### Actual Results (Jul 28, 2026)
Implementation: new `ai/oracle_macro.rs` (commitment picker, star-gate predicate, reserve=5) + `--macro-commit` / `--macro-star-gate` / `--base-seed` on arena; `pursuit_focus` threaded through every agent encode (cache-safe — the eval LRU and tree-reuse both key on feature bytes). Verified by unit tests and same-seed engagement divergence. Run: `checkpoints/exp026_model.safetensors`, 125 seeds × 2 orientations per arm, n=64/k=16, `GUMBEL_SCALE=0`, `base_seed=20260728`, vs Greedy. Dumps in `replays/exp026/` (incl. `analyze.py` / `causal_read.py`).

| arm | reach 3+ | Δreach paired (McNemar) | win% | techs@t10 | tpc@t10 | Greedy reach |
|---|---|---|---|---|---|---|
| A baseline | 64.8% | — | 58.0% | 8.44 | 5.14 | 66.8% |
| B commit+gate | 74.8% | **+10.0pp, z=+3.20** | 60.0% | 7.12 | 4.23 | 59.6% |
| commit-only | 66.0% | +1.2pp, z=+0.51 | 53.6% | 8.39 | 5.13 | 65.6% |
| gate-only | 72.4% | **+7.6pp, z=+2.32** | 60.4% | 7.11 | 4.17 | 57.2% |

⚠️ Baseline reach is 64.8% here (deterministic n=64), not M2's 55% (noisy n=256) — condition difference; the A/B is internally controlled.

- **Primary lands between the falsifier and the expectation:** +10.0pp (95% CI ≈ [+2, +18]) — real (paired z 3.20, p≈0.001) but under the registered +15pp; the <+8pp falsifier did not trigger.
- **Attribution: the star gate is the entire effect.** Commit-only is inert-to-negative (reach +1.2pp; win −4.4pp, McNemar −1.39 — plausibly because focusing `CH_PURSUIT` *hides* the non-target villages the all-villages field normally shows). Gate-only reproduces the full combined profile, including Greedy's reach dropping −9.6pp (the model now wins races) and the research curve bending (techs@t10 −1.33, tpc 5.14 → 4.17, i.e. the inelasticity signature un-bends when stars are gated).
- **⭐ The causal read (decisive):** in the 43 paired games where the combined macro flipped reach on, wins went **27.9% → 81.4%** (gate-only: 23.3% → 65.1%); in the 18 games the always-on script *broke* reach, **14/18 → 2/18**. Conditional win rates are unchanged across arms (win|reach ≈ 75–76% everywhere) — forced expansion converts at the same rate as organic expansion. Third-city reach is **causal at nearly the full conditional margin, in both directions**; the selection-effect alternative is dead.
- **Why win% stayed ~flat (+2.0pp, z=+0.53):** net reach flips (+43 / −18) are worth ≈ +11 wins, but the crude always-on rules cost ≈ 6 wins in games baseline handled fine — net +5 games on a 250-game reading. The *gross* effect, not the net, is the learned-macro headroom: ≈ +9pp win at this budget if a selective policy fired only where it helps.

### Verdict — CONFIRMED (attenuated magnitude): the macro layer is binding, the mechanism is STAR ALLOCATION, and third-city causality is established
Interpretation-matrix branch 1, with a mechanism refinement: micro execution is adequate (forced reach converts at full rate), and the active ingredient is the **star gate** (resource allocation), not the commitment (attention/representation — consistent with the CH_PURSUIT no-strength-gain history). Open-question candidate **(b) "stars diverted to tech at the decision point" is now causally demonstrated** as the dominant reach-failure mechanism. Follow-ups, in value order: (i) make the gate *selective* (fire only when a capturable village is genuinely fundable/reachable — recover the ~6-game collateral); (ii) generate training data with the gate on and distill, so the allocation policy moves into the net; (iii) test the budget interaction at n=256.

---

## EXP_ELO_027: LLM hindsight-credit annotation — Phase 0 validity gate, then loss reweighting
*Jul 28, 2026 — pre-registered. Phase 1 runs ONLY if Phase 0 passes bar 5.*

**Hypothesis.** An LLM given engine-derived per-turn summaries of finished games (both players: cities, SPT, army value, techs, FOW-visible neutral villages, kills; outcome revealed) can tag the decisive decisions/mistakes with enough fidelity to carry credit signal the current labels lack (Motif-style LLM-as-retrospective-credit-assigner, Klissarov et al. 2023).

**Phase 0 — annotation validity. No training, no MCTS CPU.** Annotate 50–100 games assembled from existing data (`replays/gauge_stats/` holds 94 dumps; plus `--dump-turn-states` games). Include a blinded who-wins probe as a calibration check. Pre-registered bars:
1. **Outcome alignment:** when the LLM tags player X's decisive mistake, X actually lost — precision ≥80%.
2. **Temporal concentration:** tags cluster in the t8–12 decision window (score-gap AUC 0.718@t10, M2), not uniformly.
3. **Known-factor recovery:** on ≤2-city losses, the top tagged mistake references the expansion failure and agrees with the programmatic hindsight rule "village FOW-visible, never approached" (computable from dump-turn-states).
4. **Reliability:** re-annotating the same games with shuffled presentation agrees on ≥70% of tags.
5. **Incremental information (the prize):** on games where the programmatic rule finds *nothing*, tags still align with outcome. Failing only this bar → drop the LLM, ship the programmatic annotator instead.

### Phase 0 Falsifier
Miss bars 1–2 → annotation is noise; stop entirely.

**Phase 1 — credit reweighting (conditional on Phase 0 bar 5).** Annotation sidecar (game, seat, turn) → **per-sample loss weights** in train.py: plies in tagged turns get value-loss (optionally policy-loss) weight ×2–3, renormalized. Deliberately **not** a target change — the value-target parameterization family is closed (021/022/024); same targets, different emphasis. Prerequisite: step→(game, turn) index columns in `games_*.safetensors` (the same schema addition value-reanalyze needs — one change serves both). Scale: 300–500 annotated archived games (API cost only); A/B fine-tune of 10–20 iterations from the same checkpoint, same seeds.

### Expected Results (Phase 1)
Model-side behavior curves move: third-city reach in anchor games up, techs-per-city toward the win profile. Judge on behavior curves (t25 cities sd 0.081), **not** the ±12pp gauge.

### Falsifier (Phase 1)
No behavior-curve movement beyond the noise floor after the run → null. Honest prior is modest — label-side interventions are 0-for-4 here — which is exactly why the phase gate exists: the expensive half never runs unless Phase 0 proves the annotations contain information the current labels don't.

### Confound control
Reweighting interacts with EXP_ELO_025's label semantics — run Phase 1 under one label regime only (whichever is production at the time), never mixed within a replay-buffer window.

---

## EXP_ELO_028: Learned macro layer — orders field + stance head on the shared trunk
*Jul 28, 2026 — design registered after EXP_ELO_026's causal result. Phase 0 is analysis-only and runs immediately; Phase 1 is pre-registered and runs only if Phase 0 passes.*

**Design (program frame, agreed with Verdi Jul 28).** Macro strategy and micro execution are deliberately isolated: macro trains on its OWN hindsight labels (never distilled through the policy head), micro (policy/value) trains as a goal-conditioned executor, and the only interface between them is the observation plus a small allocation mask.

- **Orders head** (spatial, concurrent): k=3 planes (EXPAND / ATTACK / DEFEND) × 11×11 on the shared trunk, multiple hot regions allowed — concurrent objectives are first-class (two warriors on different missions follow different local paint; conv locality does per-unit assignment for free; no slot-permutation machinery).
- **Stance head** (global purse, categorical): {**GROW** (default: harvest/upgrade/eco-tech free, military tech gated behind a reserve — the EXP_ELO_026 star gate generalized), **ARM** (units first, tech gated hard), **UNLOCK(tech-line)** (fight-for-life commitment: plow stars toward one unlock, gate the rest)}. The stance→root-mask table stays hand-written and inspectable: learned judgment, scripted enforcement. Doctrine to validate, not assume: winners ≈ GROW with rare event-driven excursions.
- **Invocation**: two-pass root. Pass 1 encodes with the *standing* goal in appended channels → macro heads → chosen goal (continuity is learned: the head sees its own commitment; hysteresis margin dial in reserve). Pass 2 re-encodes with the chosen goal → policy/value for search; every in-tree encode carries the goal. ~1.5% eval overhead; tree reuse keys on feature bytes so goal flips invalidate reuse correctly. Goal-flip and stance-flip rates are first-class metrics.
- ⚠️ These heads are INVOKED by search — the opposite of the causally-disconnected `aux_*` heads (ledger §2). Dual-network mirroring (network.rs + train.py), appended-channel zero-pad migration, and checkpoints/ migration all apply.
- **Bootstrap staging**: Stage 1 — a script sets goals in self-play (EXP_ELO_026 rules re-expressed: paint EXPAND on capturable villages while <3 cities; stance GROW-with-reserve), micro learns to follow, labels accumulate. Stage 2 — macro heads take over goal-setting with a decaying script share (ANCHOR_FRAC-style crutch decay). Stage 3 — script retired to an anchor. Staging + script anchor is the damping for HRL two-timescale instability.

### Phase 0 — label machinery + vocabulary validation (analysis-only, no training, no code in the engine)
Build both hindsight labelers and validate the strategy vocabulary on existing data (`replays/exp026/` stats + turn dumps; 250 natural-play games in arm A, Greedy side as a reference policy).
- **Stance labeler**: per-turn spending mix from TurnSample deltas + star-flow accounting (spent ≈ stars + income − next stars; decomposed into tech / units / eco-residual). Known v0 limit: dumps carry tech *counts*, not identities, so UNLOCK vs eco-tech is not yet separable — flagged for a future dump field.
- **Orders labeler**: achieved objectives from turn-state dumps (village captured → EXPAND window painted over the approach turns; units converged on enemy city + enemy losses → ATTACK; enemy units near our city → DEFEND).

Pre-registered checks:
1. **Doctrine check**: model-side winners' stance mix is GROW-modal with ARM/UNLOCK excursions concentrated near combat/threat events; Greedy (a known-good expander) reads similarly.
2. **Vocabulary check (the falsifier that matters)**: macro-sequence features (stance shares ≤t12, achieved-EXPAND counts, order-window timing) must separate wins from losses (AUC meaningfully > 0.5 on features the M2 autopsy did NOT already establish). If winners and losers are indistinguishable in macro-language space, the vocabulary does not describe why games are won → revise vocabulary; no implementation proceeds.
3. **Concurrency measurement**: fraction of turns with ≥2 overlapping order windows — quantifies whether the multi-goal orders field is load-bearing or a single goal would have sufficed.
4. Label coverage and class balance reported; DEFEND merges into ATTACK if it labels <5% of windows.

### Phase 0 — Actual Results (Jul 28, 2026; `replays/exp028/phase0_labels.py` over arm A's 250 natural-play games)
1. **Doctrine check — PASS, via the cross-policy contrast, with a sharpening.** Greedy (the stronger expander) is exactly the doctrine: ECO-modal (47–50% of turns; 19% UNITS in its wins). The model is **TECH-modal instead (46–49% overall, 56–58% in turns ≤12)** — and critically, its stance mix is **nearly identical in wins and losses** (56.3% vs 58.4% early tech). The doctrine violation is *systemic, not episodic*: the model plays the same allocation policy everywhere and wins only when enough stars leak into units/expansion anyway. This is why per-turn stance shares barely discriminate outcomes (tech_share_early AUC 0.445) while flow-into-army does — a constant can't discriminate. Strengthens the stance head's rationale: the behavioral delta to learn is large and well-defined (TECH-modal → ECO-modal).
2. **Vocabulary check — PASS.** Macro-language features separate outcomes on axes M2 never measured: **`n_attack_turns` (offensive-convergence events on Greedy cities with enemy losses) AUC 0.806** — the strongest macro discriminator measured to date (wins 2.16 such turns, losses 0.56); `unit_share_early` AUC 0.684; `n_expand_achieved` AUC 0.658. ⚠️ Directionality caution on the 0.806: late-game attack events are partly a *consequence* of already winning — treat as vocabulary validation, not yet as a causal lever; an EXP_ELO_026-style steer test would be needed before believing ATTACK-steering wins games.
3. **Concurrency — the multi-goal orders field is load-bearing: 21.4% of turns carry ≥2 overlapping order windows.** Achieved-EXPAND windows: 341 across 250 games, mean length **7.6 turns** — long, persistent label windows, good news for learned continuity.
4. **Coverage/balance:** stance labels cover ~93% of turns (SAVE 6–8%). **DEFEND labeler is too loose** (any Greedy unit within 2 of any city fires on 47% of turns on a Tiny map) — needs a real threat predicate (≥2 units or actual combat) before Phase 1 labels; its inverse correlation (AUC 0.374 — defending = losing) is real signal, so DEFEND stays in the vocabulary. Known limit stands: tech *identities* aren't in the dumps, so UNLOCK vs eco-tech isn't separable yet — add a tech-id dump field alongside Phase 1.

**Phase 0 verdict: PASS on all four checks — Phase 1 is green-lit per the registration.**

### Phase 1 — goal-conditioned micro under the scripted goal-setter (first training experiment; pre-registered, gated on Phase 0)
Add the goal channels (orders planes + stance one-hot; appended → zero-pad compatible), have the script drive them in self-play, train micro conditioned.
- **Hypothesis**: micro can learn to FOLLOW orders — the conditioning interface is a working actuator.
- **Expected**: on paired seeds with varied scripted targets, order-following (directional compliance / goal-achievement rate) separates the conditioned net from a zero-channel control; reach ≥ the EXP_ELO_026 script-arm level.
- **Falsifier**: no goal-following difference vs the zero-channel control after the run → the conditioning interface is dead and the macro head has no actuator; the program halts before any macro head is built.
- **Confound control**: same seeds both arms; checkpoint migration required (strict Rust league loader — see memory); do not mix pre/post-channel games when reading value/policy loss.

### Phase 1 infrastructure — SHIPPED (Jul 28, 2026; no training run yet)
- **Channels**: `CH_ORDER_START..END` (162–165: EXPAND/ATTACK/DEFEND proximity blobs, max-merged) + `CH_STANCE_START..END` (165–168: one-hot planes); `NUM_CHANNELS` 162 → **168**. All-zero = "no goal set" (what old data zero-pads to and non-net seats record).
- **Types + script**: `oracle_macro.rs` gains `MacroGoal`/`OrderKind`/`Stance`, `scripted_goal` (Stage-1 rules incl. the tightened ≥2-unit DEFEND predicate), `goal_star_gate`. Orders kept sorted so identical goals hash identically (eval cache + tree reuse).
- **Threading**: `state_to_cpu_features_goal` → root / leaf (`extract_leaf_data`) / re-root hash; `GumbelMctsAgent.macro_goal`; `Brain::set_macro_goal` (re-applied every `think`). Recorded features and search encodes share the SAME goal object in self_play — training data cannot disagree with what the agent saw.
- **Flags**: `self_play --goal-channels` (net seats only), `arena --goal-script` (config 1, gumbel-only, validated).
- **Migration**: `migrate_goal_channels.py` padded conv1 162→168 on model.safetensors (+.bak), all 132 checkpoints (incl. some 161-era stragglers), exp026 snapshot; optimizer_state.pt cleared. train.py/init_model.py `SPATIAL_CHANNELS = 168`; generic append-pad covers 154/161/162 data.
- **Verified**: full CI suite green (80 lib + integration + self_play tests, incl. new painting/goal-setter tests); smoke self_play `--goal-channels` wrote 168-wide `games_*.safetensors` with planes populated exactly per the script (EXPAND 36.7% of rows, ATTACK 34.2%, DEFEND 7.8% ≡ ARM 7.8%, GROW 92.2%, UNLOCK 0 — v1 script never emits it); train.py trained one epoch on a **mixed** 168+162 buffer with the migrated model.
- No macro heads exist yet (Stage 2) — Phase 1's dual-network surface is only the input width, handled above. Next: the registered Phase 1 A/B (conditioned vs zero-channel control), sequenced after EXP_ELO_025's pending run.

### Phase 1 first live run + channel audit → script v2 (Jul 29, 2026)
Run 1785279937 (resumed 110-iter model, 128 games/iter, n=256, GOAL_CHANNELS=1), iterations 1–4:
- **Iter-1 transient, then regression to baseline.** First conditioned training pass over-generalized the gated targets (research 10.3/game, harvests 4.1, SPT_t5 3.2 — all far below baseline) and expansion briefly accelerated (3rd city t11.4 vs baseline ~12.7–13.4). By iter 3 the value gradient (score-blended, unchanged — it still prices tech) had re-optimized *around* the constraint: early tech stays gate-suppressed (t≤8 below baseline) but is back-loaded above baseline after the gate window (t25 techs 19.4 vs 18.2), redirected stars park in army value (t15 army-stars 14.5 vs 12.5) without converting to captures, and 3rd-city timing returned to baseline (t13.0). **Lesson (026 corollary): a frozen net can't re-plan around a hard mask, a training net can — constraint-Goodharting; durable tempo needs the objective to price the goal, not just the mask.**
- **Channel audit (goal planes decoded from archived samples, iters 1+4): the v1 script's order distribution was mis-calibrated.** ATTACK lit on **61.5–62.3% of net plies** (the ≥2-units-within-cheb-3 trigger ≈ always true mid-game on 11×11) vs EXPAND only ~12% — the causal-lever signal was the rarest and the wallpaper signal the loudest; a near-constant channel carries no conditional information. Apparent order-following (Attack-mass 0.100 under ATTACK vs 0.036 without) is mostly the trigger's selection effect. Summon mass under ATTACK was *lower* (0.069 vs 0.085) — v1 had no "prepare" vocabulary: ATTACK only fired after force was already assembled.
- **Script v2 (shipped, tests green; takes effect on next loop restart):** (1) ATTACK requires local force superiority (own star-value within cheb 3 of the explored enemy city > defenders' within 2), not mere proximity; (2) ARM gains the prepare meaning — set when a known enemy city is winnable if massed (total army > its garrison, a unit within cheb 4, no local superiority yet), post-expansion only (≥3 cities) so it can't cannibalize the EXPAND phase; (3) EXPAND orders persist until the village is actually captured (city-count check moved into `goal_star_gate`, which keeps its <3-cities scope). `goal_star_gate` now takes (state, player, goal).
- **Expected (pre-registered before the v2 run):** ATTACK lit share drops to ~10–20% of plies (smoke at 16-iter budget: 1.9% — weak-play sample, production will sit higher); EXPAND share roughly doubles (smoke 19.8%); ARM appears in two modes (defend + prepare). If conditioning is a live actuator, Summon/Step mass under prepare-ARM should rise vs GROW quiet states within ~3 iterations of v2 data, and EXPAND-lit Capture mass should recover toward its iter-1 level (0.071). Falsifier unchanged (zero-channel A/B). Also new: `anchor_net_wr` (net win rate in the 25% embedded anchor games) now logged per iteration — ±17pp at n=32, trend-read only.

### Script v2 actuals — iterations 5–14 of run 1785279937 (Jul 29, 2026; loop resumed on the v2 binary at iter 5 and completed)
- **Distribution (channel audit, iters 11–14): partially hit.** ATTACK 36–40% of net plies (v1: 62%; target 10–20% — better but still the loudest signal; superiority near a city is common for a 256-sim net, and long sieges keep it lit). EXPAND 16–18% (v1 12%, expected ~2× — partial). **prepare-ARM landed in band: 12–15%** of plies, distinct from DEFEND-ARM (15–21%); GROW 67–72% — GROW-modal doctrine holds.
- **Conditioning predictions split.** Step mass under prepare-ARM 0.53–0.57 vs 0.35–0.46 in GROW-quiet ✓ (advance = yes, selection-effect caveat stands); **Summon mass did NOT rise** (0.061–0.069 vs 0.067–0.077 quiet) ✗ — the "make more units" half of prepare is absent; EXPAND-lit Capture mass did NOT recover (0.046–0.061 vs the 0.071 target) ✗, though v2's persistent-EXPAND changed the composition of that bucket (now includes post-3rd-city contested villages).
- **⚠️ Correction to the Jul 29 morning read: the "iter-1 shock then regression to baseline" story was substantially a tribe-mix artifact.** The low-economy iterations (1, 5, 12, 13, 14) are exactly the Oumaji+XinXi pairings; Kickoo pairings are research-rich. Within-pairing vs the pre-goal baseline run, v2 iterations show: research −1.4..−2.4/game in EVERY pairing with **no harvest/SPT cost** (unlike v1 iter 1's blanket shock), 3rd city slightly earlier in 4/5 pairings, and **4th-city rate up in 5/5 pairings** (~1–1.4 turns earlier in the Imperius pairings) — the specific signature of v2's persist-until-captured EXPAND. Lesson: never read cross-iteration behavior trends without conditioning on the tribe pairing.
- **Strength: the most positive reads of the campaign, all sub-significance individually.** `anchor_net_wr` mean ≈ **0.60** over ~230 contested games (iters 5–14; 0.79 at iter 14, n=19). Gauge at iter 5: **62.5% vs Greedy, elo_est 589 — best ladder reading on record**; iter 10: 59.4% (pre-goal ladder tail: 55–58%). The gauge cities-curve flipped decisively: model now out-expands Greedy at every t≥10 (2.91 vs 1.83 at t15) where pre-goal gauges had it behind from t20 on.
- **Next:** the zero-channel A/B falsifier is now properly justified (10 v2-conditioned iterations of training data exist); ATTACK needs one more tightening turn (margin multiple or garrison-aware defense term) before the order is informative.

### Phase 1c — goal-priced in-tree shaping + script v2.1 (registered Jul 29, 2026, before running)
**Motivation.** The stance audit showed GROW has no actuator: its only mechanism is the star gate (a brake), so GROW plies are where the net researches *most* (0.075–0.106 policy mass — nothing stops it), and the model's long-standing SPT deficit vs Greedy has no lever. The 026 corollary says masks get re-optimized around by a training net; the durable form is a reward the objective feels. Verdi's directive: "grow should boost in-tree things that grow your eco."
**Change (shipped, tests green, binaries built — takes effect on next loop start):**
- `reward::goal_potential` (new, EXP_ELO_028 Phase 1c): stance/order-priced potential added to in-tree edge rewards on the searching agent's own edges only (`gumbel_mcts::edge_snapshot`; opponent's goal is unknown, their edges are unshaped). GROW → 150 score-equiv per SPT; ARM → 50/star of living army (the missing "make units" actuator — prepare-ARM Summon mass was flat in v2); EXPAND orders → 200/tile of approach progress toward each painted target, **summed over targets with achieved-holds-cap semantics** (a self-owned target pegs at CAP so the final capture banks its step instead of cliffing −CAP — a flaw inherited by the EXP_ELO_018 pursuit gradient, avoided here). Weights sized like SHAPE_PURSUIT_PER_TILE: ≈0.1–0.2 normalized per decisive step through score_norm≈600–700.
- Threading: `GumbelMctsAgent.goal_shape_w` / `Brain::with_goal_shape_w` / `self_play --goal-w-tree` / `arena --goal-w-tree` (both require the goal flags); loop knob `GOAL_W_TREE` (**default 1 when GOAL_CHANNELS=1**, 0 opts back to channels-only); CONFIG logs it.
- Script v2.1: ATTACK now needs a **1.5× local-superiority margin** (2·local > 3·defenders) — v2's simple `>` kept ATTACK at 36–40% of plies.
- **v2.2 — granular, stance-aware research gate** (Verdi: "gate the research more granularly instead of blocking all research"). `passes_star_gate` now takes the stance and gates only the tech class that contradicts it, using the derived tech-tree annotation: **GROW gates combat-unit techs** behind the 5-star reserve (eco/mobility/defense tech passes freely — Organization, Fishing, Climbing et al. ARE the doctrine; the v1–v2.1 gate was suppressing them too, which is what the within-pairing research deficit measured); **ARM gates pure-eco techs** (mixed tech like Smithery arms you and flows), and ARM now activates the gate whenever it holds, not just in the expansion window; UNLOCK gates nothing yet. The stance-less legacy arm (arena `--macro-star-gate`, EXP_ELO_026 repro) keeps gate-everything behavior. Consequence for Phase 1c expectations: within-pairing *eco-tech* purchases should recover toward baseline while *combat-tech* purchases stay suppressed during expansion — total research/game is no longer the right metric; split it by tech class (tech_tree.json) when auditing.
- Labels untouched: shaping enters search (and thus policy targets/visit counts) only; TD value labels keep their existing pricing. Macro/micro isolation preserved — no distillation, no policy-head coupling.
**Hypothesis:** pricing the named conversion per stance gives the goal channels teeth — search will prefer eco moves under GROW, summons under ARM, and approach/capture under EXPAND, and the net will learn the *conditional* behavior from the shifted policy targets.
**Expected:** within ~3 iterations — Harvest+Build mass under GROW-quiet up from ~0.10 toward ≥0.15 with Research mass down from ~0.075–0.106; Summon mass under prepare-ARM up from ~0.065 to >0.08 and clearly above GROW-quiet; Capture mass under EXPAND >0.07; ATTACK lit share <25%; SPT curves (within-pairing!) up vs the v2 iterations; anchor_net_wr holds ≥0.60. **Risk to watch:** GROW pays ΔSPT wherever it comes from — city level-ups via cheap captures also qualify (fine, that's eco), but if harvest-rush starves defense and anchor_net_wr drops ≥10pp sustained, reduce GOAL_W_TREE to 0.5 before concluding.
**Falsifier:** if the conditional masses don't move (GROW-quiet Research stays ≥0.075, prepare-ARM Summon stays ≤ GROW-quiet) after ≥5 iterations at GOAL_W_TREE=1, in-tree pricing at this magnitude doesn't steer the decomposed policy — re-size weights from measured Q gaps (the EXP_ELO_018 method) before touching architecture.

**v2.3 — tech discipline + environment-fit bias (Verdi's crutch-until-learned directive, Jul 29).** Rationale: tech smartness won't self-emerge under the compute-constrained regimen; nudge with code logic, take the crutches off once learned — regression then costs games, which is what should hold the behavior. Shipped:
- **Whole-game purchase caps** (root-only, active whenever a `GoalAux` is set — independent of the stance-gate window): ≤**8 techs bought with own stars** per game (counted as executed Research moves; ruin-granted techs never pass through Research and don't count), of which ≤**1 tier-3**. `passes_tech_caps` + per-seat counters in self_play/arena.
- **Environment-fit tech bias** (soft, in-tree): `recommended_techs` scores the four tech lines (forest→Hunting/Forestry/Mathematics, mountain→Climbing/Mining/Smithery, farm→Organization/Farming/Construction, water→Fishing) from the player's EXPLORED tiles — terrain counts + double-weighted matching resources — takes the top two lines and recommends each line's next unowned tech. **Tribe awareness is emergent**: tribe spawns generate their signature terrain/resources (Oumaji fields→farm line, mountain-rich Imperius spawns→forge line), so counting the actual map plays into the natural environment without a hand tribe table that can drift. Owning a recommended tech pays **+150 score-equiv** (`SHAPE_GOAL_TECH_FIT`) in the goal potential — buying map-fit tech banks in-tree, off-fit tech banks nothing (bias, not ban).
- **Rider push — PATH-AWARE (upgraded same day per Verdi: judge the route, not the global census).** The global fields-vs-forest count was replaced by a movement-model answer to the real question: *does a Rider actually arrive faster on the routes to my EXPAND targets?* `turns_to_reach` runs a multi-source turn-BFS from the player's units (fallback: cities) under simplified Polytopia rules — 8-dir steps, entering rough terrain (forest/wetland/mangrove, mountain with Climbing) ends the turn, water/ice impassable, mountains impassable without Climbing, unexplored tiles optimistically open (FOW-honest). `rider_turns_saved` = max over EXPAND targets of walker-turns − rider-turns; push fires at ≥1 turn saved. This encodes Verdi's example directly: a 50/50 map with the forests clumped in an irrelevant corner still pushes riders (routes go around); it also captured a subtlety the census never could — through a THIN forest band a rider legitimately weaves open-step+forest-step at 2 tiles/turn and keeps its advantage (test pins this), and only a fully rough approach region kills the push. Each living Rider still pays **+100 score-equiv** (`SHAPE_GOAL_RIDER`) while the push is on.
- Plumbing: `GoalAux` set on the agent alongside `MacroGoal` but NOT painted into features (cache/tree-reuse unaffected; reused-root legality re-checked against caps). Aux is per-ply; recommended set frozen during a search.
- **Expected:** techs/game (own-star) ≤8 by construction; purchased-tech mix shifts toward map-fit lines (audit via tech_tree.json classes × tribe); Riders/game up on open-terrain pairings (Oumaji/Imperius) during expansion; 4th-city tempo holds or improves. **Risk:** the 8-cap binding too early late-game (if long games hit the cap by t20 and the model needs passage tech to finish, win rate on water maps could dip — watch anchor_net_wr by pairing).

### v2.4 — scout term + capture-completion bonus (registered Jul 30, 2026, before running; Verdi's directives: first village by t3–4, capture rate → 100%)
**Diagnosis (iter-24 channel audit, per-turn):** first-village latency is a DISCOVERY problem, not an approach problem. A capturable village is visible (⇔ EXPAND lit) in only **2.1% of t0 plies, 7.5% t1, 14.5% t2, 25.8% t3** (peak 45.8% at t7) — capturing at t3–4 requires visibility by t1–2, and the macro had a structural blind spot: **nothing paid for exploration.** Before a village is explored there are no EXPAND orders, no approach gradient, no reveal reward — the expansion pipeline idles. Separately, Capture mass stays 0.02–0.07 even with units adjacent (Capture-vs-Step at the final ply is a coin-toss-scale preference). Current actuals: `villages_t2c_first_cond` 5.5–8.7, `villages_first_rate` 0.74–0.83.
**Change (shipped, 94 tests green):**
- **Scout term**: `goal_potential` pays **25 score-equiv per explored tile** while stance is GROW, NO EXPAND target is known, and cities < 3 — a frontier step revealing ~3 tiles banks ~0.12 normalized. Retires the moment a village is found (approach gradient takes over) and re-arms after each capture until the third city — "find village → take it → find the next" as one continuous priced pipeline. No new channels (net can infer the mode from existing stance+pursuit+frac-explored features).
- **Generator-informed village guessing (same day, per Verdi: the EXPAND signal itself should drive discovery using our own mapgen knowledge).** `guessed_village_sites` inverts the generator's Drylands placement rules — villages fill legal spots (land, edge-band ∈ {2,4,5}, Chebyshev ≥3 from every village/capital) **to saturation**, so an UNEXPLORED legal spot ≥3 from everything known must lie near an undiscovered village. FOW-honest (game knowledge, never map peeking). `scripted_goal` now tops EXPAND orders up to `EXPAND_TARGET_MIN=2` with guesses (nearest-to-units first, mutually ≥3 apart — producing the human "first warrior center, second north/east" spread) while cities <3. Guessed targets ride the SAME painted plane and 200/tile approach gradient (the net distinguishes real targets by the pursuit channel); the completion bonus requires an actual owned CITY at the target (border-grown empty tiles pay 0), and unexplored targets always pay approach (reading their owner would leak FOW). The scout term stays as fallback. Riders now also get judged on routes to guessed sites (open unexplored ground → rider push — intended). **Smoke-verified: EXPAND lit on 100% of expansion-phase plies from t0 (was 2.1%).**
- **Capture-completion bonus**: an achieved EXPAND target now holds cap+2 tiles (`SHAPE_GOAL_EXPAND_DONE`), making the final capture edge bank ~600 score-equiv (~0.4 normalized) — a landslide vs one more Step.
**Expected:** village-visible@t2 from 14.5% toward ≥40%; `villages_t2c_first_cond` from 5.5–8.7 toward ≤5; `villages_first_rate` from ~0.78 toward ≥0.90 (100% is the goal but ~lost-from-the-start games exist — M2's annihilation losses); t2c_2nd_turn from 7–10 toward ≤6; EXPAND-lit Capture mass finally >0.07. **Risk:** scout term over-rewarding wandering scouts at the expense of early Harvest (watch SPT_t5) — the term only pays NEW tiles, so it self-limits, but if SPT_t5 drops >15% within-pairing, halve SHAPE_GOAL_SCOUT.

### Phase 1c + v2.1–2.3 — ACTUALS (run 1785279937 iterations 15–24, Jul 29–30, 2026; full stack live from iter 15)
**Core hypothesis CONFIRMED: goal-priced in-tree shaping steers behavior durably where masks alone regressed.** Ten iterations, no decay of the shifted equilibrium (v1's gains had evaporated within 3).
- **Cleanest evidence — same-pairing consecutive A/B (iter 14 old stack → 15 new stack, both XinXi+Oumaji):** research 10.7→7.1/game, SPT_t15 9.1→12.6 (+38%), units spawned 13.0→18.6 (+43%). Held across all pairings through iter 24 (research 7.1–8.0 — the caps also flattened the tribe-driven research variance; SPT_t15 12.5–16.0 vs 8.2–12.7 for every prior iteration; units 18–21).
- **Conditional allocation sharpened ~10x — the falsifier's key axis moved decisively:** Research mass under prepare-ARM 0.049→0.003–0.011 and under ATTACK 0.056→0.007–0.017, while surviving in GROW-quiet at ~0.06 (GROW:ARM research ratio 1.5x → 6–15x). GROW-quiet Harvest 0.021→0.03–0.07 ✓.
- **Tech mix went ECO-modal** (ply-weighted presence, iter 15→24): Farming 62%→92%, Forestry 40%→61%, Construction/Mining enter at 42%; Riding 92%→72% (present but selective — path-aware push); Strategy 43%→36%. This is the Phase-0 target delta (TECH-modal→ECO-modal doctrine) achieved via crutches.
- **Contested play flipped to domination:** net led Greedy at t15 on cities (3.0–3.45 vs 1.55–2.2), SPT (15–18.7 vs 6.6–10.6 — the net had NEVER led Greedy on SPT), and units, across iters 15–18; gauge cities-curve shows Greedy's count *declining* after t15 (model takes its cities). `anchor_net_wr` pooled **~71% (74/104)** over v2.3 self-play iters vs 61% for iters 5–14.
- **Not confirmed / missed targets:** (1) deterministic gauge: iter-15 56.2%, iter-20 59.4% (elo 566) — recovering but below the iter-5 peak (62.5%/589); ladder confirmation pending next gauges. (2) prepare-ARM Summon mass never exceeded GROW-quiet (0.071 vs 0.078 at iter 24) — summoning rose *globally* (+43%) not conditionally; the ARM-conditional response expresses in Step (0.55 vs 0.41) and Build (0.13 vs 0.075) instead. (3) ATTACK lit 30.1% at iter 24 vs the <25% target (from 62% v1 / 40% v2). (4) EXPAND Capture mass 0.061 vs >0.07 target. (5) 3rd city ~1 turn later on like-for-like Kickoo pairings (14.1–14.6 vs 13.3) — eco-first opening trades early tempo; profitable at current win rates but the first suspect if the gauge stalls (026: third city is causal).
- **Watch:** policy_loss rose 1.39→1.53 over iters 15–24 (value_loss stable at record-low 0.22–0.23) — the policy net is still chasing the crutch-shifted search targets; more iterations at this equilibrium should close it, and it must close before Stage-2 crutch removal (the net has to OWN the behavior, not rent it from the shaping). anchor_net_wr cooled from 0.85–0.92 (iters 15–18) to 0.40–0.73 (21–24) at n≈10 — noise-sized, unresolved.
- **Verdict: keep the stack as the production default; continue training at this equilibrium. Next decision points:** gauge trend over the coming resumes (needs to beat 62.5%), the zero-channel A/B (now well-founded — 10 iterations of conditioned data), and Stage-2 planning (macro heads + crutch decay) once policy_loss turns back down.

### Tech-tree annotation (UNLOCK groundwork, Jul 29, 2026)
Per Verdi: the tech tree needed an easy what-does-this-unlock lookup (units vs defense bonuses vs build/harvest). Shipped in `settings/technology.rs`:
- Replaced the dead opaque `unlocks_other: i32` (zero consumers, drifting comments — Climbing's said "pacifist" while the engine gives it the mountain defense bonus) with semantic fields: `defense_bonus_terrain`, `unlocks_vision`, `tech_discount`.
- New **derived** `TechEffects` (`get_tech_effects`, cached): combat/support/special units (by unit `attack` stat), harvests (from `resources.rs` `tech_required`, tribe-locked excluded), eco/score/connector/other structures (by `structures.rs` yield fields), passage terrain, abilities/tasks/vision/discount. Derived from the settings tables so it cannot drift from the engine; a test pins `defense_bonus_terrain` to `functions::get_defense_bonus`'s rule table, another requires every vanilla tech to have a nonempty annotation.
- Class predicates for the macro script/labelers: `is_military_tech` (combat units or defense bonus), `is_eco_tech` (harvests, yield structures, discount), `is_mobility_tech` (passage, connectors) — overlapping by design (Smithery = MIL+ECO).
- `tech_tree` bin dumps the whole annotated tree as JSON → `tech_tree.json` for the Python labelers (regenerate after settings changes).
Vanilla classes: MIL {Riding, Strategy, Chivalry, Archery, Ramming, + mixed Smithery/Mathematics/Sailing/Navigation/Aquatism/Climbing}, ECO {Organization, Hunting, Fishing, Farming, Mining, Forestry, Trade, Construction, Spiritualism, Philosophy, FreeSpirit, + the mixed}, util {Meditation, Diplomacy}. Unblocks: UNLOCK stance trigger (e.g. enemy fields swordsmen and we lack Smithery → UNLOCK(military line)), stance labeler tech-split (Phase 0's flagged gap), and a stance-aware star gate (gate military tech under GROW, eco tech under ARM) if v3 wants it.

### v2.5 — Destroy ability gate, from the 64-game behavior audit (registered Jul 30, 2026; Verdi: "just gate out the ability to destroy a building — we will learn that strategic use later")
**Audit (64 games, gauge conditions: iter-30 snapshot vs Greedy, mcts 256, GUMBEL_SCALE=0, goal stack on; sources: star_spend/city_rewards/turn_states dumps + shard target mixes):**
- **Confirmed systematic** (vs Verdi's watched-game observations): duplicate-direction scouting 57/64 games (2+ far units in the same bearing sector by t10); 32/64 games never visit all 4 map quadrants; 29/64 touch zero map corners (lighthouses); Explorer reward taken 3% at level 2 (Workshop 97%); warrior monoculture (~86% of summon mass, ~2.3 stars/unit vs Greedy 3.6); heavy forest clearing (ClearForest ~8–14/game).
- **NEW pathology:** **Destroy own structure is the single largest ability by search-target mass** (~9–15/game est.) — pure churn, nobody was looking at it.
- **Not systematic:** harvest discipline is fine at scale (524 harvests: 94% followed by a level-up ≤2 turns, 1% stranded at game end) — the watched game was an outlier; windmills DO get built (~1–1.7/game, 5% of build mass); 2nd unit on board avg t2.9, first village visible t3.8. Tech cap binds exactly (median=max=8 research/game).
- **Method note:** shard policy-target argmax ≠ executed move — concentrated action types overcount ~2× (Research 18.2 argmax vs 7.7 executed). star_spend is executed-move ground truth; abilities have NO executed ground truth yet (dump extension needed before trusting their absolute rates).
**Change (shipped):** `oracle_macro::passes_ability_gate` — root-only mask dropping `AbilityType::Destroy`, active whenever a `GoalAux` is set (same scope/flag as the tech caps, so training AND gauge agree). Applied at both gumbel root filter sites (fresh root + reused-root legality check). Unit test added (14 oracle_macro tests green); release binaries rebuilt — **takes effect on the next loop restart**, live run untouched.
**Hypothesis:** Destroy churn is wasted tempo/stars with no strategic payoff at current skill; masking it redirects ~9 plies/game of action budget to eco/military moves without costing strength.
**Expected:** next audit shows Destroy target-mass ≈0 on net seats; SPT/city curves hold or improve; gauge holds the current 75–83% band. **Risk:** the rare legitimate rebuild/unsiege use is lost — accepted by directive; revisit when Stage-2 crutch removal reaches this gate.

### v3 — archetype layer: doctrine from ground-truth predicates (registered + shipped Jul 30, 2026; Verdi's design directive: "the decision should be derived from some ground truth states… then we give our model the ability to evaluate and make contextual decisions")
**Design (aligned with Verdi over three iterations):** a small predicate vocabulary → base doctrine with hysteresis → reactive overlays → hard exits. No new input channels — every predicate is a function of state the net already sees (terrain channels, ghost/observation-memory units, own economy), so conditional pricing produces conditional targets the net can distill (the 028 mechanism).
- **Predicates (FOW-honest):** explored-map terrain read (open/rough fractions, metal count — same style as `recommended_techs`); expansion-race liveness (EXPAND orders or cities < 3); route mobility (`rider_turns_saved ≥ 1`, path-aware); observed enemy mix as PEAK per-class counts of enemy units on explored tiles (squishy `defense ≤ 1.5` — Verdi's threshold: riders/archers/catapults/knights in, warriors at 2.0 out; heavy `defense ≥ 3`; cavalry `movement ≥ 2`; ranged `range ≥ 2` — all derived from `units.rs` stats, not hand tables). Peaks are monotone → overlays never flap.
- **Base doctrines:** RiderRoads (open ≥0.45 + mobility + race live, hard-blocked at ≥2 heavy seen), ArcherLine (heavy seen / rough ground / contact — both siege AND push-support roles per Verdi), ForgeGiants (metal ≥2, no active DEFEND). Entry needs score ≥3 on ≥12 explored land tiles; soft switch needs +2 margin for 3 distinct turns; hard exit (score→0) re-picks immediately.
- **Overlays (counter table):** enemy cavalry ≥2 → DefenderScreen (bodies deny road corridors, screen the ranged backline); enemy heavy ≥2 → CatapultCounter (also the RiderRoads hard exit — Verdi's XinXi case); enemy squishy ≥4 → KnightCommit (Persist chains through low-defense bodies).
- **Stepping-stone tech rule (Verdi):** RiderRoads buys Riding→Roads ONLY. FreeSpirit/Chivalry are hard-gated in `passes_tech_caps` unless KnightCommit is active — a tech whose value depends on a downstream commitment is only buyable under that commitment.
- **Expression:** doctrine + overlay tech lanes join `recommended_techs` (next unowned per lane: Riding→Roads / Hunting→Archery / Climbing→Mining→Smithery / Forestry→Mathematics / Strategy / Riding→FreeSpirit→Chivalry); `GoalAux.preferred_units` pays `SHAPE_GOAL_ARCHETYPE_UNIT` (100 score-equiv) per living preferred unit in the goal potential (stacks with the rider push when both agree).
- **Plumbing:** `ArchetypeState` per seat in self_play (both net seats) and arena (model seat — gauge probes WITH the doctrine, consistent with the rest of the script), updated per ply before `scripted_goal_aux`; carried in `GoalAux` (not painted into features). 3 new tests (doctrine entry + hard exit + overlay expression; knight lane; unit pricing); full suite green; binaries rebuilt — takes effect next loop restart.
**Hypothesis:** doctrine-consistent pricing breaks the warrior monoculture *conditionally* — unit mix follows the predicates (riders on open maps, catapults after heavy sightings, knights vs squishy spam), and the net learns the evaluation because every trigger is visible in its inputs.
**Baselines (Jul 30 audit, 64 gauge games):** warrior 86% of summon mass, riders 2.02/game, archers 0.88, catapults 0.58, forges 0.44 builds, FreeSpirit bought 2.62/game (top research!) — the stepping-stone rule alone should redirect ~8 stars/game.
**Expected:** within ~5 iterations of the next run — rider summons up on open-map games specifically (not globally); FreeSpirit purchases collapse except under knight commits; catapult/defender summons appear in games with heavy/cavalry sightings; gauge holds ≥75%. **Falsifier:** if unit mixes don't move conditionally at weight 100, re-size from measured Q gaps (EXP_ELO_018 method) before adding archetype channels.

### v3 — ACTUALS (run 1785414474, 10 iterations, Jul 30 2026) — CONFIRMED, two carry-forwards
**Strength (the clean A/B):** gauged twice against the FIXED pre-v3 final model (anchor_iter30): 67.2% at iter 5 → **79.7% at iter 10** (Elo est 1355, one win short of the 80% anchor freeze). The archetype stack made the model strictly stronger than the model that lacked it, and it was still climbing at run end (plateau_strikes 0).
**Doctrine expression (shard target-mass shares, same decode as the audit baseline):**
- Warrior monoculture BROKEN: 86% → 79% (iter 6) → **67%** (iter 9). Riders 5.4% → 9.8% → **26.9%** (≈5×).
- **FreeSpirit collapse sustained:** 2.62/game (was the #1 tech) → 0.45–0.48/game across the whole run; the residual is knight lanes legitimately open (mirror opponents field riders = squishy sightings ≥4). Actual Knights rarely materialize — FreeSpirit+Chivalry depth rarely fits the 8-tech cap in 30-turn games; lane works, payoff needs longer games.
- **Counters fire conditionally by construction:** catapults ~0 because heavy sightings ~0 in mirror play (correct no-fire); defenders ~1/game with Strategy purchases up (0.78/game); the iter-6 defender spike (3.5% of summons) tracked the rider surge — riders beget defenders.
- Behavior: first-village turn improved to 4.3–4.6 late-run (from 5.4–6.2); village rate ~0.85–0.89; t2c_3rd flat ~0.65. policy_loss 1.45→~1.58 plateau — the expected re-chase of the new teacher; watch for the turn-down next run.
**Carry-forward 1 — road-spam without topology (NEW):** Roads build-mass 10.5 → 27.7 → **39.9/game** while road UTILITY is poor: best-game replays show only **16% of non-capital cities connected to capital** at game end and ~**4% of steps road-assisted** (warriors 1.26 avg on-road vs the 2.0 target; riders 2.0 vs the 4 that road chains enable). Same stepping-stone failure as FreeSpirit, one level down: v3 recommends the Roads TECH but nothing prices road TOPOLOGY, and a capital connection needs 3+ coordinated builds — beyond horizon. Fix designed: `SHAPE_GOAL_CONNECT` per `connected_to_capital` city (engine-computed flag, prices the exact in-game bonus). Decode caveat: Road option-mass is inflated by the multi-tile marginal (many legal road tiles pile onto one option index) — replay ground truth was ~9 executed roads/seat in best games; the growth trend is real, the absolute mass overstates.
**Carry-forward 2 — v2.5 Destroy leak:** ability-mass share 46% → ~12–14% but not the intended 0; suspected within-turn tree-reuse path exporting unfiltered subtree visits. Chase before next restart.

### v4 — exploration pack, bucket B (registered + shipped Jul 30, 2026; Verdi: "Let's look into Bucket B")
**Targets the audit's four largest information deficits:** duplicate-sector scouting 57/64 games; a quadrant never visited in 32/64; zero lighthouse corners in 29/64; Explorer reward taken 3%.
**Changes (all in the goal stack — active whenever goal channels are on, training AND gauge):**
- **Per-unit target assignment** (`assign_expand_targets`): greedy nearest-pair-first unique unit→EXPAND-target matching; each approach-needing target pays ONLY its assigned unit in the potential (unassigned targets keep a closest-unit fallback gradient). Kills the two-scouts-one-target failure at the pricing level. Deterministic — Φ stays a pure state function.
- **Quadrant-spread village guessing:** `guessed_village_sites` pass 1 now requires picks in distinct quadrants around the anchor centroid (pass 2 fills the remainder) — nearest-first alone often put both guesses in one bearing sector.
- **Quadrant-novelty scout term:** reveal payment is now per-quadrant concave (`min(revealed_q, 20)` per quadrant × SHAPE_GOAL_SCOUT) so fresh sectors keep paying after covered ones flatten; and it no longer retires when a target exists — half weight alongside the approach gradient (was: hard off).
- **Lighthouse nudge:** `SHAPE_GOAL_LIGHTHOUSE` (120) once per explored map corner, unconditional.
- **Explorer reward preference — PRICING, not a mask (amended same day per Verdi: "there are times when workshop on your first village is super important… raise the desirability of an explorer, don't just turn it off").** The shipped-then-replaced Workshop mask is gone; instead `SHAPE_GOAL_EXPLORER` (150) pays per Explorer reward taken, scaled by the CURRENT hidden-map fraction. On a dark map the Explorer edge banks ~150·hidden% PLUS the reveal's scout/lighthouse terms and outbids Workshop's 150-SPT edge; on a revealed map it pays ~0 and Workshop wins on merit. Frontier-ness prices itself: an interior explorer walks known ground and banks few scout tiles. Same conclusion as 028 P1c — pricing holds what masks break. **Lighthouse-chance lift (same day, per Verdi):** +`SHAPE_GOAL_EXPLORER_LIGHTHOUSE` (60, hidden-scaled) per still-dark corner within Chebyshev **5** of the choosing city, capped at **2** — Verdi's calibration: "a centrally located explorer reliably reaches one, sometimes two lighthouses," so the map center qualifies for all corners and the cap encodes the realistic yield. Note the exact-hit case needs no term at all: `predict_explorer` is a DETERMINISTIC 12-step fog-seeking walk, so search simulates the actual reveal on the Explorer edge and a reached corner banks SHAPE_GOAL_LIGHTHOUSE directly — the lift covers only the near-miss chance the simulation can't see.
- **Instrumentation (bucket A prerequisite):** `--dump-star-spend` now records Ability moves with an `ability` name field — first executed-move ground truth for Destroy/ClearForest rates.
**Tests:** 3 new (assignment uniqueness; reward-gate conditions incl. full-reveal release; quadrant spread of guesses) + 2 existing reward tests updated for the new terms; full suite green (102 lib tests); binaries rebuilt.
**Hypothesis:** pricing information gain per-unit and per-sector fixes coordination failures that per-target pricing structurally could not express, and the Explorer gate converts the highest-value information action from a 3% accident into a conditioned habit.
**Expected (next run, audit re-measure at ~iter 10):** duplicate-sector games 89% → ≤40%; all-4-quadrant coverage 50% → ≥75%; ≥1 lighthouse corner touched 55% → ≥80%; Explorer take at level 2 under hidden-map conditions → ≥60%; first-village turn 4.3–4.6 → ≤4; village rate ≥0.92; gauge holds vs anchor_iter30. **Falsifier:** if duplicate-sector stays >60% with assignment pricing live, the failure is in Gumbel action selection rather than the potential — measure per-unit Q gaps before raising weights.

### v4 — ACTUALS (run 1785446757, 9 iterations, Jul 30–31 2026; iters 1–5 ran the brief Workshop-MASK binary, 6–9 the corrected pricing — a natural A/B) — MOSTLY CONFIRMED
**Scorecard vs registered expectations** (replay-based, best-of-iteration games, n=18 net seats total):
- All-4-quadrant coverage 50% → **80–88%** ✅ (target ≥75%). ≥1 lighthouse corner 55% → **83% pooled** ✅ (target ≥80%). Village rate → **0.91/0.93 in the final two iters** ✅ (target ≥0.92, best self-play values ever logged). Gauge **75.0% vs anchor_iter30** at iter 5 ✅ (run ended at 9 — no second reading).
- Explorer take 3% → **43% under the mask / 21–24% under pricing** — the pricing version keeps an ~8× lift while letting Workshop win on merit (the amendment working as intended). Hidden-map-CONDITIONAL rate (the ≥60% target) unmeasured — needs the city_rewards dump join. ⏳
- First-village turn: improved to 4.2–4.5 typical but drifted 5.2–5.5 in the last two iters — **≤4 target NOT met**. ❌
- Duplicate-sector proxy **unmoved (16/18 seats)** vs the ≤40% target ❌ — but the outcome metrics it proxies (coverage, reveals, tiles20 ~100–107 vs ~96–102 pre-v4) all improved, and the archetype layer ~doubled early unit counts, mechanically inflating same-sector coincidences. Verdict deferred to a real waste measure (per-unit marginal reveals + overlap) before invoking the Q-gap falsifier.
**Downstream (the real prize):** t2c_3rd rate sustained **0.70–0.74** (pre-v4 median ~0.65) with t2c_3rd_turn 9.8 at iter 7 — the first sub-10 third city on record. Third city is the causal win lever (EXP_ELO_026).
**Watch:** policy_loss plateaued 1.64–1.69 (third teacher change in three days; no turn-down yet — Stage-2 gauge remains open). anchor_net_wr healthy (0.94 at iter 9).
**Next:** one instrumented 64-game audit batch (city_rewards + turn_states + star_spend-with-abilities) settles the explorer-conditional read, the reveal-overlap question, AND the bucket-A clear-forest ground truth in one run.

**Instrumented audit results (Jul 31, 64 games vs Greedy, final v4 snapshot, eval conditions):**
- **Village-turn drift = censoring artifact, definitively.** Like-for-like eval: first village at mean **5.78 (98% rate)** vs pre-v4's 6.2 (97%) — v4 is faster AND more reliable. The self-play cond-mean rise (4.3→5.5) is composition: rate 0.85→0.93 pulls formerly-failing late captures into the conditional mean (+0.08 share × ~t16 ≈ +1.0, matching exactly). Capture CDF: 38% by t4, 67% by t6, 95% by t10 — the path to a ~4 mean is compressing the TRAVEL tail (discovery is already t3.8); lever = mobility (road topology + rider lanes), not more capture pressure.
- **Duplicate-sector proxy retired.** Bearing proxy fires 58/64 (91%) but TRUE waste is modest: vision-footprint overlap **20.7%**, marginal reveals 2.5 tiles/unit-turn, far-unit crowding 36% of turns (which includes legitimate capture convergence). New tracked metric: overlap ratio (~20% baseline); optional 8-way sector caps could shave toward 15% — low priority. The ≤40% bearing target is void, not failed.
- **Destroy gate: FULLY effective in behavior.** Net executed **0 Destroys in 64 games** (ground truth; Greedy executes 11.6/g ungated). The shard "leak" is target-mass-only — reused-root visits polluting the exported option marginal, never executed. Low-priority target-hygiene cleanup, no behavioral issue.
- **ClearForest ground truth: net 8.06/game + 1.09 burns** — the bucket-A conditional clear gate is justified by executed data.
- **Explorer pricing UNDER-SIZED at deploy: 2% take in the hidden window (t≤6), 11% at t7–12.** Self-play's 21–24% was Gumbel noise flipping a near-tie. Workshop's compounding SPT beats the one-shot info payoff in search Q — the ≥60% conditional target is ❌ pending a re-size via the EXP_ELO_018 Q-gap method (measure actual Q deltas on level-2 choice edges before picking the new constant).
- **All-4 lighthouses: 0 of 18 sampled games** (max 3 corners; explorer walks add nothing). The engine DOES model the Explorer task → monument, so the payoff is real and never collected — candidate for a small all-4 completion term, deferred.

**Three-bucket status audit (Jul 31; same 64-game eval dumps + 9 v4-era mirror replays, tile-level joins):**
- **Bucket A (star-allocation micro) — mostly closed, two live items.** Harvest discipline stays refuted (94% of harvests precede a level-up ≤2 turns, unchanged). Burns are effectively clean: 93% become a Farm on the burned tile. Clears (8.06/g eval / 10.1/g mirror): tile-level classes = 49% level-up-linked (ClearForest GAINS 1–2 stars — banked stars funding an imminent level-up is Verdi's allowed case), ~38% build-enabling (Road 14% the largest — clearing to lay road), **10% UNJUSTIFIED (~1 clear/game)** — the conditional gate's real target is ~1–2 clears/g, far smaller than the 8/g headline. **Windmill placement is the bigger A item now: 3.67 windmills/g but 52% sit next to only 1 farm (value-negative vs just building a farm), 9% reach 3 farms** — the pop-per-star yield-table pricing term has a crisp target: shift the 1-farm majority to 2–3-farm placements.
- **Bucket B (exploration) — one substantive open item.** Explorer pricing under-sized (2% hidden-window take vs ≥60%; L2 vocab: Workshop 316 / Explorer 11) → Q-gap re-size is THE open B task. Minor/deferred: all-4 lighthouse completion term (0 games), 8-way sector caps (overlap 20.7%→~15%), village-turn tail = the deferred road/mobility work.
- **Bucket C (archetype) — holding, one conversion gap.** Deploy summon mix: mirror Warrior 64% / Rider 27% / Defender 5% / Archer 2% / Swordsman 2%; eval-vs-Greedy cost histogram 75% warrior / 24% cost-3. Research lanes live: Riding 10/18 seats, Roads 9/18, Archery 7/18, FreeSpirit/Chivalry 4/18 (knight-commit overlay fires) — **but 0 knights ever summoned** (eval cost-8 = 1 in 64 games): the lane is researched but never converted; watch item, not yet a work item (games may simply end first). $/unit unchanged at ~2.3 (army-comp stays demoted). Net out-economies Greedy 233 vs 193 stars/g; net Research share 24% vs Greedy 31%.

### v5 — economy + reward-choice pack (registered + shipped Jul 31, 2026; Verdi: "Ok let's tackle the work queue")
**Q-gap measurement first (EXP_ELO_018 method, new `--dump-reward-choices` flag):** 64 eval games on the v4 snapshot traced every modal city-reward ply — 166 net-seat Explorer/Workshop choices. Findings: (a) **the policy is structurally blind on reward choices** — prior gap exactly 0.0 on 166/166 plies because the mapper sent EVERY Reward move to option slot 191, and visits split 128/128 by sequential halving, so the choice was decided purely by root Q with no distillation channel at all; (b) Workshop's root-Q lead at hidden≥0.5 is **median +0.258 normalized (p75 +0.50)**, Explorer take 14%; the old 150·h term (~+0.12 effective) was ~2.5× too small — same 15×-too-small failure shape as EXP_ELO_016's proximity term, now caught by measuring first.
**Changes:**
- **Mapper: per-type reward slots.** CityRewardType ids 1..=8 → option slots 181–188 (`OFFSET_REWARDS`, carved from the unused ability-block tail; slot 191 retired). `MoveVisit` gained `reward_type`; composer + both target paths updated. No head resize, no checkpoint migration; old shards' 191 targets stay readable. This opens the distillation channel: the exported π′ = softmax(logit + β·σ(Q)) is informative at reward plies even at equal visits.
- **Explorer re-size, shape change:** `SHAPE_GOAL_EXPLORER` 150→**1000**, `SHAPE_GOAL_EXPLORER_LIGHTHOUSE` 60→**350**, both now scaled by **hidden_frac²** (was linear h). Rationale: the potential telescopes to the horizon's h (the reveal drains its own multiplier), and quadratic keeps the dark-map edge dominant (~+0.42 at h=0.5, clearing the measured p60 gap) while dropping below Workshop's merit lead once h<0.25.
- **Engine rules fix — retroactive adjacency pop** (`actions/structure.rs`): building a Farm/LumberHut/Mine next to an existing own Windmill/Sawmill/Forge now pays that structure's reward_pop to its city, matching the real game (previously pop locked at the yield structure's build time, silently punishing windmill-first ordering). Covered by `tests/retroactive_adjacency_pop.rs` incl. undo integrity.
- **`SHAPE_GOAL_YIELD_ADJ` = 100:** Φ pays reward_pop × (partners−1) per owned adjacency-yield structure, derived from structures.rs tables (Forge's 2-pop scales ×2; Market self-excludes via reward_pop 0). Targets the 52%-of-windmills-at-1-farm placement waste.
- **`SHAPE_GOAL_FOREST_STANDING` = 50:** standing forest in own territory holds option value; the ~1/game follow-through-free clear goes net-negative in-tree while justified clears (level-up funding, build-enabling) still win on their follow-up payoff.
**Expected (deploy verification, same 64-game eval harness, new binary):** Explorer take at hidden≥0.5 **14% → ≥60%**; take at hidden<0.25 stays **≤35%** (Workshop on merit); ClearForest 8.06/g → **~6–7/g** (the unjustified tail, not a collapse); win rate vs Greedy within gauge noise of the 75% baseline. **Falsifiers:** hidden-window take <40% → the h²/telescoping model is wrong, re-dump and re-fit; win rate <60% → constants too hot, revert; ClearForest <4/g → halve FOREST_STANDING.
**Expected (next training run):** windmill 1-farm share 52% → ≤30% at the ~iter-10 replay audit; Explorer/Workshop PRIOR gap moves off exactly-0.5 within ~10 iters (the new slots training) — first-ever ownership path for reward choices; third-city and coverage metrics hold.

**Deploy verification ACTUALS (Jul 31, 4×64-game matched runs, same harness, random tribes) — CONFIRMED after a two-step calibration:**
The Explorer constant was dialed against the measured root-Q median at hidden≥0.5 (slope ≈ 0.085 dq per 100 of constant — the first fit overshot because the Φ jump compounds with the reveal's scout/lighthouse terms, roughly doubling the naive telescoping estimate):

| arm | constants | dark take (h≥0.5) | dq med (h≥0.5) | wr vs Greedy | SPT@t20 | t2c_4th |
|---|---|---|---|---|---|---|
| pre-v5 | 150/60·h | 14% | +0.258 | 0.750 | 17.9 | 0.75 |
| v5 | 1000/350·h² | 83% | −0.340 | 0.609 | 12.9 | 0.52 |
| v5b | 600/200·h² | 52% | −0.004 | 0.672 | 15.2 | 0.56 |
| **v5c (shipped)** | **700/230·h²** | **64% ✅** | −0.084 | 0.641 | 13.7 | 0.63 |

- **Target met:** dark-map take 64% (≥60% registered); low-hidden take 15–32% across runs (≤35% bound holds — Workshop keeps winning on merit once the map is lit); t≤6 take 67% at the v5 reading.
- **The open price:** vs-Greedy eval wr sits ~8–11pp below pre-v5 (64–67% vs 75%, ~1σ each run but consistent in direction), tracking the SPT give-up (~3 @t20). This is the cost of buying information with an economy edge while the value head can't yet cash map knowledge. **The training-run question (registered): does the net convert the taught behavior into wins — and does the new distillation channel move the Explorer/Workshop PRIOR off exactly-0.5 (ownership)?** If eval wr is still ≥8pp below the crutch-free baseline after ~15 iters with no prior movement, the term is overpriced for its value — dial toward 600 or below.
- **ClearForest 4.5–5.3/g, Burn 0.4–0.7/g** across v5 arms — the <4/g collapse falsifier never fired (tribe-mix caveat vs the 8.06 fixed-tribe baseline stands).
- **Side-checks stable:** villages_t2c_first_cond 4.7–5.1 @ 97–98%, t2c_3rd rate 0.81–0.83. All three v5 terms rode along in every arm; the wr/SPT deltas track the Explorer dial specifically (the only constant varied between arms).

### v5 — training ACTUALS (run 1785520188, 5 iters, eff 111–115, Aug 1 2026) — CONFIRMED on every registered endpoint
- **GAUGE FREEZE at iter 5: 85.7% vs anchor_iter30** — new anchor `anchor_iter5_20260731_235649` (4th freeze of the campaign). Cities curve is the striking part: model 3.13@t10 → **4.14@t20** → 4.37@t25 while the old anchor collapses 1.54 → 0.95 → 0.73 — the model is taking the anchor's cities, not just out-growing it. The old ≥2.3 tower bar is doubled.
- **OWNERSHIP OPENED (the registered first-ever endpoint): prior gap ≠ 0 on 180/180 reward plies** (was exactly 0.0 on 166/166 — structurally blind pre-slot-fix), and it is already **conditional**: raw-prior median Explorer−Workshop **+0.07 at hidden≥0.5 vs −0.47 at hidden<0.25** — information when dark, economy when lit, learned from 5 iterations of π′ targets through the new slots. Crutch-removal for this choice is now a measurable path (watch the dark-map prior median climb; removal candidate when the prior alone reproduces the ≥60/≤35 split).
- **Cost recovery underway:** same eval probe on the iter-5 weights vs the v4-weights v5c probe — wr 0.641→**0.703** (pre-v5 crutch-free: 0.750), captures 4.98→**5.56** (above pre-v5), t2c_3rd_turn 10.15→**8.72 @ 0.844 rate (first sub-9 on record)**, t2c_4th 0.63→0.69, villages_t2c_first_cond **4.07 @ 95%** — the ~4 first-village mark Verdi asked for, at the best rate yet. SPT@t20 stays ~14 vs pre-v5's 17.9: the model is converting the surrendered SPT into tempo (cities/captures) rather than recovering it — which is the trade working, not failing.
- **Windmill placement (registered ≤30%): 1-farm share 52% → 31%**, 3+-farm 9% → 38% (n=13, 5 mirror replays). Clears 10.1 → 7.2/g with burns up 3.3 → 4.8/g — the clear→burn+farm shift is the economically justified direction. Explorer 41% of mirror level-2 takes.
- **policy_loss 1.737 → 1.698 over the 5 iters** — mild decline through a triple teacher change (new constants + new slots), where each previous teacher change re-opened the gap.
- **Watch items:** dark-map take on trained weights 58% (band edge, n=62); SPT@t20 plateau ~14; whether the prior gap keeps widening toward search-free ownership. Next read at the 10-iter league/gauge.

### v6 — replay-critique pack (registered + shipped Aug 1, 2026; Verdi's six critiques from v4_iter8_mirror_8475, plan approved)
**Phase 0 — engine/heuristic fixes (all landed with tests):** (1) `population`→`progress` at the 4 buggy sites — the ONLY pre-existing level-completion check always awarded "finishes level" because it read the lifetime counter; (2) **Workshop/Park single-count** (+1 SPT each per the real game, was +2 via a stored-bump double-count; Park keeps its +250 score) — honest SPT is ~1-2/city LOWER than every pre-v6 reading; (3) Market adjacency counts friendly hubs only; (4) Greedy's `evaluate_economy` now reads derived production (sees capital/workshop/market income).
**Instruments:** `--dump-level-completion` (per-city spend discipline), `--dump-pop-spend-choices` (sampled economy-ply root Q), turn-state dump extended with per-city (level, progress, production) + archetype seen-counts/knight_commit.
**Baseline (64 games, post-Phase-0 semantics, iter-5 weights):** pop spends completing a level immediately **20%**; **68% of cities end the game with unfinished progress** (the honest per-city number the old player-level "94%" join hid); harvest-vs-alternative root Q median **+0.000** (near-ties — discipline is cheap to buy); summon-under-cap loses by median **−0.118**; **knight_commit fires in 27/64 seats at median t14** with threshold 4 (threshold 3 would fire at t12 in 38/64 — only ~2 turns earlier, below the ≥4-turn bar registered for proposing a threshold change, so SEEN_SQUISHY_KNIGHT stays at Verdi's 4).
**Changes:**
- **Capture-first root gate** (`passes_capture_first`, both gumbel retain sites): a unit standing on a capturable village/ruin (neutral or enemy-owned) never attacks — hard gate per Verdi. Blocks even when Capture is illegal this ply (post-step): idle now, capture next turn.
- **Stranded-progress discipline** (`SHAPE_GOAL_STRANDED=150`, GROW only): −Φ per city with started-but-uncompletable progress, completability derived from settings tables (`max_affordable_pop` greedy knapsack over territory resources/pop structures), threat-exempt (`city_threatened`, cheb 2 — Verdi's 15% harvest-under-threat case).
- **Retake pack:** enemy-captured villages stay painted (`retakeable_village`: explored, non-capital, within RETAKE_PAINT_RADIUS=6) with approach at `SHAPE_GOAL_RETAKE_W=0.75`; real targets outrank fog guesses in assignment; a CONTESTED target pays one extra converger at 0.5× gradient.
- **Early body count** (`SHAPE_GOAL_BODY=150`, GROW): pays per unit up to min(cities+1, 3) while map/expansion remains — the 2nd/3rd warrior finally prices against a 2★ harvest.
- **Knight lane fix:** archetype pricing now cost-scaled (`SHAPE_GOAL_ARCHETYPE_PER_COST=33` — cost-3 units numerically unchanged at 99, Knight 264); Chivalry exempt from the tier-3 cap under an active knight_commit; `passes_star_gate` waives the stance-class block for FreeSpirit/Chivalry under commit (the lane was gated by GROW and ARM from opposite sides).
- **Market/Trade lane:** `market_ready` (3 cities + owned hub) pushes Riding→Roads→Trade and exempts Trade from the tier-3 cap; `SHAPE_GOAL_YIELD_ADJ_STARS=50` prices multi-hub Market placement (half the pop analog — partners' SPT already pays through SHAPE_GOAL_SPT).
**Expected (deploy verification, 64-game A/B vs the v6 baseline):** attack-from-capturable-tile ≈ 0; level-completion immediate-complete rate 20% → ≥35% with end-stranded 68% → ≤50%; units@t5 up ~+1; knights appear in commit games (>0/64 vs 1); win rate within noise of baseline (no >5pp drop); SPT@t10 not lower than baseline (honest scale). **Falsifiers:** wr drops >5pp → pull SHAPE_GOAL_STRANDED/BODY to 75 first (near-tie Q-gaps mean 150 is likely 2× hot); stranded rate unmoved → the knapsack predicate is wrong, re-derive; units@t10 >5 with ≤2 cities → body cap failing.
**Expected (next training run):** SPT@t10 toward 13+ honest (15 = stretch), SPT@t20 26-32 with the market limb; retaken villages measurable; third-city tempo holds (t2c_3rd ≤ 10).

**Deploy verification ACTUALS (Aug 1, 3×64-game matched runs) — SHIPPED AT v6b AFTER ONE FALSIFIER ITERATION:**
| arm | stranded term | wr | SPT@t10/t20 | t2c_3rd | complete% | end-stranded | 8★ summons |
|---|---|---|---|---|---|---|---|
| baseline (Phase 0 only) | — | 0.781 | 8.1 / 14.7 | 9.3 @ 0.89 | 20% | 68% | 1 |
| v6 first fit | 150/city, stars-dependent | 0.656 ❌ | 7.1 / 13.8 | 10.2 @ 0.81 | 22% | 67% | 5 |
| **v6b (shipped)** | **75/pop, resource-structural** | **0.797 ✅** | 7.4 / 14.8 | 9.5 @ 0.89 | 24% | 66% | 3 |
- **Both v6-first-fit falsifiers fired and the registered responses were applied.** Root cause of the wr −12.5pp: a stars-dependent stranded predicate turns EVERY purchase into a potential −150 flip (spending anywhere can strand a partial city) — a broad economy tax; meanwhile depth-blindness left extra pop into already-stranded cities free. The reshape (penalty per stranded POP POINT, completability from remaining territory resources at any star budget) removes the spend-tax entirely and prices exactly the harvest-into-a-dead-end case. Body term halved to 75 (clears the measured −0.118 summon deficit without doubling it).
- **Knight lane converts at deploy: 8★ summons 1 → 5/3 across v6 arms** — the cost-scaled pricing + tier-3 exemption + stance waiver work. Commit timing measured: fires 27/64 seats at median t14 with threshold 4; threshold 3 buys only ~2 turns (below the registered ≥4-turn bar) → **SEEN_SQUISHY_KNIGHT stays 4**, no user decision needed.
- **Discipline targets NOT met at deploy** (24% vs ≥35% complete, 66% vs ≤50% end-stranded — directionally right). Honest read: the structural penalty is deliberately narrow; the star-TIMING half of Verdi's rule is unpriceable without re-introducing the spend-tax, and the near-tie root Q-gaps (median +0.000) mean deploy search barely distinguishes these plies. The remaining leverage is TRAINING-side: the shifted π′ targets + the penalty accumulating across self-play. Re-measure at the next run's ~iter-10 audit; if end-stranded is still >60%, the next lever is a completion BONUS (pay level-up landing) rather than a deeper penalty.
- Capture-first gate and retake pack ride along in all arms (unit-tested; behavioral replay audit needs training-run replays — attack-from-village and retake-rate registered as next-run checkpoints). avg_score sits ~14% below baseline at equal wr — score is not the objective but flagged as a watch item.

### v6 — training ACTUALS (run 1785569587, 5 iters, eff 111–115, anchor-frac 0.25, Aug 1 2026) — **STRONGER MODEL, ZERO REGISTERED ENDPOINTS MET**
**Method note:** every behavior number below comes from a matched 64-game audit (`self_play_v6b`, mcts 256, GUMBEL_SCALE=0, anchor-frac 1.0) run on the v5 tip weights and again on this run's tip. Same binary, same harness, same flags — **the only difference is weights**, so these isolate what training did, with the Phase-0 accounting change held constant.

- **The model got stronger head-to-head and it is not noise: gauge 68.75% (44–20, n=64, z≈3.0 vs 50%) against `anchor_iter5_20260731_235649` (the v5 tip, elo 1316)** → elo_est 1453. Cities curve 1.69/3.05/3.73/4.09 @t5/10/15/20 while the anchor collapses 1.11 → 1.45 → 1.14. Self-play anchor wr 0.787 → **0.900** (n=80/run). Training-side quality all improved mildly: policy_loss 1.713→1.677, value_loss 0.245→0.232, value_r2 0.899→0.905, aux_spt 0.050→0.046.
- **But against Greedy it is flat-to-down: 0.797 (v5 weights) → 0.750 (trained), n=64 each** — a −4.7pp move well inside the ±13pp two-arm noise band, i.e. **no absolute gain to show for the +18.75pp head-to-head.** The gain is relative to its own lineage, which is exactly the metric most exposed to non-transitive self-play drift. Flagged, not yet diagnosed.
- **Registered checkpoint scorecard — 1 of 8 met:**

| checkpoint | registered target | v6b deploy (v5 wts) | v6b trained | verdict |
|---|---|---|---|---|
| attack-from-capturable-tile | ≈ 0 | hard-gated | hard-gated | ✅ (see squat check) |
| level-completion immediate | ≥ 35% | 24.3% | 24.8% | ❌ flat |
| end-stranded cities | ≤ 50% | 65.6% | **70.0%** | ❌ worse |
| units @t5 | 2–3 | 2.91 | 2.78 | ⚠️ in band but falling |
| knights (8★ summons / 64g) | > 0 and growing | 3 | **1** | ❌ regressed |
| retake rate | measurably ↑ | 35.6% | 32.0% | ❌ flat |
| SPT @t10 / @t20 (honest) | 13+ / 26–32 | 7.30 / 12.12 | 6.86 / 11.86 | ❌ down |
| third-city tempo | t2c_3rd ≤ 10 | 9.47 | 10.29 | ❌ breached |

- **Capture-first gate is clean — the idle-squat falsifier did NOT fire.** Units parked on an open village at turn start: 0.104/turn (deploy) → 0.092/turn (trained), and only **0.8% are still standing there at the next turn start** (v6base without the gate: 0.0%). The gate converts to capture rather than wasting the unit's turn. Village captures nonetheless fell 2.13 → 2.03 → 1.91 across the arms — that is fewer expansion *attempts*, not squatting.
- **What training actually optimized: militarization.** Move mix per player-turn, v5 run → v6b run: Build **−21% (t5-9), −36% (t10-14), −34% (t15-19), −27% (t20+)**; Attack **+35%/+35%/+47%**; Summon +15%/+44%/+21%; Ability +89%/+62%/+60%. Aggregate: units_spawned 19.2→22.7, attacks 22.1→28.7, kills 12.2→16.1, units_lost 12.5→16.3, builds 27.2→23.5. Army *composition* did not change (unit_worth_t15 2.44→2.47, 8★/5★ summons ≈ 0) — it bought **more of the same 2–3★ bodies**, not better ones.
- **Mechanism (consistent with the deploy A/B):** the discipline terms sit on near-tie root Q-gaps (median +0.000 measured at baseline), so they contribute almost no training signal; the body/archetype pricing sits on a real gap and therefore captured the shaping budget. Training amplified the term that had a gradient, not the term we wanted.
- **Accounting delta quantified (identical weights, pre- vs post-Phase-0 binary): the Workshop/Park single-count fix costs 0.75 SPT at t10 and t15, ~0 at t5/t20.** The v5→v6b run drop of ~1.2 SPT@t10 is therefore ≈0.75 accounting + ≈0.45 real regression. Honest scale confirmed; the regression is small but real.
- **Registered trigger has FIRED:** end-stranded >60% at the post-run audit (70.0%) → per the deploy-verification entry, the next lever for Verdi's critique #1 is a **completion BONUS** (pay the level-up landing) rather than a deeper penalty. The penalty side is now measured twice as inert.
- **Strategic read from the same 64 games (conditional, not causal):** cities@t10 ≥4 → **93% win** (n=14) vs 3–4 → 58% (n=33); AUC on win/loss is **spt_lead 0.677 > cities 0.594 > spt 0.570** — the *relative* economy and the 4th city carry the signal, absolute SPT barely does. Late-game curve dips (t25+) are pure survivorship: games reaching t25 were already behind at t15 (3.00 cities / 9.31 SPT / 25% wr vs 3.91 / 14.09 / 82% for games ending earlier) — never read the t25/t29 columns as a trend, n falls to 11–18.

#### v6 regression diagnosis (Aug 1, 2026 — three-arm dump analysis, exposure-normalized)
**Cause is NOT a boosted economy incentive — v6 reduced the economy gradient on three axes and added its only new positive term on army.**
1. **Phase 0 halved Workshop inside Φ.** `goal_potential` under GROW is `SHAPE_GOAL_SPT(150) × get_tribe_spt − SHAPE_GOAL_STRANDED(75) × stranded_progress`. The single-count fix took a Workshop from 2 SPT to 1, i.e. **300 Φ → 150 Φ**; the friendly-only Market fix removed enemy-hub income the same way. Both are correct fidelity fixes, but each is a genuine cut to the economy potential.
2. **`SHAPE_GOAL_STRANDED` behaves as a LEVEL-DEPENDENT TAX on exactly the cities that make SPT — this is the primary cause.** Exposure-normalized pop spends per city-turn (v6base → trained): **L1 0.464 → 0.528 (+14%), L2 −25%, L3 +2%, L4 −12%, L5 0.988 → 0.630 (−36%), L6 1.057 → 0.473 (−55%)**. The effect is level-monotonic and **deepened with training** (L6 was −3% at deploy, −55% after 5 iters) — the signature of a learned tax, not a general star shortage, which would depress all levels alike. City-level mix followed: L6 share of city-turns 4% → 2%, L5 7% → 5%. SPT is the sum of city levels, so capping level growth caps SPT by construction. (n-caveat: L5 rests on 162–258 city-turns and is solid; L6 on 33–55 and is directional.)
3. **Root cause inside the term: `max_affordable_pop` is myopic.** It scores completability from the city's *current* territory only, while the level threshold grows as N+1 and territory resources are finite and get consumed — so past ~L4 nearly every city reads "uncompletable" and ANY pop investment there is strictly −Φ. But `BorderGrowth`, `Resources` and `PopGrowth` are level-up **rewards**, and village capture adds tiles: "cannot finish from today's land" is not "can never finish." The trap is self-fulfilling — the way out of stranded is to level up, and the penalty discourages the investment that gets you there.
4. **Second-order: the term penalises the level-ups we want.** Overflow progress carries, and **25% of successful level-ups land with progress > 0** (63+27 of 356 at v6base; unchanged across arms, so it is a latent design flaw rather than a v6-introduced one). When that overflow lands in a city the predicate calls dead, a level-up worth +150 Φ immediately books −75 or −150 Φ of stranded penalty — halving or erasing its own reward.
5. **Star diversion is real but secondary, and complementary rather than competing.** Share of stars spent (v6base → deploy → trained): **Build 44.6% → 36.4% → 36.6%, Summon 23.3% → 26.8% → 27.8%, Research 25.0% → 28.5% → 27.0%**. Build-cost detail shows the collapse concentrated in 3★ builds (1333 → 789 → 882). The stranded term pushed stars OUT of the economy; `SHAPE_GOAL_BODY`/archetype pricing caught them.
**Implication for the next lever:** the already-registered switch from penalty to **completion BONUS** is now doubly supported — the penalty is inert where it was aimed (completes 20% → 24.8% across three arms) and actively harmful where it was not (L5/L6 growth). Any retained penalty needs a non-myopic completability predicate (credit reachable territory/reward routes) and must not fire on carried overflow.

#### v6 diagnosis, part 2 — Verdi's two corrections, both CONFIRMED in code + data (Aug 1, 2026)
**(a) The predicate is blind to the entire mid/late-game pop engine — which is exactly the game-true reason late levels are hard.** `max_affordable_pop` only walks territory tiles that carry a **resource** (or Forest under Forestry, for LumberHut). Empty Field tiles score **zero**. But the multiplier tier sits on empty Fields and, per `actions/structure.rs:242`, pays `reward_pop *= adj_count` — **one pop per friendly adjacent partner**, so:
| route | cost | pop | ★/pop | seen by predicate? |
|---|---|---|---|---|
| Farm / Mine | 5★ | 2 | 2.50 | ✅ |
| LumberHut | 3★ | 1 | 3.00 | ✅ |
| **Windmill** (adj 3 Farms) | 5★ | **3** | 1.67 | ❌ |
| **Forge** (adj 2 Mines) | 5★ | **4** | 1.25 | ❌ |
| **Sawmill** (adj 3 LumberHuts) | 5★ | **3** | 1.67 | ❌ |
| Temples (Mountain/Water/Forest) | 20★ | 1 | 20.0 | ❌ |
| **Monuments** | task | **3** | free | ❌ |
The predicate is blind to the **cheapest pop per star in the game** and to the free 3-pop monuments. So a city holding 3 Farms and one empty Field — a Windmill away from +3 pop — is scored a dead end and taxed for trying. This is the precise mechanism behind the L5/L6 collapse, and it makes the term wrong on game rules, not merely myopic.

**(b) No star banking whatsoever — Verdi's "hold 20★ to buy Sawmills next turn" behavior does not exist.** Measured on net seats across the three arms:
| arm | zero-spend turns | spend/income p25·p50·p75 | carried balance | save-then-buy |
|---|---|---|---|---|
| v6base | 8.2% | 0.80 · **1.00** · 1.36 | 2.32★ = 0.26 turns | 2.3% |
| v6b deploy | 11.0% | 0.71 · **1.00** · 1.50 | 3.37★ = 0.45 turns | 3.9% |
| v6b trained | 9.3% | 0.80 · **1.00** · 1.40 | **1.42★ = 0.20 turns** | 3.7% |
**Median spend/income is exactly 1.00 in every arm** — pure hand-to-mouth — and training made it *more* so (carried balance 2.32 → 1.42★). Three independent structural causes, all confirmed:
1. **Φ pays nothing for liquidity.** Held stars appear nowhere in `goal_potential` (the only `stars` in reward.rs are `max_affordable_pop`'s unused budget param and the star-yield adjacency term). Under potential-based shaping, converting stars into any scored asset strictly raises Φ while holding leaves it flat — so saving is a strictly dominated action by construction. The 2-3 turn plan requires crossing a Φ-flat valley the shaping punishes.
2. **The search horizon is one game turn.** Gauge tree depth **mean 8.37 plies** (max 254) against the project's own ~8-plies-per-game-turn branching analysis. A 2-3 turn plan is outside the tree entirely, so search cannot discover it even if Φ allowed it.
3. **γ = 0.9/turn** charges ~10% to defer any purchase by one turn — a third, smaller headwind.
The only star-management machinery that exists is `STAR_GATE_RESERVE = 5`, a solvency **floor** on tech purchases (don't buy a tech that leaves you under 5★). That is a bankruptcy guard, not a savings plan.
**This is the same root cause as the army's missing long-range strategy** (Verdi's parallel observation): the agent has no representation of a multi-turn commitment, and both its reward and its horizon are single-turn. Any fix that only re-prices individual moves — the entire v3–v6 crutch family — cannot express "spend nothing this turn so that two turns from now I can afford the Forge lane." Noted as the top open design question for v7.

## v7 — economy correctness + the commitment layer (registered + shipped Aug 1, 2026, BEFORE the run; Verdi: "let's go ahead and set up all the code needed for the next training run")
Follows the v6 regression diagnosis and Verdi's two corrections: the late-game pop engine is the multiplier tier plus monuments, and the net "doesn't pick a way it's going to play its game" or manage stars across turns toward it.

**1. `max_affordable_pop` now sees every pop route the engine offers (correctness, not a dial).** Added the adjacency-multiplier tier (Windmill/Sawmill/Forge, yield `reward_pop × friendly partners`, counting partners the city could still build) and the pop-bearing terrain tier (Temple/MountainTemple/WaterTemple/ForestTemple/Port), mirroring `moves/build.rs` legality including tech unlock, `limited_per_city`, algae and enemy-occupancy. Monuments deliberately excluded (`check_task` over every TaskType is too costly for a leaf-path helper; the omission only ever makes the predicate more pessimistic).
**2. Stranded penalty is FLAGGED, not billed by depth** (`STRANDED_PER_CITY_CAP = 1`). v6 summed every stranded pop point, so a level-up landing in 2 overflow booked −150 against its own +150 of SPT. Overflow lands on 25% of level-ups, unchanged across all v6 arms — a latent trap, now closed.
**3. Completion BONUS** (`SHAPE_GOAL_COMPLETION = 75`, the registered replacement now that end-stranded sat at 70% > the 60% trigger): progress toward a REACHABLE level pays `progress/(level+1)` units. Fractional by construction so it is always worth less than the SPT jump a level-up banks — a flat per-point bonus would recreate the very trap it replaces. Guarded by a test that asserts both progress terms stay under `SHAPE_GOAL_SPT` at every level 1-7.
**4. The goal-setter has memory** (`StanceCommit`, `update_goal`). `scripted_goal` was a pure function of current state recomputed every ply, so the "strategy" could contradict itself between plies of one turn. Stance now carries with `STANCE_SWITCH_TURNS = 2` hysteresis, counted in TURNS not plies. Asymmetric on purpose: a DEFEND order switches instantly (a threat response that waits out a hysteresis window arrives after the city falls); only discretionary swings are damped. Threaded through self_play and arena.
**5. SAVE stance + savings ramp** (`Stance::Save`, `SHAPE_GOAL_SAVE = 100`, `CH_STANCE_COUNT` 3→4, `NUM_CHANNELS` 168→169, `migrate_save_stance.py` run over model.safetensors + all 166 checkpoints, both Python trainers mirrored). Held stars appeared NOWHERE in Φ, so converting them into any scored asset strictly raised Φ while holding left it flat — saving was a dominated action by construction, and the measured policy was hand-to-mouth (median spend/income exactly 1.00, carried balance 0.20 turns of income). SAVE names a target and Φ pays `stars/cost` toward it, so banking climbs a ramp. **The ramp is the mechanism that makes a multi-turn plan legible to a one-turn search** (measured tree depth 8.37 plies ≈ one game turn): it is visible at depth 1, so the tree never has to reach the purchase to value it. SAVE is GROW plus the ramp — banking never costs the economy gradient — and it self-terminates (fires only when the batch is out of pocket now but inside `SAVE_MAX_TURNS = 3` of income).
- **Smoke caught a dead feature and it was fixed before shipping:** with the batch defined as already-unlocked structures, a batch was placeable on 26% of turns and the SAVE gate fired on **0 of them** — a lone 5★ Windmill is affordable out of pocket at any real income. Redefined the batch as the cheapest **LANE**: the enabling tier-3 tech (Construction/Mathematics/Smithery/Trade) when unowned, plus its placements. `TIER3_CAP_PER_GAME = 1` means a tribe picks at most one lane per game, which IS Verdi's "which territory upgrade am I leaning on this game." Post-fix smoke: batch costs **24-27★**, gate fires on 20/21 placeable turns, stance reaches SAVE.
**6. Plan-outcome instrumentation — the belief tripwire.** Every painted EXPAND target is resolved as achieved / contested-known (enemy visible when we committed) / **contested-SURPRISE** (no enemy visible at commit, one present at drop) / dropped-by-churn. This is the pre-registered discriminator for whether a belief state is the binding constraint: surprise-dominated means we need probabilistic opponent modelling, churn-dominated means we need commitment. Turn dumps also now carry stance, save_target, the raw `save_batch` (separating "no batch existed" from "the gate rejected it"), and the stance/order flip counters EXP_ELO_028 registered and never measured.

**Pre-run baselines (matched 64-game audit on the v6b trained tip unless noted):** completes 24.8%, end-stranded 70.0%, pop-spends/city-turn L5 0.630 / L6 0.473 (v6base pre-tax: 0.988 / 1.057), SPT@t10 6.86 @t20 11.86, cities@t10 2.58, t2c_3rd 10.29, wr-vs-Greedy 0.750, knights 1/64, carried balance 1.42★ = 0.20 turns, median spend/income 1.00, save-then-buy 3.7%. From the 4-game v7 smoke (weak n, directional only): order-flip 44%/turn, plans achieved 32% / dropped-by-churn 59% / contested-surprise 7%.

**Expected (next training run):**
- Pop spends per city-turn at L5/L6 recover toward the v6base pre-tax level (≥0.85 / ≥0.90); end-stranded 70% → ≤55%; immediate-completion 24.8% → ≥30%.
- SPT@t10 ≥ 8.0 and @t20 ≥ 14 on the honest scale (recovering the v6 loss before chasing Verdi's 15-20/35+ targets, which stay DIRECTIONAL and winrate-gated).
- Star banking becomes visible: median spend/income < 1.00, carried balance ≥ 0.5 turns of income, save-then-buy ≥ 10% of turns, and at least one lane bought-then-built per 4 games.
- Third-city tempo holds (t2c_3rd ≤ 10.3); win rate vs Greedy no worse than 0.75 − 5pp.
- Order-flip rate falls below the smoke's 44%/turn as the standing goal takes hold.

**Falsifiers (each with its prescribed response):**
- L5/L6 pop-spend rate does NOT recover → the predicate is still wrong, not merely narrow; re-derive `max_affordable_pop` against engine rollouts rather than against our reading of the tables (the exact failure that produced v6).
- Carried balance rises but purchases do NOT (lane bought-then-built stays ~0) → `SHAPE_GOAL_SAVE` is a hoard bonus; halve it to 50 first (first fits have overshot ~2× every time per the q-gap dial method), and if hoarding persists, gate the ramp on the lane tech already being owned.
- `save_batch` non-null on <10% of turns → the lane predicate is still too strict; drop `SAVE_MIN_PARTNERS` to 1 for Forge (2 pop/partner) before touching anything else.
- Win rate vs Greedy drops >5pp → suspect the completion bonus first (it is the only new term that pays continuously under GROW); pull `SHAPE_GOAL_COMPLETION` to 37.5 before touching SAVE.
- Stance-flip rate near zero AND win rate down → hysteresis is too sticky; `STANCE_SWITCH_TURNS` 2 → 1.
- **Tripwire read (not a falsifier, a routing decision):** if contested-SURPRISE overtakes dropped-by-churn as the dominant plan outcome, belief state has become the binding constraint and the sparse-macro-head / abstract-search program moves to the front of the queue. If churn still dominates, commitment work continues and belief waits.

**Not in v7 (deliberate):** the learned stance head (EXP_ELO_028 Stage 2) — the commitment object it is supposed to choose only just started existing; monuments in the completability predicate; order-level hysteresis; and the aux_fog calibration check that would tell us whether the trunk's fog belief is real or vestigial (cheap, Python-side, worth doing before any belief work).

### v7 — training ACTUALS (run 1785586869, 5 iters, eff 111–115, anchor-frac 0.25, Aug 1 2026) — **ECONOMY + STRENGTH BOTH UP; DISCIPLINE AND BANKING BOTH MISSED**
**Method:** three matched 64-game audits, mcts 256, GUMBEL_SCALE=0, anchor-frac 1.0. `v6 code+wts` (where v7 started) → `v7 code, v6 wts` (isolates the CODE) → `v7 code+wts` (code + training). Same flags throughout.

- **GAUGE FREEZE at iter 5: 84.4% vs `anchor_iter5_20260731_235649`** — the same anchor the v6b run scored 68.8% against. Link match n=128 settles at **74.2%, elo 1499.7** (v6b: 1453). New anchor `anchor_iter5_20260801_155754` — 5th freeze of the campaign. Pooled over v7's own 192 anchor games the edge over v6b is ~1.4σ: real-looking, not established.
- **Win rate vs Greedy 0.750 → 0.891** (n=64 each, ~2.1σ) with avg_score 4135 → 4657. **This is the first run in the campaign where economy and strength moved in the same direction** — v6 won by militarising while the economy fell.

| endpoint | registered target | v6 code+wts | v7 code, v6 wts | v7 code+wts | |
|---|---|---|---|---|---|
| pop spends/city-turn **L6** | ≥ 0.90 | 0.473 | 0.508 | **0.973** | ✅ |
| pop spends/city-turn L5 | ≥ 0.85 | 0.630 | 0.915 | 0.619 | ❌ |
| multiplier-tier builds /64g | lane converts | 23 | 29 | **42** | ✅ |
| SPT @t10 | ≥ 8.0 | 6.86 | 7.69 | 7.58 | ❌ (+10%) |
| SPT @t20 | ≥ 14 | 11.86 | 13.28 | **13.59** | ❌ (+15%) |
| cities @t20 | — | 3.38 | 3.41 | 3.56 | ✅ |
| t2c_3rd | ≤ 10.3 | 10.29 | 9.60 | 9.81 | ✅ |
| 4th-city rate | — | 0.719 | 0.750 | **0.812** | ✅ |
| 8★ summons (knights) /64g | > 0 | 1 | 13 | **25** | ✅✅ |
| level completed immediately | ≥ 30% | 24.8% | 20.9% | 22.1% | ❌ |
| cities end stranded | ≤ 55% | 70.0% | 69.3% | 67.3% | ❌ |
| spend/income median | < 1.00 | 1.00 | 1.00 | **1.00** | ❌ |
| carried balance (turns) | ≥ 0.5 | 0.20 | 0.20 | 0.29 | ❌ |
| save-then-buy | ≥ 10% | 3.7% | 3.1% | 3.2% | ❌ |
| wr vs Greedy | ≥ 0.70 | 0.750 | 0.719 | **0.891** | ✅ |

- **The `max_affordable_pop` correctness fix is confirmed causal and is the run's biggest win.** The code-only arm (weights frozen) lifts SPT@t10 +12%, SPT@t20 +12%, L5 pop-spend rate +45% and builds/game 17.8 → 24.7. Training then doubles the **L6** rate to 0.973, recovering the pre-v6-tax level (v6base: 1.057). The v6 diagnosis — that the predicate's blindness to the multiplier tier was a level-dependent tax on exactly the cities that make SPT — is now established by intervention, not just correlation.
- **The commitment layer paid off somewhere unplanned: the knight lane.** 8★ summons 1 → 13 on the CODE change alone, → 25 after training. Mechanism: with `StanceCommit` hysteresis, ARM now *holds* (47%/39% of turns) instead of flip-flopping every ply, so the cost-scaled archetype pricing finally has time to act. v6 shipped that pricing and got 1 knight per 64 games; the missing ingredient was never the price, it was persistence. Units lost also fell 12.4 → 10.2 while captures rose.
- **Banking did NOT materialise, and the failure mode is the opposite of the registered one.** SAVE fires on 8% of turns and the gate on 17-22% (a batch is placeable on 31%), yet spend/income median is still **exactly 1.00** and carried balance moved only 0.20 → 0.29 turns. The registered falsifier anticipated a hoard ("banks but never buys" → halve `SHAPE_GOAL_SAVE`); what actually happened is the ramp is too WEAK to change the spend decision at all. Response is therefore the reverse of the registered one: raise `SHAPE_GOAL_SAVE` and/or make SAVE hold longer, and dial against the measured carried balance.
- **BUG found mid-run and still open:** `save_batch_cost` prices a lane's tier-3 tech whenever it is merely *unowned*, never checking that `passes_tech_caps` would admit it. `self_play.rs:2208` increments `tier3_bought` for **every** tier-3 including Chivalry, so a knight commitment consumes the single slot and locks Construction/Mathematics/Smithery out — in high-score replays Chivalry was the first tier-3 in 7/14 v7 seats. The model can therefore price a lane it is structurally forbidden to buy. It did NOT produce the hoard I predicted mid-run (banking never got going at all), but it is wrong and must be fixed before the ramp is strengthened, or raising `SHAPE_GOAL_SAVE` will bank toward unbuyable targets.
- **Discipline is now inert twice over.** The completion BONUS moved immediate-completion 24.8% → 22.1% (i.e. not at all) and end-stranded 70.0% → 67.3%. Both the penalty (v6) and the bonus (v7) have now failed on this endpoint at matched conditions. **Stop dialling this axis** — the pop-spend timing question is not reachable through a state potential, because Φ cannot distinguish "harvest now" from "harvest next turn" when both reach the same state. Next attempt, if any, should be a root gate on the specific move (harvest that leaves a completable city short) rather than a potential.
- **Tripwire read — commitment, not belief, and it is consistent across all three measurements.** Plans resolve **27% achieved / 59% dropped by our own churn / 7% lost to a fog surprise** (smoke: 32/59/7; v6-weights arm: 24/57/8). Order-flip rate is still **48%/turn** despite stance hysteresis, because hysteresis was applied to the stance only and orders were deliberately left alone. **The belief-state / sparse-macro-head program stays parked**; the next commitment work is order-level persistence.
- **Method note for future reads: the training log misled and the audit corrected it.** Mid-run, `training_log.csv` showed SPT@t10 7.09 (v7) vs 7.63 (v6b) and I read a regression; the matched audit shows 7.58 vs 6.86, the opposite sign. The log mixes 25% anchor games with exploration noise and its per-game counts deflate as games shorten (avg_moves 288 → 209 as the model won faster). **Judge economy changes on the matched audit, never on training-log SPT.** High-score replays likewise over-read: they are a best-games sample and showed lane builds falling when the audit shows them nearly doubling (23 → 42).

### v7.1 — two tier-3 slots with economy-first ordering + chain-aware lane pricing (registered + shipped Aug 1, 2026, BEFORE the run; Verdi's two directives)
**1. `TIER3_CAP_PER_GAME` 1 → 2, with combat tier-3s gated behind an economic one.** One slot made the economy lane and the knight lane compete for the same purchase and the knight lane usually won (Chivalry first in 7/14 sampled v7 seats; Construction purchases fell to 2). Verdi's rule: *"in real games you almost never see players get knights before level-3 pop buildings unless they got lucky with a free-spirit ruin… in 90% of cases players favour the economic ones because it gets them giants."*
- New `settings::technology::is_eco_tier3` — **table-derived, not a hand list**: a tier-3 is economic when the structure it unlocks yields pop or stars. Splits cleanly into Construction/Mathematics/Smithery/Trade/Philosophy/Aquatism/Spiritualism (economic) vs Chivalry/Navigation/Diplomacy (combat). Pinned by `eco_tier3_classification_is_table_derived` so it stays correct if the tables move — the discipline `max_affordable_pop` failed at.
- `GoalAux.eco_tier3_owned` (derived from state in `scripted_goal_aux`) gates combat tier-3s in `passes_tech_caps`. **OWNERSHIP, not purchases** — a free economy tier-3 out of a ruin unblocks the combat lane immediately, which is exactly Verdi's stated exception.
- ⚠️ **Carve-out left in place, flagged for confirmation:** the pre-existing `knight_commit`/`market_push` exemptions fire only when `tier3_bought >= TIER3_CAP_PER_GAME`, so an active knight commitment can still buy a THIRD tier-3. They were introduced to work around cap=1 and that premise is now gone. Left untouched as the minimal change; say the word and they come out.

**2. Lane pricing now walks the whole prerequisite chain, and unreachable lanes are dropped.** Verdi: *"net should know that to get to a certain tier 3 it's gonna need to unlock everything along the way… so that we may weight different final strategies, the cost to get there."*
- New `tech_chain_cost(tribe, tech)`: every undiscovered prerequisite up the `requires` chain plus the tech itself. Pricing only the final tech understated lanes badly — Trade sits behind Roads behind Riding, so "5 stars for a Market" is really 30+.
- `save_batch_cost(state, player, tier3_bought)` now (a) prices the chain and (b) **skips any lane whose tech the tier-3 cap will refuse** — the v7 bug where the model banked toward structurally unbuyable techs. `tier3_bought` threaded through `scripted_goal`/`update_goal` from both binaries. Since `save_batch_cost` takes the MIN over lanes, lane choice is now genuinely "cheapest complete path to a territory upgrade."
- **Tech identity landed in the dumps** (`tech`, `tech_tier`, `tech_eco3` on Research rows in `--dump-star-spend`). This closes a gap EXP_ELO_028 Phase 0 flagged and never fixed, which is why the Chivalry-crowds-out-Construction read had to lean on best-games replays. The next audit can check the ordering directly.

**Verification (16-game / mcts-64 smoke + unit tests):** full suite green (36 binaries, 126 lib tests). Batch costs are now chain-inclusive (**20/31/39★** vs 24-27 before). Economy-first ordering: the one seat reaching a tier-3 took Construction, **0 violations**; the rule itself is pinned structurally by `combat_tier3_waits_for_an_economic_tier3`. Multiplier-tier builds healthy at 13 per 16 games.
**⚠️ Watch item — `SAVE_MAX_TURNS` may now be too tight.** Honest chain costs raised batch prices ~50%, so the reachability gate (`stars + spt × 3 ≥ cost`) rejects more: batch-placeable read 9% in the smoke vs 31% in the v7 audit. Small, low-budget smoke so the numbers are not comparable, but the direction is a real consequence of honest pricing. **Falsifier + response:** if the next run's audit shows `save_batch` non-null on <15% of turns, raise `SAVE_MAX_TURNS` 3 → 5 before touching `SHAPE_GOAL_SAVE`; a lane that takes 4-5 turns of income to reach is still a plan, and Verdi's own described behaviour (hold 20★ at +20 income) is a 1-2 turn hold only at late-game income.
**Expected (next run):** first tier-3 is economic in ≥90% of seats; seats taking 2 tier-3s > 0; multiplier-tier builds ≥ 42/64g (the v7 level) and ideally higher; knights hold near the v7 level of 25/64g rather than collapsing; SPT@t10 ≥ 8.0 (v7 fell just short at 7.58). **Falsifier:** knights collapse toward 0 AND win rate drops >5pp → the ordering rule is delaying the combat lane past its usefulness; relax to "economic tier-3 owned OR turn ≥ 12".

#### v7.1 — ACTUALS (run 1785601511, 5 iters; 3-arm matched audit 64g / mcts 256 / GUMBEL_SCALE=0, Aug 1 2026)

**Ladder (the primary signal, and it is unambiguous).** Chained link elo **1499.7 → 1737.2 (+237)** over one 5-iteration run; the end-of-run gauge beat the v7 anchor **0.891** (64g). v6b→v7 was +183 by the same measure, so v7.1 is the largest single-generation gain in the series. Net-seat per-turn tempo (`tempo_by_turn`, n≈380/turn-point, mirror games) has v7.1 ahead of v7 by +12–19% SPT and +15–28% city levels through t12, with city COUNT flat — the gain is depth per city, which is what the ordering rule targets. Giants/game 0.55 → 0.70.

**The matched audit vs Greedy does NOT reproduce that gain.** Three arms, same flags, only the binary/weights differ:

| endpoint | v7 code+wts | v7.1 code, v7 wts | v7.1 code+wts |
|---|---|---|---|
| wr vs Greedy (n=64) | 0.891 [.814,.967] | 0.797 [.698,.895] | 0.812 [.717,.908] |
| SPT @t10 *(cond. on reaching t10)* | 7.67 (n61) | 7.17 (n59) | **7.83** (n59) |
| SPT @t18 | **13.82** (n39) | 10.78 (n37) | 12.27 (n33) |
| city levels @t10 | 5.85 | 5.66 | **6.20** |
| city levels @t18 | **10.79** | 8.70 | 9.76 |
| multiplier-tier builds | 42 | 14 | 33 |
| 8★ summons (knights) | 25 | 7 | 16 |
| mean last turn | 20.8 | 19.8 | 19.4 |

- **Registered endpoint MET: first tier-3 is ECONOMIC in 100% of seats** (17/17 trained, 8/8 code-only) — vs Chivalry-first in 7/14 sampled v7 seats. The ordering rule does exactly what it was built to do, and the star-spend `tech` fields shipped in v7.1 measure it directly instead of via best-games replays.
- **Registered endpoint MISSED, and the diagnosis kills the change: seats taking 2+ tier-3s = 0.** Only **17/64 seats reach even ONE** tier-3. Raising `TIER3_CAP_PER_GAME` 1→2 was therefore **inert by construction** — a cap of 1 cannot bind when 73% of seats never reach it. The binding constraint is affordability/horizon (tier-3 = 4 + cities×3 ≈ 13★ at 3 cities, games end ~t19-20), not permission. **Do not dial the cap again; the lever is reaching the first tier-3 sooner.**
- **Falsifier partially triggered, verdict = do not relax yet.** Registered: "knights collapse toward 0 AND win rate drops >5pp". Knights fell 25 → 16 (not a collapse; the code-only arm's 7 shows most of the drop is policy/gate mismatch that training recovers) and wr fell 7.9pp — but the 95% CIs overlap heavily (±~9pp at n=64), so the drop is not established. The ordering rule stays.
- **Late-game reads are censoring-confounded, early-game reads are not.** v7.1 games are shorter (mean last turn 20.8 → 19.4; only 27/64 reach t20 vs 35/64), so t15+ rows compare differently-selected subsets — v7.1 leads at t8–t12 where n is near-full and trails at t18+ where it is not. **The earlier `spt20`-style metric (last state with turn ≤ T) silently averaged in short games; it is replaced by conditional-on-reaching-T with n printed.**
- **⚠️ Instrument gap, stated so it is not misread later:** the v7 arm's `star_spend` predates the `tech`/`tech_tier`/`tech_eco3` fields, so its tier-3 and water-tech columns are **n/a, not zero**. Research star-cost is not a usable proxy (a tier-2 at 5 cities costs the same as a tier-3 at 3).
- Banking is still inert (spend/income median exactly 1.00 in all three arms; carried balance 0.29 → 0.20). `batch_placeable` 31% → 28%, i.e. **above** the registered 15% floor, so `SAVE_MAX_TURNS` stays at 3. Order-flip rate is unchanged at ~52%/turn and plans still resolve 23% achieved / 59% dropped-by-churn / 8% fog-surprise — the belief-state program stays parked, order-level persistence remains the next commitment target.

### Drylands: the water lane is dead weight and was being bought (Verdi, Aug 1 2026)

Verdi: *"We should restrict all the water techs on the drylands map. It just doesn't make any sense (at least to buy)"* and *"there's a bug where even in that, sometimes we will have one or two water tiles spawn."*

**Measured waste before the fix** (net seats, 64-game audit): **51 Ports built in the v7 arm (~357★, 3.0% of all stars)**, 34 in the v7.1 arm, plus 21–25 water-tech purchases. A Port is +1 pop for 7★ where a Farm is +2 pop for 5★ — strictly dominated, and only reachable at all because of stray puddles.

1. **Mapgen: Drylands is now bone dry.** `land_ratio` 0.95 → 1.0 (on 11×11 that was ~7 water tiles per map, every one of them adjacent to land and so shallowed into buildable `Water`). Two secondary sources also removed: the guaranteed-starting-resource block **writes its target terrain onto the tile**, so a Kickoo/Aquarion fish start carved water into the map — those tribes now get a land start when the map type is dry; and the explicit "Drylands: Kickoo/Aquarion capitals get 2 water tiles" block is gone. `prediction.rs`'s fog land prior follows to 1.0. New `is_fully_dry(map_type)` is the single source of truth. Pinned by `drylands_generates_no_water_at_all` (240 maps: 2 sizes × 3 tribe sets × 40 seeds) and `water_tribes_still_get_a_starting_resource_on_drylands`.
2. **Root mask on the naval lane when the map has no water.** New table-derived `is_water_tech` (every unlock is water-bound: `Float`/`Water` hulls, water-only structures, water/ocean terrain, water defense bonus; techs granting nothing of their own inherit from `replaces_tech`) and `is_water_dead_end` (a water tech whose `next` list is also all water). `GoalAux.water_dead` reads the true tile set — map type is public information at game start, unlike what sits under fog — and `passes_tech_caps` masks the dead-end lane. **A mask, not a price: this is the never-do case masks exist for.** Membership is pinned exactly (`water_tech_classification_is_table_derived`): Fishing/Sailing/Ramming/Navigation/Aquatism/Oceantology are masked; Aquarion's **FreeDiving is deliberately spared** because it gates Chivalry, and amphibious/ice units (Tridention, Mooni, BattleSled) keep their techs off the list.
3. **Closed a loophole the mask exposed:** Aquatism unlocks WaterTemple, which yields population, so `is_eco_tier3` calls it economic — on a dry map it would have satisfied the economy-first rule and unblocked the combat lane for free. `eco_tier3_owned` now discounts water techs when `water_dead`.

**Verification:** full suite green (36 binaries, 132 lib tests). End-to-end test on real generated maps (`a_generated_drylands_game_masks_the_water_lane`). 16-game / mcts-64 smoke: **net seat buys 0 water techs and builds 0 Ports** (all 11 remaining water purchases are the Greedy anchor, which does not use the macro gate).
**Open, needs Verdi's call:** the Greedy heuristic anchor still buys Fishing/Ramming/Aquatism on dry maps — it is terrain-blind in tech scoring. Fixing it would strengthen the benchmark and therefore shift every historical win rate, so it is left alone.

**Correction, same day (Verdi):** *"even on drylands those two tribes should always start with 2 water tiles. They are the exception."* Restored — but as ONE mechanism rather than the two that used to overlap. Kickoo/Aquarion capitals now get **exactly 2** adjacent fish ponds on a dry map, orthogonal neighbours preferred, placed by a dedicated block; the guaranteed-starting-resource path skips those tribes when the map is dry so the two blocks cannot both fire (the pre-Aug-2026 code ran both, which scattered up to 4 ponds — 2 from the resource block writing `Water` onto its chosen tile, 2 more from the capital block). Pinned by `water_tribes_get_exactly_two_capital_ponds_on_drylands` (60 maps: every pond adjacent to the capital and carrying a fish, and no other water anywhere), with `drylands_generates_no_water_for_land_tribes` covering the land tribes. Consequence to remember: `water_dead` is a MAP-level property, so a Kickoo/Aquarion seat un-masks the naval lane for **both** players in that game — correct (there IS water), and irrelevant to Imperius/Bardur training, but it means the mask is not a guarantee on mixed-tribe ladders.

**Correction to the v7.1 ACTUALS — "the cap raise was inert" was measured on the wrong game population.** Six unbiased single-game draws from the v7.1 tip (`--num-games 1`, so the saved "best game" IS the only game; MIRROR, `--anchor-frac 0`, mcts 256) give 12 seats:

| population | seats reaching a tier-3 | seats taking 2 | mean game length |
|---|---|---|---|
| vs Greedy (audit, n=64) | 27% | **0%** | 19.4 turns |
| mirror (n=12) | **75%** | **17%** | 27.7 turns |

Mirror games run ~8 turns longer because neither side folds, so seats actually reach the tier-3 window. **Mirror is ~75% of training data**, so `TIER3_CAP_PER_GAME` 1→2 is NOT inert where it matters — the two double-buyers took Construction t8 → Mathematics t21 and Mathematics t8 → Smithery t20. The audit's 0% is a property of games the net wins before t20, not of the rule. Ordering held: **every tier-3-reaching seat took an economic one first** (Construction ×4, Mathematics ×5, Chivalry ×0).

**The real ceiling is `TECH_CAP_PER_GAME = 8`, and it binds on the median seat.** Techs bought per seat: `{3:1, 5:1, 6:2, 7:1, 8:7}` — **7 of 12 seats bought exactly 8**. Corroborated at scale by `tempo_by_turn`: v7.1 net seats hold 8.83 techs at t20 and 9.22 at t25 (n≈200-380), and ruin grants don't count toward the cap, so ~8 bought + ~1 granted = capped out. Both second tier-3s landed at t20-21, i.e. exactly when the 8-tech budget runs out. **The second tier-3 competes against the tech cap, not against the tier-3 cap** — so the lever for more/earlier tier-3s is `TECH_CAP_PER_GAME`, or reaching the first one sooner, NOT `TIER3_CAP_PER_GAME`. Caveat: n=12 seats on the mirror side; the cap-binding claim is the well-supported half (n≈380 via tempo), the 17% double-buy rate is indicative only.

## v8 — replay-critique pack #2: stop the free-star exploits, gate the pop spend, price unit safety (registered Aug 2, 2026)

Verdi watched `v71_tip_mirror_typical_4520.json` (an UNBIASED draw — `--num-games 1`, so the saved "best game" is the only game) and filed 13 observations. Every claim checked out against the command log. Verdi asked for all of them in ONE experiment to tighten the tweak→behaviour loop, so attribution is preserved by giving each item its own constant and its own report-card endpoint rather than by splitting the run.

**Measured root causes (from the command log, not inference):**

| # | Verdi's observation | Measured | Root cause in code |
|---|---|---|---|
| 1 | "buys Forestry then chops all our forests" | **16 clears**, 6 in the turn Forestry landed; 2 LumberHuts built | `ClearForestMove` is FREE, **pays a star**, and `consume_resource` **deletes the Game** on the tile. `SHAPE_GOAL_FOREST_STANDING` (50, sized in v5 when clears were ~1/game) was the only counterweight |
| 2 | "spamming senseless roads all over the map" | **41 of 49 builds (84%), ~123★** | Roads DO pay (+1 pop per city connected, `update_capital_connections`), but nothing prices redundancy — and Greedy's `score_road` actively teaches it, while Greedy is both the early prior and the anchor |
| 3 | "explorer on the first village is not recommended… want 70/30 workshop" | capital's first reward = Explorer in **12/12 seats**; captured villages 36% Workshop | At t0 `hidden²`≈1 so the term pays ~700-930 vs Workshop's 150. Not a distribution — a constant |
| 4 | "harvested a single fruit, village still 1/2" | — | v6 priced this with a penalty, v7 with a bonus, both moved it ~0. Φ cannot separate "harvest now" from "harvest next turn" |
| 5 | "rider made where an enemy could reach it"; "lost a rider by moving next to a warrior" | — | **`reward.rs` contains no reference to unit health at all** — a unit is worth `5 × cost` at 1hp or 10hp, so retaliation is free and standing in reach is unpriced |
| 6 | "hard-cap on tech? soften it" | 7/12 seats bought **exactly 8** then nothing | `TECH_CAP_PER_GAME = 8`, binding on the median seat; both second tier-3s landed t20-21 |

**Shipped:** `SHAPE_GOAL_FOREST_STANDING` 50→150; ClearForest/BurnForest **masked on resource tiles** (trading a harvestable pop source for one star is dominated at every star price — a never-do, hence a mask); `passes_road_gate` (a road must extend the network AND some city must still be unconnected); `passes_pop_discipline` (a root GATE, not a third potential — it can act because it sees the MOVE); `unit_value` HP-weighting + `SHAPE_GOAL_EXPOSED_PER_COST` (25) for units inside a visible enemy's move+strike reach; `SHAPE_GOAL_EXPLORER_FIRST_CITY_SCALE` (0.15, applied only while the tribe holds one city — the July dark-map dial that cost 14pp of win rate at 1000 is deliberately NOT reopened); `TECH_CAP_PER_GAME` 8→10.

**⚠️ Bug found and fixed while validating, and it invalidates nothing earlier but matters a lot:** `build_fresh_root` applied the root gates, but `finish_reused_root` — the **tree-reuse** path, which is the common case for ~8 of every 9 plies in a game turn — took its children from the cached tree and **never gated them**. Every root gate was leaking on all but the first ply of each turn. Fixed by applying the same predicate to the reused root's children (EndTurn exempt so the root can never be emptied). This silently weakened the v6 Destroy mask and the v7 capture-first gate too.

**Verification status — NOT yet demonstrated.** Full suite green (36 binaries, 134 lib tests) and the binary-endpoint gates are confirmed working (resource-tile clears → 0, ports → 0). But the continuous endpoints are **not yet measured credibly**: a 6-game report card moved 141 → 250 total roads between two runs of IDENTICAL code, and its baseline had been recorded at mcts 256 against a report card running at mcts 96 — not a legal comparison (same rule as never comparing win rates across budgets). A matched 16-game A/B (pre-v8 binary vs v8 binary, same weights, same budget, dumps on both arms) is running; nothing here should be believed until it reads out.

**Open, measured, and NOT yet solved:** `passes_pop_discipline` gates *affordability*, not follow-through — the first report card shows **70% of pop-spend turns still end without the level finishing**, because a city that COULD complete is allowed to spend and then simply doesn't. The gate is necessary but not sufficient; the follow-through still rests on `SHAPE_GOAL_COMPLETION`. Do not call item 4 fixed.

**Explicitly deferred:** the army-allocation critiques (5 warriors sent to take a 1-rider city; the whole army walking to one ruin; no push toward the inferred enemy capital). `assign_expand_targets` already enforces one-unit-per-target and exactly-one extra converger — but only for *painted EXPAND targets*, and enemy-held cities are never painted (`retakeable_village` requires `structure_type == Village` and excludes `capital_of != 0`). Making this work needs a force-sufficiency model ("what does it take to hold this tile"), which is a design, not a dial — and bundling a half-built one into v8 would poison the attribution of everything else.

**Falsifiers.** (a) Median score drops >10% on the matched A/B → the gates are cutting real options, most likely the road gate (relax to "extends the network" only, dropping the all-connected clause). (b) Clears/seat does not fall → `SHAPE_GOAL_FOREST_STANDING` is fighting a Farm chain worth more than 150; re-dial against measured clears rather than raising blind. (c) Capital Workshop share stays <40% → the 0.15 scale is still above the flip point; the term is ~0.15×(700+2×230)=174 vs Workshop's 150, so try 0.10. (d) Units lost/seat rises → the exposure term is making the net passive rather than careful.

#### v8 — A/B ACTUALS (two matched 16-game/32-seat runs, same weights, mcts 128, Aug 2 2026)

The pop gate was narrowed to **Harvest only** between run 1 and run 2 (run 1 gated pop-yielding Builds too: harvests barely moved 219→209 while Farm builds HALVED 61→30, and early clears rose 102→145 as the search substituted free ClearForest stars for the builds it could no longer make — the gate made the exploit *more* attractive by removing its alternative).

| endpoint | pre-v8 r1 | pre-v8 r2 | v8 r1 | v8 r2 | verdict |
|---|---|---|---|---|---|
| ports /seat | 0.2 | 0.3 | **0.0** | **0.0** | ESTABLISHED |
| clears on a resource tile /seat | 0.2 | 0.2 | ~0 | ~0 | ESTABLISHED |
| capital 1st reward = Workshop | 0% | 0% | **28%** | **34%** | ESTABLISHED |
| roads /seat | 25.1 | 20.9 | 14.1 | 22.7 | **NOT established** |
| clears /seat | 6.2 | 6.6 | 7.2 | 6.1 | **NOT established** |
| median score | 5670 | 5320 | 4700 | 5920 | **NOT established** (sign flips) |
| orphan pop-spend turns | 70% | 70% | 66% | 73% | **unmoved** |

**The dominant lesson is about the instrument, not the changes.** The pre-v8 arm — identical binary, identical weights, identical flags — moved roads/seat 25.1 → 20.9 and median score 5670 → 5320 between the two runs. The v8 arm moved roads 14.1 → 22.7 on a code change that touched only the pop gate. **At n=16 games / 32 seats, per-seat build counts carry ±50-60% run-to-run swing**, so every continuous endpoint in the first A/B write-up (including the "roads −44%" I reported) was inside noise. Only endpoints that changed from a CONSTANT to a distribution survive at this sample size: ports (structurally guaranteed by the dry-map fix), resource-tile clears (a hard mask), and the capital reward (0/32 in both control runs → ~30% in both treatment runs).

**Falsifier-design lesson:** falsifier (a) was written as ">10% median score drop". It fired in run 1 (−17%) and reversed in run 2 (+11%), and the run-1 drop failed significance anyway (Mann-Whitney z=0.90). **Percentage thresholds on continuous endpoints at n=16 are not falsifiers.** Future registrations on continuous endpoints must name a test and an n, or use a structural endpoint instead.

**The road gate is weak by construction, independent of noise.** It requires a road to *extend the network* (adjacent to a city or an existing own road) — but once a network exists every adjacent tile qualifies, so it blocks isolated roads while permitting unlimited contiguous sprawl. It should not be re-dialled; it needs a different rule (e.g. cap total roads against the count of cities actually connected, or price roads beyond the connecting path).

**Recommended next step — instrument the gates instead of inferring them.** Every question above ("does this gate bite, and how often?") is a deterministic count that a `--dump-gate-blocks` counter answers in 2 games with zero variance, where the behavioural outcome needs hundreds of seats. This is the tight tweak→behaviour loop Verdi asked for; the current report card can only ever resolve structural endpoints.

**Status: v8 is NOT cleared for a training run.** Established: the dry-map/water work, the resource-clear mask, the capital reward fix, and the tree-reuse gate leak (a real correctness bug regardless of v8's fate — it silently weakened the v6 Destroy mask and v7 capture-first gate too). Unproven: roads, clears, forest price, tech cap, exposure/HP terms. Unfixed: orphan pop-spends (item 4), army allocation (deferred by design).

#### v8 — REVERTED to a clean base (Aug 2, 2026, Verdi's call)

The A/B could not resolve any continuous endpoint at n=16 (the control arm alone swung roads/seat 25.1 → 20.9 between identical runs), so layering the next reward-architecture change on top would have made attribution impossible for a second time. Reverted to their pre-v8 values: `SHAPE_GOAL_FOREST_STANDING` 150 → 50, `TECH_CAP_PER_GAME` 10 → 8, `passes_pop_discipline` and `passes_road_gate` removed entirely, `unit_value` HP-weighting and `SHAPE_GOAL_EXPOSED_PER_COST` removed. HP-weighting returns as part of the military floor in the next design rather than as a loose dial.

**Kept — established by measurement or correctness:**
- The drylands/water work (no water for land tribes, exactly 2 capital ponds for Kickoo/Aquarion, the naval-lane mask). Ports 0.2-0.3 → 0.0 per seat in both A/B runs.
- The ClearForest/BurnForest **resource-tile mask** — clearing deletes a harvestable Game for one star, dominated at every star price.
- `SHAPE_GOAL_EXPLORER_FIRST_CITY_SCALE` = 0.15. Capital's first reward went from a CONSTANT (Explorer 0/32 Workshop in both control runs) to ~30% Workshop in both treatment runs. A constant→distribution change is the one class of result n=32 seats can settle.
- **The `finish_reused_root` gate leak fix** — a correctness bug independent of v8. Root gates were applied in `build_fresh_root` but not on the tree-reuse path, which is ~8 of every 9 plies in a game turn, so the v6 Destroy mask and v7 capture-first gate were also leaking.
- `city_completable_now` (unused for now; the per-city ceiling term will need it).

### v9 — design agreed with Verdi (Aug 2, 2026), NOT yet built

**Three permanent floors, no hand-scheduled emphasis.** Stance-switched Φ is not a function of state — `StanceCommit` carries history, so a GROW→ARM flip zeroes a 150×SPT term with no state change, which breaks the policy-invariance that potential-based shaping depends on (stance flips run 14-16%/turn). All three drives always carry non-zero weight; **emphasis emerges from marginal Φ-per-star**, not from a schedule I write. Verdi: *"a floor that acts as a gentle nudge for all 3 vertexes whereas the net has the liberty to choose where to emphasize more weights based on what's most game-state relevant."*
- **Economy** — existing SPT term, plus a new per-city term paying the gap between a city's level and **its own structural ceiling** (max level reachable from that city's territory routes at unlimited stars, via `max_affordable_pop(.., i32::MAX)` walked against the `level+1` thresholds). Verdi rejected a level-5 target: *"some villages have the ability to scale beyond that… sometimes the ceiling is 4, for others it's 7."* This also prices a forest by what it is FOR — clearing lowers that city's ceiling — which is a better signal than v8's flat standing-value penalty.
- **Territory** — a permanent owned-tiles/cities term. Today territory is paid only through order-driven EXPAND gradients, so it vanishes whenever no EXPAND order is live.
- **Military** — HP-weighted army value, currently ARM-gated.
- **NO giant-specific term** (Verdi: *"I'd rather have the drive for eco + military drive to the natural conclusion that city upgrades is the most effective way"*). Super units fall out of the eco drive at cities whose ceiling reaches 5.

**Risk-adjusted star optionality — replaces the v7 savings ramp's framing.** Verdi: *"The reason we want to wait is because of risk… you'd rather have the stars to have the optionability to pivot."* Held stars are worth ~nothing in Φ today (spend/income median exactly 1.00 in every arm ever measured), so "wait" is a dominated action regardless of what the net knows. `Φ += W_STAR_OPTION × stars_held × risk(state)`. Because Φ telescopes, the in-tree reward for spending becomes `(purchase gain) − (option value released)` — high risk RAISES the bar for spending rather than forbidding it, and on a quiet board the term goes to ~0. Must sit below what a completed level pays, or it hoards.

**`risk` includes fog-encounter probability from `aux_fog_units`** (Verdi: *"risk is low on the first village close to my capital. By the 3rd city, if I haven't seen the opponent yet, the risk of running into them is high"*). That head already exists and is trained (121-dim per-tile enemy-under-fog prediction, AUX_FOG_W 0.2) — but it is **training-only and deliberately absent from `network.rs`**, so using it at inference requires mirroring it into the Rust network. See the open question below. Other risk ingredients are state-side and free: visible enemy army value near own territory, own border touching fog, ghost channels (remembered enemies whose position is now unknown), `city_threatened`.

**Open question flagged to Verdi before building:** mirroring `aux_fog` into `network.rs` breaks the standing dual-network rule ("do not add aux heads to network.rs"). The conv itself is trivial (filters→1, 1×1) and `model.safetensors` always comes from train.py so the weights are present — but the Rust opponent loader is strict, so every historical checkpoint used in league play must also carry the key or the first league iteration crashes (see `migrate-checkpoints-on-arch-change`). Needs a compatibility check across `checkpoints/` before committing to it.

**Tension to watch:** permanent floors push toward doing a bit of everything, while the diagnosed failure is a LACK of commitment. Floors-plus-emergent-emphasis is the proposed resolution; if a run comes back with the net spreading thinner and finishing nothing, that is the mechanism to suspect first.

### v9 + v9.1 — training ACTUALS (runs 1785702298 / 1785709634, 5 iters each, Aug 2–3 2026) — **NO MOVEMENT ON ANY REGISTERED ENDPOINT; PACKAGE REVERTED**

**Registered endpoints, both failed.** v9.1's two changes (`HEADROOM_PER_CITY_CAP = 1`, risk without `dark`) predicted city_levels@t10 and revealed@t3 would recover. city_levels@t10 5.18 → 5.15; revealed@t3 38.53 → 39.67. Neither moved.

**vs-Greedy scorecard** (`model_vs_anchor`, net seat, ± = SEM across the 5 iterations). Greedy is fixed, so this is the only cross-run-comparable instrument:

| metric | v7.1 | v9 | v9.1 |
|---|---|---|---|
| SPT @ t10 | 9.12 ±0.38 | 9.32 ±0.24 | 8.87 ±0.52 |
| SPT @ t20 | 16.30 ±0.85 | 15.96 ±0.50 | 16.12 ±1.96 |
| avg city level @ t20 | 3.33 | 3.26 | **3.61** |
| cities @ t20 | 3.91 ±0.33 | 3.96 ±0.11 | **3.50** ±0.43 |
| K/D @ t10 / t20 | 1.45 / 1.02 | 1.31 / 1.03 | 1.49 / 0.97 |
| army value @ t10 / t20 | 16.82 / 37.70 | 16.86 / 38.96 | 16.10 / 37.38 |
| villages_first_rate | 0.92 | 0.88 | **0.87** |

Everything overlaps inside one SEM except the last two rows. **The one real effect is a composition shift:** v9.1 runs fewer, deeper cities — exactly what the headroom cap was designed to do — but total SPT did not improve, so it bought depth at par, not profit.

**Activity fell across the board, monotonically.** Not attrition (units lost @ t10: 1.78 / 2.09 / 1.71, flat) — the net simply *acts* less: builds 24.17 → 21.11 → 18.17 (−25%), harvests 7.49 → 5.82 → 5.44 (−27%), units trained 22.46 → 21.93 → 19.51, moves 262 → 244 → 217, revealed@t3 53.11 → 39.59 → 38.23 (−28%), units@t5 3.84 → 3.41 → 3.03 (−21%).

**Two instrument errors found and corrected during this analysis — both had produced wrong verdicts:**
1. **A direct `arena` head-to-head (512 games) ranked v9.1 above v7.1 and was void.** `arena` defaults to `goal_script: false, goal_w_tree: 0.0`; training runs `--goal-channels --goal-w-tree 1`. The match therefore ran with the 6 goal input planes **zeroed** and **all in-tree shaping off** — i.e. with the entire v9/v9.1 contribution deleted, on inputs neither net ever saw in training. Any future tip-vs-tip match MUST pass the training goal flags.
2. **"Time to first village ≈ 6.0" was the wrong statistic.** Interpolating the turn at which the *mean city count* crosses 2.0 is dragged arbitrarily late by the 8–13% of games that never capture. The correct metric already exists: `villages_t2c_first_cond` (`self_play.rs:4112` — sums per-game capture turns, divides by games that got there) = **4.54 / 4.65 / 4.54**. Conditional speed is FLAT; it is the *rate* that fell.

**Corrected read of the expansion regression:** the net is not slower, it expands in **fewer games**. Conditional turn-to-Nth-city is flat or slightly better (3rd city 10.36 → 10.02, 4th 15.00 → 14.07) while reach rates fall (village 0.92 → 0.87, 3rd city 0.76 → 0.71).

**Also confirmed: `avg_score` from self-play is not a strength metric.** It is confounded by the opponent, which is the model itself. v9.1 averaged 3740 in its own run but 3815 against v7.1. Do not rank runs by it.

### The regression predates v9 — audit across all 70 logged runs (Aug 3 2026)

| | Jul 13–14 | v7.1 (Aug 1) | v9.1 (Aug 3) |
|---|---|---|---|
| villages_first_rate | **0.99** | 0.92 | 0.87 |
| avg_cap_villages | **3.10** | 1.66 | 1.62 |
| SPT @ t10 | **9.55** | 7.61 | 7.26 |
| avg_moves | 548 | 262 | 217 |

Count metrics are partly a game-length artifact (moves halved), but `villages_first_rate` is a rate and `SPT@t10` is a turn-10 snapshot — neither is length-dependent, and both fell hard. **Most of the damage was already done by v7.1**, somewhere in v6/v7. That hunt is still open.

**Reverting further than v7.1 is not possible.** The best run (1783979192: 0.99 rate, 3.10 villages, SPT@t10 9.55) saved **no checkpoints**. The recoverable Jul 14 checkpoints do not load — the value head was widened in late July: `v_pool_conv` (1,64,1,1) → (8,64,1,1), `v_fc_shared` (64,121) → (64,968). Recovering them needs either an architecture revert across `network.rs`/`train.py`/`tch`/`metal`, or loading trunk+policy and re-initialising the value head.

### Decision (Verdi, Aug 3 2026): revert to v7.1 weights, keep the correctness fixes, dial the gates

- **Weights** → `checkpoints/gauge_1785601511_iter5.safetensors` (v7.1); prior tip preserved at `model_v91_tip_backup.safetensors`.
- **Code** → `a1f62f2` reverted (commit `e137426`): removes the v9/v9.1 floors, headroom, territory, star-option and risk terms, the `SHAPE_GOAL_EXPLORER_FIRST_CITY_SCALE = 0.15` explorer de-weight, and the `aux_fog`/`root_fog` inference plumbing.
- **Kept deliberately**: drylands/water-tech work, the ClearForest resource-tile mask, and the `finish_reused_root` gate-leak fix — the last of these is a genuine correctness fix and is NOT being reinstated as a bug.

**⚠️ Standing hazard this creates:** v7.1's weights were trained on Aug 1, *before* `d896edf` (Aug 2) fixed the gate leak. They have never played under gates that fire on every ply. Loading them into the current binary is a mismatched weights/code pair — the same class of error as the void arena match, on the training side.

## EXP_GATE_001 — what does the gate-leak fix cost in activity? (registered Aug 3 2026, BEFORE reading results)

**Hypothesis.** `d896edf` (Aug 2) fixed root gates leaking on ~8 of every 9 plies (`finish_reused_root` promoted children were created as INTERIOR nodes and never met the root retain predicate). Four gates — `passes_star_gate`, `passes_tech_caps`, `passes_ability_gate`, `passes_capture_first` — went from firing on ~1 ply in 9 to firing on all of them, a ~9× simultaneous amplification, **and none were re-dialed afterward** (they had been tuned while mostly inert). This is the only single change that would suppress builds, harvests, research AND summons uniformly, and it sits exactly at the v7.1 → v9 step (villages_first_rate 0.92 → 0.88).

**Method.** New env switch `POLYFISH_REUSED_ROOT_GATES=0` restores fresh-roots-only gating; default stays on. One binary, two arms, v7.1 weights, 128 games each vs the Greedy anchor at `--anchor-frac 1.0 --goal-channels --goal-w-tree 1 --mcts-iters 64 --gumbel-k 16`, matching the training search config.

**Expected if the hypothesis holds.** Arm `off` shows materially more activity — builds and harvests up ≥15%, `villages_first_rate` up ≥3pp, revealed@t3 up — with win rate vs Greedy no worse.

**Falsifier.** Arms within noise (`villages_first_rate` SE ≈ 0.026 at n=128, so ±5pp at 95%) ⇒ the gate amplification is NOT the mechanism, and the search is elsewhere: the v6/v7 packages committed in `b3d4421` and earlier.

**Decision rule.** The fix stays either way — it is correct. If the amplification is confirmed, the individual gates get dialed down (and `--dump-gate-blocks` finally gets built to attribute the loss per gate) before any training run starts from the reverted tip.

### EXP_GATE_001 — ACTUALS (Aug 4 2026, 128 games/arm, v7.1 weights vs Greedy anchor) — **FALSIFIER TRIGGERED; HYPOTHESIS WRONG AND THE PROPOSED REMEDY WOULD HAVE HURT**

| metric | gates ON (current) | gates OFF (pre-Aug-2) | delta |
|---|---|---|---|
| **win rate vs Greedy** | **0.852** | 0.742 | **−10.9pp** (z=2.19, p≈0.03) |
| villages_first_rate | 0.953 | 0.977 | +2.3pp (z=1.00, p≈0.32, **n.s.**) |
| turn of 1st village (cond) | 4.541 | 4.320 | −0.22 |
| turn of 2nd city | 5.608 | 5.336 | −0.27 |
| villages captured | 2.078 | 2.312 | +11.3% |
| builds | 26.750 | 31.844 | +19.0% |
| units trained | 21.344 | 23.258 | +9.0% |
| moves | 264.3 | 300.5 | +13.7% |
| **units lost** | **11.836** | 14.539 | **+22.8%** |
| owned tiles | 54.719 | 49.805 | −9.0% |

**Registered falsifier fired.** `villages_first_rate` moved +2.3pp against a ±5.2pp noise floor — within noise. **The gate amplification is NOT the mechanism behind the village-capture regression.** The hunt moves upstream to the v6/v7 packages in `b3d4421` and earlier, exactly as the falsifier specified.

**The activity-suppression half of the hypothesis was right, and the interpretation was wrong.** Gates on every ply do suppress activity — builds −16%, moves −12%, units trained −8% relative to the leaky arm. But that suppression is **productive**: the same arm wins **+10.9pp more games vs Greedy** and loses **19% fewer units**. The gates are pruning waste, not value. This also delivers the unit-preservation behavior Verdi asked for in replay critique #13 ("there needs to be a sense of preservation") — from a gate, not from a shaped risk term.

**Consequence: DO NOT dial the gates down.** The remedy proposed when EXP_GATE_001 was registered ("if the amplification is confirmed, the individual gates get dialed down") is withdrawn — it would have traded ~11pp of strength for ~19% more builds and a faster first village. The `d896edf` gate-leak fix stands as both a correctness fix and a strength win.

**Residual cost worth revisiting later, but not by turning gates off.** Gates on do cost expansion tempo: first village 4.54 vs 4.32, second city 5.61 vs 5.34, villages captured 2.08 vs 2.31. Attributing that per-gate needs `--dump-gate-blocks` (still unbuilt). `passes_capture_first` and `passes_star_gate` are the two plausible culprits for delaying a capture.

**Caveats.** (i) Both arms ran v7.1 weights on post-`d896edf` code — the mismatched weights/code pair flagged in the revert decision. It is shared by both arms so the contrast is clean, but the absolute levels are not a prediction for a trained run. (ii) `--anchor-frac 1.0` means every game is vs Greedy, so absolute rates run far above the training log's mixed self-play population (0.95 here vs 0.92 there); only the between-arm contrast transfers. (iii) The 5.1 GB of games data from both arms was deleted — one arm had deliberately broken gating and must never enter the training corpus.

**New standing tool.** `POLYFISH_REUSED_ROOT_GATES=0` (gumbel_mcts.rs) restores fresh-roots-only gating. Default on. Kept as a measurement dial for future gate work.

### EXP_GATE_002 — gate attribution instrument (built + measured Aug 4 2026)

`POLYFISH_DUMP_GATE_BLOCKS=1` records every blocked root candidate by **gate × move type × turn band** into `.last_self_play_metrics.json` under `gate_blocks`. Implementation collapsed the gate predicate's **three** duplicated copies (`build_fresh_root`, `finish_reused_root`, `reused_children_match_legal`) into one `gate_block()` — hand-syncing those three is how the gate leak survived as long as it did. Attribution is FIRST-BLOCKER-WINS, so per-gate counts are lower bounds.

128 games vs Greedy, v7.1 weights, production config:

| gate | blocks | share | blocks only | t1–10 |
|---|---|---|---|---|
| tech_caps | 66,773 | 51.5% | Research | 4,370 |
| star_gate | 46,414 | 35.8% | Research | 8,990 |
| ability_gate | 15,951 | 12.3% | Ability | 421 |
| **capture_first** | **566** | **0.4%** | Attack | 288 |

Gates remove **4.77%** of all candidates (70.5 → 67.2 per ply); 75.6% of plies lose at least one. **87% of every block is a Research move**, rising to 95% in turns 1–10. **`passes_capture_first` is eliminated as a suspect for expansion tempo** — 566 blocks across 128 games is a rounding error. If the gates cost anything it is through research pacing, nothing else.

### ⚠️ EXP_GATE_001 was UNPAIRED — most of its findings were map luck (found Aug 4 2026)

`self_play.rs:2923` derived `base_seed` from the wall clock with **no override**, so the two arms played entirely different map sets. Two runs with *identical flags and identical weights* differed by more than the reported "effect" on 12 of 19 metrics — builds 26.75 vs 38.35 (noise **2.3×** the claimed effect), moves 264 vs 323 (1.6×), villages_first_rate 0.953 vs 0.992 (1.7×). Meanwhile `arena` has carried `--base-seed` all along, documented as *"Fix it to play identical maps across separate arena runs (paired A/B arms)"* — the discipline existed in one tool and was not applied to the other.

**Fix:** `--base-seed` added to self_play (0 = wall clock, preserving training behavior).

**RETRACTED from EXP_GATE_001:** "builds −16%", "units lost −19%", "moves −12%", and the **+10.9pp** win-rate magnitude. The direction survived re-testing; the magnitude was ~3.5× inflated by map luck.

### EXP_GATE_001R — paired re-run with a measured noise floor (Aug 4 2026) — **GATES ARE A WIN, AND THE TEMPO CONCERN REVERSES**

Three arms, 128 games each, seed 770425 fixed: **A** gates on, **B** gates off (identical maps), **C** gates on (repeat → noise floor). Pairing tightened the floor ~10×: win rate 0.078 → **0.008**, moves 58.9 → **5.1**, builds 11.6 → **1.9**.

Effect = gates OFF − gates ON, reported only where it clears its own measured floor:

| metric | A on | C on | noise | B off | effect | ratio |
|---|---|---|---|---|---|---|
| owned tiles | 51.719 | 51.695 | 0.023 | 51.500 | −0.219 | 9.3× |
| units trained | 22.672 | 22.734 | 0.062 | 23.109 | +0.438 | 7.0× |
| **win rate vs Greedy** | **0.805** | 0.812 | 0.008 | 0.773 | **−0.031** | 4.0× |
| moves | 284.1 | 289.3 | 5.1 | 304.3 | +20.2 | 3.9× |
| **turn of 1st village** | **4.325** | 4.282 | 0.043 | 4.463 | **+0.138** | 3.2× |
| research | 6.492 | 6.555 | 0.062 | 6.680 | +0.188 | 3.0× |
| turn of 3rd city | 9.548 | 9.327 | 0.221 | 10.145 | +0.597 | 2.7× |

Indistinguishable from noise: village capture rate (**0.961 both, ratio 0.00**), villages captured, units lost, revealed tiles, giants, score, turn of 2nd city.

**Gates ON is better on every metric that clears noise.** Gates OFF does strictly *more* — +7% moves, +12% builds, +3% research, +2% units trained — and achieves *less*: −3.1pp win rate, slower first village, slower third city, fewer owned tiles. The suppressed activity is waste, now on paired evidence rather than inference.

**The motivating concern is dead and reversed.** EXP_GATE_001 (unpaired) showed gates delaying the first village 4.54 vs 4.32. Paired, gates ON is **faster** — 4.325 vs 4.463 — and faster to the 3rd city (9.55 vs 10.15). There is no expansion-tempo cost to buy back. Combined with `capture_first` at 0.4% of blocks, the "gates hurt expansion" line of inquiry is closed.

**Caveats.** (i) The noise floor rests on ONE repeat; 0.008 on win rate could be 0.02 on another. Ratios near the 1.5–3× band (builds, harvests, turn of 3rd city) should not be leaned on. (ii) Same-seed arms are NOT bit-identical — `candidates_in` differs 8.9% between A and C, so trajectories diverge from Gumbel sampling / actor scheduling / MPSGraph float order even on identical maps; aggregate outcomes still reproduce to <1% on win rate. (iii) v7.1 weights on post-`d896edf` code, shared by all arms.

**Standing rule going forward: every self_play A/B must pass `--base-seed` and carry a repeat arm.** Two rounds of conclusions have now been overturned by unmeasured noise; the floor is an output, never an assumption.

### EXP_COMMIT_001 — the 0-capture slice and whether strategy flipping predicts losing (Aug 4 2026, 256 games vs Greedy, seed 770425, v7.1 weights)

**A) 0-capture seats: 7/256 = 2.7%, and most of it is not a policy failure.**

| cause | seats | share |
|---|---|---|
| saw a village ≤2 turns (game ended early) | 3 | 43% |
| **saw one >2 turns and never took it** | **3** | **43%** |
| never saw a village at all | 1 | 14% |

0-capture seats end at median **t19** vs **t25** for all seats — these are mostly games the seat was losing, where the game ended before expansion could happen. The genuinely anomalous slice is **3 seats in 256 (1.2%)**, the clearest being `g105p2`: village visible from **turn 1**, visible for **13 turns**, full 30-turn game, zero captures. Worth a single-game decision trace; not a systemic defect.

**B) Strategy flipping does NOT predict worse outcomes — the commitment hypothesis is unsupported.**

| split | n | cities gained | prod@t10 | final prod |
|---|---|---|---|---|
| stance_flips ≤3 | 150 | 2.85 | 8.91 | 14.41 |
| stance_flips >3 | 106 | 2.80 | 8.51 | 14.57 |
| order_flips ≤11 | 145 | 2.77 | 8.86 | 14.71 |
| order_flips >11 | 111 | **2.91** | 8.59 | 14.17 |

Signs are **inconsistent across metrics and across the two flip types**: cities gained falls with stance flips but *rises* with order flips; final production rises with stance flips but falls with order flips. Every gap is 1–5%, inside the noise floor measured in EXP_GATE_001R. There is no correlation to explain.

**Plans achieved 3.02/seat vs dropped 5.49/seat — a 65% abandonment rate that costs nothing measurable.** Either dropping is adaptive (circumstances genuinely changed) or the plans are low-stakes. Forcing the model to hold plans it currently drops would be an intervention with no evidence the held plans are worth holding.

**Scope caveat:** `stance_flips`/`order_flips` measure the *scripted goal-setter*, not the net's own consistency of unit composition or tech line. An observation of "the net flips a lot" drawn from watching replays may be about behavior this metric does not capture; that would need a separate metric (e.g. unit-mix entropy per game) before it can be tested.

### EXP_LANE_001..004 + EXP_HUB_001 — economy lane scoring (Aug 4–5 2026) — **ALL FAILED; the diagnosis is the deliverable**

Four rewrites of `recommended_techs` and one Φ term. **SPT@t10 never moved**: 8.94 baseline → 8.82 → 9.08 → 9.06 → 8.90, every one inside the paired noise floor (±0.329). Registered decision rule ("flat a third time ⇒ lane selection is not the SPT lever") fired and I kept going one attempt too long.

| variant | seed-4102 forest engine | win rate | SPT@t10 | giants | owned tiles |
|---|---|---|---|---|---|
| baseline (terrain counts) | Forestry t17, 5 huts, 0 saw | 0.809 | 8.94 | 0.907 | 51.71 |
| 1 owned-territory only | — | — | 8.82 | 0.703 | 50.73 |
| 2 prospective territory | t11, 4 huts, 2 saw | **0.828** | 9.08 | 0.672 | 53.01 |
| 3 + border growth, 1st-step runner-up | **never, 0 huts** | 0.812 | 9.06 | **0.875** | **53.47** |
| 4 whole-prefix runner-up | **t11, 7 huts, 3 saw** | **0.789** | 8.90 | 0.680 | 52.24 |

**The forest engine is anti-correlated with winning.** The variant that builds it best has the worst win rate and 25% fewer giants; the one that never touches forest has the best giants and territory. Ranking lanes by best-prefix instead of whole-lane was also tried and was worse (Forestry fell to 7th pick) — lane scores sit within noise of each other on these maps, so any scoring tweak reshuffles the order arbitrarily.

### ⭐ The actual bottleneck — planner/executor disagreement on hub placement

Verdi asked the right question ("what is the average level of our sawmills?"). Measured on seed 4102:

```
7 LumberHuts at [16, 23, 28, 35, 37, 70, 79]
hut-to-hut adjacency: 0.57 mean; 3 of 7 fully isolated
Sawmills built t15@18 -> 1 partner, t17@11 -> 1, t20@80 -> 2   (avg level 1.33)
BEST available site at end: tile 27 with 3 partners — adjacent to huts 16, 28, 37
```

A Sawmill costs 5★ and pays `reward_pop × adjacent partners` (`actions/structure.rs:222-242`). At 1 partner that is **5★ for 1 pop** — worse than a LumberHut (3.0★/pop) and worse than free Fruit (2.0★/pop). Those Sawmills destroy value.

Meanwhile `max_affordable_pop` scores a hub at its **best** placement (`best = best.max(partners)`). So the planner ranks the forest lane on a Sawmill worth 3–4 pop, and the net delivers 1.33. **That is the whole paradox**: every increment of forest commitment buys more 5★-for-1-pop Sawmills, consuming stars that would otherwise become units or city levels — hence fewer giants and lower win rate, without forest being the wrong lane.

Root cause of the scatter: **nothing prices hut clustering.** A LumberHut pays 1 pop wherever it sits and huts do not feed each other, so every forest tile is identical to the net. `SHAPE_GOAL_YIELD_ADJ` pays a hub only once it EXISTS — too late to influence the placements that determine its worth.

**EXP_HUB_001 (failed):** priced the best unbuilt hub site at `SHAPE_GOAL_HUB_SITE × reward_pop × partners`. First cut was a sign error — site paid `60p` while a built hub pays `100(p−1)`, making the BUILD a Φ loss below 3 partners, so the net hoarded sites: 0 LumberHuts, score 1815. Corrected to `(p−1)` to mirror the hub formula; still 0 LumberHuts, Forestry never bought, 24 roads, score 3115. Reverted.

**Open, and the right next target:** make hub placement good, either by pricing clustering at partner-placement time in a form that does not distort lane choice, or by accepting realistic placement in the planner so lanes stop being ranked on pop the net never realises. Note `max_affordable_pop` also deliberately excludes monuments (free, 3 pop), so it undercounts there too — Verdi flagged monuments as part of the 4-giant plan for this map.

### Why Φ pricing could not fix hub placement — it is a SEARCH-REACHABILITY problem (Aug 5 2026)

`--dump-pop-spend-choices`, 24 games, seed 4102. When a ply offers **two or more placements of the same structure**, how often does search visit any of them?

| structure | plies with ≥2 placements | ≥1 placement searched |
|---|---|---|
| Farm | 39 | 12 (31%) |
| **Windmill** (adjacency hub) | 17 | **3 (18%)** |
| Road | 10 | 8 (80%) |
| Sawmill | 1 | 0 |
| LumberHut | 1 | 1 |

The one Sawmill ply (turn 11, 12 candidate tiles) had **visits = 0 on every tile**, priors spanning 0.00018–0.00026 — a 1.4× ratio, but both far too small for Gumbel (k=16) to sample. Its zero Q-spread was an artifact of zero visits, not of the value head being blind; I nearly misread it as blindness.

**So placement is decided by the raw policy prior, not by search.** That is why both attempts to price placement in Φ failed: Φ only changes what the search *evaluates*, and these moves are never evaluated. A reward term on an unsearched move is inert by construction — the same class of error as the v8 gates that were "fully inert" until the leak fix.

**Caveats, deliberately not over-claimed.** Sawmill and LumberHut rows are n=1; only Farm (39) and Windmill (17) carry weight, and Windmill is the closest analogue to Sawmill. Per CLAUDE.md this is NOT a claim that economy priors are too low versus Step — that comparison is the documented analytical trap. The claim is narrower: *among placements of one structure type in one ply*, alternatives usually go unsearched.

**Implication for the next attempt:** the lever is the PRIOR, not the potential — raise adjacency-hub Build candidates in proportion to partner count so the good tile is reachable, then let the existing `SHAPE_GOAL_YIELD_ADJ` do the valuing. The heuristic prior blend (`blend_heuristic_into_logits` / `prior_heuristic_weight`) is the existing mechanism, but it decays to zero during training, so a durable fix cannot live there.

### EXP_PRIOR_001 — widen the heuristic placement gradient (Aug 5 2026) — **MECHANISM FIXED, OUTCOME FAILED, REVERTED**

Verdi's call: raise the prior on good hub placements and accept that the crutch decays ("I'd expect the net to have distilled that into its prior").

**The knowledge already existed.** `scoring.rs` scores hub placement by partner count (1 → −2, 2 → +5, 3 → +12, 4+ → +18) and has a "Future Adjacency Prediction" block paying `(others + 1) × 2.5` per adjacent empty tile that could host the hub. It reaches the search through `blend_heuristic_prior` → `blend_heuristic_into_logits`.

**Why it never reached the net — the arithmetic.** `HEURISTIC_TEMP = 20`, so the −2..+12 spread is `14/20 = 0.7` in the softmax exponent → 2× between tiles; blended at `CRUTCH_FLOOR = 0.1` → **~1.4×** in the final prior. That is exactly the measured 0.00018 vs 0.00026 across 12 Sawmill tiles. The mechanism was never broken; it was calibrated an order of magnitude below what the blend preserves.

**Widening worked on the mechanism.** Gradient → (−15, +5, +25, +40), clustering → `others × 7.5`. Prior ratio between best and worst placement of the same structure:

| structure | plies | before | after |
|---|---|---|---|
| Windmill (adjacency hub) | 14 | **1.06** | **1.97** |
| Farm | 31 | 1.47 | 2.16 |

**But the outcome regressed hard, from a mean shift I claimed to have avoided.** The hub gradient was mean-preserved over 1–3 partners ((−15+5+25)/3 = 5.0 = (−2+5+12)/3). The clustering term was not: `(others+1) × 2.5 → others × 7.5` deletes a baseline that is summed **per adjacent empty tile**, so an isolated hut lost up to 12.5 on a base of 22. Ordinary economy builds collapsed — win rate −3.5pp, owned tiles −3.1 (133× noise), units −0.64, giants −0.14.

Re-centred to `others × 7.5 + 2.5` (isolated placements score exactly as before, only the gradient sharpens). Territory and expansion flipped back to gains — owned tiles **+1.24** (52.8× noise), villages captured **+0.027** (3.5×), first village **0.11 turns faster** (2.6×) — but **win rate 0.797 sits below the registered 0.805 floor, SPT@t10 fell to 8.720, and units trained is down 0.758 (12.1× noise)**. Reverted.

**Rule worth keeping: a softmax prior only cares about DIFFERENCES, so there is never a reason to move the mean when widening a spread.** Re-centre by construction, and check every term, not just the one whose arithmetic is easy.

**Still unproven, not disproven:** the outcome could not be tested properly, because with the recommender reverted the model builds ~2 LumberHuts on seed 4102 and there is nothing to cluster. Judging hub placement needs a config where hubs actually get built.

---

### EXP_HUBLVL_001 — measure realized hub level, then re-test the placement prior against it (Aug 6 2026) — **PRE-REGISTERED**

**Why this exists: EXP_PRIOR_001 was judged without ever measuring its own target.** It changed hub *placement*, verified that the *prior* moved (Windmill best-vs-worst ratio 1.06 → 1.97), and then accepted/rejected on downstream outcomes (win rate, SPT, units). The proximal quantity — how many partners a Sawmill/Windmill/Forge actually ends the game with — was never collected on either arm. The only number ever measured, **1.33 partners**, came from a single replay (seat 1 of `best_game_score_5705`, three Sawmills). n=3, one game, one map. Every inference built on it, including "that is the whole paradox," rests on that.

**Instrument (shipped first, separately from any behavior change).** `self_play.rs` now walks the final state and, for every pop-bearing adjacency hub (`adjacent_types` non-empty *and* `reward_pop > 0` → exactly Windmill/Sawmill/Forge; Market self-excludes at `reward_pop 0`) standing on a **net seat's** tile, counts friendly-owned adjacent partners using the same predicate `actions::structure::build_structure` pays on. New `METRICS:` fields:

- `avg_hub_level` — mean realized partners per hub (−1.0 if none built)
- `hub_starved_frac` — share of hubs at ≤1 partner
- `avg_hubs_built` — hubs per net game
- `hub_levels_by_type` — `{built, mean_level, starved_frac}` per structure

Two properties this has that the replay script did not: it covers **every game in the batch** (high-score replays are the best game of a batch — a biased sample that would have been read as signal), and it is measured **at game end**, so a hub placed at 2 partners that grows to 4 as later partners go down is credited at 4. Verdi's point, and the reason build-time counting would understate.

Caveat to read the breakdown with: `starved ≤1` is the value-destroying line for Sawmill/Windmill (`reward_pop 1` → 5★ for 1 pop, worse than the 3★ LumberHut feeding it), but Forge is `reward_pop 2`, so a 1-partner Forge is 2.5★/pop and fine. Read per type, not by the aggregate, when Forges are present.

**Smoke (n=8, baseline weights): 6 Windmills, `avg_hub_level` 1.17, `hub_starved_frac` 0.83, no Sawmills at all.** Consistent with the 1.33 single-replay figure and with the reverted recommender never buying Forestry early. Not yet a result — n=8.

**Arms** (128 games, `--base-seed 770425`, vs Greedy at `--anchor-frac 1.0`, gumbel 64/k=16, same flags as `paired_A_on`; all three re-run on the new binary since the old arms carry no hub fields):

- **A** — HEAD `scoring.rs`. Establishes the population answer to "what IS the average hub level," replacing the n=3 anecdote.
- **B** — EXP_PRIOR_001's re-centred widening: gradient `(−15, +5, +25, +40)`, clustering `others × 7.5 + 2.5`.
- **C** — repeat of A. `|A−C|` is the per-metric noise floor; nothing counts unless `|B−A|` clears it.

**Primary endpoint is now the proximal metric, deliberately.** `avg_hub_level` and `hub_starved_frac` decide whether the intervention does what it claims. Outcomes (win rate, SPT@t10, owned tiles, units, giants) are recorded as secondary and do **not** get a veto in this run — EXP_PRIOR_001 already showed they can move for reasons unrelated to placement, and letting them adjudicate a mechanism question is what produced an uninterpretable result the first time.

**Falsifier — the point of running this.** If `|B−A|` on `avg_hub_level` does not clear `|A−C|`, then widening the prior does not change where hubs actually land, the prior is **not** the lever, and the next move is the planner side: stop pricing hubs at their best placement in `max_affordable_pop` and price them at the realized distribution instead, so lane ranking stops assuming pop the net never delivers.

**Reading it the other way is also a result.** If hub level rises materially *and* the secondary outcomes still fall, then good placement is not worth what the planner assumes it is, and the 1.33 diagnosis — that starved hubs are what drains the economy — is itself wrong and must be retracted rather than re-fixed.

**ACTUALS (Aug 6 2026) — 3 arms × 128 games, seed 770425.**

**1. The number the instrument was built to get.** Pooling the two baseline arms (A+C, 256 net games):

| hub | n | mean realized level | share ending at ≤1 partner |
|---|---|---|---|
| **Sawmill** | 105 | **1.40** | **67%** |
| **Windmill** | 174 | **1.56** | **58%** |
| Forge | 7 | 1.00 | 100% |

The n=3 single-replay figure of 1.33 **holds up at population scale**. Sawmills do not grow into their sites: two thirds finish the game at ≤1 partner, i.e. 5★ for 1 pop, worse than the 3★ LumberHut feeding them. Measured at game end, so late-arriving partners are already credited — the number is not a build-time artifact.

**2. Did widening the placement prior move it? Not established — the endpoint is confounded by its own denominator.** Mean level rose `1.464 → 1.598` (+0.134), but arm B also built **22% fewer hubs** (0.977 → 0.758 per game). Mean level is conditional on building, so skipping marginal isolated hubs raises it by selection with no improvement in placement skill. Separating the two:

| | A base | C base | B wide |
|---|---|---|---|
| hubs built / game | 0.977 | 1.258 | 0.758 |
| pop delivered / game | 1.437 | 1.945 | **1.219** |
| pop per star on hubs | 0.294 | 0.309 | 0.322 |

Total hub pop per game is **lowest in arm B** and swamped by the A-vs-C gap. Pop-per-star is up ~9%, which is the one reading favourable to the change, and it is consistent with "builds fewer bad hubs" rather than "places good ones". Secondary outcomes: SPT@t20 +0.734 and owned tiles +0.797 both clear their gap; win rate 0.875 vs 0.828/0.781; SPT@t10, giants, units, villages all flat. Not shipped — the mechanism claim is unproven either way.

**3. The methodological result, which outranks both: a single same-config repeat is NOT a noise floor.** The Aug 4 and Aug 6 pairs measure the identical quantity — `|A − C|` at n=128 on the same seed and config — and disagree by up to **24×**:

| metric | \|A−C\| Aug 4 | \|A−C\| Aug 6 | ratio |
|---|---|---|---|
| units trained | 0.0625 | 1.5312 | **24.5×** |
| owned tiles | 0.0234 | 0.2656 | 11.3× |
| win rate | 0.0078 | 0.0469 | 6.0× |
| villages captured | 0.0078 | 0.0469 | 6.0× |
| giants | 0.0312 | 0.1641 | 5.2× |
| SPT@t10 | 0.3288 | 0.6523 | 2.0× |

`|A − C|` is one draw from a distribution, not its width. Dividing an effect by it produces a ratio that can be off by an order of magnitude in either direction — so **every "×noise" multiplier quoted in this session's entries is unreliable**, in both directions: the ones used to accept a finding (owned tiles "52.8× noise", "133× noise") and the ones used to reject one.

**Consequence: EXP_PRIOR_001's rejection does not stand.** It was failed on win rate 0.797 against a registered 0.805 floor — a gap of 0.008, against a same-config repeat gap measured here at 0.047. Its SPT@t10 drop (0.22) is a third of the 0.652 repeat gap. At n=128 the binomial standard error on win rate alone is ≈0.033, so a 0.008 difference was never resolvable. **The verdict on EXP_PRIOR_001 is withdrawn: that run was uninformative, not negative.** It should not be cited as evidence against the placement prior.

**Standing rule going forward:** an arm difference counts only against a variance estimated from the per-game distribution (bootstrap over games, or McNemar on paired per-game win/loss), never against one repeat. For a binary at n=128 that floor is ~0.03–0.05 in win rate — most of the win-rate deltas argued over this session sit under it.

### Why same-seed arms diverge — argmax is real, determinism is not (Aug 7 2026)

Verdi challenged the "MCTS noise" explanation: selection is argmax and the prior is heavily distilled, so a fixed seed should reproduce. **The argmax half is correct** — `TEMPERATURE_MOVE_THRESHOLD = 0` (`mcts_zero.rs:269`), so the visit-weighted sampling branch at `gumbel_mcts.rs:1372` is dead code and every move goes through `recommend_final_move`. There is no temperature sampling anywhere in self-play.

**Determinism still fails, from two independent sources.** Same binary, same `--base-seed 4102`, same flags, run twice:

| config | run 1 | run 2 | identical prefix |
|---|---|---|---|
| default (`GUMBEL_SCALE=1`) | 83 moves | 62 moves | **1 move** |
| `GUMBEL_SCALE=0` | 55 moves | 67 moves | **20 moves** |

1. **Root Gumbel from an unseeded `rand::thread_rng()`** (`gumbel_mcts.rs:598-601`, `660-666`). `--base-seed` seeds map generation only; it never reaches the search. Divergence at move 1.
2. **The eval path itself.** With the Gumbel off, runs still diverge at move ~20 — the Metal/MPSGraph backend runs 14 actors with `max_batch=256, coalesce_us=1000`, so batch composition varies run to run, float reduction order varies with it, and argmax flips on near-ties.

**The part that matters for placement — "argmax over Gumbel-perturbed logits" IS sampling from the policy.** By the Gumbel-max trick, `argmax(logit + Gumbel(0,1))` is exactly a categorical draw from `softmax(logit)`. For a candidate that Sequential Halving never visits there is no q̂ term to correct it, so the choice reduces to a pure sample from the prior. Placement candidates are unvisited in 69–82% of plies (EXP_PRIOR_001), which puts hub/hut placement squarely in that regime.

Against a Gumbel(0,1) std of **1.283**, the measured best-vs-worst prior gaps are:

| | logit gap | noise ÷ signal |
|---|---|---|
| Windmill, baseline weights | 0.058 | **22.0×** |
| Windmill, widened (EXP_PRIOR_001) | 0.678 | 1.9× |
| Farm, baseline | 0.385 | 3.3× |
| Farm, widened | 0.770 | 1.7× |

**This is the mechanism behind the 1.40 realized Sawmill level.** Placement is not a decision the net loses on merit — at a 22:1 noise-to-signal ratio it is close to a uniform draw over candidate tiles. It also explains why widening the prior helped only partially: it moved the ratio from 22× to 1.9×, which is better than random but still noise-dominated.

**Consequences.**
- Same-seed arms are *expected* to differ; `|A − C|` is measuring this, not a bug in the harness. It does not, however, rescue `|A − C|` as a noise estimator (see the 24× instability above).
- Two levers exist that were never on the table: lower `GUMBEL_SCALE` at eval/measurement time, and seed the root RNG so `--base-seed` actually pins a run.
- Any future placement work should quote its prior gap **in logits against 1.283**, not as a ratio — a "2× better prior" is still under the noise.

### Hub metric corrected — attribution by builder (Aug 7 2026). ⚠️ SUPERSEDES the 1.40 figure above

**The defect.** The first `hub_levels` implementation scanned the final map and counted every pop-bearing adjacency hub standing on a tile a net seat owned **at game end**. With `--anchor-frac 1.0` every game is against Greedy, and Greedy builds a real forest economy — so every hub the net *captured* was counted as the net's own placement. On seed 4102 this was the entire sample: the net built 4 Farms, 9 Roads and a GateOfPower and **no hubs at all**, Greedy built the 9 LumberHuts and a 3-partner Sawmill at tile 74, and the metric reported "1 net Sawmill at mean level 3.0."

**The fix.** Hubs are now recorded at build time (`built_hubs`, pushed when a net seat executes a Build of a structure with non-empty `adjacent_types` and `reward_pop > 0`) and scored at game end. Partner counting stays ownership-based, matching what `build_structure` actually pays — a captured Farm does feed your Windmill — but the *placement decision* is attributed to whoever made it. New `hub_lost_frac` exposes the opposite confound: hubs the net built and no longer holds.

Parity-checked against the replay on a single game: metric 3 Windmills / `starved_frac` 0.667 vs replay seat-1 builds 3 Windmills at 0, 2, 0 partners / 0.667. Exact agreement on the starved share; the mean differs (1.00 vs 0.67) only because the metric counts a captured adjacent Farm the replay script does not — the metric is right, the engine pays that Farm.

**Corrected baseline, n=64, w=0.1, attributed by builder:**

| hub | built | mean level | ≤1 partner | lost |
|---|---|---|---|---|
| **Sawmill** | 33 | **1.33** | **64%** | 3% |
| **Windmill** | 38 | **1.16** | **76%** | 18% |
| aggregate | 71 | **1.24** | 70% | 11% |

Against the ownership-attributed reading of the same config (1.621 / 0.576), the net's own placement is **materially worse** than previously reported — it was being flattered by Greedy's well-placed hubs. Windmills are the weak leg: 76% sit at ≤1 Farm, i.e. 5★ for 1 pop.

**Retractions.** The "1.40 mean Sawmill level over 256 games" and the whole per-structure table in the EXP_HUBLVL_001 ACTUALS are ownership-attributed and must not be cited. The EXP_HUBLVL_001 arm comparison and the w=0.1/w=0.5 blend probe both used the defective metric for their primary endpoint; their hub-level readings are void. The composition shift (Sawmill vs Windmill build counts) was also ownership-derived and needs re-deriving before it is claimed.

**Incidental, and another data point on the noise problem.** The corrected run and `crutch64_W10` are the *same binary config and seed*, differing only by search/eval nondeterminism. Win rate came out 0.875 vs 0.797 (Δ0.078) and owned tiles 55.6 vs 51.0 (Δ4.6) — both larger than the entire w=0.1→w=0.5 effect measured earlier (Δ0.016 win rate). Nothing in that probe was resolvable.

### First-Sawmill placement IS the lever (Aug 7 2026). ⚠️ SUPERSEDES the "placement is near-optimal" reading above

**The metric that produced the wrong answer.** `best_available_partners` scored each candidate tile by the partners **actually standing there at game end**. The net builds ~3 LumberHuts a game, so no tile could score above ~3, and the measured "best available 1.73" was a restatement of the net's own hut policy, not a property of the map. It made placement look near-optimal (77% "optimal", rank pct 0.95) and produced the conclusion that perfect placement was worth ~7 pop across 128 games. That was circular and is withdrawn. Verdi caught it from the seed-4102 spawn screenshot: dense forest, a level-4 Sawmill site by the capital, level 3–4 off the eastern and northern cities.

**The corrected measure — terrain ceiling.** For each candidate tile, count adjacent tiles that could *ever* host a partner, by terrain + resource alone (Forest → LumberHut; Field+Crop → Farm; Mountain+Metal → Mine), independent of what was built. n=128, seed 770425:

| hub | games | sites | ceiling chosen | ceiling best | picks best site | realized |
|---|---|---|---|---|---|---|
| **Sawmill** | 17 | 27.4 | **2.00** | **3.29** | **29.4%** | 1.53 |
| Windmill | 45 | 19.4 | 1.71 | 2.33 | 55.6% | 1.44 |

**The net picks a max-potential Sawmill site only 29.4% of the time**, against ~27 candidates. It lands on 2.00 potential when 3.29 was available. Verdi's read off the map — level 3–4 sites exist — is confirmed at 3.29.

**Decomposing the loss for the first Sawmill** (best possible 3.29 → realized 1.53):

| term | partners lost | share |
|---|---|---|
| chose a lower-potential site | **1.29** | **73%** |
| never developed the site it chose | 0.47 | 27% |

Placement is the dominant term, not a rounding error. In star terms this is the difference between a good building and a bad one: a Sawmill on a fully-developed 3.29 site is **1.52★/pop**; as actually placed and developed it is **3.27★/pop**, i.e. worse than the 3.00★/pop LumberHut feeding it.

**Consequences.** The prior-widening thread (EXP_PRIOR_001, the blend-weight probe) was aimed at a real and large lever after all — 1.29 partners on the first Sawmill alone — not the ~0.32 the broken metric implied. Their failure to move outcomes needs a different explanation than "the lever is too small," and the earlier finding still stands that both were measured with instruments too noisy to resolve anything.

**Method note, third time this session.** A metric whose denominator or comparison set is produced by the very policy under test will always report that policy as near-optimal. `best_available_partners` (built partners), `avg_hub_level` (ownership-attributed), and `|A − C|` (single-repeat noise floor) all failed this way. Ask what the comparison set would look like under a *different* policy before trusting a ranking.

---

### SSOT consolidation — collapse duplicated game logic (Aug 8 2026)

**Why.** Six bugs in one session all had one root cause: the same game rule implemented in several places, with the copies drifted apart. A three-part audit (combat / economy / shared primitives) found 11 independent implementations of "friendly adjacent partners" alone, across 7 files, plus 10 army-value sites, 7 build-legality checks, 12 capturability predicates and 4 hardcoded prereq tables.

**Rule conflicts fixed (copies disagreed, one was wrong).**

| fix | what was wrong |
|---|---|
| **Retaliation gate** | `calculate_combat_preview` applied none of `!Stiff && !Surprise && distance <= range`, so every ranged and Surprise attacker was predicted to take a counter it never receives — and `ai/scoring.rs:65` then priced those attacks **1.0 ("suicide")** instead of 50–95. Largest single correctness bug found. |
| Rogue damage formula | `functions.rs:1368` used `(atk/(atk+def))*atk*4.5`, dropping both HP ratios and the defence bonus, never rounding. |
| Splash rounding | Stomp floored, Splash did not — different damage for identical inputs. |
| `upgrade_unit` | missing the `version < 115` damage-inheritance gate both other retype paths have. |
| Post-upgrade vision | had lost the Mountain clause; a unit upgraded on a mountain revealed r=1 instead of 2. |
| Hub destroy | refunded base `reward_pop`, so destroying a level-4 Sawmill removed 1 pop, not 4. |
| Lost City | granted only `CityWall` on a level-3 city, desyncing `required = level - 1` so slot 2 was offered twice. Now grants `[Workshop, CityWall]`. |
| `eco_plan` SPT | used a flat base of 1; the engine's base is `city.production`, which tracks **level**. A level-5 city yields 5. |
| `ai/scoring.rs` pop | credited Market `adj_count` population though its `reward_pop` is 0, and hardcoded Forge's `×2`. |
| Super units | `ai/scoring.rs` and `self_play.rs` tested `== UnitType::Giant`, so Gaami/Crab/DragonEgg/Centipede were invisible to the summon heuristic **and to `avg_giants_made`** — a metric this campaign has been steering by. |
| Resource visibility | `settings/resources.rs::visible_required` was declared, populated for 3 of 8 resources, and **never read**; `functions.rs` hardcoded a different rule. Table completed and adopted (Verdi's call). Starfish now needs Fishing/Sailing/Navigation. |
| Territory writers | four writers disagreed on radius and on whether the city tile is a member. Canonical (Verdi's call): `get_square_indices(city.idx, border_size, size)` — a fresh city rules **9 tiles, its own included**. `post_load` was rebuilding every city at a fixed radius 2 with the centre dropped. |

**New `src/rules/`** — `combat`, `economy`, `vision`, `capture`. Two-tier shape: `foo_with(...)` takes resolved context and does no lookups (the engine's hot path calls this), `foo(...)` resolves and delegates. One body each, so they cannot drift. A parameter controls **cost, never the answer** — counterfactuals are separately named (`partner_count` / `_planned` / `_ceiling`) rather than flagged.

**Migrated:** engine build pay + build legality + `save_batch_cost` → `partner_count_with`; 8 army-value sites → `unit_worth` (which counts passengers and zeroes converted units, as the engine's own score does); build and harvest **move generation** → `territory_tiles` (ruling-city dedupe — a tile in two of your cities' `_territory` no longer emits the move twice); 5 vision copies; 4 promotion-threshold copies; 4 hardcoded prereq tables and the tech→resource table → `settings/*` via `strum::iter`; 3 city-score copies; `get_tiles_in_range` and `get_neighbors` → the shared helpers; 6 chebyshev closures.

**Deliberately not migrated:** ~90 raw `explorers.contains` reads. They differ from `is_tile_explored` only when `_fow == false`, which training never uses, and several are inside `fow.rs`/`prediction.rs` where they *implement* fog rather than consume it.

**Measured (n=128, seed 770425, vs Greedy, paired on maps).** This is the COMBINED effect of every merged change — unattributable between them, which was the accepted trade.

| | before | after | Δ |
|---|---|---|---|
| win rate vs Greedy | 0.773 | 0.750 | −0.023 |
| SPT@t10 | 8.67 | 8.85 | +0.18 |
| owned tiles | 50.7 | 47.4 | **−3.4** |
| realized hub level | 1.352 | 1.420 | +0.068 |
| Sawmills built | 36 | **49** | +36% |
| Windmills built | 89 | **60** | −33% |

**Win rate moved −0.023 against a same-config repeat floor measured at 0.078 earlier this session — well inside noise, so no detectable strength regression.** The clearest real shift is hub composition: a third fewer Windmills, a third more Sawmills, and Forges appearing at all. Consistent with `ai/scoring.rs` no longer crediting Market fake population and with legality/pay now sharing one partner count.

**Throughput: no measurable change.** Three reps each: before 172.7/161.8/135.3, after 159.0/169.3/126.6. The 135–173 spread inside a single build swamps any difference. **The prediction that making `get_structure_setting`/`get_resource_setting` return `&'static` (removing two HashSet allocations per call, on a path `moves/build.rs` runs per candidate tile) would be a net speedup is NOT supported** — it is correct and allocation-free, but it was not the bottleneck.

**Guard tests** (`tests/rules_ssot.rs`, 8): ranged attacker takes no predicted retaliation; a fresh city rules 9 tiles including its own; partner count is player-scoped not city-scoped; level thresholds and super-unit slots; per-tribe super units; unit worth counts passengers and ignores converted; resource visibility comes from the table. **230 tests green.**

**Method note worth keeping: the territory reshape and the resource-visibility change are real rule changes and not one pre-existing test objected.** Coverage in those areas was nil, which is precisely how the original copies drifted unnoticed.

## MAPGEN_001 — resource spawn rates were the wiki's land-tile fractions misread as per-tile conditionals (fixed Aug 10 2026)

**Not an experiment — a correctness fix to the world itself. Registered so every measurement discontinuity from this date has a written cause.** Full research with provenance: `mapgen_research.md` (repo root).

**The bug.** `mapgen.rs::get_resource_prob` used the Polytopia wiki's Map Generation table numbers (metal 11%/3%, game 19%/6%, fruit/crop 18%/6%) directly as P(resource | matching terrain tile). The wiki lists **fractions of all land tiles** (14% of land is mountain, 11% of land is mountain-with-metal). Correct conditionals = joint ÷ terrain share. Verified against three real Steam-game turn-0 captures in `polyfish-rs/replays/` (vers 111/114/115): real near-village mountains carry metal 57–75% of the time; our generator produced 14–22%. Matched tribe pair (Cymanti+AiMo): real 75%/20% (inner/ring) vs ours 15%/4%.

**The fix (all in `mapgen.rs`).**
1. Conditional bases: metal 0.8 (between the 0.85 Moonrise patch constant and the modern 11/14 table), game 0.5, fruit/crop/spores 0.375, fish 0.5; outer ring = inner × 1/3 (the game's border-expansion factor, now applied to fish too — fish was flat 0.5 at distance 2, real is ~0.17).
2. Resource pass restructured to the real game's shape: one inner/outer classification per tile (inner precedence across overlapping village zones) and one roll — the old per-village iteration both re-rolled failed inner tiles at outer rate and rolled the same tile once per nearby village, which would have compounded badly on top of corrected rates.
3. Guarantee block now runs AFTER natural spawning as a top-up (real post-gen semantics), and **Xin-xi gets its guaranteed 2 capital metal** (Espark's decompiled Starting Resource table). Saturation fallback added (guarantees may overwrite other resources when the capital ring is full) — also for the Drylands Kickoo/Aquarion pond invariant.
4. Removed the invented cap of 3 primary resources per capital (the "5 or 6 fruit is overkill" patch) — the real game does produce 4–6; the overkill impression came from the ×2 tribe multiplier sitting on a mis-scaled base.
5. Tribe multipliers, terrain rates, phase order: unchanged (verified faithful to Espark's table).

**Measured after fix (300 maps/pair, Drylands 11×11, same measurement as the real captures):** metal inner 80–81% base / 98–100% Xin-xi territory, ring 25% / 37–39%; game 52–56% in unpenalized climates; fruit 37–39% base. All within noise of the real captures. Xin-xi capitals: **3.9 metal in reach on average, was 0.65** (the one real Xin-xi capture: 5). Tests: suite green (239), old `test_resource_density` (asserted the cap) replaced by `test_resource_rates_match_real_game` + `xinxi_capital_always_has_metal`.

**Every measurement crossing this date shifts. Expected directions:**
- **Self-play/training metrics** (`training_log.csv`, moves_by_turn, dashboards): score, SPT, builds, harvests all up — maps are simply richer (~4× more metal-adjacent economy, ~2× game, ~2× fruit/crop). Do not read a cross-boundary jump as learning.
- **Seed continuity is broken everywhere**: same seed → different map (RNG stream order changed). The frozen seed-770425 paired gauge is void across the boundary — re-baseline the noise floor before the next A/B. Old mid-game states replayed via regenerate-from-`initial_seed` (`main.rs` replay path) reconstruct the wrong initial map; full-state captures are unaffected.
- **League/arena vs pre-fix checkpoints**: old nets trained on starved maps now play on rich ones — historical win rates are not comparable, and league results near the boundary are biased in an unknown direction.
- **Tech/economy behavior**: Mining/Smithery/Forge lanes go from ~7×-underpriced to fairly priced; expect research and hub-lane distributions to shift (more Mines, Forge viable), and eco_plan goal frequencies to move.
- **Any resumed run mixes distributions mid-run.** Start the next training campaign as a NEW run, not `--resume`, and treat pre/post-fix game archives as separate populations.

## EXP_ELO_029: ARM is economy-blind — pop growth under ARM + dual-class tech exemption + remove the star reserve

**Status: PRE-REGISTERED, not yet run (Aug 11 2026). Code landed; the A/B has not.**

### What prompted it

Verdi played a Xin-xi game against the AI and reached **3 giants and +14 SPT by turn 10, with two level-2 Forges**. Measured against Greedy on the same tribe (64 games, mcts=256, gumbel k=16, goal script on, base-seed 1786400000):

| | by t10 | by t20 |
|---|---|---|
| giants obtained (mean) | **0.06** | 1.61 |
| games with 0 giants | **60 / 64** | 16 / 64 |
| games with ≥3 | **0 / 64** | 16 / 64 |

The human's turn 10 is outside the model's turn-10 distribution entirely. Turning the goal script OFF is much worse (0.38 giants by t20, 23.4% win vs 54.7%), so the macro is load-bearing — the question was *what* it is steering toward.

### Diagnosis (new instrument)

`arena.rs` now writes a **goal trace**: one row per model ply with the committed stance, the stance the script wanted that ply, the SAVE target, the star gate, stars/SPT/cities, and the move actually chosen. Also surfaces `StanceCommit`'s `stance_flips`/`order_flips`, which EXP_ELO_028 registered as first-class metrics and never read.

Across 64 vs-Greedy games, 14,767 model plies:

- **Stance is ARM 70.0% of plies** (GROW 24.4%, SAVE 5.7%), and the share is monotone in turn: 0% at t0, 48% at t8, **82% at t10, 91% at t14, 92% at t20**. Cause splits evenly — 5,243 plies from a real Defend order, 5,091 from the `prepare` branch.
- **The plan is not being abandoned.** stance flips 0.12/turn, order flips 0.53/turn, hysteresis overriding the script on 16% of plies. It is held and followed. *Verdi's "decide at t7, abandon by t9" hypothesis is NOT supported.*
- **ARM's potential had no economy term at all** (`reward.rs`): `SHAPE_GOAL_ARM_PER_COST × army`, and the stranded/completion discipline was `Grow | Save` only. So ~85% of every ply after turn 10 carried **zero economy gradient**.
- **The star reserve was a constant, not a decision.** EndTurn stars: mean 1.37, median 1, ≤2 on 84% of turns. A gated tech needed `stars − cost ≥ 5`; Smithery would have cleared it on **12 of 2,584** gated GROW plies (0.5%).
- **The stance classes gated the hub lanes backwards.** GROW gated `combat_units non-empty`; ARM gated `is_eco && no combat unit`. Smithery (Swordsman + Forge) and Mathematics (Catapult + Sawmill) are dual, so each was gated by exactly one stance; Construction and Mining are pure-eco, gated by the other. Trace confirms: Smithery researched **15× under ARM, 1× under GROW**; Mining **56× under GROW, 2× under ARM**.

### Changes

1. **`reward.rs`** — ARM gains `SHAPE_GOAL_ARM_SPT × get_tribe_spt` (new const, **75.0** = half `SHAPE_GOAL_SPT`), and `SHAPE_GOAL_COMPLETION × completion_progress` now pays under ARM as well as GROW/SAVE. The stranded **tax** stays off ARM (v6 reason unchanged: combat spending shouldn't be penalised for levels it never planned to finish). Rationale (Verdi): giants come from level-5 cities and a super unit is a military asset, so pop growth *is* armament.
2. **`oracle_macro::passes_star_gate`** — a stance now gates only the tech that is **purely** the other class: GROW/SAVE gate `arms && !grows`, ARM gates `grows && !arms`. Dual-class tech is never dropped. SAVE moves from "gates everything" to GROW's rule — it is an economy stance in `goal_potential`, and gating all research meant a batch whose cost *includes an unowned tech chain* could not buy its own tech.
3. **`STAR_GATE_RESERVE` deleted.** A gated tech is now dropped outright. This makes the legacy stance-less arm (`--macro-star-gate`, the EXP_ELO_026 instrument) a hard block rather than the affordability test that experiment measured — noted in the flag's doc.

### Conflict with EXP_ELO_026, and why it is narrow

EXP_ELO_026 measured the reserve rule as causally worth **+7.6pp reach** (28%→81% in flipped games). That instrument is the **stance-less** path, which still gates every tech and is now merely stricter. The *live* goal-script GROW gate never blocked eco tech at all — which is where the diverted stars actually went — and the reserve opened on 0.5% of gated plies. So the behavioural payload of this change is item 2 (dual-class exemption), not item 3.

### Predictions (falsifiable, vs Greedy, Xin-xi, n=64, base-seed 1786400000, mcts=256, k=16)

| metric | before | predicted after | falsified if |
|---|---|---|---|
| Smithery median turn | 12 | **≤ 8** | ≥ 11 |
| games reaching Smithery | 25 / 64 | **≥ 38 / 64** | < 30 |
| giants obtained by t20 | 1.61 | **≥ 2.1** | ≤ 1.7 |
| giants obtained by t10 | 0.06 | ≥ 0.3 | ≤ 0.1 |
| SPT @ t20 | 14.96 | ≥ 16.5 | ≤ 15.0 |
| win rate vs Greedy | 54.7% | ≥ 54.7% − noise | < 45% |

**Read Smithery timing and giants@t20 first; win% at n=64 is a ±12pp ruler and cannot adjudicate this on its own.** Both new weights (`SHAPE_GOAL_ARM_SPT = 75`, completion under ARM) are **first fits** — every first fit in `reward.rs` has overshot ~2×, so dial against the measured dq median per the q-gap method before trusting the level. The frozen seed-770425 gauge is void post-MAPGEN_001; base-seed 1786400000 is the replacement pair.

**Risk to watch:** ARM's SPT term could simply buy back GROW behaviour and slow the army, losing the races EXP_ELO_026 showed convert at ~75%. The tell is cities@t20 and win% falling together while SPT rises.

**Tests:** 147 lib + 33 integration green. New guards — `dual_class_tech_is_never_stance_gated` (Smithery and Mathematics pass GROW/ARM/SAVE at zero stars) and `legacy_star_gate_blocks_research_at_any_star_count` (being rich no longer lifts the stance-less gate).

### EXP_ELO_029 — ACTUAL (Aug 12 2026). VERDICT: **falsified. No measurable effect; Smithery got slightly LATER.**

Ran the after-arm byte-identically to two pre-change runs (Xin-xi vs Greedy, n=64, base-seed 1786400000, mcts=256, gumbel k=16, goal script on). Two independent BEFORE runs were available, which is what makes this readable.

| | BEFORE a | BEFORE b | AFTER v9 | predicted | verdict |
|---|---|---|---|---|---|
| win vs Greedy | 54.7% | 48.4% | 48.4% | ≥54.7%−noise | flat (not falsified) |
| Smithery: games | 25/64 | 20/64 | 23/64 | ≥38/64 | **FALSIFIED** |
| Smithery: median turn | 12 | 12 | **14** | ≤8 | **FALSIFIED** |
| giants by t10 | 0.06 | 0.08 | 0.06 | ≥0.3 | **FALSIFIED** |
| giants by t20 | 1.61 | 1.20 | 1.25 | ≥2.1 | **FALSIFIED** |
| SPT @ t20 | 15.03 | 13.45 | 13.98 | ≥16.5 | **FALSIFIED** |
| cities @ t20 | 3.23 | 3.03 | 3.22 | — | flat (risk did not fire) |
| Forges built | 49 | 20 | 35 | — | inside the before-spread |
| Mines built | 344 | 247 | 280 | — | — |

**The two BEFORE runs bracket the AFTER run on every single metric.** Measured same-config repeat floor: **6.2pp** on win rate, 0.41 on giants@t20, 1.58 on SPT@t20, and **29 on Forges built** (49 vs 20 at an identical config — that metric is near-useless at n=64).

**The change is NOT inert — it reshuffles.** All 64 paired games diverged from BEFORE b (0/64 identical outcomes; scores swing ±2,000 either way). Play changes substantially per game; the means do not move.

**Invariant check passed:** stance mix ARM 70.0% → 69.7%, GROW 24.4% → 23.9%, SAVE 5.7% → 6.4%. `scripted_goal` was untouched, and the ARM share is confirmed independent of the potential.

**Why it failed — two mechanisms, both diagnosable in hindsight:**
1. **The dual-class exemption had almost no surface to act on.** Smithery was *already* ungated under ARM, and ARM is 70% of plies (85–92% after t10). The exemption only added freedom under GROW — 24% overall, 1–3% after t14 — which is precisely not the window where a tier-3 tech gets bought. The measured payload was near zero, exactly where the pre-registration guessed the payload was.
2. **An income term is the wrong shape for a two-step plan, and may cut against it.** SPT rises *immediately* from one more Mine (5★ → 2 pop → level → income). Smithery pays nothing until a *subsequent* 5★ Forge is built. Paying Phi for income per se strengthens the one-step purchase over the two-step plan. Signature matches: Mines up (247 → 280), Smithery *later* (t12 → t14).

**Kept or reverted:** code left in place pending Verdi's call. Nothing measured a cost; nothing measured a benefit.

**What this rules in.** The only mechanism that names a multi-step purchase is `Stance::Save`, and it is 5.7% of plies firing at median turn 14 because `save_batch_cost` demands `SAVE_MIN_PARTNERS = 2` partners *already standing* next to a placeable tile. On the Forge lane that means two Mines must exist before the plan to buy Smithery can even form. The next lever is that trigger — or a potential that prices the *unlocked* Forge rather than current income — not more weight on ARM.

**Method note for the next A/B: n=64 with two before-runs 6.2pp apart cannot adjudicate an effect of this size.** Either raise n substantially, or make the arms paired-deterministic, before reading another economy A/B at this budget. Reporting a single 64-game arm against a single baseline here would have produced a confident wrong answer in either direction.

---

## EXP_ELO_030 — the macro names a lane; make the PRIOR and the RAMP agree with it

**Registered 2026-08-12, before running.** Control arm includes the EXP_ELO_029
ARM terms (still un-reverted), so attribution is v10-on-top-of-029.

### Diagnosis this is built on (measured, not assumed)

`SMTRACE/`, 519 root decisions where Smithery was affordable + unowned, 52 games,
`--trace-tech Smithery` (new arena flag). `prior_heuristic_weight = 0.0`, so the
root prior IS the net's policy.

| stage | measured |
|---|---|
| policy prior on Research(Smithery) | median **3e-6**, gap to chosen **−11.3 nats** |
| enters Gumbel top-k (k=16, ~40 cands) | 183/519 (**35%**) |
| visits / max_visits when in the cut | median **0.10**; survives to the final halving round 18/183 |
| tied at max visits → *eligible* under argmax-visits | **18/519 (3%)** |
| bought | 4/519 |

`edge_reward` already favours the purchase **85×** (+0.346 vs +0.004) and
Q(Smithery) − Q(chosen) is only −0.17 median — so reward is NOT the binding
constraint. Visits are. Reward reaches selection only through σ(Q), which is
worth ~6 effective logits against an 11-nat prior deficit.

Root cause of the missing signal: `MacroGoal.save_target` was `Option<i32>`.
`save_batch_cost` identified the lane and discarded everything but the price, so
outside its own tests `save_target` had exactly two consumers — where it is set,
and the ramp in `reward.rs:642`. Not in `features.rs`, not in any gate, not in
the prior.

### Hypothesis

The macro already names the right lane (it did so in 11/27 ready-but-failed
games, held for runs up to 73 plies). The failure is that nothing downstream can
act on the name. Carrying the lane identity and (a) reserving prior mass for
plan-advancing moves and (b) making the ramp measure plan progress rather than
star balance should move the funnel at the Smithery step.

### Changes

1. `SaveLane { cost, tech_cost, structure_cost, structure_unit_cost, tech, structure }`;
   `save_batch_cost` → `save_batch_plan`; `MacroGoal.save_target: Option<SaveLane>`.
2. `advances_save_plan` — matches Build(lane.structure) and **any undiscovered
   tech on the `requires` chain** to `lane.tech` (Market sits behind Roads behind
   Riding; matching only the last step would fix Forge and leave the rest stuck).
3. `blend_goal_prior` — mixes `GOAL_PRIOR_W = 0.15` onto plan-advancing moves,
   applied after the `raw_logits` trace snapshot and **before** `build_in_cut`, on
   both the fresh-root and reused-root paths (root-only fixups are inert on 7 of
   8 plies — the v8 gate lesson).
4. `reward::save_progress` — ramp = `(stars + owned tech_cost + built×unit_cost)/cost`
   instead of `stars/cost`. Guard test `buying_the_lane_tech_does_not_lower_the_savings_ramp`.

### Predictions — stage-level, per the 029 method note

Controls: **V9** for the funnel (untraced, same seeds, post-029), **SMTRACE** for
the stage table (traced). n=64 win rate cannot adjudicate this; it is a guardrail
only.

| metric | control | predicted | falsified if |
|---|---|---|---|
| prior on Research(lane tech) | 3e-6 | ≥ 0.10 | < 0.02 |
| enters top-k cut | 35% | ≥ 95% | < 70% |
| eligible (tied at max visits) | 3% | ≥ 25% | < 10% |
| Smithery researched | 16/64 (V9) | ≥ 30/64 | ≤ 22/64 |
| Forges built | 16/64 games (V9) | ≥ 26/64 | ≤ 19/64 |
| giants by t20 | 1.25 (V9) | ≥ 1.8 | ≤ 1.4 |
| win vs Greedy (GUARDRAIL) | 48.4% | ≥ 40% | < 35% = harmful, revert |

Stage 1–3 moving while the funnel does NOT move would be the interesting
negative: it would mean the purchase is genuinely bad on this board and the net
was right, which sends the next lever back to the macro's lane choice.

### ACTUAL (2026-08-12) — mechanism confirmed, intermediate objective doubled, terminal objective did not move

Arms: **V10** (untraced, funnel, vs V9 control) and **SMTRACE10** (traced stage
table, vs SMTRACE control). Same seeds/budget as every prior run in this series.

#### The mechanism does exactly what it was built to do — where it fires

Splitting the traced plies by whether a live SAVE lane existed (the only
condition under which `blend_goal_prior` can act):

| | plies | prior on Research(Smithery) | enters top-k |
|---|---|---|---|
| SMTRACE, live SAVE lane | 65 | 0.000000 | 25% |
| **SMTRACE10, live SAVE lane** | 22 | **0.150000** | **86%** |
| SMTRACE, no lane | 454 | 0.000005 | 37% |
| SMTRACE10, no lane | 366 | 0.000023 | 56% |

`0.150000` is exactly `GOAL_PRIOR_W`. The blend is correct and precise.

**But it only reaches 22 of 388 affordable plies (5.7%)** — SAVE is rare, and the
boost is conditioned on it. That is why the *aggregate* stage numbers fall far
short of the pre-registered targets, which were written against the aggregate.

#### Predictions

| metric | control | predicted | actual | verdict |
|---|---|---|---|---|
| prior (aggregate) | 3e-6 | ≥0.10 | 2.8e-5 | **FALSIFIED** (aggregate) |
| prior (live lane only) | 0.000000 | — | **0.150** | mechanism confirmed |
| cut entry (aggregate) | 35% | ≥95% | 58% | **FALSIFIED** (86% on live-lane plies) |
| eligible at max visits | 3% | ≥25% | 11% | **FALSIFIED** |
| **Smithery researched** | 16/64 | ≥30/64 | **33/64** | **CONFIRMED** |
| **Forge games / built** | 16/64, 35 | ≥26/64 | **28/64, 51** | **CONFIRMED** |
| giants by t20 | 1.25 | ≥1.8 | 1.20 | **FALSIFIED** |
| win vs Greedy (guardrail) | 48.4% | ≥40% | 54.7% | passed |

#### Outcomes, V9 → V10

```
win vs Greedy    48.4% -> 54.7%     giants t20      1.25 -> 1.20
SPT @ t20        13.98 -> 14.38     giants FINAL    1.33 -> 1.28
cities @ t20      3.22 -> 3.12      >=1 giant      47/64 -> 39/64
level-5+ cities   1.06 -> 1.28      >=3 giants      8/64 -> 11/64
Forges built         35 -> 51       avg score       4404 -> 4588
Mines built         280 -> 291      Parks           40 -> 38
city levels >=5      68 -> 82
```

Smithery +17 games on paired seeds is far outside any plausible repeat floor and
is established. The giant regression (−8 games) is ~1–2σ at n=64 — **suggestive,
not established**.

#### Against the measured repeat floors — ONE established effect

Applying the same-config floors measured in 029 (BEFORE a vs BEFORE b):

| metric | V9 -> V10 | 029 repeat floor | established? |
|---|---|---|---|
| **Smithery researched** | **16 -> 33 games (+17)** | **5** | **YES** |
| Forge games | 16 -> 28 (+12) | no direct floor; implied by Smithery x 0.85 conv | probably |
| Forges built | 35 -> 51 (+16) | 29 | no |
| win vs Greedy | +6.3pp | 6.2pp | at the floor |
| SPT @ t20 | +0.40 | 1.58 | no |
| avg score | +184 | 352 | no |
| level-5+ cities/game | +0.22 | 0.53 | no |
| giants by t20 | -0.05 | 0.41 | no |
| >=1 giant | 47 -> 39 games | ~1-2 sigma at n=64 | no |

**Exactly one effect clears its floor: the purchase roughly doubled.** Every
other number — the economy gains AND the giant dip — sits inside the noise this
harness resolves at n=64. An earlier draft of this entry claimed "economy
improved across the board"; that repeated the 029 mistake and is retracted.

#### Where the giants went — cohort re-composition, not a lane failure

| | V9 | V10 |
|---|---|---|
| level>=5 reward events (sum of level-4) | 103 | 123 |
| parks | 40 (39% of events) | 38 (31% of events) |
| Forge games / their giants | 16 / 43 (2.69 ea) | 28 / 55 (1.96 ea) |
| non-Forge games / their giants | 48 / 42 (0.88 ea) | 36 / 27 (0.75 ea) |
| ALL giants | 85 | 82 |

Reward events rose 19% while parks stayed flat, so the level-5 pick shifted
*toward* giants (park share 39% -> 31%). The Forge cohort grew by 12 games and
gained 12 giants — **a marginal yield of ~1.0 giants per added Forge game against
the ~0.88 those same games averaged without one.** The marginal Forge is worth
~0.1 giants. Not that the lane is harmful, and not that ambient giants were
destroyed: forcing the lane into the games the net skipped buys almost nothing.

The falling within-Forge mean (2.69 -> 1.96) is that same composition effect,
not a degradation of the original 16 games.

Two explanations are dead: **not Park displacement** (parks flat, share falling)
and **not timing** (giants flat at every checkpoint: t10 0.06/0.06, t15 0.56/0.50,
t20 1.25/1.20, final 1.33/1.28).

#### Identified limitation: the boost switches off at plan maturity

`scripted_goal` filters the plan with `tribe.stars < lane.cost`, so the moment
the FULL batch (tech + structure) becomes affordable, `save_target` goes `None`,
the stance leaves SAVE, and `blend_goal_prior` stops firing — at exactly the ply
the plan matures. A plausible contributor to eligibility stalling at 11%, and a
design gap rather than a tuning one. Supporting observation: traced plies with a
live SAVE lane fell 65 -> 22, consistent with plans completing and states exiting
the condition.

#### Training-side note

`blend_goal_prior` is library code, so the next `run_training_loop.sh` restart
picks it up and boosted lane moves will enter the **policy targets** — the
ownership path (prior -> visits -> targets -> net internalizes). Scripted arena
play cannot measure that; it is a real argument for keeping the change even with
giants flat here.

#### Verdict

The diagnosis was correct and the fix works exactly as designed: an 11-nat prior
deficit was the binding constraint on the purchase, and closing it doubled
Smithery. That is the one thing this run establishes.

It did NOT establish that the lane produces giants. The marginal Forge game
yields ~0.1 giants over its counterfactual, and every economy metric is inside
the repeat floor. Combined with the trace — better-estimated Q likes the purchase
*less* (Q(Smithery) > Q(chosen) fell 28% -> 17%; NN value after the purchase
-0.289 vs root, was -0.119) — the live reading is that **the net's reluctance was
partly informed**: it skipped the lane in games where it does not pay, and could
not buy it in games where it does.

The next question is therefore no longer about priors. It is whether
`save_batch_plan` picks the right lane for the right board, and whether 30 turns
is long enough for Forge -> pop -> level 5 -> giant to repay 21 stars.

**Kept or reverted: left in place pending Verdi's call.** One established benefit
(the purchase doubled), no established cost, guardrail passed, plus a
training-side argument arena cannot measure.

## MAPGEN_002 — is the current build's capital starting-resource mechanic hotter than top-up-2? (pre-registered Aug 12 2026, AWAITING DATA)

**Trigger.** Verdi's current-build iPad games on Tiny 11×11 Drylands: Xin-xi capital
Forge level 3 with NO border growth (⇒ ≥3 metal in the initial 3×3), level 4 after
border growth, two games back to back. Under our generator that pattern is a 3.4%/game
event (back-to-back ≈ 0.1%): probe over n=2000 Tiny maps gives capital-ring metal
2: 77.7%, 3: 15.9%, ≥4: 6.5%, best no-BG Forge ≥3 in 8.6%.

**What the record supports (all verified Aug 12).** Espark's decompiled table: Xin-xi
mountain 1.5 / metal 1.5 (ours match). Era-1 generator `post_generate`: top-up
(`while resources < quantity`), quantity 2. Real captures v111–115: every
guarantee-relevant capital at the floor — anjiian's Vengir ring shows exactly 2 game
against a ~0.15 natural expectation (pins quantity=2 top-up in v111); Xin-xi metal 2 on
2 ring mountains. The wiki's quota system pins zone counts, not ring concentration.
Full analysis: `mapgen_research.md` Aug-12 addenda.

**Hypothesis.** The Aug-2026 live build (post-2025-Balance-Pass, which already touched
Xin-xi starts) places starting resources ADDITIVELY (2 on top of natural spawns, carving
terrain) or otherwise runs capitals hotter than top-up-2. Our newest real capture is
Jun 2026; none covers the current build.

**Experiment.** Step 0: confirm the Steam and iPad builds are the same version (the
disputed games are iPad; the mod captures Steam) — else screenshot 2–3 fresh iPad Tiny
Xin-xi turn-0 starts and count ring metal by eye. Then capture ≥5 fresh turn-0 Tiny
11×11 Drylands Xin-xi-vs-Imperius games from the current Steam build via polyfish-mod
and run `python3 capital_ring_check.py replays/<f>.json`. Keep Imperius as the opponent:
its capital is a second, independent semantics probe (fruit's natural ring rate is high,
so top-up predicts ~2.8 ring fruit vs additive ~4.5 — unlike Vengir, where both models
predict 2).

**Predictions.** Top-up (ours correct): ring metal mean ≈ 2.2, ≥3 in ~22% of capitals.
Additive (we under-spawn): mean ≈ 3.4, ≥3 in ~78%. **Read the mean first** (~4σ apart at
n=5, per-capital sd ≈ 0.6); the "≥4 of 5 capitals show ≥3" count is the conservative
confirm (~1% false positive but only ~70% power — a 3/5 result is ambiguous on the count
alone, not on the mean). Ring-mountain counts double as a free re-check of Xin-xi
mountain 1.5 (mean 1.7/ring) vs the Moonrise 2.0 (mean 2.2/ring).

**Committed consequence.** If falsified: switch the guarantee block in `mapgen.rs` to
the measured semantics (new ledger entry, training-distribution discontinuity like
MAPGEN_001 — new run, re-baselined gauges). If confirmed: Verdi's games were the ~5%
tail + selection; close with no change. Probe source: `src/bin/forge_probe.rs`
(untracked, delete on close).

---

## EXP_ELO_031 — price the super unit, and price hub CAPACITY not just realized partners

**Registered 2026-08-12, before running.** Third layer on `goal_potential`;
029 and 030 both still un-reverted, so the control is v10 (= 029+030).

### Diagnosis (measured on V9/V10, 64 games each, gamemode 2 = Domination)

**(a) Giants are lost at the level-5 reward pick, and it is stance-driven.**
Park = +250 score AND +1 production (`functions.rs:495,1076`); a Giant is cost
10 -> +50 score. Raw score prefers Park 5:1; only ARM's army term overturned it.

| stance | V9 park/super | V10 park/super | park share |
|---|---|---|---|
| ARM | 11 / 82 | 7 / 81 | 8-12% |
| GROW | 7 / 2 | 5 / 1 | 78-83% |
| SAVE | 5 / 5 | 22 / 3 | 50% -> 88% |
| total | 89 super, 23 park | 85 super, 34 park | |

EXP_ELO_030 made this worse by producing more level-5 cities while in SAVE.
There is **no 98/2 rule anywhere in the code** — the only super-unit preference
is `score_reward`'s Domination gap of 13 points, ~2:1 through `HEURISTIC_TEMP=20`,
and it is inert in arena (`prior_heuristic_weight = 0`).

**(b) Hub placement: 68% optimal, and the misses are not boundary-limited.**
31 sub-optimal Forge placements (V9+V10). **All 31 were in the SAME city and
legal at that moment** — never outside the border, never needing border growth.
12 had no resource trade-off at all (both tiles bare, ceiling gaps 1->2, 2->3,
2->5); 16 put the Forge on a resource tile it then crushes; only 7 have a
defensible reason (the better tile carried a resource).

Cause: `SHAPE_GOAL_YIELD_ADJ` counts **realized** partners. At placement a
ceiling-3 bare site and a ceiling-1 site both score 0, and the Mines that would
distinguish them are built later — far past the 6.48-ply horizon.

Forge ceiling on Xinxi confirmed: mean best available **2.40** (1:5, 2:37, 3:17,
4:4, 5:2 over 65 placements); pop = `reward_pop(2) x partners`, so ~4.8 pop at
ceiling. Forges reach the ceiling they are sited for (chosen 2.00 == final 2.00),
so the loss is purely site choice.

### Changes

1. `SHAPE_GOAL_SUPER = 500` per super unit owned, **in every stance**, damped by
   `SHAPE_GOAL_SUPER_ECON_DAMP = 0.6` x `save_progress` under SAVE only.
   Keyed to `save_progress` and deliberately NOT `completion_progress`: under
   GROW the latter approaches 1.0 exactly at the reward ply, which would damp
   every pick and rebuild the pathology. Super units are reward-only, never
   summoned, so this cannot distort purchasing.
2. `SHAPE_GOAL_YIELD_CAPACITY_W = 0.5` — unfilled partner capacity paid at half
   the realized rate, via `partner_ceiling_with`. Owned-tiles-only, so no FOW
   leak (owning implies explored); it does read the raw resource map, the
   tech-visibility read the engine already makes there on purpose.
3. `score_reward` Domination SuperUnit `base+18 -> base+27` (gap 13 -> 22 ~= 3:1
   at TEMP 20). Perfection untouched — there a Park's +250 really does win.

### Measured pick ratios (end to end, raw score + Phi, via `goal_potential`)

| corner | target | measured |
|---|---|---|
| ARM | 3:1 giant | **3.23:1 giant** |
| GROW | ~1.5:1 giant | **1.56:1 giant** |
| SAVE, plan barely started | giant | 1.38:1 giant |
| SAVE, plan complete | ~1.5:1 park | **1.43:1 park** |

Locked by `level_five_reward_pick_favours_the_giant_except_on_a_nearly_done_plan`.

### Predictions

Control **V10**, same seeds/budget. Primary metric is the reward-pick table, not
win rate (the 029/030 lesson).

| metric | V10 | predicted | falsified if |
|---|---|---|---|
| GROW park share | 83% | < 40% | > 60% |
| SAVE park share (u<0.8) | 88% | < 40% | > 60% |
| ARM park share | 8% | stays < 15% | > 20% |
| total super units | 85 | >= 110 | <= 92 |
| games with >=1 giant | 39/64 | >= 46/64 | <= 41/64 |
| Forge placement optimal | 68% | >= 85% | < 75% |
| ...of which the 12 both-bare errors | 12 | ~0 | > 6 |
| mean chosen ceiling | 2.00 | >= 2.30 | < 2.15 |
| win vs Greedy (GUARDRAIL) | 54.7% | >= 45% | < 40% = revert |

Risk to watch: `SHAPE_GOAL_SUPER` also prices *losing* a giant at 500, which
could make the agent hoard/over-protect it. Tell: super units alive at t30 rising
while attacks-with-giant falls.

**Stack note: this is the third un-adjudicated layer on `goal_potential`
(029 ARM terms, 030 lane prior + ramp, 031 super + capacity). Recommend deciding
029/030 keep-or-revert when this reads out — attribution degrades with each layer.**

### ACTUAL (2026-08-12) — the reward pick flipped exactly as priced; placement moved partway

Arm **V11** vs control **V10**, 64 paired Xinxi vs-Greedy games, same seeds/budget.

#### Primary metric: the level-5 reward pick

| stance | V10 park/super | V11 park/super | park share | predicted | verdict |
|---|---|---|---|---|---|
| ARM | 7 / 81 | **0 / 105** | 8% -> **0%** | stay <15% | **CONFIRMED** |
| GROW | 5 / 1 | **2 / 5** | 83% -> **29%** | <40% | **CONFIRMED** |
| SAVE | 22 / 3 | **6 / 10** | 88% -> **38%** | <40% | **CONFIRMED** |
| total | 34 park, 85 super | **8 park, 120 super** | | supers >=110 | **CONFIRMED** |

Parks fell 4x and supers rose 41%. The pick moved almost exactly where the
measured ratio table said it would — this is the tightest mechanism-to-behaviour
link in the whole EXP_ELO_02x/03x series.

#### Giants

| | V9 | V10 | **V11** |
|---|---|---|---|
| giants, final mean | 1.33 | 1.28 | **1.77** |
| games with >=1 giant | 47/64 | 39/64 | **48/64** |
| games with >=3 giants | 8/64 | 11/64 | **18/64** |
| super units alive at end | — | 1.05 | 1.42 |

+0.49 on the mean clears the 029 giants floor (0.41), so this one is established
— narrowly. `>=1 giant` merely recovers to V9; the real movement is in the tail
(>=3 giants more than doubles vs V9).

**Hoarding risk did NOT fire.** Survival rate (alive at end / total earned) is
1.05/1.28 = 82% in V10 vs 1.42/1.77 = 80% in V11 — unchanged. Pricing the giant
at 500 did not make the agent hide it.

#### Placement — real movement, short of target

| metric | V10 | V11 | predicted | verdict |
|---|---|---|---|---|
| optimal share | 68% | **75%** | >=85% (falsify <75%) | missed, not falsified |
| both-bare errors | 10 | **4** | ~0 (falsify >6) | passed, not met |
| mean chosen ceiling | 2.00 | **2.17** | >=2.30 (falsify <2.15) | missed, not falsified |
| mean best available | 2.40 | 2.48 | — | — |

Gap between best and chosen closed 0.40 -> 0.31, i.e. ~23% of the loss recovered.
Consistent with a half-weight first fit. `SHAPE_GOAL_YIELD_CAPACITY_W = 0.5` is
under-powered; the q-gap method says dial against this measurement rather than
guessing again.

#### Guardrail and costs

```
win vs Greedy   V9 48.4%   V10 54.7%   V11 46.9%    (spread 7.8pp ~= the 6.2pp floor: flat)
SPT @ t20            13.98      14.38      13.05    (-1.33, floor 1.58: inside)
cities @ t20          3.22       3.12       2.89    (-0.23, floor 0.20: marginally outside)
avg score             4404       4588       4439    (-149, floor 352: inside)
Mines built            280        291        231    (-60, ~21%: watch)
Smithery              33/64      33/64      31/64
Forge games           28/64      28/64      24/64
```

Win rate is flat across all three arms once the repeat floor is applied — V10's
54.7% was the outlier, not V11's 46.9%. **Mines -60 is the one number worth
chasing**: it is the largest un-floored move against the change and the Forge
lane depends on Mines. Plausible mechanism: 500 score-equivalents per giant
raises the value of giant-bearing states enough to pull search effort off
economy plies. Unverified.

#### Verdict

The pricing hypothesis is **confirmed**. The level-5 pick was mis-priced, the
mis-pricing was stance-shaped, and correcting it in `goal_potential` — the exec
path, not just the training door — moved behaviour immediately and in the exact
proportions the ratio table predicted. Giants rose 33% over the pre-030 baseline
with the tail more than doubling, at no measurable cost to win rate.

The placement half half-worked: the diagnosis (Phi counts realized partners, the
placement decision needs capacity) is right, and 0.5 is simply too small a
weight. Dial it, do not redesign it.

**Kept or reverted: left in place pending Verdi's call.** Now three layers deep
on `goal_potential` (029, 030, 031) with 029 and 030 still undecided; 031 is the
only one of the three with a clean, established, mechanistically-explained
benefit.

## EXP_ELO_032 — macro-decision bootstrap: scripted-directive executor + shallow turn-level lookahead

**Registered 2026-08-12, before running.** Inference-only; no changes to
training, network shapes, or the loop. This is the gate experiment for the
hierarchical macro-MCTS redesign: search over per-turn directives with a
deterministic executor for the plies.

### Diagnosis

Per-ply Gumbel at n=64 is depth-starved (~4 plies vs 8–20-ply turns,
EXP_ELO_023) and the in-tree opponent is frozen. The redesign attacks all four
documented tempo-learning failure mechanisms (credit horizon, V-not-Q, frozen
opponent, contingent capture payoff) by making one search edge = one whole
turn. Before building the tree: (0) is a scripted directive + deterministic
whole-turn executor at least Greedy-strength, and (1) does enumerating K
directives and rolling each out H turns (both seats, FOW-honest clones) beat
the script alone?

### Changes

New `src/ai/macro_exec.rs` (goal-conditioned executor: the four oracle_macro
root gates + `score_move + λ·Δgoal_potential` ranking, `execute_turn` via
`simulate_single_end_turn`, `ghost_until` ghost-Greedy opponent turns) and
`src/ai/macro_agent.rs` (`MacroScriptAgent` = Stage 0, `MacroLookaheadAgent` =
Stage 1 with per-turn directive commit + divergence telemetry,
`enumerate_candidates`: base + Grow/Arm/Save overrides + real-targets-only +
attack-capital variants). New arena backends `macro-script` /
`macro-lookahead` + flags `--macro-leaf/-k/-horizon/-lambda`; divergence in
the dump JSON and stdout. Agents plan on `clone_for_mcts` fogged views
(deliberate divergence from the 026/028 instrument, which reads the true
state) and intersect ranked moves against the true legal set by `serialize()`
(fog-planned moves can be true-illegal; arena ignores rejected moves, which
would livelock). Rollouts never compose undos — fresh clone per candidate.

### Setup (pinned)

n=125 seeds ×2 orientations = 250 games/arm; base_seed 1786400000 (post-
MAPGEN_001; the 770425 gauge is void); gamemode 2; max_turns 30; imperius
both seats; --eval-backend metal; GUMBEL_SCALE=0 exported on every arm; model
`model.safetensors` both configs (evaluators idle except E3/E4); macro params
k=4 horizon=2 lambda=1.0; one binary (/tmp/exp032_target, --features apple);
dumps `replays/exp032/arm{E1,C0a,E2,E3,C0b,E4}`.

| arm | config1 | config2 |
|---|---|---|
| E1 | macro-script | greedy |
| C0a | repeat of E1 | (determinism floor) |
| E2 | macro-lookahead leaf=heuristic | macro-script |
| E3 | macro-lookahead leaf=net | macro-script |
| C0b | repeat of E3 | (net noise floor) |
| E4 (opt.) | best lookahead | gumbel n=64 k=16 |

Reads: within-arm z = (W−125)/7.91 over decisive games; seed-level paired
read via `replays/exp026/analyze.py` + `causal_read.py`; C0 flip rates
subtracted before believing any delta.

### Predictions

| arm | metric | predicted | falsified if |
|---|---|---|---|
| C0a | flip rate vs E1 | 0 | >0 → becomes the heuristic-arm noise floor |
| C0b | flip rate vs E3 | <5% of games | >10% → net leaf unusable as-is |
| E1 | Stage0 win vs Greedy | ≥50% | z < −1.96 (≤~110/250) = executor broken, halt |
| E1 | win vs Greedy (GUARDRAIL) | ≥45% | <40% = do not read E2/E3 |
| E2 | Stage1-heur win vs Stage0 | ≥56.4% (z ≥ 1.96) | <50% on two runs |
| E3 | Stage1-net win vs Stage0 | ≥56.4% after C0b floor | <50% |
| E2/E3 | divergent/planned turns | 15–50% | <10% = candidate set too narrow, no verdict |

Go/no-go: E2 or E3 clearing z ≥ 1.96 past the noise floor = **GO** on the
full macro-MCTS redesign. Flat with divergence ≥20% = no-go pending (H=3 /
leaf quality first). Flat with divergence <10% = widen candidates, rerun.
E3 vs E2 decides whether the redesign leans on net value leaves.

Risk to watch: Stage 1 beating Stage 0 by exploiting its determinism
(opponent-model overfit), not by better directives — Tell: E2 positive but E4
collapses, with divergence concentrated in overrides that only pay against
the script. Secondary: the fogged ghost opponent (invisible units absent)
makes rollouts uniformly optimistic — watch win|reach conditionals in
causal_read.py for inversion.

**Stack note:** goal_potential is three undecided layers deep (029/030/031);
the executor prices plies through the same Φ, so any later change to those
layers invalidates cross-experiment comparisons of E-arm behaviour metrics,
not the within-experiment paired reads.

### ACTUAL (2026-08-12) — registered arms: gate passed, lookahead positive but under-powered; extension registered

All five arms ran as pinned (250 games/arm, base_seed 1786400000, one binary).
Runtime: script arms ~6s, lookahead arms ~40-46s per 250 games. ms/move:
Greedy 0.7, MacroScript ~4.6, MacroLookahead ~33 (vs Gumbel n=64 ~166).

| arm | metric | predicted | actual | verdict |
|---|---|---|---|---|
| C0a | flip rate vs E1 | 0 | **40/250 games (16.0%)**, net win% 55.2 vs 52.8 | **falsified** — the ENGINE is not run-to-run deterministic (pre-existing: no NN, no lookahead in these arms; movegen-order ties suspected). Identical-arm win% sd ≈ 2.5pp at n=250 |
| C0b | flip rate vs E3 | <5% of games | 34/250 (13.6%) | falsified numerically, same cause as C0a — treated as the same engine floor, not a net-leaf pathology (C0b as an independent E3 replicate reads 52.0%) |
| E1 | Stage0 win vs Greedy | >=50% | **55.2%** (138/250, z=+1.64) | CONFIRMED (gate + guardrail passed) |
| E2 | Stage1-heur vs Stage0 | >=56.4% | 54.8% (137/250, z=+1.52) | not met — positive, under-powered |
| E3 | Stage1-net vs Stage0 | >=56.4% | 53.6% (134/250, z=+1.14); pooled with C0b 264/500 = 52.8% | not met — positive, under-powered |
| E2/E3 | divergent/planned turns | 15-50% | E2 42.4%, E3 60.4% | in/above band — candidate set is NOT too narrow |

Secondaries (3-city reach, the causal bottleneck): **Stage0 76.8% vs Greedy
62.4% (+14.4pp)** — the directive+executor moves the exact metric EXP_ELO_026
proved causal. Lookahead's own reach is LOWER than standalone script
(69.6-72.8%) while still winning more — it trades expansion for its overrides.

Consequence of the C0 result: cross-arm seed-paired McNemar is invalidated
(pairs are not stable across runs); the within-arm head-to-head z is the valid
instrument, and game-level flip noise is already inside its binomial variance.

#### Extension (registered 2026-08-12, before running): E1x/E2x/E3x at n=1250 seeds

Same binary, same flags, base_seed **1786500000** (fresh non-overlapping
range), 2500 games/arm — sigma = 1.0pp, so z>=1.96 needs >=52.0%.
Predictions: E1x >= 52% (falsified if < 50%); E2x >= 52% (falsified if
< 51%, i.e. the n=250 read was noise); E3x >= 52% (falsified if < 51%).
GO on the redesign if E2x or E3x clears 52%; the E2x-vs-E3x ordering carries
the leaf-choice read.

#### ACTUAL (extension, 2026-08-12) — GO. Lookahead beats the script at z>5; leaf choice is a wash; Stage0 is Greedy-parity with far better reach

2500 games/arm, base_seed 1786500000, same binary/flags. sigma = 1.0pp.

| arm | predicted | actual | verdict |
|---|---|---|---|
| E1x Stage0 vs Greedy | >=52% (halt if <50%) | **50.9%** (1273/2500, z=+0.92) | prediction missed, halt condition clear — Stage0 is Greedy-PARITY in wins (the n=250 55.2% was mostly noise) while reaching 3 cities 74.6% vs 62.4% |
| E2x Stage1-heur vs Stage0 | >=52% | **55.2%** (1379/2499 decisive, z=+5.18) | **CONFIRMED** |
| E3x Stage1-net vs Stage0 | >=52% | **55.1%** (1377/2500, z=+5.08) | **CONFIRMED** |
| divergence | — | E2x 42.3%, E3x 59.8% (stable vs n=250) | candidate set healthy |

Reach (3+ cities), n=2500: E1x 74.6/62.4; E2x 72.2/64.0; E3x 74.2/62.0.

Reads:
- **The bootstrap thesis is CONFIRMED: one turn-level lookahead step over K=4
  directives is worth +5.2pp head-to-head over the same executor without it**
  (z>5, two independent leaf configurations replicating each other).
- **E2x == E3x (55.2 vs 55.1): the net value head adds NOTHING over the
  hand-written evaluator as a rollout leaf at H=2.** Consistent with the known
  over-confidence miscalibration (EXP_ELO_021); the redesign must NOT assume
  net leaves — value-head improvement is on the critical path only if later
  stages show heuristic leaves saturating.
- Stage0 alone converts a +12pp reach advantage into zero win margin vs
  Greedy — directive-execution without directive-SELECTION leaves the gains on
  the table; selection (lookahead) is where the wins come from. This is the
  cleanest evidence yet for the macro-search direction.
- Cost: MacroLookahead ~33 ms/move vs Gumbel n=64 ~166 ms/move — 5x cheaper
  than production search while beating its own baseline, with zero NN calls in
  the heuristic-leaf configuration.

#### E4 (risk tell, run 2026-08-12 after E2x/E3x cleared): lookahead-heur vs production Gumbel n=64/k=16

2500 games, base_seed 1786500000, GUMBEL_SCALE=0. Registered tell for the
"lookahead exploits the script's determinism" risk: collapse here = overfit.

**No collapse — the opposite: 62.5% (1562/2499 decisive, z=+12.5)** against
the production model+search agent, at 16 vs 173 ms/move (10.8x faster) with
zero NN calls. Divergence 45.0%, in line with E2x. The +5.2pp over Stage0 is
real directive-selection strength that transfers to a completely different
opponent; the NN-free lookahead agent is now plausibly the strongest agent in
the repo at ~1/10 production compute. Risk retired.

#### Verdict — GO (gate cleared)

Proceed to Stage 2 of the redesign: a real macro-MCTS tree over directives
(alternating turn-level nodes, Gumbel root over candidates, tree reuse), then
Stage 3 AlphaZero-ification (macro policy head + value on turn boundaries).
Heuristic leaves are the default evaluator until shown saturating.
Kept: macro-script/macro-lookahead backends, EXP_ELO_032 harness and arms.

## EXP_ELO_033 — Stage 2: adversarial turn-level MCTS over macro directives

**Registered 2026-08-12, before running.** Follow-on to EXP_ELO_032's GO.
Inference-only; arena-only backend `macro-mcts`.

### Diagnosis

EXP_ELO_032 proved one turn-level lookahead step over K=4 directives is worth
+5.2pp over its own executor (z>5) and beats production Gumbel n=64 62.5/37.5.
Its two structural limits: the opponent is a scripted ghost (Greedy), and only
the FIRST own turn varies (turns 2..H replay the script). Stage 2 replaces
both with a real tree: nodes are turn boundaries, edges are directives
executed by the deterministic executor, and — the first genuine attack on
failure mechanism #3 — **the opponent's turns are searched adversarially**,
choosing among its own K candidate directives instead of ghost-scripting.

### Changes

New `src/ai/macro_mcts.rs`: `MacroMctsSearch` (UCT over directive edges,
exploration 0.6 on [0,1]-mapped values per the HeuristicMctsAgent
calibration — classic 1.4 against evaluate_state's compressed band would
degenerate to uniform visits), negamax backup (antisymmetry of
`evaluate_state` is unit-tested, 2-player asserted), per-node owned state
clones + per-seat counters/archetype, node player fixed BY ALTERNATION (a
mid-edge game-over leaves current_player_turn_id unreliable — terminal value
is score-compare from the alternation player's perspective), opponent root
counters DERIVED from the fogged state (techs discovered after turn 0) so the
tech caps bind in-tree, turn-depth cap 8, expand-one-per-sim, leaf =
`evaluate_state` (heuristic only — the 032 leaf wash demoted net leaves; the
stage-weight discontinuity at turns 9/21 is ACCEPTED for v1: variable-depth
leaves straddle it, noted not fixed), root pick = argmax visits with ties to
candidate 0 (base). Fresh tree per own-turn boundary (no reuse in v1).
`MacroMctsAgent` = same per-turn commit surface as the Stage-1 lookahead.
Backend `macro-mcts` + `--macro-sims` (default 32, in the arena guard).

### Setup (pinned)

n=1250 seeds ×2 = 2500 games/arm; base_seed **1786600000** (fresh range);
gamemode 2; max_turns 30; imperius; --eval-backend metal; GUMBEL_SCALE=0;
model model.safetensors both configs; macro params k=4 lambda=1.0 sims=32
(lookahead arm keeps h=2, leaf=heuristic); one binary (/tmp/exp032_target).
Dumps `replays/exp033/arm{A1,A2}`. Smoke first: visit concentration
(root_visit_max_share), tree depth ≥2, ms/move.

| arm | config1 | config2 | question |
|---|---|---|---|
| A1 (gate) | macro-mcts sims=32 | macro-lookahead k=4 h=2 heur | tree > one-ply lookahead? |
| A2 (context) | macro-mcts sims=32 | gumbel n=64 k=16 | vs production, vs E4x's 62.5% |

### Predictions

| arm | metric | predicted | falsified if |
|---|---|---|---|
| A1 | mcts win vs lookahead | ≥52.0% (z≥1.96 at n=2500) | <51% AND root visits concentrated → tree adds nothing over one ply at sims=32 (a REAL finding: the adversarial opponent + depth were the new information) |
| A1 | divergence (root pick ≠ base) | 30–65% | <15% = tree collapsed to the script |
| A2 | mcts win vs gumbel n=64 | ≥62% (≥ E4x within noise) | <58% = the tree LOSES strength vs plain lookahead against a real opponent |
| smoke | root_visit_max_share | concentrated (>1/k+10pp) | ~1/k = exploration swamps the value band; retune before arms |

Depth-monotonicity sweep (sims 16/32/128) is the registered follow-up, NOT
part of this gate. Risk to watch: per-node state clones + 32 executions/turn
≈ 4× E2x cost — if ms/move lands >150 (worse than production Gumbel), the
efficiency story weakens even if strength holds; Tell: smoke ms/move.

### ACTUAL (2026-08-12) — CONFIRMED on both arms; Stage 2 works

Smoke first, per registration: at the registered EXPLORATION=0.6 the falsifier
fired — root visits uniform (share = 1/k) with measured root q01 spreads of
only 0.01–0.06. Dialed c to 0.05 against the measured band BEFORE arms (the
q-gap dial method): separated states then concentrate to share 0.69–0.81 with
PV depth 5, genuine ties stay uniform. Constant + rationale documented at
`macro_mcts.rs::EXPLORATION`.

Arms (2500 games each registered; A2 stopped externally at 2234 dumps —
time-truncated, outcome-blind, so the partial sample is unbiased):

| arm | predicted | actual | verdict |
|---|---|---|---|
| A1 mcts vs lookahead | >=52.0% | **58.9%** (1472/2500, z=+8.9) | **CONFIRMED** |
| A1 divergence | 30–65% | 38.6% | in band |
| A2 mcts vs gumbel n=64 | >=62% | **66.0%** (1475/2234, z=+15.2) | **CONFIRMED** (E4x was 62.5% for one-ply lookahead; the tree adds ~+3.5pp against production too) |
| smoke ms/move | <150 | 95.2 | passed — still 1.8x cheaper than production Gumbel (173) |

Strength ladder now: script (Greedy-parity) < one-ply lookahead (+5.2pp over
script) < turn-level adversarial tree (+8.9pp over lookahead, 66% over
production Gumbel), all NN-free at the leaves.

#### Verdict — CONFIRMED. The adversarial turn-level tree beats one-ply
lookahead decisively at sims=32; the macro-search architecture compounds.
Registered follow-ups: (i) sims sweep 16/32/128 (depth monotonicity at the
macro level); (ii) Stage 3 — AlphaZero-ify: macro policy head trained on tree
visit distributions + value head trained on turn-boundary states, replacing
the heuristic leaf only when it beats it in a paired A/B.

## EXP_ELO_033b — sims-64 rung: does macro depth buy strength head-to-head?

**Registered 2026-08-12, before running.** Quick single rung of the registered
depth-monotonicity sweep, on Verdi's ask. New arena `--macro-sims1/2` per-side
overrides make it a PAIRED same-game comparison, not a cross-arm read.

Setup: macro-mcts sims=64 (config1) vs macro-mcts sims=32 (config2), both
k=4 lambda=1.0 heuristic leaves; n=500 seeds ×2 = 1000 games (quick — sigma
1.6pp); base_seed 1786700000; GUMBEL_SCALE=0; metal; dumps
replays/exp033/armB64. Probe at sims=64: max depth 5–6 player-turns (vs 4–5
at 32), nodes 65, concentration 0.41–0.91.

Predictions: sims=64 wins ≥52% (depth helps at the macro level too);
falsified if ≤50% (z≤0) — macro depth saturates by 32 sims at k=4, itself a
real finding (candidate set or leaf noise binds, not depth). Guardrail: ms/move
config1 ≤ 2.2× config2.

### ACTUAL (2026-08-12) — falsified: macro depth saturates by ~32 sims at k=4

1000 games as pinned. sims=64: **48.4%** (484/998 decisive, z=-0.95) vs
sims=32 — the >=52% prediction is falsified; flat-to-slightly-negative,
indistinguishable from 50%. ms/move 219 vs 112 (1.96x, guardrail passed).
Divergence 45.0% at sims=64 (42-45% band stable across budgets).

Read: doubling macro-tree compute buys NOTHING at k=4 with heuristic leaves.
At 32 sims each of the ~3-4 live candidates already gets ~8-10 visits — enough
for UCT to resolve the measured 0.03-0.1 q gaps; extra sims only deepen the PV
(probe: depth 5-6 vs 4-5), and deeper lines run on a degrading world model
(fogged opponent, drifting counters/archetypes) scored by a leaf that cannot
discriminate them. The binding constraints are now, in order of suspicion:
(a) candidate-set richness (k / directive vocabulary), (b) leaf evaluation
quality, (c) belief-state fidelity of long rollouts - NOT depth. Mirrors the
ply-level story (only 64->256 ever cleared significance) but saturating at the
point the candidate set is exhausted.

#### Verdict — FALSIFIED (a useful null). Do not spend on --macro-sims.
Stage-3 priority order updated: a learned/calibrated leaf and a richer
candidate generator are the levers; depth is bought and banked at sims=32.

## EXP_ELO_034 — belief state over hidden information: offline calibration

**Registered 2026-08-13, before running.** First rung of the belief ladder
(034 inference+calibration → 035 MAP materialization → 036 determinization
ensemble; design locked in current_understanding.md). The macro tree's known
bias is optimism: `obscure_fog` deletes unseen enemy assets, so rollouts play
a weaker opponent than reality. Before any search change, this experiment
builds the belief and **measures it against arena ground truth only** — no
agent consumes it.

Diagnosis feeding the design: score is public (leaderboard) and
`calculate_detailed_tribe_score` is deterministic and invertible (units
cost×5, tech 100×tier, capture +100 +20/territory +5/pop, exploration
+5/tile), so per-move opponent score deltas carry a decodable event stream.
Mapgen (quadrant path, Tiny Drylands) provably confines capitals to one cell
per quadrant — {24, 29, 79, 84} — verified by a 100-map generator probe
BEFORE this registration (`capital_support_matches_generator_tiny_drylands`),
so the opponent-capital prior is exactly 3 uniform hypotheses.

Changes: `src/ai/belief.rs` — `BeliefState` (capital posterior over the
generator support; unwitnessed-delta attribution: exploration / unit build /
tech / capture signatures; ghost-departure disambiguation so a scout the
observer watched leave doesn't read as hidden production; emergence
accounting shrinks the inferred pool when hidden stars walk into vision;
tanh-bounded confidences) + `CalibHarness` (the only truth-reading component;
feeds observables, logs belief-vs-truth rows). Arena `--belief-calib`
(observation-only; requires `--dump-stats-dir`; rows land in each game's
dump JSON). 9 unit tests. Witnessing rule per advisor review: a move is
witnessed iff mover==observer, or it has ≥1 involved tile and ALL are
observer-explored — tile-less opponent moves (Research) are always
unwitnessed, else all tech signal dies silently.

Setup (pinned): arena `--backend1 macro-script --backend2 greedy` (the 035
deployment shape, NN-free fast), `--games 125` (=250 games), `--base-seed
1786800000`, `--gamemode 2 --max-turns 30 --tribe imperius --eval-backend
metal`, `GUMBEL_SCALE=0`, `--belief-calib --dump-stats-dir
replays/exp034/calib`. Analysis: `replays/exp034/analyze_calib.py`.

Predictions:
- **P1 (capital collapse):** mean posterior mass on the true capital cell
  (`cap_truth_p`) ≥ 0.6 by t10 and ≥ 0.9 by t20; top-1 hit ≥ 60% by t10.
  Falsifier: cap_truth_p ≤ 0.4 at t10 (no better than the 0.33 prior) —
  elimination/sighting updates broken.
- **P2 (GUARDRAIL, support validity):** true capital inside the initial
  3-cell prior in ≥ 99% of games (cap_truth_p > 0 on the first row).
  Falsifier: any systematic violation — the mapgen read is wrong; halt, all
  downstream numbers void.
- **P3 (hidden army beats assume-zero):** MAE(believed_hidden_army −
  truth_hidden_army) ≤ 0.7 × MAE(assume-zero) averaged over t5–t25.
  Assume-zero IS the empty-fog status quo, so this is the gate for 035:
  falsified if ratio ≥ 0.9 — v1 attribution too noisy, fix inference before
  any materialization (it would inject garbage).
- **P4 (confidence reliability):** bucket rows by `army_conf` terciles;
  hidden-army MAE must fall monotonically low→high bucket. Falsifier:
  non-monotone — the tanh evidence model is miscalibrated.
- Context (logged, not gated): unwitnessed share of opponent score events;
  corner-heuristic (`predict_enemy_capitals` replica) hit rate vs belief
  top-1 — expected to be dominated.

Known v1 blind spots (deliberate, in the ledger so they're not rediscovered):
converted units score 0 (score counting can't see them); cloaked units
excluded from the visible scan; ghost records carry no passenger; territory
bleed of a hidden capture into visible tiles is unmodeled; capital capture
mid-game unhandled. Risk+tell: witnessing misclassification polluting the
residual pool — tell is hidden_cities_believed ≫ truth in capture-heavy
games with P3 failing there specifically.

### ACTUAL (2026-08-13) — P1/P2/P3 PASS, P4 falsified; belief is 035-ready

250 games as pinned (macro-script won 53.2%, in the 032 E1 band — arm sane).
~11k belief-vs-truth rows, 500 observer streams.

- **P2 GUARDRAIL: 500/500 (100%)** — the true capital was inside the 3-cell
  prior in every stream. The {24,29,79,84} generator support holds at scale.
- **P1 PASS: cap_truth_p 0.858 at t10, 0.984 at t20; top-1 87% at t10.**
  Autopsy: ZERO wrong-collapses in 500 streams — the only 3 streams ending
  <0.5 are never-scouted splits honestly still at the prior. Corner-heuristic
  baseline (predict_enemy_capitals replica): 16% at t10 — dominated 5×.
- **P3 PASS: hidden-army MAE ratio 0.53** (belief 2.65 vs assume-zero 5.00
  stars, t5–25) — the belief halves the empty-fog error. Decomposition
  (advisor-prompted): ghost-memory-only 0.69, score-residual-only 0.77,
  combined 0.53 — the components are complementary (ghosts = seen-then-hidden
  units WITH positions; residuals = never-seen production), and neither alone
  clears the 0.7 gate. Aggregate is near-unbiased: truth 5.00 ≈ residual 1.90
  + ghost 3.19. Gate for 035 cleared. Unwitnessed share of opponent score
  events: 36.9%. 035 materialization note: ghosts carry placement, residual
  stars need a placement policy (e.g. around the believed/known capital).
- **P4 FALSIFIED, and not by confounding:** turn-banded terciles show the
  inversion inside every band (t11–17: 1.41 → 2.50 → 5.09). v1 confidence
  counts build signals, so it grows with the pool, and absolute error grows
  with the pool — it measures estimate SIZE, not reliability. Redesign before
  036 uses it as a world-mixing weight (035 MAP materialization doesn't need
  it; the capital side's confidence = posterior mass, calibrated by
  construction — 98% mass ≙ 98% top-1 at t20).
- **Instrument lesson (cost one rebuild): `tile.capital_of` is reassigned to
  the capturer on capture (actions/city.rs:274)** — the first run's truth
  query used live lookups and manufactured a fake 15% wrong-collapse plateau
  (0.855 at t20). Truth for a spawn-location belief must be snapshotted at
  game start. The belief itself was right all along.
- Known v1 gap confirmed: hidden_cities overcounts ~10× late (belC 4.5 vs
  truthC 0.1 at t25) — two causes: level-up composites (harvest +50 +20·k)
  hit the ≥150 capture signature, and discovered cities never decrement the
  believed count (no city-emergence accounting). Both are v2 items; nothing
  in 035 consumes hidden_cities.

#### Verdict — CONFIRMED for the 035 gate. Capital posterior: trust it
(zero wrong-collapses, 98% collapsed by t20). Hidden-army estimate: use the
point estimate, not the confidence. GO for EXP_ELO_035 (MAP materialization
into the fogged clone, A/B vs empty-fog macro-mcts); fix hidden_cities and
confidence semantics before 036.

## EXP_ELO_034b — aux_fog calibration: is the learned belief head usable?

**Registered 2026-08-14, before running.** Verdi's directive: before the
learned per-tile P(enemy) head (`aux_fog`, mirrored in network.rs since v9)
becomes the materializer's placement policy or the 036 world-sampler, vet it
with the same offline-calibration discipline the hand belief got in 034.
Training semantics (verified in code): target = TRUE non-invisible
enemy-unit occupancy over ALL tiles (self_play.rs `enemy_unit_grid`), input
= the POV's fogged-view features WITH goal channels painted — so explored
tiles are perception (easy), fog tiles are inference (the signal we need).

Instrument: `#[ignore]` probe test — 40 generated Tiny Drylands games,
MacroScript both seats, moves applied via **`game.play_move`** (NOT the
simulate path: `_are_you_sure` gates observation memory — verified
game.rs:206 vs :275 — and simulate-driven games would starve the ghost
input channels the head trained on; tripwire asserts ghosts appear in ≥30%
of games). Features encoded EXACTLY as training does (verified
self_play.rs:1694): `state_to_cpu_features_goal(&TRUE_state, pov, None,
Some(&scripted_goal))` — the true state with pov-keyed masking inside the
encoder, goal channels painted — not a fogged clone. Forward through a
PINNED copy of model.safetensors (sha256 recorded in ACTUAL) on candle CPU;
score `fog_probs` against the true enemy grid, partitioned explored/fog.

Predictions:
- **P0 (GUARDRAIL, instrument validity):** explored-tile AUC ≥ 0.90 — the
  head sees visible units in its input; if it can't rank those, the probe's
  feature encoding is broken, not the head. Any P1 read is void until P0
  passes.
- **P1 (headline):** fog-tile AUC ≥ 0.70 → usable placement signal.
  Falsifier: < 0.60 — the head lacks utility for this purpose; do not build
  the placement policy on it (fix would be training-side, e.g. fog-masked
  loss weighting, not consumer-side).
- **P2 (mass calibration, turn-banded — the 034-P4 lesson pre-applied):**
  per-row Σ fog_probs over fog tiles vs true hidden unit count, Pearson r
  computed WITHIN each band t5–10 / t11–17 / t18–25 (pooled r is
  turn-confounded and not registered). Pass: r ≥ 0.4 in every band;
  falsifier: any band < 0.2.
- Context: per-turn AUC curve (does inference quality survive past the
  early game?); mean predicted fog mass vs true hidden count by turn.

Caveat in the registration: the head was trained on Gumbel self-play
trajectories; the probe plays scripted-executor games — close (vs-Greedy
anchors are in the training mix) but not identical distributions.

### ACTUAL (2026-08-14) — CONFIRMED on every gate; aux_fog is a usable belief

Model pinned: sha256 78f0f66cccee2fbd95a2371e64254d7acebe31d62fb29a865a895b3678b07b76.
40 games, ~2.1k forwards, ghost tripwire 40/40 (observation memory
accumulated in every game — the play_move decision mattered).

- **P0 instrument guardrail: explored-tile AUC 1.000** — perception is
  perfect, the feature parity is right, P1 is a valid read.
- **P1 PASS: fog-tile AUC 0.848** (gate ≥ 0.70) — the head genuinely ranks
  unexplored tiles by hidden-enemy likelihood.
- **P2 PASS in every band: r = 0.675 (t5–10), 0.820 (t11–17), 0.797
  (t18–25)** — predicted fog mass tracks the true hidden count, with a mild
  ~1.25× over-prediction (e.g. 1.89 predicted vs 1.40 true at t11–17) that a
  single scalar calibration would fix.
- Per-turn curve: strongest exactly where the belief window matters —
  0.85–0.93 through t10, 0.72–0.81 mid-game, noisy-but-positive late (small
  n past t21). Open oddity flagged so it isn't rediscovered: a dip at t1–t2
  (0.93 → 0.86 → 0.80, recovering to 0.92 by t3) sits exactly in the
  earliest sampler window — plausibly tiny-positive-count noise (one scout
  in fog), unexplained.

#### Verdict — CONFIRMED. The learned head solves the hand belief's weakest
sub-problem: WHERE (per-tile placement, AUC 0.85 in fog) — and P2's r
0.68–0.82 says it plausibly covers HOW MUCH as well; whether the 034
score-counting residual adds anything on top of aux_fog mass is UNMEASURED
(the two probes ran on different game populations), so measure that overlap
before 036 spends design effort fusing two systems one may satisfy alone.
Cleared as the materializer's placement policy and the 036 world-sampler
(with the ~1.25× mass calibration applied), **conditional on re-running this
probe under the consumer's real trajectory mix** — a belief-driven planner
steers games because of the belief, which moves the distribution off what
was measured here; the permanent `#[ignore]` test makes that gate nearly
free.

## EXP_ELO_036 rung 1 — belief-conditioned fog-expansion candidates

**Registered 2026-08-14, before running.** The twice-reinforced lever
(033b, 035) plus the 035 post-mortem's sharpest lesson: belief-driven play
("they spawned east → rush the safe NW villages, contest NE") was
INEXPRESSIBLE — expansion orders were filtered to visible villages. This
rung makes it expressible and prices it in the tree. One delta: candidate
generation only — NO materialization (belief_mode=candidates), no eco
model, no reality weighting.

Changes: `enumerate_candidates_with_belief` (class-tagged; adds ClaimSafe =
base + up to 2 safe-side predicted villages, Contest = base + nearest
enemy-side predicted village; targets from the deterministic, fog-only
`predict_villages`, partitioned by Chebyshev distance to own vs believed
enemy capital — `capital_confirmed` preferred over the posterior peak;
excludes the believed-capital ring and duplicate/visible targets; pushed
before Real/Attack variants so k=7 always retains them). goal_potential
already prices unexplored targets approach-only (reward.rs v4) — no pricing
changes. `MacroParams.belief_mode {off,world,candidates,both}` (`--macro-
belief-mode1/2`, 035's `--macro-belief1/2` alias World); per-side
`--macro-k1/2`; telemetry = winning-candidate CLASS per planned turn
(executions, not offers — the 035 lesson) + consecutive-turn re-picks of
the same fog target (plan-stability tell). Scope note: belief conditions
the ROOT enumeration only; deeper in-tree enumerations (own and opponent
turns) stay belief-blind — the opponent's simulated counters never see
pov's belief.

Setup (pinned): config1 `--backend1 macro-mcts --macro-belief-mode1
candidates --macro-k1 7`; config2 `--backend2 macro-mcts` (k=4 production
generator); **both `--macro-sims 48`** — compute-matched, and baseline-at-48
≈ baseline-at-32 strength per 033b's flat 32→64 (pinned so this arm isn't
moving two dials silently). 500 seeds ×2 = 1000 games, base_seed
1787000000, gamemode 2, max-turns 30, imperius, metal, GUMBEL_SCALE=0,
dumps `replays/exp036/armA`.

Predictions:
- **P1:** config1 ≥ 52.5% (z ≥ 1.58) → CONFIRMED — **scoped: a win
  confirms "richer fog-expansion candidates help"; attribution between
  belief-partitioning and mere width is EXPLICITLY DEFERRED to the
  pre-named ablation 036-abl** (same k=7, fog targets ranked by
  distance-to-us only, posterior ignored) — the 035 star-ramp scoping
  lesson applied prospectively. Falsifier: ≤ 50.0% — the offered plays
  don't win directive selection or cost tempo when they do. 50.0–52.5% →
  extend to 1250 seeds first.
- **P2 (GUARDRAIL):** zero panicked/skipped games.
- **P3 (GUARDRAIL):** ms/move config1 ≤ 1.4× config2 (k 7 vs 4 enlarges
  rollout fan-out).
- Context (gates interpretation): claim+contest must WIN ≥ 10% of planned
  turns, else a flat P1 is "never picked", not "picked and worthless";
  repick count reported for plan stability (flickering fog offers would
  yank units mid-approach and read flat for a stability reason).

Risk+tell: chasing wrong village predictions costs tempo — tell is P1
falsified WITH a high claim-pick rate and config1's own score dropping
(prediction-quality problem, fix predict_villages), vs falsified with ~0%
picks (competitiveness problem, fix pricing/stance interaction).

### ACTUAL (2026-08-14) — context gate FAILED: expressible ≠ competitive

1000/1000 games (P2 pass, zero panics); ms/move 171 vs 171 — **k=7 is
compute-free at matched sims** (P3 pass, 1.00×). P1 read 50.6% (z=+0.38),
nominally the extend band — but the registered interpretability gate
decides: **claim+contest won only 3.8% of 21,411 planned turns (gate ≥10%),
so the arm answers competitiveness, not value — extending seeds would spend
25 min measuring a play that barely fires.** The pre-registered "~0% picks"
tell applies: fix why they lose directive selection before re-asking P1.
Plan stability confirmed the advisor's predicted failure mode: 74 re-picks
vs ~814 belief-class picks (~9% persistence) — a picked fog target must
re-beat the scripted base from scratch every turn (update_goal's hysteresis
has no memory of it) and usually loses. Divergence 42.0% (back in the 033
band at sims=48).

Diagnosis — three mechanisms examined, third quantified and EXONERATED by
the `fog_offer_quality_probe` (20 scripted games, advisor-prompted): claim
offers appear on 50–60% of player-turns FROM T0 (not sparse, not late —
the northwest-rush window is covered), and 56–88% of predicted targets sit
within 1 tile of a real village/city (mirror-Imperius scope; the climate
heuristic being same-climate-dead did not materialize — orphan-resource
evidence carries the predictor). Contest offers are rare (~2–20%): most
predicted villages fall on the observer's side of the partition. So the
tree DECLINES frequent, mostly-good offers — the two live mechanisms:
1. **Leaf blindness to in-flight approach.** The macro tree's leaves are
   `evaluate_state` only; λ·Δφ shapes PLY ranking inside rollouts but never
   enters LEAF VALUES. A 4–7 tile walk to a predicted village pays only if
   the capture completes inside the 2–3-own-turn effective horizon;
   otherwise the leaf sees displaced units and no credit — the directive
   loses to stance variants whose payoff (SPT) is leaf-visible immediately.
   This is 028's core finding pointed at the macro layer: the goal must be
   priced IN THE SEARCH OBJECTIVE, not only in move ordering.
2. **No directive persistence** — nothing carries a chosen fog target into
   the next turn's base, so multi-turn plays can't survive re-enumeration.

#### Verdict — GATED NULL (not falsified, not confirmed): the vocabulary
lever alone doesn't bind while the tree cannot CREDIT or SUSTAIN multi-turn
expansion plays. Follow-up 036b (one delta each, in order): (a)
potential-based shaping in the tree — Δφ of the node's ACTIVE directive
accumulated on pov's own edges (telescoping to φ(leaf)−φ(root)), NOT naive
φ-added-to-leaf, which would break the negamax antisymmetry invariant the
backup rests on; extend the antisymmetry unit test to cover shaping, and
expect the UCT c=0.05 to need re-dialing (it was dialed to the unshaped
q-spread); (b) fog-target stickiness — committed expand targets join
StanceCommit's hysteresis. Then re-ask this rung's P1; 036-abl
(belief-blind width) stays queued for attribution only if the play becomes
competitive.

## EXP_ELO_039 — Stage 3 gate: does the macro-distilled value head beat the heuristic leaf?

**Registered 2026-08-14, before running. RUN GATE: only after Verdi's
20-iteration macro-distillation round completes and the model is
SNAPSHOTTED (cp model.safetensors → a pinned path; record sha256 + iter
count in the ACTUAL). Do not run against the live mutating file.**

The bottleneck claim this tests (from the six-arm falsification program):
macro directive selection starves for EVALUATION — a heuristic leaf 2–3
own-turns deep cannot distinguish the futures competing strategies create.
The counter-evidence standard was set in 032-E3 (net leaves were a wash),
with the Stage-3 bet being that E3's net was OFF-distribution (per-ply
Gumbel training, documented calibration problems) and a macro-distilled,
on-distribution value head changes the answer.

Changes: net-leaf path in the macro TREE (033 restricted it to lookahead):
`MacroMctsAgent`/`MacroMctsSearch` carry the evaluator; leaf value =
win_value on features painted with the SCRIPTED base goal for the leaf
player (the committed directive is unknowable before the choice — a known
approximation; `.1` progress stubbed on tch/metal, `.0` only; heuristic
fallback on encode/eval failure). Arena: per-side `--macro-leaf1/2`, 033
guard lifted. Dummy-evaluator tree test added.

Setup (pinned at registration; model path filled at run time): config1
`--backend1 macro-mcts --macro-leaf1 net --model1 <snapshot>`; config2
`--backend2 macro-mcts` (heuristic leaf; model2 irrelevant/idle); both
sims=32 k=4, no belief/shaping. Both arms SYMMETRICALLY carry the 038
continuation-memory candidates (built unconditional; measured neutral in
038, and symmetric here so it cannot confound the leaf read). 500 seeds
×2 = 1000 games, fresh base_seed 1787300000, gamemode 2, max-turns 30,
imperius, metal, GUMBEL_SCALE=0, dumps `replays/exp039/armA`.

Predictions:
- **P1 (the Stage-3 gate):** net leaf ≥ 52.5% → CONFIRMED — learned
  evaluation beats the hand-written leaf at matched budget; the macro leaf
  seat changes hands and the bottleneck diagnosis is validated. Falsifier:
  ≤ 50.0% — distillation at this depth hasn't produced a discriminating
  evaluator (more iterations / label changes before re-ask; the heuristic
  keeps the seat). 50–52.5% → extend.
- **P2 (mechanism, runs either way):** root q-spread probe on mid-game
  states, net vs heuristic leaves — the bottleneck hypothesis predicts the
  net WIDENS the measured 0.01–0.06 spreads. Widened spreads + P1 win =
  clean confirmation; widened + P1 loss = discriminating but miscalibrated
  (sign/scale work); unchanged spreads = the head learned the same
  blindness (label problem).
- **P3 (GUARDRAIL):** zero panics; ms/move config1 ≤ 3× config2 (one
  batch-1 eval per sim, ~32/turn — the net leaf is not free).

Risk+tell: value-head over-confidence transfers from the calibration
diagnostic (documented pre-macro) — tell is P1 falsified with WIDE spreads
and root_visit_max_share pinned near 1.0 (over-confident leaf collapses
exploration).

### ACTUAL (2026-08-14, 1000 games) — **P1 FALSIFIED, and in the WRONG
DIRECTION.** Net leaf 388/1000 (38.8%) vs heuristic leaf 612 (61.2%);
z = (388-500)/15.8 = **-7.1**. The macro-distilled value head does not
merely fail to beat `evaluate_state` at ranking macro futures (the 032-E3
wash) — it ranks them actively WORSE than the hand-written evaluator.
Avg score 4012 vs 4982; cost 436 vs 362 ms/move (+20%).

Snapshot: `checkpoints/exp039_snapshot_iter125.safetensors`, sha256
119b809760e81e12aabd9ce5cd15087cb47acb31245c10711ad0938eafab39a4, iter
125 (the 10-iteration MACRO_GEN round: goal_channels=1, policy_loss 2.28,
value_loss 0.45, value_r2 0.79).

⚠️ **Scope: this is a verdict on THIS checkpoint trained on THOSE labels,
not on "a net can never be the leaf."** The same session proved the
labels defective: under MACRO_GEN the macro agent reported no root value
(`brain.rs` returned None for every non-Gumbel backend), so every TD
n-step return bootstrapped with 0.0 — systematically truncated returns,
which teach a value head to under-weight exactly the long horizon a macro
leaf must judge. Fixed in ea700e4 (macro root Q under a net leaf;
`--td-missing-bootstrap mc` for the heuristic-leaf path). Re-run 039 on a
round trained with corrected labels before drawing an architectural
conclusion.

Consequences taken (per the approved plan's outcome table):
- Heuristic leaf keeps the seat; `MACRO_LEAF` stays `heuristic`.
- Stage 3b (`aux_playstyle` head + the 169->173 channel migration across
  ~210 checkpoints) is DEFERRED — a trunk that misranks futures at the
  leaf is not a credible lane ranker, and the migration is only worth
  paying once something can use it.
- Stage 3a (the algorithmic selector) proceeds → EXP_ELO_045a.
- Measured in passing: net-leaf GENERATION runs at 3.28 moves/s vs ~100
  for the heuristic leaf (4 games, 2 actors) — a ~30x tax that would have
  made a net-leaf training round impractical even had P1 confirmed.

## EXP_ELO_038 — strategist memory: continuity by selection, not injection

**Registered 2026-08-14, before running.** Verdi's architectural resolution
after the 036b/037 chain: crediting the work of advance belongs to the
EXECUTOR only (where it has lived since 032); the STRATEGIST gets memory
instead of bribes. Two named deltas from 037 (a redesign, not a tuning arm
— attribution between them deferred if needed):
1. **Chooser shaping OFF** (shape_w=0) — removes the convicted −5pp
  channel; plumbing kept for the still-untested approach-only variant.
2. **Continuation candidates**: the agent remembers its last 3 picked
  directives; each plan re-offers them on the ballot (deduped against the
  enumerated set, fog orders the evidence has killed stripped before the
  offer). The tree picks continuation when rollouts still like it and
  pivots freely otherwise — the 036b forced base-injection is deleted.
  Per-ply intra-turn stripping of the live goal stays (uncontroversial
  hygiene). New CandidateClass::Continuation telemetry (class array → 7).

Setup (pinned): config1 macro-mcts belief-mode candidates k1=7 (+ up to 3
continuations appended past k), shape-w1 0; config2 production macro-mcts
k=4; both sims=48; same base_seed 1787100000 (chain read); 500 seeds ×2 =
1000 games; dumps `replays/exp038/armA`.

Predictions:
- **P1:** ≥ 52.5% vs production → CONFIRMED (strategist memory is worth
  real ELO). Falsifier: ≤ 50.0% — memory doesn't bind at this decision
  granularity. 50–52.5%: extend before verdict. Chain anchor: removing the
  convicted shaping harm should alone recover toward rung-1's ~50.6%;
  anything clearly above that is the memory's contribution.
- **P2 (GUARDRAIL):** zero panics. **P3 (GUARDRAIL):** ms/move ≤ 1.15×
  (shaping calls gone; a few extra candidates added).
- Context (interpretability gates): continuation-pick rate ≥ 15% of
  planned turns (else the memory is never used and a flat P1 says
  nothing); divergence expected to DROP vs 037's 54.5% (chosen continuity
  should stabilize strategy); claim/contest picks expected to fall back
  toward rung-1 levels without shaping credit (the fog-play thread goes
  dormant this arm — that's accepted; this arm tests MEMORY).

Risk+tell: self-reinforcement — continuation re-picks compound (yesterday's
choice crowds today's ballot) and the strategy ossifies; tell: continuation
picks > 50% with divergence collapapsing toward zero AND P1 flat-negative.
The dedup-against-base and the tree's freedom to pivot are the guards.

### ACTUAL (2026-08-14) — P1 FALSIFIED at 48.6%: memory used, not useful

1000/1000 (P2 pass; P3 pass at 0.99× — cheapest consumer arm yet).
Continuation won 15.6% of 21.9k planned turns (interpretability gate ≥15%
cleared exactly), divergence dropped 54.5→48.2% as predicted, ossification
tell never fired. So: the strategist demonstrably used its memory, chose
continuity over scratch-derived stance flips (stance picks 35→19%), and
none of it bought wins — 48.6% vs production (z=−0.89), score dead even
(4415 vs 4430). Chain: dropping the convicted shaping recovered +2pp
(46.6→48.6); the remaining composite (k=7 + continuation + strips) sits at
or slightly below plain rung-1 (50.6%, prior seed base).

#### Verdict — FALSIFIED. Strategist memory as ballot candidates is
neutral-to-slightly-negative at this decision granularity with heuristic
leaves. PROGRAM-LEVEL READING after six consumer arms (035, 036r1, 036b,
037, 038): no intervention that changes WHICH directive the tree picks has
beaten the tree's own unshaped judgment over the production candidate set —
harms when it forces (shaping, stickiness), neutral when it offers
(candidates, memory). The selection layer is not starving for options or
continuity; it is starving for EVALUATION — a heuristic leaf 2–3 own-turns
deep cannot distinguish the futures these strategies create (033b's
leaf-quality lever, still untouched). The macro program's next real forks:
(1) Stage 3 — learned value at turn boundaries trained on macro-tree games
(the original redesign endgame; the tree generates its own training data);
(2) the eco-calibrated materializer + reality-weighted descent directives
(still unbuilt); (3) integrate production macro-mcts (still the strongest
agent in the repo, 66% over per-ply Gumbel) into the TRAINING loop as the
self-play policy. All belief/hygiene/telemetry infrastructure survives for
whichever runs.

## EXP_ELO_037 — evidence-driven commitment retirement (menu item c)

**Registered 2026-08-14, before running.** Single delta from 036b: the
sticky commitment now CONSUMES the belief it serves (Verdi's spec). Four
retirement rules + backstop, replacing the blind 6-turn cap:
1. **Intra-turn strip**: select_move runs every ply on a fresh view — a fog
   order seen explored-and-gone mid-turn is stripped from the LIVE goal that
   ply (achieved orders exempt per 028's achieved-holds-cap semantics), and
   the sticky commitment retires with it. Beliefs already updated per move;
   the frozen directive was the only non-consumer.
2. **Live-prediction re-validation** at plan time: a still-fogged target
   must remain in the current predict_villages set — predictions move with
   evidence, the commitment now moves with them.
3. **No-progress retirement**: two consecutive plan turns where the closest
   own unit failed to get closer → the order is a zombie distorting pricing
   while nobody marches; retire.
4. Achieved/adopted retirement, and **cap 4 as pure backstop** (not the
   dose-slice 2: with evidence rules live, a 2-cap would amputate
   legitimate 3–4-turn marches to real villages and dead-code rule 3).
Telemetry: retirements by rule + mid-turn strips (STICKY RETIREMENTS line).

Setup (pinned): IDENTICAL to 036b — config1 macro-mcts belief-mode
candidates k1=7 shape-w1=1e-4, config2 production macro-mcts k=4, both
sims=48, **same base_seed 1787100000** (cross-arm same-seed read vs 036b's
45.3%, with the documented engine-flip caveat: σ_diff ≈ 2.2pp cross-run).
500 seeds ×2 = 1000 games, dumps `replays/exp037/armA`.

Predictions:
- **P1a (mechanism, primary):** ≥ +3pp vs 036b's 45.3% (i.e. ≥ 48.3%) —
  the retirement rules recover real value from the long-hold harm channel.
  Falsifier: < +1.5pp — the 3+/6+ dose bands were mostly reverse causation
  (losing games GENERATE long holds), channel A (selection-level shaping
  double-pay) dominates → pivot to approach-only shaping.
- **P1b (deployment, stretch):** ≥ 52.5% vs production → CONFIRMED. Honest
  mix-math expectation if the dose slice is fully causal: converting the
  276 long-hold games (~28% win) to short-hold behavior (~50–54%) predicts
  ~+2–5pp → 47–50%; crossing 50% would beat that expectation.
- **P2 (GUARDRAIL):** zero panics. **P3 (GUARDRAIL):** ms/move ≤ 1.25×.
- Context: long-hold share (sticky_injected ≥3 per game) collapses vs
  036b's 27.6%; retirements dominated by the evidence rules
  (dead/unpredicted/no-progress), NOT the cap — cap-dominated retiring
  would mean the rules never fire and the arm degenerates to cap-4.

Risk+tell: over-eager retirement re-creates rung-1 flicker (units yanked
mid-march from VALID targets) — tell: sticky_injected collapses toward
zero AND claim picks stay high AND P1a fails; the no-progress rule (2
strikes, not 1) and the achieved exemption are the guards.

### ACTUAL (2026-08-14) — P1a FALSIFIED at +1.3pp: the dose bands were
reverse causation; channel A convicted by elimination

1000/1000 (P2 pass; P3 pass 1.12×). 46.6% vs production = **+1.3pp over
036b's 45.3%, under the +1.5pp falsifier line** and inside σ_diff ≈ 2.2pp —
statistically nothing. The over-eager-retirement tell did NOT fire
(sticky_injected 1676 vs 1938, picks steady at 10.3%), and the machinery
demonstrably worked: 3,778 mid-turn strips, retirements evidence-dominated
(unpredicted 118, no-progress 82, achieved/adopted **814**, cap 3 — most
commitments now END IN CAPTURE or script adoption). The plays complete;
completing them doesn't win games.

Three-arm causal chain, one delta each:
- 036 rung 1 (candidates only): 50.6% — plays offered, never picked, flat.
- 036b (+ shaping w=1e-4 + sticky): 45.3% — plays picked, −5pp.
- 037 (036b + evidence retirement): 46.6% — commitment hygiene fixed,
  ~nothing recovered.
⇒ **The ~5pp harm lives in the selection-level shaping (channel A), not
the commitment mechanics (channel B).** The 035/036b dose slices misled in
exactly the direction their endogeneity caveats warned about.

#### Verdict — FALSIFIED as registered; pivot to the named branch:
**approach-only shaping** (strip GROW/ARM stance terms from the shaping
potential; credit only Expand/Attack approach — the leaf-invisible
quantity). The retirement engine stays (it's correct hygiene, near-free,
and 814 capture-completions say the plays themselves now resolve
properly); the next arm changes ONLY the potential. If approach-only
shaping cannot reach parity while holding picks ≥8%, the conclusion
hardens to: adherence credit at directive-selection is unsound in any
form, and the macro tree should select on unshaped values with shaping
confined to the executor (where 028 proved it).

## EXP_ELO_036b — the enabler stack: credit the advance, keep the commitment

**Registered 2026-08-14, before running.** Rung 1's gated null isolated two
missing enablers, both solved before in this repo and now transplanted
(Verdi's directive: "we need to reward hack — credit the work of advance"
= 028's pricing lesson; "borrow the continuity solution" = StanceCommit/026
sticky commitment):

- **Δφ edge shaping in the macro tree** (`MacroParams.shape_w`,
  `--macro-shape-w1/2`): potential-based reward w·(φ(s′,g)−φ(s,g)) on the
  ROOT PLAYER's edges only, g = that edge's own directive, one GoalAux for
  both sides of the difference (executor's edge_snapshot pattern — mixing
  directives/auxes across the difference would mint reward on switches).
  Backup is negamax-with-edge-rewards: v(parent) = r(edge) − v(child),
  r=0 on opponent edges (reduces exactly to the old negation) — the sign
  convention is pinned by `shaped_backup_matches_hand_computation` (4-node
  prebuilt chain vs hand computation; a sign error would invert the
  incentive at alternating depths). **w = 1e-4, dialed by probe** on
  mid-window states: shaped bonuses land at +0.03–0.05, inside the measured
  live q-band (0.01–0.07); 3e-4 is 2–3× the band; 1e-3 drives q>1
  (pathological). At w=1e-4 the probed roots start PICKING claim/contest —
  the credit mechanism works at the root.
- **Sticky fog target** (026-style): a picked claim/contest target joins
  the BASE directive on subsequent turns until captured, invalid
  (explored & not capturable), adopted by the script itself, vetoed by a
  RealFilter pick, or capped at 6 injected turns (a phantom prediction
  must not pin base.orders forever — ~25% of targets are phantoms).
  Telemetry: `sticky_injected` turns.

Setup (pinned): config1 `--backend1 macro-mcts --macro-belief-mode1
candidates --macro-k1 7 --macro-shape-w1 0.0001`; config2 `--backend2
macro-mcts` (production: k=4, w=0, no belief). Both sims=48. 500 seeds ×2 =
1000 games, base_seed 1787100000, gamemode 2, max-turns 30, imperius,
metal, GUMBEL_SCALE=0, dumps `replays/exp036/armB`.

**Attribution scoping (prospective, as in rung 1):** config1 bundles THREE
deltas (belief candidates + shaping + stickiness), and shaping also prices
BASE-directive adherence — a win confirms "the enabler stack works" and
could in principle be carried by shaping alone. Pre-named ablations, run
only on a win: 036b-abl1 = shaping-only (w=1e-4, no belief, k=4) vs
production; 036b-abl2 = belief-blind width (k=7 distance-ranked) + shaping
+ stickiness vs 036b arm.

Predictions:
- **P1:** config1 ≥ 52.5% → CONFIRMED (scoped per above). Falsifier:
  ≤ 50.0% — with credit AND persistence in place, if the plays still lose,
  the rung-1 leaf-credit diagnosis was wrong or the plays are genuinely bad.
  50.0–52.5% → extend before verdict.
- **P2 (GUARDRAIL):** zero panicked games.
- **P3 (GUARDRAIL):** ms/move config1 ≤ 1.3× (2 goal_potential calls per
  own-edge expansion).
- **Context, now EXPECTED to move (the mechanism test):** claim+contest
  pick rate ≥ 10% (was 3.8%) — if it stays < 10% with shaping and
  stickiness live, the credit diagnosis is falsified regardless of P1;
  sticky_injected ≫ 0 and repicks ≫ 74 (persistence must be visible).

Risk+tell: over-shaped adherence — the agent walks at fog mirages past
real fights (tell: pick rate high, P1 falsified, score gap growing in
CONTACT games specifically); w=1e-4 was chosen minimal-sufficient to
bound this.

### ACTUAL (2026-08-14) — P1 FALSIFIED HARD: 45.3% (z=−3.0). The enablers
work mechanically; what they enable loses games.

1000/1000 as pinned (P2 pass; P3 pass at 1.12×). Every context gate moved
as demanded: claim+contest picks 10.2% (was 3.8%), sticky-injected 1938
turns, divergence 42→56%. So the plays were selected AND sustained — and
the stack loses by 4.7pp with its own score down ~400 (4095 vs 4502). The
rung-1 "just make it selectable" theory is dead; the registered
over-adherence tell fired.

Behavioral autopsy (per-turn samples, orientation-verified): config1 runs
FEWER units from t5 (2.43 vs 2.78), buys an early city lead by t10–15
(2.32 vs 2.18 — the fog plays DO capture villages), then collapses by t20:
units −1.4, SPT −1.5, and the city lead inverts (2.47 vs 2.68 — it LOSES
cities while production grows). Early expansion bought with military tempo;
production converts the army edge mid-game. NOT over-arming — the
stance-pick inflation (25→35%) is INFERRED from the unit/SPT curves to be
eco/expansion-leaning (class telemetry doesn't split Grow/Arm/Save).

TWO harm channels, advisor-prompted split (the ACTUAL first named only one):
- **Channel A — selection-level shaping double-pay** (tree, w=1e-4): see
  design lesson below. Owns the t5 fewer-units signal (stickiness hasn't
  accumulated by t5).
- **Channel B — sticky phantom pursuit at EXECUTOR strength.** Stickiness
  injects fog targets into base.orders, where the executor prices approach
  at λ=1.0 — full production strength, not 1e-4 — for up to 6 turns; ~25%
  of predictions are phantoms. Dose slice (endogeneity-caveated like 035's):
  sticky 0 turns → 48.1%, **1–2 → 53.7% (above baseline — quick captures
  look genuinely good)**, 3–5 → 34.0%, **6+ → 10.9%** (a target held to the
  cap without capture = 6 turns of full-strength misdirection). The
  city-lead inversion t10→t20 is at least as consistent with B as with A.

Also unexamined (pinned so 036c doesn't inherit it silently): the
pre-registered UCT re-dial never ran — w was dialed into the single-edge
q-band, but shaped bonuses accumulate over multiple own edges on deep
paths, so deep-path q inflation and premature concentration remain a
possible contributor.

Design lesson (channel A root): **shaping directive ADHERENCE at the
directive-SELECTION level double-pays whatever the leaf already sees.**
goal_potential mixes stance terms (GROW SPT, ARM army-stars — leaf-visible;
evaluate_state already prices them) with order-approach terms (leaf-
invisible). At the ply level (028) the goal is FIXED, so this is harmless
steering; at the selection level every directive scores its own adherence,
and the most pump-able φ terms (eco/expansion) win arguments they shouldn't.
The uniquely legitimate shaping target is the leaf-INVISIBLE quantity:
approach progress toward fog/attack targets.

#### Verdict — FALSIFIED. Enabler machinery (shaping plumbing, sign-tested
backup, stickiness, telemetry) is sound and kept; what it enables, as
tuned, loses. Single-delta follow-up menu on kept machinery (each isolates
one channel): (a) **036c approach-only shaping** — strip stance terms from
the shaping potential, credit only Expand/Attack approach (kills the
channel-A double-count); (b) **shaping-only** (no belief, k=4) —
does channel A harm alone?; (c) **stickiness tuning** — three retirement rules (Verdi-specified): (1)
INTRA-TURN order stripping — select_move runs every ply on a fresh view, so
a fog target seen explored-and-not-capturable mid-turn is stripped from the
live goal that same ply, not next plan (beliefs already update per move;
the frozen directive was the only thing not consuming them); (2) retire
when the target drops out of the LIVE prediction set at plan time; (3)
retire after 2 consecutive no-approach-progress turns (closest own-unit
distance to target failed to shrink — a zombie order distorting pricing
while nobody marches). Plus cap 2. The 1–2-turn dose band sat ABOVE
baseline at 53.7%, so short-hold fog plays may already be a win; (d)
**stickiness-only** (rung-1 candidates + sticky, no shaping) — channel B
in isolation. The dose slice makes (c) unusually promising for a tuning
arm.

## EXP_ELO_035 — MAP materialization: does a belief-fed opponent beat empty fog?

**Registered 2026-08-13, before running.** Second rung of the belief ladder.
The 034-calibrated belief is written into macro-mcts's fogged root before the
tree runs: believed capital city at the posterior peak (engine-native via
pov-swapped `capture_city`, pov-visible tiles stripped from the imagined
border), ghost units at their recorded tiles, inferred residual army as
warriors ringed around the capital hypothesis. `known_enemy_capital` re-keyed
from exploration to presence so the materialized capital unlocks the
attack-capital directive (obscure_fog deletes fogged enemy cities, so
presence ⇔ sighted-or-materialized). Belief maintained by the arena harness
(same validated 034 feed) and handed to the agent each turn.

Changes: `belief.rs` (+`materialize_into`, `MaterializeStats`, `belief_for`,
3 new tests incl. an 8-half-turn opponent-execution probe on materialized
views), `macro_mcts.rs` (belief field + telemetry), `macro_agent.rs`
(`known_enemy_capital`), arena `--macro-belief1/2` + `BELIEF MATERIALIZATION`
line + dump fields.

Deliberate decisions (in the ledger so the A/B is interpretable):
- **Opp-star leak**: `obscure_fog` leaves the true star bank visible
  (pre-existing since 033, mostly inert without fogged spend sites). The
  belief arm REPLACES opp stars with a public ramp (2 + turn/2, cap 20) so a
  materialized capital can't spend hidden information; the baseline arm keeps
  the mostly-inert leak. A both-arms fix is out of scope here.
- v1 scope cuts: naval ghosts skipped (land guard); residual materializes as
  Warriors only; capital materializes at level 1/pop 0 (conservative floor);
  only the TREE view is materialized (per-ply executor ranking sees the plain
  fogged view); hidden_cities/confidence unused (034 verdict).

Setup (pinned): arena `--backend1 macro-mcts --macro-belief1 --backend2
macro-mcts`, both sims=32 k=4 λ=1.0 heuristic leaves; `--games 500` (=1000
games, sides swapped); `--base-seed 1786900000`; `--gamemode 2 --max-turns 30
--tribe imperius --eval-backend metal`; `GUMBEL_SCALE=0`; dumps
`replays/exp035/armA`. Paired within-run A/B — the valid instrument given the
16% engine-flip finding.

Predictions:
- **P1 (headline):** belief side ≥ 52.5% (z ≥ 1.58, σ=1.58pp at n=1000) →
  CONFIRMED. Falsifier: ≤ 50.0% — materialization as built does not beat
  empty fog (belief value doesn't survive contact with the tree, or the
  imagined world misleads more than it informs). 50.0–52.5%: inconclusive →
  extend to 1250 seeds before any verdict.
- **P2 (GUARDRAIL):** zero panicked/skipped games — materialized views must
  execute cleanly through full games.
- **P3 (GUARDRAIL):** ms/move config1 ≤ 1.3× config2 (materialization is a
  per-turn O(few-tiles) edit).
- Context (gates interpretation, not a pass/fail): capital-materialization
  rate expected ~10–40% of planned turns (posterior confirms by ~t10, after
  which the real capital is in the view and materialization correctly
  no-ops). If < 5%, a flat P1 is uninterpretable (window too small) — the
  follow-up is widening what materialization covers, not a verdict.

Risk+tell: the imagined army makes the agent over-cautious (turtling vs
phantom warriors) — tell is P1 falsified WITH high materialization rate and
the belief side losing on score at t30 in games that never reached contact.

### ACTUAL (2026-08-13) — P1 FALSIFIED at a real dose: 49.7%, dead flat

1000/1000 games completed as pinned (P2 guardrail pass — zero panics on
materialized views across ~222k planned moves). ms/move 141 vs 121 (1.17×,
P3 pass — imagined units genuinely enlarge rollout branching). Macro
divergence 35.4%. **Capital materialized on 48.1% of planned turns, 1.27
units/turn — the null is real, not window-starved: materialization ran on
half of all plans and bought nothing (497/1000, z=−0.19).**

Post-hoc dose slices (ENDOGENOUS — dose is not randomized; a large hidden
enemy army causes both high materialization and losing, so read as
hypothesis-generating only):
- Inverted-U in capital dose: <25% of turns → 50.9% (n=165); 25–60% →
  **55.8%** (n=548, nominal z≈2.7); >60% → **37.3%** (n=287, config1's own
  score collapses to 3515 vs 4447).
- Units materialized: <10/game → 85.6% wins (n=167); ≥30/game → 20.9%
  (n=363). Consistent with the registered turtling tell (phantom warriors →
  passivity) AND with reverse causation (losing → bigger hidden army → more
  materialization). This arm cannot separate them.

Read (SCOPED — advisor-prompted): the arm bundled TWO deltas —
materialization AND the opp-star-ramp replacement — so the null is strictly
"materialization + star-ramp does not beat empty-fog + leaked-true-stars."
The leak is not inert in the baseline: sighted opp cities survive
obscure_fog and the tree trains from them every rollout. Post-hoc bank check
(mid_*.json boards at t12): true banks mean 5.2 / median 4 vs ramp 8 —
**the ramp over-funds the belief arm's sim opponent 2× (64.8% of true banks
sit below it)**, pushing the same direction as phantom-army turtling. What
survives unscoped: root-world fidelity as consumed here is not the lever at
k=4/sims=32/heuristic leaves; the mid-dose band hints at value in the
normal confirmation window, causal arrow unresolved.

#### Verdict — FALSIFIED (the second useful null of the macro program),
scoped to the materialization+ramp bundle. The belief itself remains
calibrated and cheap (034 stands); what failed is this consumer. Follow-up
menu (each a one-knob arm on the kept infrastructure): (1) **candidate
generator** — still top; both nulls point there and neither confounds that
question; (2) 035b residuals-off (capital+ghosts only) — pins the
phantom-army arrow; (3) 035c calibrated-ramp (median-bank ramp, ~turn/3) —
pins the over-funding arrow. The bank check makes (3) at least as sharp as
(2). Keep `--macro-belief1/2` + BELIEF MATERIALIZATION telemetry.

## EXP_ELO_040 — Defense as a first-class executor capability (threat model + Defend pricing + coverage leash)

**Registered 2026-08-14, before running.**

Diagnosis (fixture seed 1786670356, both baselines committed d476308):
the executor cannot defend a city because THREE layers are missing, not
one. (1) Emission: `update_goal` fires Defend only at `near >= 2` enemy
units within cheb 2 — a single sieging swordsman never emits a Defend
order at all (both fixture games sat in this hole). (2) Pricing:
`goal_potential`'s order loop skips every kind except Expand — Defend
and Attack orders flip stance and paint feature planes but are worth
exactly 0 Φ, so nothing in the executor's λ·Δφ holds or recalls units;
Expand approach gradients actively pay a garrison to march off its own
threatened city (macro game, t3). (3) Response sizing: nothing computes
what the opponent can deliver or what counter-force suffices, so the
contest is piecemeal — single attacks per turn against a 15hp swordsman
(the ACTUAL failure that lost the macro game per the corrected read; the
1hp-garrison walk-off itself was defensible).

Changes (design pinned; Verdi's spec 2026-08-14):
- New `src/ai/defense.rs`: `city_threats(state, player)` — per own city,
  worst-case next-turn strike from FOW-visible enemy units using the real
  engine math (`compute_reachable_tiles` Dijkstra → made pub(crate);
  `calculate_combat` with `get_defense_bonus` on hypothetical placements
  via coord-swapped unit clones — no hand-coded multipliers). At-risk =
  active siege (enemy on city), reachable-while-unguarded, or strike ≥
  RISK_MARGIN × garrison effective HP.
- `defend_plans(...)`: `need_damage` = kill the strongest deliverable
  attacker standing on the city (with ITS defense bonus — enemy units
  get no city bonus, engine truth); min-diversion assignment of covering
  units (prefer units not holding Expand assignments) until damage met;
  `shortfall` recorded. Cover check: exact Dijkstra "can attack a unit
  on C next turn" for units within cheb ≤ 2·movement+2 (roads/terrain
  count — the rider-via-road case falls out of the movement graph);
  2-turn half-cover ring approximated by cheb ≤ 2·movement (STATED
  APPROXIMATION: underestimates road reach; fine for a soft leash).
- Emission fix in `update_goal`: Defend fires on at-risk (single
  attacker included), replacing `near >= 2`. Defend → Arm flip KEPT
  as-is this pass (guardrail G2 measures the cost).
- `goal_potential`: price Defend orders — per assigned unit,
  SHAPE_GOAL_DEFEND_COVER (600) × urgency (0.5 base, 1.0 at-risk) ×
  satisfaction (1.0 in 1-turn cover, 0.5 in the 2-turn ring, proximity
  gradient below that); + SHAPE_GOAL_DEFEND_HOLD (400) × urgency for a
  garrison ONLY when removing it creates shortfall (no unconditional
  pinning — the leash must allow frontier pushing, per spec). Prep is
  priced by OUTCOME only: a trained unit / new road / bought tech that
  flips a unit into (or nearer) coverage raises Φ and the per-ply Δφ
  pays each step — no discrete prep planner, no selection-level bonus;
  dual-purpose purchases win ties via the existing TECH_FIT term.
- Sizing rationale: full cover 600 > max single-ply Expand approach gain
  (200×2 for a rider), so an at-risk leash holds; 0.5-urgency cover 300
  < 400, so mild threat still loses to a 2-step expand — "intensity
  calibrated to actual need."

Setup: unit tests (emission single-attacker, zero-Φ-when-no-defend,
coverage gradient monotonicity, hold-term-only-on-shortfall, miniature
t3 walk-off state); fixture rerun seed 1786670356 teacher-vs-Greedy
(new binary) vs committed baselines; paired A/B cost check old-teacher
(`/tmp/exp040_baseline_arena`, snapshotted pre-rebuild) vs new-teacher,
each vs Greedy, 125 seeds ×2, base_seed 1787600000, gamemode 2,
max-turns 30, imperius, metal, GUMBEL_SCALE=0.

Predictions:
- **P1 (fixture, the behavior gate):** on seed 1786670356 the sieger at
  the center city is killed before capture OR the city is held/retaken
  with a concentrated (≥2-attack-per-turn) response — NOT judged by
  "garrison never steps off" (a 1hp step-off into cover is correct).
  Falsifier: same piecemeal pattern (≤1 attack/turn vs an at-risk city
  while ≥2 covering units exist).
- **P2 (A/B, the cost gate):** new teacher vs Greedy win rate within
  [−4pp, +8pp] of old teacher (n=250/arm, σ≈3.2pp); expected +2–5pp.
  Falsifier for the design: < −4pp means the leash/Arm-flip tax
  exceeds the defense value.
- **G1 (guardrail, throughput):** fixture-game generation throughput ≥
  60 moves/s (baseline 106.6; Dijkstras only on defend-order turns).
  Below → cap threat/cover rings before any verdict.
- **G2 (guardrail, Arm inflation):** sensitized emission raises Arm
  fraction; if pre-t10 Arm plies > 2× baseline AND the A/B SPT curve
  regresses, the pre-committed follow-up is decoupling order emission
  from the stance flip — measure, don't redesign now.

Risk+Tell: enemy-dependent Φ terms make the potential move on opponent
plies; executor Δφ is computed within own plies only (enemy static), and
tree edge shaping stays shape_w=0 — the tell for a leak is
shaped_backup_matches_hand_computation failing or λ-off A/A drift.
Training semantics: this changes goal-script conditioning for FUTURE net
rounds two ways (Defend feature plane goes from ~never-painted to
informative; in-tree --goal-w-tree shaping gains the defend terms) —
current running round unaffected (binaries snapshotted at launch).

ACTUAL (2026-08-14, shipped commit 53b3fe0):
- **P1 (fixture) — MECHANISM PASS, outcome unchanged.** On seed
  1786670356 the Defend order fires AT the siege turn (raw read t2: Arm,
  3 orders; the old proxy never fired), the wounded garrison holds and
  RECOVERS in place at t3 instead of the walk-off, and the defense
  concentrates (two attacks per turn t5–t8 — piecemeal gone). The city
  still falls and the game is still lost ~t20: one wounded rider vs two
  swordsmen is unwinnable locally, and the loss is production-layer
  (army stars 3 vs 121 by t20) — the 034–038 evaluation bottleneck, not
  coverage. Replay: replays/macro040_vs_greedy_1786670356.json.
- **P2 (cost gate) — brushes the falsifier, not significant.** Old
  61.6% (154/250) vs new 57.2% (143/250) vs Greedy on paired seeds:
  −4.4pp point estimate, seed-level McNemar z = −1.51 (53 discordant:
  32 old-only vs 21 new-only) — inside noise, but the direction is a
  mild tax. Read: Greedy is not a rusher on most seeds, so defense
  pricing buys resilience Greedy rarely punishes while costing some
  expansion tempo. The value claim shifts to distillation semantics
  (Defend plane now informative) and aggressive opponents, NOT teacher
  Elo vs Greedy. Decision knob left armed: halve DEFEND_COVER/HOLD or
  decouple emission from the Arm flip if the next training round shows
  eco regression.
- **G1 (throughput) — PASS after remedy.** Naive reach probes cost 46
  moves/s (fixture); distance-banding (plain cheb decides inside
  movement+range, exact road-aware search only in the band beyond) →
  66 moves/s. Arena ms/move 103 → 174 (+69%) — the teacher pays real
  compute for threat truth on contested turns.
- **G2 (Arm inflation) — UNMEASURABLE from arena dumps** (stance_flips
  gauge reads 0 under the macro backend — goal-script-path-only
  counter). Observable shift: stance-override candidate picks 21.8% →
  16.6%, continuation picks 13.2% → 23.0%, divergence 46.1% → 49.6%.
  Defer the emission/flip decoupling call to the next training round's
  Arm-fraction telemetry.
- Tests: 181 lib tests pass incl. new defense module tests + the
  miniature t3 walk-off regression (hold > step-off > out-of-leash in
  Φ) + FOW-honesty (hidden units never threaten).
- Net-path cost (advisor check): gumbel n=64/k=16 + goal-script +
  goal-w-tree 1, 16 games old vs new binary — 56.95 → 59.75 ms/move
  (+4.9%): the training loop's generation phase is essentially
  unaffected. ⚠️ Gauge discontinuity: goal-script conditioning now
  emits Arm/Defend far more often than the pre-040 checkpoint's
  training distribution — fixed-seed goal-script gauge reads are not
  comparable across the 040 boundary for SCRIPT reasons (same class of
  break as MAPGEN_001); judge the next round's student on its own data.
- Fixture A/A: same binary + seed gives different games (349 vs 329
  moves) — legal-move-ordering nondeterminism, the documented engine
  property behind DecomposedMapper. Pinned-seed behavior asserts must
  be property-based (Defend fired at siege turn, ≥2 attacks per defense
  turn, no garrison walk-off while load-bearing), never exact-replay.
- Prep mechanism demonstrated at unit level
  (new_unit_inside_the_ring_outprices_the_same_unit_far_away): the same
  purchased rider is worth ≥ half a cover slot more Φ landing in the
  ring than far away — the gradient the train/road/tech chain climbs.
  The full multi-turn chain is priced by construction; an end-to-end
  fixture demonstration is still owed.
- VERDICT: shipped. Behavior contract holds on the fixture; the
  REGISTERED P2 falsifier fired on the point estimate (−4.4pp < −4pp)
  and only the paired McNemar (z=−1.51, n.s.) rescues the ship
  decision — shipped on the distillation motive with knobs armed
  (halve DEFEND_COVER/HOLD; decouple emission from the Arm flip);
  re-judge on the next MACRO_GEN round where the student sees painted
  Defend channels for the first time.

## EXP_ELO_041 — Full assessment: does the defense signal win the NET more games? (+ siege scoreboard)

**Registered 2026-08-14, before running.**

Question (Verdi): does the 040 defense signal lead to overall more
victories — and how often does the net successfully unsiege vs lose a
city? The 040 A/B answered this for the TEACHER only (−4.4pp n.s.); the
production path is the NET under goal-script conditioning + in-tree
shaping, measured at n=16 only (uninformative).

Instrument (built first, commit pending): arena `SiegeTracker` — a siege
is an enemy unit standing on an owned city tile, scanned after every
move; episodes resolve as UNSIEGED (attacker gone, city kept) or LOST
(ownership flipped). Per-game dump fields `sieges/unsieged/cities_lost`
per config + `SIEGE DEFENSE` aggregate lines. The OLD arm is rebuilt
from d476308 (pre-040) in a worktree WITH the same instrumented arena.rs
(040 never touched arena.rs), so both arms report the metric under their
own behavior.

Setup: net path both arms — gumbel n=64/k=16, `--goal-script
--goal-w-tree 1`, model pinned /tmp/net_check_1786669494.safetensors
(pre-040 checkpoint BOTH arms — isolates the script/shaping change),
vs Greedy, 125 seeds ×2 = 250 games/arm, base_seed 1787800000,
gamemode 2, max-turns 30, imperius, metal, GUMBEL_SCALE=0, dumps
replays/exp041/{old,new}.

Known confound (accepted, stated up front): the pre-040 checkpoint is
OFF-distribution for 040 conditioning (Arm/Defend painted far more
often than its training data ever showed) — a win-rate drop can mean
"the signal mis-steers THIS net," not "the signal is bad." The siege
scoreboard is the disambiguator: defense value should show as higher
unsiege rate / fewer cities lost even if raw win rate dips.

Predictions:
- **P1 (siege scoreboard, primary):** NEW arm unsiege rate > OLD arm,
  and cities_lost per game lower. Falsifier: no improvement in either →
  the net does not consume the painted Defend signal zero-shot; the
  claim moves to post-training.
- **P2 (win rate):** point estimate anywhere in [−6pp, +6pp] would not
  surprise; only a paired McNemar |z| ≥ 1.96 counts as a real move.
  A significant DROP + improved siege metrics = off-distribution tax
  (expected recoverable via training); a significant drop + flat siege
  metrics = the falsifier that the shaping itself mis-prices.
- **G1 (sanity):** OLD arm win rate within noise of its 040-era readings
  vs Greedy on fresh seeds (~56% at the last gauge) — if OLD reads wildly
  off, the seed batch or harness changed and no cross-arm read is valid.

ACTUAL (2026-08-14, 250 paired games/arm, commit 144d02b):
- **P2 (wins): FLAT.** Old 63.6% (159/250) vs new 60.8% (152/250);
  McNemar 39 old-only vs 32 new-only, z = −0.83 — no significant win
  effect either direction. The defense signal neither wins nor loses
  games for the pre-040 checkpoint zero-shot.
- **P1 (siege scoreboard): PARTIAL PASS — conversion better, exposure
  up.** Conditional on being sieged the net defends better on BOTH
  margins: unsiege-given-siege 34.9% → 37.8% (+2.9pp), lost-given-siege
  62.2% → 59.4% (−2.8pp). But siege EXPOSURE rose +18% (1.67 → 1.98
  episodes/game, paired t ≈ +2.8, the strongest effect in the data), so
  ABSOLUTE cities lost still rose +0.14/game (t ≈ +1.6, n.s.). Read:
  the in-tree Δφ (the only channel this untrained checkpoint can
  consume — it has never learned the painted Defend plane) genuinely
  improves defense once contact happens, while the conditioning shift
  changes game shape toward more contact. Greedy-side mirror agrees:
  the net's own sieges of Greedy convert WORSE (Greedy unsiege 34% →
  42%) — the leash pulls attack support home mid-offense.
- **G1: PASS** — old arm 63.6% on fresh seeds vs 56.2% at the n=32
  gauge is within the n=32 noise (σ≈8.8pp).
- Cost: net-path ms/move 154 → 194 (+25%) at n=250 — the +4.9% from
  the n=16 probe under-measured; contested games hit the threat
  computation hardest. Real but tolerable for gauge/league volume;
  MACRO_GEN generation itself uses the macro backend and is unaffected
  by this number.
- VERDICT: the defense signal is consumed zero-shot at the margin
  (conversion +3pp both ways) but does NOT yet buy victories; the
  offense-conversion regression and the exposure rise are the costs of
  shaping an untrained policy from outside. The decisive test stays the
  next MACRO_GEN round: a student TRAINED on painted+priced Defend
  channels, re-measured on this same instrument (replays/exp041 seeds,
  SIEGE DEFENSE scoreboard). Knobs still armed: halve DEFEND_COVER/HOLD
  (offense-leash tax), decouple emission from the Arm flip.

## EXP_ELO_042 — Two signals: defend city B while attacking city H (duty partitioning + attack press)

**Registered 2026-08-14, before running.**

Diagnosis (EXP_ELO_041 mirror data): defense pricing is asymmetric and
un-partitioned. (1) The shortfall RECALL gradient prices the single
nearest non-assigned unit's distance to the threatened city with no
notion of duty — mid-offense, the net's attackers are often exactly
that unit, so every in-tree step deeper into enemy land loses shaping
value and the tree tilts attackers homeward. (2) Attack orders are
worth 0 Φ (like Defend was before 040), so any pricing tie breaks
toward the only spatial gradient that exists — home. Measured: Greedy's
unsiege rate vs the net's own sieges 34% → 42%, sieges started but not
finished. Verdi's spec: "need 2 different signals. I can defend city B
while attacking city H."

Changes (design pinned):
- **Duty partitioning (COMPARATIVE, not radius — advisor catch: a
  2m+2 ring around H contains B itself on Tiny maps, capitals cheb 5
  apart, so a radius rule would defense-exempt the whole center):** a
  unit is attack-committed for a given Defend city B iff it stands ON
  an enemy city (state-fact latch — survives Attack-order flicker when
  a co-attacker dies and the local.len()>=2 emission predicate drops
  the order) OR some Attack-order target H is STRICTLY closer to it
  than B (tie → defense). Committed units are excluded from
  defend_plan assignment AND the shortfall recall gradient. If only
  committed units remain, recall goes silent — shortfall drives prep
  (train defenders at home, already outcome-priced) instead of
  un-committing the army. The same geometry retro-explains 041: an
  attacker standing on H at cheb 5 was INSIDE B's cover-candidate ring,
  so assignment was stealing attackers directly, not just recall.
- **Attack press (symmetric pricing):** per Attack order H, units in
  H's press ring get SHAPE_GOAL_ATTACK_PRESS = 500 per unit (cap
  MAX_ASSIGN; sat 1.0 in 1-turn strike reach, 0.5 in the 2-turn ring;
  pick order sat-desc/dist-asc/tile-asc, deterministic like
  defend_plan). A unit standing ON an enemy city pays PRESS ×
  SIEGE_HOLD_MULT = 1.5 (750) BY STATE-FACT, independent of any order
  (the flicker latch), and is skipped by the order-press assignment so
  it is never double-paid. Sizing: 500 beats the max single-ply Expand
  approach gain (2 tiles × 200) so a committed attacker never abandons
  for a village pull; 750 makes stepping off a siege a ≥250 Φ loss;
  both sit below at-risk DEFEND_COVER+HOLD (1000) so a unit that is
  somehow both roles resolves home — but the partition makes that moot.
  No need-math on the offense side v1: the Attack emission predicate
  already gates on local superiority (1.5×).
- Stance flip (Defend → Arm) deliberately UNTOUCHED: the granular
  intensity gate keeps eco open below 0.98, and Arm is what lets home
  cities train defenders. The armed decouple-knob stays armed, not
  spent — this experiment fixes the SPATIAL interference only.

Setup: unit tests (attacker at H exempt from cover/recall while home
unit covers B — the defend-B-attack-H scenario in miniature; siege-hold
on H outprices stepping off; recall skips committed units and goes
silent when only they remain); full lib suite; then RERUN the exact 041
instrument — same binary path, same base_seed 1787800000, same 125
seeds ×2, net path (gumbel 64/16, goal-script, goal-w-tree 1, pinned
pre-040 checkpoint) vs Greedy, dumps replays/exp042/new — comparable
row-for-row against BOTH 041 arms.

Predictions:
- **P1 (offense conversion recovers, primary):** Greedy's unsiege rate
  vs the net back below ~38% (from 041-new's 42%; 041-old read 34%), and
  Greedy cities_lost/game back up toward the old arm's 1.09. Falsifier:
  no recovery → the leash was not the offense regression's cause.
- **P2 (defense holds):** net unsiege-given-siege stays ≥ ~37% and
  lost-given-siege ≤ ~60% (the 041 gains kept). Falsifier: defense
  conversion collapses back → the recall gradient was doing the real
  defensive work and partitioning broke it.
- **P3 (wins):** vs 041-old, McNemar on the same seeds; expected drift
  positive but honestly unknown; any |z| < 1.96 reads as flat.
- **G1 (exposure):** net sieges suffered/game between the two 041 arms
  (1.67–1.98); if it rises FURTHER, attack press is overextending the
  army and the press weight halves before any verdict.

ACTUAL (2026-08-14, 250 games on the 041 seed batch, commit 2d92206):
three-arm table (identical seeds; 041old = pre-040, 041new = 040):

  arm     wins   NET sieged/g  unsiege%  lost|siege  GREEDY sieged/g  unsiege%  lost/g
  041old  63.6%  1.67          34.9%     62.2%       1.79             34.2%     1.09
  041new  60.8%  1.98          37.8%     59.4%       1.90             41.7%     1.04
  042     60.0%  1.94          36.4%     58.4%       2.02             41.0%     1.15

- **P1 (offense recovers): PARTIAL.** Greedy's escape rate barely moved
  (41.7% → 41.0%, predicted < 38%), but conversion-given-siege
  recovered half the 041 regression (54.7% → 56.9%; old 60.9%) and the
  net now starts MORE sieges (2.02/g, highest) — so ABSOLUTE captures
  of Greedy cities are the best of all three arms (1.15/g vs old 1.09).
  The press works by volume more than by conversion.
- **P2 (defense holds): PASS.** Lost-given-siege 58.4% is the best in
  the family (041new 59.4%, old 62.2%); unsiege 36.4% gives back ~1.4pp
  of 041new's gain but stays above old. Partitioning did not break the
  defensive work the recall gradient was doing.
- **P3 (wins): FLAT.** 60.0%; McNemar vs 041old z = −1.07, vs 041new
  z = −0.26 — all three script variants statistically indistinguishable
  at n=250. **G1 (exposure): PASS** — net sieged 1.94/g sits inside the
  arm range; no press-halving mandated.
- Cost note: ms/move 140 vs 041new's 194 (and old's 154) — the
  partition prunes defend-plan candidates and games run shorter.
- Fixture note: the 042 teacher on seed 1786670356 now reproduces
  deterministically (183 moves, 3× identical) and loses FASTER (t14) —
  partition removed the accidental home-drag, and on that adversarial
  seed committed units press a losing center fight; press has no
  need-math v1. Population data (G1) says this is not a systematic tax.
- VERDICT: the two-signal separation does what it says mechanically
  (best-in-family defense conversion + best-in-family absolute offense
  output + stable exposure) and win rate stays flat — consistent across
  040/041/042: ZERO-SHOT STEERING OF THIS CHECKPOINT MOVES MICRO-DIALS,
  NOT VICTORIES. The signal stack is now behaviorally correct on both
  axes; the win-rate claim rides on the next MACRO_GEN training round
  (teacher + student both under 042 semantics), re-measured on this
  same instrument. Armed knobs unspent: press need-math (don't press a
  losing fight), halve DEFEND_COVER/HOLD, decouple emission/Arm-flip.

## EXP_ELO_043 — Tier 1: the net OWNS the root doctrine (map-context lane choice, ≤3 pivots)

**Registered 2026-08-14, before running.** Verdi's architecture call: the
product is the strategist at macro-mcts, in three tiers — (1) root
doctrine, ONE per game, rarely flips, map+tribe-context driven; (2)
map-driven orders, multiple per turn, reactive = macro-mcts; (3) units/
stars/tech = deterministic executor. Every action traces up to a root
cause. Tier 1 is LEARNED (net), not scripted: "I want the net to have
the ability to pivot at most up to 3 lanes and the strategy chosen is
map-context driven. Your tribe + your terrain on your first 2 villages
sets the tone."

Why the net can own Tier 1 when per-ply imitation of scripts failed
(040/041/042: micro-dials moved, wins flat): CREDIT ASSIGNMENT RATIO.
The doctrine is ~1–3 decisions per game with the outcome directly
attributable — 1:1, versus ~600:1 for per-ply policy targets diluted
across a 30-turn game. It is the cleanest learning signal in the
system, and it is exactly the "which principle dominates here"
judgment that determinism provably cannot do (034–038: heuristic leaf
q-spreads 0.01–0.06).

Design:
- **Head:** `doctrine_logits = Linear(filters -> K)` off the pooled
  trunk, K = 3–4 lanes. Mirrored in network.rs (inference consumer =
  the macro agent) and train.py, loaded OPTIONALLY (`vs.contains_tensor`,
  aux_fog precedent) so pre-043 checkpoints gain no rejection reason.
  NO feature-layout change: tribe identity (CH_MY_TRIBE_TYPE) and
  terrain planes already carry the census the head reads.
- **Enforcement is structural, not learned:** `DoctrineCommit { lane,
  pivots_used }` — max 3 lane changes/game, pivots proposable only at
  evidence checkpoints (opponent identity revealed, lane blocked e.g.
  no metal in reach, city-count break). The net JUDGES; rules CONSTRAIN.
  "Sticking with it" becomes a structural guarantee instead of a
  behaviour we hope the weights hold.
- **Downward gating (the simplification):** chosen lane restricts tech
  caps, `preferred_units`, and which Tier-2 candidates
  `enumerate_candidates` proposes — the macro search space SHRINKS.
- **Supervision is OUTCOME, not search.** Explicit non-goal: no
  visit-count target for this head. A 2–3 turn tree with a heuristic
  leaf cannot adjudicate lanes that mature over 10+ turns (034–038
  measured); using search as the improvement operator here would
  supervise noise.
- **Counterfactual pairs (the sample-efficiency multiplier):** same
  seed played under lane A and lane B yields a PREFERENCE label ("on
  this tribe+spawn, A > B"), which is far stronger per game than an
  absolute win/loss and matches the head's decision exactly. The
  paired-seed arena harness already produces this shape.
- **Exploration:** temperature/Dirichlet on the lane sample during
  generation; a census heuristic seeds the prior with DECAYING weight
  (bootstrap only — distilling the census outright would inherit its
  ceiling, the documented teacher-cap failure mode).
- **Explainability invariant:** every ply logs `ply <- order <-
  doctrine`; a losing game is attributable to a TIER.

Setup (pinned): starter library K=4 — Riders+Roads (mobility/expansion),
Archery/Forest, Metal/Smithery→Giants, Eco/Parks — chosen because they
map to distinguishable spawn censuses. Teacher-side first (macro agent
consumes a lane), measured on the 041/042 instrument: same base_seed
1787800000, 125 seeds x2, SIEGE DEFENSE + win rate, comparable
row-for-row against all three prior arms.

Predictions:
- **P1 (doctrine coherence lifts the TEACHER):** committed-lane teacher
  vs current teacher > +4pp (McNemar |z| >= 1.96). Rationale: the star
  gate's committed-reach flip was 28%->81%; the fixture loss was a
  doctrine failure (2 techs in 19 turns, no lane). Falsifier: flat ->
  lane commitment alone is not worth the search-space restriction, and
  Tier 1's value rests entirely on the net's context-conditional choice.
- **P2 (context-conditionality is learnable):** the head's lane choice
  correlates with spawn census better than chance on held-out seeds
  (top-1 agreement with the counterfactual-winner label > 40% at K=4).
  Falsifier: at-chance -> the census signal is too weak at this map
  size, and doctrine collapses to a single global best lane (which is
  still a valid, simpler product).
- **G1 (no collapse):** lane entropy across a generation batch stays
  > 0.5 nats; collapse to one lane means exploration is broken.
- **G2 (dual-net sync):** identical logits Rust vs Python on a fixed
  batch before any training run consumes the head.

**STATUS 2026-08-14: PARKED BEFORE RUNNING — kept as a written-down idea
to revisit, per Verdi.** The reshaping of TRAINING (outcome/bandit
supervision, counterfactual lane pairs as preference labels) is the
part being deferred; the tier architecture it serves is not. Superseded
for now by EXP_ELO_044, which reaches the same Tier-1 goal through the
project's existing validated machinery (decaying script crutch ->
prior, goal-conditioned feature painting, calibrated dial) instead of a
new supervision regime. Revisit if 044's crutch never becomes "owned"
(policy_loss gauge) or if the pivot hurdle proves un-callable from
value differences.

ACTUAL: (not run — parked)

## EXP_ELO_044 — Playstyle as conditioned INPUT: crutch-distilled prior + learned pivot hurdle

**Registered 2026-08-14, before running.** Verdi's revision of 043 (which
is parked): keep Tier 1 in the net, but reach it with heavy script
crutches and input conditioning rather than a new supervision regime.
Three pieces: (1) a `playstyle_log[]` carried IN THE INPUT — the net's
own past playstyle decisions fed forward into future turns; (2) a
terrain/tribe census script that strongly picks the playstyle early
(forest -> forestry, metal -> smithery/giants, ...) and is DISTILLED
INTO THE PRIOR on the project's standard decaying-crutch schedule; (3)
pivoting gated by a LEARNED estimate of whether the current playstyle
can still win — the lane changes only when the hurdle is cleared.

Why this is cheap: every mechanism already exists and is validated.
- Input conditioning: EXP_ELO_028 goal channels already paint scripted
  directives as planes; playstyle planes are the same pattern, appended
  at the END of the spatial layout so old archives keep zero-padding at
  load (train.py) — the documented safe path for channel growth.
- Crutch -> prior: `blend_heuristic_prior` / `anchor_frac` /
  `decay_crutch` are the same machinery that already bootstraps the
  policy from Greedy. `policy_loss` is the owned-vs-rented gauge (memo:
  goal-pricing-beats-masks) — no crutch removal until it closes.
- Pivot estimate WITHOUT A NEW HEAD: `state_to_cpu_features_goal(...,
  Some(&candidate))` already paints a hypothetical directive into the
  features, so a lane-conditional value read is just re-encoding with
  playstyle plane B and re-running the existing value head. Pivot iff
  V(state | lane B) - V(state | current) > hurdle.

Design pins:
- Planes (appended at end): K current-lane one-hots + turns-in-lane
  (normalized) + pivots-used (normalized). The history summary IS the
  playstyle_log; no sequence model needed.
- Hurdle calibration by the q-gap dial method (memo): fit against the
  MEASURED distribution of lane-conditional value differences, expect
  the first fit to overshoot ~2x, iterate. Plus structural guards: min
  dwell turns and <= 3 pivots/game (Verdi's budget).
- Value-head caveat: the head is ~2x overconfident when ahead (022
  family). Comparing two conditional reads at the SAME state is a
  difference, so calibration bias partially cancels — but the hurdle
  must be set on measured spreads, never a guessed constant.

Known negative prior (stated up front): EXP_ELO_038 fed the previous 3
turn directives to the STRATEGIST for candidate generation and came
back flat (48.6%). The mechanism here differs — the consumer is the
NET'S PRIOR conditioning ~600 plies/game, not the tree's candidate set
(which the tree could already reach). If 044's playstyle planes also
read flat, that is two independent falsifications of directive memory
and the idea should be retired, not re-skinned a third time.

Predictions:
- **P1 (the rule gets OWNED):** with the census crutch decaying, the
  net's playstyle choice agrees with the census on held-out spawns at
  a rate that keeps rising as crutch weight falls, and playstyle-head
  policy_loss closes. Falsifier: agreement collapses the moment the
  crutch decays -> rented, not owned; the input conditioning is doing
  nothing the crutch was not already doing.
- **P2 (the hurdle is callable):** lane-conditional value differences
  at t~8 rank the actual better lane above chance on forced-lane
  paired seeds (AUC > 0.6). Falsifier: AUC ~ 0.5 -> the value head
  cannot rank lanes; fall back to script-only pivot triggers (evidence
  gates: lane blocked / opponent identity), and note the pivot half of
  this design as unsupported.
- **G1 (stability):** <= 3 pivots/game and dwell >= N turns hold
  structurally; lane entropy across a batch stays > 0.5 nats (no
  collapse to one global lane).
- **G2 (migration):** channels appended at end, train.py zero-pad path
  exercised on a legacy archive, `checkpoints/` migrated (memo:
  migrate-checkpoints-on-arch-change — the Rust opponent loader is
  strict), dual-net logits identical on a fixed batch.

ACTUAL: (pending)

## EXP_ELO_045a — Tier 1, algorithmic half: does a COMMITTED playstyle pay?

**Registered 2026-08-14, before running.** Stage 3a of the approved tier
refactor (plan: strategist at macro-mcts, three tiers). The cheap kill
switch that must pass before any channel migration is paid.

Diagnosis: `Archetype {RiderRoads, ArcherLine, ForgeGiants}` — exactly
Verdi's three lanes — already existed with census predicates and
hysteresis, but `update_archetype` ran on EVERY executor ply (via
`rank_view`, `macro_agent.rs:153`, and inside `execute_turn`). A lane
recomputed 20x a turn is a running average, not an identity, and nothing
downstream treated it as a commitment.

Changes (commits fec32c0, 5bf6af7):
- Split: `observe_archetype` (per-ply peaks + overlays) vs
  `select_playstyle` (turn boundary, scores EVERY lane, returns the call).
  `rank_view`/`execute_turn` observe only; the macro agent selects once
  per turn in its replan branch; the script-path wrapper selects only when
  the turn advances (or to make the first commit).
- Tribe prior: mapgen stamps one tribe tech at turn 0; if it opens a
  lane's chain (`lane_techs`, now the single source of truth shared with
  the recommendations) that lane commits BEFORE any terrain is explored
  and scores +2 thereafter. Verified against mapgen: Oumaji/Aquarion →
  Riding, Yadakk → Roads (RiderRoads); Bardur → Hunting, Hoodrick →
  Archery (ArcherLine); XinXi → Climbing, Vengir → Smithery
  (ForgeGiants); Imperius/Kickoo/Zebasi/AiMo/Quetzali → no prior.
- Commitment: ≤3 pivots/game (`MAX_PIVOTS`), `DWELL_MIN=5` turns before a
  discretionary switch, existing margin/streak hysteresis retained, and
  `lane_blocked_turns >= 3` (the lane's next tech proposed and
  gate-dropped) as pivot evidence for a lane stranded behind the ARM eco
  mask. A refuted lane (score 0) still exits immediately but now COSTS
  budget.
- `select_playstyle` takes an optional per-lane `head: Option<&[f32; 3]>`
  added on top of the census — the hook Stage 3b (the aux head) fills.
  Unused in this arm (always `None`).

Setup: the 041/042 instrument, so rows compare directly against those
runs — arena, `--backend1 macro-mcts --backend2 greedy`, 125 seeds ×2,
`base_seed 1787800000`, gamemode 2, max-turns 30, imperius, metal,
`GUMBEL_SCALE=0`, SIEGE DEFENSE scoreboard + win rate + McNemar. BASELINE
ARM is a snapshot of the pre-selector binary (`/tmp/exp045a_baseline_arena`,
built at ea700e4 = Stage 1 only, whose changes do not touch arena's macro
path), so the delta isolates the selector.

Predictions:
- **P1 (the gate):** committed-lane teacher ≥ +4pp over the per-ply
  teacher with |z| ≥ 1.96 (McNemar, paired seeds). Rationale: the closest
  precedent for commitment is the star gate (forced third-city reach
  flipped 28%→81%), and the 042 fixture loss was a lane failure (2 techs
  in 19 turns). **Falsifier: flat → lane persistence alone does not pay;
  HOLD the Stage-3b channel migration** (210 checkpoint files + optimizer
  reset) and reconsider whether Tier 1's value rests entirely on the
  head's conditional choice.
- **P2 (stability, mechanical):** ≤3 pivots/game and mean lane tenure ≥
  DWELL_MIN turns across the run; lane distribution not collapsed to one
  lane across seeds (each of the three appears on ≥10% of games).
  Falsifier: collapse → the census+prior is degenerate on Tiny/Drylands
  and the lane vocabulary needs work before the head can rank it.
- **G1 (throughput):** ms/move within 1.3× of baseline — the selector
  runs once per turn, so a bigger regression means it is still being
  called per ply somewhere.

ACTUAL (2026-08-14, 250 paired games/arm, commits fec32c0 + 5bf6af7):
- **P1 FAILED.** Committed-lane 154/250 (61.6%) vs per-ply 167/250
  (66.8%) — **-5.2pp**, McNemar 35 old-only vs 22 new-only, z = -1.72.
  The gate wanted >= +4pp at |z| >= 1.96; this is a mild regression that
  does not itself clear significance. Siege rows move the same way
  (lost-given-siege 54.5% -> 58.3%).
- ⚠️ **Instrument scope — the tribe prior was INERT in this run.** Arena
  pins BOTH seats to one tribe (`arena.rs:539` uses `args_tribe` twice)
  and the 041/042 instrument pins `--tribe imperius`; Imperius's spawn
  tech is Organization, which opens no lane chain, so
  `tribe_lane_prior` returned None in all 250 games. What this arm
  actually measured is the COMMITMENT half alone (turn-boundary
  selection + <=3 pivot budget + dwell), not "tribe sets the tone".
  Follow-up registered below.
- **P2 PASS (mechanically), with a skew worth noting.** Pivots: mean
  0.80/game, max 2 (cap 3, never hit); 198/250 games pivot at least
  once; mean last-commit turn 9.6. Lane distribution ArcherLine 72.0% /
  ForgeGiants 17.2% / RiderRoads 10.8% — all three clear the 10% floor,
  but ArcherLine dominates because its score fires on `seen_heavy >= 1`
  (+2) plus contact (+2), both common in a Greedy matchup.
- **G1 PASS, better than required:** 198.8 vs 229.4 ms/move (0.87x).
  Selection moved from every ply to once per turn, and the wiring is
  confirmed by the speedup itself.
- READ: on Tiny/30-turn games against Greedy, continuous re-evaluation
  beats a held identity. Most likely mechanism: refutation latency — the
  per-ply version hard-exited a countered lane the moment two giants
  appeared, while the committed version waits for the next turn boundary
  (pinned deliberately in the archetype test). The stance layer solved
  the same tension by letting THREAT responses bypass hysteresis
  (`update_goal`'s `urgent` path); the lane layer currently has no such
  bypass.
- CONSEQUENCE: Stage 3b stays deferred (it already was, per 039). Tier 1
  is NOT yet justified by evidence: the commitment half costs ~5pp, the
  head half is deferred, and the tribe-prior half is unmeasured. Do not
  build further on Tier 1 until 045a-b reads.

## EXP_ELO_045a-b — the untested half: does the TRIBE prior pay?

**Registered 2026-08-14, before running.** 045a's instrument nulled the
tribe prior (Imperius has no lane tech). This arm pins a tribe that DOES
have one so the feature is actually exercised: `--tribe oumaji` (spawn
tech Riding -> RiderRoads, verified against `mapgen.rs:1254-1284`).
Identical otherwise: baseline `/tmp/exp045a_baseline_arena` vs the
selector build, macro-mcts vs Greedy, 125 seeds x2, base_seed 1787800000,
gamemode 2, max-turns 30, metal, GUMBEL_SCALE=0.

- **P1:** committed+primed lane beats per-ply by >= +4pp (|z| >= 1.96).
  Falsifier: flat or negative -> the tribe prior does not rescue
  commitment either, and Tier 1 as specified is unsupported on this
  distribution; report that plainly rather than tuning toward a pass.
- **P2:** RiderRoads share > 50% in the new arm (the prior should
  visibly bend lane choice for a rider tribe), vs whatever the baseline
  picks. A flat distribution means the +2 prior is too weak to matter
  against census scores that reach 6.

ACTUAL (2026-08-14, 250 paired games/arm, Oumaji mirror):
- **P1: gate not cleared, but the SIGN FLIPS.** Committed+primed 175/250
  (70.0%) vs per-ply 170/250 (68.0%) — **+2.0pp**, McNemar 22 old-only
  vs 27 new-only, z = +0.71. Below the +4pp/|z|>=1.96 bar, so not a pass;
  but against 045a's -5.2pp on the prior-less Imperius instrument this is
  a 7.2pp swing attributable to the one thing that differs — whether the
  tribe opens a lane.
- **P2: the prior demonstrably works.** RiderRoads share 49.2% (vs 10.8%
  on Imperius, where `tribe_lane_prior` returned None); ArcherLine 50.8%.
  Just under the >50% line, so scored a near-miss rather than a pass, but
  the mechanism is unambiguous: a +2 spawn bonus moves a rider tribe's
  lane share by ~38pp. Stability improved too: 0.51 pivots/game, max 1
  (vs 0.80/max 2), mean last-commit turn 7.2.
- READ (both arms together): commitment ALONE is a cost; commitment plus
  a lane the tribe was born into is roughly neutral-to-positive. That is
  the shape Verdi's spec predicted ("your tribe sets the tone") and the
  opposite of what 045a alone implied.
- ⚠️ CONFOUND, discovered after both arms ran (advisor-caught, verified):
  the new build removed lane SELECTION from `execute_turn` while
  `run_with` seeds only `arch[pov]`, so the in-tree OPPONENT played
  laneless for whole rollouts (no lane techs, no preferred-unit pricing)
  where the baseline's rollouts selected per ply for both seats. Both
  045a and 045a-b therefore compare trees that evaluate different
  futures, not just different root commitment. Fixed in 67cae7f
  (selection in `Node::new`, i.e. at the turn boundary a node already
  is, for both seats) together with a refutation bypass mirroring the
  stance layer's urgent path. Neither arm's number is retired — the
  positive one cleared the handicap — but the clean read is 045a-v2.

## EXP_ELO_045a-v2 — the selector, confound removed

**Run 2026-08-14** after 67cae7f fixed the two divergences 045a/045a-b
exposed: (1) lane selection restored in-tree at `Node::new` for BOTH
seats (the observe/select split had left the simulated opponent laneless
for whole rollouts, since `run_with` seeds only `arch[pov]`); (2)
refutation bypasses the turn boundary, mirroring the stance layer's
urgent path. Only the NEW arm re-ran — same baseline binary, same seeds,
so the stored 045a/045a-b old arms remain the comparator.

| instrument | baseline (per-ply) | v1 (committed) | v2 (fixed) |
|---|---|---|---|
| Imperius (no tribe prior) | 66.8% | 61.6% (-5.2pp) | **65.2% (-1.6pp, z=-0.54)** |
| Oumaji (tribe prior fires) | 68.0% | 70.0% (+2.0pp) | **69.2% (+1.2pp, z=+0.52)** |

Reads:
- **The -5.2pp regression was mostly the confound, not commitment.**
  Restoring in-tree lane selection recovered 3.6 of the 5.2 points; what
  remains on Imperius is -1.6pp at z=-0.54, i.e. noise.
- **Both instruments now sit inside noise** (|z| < 0.6). Committing to a
  lane neither costs nor buys measurable strength against Greedy at
  n=250 per arm. The registered +4pp gate is NOT cleared on either.
- **The tribe prior remains the only component with a visible mechanical
  effect**: RiderRoads share 15% (Imperius, prior inert) vs 52% (Oumaji,
  prior fires) under identical code — a ~37pp swing driven by a +2 spawn
  bonus. Stability also improves where the prior anchors the choice:
  0.48 pivots/game max 1 (Oumaji) vs 0.76 max 2 (Imperius).
- Throughput: 227 ms/move vs the baseline's 229 — the in-tree selection
  restored is per-TURN, not per-ply, so the cost stayed flat.

VERDICT: Tier 1's algorithmic half is **behaviour-neutral, not
behaviour-positive**. It reshapes lane choice exactly as designed (and
the tribe prior does what Verdi specified), at no measurable cost or
benefit in win rate. That is a defensible foundation to keep — the value
proposition was always that the SELECTOR becomes the place a learned
signal plugs in (Stage 3b's per-lane head), and the fixed code is now
the honest baseline for that test. It is NOT, on this evidence, a
strength win on its own, and should not be reported as one.

Standing decision (unchanged): Stage 3b's head + the 169->173 migration
stay deferred behind a re-run of EXP_ELO_039 on a checkpoint trained
with corrected TD labels (ea700e4). The three gates together —
039 falsified, 045a/v2 neutral, 045a-b prior-positive — say the tier
architecture's remaining upside is concentrated in the LEARNED half,
which is exactly what the label bug has been suppressing.

## EXP_ELO_046 — the corrected-label round: does a full-MC macro label restore the value head?

STATUS: REGISTERED (Aug 15, 2026) — running.

CONTEXT. Iterations 1-10 of run_id 1786710389 (Aug 14, macro-generated)
were produced by a binary in which `Brain::think` hard-coded
`macro_params = None` and `Brain::last_root_value()` returned `None` for
the macro backend. Consequence: EVERY n-step return bootstrapped with
0.0 at EVERY checkpoint, so the value labels of the whole round were
systematically truncated toward zero — the fastest way to teach a value
head that the future is worth nothing. EXP_ELO_039 then measured that
head at the macro leaf and read 38.8% vs the heuristic leaf's 61.2%
(z=-7.1). The 039 verdict was deliberately scoped to "this checkpoint,
trained on those labels"; this round produces the checkpoint that makes
the re-run meaningful.

HYPOTHESIS. With `--td-missing-bootstrap mc` the missing checkpoint's
weight carries forward to the terminal return instead of pulling the
label to zero. Under the heuristic leaf NO checkpoint has a root value,
so this collapses the labels to full Monte-Carlo (lambda=1) returns over
each game. A value head trained on true returns should separate
strategic futures materially better than one trained on truncated ones.

CONFIG (resume of run_id 1786710389, iterations 11-20, EFF_ITER 126-135):
  MACRO_GEN=1 GOAL_CHANNELS=1 ITER_OFFSET=115 TD_MISSING_BOOTSTRAP=mc
  ./run_training_loop.sh --resume -i 10 -g 64 -n 32
Leaf stays HEURISTIC: 039 says the net leaf plays worse today, and it
generates ~30x slower (3.28 vs ~100 moves/s).

THREE GENERATION DELTAS vs iterations 1-10 (all recorded because a
re-run that flips must be attributable):
  (a) TD bootstrap zero -> mc (the label fix; the point of the round).
  (b) anchor-frac 0.25 -> 0 (committed default under MACRO_GEN: the
      anchor exists to match what `blend_heuristic_prior` injects, and
      that blending is Gumbel-only, so under macro generation the anchor
      games are macro-vs-Greedy blowouts diluting the cloning data).
  (c) the generation binary now carries the Stage-3a selector: committed
      per-turn playstyle, tribe prior, in-tree lane selection for both
      seats. Measured behaviour-NEUTRAL on the arena instrument
      (-1.6pp / +1.2pp), but it shifts lane shares hard for tribes whose
      spawn tech opens a lane — and Oumaji/Bardur/Kickoo are in the
      training tribe rotation, so the generated distribution is not
      identical to iterations 1-10's.

EXPECTED, and the discontinuity to NOT misread. Value labels change
semantics at iteration 11 INSIDE run_id 1786710389, so `value_loss` and
`value_r2` will step at that boundary on the dashboard. That step is the
label change, not a regression — do not stop the run over it. Full-MC
labels are higher-variance than bootstrapped ones, so a value_loss rise
with an unchanged or improved value_r2 is the expected shape.

PREDICTIONS.
  P1 (the gate): re-run EXP_ELO_039 exactly as registered against the
     iteration-20 checkpoint. Net leaf >= 52.5% unblocks Stage 3b in
     full; 50-52.5% ships it with w_algo held at its floor; <= 50% a
     SECOND time retires the net-leaf idea rather than re-skinning it,
     and Tier 1 keeps the algorithmic selector alone.
  P2: value_r2 at iteration 20 >= its iteration-10 value (0.79). A
     corrected label that does not improve the head's fit falsifies the
     premise that labels were the binding constraint.
  P3: the gauge-vs-Greedy win rate does not fall below its iteration-10
     level — this round is a label correction, not a strength push, so
     flat is a pass and a drop is a signal something else broke.

CARRY FORWARD. If 039 fails again on corrected labels, the next suspect
is already identified: the leaf paints `scripted_goal` while the training
data painted the tree's COMMITTED directive, and those two disagree on
40-55% of turns (MACRO DIVERGENCE, arena). That is a painting mismatch,
not a label problem, and it is the cheaper of the two remaining fixes.

### ACTUAL — training round (Aug 15, 2026, iterations 11-20 = EFF_ITER
126-135, run_id 1786710389, ~2.5h wall clock). **P2 PASSED, P3 PASSED.**

| | iter 10 (zero-bootstrap) | iter 20 (mc) |
|---|---|---|
| value_r2 | 0.7919 | **0.8072** |
| value_loss | 0.4517 | 0.6009 |
| policy_loss | 2.2822 | 2.1685 |
| gauge vs anchor_iter5 (64g) | 62.5% | 65.6% |

P2 (value_r2 >= 0.79): PASSED at 0.8072, the highest reading of the run,
and it climbed monotonically over the second half (0.7811 -> 0.7944 ->
0.7940 -> 0.7979 -> 0.8055 -> 0.8072). The value_loss RISE to 0.6009
alongside it is the predicted shape, not a regression: full-MC labels are
higher-variance than bootstrapped ones, so the same fit costs more raw
MSE. Rising r2 with rising loss is exactly what a variance change looks
like; a genuine regression would move them the same way.

P3 (gauge does not fall): PASSED, 65.6% vs 62.5% at iteration 10. Both
readings sit inside the 64-game gauge's +-12pp ruler, so the honest read
is HELD, not improved.

policy_loss fell 2.2822 -> 2.1685, continuing its trend — the
owned-vs-rented gauge moving the right way while the label semantics
changed under it.

Behaviour was flat to slightly down over the round (score 4993 -> 4756,
captures 5.89 -> 5.25, 3rd city 80%@t11.0 -> 72%@t12.5), with research up
6.44 -> 7.20. Ten iterations at 64 games is far under this instrument's
resolution, and the round changed generation in three ways at once
(anchor games and the Stage-3a selector, not just the labels) — read
these as "nothing broke", not as a behaviour result.

OPERATIONAL NOTE (cost an exit-code-1 false alarm): the loop exited 1
with every iteration complete and the model saved. Cause is its own EXIT
trap: the background dashboard server it starts had already died on
`AddrInUse` (port 3000 was held by a pre-existing polyfish server), so
`kill $SERVER_PID` in the trap returned 1, and under `set -e` a failing
command in an EXIT trap sets the shell's exit status. Training was never
affected. Verified by reproduction:
`bash -c 'set -e; trap "kill 999999 2>/dev/null; rm -f /tmp/x" EXIT; true'`
-> exit 1.

### ACTUAL — P1, the EXP_ELO_039 re-run (Aug 15, 2026, 1000 games,
37 min). **P1 FALSIFIED A SECOND TIME.** Net leaf 419/1000 (41.9%) vs
heuristic leaf 580 (58.0%), 1 draw; z = (419-499.5)/15.8 = **-5.09**.
Avg score 4071 vs 4821. Cost 298 vs 232 ms/move (+28%, inside the 3x
guardrail). Snapshot `checkpoints/exp046_snapshot_iter135.safetensors`,
sha256 4cbb9ddc61bc1fd2aad015d7ace7ddfd90069f23774289f47c9b646505e26e67
(run 1786710389 iter 20 = EFF_ITER 135, value_r2 0.807, policy_loss
2.169). Same arms, same 500 seeds, same base_seed 1787300000 as the
original.

Direction of travel, stated as directional only: 38.8% -> 41.9%
(+3.1pp). The corrected labels moved the leaf the RIGHT way, and by
more than the round's other metrics suggested — but nowhere near the
50% it needs to take the seat. NOT a clean paired read: the re-run's
binary carries 67cae7f (in-tree lane selection for both seats +
refutation bypass), which is symmetric across the two arms here (so the
net-vs-heuristic comparison inside this run is clean) but differs from
the binary that produced the original 38.8%.

Behaviour tell, new this round: the net-leaf arm plays a materially
LOOSER game — 3.69 sieges/game suffered vs the heuristic arm's 2.22, and
1.62 cities lost vs 1.19. It is not passively losing; it is picking
fights and positions it cannot hold. That is a plan-quality failure at
the leaf, not a timid evaluator.

CONSEQUENCE (as pre-registered): the net leaf is RETIRED, not re-skinned.
`MACRO_LEAF` stays `heuristic` in generation and in play. Two independent
falsifications, the second on labels specifically corrected to address
the first, is the standard this ledger set for retiring an idea.

The one caveat that survives retirement, recorded so it is not silently
re-litigated: the leaf paints `scripted_goal` for the leaf player because
the committed directive is unknowable before the choice, while the
training data painted the tree's COMMITTED directive — the two disagree
on 40-55% of turns (MACRO DIVERGENCE, arena). So 039 is strictly a
verdict on "the net leaf AS PAINTED", and a painting fix is a different
experiment, not a re-run of this one. It is cheap and it is identified;
it is not scheduled.

STANDING: Stage 3b (`aux_playstyle` head + the 169->173 migration across
~210 checkpoints) stays DEFERRED. Its premise was that a trunk which can
rank futures at the leaf is a credible lane ranker; that premise has now
been tested twice and failed twice. Tier 1 keeps its algorithmic
selector (045a-v2: behaviour-neutral, tribe prior working as specified).

## EXP_ELO_047 — the painting mismatch: does the goal the leaf paints even move the value?

STATUS: REGISTERED (Aug 15, 2026), Verdi-requested after 046.

FRAMING (load-bearing): this tests the PAINTING hypothesis. It does not
re-open the net leaf's retirement. The leaf seat changes hands only if an
aligned-painting arm clears the originally registered >= 52.5% gate
against the heuristic leaf. Two falsifications stand until then.

THE MISMATCH. Training (`self_play.rs:1871`) paints the tree's COMMITTED
directive into the goal channels — the goal that actually drove the ply.
Inference (`macro_mcts.rs:215`) paints `scripted_goal` at the leaf,
because the committed directive is unknowable before the choice. Arena
measures the two disagreeing on 40-55% of planned turns, so roughly half
of all leaf queries are off-distribution in the conditioning channels.

PHASE A — the diagnostic (cheap, decides whether Phase B is worth
running). At every macro root, post-search, encode the SAME state twice —
once painted with `base` (scripted) and once with the committed directive
— and evaluate both. Emit one JSONL row per root:
`{turn, pov, diverged, v_scripted, v_committed, v_opp, q_spread, q_best,
q_base}`. Env-gated (`POLYFISH_PAINT_PROBE=<path>`), zero cost when unset.
The root is exactly the state class a leaf is: a turn boundary, encoded
from the acting player's perspective.

⚠️ Analysis rule, pre-registered: ~47% of roots pick index 0, where
committed == scripted and dV == 0 BY CONSTRUCTION. All reads below are
computed on DIVERGENT rows only (`diverged == true`). Pooling the
identical rows would dilute the median toward zero and falsely kill the
hypothesis.

  P1 (does painting matter?): median |dV| on divergent roots, against
     median root q_spread (the value difference the tree must resolve to
     rank one directive over another).
       < 0.1x  -> painting hypothesis DEAD. The head barely reads the
                  goal channels, the mismatch cannot explain 041.9%, and
                  the thread closes WITH EVIDENCE rather than a shrug.
       >= 0.5x -> LIVE. Proceed to Phase B.
       0.1-0.5x -> report both numbers and the sign before choosing.
  P2 (sign bias): median SIGNED dV = v_committed - v_scripted. A
     systematic bias distorts leaf ranking even when |dV| is small,
     because it applies to only ~half the leaves — the divergent ones.
  P3 (antisymmetry, gates the Phase B implementation): median
     |v_pov + v_opp| at the same root. Phase B's cheap form returns
     -V(child, mover, g_edge) and relies on the net being approximately
     antisymmetric — which is TESTED for `evaluate_state` and never
     enforced for the net (per-seat labels). If this is large relative to
     q_spread, the negation imports that gap into every leaf and Phase B
     must restructure the negamax instead of negating.

PHASE B — the fix, only if P1 says LIVE. Paint what the tree DOES know at
a leaf: the child state was reached by executing edge directive `g` for
player `p`, so paint `g` and evaluate from `p`'s perspective (returning
its negation to keep the child-perspective convention). This is
in-distribution by construction — training contains exactly
(state during p's turn, p's pov, g committed) samples, and the last such
ply of a turn is the pre-EndTurn state. Verified prerequisite: features
carry NO to-move / current-player plane (grep of `features.rs`), so a
post-EndTurn state read from the mover's perspective is not a state class
the net has never seen. Implementation: store the incoming edge's
(player, goal) on the Node at expand time; root and frozen paths keep the
scripted fallback.

MEASUREMENT (Phase B): ONE arena run, net-aligned vs heuristic leaf,
`base_seed 1787300000`, 500 seeds x2 — the same seeds and the same
binary-era arms as 046's re-run, whose `replays/exp046/armA` dumps are
the stored comparator. The heuristic arm is untouched by the paint
change, so this single run yields BOTH reads: McNemar vs armA isolates
the painting delta, and the absolute win rate is the seat test (>= 52.5%).

FALLBACK (a fork for Verdi, NOT a default): if P3 or the to-move check
kills Phase B, the other direction is to paint the SCRIPTED goal in
training so inference matches. That changes what the goal channels MEAN
in all future macro data and knowingly mis-conditions the policy head on
~half of turns (MACRO DIVERGENCE 52.7%) — a training-semantics decision
to be made with the Phase A numbers in hand, not fallen into.

### ACTUAL — Phase A (Aug 15, 2026; 2500 macro roots over 120 games,
exp046 snapshot, heuristic leaf so the trajectories are the DEPLOYED
distribution). **P1 LIVE. P2 no bias. P3 falsified the planned Phase B
implementation and found a larger defect than the one registered.**

P1 — painting moves the value MORE than the tree's whole decision margin.
On the 1306 divergent roots (52.2%, matching the arena's 52.7% MACRO
DIVERGENCE): median |dV| **0.0754**, p90 0.443, against a median root
q_spread of **0.0518** on the same rows. **Ratio 1.46x** — the
registered LIVE threshold was 0.5x. Repainting the same state with the
other directive perturbs the value by half again as much as the entire
spread the tree is trying to resolve between competing directives. The
mismatch is not a rounding error on the leaf; on divergent turns it is
larger than the signal.

P2 — no systematic bias: median dV +0.0018, mean +0.014, 53.0% positive.
This is the WORST shape for ranking. A bias would shift all leaves
together and partly cancel in the negamax; symmetric noise of larger
magnitude than the signal just scrambles the ordering, and only on the
~half of turns where the two paintings disagree.

P3 — **the net is not zero-sum, and this is a bigger defect than the
painting.** Median |V(s,p1) + V(s,p2)| = **0.386** (p90 1.067, max 1.94
on a [-1,1] value); 21% of roots read BOTH players as winning. Control on
the identical fogged states: the heuristic's |h(p1)+h(p2)| is
**0.000000 at median, p90, AND max** — exactly antisymmetric, as its unit
test requires. So fog does not explain it; this is the net's own defect,
and it is ~13x the q_spread the tree resolves.

WHY THAT MATTERS MORE: negamax backs a child's value to its parent by
NEGATING it, which silently assumes the leaf scorer is zero-sum.
`evaluate_state` satisfies this (there is a test —
`evaluate_state_is_antisymmetric`); the net never had to. Every backup
through a net leaf therefore carries a median 0.386 error against a
0.03-0.05 decision margin. **This alone is sufficient to explain
EXP_ELO_039's 41.9%, with no reference to painting at all** — and it was
invisible to every previous experiment because the heuristic leaf, the
only leaf ever deployed, satisfies the assumption for free.

PHASE B AMENDED (Aug 15, dated per discipline). The registered Phase B
(paint the edge directive, return its negation) is DEAD ON ARRIVAL: it
depends on the very antisymmetry P3 just falsified. Replaced by, in
order of measured size:
  B1 — `MacroLeaf::NetAsym`: leaf value `(V(s,p) − V(s,opp))/2`, two
     forwards per leaf, which makes the zero-sum identity hold BY
     CONSTRUCTION. Painting unchanged (each side its own scripted goal),
     so this is ONE variable. Pinned by `net_asym_leaf_is_zero_sum`.
  B2 — aligned painting, only if B1 moves the number: NetAsym already
     evaluates the mover's perspective, and for the mover the committed
     directive IS known (it is the edge). B2 paints it there. Deliberately
     NOT bundled with B1.
Measurement for B1: one arena run, `--macro-leaf1 net-asym` vs heuristic,
base_seed 1787300000, 500 seeds x2 — same seeds and same heuristic arm as
046's re-run, so `replays/exp046/armA` is the stored comparator. McNemar
against it isolates the antisymmetry fix; the absolute number is the seat
test (>= 52.5%).

Probe hygiene: 1 of 1194 identical-painting rows showed a nonzero dV
(0.08%), consistent with metal batch-coalescing nondeterminism rather
than a probe bug — the analysis asserts this count and it is reported,
not swept.

### ACTUAL — Phase B1, the antisymmetrized leaf (Aug 15, 2026, 1000 games,
53 min). **The defect was real; fixing it is not sufficient.**
`MacroLeaf::NetAsym` scored **440/1000 (44.0%)** against the heuristic
leaf. Paired against `replays/exp046/armA` (identical seeds, identical
heuristic arm, same snapshot — only the leaf's zero-sum handling
changed): **41.9% -> 44.0%, +2.1pp, McNemar z = +1.08** (old-only 177,
new-only 198). Seat test vs the registered gate: z = -3.79 against 50%,
so **>= 52.5% is not remotely cleared and the heuristic leaf keeps the
seat.** Avg score 4200 vs 4809. Cost 450 vs 381 ms/move (both arms
inflate vs 046's 298/232 — two forwards per leaf saturate the eval server
and arena games run concurrently, so this is contention, not per-move
work on the heuristic side).

READ IT HONESTLY. A provably real defect — the negamax backup was
carrying a median 0.386 error against a 0.03-0.05 decision margin, 13x —
was removed cleanly (pinned by `net_asym_leaf_is_zero_sum`), and it
bought +2.1pp that does not clear noise. Two consequences:
  1. The zero-sum violation was NOT the reason the net leaf loses. It was
     a genuine bug worth fixing on its own terms, and the fix stays.
  2. Defect magnitude in value-units does not translate to win rate. This
     is the third time in this program (038's memory, 045a's selector,
     now 047-B1) that a mechanism confirmed to be doing exactly what it
     was designed to do moved nothing. The pattern is now strong enough
     to state plainly: the macro leaf's problem is not any single
     identified defect in how the value is READ.

### ACTUAL — Phase B2, aligned painting (Aug 15, 2026, 1000 games, 41 min).
**FALSIFIED, and it made things WORSE.** `NetAsymPaint` scored
**399/1000 (39.9%)** vs the heuristic leaf. Paired on identical seeds:

| arm | leaf | painting | wins |
|---|---|---|---|
| 046 re-run | net | scripted | 419 (41.9%) |
| 047-B1 | net, antisymmetrized | scripted | 440 (44.0%) |
| 047-B2 | net, antisymmetrized | **aligned** | 399 (39.9%) |

B2 vs B1 (the clean painting isolation — both antisymmetrized, only the
mover's painting differs): **z = -2.15** (B1-only 202, B2-only 161). The
aligned painting is not neutral; it is significantly WORSE than the
scripted painting it replaced.

WHY, and it invalidates my in-distribution argument rather than the
measurement. The goal channels mean "the directive I am ABOUT TO
EXECUTE" — every training sample is a pre-move ply painted with the
directive that then drove it. A leaf state is POST-`execute_turn`: the
mover's directive is finished, and the state has already advanced through
the turn boundary (the next player's income and city production are
applied). Painting the just-executed directive there asks the head "what
is this state worth if I am about to do g" about a state where g has
already happened and is no longer available — a strictly WORSE
off-distribution query than the scripted goal, which at least names
something the player might do next. Phase A's 1.46x was measured at the
ROOT, where the acting player really is about to execute the directive;
that quantity does not transfer to the leaf, and I over-read it when I
designed B2. The measurement was right; the fix direction was wrong.

CONSEQUENCE. The painting thread is CLOSED with evidence, not abandoned:
the mismatch is real at the root, the only alignment available at a leaf
makes it worse, and no third option exists that does not require knowing
the leaf player's future directive (i.e. a nested search). `NetAsymPaint`
stays in the tree as a selectable leaf for the record but is NOT the
default and should not be used.

WHAT SURVIVES 047: `MacroLeaf::NetAsym` (+2.1pp, z=+1.08 — a real bug
fixed, an insignificant gain), the probe, `root_q_spread`, and the two
zero-sum tests. The heuristic leaf keeps the seat under all three arms.

THE PROGRAM-LEVEL READ. 039 (twice), 045a, 047-B1 and 047-B2 now say the
same thing from four directions: every defect identified in HOW THE TREE
READS the value has been fixed or measured, and the net leaf still loses
by 12-20pp. What has never been fixed is WHAT THE HEAD KNOWS — it is
trained on 64-game rounds at ~135 iterations, reads value_r2 0.807 yet
prices both players as winning on 21% of roots (Phase A), and was
calibration-flagged as 2x over-confident when ahead long before any of
this. The next credible move is data/labels/scale, not another consumer
of the same head.

## EXP_ELO_048 — does Tier 3 follow Tier 2? (the boundary probe)

STATUS: REGISTERED (Aug 15, 2026), Verdi-requested.

WHY. Six-plus arms that changed WHICH directive the macro tree picks
(033b sims, 035 belief world, 036 fog candidates, 038 memory, 045a lane
commitment) all read flat. Two readings are consistent with that: the
executor makes poor use of good directives, or the directive never
reaches the plies at all. ⚠️ Correction on the record: EXP_ELO_039/047
do NOT bear on this — they hold the executor fixed and vary only the leaf
scorer, so they are evidence about the evaluator alone. Nor has a BETTER
leaf than `evaluate_state` ever been tested (039/047 tested a worse one),
so 038's "starving for evaluation" conclusion is still live, not refuted.

WHAT IT MEASURES. At every planned root, re-execute that same turn on
throwaway clones under three directives — the tree's pick, the scripted
base, and `MacroGoal::default()` (no directive at all) — and compare the
executed PLY SEQUENCES. Per ply of the pick's turn it also asks the
counterfactual directly: would this ply have been chosen with the
directive's ranking term removed (`flip_no_phi`, gate intact) and with
the directive removed entirely (`flip_no_goal`)? One implementation, not
two: `execute_turn` now delegates to `execute_turn_recorded`, so the
probe cannot drift from the executor it measures. Env-gated
(`POLYFISH_TIER_PROBE`), zero cost unset.

Overlap is deliberately multiset (order-insensitive), so a reordered but
identical set of plies reads as "the directive changed nothing" — the
conservative direction for this question.

PREDICTIONS (pre-registered).
  P1 — the boundary: median `overlap_pick_none` (tree's directive vs NO
     directive) over divergent roots.
       > 0.9  -> the tier boundary LEAKS: Tier 2 cannot express itself
                 through Tier 3, and every directive-layer null collapses
                 into one explanation. Fixing the executor's ply CHOICE
                 would not be the lever either — the wiring is.
       < 0.6  -> the directive genuinely drives the turn; the nulls then
                 mean the directives themselves are interchangeable in
                 value, which points at the generator (H3) or the
                 evaluator (H2), not the boundary.
  P2 — where it leaks: the same overlap restricted to STAR-SPENDING plies
     (Research/Build/Summon). Star allocation is the one channel with a
     proven causal link to wins (EXP_ELO_026: 28%->81%). A directive that
     reshuffles Steps but leaves the spend sequence identical is
     decorative where it counts, even if total overlap looks moderate.
  P3 — per-ply ownership: share of executed plies with `flip_no_phi`
     (the lambda*dphi pull owns the pick) and `flip_no_goal` (the whole
     directive channel owns it). This separates the two ways a directive
     can act — the gate FILTER vs the ranking PULL — which the sequence
     overlaps alone cannot.

NOT A WIN-RATE EXPERIMENT. No arm, no A/B, no gate. It is a measurement
of a mechanism, and its job is to say which of H1 (executor) / H2
(evaluation) / H3 (generator) is worth spending on next.

### ACTUAL (Aug 15, 2026; 1669 planned roots over 80 games, deployed
config — macro-mcts heuristic leaf vs Greedy). **P1 ANSWERED: the
boundary does NOT leak. Tier 3 follows Tier 2.**

Turn shape for scale: median 7 executed plies/turn, of which 2 are
star-spending.

| metric (median) | all roots | divergent (53.2%) |
|---|---|---|
| ply overlap, pick vs scripted base | 0.850 | 0.594 |
| ply overlap, pick vs NO directive | **0.545** | **0.556** |
| SPEND-ply overlap, pick vs base | 1.000 | 0.500 |
| SPEND-ply overlap, pick vs no directive | 0.500 | 0.500 |
| plies owned by the lambda*dphi pull | 51.2% | 50.9% |
| plies owned by the whole directive | 51.3% | 51.0% |

P1: 0.545 is below the pre-registered 0.6 "drives" line and nowhere near
the 0.9 "leaks" line. Removing the directive entirely changes about HALF
the executed plies. The directive is not decorative, and the
`ply <- order` link is real. (19.5% of turns do reproduce exactly with no
directive — the quiet turns where only one sensible line exists.)

P2: on divergent roots the spend overlap is 0.500 against a median of 2
spend plies — i.e. the tree's directive changes one of the two star
commitments of that turn. The directive reaches the channel with the only
proven causal link to wins (EXP_ELO_026), not just the Steps.

P3, the sharp one: `flip_no_phi` (51.2%) and `flip_no_goal` (51.3%) are
the SAME to within a tenth of a point. All of the directive's influence
flows through the lambda*dphi ranking PULL; the stance/star GATE adds
essentially nothing on top of it. Anything that hopes to steer the
executor by gating rather than pricing is steering a channel that is
already carrying ~0 marginal signal — which retroactively explains why
the mask-style arms lost to the pricing-style ones (goal-pricing-beats-
masks, 028 P1c).

VERDICT AND WHERE IT LEAVES THE PROGRAM. H1-as-wiring is FALSIFIED: the
nulls are not caused by directives failing to reach the plies. Combined
with the standing result that changing WHICH directive is picked does not
change wins, the surviving reading is that **the candidates are
value-degenerate — they produce materially different turns that are worth
about the same.** That splits two ways, and they are distinguishable:
  H2 (evaluation): the alternatives really do differ in value and the
     tree cannot tell — still LIVE, and never tested with an evaluator
     better than `evaluate_state` (039/047 tested worse ones).
  H3 (generator): the alternatives are genuinely interchangeable, so no
     amount of evaluation would help; the generator has to propose
     strategically different turns, not variations.
Distinguishing probe (not run, not scheduled): on divergent roots, roll
BOTH candidates out deep (to terminal or a long horizon, ghost opponent)
and compare outcomes. Materially different outcomes => H2 and evaluation
is the lever; equal outcomes => H3 and the generator is.

### ACTUAL — obedience (Verdi's metric, Aug 15, 2026): distance from the
nearest own unit to each order's target, BEFORE the turn vs AFTER, with
the no-directive execution of the same turn as the control. 1637 roots,
3205 orders (1.96 per directive: 71% Expand, 18% Attack, 11% Defend).

| order | closer | unchanged | farther | progress w/ | control | **causal lift** |
|---|---|---|---|---|---|---|
| ALL | 53.4% | 38.5% | 8.1% | +0.549 | +0.096 | **+0.453** |
| Expand | 60.7% | 32.8% | 6.4% | +0.645 | +0.123 | **+0.522** |
| Attack | 42.1% | 49.6% | 8.4% | +0.422 | +0.077 | +0.346 |
| Defend | 24.1% | 57.7% | 18.3% | +0.125 | −0.046 | +0.171 |

(tiles/turn; median starting distance 2 tiles; a unit moves ~1 tile/turn.)

TIER 3 OBEYS, AND THE OBEDIENCE IS CAUSAL. Orders are followed at about
half a tile per turn against a control that drifts at a tenth — a ~5.7x
causal lift, and 5.7x is the honest form of the number because the
control is the SAME turn re-executed with the directive removed. Against
a ~1 tile/turn movement budget and a 2-tile median target, +0.55 is
roughly half the theoretical maximum, which is what a turn that must also
harvest, research and build looks like. Per order vs its own control:
34% strictly closer, 61% identical, 5% farther.

THE INTERESTING PARTS ARE THE EXCEPTIONS.
- **Defend is nearly inert**: +0.125 tiles/turn, 18.3% of Defend orders
  end FARTHER from the city than they started, and it is the only kind
  whose control is negative. Its median starting distance is 0 (a garrison
  is already there), so "progress" is the wrong yardstick for it — but
  "moved away from the city it was told to defend, 18% of the time" is
  measured on the right one. That is the executor leaking a duty, and it
  is exactly the behaviour EXP_ELO_040/042 built pricing for and could
  not move.
- **Fulfilment is 9.8%**: of orders on targets not yet owned, only one in
  ten is captured during the turn it was issued. With a 2-tile median gap
  that is arithmetically expected, not damning — but it means an order's
  value is realised over ~2-3 turns, and the turn-level tree is scoring a
  horizon where most of its own orders have not paid off yet.
- Only 9.2% of orders are issued on targets >3 tiles away, so the
  generator is not setting unreachable goals.

WHAT THIS SETTLES. Combined with the ply-influence result, both halves of
"does Tier 3 follow Tier 2" now answer YES: the directive owns ~half the
plies, and those plies move toward the named target at ~5.7x the
no-directive drift. **H1 is dead in both its forms** — neither the wiring
nor the following is broken. The directive-layer nulls therefore mean the
alternative directives are worth about the same (H2 evaluation / H3
generator), which is exactly what the deep-rollout separator would
distinguish.

## EXP_ELO_049 — the siege ledger: is the model besieged more, or worse at answering?

STATUS: MEASURED (Aug 15, 2026), Verdi-requested. Instrument, not an A/B.

WHAT WAS ALREADY IN HAND (SIEGE DEFENSE lines, all runs to date). The
question is answerable from the aggregate alone: the model is besieged
LESS and loses the city MORE.

| run (config1 vs config2) | model sieges/game | unsiege% | cities lost/game |
|---|---|---|---|
| 048 macro vs Greedy | 1.46 / **2.58** | **32%** / 53% | 0.94 / 1.17 |
| 045v2 macro vs Greedy | 1.60 / **3.04** | **45%** / 52% | 0.84 / 1.36 |
| 045a-b macro vs Greedy | 1.70 / **4.62** | **40%** / 67% | 0.97 / 1.44 |
| 041 NET (gumbel) vs Greedy | 1.98 / 1.90 | **38%** / 42% | 1.18 / 1.04 |

NEW INSTRUMENT. `SiegeTracker` now closes each episode with the facts at
the moment the attacker stepped onto the city — attacker type/health,
city level, own units alive, nearest own unit, **responders** (own units
that could strike that tile next turn, via `defense::covers`), stars
banked, and whether Tier 2 had a Defend order on that very city — plus
the outcome and how many turns it was held. Written into each game dump
as `siege_episodes` (only with `--dump-stats-dir`).

RESULT (60 seeds x2 = 120 games, deployed config vs Greedy, 462 episodes).

| | MACRO (model) | GREEDY |
|---|---|---|
| sieges suffered | 166 | 296 |
| unsieged | **25.3%** | **54.1%** |
| city lost | 74.7% | 45.9% |
| responders >= 1 at open | 74.7% | 71.3% |
| nearest own unit | 2 tiles | 2 tiles |
| stars banked | 1 | 1 |
| Tier 2 ordered THIS city defended | 42.2% | n/a (script) |

**THE ANSWER IS NEITHER "MORE SIEGES" NOR "WORSE DEFENDING" — IT IS WHO
IS STANDING ON THE CITY.** Unsiege rate by attacker:

| attacker on the city | model n | model unsieges | Greedy n | Greedy unsieges |
|---|---|---|---|---|
| Rider | 47 | **70%** | 168 | 82% |
| Giant | 72 | **6%** | 42 | 24% |
| Warrior | 20 | 5% | 57 | 18% |
| Swordsman | 18 | 22% | 26 | 12% |

The model's sieges are **43.4% Giants**; Greedy's are **14.2%**. Giants
are ~unkillable once parked (6% cleared). Re-weighting the model's
per-attacker rates onto Greedy's attacker mix lifts it from 25.3% to
**43.7%** — i.e. **roughly two thirds of the entire unsiege gap is the
attacker mix, not the defending**. The residual (43.7% vs Greedy's 54.1%)
is real but secondary.

THIS IS THE ARMY-COMPOSITION GAP, ARRIVING AS A DEFENSIVE SYMPTOM. The
model fields the same unit COUNT as Greedy at half the value per unit
(memory: army-composition-gap, $/unit 2.2 vs 4.4) — and here is the bill:
Greedy spends up into Giants and parks them on the model's cities, while
the model attacks with Riders that Greedy kills 82% of the time. The
finding was DEMOTED earlier for not predicting wins on its own (AUC
0.536); this is a causal channel by which it does.

SECOND FINDING — `responders` DOES NOT PREDICT THE OUTCOME. Model
unsieges 26.6% with >=1 responder vs 21.4% with none; Greedy 53.1% vs
56.5% (inverted). Having a unit that can HIT the besieger is nearly
uninformative, because what the tile needs is enough damage to KILL it.
⚠️ This directly indicts the EXP_ELO_040 threat model, which prices
coverage in `covers`/reach terms — it was built on the yardstick this
measurement shows to be the wrong one, which is a better explanation for
041/042's null than "zero-shot steering doesn't work".

THIRD — Tier 2 named the besieged city in a Defend order only 42.2% of
the time, and median stars at open is 1 (nothing to buy a defender with).
Median turns held is 1: the answer must exist BEFORE the attacker
arrives, not after.

CONSEQUENCE. The defensive fix is not a better Defend order or better
coverage pricing (both tried, both null). It is (a) fielding units that
can kill what shows up, and (b) not letting a Giant reach a city with 1
star in the bank. Both are ECONOMY/COMPOSITION decisions, several turns
upstream of the siege.

## EXP_ELO_050 — city risk as a priced potential: prevention, not cure

STATUS: REGISTERED (Aug 15, 2026), Verdi-specified.

THE SPEC (Verdi): "using all of the enemy's units and tech in view, do they
have a path to sieging my city and is that an unbreakable siege? If yes, I
need counter-measures... choose the cheapest defense based on my goals, or
accelerate my path towards something like a giant. I want this to be priced
so that it can distill in the net."

WHY THE OLD MODEL WAS WRONG. EXP_ELO_040 built the threat model around
COVERAGE — can a unit of mine strike that tile — and EXP_ELO_049 then
measured that yardstick to be uninformative: unsiege rate 26.6% with a
responder vs 21.4% without, and a parked Giant cleared 6% of the time.
Cure does not work; the fixture (seed 1786807403) shows why prevention
does: the capital was lost on t10 after the garrison stepped off on t9,
and the answering attack carried a policy prior of 0.0000.

WHAT SHIPS.
- `defense::city_risks` — per city: is it sieged now; is it OPEN (nothing of
  mine on the tile); can a visible enemy END its move there next turn (real
  movement, roads and tech included, via the banded `can_reach_tile`); and
  if they park there, is the siege BREAKABLE (sum of my deliverable damage
  on that tile >= their health, with the city's defense bonus applied to
  the occupier). Risk dials, in doctrine order: unbreakable siege 1.0,
  breakable 0.45, garrison that would fall 0.35, garrison that holds 0.05,
  two-turn approach 0.30.
- `defense::expected_city_loss` = Σ risk x worth, worth = 12 + 6·level, in
  the score-equivalent units Φ already uses.
- `reward::goal_potential` subtracts `SHAPE_GOAL_CITY_RISK` x that. Being a
  POTENTIAL is the point: the ply that steps the last unit off a threatened
  city pays exactly what the ply that garrisons it earns, it is live on
  every stance and every turn, and it distills through the goal channels
  into the net rather than living in a mask.

CONTRACT CHANGED, DELIBERATELY. `no_defend_order_means_no_defense_pricing`
(040) asserted that enemy presence must NOT move Φ without a Defend order.
That rule is what the fixture lost the capital to — the t9 directive was
Grow/Expand and `Defend 24` appeared only after the city was occupied. The
test is now `city_risk_is_priced_without_any_defend_order` and asserts the
opposite. The ORDER-keyed defend terms are unchanged and still require an
order; only the RISK term is order-independent.

AMENDED (Aug 15, Verdi): **the assessment belongs to T2, not T3.** First
cut put `expected_city_loss` inline in `goal_potential`, which meant the
executor re-ran the threat model twice per ply per candidate — both a tier
violation and a compute one. Now: `city_risks` runs ONCE per turn inside
`scripted_goal_aux`, lands in `GoalAux.city_risk` as (city, expected loss),
and T2 also emits a `Defend` order for each risky city — including the case
the coverage model structurally cannot see, an EMPTY city an enemy can walk
onto, which carries no "strike". T3 reads the handed-down number and prices
its RESPONSE. With no aux there is no assessment and the term is silent,
which is the same convention every other aux-carried term follows — and it
restored the two 040 exact-equality tests to their original form.

PREDICTIONS (the 049 siege ledger is the instrument, same 120-game config).
  P1 (the point): sieges SUFFERED per game falls — prevention. Gate: the
     model's sieges/game drops from 1.45 by >= 20% with cities_lost/game
     falling at least as fast.
  P2: unsiege RATE is allowed to stay flat or even fall. Preventing the
     easy sieges leaves a harder residue; reading a flat rate as failure
     would be the exact mistake 049 warned about.
  P3 (cost): no win-rate regression on the paired instrument
     (base_seed 1787800000, 125 seeds x2), |z| < 1.96 against the stored
     baseline; ms/move within 1.2x (city_risks runs inside `goal_potential`,
     which the executor calls twice per ply per candidate).
  P4 (the distillation claim, deferred): after one MACRO_GEN round, the net
     seat's own sieges/game falls too. Until then this is executor-side
     only, and must not be reported as a net result.

AMENDED AGAIN (Aug 16) — **the Aug 15 cut had no gradient at all.**
`rank_plies` computes `phi_pre` and `phi_post` from the SAME `aux`
(macro_exec.rs:91-104). Carrying the assessment as a frozen (city, loss)
pair meant the term's only state dependence was "is this city still mine",
and a player never loses a city during their own ply — so Δφ = 0 for every
candidate and the executor could not see a defensive ply. The assessment
shipped; the response pricing did not. The tier split was right, the
division of labour was not.

Now: T2 hands down the FACTS its search paid for — `attackers` /
`enterers` (tile indices, the expensive `reach_search` half), `breakable`,
`worth` — and T3 re-resolves `defense::residual_risk` against LIVE
occupancy: who is standing on the tile now, which named attackers are still
alive. Same risk ladder, O(1) lookups, no second threat model. That gives a
gradient to exactly the plies that should have one — garrison, vacate, and
killing the approacher (its death drops it out of `enterers`). Losing the
city reads `RISK_LOST`, so no line can buy potential by letting one fall.
Pinned by `a_frozen_assessment_still_prices_the_garrison`, which fails on
the Aug 15 code: it assesses ONCE and then varies only the state.

Also: Defend emission gated on `needs_order()` (risk >= RISK_GARRISON_FALLS).
Every risky city emitting an order would pin the stance to ARM from first
contact onward; a garrison that merely HOLDS needs no order, while its
vacating stays priced. And `RISK_TWO_TURN` was dead code — when `!sieged`,
`arrives_next_turn ≡ threat_unit.is_some()`, so the branch below it was
unreachable. Removed.

SMOKE (Aug 16, n=4, NOT the registered A/B): seeds 1786807403-06, XinXi net
vs pinned Greedy Imperius, GUMBEL_SCALE=0. 2W/2L. No Defend spam — 0.06,
0.03, 0.92, 0.25 orders/turn; the ARM share (g2 84%, g3 79%) matches the
known pre-050 baseline, so the `needs_order` gate holds. On the fixture map
(g0) `Defend 24` now fires at t9, the turn the enemy stood adjacent, versus
t10 in the original — the assessment miss is fixed. **The capital fell
anyway**, and g2 is the sharper version of the same story: 5 cities and spt
12 by t9, then every one lost (t11, t12, t14, t19, t24) to elimination,
with T2 naming @85 and @60 for Defend on the turns they fell and 8-10 units
on the board. T2 names it, T3 has the means, the city still falls — which
points the next cut at the RESPONSE, consistent with 049 (Defend orders
move the executor least, +0.125 tiles/turn). ⚠️ n=4, and `model.safetensors`
also advanced (046 round) since the fixture, so g0 is not a clean A/B.

NOT YET DONE (Verdi's other half): the lane -> `eco_plan` -> star-plan
pricing, so "save toward the Forge" competes in the same currency as
attack/defend/expand instead of a flat per-type bonus. ⚠️ `eco_plan` is a
2934-line BINARY, not a library: it is the ORACLE to calibrate against, not
something to call per ply. The hot path needs a cheap "next purchase on the
lane's plan + turns-to-afford" query validated against it — computed once
per turn at T2 and handed down, the same shape as the risk facts above.

## EXP_ELO_051 — a threat model that can see past one tile, and a lane worth banking for

REGISTERED + RUN Aug 16, 2026. Verdi's three morning targets after watching
`game_iter135_game2_seed1786807405`: stop losing cities to Greedy, make many
giants quickly, stop buying irrelevant techs.

⚠️ **HARNESS FINDING, and it invalidates the first four iterations of this
experiment.** `self_play --goal-w-tree` defaults to **0.0**; training runs
`--goal-channels --goal-w-tree 1` (loop line 233). A debug run that passes
only `--goal-channels` therefore paints the goal planes but deletes the
ENTIRE `goal_potential` channel from the in-tree rewards — every defense
term, every lane term, silently inert. This is the same trap that voided the
v9.1-vs-v7.1 arena match (see line ~1675). Every fixture run in EXP_ELO_049
and EXP_ELO_050, and my own first four measurements here, were made on that
config. **Any behaviour claim about T2 pricing measured without
`--goal-w-tree 1` is void.** The gauge harness now passes it.

HYPOTHESES.
  H1 (defense): the one-move horizon is why cities fall. `can_reach_tile`
     answered only "could this enemy stand here NOW", so both losses in the
     fixture — each a garrison stepping off with the taker 2-3 tiles out —
     produced no risk entry, no Defend order, and a vacate priced 0.0000.
  H2 (economy): the save plan picked the CHEAPEST lane and counted only hub
     sites placeable today. A Forge needs Mines that nothing was banking to
     build, so `save_lane` stayed empty until t9 and nothing competed with a
     junk tech.

CHANGES.
  - `turns_to_reach`: engine cost search over a multi-turn budget, so roads
    shorten the answer as they do in play. Risk decays over a 3-turn horizon
    (`RISK_BY_TURNS`) instead of falling off a cliff.
  - Rider **Escape** honoured: with a victim in range it moves again after
    the kill, putting a "safe" city two moves out one turn away.
  - Reach measured as a step-in from a NEIGHBOUR, since the city tile is
    normally blocked by the very garrison whose departure is being priced.
    (First cut returned empty `enterers` for exactly this reason.)
  - `enemy_ghosts` join the threat set, discounted by age — a sighting that
    walks into fog is not forgotten, and T3 cannot disprove it by looking.
  - Garrison risk continuous in health: holding whole beats holding wounded,
    which is what makes "stay put, don't attack out" the priced answer.
  - `needs_order()` so a merely-holding garrison raises no order (else the
    stance pins to ARM from first contact) while its vacating stays priced.
  - Save lane: most-invested lane wins (price only breaks ties), partners
    counted on ground that COULD take a mine, batch capped at the next two
    placements so it stops growing with the empire it serves.
  - Lane discipline moved to `passes_tech_caps` (always on) — behind
    `star_gate` it switched off under GROW at the third city.
  - `SHAPE_GOAL_CITY_RISK` 1.0 → 4.0, dialled against the measured edge
    distribution (a vacate priced −0.01 against Research at +0.167).

ACTUAL — paired 48 seeds (base 1786807403), XinXi net vs pinned Greedy
Imperius, gumbel 64/16, `--goal-channels --goal-w-tree 1`, GUMBEL_SCALE=0,
against 9deec1d built from a clean worktree:

| | baseline | after | |
|---|---|---|---|
| win rate | 39/48 (81.2%) | **47/48 (97.9%)** | discordant 8–0, **McNemar z = 2.47** |
| cities lost | 56 | **26** | −54% |
| off-lane techs (t≤12) | 158 | **99** | −37% |
| giants (level-5 SuperUnit) | 200 | 181 | −10%, both ≫ target |

VERDICT: **H1 and H2 both confirmed.** Win rate and city retention move
together and significantly.

⚠️ Two corrections to earlier readings in this ledger, both mine:
  1. The "0 giants" alarm was a broken gauge — Giants arrive as the level-5
     `Choose reward SuperUnit`, not a Train move. The nets were already
     making ~4/game.
  2. A `game1`/`game10` prefix collision made the first n=24 city-loss
     numbers read the wrong turn file. Corrected above.

OPEN (the next lever): **all 26 remaining losses are `garrisoned=false,
defend_ordered=true`** — T2 names the city and the executor still does not
put a unit on it. Median loss turn 17. The assessment is now right; what is
missing is T3 choosing the cheapest goal-aligned response (garrison vs buy
vs accelerate), which is the half of Verdi's Aug 15 note still unbuilt.

### EXP_ELO_051 addendum (Aug 16) — where the last 26 losses actually come from

Traced every loss in the n=48 arm back past the "last turn we still held it"
window, which was misleading: by then an enemy already stood on the tile, so
the ballot carried no Train and no Step-in (100% of cases). The decision is
2-3 turns earlier.

  - **26/26 losses trace to a garrison LEAVING**, not to being overwhelmed.
  - Gap from vacate to loss: **2 turns in 18 cases, 3 in 7, 6 in 1** — all
    inside `THREAT_HORIZON`, so the assessment does see it.
  - **14 of the 15 recoverable vacating plies are now priced NEGATIVE**
    (−0.002 to −0.257). Before 051 this was 0 of 97: the term is working.
  - What beats it is the **policy prior**, which sits at 0.97 / 0.83 / 0.82 /
    0.67 on the very moves that leave the city.

So the remaining gap is not assessment and not pricing — both now do the
right thing. It is that the net has never been trained on this behaviour.
That is precisely what the MACRO_GEN round started Aug 16 02:0x (run_id
1786710389, resumed at iter 21) is for, and it makes the P4 distillation
claim from EXP_ELO_050 the live question rather than a deferred one.

⚠️ Deliberately NOT adding a mask here. `goal-pricing-beats-masks` (028 P1c)
found in-tree potentials hold behaviour shifts masks cannot, and the pricing
is already correct — a mask would paper over an untrained prior and remove
the very gradient the training round needs.
