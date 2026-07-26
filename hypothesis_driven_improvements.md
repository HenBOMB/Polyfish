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

