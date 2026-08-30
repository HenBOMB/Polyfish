# Seed0 improvement loop — status

Standing loop per Verdi's request (2026-08-30): generate seed0 (XinXi vs
Imperius, net vs Greedy-anchor) with full debug traces, find the clearest
real mistake, root-cause it, fix reward/scoring, verify (flagged decision
+ paired gauge), regenerate, repeat. No time limit. This file is the
resume point across compactions — update it every iteration.

## Operating rules (how "no mistakes" gets operationalized)

- Whole-turn analysis only (never judge a single ply against its
  same-ply competing candidates — CLAUDE.md's own stated trap).
- Every fix ships ONLY if it passes both: (a) the flagged decision
  actually improves on a fresh regenerated game, (b) an n=128 paired
  gauge (seed 770425 Imperius-mirror harness) shows no clear regression.
  If (b) fails, revert and try a different mechanism — do not force a
  fix through because it makes the one flagged game look better.
- Every attempt (shipped or reverted) gets a numbered entry in
  `hypothesis_driven_improvements.md`, continuing the EXP_ELO_NNN
  sequence.
- Stop condition per pass: a full turn-by-turn read of the current best
  game finds nothing else clearly wrong. That ends ONE pass, not the
  loop — regenerate and do another pass, since a fix can change what the
  NEXT mistake looks like.

## Concrete success targets (Verdi, mid-loop)

Not "no mistakes" in the abstract — driving toward four measurable
numbers on the seed0 game (and ultimately the broader eval, for #4):

1. Game ends by turn 15 or sooner.
2. Net loses fewer than 3 units in combat; kills many more of the
   enemy's than it loses.
3. Net has at least 3 giants before turn 12.
4. Net wins 100% of the time (read as: at/near 100% on the standard
   eval harnesses, not just this one seed — literal eternal 100% isn't
   a provable target).

## KPI tooling note (read before trusting any units_lost/units_killed number)

`examples/game_kpis.rs` originally compared unit COUNTS once per turn —
a death masked by a same-turn Train/Summon (net count unchanged) was
silently invisible. It reported "5 lost" for both the iteration-1 and
iteration-2 games, which made EXP_ELO_101 look like it hadn't touched
units_lost at all. Fixed (2026-08-30, during iteration-2 analysis) to
diff unit-ID SETS per move instead of counts per turn; verified against
an independent full-replay roster diff. All units_lost/units_killed
numbers below are the CORRECTED ones — if you find an older number
quoted as "5" anywhere, it's stale.

## Baseline (iteration 1, commit 23ee7a1, before this loop's first fix)

- Game length: ends turn 32 (target: <=15) — **MISS, large gap**
- Units lost: 29 (target: <3) — **MISS, large gap**
- Units killed: 25 (favorable ratio, not a target itself)
- Giants by turn 12: 2 (target: >=3) — **MISS, close**
- Win rate: this game won; eval_seeds broader win rate 47.5% (100
  seeds/200 games, see hypothesis_driven_improvements.md) — **MISS,
  large gap** on the "100% of the time" framing

## Recovered "vs-Greedy watch" recipe (needed for every future iteration)

Misremembered twice this session before being recovered verbatim from
the pre-compaction transcript. This is the canonical single-game debug
recipe -- use it exactly, every iteration, or the "opponent" silently
stops being Greedy or the network stops being consulted at leaves:

```
self_play --num-games 1 --search-backend macro-mcts --macro-leaf net-asym \
  --macro-sims 64 --macro-k 8 --macro-rollout-lambda 0.0 \
  --goal-channels --goal-w-tree 1 \
  --base-seed 1787500020 --tribe1 XinXi --tribe2 Imperius --gamemode 2 \
  --anchor-frac 1.0 --iteration 100 --anchor-decay-start 100 --anchor-seat 2 \
  --dump-games-dir <dir>
```
`POLYFISH_PLY_TRACE`/`--dump-macro-policy` are auto-set to `<dir>` by
`--dump-games-dir` for macro-mcts games -- no need to pass separately.
`--anchor-decay-start` MUST equal `--iteration` (both 100 here) or
`anchor_frac` silently decays to its ~10% floor and BOTH seats end up
macro-mcts (no Greedy opponent at all) despite `--anchor-frac 1.0`
looking right in the banner. Verify by checking `ply_trace.jsonl` has
ONLY player 1 rows (macro-mcts's own trace writer) -- if player 2 shows
up too, the anchor pin didn't take.

The n=128 paired-gauge harness (Imperius mirror, from EXP_ELO_100) uses
the identical flags with `--num-games 128 --tribe1 Imperius --tribe2
Imperius --base-seed 770425 --actors 14` and no `--anchor-seat` (mirror
matchup, no fixed anchor seat needed). This harness showed a real
run-to-run gap on the `net-asym` leaf during EXP_ELO_101's gauge (two
treatment-arm rereads of the identical config: 0.3828 then 0.3672) —
unlike the fully-deterministic `heuristic`-leaf harness EXP_ELO_100 used.
Always rerun at least the arm you're about to ship on before trusting a
single reading; baseline reproduced exactly both times, so this is
specific to something `net-asym` touches, not a regression of
EXP_ELO_091's move-gen determinism fix.

## Current state

- Iteration: 4 in progress. EXP_ELO_101 and EXP_ELO_102 both shipped.
  A stop-hook check (2026-08-30) redirected focus mid-iteration-3 to a
  specific behavioral demand: demonstrate the net (1) exploiting a
  weakly-defended ENEMY city and (2) emitting a visible defense signal
  when search reveals ITS OWN city is at risk. Both are now satisfied:

  **Condition (1), offense — already satisfied by existing evidence,
  assembled here rather than re-investigated (per advisor review):**
  EXP_ELO_101's fix is exactly this mechanism. Ground-truth verified on
  a real seed0 ply: a lethal Attack on a contested-but-weakly-defended
  city flips from -75.4 (Step alternative scored higher) to +299.6
  once the pricing correctly treats "killed the garrison, city not yet
  captured" as still-contested rather than falsely "safe" (the old
  `occupied`-vs-`contested` bug read tile ownership, not live-enemy
  presence, and zeroed contest value the instant the garrison died).
  On the regenerated EXP_ELO_101 game this produces a clean
  capture-and-convert: city 79 is captured cleanly one turn after the
  kill, instead of the historical pattern where the wounded occupier
  died before capturing and the opportunity was lost. This is search
  correctly recognizing and exploiting an opening the moment a city's
  defender is beatable — the offense-side demonstration.

  **Condition (2), defense — EXP_ELO_103, SHIPPED this iteration:**
  found and fixed a severe reward-collapse-on-success bug in
  `defend_cover`/`defend_hold`/`defend_recall` (structurally the same
  class as EXP_ELO_101 but on defense, and larger — see full writeup
  below and the ledger entry). Regenerated-game verification is
  unambiguous: city 49's tile ownership never flips to the enemy for
  the ENTIRE 23-turn game post-fix (was captured/lost 5 times
  pre-fix), stays garrisoned nearly every turn from turn 12 onward
  including by two different Giants, and the game also hit
  giants-by-turn-12 = 3 for the first time this loop (Verdi's target).
  Paired gauge in progress (see EXP_ELO_103 entry).
- Baseline game file (iteration 1, pre-fix): `replays/xinxi_seed0_vs_greedy_exp100_8480.json`
  (score 8480, 741 moves, turn 32). Reproduced byte-for-byte against the
  real historical run once the correct generation recipe (above) was
  recovered — confirms the recipe and that EXP_ELO_101 is the only
  variable in the iteration-2 comparison below.
- Treatment game file (iteration 2, EXP_ELO_101 applied): `replays/exp101_seed0_watch/game_iter100_game0_seed1787500020.replay.json`
  (score 8580, 514 moves, turn 25, decisive win) -- same seed, same
  recipe, single-variable A/B against the baseline above. Debug traces
  in the same directory (decisions.json, ply_trace.jsonl, game0.jsonl).
- Iteration-1 debug traces (baseline): `replays/exp100_seed0_watch/`.
- Issues fixed so far (this loop):
  1. **EXP_ELO_101** (`unit_goal_contest_second`/`expand_contest_second`
     occupied->contested fix, SHIPPED). Root cause: the "occupied" gate
     keyed off live-enemy-unit presence, so it collapsed to zero the
     instant a lethal attack killed a contested target's garrison
     (capturing a city is a separate move; tile ownership stays enemy
     until then) -- exactly the discrete-pricing anti-pattern CLAUDE.md
     flags. NOT the `attack_press`/`attack_siege_hold` swap, which
     turned out to be a correct, intentional design (EXP_ELO_042).
     Ground-truth ply verification: flips the flagged decision
     (Attack -75.4 -> +299.6 vs Step's flat 86.0). Controlled
     regenerated-game A/B: wins 7 turns faster (25 vs 32), 227 fewer
     moves, city 79 captured cleanly one turn after the kill instead of
     the historical wounded-occupier-dies-before-capturing failure.
     Corrected KPIs: units_lost 29->19, units_killed 25->20 (real
     ~34% reduction, not the "unchanged" the buggy KPI tool first
     suggested). Paired gauge (n=128, seed 770425): baseline 0.3516
     (exact match to EXP_ELO_100's own historical reading), treatment
     0.3828 / 0.3672 across two runs, both +1.6 to +3.1pp, within the
     ~7.8pp noise floor. Committed (`8d4e5c1`, `3e260b2`).
  2. **EXP_ELO_102** (`revive_endturn_for_lone_doomed_unit`, narrow
     conditional EndTurn revival, SHIPPED). Root cause: the flat -700
     EndTurn-revival floor can't tell "one doomed unit, zero
     opportunity cost, EndTurn should always win" from "mediocre ply
     with real opportunity cost" (the EXP_ELO_075 regression shape).
     New mechanism fires ONLY when every remaining candidate shares one
     source unit, is Attack, and is provably lethal to the attacker.
     Controlled regenerated-game verification: fires exactly at the
     predicted ply (turn 10, global idx 146), units_lost 19->9 (more
     than halved), game length 25->22 turns. Paired gauge (n=128, seed
     770425): exact tie (46/128 both arms) but per-game logs clearly
     diverge and the shared EndTurn-chosen-despite-alternatives counter
     rose 1.737%->2.367%, confirming the mechanism fired more and
     reshaped games without moving the aggregate win rate — a clean
     wash, no regression. Committed.
  3. **EXP_ELO_103** (`defend_plan_open_framing` + `defend_garrison_hold`
     + attacker-pressure decoupling, SHIPPED). Root cause: same class
     of bug as EXP_ELO_101 (collapse-on-success) but on defense and
     bigger — `defend_cover`/`defend_hold`/`defend_recall` all read a
     garrison-conditioned `need_damage`, so the instant any covering
     unit's arrival satisfied it, EVERY unit's anticipatory credit
     (not just the mover's) collapsed together. Fixed by giving
     cover/hold/recall a garrison-independent `need_damage` basis and
     scaling all four defend terms by `attacker_pressure` (reachability)
     instead of risk-derived `urgency`, which stays garrison-coupled.
     Ground-truth ply verification matched the advisor's independent
     hand-computed estimate to within 1 point (-2800.175 -> -95.413 for
     the flagged Step; the real demonstration is a same-ply Summon
     candidate, -2515.470 -> +158.670, now the top-ranked option).
     Regenerated-game verification: city 49 held for the ENTIRE 23-turn
     game (was captured/lost 5 times pre-fix), garrisoned nearly every
     turn from turn 12 on including by two Giants; giants-by-turn-12
     hit 3 for the first time this loop. Paired gauge (n=128, seed
     770425, two runs per arm): baseline avg 0.359375, treatment avg
     0.488281, **+12.89pp**, clearing the ~7.8pp noise floor in every
     pairing checked. Known open risk (not blocking, registered in the
     ledger): killing a city's last besieger could deflate the same
     defend pool the same way a garrison landing used to — no real-game
     ply found yet to test it. Committed.
  4. **EXP_ELO_104** (`defend_plan` waterfall garrison-exclusion +
     `defend_recall`'s own-tile exclusion, SHIPPED). Found by an
     `ml-expert` pass-3 analysis of the EXP_ELO_103 game: healing a
     threatened garrison scored -331.43 (Δφ -371.43) vs abandoning it
     at +5.68 — bit-verified against HEAD. Root cause: EXP_ELO_103
     excluded the garrison from RECEIVING defend_cover credit but not
     from CONSUMING waterfall budget, so a healthier garrison (more
     `hypo_damage`, HP-scaled) crowded out other units' credit for
     nothing in return. Fixed by excluding the garrison from the
     waterfall entirely. Caught a second-order bug while testing: doing
     so also let the garrison win `defend_recall`'s "nearest unassigned
     unit" search against its own city (distance 0), masking the real
     recall signal — 2 unit tests failed and pinpointed this exactly;
     fixed by excluding the city tile itself from that search. Verified:
     t6 heal ply Δφ 0.000 exactly (no longer punished); both EXP_ELO_103
     reference plies unaffected or improved as predicted (Summon-at-49
     +158.670 -> +715.813). Paired gauge (n=128, seed 770425, 2 runs/arm):
     -1.56pp average, comfortably inside the noise floor — baseline
     reproduced 103's own gauge average almost exactly, confirming this
     isn't 103's gain being given back. Committed.
- Current best game for the next analysis pass: the EXP_ELO_104 game
  (turn 21, 509 moves, score 8245, 11 lost, 14 killed, 4 giants by t12).
  Still misses turn-count (<=15) and units-lost (<3) targets, though by
  a smaller margin than every prior iteration.
- **EXP_ELO_105 was attempted and REVERTED** (2026-08-30, see the
  ledger for the full writeup) — the retaliation/exposure pricing gap
  below is still the diagnosed primary lever, but the specific fix
  tried (a broad, always-on per-unit `unit_exposure` Φ term, gated to
  Arm stance, `lethality^2`-damped) failed its own paired gauge:
  -8.59pp average win rate across 2x2 cross-pairings (every pairing
  negative, not noise-shaped), PLUS a real ~30-45% self-play throughput
  regression from the added `hypo_damage` calls -- this despite the
  mechanism being individually verified correct on 8/8 ground-truth
  reference plies, and despite the single regenerated seed0 game
  looking spectacular (turn 21->14, units_lost 11->3, clearing the
  turn-count target for the first time). This is exactly the kind of
  gap the loop's own paired-gauge discipline exists to catch — do NOT
  ship a fix on the strength of one flagged game looking better.
  Leading hypothesis for the disagreement (unverified): the seed0 game
  is asymmetric (net usually pressing an advantage, where caution costs
  little); the gauge is a symmetric mirror where the same caution may
  cost real tempo in an evenly-matched fight. Code fully reverted to
  EXP_ELO_104's committed state. Candidate directions for a future
  retry are in the ledger entry — none yet designed.
- Next candidate fix, identified by pass-3 `ml-expert` analysis of the
  EXP_ELO_103 game (2026-08-30, diagnosis still believed correct; the
  EXP_ELO_105 attempt above was the wrong IMPLEMENTATION of this fix,
  not evidence the diagnosis itself is wrong):
  **retaliation damage and post-move exposure are priced at zero.**
  9-10 of 12 net-seat losses in the EXP_ELO_103 game trace directly or
  upstream to the same shape: a non-lethal chip Attack into a healthy
  Defender wins its ply by 5-25 points over a safe Step (e.g. t15 ply
  idx252, Attack 68->57: 45.00 vs 40.00; reproduces bit-exact under
  HEAD), the unit eats 7-22 retaliation, ends at 1-2hp adjacent to what
  it just chipped, and dies on the enemy's next ply — in one case
  (id=30, t18) to the exact Defender it chipped. Root cause, all
  code-verified: (1) `scoring.rs`'s Attack base (lines ~82-117) reads
  `preview.damage_to_attacker` ONLY via a binary `attacker_dies` check
  — 0 retaliation and near-lethal retaliation price identically; (2)
  the Step branch (~551-715) has no threat/danger term of any kind —
  a 1hp unit's step into enemy reach and its step away differ only by
  curiosity/territory pulls; (3) the φ layer only prices exposure
  through CITIES (`city_risk`, the defend family) — a unit in open
  field carries zero risk potential; (4) `rank_plies` is one-ply greedy,
  so the death (next enemy ply) is beyond every simulation's horizon —
  no tree branch ever proposes the safe alternative. Proposed fix
  shape (not yet designed in detail): a continuous per-unit exposure
  potential — Φ charged per own unit as (missing HP) × (enemy lethality
  within strike reach), from the already-frozen `threat_units` list
  `rank_plies` builds (don't rescan per-candidate, per EXP_ELO_061).
  Must-not-regress probes before shipping: EXP_ELO_101's capture chain
  (an exposure charge on a wounded contested-city occupier must not
  re-suppress it), the t11 Giant kill (Attack 41->51, defender_dies,
  319.66) must stay above a same-tile suicide chip, and fire-rate/
  distribution across the n=128 gauge per the EXP_ELO_075 lesson (this
  term will fire on nearly every ply, not just the 9 flagged ones).
  - **Also checked and REFUTED this pass**: a claimed reproducibility
    gap in two suicide-attack candidate scores (+840.00 off from HEAD,
    twice, both first-ply-of-turn) is a probe-side turn-start context
    reconstruction issue, not a production/dirty-binary problem — the
    lib rebuilds incrementally in 0.4s against this exact source,
    ruling out a stale binary. Standing recipe fix: record the
    generating binary's MD5 into the dump dir going forward; keep
    trace-score forensics to mid-turn plies until this is chased down.

## Log (append one entry per iteration, newest last)

- **2026-08-30, iteration 2 (EXP_ELO_101)**: fixed
  `unit_goal_contest_second`/`expand_contest_second`'s occupied-vs-
  contested confusion. Ground-truth ply verification: flips the flagged
  decision (Attack -75.4 -> +299.6 vs Step's flat 86.0). Controlled
  regenerated-game verification: wins 7 turns faster, 227 fewer moves,
  clean capture-and-convert instead of the historical wounded-occupier
  death; corrected KPIs show units_lost 29->19, units_killed 25->20.
  Paired gauge (n=128, seed 770425): baseline 0.3516 (matches
  EXP_ELO_100's own reading exactly), treatment 0.3828 / 0.3672 across
  two runs (favorable, within the ~7.8pp noise floor, not a
  regression). **SHIPPED** — see ledger EXP_ELO_101 for full writeup.
  Committed.
- **2026-08-30, iteration 3, pass 2 analysis (pre-fix)**: ml-expert
  agent analysis of the EXP_ELO_101 game found the EndTurn-revival-floor
  issue above (candidate for EXP_ELO_102) and refuted the carried-over
  `defender_dies` lead from pass 1. Also surfaced and fixed the
  `game_kpis.rs` undercounting bug (see KPI tooling note above).
  Implementation of EXP_ELO_102 in progress.
- **2026-08-30, iteration 3 (EXP_ELO_102)**: implemented
  `revive_endturn_for_lone_doomed_unit`. Controlled regenerated-game
  verification: fires exactly at the predicted ply, units_lost 19->9,
  game length 25->22. Paired gauge (n=128, seed 770425): exact tie
  (46/128 both arms) but per-game logs diverge and the shared
  EndTurn-chosen-despite-alternatives counter rose 1.737%->2.367% —
  confirmed the mechanism fired and reshaped games without moving the
  aggregate win rate, a clean wash. **SHIPPED** (`ab80424`).
- **2026-08-30, iteration 4 (stop-hook redirect + EXP_ELO_103)**: a
  stop-hook check demanded concrete evidence of two behaviors: the net
  exploiting weakly-defended enemy cities (offense), and emitting a
  visible defense signal when its own city is at risk (defense).
  Offense was already satisfied by EXP_ELO_101's capture-chain evidence
  (assembled into "Current state" above, not re-investigated). Defense
  required real work: built `city49_probe.rs`, found a real 5-flip
  contested-city pattern, root-caused it with `attack_pricing_probe3.rs`
  to the same collapse-on-success bug class as EXP_ELO_101 but on
  defense (`defend_cover`/`defend_hold`/`defend_recall` all collapsing
  together the instant a garrison landed). First attempt (scaling a new
  garrison-hold term by `urgency`) was circular; advisor review caught
  it, corrected to `attacker_pressure`; a second advisor pass then
  caught that cover/hold/recall were STILL on the circular `urgency`
  signal even after garrison_hold was fixed, and the full decoupling
  (fix described in ledger EXP_ELO_103) is what actually worked.
  Regenerated-game verification: city 49 held for the entire 23-turn
  game (was lost 5 times), giants-by-12 hit 3 (first time this loop).
  Paired gauge (n=128, seed 770425, 2 runs/arm): +12.89pp average,
  clearing the noise floor in every pairing. **SHIPPED**. This is the
  largest single behavioral/gauge improvement of the loop so far.
- **2026-08-30, iteration 5 (pass-3 analysis + EXP_ELO_104)**:
  `ml-expert` pass-3 analysis of the EXP_ELO_103 game found three
  things: (1) the retaliation/exposure pricing gap above — the primary
  remaining lever, not yet fixed; (2) a bit-verified sign inversion in
  the just-shipped EXP_ELO_103 code (healing a threatened garrison
  priced at -371.43); (3) a probe-side reproducibility anomaly on two
  suicide-attack scores, checked and attributed to turn-start context
  reconstruction, not production. Independently reproduced (2) against
  HEAD before trusting it, then fixed it as EXP_ELO_104: excluded the
  garrison from `defend_plan`'s waterfall entirely (not just from
  receiving credit), which surfaced a second-order bug caught by 2
  failing unit tests (the garrison-free `assigned` list also broke
  `defend_recall`'s own-tile exclusion) — fixed alongside. Verified via
  3 pre-registered falsifiers, all met. Paired gauge (2 runs/arm):
  -1.56pp average, a clean noise-floor wash, baseline reproducing
  103's own gauge average almost exactly. **SHIPPED**. Next: design and
  implement the exposure-pricing fix (Finding 1), the loop's biggest
  remaining lever for units_lost<3.
- **2026-08-30, iteration 6 (EXP_ELO_105, REVERTED)**: designed and
  implemented `unit_exposures`/`unit_exposure` (per-unit lethality Φ
  term) across two advisor passes -- the first pass caught a
  suicide-relief exploit before any code was written (a doomed unit's
  death is exposure RELIEF, so it must be priced strictly below the
  unit's own `arm_value` death charge or dying becomes the cheap way
  out) and resolved the cost-weighting question as the mechanism that
  enforces that invariant. Ground-truth verified on 8 reference plies:
  5 clean fixes (flagged chip attacks flip from winning their ply to
  losing it, e.g. t13 idx191 49.32->-8.28 vs 39.81), 1 correctly
  untouched (Grow-stance ply, outside this fix's deliberate scope), 2
  unchanged as required (Giant kill, Summon-at-49 stay on top). Single
  regenerated seed0 game: turn 21->14 (first time clearing the <=15
  target), units_lost 11->3. Paired gauge (n=128, seed 770425, 2x2
  cross-pairings): -8.59pp average, EVERY pairing negative (not a
  noise pattern), plus every treatment throughput reading below every
  baseline reading (~30-45% slower generation). **REVERTED** per the
  loop's own rule -- a real, measured gauge regression beats one good
  flagged game, however dramatic. Full writeup and candidate future
  directions in the ledger; diagnosis (Finding 1) still believed
  correct, this specific implementation was not the right fix.
