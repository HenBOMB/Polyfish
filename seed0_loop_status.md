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
  5. **EXP_ELO_106** (`CityRisk.attackers` frozen-basis + `defend_kill_advance`
     state-fact term, SHIPPED). Found by an `ml-expert` pass-4 analysis of
     the EXP_ELO_104 game: 9/11 net-seat deaths traced to one Catapult/
     Archer force besieging city 49 t9-t17, with EVERY opportunity to
     remove it priced negative — exactly the "kill-the-besieger" risk
     pre-registered as untested when EXP_ELO_103 shipped. Two distinct
     mechanisms: (a) `need_damage`/`defend_cover`'s waterfall budget was
     derived via a LIVE re-lookup of the attacker's health, so a
     candidate that itself chipped the attacker shrank its own
     comparison's budget mid-eval; (b) a melee kill-and-advance vacates
     the garrison tile the SAME ply it earns `defend_garrison_hold`'s
     credit hardest. Fixed by storing the frozen `UnitState` snapshot in
     `CityRisk.attackers` (no more live re-derivation) and adding
     `defend_kill_advance`, a new state-fact term (042/103 lineage)
     paying a friendly unit that lands on a frozen attacker's own tile.
     A third flagged ply (Giant reinforcement) was deliberately left
     unfixed — it's the same finite-budget zero-sum invariant EXP_ELO_054
     depends on, not a bug. Ground truth: all 3 flagged plies behave
     exactly as predicted (2 flip strongly positive, 1 correctly
     unchanged); 4 must-not-regress reference plies from 101/103/104
     hold bit-exact or improve. Regenerated game confirms the causal
     story directly (not just KPIs): the siege is gone by t14-15 (was
     t17), units_lost 11->7, giants-by-12 4->5 (new high). Two
     independent seed-block gauges (770425 rerun + disjoint 770553) both
     read positive: +5.47pp, +4.69pp. Known open risk (registered, not
     blocking): an age-0 ghost sighting can pass `defend_kill_advance`'s
     filter the same as a visible unit, so the latch could in principle
     fire on an empty remembered tile with no actual kill — not observed
     in any measured game yet. Committed (`ceb429f`).
  6. **EXP_ELO_107** (`attack_capture_complete` state-fact latch,
     SHIPPED). Found by an `ml-expert` pass-5 analysis of the EXP_ELO_106
     game, launched specifically to check the carried-forward
     goal-ballot lead — REFUTED (the ballot converges hard on `Attack
     24` from t15 on). The real mistake was one level downstream: by
     t17 the net had killed the capital's last defender and stood on
     it (p2 down to zero units), but `Capture` scored -160 against
     Recover's +20 every turn for 5 straight turns, because
     `attack_siege_hold`'s state-fact latch (a unit standing on a
     still-ENEMY city) is forfeited the instant Capture itself flips
     ownership — the offense-side mirror of EXP_ELO_101/103/104/106's
     collapse-on-success family, on the terminal action this time.
     Fixed with `attack_capture_complete`, a sibling latch paying the
     same rate once the Attack-ordered target is OWNED by the player
     (not occupied) — can't be re-forfeited by stepping off, and the
     order generator naturally stops re-issuing Attack for cities
     already in the player's own tribe. Ground truth: flagged ply
     -160->+590 bit-exact, matching the agent's own hand-estimate.
     Regenerated game: the stuck loop is gone — Capture fires
     immediately at **turn 18** (was 23), the best turn-count result
     this loop has ever shipped; units_lost/killed/giants-by-12 all
     unchanged (a pure conversion-speed fix). Paired gauge, two
     independent seed blocks: BOTH landed on an exact win-rate tie
     (0.507813/0.507813, 0.546875/0.546875) with a consistent drop in
     average game length (-2.9%, -9.4%) — the fix changes when an
     already-won game closes out, not whether it's won, the same
     "clean wash on win rate, verified underneath" shape EXP_ELO_102
     shipped on. Committed (`a0b1917`).
  7. **EXP_ELO_108** (`unit_goal_approach_unassigned`'s `assigned` set
     read from the frozen `UnitGoalStore` instead of live `tribe.units`,
     SHIPPED). Found by an `ml-expert` pass-6 analysis of the
     EXP_ELO_107 game, launched to check whether the still-open
     EXP_ELO_105 retaliation/exposure diagnosis was still the dominant
     units-lost cause (confirmed: 6/7 losses trace to it) — but the
     agent ALSO found a distinct, sharper bug along the way: a known-
     lethal suicide Attack (engine-confirmed `attacker_dies`, base
     floored at 1.0) still won its ply 101.000 vs a safe Step's 48.104,
     because the dying pursuer dropping out of live `tribe.units`
     un-claimed its Expand target for `unit_goal_approach_unassigned`,
     which then paid a DIFFERENT idle unit's sudden "closest to an
     unassigned target" credit — a death subsidizing itself, the
     collapse-on-transition family inverted. Fixed by reading
     `UnitGoalStore.active_targets()` (frozen for the whole ply)
     instead of re-deriving `assigned` from live unit positions.
     Ground truth: flagged ply 101.000 -> **-1099.000**, matching the
     agent's own hand-estimate. Both must-not-regress reference plies
     (101's capture chain, 107's capture ply) held bit-exact. Added 1
     pinning test (338/338 total). Regenerated game: units_lost
     **7 -> 5** — real progress on the loop's most stubborn KPI.
     Paired gauge, two independent seed blocks: both landed on the
     IDENTICAL small delta (-0.78pp, exactly one game/128 in each),
     smaller than EXP_ELO_104's own shipped -1.56pp wash. Committed
     (`2beabb0`).
  8. **EXP_ELO_110** (`defend_plan_impl`'s live own-roster health floored
     against a per-ply `pre_health` snapshot, SHIPPED). Fixed exactly
     the "category (b)" subsidy EXP_ELO_109's ledger entry registered
     as the next lever: a covering unit's own health is read LIVE, so
     a self-wounding chip shrinks its own waterfall contribution and
     frees budget that re-credits OTHER covering units — `defend_cover`
     pays a FRACTION while the waterfall consumes ABSOLUTE damage, so
     the actor's own pay stays unchanged while it eats far less of the
     shared budget. Fixed with `max(pre-ply health, live health)` — a
     floor, not a clamp, so healing/reinforcement still raises the
     contribution unchanged (EXP_ELO_104/106's gradients provably
     untouched: the flagship t6 heal ply stays Δφ=0.000 exactly) while
     a same-ply self-wound can't shrink it. Ground truth: all 3
     EXP_ELO_109-flagged plies behave exactly as a pass-8 `ml-expert`
     design review predicted — idx122 125→45 (Research now wins by
     115), idx180 623.570→45.000 (the safe Step now wins by 482.638,
     a full reversal), idx266 genuinely unchanged (no Defend order
     active at t15, correctly out of this fix's scope). Two NEW
     reference plies found and confirmed (idx241, idx242). 5 standard
     must-not-regress plies held bit-exact. Added 2 pinning tests
     (342/342 total). Regenerated game: turn 17 held, units_lost 5
     held, units_killed 9 held, giants **5→6** improved, throughput
     UP (52.19 vs 38.56 moves/sec — an O(1) lookup, not a rescan, so
     none of EXP_ELO_105's cost problem). units_lost holding rather
     than dropping was checked, not assumed: id14/id16 still die, but
     traced concretely to whole-turn ply-by-ply sequencing (the chip's
     turn-priority correctly moved after Research, but still gets
     played once nothing better remains) — a separate, known search
     limitation, not a flaw in this fix. Paired gauge, two independent
     seed blocks: BOTH positive (+2.34pp, +3.13pp), same direction and
     similar magnitude. Committed.
- Current best game for the next analysis pass: the **EXP_ELO_110** game
  (`replays/exp110_seed0_watch/`, turn **17**, 5 lost, 9 killed, 6
  giants by t12). Units-lost is still 5 (down from a starting-loop
  value of 29) but still misses the <3 target — the clearest remaining
  gap. id28's t15 death (idx266) is confirmed genuinely unpriced by
  every fix shipped so far (no Defend order active at that ply) and is
  the next concrete lever; 2 of the 5 losses (id7 t6, id12 t12) are
  pass-7-classified as structurally forced and out of scope for a
  pricing fix.
- **EXP_ELO_109 was attempted and REVERTED** (2026-08-31, see the ledger
  for the full writeup) — pass-6/7's proposed move-level lethality-
  exposure penalty (a flat per-star charge on an Attack that leaves its
  own actor newly single-threat-lethal) was ground-truth verified
  bit-exact on all 3 flagged plies AND left all 5 established
  must-not-regress plies untouched, but the regenerated canonical game
  got WORSE in both scopings tried: all-move-types gated units_lost
  5->9 (turn 17->20); narrowed to Attack-only, fire count dropped 4x
  (163->39) but units_lost went to **11** (worse again, turn still 20).
  Confirmed concretely: the baseline's turn-5 `Attack 39->27` (base 110,
  a real kill) got penalized to net 10 and a different move won instead
  — one suppressed correct kill, cascading through the rest of the
  fully-deterministic game. Root cause (advisor, second review): the
  flagged-ply margins (6.2, 5.0, 95.9) and ordinary early-game attack
  values (45-110) occupy the same numeric range, so no flat per-star
  multiplier can separate "flip the 3 flagged chips" from "don't
  suppress ordinary good attacks" — and idx180's outsized 95.9 margin is
  itself mostly an artifact of the STILL-UNFIXED `defend_cover` own-side
  live-read subsidy (pass-7's "category (b)"), not a genuine tactical
  gap. Third empirical failure of the exposure/retaliation-pricing
  family (105's Φ-term, 109 broad, 109 narrow) — no fourth flat-penalty
  attempt without fixing category (b) first. Reverted the `rank_plies`
  wiring and `attack_pricing_probe3`'s parity block together; KEPT
  `combat::lethal_threat_weight` + 2 pinning tests (sound primitive,
  zero false positives on 8 real plies) and `lethality_gate_probe.rs`.
  **Next candidate lever**: fix `defend_plan_impl`'s live own-side
  `hypo_damage` read directly — an asymmetric floor,
  `max(pre-ply value, live value)`, on the acting unit's own waterfall
  contribution (healing still raises it, self-wounding can't shrink
  it). Re-probe idx122/180/266 after that lands before deciding whether
  any residual move-level penalty is needed at all.
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
- **Second candidate fix, unproven, flagged by pass-4 `ml-expert`
  analysis (2026-08-30, not yet investigated)**: goal-ballot
  near-indifference to conversion. In the EXP_ELO_104 game, `Attack 24`
  (the enemy capital) was a live ballot candidate every turn t8-t17 but
  rarely won outright, showing near-ties in the endgame window despite
  a growing force advantage (8 Giants vs 3 enemy units by the late
  game). May explain part of the EXP_ELO_106 game's game-length
  regression (21->23 turns despite the city-49 siege resolving 3 turns
  earlier) — worth checking whether this is unchanged, better, or worse
  post-106 before designing a fix. Possible mechanism, unverified:
  goal-rollout horizon or leaf-value blindness to conversion speed —
  the search may correctly PRICE the capital as worth attacking without
  distinguishing "attack now" from "attack in 5 turns" once the
  attacker's own safety is comparable either way.

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
- **2026-08-30, iteration 7 (pass-4 analysis + EXP_ELO_106)**: `ml-expert`
  pass-4 analysis of the EXP_ELO_104 game found the pre-registered
  "kill-the-besieger" risk from EXP_ELO_103 had actually manifested: a
  Catapult/Archer force besieging city 49 t9-t17 caused 9/11 net-seat
  deaths, and every candidate that damaged, killed, or reinforced past
  it priced negative. My first read of the flagged kill ply misattributed
  it to `attacker_pressure` hitting zero; the advisor corrected this to
  the real mechanism (a melee kill-and-advance physically vacates the
  garrison tile, confirmed by `city_train_blocked` flipping in lockstep)
  and separately identified a SECOND, independent mechanism behind a
  different flagged ply (a live re-lookup of attacker health letting a
  chip attack shrink its own comparison's defend_cover budget), plus
  correctly told me NOT to fix the third flagged ply (a Giant
  reinforcement) since that's the EXP_ELO_054 finite-budget invariant,
  not a bug. Fixed both real mechanisms: `CityRisk.attackers` now stores
  the frozen `UnitState` snapshot instead of a tile index to re-resolve
  live, and a new `defend_kill_advance` state-fact term pays a friendly
  unit landing on a frozen attacker's own tile. Ground truth: all 3
  flagged plies behave exactly as predicted; 4 must-not-regress
  reference plies from 101/103/104 hold bit-exact or (in one case)
  improve further via legitimate compounding with the new term. Added 2
  pinning tests (336/336 total). Regenerated game: siege cleared 3 turns
  earlier (t14-15 vs t17), units_lost 11->7, giants-by-12 4->5 (new
  high). Discovered mid-gauge that same-seed reruns are now BIT-EXACT
  identical (not just "within noise" — every metric matched except the
  games-file name) — recorded as a methodology finding: same-seed
  reruns are a determinism check now, not a second noise sample; used a
  disjoint seed block (770553) for a genuine second data point instead.
  Both independent seed blocks read positive: +5.47pp, +4.69pp.
  **SHIPPED** (`ceb429f`). One open risk registered (ghost-tile hole in
  the new latch, not yet observed in any measured game) and one loose
  end carried forward (game length went up 2 turns despite the earlier
  siege resolution — possibly related to the still-open goal-ballot
  conversion-indifference lead from the pass-4 report). Next: pass 5 on
  the EXP_ELO_106 game, and/or a fresh attempt at the retaliation/
  exposure pricing lever (EXP_ELO_105's diagnosis, still believed
  correct, still without a working implementation).
- **2026-08-31, iteration 8 (pass-5 analysis + EXP_ELO_107)**: `ml-expert`
  pass-5 analysis of the EXP_ELO_106 game, launched specifically to
  check the carried-forward goal-ballot conversion-indifference lead.
  REFUTED as stated — the ballot converges hard on `Attack 24` from
  t15 onward. The real mistake was one level downstream: by t17 the
  net had killed the enemy capital's last defender and stood on it
  (p2 down to zero units, one city) — a won position — but `Capture`
  scored -160 against `Recover`'s +20 for 5 STRAIGHT TURNS (t18-t22),
  only resolving when the enemy's own city growth accidentally
  repriced `spt` enough to tip it positive at t23. Root cause:
  `attack_siege_hold` (EXP_ELO_042) pays a unit standing on a
  still-ENEMY city, so Capture itself — the action the whole order
  exists to produce — flips ownership and forfeits its own credit,
  the offense-side mirror of EXP_ELO_101/103/104/106's collapse-on-
  success family, now on the terminal action rather than a supporting
  one. Fixed with `attack_capture_complete`, a sibling latch paying
  the same rate once the target is player-owned. Ground truth: flagged
  ply -160->+590 bit-exact, matching the agent's own independent
  hand-estimate exactly. Both must-not-regress reference plies (101's
  capture chain, 106's idx291) held bit-exact. Added 1 pinning test
  (337/337 total). Regenerated game: the stuck loop is gone — Capture
  fires immediately at turn 18 (was 23), the best turn-count result
  this loop has ever shipped, with units_lost/killed/giants-by-12 all
  unchanged (a pure conversion-speed fix, as expected). Paired gauge,
  two independent seed blocks: BOTH landed on an exact win-rate tie
  with a consistent drop in average game length (-2.9%, -9.4%) — not
  noise (two disjoint 128-game samples don't land on identical win
  rates by chance), and exactly the shape the mechanism predicts (an
  already-won game closes out faster, doesn't become "more won").
  **SHIPPED** (`a0b1917`). Next: pass 6 on the EXP_ELO_107 game
  (turn-18, 7 lost), and/or the still-unimplemented retaliation/
  exposure pricing lever, now the single clearest remaining path to
  the units-lost <3 target.
- **2026-08-31, iteration 9 (pass-6 analysis + EXP_ELO_108)**: `ml-expert`
  pass-6 analysis of the EXP_ELO_107 game, launched to check whether
  the still-open EXP_ELO_105 retaliation/exposure diagnosis was still
  the dominant units-lost cause. Confirmed: 6 of 7 losses trace to that
  family. But it ALSO found a distinct, sharper bug while investigating:
  a Warrior with an active Expand goal attacked a Defender KNOWING it
  would die to retaliation (engine-confirmed `attacker_dies`, base
  floored at 1.0) — and the ply still won outright, 101.000 vs a safe
  Step's 48.104. Root cause: `unit_goal_approach_unassigned`'s
  "assigned" set was derived from LIVE `tribe.units`, so the dying
  pursuer simply dropping out of the roster this candidate's own move
  un-claimed its Expand target — and a DIFFERENT idle unit's sudden
  "closest to an unassigned target" credit (+1200) outweighed both the
  pursuer's own lost credit (-1000) and its death charge (-100)
  combined. Same collapse-on-transition family as 101/103/104/106/107,
  but inverted into a death subsidy instead of a success forfeiture.
  Fixed by reading `UnitGoalStore.active_targets()` (frozen for the
  whole ply, unaffected by which units are alive in any one candidate's
  simulated state) instead of re-deriving `assigned` from live unit
  positions. Ground truth: flagged ply 101.000 -> **-1099.000**,
  matching the agent's own independent hand-estimate essentially
  exactly. Both must-not-regress reference plies (101's capture chain,
  107's capture ply) held bit-exact. Added 1 pinning test (338/338
  total). Regenerated game: units_lost **7 -> 5**, real progress on
  the loop's most stubborn KPI (down from 29 at the start of this
  loop), turn count also improved slightly (18->17). Paired gauge, two
  independent seed blocks: both landed on the IDENTICAL small delta
  (-0.78pp, exactly one game out of 128 in each) — smaller than
  EXP_ELO_104's own shipped -1.56pp wash. **SHIPPED** (`2beabb0`).
  Next: pass 7 on the EXP_ELO_108 game (turn-17, 5 lost), and/or the
  pass-6-recommended narrower re-attempt of the retaliation/exposure
  lever (a move-level, acting-unit-only post-move lethality penalty
  reusing the already-frozen `threat_units` snapshot rather than
  EXP_ELO_105's broad per-unit-per-candidate `hypo_damage` rescan) —
  still the single clearest remaining path to units-lost <3.
- **2026-08-31, iteration 10 (pass-7 analysis + EXP_ELO_109, REVERTED)**:
  implemented pass-6/7's proposed move-level lethality-exposure penalty
  (a flat per-star charge on an Attack whose actor becomes newly
  single-threat-lethal-exposed post-move). Ground-truth verified
  bit-exact on all 3 flagged plies (idx122 t9, idx180 t12, idx266 t15)
  BEFORE writing code, per standing discipline; a design review first
  computed the sizing box a flat penalty would need (lower bound
  `m > 48` from idx180's 95.93 margin) and flagged a possible upper
  bound from EXP_ELO_106's own shipped reference plies — checked
  empirically via a new `combat::lethal_threat_weight` gate primitive
  and `lethality_gate_probe.rs`, and the feared ceiling never actually
  bound (both reference plies turned out not to gate at all). Wired the
  penalty into `rank_plies` + kept `attack_pricing_probe3` in parity;
  all 3 flagged plies flipped exactly as predicted, all 5
  must-not-regress plies stayed bit-exact, 340/340 tests passed.
  Regenerated the canonical game and got a REGRESSION: units_lost
  5->9, turn 17->20 (all move types gated). Narrowed the gate to
  Attack-only (matching what the 3 flagged plies actually are) — fire
  count dropped 4x (163->39 in a `POLYFISH_DEBUG_LETHALITY` diagnostic)
  but the game got WORSE AGAIN: units_lost 5->11, turn still 20.
  Traced concretely to the baseline's turn-5 `Attack 39->27` (base 110,
  a real kill) getting penalized to net 10 and a worse move winning
  instead, cascading through the rest of the deterministic game. A
  second advisor review found the real root cause: the flagged-ply
  margins (6.2, 5.0, 95.9) overlap ordinary early-game attack values
  (45-110) entirely, so no flat per-star multiplier can separate them —
  and idx180's outsized margin turned out to be mostly an artifact of
  the STILL-UNFIXED `defend_cover` own-side live-read subsidy pass-7
  itself had flagged (and recommended NOT fixing directly, in favor of
  this move-level penalty instead — the empirical result says that
  priority was backwards). This also closes EXP_ELO_105's own open
  "why does the single game disagree with the gauge" hypothesis: both
  109 failures happened on the asymmetric canonical game itself, never
  even reaching the gauge, refuting the idea that the asymmetric game
  specifically tolerates this class of caution. Reverted the
  `rank_plies` wiring and probe3's parity block together in one commit;
  kept `combat::lethal_threat_weight` (2 new pinning tests, zero false
  positives on 8 real plies) and `lethality_gate_probe.rs` as
  independently useful, verified primitives. Third empirical failure of
  the exposure/retaliation-pricing family on this loop (105, 109
  broad, 109 narrow) — next attempt targets category (b) directly
  (the live own-side read in `defend_plan_impl`) instead of a fourth
  flat-penalty variant. `replays/exp109_seed0_watch/` kept on disk as
  the regression record.
- **2026-08-31, iteration 11 (pass-8 design review + EXP_ELO_110,
  SHIPPED)**: pass-8 `ml-expert` was launched specifically to verify
  and design EXP_ELO_109's registered next lever (the live own-side
  `hypo_damage` read in `defend_plan_impl`) BEFORE any code was
  written — confirmed the mechanism against actual source (not just
  the description), ground-truth-verified it against real plies, and
  delivered a precise fix design plus predictions for all 3 previously
  -flagged plies. Independently re-verified the two load-bearing claims
  (the live-health read itself; the fraction-pay-vs-absolute-consumption
  asymmetry in `defend_cover`) against source before implementing.
  Fixed with `max(pre-ply health, live health)` — a floor, not a
  clamp — threaded as a new `pre_health` map through
  `defend_plan_impl`/`goal_potential_inner`/`rank_plies`, built once
  per ply (O(1) lookup per candidate, not a rescan). All 3 flagged
  plies behaved EXACTLY as predicted: idx122 and idx180 both flip
  cleanly (Research/Step now win by 115/482.638 respectively — idx180
  going from losing by 95.9 to winning by 482.638, since most of that
  margin turned out to be this exact subsidy), idx266 correctly stays
  unchanged (no Defend order active, out of scope by design). 2 new
  reference plies found (idx241, idx242), both confirmed. 5 standard
  must-not-regress plies + EXP_ELO_104's flagship heal ply all held
  bit-exact. Added 2 pinning tests (342/342 total). Regenerated game:
  held on turn/units_lost/units_killed, improved on giants (5→6),
  throughput UP not down (O(1) fix, unlike EXP_ELO_105's O(units×
  threats) rescan). units_lost holding rather than dropping was
  investigated, not just accepted: id14/id16 still die, traced
  concretely to a real but separate limitation (ply-by-ply greedy
  turn sequencing — the chip's priority correctly demotes below
  Research, but still plays once nothing better remains later in the
  same turn). Paired gauge, two independent seed blocks: BOTH
  positive (+2.34pp, +3.13pp) — a real signal, not the usual
  wash/tie shape most of this loop's fixes have shown. Also closes
  EXP_ELO_105's asymmetric-vs-symmetric hypothesis from the other
  direction: the correct fix improves both the asymmetric canonical
  game and the symmetric gauge, confirming the tension was in
  EXP_ELO_109's specific (abandoned) implementation, not the
  underlying diagnosis. Committed. Next: id28's t15 death (idx266) is
  now the clearest remaining units-lost lever — genuinely unpriced,
  needs a different mechanism than the Defend-waterfall family; the
  other 2 losses (id7, id12) are structurally forced per pass-7 and
  out of scope for any pricing fix.
