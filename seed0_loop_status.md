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
  when search reveals ITS OWN city is at risk. (2) is under active
  investigation as EXP_ELO_103 (see below) — found and partially fixed
  a severe reward-collapse-on-success bug in the Defend-order pricing,
  structurally the same class as EXP_ELO_101 but on defense and bigger.
  (1) has not been investigated yet this iteration.
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
- Current best game for the next analysis pass: the EXP_ELO_101
  treatment game above (turn 25, 19 lost, 20 killed, 2 giants by t12) --
  still misses all four KPI targets, though by a smaller margin on #1
  and #2 than the iteration-1 baseline.
- Next candidate fix, identified by pass-2 `ml-expert` analysis
  (2026-08-30, not yet implemented — see EXP_ELO_102 once registered):
  **EndTurn-revival floor is a flat constant that can't tell "one
  trapped unit with no non-lethal option, EndTurn is free" from
  "mediocre mid-turn ply."** `macro_exec.rs`'s `ENDTURN_REVIVE_PRICE_DEFAULT
  = -700.0` (line ~246) only revives EndTurn when the best surviving
  candidate scores below a flat -700 (EXP_ELO_075 found flat revival at
  every tested strength net-negative, since it also fires on ordinary
  mediocre plies with real opportunity cost — this is why the floor is
  flat and conservative today). But when the ONLY legal candidates left
  are one already-fully-acted unit's self-lethal attacks, EndTurn has
  ZERO opportunity cost (everyone else already moved) and should always
  win regardless of score. Confirmed at 3 real plies this game (turn 18
  ply 162 unit 3, turn 10 ply 71 unit 8, turn 20 ply 195 unit 36) — one
  (-699.0, a single candidate) missed the -700 floor by exactly 1.0
  point. Proposed fix: a narrow, CONDITIONAL revival (candidate count
  small, all remaining candidates are Attack from the same src,
  attacker_dies=true) — structurally different from the flat floor
  already measured net-negative, so this should be a new mechanism, not
  a tuning of -700's value.
  - **Also checked and REFUTED this pass**: the `defender_dies` flat
    HP-blind pricing lead from pass 1 (`scoring.rs` ~99-106). The
    engine's own combat model already zeroes retaliation damage whenever
    an attack is lethal (`actions/units.rs` ~773-781), independent of
    which unit lands the kill — confirmed across 20 same-target
    alternate-attacker plies in this game, `dmg_to_atk=0.0` in every
    lethal case regardless of attacker choice. This is NOT the mechanism
    behind any of the game's unit losses; drop it from future passes.

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
