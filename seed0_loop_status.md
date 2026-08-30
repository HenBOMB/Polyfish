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

## Baseline (iteration 1, commit 23ee7a1, before this loop's first fix)

- Game length: ends turn 32 (target: <=15) — **MISS, large gap**
- Units lost: 5 (target: <3) — **MISS**
- Units killed: 17 (favorable ratio, not a target itself)
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
matchup, no fixed anchor seat needed).

## Current state

- Iteration: 2 (EXP_ELO_101 shipped/pending -- see ledger for final
  disposition once the paired gauge lands)
- Baseline game file (iteration 1, pre-fix): the byte-identical
  reproduction lives at
  `/private/tmp/.../scratchpad/gauge_101/results/`-adjacent scratch
  copies; canonical repo copy is
  `replays/xinxi_seed0_vs_greedy_exp100_8480.json` (score 8480, 741
  moves, turn 32).
- Treatment game file (iteration 2, post-fix): `replays/fix1_final_seed0_run1.json`
  (score 8580, 514 moves, turn 25, decisive win) -- same seed, same
  recipe, single-variable A/B against the baseline above.
- Debug traces (baseline): `replays/exp100_seed0_watch/` (replay.json,
  decisions.json, game0.jsonl macro-goal ballots, ply_trace.jsonl).
- First finding (ml-expert agent pass 1): siege-pricing inversion at
  city 79, turns 22-29. Root-caused via ground-truth `POLYFISH_PLY_TRACE`
  reconstruction (not the agent's approximate numbers -- see EXP_ELO_101):
  `unit_goal_contest_second`'s "occupied" gate keyed off live-enemy-unit
  presence, so it collapsed to zero the instant a lethal attack killed
  the garrison (capturing a city is a separate move; tile ownership
  stays enemy until then). This, not the `attack_press`/`attack_siege_hold`
  swap (which is a correct, intentional design), was the actual bug.
- Issues fixed so far (this loop):
  1. **EXP_ELO_101** (`unit_goal_contest_second`/`expand_contest_second`
     occupied->contested fix) -- see ledger for full writeup. Controlled
     single-game A/B: wins 7 turns faster (25 vs 32), 227 fewer moves,
     city 79 captured cleanly one turn after the kill instead of the
     wounded-occupier-dies-before-capturing failure mode. Paired gauge
     pending at status-file-write time.
- Still-open findings from pass 1, NOT yet addressed (carry into next
  iteration's analysis, don't silently drop):
  - `defender_dies` pricing (`scoring.rs` ~99-102) is flat (95+15)
    regardless of attacker HP post-retaliation, so lethal-attack
    selection doesn't prefer the healthier attacker when multiple units
    can make the same kill. Plausibly related to why `units_lost` stayed
    at 5 in the iteration-2 game even though the turn-22 decision itself
    is now fixed.

## Log (append one entry per iteration, newest last)

- **2026-08-30, iteration 2 (EXP_ELO_101)**: fixed
  `unit_goal_contest_second`/`expand_contest_second`'s occupied-vs-
  contested confusion. Ground-truth ply verification: flips the flagged
  decision (Attack -75.4 -> +299.6 vs Step's flat 86.0). Controlled
  regenerated-game verification: wins 7 turns faster, 227 fewer moves,
  clean capture-and-convert instead of the historical wounded-occupier
  death. Paired gauge (n=128, seed 770425): baseline 0.3516 (matches
  EXP_ELO_100's own reading exactly), treatment 0.3828, delta +3.13pp
  (favorable, within the ~7.8pp noise floor, not a regression).
  **SHIPPED** — see ledger EXP_ELO_101 for full writeup. Committed.
