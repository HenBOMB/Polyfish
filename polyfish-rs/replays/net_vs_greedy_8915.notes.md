# Behavior test fixture: net_vs_greedy_8915 (Verdi, Aug 14)

**Map seed: 1786670356** (Tiny Drylands, gamemode 2, max_turns 30; net=P1
Oumaji-side seat vs Greedy anchor). Regenerate the exact map:
`MapGenSettings { seed: 1786670356, size: Tiny, map_type: Drylands, ... }`.
Replay: `replays/net_vs_greedy_8915.json` (+ stance/checkpoint probes work on it).

Observed mistakes to repeatedly test until fixed (net's own moves — the net
is both strategist-consumer and micro here; the macro executor is NOT in
this loop):
1. **t2–3: center village sieged by a swordsman; two riders available and
   the net declined to unsiege.** Desired behavior: unsiege, then bounce
   the rider back out along an exploration vector — defense and fog
   advance are not exclusive. Stance context (stance_probe): stance GROW
   the whole window, arm intensity 0.50 at t2 (the siege registered!) —
   the signal saw it; the policy didn't act.
2. **t6: wounded rider stepped into enemy range and died for nothing**
   while healthy riders nearby could have sieged the enemy city instead.
3. **Slow Farming, rider over-spam**: stars misallocated to surplus riders
   instead of Farming → pop → levels → giants.

Test harness idea: arena/self_play pinned to seed 1786670356, assert (a)
village unsieged within 1 turn of siege when defender-adjacent army ≥
attacker, (b) no wounded-unit no-gain sacrifices, (c) Farming bought by tN.

Command-record correction (Aug 14): the original game's exact record is
capture-of-60 by P1 at t2 → swordsman attacks t1/t2 → P2 recaptures t3 →
P1 contests piecemeal (t4, t5 single attacks) → P1 retakes t8 and holds.

## Macro-teacher baseline on the same seed (Aug 14)

`replays/macro_vs_greedy_1786670356.json` — same map/tribes/seats
(self_play --base-seed 1786670356 --num-games 1 --search-backend
macro-mcts --anchor-frac 1.0; teacher = sims 32/k 4, heuristic leaf,
λ=1.0). Result: **teacher LOST decisively t19** (capital kill; Vengir
5775). Same opening (rider→60 t1, capture t2, swordsman attacks t2), then:

1. **t3: executor walked the wounded garrison OUT of the center city
   (60→48) with the swordsman adjacent** — gifted the city, P2 recaptured
   t4 with zero combat. Strictly worse than the net (whose garrison at
   least died defending). Raw script read at t3 was **Arm i=0.42** — the
   defend signal was live. Verified mechanism (reward.rs
   `goal_potential`): the order loop skips every kind except Expand —
   **Defend/Attack orders are enumerated, flip stance, paint the net's
   feature planes, and are worth exactly 0 Φ to the executor.** A parked
   garrison scores nothing, while Expand approach gradients actively pay
   it to march off the city. The signal-exists-but-nothing-consumes-it
   pattern again (same shape as the stance-intensity gate finding).
2. Same piecemeal-contest pattern after (2 attacks t7, 1 t9, retake t11,
   lost again t12), then full collapse: villages 25/93/63/96 all Vengir
   by t12, P1 ends 2 techs, 2 SPT, army stars 0 vs 101.
3. Farming: never bought (downstream of eco collapse, not a planner
   choice to test on this branch).

Fixture verdict: the unsiege/garrison gap is NOT a net-distillation
artifact — the teacher itself fails it, harder. Fix belongs in the
executor's ply pricing (hold/garrison term or defend-order consumption),
then re-distill. (stance_probe rows are the raw script read — fresh
StanceCommit per turn, not the search-committed goal.)
