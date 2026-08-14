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
