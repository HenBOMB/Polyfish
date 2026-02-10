# OPENING BOOK
Opening book is a must, 99% always the best moves.
`polyfish-rs/src/ai/book.rs` [PARTIAL]

# GAME MODES
#### Domination
Maximize control and military potential? [PARTIAL] (Via Late Game Weights)

#### Perfection
Maximize economy and score potential? [PARTIAL] (Via Early/Mid Game Weights)

Score is all that matters?

# MOVE ORDERING
[IMPLEMENTED] Move ordering logic in MCTS/Search.
- Forced rewards
- Captures (enemy capitals -> ruins -> cities -> villages)
- Unit attacks and kills
- Harvest & Builds that generate instant pop gain / Free task reward structures
- Adjacency structures sorted by most gain [TODO]
- Self healing units, Heal others [TODO]
- Unit steps (steps that explore tiles descending)
- Unit promotions (veterancy +3 kills)
- Unit abilities (explode, boost, freeze, convert)
- Economy abilities (decompose, destroy, diplomacy, enchant animal) [TODO]
- All other harvest moves
- Rewardful abilities (clear forest) [TODO]
- Disband / destructive abilities (disband, burn forest) [TODO]
- Research: [IMPLEMENTED] Scored via `evaluate_tech_utility` ROI + buy-before-capture bonus
- Unit attack suicides

# REWARDING
- Explored tiles value: [IMPLEMENTED]
```
maxExploration = 0.8
formula: (explored - total * (1 - maxExploration)) / (total * maxExploration)
    Will reward 1 when exploring (maxExploration * 100)% of the map
```
- Map control [TODO]

# PENALTIES [PARTIAL]
- Partially upgraded cities [IMPLEMENTED] (Via `evaluator::economy::penalty_partial_cities`)
- When buying tech, USE it, if not penalty [IMPLEMENTED] (Via `evaluator::economy::penalty_unused_tech` — resources, structures, terrain-less chains)
- Frozen, poisoned units are worth less [IMPLEMENTED] (Via `heuristics::assess_unit_power` status modifiers)
- Boosted is worth more [IMPLEMENTED] (Via `heuristics::assess_unit_power` status modifiers)
- Weak units, alone with no other units nearby are worth less [TODO]
- Poorly placed roads that dont connect to nowhere [IMPLEMENTED] (Via `evaluator::economy::penalty_bad_structures`)
- Bad placed structures [IMPLEMENTED] (Via `evaluator::economy::penalty_bad_structures` — lonely adjacency structures)

# HEURISTICS

### Multipliers
- **Game Stage**: [IMPLEMENTED] (Dynamic weights in `evaluator::player`)
    - Early game: Prioritize economic development (SPT), expansion (villages) and exploration (FOW)
    - Mid game: Balanced
    - Late game: Military Dominance

### Rewards: [IMPLEMENTED] (Via `ordering::score_reward` — context-aware per slot)
- **Workshop:** Safest best, +1 SPT.
- **Explorer**: Not best on the first turn, unless there are many tribes to gain discovery stars from.
- **Walls**: Preferred if there are enemies nearby or city is under attack.
- **Resources**: +5 stars, perfect for early game. 
- **Population Growth**: If border growth doesnt give access to terrain that is worth is, choose this.
- **Border Growth**: Only if worth +3 population or more.
- **Park**: Always choose in perfection gamemode, unless strictly losing and army needs more potential score.
- **Super Unit**: Always choose in domination gamemode.

### Tribes

- Use initial starting score and incrementing score per turn to deduce enemy tribe type. [TODO]

### Technology

- Dont waste stars on technology that doesnt favor the explored territory, eg: [IMPLEMENTED] (Via `evaluator::economy::penalty_unused_tech` — terrain-less tech chains penalized)
    - If we dont have any forests, forestry tech is pointless, unless going for mathematics.
    - If no mountains climbing is useless.
    - If no water, sailing is useless.
    - If no game (animal resource), hunting is useless.

- Always buy tech before capturing cities, capturing increases tech cost substantially depending on the tech tier.

### Structures

- Sawmill, Forge, Market, Windmill: [IMPLEMENTED] (Via `ordering.rs` adjacency count scoring — lonely=-2, 2adj=+5, 3adj=+12, 4+=+18)
    - Always build these in the spots where you can maximise the amount of population gained. 
    - Never build alone, unless really worth it, try to build in a cluster of 2 or more.

- Roads: [IMPLEMENTED] (Via `ordering.rs` `score_road()` — manhattan-distance path scoring between city pairs, unconnected city bonus +8, on-path +5, adj road +2, adj city +3)
    - Prioritize roads that connect unconnected cities to capital
    - Roads on the shortest path between two cities are scored highest
    - Deprioritize roads with only 1 city or not on any useful path

- Temple Timing: [IMPLEMENTED] IN PERFECTION MODE, build temples by **Turn 19** to ensure they reach max level (Level 5) by Turn 30. Temples built on Turn 29/30 are useless for leveling. (Via `ordering.rs` temple bonus in `MoveType::Build`)

### Military

- **Units**: [IMPLEMENTED] Direct meta-scoring `UnitValues` (all unit types covered, Polaris disabled) + `assess_unit_power` for HP/status/defense modifiers

- **Veterancy**:
    - Prioritize stacking kills with the same unit, to reach veteran and gain +5 health and fully heal up when near death. [PARTIAL] (Evaluator rewards high HP/Veteran)

- **Unit Combinations**: [TODO]
    - **Rider + Roads**: Essential for hit-and-run tactics. High mobility allows Riders to attack and retreat to safety.
    - **Shields + Archers**: Use Shields (Defenders) to tank damage while Archers deal damage from behind.
    - **Swordsmen spam**: Effective in late game vs almost anything except battleships.

- **Defense**: [TODO]
    - Always end turn on defensive terrain (Mountains with Climbing, Forests with Archery, Cities with Walls) if possible.
    - **Defense Bonus**: 1.5x multiplier is huge. A Defender in a walled city is nearly invincible in early game. Best in ally territory to maximise heal.
    - **Zone of Control**: Remember that enemy units exert a "zone of control" (ZOC) on adjacent tiles, stopping movement. Use this to block enemies from reaching your cities. Also use it to detect enemies in FOG.

### Economy Management [IMPLEMENTED] (Via `evaluator::economy` - Income/Stars/Tech)

- **Customs Houses**: The engine of late-game economy. Plan port locations early to maximize Customs House adjacency (up to 8 ports = +16 stars/turn). [TODO]
- **Sawmills vs Lumber Huts**: Lumber Huts are quick population, but Sawmills offer scalable population growth if you have clusters of forests.
- **Save Stars**: Don't spend to 0 every turn. Saving for a glorious tech (like Trade or Philosophy) is better than buying a Warrior you don't need. [TODO]

### Naval Superiority [TODO]

- **Control the Seas**: In Archipelago or Water World maps, he who rules the waves rules the game.
- **Battleships**: Expensive (15+ stars) but essential. Their 4-range visibility across water (and 2-range attack) are unmatched.
- **Giant Battleships**: The ultimate unit. Upgrade a Giant into a Battleship for massive HP and free stomp damage.

### Fog of War (FOW) [IMPLEMENTED] (Via `evaluator::exploration`)

- **Exploration Value**: Every revealed tile is information. Information reduces the risk of ambushes.
- **Scouting**: Use cheap units (Riders, non-veteran Warriors) to scout ahead before moving valuable units like Giants or Knights.
- **Danger Heuristic**: Be wary of moving into the "fog" with your last unit. If you can't see it, assume there is a Knight waiting to kill you. (depending on the current turn) On early turns the enemy will not have advanced units, so it is safer to move into the fog.

### Advanced Tactics (Web Research) [TODO]

- **Leapfrogging**: Use two Riders to explore Fog of War twice as fast suitable for map control. One moves, expanding vision, the second moves past the first.
- **Vet Assigning**: Don't promote units instantly. Attack with a unit, and if it survives with low HP, promote it to fully heal it and attack again (or survive the retaliation).
- **Damage Calculator Vision**: If you shoot an enemy (e.g., hidden Archer) and they *don't* shoot back next turn, it means you are in their FOW.
- **Markets**: Crucial for high scores/late game.
- **Aggressive Opening**: "Steal" villages. If you see a village an enemy is moving towards but hasn't claimed, get there first even if it delays your own development slightly.
- **Tribe Tiers**:
    - **S-Tier**: Bardur (Economy/Combat balance), Kickoo (Water dominance), Yadakk (Roads/Expansion).
    - **Counters**: Use Riders vs. Archers/Warriors (hit & run). Use Knights vs. spam (chain kill).