# OPENING BOOK
Opening book is a must, 99% always the best moves.
`polyfish-rs/src/ai/opening.rs`

# GAME MODES
#### Domination
Maximize control and military potential?

#### Perfection
Maximize economy and score potential?

Score is all that matters?

# MOVE ORDERING
- Forced rewards
- Captures (enemy capitals -> ruins -> cities -> villages)
- Unit attacks and kills
- Harvest & Builds that generate instant pop gain / Free task reward structures
- Adjacency structures sorted by most gain
- Self healing units, Heal others
- Unit steps (idk how to sort) *TODO*
- Unit promotions (veterancy +3 kills)
- Unit abilities (explode, boost, freeze, convert)
- Economy abilities (decompose, destroy, diplomacy, enchant animal)
- Rewardful abilities (clear forest)
- Disband / destructive abilities (disband, burn forest)
- Research (least of priorities, but crucial, not sure where this goes) *TODO*
- Unit attack suicides

# REWARDING
- Explored tiles value:
```
maxExploration = 0.8
formula: (explored - total * (1 - maxExploration)) / (total * maxExploration)
    Will reward 1 when exploring (maxExploration * 100)% of the map
```
- Map control

# PENALTIES
- Partially upgraded cities
- When buying tech, USE it, if not penalty
- Frozen, poisoned units are worth less
- Boosted is worth more
- Weak units, alone with no other units nearby are worth less
- Poorly placed roads that dont connect to nowhere
- Bad placed structures

# HEURISTICS

### Rewards:
- **Workshop:** Safest best, +1 SPT.
- **Explorer**: Not best on the first turn, unless there are many tribes to gain discovery stars from.
- **Walls**: Preferred if there are enemies nearby or city is under attack.
- **Resources**: +5 stars, perfect for early game. 
- **Population Growth**: If border growth doesnt give access to terrain that is worth is, choose this.
- **Border Growth**: Only if worth +3 population or more.
- **Park**: Always choose in perfection gamemode, unless strictly losing and army needs more potential score.
- **Super Unit**: Always choose in domination gamemode.

### Tribes

- Use initial starting score and incrementing score per turn to deduce enemy tribe type.

### Technology

- Dont waste stars on technology that doesnt favor the explored territory, eg:
    - If we dont have any forests, forestry tech is pointless, unless going for mathematics.
    - If no mountains climbing is useless.
    - If no water, sailing is useless.
    - If no game (animal resource), hunting is useless.

- Always buy tech before capturing cities, capturing increases tech cost substantially depending on the tech tier.

### Structures

- Sawmill, Forge, Market, Windmill:
    - Always build these in the spots where you can maximise the amount of population gained. 
    - Never build alone, unless really worth it, try to build in a cluster of 2 or more.

- Temple Timing: IN PERFECTION MODE, build temples by **Turn 19** to ensure they reach max level (Level 5) by Turn 30. Temples built on Turn 29/30 are useless for leveling.

### Military Strategy

- **Veterancy**:
    - Prioritize stacking kills with the same unit, to reach veteran and gain +5 health and fully heal up when near death.

- **Unit Combinations**:
    - **Rider + Roads**: Essential for hit-and-run tactics. High mobility allows Riders to attack and retreat to safety.
    - **Shields + Archers**: Use Shields (Defenders) to tank damage while Archers deal damage from behind.
    - **Swordsmen spam**: Effective in late game vs almost anything except battleships.

- **Defense**:
    - Always end turn on defensive terrain (Mountains with Climbing, Forests with Archery, Cities with Walls) if possible.
    - **Defense Bonus**: 1.5x multiplier is huge. A Defender in a walled city is nearly invincible in early game. Best in ally territory to maximise heal.
    - **Zone of Control**: Remember that enemy units exert a "zone of control" (ZOC) on adjacent tiles, stopping movement. Use this to block enemies from reaching your cities. Also use it to detect enemies in FOG.

### Economy Management

- **Customs Houses**: The engine of late-game economy. Plan port locations early to maximize Customs House adjacency (up to 8 ports = +16 stars/turn).
- **Sawmills vs Lumber Huts**: Lumber Huts are quick population, but Sawmills offer scalable population growth if you have clusters of forests.
- **Save Stars**: Don't spend to 0 every turn. Saving for a glorious tech (like Trade or Philosophy) is better than buying a Warrior you don't need.

### Naval Superiority

- **Control the Seas**: In Archipelago or Water World maps, he who rules the waves rules the game.
- **Battleships**: Expensive (15+ stars) but essential. Their 4-range visibility across water (and 2-range attack) are unmatched.
- **Giant Battleships**: The ultimate unit. Upgrade a Giant into a Battleship for massive HP and free stomp damage.

### Fog of War (FOW)

- **Exploration Value**: Every revealed tile is information. Information reduces the risk of ambushes.
- **Scouting**: Use cheap units (Riders, non-veteran Warriors) to scout ahead before moving valuable units like Giants or Knights.
- **Danger Heuristic**: Be wary of moving into the "fog" with your last unit. If you can't see it, assume there is a Knight waiting to kill you. (depending on the current turn) On early turns the enemy will not have advanced units, so it is safer to move into the fog.

### Advanced Tactics (Web Research)

- **Leapfrogging**: Use two Riders to explore Fog of War twice as fast suitable for map control. One moves, expanding vision, the second moves past the first.
- **Vet Assigning**: Don't promote units instantly. Attack with a unit, and if it survives with low HP, promote it to fully heal it and attack again (or survive the retaliation).
- **Damage Calculator Vision**: If you shoot an enemy (e.g., hidden Archer) and they *don't* shoot back next turn, it means you are in their FOW.
- **Markets**: Crucial for high scores/late game.
- **Aggressive Opening**: "Steal" villages. If you see a village an enemy is moving towards but hasn't claimed, get there first even if it delays your own development slightly.
- **Tribe Tiers**:
    - **S-Tier**: Bardur (Economy/Combat balance), Kickoo (Water dominance), Yadakk (Roads/Expansion).
    - **Counters**: Use Riders vs. Archers/Warriors (hit & run). Use Knights vs. spam (chain kill).