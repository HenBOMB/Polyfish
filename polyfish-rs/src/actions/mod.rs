//! Action helper functions for game state manipulation
//!
//! These functions modify game state and return undo callbacks for reversibility.

pub mod city;
pub mod connection;
pub mod discovery;
pub mod resource;
pub mod structure;
pub mod tech;
pub mod units;

use crate::coords::Coords;
use crate::functions::{get_adjacent_indices, get_enemy_at};
use crate::states::*;
use crate::types::*;

/// Type alias for undo callback
pub type UndoCallback = Box<dyn FnOnce(&mut GameState)>;

/// No-op undo callback
pub fn noop_undo() -> UndoCallback {
    Box::new(|_| {})
}

/// Chain multiple undo callbacks into one
pub fn chain_undos(undos: Vec<UndoCallback>) -> UndoCallback {
    Box::new(move |state| {
        for undo in undos.into_iter().rev() {
            undo(state);
        }
    })
}

/// Modify terrain type at a tile index
pub fn modify_terrain(state: &mut GameState, idx: i32, new_terrain: TerrainType) -> UndoCallback {
    let tile = match state.tiles.get_mut(&idx) {
        Some(t) => t,
        None => return noop_undo(),
    };

    let old_terrain = tile.terrain_type;
    tile.terrain_type = new_terrain;

    Box::new(move |s| {
        if let Some(tile) = s.tiles.get_mut(&idx) {
            tile.terrain_type = old_terrain;
        }
    })
}

/// Gain stars for the current player
pub fn gain_stars(state: &mut GameState, amount: i32) -> UndoCallback {
    spend_stars(state, -amount)
}

/// Spend stars for the current player
pub fn spend_stars(state: &mut GameState, amount: i32) -> UndoCallback {
    if amount == 0 {
        return noop_undo();
    }

    let pov_id = state.settings.current_player_turn_id;
    if let Some(tribe) = state.tribes.get_mut(&pov_id) {
        let old_stars = tribe.stars;

        tribe.stars -= amount;

        // println!(
        //     "🐛 Tribe {} {} {} stars: {} -> {} (Turn {})",
        //     pov_id,
        //     if amount > 0 { "spent" } else { "gained" },
        //     amount.abs(),
        //     old_stars,
        //     tribe.stars,
        //     state.settings.turn
        // );

        // Debt is impossible in Polytopia so this means we made an illegal move.
        if tribe.stars < 0 && amount > 0 && state.settings._are_you_sure {
            panic!(
                "BUG: illegal move executed in real game: spent {} stars with only {} available (turn {}, player {})",
                amount, old_stars, state.settings.turn, pov_id
            );
        }

        Box::new(move |s| {
            if let Some(tribe) = s.tribes.get_mut(&pov_id) {
                tribe.stars = old_stars;
            }
        })
    } else {
        noop_undo()
    }
}

/// Try to add an effect to a unit
pub fn try_add_effect(
    state: &mut GameState,
    owner: PlayerId,
    unit_idx: usize,
    effect: UnitEffect,
) -> UndoCallback {
    if let Some(tribe) = state.tribes.get_mut(&owner) {
        if let Some(unit) = tribe.units.get_mut(unit_idx) {
            if unit.effects.contains(&effect) {
                return noop_undo();
            }
            unit.effects.insert(effect);
            return Box::new(move |s| {
                if let Some(tribe) = s.tribes.get_mut(&owner) {
                    if let Some(unit) = tribe.units.get_mut(unit_idx) {
                        unit.effects.remove(&effect);
                    }
                }
            });
        }
    }
    noop_undo()
}

/// Try to remove an effect from a unit
pub fn try_remove_effect(
    state: &mut GameState,
    owner: PlayerId,
    unit_idx: usize,
    effect: UnitEffect,
) -> UndoCallback {
    if let Some(tribe) = state.tribes.get_mut(&owner) {
        if let Some(unit) = tribe.units.get_mut(unit_idx) {
            if !unit.effects.contains(&effect) {
                return noop_undo();
            }
            unit.effects.remove(&effect);
            return Box::new(move |s| {
                if let Some(tribe) = s.tribes.get_mut(&owner) {
                    if let Some(unit) = tribe.units.get_mut(unit_idx) {
                        unit.effects.insert(effect);
                    }
                }
            });
        }
    }
    noop_undo()
}

/// End a unit's turn (set moved and attacked to true)
pub fn end_unit_turn(state: &mut GameState, owner: PlayerId, unit_idx: usize) -> UndoCallback {
    if let Some(tribe) = state.tribes.get_mut(&owner) {
        if let Some(unit) = tribe.units.get_mut(unit_idx) {
            let old_moved = unit.moved;
            let old_attacked = unit.attacked;

            unit.moved = true;
            unit.attacked = true;

            return Box::new(move |s| {
                if let Some(tribe) = s.tribes.get_mut(&owner) {
                    if let Some(unit) = tribe.units.get_mut(unit_idx) {
                        unit.moved = old_moved;
                        unit.attacked = old_attacked;
                    }
                }
            });
        }
    }
    noop_undo()
}

/// Start a unit's turn (reset moved and attacked to false)
pub fn start_unit_turn(state: &mut GameState, owner: PlayerId, unit_idx: usize) -> UndoCallback {
    if let Some(tribe) = state.tribes.get_mut(&owner) {
        if let Some(unit) = tribe.units.get_mut(unit_idx) {
            let old_moved = unit.moved;
            let old_attacked = unit.attacked;
            let old_attacks_performed = unit.attacks_performed;

            unit.moved = false;
            unit.attacked = false;
            unit.attacks_performed = 0;

            return Box::new(move |s| {
                if let Some(tribe) = s.tribes.get_mut(&owner) {
                    if let Some(unit) = tribe.units.get_mut(unit_idx) {
                        unit.moved = old_moved;
                        unit.attacked = old_attacked;
                        unit.attacks_performed = old_attacks_performed;
                    }
                }
            });
        }
    }
    noop_undo()
}

/// Check if a unit has an effect
pub fn has_effect(unit: &UnitState, effect: UnitEffect) -> bool {
    unit.effects.contains(&effect)
}

/// Update exploration for a player based on their units and cities.
/// This marks tiles as explored (permanent, no undo).
/// Only runs during real moves (_are_you_sure = true) to prevent MCTS cheating.
/// Update exploration for a player based on their units and cities.
/// This marks tiles as explored.
/// Only runs during real moves (_are_you_sure = true) to prevent MCTS cheating.
pub fn update_exploration(state: &mut GameState, player_id: PlayerId) -> UndoCallback {
    // CRITICAL: Only modify explorers during real moves, not MCTS simulations
    if !state.settings._are_you_sure {
        return noop_undo();
    }

    let mut modified_tiles: Vec<i32> = Vec::new();

    // Check Internal FOW Toggle (God Mode for AI Training)
    if !state.settings._fow {
        for (idx, tile) in state.tiles.iter_mut() {
            if !tile.explorers.contains(&player_id) {
                tile.explorers.insert(player_id);
                modified_tiles.push(*idx);
            }
        }
    } else if let Some(tribe) = state.tribes.get(&player_id) {
        let map_size = state.settings.size;

        // Collect tiles to explore from cities
        let mut tiles_to_explore: Vec<i32> = Vec::new();

        for city in &tribe.cities {
            let city_coords = Coords::from_index(city.idx, map_size);
            let range = if state.tiles[city.idx as usize].capital_of > 0 {
                2
            } else {
                city.border_size
            };

            for dy in -range..=range {
                for dx in -range..=range {
                    let nx = city_coords.x + dx;
                    let ny = city_coords.y + dy;
                    if nx >= 0 && nx < map_size && ny >= 0 && ny < map_size {
                        let idx = ny * map_size + nx;

                        // Corner hidden rule: Lighthouses are hidden from capitals
                        let is_corner = idx == 0
                            || idx == map_size - 1
                            || idx == map_size * (map_size - 1)
                            || idx == map_size * map_size - 1;
                        if state.tiles[idx as usize].explorers.contains(&player_id)
                            || (is_corner && idx != city.idx)
                        {
                            continue;
                        }

                        tiles_to_explore.push(idx);
                    }
                }
            }
        }

        // Collect tiles to explore from units
        for unit in &tribe.units {
            // Standard vision range: 2 if Scout or on Mountain, else 1
            let vision_range = if crate::functions::has_skill(unit, SkillType::Scout)
                || state
                    .tiles
                    .get(&unit.coords.idx)
                    .map_or(false, |t| t.terrain_type == TerrainType::Mountain)
            {
                2
            } else {
                1
            };
            for idx in get_adjacent_indices(state, unit.coords.idx, vision_range) {
                tiles_to_explore.push(idx);
            }
            tiles_to_explore.push(unit.coords.idx);
        }

        // Mark all collected tiles as explored
        for idx in tiles_to_explore {
            if let Some(tile) = state.tiles.get_mut(&idx) {
                if !tile.explorers.contains(&player_id) {
                    tile.explorers.insert(player_id);
                    modified_tiles.push(idx);
                }
            }
        }
    }

    if modified_tiles.is_empty() {
        noop_undo()
    } else {
        let score_gain = 5 * modified_tiles.len() as i32;
        if let Some(tribe) = state.tribes.get_mut(&player_id) {
            tribe.score += score_gain;
        }

        Box::new(move |s| {
            if let Some(tribe) = s.tribes.get_mut(&player_id) {
                tribe.score -= score_gain;
            }
            for idx in modified_tiles {
                if let Some(t) = s.tiles.get_mut(&idx) {
                    t.explorers.remove(&player_id);
                }
            }
        })
    }
}

/// Legacy wrapper for compatibility - calls update_exploration
/// Deprecated: Use update_exploration directly
#[allow(dead_code)]
pub fn set_visible_tiles(state: &mut GameState, player_id: PlayerId) -> UndoCallback {
    update_exploration(state, player_id)
}

/// Try to discover other tribes that are now visible and reward with star exchange
pub fn try_discover_other_tribes(state: &mut GameState) -> UndoCallback {
    let pov_id = state.settings.current_player_turn_id;
    let max_tribes = state.settings._max_tribe_count;

    // Get our known players count
    let known_count = state
        .tribes
        .get(&pov_id)
        .map(|t| t.known_players.len())
        .unwrap_or(0);

    // Already discovered all tribes
    if known_count >= (max_tribes - 1) as usize {
        return noop_undo();
    }

    let mut undos: Vec<UndoCallback> = Vec::new();

    // Check explored tiles for enemy units
    let explored_tiles: Vec<i32> = state
        .tiles
        .iter()
        .filter(|(_, t)| t.explorers.contains(&pov_id))
        .map(|(&idx, _)| idx)
        .collect();

    for idx in explored_tiles {
        if let Some(enemy) = get_enemy_at(state, idx, pov_id) {
            let enemy_owner = enemy.owner;

            // Check if we already know this tribe
            let already_known = state
                .tribes
                .get(&pov_id)
                .map(|t| t.known_players.contains(&enemy_owner))
                .unwrap_or(true);

            if !already_known {
                // Discover and get stars
                // Star exchange based on MET tribe's score
                let stars = crate::functions::get_star_exchange(state, enemy_owner);

                undos.push(gain_stars(state, stars));

                if state.settings._verbose {
                    if let Some(enemy_tribe) = state.tribes.get(&enemy_owner) {
                        state._messages.push(format!(
                            "Met the {:?}! ({}+ Stars) 🤝",
                            enemy_tribe.tribe_type, stars
                        ));
                    } else {
                        state
                            ._messages
                            .push(format!("Met a new tribe! ({}+ Stars) 🤝", stars));
                    }
                }

                if let Some(tribe) = state.tribes.get_mut(&pov_id) {
                    tribe.known_players.insert(enemy_owner);

                    // Diplomacy Metadata: Record first meeting turn
                    let current_turn = state.settings.turn;
                    let relation_existed = tribe.relations.contains_key(&enemy_owner);
                    let relation = tribe.relations.entry(enemy_owner).or_default();
                    let old_first_meet = relation.first_meet;
                    relation.first_meet = current_turn;

                    undos.push(Box::new(move |s| {
                        if let Some(t) = s.tribes.get_mut(&pov_id) {
                            t.known_players.remove(&enemy_owner);
                            if relation_existed {
                                if let Some(r) = t.relations.get_mut(&enemy_owner) {
                                    r.first_meet = old_first_meet;
                                }
                            } else {
                                t.relations.shift_remove(&enemy_owner);
                            }
                        }
                    }));
                }
            }
        }
    }

    chain_undos(undos)
}

// Redundant get_city_production removed, use crate::functions::get_total_production

/// Freeze water tiles and enemies in an area around a unit
pub fn freeze_area(
    state: &mut GameState,
    freezer_owner: PlayerId,
    freezer_idx: i32,
) -> UndoCallback {
    let mut undos: Vec<UndoCallback> = Vec::new();
    let adjacent = get_adjacent_indices(state, freezer_idx, 1);

    for idx in adjacent {
        // Freeze any water
        if let Some(tile) = state.tiles.get_mut(&idx) {
            match tile.terrain_type {
                TerrainType::Water | TerrainType::Ocean => {
                    if !tile.is_frozen() {
                        undos.push(Box::new(move |s| {
                            if let Some(t) = s.tiles.get_mut(&idx) {
                                t.terrain_type = TerrainType::Ice;
                            }
                        }));
                    }
                }
                _ => {}
            }
        }

        // Freeze enemy units
        if let Some(enemy) = get_enemy_at(state, idx, freezer_owner) {
            let enemy_owner = enemy.owner;
            // Find unit index
            if let Some(tribe) = state.tribes.get(&enemy_owner) {
                for (unit_idx, unit) in tribe.units.iter().enumerate() {
                    if unit.coords.idx == idx {
                        undos.push(try_add_effect(
                            state,
                            enemy_owner,
                            unit_idx,
                            UnitEffect::Frozen,
                        ));
                        break;
                    }
                }
            }
        }
    }

    chain_undos(undos)
}

/// Process effects at the start of a player's turn
pub fn process_start_turn_effects(state: &mut GameState, player_id: PlayerId) -> UndoCallback {
    let mut undos: Vec<UndoCallback> = Vec::new();
    let mut growth_candidates = Vec::new();
    let mut boat_indices = Vec::new();
    let tribe_type;

    // Scope for immutable borrow of state.tribes
    {
        let tribe = match state.tribes.get(&player_id) {
            Some(t) => t,
            None => return noop_undo(),
        };
        tribe_type = tribe.tribe_type;

        // Dragon Growth Logic
        // Iterate through units to find growing dragons
        for (unit_idx, unit) in tribe.units.iter().enumerate() {
            if crate::functions::has_skill(unit, SkillType::Grow) {
                growth_candidates.push(unit_idx);
            }
            // Also collect boat indices
            if unit.passenger_type.is_some() {
                boat_indices.push(unit_idx);
            }
        }
    }

    for unit_idx in growth_candidates {
        // Re-borrow for mutation
        if let Some(tribe) = state.tribes.get_mut(&player_id) {
            if let Some(unit) = tribe.units.get_mut(unit_idx) {
                let age = state.settings.turn - unit.created_turn;

                // Checks for evolution
                let mut new_type = None;

                match unit.unit_type {
                    UnitType::DragonEgg => {
                        // Egg -> Baby Dragon after 3 turns
                        if age > 3 {
                            new_type = Some(UnitType::BabyDragon);
                        }
                    }
                    UnitType::BabyDragon => {
                        // Baby -> Fire Dragon after 3 more turns (total 6 turns from creation)
                        if age > 6 {
                            new_type = Some(UnitType::FireDragon);
                        }
                    }
                    UnitType::InsectEgg => {
                        if age > 2 {
                            new_type = Some(UnitType::Larva);
                        }
                    }
                    UnitType::Larva => {
                        // Larva -> Moth after 3 turns (Total 5 turns: 2 as Egg + 3 as Larva)
                        if age > 5 {
                            new_type = Some(UnitType::Moth);
                        }
                    }
                    _ => {}
                }

                if let Some(target_type) = new_type {
                    let old_type = unit.unit_type;
                    let old_health = unit.health;

                    // Inherit damage: new_hp = new_max - (old_max - current_hp)
                    let old_max_hp = crate::functions::get_unit_max_health(unit);
                    let mut damage = old_max_hp - unit.health;

                    // Support for bug in versions < 115: Damage not inherited
                    if state.settings.version < 115 {
                        damage = 0.0;
                    }

                    unit.unit_type = target_type;
                    let new_max_hp = crate::functions::get_unit_max_health(unit);
                    unit.health = (new_max_hp - damage).max(1.0); // Minimum 1 HP

                    // Add undo
                    undos.push(Box::new(move |s| {
                        if let Some(t) = s.tribes.get_mut(&player_id) {
                            if let Some(u) = t.units.get_mut(unit_idx) {
                                u.unit_type = old_type;
                                u.health = old_health;
                            }
                        }
                    }));
                }
            }
        }
    }

    for boat_idx in boat_indices {
        if let Some(tribe) = state.tribes.get_mut(&player_id) {
            if let Some(_boat) = tribe.units.get_mut(boat_idx) {
                // If passenger is Egg/Baby
                // But wait, passenger is ONLY a UnitType. We don't have its `created_turn`!
                // Major Issue: The current engine implementation of `passenger_type: Option<UnitType>` loses state (health, age, effects) of the passenger.
                // This means we CANNOT track an Egg's age while it is inside a boat.
                // The implementation plan assumes state tracking.
                // However, fixing the entire carrier system to store full UnitState is out of scope for this task.
                // For now, we only implement growth for units on the map.

                // If the user insists "Dragon Egg can be carried by a Raft and continue growing", we'd need to store age in the carrier.
                // But with `processed_start_turn_effects`, we only see on-board units.
                // I will stick to map units for now as per current data model constraints.
            }
        }
    }

    // Temple Growth Logic
    let mut growing_temples = Vec::new();
    // Only check temples owned by current player
    // Iterate all structures is inefficient but simplest for now.
    // Better: Iterate city territories?
    // Given structure map is global, iterating it is O(S). S is small-ish.
    for (idx, structure_opt) in state.structures.iter() {
        if let Some(structure) = structure_opt {
            let is_temple = matches!(
                structure.structure_type,
                StructureType::Temple
                    | StructureType::WaterTemple
                    | StructureType::ForestTemple
                    | StructureType::MountainTemple
                    | StructureType::IceTemple
            );

            if is_temple {
                // Check owner
                if let Some(tile) = state.tiles.get(idx) {
                    if tile.owner == player_id {
                        let age = state.settings.turn - structure.founded;
                        // Level 1: Age 0-2 (or <3)
                        // Level 2: Age 3-5 (or >=3)
                        // Level 3: Age 6-8 (or >=6)
                        // Level 4: Age 9-11 (or >=9)
                        // Level 5: Age 12+ (or >=12)
                        let mut expected_level = 1 + (age / 3);
                        if expected_level > 5 {
                            expected_level = 5;
                        } else if expected_level < 1 {
                            expected_level = 1;
                        }

                        if structure.level < expected_level {
                            growing_temples.push((*idx, expected_level));
                        }
                    }
                }
            }
        }
    }

    // Apply growth
    for (idx, new_level) in growing_temples {
        if let Some(structure) = state.structures.get_mut(&idx).and_then(|s| s.as_mut()) {
            let old_level = structure.level;
            let levels_gained = new_level - old_level;
            structure.level = new_level;

            // Score gain: 100 per level gained
            let score_gain = 100 * levels_gained;
            if let Some(tribe) = state.tribes.get_mut(&player_id) {
                tribe.score += score_gain;
            }

            undos.push(Box::new(move |s| {
                if let Some(tribe) = s.tribes.get_mut(&player_id) {
                    tribe.score -= score_gain;
                }
                if let Some(st) = s.structures.get_mut(&idx).and_then(|x| x.as_mut()) {
                    st.level = old_level;
                }
            }));
        }
    }

    // 2. Fungi Logic (Cymanti Growth)
    let (has_recycling, recycling_discovered_turn) =
        if let Some(tribe) = state.tribes.get(&player_id) {
            let tech = tribe
                .tech_vanilla
                .iter()
                .find(|t| t.tech_type == TechnologyType::Recycling);
            (tech.is_some(), tech.map(|t| t.discovered_turn).unwrap_or(0))
        } else {
            (false, 0)
        };

    let fungi_indices: Vec<i32> = state
        .structures
        .iter()
        .filter(|(_, s)| {
            s.as_ref()
                .map(|st| st.structure_type == StructureType::Fungi)
                .unwrap_or(false)
        })
        .map(|(&idx, _)| idx)
        .collect();

    for f_idx in fungi_indices {
        // Growth can only happen if you own the tile
        let owner_id = state.tiles.get(&f_idx).map(|t| t.owner).unwrap_or(0);
        if owner_id == player_id {
            let mut leveled_up = false;
            let mut new_level = 0;

            if let Some(structure) = state.structures.get_mut(&f_idx).and_then(|s| s.as_mut()) {
                // Growth requirement: turn - founded > 0 (prevents same-turn growth)
                let age = state.settings.turn - structure.founded;
                let max_level = if state.settings.version < 115 {
                    3
                } else {
                    if has_recycling { 3 } else { 2 }
                };

                let can_grow = if structure.level == 1 {
                    age > 0
                } else if structure.level == 2 {
                    // Force wait +1 turn after research: turn must be > discovery turn
                    has_recycling && state.settings.turn > (recycling_discovered_turn + 1)
                } else {
                    false
                };

                if structure.level < max_level && can_grow {
                    structure.level += 1;
                    new_level = structure.level;
                    leveled_up = true;
                }
            }

            if leveled_up {
                println!("Fungi leveled up to level {}", new_level);
                // Undo level change
                undos.push(Box::new(move |s| {
                    if let Some(st) = s.structures.get_mut(&f_idx).and_then(|x| x.as_mut()) {
                        st.level -= 1;
                    }
                }));

                // Add population to city whenever it levels up
                if let Some(city) = crate::functions::get_city_owning_tile(state, f_idx) {
                    undos.push(crate::actions::city::add_population(state, city.idx, 1));
                }
            }
        }
    }

    if tribe_type == TribeType::Polaris {
        // TODO: Polaris disabled
    }

    // Sanctuary Animal Spawning Logic
    let mut spawns = Vec::new();
    for (&idx, structure_opt) in state.structures.iter() {
        if let Some(structure) = structure_opt {
            if structure.structure_type == StructureType::Sanctuary {
                if let Some(tile) = state.tiles.get(&idx) {
                    if tile.owner == player_id {
                        let age = state.settings.turn - structure.founded;
                        // Spawns every 3 turns (turns 3, 6, 9... after building)
                        if age > 0 && age % 3 == 0 {
                            // Find empty adjacent forests
                            let adj = get_adjacent_indices(state, idx, 1);
                            let mut forest_candidates = Vec::new();
                            for a_idx in adj {
                                if let Some(a_tile) = state.tiles.get(&a_idx) {
                                    if a_tile.terrain_type == TerrainType::Forest {
                                        let has_resource = state
                                            .resources
                                            .get(&a_idx)
                                            .and_then(|r| r.as_ref())
                                            .is_some();
                                        // Empty forest: No resource and no structure
                                        if !has_resource
                                            && crate::functions::get_structure_at(state, a_idx)
                                                .is_none()
                                        {
                                            forest_candidates.push(a_idx);
                                        }
                                    }
                                }
                            }
                            if !forest_candidates.is_empty() {
                                // Select one at random using state seed for determinism
                                use rand::SeedableRng;
                                use rand::seq::SliceRandom;
                                let mut rng = rand::rngs::StdRng::seed_from_u64(
                                    (state.initial_seed as u64)
                                        ^ (state.settings.turn as u64)
                                        ^ (idx as u64),
                                );
                                if let Some(&spawn_idx) = forest_candidates.choose(&mut rng) {
                                    spawns.push(spawn_idx);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Apply spawning
    for spawn_idx in spawns {
        undos.push(crate::actions::resource::create_resource(
            state,
            spawn_idx,
            ResourceType::Game,
        ));
    }

    chain_undos(undos)
}

/// Process effects at the end of a player's turn
pub fn process_end_turn_effects(state: &mut GameState, _player_id: PlayerId) -> UndoCallback {
    let mut undos: Vec<UndoCallback> = Vec::new();

    // 1. Mycelium healing (Cymanti)
    let mycelium_tiles: Vec<i32> = state
        .structures
        .iter()
        .filter(|(_, s)| {
            s.as_ref()
                .map(|st| st.structure_type == StructureType::Mycelium)
                .unwrap_or(false)
        })
        .map(|(&idx, _)| idx)
        .collect();

    for m_idx in mycelium_tiles {
        let m_owner = state
            .tiles
            .get(&m_idx)
            .and_then(|t| if t.owner != 0 { Some(t.owner) } else { None });
        if let Some(owner_id) = m_owner {
            let adj = get_adjacent_indices(state, m_idx, 1);
            let mut targets = adj;
            targets.push(m_idx);

            for t_idx in targets {
                if let Some(unit_owner) = state.tiles.get(&t_idx).and_then(|t| t._unit_owner_id) {
                    if unit_owner == owner_id {
                        if let Some(tribe) = state.tribes.get(&unit_owner) {
                            if let Some(unit_pos) =
                                tribe.units.iter().position(|u| u.coords.idx == t_idx)
                            {
                                undos.push(crate::actions::units::heal_unit(
                                    state, unit_owner, unit_pos, 4.0,
                                ));
                            }
                        }
                    }
                }
            }
        }
    }
    // 2. Fungi Logic (Cymanti Poisoning only)
    let fungi_indices: Vec<i32> = state
        .structures
        .iter()
        .filter(|(_, s)| {
            s.as_ref()
                .map(|st| st.structure_type == StructureType::Fungi)
                .unwrap_or(false)
        })
        .map(|(&idx, _)| idx)
        .collect();

    for f_idx in fungi_indices {
        let owner_id = state.tiles.get(&f_idx).map(|t| t.owner).unwrap_or(0);

        // B. Poison
        // Poisons non-Cymanti units that step on it (except flying)
        if let Some(unit) = crate::functions::get_unit_at(state, f_idx) {
            // Find unit details
            let unit_owner = unit.owner;
            // Unit index
            let unit_idx_opt = state
                .tribes
                .get(&unit_owner)
                .and_then(|t| t.units.iter().position(|u| u.coords.idx == f_idx));

            if let Some(unit_idx) = unit_idx_opt {
                // Check exemptions
                let is_cymanti = state
                    .tribes
                    .get(&unit_owner)
                    .map(|t| t.tribe_type == crate::types::TribeType::Cymanti)
                    .unwrap_or(false);
                let is_flying = crate::functions::has_skill(unit, SkillType::Fly);
                let is_ally = state
                    .tribes
                    .get(&owner_id)
                    .and_then(|t| t.relations.get(&unit_owner))
                    .map(|r| r.state == 1)
                    .unwrap_or(false);

                if !is_cymanti && !is_flying && !is_ally && unit_owner != owner_id {
                    undos.push(crate::actions::try_add_effect(
                        state,
                        unit_owner,
                        unit_idx,
                        UnitEffect::Poison,
                    ));
                }
            }
        }
    }

    // 3. Auto-heal idle units (units that didn't move or attack this turn)
    // In Polytopia, if a unit does nothing during its turn, it automatically heals
    // at the end of the turn, same as Recover (4 HP in territory, 2 HP outside).
    let active_player = state.settings.current_player_turn_id;
    if let Some(tribe) = state.tribes.get(&active_player) {
        let idle_units: Vec<(usize, i32)> = tribe
            .units
            .iter()
            .enumerate()
            .filter(|(_, u)| !u.moved && !u.attacked)
            .map(|(idx, u)| (idx, u.coords.idx))
            .collect();

        for (unit_idx, tile_idx) in idle_units {
            let in_territory =
                crate::functions::is_in_own_territory(state, tile_idx, active_player);
            let amount = if in_territory { 4.0 } else { 2.0 };
            undos.push(crate::actions::units::heal_unit(
                state,
                active_player,
                unit_idx,
                amount,
            ));
        }
    }

    // 4. End of turn queue
    let queue: Vec<EndOfTurnAction> = state._end_of_turn_queue.drain(..).collect();
    let old_queue = queue.clone();

    for action in queue {
        match action {
            EndOfTurnAction::Decompose {
                tile_index,
                owner_id,
            } => {
                if let Some(structure) = crate::functions::get_structure_at(state, tile_index) {
                    let settings = crate::settings::structures::get_structure_setting(
                        structure.structure_type,
                    );
                    let cost = settings.cost.unwrap_or(0);
                    if cost > 0 && state.settings.current_player_turn_id == owner_id {
                        undos.push(gain_stars(state, cost));
                    }
                }
                undos.push(crate::actions::structure::destroy_structure(
                    state, tile_index,
                ));
            }
        }
    }

    undos.push(Box::new(move |s| {
        s._end_of_turn_queue = old_queue;
    }));

    chain_undos(undos)
}
