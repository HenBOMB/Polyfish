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

        // Polytopia/TS doesn't usually allow debt, but if we are here we already validated
        // Let's clamp to 0 just in case to match TS state behavior if moves were forced.
        if tribe.stars < 0 {
            tribe.stars = 0;
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
    effect: EffectType,
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
    effect: EffectType,
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

            unit.moved = false;
            unit.attacked = false;
            unit.attacks_performed = 0;

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

/// Check if a unit has an effect
pub fn has_effect(unit: &UnitState, effect: EffectType) -> bool {
    unit.effects.contains(&effect)
}

/// Set visible tiles for a player based on their units and cities
pub fn set_visible_tiles(state: &mut GameState, player_id: PlayerId) -> UndoCallback {
    let old_visibility = state._visible_tiles.clone();

    // Clear current visibility
    state._visible_tiles.clear();

    // Get the tribe
    if let Some(tribe) = state.tribes.get(&player_id) {
        let map_size = state.settings.size;

        // Vision from cities
        for city in &tribe.cities {
            // Cities see their territory plus adjacent
            let city_coords = Coords::from_index(city.tile_index, map_size);
            for dy in -(city.border_size + 1)..=(city.border_size + 1) {
                for dx in -(city.border_size + 1)..=(city.border_size + 1) {
                    let nx = city_coords.x + dx;
                    let ny = city_coords.y + dy;
                    if nx >= 0 && nx < map_size && ny >= 0 && ny < map_size {
                        let idx = ny * map_size + nx;
                        state._visible_tiles.insert(idx, true);
                    }
                }
            }
        }

        // Vision from units
        for unit in &tribe.units {
            // Standard vision range (could be enhanced by Scout skill)
            let vision_range = if crate::functions::has_skill(unit, SkillType::Scout) {
                2
            } else {
                1
            };
            for idx in get_adjacent_indices(state, unit.coords.idx, vision_range) {
                state._visible_tiles.insert(idx, true);
            }
            // Unit's own tile
            state._visible_tiles.insert(unit.coords.idx, true);
        }
    }

    Box::new(move |s| {
        s._visible_tiles = old_visibility;
    })
}

/// Get star exchange amount when discovering a new tribe
pub fn get_star_exchange(state: &GameState, player_id: PlayerId) -> i32 {
    crate::functions::get_star_exchange(state, player_id)
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

    // Check visible tiles (not revelation score, just meeting)
    // REVELATION SCORE is handled in discover_tiles action.
    let visible_tiles: Vec<i32> = state._visible_tiles.keys().cloned().collect();

    for idx in visible_tiles {
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
                // Star exchange based on recipients score
                let stars = get_star_exchange(state, pov_id);
                undos.push(gain_stars(state, stars));

                if let Some(tribe) = state.tribes.get_mut(&pov_id) {
                    tribe.known_players.insert(enemy_owner);
                    undos.push(Box::new(move |s| {
                        if let Some(t) = s.tribes.get_mut(&pov_id) {
                            t.known_players.remove(&enemy_owner);
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
        // Freeze terrain (All except Mountain)
        if let Some(tile) = state.tiles.get_mut(&idx) {
            match tile.terrain_type {
                TerrainType::Water
                | TerrainType::Ocean
                | TerrainType::Forest
                | TerrainType::Field => {
                    if !tile.frozen {
                        tile.frozen = true;
                        undos.push(Box::new(move |s| {
                            if let Some(t) = s.tiles.get_mut(&idx) {
                                t.frozen = false;
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
                            EffectType::Frozen,
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
                        if age >= 3 {
                            new_type = Some(UnitType::BabyDragon);
                        }
                    }
                    UnitType::BabyDragon => {
                        // Baby -> Fire Dragon after 3 more turns (total 6 turns from creation)
                        if age >= 6 {
                            new_type = Some(UnitType::FireDragon);
                        }
                    }
                    _ => {}
                }

                if let Some(target_type) = new_type {
                    let old_type = unit.unit_type;
                    unit.unit_type = target_type;

                    // Add undo
                    undos.push(Box::new(move |s| {
                        if let Some(t) = s.tribes.get_mut(&player_id) {
                            if let Some(u) = t.units.get_mut(unit_idx) {
                                u.unit_type = old_type;
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

    if tribe_type == TribeType::Polaris {
        // TODO: Polaris disabled
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
        let m_owner =
            state
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
                                    state, unit_owner, unit_pos, 4,
                                ));
                                undos.push(try_remove_effect(
                                    state,
                                    unit_owner,
                                    unit_pos,
                                    EffectType::Poison,
                                ));
                            }
                        }
                    }
                }
            }
        }
    }

    // 2. End of turn queue
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
