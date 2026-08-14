//! Core game functions and utilities
//!
//! This module contains the helper functions for querying and manipulating game state.

/// Check if a tile is in the player's own territory
pub fn is_in_own_territory(state: &GameState, idx: i32, owner: i32) -> bool {
    let result = state
        .tiles
        .get(&idx)
        .map(|t| t.owner == owner)
        .unwrap_or(false);
    // if idx == 147 {
    //     println!(
    //         "🐛 Tile 147, Owner={}, QueryOwner={}, Result={}",
    //         state.tiles.get(&idx).map(|t| t.owner).unwrap_or(-1),
    //         owner,
    //         result
    //     );
    // }
    result
}

use crate::coords::Coords;
use crate::settings::get_unit_setting;
use crate::settings::has_technology;
use crate::states::*;
use crate::types::*;

/// Get the current player's tribe (POV = Point of View)
pub fn get_pov_tribe(state: &GameState) -> Option<&TribeState> {
    state.tribes.get(&state.settings.current_player_turn_id)
}

/// Get the current player's tribe mutably
pub fn get_pov_tribe_mut(state: &mut GameState) -> Option<&mut TribeState> {
    state.tribes.get_mut(&state.settings.current_player_turn_id)
}

/// Check if the game is over
pub fn is_game_over(state: &GameState) -> bool {
    if state.settings._game_over {
        return true;
    }

    // Count alive tribes
    let alive_count = state
        .tribes
        .values()
        .filter(|t| t.killed_turn <= 0 && t.resigned_turn <= 0)
        .count();

    // Game over if only one tribe left
    if alive_count <= 1 {
        return true;
    }

    // Check turn limit as a safety boundary for all modes (prevents infinite training games)
    if state.settings.turn > state.settings.max_turns {
        return true;
    }

    false
}

/// Get indices in a plus sign pattern (4 cardinal neighbors)
pub fn get_plus_sign_indices(idx: i32, size: i32) -> Vec<i32> {
    let mut indices = Vec::new();
    let (x, y) = (idx % size, idx / size);

    if x > 0 {
        indices.push(idx - 1);
    }
    if x < size - 1 {
        indices.push(idx + 1);
    }
    if y > 0 {
        indices.push(idx - size);
    }
    if y < size - 1 {
        indices.push(idx + size);
    }

    indices
}

/// Get indices in a square pattern (radius R, includes center)
pub fn get_square_indices(idx: i32, radius: i32, size: i32) -> Vec<i32> {
    let mut indices = Vec::new();
    let cx = idx % size;
    let cy = idx / size;
    for y in (cy - radius)..=(cy + radius) {
        for x in (cx - radius)..=(cx + radius) {
            if x >= 0 && x < size && y >= 0 && y < size {
                indices.push(y * size + x);
            }
        }
    }
    indices
}

/// Calculate chebyshev distance between two flat indices
pub fn get_chebyshev_distance(a: i32, b: i32, size: i32) -> i32 {
    let (ax, ay) = (a % size, a / size);
    let (bx, by) = (b % size, b / size);
    (ax - bx).abs().max((ay - by).abs())
}

/// Calculate squared euclidean distance between two flat indices (avoids sqrt)
pub fn get_squared_euclidean_distance(a: i32, b: i32, size: i32) -> i32 {
    let (ax, ay) = (a % size, a / size);
    let (bx, by) = (b % size, b / size);
    let dx = ax - bx;
    let dy = ay - by;
    dx * dx + dy * dy
}

/// Get adjacent tile indices
pub fn get_adjacent_indices(state: &GameState, idx: i32, range: i32) -> Vec<i32> {
    let size = state.settings.size;
    let coords = Coords::from_index(idx, size);
    let mut result = Vec::new();

    for dy in -range..=range {
        for dx in -range..=range {
            if dx == 0 && dy == 0 {
                continue;
            }
            let nx = coords.x + dx;
            let ny = coords.y + dy;
            if nx >= 0 && nx < size && ny >= 0 && ny < size {
                result.push(ny * size + nx);
            }
        }
    }

    result
}

/// Get adjacent tiles
pub fn get_adjacent_tiles<'a>(state: &'a GameState, idx: i32, range: i32) -> Vec<&'a TileState> {
    get_adjacent_indices(state, idx, range)
        .into_iter()
        .filter_map(|i| state.tiles.get(&i))
        .collect()
}
/// Check if a resource type is visible to a tribe (based on tech requirements)
pub fn is_resource_visible_to_tribe(
    state: &GameState,
    resource_type: ResourceType,
    tribe_id: PlayerId,
    idx: Option<i32>,
) -> bool {
    use crate::settings::technology::has_technology;
    use crate::types::TechnologyType;

    let tribe = match state.tribes.get(&tribe_id) {
        Some(t) => t,
        None => return false,
    };

    // Check FOW: must be explored by current player to see resources
    if idx.is_some() && !is_tile_explored(state, idx.unwrap(), tribe_id) {
        return false;
    }

    // Cannot see the resource if its blocked by a structure
    if idx.is_some() && get_structure_at(state, idx.unwrap()).is_some() {
        return false; // strict? should be only ruins but visually its the same thing
    }

    match resource_type {
        // Fruit, Starfish, and Spores are always visible
        ResourceType::Fruit
        | ResourceType::Starfish
        | ResourceType::Spores
        | ResourceType::None => true,
        // Crop requires Organization OR Farming
        ResourceType::Crop => {
            has_technology(&tribe.tech_vanilla, TechnologyType::Organization)
                || has_technology(&tribe.tech_vanilla, TechnologyType::Farming)
        }
        // Metal requires Climbing
        ResourceType::Metal => has_technology(&tribe.tech_vanilla, TechnologyType::Climbing),
        // Game (animal) requires Hunting
        ResourceType::Game => has_technology(&tribe.tech_vanilla, TechnologyType::Hunting),
        // Fish requires Fishing
        ResourceType::Fish => has_technology(&tribe.tech_vanilla, TechnologyType::Fishing),
        // AquaCrop (Aquarion) - requires FreeDiving
        ResourceType::AquaCrop => has_technology(&tribe.tech_vanilla, TechnologyType::FreeDiving),
    }
}

/// Get resource at a tile (respects tech visibility and FOW for current player)
pub fn get_resource_at(state: &GameState, idx: i32) -> Option<ResourceType> {
    let pov_id = state.settings.current_player_turn_id;

    let resource_type = state
        .resources
        .get(&idx)
        .and_then(|r| r.as_ref())
        .map(|r| r.resource_type)?;

    // Check tech visibility for current player
    if is_resource_visible_to_tribe(state, resource_type, pov_id, Some(idx)) {
        Some(resource_type)
    } else {
        None
    }
}

/// Get structure at a tile
pub fn get_structure_at(state: &GameState, idx: i32) -> Option<&StructureState> {
    state.structures.get(&idx).and_then(|s| s.as_ref())
}

/// Get structure type at a tile
pub fn get_structure_type_at(state: &GameState, idx: i32) -> Option<StructureType> {
    get_structure_at(state, idx).map(|s| s.structure_type)
}

/// Find unit at a tile index
pub fn get_unit_at<'a>(state: &'a GameState, idx: i32) -> Option<&'a UnitState> {
    for tribe in state.tribes.values() {
        for unit in &tribe.units {
            if unit.coords.idx == idx {
                // // Skip ships - return the passenger conceptually
                // let setting = get_unit_setting(unit.unit_type);
                // if setting.skills.contains(&SkillType::Carry) {
                //     if let Some(passenger) = unit.passenger_type {
                //         if passenger != UnitType::None {
                //             continue; // The "true" unit is the passenger
                //         }
                //     }
                // }
                return Some(unit);
            }
        }
    }
    None
}

/// Find the actual unit at a tile (including ships, not just passengers)
pub fn get_true_unit_at<'a>(state: &'a GameState, idx: i32) -> Option<&'a UnitState> {
    for tribe in state.tribes.values() {
        for unit in &tribe.units {
            if unit.coords.idx == idx {
                return Some(unit);
            }
        }
    }
    None
}

/// Find unit at a tile index (mutable)
pub fn get_unit_at_mut<'a>(state: &'a mut GameState, idx: i32) -> Option<&'a mut UnitState> {
    for tribe in state.tribes.values_mut() {
        for unit in &mut tribe.units {
            if unit.coords.idx == idx {
                return Some(unit);
            }
        }
    }
    None
}

/// Get city at a tile index
pub fn get_city_at<'a>(state: &'a GameState, idx: i32) -> Option<&'a CityState> {
    for tribe in state.tribes.values() {
        for city in &tribe.cities {
            if city.idx == idx {
                return Some(city);
            }
        }
    }
    None
}

/// Get the city that owns a tile
pub fn get_city_owning_tile<'a>(state: &'a GameState, idx: i32) -> Option<&'a CityState> {
    let tile = state.tiles.get(&idx)?;
    let ruling_coords = tile.ruling_city_coords.as_ref()?;
    get_city_at(state, ruling_coords.idx)
}

/// Get enemy unit at a tile (not matching given owner)
/// Check if a player is an enemy of another player
pub fn is_enemy(state: &GameState, pov: PlayerId, other: PlayerId) -> bool {
    if pov == other || other == 0 {
        return false;
    }
    !is_at_peace(state, pov, other)
}

pub fn get_enemy_at<'a>(
    state: &'a GameState,
    idx: i32,
    not_owner: PlayerId,
) -> Option<&'a UnitState> {
    get_unit_at(state, idx).filter(|u| {
        is_enemy(state, not_owner, u.owner) && !u.effects.contains(&UnitEffect::Invisible)
    })
}

pub fn get_true_enemy_at<'a>(
    state: &'a GameState,
    idx: i32,
    not_owner: PlayerId,
) -> Option<&'a UnitState> {
    get_true_unit_at(state, idx).filter(|u| is_enemy(state, not_owner, u.owner))
}

/// Check if a unit has an effect
pub fn has_effect(unit: &UnitState, effect: UnitEffect) -> bool {
    unit.effects.contains(&effect)
}

pub fn has_skill(unit: &UnitState, skill: SkillType) -> bool {
    // Check base unit skills
    if crate::settings::units::has_skill(unit.unit_type, skill) {
        return true;
    }
    // Check passenger skills if any
    if let Some(passenger) = unit.passenger_type {
        if crate::settings::units::has_skill(passenger, skill) {
            return true;
        }
    }
    false
}

/// Check if a unit is considered "Amphibious"
pub fn is_amphibious(unit: &UnitState) -> bool {
    has_skill(unit, SkillType::Amphibious)
}

/// Get the maximum health of a unit (accounting for veteran status and pass-through)
pub fn get_unit_max_health(unit: &UnitState) -> f32 {
    // Health is determined by the passenger (inner unit) if available
    let u_type = unit.passenger_type.unwrap_or(unit.unit_type);
    let mut health = crate::settings::units::get_unit_setting(u_type).health;
    if health < 0.0 {
        health = 10.0;
    }
    if unit.veteran {
        health += 5.0;
    }
    health
}

/// Get unit attack strength (accounting for Boost)
pub fn get_unit_attack(state: &GameState, unit: &UnitState) -> f32 {
    let mut atk = crate::settings::units::get_unit_setting(unit.unit_type).attack;

    // If the unit type itself has no attack (like Transportship), inherit from passenger
    if atk < 0.0 {
        if let Some(passenger) = unit.passenger_type {
            atk = crate::settings::units::get_unit_setting(passenger).attack;
        }
    }

    if has_effect(unit, UnitEffect::Boosted) && state.settings.version < 115 {
        atk += 0.5;
    }
    atk
}

/// Get unit defense strength (accounting for Poison)
pub fn get_unit_defense(unit: &UnitState) -> f32 {
    let mut def = crate::settings::units::get_unit_setting(unit.unit_type).defense;
    if has_effect(unit, UnitEffect::Poison) {
        def *= 0.7; // 30% defense reduction
    }
    def
}

/// Get unit movement (accounting for Boost, Poison, and Skate)
pub fn get_unit_movement(state: &GameState, unit: &UnitState) -> i32 {
    let mut movement = crate::settings::units::get_unit_setting(unit.unit_type).movement;

    // Boost bonus
    if has_effect(unit, UnitEffect::Boosted) {
        movement += 1;
    }

    // Skate bonus: +1 movement on Ice (Resulting in 3 movement for standard skating units)
    if has_skill(unit, SkillType::Skate) {
        if let Some(tile) = state.tiles.get(&unit.coords.idx) {
            if tile.is_frozen() {
                movement += 1;
            }
        }
    }

    // Poison slows down enemy units: reduces movement by 1 (but cannot reduce to 0)
    // not sure about this
    // if has_effect(unit, UnitEffect::Poison) && movement > 0 {
    //     movement = (movement - 1).max(1);
    // }
    movement
}

/// Get tech cost tier from a technology
pub fn get_tech_tier(tech_type: TechnologyType) -> i32 {
    crate::settings::technology::get_technology_setting(tech_type)
        .tier
        .unwrap_or(1)
}

/// Get defense bonus multiplier for a unit on its current tile
pub fn get_defense_bonus(state: &GameState, unit: &UnitState) -> f32 {
    // Poisoned units can now receive defense bonuses (defense is just reduced)

    let tribe = state.tribes.get(&unit.owner).unwrap();
    let tile = state.tiles.get(&unit.coords.idx).unwrap();

    let bonus = match tile.terrain_type {
        TerrainType::Water | TerrainType::Ocean => {
            let aquatism = crate::settings::technology::resolve_tech_for_tribe(
                TechnologyType::Aquatism,
                tribe.tribe_type,
            );
            if crate::settings::technology::has_technology(&tribe.tech_vanilla, aquatism) {
                1.5
            } else {
                1.0
            }
        }
        TerrainType::Forest => {
            let archery = crate::settings::technology::resolve_tech_for_tribe(
                TechnologyType::Archery,
                tribe.tribe_type,
            );
            if crate::settings::technology::has_technology(&tribe.tech_vanilla, archery) {
                1.5
            } else {
                1.0
            }
        }
        TerrainType::Mountain => {
            let climbing = crate::settings::technology::resolve_tech_for_tribe(
                TechnologyType::Climbing,
                tribe.tribe_type,
            );
            if crate::settings::technology::has_technology(&tribe.tech_vanilla, climbing) {
                1.5
            } else {
                1.0
            }
        }
        _ => {
            // City defense: applies to friendly units (owner or ally) with Fortify skill
            if let Some(city) = get_city_at(state, unit.coords.idx) {
                // Must be a friendly city and require skill
                if has_skill(unit, SkillType::Fortify)
                    && !is_enemy_city(state, city.idx, unit.owner)
                {
                    if city.has_walls() { 4.0 } else { 1.5 }
                } else {
                    1.0
                }
            } else {
                1.0
            }
        }
    };

    if has_effect(unit, UnitEffect::Poison) {
        if bonus == 4.0 {
            return 2.0;
        } else if bonus == 1.5 {
            return 0.75;
        } else {
            return 0.5;
        }
    }

    bonus
}

/// Get city production (stars per turn)
pub fn get_city_production(state: &GameState, city: &CityState) -> i32 {
    // If city is on riot or the tile is occupied by an enemy then production is nullified
    if is_under_siege(state, city.idx) {
        return 0;
    }

    let mut prod = city.production;
    // Capitals get a +1 star bonus
    if let Some(tile) = state.tiles.get(&city.idx) {
        if tile.capital_of == city.owner && tile.capital_of != 0 {
            prod += 1;
        }
    }

    let rewards = city
        .rewards
        .iter()
        // workshops or parks give +1 production
        .filter(|r| **r == CityRewardType::Workshop || **r == CityRewardType::Park);

    prod += rewards.count() as i32;

    // Clathrus and Market bonuses
    for &idx in &city._territory {
        if let Some(structure) = crate::functions::get_structure_at(state, idx) {
            let setting =
                crate::settings::structures::get_structure_setting(structure.structure_type);

            // Adjacency Bonuses
            match structure.structure_type {
                StructureType::Clathrus => {
                    // +X star for each adjacent Algae in friendly territory
                    let adj = crate::functions::get_adjacent_indices(state, idx, 1);
                    for n_idx in adj {
                        if let Some(n_tile) = state.tiles.get(&n_idx) {
                            if n_tile.effects.contains(&TileEffect::Algae)
                                && n_tile.owner == city.owner
                            {
                                prod += setting.reward_stars;
                            }
                        }
                    }
                }
                StructureType::Market => {
                    // +X star for each adjacent "production" building (Windmill, Sawmill, Forge)
                    let adj = crate::functions::get_adjacent_indices(state, idx, 1);
                    for n_idx in adj {
                        if let Some(n_struct) = crate::functions::get_structure_at(state, n_idx) {
                            if setting.adjacent_types.contains(&n_struct.structure_type) {
                                prod += setting.reward_stars;
                            }
                        }
                    }
                }
                StructureType::IceBank => {
                    // +2 stars for every 20 frozen tiles globally
                    let frozen_count = count_frozen_tiles(state);
                    let income = ((frozen_count / 20) * 2).min(60); // Max level 30 = 60 stars
                    prod += income;
                }
                StructureType::Sanctuary => {
                    // +1 star for every adjacent wild animal in friendly territory
                    let adj = crate::functions::get_adjacent_indices(state, idx, 1);
                    for n_idx in adj {
                        if let Some(res) = state.resources.get(&n_idx) {
                            if let Some(r) = res {
                                if r.resource_type == ResourceType::Game {
                                    // Must be in player's territory
                                    if let Some(n_tile) = state.tiles.get(&n_idx) {
                                        if n_tile.owner == city.owner {
                                            prod += 1;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    // Apply income penalty for negative population (-1 star per turn per red bar, minimum 0)
    if city.progress < 0 {
        prod = (prod + city.progress).max(0);
    }

    prod
}

/// Get total production for a list of cities
pub fn get_total_production(state: &GameState, cities: &[CityState]) -> i32 {
    cities.iter().map(|c| get_city_production(state, c)).sum()
}

/// Get total tribe SPT (stars per turn)
pub fn get_tribe_spt(state: &GameState, tribe: &TribeState) -> i32 {
    get_total_production(state, &tribe.cities)
}

/// Check if two tribes are at peace
pub fn is_at_peace(state: &GameState, tribe_a: PlayerId, tribe_b: PlayerId) -> bool {
    if let Some(tribe) = state.tribes.get(&tribe_a) {
        if let Some(relation) = tribe.relations.get(&tribe_b) {
            return relation.state == 1;
        }
    }
    false
}

/// Check if a city is under siege (has enemy unit on its center)
pub fn is_under_siege(state: &GameState, city_idx: i32) -> bool {
    if let Some(tile) = state.tiles.get(&city_idx) {
        return get_enemy_at(state, city_idx, tile.owner).is_some();
    }
    false
}

/// Check if a tile is a city center
pub fn is_city(state: &GameState, idx: i32) -> bool {
    state
        .tiles
        .get(&idx)
        .and_then(|t| t.ruling_city_coords.as_ref())
        .map_or(false, |c| c.idx == idx)
}

/// Check if a tile is an enemy city center
pub fn is_enemy_city(state: &GameState, idx: i32, pov_id: PlayerId) -> bool {
    if !is_city(state, idx) {
        return false;
    }
    state
        .tiles
        .get(&idx)
        .map_or(false, |t| is_enemy(state, pov_id, t.owner))
}

pub fn is_enemy_capital(state: &GameState, idx: i32, pov_id: PlayerId) -> bool {
    if !is_enemy_city(state, idx, pov_id) {
        return false;
    }
    state
        .tiles
        .get(&idx)
        .map_or(false, |t| t.capital_of != pov_id)
}

/// Check if a tile is frozen (has Ice terrain or Polaris climate)
pub fn is_tile_frozen(state: &GameState, idx: i32) -> bool {
    state.tiles.get(&idx).map_or(false, |t| {
        t.terrain_type == TerrainType::Ice || t.climate == TribeType::Polaris
    })
}

/// Count total frozen tiles globally
pub fn count_frozen_tiles(state: &GameState) -> i32 {
    let mut count = 0;
    for idx in state.tiles.keys() {
        if is_tile_frozen(state, *idx) {
            count += 1;
        }
    }
    count
}

/// Check if coordinate is in bounds
pub fn is_in_bounds(x: i32, y: i32, size: i32) -> bool {
    x >= 0 && x < size && y >= 0 && y < size
}

/// Check if a tile is occupied by a unit
pub fn is_tile_occupied(state: &GameState, idx: i32) -> bool {
    crate::functions::get_unit_at(state, idx).is_some()
}

/// Check if a tile is explored by a player (Fog of War)
pub fn is_tile_explored(state: &GameState, idx: i32, player_id: PlayerId) -> bool {
    if !state.settings._fow {
        return true;
    }
    state
        .tiles
        .get(&idx)
        .map(|t| t.explorers.contains(&player_id))
        .unwrap_or(false)
}

/// Get the capital city of a tribe
pub fn get_capital_city(state: &GameState, player_id: PlayerId) -> Option<&CityState> {
    state.tribes.get(&player_id).and_then(|tribe| {
        tribe.cities.iter().find(|c| {
            state
                .tiles
                .get(&c.idx)
                .map(|t| t.capital_of == player_id)
                .unwrap_or(false)
        })
    })
}

/// Get the "best" capital for a tribe.
/// Priority: Original capital > Highest level city that is a capital.
pub fn get_best_capital(state: &GameState, player_id: PlayerId) -> Option<&CityState> {
    let tribe = state.tribes.get(&player_id)?;

    // 1. Try to find the original capital
    if let Some(orig) = get_capital_city(state, player_id) {
        return Some(orig);
    }

    // 2. Otherwise, find the highest level city that is currently a capital
    // TODO: fix
    tribe
        .cities
        .iter()
        .filter(|c| {
            state
                .tiles
                .get(&c.idx)
                .map(|t| t.capital_of != 0)
                .unwrap_or(false)
        })
        .max_by_key(|c| c.level)
}

/// Get the number of units in a city
pub fn get_city_unit_count(state: &GameState, city: &crate::states::CityState) -> i32 {
    let mut count = 0;
    for tribe in state.tribes.values() {
        for unit in &tribe.units {
            if let Some(home) = &unit.home_coords {
                if home.idx == city.idx {
                    count += 1;
                }
            }
        }
    }
    count
}

/// Calculate a valid position to push a unit to
pub fn calculate_pushable_position(state: &GameState, unit: &UnitState) -> Option<i32> {
    let size = state.settings.size;
    let initial_x = unit.coords.x;
    let initial_y = unit.coords.y;

    let center_x = size / 2;
    let center_y = size / 2;

    let (dx, dy) = if let Some(target) = unit.last_attack_coords {
        // Rule 3: Ranged units (or any unit that attacked)
        let settings = get_unit_setting(unit.unit_type);
        let is_ranged = settings.range > 1;

        // "last move or last attack"
        // If it attacked, the attack direction is typically the "last" action for ranged units
        // unless they have Escape and moved after.
        let attacker_moved_after = has_skill(unit, SkillType::Escape) && unit.moved;

        if is_ranged && unit.attacked && !attacker_moved_after {
            let mut dx = if target.x > initial_x {
                1
            } else if target.x < initial_x {
                -1
            } else {
                0
            };
            let mut dy = if target.y > initial_y {
                1
            } else if target.y < initial_y {
                -1
            } else {
                0
            };

            // Apply flip for enemies (treating attack direction as movement direction for consistency)
            if unit.owner != state.settings.current_player_turn_id {
                dx = -dx;
                dy = -dy;
            }

            if state.settings._verbose {
                println!("Chosen ranged using attacked and !moved after");
            }

            (dx, dy)
        } else if unit.moved
            && unit.prev_coords.is_valid()
            && unit.prev_coords.idx != unit.coords.idx
        {
            if state.settings._verbose {
                println!("Chosen ranged using moved");
            }
            // Revert to move logic if move was last or not a ranged attack
            calculate_move_push_direction(state, unit)
        } else {
            if state.settings._verbose {
                println!("Chosen ranged using center");
            }
            get_direction_toward_center(initial_x, initial_y, center_x, center_y)
        }
    } else if unit.prev_coords.is_valid() && unit.prev_coords.idx != unit.coords.idx {
        if state.settings._verbose {
            println!("Chosen using moved");
        }
        calculate_move_push_direction(state, unit)
    } else {
        if state.settings._verbose {
            println!("Chosen using center");
        }
        // Rule 4: Not previously moved or attacked: push towards center
        get_direction_toward_center(initial_x, initial_y, center_x, center_y)
    };

    // If tile occupied or impassable, try CCW then CW one tile at a time.
    // 8 neighbors. Direction (dx, dy) is target.
    // Order: Target, Target rotated CCW 45, Target rotated CW 45, Target CCW 90, ...
    // Wait, "try counterclockwise and then clockwise one tile at a time".
    // Does this mean: [Target, Target+CCW1, Target+CW1, Target+CCW2, Target+CW2...] ?
    // Or strictly spiraling one way?
    // "counterclockwise and then clockwise" implies an alternating search or preference?
    // Let's implement an alternating search starting from the target direction vector.
    // Directions are 8 neighbors.
    // Neighbors in order: (0,-1) N, (1,-1) NE, (1,0) E, (1,1) SE, (0,1) S, (-1,1) SW, (-1,0) W, (-1,-1) NW.

    let candidates = get_push_search_order(initial_x, initial_y, dx, dy, center_x, center_y);
    for (nx, ny) in candidates {
        if is_in_bounds(nx, ny, size) {
            let idx = ny * size + nx;
            if is_steppable_for_push(state, unit, idx) {
                return Some(idx);
            }
        }
    }

    None
}

fn get_direction_toward_center(x: i32, y: i32, cx: i32, cy: i32) -> (i32, i32) {
    let dx = if x < cx {
        1
    } else if x > cx {
        -1
    } else {
        0
    };
    let mut dy = if y < cy {
        1
    } else if y > cy {
        -1
    } else {
        0
    };

    // If exact center, push South
    if dx == 0 && dy == 0 {
        dy = 1;
    }
    (dx, dy)
}

fn calculate_move_push_direction(_state: &GameState, unit: &UnitState) -> (i32, i32) {
    let initial_x = unit.coords.x;
    let initial_y = unit.coords.y;
    let prev = unit.prev_coords;

    let mut dx = if initial_x > prev.x {
        1
    } else if initial_x < prev.x {
        -1
    } else {
        0
    };
    let mut dy = if initial_y > prev.y {
        1
    } else if initial_y < prev.y {
        -1
    } else {
        0
    };

    if _state.settings._verbose {
        println!("Using prev_coords: {}", unit.prev_coords.idx);
        println!("Chosen using moved: dx = {}, dy = {}", dx, dy);
    }

    // Rule 2: Enemy units pushed in opposite direction
    if unit.owner != _state.settings.current_player_turn_id {
        dx = -dx;
        dy = -dy;
    }
    (dx, dy)
}

fn get_push_search_order(x: i32, y: i32, dx: i32, dy: i32, cx: i32, cy: i32) -> Vec<(i32, i32)> {
    // 8 directions. Finding index of (dx, dy) in standard loop.
    let dirs = [
        (0, -1),
        (1, -1),
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
    ];

    let start_idx = dirs
        .iter()
        .position(|&(d_x, d_y)| d_x == dx && d_y == dy)
        .unwrap_or(0);

    // Alternating expansion: 0, 1, -1, 2, -2 ...
    // prioritizing tiles that are closer to the map center as tie-breakers
    let mut search_dirs = Vec::new();
    search_dirs.push(dirs[start_idx]);

    for i in 1..=4 {
        let ccw_idx = (start_idx as i32 - i).rem_euclid(8) as usize;
        let cw_idx = (start_idx as i32 + i).rem_euclid(8) as usize;

        if ccw_idx == cw_idx {
            search_dirs.push(dirs[ccw_idx]);
        } else {
            let (ccw_dx, ccw_dy) = dirs[ccw_idx];
            let (cw_dx, cw_dy) = dirs[cw_idx];

            let ccw_x = x + ccw_dx;
            let ccw_y = y + ccw_dy;
            let cw_x = x + cw_dx;
            let cw_y = y + cw_dy;

            // 1. Distance Tie-break: Which one is Euclidean-closer to the map center?
            let dist_ccw = (ccw_x - cx).pow(2) + (ccw_y - cy).pow(2);
            let dist_cw = (cw_x - cx).pow(2) + (cw_y - cy).pow(2);

            if dist_cw < dist_ccw {
                search_dirs.push(dirs[cw_idx]);
                search_dirs.push(dirs[ccw_idx]);
            } else if dist_ccw < dist_cw {
                search_dirs.push(dirs[ccw_idx]);
                search_dirs.push(dirs[cw_idx]);
            } else {
                // 2. Visual Tie-break: Prioritize visual horizontal center (x-y closer to 0)
                let visual_ccw = (ccw_x - ccw_y).abs();
                let visual_cw = (cw_x - cw_y).abs();
                if visual_cw < visual_ccw {
                    search_dirs.push(dirs[cw_idx]);
                    search_dirs.push(dirs[ccw_idx]);
                } else {
                    search_dirs.push(dirs[ccw_idx]);
                    search_dirs.push(dirs[cw_idx]);
                }
            }
        }
    }

    search_dirs
        .into_iter()
        .map(|(d_x, d_y)| (x + d_x, y + d_y))
        .collect()
}

pub fn is_steppable_for_push(state: &GameState, unit: &UnitState, idx: i32) -> bool {
    // Hidden enemies block pushing
    if get_enemy_at(state, idx, unit.owner).is_some() {
        return false;
    }
    // Our own units also block pushing
    if let Some(tribe) = state.tribes.get(&unit.owner) {
        if tribe.units.iter().any(|u| u.coords.idx == idx) {
            return false;
        }
    }

    // Terrain validity
    let settings = crate::settings::units::get_unit_setting(unit.unit_type);
    let tile = match state.tiles.get(&idx) {
        Some(t) => t,
        None => return false,
    };
    let tech = &state.tribes.get(&unit.owner).unwrap().tech_vanilla;

    if settings.skills.contains(&SkillType::Fly) {
        return true;
    }

    match tile.terrain_type {
        TerrainType::Water | TerrainType::Ocean => {
            let has_bridge = get_structure_type_at(state, idx) == Some(StructureType::Bridge);
            let has_port = get_structure_type_at(state, idx) == Some(StructureType::Port);
            let is_our_port = has_port && tile.owner == unit.owner;
            if !settings.skills.contains(&SkillType::Float)
                && !tile.is_algae()
                && !has_bridge
                && !is_our_port
            {
                return false;
            }
        }
        TerrainType::Mountain => {
            let tribe_type = state.tribes.get(&unit.owner).unwrap().tribe_type;
            let climbing = crate::settings::technology::resolve_tech_for_tribe(
                TechnologyType::Climbing,
                tribe_type,
            );
            if !has_technology(&tech, climbing) {
                return false;
            }
        }
        _ => {
            if settings.skills.contains(&SkillType::Water) && !tile.is_flooded() {
                return false;
            }
        }
    }

    true
}

/// Convert tile index to coords
pub fn idx_to_coords(idx: i32, size: i32) -> (i32, i32) {
    (idx % size, idx / size)
}

/// Get the tech cost based on number of cities and tier
pub fn get_tech_cost(tribe: &TribeState, tech: TechnologyType) -> i32 {
    let tier = crate::settings::technology::get_technology_setting(tech)
        .tier
        .unwrap_or(1);
    let cities_count = tribe.cities.len() as i32;
    let has_philo = crate::settings::technology::has_technology(
        &tribe.tech_vanilla,
        TechnologyType::Philosophy,
    );
    crate::settings::technology::get_tech_cost(cities_count, tier, has_philo)
}

/// Calculate the detailed tribe score as per Polytopia rules
pub fn calculate_detailed_tribe_score(state: &GameState, player_id: PlayerId) -> i32 {
    let tribe = match state.tribes.get(&player_id) {
        Some(t) => t,
        None => return 0,
    };

    let mut score = 0;

    // 100 per level, 20 per territory
    for city in &tribe.cities {
        // City score: 100 + 50 per level above 1
        let city_score = if city.level >= 1 {
            100 + (city.level - 1) * 50
        } else {
            0
        };
        // Territory: 20 per tile
        score += city_score + (city._territory.len() as i32 * 20);

        // Structure scores (Temples & Monuments)
        for &tile_idx in &city._territory {
            if let Some(structure) = crate::functions::get_structure_at(state, tile_idx) {
                match structure.structure_type {
                    StructureType::Temple
                    | StructureType::WaterTemple
                    | StructureType::ForestTemple
                    | StructureType::MountainTemple
                    | StructureType::IceTemple => {
                        // Temples: 100 per level
                        score += structure.level * 100;
                    }
                    _ => {
                        // Monuments and others
                        let settings = crate::settings::structures::get_structure_setting(
                            structure.structure_type,
                        );
                        score += settings.reward_score;
                    }
                }
            }
        }

        // Park: 250 points
        if city.has_park() {
            score += 250;
        }

        // Population: 5 points per population
        score += city.population * 5;
    }

    // Except for Luxidoor, which already starts with a level 3 city (capital)
    // lvl 1: 2 pop
    // lvl 2: 3 pop
    if state.tribes.get(&player_id).unwrap().tribe_type == TribeType::Luxidoor {
        score -= 5 * 5;
    }

    // 5 per revealed tile (explored by our explorers)
    let explored_count = state
        .tiles
        .values()
        .filter(|t| t.explorers.contains(&player_id))
        .count() as i32;
    score += explored_count * 5;

    // 5 per star of unit cost
    for unit in &tribe.units {
        // Converted units do not change score (worth 0 for new owner)
        if unit.converted {
            continue;
        }

        // Score = 5 * (Stars spent on unit + Stars spent on its passenger/original carrier)
        let cost = crate::settings::units::get_unit_setting(unit.unit_type).cost
            + unit
                .passenger_type
                .map(|p| crate::settings::units::get_unit_setting(p).cost)
                .unwrap_or(0);

        score += cost * 5;
    }

    // 100 per tech tier
    for tech in &tribe.tech_vanilla {
        let tier = crate::settings::technology::get_technology_setting(tech.tech_type)
            .tier
            .unwrap_or(1);
        score += 100 * tier;
    }

    score
}

/// Sync all tribes' scores based on current state
pub fn sync_scores(state: &mut GameState) {
    let ids: Vec<PlayerId> = state.tribes.keys().cloned().collect();
    for id in ids {
        let score = calculate_detailed_tribe_score(state, id);
        if let Some(tribe) = state.tribes.get_mut(&id) {
            tribe.score = score;
        }
    }
}

/// Get the star exchange rate based on score
pub fn get_star_exchange(state: &GameState, player_id: PlayerId) -> i32 {
    let score = state.tribes.get(&player_id).map(|t| t.score).unwrap_or(0);
    // Modern formula (v >= 83)
    // C#: 1 + Math.Min((int)Math.Ceiling(score / 1000f), 5) * 2;
    let multiplier = if score <= 0 {
        0
    } else {
        ((score - 1) / 1000 + 1).min(5)
    };
    1 + 2 * multiplier
}

/// Calculate a combat preview between an attacker and defender unit.
/// Returns (attacker_damage_dealt, attacker_damage_taken, defender_dies, attacker_dies)
pub fn calculate_combat_preview(
    state: &GameState,
    attacker_idx: i32,
    defender_idx: i32,
) -> Option<CombatPreview> {
    let attacker = get_unit_at(state, attacker_idx)?;
    let defender = get_enemy_at(state, defender_idx, attacker.owner)?;

    let atk_atk = get_unit_attack(state, attacker);
    let atk_health = attacker.health;
    let atk_max = get_unit_max_health(attacker);

    let def_def = get_unit_defense(defender);
    let def_health = defender.health;
    let def_max = get_unit_max_health(defender);
    let defense_bonus = get_defense_bonus(state, defender);

    let result = crate::actions::units::calculate_combat(
        atk_atk,
        atk_health,
        atk_max,
        def_def,
        def_health,
        def_max,
        defense_bonus,
    );

    let damage_to_defender = result.attack_damage;
    let damage_to_attacker = result.defense_damage;
    let defender_dies = def_health - damage_to_defender <= 0.0;
    let attacker_dies = atk_health - damage_to_attacker <= 0.0;

    Some(CombatPreview {
        attacker_tile: attacker_idx,
        defender_tile: defender_idx,
        damage_to_defender,
        damage_to_attacker,
        defender_dies,
        attacker_dies,
    })
}

/// Combat preview result for UI display
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CombatPreview {
    pub attacker_tile: i32,
    pub defender_tile: i32,
    pub damage_to_defender: f32,
    pub damage_to_attacker: f32,
    pub defender_dies: bool,
    pub attacker_dies: bool,
}

use std::collections::HashMap;

/// Phase 4: Expansion Analysis Result
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExpansionAnalysis {
    pub tile_values: HashMap<i32, f32>,
    pub threats: Vec<Threat>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Threat {
    pub target_tile: i32,
    pub threat_level: f32,   // 0.0 to 1.0 (1.0 = lethal)
    pub attackers: Vec<i32>, // tile indices of attackers
}

pub fn analyze_expansion(state: &GameState, player_id: PlayerId) -> ExpansionAnalysis {
    let mut tile_values = HashMap::new();
    let mut threats = Vec::new();

    let tribe = match state.tribes.get(&player_id) {
        Some(t) => t,
        None => {
            return ExpansionAnalysis {
                tile_values,
                threats,
            };
        }
    };

    // 1. Expansion Analysis
    // Identify tiles adjacent to our territory or units that are valuable
    for tile in state.tiles.values() {
        if tile.owner == player_id {
            continue; // Already owned
        }

        // Anti-Cheat: Must have explored the tile to know its value
        if !tile.explorers.contains(&player_id) {
            continue;
        }

        // Simple check for adjacency to our territory/units
        // This is expensive if we do it for all tiles.
        // Optimization: Only check tiles near our cities/units.
        // For now, iterate all tiles but fast-fail if far? No, map is small enough (256-400 tiles).

        let mut value = 0.0;

        // Resource Value
        if let Some(res_opt) = state.resources.get(&tile.coords.idx) {
            if let Some(res) = res_opt {
                match res.resource_type {
                    ResourceType::Fruit | ResourceType::Game | ResourceType::Fish => value += 1.0,
                    ResourceType::Crop | ResourceType::AquaCrop => value += 1.5,
                    ResourceType::Metal => value += 2.0,
                    // Whale/Gold not in enum
                    _ => {}
                }
            }
        }

        // Village/Structure Value
        if let Some(struct_type) = get_structure_type_at(state, tile.coords.idx) {
            match struct_type {
                StructureType::Village => value += 5.0,
                StructureType::Ruin => value += 3.0,
                StructureType::Port => value += 2.0,
                _ => {}
            }
        }

        // Enemy City Value
        if tile.capital_of > 0 && tile.owner != player_id {
            value += 8.0;
        }

        // Territory Connection Value (Adjacency)
        let adj_indices = get_adjacent_indices(state, tile.coords.idx, 1);
        let mut is_adj = false;
        for adj_idx in &adj_indices {
            if let Some(adj_tile) = state.tiles.get(adj_idx) {
                if adj_tile.owner == player_id {
                    value += 1.0;
                    is_adj = true;
                }
            }
        }

        if value > 0.0 && is_adj {
            tile_values.insert(tile.coords.idx, value);
        }
    }

    // 2. Threat Analysis
    // Iterate over our units and check for enemy threats
    for unit in &tribe.units {
        let u_idx = unit.coords.idx;
        let mut attackers = Vec::new();
        let mut max_damage = 0.0;

        // Find all enemies
        for (other_id, other_tribe) in &state.tribes {
            if *other_id == player_id {
                continue;
            }

            for enemy in &other_tribe.units {
                // Anti-Cheat: Enemy unit is only a threat if visible
                if !state.tiles[&enemy.coords.idx]
                    .explorers
                    .contains(&player_id)
                {
                    continue;
                }

                // Approximate Threat: Distance <= Move + Range (Chebyshev for grid movement)
                let dist = unit.coords.chebyshev_distance_to(&enemy.coords);
                let enemy_settings = get_unit_setting(enemy.unit_type);
                let threat_range = enemy_settings.movement + enemy_settings.range;

                if dist <= threat_range {
                    // Potential attacker
                    // Calculate simplified damage
                    let atk = get_unit_attack(state, enemy);
                    let def = get_unit_defense(unit);

                    // Polytopia simplified non-retaliation calculation
                    let dmg = (atk / (atk + def)) * atk * 4.5;

                    max_damage += dmg;
                    attackers.push(enemy.coords.idx);
                }
            }
        }

        if !attackers.is_empty() {
            let threat_level = (max_damage / unit.health as f32).min(1.0);
            threats.push(Threat {
                target_tile: u_idx,
                threat_level,
                attackers,
            });
        }
    }

    ExpansionAnalysis {
        tile_values,
        threats,
    }
}
