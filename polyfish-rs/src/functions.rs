//! Core game functions and utilities
//!
//! This module contains the helper functions for querying and manipulating game state.

/// Check if a tile is in the player's own territory
pub fn is_in_own_territory(state: &GameState, idx: i32, owner: i32) -> bool {
    state
        .tiles
        .get(&idx)
        .map(|t| t.owner == owner)
        .unwrap_or(false)
}

use crate::coords::Coords;
use crate::settings::get_unit_setting;
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

    // Check turn limit for Perfection mode
    if state.settings.mode == ModeType::Perfection && state.settings.turn > state.settings.max_turns
    {
        return true;
    }

    false
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

/// Get resource at a tile
pub fn get_resource_at(state: &GameState, idx: i32) -> Option<ResourceType> {
    state
        .resources
        .get(&idx)
        .and_then(|r| r.as_ref())
        .map(|r| r.resource_type)
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
                // Skip ships - return the passenger conceptually
                let setting = get_unit_setting(unit.unit_type);
                if setting.skills.contains(&SkillType::Carry) {
                    if let Some(passenger) = unit.passenger_type {
                        if passenger != UnitType::None {
                            continue; // The "true" unit is the passenger
                        }
                    }
                }
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

/// Get city at a tile index
pub fn get_city_at<'a>(state: &'a GameState, idx: i32) -> Option<&'a CityState> {
    for tribe in state.tribes.values() {
        for city in &tribe.cities {
            if city.tile_index == idx {
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
        is_enemy(state, not_owner, u.owner) && !u.effects.contains(&EffectType::Invisible)
    })
}

/// Check if a unit has an effect
pub fn has_effect(unit: &UnitState, effect: EffectType) -> bool {
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
/// Logic: Any Aquarion unit that has the Float skill.
pub fn is_amphibious(state: &GameState, unit: &UnitState) -> bool {
    let tribe_type = state
        .tribes
        .get(&unit.owner)
        .map(|t| t.tribe_type)
        .unwrap_or(TribeType::None);
    tribe_type == TribeType::Aquarion && has_skill(unit, SkillType::Float)
}

/// Get the maximum health of a unit (accounting for veteran status and pass-through)
pub fn get_max_health(unit: &UnitState) -> i32 {
    let mut hp = get_real_unit_setting(unit).health;
    if unit.veteran {
        hp += 5;
    }
    hp * crate::states::HEALTH_SCALE
}

/// Get the real unit setting (ignoring naval types if carrying a passenger)
pub fn get_real_unit_setting(unit: &UnitState) -> crate::settings::units::UnitSetting {
    let u_type = unit.passenger_type.unwrap_or(unit.unit_type);
    crate::settings::units::get_unit_setting(u_type)
}

/// Get unit attack strength (accounting for Boost)
pub fn get_unit_attack(unit: &UnitState) -> f32 {
    let mut atk = get_real_unit_setting(unit).attack;
    if has_effect(unit, EffectType::Boost) {
        atk += 0.5;
    }
    atk
}

/// Get unit defense strength (accounting for Poison)
pub fn get_unit_defense(unit: &UnitState) -> f32 {
    let mut def = get_real_unit_setting(unit).defense;
    if has_effect(unit, EffectType::Poison) {
        def *= 0.7; // 30% damage reduction
    }
    def
}

/// Get unit movement (accounting for Boost)
pub fn get_unit_movement(unit: &UnitState) -> i32 {
    let mut movement = crate::settings::units::get_unit_setting(unit.unit_type).movement;
    if has_effect(unit, EffectType::Boost) {
        movement += 1;
    }
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
    // Poisoned units cannot receive defense bonus
    if has_effect(unit, EffectType::Poison) {
        return 1.0;
    }

    let tribe = match state.tribes.get(&unit.owner) {
        Some(t) => t,
        None => return 1.0,
    };

    let tile = match state.tiles.get(&unit.coords.idx) {
        Some(t) => t,
        None => return 1.0,
    };

    match tile.terrain_type {
        TerrainType::Water | TerrainType::Ocean => {
            if crate::settings::technology::has_technology(
                &tribe.tech_vanilla,
                TechnologyType::Aquatism,
            ) {
                return 1.5;
            }
        }
        TerrainType::Forest => {
            if crate::settings::technology::has_technology(
                &tribe.tech_vanilla,
                TechnologyType::Archery,
            ) {
                return 1.5;
            }
        }
        TerrainType::Mountain => {
            if crate::settings::technology::has_technology(
                &tribe.tech_vanilla,
                TechnologyType::Climbing,
            ) {
                return 1.5;
            }
        }
        _ => {
            // City defense
            if let Some(city) = tribe
                .cities
                .iter()
                .find(|c| c.tile_index == unit.coords.idx)
            {
                if has_skill(unit, SkillType::Fortify) {
                    if city._walls {
                        return 4.0;
                    } else {
                        return 1.5;
                    }
                }
            }
        }
    }

    1.0
}

/// Get city production (stars per turn)
pub fn get_city_production(state: &GameState, city: &CityState) -> i32 {
    // If city is on riot or the tile is occupied by an enemy then production is nullified
    if city._riot || crate::functions::get_enemy_at(state, city.tile_index, city.owner).is_some() {
        return 0;
    }

    let mut prod = city.production;
    // Capitals get a +1 star bonus
    if let Some(tile) = state.tiles.get(&city.tile_index) {
        if tile.capital_of == city.owner && tile.capital_of != 0 {
            prod += 1;
        }
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
    state.tiles.get(&idx).map_or(false, |t| t.owner != pov_id)
}

/// Check if a tile is frozen (has Ice terrain)
pub fn is_tile_frozen(state: &GameState, idx: i32) -> bool {
    state.tiles.get(&idx).map_or(false, |t| t.frozen)
}

/// Check if coordinate is in bounds
pub fn is_in_bounds(x: i32, y: i32, size: i32) -> bool {
    x >= 0 && x < size && y >= 0 && y < size
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
                .get(&c.tile_index)
                .map(|t| t.capital_of == player_id)
                .unwrap_or(false)
        })
    })
}

/// Calculate a valid position to push a unit to
/// Calculate a valid position to push a unit to
pub fn calculate_pushable_position(state: &GameState, unit: &UnitState) -> Option<i32> {
    let size = state.settings.size;
    let initial_x = unit.coords.x;
    let initial_y = unit.coords.y;

    let center_x = size / 2;
    let center_y = size / 2;

    let (dx, dy) = if unit.moved || unit.prev_coords.idx != -1 {
        // Determine vector of last move
        let prev = if unit.prev_coords.idx != -1 {
            unit.prev_coords
        } else {
            // Fallback if moved but no prev coords recorded (shouldn't happen usually)
            unit.coords
        };

        if prev.idx == unit.coords.idx {
            // Moved but staying in place? Treat as not moved.
            get_direction_toward_center(initial_x, initial_y, center_x, center_y)
        } else {
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

            // Enemy units pushed in opposite direction
            if unit.owner != state.settings.current_player_turn_id {
                dx = -dx;
                dy = -dy;
            }
            (dx, dy)
        }
    } else {
        // Not previously moved: push towards center
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

    let candidates = get_push_search_order(initial_x, initial_y, dx, dy);
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

fn get_push_search_order(x: i32, y: i32, dx: i32, dy: i32) -> Vec<(i32, i32)> {
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

    // Alternating expansion: 0, -1, +1, -2, +2 ...
    // i.e. start, start-1 (CCW), start+1 (CW), start-2 ...
    // Indices modulo 8.

    let mut search_dirs = Vec::new();
    // 0
    search_dirs.push(dirs[start_idx]);

    for i in 1..=4 {
        // CCW (assuming array is CW ordered N->NE->E...)
        // Array is N, NE, E, SE, S, SW, W, NW. This is CW order.
        // So CCW is -1.
        let ccw_idx = (start_idx as i32 - i).rem_euclid(8) as usize;
        let cw_idx = (start_idx as i32 + i).rem_euclid(8) as usize;

        search_dirs.push(dirs[ccw_idx]);
        if cw_idx != ccw_idx {
            // Don't add opposite twice (at i=4)
            search_dirs.push(dirs[cw_idx]);
        }
    }

    search_dirs
        .into_iter()
        .map(|(d_x, d_y)| (x + d_x, y + d_y))
        .collect()
}

fn is_steppable_for_push(state: &GameState, unit: &UnitState, idx: i32) -> bool {
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

    match tile.terrain_type {
        TerrainType::Water | TerrainType::Ocean => {
            if !settings.skills.contains(&SkillType::Float)
                && !settings.skills.contains(&SkillType::Fly)
            {
                return false;
            }
        }
        _ => {}
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

        // Park: 250 points
        if city.rewards.contains(&RewardType::Park) {
            score += 250;
        }
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
        let cost = crate::settings::units::get_unit_setting(unit.unit_type).cost;
        score += cost * 5;
    }

    // 100 per tech tier
    for tech in &tribe.tech_vanilla {
        if tech.discovered {
            let tier = crate::settings::technology::get_technology_setting(tech.tech_type)
                .tier
                .unwrap_or(1);
            score += 100 * tier;
        }
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
    if score < 1000 {
        3
    } else if score < 2000 {
        6
    } else if score < 3000 {
        9
    } else {
        12
    }
}
