use crate::functions::{
    get_adjacent_indices, get_chebyshev_distance, get_tech_cost, is_resource_visible_to_tribe,
    is_tile_explored,
};
use crate::states::{GameState, PlayerId, TribeState};
use crate::types::{ModeType, ResourceType, StructureType, TechnologyType, TerrainType, TribeType};

/// Evaluates the utility of a technology for a given player.
/// Returns a score representing the "Return on Investment" (ROI).
///
/// Pruning & Valuation Model:
/// 1. Physical Impossibility (Tier 1): If the required terrain/resource does not exist,
///    returns `f32::NEG_INFINITY` (hard masked in search).
/// 2. Dependency / Adjacency Deadends (Tier 2): If required foundation terrain does not exist,
///    or treasury cannot support activation capital, returns `f32::NEG_INFINITY` or `-15.0`.
/// 3. Climbing Overhaul: Values mountain vision discovery (unrevealed fog in radius 2)
///    and tactical combat defense rather than static border counting.
/// 4. Modern Trade: Evaluates Market potential adjacent to Sawmills, Windmills, and Forges.
pub fn evaluate_tech_utility(state: &GameState, player_id: PlayerId, tech: TechnologyType) -> f32 {
    let tribe_opt = state.tribes.get(&player_id);
    if tribe_opt.is_none() {
        return 0.0;
    }
    let tribe = tribe_opt.unwrap();
    let is_polaris = tribe.tribe_type == TribeType::Polaris;

    match tech {
        // --- Tier 1 Resource Techs ---
        TechnologyType::Organization => {
            let count = count_resources_fair(state, tribe, ResourceType::Fruit);
            if count == 0.0 {
                f32::NEG_INFINITY
            } else {
                count * 2.5 - 2.0
            }
        }
        TechnologyType::Hunting => {
            let count = count_resources_fair(state, tribe, ResourceType::Game);
            if count == 0.0 {
                f32::NEG_INFINITY
            } else {
                count * 2.5 - 2.0
            }
        }
        TechnologyType::Fishing => {
            if is_polaris {
                let ice_count = count_terrain(state, tribe, TerrainType::Ice);
                if ice_count == 0 {
                    f32::NEG_INFINITY
                } else {
                    (ice_count as f32 * 1.5) - 2.0
                }
            } else {
                let has_water = has_water_access(state, tribe);
                let fish_count = count_resources_fair(state, tribe, ResourceType::Fish);
                if !has_water && fish_count == 0.0 {
                    f32::NEG_INFINITY
                } else {
                    (fish_count * 2.5 + if has_water { 1.0 } else { 0.0 }) - 2.0
                }
            }
        }
        TechnologyType::Farming => {
            let count = count_resources_fair(state, tribe, ResourceType::Crop);
            if count == 0.0 {
                f32::NEG_INFINITY
            } else {
                count * 3.0 - 3.0
            }
        }
        TechnologyType::Mining => {
            let count = count_resources_fair(state, tribe, ResourceType::Metal);
            if count == 0.0 {
                f32::NEG_INFINITY
            } else {
                count * 4.0 - 3.0
            }
        }

        // --- Terrain & Vision Techs ---
        TechnologyType::Forestry => {
            let forests = count_terrain(state, tribe, TerrainType::Forest);
            if forests == 0 {
                f32::NEG_INFINITY
            } else {
                (forests as f32 * 1.5) - 3.0
            }
        }
        TechnologyType::Climbing => evaluate_climbing_utility(state, tribe),

        TechnologyType::Sailing => {
            if is_polaris {
                if !has_ice_access(state, tribe) {
                    f32::NEG_INFINITY
                } else {
                    (count_terrain(state, tribe, TerrainType::Ice) as f32 * 2.0) - 3.0
                }
            } else {
                if !has_water_access(state, tribe) {
                    f32::NEG_INFINITY
                } else {
                    (count_terrain(state, tribe, TerrainType::Water) as f32 * 2.0) - 3.0
                }
            }
        }
        TechnologyType::Navigation => {
            if is_polaris {
                if !has_ice_access(state, tribe) {
                    f32::NEG_INFINITY
                } else {
                    (count_terrain(state, tribe, TerrainType::Ice) as f32 * 2.5 + 2.0) - 5.0
                }
            } else {
                if !has_water_access(state, tribe) {
                    f32::NEG_INFINITY
                } else {
                    (count_terrain(state, tribe, TerrainType::Ocean) as f32 * 2.5 + 2.0) - 5.0
                }
            }
        }
        TechnologyType::Ramming => {
            if is_polaris {
                if !has_ice_access(state, tribe) {
                    f32::NEG_INFINITY
                } else {
                    (count_terrain(state, tribe, TerrainType::Ice) as f32 * 1.5 + 1.0) - 3.0
                }
            } else {
                if !has_water_access(state, tribe) {
                    f32::NEG_INFINITY
                } else {
                    (count_terrain(state, tribe, TerrainType::Water) as f32 * 1.5 + 2.0) - 3.0
                }
            }
        }
        TechnologyType::Aquatism => {
            if is_polaris {
                if !has_ice_access(state, tribe) {
                    f32::NEG_INFINITY
                } else {
                    (count_terrain(state, tribe, TerrainType::Ice) as f32 * 1.5) - 3.0
                }
            } else {
                if !has_water_access(state, tribe) {
                    f32::NEG_INFINITY
                } else {
                    (count_terrain(state, tribe, TerrainType::Water) as f32 * 1.5) - 3.0
                }
            }
        }

        // --- Tier 2 / 3 Dependency Techs (Hybrid Model) ---
        TechnologyType::Trade => evaluate_trade_utility(state, tribe),
        TechnologyType::Mathematics => evaluate_mathematics_utility(state, tribe),
        TechnologyType::Construction => evaluate_construction_utility(state, tribe),
        TechnologyType::Smithery => evaluate_smithery_utility(state, tribe),

        // --- Military Techs ---
        TechnologyType::Riding => {
            let field_count = count_terrain(state, tribe, TerrainType::Field);
            2.0 + field_count as f32 * 0.2 - 2.0
        }
        TechnologyType::Archery => {
            let forests = count_terrain(state, tribe, TerrainType::Forest);
            1.5 + (forests.min(3) as f32 * 0.5) - 3.0
        }
        TechnologyType::Strategy => {
            let is_threatened = tribe.cities.iter().any(|city| {
                state.tribes.iter().any(|(&other_id, other_tribe)| {
                    other_id != tribe.id
                        && other_tribe.units.iter().any(|u| {
                            get_chebyshev_distance(u.coords.idx, city.idx, state.settings.size) <= 3
                        })
                })
            });
            if is_threatened {
                4.0 - 3.0
            } else {
                1.5 - 3.0
            }
        }
        TechnologyType::Chivalry => {
            let tech_cost = get_tech_cost(tribe, TechnologyType::Chivalry);
            if tribe.stars < tech_cost + 8 {
                -15.0
            } else {
                5.0 - 6.0
            }
        }

        // --- Infrastructure & Contextual Techs ---
        TechnologyType::Roads => {
            if tribe.cities.len() <= 1 {
                if !has_nearby_village(state, tribe) {
                    f32::NEG_INFINITY
                } else {
                    -15.0
                }
            } else {
                (tribe.cities.len() as f32 * 1.2) - 3.0
            }
        }
        TechnologyType::Diplomacy => {
            if tribe.known_players.is_empty() {
                f32::NEG_INFINITY
            } else {
                (tribe.known_players.len() as f32 * 1.5) - 3.0
            }
        }
        TechnologyType::FreeSpirit => {
            if state.settings.turn < 15 && tribe.stars < 20 {
                -15.0
            } else {
                2.0 - 3.0
            }
        }
        TechnologyType::Meditation => {
            if state.settings.turn < 15 && state.settings.mode != ModeType::Perfection {
                -15.0
            } else {
                2.0 - 3.0
            }
        }
        TechnologyType::Spiritualism => {
            if state.settings.mode != ModeType::Perfection || state.settings.turn < 20 {
                -15.0
            } else {
                3.0 - 5.0
            }
        }
        TechnologyType::Philosophy => {
            let techs_left = 25 - tribe.tech_vanilla.iter().filter(|t| t.discovered).count();
            (techs_left as f32 * 0.3) - 3.0
        }

        // --- Polaris Replacements Fallback ---
        TechnologyType::Frostwork => {
            evaluate_tech_utility(state, player_id, TechnologyType::Fishing)
        }
        TechnologyType::Sledding => {
            evaluate_tech_utility(state, player_id, TechnologyType::Sailing)
        }
        TechnologyType::IceFishing => {
            evaluate_tech_utility(state, player_id, TechnologyType::Ramming)
        }
        TechnologyType::PolarWarfare => {
            evaluate_tech_utility(state, player_id, TechnologyType::Navigation)
        }
        TechnologyType::Polarism => {
            evaluate_tech_utility(state, player_id, TechnologyType::Aquatism)
        }

        _ => 1.0 - 2.0,
    }
}

pub fn has_water_access(state: &GameState, tribe: &TribeState) -> bool {
    let mut explored_water = false;
    for tile in state.tiles.values() {
        if (tile.terrain_type == TerrainType::Water || tile.terrain_type == TerrainType::Ocean)
            && tile.explorers.contains(&tribe.id)
        {
            explored_water = true;
            break;
        }
    }
    if !explored_water {
        return false;
    }

    for city in &tribe.cities {
        for &tile_idx in &city._territory {
            if let Some(tile) = state.tiles.get(&tile_idx) {
                if tile.terrain_type == TerrainType::Water
                    || tile.terrain_type == TerrainType::Ocean
                {
                    return true;
                }
            }
        }
    }

    for unit in &tribe.units {
        if let Some(tile) = state.tiles.get(&unit.coords.idx) {
            if tile.terrain_type == TerrainType::Water || tile.terrain_type == TerrainType::Ocean {
                return true;
            }
        }
    }

    for city in &tribe.cities {
        let adj = get_adjacent_indices(state, city.idx, 2);
        for idx in adj {
            if let Some(tile) = state.tiles.get(&idx) {
                if tile.terrain_type == TerrainType::Water
                    || tile.terrain_type == TerrainType::Ocean
                {
                    return true;
                }
            }
        }
    }

    for tile in state.tiles.values() {
        if tile.explorers.contains(&tribe.id) {
            if let Some(s) = crate::functions::get_structure_at(state, tile.coords.idx) {
                if s.structure_type == StructureType::Village {
                    let adj = get_adjacent_indices(state, tile.coords.idx, 1);
                    for n_idx in adj {
                        if let Some(n_tile) = state.tiles.get(&n_idx) {
                            if n_tile.terrain_type == TerrainType::Water
                                || n_tile.terrain_type == TerrainType::Ocean
                            {
                                return true;
                            }
                        }
                    }
                }
            }
        }
    }

    false
}

pub fn has_ice_access(state: &GameState, tribe: &TribeState) -> bool {
    let mut explored_ice = false;
    for tile in state.tiles.values() {
        if tile.terrain_type == TerrainType::Ice && tile.explorers.contains(&tribe.id) {
            explored_ice = true;
            break;
        }
    }
    if !explored_ice {
        return false;
    }
    for city in &tribe.cities {
        for &tile_idx in &city._territory {
            if let Some(tile) = state.tiles.get(&tile_idx) {
                if tile.terrain_type == TerrainType::Ice {
                    return true;
                }
            }
        }
    }
    for unit in &tribe.units {
        if let Some(tile) = state.tiles.get(&unit.coords.idx) {
            if tile.terrain_type == TerrainType::Ice {
                return true;
            }
        }
    }
    false
}

fn evaluate_climbing_utility(state: &GameState, tribe: &TribeState) -> f32 {
    let size = state.settings.size;
    let mut total_explored_mountains = 0;
    let mut accessible_mountains = Vec::new();

    for tile in state.tiles.values() {
        if tile.terrain_type == TerrainType::Mountain && tile.explorers.contains(&tribe.id) {
            total_explored_mountains += 1;
            let is_in_territory =
                tribe.cities.iter().any(|c| c._territory.contains(&tile.coords.idx));
            let is_near_unit = tribe.units.iter().any(|u| {
                get_chebyshev_distance(u.coords.idx, tile.coords.idx, size) <= 2
            });
            if is_in_territory || is_near_unit {
                accessible_mountains.push(tile.coords.idx);
            }
        }
    }

    if total_explored_mountains == 0 {
        return f32::NEG_INFINITY;
    }

    let mut max_fog_revealed = 0;
    for &m_idx in &accessible_mountains {
        let mut fog_count = 0;
        let mx = m_idx % size;
        let my = m_idx / size;
        for dy in -2..=2 {
            for dx in -2..=2 {
                let nx = mx + dx;
                let ny = my + dy;
                if nx >= 0 && nx < size && ny >= 0 && ny < size {
                    let n_idx = ny * size + nx;
                    if !is_tile_explored(state, n_idx, tribe.id) {
                        fog_count += 1;
                    }
                }
            }
        }
        if fog_count > max_fog_revealed {
            max_fog_revealed = fog_count;
        }
    }
    let vision_u = (max_fog_revealed as f32 * 0.35).min(5.0);

    let mut defense_u = 0.0;
    for city in &tribe.cities {
        let is_threatened = state.tribes.iter().any(|(&other_id, other_tribe)| {
            other_id != tribe.id
                && other_tribe.units.iter().any(|u| {
                    get_chebyshev_distance(u.coords.idx, city.idx, size) <= 3
                })
        });
        if is_threatened {
            let has_adj_mountain = accessible_mountains.iter().any(|&m_idx| {
                get_chebyshev_distance(m_idx, city.idx, size) <= 1
            });
            if has_adj_mountain {
                defense_u = 2.0;
                break;
            }
        }
    }

    let mountains_in_territory = count_terrain(state, tribe, TerrainType::Mountain) as f32;
    let metal_pot = mountains_in_territory * 0.8;

    let total_utility = vision_u + defense_u + metal_pot;

    if vision_u == 0.0 && defense_u == 0.0 && mountains_in_territory == 0.0 {
        return -15.0;
    }

    total_utility - 2.0
}

fn evaluate_trade_utility(state: &GameState, tribe: &TribeState) -> f32 {
    let has_forest = count_terrain(state, tribe, TerrainType::Forest) > 0;
    let has_field = count_terrain(state, tribe, TerrainType::Field) > 0;
    let has_mountain = count_terrain(state, tribe, TerrainType::Mountain) > 0;

    if !has_forest && !has_field && !has_mountain {
        return f32::NEG_INFINITY;
    }

    let mut existing_hubs = 0;
    for city in &tribe.cities {
        for &tile_idx in &city._territory {
            match crate::functions::get_structure_type_at(state, tile_idx) {
                Some(StructureType::Sawmill)
                | Some(StructureType::Windmill)
                | Some(StructureType::Forge) => {
                    existing_hubs += 1;
                }
                _ => {}
            }
        }
    }

    if existing_hubs > 0 {
        let mut market_utility = 0.0;
        for city in &tribe.cities {
            for &tile_idx in &city._territory {
                if crate::functions::get_structure_type_at(state, tile_idx).is_none() {
                    let neighbors = get_neighbors(tile_idx, state.settings.size);
                    let adj_hubs = neighbors
                        .iter()
                        .filter(|&&n| match crate::functions::get_structure_type_at(state, n) {
                            Some(StructureType::Sawmill)
                            | Some(StructureType::Windmill)
                            | Some(StructureType::Forge) => true,
                            _ => false,
                        })
                        .count();
                    market_utility += adj_hubs as f32 * 2.5;
                }
            }
        }
        market_utility.clamp(2.0, 10.0) - 5.0
    } else {
        let tech_cost = get_tech_cost(tribe, TechnologyType::Trade);
        if tribe.stars < tech_cost + 8 {
            f32::NEG_INFINITY
        } else {
            -15.0
        }
    }
}

fn evaluate_mathematics_utility(state: &GameState, tribe: &TribeState) -> f32 {
    let forests = count_terrain(state, tribe, TerrainType::Forest);
    let huts = count_structures(state, tribe, StructureType::LumberHut);

    if huts > 0 {
        let mut sawmill_u = 0.0;
        for city in &tribe.cities {
            for &tile_idx in &city._territory {
                if crate::functions::get_structure_type_at(state, tile_idx).is_none() {
                    let neighbors = get_neighbors(tile_idx, state.settings.size);
                    let adj_huts = neighbors
                        .iter()
                        .filter(|&&n| {
                            crate::functions::get_structure_type_at(state, n)
                                == Some(StructureType::LumberHut)
                        })
                        .count();
                    sawmill_u += adj_huts as f32 * 3.0;
                }
            }
        }
        sawmill_u.clamp(3.0, 10.0) - 3.0
    } else {
        let tech_cost = get_tech_cost(tribe, TechnologyType::Mathematics);
        if forests == 0 {
            if tribe.stars < tech_cost + 8 {
                f32::NEG_INFINITY
            } else {
                1.5 - 3.0
            }
        } else if tribe.stars < tech_cost + 8 {
            f32::NEG_INFINITY
        } else {
            -15.0
        }
    }
}

fn evaluate_construction_utility(state: &GameState, tribe: &TribeState) -> f32 {
    let farms = count_structures(state, tribe, StructureType::Farm);

    if farms > 0 {
        let mut windmill_u = 0.0;
        for city in &tribe.cities {
            for &tile_idx in &city._territory {
                if crate::functions::get_structure_type_at(state, tile_idx).is_none() {
                    let neighbors = get_neighbors(tile_idx, state.settings.size);
                    let adj_farms = neighbors
                        .iter()
                        .filter(|&&n| {
                            crate::functions::get_structure_type_at(state, n)
                                == Some(StructureType::Farm)
                        })
                        .count();
                    windmill_u += adj_farms as f32 * 2.0;
                }
            }
        }
        windmill_u.clamp(2.0, 8.0) - 3.0
    } else {
        let forests = count_terrain(state, tribe, TerrainType::Forest);
        let crops = count_resources_fair(state, tribe, ResourceType::Crop);
        if forests == 0 && crops == 0.0 {
            f32::NEG_INFINITY
        } else {
            let tech_cost = get_tech_cost(tribe, TechnologyType::Construction);
            if tribe.stars < tech_cost + 10 {
                f32::NEG_INFINITY
            } else {
                -15.0
            }
        }
    }
}

fn evaluate_smithery_utility(state: &GameState, tribe: &TribeState) -> f32 {
    let mines = count_structures(state, tribe, StructureType::Mine);

    if mines > 0 {
        let mut forge_u = 0.0;
        for city in &tribe.cities {
            for &tile_idx in &city._territory {
                if crate::functions::get_structure_type_at(state, tile_idx).is_none() {
                    let neighbors = get_neighbors(tile_idx, state.settings.size);
                    let adj_mines = neighbors
                        .iter()
                        .filter(|&&n| {
                            crate::functions::get_structure_type_at(state, n)
                                == Some(StructureType::Mine)
                        })
                        .count();
                    forge_u += adj_mines as f32 * 3.5;
                }
            }
        }
        forge_u.clamp(3.0, 12.0) + 2.0 - 6.0
    } else {
        let mountains = count_terrain(state, tribe, TerrainType::Mountain);
        let tech_cost = get_tech_cost(tribe, TechnologyType::Smithery);
        if mountains == 0 {
            if tribe.stars < tech_cost + 5 {
                f32::NEG_INFINITY
            } else {
                3.0 - 6.0
            }
        } else if tribe.stars < tech_cost + 5 {
            f32::NEG_INFINITY
        } else {
            -15.0
        }
    }
}

fn count_structures(state: &GameState, tribe: &TribeState, s_type: StructureType) -> usize {
    let mut count = 0;
    for city in &tribe.cities {
        for &tile_idx in &city._territory {
            if crate::functions::get_structure_type_at(state, tile_idx) == Some(s_type) {
                count += 1;
            }
        }
    }
    count
}

fn has_nearby_village(state: &GameState, tribe: &TribeState) -> bool {
    let size = state.settings.size;
    for tile in state.tiles.values() {
        if tile.explorers.contains(&tribe.id) {
            if let Some(s) = crate::functions::get_structure_at(state, tile.coords.idx) {
                if s.structure_type == StructureType::Village {
                    let near_capital = tribe.cities.iter().any(|c| {
                        get_chebyshev_distance(c.idx, tile.coords.idx, size) <= 4
                    });
                    if near_capital {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Fairly counts resources by using terrain-based estimates if the resource is hidden.
/// This prevents MCTS from "probing" the fog by simulating tech purchases.
fn count_resources_fair(
    state: &GameState,
    tribe: &crate::states::TribeState,
    res_type: ResourceType,
) -> f32 {
    let has_tech = is_resource_visible_to_tribe(state, res_type, tribe.id, None);
    let mut count = 0.0;

    for city in &tribe.cities {
        for &tile_idx in &city._territory {
            if has_tech {
                if let Some(Some(res)) = state.resources.get(&tile_idx) {
                    if res.resource_type == res_type {
                        count += 1.0;
                    }
                }
            } else if let Some(tile) = state.tiles.get(&tile_idx) {
                let is_proxy = match res_type {
                    ResourceType::Crop => tile.terrain_type == TerrainType::Field,
                    ResourceType::Metal => tile.terrain_type == TerrainType::Mountain,
                    ResourceType::Fish => tile.terrain_type == TerrainType::Water,
                    ResourceType::Game => tile.terrain_type == TerrainType::Forest,
                    ResourceType::Fruit => tile.terrain_type == TerrainType::Field,
                    _ => false,
                };
                if is_proxy {
                    count += 0.5;
                }
            }
        }
    }
    count
}

fn count_terrain(
    state: &GameState,
    tribe: &crate::states::TribeState,
    terrain_type: TerrainType,
) -> usize {
    let mut count = 0;
    for city in &tribe.cities {
        for &tile_idx in &city._territory {
            if let Some(tile) = state.tiles.get(&tile_idx) {
                if tile.terrain_type == terrain_type {
                    count += 1;
                }
            }
        }
    }
    count
}

fn get_neighbors(idx: i32, size: i32) -> Vec<i32> {
    let x = idx % size;
    let y = idx / size;
    let mut neighbors = Vec::new();
    for dy in -1..=1 {
        for dx in -1..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let nx = x + dx;
            let ny = y + dy;
            if nx >= 0 && nx < size && ny >= 0 && ny < size {
                neighbors.push(ny * size + nx);
            }
        }
    }
    neighbors
}
