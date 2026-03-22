use crate::functions::is_resource_visible_to_tribe;
use crate::states::{GameState, PlayerId};
use crate::types::{ResourceType, TechnologyType, TerrainType};

/// Evaluates the utility of a technology for a given player.
/// Returns a score representing the "Return on Investment" (ROI).
///
/// Heuristic Logic:
/// 1. Unlockable Resources: Count owned/visible resources that the tech unlocks.
/// 2. Terrain Utility: Climbing (Mountains), Sailing (Water).
/// 3. Cost-Benefit: Value = (UnlockableCount * ResourceScore) + StrategicValue.
///
/// If Value < Cost (in logical units), the tech is considered wasteful.
pub fn evaluate_tech_utility(state: &GameState, player_id: PlayerId, tech: TechnologyType) -> f32 {
    let tribe_opt = state.tribes.get(&player_id);
    if tribe_opt.is_none() {
        return 0.0;
    }
    let tribe = tribe_opt.unwrap();

    let (utility, cost_offset) = match tech {
        // --- Resource Techs ---
        TechnologyType::Organization => (
            count_resources_fair(state, tribe, ResourceType::Fruit) * 2.5,
            2.0,
        ),
        TechnologyType::Hunting => (
            count_resources_fair(state, tribe, ResourceType::Game) * 2.5,
            2.0,
        ),
        TechnologyType::Fishing => (
            count_resources_fair(state, tribe, ResourceType::Fish) * 2.5,
            2.0,
        ),
        TechnologyType::Farming => (
            count_resources_fair(state, tribe, ResourceType::Crop) * 3.0,
            3.0,
        ),
        TechnologyType::Mining => (
            count_resources_fair(state, tribe, ResourceType::Metal) * 4.0,
            3.0,
        ),

        // --- Terrain Techs ---
        TechnologyType::Forestry => (
            count_terrain(state, tribe, TerrainType::Forest) as f32 * 1.5,
            3.0,
        ),
        TechnologyType::Climbing => (
            count_terrain(state, tribe, TerrainType::Mountain) as f32 * 1.0,
            2.0,
        ),
        TechnologyType::Sailing => (
            count_terrain(state, tribe, TerrainType::Water) as f32 * 2.0,
            3.0,
        ),
        TechnologyType::Navigation => (
            count_terrain(state, tribe, TerrainType::Ocean) as f32 * 2.5,
            5.0,
        ),

        // --- Military Techs ---
        TechnologyType::Riding => (
            2.0 + count_terrain(state, tribe, TerrainType::Field) as f32 * 0.2, // Base + Rider preference
            2.0,
        ),
        TechnologyType::Archery => (1.5, 3.0),
        TechnologyType::Strategy => (2.0, 3.0),
        TechnologyType::Chivalry => (5.0, 6.0),
        TechnologyType::Smithery => (6.0, 6.0),

        // --- Infrastructure ---
        TechnologyType::Roads => (tribe.cities.len() as f32 * 0.5, 3.0),
        TechnologyType::Trade => {
            let mut u = 0.0;
            // Reward based on Customs House potential (water adjacency to cities)
            for city in &tribe.cities {
                let neighbors = get_neighbors(city.idx, state.settings.size);
                for n_idx in neighbors {
                    if let Some(tile) = state.tiles.get(&n_idx) {
                        if tile.terrain_type == TerrainType::Water
                            || tile.terrain_type == TerrainType::Ocean
                        {
                            u += 1.0;
                        }
                    }
                }
            }
            (u, 5.0)
        }

        // --- Others ---
        TechnologyType::Philosophy => {
            let techs_left = 25 - tribe.tech_vanilla.iter().filter(|t| t.discovered).count();
            (techs_left as f32 * 0.3, 3.0)
        }
        TechnologyType::Diplomacy => (tribe.known_players.len() as f32 * 1.5, 3.0),

        _ => (1.0, 2.0),
    };

    utility - cost_offset
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
                // Truth: Tech is REALLY discovered, use actual resources
                if let Some(Some(res)) = state.resources.get(&tile_idx) {
                    if res.resource_type == res_type {
                        count += 1.0;
                    }
                }
            } else {
                // Estimation: Tech is NOT discovered (or in simulation), use terrain proxy
                if let Some(tile) = state.tiles.get(&tile_idx) {
                    let is_proxy = match res_type {
                        ResourceType::Crop => tile.terrain_type == TerrainType::Field,
                        ResourceType::Metal => tile.terrain_type == TerrainType::Mountain,
                        ResourceType::Fish => tile.terrain_type == TerrainType::Water,
                        ResourceType::Game => tile.terrain_type == TerrainType::Forest,
                        ResourceType::Fruit => tile.terrain_type == TerrainType::Field,
                        _ => false,
                    };
                    if is_proxy {
                        // Discounted value (0.5) because it's just a potential/estimated resource
                        count += 0.5;
                    }
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
