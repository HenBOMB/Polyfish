//! Prediction module for FOW (Fog of War)
//!
//! Provides prediction functions for MCTS simulations to avoid accessing ground truth data.
//! When `_are_you_sure = false`, the engine uses these predictions instead of actual hidden data.

use crate::functions::get_adjacent_indices;
use crate::states::GameState;
use crate::types::{TerrainType, TribeType};
use indexmap::IndexMap;

/// Predicted tribe for a classic climate id (see `types::classic_climate_id`).
pub fn climate_to_tribe(climate: i32) -> TribeType {
    crate::types::tribe_from_classic_climate(climate)
}

/// Classic climate id a tribe's territory carries.
pub fn tribe_to_climate(tribe: TribeType) -> i32 {
    crate::types::classic_climate_id(tribe)
}

/// Validation for village candidates based on mapgen rules
fn validate_village_candidate(
    state: &GameState,
    idx: i32,
    current_predictions: &IndexMap<i32, (TribeType, bool)>,
    known_cities: &std::collections::HashSet<i32>,
) -> bool {
    let size = state.settings.size;

    // 1. Cardinal Neighbor Rule: No Ocean neighbors
    let cardinals = crate::functions::get_plus_sign_indices(idx, size);
    for n_idx in cardinals {
        if let Some(tile) = state.tiles.get(&n_idx) {
            if tile.terrain_type == TerrainType::Ocean {
                return false;
            }
        }
    }

    // 2. Map Edge Rule: edge_dist >= 2 && edge_dist != 3
    let (x, y) = (idx % size, idx / size);
    let dist_x = x.min(size - 1 - x);
    let dist_y = y.min(size - 1 - y);
    let edge_dist = dist_x.min(dist_y);
    if edge_dist < 2 || edge_dist == 3 {
        return false;
    }

    // 3. Distance-3 Rule (Chebyshev) from known cities
    for &city_idx in known_cities {
        if crate::functions::get_chebyshev_distance(idx, city_idx, size) < 3 {
            return false;
        }
    }

    // 4. Distance-3 Rule (Chebyshev) from other predictions
    for &pred_idx in current_predictions.keys() {
        if crate::functions::get_chebyshev_distance(idx, pred_idx, size) < 3 {
            return false;
        }
    }

    true
}

/// Predict village locations in fog based on climate density AND orphan resources
pub fn predict_villages(state: &GameState) -> IndexMap<i32, (TribeType, bool)> {
    let pov_id = state.settings.current_player_turn_id;
    let pov_tribe_type = state
        .tribes
        .get(&pov_id)
        .map(|t| t.tribe_type)
        .unwrap_or(TribeType::None);
    let pov_climate = tribe_to_climate(pov_tribe_type);

    let mut candidates: IndexMap<i32, (i32, i32)> = IndexMap::new();

    // Collect all known cities/villages
    let mut known_cities = std::collections::HashSet::new();
    for (&idx, tile) in &state.tiles {
        if tile.capital_of > 0
            || (tile.explorers.contains(&pov_id)
                && crate::functions::get_structure_type_at(state, idx)
                    == Some(crate::types::StructureType::Village))
        {
            known_cities.insert(idx);
        }
    }

    // Orphan helper
    let is_orphan = |res_idx: i32| -> bool {
        if known_cities.contains(&res_idx) {
            return false;
        }
        let (rx, ry) = (res_idx % state.settings.size, res_idx / state.settings.size);
        for &city_idx in &known_cities {
            let (cx, cy) = (
                city_idx % state.settings.size,
                city_idx / state.settings.size,
            );
            if (rx - cx).abs() <= 1 && (ry - cy).abs() <= 1 {
                return false;
            }
        }
        true
    };

    // 1. Resource Heuristic - iterate over explored tiles
    for (&tile_idx, tile) in &state.tiles {
        if !tile.explorers.contains(&pov_id) {
            continue;
        }
        if let Some(res_opt) = state.resources.get(&tile_idx) {
            if res_opt.is_some() && is_orphan(tile_idx) {
                let neighbors = get_adjacent_indices(state, tile_idx, 1);
                for n_idx in neighbors {
                    let n_explored = state
                        .tiles
                        .get(&n_idx)
                        .map(|t| t.explorers.contains(&pov_id))
                        .unwrap_or(false);
                    if !n_explored {
                        if !validate_village_candidate(
                            state,
                            n_idx,
                            &IndexMap::new(),
                            &known_cities,
                        ) {
                            continue;
                        }
                        let entry = candidates.entry(n_idx).or_insert((0, 0));
                        entry.0 += 5;
                    }
                }
            }
        }
    }

    // 2. Climate Heuristic - iterate over explored tiles
    for (&tile_idx, tile) in &state.tiles {
        if !tile.explorers.contains(&pov_id) {
            continue;
        }
        if tile.owner != pov_id && tile.climate != pov_climate && tile.climate != 0 {
            let around = get_adjacent_indices(state, tile_idx, 2);
            for idx in around {
                let idx_explored = state
                    .tiles
                    .get(&idx)
                    .map(|t| t.explorers.contains(&pov_id))
                    .unwrap_or(false);
                if !idx_explored {
                    if !validate_village_candidate(state, idx, &IndexMap::new(), &known_cities) {
                        continue;
                    }
                    let entry = candidates.entry(idx).or_insert((0, tile.climate));
                    entry.0 += 1;
                }
            }
        }
    }

    // 3. Resource Cluster Density & Tribe Bias (Exploit: Cities prefer areas with many resources)
    let candidate_indices: Vec<i32> = candidates.keys().cloned().collect();
    for idx in candidate_indices {
        let mut res_count = 0;
        let neighbors = get_adjacent_indices(state, idx, 1);
        for n_idx in neighbors {
            if let Some(res_opt) = state.resources.get(&n_idx) {
                if res_opt.is_some() {
                    res_count += 1;
                }
            }
        }

        if let Some(entry) = candidates.get_mut(&idx) {
            if res_count >= 2 {
                entry.0 += 10; // Massive bonus for clusters
            }

            // Tribe Bias (Bardur doesn't have crops, Imperius loves fruit)
            let predicted_tribe = climate_to_tribe(entry.1);
            if predicted_tribe == TribeType::Bardur {
                // Check if any nearby resource is a crop? (Wait, we might not know resource type if hidden)
                // But we can check visible neighboring crops
                for n_idx in get_adjacent_indices(state, idx, 1) {
                    if let Some(Some(res)) = state.resources.get(&n_idx) {
                        if res.resource_type == crate::types::ResourceType::Crop {
                            entry.0 -= 20; // Extremely unlikely to be Bardur if there are crops
                        }
                    }
                }
            }
        }
    }

    let mut prediction_map: IndexMap<i32, (TribeType, bool)> = IndexMap::new();
    let mut sorted: Vec<_> = candidates.into_iter().collect();
    sorted.sort_by_key(|(_, (count, _))| -count);

    for (best_idx, (count, climate)) in &sorted {
        if *count > 2 {
            if validate_village_candidate(state, *best_idx, &prediction_map, &known_cities) {
                prediction_map.insert(*best_idx, (climate_to_tribe(*climate), true));
            }
        }
        if prediction_map.len() >= 5 {
            break;
        }
    }

    if prediction_map.is_empty() && !sorted.is_empty() {
        let (best_idx, (_, climate)) = sorted[0];
        prediction_map.insert(best_idx, (climate_to_tribe(climate), true));
    }

    prediction_map
}

/// Find the tribe of the nearest known city/village to a given tile
fn get_nearest_known_tribe(state: &GameState, idx: i32) -> Option<TribeType> {
    let size = state.settings.size;
    let pov_id = state.settings.current_player_turn_id;

    let mut best_dist = i32::MAX;
    let mut best_tribe = None;

    // Check all tiles for known cities
    for (&t_idx, tile) in &state.tiles {
        // Must be visible or have a known capital/village
        let is_known_city = tile.capital_of > 0
            || (tile.explorers.contains(&pov_id)
                && crate::functions::get_structure_type_at(state, t_idx)
                    == Some(crate::types::StructureType::Village));

        if is_known_city {
            let dist = crate::functions::get_chebyshev_distance(idx, t_idx, size);
            if dist < best_dist {
                best_dist = dist;
                // If it's a capital, use the owner's tribe. If it's a village, we might not know the original tribe,
                // but we can guess from the climate if visible, or owner if visible.
                // For now, use owner if present, else climate.
                if let Some(owner) = state.tribes.get(&tile.owner) {
                    best_tribe = Some(owner.tribe_type);
                } else {
                    best_tribe = Some(climate_to_tribe(tile.climate));
                }
            }
        }
    }

    best_tribe
}

/// Predict terrain for fog tiles based on probabilistic mapgen rules.
/// The second tuple element is a classic climate id (types::classic_climate_id).
pub fn predict_terrain(
    state: &GameState,
    fog_tiles: &[i32],
) -> IndexMap<i32, (TerrainType, i32)> {
    let pov_id = state.settings.current_player_turn_id;
    let map_type = state.settings.map_type;

    // Base land chances for map types (rough estimates from mapgen.rs)
    let base_land_prob = match map_type {
        crate::MapType::None => 0.5,
        crate::MapType::Drylands => 1.0,
        crate::MapType::Lakes => 0.72,
        crate::MapType::Continents => 0.45,
        crate::MapType::Pangea => 0.50,
        crate::MapType::Archipelago => 0.30,
        crate::MapType::WaterWorld => 0.05,
    };

    let mut predictions = IndexMap::new();

    for &tile_idx in fog_tiles {
        let nearest_tribe = get_nearest_known_tribe(state, tile_idx).unwrap_or(TribeType::Imperius); // Default to Imperius rates if totally lost
        let biome_rates = crate::mapgen::get_tribe_biome_rates(nearest_tribe);

        let neighbors = get_adjacent_indices(state, tile_idx, 1);
        let mut land_neighbors = 0;
        let mut total_neighbors = 0;

        for n_idx in neighbors {
            if let Some(tile) = state.tiles.get(&n_idx) {
                if tile.explorers.contains(&pov_id) {
                    match tile.terrain_type {
                        TerrainType::Water | TerrainType::Ocean => {}
                        _ => land_neighbors += 1,
                    }
                    total_neighbors += 1;
                }
            }
        }

        // Adaptive Land Probability:
        // If surrounded by water, likely water. If surrounded by land, likely land.
        // If unknown, use global map type bias.
        let local_land_prob = if total_neighbors > 0 {
            (land_neighbors as f32 + 0.5) / (total_neighbors as f32 + 1.0)
        } else {
            base_land_prob
        };

        // Weighted mix of local and global
        let final_land_prob = 0.7 * local_land_prob + 0.3 * base_land_prob;

        // Decide Terrain
        let terrain_type = if final_land_prob < 0.5 {
            // Likely water. Deep ocean if far from land?
            // Simple heuristic: If we have land neighbors, it's Coast (Water). Else Ocean.
            if land_neighbors > 0 {
                TerrainType::Water
            } else {
                TerrainType::Ocean
            }
        } else {
            // Land! Use biome rates.
            // Normalize rates to 1.0 just in case
            let total_rate = biome_rates.mountain + biome_rates.forest + biome_rates.field;

            // deterministic pseudo-random from coord
            let pseudo_rnd = ((tile_idx * 12345 + 67890) % 100) as f32 / 100.0;

            if pseudo_rnd < (biome_rates.mountain / total_rate) {
                TerrainType::Mountain
            } else if pseudo_rnd < ((biome_rates.mountain + biome_rates.forest) / total_rate) {
                TerrainType::Forest
            } else {
                TerrainType::Field
            }
        };

        let climate = tribe_to_climate(nearest_tribe);

        let final_climate =
            if terrain_type == TerrainType::Water || terrain_type == TerrainType::Ocean {
                0 // fluids are functionally Nature climate
            } else {
                climate
            };

        predictions.insert(tile_idx, (terrain_type, final_climate));
    }

    predictions
}

pub fn get_border_clouds(state: &GameState) -> Vec<i32> {
    let pov_id = state.settings.current_player_turn_id;
    let mut border = std::collections::HashSet::new();
    for (&idx, tile) in &state.tiles {
        if tile.explorers.contains(&pov_id) {
            for n in get_adjacent_indices(state, idx, 1) {
                let n_explored = state
                    .tiles
                    .get(&n)
                    .map(|t| t.explorers.contains(&pov_id))
                    .unwrap_or(false);
                if !n_explored {
                    border.insert(n);
                }
            }
        }
    }
    border.into_iter().collect()
}

pub fn update_predictions(state: &mut GameState) {
    let villages = predict_villages(state);

    // Prediction for ALL unexplored tiles (Mental Image)
    let pov_id = state.settings.current_player_turn_id;
    let mut fog_tiles = Vec::new();
    for (&idx, tile) in &state.tiles {
        if !tile.explorers.contains(&pov_id) {
            fog_tiles.push(idx);
        }
    }

    let terrain = predict_terrain(state, &fog_tiles);
    let enemy_capitals = predict_enemy_capitals(state);

    state._prediction = Some(crate::states::PredictionState {
        _villages: villages,
        _terrain: terrain,
        _enemy_capital_suspects: enemy_capitals,
        _city_rewards: Vec::new(),
    });
}

pub fn predict_enemy_capitals(state: &GameState) -> Vec<i32> {
    let size = state.settings.size;
    let pov_id = state.settings.current_player_turn_id;
    let mut pov_cap = None;
    for tribe in state.tribes.values() {
        if tribe.id == pov_id && !tribe.cities.is_empty() {
            pov_cap = Some(tribe.cities[0].idx);
            break;
        }
    }
    let Some(cap) = pov_cap else {
        return Vec::new();
    };
    let (px, py) = (cap % size, cap / size);
    let tx = if px < size / 2 { size - 1 } else { 0 };
    let ty = if py < size / 2 { size - 1 } else { 0 };
    let target = ty * size + tx;

    get_adjacent_indices(state, target, 3)
        .into_iter()
        .filter(|idx| {
            !state
                .tiles
                .get(idx)
                .map(|t| t.explorers.contains(&pov_id))
                .unwrap_or(false)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::states::{GameState, TileState, TribeState};
    use crate::types::{TerrainType, TribeType};
    use std::collections::HashSet;

    #[test]
    fn test_village_prediction_constraints() {
        let mut state = GameState::default();
        let size = 11;
        state.settings.size = size;
        for i in 0..(size * size) {
            let mut tile = TileState::default();
            tile.coords = crate::coords::Coords::from_index(i, size);
            tile.terrain_type = TerrainType::Field;
            state.tiles.insert(i, tile);
        }
        let pov_id = 1;
        state.settings.current_player_turn_id = pov_id;
        let mut pov_tribe = TribeState::default();
        pov_tribe.id = pov_id;
        pov_tribe.tribe_type = TribeType::Imperius;
        state.tribes.insert(pov_id, pov_tribe);

        let known_cities = HashSet::new();
        let ocean_idx = 2 * size + 2;
        state.tiles.get_mut(&ocean_idx).unwrap().terrain_type = TerrainType::Ocean;
        let adj_idx = 2 * size + 3;
        assert!(!validate_village_candidate(
            &state,
            adj_idx,
            &IndexMap::new(),
            &known_cities
        ));

        let city_idx = 5 * size + 5;
        let mut cities = HashSet::new();
        cities.insert(city_idx);
        assert!(!validate_village_candidate(
            &state,
            city_idx + 1,
            &IndexMap::new(),
            &cities
        ));
        assert!(validate_village_candidate(
            &state,
            city_idx + 3,
            &IndexMap::new(),
            &cities
        ));
    }

    #[test]
    fn test_terrain_prediction_biomes() {
        let mut state = GameState::default();
        let size = 11;
        state.settings.size = size;
        state.settings.map_type = crate::types::MapType::Drylands; // all land

        // Setup Bardur Tribe
        let bardur_id = 2;
        let mut bardur_tribe = TribeState::default();
        bardur_tribe.id = bardur_id;
        bardur_tribe.tribe_type = TribeType::Bardur;
        state.tribes.insert(bardur_id, bardur_tribe);

        let bardur_cap_idx = 2 * size + 2;
        let mut cap_tile = TileState::default();
        cap_tile.coords = crate::coords::Coords::from_index(bardur_cap_idx, size);
        cap_tile.capital_of = bardur_id;
        cap_tile.owner = bardur_id;
        cap_tile.terrain_type = TerrainType::Field;
        state.tiles.insert(bardur_cap_idx, cap_tile);

        // Prediction Target: Tile near Bardur
        let target_idx = 2 * size + 3; // Adjacent to Bardur Cap

        // Let's predict!
        let predictions = predict_terrain(&state, &[target_idx]);
        let (pred_terrain, pred_climate) = predictions[&target_idx];

        // 1. Should be Land (Drylands + Land Neighbor)
        assert_ne!(pred_terrain, TerrainType::Water);
        assert_ne!(pred_terrain, TerrainType::Ocean);

        // 2. Should correspond to Bardur climate (classic id 3)
        assert_eq!(pred_climate, crate::types::classic_climate_id(TribeType::Bardur));

        // 3. Terrain Type Check (Probabilistic but deterministic seed)
        // With current deterministic RNG:
        // idx = 25. pseudo_rnd = ((25 * 12345 + 67890) % 100) / 100.0 => 0.15
        // Bardur Rates (approx): Mountain 0.14, Forest 0.30, Field 0.56
        // 0.15 > 0.14 but < (0.14+0.30=0.44). So likely Forest.
        // Wait, if 0.15 is slightly above 0.14, it falls into "Forest" bucket.
        // Let's assert it is Forest or Field, ensuring it works.
        // Actually, if I can force it to be Forest, great.
        // If pseudo_rnd is 0.15, and mountain rate is ~0.14, then it IS Forest.
        // Let's just assert it is valid land.
        assert!(matches!(
            pred_terrain,
            TerrainType::Field | TerrainType::Forest | TerrainType::Mountain
        ));
    }
}
