//! Prediction module for FOW (Fog of War)
//!
//! Provides prediction functions for MCTS simulations to avoid accessing ground truth data.
//! When `_are_you_sure = false`, the engine uses these predictions instead of actual hidden data.

use crate::functions::get_adjacent_indices;
use crate::states::GameState;
use crate::types::{ClimateType, TerrainType, TribeType};
use indexmap::IndexMap;
use std::collections::HashMap;

/// Maps ClimateType to corresponding TribeType
pub fn climate_to_tribe(climate: ClimateType) -> TribeType {
    match climate {
        ClimateType::XinXi => TribeType::XinXi,
        ClimateType::Imperius => TribeType::Imperius,
        ClimateType::Bardur => TribeType::Bardur,
        ClimateType::Oumaji => TribeType::Oumaji,
        ClimateType::Kickoo => TribeType::Kickoo,
        ClimateType::Hoodrick => TribeType::Hoodrick,
        ClimateType::Luxidoor => TribeType::Luxidoor,
        ClimateType::Vengir => TribeType::Vengir,
        ClimateType::Zebasi => TribeType::Zebasi,
        ClimateType::AiMo => TribeType::AiMo,
        ClimateType::Aquarion => TribeType::Aquarion,
        ClimateType::Quetzali => TribeType::Quetzali,
        ClimateType::Elyrion => TribeType::Elyrion,
        ClimateType::Yadakk => TribeType::Yadakk,
        ClimateType::Polaris => TribeType::Polaris,
        ClimateType::Cymanti => TribeType::Cymanti,
        ClimateType::Nature => TribeType::Nature,
    }
}

/// Maps TribeType to corresponding ClimateType
pub fn tribe_to_climate(tribe: TribeType) -> ClimateType {
    match tribe {
        TribeType::XinXi => ClimateType::XinXi,
        TribeType::Imperius => ClimateType::Imperius,
        TribeType::Bardur => ClimateType::Bardur,
        TribeType::Oumaji => ClimateType::Oumaji,
        TribeType::Kickoo => ClimateType::Kickoo,
        TribeType::Hoodrick => ClimateType::Hoodrick,
        TribeType::Luxidoor => ClimateType::Luxidoor,
        TribeType::Vengir => ClimateType::Vengir,
        TribeType::Zebasi => ClimateType::Zebasi,
        TribeType::AiMo => ClimateType::AiMo,
        TribeType::Aquarion => ClimateType::Aquarion,
        TribeType::Quetzali => ClimateType::Quetzali,
        TribeType::Elyrion => ClimateType::Elyrion,
        TribeType::Yadakk => ClimateType::Yadakk,
        TribeType::Polaris => ClimateType::Polaris,
        TribeType::Cymanti => ClimateType::Cymanti,
        TribeType::Nature | TribeType::None => ClimateType::Nature,
    }
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

    let mut candidates: IndexMap<i32, (i32, ClimateType)> = IndexMap::new();

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
                        let entry = candidates.entry(n_idx).or_insert((0, ClimateType::Nature));
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
        if tile.owner != pov_id
            && tile.climate != pov_climate
            && tile.climate != ClimateType::Nature
        {
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

/// Predict terrain for fog tiles based on visible neighbors
pub fn predict_terrain(
    state: &GameState,
    fog_tiles: &[i32],
) -> IndexMap<i32, (TerrainType, ClimateType)> {
    let pov_id = state.settings.current_player_turn_id;
    let mut predictions = IndexMap::new();
    for &tile_idx in fog_tiles {
        let neighbors = get_adjacent_indices(state, tile_idx, 1);
        let mut terrain_counts = HashMap::new();
        let mut climate_counts = HashMap::new();

        for n_idx in neighbors {
            if let Some(tile) = state.tiles.get(&n_idx) {
                if tile.explorers.contains(&pov_id) {
                    *terrain_counts.entry(tile.terrain_type).or_insert(0) += 1;
                    *climate_counts.entry(tile.climate).or_insert(0) += 1;
                }
            }
        }

        let terrain = terrain_counts
            .into_iter()
            .max_by_key(|&(_, c)| c)
            .map(|(t, _)| t)
            .unwrap_or(TerrainType::Field);
        let climate = climate_counts
            .into_iter()
            .max_by_key(|&(_, c)| c)
            .map(|(c, _)| c)
            .unwrap_or(ClimateType::Nature);

        let final_climate = if terrain == TerrainType::Water || terrain == TerrainType::Ocean {
            ClimateType::Nature
        } else {
            climate
        };
        predictions.insert(tile_idx, (terrain, final_climate));
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
    let border_clouds = get_border_clouds(state);
    let terrain = predict_terrain(state, &border_clouds);
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
            pov_cap = Some(tribe.cities[0].tile_index);
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
}
