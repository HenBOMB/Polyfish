//! Prediction module for FOW (Fog of War)
//!
//! Provides prediction functions for MCTS simulations to avoid accessing ground truth data.
//! When `_are_you_sure = false`, the engine uses these predictions instead of actual hidden data.

use crate::functions::get_adjacent_indices;
use crate::states::GameState;
use crate::types::{ClimateType, TerrainType, TribeType};
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

/// Predict village locations in fog based on climate density AND orphan resources
///
/// Algorithm:
/// 1. Resource Heuristic: Find visible resources that are too far from any known city/village.
///    These must belong to a hidden village adjacent to them.
/// 2. Climate Heuristic: Find visible tiles with enemy climates (not owned by POV).
/// 3. Combine scores and return best candidates.
pub fn predict_villages(state: &GameState) -> HashMap<i32, (TribeType, bool)> {
    let pov_id = state.settings.current_player_turn_id;
    // ...
    let pov_tribe_type = state
        .tribes
        .get(&pov_id)
        .map(|t| t.tribe_type)
        .unwrap_or(TribeType::None);
    let pov_climate = tribe_to_climate(pov_tribe_type);

    let mut candidates: HashMap<i32, (i32, ClimateType)> = HashMap::new();

    // Collect all known cities/villages (visible or explored) to check for orphan resources
    let mut known_cities = std::collections::HashSet::new();
    for (&idx, tile) in &state.tiles {
        // If we own it, we know it
        // If we explored it and found a village/city, we know it
        // If it's visible and has a village/city, we know it
        if tile.capital_of > 0
            || (tile.explorers.contains(&pov_id)
                && crate::functions::get_structure_type_at(state, idx)
                    == Some(crate::types::StructureType::Village))
            || (state._visible_tiles.contains_key(&idx)
                && crate::functions::get_structure_type_at(state, idx)
                    == Some(crate::types::StructureType::Village))
        {
            known_cities.insert(idx);
        }
    }

    // Helper to check if a resource is orphaned
    let is_orphan = |res_idx: i32| -> bool {
        if known_cities.contains(&res_idx) {
            return false;
        } // Resource ON city?
        let res_coords = crate::coords::Coords::from_index(res_idx, state.settings.size);
        for &city_idx in &known_cities {
            let city_coords = crate::coords::Coords::from_index(city_idx, state.settings.size);
            if res_coords.distance_to(&city_coords) <= 1 {
                return false; // Belong to this city
            }
        }
        true
    };

    // 1. Resource Heuristic
    for (&tile_idx, _) in &state._visible_tiles {
        if let Some(res_opt) = state.resources.get(&tile_idx) {
            if res_opt.is_some() && is_orphan(tile_idx) {
                // This resource needs a home!
                // Check neighbors for potential hidden village
                let neighbors = get_adjacent_indices(state, tile_idx, 1);
                for n_idx in neighbors {
                    // Must be unexplored fog
                    if let Some(t) = state.tiles.get(&n_idx) {
                        if t.explorers.contains(&pov_id) {
                            continue;
                        }
                        if state._visible_tiles.contains_key(&n_idx) {
                            continue;
                        }

                        // Skip edges (villages don't spawn on map borders)
                        let size = state.settings.size;
                        let x = n_idx % size;
                        let y = n_idx / size;
                        if x == 0 || x == size - 1 || y == 0 || y == size - 1 {
                            continue;
                        }

                        // Boost score significantly
                        let entry = candidates.entry(n_idx).or_insert((0, t.climate));
                        entry.0 += 5; // Strong evidence
                    }
                }
            }
        }
    }

    // 2. Climate Heuristic (existing)
    // Find visible tiles with enemy climates
    for (&tile_idx, _) in &state._visible_tiles {
        if let Some(tile) = state.tiles.get(&tile_idx) {
            // Skip tiles we own
            if tile.owner == pov_id {
                continue;
            }
            // Skip our own climate
            if tile.climate == pov_climate || tile.climate == ClimateType::Nature {
                continue;
            }

            let target_climate = tile.climate;

            // Check adjacent fog tiles within range 2
            let around = get_adjacent_indices(state, tile_idx, 2);
            for idx in around {
                // Skip if already visible
                if state._visible_tiles.contains_key(&idx) {
                    continue;
                }
                // Skip edge tiles
                let size = state.settings.size;
                let x = idx % size;
                let y = idx / size;
                if x <= 1 || x >= size - 2 || y <= 1 || y >= size - 2 {
                    continue;
                }
                // Check if explored by POV
                if let Some(t) = state.tiles.get(&idx) {
                    if t.explorers.contains(&pov_id) {
                        continue;
                    }
                }

                let entry = candidates.entry(idx).or_insert((0, target_climate));
                entry.0 += 1;
            }
        }
    }

    // Sort by density (highest first) and return top predictions
    let mut prediction_map: HashMap<i32, (TribeType, bool)> = HashMap::new();

    // Return ALL likely candidates, not just one?
    // Frontend handles multiple. Let's return top 5 or anything with score > 2?
    // Original code returned just the MAX.
    // Let's filter by threshold if we have strong resource hits.

    let mut sorted: Vec<_> = candidates.into_iter().collect();
    sorted.sort_by_key(|(_, (count, _))| -count); // Descending

    for (best_idx, (count, climate)) in sorted.iter().take(5) {
        if *count > 2 {
            // Threshold
            let predicted_tribe = climate_to_tribe(*climate);
            prediction_map.insert(*best_idx, (predicted_tribe, true));
        }
    }

    // Safety: If resource check found nothing but climate found something, return top 1
    if prediction_map.is_empty() && !sorted.is_empty() {
        let (best_idx, (_, climate)) = sorted[0];
        let predicted_tribe = climate_to_tribe(climate);
        prediction_map.insert(best_idx, (predicted_tribe, true));
    }

    // println!("Predicted villages: {:?}", prediction_map);
    prediction_map
}

/// Predict terrain for fog tiles based on visible neighbors (neighbor voting)
pub fn predict_terrain(
    state: &GameState,
    fog_tiles: &[i32],
) -> HashMap<i32, (TerrainType, ClimateType)> {
    let mut predictions: HashMap<i32, (TerrainType, ClimateType)> = HashMap::new();

    for &tile_idx in fog_tiles {
        let neighbors = get_adjacent_indices(state, tile_idx, 1);

        let mut terrain_counts: HashMap<TerrainType, i32> = HashMap::new();
        let mut climate_counts: HashMap<ClimateType, i32> = HashMap::new();

        for n_idx in neighbors {
            if state._visible_tiles.contains_key(&n_idx) {
                if let Some(tile) = state.tiles.get(&n_idx) {
                    *terrain_counts.entry(tile.terrain_type).or_insert(0) += 1;
                    *climate_counts.entry(tile.climate).or_insert(0) += 1;
                }
            }
        }

        // Find most common terrain
        let terrain = terrain_counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(t, _)| *t)
            .unwrap_or(TerrainType::Field);

        // Find most common climate
        let climate = climate_counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(c, _)| *c)
            .unwrap_or(ClimateType::Nature);

        // Water/Ocean tiles have Nature climate
        let final_climate = if terrain == TerrainType::Water || terrain == TerrainType::Ocean {
            ClimateType::Nature
        } else {
            climate
        };

        predictions.insert(tile_idx, (terrain, final_climate));
    }

    predictions
}

/// Get border clouds (fog tiles adjacent to visible tiles)
pub fn get_border_clouds(state: &GameState) -> Vec<i32> {
    let size = state.settings.size;
    let total_tiles = size * size;
    let mut border_clouds = std::collections::HashSet::new();

    for &tile_idx in state._visible_tiles.keys() {
        let neighbors = get_adjacent_indices(state, tile_idx, 1);
        for n_idx in neighbors {
            if n_idx >= 0 && n_idx < total_tiles && !state._visible_tiles.contains_key(&n_idx) {
                border_clouds.insert(n_idx);
            }
        }
    }

    border_clouds.into_iter().collect()
}

/// Update all predictions in the game state
pub fn update_predictions(state: &mut GameState) {
    // 1. Predict Villages
    let villages = predict_villages(state);

    // 2. Predict Terrain for border clouds
    let border_clouds = get_border_clouds(state);
    let terrain = predict_terrain(state, &border_clouds);

    // 3. Predict Enemy Capitals
    let enemy_capitals = predict_enemy_capitals(state); // Wait, I need to implement this or find it

    // Update state
    let prediction = crate::states::PredictionState {
        _villages: villages,
        _terrain: terrain,
        _enemy_capital_suspects: enemy_capitals,
        _city_rewards: Vec::new(), // Not implemented yet
    };

    state._prediction = Some(prediction);
}

/// Predict likely enemy capital locations based on heuristics
pub fn predict_enemy_capitals(state: &GameState) -> Vec<i32> {
    // Simply return empty for now if logic is missing, or implement basic heuristic
    // Heuristic: Center of large unexplored areas?
    // For now, let's check if we have the advanced logic.
    // I can port the TS logic later.
    Vec::new()
}
