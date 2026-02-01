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

/// Predict village locations in fog based on climate density
///
/// Algorithm (from TS):
/// 1. Find visible tiles with enemy climates (not owned by POV)
/// 2. For each, check adjacent fog tiles within 2 range
/// 3. Count density of candidates
/// 4. Return highest density tile as predicted village
pub fn predict_villages(state: &GameState) -> HashMap<i32, (TribeType, bool)> {
    let pov_id = state.settings.current_player_turn_id;
    let pov_tribe_type = state
        .tribes
        .get(&pov_id)
        .map(|t| t.tribe_type)
        .unwrap_or(TribeType::None);
    let pov_climate = tribe_to_climate(pov_tribe_type);

    let mut candidates: HashMap<i32, (i32, ClimateType)> = HashMap::new();

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

    // Sort by density (highest first) and return top prediction
    let mut prediction_map: HashMap<i32, (TribeType, bool)> = HashMap::new();
    if let Some((&best_idx, (_, climate))) = candidates.iter().max_by_key(|(_, (count, _))| *count)
    {
        let predicted_tribe = climate_to_tribe(*climate);
        prediction_map.insert(best_idx, (predicted_tribe, true));
    }

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
