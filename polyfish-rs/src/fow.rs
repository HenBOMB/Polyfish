//! Fog of War (FOW) Abstraction Layer
//!
//! Provides accessor functions that prevent MCTS from accessing ground truth
//! data for unexplored tiles. During MCTS simulations (`_are_you_sure = false`),
//! these functions return predicted data instead of actual data.

use crate::states::{GameState, PlayerId};
use crate::types::{ResourceType, TerrainType};

/// Get terrain at a tile, using prediction for unexplored tiles during MCTS
///
/// - Returns actual terrain if tile is explored by player
/// - Returns actual terrain during real moves (`_are_you_sure = true`)
/// - Returns predicted terrain during MCTS for unexplored tiles
pub fn get_terrain_at(state: &GameState, idx: i32, pov_id: PlayerId) -> TerrainType {
    let tile = match state.tiles.get(&idx) {
        Some(t) => t,
        None => return TerrainType::None,
    };

    // If tile is explored by this player, return actual terrain
    if tile.explorers.contains(&pov_id) {
        return tile.terrain_type;
    }

    // If real move, return actual (we're revealing now)
    if state.settings._are_you_sure {
        return tile.terrain_type;
    }

    // MCTS simulation: return predicted terrain
    get_predicted_terrain(state, idx)
}

/// Get predicted terrain for a fog tile (for MCTS use)
///
/// Uses cached prediction from `_prediction` if available,
/// otherwise falls back to Field (optimistic assumption).
pub fn get_predicted_terrain(state: &GameState, idx: i32) -> TerrainType {
    // Check cached prediction first
    if let Some(pred) = &state._prediction {
        if let Some(&(terrain, _)) = pred._terrain.get(&idx) {
            return terrain;
        }
    }

    // Fallback: assume Field (passable, optimistic)
    TerrainType::Field
}

/// Get resource at a tile, using prediction for unexplored tiles during MCTS
///
/// Returns None for unexplored tiles during MCTS (resources are hidden in fog).
pub fn get_resource_at(state: &GameState, idx: i32, pov_id: PlayerId) -> Option<ResourceType> {
    let tile = match state.tiles.get(&idx) {
        Some(t) => t,
        None => return None,
    };

    // If tile is explored by this player, check actual resource
    if tile.explorers.contains(&pov_id) {
        return state
            .resources
            .get(&idx)
            .and_then(|r| r.as_ref())
            .map(|r| r.resource_type);
    }

    // If real move, return actual
    if state.settings._are_you_sure {
        return state
            .resources
            .get(&idx)
            .and_then(|r| r.as_ref())
            .map(|r| r.resource_type);
    }

    // MCTS simulation: resources are hidden in fog
    None
}

/// Check if a tile is passable for a unit type, using prediction for fog tiles
///
/// This is the safe wrapper around terrain passability checks during MCTS.
pub fn is_terrain_passable(
    state: &GameState,
    idx: i32,
    pov_id: PlayerId,
    has_climbing: bool,
    has_sailing: bool,
    has_navigation: bool,
) -> bool {
    let terrain = get_terrain_at(state, idx, pov_id);

    match terrain {
        TerrainType::None => false,
        TerrainType::Mountain => has_climbing,
        TerrainType::Water => has_sailing,
        TerrainType::Ocean => has_navigation,
        _ => true, // Field, Forest, Ice, etc.
    }
}

/// Check if water terrain (for unit placement checks)
pub fn is_water_terrain_at(state: &GameState, idx: i32, pov_id: PlayerId) -> bool {
    let terrain = get_terrain_at(state, idx, pov_id);
    matches!(terrain, TerrainType::Water | TerrainType::Ocean)
}
