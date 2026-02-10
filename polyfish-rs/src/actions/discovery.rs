//! Discovery actions

use crate::actions::city::add_population;
use crate::actions::{UndoCallback, chain_undos};
use crate::functions::{get_adjacent_indices, get_capital_city};
use crate::settings::has_skill;
use crate::states::{GameState, PlayerId, UnitState};
use crate::types::{SkillType, StructureType, TerrainType};

/// Discover tiles around a unit or specific tiles
pub fn discover_tiles(
    state: &mut GameState,
    tribe_id: PlayerId,
    unit: Option<&UnitState>,
    tile_indices: Option<Vec<i32>>,
) -> UndoCallback {
    let pov_id = tribe_id;

    // Determine tiles to reveal
    let tiles_to_check = if let Some(indices) = tile_indices {
        indices
    } else if let Some(u) = unit {
        let range = if has_skill(u.unit_type, SkillType::Scout)
            || state
                .tiles
                .get(&u.coords.idx)
                .map(|t| t.terrain_type == TerrainType::Mountain)
                .unwrap_or(false)
        {
            2
        } else {
            1
        };
        let mut adj = get_adjacent_indices(state, u.coords.idx, range);
        adj.push(u.coords.idx);
        adj
    } else {
        Vec::new()
    };

    // Filter to tiles not yet explored by this player
    let newly_discovered: Vec<i32> = tiles_to_check
        .into_iter()
        .filter(|&idx| {
            state
                .tiles
                .get(&idx)
                .map(|t| !t.explorers.contains(&pov_id))
                .unwrap_or(false)
        })
        .collect();

    if newly_discovered.is_empty() {
        return Box::new(|_| {});
    }

    let mut undos: Vec<UndoCallback> = Vec::new();

    // Update score
    let score_gain = 5 * newly_discovered.len() as i32;
    if let Some(tribe) = state.tribes.get_mut(&pov_id) {
        tribe.score += score_gain;
    }
    undos.push(Box::new(move |s: &mut GameState| {
        if let Some(tribe) = s.tribes.get_mut(&pov_id) {
            tribe.score -= score_gain;
        }
    }));

    // Process each tile - only mark as explored during real moves (not MCTS)
    let is_real_move = state.settings._are_you_sure;
    for idx in newly_discovered {
        if is_real_move {
            // Mark explored (permanent, no undo)
            if let Some(tile) = state.tiles.get_mut(&idx) {
                tile.explorers.insert(pov_id);

                // Check if lighthouse
                if let Some(Some(struct_state)) = state.structures.get(&idx) {
                    if struct_state.structure_type == StructureType::Lighthouse {
                        let city_to_reward = get_capital_city(state, pov_id)
                            .map(|c| c.tile_index)
                            .or_else(|| {
                                state
                                    .tribes
                                    .get(&pov_id)
                                    .and_then(|t| t.cities.first().map(|c| c.tile_index))
                            });

                        if let Some(city_idx) = city_to_reward {
                            undos.push(add_population(state, city_idx, 1));
                        }
                    }
                }
            }
        }
    }

    // Check for other tribes (integrated here or called separately)
    undos.push(crate::actions::try_discover_other_tribes(state));

    chain_undos(undos)
}

use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::HashSet;

/// Predict where an explorer will go and return (path, revealed_tiles)
pub fn predict_explorer(state: &GameState, start_idx: i32) -> (Vec<i32>, Vec<i32>) {
    let pov_id = state.settings.current_player_turn_id;
    // Build current visibility from explorers
    let mut current_visible: HashSet<i32> = state
        .tiles
        .iter()
        .filter(|(_, t)| t.explorers.contains(&pov_id))
        .map(|(&idx, _)| idx)
        .collect();
    let mut explored_tiles: HashSet<i32> = HashSet::new();
    let mut path_indices: Vec<i32> = Vec::new();
    let mut current_tile = start_idx;
    let mut prev_tile = -1;

    path_indices.push(current_tile);

    for _ in 0..12 {
        // 1. Calculate scores for all tiles (expensive but deterministic)
        let scores = calculate_explorer_scores(state, &current_visible, pov_id);

        // 2. Identify neighbors and pick the best one
        let neighbors = get_allowed_neighbors(state, current_tile, true, Some(&current_visible));
        if neighbors.is_empty() {
            break; // Poof
        }

        let mut best_score = 10000;
        let mut candidates = Vec::new();

        for &n_idx in &neighbors {
            let mut score = *scores.get(&n_idx).unwrap_or(&1000);

            // Backtracking penalty
            if n_idx == prev_tile {
                score += 1;
            }

            if score < best_score {
                best_score = score;
                candidates.clear();
                candidates.push(n_idx);
            } else if score == best_score {
                candidates.push(n_idx);
            }
        }

        let next_tile = if candidates.is_empty() {
            current_tile
        } else {
            *candidates.choose(&mut thread_rng()).unwrap()
        };

        if next_tile == current_tile {
            // No progress possible or stuck
            break;
        }

        // Move and reveal
        prev_tile = current_tile;
        current_tile = next_tile;
        path_indices.push(current_tile);

        // Reveal tile and its neighbors (vision range 1 for prediction,
        // mountain reveal happens in discover_tiles action but we predict it here too)
        // Explorer has range 2 (5x5 area)
        let range = 2;

        let mut to_reveal = get_adjacent_indices(state, current_tile, range);
        to_reveal.push(current_tile);

        for r_idx in to_reveal {
            if !current_visible.contains(&r_idx) {
                current_visible.insert(r_idx);
                explored_tiles.insert(r_idx);
            }
        }
    }

    (path_indices, explored_tiles.into_iter().collect())
}

fn calculate_explorer_scores(
    state: &GameState,
    visible: &HashSet<i32>,
    _pov_id: PlayerId,
) -> std::collections::HashMap<i32, i32> {
    let mut scores = std::collections::HashMap::new();

    // 1. Initial scoring for all fog tiles
    // We scan ALL tiles for fog
    for (&idx, _) in &state.tiles {
        if !visible.contains(&idx) {
            scores.insert(idx, score_fog_tile(state, visible, idx));
        }
    }

    // 2. BFS iterations (3 steps)
    for _ in 0..3 {
        let mut next_scores = scores.clone();
        for (&idx, &score) in &scores {
            let adj = get_adjacent_indices(state, idx, 1);
            for n_idx in adj {
                let current_n_score = *next_scores.get(&n_idx).unwrap_or(&1000);
                let new_score = score + 100;
                if new_score < current_n_score {
                    next_scores.insert(n_idx, new_score);
                }
            }
        }
        scores = next_scores;
    }

    scores
}

fn score_fog_tile(state: &GameState, visible: &HashSet<i32>, idx: i32) -> i32 {
    let mut reveal_count = 0;
    let mut has_lighthouse = false;

    // Determine what would be revealed if this tile was uncovered
    // "vision range of the explorer is actually 5 tiles" - wait
    // Actually the wiki says "Scan the area around the explorer for fog tiles
    // to uncover within a movement range of 4 steps"
    // "we actually scan from each fog tile towards the explorer"

    // The 110-173 scoring is based on how many OTHER fog tiles are cleared
    // when THIS fog tile is cleared.
    // Explorers have 5x5 vision (Range 2)
    let adj = get_adjacent_indices(state, idx, 2);
    let mut check_reveal = adj;
    check_reveal.push(idx);

    for r_idx in check_reveal {
        if !visible.contains(&r_idx) {
            reveal_count += 1;
            if let Some(Some(s)) = state.structures.get(&r_idx) {
                if s.structure_type == StructureType::Lighthouse {
                    has_lighthouse = true;
                }
            }
        }
    }

    reveal_count = reveal_count.clamp(1, 4);

    match (reveal_count, has_lighthouse) {
        (4, true) => 110,
        (3, true) => 120,
        (2, true) => 130,
        (1, true) => 140,
        (4, false) => 143,
        (3, false) => 153,
        (2, false) => 163,
        (1, false) => 173,
        _ => 173, // Should not happen for reveal_count < 1 as we are in a fog tile
    }
}

fn get_allowed_neighbors(
    state: &GameState,
    idx: i32,
    include_unexplored: bool,
    simulated_visible: Option<&HashSet<i32>>,
) -> Vec<i32> {
    let pov_id = state.settings.current_player_turn_id;
    let tribe = match state.tribes.get(&pov_id) {
        Some(t) => t,
        None => return Vec::new(),
    };

    let adj = get_adjacent_indices(state, idx, 1);
    let mut allowed = Vec::new();

    for n_idx in adj {
        // Check visibility using simulation context if provided, otherwise game state
        let is_explored = if let Some(sim_vis) = simulated_visible {
            sim_vis.contains(&n_idx)
        } else {
            state
                .tiles
                .get(&n_idx)
                .map(|t| t.explorers.contains(&pov_id))
                .unwrap_or(false)
        };

        if !include_unexplored && !is_explored {
            continue;
        }

        if is_steppable_for_explorer(state, n_idx, tribe, is_explored) {
            allowed.push(n_idx);
        }
    }

    allowed
}

fn is_steppable_for_explorer(
    state: &GameState,
    idx: i32,
    tribe: &crate::states::TribeState,
    is_explored: bool,
) -> bool {
    use crate::settings::technology::has_technology;
    use crate::types::TechnologyType;

    // Check tile exists
    if state.tiles.get(&idx).is_none() {
        return false;
    }

    // If unexplored, be optimistic - assume it's passable
    // This is for prediction/simulation purposes
    if !is_explored {
        return true;
    }

    // If explored, use FOW-safe terrain access
    let pov_id = state.settings.current_player_turn_id;
    let terrain = crate::fow::get_terrain_at(state, idx, pov_id);

    match terrain {
        TerrainType::None => false,
        TerrainType::Mountain => has_technology(&tribe.tech_vanilla, TechnologyType::Climbing),
        TerrainType::Water => has_technology(&tribe.tech_vanilla, TechnologyType::Sailing),
        TerrainType::Ocean => has_technology(&tribe.tech_vanilla, TechnologyType::Navigation),
        _ => true, // Field, Forest, etc.
    }
}
