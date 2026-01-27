//! Discovery actions

use crate::actions::city::add_population;
use crate::actions::{chain_undos, UndoCallback};
use crate::functions::{get_adjacent_indices, get_capital_city, get_pov_tribe};
use crate::settings::has_skill;
use crate::states::{GameState, UnitState};
use crate::types::{SkillType, StructureType, TerrainType};

/// Discover tiles around a unit or specific tiles
pub fn discover_tiles(
    state: &mut GameState,
    unit: Option<&UnitState>,
    tile_indices: Option<Vec<i32>>,
) -> UndoCallback {
    let pov_id = state.settings.current_player_turn_id;

    // Determine tiles to reveal
    let tiles_to_check = if let Some(indices) = tile_indices {
        indices
    } else if let Some(u) = unit {
        let range = if state
            .tiles
            .get(&u.coords.idx)
            .map(|t| t.terrain_type == TerrainType::Mountain)
            .unwrap_or(false)
            || has_skill(u.unit_type, SkillType::Scout)
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

    let newly_discovered: Vec<i32> = tiles_to_check
        .into_iter()
        .filter(|&idx| !state._visible_tiles.contains_key(&idx))
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

    // Process each tile
    for idx in newly_discovered {
        // Set visible
        state._visible_tiles.insert(idx, true);

        // Mark explored
        if let Some(tile) = state.tiles.get_mut(&idx) {
            if !tile.explorers.contains(&pov_id) {
                tile.explorers.insert(pov_id);
                // Undo explorer mark
                let tribe_id = pov_id;
                undos.push(Box::new(move |s: &mut GameState| {
                    if let Some(t) = s.tiles.get_mut(&idx) {
                        t.explorers.remove(&tribe_id);
                    }
                }));

                // Check if lighthouse
                if let Some(Some(struct_state)) = state.structures.get(&idx) {
                    if struct_state.structure_type == StructureType::Lighthouse {
                        if let Some(capital) = get_capital_city(state, pov_id) {
                            let cap_idx = capital.tile_index;
                            undos.push(add_population(state, cap_idx, 1));
                        }
                    }
                }
            }
        }
        // Undo visibility
        undos.push(Box::new(move |s| {
            s._visible_tiles.remove(&idx);
        }));
    }

    // Check for other tribes (integrated here or called separately)
    undos.push(crate::actions::try_discover_other_tribes(state));

    chain_undos(undos)
}

use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::{HashSet, VecDeque};

/// Predict where an explorer will go and return revealed tile indices
pub fn predict_explorer(state: &GameState, start_idx: i32) -> Vec<i32> {
    let mut current_visible = state._visible_tiles.clone();
    let mut explored_tiles: HashSet<i32> = HashSet::new();
    let mut current_tile = start_idx;
    let pov_id = state.settings.current_player_turn_id;
    let map_size = state.settings.size;

    for _ in 0..15 {
        // Find nearest cloud within 4 moves
        let path = find_nearest_cloud(state, &current_visible, current_tile, 4);

        let next_tile = if let Some(p) = path {
            if p.len() > 1 {
                p[1]
            } else {
                current_tile
            }
        } else {
            // Random allowed neighbor
            let allowed = get_allowed_neighbors(state, current_tile, false);
            if !allowed.is_empty() {
                *allowed.choose(&mut thread_rng()).unwrap()
            } else {
                current_tile
            }
        };

        // Move and reveal
        current_tile = next_tile;

        // Reveal tile and its neighbors
        let to_reveal = {
            let mut adj = get_adjacent_indices(state, next_tile, 1);
            adj.push(next_tile);
            adj
        };

        for r_idx in to_reveal {
            if !current_visible.contains_key(&r_idx) {
                current_visible.insert(r_idx, true);
                explored_tiles.insert(r_idx);
            }
        }
    }

    explored_tiles.into_iter().collect()
}

fn find_nearest_cloud(
    state: &GameState,
    visible: &std::collections::HashMap<i32, bool>,
    start_idx: i32,
    max_dist: i32,
) -> Option<Vec<i32>> {
    let mut queue = VecDeque::new();
    let mut visited = HashSet::new();

    queue.push_back((start_idx, vec![start_idx]));
    visited.insert(start_idx);

    let mut candidates = Vec::new();
    let mut current_dist = 0;

    while !queue.is_empty() && current_dist <= max_dist {
        let level_size = queue.len();
        for _ in 0..level_size {
            let (idx, path) = queue.pop_front().unwrap();

            // Check if cloud
            if path.len() > 1 && !visible.contains_key(&idx) {
                candidates.push(path.clone());
            }

            // Explore neighbors
            if (path.len() as i32 - 1) < max_dist {
                let mut neighbors = get_allowed_neighbors(state, idx, true);
                neighbors.shuffle(&mut thread_rng());
                for n_idx in neighbors {
                    if !visited.contains(&n_idx) {
                        visited.insert(n_idx);
                        let mut new_path = path.clone();
                        new_path.push(n_idx);
                        queue.push_back((n_idx, new_path));
                    }
                }
            }
        }

        if !candidates.is_empty() {
            break;
        }
        current_dist += 1;
    }

    if candidates.is_empty() {
        return None;
    }

    candidates.choose(&mut thread_rng()).cloned()
}

fn get_allowed_neighbors(state: &GameState, idx: i32, include_unexplored: bool) -> Vec<i32> {
    let pov_id = state.settings.current_player_turn_id;
    let tribe = match state.tribes.get(&pov_id) {
        Some(t) => t,
        None => return Vec::new(),
    };

    use crate::settings::{has_skill, has_technology};
    use crate::types::{SkillType, TechnologyType};

    let mut odds = 0.45;
    if has_technology(&tribe.tech_vanilla, TechnologyType::Fishing) {
        odds += 0.25;
    }
    if has_technology(&tribe.tech_vanilla, TechnologyType::Sailing) {
        odds += 0.10;
    }
    if has_technology(&tribe.tech_vanilla, TechnologyType::Climbing) {
        odds += 0.10;
    }

    let adj = get_adjacent_indices(state, idx, 1);

    let mut allowed = Vec::new();
    let mut rng = thread_rng();

    for n_idx in adj {
        let is_visible = state._visible_tiles.contains_key(&n_idx);

        if !is_visible && !include_unexplored {
            continue;
        }

        // Check if explorer already been there (mark in tile)
        let been_there = state
            .tiles
            .get(&n_idx)
            .map(|t| t.explorers.contains(&pov_id))
            .unwrap_or(false);

        if been_there {
            // If been there, use standard steppability (cheating slightly for simplicity as per TS)
            // TS: state.tiles[x].explorers.has(pov.id)? isTribeSteppable(state, x)
            if is_steppable_for_explorer(state, n_idx) {
                allowed.push(n_idx);
            }
        } else {
            // Random chance
            use rand::Rng;
            if rng.gen::<f64>() < odds {
                allowed.push(n_idx);
            }
        }
    }

    allowed
}

fn is_steppable_for_explorer(state: &GameState, idx: i32) -> bool {
    // Simplification of steppable for dummy explorer
    if let Some(tile) = state.tiles.get(&idx) {
        match tile.terrain_type {
            TerrainType::None => false,
            _ => true,
        }
    } else {
        false
    }
}
