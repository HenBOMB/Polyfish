//! Capital connection logic
//!
//! Handles detecting networks of cities connected by roads and ports.

use crate::actions::UndoCallback;
use crate::actions::chain_undos;
use crate::actions::city::add_population;
use crate::functions::get_capital_city;
use crate::states::GameState;
use crate::types::{StructureType, TerrainType, TribeType};
use std::collections::{HashSet, VecDeque};

/// Update capital connections for a tribe
///
/// Should be called after building roads or ports.
pub fn update_capital_connections(state: &mut GameState, tribe_id: i32) -> UndoCallback {
    let _map_size = state.settings.size;

    // Find capital
    let capital_idx = if let Some(cap) = get_capital_city(state, tribe_id) {
        cap.idx
    } else {
        return Box::new(|_| {});
    };

    // Identify all cities
    let tribe = match state.tribes.get(&tribe_id) {
        Some(t) => t,
        None => return Box::new(|_| {}),
    };

    let city_indices: HashSet<i32> = tribe.cities.iter().map(|c| c.idx).collect();

    // BFS to find all connected tiles from Capital
    // Connectivity rules:
    // - Roads connect to adjacent Roads or Cities.
    // - Ports connect to other Ports via Water/Ocean.

    let mut connected_cities = HashSet::new();
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();

    // Start at capital
    queue.push_back(capital_idx);
    visited.insert(capital_idx);
    connected_cities.insert(capital_idx);

    while let Some(current_idx) = queue.pop_front() {
        let tribe_type = tribe.tribe_type;

        let is_port = has_port(state, current_idx);
        let current_has_road = has_road(state, current_idx)
            || city_indices.contains(&current_idx)
            || has_algae(state, current_idx)
            || is_port;
        let current_has_mycelium =
            has_mycelium(state, current_idx) || city_indices.contains(&current_idx);

        let neighbors = crate::functions::get_adjacent_indices(state, current_idx, 1);
        for n_idx in neighbors {
            if visited.contains(&n_idx) {
                continue;
            }

            let n_has_road =
                has_road(state, n_idx) || city_indices.contains(&n_idx) || has_port(state, n_idx);
            let n_has_algae = has_algae(state, n_idx);
            let n_has_mycelium = has_mycelium(state, n_idx);

            let mut connected = false;

            if tribe_type == TribeType::Cymanti {
                // Cymanti connections: Mycelium/Cities connect via Algae/Roads (if any) or adjacency
                // But specifically Mycelium has 4-range connection.
                if (current_has_mycelium || current_has_road || has_algae(state, current_idx))
                    && (n_has_road
                        || n_has_algae
                        || city_indices.contains(&n_idx)
                        || n_has_mycelium)
                {
                    connected = true;
                }
            } else if tribe_type == TribeType::Polaris {
                // TODO: Polaris disabled
                // TODO Outposts 5x5 ice connections
            } else {
                // Standard road connection
                if current_has_road && n_has_road {
                    connected = true;
                }
            }

            if connected {
                visited.insert(n_idx);
                queue.push_back(n_idx);
                if city_indices.contains(&n_idx) {
                    connected_cities.insert(n_idx);
                }
            }
        }

        // Cymanti Mycelium long-range connection (4 radius)
        if tribe_type == TribeType::Cymanti && current_has_mycelium {
            let radius_indices = crate::functions::get_adjacent_indices(state, current_idx, 4);
            for target_idx in radius_indices {
                if visited.contains(&target_idx) {
                    continue;
                }

                // Can connect to City or Mycelium
                if city_indices.contains(&target_idx) || has_mycelium(state, target_idx) {
                    // Path check: through field, forest, or algae
                    if has_cymanti_path(state, state.settings.size, current_idx, target_idx) {
                        visited.insert(target_idx);
                        queue.push_back(target_idx);
                        if city_indices.contains(&target_idx) {
                            connected_cities.insert(target_idx);
                        }
                    }
                }
            }
        }

        // Port-to-Port connection (4 radius)
        if is_port && tribe_type != TribeType::Cymanti {
            let radius_indices = crate::functions::get_adjacent_indices(state, current_idx, 4);
            for target_idx in radius_indices {
                if visited.contains(&target_idx) {
                    continue;
                }

                if has_port(state, target_idx) {
                    // Path check: only through water or ocean
                    if has_water_path(state, current_idx, target_idx) {
                        visited.insert(target_idx);
                        queue.push_back(target_idx);
                        if city_indices.contains(&target_idx) {
                            connected_cities.insert(target_idx);
                        }
                    }
                }
            }
        }
    }

    // Apply changes
    let mut undos = Vec::new();
    let mut capital_pop_gain = 0;

    // We need to borrow tribe mutably again.
    // BUFFER updates to avoid borrow conflict
    let mut city_pop_changes: Vec<(i32, i32)> = Vec::new();
    let mut updates = Vec::new(); // city_idx (in array or tile_idx?)

    if let Some(tribe) = state.tribes.get_mut(&tribe_id) {
        for (_idx, city) in tribe.cities.iter_mut().enumerate() {
            if city.idx == capital_idx {
                continue;
            }

            let is_connected_now = connected_cities.contains(&city.idx);
            if is_connected_now && !city.connected_to_capital {
                city.connected_to_capital = true;
                city_pop_changes.push((city.idx, 1));
                updates.push(city.idx);
                capital_pop_gain += 1;
            } else if !is_connected_now && city.connected_to_capital {
                // Connection lost
                city.connected_to_capital = false;
                city_pop_changes.push((city.idx, -1));
                updates.push(city.idx);
                capital_pop_gain -= 1;
            }
        }
    }

    // Apply population bonus for newly connected/disconnected cities
    for (city_tile_idx, amount) in city_pop_changes {
        undos.push(add_population(state, city_tile_idx, amount));
    }

    // Apply population bonus to Capital
    if capital_pop_gain != 0 {
        if let Some(cap) = crate::functions::get_capital_city(state, tribe_id) {
            // Debug logic
            if capital_pop_gain > 2 {
                println!(
                    "DEBUG: Massive population gain for capital! +{}",
                    capital_pop_gain
                );
            }
            if capital_pop_gain < -2 {
                // println!("DEBUG: Massive population loss for capital: {}", capital_pop_gain);
            }

            undos.push(add_population(state, cap.idx, capital_pop_gain));
        }
    }

    // Note: Undo for `connected_to_capital` flag is needed.
    // add_population undo handles pop. But flag undo is manual.
    // We didn't change the flag in `match tribe...` above?
    // Wait, line 125: `city.connected_to_capital = true;`. Yes we did.
    // We need to undo that.

    let _changed_cities: Vec<i32> = connected_cities.into_iter().collect();
    // Actually we only changed those in `updates`.

    // Correct logic: we used `updates` to track those who FLIPPED from false to true.
    // We need to store that list for undo.
    let flipped_cities = updates.clone(); // indices (tile_idx)

    undos.push(Box::new(move |s| {
        if let Some(t) = s.tribes.get_mut(&tribe_id) {
            for c in t.cities.iter_mut() {
                if flipped_cities.contains(&c.idx) {
                    c.connected_to_capital = false;
                }
            }
        }
    }));

    chain_undos(undos)
}

fn has_port(state: &GameState, idx: i32) -> bool {
    state
        .structures
        .get(&idx)
        .and_then(|s| s.as_ref())
        .map(|s| s.structure_type == StructureType::Port)
        .unwrap_or(false)
}

fn has_road(state: &GameState, idx: i32) -> bool {
    state.tiles.get(&idx).map(|t| t.has_road).unwrap_or(false)
}

fn has_mycelium(state: &GameState, idx: i32) -> bool {
    state
        .structures
        .get(&idx)
        .and_then(|s| s.as_ref())
        .map(|s| s.structure_type == StructureType::Mycelium)
        .unwrap_or(false)
}

fn has_algae(state: &GameState, idx: i32) -> bool {
    state
        .tiles
        .get(&idx)
        .map(|t| t.has_effect(crate::types::TileEffect::Algae))
        .unwrap_or(false)
}

fn has_water_path(state: &GameState, src: i32, dest: i32) -> bool {
    let size = state.settings.size;
    let sx = src % size;
    let sy = src / size;

    // BFS path check
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    queue.push_back(src);
    visited.insert(src);

    while let Some(curr) = queue.pop_front() {
        if curr == dest {
            return true;
        }

        let dist = (curr % size - sx).abs().max((curr / size - sy).abs());
        if dist >= 4 && curr != src {
            continue;
        } // Max radius 4

        let adj = crate::functions::get_adjacent_indices(state, curr, 1);
        for n in adj {
            if visited.contains(&n) {
                continue;
            }
            let tile = match state.tiles.get(&n) {
                Some(t) => t,
                None => continue,
            };

            // Path must be water or ocean
            if is_water(tile.terrain_type) || n == dest {
                visited.insert(n);
                queue.push_back(n);
            }
        }
    }

    false
}

fn is_water(terrain: TerrainType) -> bool {
    matches!(terrain, TerrainType::Water | TerrainType::Ocean)
}

fn has_cymanti_path(state: &GameState, size: i32, src: i32, dest: i32) -> bool {
    let sx = src % size;
    let sy = src / size;

    // BFS path check
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    queue.push_back(src);
    visited.insert(src);

    while let Some(curr) = queue.pop_front() {
        if curr == dest {
            return true;
        }

        let dist = (curr % size - sx).abs().max((curr / size - sy).abs());
        if dist >= 4 && curr != src {
            continue;
        } // Max radius 4

        let adj = crate::functions::get_adjacent_indices(state, curr, 1);
        for n in adj {
            if visited.contains(&n) {
                continue;
            }
            let tile = match state.tiles.get(&n) {
                Some(t) => t,
                None => continue,
            };

            // Path must be any land terrain or algae on water
            let valid_terrain = match tile.terrain_type {
                TerrainType::Field
                | TerrainType::Forest
                | TerrainType::Mountain
                | TerrainType::Ice => true,
                _ => has_algae(state, n),
            };

            if valid_terrain || n == dest {
                visited.insert(n);
                queue.push_back(n);
            }
        }
    }

    false
}
