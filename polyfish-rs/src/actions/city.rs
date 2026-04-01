//! City-related actions (population, territory)

use crate::actions::discovery::discover_tiles;
use crate::actions::{UndoCallback, chain_undos};
use crate::coords::Coords;
use crate::states::GameState;

/// Add population to a city
///
/// This handles:
/// - Adding population
/// - Leveling up if threshold reached
/// - Generating rewards (stubbed for now)
/// - Updating score
pub fn add_population(state: &mut GameState, city_tile_idx: i32, amount: i32) -> UndoCallback {
    let pov_id = state.settings.current_player_turn_id;

    // Find the city
    let (city_idx, old_city) = {
        let tribe = match state.tribes.get(&pov_id) {
            Some(t) => t,
            None => return Box::new(|_| {}),
        };
        match tribe
            .cities
            .iter()
            .enumerate()
            .find(|(_, c)| c.idx == city_tile_idx)
        {
            Some((idx, c)) => (idx, c.clone()),
            None => return Box::new(|_| {}),
        }
    };

    let mut undos: Vec<UndoCallback> = Vec::new();

    // Update city
    if let Some(tribe) = state.tribes.get_mut(&pov_id) {
        if let Some(city) = tribe.cities.get_mut(city_idx) {
            city.population += amount;
            city.progress += amount;

            // Check for level up (loop for consecutive level-ups)
            // A large pop gain (e.g. connecting 3 cities at once) can cross multiple thresholds
            let old_struct_level =
                state.structures.get(&city_tile_idx).and_then(|s| s.as_ref()).map(|s| s.level);
            let mut total_score = 0;
            let mut levels_gained = 0;

            while city.progress >= city.level + 1 {
                let next = city.level + 1;
                city.level += 1;
                city.progress -= next;
                city.production += 1; // Level up bonus
                levels_gained += 1;

                // Update structure level if it exists
                if let Some(Some(struct_state)) = state.structures.get_mut(&city_tile_idx) {
                    struct_state.level += 1;
                }

                // Score is ONLY awarded on level up!
                // Formula: (level > 1 ? 50 - (level * 5) : 0) + amount * 5
                let amount_score = (if city.level > 1 {
                    50 - (city.level * 5)
                } else {
                    0
                }) + amount * 5;

                total_score += amount_score;
            }

            if levels_gained > 0 {
                tribe.score += total_score;

                undos.push(Box::new(move |s: &mut GameState| {
                    if let Some(t) = s.tribes.get_mut(&pov_id) {
                        t.score -= total_score;
                    }
                    // Restore structure level
                    if let Some(old_lvl) = old_struct_level {
                        if let Some(Some(st)) = s.structures.get_mut(&city_tile_idx) {
                            st.level = old_lvl;
                        }
                    }
                }));
            }
            // No level up = NO score awarded (matching TS behavior)
        }
    }

    // Restore city state on undo
    undos.push(Box::new(move |s: &mut GameState| {
        if let Some(tribe) = s.tribes.get_mut(&pov_id) {
            if let Some(city) = tribe.cities.get_mut(city_idx) {
                *city = old_city;
            }
        }
    }));

    chain_undos(undos)
}

/// Claim territory for a city
pub fn claim_territory(
    state: &mut GameState,
    territory: &[i32],
    city_tile_idx: i32,
    force: bool,
) -> UndoCallback {
    let pov_id = state.settings.current_player_turn_id;
    let city_coords = Coords::from_index(city_tile_idx, state.settings.size);

    // Filter territory if needed (only unowned)
    let tiles_to_claim: Vec<i32> = if !force {
        territory
            .iter()
            .cloned()
            .filter(|&idx| state.tiles.get(&idx).map(|t| t.owner == 0).unwrap_or(false))
            .collect()
    } else {
        territory.to_vec()
    };

    if tiles_to_claim.is_empty() {
        return Box::new(|_| {});
    }

    // Claim tiles
    let mut old_owners: Vec<(i32, i32, Option<Coords>)> = Vec::with_capacity(tiles_to_claim.len());

    for &idx in &tiles_to_claim {
        if let Some(tile) = state.tiles.get_mut(&idx) {
            old_owners.push((idx, tile.owner, tile.ruling_city_coords.clone()));
            tile.owner = pov_id;
            tile.ruling_city_coords = Some(city_coords.clone());
        }
    }

    // Update city's own territory list
    let mut added_indices = Vec::new();
    let mut city_idx_in_tribe = None;
    if let Some(tribe) = state.tribes.get_mut(&pov_id) {
        if let Some(pos) = tribe.cities.iter().position(|c| c.idx == city_tile_idx) {
            let city = &mut tribe.cities[pos];
            for &idx in &tiles_to_claim {
                if !city._territory.contains(&idx) {
                    city._territory.push(idx);
                    added_indices.push(idx);
                }
            }
            city_idx_in_tribe = Some(pos);
        }
    }

    // Update score
    let score_gain = 20 * tiles_to_claim.len() as i32;
    if let Some(tribe) = state.tribes.get_mut(&pov_id) {
        tribe.score += score_gain;
    }

    let num_newly_added = added_indices.len();
    Box::new(move |s| {
        // Undo score
        if let Some(tribe) = s.tribes.get_mut(&pov_id) {
            tribe.score -= score_gain;
        }

        // Restore city's own territory list
        if let Some(tribe_idx) = city_idx_in_tribe {
            if let Some(tribe) = s.tribes.get_mut(&pov_id) {
                if let Some(city) = tribe.cities.get_mut(tribe_idx) {
                    // Remove the last N indices that were added
                    let new_len = city._territory.len().saturating_sub(num_newly_added);
                    city._territory.truncate(new_len);
                }
            }
        }

        // Restore tiles
        for (idx, owner, ruling_coords) in old_owners {
            if let Some(tile) = s.tiles.get_mut(&idx) {
                tile.owner = owner;
                tile.ruling_city_coords = ruling_coords;
            }
        }
    })
}

/// Capture a city or village
pub fn capture_city(state: &mut GameState, tile_idx: i32) -> Result<UndoCallback, String> {
    use crate::actions::units::remove_unit;
    use crate::functions::get_square_indices;
    use crate::states::CityState;
    use crate::types::TribeType;

    let pov_id = state.settings.current_player_turn_id;
    let mut undos: Vec<UndoCallback> = Vec::new();
    let tile_owner = state.tiles.get(&tile_idx).map(|t| t.owner).unwrap_or(0);

    // Case 1: Capture Enemy City
    if tile_owner > 0 && tile_owner != pov_id {
        let mut old_city: Option<CityState> = None;
        let mut old_city_idx: Option<usize> = None;

        // Remove from old tribe
        if let Some(old_tribe) = state.tribes.get_mut(&tile_owner) {
            if let Some(pos) = old_tribe.cities.iter().position(|c| c.idx == tile_idx) {
                old_city_idx = Some(pos);
                old_city = Some(old_tribe.cities.remove(pos));
            }
        }

        if let Some(mut city) = old_city {
            let city_name_old = city.name.clone();
            let old_capital_val = state
                .tiles
                .get(&tile_idx)
                .map(|t| t.capital_of)
                .unwrap_or(0);

            // Update city ownership and name
            city.owner = pov_id;
            let tribe_type = state
                .tribes
                .get(&pov_id)
                .map(|t| t.tribe_type)
                .unwrap_or(TribeType::None);

            city.name = format!(
                "{:?} {}",
                tribe_type,
                if old_capital_val > 0 {
                    "Capital"
                } else {
                    "City"
                }
            );

            // Update tile
            if let Some(tile) = state.tiles.get_mut(&tile_idx) {
                tile.owner = pov_id;
                if tile.capital_of > 0 {
                    tile.capital_of = pov_id;
                }
            }

            // Add to new owner
            if let Some(new_tribe) = state.tribes.get_mut(&pov_id) {
                new_tribe.cities.push(city.clone());
            }

            // Tribe elimination check
            let is_eliminated = state
                .tribes
                .get(&tile_owner)
                .map(|t| t.cities.is_empty())
                .unwrap_or(false);

            if is_eliminated {
                if let Some(old_tribe) = state.tribes.get_mut(&tile_owner) {
                    old_tribe.killed_turn = state.settings.turn;
                    old_tribe.killer_id = pov_id;

                    // Remove all units
                    let unit_count = old_tribe.units.len();
                    for i in (0..unit_count).rev() {
                        undos.push(remove_unit(state, tile_owner, i, Some(pov_id), None));
                    }
                }
            }

            // Undo logic
            let c_clone = city.clone();
            undos.push(Box::new(move |s| {
                // Restore tribe status
                if let Some(ot) = s.tribes.get_mut(&tile_owner) {
                    if ot.killer_id == pov_id && ot.killed_turn == s.settings.turn {
                        ot.killer_id = 0;
                        ot.killed_turn = 0;
                    }
                }
                // Remove from new
                if let Some(nt) = s.tribes.get_mut(&pov_id) {
                    if let Some(p) = nt.cities.iter().position(|c| c.idx == c_clone.idx) {
                        // println!(
                        //     "🐛 CaptureUndo: Removing city at {} from tribe {}",
                        //     c_clone.idx, pov_id
                        // );
                        nt.cities.remove(p);
                    } else {
                        eprintln!(
                            "🐛 CaptureUndo: FAILED to find city at {} in tribe {}",
                            c_clone.idx, pov_id
                        );
                    }
                }
                // Restore tiles
                if let Some(tile) = s.tiles.get_mut(&c_clone.idx) {
                    tile.owner = tile_owner;
                    if old_capital_val > 0 {
                        tile.capital_of = tile_owner;
                    }
                }
                // Add back to old
                if let Some(ot) = s.tribes.get_mut(&tile_owner) {
                    let mut restored = c_clone.clone();
                    restored.owner = tile_owner;
                    restored.name = city_name_old;
                    if let Some(pos) = old_city_idx {
                        ot.cities.insert(pos, restored);
                    } else {
                        ot.cities.push(restored);
                    }
                }
            }));

            // Claim territory
            undos.push(claim_territory(state, &city._territory, tile_idx, true));
            undos.push(discover_tiles(
                state,
                pov_id,
                None,
                Some(city._territory.clone()),
            ));

            // Refresh connections for BOTH tribes
            undos.push(crate::actions::connection::update_capital_connections(
                state, pov_id,
            ));
            undos.push(crate::actions::connection::update_capital_connections(
                state, tile_owner,
            ));
        }
    } else {
        // Case 2: Capture Neutral Village (New City)
        let territory = get_square_indices(tile_idx, 1, state.settings.size);
        let tribe_type = state
            .tribes
            .get(&pov_id)
            .map(|t| t.tribe_type)
            .unwrap_or(TribeType::None);

        let created_city = CityState {
            idx: tile_idx,
            name: format!("{:?} City", tribe_type),
            population: 0,
            progress: 0,
            border_size: 1,
            connected_to_capital: false,
            level: 1,
            production: 1,
            owner: pov_id,
            rewards: Vec::new(),
            _territory: territory.clone(),
        };

        if let Some(tribe) = state.tribes.get_mut(&pov_id) {
            tribe.cities.push(created_city.clone());
        }

        undos.push(Box::new(move |s| {
            if let Some(tribe) = s.tribes.get_mut(&pov_id) {
                tribe.cities.pop();
            }
        }));

        undos.push(claim_territory(state, &territory, tile_idx, false));
        undos.push(discover_tiles(state, pov_id, None, Some(territory)));

        // Refresh connection for capturer
        undos.push(crate::actions::connection::update_capital_connections(
            state, pov_id,
        ));
    }

    Ok(chain_undos(undos))
}
