//! City-related actions (population, territory)

use crate::actions::{chain_undos, UndoCallback};
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
            .find(|(_, c)| c.tile_index == city_tile_idx)
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

            // Check for level up
            let next = city.level + 1;
            if city.progress >= next {
                city.level += 1;
                city.progress -= next;
                city.production += 1; // Level up bonus

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

                tribe.score += amount_score;

                undos.push(Box::new(move |s: &mut GameState| {
                    if let Some(t) = s.tribes.get_mut(&pov_id) {
                        t.score -= amount_score;
                    }
                }));

                // TODO: Generate rewards (Workshop, etc.) - this usually happens via Moves
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

    // Update score
    let score_gain = 20 * tiles_to_claim.len() as i32;
    if let Some(tribe) = state.tribes.get_mut(&pov_id) {
        tribe.score += score_gain;
    }

    Box::new(move |s| {
        // Undo score
        if let Some(tribe) = s.tribes.get_mut(&pov_id) {
            tribe.score -= score_gain;
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
    use crate::functions::get_adjacent_indices;
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
            if let Some(pos) = old_tribe
                .cities
                .iter()
                .position(|c| c.tile_index == tile_idx)
            {
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

            // Remove units belonging to this city from ALL tribes (Disband rule)
            let mut all_units_to_remove: Vec<(i32, usize)> = Vec::new();
            for (&tribe_id, tribe) in &state.tribes {
                for (i, unit) in tribe.units.iter().enumerate() {
                    if let Some(home) = &unit.home_coords {
                        if home.idx == tile_idx {
                            all_units_to_remove.push((tribe_id, i));
                        }
                    }
                }
            }

            // Remove starting from highest index per tribe to prevent index shifting issues
            all_units_to_remove.sort_by(|a, b| b.1.cmp(&a.1));
            for (t_id, u_idx) in all_units_to_remove {
                undos.push(crate::actions::units::remove_unit(
                    state,
                    t_id,
                    u_idx,
                    Some(pov_id),
                    None,
                ));
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
                    if let Some(p) = nt
                        .cities
                        .iter()
                        .position(|c| c.tile_index == c_clone.tile_index)
                    {
                        nt.cities.remove(p);
                    }
                }
                // Restore tiles
                if let Some(tile) = s.tiles.get_mut(&c_clone.tile_index) {
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
            undos.push(claim_territory(state, &city._territory, tile_idx, false));
        }
    } else {
        // Case 2: Capture Neutral Village (New City)
        let territory = get_adjacent_indices(state, tile_idx, 1);
        let tribe_type = state
            .tribes
            .get(&pov_id)
            .map(|t| t.tribe_type)
            .unwrap_or(TribeType::None);

        let created_city = CityState {
            id: tile_idx,
            name: format!("{:?} City", tribe_type),
            population: 0,
            progress: 0,
            border_size: 1,
            connected_to_capital: false,
            level: 1,
            production: 1,
            owner: pov_id,
            tile_index: tile_idx,
            rewards: std::collections::HashSet::new(),
            _territory: territory.clone(),
            _riot: false,
            _walls: false,
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
    }

    Ok(chain_undos(undos))
}
