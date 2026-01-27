//! City-related actions (population, territory)

use crate::actions::{chain_undos, UndoCallback};
use crate::coords::Coords;
use crate::functions::get_pov_tribe;
use crate::moves::Move;
use crate::states::{CityState, GameState};
use crate::types::MoveType;
use crate::types::RewardType;

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
    let (city_idx, mut old_city) = {
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
