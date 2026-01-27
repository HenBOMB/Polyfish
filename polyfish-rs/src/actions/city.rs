//! City-related actions (population, territory)

use crate::actions::{chain_undos, UndoCallback};
use crate::coords::Coords;
use crate::functions::get_pov_tribe;
use crate::moves::Move;
use crate::types::MoveType;
use crate::states::{CityState, GameState};
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
        match tribe.cities.iter().enumerate().find(|(_, c)| c.tile_index == city_tile_idx) {
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
            
            // Check for level up/down
            if amount > 0 {
                let next_level = city.level + 1;
                if city.progress >= next_level {
                    let overflow = city.progress - next_level;
                    city.level += 1;
                    city.progress = overflow;
                    city.production += 1; 
                    
                    tribe.score += 100;
                    undos.push(Box::new(move |s: &mut GameState| {
                        if let Some(t) = s.tribes.get_mut(&pov_id) { t.score -= 100; }
                    }));
                }
            } else if amount < 0 {
                if city.progress < 0 {
                    city.level -= 1;
                    city.progress += city.level + 1; // Simplify: back to previous level progress state
                    city.production -= 1;
                    
                    tribe.score -= 100;
                    undos.push(Box::new(move |s: &mut GameState| {
                        if let Some(t) = s.tribes.get_mut(&pov_id) { t.score += 100; }
                    }));
                }
            }
            
            // Score increase for population: 5 points per pop
            let pop_score = amount * 5;
            tribe.score += pop_score;
            undos.push(Box::new(move |s: &mut GameState| {
                if let Some(t) = s.tribes.get_mut(&pov_id) {
                    t.score -= pop_score;
                }
            }));
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
pub fn claim_territory(state: &mut GameState, territory: &[i32], city_tile_idx: i32, force: bool) -> UndoCallback {
    let pov_id = state.settings.current_player_turn_id;
    let city_coords = Coords::from_index(city_tile_idx, state.settings.size);
    
    // Filter territory if needed (only unowned)
    let tiles_to_claim: Vec<i32> = if !force {
        territory.iter()
            .cloned()
            .filter(|&idx| {
                state.tiles.get(&idx).map(|t| t.owner == 0).unwrap_or(false)
            })
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
