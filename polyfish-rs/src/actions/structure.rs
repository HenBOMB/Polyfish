//! Structure-related actions (create, destroy)

use crate::actions::city::add_population;
use crate::actions::{UndoCallback, chain_undos};
use crate::functions::get_city_owning_tile;
use crate::settings::structures::get_structure_setting;
use crate::states::{GameState, StructureState};
use crate::types::StructureType;

/// Crate-local RNG helper (Linear Congruential Generator)
/// Constants from MMIX by Knuth
fn next_rng(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed
}

/// Create a structure at a tile
pub fn create_structure(
    state: &mut GameState,
    idx: i32,
    structure_type: StructureType,
    level: i32,
) -> UndoCallback {
    let old_struct = state.structures.get(&idx).cloned();

    let structure = StructureState {
        structure_type,
        level,
        founded: state.settings.turn,
        tile_index: idx,
        score: 0,
    };

    state.structures.insert(idx, Some(structure));

    // Valid references for move closure
    let pov_id = state.settings.current_player_turn_id;
    let old_has_road = state.map.tiles.get(&idx).map(|t| t.has_road).unwrap_or(false);

    // Sync has_road
    if structure_type == StructureType::Road {
        if let Some(tile) = state.map.tiles.get_mut(&idx) {
            tile.has_road = true;
        }
    }

    let mut undos: Vec<UndoCallback> = Vec::new();

    // Update connections if Road or Port
    if structure_type == StructureType::Road || structure_type == StructureType::Port {
        undos.push(crate::actions::connection::update_capital_connections(
            state, pov_id,
        ));
    }

    // Award score for structure (e.g. Monuments giving 400)
    let settings = get_structure_setting(structure_type);
    if settings.reward_score > 0 {
        if let Some(tribe) = state.tribes.get_mut(&pov_id) {
            tribe.score += settings.reward_score;
            tribe.built_unique_improvements.insert(structure_type);
        }
        let score_gain = settings.reward_score;
        undos.push(Box::new(move |s| {
            if let Some(t) = s.tribes.get_mut(&pov_id) {
                t.score -= score_gain;
                t.built_unique_improvements.remove(&structure_type);
            }
        }));
    }

    undos.push(Box::new(move |s| {
        // Undo connection logic handled by its own closure in chain

        // Restore has_road
        if structure_type == StructureType::Road {
            if let Some(tile) = s.map.tiles.get_mut(&idx) {
                tile.has_road = old_has_road;
            }
        }

        if let Some(old) = old_struct {
            s.structures.insert(idx, old);
        } else {
            s.structures.shift_remove(&idx);
        }
    }));

    chain_undos(undos)
}

/// Destroy a structure at a tile
pub fn destroy_structure(state: &mut GameState, idx: i32) -> UndoCallback {
    let structure = match state.structures.get(&idx).cloned().flatten() {
        Some(s) => s,
        None => return Box::new(|_| {}),
    };

    // Remove structure
    state.structures.shift_remove(&idx);

    let mut undos: Vec<UndoCallback> = Vec::new();
    let pov_id = state.settings.current_player_turn_id;

    // Handle score reduction (e.g. Monuments)
    let settings = get_structure_setting(structure.structure_type);
    if settings.reward_score > 0 {
        if let Some(tribe) = state.tribes.get_mut(&pov_id) {
            tribe.score -= settings.reward_score;
        }
        let score_loss = settings.reward_score;
        undos.push(Box::new(move |s| {
            if let Some(t) = s.tribes.get_mut(&pov_id) {
                t.score += score_loss;
            }
        }));
    }

    // Restore structure on undo
    let undo_structure = structure.clone();
    undos.push(Box::new(move |s: &mut GameState| {
        s.structures.insert(idx, Some(undo_structure.clone()));
    }));

    // Handle population reduction for city structures
    if structure.structure_type != StructureType::Ruin {
        if let Some(city) = get_city_owning_tile(state, idx) {
            let city_tile_idx = city.tile_index;
            let settings = get_structure_setting(structure.structure_type);

            if settings.reward_pop > 0 {
                // Reduce population (negative add)
                undos.push(add_population(state, city_tile_idx, -settings.reward_pop));
            }
        }
    }

    // Valid stats for undo
    let pov_id = state.settings.current_player_turn_id;
    let old_has_road = state.map.tiles.get(&idx).map(|t| t.has_road).unwrap_or(false);

    // Sync has_road
    if structure.structure_type == StructureType::Road {
        if let Some(tile) = state.map.tiles.get_mut(&idx) {
            tile.has_road = false;
        }
    }

    // Update connections if Road or Port
    if structure.structure_type == StructureType::Road
        || structure.structure_type == StructureType::Port
    {
        undos.push(crate::actions::connection::update_capital_connections(
            state, pov_id,
        ));
    }

    undos.push(Box::new(move |s: &mut GameState| {
        // Restore has_road
        if structure.structure_type == StructureType::Road {
            if let Some(tile) = s.map.tiles.get_mut(&idx) {
                tile.has_road = old_has_road;
            }
        }
    }));

    chain_undos(undos)
}

/// Build a structure (spend stars, create, add pop)
pub fn build_structure(
    state: &mut GameState,
    idx: i32,
    structure_type: StructureType,
) -> UndoCallback {
    use crate::actions::spend_stars;
    use crate::settings::structures::get_structure_setting;

    let mut undos = Vec::new();
    let settings = get_structure_setting(structure_type);

    // 1. Spend stars
    if let Some(cost) = settings.cost {
        undos.push(spend_stars(state, cost));
    }

    // 2. Create structure
    undos.push(create_structure(state, idx, structure_type, 1));

    // 3. Add population
    if let Some(city) = get_city_owning_tile(state, idx) {
        let city_tile_idx = city.tile_index;
        let mut reward_pop = settings.reward_pop;

        // Handle adjacent multipliers (Windmill, Sawmill, Forge)
        if !settings.adjacent_types.is_empty() {
            use crate::functions::get_adjacent_indices;
            use crate::functions::get_structure_at;

            let adj = get_adjacent_indices(state, idx, 1);
            let adj_count = adj
                .iter()
                .filter(|&&adj_idx| {
                    if let Some(s) = get_structure_at(state, adj_idx) {
                        settings.adjacent_types.contains(&s.structure_type)
                    } else {
                        false
                    }
                })
                .count() as i32;
            reward_pop *= adj_count;
        }

        if reward_pop > 0 {
            undos.push(add_population(state, city_tile_idx, reward_pop));
        }
    }

    chain_undos(undos)
}
/// Capture a ruin and gain rewards
pub fn capture_ruin(state: &mut GameState, tile_idx: i32) -> UndoCallback {
    use crate::actions::discovery::discover_tiles;
    use crate::actions::tech::unlock_tech;
    use crate::actions::{UndoCallback, gain_stars};
    use crate::functions::{get_adjacent_indices, get_capital_city};
    use crate::types::TechnologyType;

    let pov_id = state.settings.current_player_turn_id;
    let mut undos: Vec<UndoCallback> = Vec::new();

    // 0. Capture Seed & Setup RNG
    let original_seed = state.settings.seed;
    let mut current_seed = original_seed;

    // Helper to restore seed
    undos.push(Box::new(move |s: &mut GameState| {
        s.settings.seed = original_seed;
    }));

    // Destroy ruins
    undos.push(destroy_structure(state, tile_idx));

    let mut possible_rewards: Vec<Box<dyn FnOnce(&mut GameState) -> UndoCallback>> = Vec::new();

    // 1. Stars: 10 stars
    possible_rewards.push(Box::new(|s: &mut GameState| {
        if s.settings.verbose {
            s._messages.push("Ruin reward: 10 Stars! ⭐".to_string());
        }
        gain_stars(s, 10)
    }));

    // 2. Tech: random unlockable
    if let Some(tribe) = state.tribes.get(&pov_id) {
        let mut unlockable_cand = Vec::new();
        for t_val in 1..=24 {
            // Standard tech tree range (excluding 11-Start)
            if t_val == 11 {
                continue;
            }
            let t_type: TechnologyType = unsafe { std::mem::transmute(t_val as i8) };
            if !crate::settings::technology::has_technology(&tribe.tech_vanilla, t_type) {
                unlockable_cand.push(t_type);
            }
        }
        if !unlockable_cand.is_empty() {
            // Pre-calculate which tech to unlock using current seed state
            // Note: We need to use 'next_rng' here locally if we want to pick *now*.
            // But we only want to pick IF this reward is chosen.
            // Actually, for "Bias Fairness", we should pick the tech *now* (speculatively)
            // or we use the seed *after* we pick this reward category?
            // To be purely seed based: We use the seed state AT THE MOMENT of execution.

            // Let's defer execution but capture the *seed*? No, state.seed changes.
            // Correct approach:
            // The "Ruin Reward Choice" event is one RNG step.
            // The "Which Tech" event is a sub-step.

            // So it's okay to advance seed "conditionally" deep inside.

            // Let's PRE-SELECT the tech candidate deterministically *outside*, so the closure captures it.
            // We use 'next_rng' to pick it.
            // Wait, if we use 'next_rng' here, we advance state.seed even if we DON'T pick Tech reward?
            // That would mean "Ruin contents" affect global RNG even if not picked?
            // Actually, we usually want:
            // 1. RNG pick Reward Category.
            // 2. If Category == Tech, RNG pick Tech.

            // So:
            possible_rewards.push(Box::new(move |s: &mut GameState| {
                // We utilize the state seed (which has been updated by the Reward Category pick)
                // to pick the tech.
                // We need to mutate the seed inside here too.

                let mut seed = s.settings.seed;
                let r = next_rng(&mut seed);
                s.settings.seed = seed; // Commit update

                let index = (r as usize) % unlockable_cand.len();
                let picked = unlockable_cand[index];

                if s.settings.verbose {
                    s._messages
                        .push(format!("Ruin reward: Discovered {:?}! 💡", picked));
                }
                unlock_tech(s, picked, true).unwrap_or_else(|_| Box::new(|_| {}))
            }));
        }
    }

    // 3. Pop growth: 3 to capital
    if let Some(cap) = get_capital_city(state, pov_id) {
        let cap_tile_idx = cap.tile_index;
        possible_rewards.push(Box::new(move |s: &mut GameState| {
            if s.settings.verbose {
                s._messages
                    .push("Ruin reward: Population growth! 👨‍👩‍👧‍👦".to_string());
            }
            crate::actions::city::add_population(s, cap_tile_idx, 3)
        }));
    }

    // 4. Explorer: if nearby is fog
    let mut fog_nearby = false;
    let around = get_adjacent_indices(state, tile_idx, 2);
    for &idx in &around {
        let is_explored = state
            .map.tiles
            .get(&idx)
            .map(|t| t.explorers.contains(&pov_id))
            .unwrap_or(false);
        if !is_explored {
            fog_nearby = true;
            break;
        }
    }
    if fog_nearby {
        possible_rewards.push(Box::new(move |s: &mut GameState| {
            if s.settings.verbose {
                s._messages.push("Ruin reward: Explorer! 🧭".to_string());
            }
            let (_, revealed) = crate::actions::discovery::predict_explorer(s, tile_idx);
            discover_tiles(s, pov_id, None, Some(revealed))
        }));
    }

    // 5. Veteran Unit (Swordsman or Mantis for Cymanti)
    let unit_reward_type = if let Some(t) = state.tribes.get(&pov_id) {
        if t.tribe_type == crate::types::TribeType::Cymanti {
            crate::types::UnitType::Mantis
        } else {
            crate::types::UnitType::Swordsman
        }
    } else {
        crate::types::UnitType::Swordsman
    };

    possible_rewards.push(Box::new(move |s: &mut GameState| {
        let mut undos = Vec::new();
        if s.settings.verbose {
            s._messages
                .push(format!("Ruin reward: Veteran {:?}! ⚔️", unit_reward_type));
        }
        // Spawn unit
        let res = crate::actions::units::summon_unit(s, unit_reward_type, tile_idx, false, false);
        match res {
            Ok(r) => undos.push(r.undo),
            Err(e) => {
                if s.settings.verbose {
                    s._messages.push(format!("Failed to spawn veteran: {}", e));
                }
            }
        }

        // Make veteran: find the unit we just spawned (it will be at tile_idx)
        if let Some(u) = crate::functions::get_unit_at_mut(s, tile_idx) {
            let old_veteran = u.veteran;
            let old_health = u.health;

            u.veteran = true;
            u.health = crate::functions::get_unit_max_health(u);

            undos.push(Box::new(move |st| {
                if let Some(u) = crate::functions::get_unit_at_mut(st, tile_idx) {
                    u.veteran = old_veteran;
                    u.health = old_health;
                }
            }));
        }
        crate::actions::chain_undos(undos)
    }));

    // Pick one reward using Seed RNG
    if !possible_rewards.is_empty() {
        let r = next_rng(&mut current_seed);
        let index = (r as usize) % possible_rewards.len();

        let reward_fn = possible_rewards.remove(index);

        // Execute reward
        // NOTE: we do NOT update state.settings.seed here yet, because
        // passing 'current_seed' locally tracked the change.
        // We must sync the state seed before executing the reward,
        // in case the reward itself (e.g. Tech) needs the updated seed.
        state.settings.seed = current_seed;

        undos.push(reward_fn(state));
    }

    chain_undos(undos)
}
