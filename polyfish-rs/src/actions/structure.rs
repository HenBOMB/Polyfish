//! Structure-related actions (create, destroy)
use crate::actions::city::add_population;
use crate::actions::discovery::discover_tiles;
use crate::actions::tech::unlock_tech;
use crate::actions::{UndoCallback, chain_undos, gain_stars};
use crate::functions::{get_adjacent_indices, get_best_capital, get_city_owning_tile};
use crate::settings::structures::get_structure_setting;
use crate::states::{GameState, StructureState};
use crate::types::{
    CityRewardType, RuinsRewardType, StructureType, TechnologyType, TerrainType, TribeType,
    UnitType,
};

/// Helper to generate the next seed using xxHash32, matching the game's deterministic RNG
fn next_rng_xxhash(current_seed: &mut i64, initial_seed: i64) -> i64 {
    let next = crate::hash::get_hash(initial_seed as u32, &[*current_seed as i32]);
    *current_seed = next as i64;
    *current_seed
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
    };

    if structure_type != StructureType::Road {
        state.structures.insert(idx, Some(structure));
    }

    // Valid references for move closure
    let pov_id = state.settings.current_player_turn_id;
    let old_has_road = state.tiles.get(&idx).map(|t| t.has_road).unwrap_or(false);

    // Sync has_road and Algae effect
    if structure_type == StructureType::Road {
        if let Some(tile) = state.tiles.get_mut(&idx) {
            tile.has_road = true;
        }
    } else if structure_type == StructureType::Algae {
        if let Some(tile) = state.tiles.get_mut(&idx) {
            tile.effects.insert(crate::TileEffect::Algae);
        }
    }

    let mut undos: Vec<UndoCallback> = Vec::new();

    // Update connections if Road or Port
    if structure_type == StructureType::Road
        || structure_type == StructureType::Port
        || structure_type == StructureType::Bridge
    {
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

        // Restore has_road and Algae effect
        if structure_type == StructureType::Road {
            if let Some(tile) = s.tiles.get_mut(&idx) {
                tile.has_road = old_has_road;
            }
        } else if structure_type == StructureType::Algae {
            if let Some(tile) = s.tiles.get_mut(&idx) {
                tile.effects.remove(&crate::TileEffect::Algae);
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
            let city_tile_idx = city.idx;
            let settings = get_structure_setting(structure.structure_type);

            if settings.reward_pop > 0 {
                // Refund what the structure actually paid, not its base yield:
                // an adjacency hub was paid `reward_pop x partners` on build, so
                // removing a level-4 Sawmill must remove 4 pop, not 1.
                let pop = if settings.adjacent_types.is_empty() {
                    settings.reward_pop
                } else {
                    let owner = state.tiles.get(&idx).map_or(0, |t| t.owner);
                    settings.reward_pop
                        * crate::rules::economy::partner_count_with(
                            state,
                            idx,
                            &settings.adjacent_types,
                            owner,
                        )
                };
                if pop > 0 {
                    undos.push(add_population(state, city_tile_idx, -pop));
                }
            }
        }
    }

    // Valid stats for undo
    let pov_id = state.settings.current_player_turn_id;
    let old_has_road = state.tiles.get(&idx).map(|t| t.has_road).unwrap_or(false);

    // Sync has_road and Algae effect
    if structure.structure_type == StructureType::Road {
        if let Some(tile) = state.tiles.get_mut(&idx) {
            tile.has_road = false;
        }
    } else if structure.structure_type == StructureType::Algae {
        if let Some(tile) = state.tiles.get_mut(&idx) {
            tile.effects.remove(&crate::TileEffect::Algae);
        }
    }

    // Update connections if Road or Port
    if structure.structure_type == StructureType::Road
        || structure.structure_type == StructureType::Port
        || structure.structure_type == StructureType::Bridge
    {
        undos.push(crate::actions::connection::update_capital_connections(
            state, pov_id,
        ));
    }

    undos.push(Box::new(move |s: &mut GameState| {
        // Restore has_road and Algae effect
        if structure.structure_type == StructureType::Road {
            if let Some(tile) = s.tiles.get_mut(&idx) {
                tile.has_road = old_has_road;
            }
        } else if structure.structure_type == StructureType::Algae {
            if let Some(tile) = s.tiles.get_mut(&idx) {
                tile.effects.insert(crate::TileEffect::Algae);
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
    let pov_id = state.settings.current_player_turn_id;

    // Adjacency yield: one population per FRIENDLY adjacent partner. Uses the
    // shared rule so pop payout, Market income, legality and every AI consumer
    // read one definition of "partner". A hub's level IS this count.
    let partners = if settings.adjacent_types.is_empty() {
        None
    } else {
        Some(crate::rules::economy::partner_count_with(
            state,
            idx,
            &settings.adjacent_types,
            pov_id,
        ))
    };

    // 1. Spend stars
    if let Some(cost) = settings.cost {
        undos.push(spend_stars(state, cost));
    }

    // 2. Create structure
    undos.push(create_structure(
        state,
        idx,
        structure_type,
        partners.unwrap_or(1),
    ));

    // 3. Add population
    if let Some(city) = get_city_owning_tile(state, idx) {
        let city_tile_idx = city.idx;
        let reward_pop = settings.reward_pop * partners.unwrap_or(1);

        if reward_pop > 0 {
            undos.push(add_population(state, city_tile_idx, reward_pop));
        }
    }

    // Retroactive adjacency pop (real-game rule): a new partner structure
    // (e.g. Farm) also feeds already-built neighbors that list its type in
    // adjacent_types (Windmill/Sawmill/Forge), paying their owning city.
    // add_population only reaches the acting player's cities, so this pays
    // pov-owned neighbors only (cross-border feeding stays unmodeled).
    // The neighbor's stored level tracks the same partner count, so it is bumped
    // here too — nothing in the engine reads it (income recomputes on read), but
    // the UI selects hub sprites by level.
    {
        use crate::functions::{get_adjacent_indices, get_structure_at};
        for adj_idx in get_adjacent_indices(state, idx, 1) {
            let feeds = get_structure_at(state, adj_idx).map(|s| s.structure_type);
            if let Some(neighbor_type) = feeds {
                let neighbor = get_structure_setting(neighbor_type);
                if !neighbor.adjacent_types.contains(&structure_type) {
                    continue;
                }
                if neighbor.reward_pop > 0 {
                    let owned_city_idx = get_city_owning_tile(state, adj_idx)
                        .filter(|c| c.owner == pov_id)
                        .map(|c| c.idx);
                    if let Some(city_tile_idx) = owned_city_idx {
                        undos.push(add_population(state, city_tile_idx, neighbor.reward_pop));
                    }
                }
                if state.tiles.get(&adj_idx).is_some_and(|t| t.owner == pov_id) {
                    if let Some(st) = state.structures.get_mut(&adj_idx).and_then(|s| s.as_mut()) {
                        let old_level = st.level;
                        st.level = old_level + 1;
                        undos.push(Box::new(move |s: &mut GameState| {
                            if let Some(st) = s.structures.get_mut(&adj_idx).and_then(|x| x.as_mut())
                            {
                                st.level = old_level;
                            }
                        }));
                    }
                }
            }
        }
    }

    chain_undos(undos)
}
/// Capture a ruin and gain rewards
pub fn capture_ruin(
    state: &mut GameState,
    tile_idx: i32,
    reward_hint: Option<RuinsRewardType>,
    revealed_tiles_hint: Option<Vec<i32>>,
    tech_hint: Option<TechnologyType>,
) -> UndoCallback {
    let pov_id = state.settings.current_player_turn_id;
    let mut undos: Vec<UndoCallback> = Vec::new();

    // 0. Setup RNG
    let original_seed = state.settings.seed;
    let mut current_seed = original_seed;

    undos.push(Box::new(move |s: &mut GameState| {
        s.settings.seed = original_seed;
    }));

    // Destroy ruins
    undos.push(destroy_structure(state, tile_idx));

    // If a specific reward was requested (from replay), execute it immediately
    if let Some(hint) = reward_hint {
        let ruin_tile = state.tiles.get(&tile_idx).unwrap();
        let is_water = ruin_tile.is_water_terrain();

        match hint {
            RuinsRewardType::Resources => {
                if state.settings._verbose {
                    state
                        ._messages
                        .push("Hint: Ruin reward: 10 Stars! ⭐".to_string());
                }
                undos.push(gain_stars(state, 10));
            }
            RuinsRewardType::FreeTech => {
                if let Some(tribe) = state.tribes.get(&pov_id) {
                    let researchable = crate::settings::technology::get_researchable_techs(
                        &tribe.tech_vanilla,
                        tribe.tribe_type,
                    );
                    if !researchable.is_empty() {
                        let picked = if let Some(tech) = tech_hint {
                            tech
                        } else {
                            let mut seed = state.settings.seed;
                            let r = next_rng_xxhash(&mut seed, state.initial_seed);
                            state.settings.seed = seed;
                            let index = (r as usize) % researchable.len();
                            researchable[index]
                        };

                        if state.settings._verbose {
                            state
                                ._messages
                                .push(format!("Hint: Ruin reward: Discovered {:?}! 💡", picked));
                        }
                        if let Ok(u) = unlock_tech(state, picked, true) {
                            undos.push(u);
                        }
                    } else {
                        // Fallback to stars if nothing to research
                        undos.push(gain_stars(state, 10));
                    }
                }
            }
            RuinsRewardType::PopGrowth => {
                let cap_idx = get_best_capital(state, pov_id).map(|c| c.idx);
                if let Some(idx) = cap_idx {
                    if state.settings._verbose {
                        state
                            ._messages
                            .push("Hint: Ruin reward: Population growth! 👨‍👩‍👧‍👦".to_string());
                    }
                    undos.push(crate::actions::city::add_population(state, idx, 3));
                }
            }
            RuinsRewardType::Explorer => {
                if state.settings._verbose {
                    state
                        ._messages
                        .push("Hint: Ruin reward: Explorer! 🧭".to_string());
                }
                let revealed = if let Some(tiles) = revealed_tiles_hint.clone() {
                    tiles
                } else {
                    let (_, r) = crate::actions::discovery::predict_explorer(state, tile_idx);
                    r
                };
                undos.push(discover_tiles(state, pov_id, None, Some(revealed)));
            }
            RuinsRewardType::Swordsman => {
                if is_water {
                    // Veteran Rammer
                    if let Ok(res) = crate::actions::units::summon_unit(
                        state,
                        UnitType::Rammership,
                        tile_idx,
                        false,
                        false,
                    ) {
                        undos.push(res.undo);
                        if let Some(u) = crate::functions::get_unit_at_mut(state, tile_idx) {
                            u.veteran = true;
                            u.health = crate::functions::get_unit_max_health(u);
                            u.passenger_type = Some(UnitType::Warrior);
                        }
                    }
                } else {
                    let unit_type = if let Some(t) = state.tribes.get(&pov_id) {
                        if t.tribe_type == TribeType::Cymanti {
                            UnitType::Mantis
                        } else {
                            UnitType::Swordsman
                        }
                    } else {
                        UnitType::Swordsman
                    };
                    if let Ok(res) =
                        crate::actions::units::summon_unit(state, unit_type, tile_idx, false, false)
                    {
                        undos.push(res.undo);
                        if let Some(u) = crate::functions::get_unit_at_mut(state, tile_idx) {
                            u.veteran = true;
                            u.health = crate::functions::get_unit_max_health(u);
                        }
                    }
                }
            }
            RuinsRewardType::City => {
                // Aquarion Lost City
                let territory =
                    crate::functions::get_square_indices(tile_idx, 1, state.settings.size);
                let city = crate::states::CityState {
                    idx: tile_idx,
                    name: "Lost City".to_string(),
                    population: 0,
                    progress: 0,
                    border_size: 1,
                    connected_to_capital: false,
                    level: 3,
                    production: 3,
                    owner: pov_id,
                    // A level-3 city has two reward slots consumed
                    // (`required = level - 1`); granting only CityWall left the
                    // counter short, so slot 2 was offered a second time.
                    rewards: vec![CityRewardType::Workshop, CityRewardType::CityWall],
                    _territory: territory.clone(),
                };
                if let Some(tribe) = state.tribes.get_mut(&pov_id) {
                    // Derived, not a literal: population is 0 at founding, so
                    // `get_city_score` is exactly base + per-level here.
                    let city_score = crate::score::get_city_score(&city);
                    tribe.score += city_score;
                    tribe.cities.push(city.clone());

                    undos.push(Box::new(move |st: &mut GameState| {
                        if let Some(t) = st.tribes.get_mut(&pov_id) {
                            t.score -= city_score;
                            t.cities.pop();
                        }
                    }));
                }
                if let Some(tile) = state.tiles.get_mut(&tile_idx) {
                    let old_owner = tile.owner;
                    tile.owner = pov_id;
                    undos.push(Box::new(move |st: &mut GameState| {
                        if let Some(t) = st.tiles.get_mut(&tile_idx) {
                            t.owner = old_owner;
                        }
                    }));
                }
                undos.push(crate::actions::city::claim_territory(
                    state, &territory, tile_idx, true,
                ));
                undos.push(crate::actions::connection::update_capital_connections(
                    state, pov_id,
                ));
            }
            _ => {}
        }

        return chain_undos(undos);
    }

    // --- Dynamic/RNG logic if no hint provided ---
    // Type definition to help compiler unify closures
    type RuinRewardFn = Box<dyn FnOnce(&mut GameState) -> UndoCallback>;
    let mut possible_rewards: Vec<RuinRewardFn> = Vec::new();

    // 1. Resources (10 Stars) - Always available
    possible_rewards.push(Box::new(|s: &mut GameState| -> UndoCallback {
        if s.settings._verbose {
            s._messages.push("Ruin reward: 10 Stars! ⭐".to_string());
        }
        gain_stars(s, 10)
    }) as RuinRewardFn);

    // 2. Scrolls of Wisdom (Free Tech) - Only if researchable
    if let Some(tribe) = state.tribes.get(&pov_id) {
        let researchable = crate::settings::technology::get_researchable_techs(
            &tribe.tech_vanilla,
            tribe.tribe_type,
        );

        if !researchable.is_empty() {
            possible_rewards.push(Box::new(move |s: &mut GameState| -> UndoCallback {
                let mut seed = s.settings.seed;
                let r = next_rng_xxhash(&mut seed, s.initial_seed);
                s.settings.seed = seed;

                let index = (r as usize) % researchable.len();
                let picked = researchable[index];

                if s.settings._verbose {
                    s._messages
                        .push(format!("Ruin reward: Discovered {:?}! 💡", picked));
                }
                unlock_tech(s, picked, true).unwrap_or_else(|_| Box::new(|_| {}))
            }) as RuinRewardFn);
        }
    }

    // 3. Population (3 Population to capital) - Only if owns a capital
    if let Some(cap) = get_best_capital(state, pov_id) {
        let cap_tile_idx = cap.idx;
        possible_rewards.push(Box::new(move |s: &mut GameState| -> UndoCallback {
            if s.settings._verbose {
                s._messages
                    .push("Ruin reward: Population growth! 👨‍👩‍👧‍👦".to_string());
            }
            crate::actions::city::add_population(s, cap_tile_idx, 3)
        }) as RuinRewardFn);
    }

    // 4. Explorer (Free Explorer) - If nearby is fog AND not a Mountain
    let ruin_tile = state.tiles.get(&tile_idx).unwrap();
    let is_mountain = ruin_tile.terrain_type == TerrainType::Mountain;
    let is_water = ruin_tile.is_water_terrain();

    let mut can_discovery = !is_mountain;

    // Cymanti cannot get Explorer on Water
    if let Some(t) = state.tribes.get(&pov_id) {
        if t.tribe_type == TribeType::Cymanti && is_water {
            can_discovery = false;
        }
    }

    if can_discovery {
        let mut fog_nearby = false;
        let around = get_adjacent_indices(state, tile_idx, 2); // 5x5
        for &idx in &around {
            if !crate::functions::is_tile_explored(state, idx, pov_id) {
                fog_nearby = true;
                break;
            }
        }
        if fog_nearby {
            let tiles_hint = revealed_tiles_hint.clone();
            possible_rewards.push(Box::new(move |s: &mut GameState| -> UndoCallback {
                if s.settings._verbose {
                    s._messages.push("Ruin reward: Explorer! 🧭".to_string());
                }
                let revealed = if let Some(tiles) = tiles_hint {
                    tiles
                } else {
                    let (_, r) = crate::actions::discovery::predict_explorer(s, tile_idx);
                    r
                };
                discover_tiles(s, pov_id, None, Some(revealed))
            }) as RuinRewardFn);
        }
    }

    // 5. Units (Veteran Swordsman or Rammer)
    if is_water {
        // Veteran Rammer (carrying Warrior)
        possible_rewards.push(Box::new(move |s: &mut GameState| -> UndoCallback {
            let mut undos = Vec::new();
            if s.settings._verbose {
                s._messages
                    .push("Ruin reward: Veteran Rammer (Warrior)! ⛵".to_string());
            }

            match crate::actions::units::summon_unit(
                s,
                UnitType::Rammership,
                tile_idx,
                false,
                false,
            ) {
                Ok(res) => {
                    undos.push(res.undo);
                    // Make veteran and add passenger
                    if let Some(u) = crate::functions::get_unit_at_mut(s, tile_idx) {
                        let old_vet = u.veteran;
                        let old_hp = u.health;
                        let old_pass = u.passenger_type;

                        u.veteran = true;
                        u.health = crate::functions::get_unit_max_health(u);
                        u.passenger_type = Some(UnitType::Warrior);

                        undos.push(Box::new(move |st: &mut GameState| {
                            if let Some(u) = crate::functions::get_unit_at_mut(st, tile_idx) {
                                u.veteran = old_vet;
                                u.health = old_hp;
                                u.passenger_type = old_pass;
                            }
                        }) as UndoCallback);
                    }
                }
                Err(_) => {}
            }
            chain_undos(undos)
        }) as RuinRewardFn);
    } else {
        // Veteran Swordsman
        let unit_type = if let Some(t) = state.tribes.get(&pov_id) {
            if t.tribe_type == TribeType::Cymanti {
                UnitType::Mantis
            } else {
                UnitType::Swordsman
            }
        } else {
            UnitType::Swordsman
        };

        possible_rewards.push(Box::new(move |s: &mut GameState| -> UndoCallback {
            let mut undos = Vec::new();
            if s.settings._verbose {
                s._messages
                    .push(format!("Ruin reward: Veteran {:?}! ⚔️", unit_type));
            }
            match crate::actions::units::summon_unit(s, unit_type, tile_idx, false, false) {
                Ok(res) => {
                    undos.push(res.undo);
                    if let Some(u) = crate::functions::get_unit_at_mut(s, tile_idx) {
                        let old_vet = u.veteran;
                        let old_hp = u.health;

                        u.veteran = true;
                        u.health = crate::functions::get_unit_max_health(u);

                        undos.push(Box::new(move |st: &mut GameState| {
                            if let Some(u) = crate::functions::get_unit_at_mut(st, tile_idx) {
                                u.veteran = old_vet;
                                u.health = old_hp;
                            }
                        }) as UndoCallback);
                    }
                }
                Err(_) => {}
            }
            chain_undos(undos)
        }) as RuinRewardFn);
    }

    // 6. Lost City (Aquarion on Ocean)
    let is_aquarion = state
        .tribes
        .get(&pov_id)
        .map_or(false, |t| t.tribe_type == TribeType::Aquarion);
    let is_ocean = ruin_tile.terrain_type == TerrainType::Ocean;

    if is_aquarion && is_ocean {
        possible_rewards.push(Box::new(move |s: &mut GameState| -> UndoCallback {
            let mut undos = Vec::new();
            if s.settings._verbose {
                s._messages.push("Ruin reward: LOST CITY! 🏛️".to_string());
            }

            // Spawn level 3 city
            // 1. Create territory
            let territory = crate::functions::get_square_indices(tile_idx, 1, s.settings.size);

            // 2. Resolve name
            let city_name = "Lost City".to_string();

            let city = crate::states::CityState {
                idx: tile_idx,
                name: city_name,
                population: 0,
                progress: 0,
                border_size: 1,
                connected_to_capital: false,
                level: 3,
                production: 3,
                owner: pov_id,
                // Both slots a level-3 city has already spent — see above.
                rewards: vec![CityRewardType::Workshop, CityRewardType::CityWall],
                _territory: territory.clone(),
            };

            // 3. Add to tribe
            if let Some(tribe) = s.tribes.get_mut(&pov_id) {
                tribe.cities.push(city.clone());
                undos.push(Box::new(move |st: &mut GameState| {
                    if let Some(t) = st.tribes.get_mut(&pov_id) {
                        t.cities.pop();
                    }
                }) as UndoCallback);
            }

            // 4. Update tile
            if let Some(tile) = s.tiles.get_mut(&tile_idx) {
                let old_owner = tile.owner;
                tile.owner = pov_id;
                undos.push(Box::new(move |st: &mut GameState| {
                    if let Some(t) = st.tiles.get_mut(&tile_idx) {
                        t.owner = old_owner;
                    }
                }) as UndoCallback);
            }

            // 5. 4 adjacent shallow water tiles
            let adj = crate::functions::get_adjacent_indices(s, tile_idx, 1);
            let mut water_changed = 0;
            for n_idx in adj {
                if water_changed >= 4 {
                    break;
                }
                if let Some(tile) = s.tiles.get_mut(&n_idx) {
                    if tile.terrain_type == TerrainType::Ocean {
                        tile.terrain_type = TerrainType::Water; // Shallow
                        undos.push(Box::new(move |st: &mut GameState| {
                            if let Some(t) = st.tiles.get_mut(&n_idx) {
                                t.terrain_type = TerrainType::Ocean;
                            }
                        }) as UndoCallback);
                        water_changed += 1;
                    }
                }
            }

            // 6. Claim territory
            undos.push(crate::actions::city::claim_territory(
                s, &territory, tile_idx, true,
            ));

            // 7. Refresh connections
            undos.push(crate::actions::connection::update_capital_connections(
                s, pov_id,
            ));

            chain_undos(undos)
        }) as RuinRewardFn);
    }

    // Pick one reward using Seed RNG
    if !possible_rewards.is_empty() {
        let r = next_rng_xxhash(&mut current_seed, state.initial_seed);
        let index = (r as usize) % possible_rewards.len();

        let reward_fn = possible_rewards.remove(index);

        state.settings.seed = current_seed;
        undos.push(reward_fn(state));
    }

    chain_undos(undos)
}
