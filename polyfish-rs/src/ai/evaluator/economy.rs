use crate::functions::{get_adjacent_indices, get_structure_at};
use crate::states::{GameState, PlayerId};
use crate::types::{ModeType, ResourceType, StructureType, TechnologyType};

/// Evaluates the economy score for a given player.
/// Returns a score roughly between 0.0 and 1.0.
pub fn evaluate_economy(state: &GameState, player_id: PlayerId) -> f32 {
    let tribe_opt = state.tribes.get(&player_id);
    if tribe_opt.is_none() {
        return 0.0;
    }
    let tribe = tribe_opt.unwrap();

    // --- 1. Economy Score (0.0 - 1.0) ---
    // SPT (Stars Per Turn) - Soft cap around 25
    // Note: city.production already includes the base 1 star per city
    let mut income = 0.0;
    for city in &tribe.cities {
        income += city.production as f32;
    }
    let spt_score = (income / crate::states::default_max_spt() as f32).clamp(0.0, 1.0);

    // Stars (Liquid Cash) - Soft cap around 40
    let stars_score =
        (tribe.stars as f32 / crate::states::default_max_stars() as f32).clamp(0.0, 1.0);

    // Tech - Sum net utility of all owned techs
    let mut total_tech_utility = 0.0;
    for tech_state in &tribe.tech_vanilla {
        if tech_state.discovered {
            total_tech_utility += crate::ai::evaluator::research::evaluate_tech_utility(
                state,
                player_id,
                tech_state.tech_type,
            );
        }
    }

    // Normalization:
    // If a player has ~10 useful techs, score should be around 0.5.
    // Max utility per tech is roughly 5-10, but average is lower.
    // Let's use 20.0 as a denominator for "good" tech progress.
    let tech_score = (total_tech_utility / 20.0).clamp(-1.0, 1.0);

    // --- 2. Bonuses ---
    let capital_bonus = bonus_capital_connections(tribe);

    // --- 3. Penalties ---
    let partial_city_penalty = penalty_partial_cities(tribe, state.settings.mode);
    let unused_tech_penalty = penalty_unused_tech(state, tribe);
    let bad_struct_penalty = penalty_bad_structures(state, tribe);
    let low_stars_penalty = penalty_low_stars(tribe);
    let urgency_penalty = penalty_urgency(state);
    // Weighted Eco
    // Income is most important, then tech/stars
    // Weight income=50% stars=20% tech=30%
    // let raw = (spt_score * 0.5) + (stars_score * 0.2) + (tech_score * 0.3);
    let base_score = (spt_score * 0.6) + (stars_score * 0.1) + (tech_score * 0.3);

    let score = base_score + capital_bonus
        - partial_city_penalty
        - unused_tech_penalty
        - bad_struct_penalty
        - low_stars_penalty
        - (urgency_penalty * 0.1);
    score.clamp(0.0, 1.0)
}

/// Bonus for cities connected to the capital.
/// Each connected city earns a small bonus, encouraging road networks.
/// Max bonus: ~0.10 (capped to avoid dominating the score).
fn bonus_capital_connections(tribe: &crate::states::TribeState) -> f32 {
    if tribe.cities.len() <= 1 {
        return 0.0; // No bonus needed with 0-1 cities
    }
    let connected = tribe
        .cities
        .iter()
        .filter(|c| c.connected_to_capital)
        .count() as f32;
    let total = tribe.cities.len() as f32;
    // Ratio of connected cities (excluding the capital itself which is always "connected")
    let ratio = connected / total;
    // Scale: fully connected = 0.10 bonus
    (ratio * 0.10).min(0.10)
}

/// Penalty/bonus for cities with population progress.
/// - **Domination**: Penalizes stuck population (investment not yet converted to a level-up).
///   Encourages completing level-ups quickly rather than spreading pop thinly.
/// - **Perfection**: Rewards population progress (each pop point is progress toward score).
///   Encourages adding population even if the level-up isn't immediate.
fn penalty_partial_cities(tribe: &crate::states::TribeState, mode: ModeType) -> f32 {
    if tribe.cities.is_empty() {
        return 0.0;
    }

    match mode {
        ModeType::Perfection => {
            // In Perfection, reward population progress (negative penalty = bonus)
            let mut total_bonus = 0.0;
            for city in &tribe.cities {
                if city.population > 0 {
                    let needed = city.level + 1;
                    let progress = (city.population as f32 / needed as f32).clamp(0.0, 0.99);
                    // Small bonus per city with progress
                    total_bonus += progress * 0.03;
                }
            }
            // Return negative (acts as bonus when subtracted)
            -(total_bonus.min(0.10))
        }
        ModeType::Domination => {
            // In Domination, penalize stuck population
            let mut total_penalty = 0.0;
            for city in &tribe.cities {
                if city.population > 0 {
                    let needed = city.level + 1;
                    let stuck_ratio = (city.population as f32 / needed as f32).clamp(0.0, 0.99);
                    total_penalty += stuck_ratio * 0.05;
                }
            }
            total_penalty.min(0.15)
        }
        ModeType::Glory
        | ModeType::Custom
        | ModeType::Might
        | ModeType::Sandbox
        | ModeType::Tutorial
        | ModeType::None => {
            todo!(
                "penalty_partial_cities not implemented for Glory/Custom/Might/Sandbox/Diplomacy/Tutorial mode"
            );
        }
    }
}

/// Penalty for techs that have been discovered but the player
/// hasn't used the resources/structures they unlock.
/// "You bought the tech — USE it."
///
/// Covers:
/// - Resource techs: Organization→Fruit, Hunting→Game, Fishing→Fish, Farming→Crop, Mining→Metal
/// - Structure techs: Forestry→LumberHut, Sailing→Port, FreeSpirit→Temple, etc.
/// - Adjacency techs: Mathematics→Sawmill, Smithery→Forge, Construction→Windmill, Trade→Market
fn penalty_unused_tech(state: &GameState, tribe: &crate::states::TribeState) -> f32 {
    use crate::settings::technology::has_technology;

    let mut penalty = 0.0;

    // --- 1. Resource techs: penalize unharvested resources in territory ---
    let resource_techs: &[(TechnologyType, ResourceType)] = &[
        (TechnologyType::Organization, ResourceType::Fruit),
        (TechnologyType::Hunting, ResourceType::Game),
        (TechnologyType::Fishing, ResourceType::Fish),
        (TechnologyType::Farming, ResourceType::Crop),
        (TechnologyType::Mining, ResourceType::Metal),
    ];

    for &(tech, res_type) in resource_techs {
        if !has_technology(&tribe.tech_vanilla, tech) {
            continue;
        }
        let mut unharvested = 0;
        for city in &tribe.cities {
            for &tile_idx in &city._territory {
                if let Some(Some(res)) = state.resources.get(&tile_idx) {
                    if res.resource_type == res_type {
                        unharvested += 1;
                    }
                }
            }
        }
        if unharvested > 0 {
            penalty += (unharvested as f32).min(4.0) * 0.01;
        }
    }

    // --- 2. Structure techs: penalize having the tech but no structures built ---
    // For basic structures: check if buildable terrain exists but structure is absent
    // Harsher penalty if the terrain doesn't even exist in territory (total waste)
    let structure_techs: &[(TechnologyType, StructureType, &[crate::types::TerrainType])] = &[
        (
            TechnologyType::Forestry,
            StructureType::LumberHut,
            &[crate::types::TerrainType::Forest],
        ),
        (
            TechnologyType::Sailing,
            StructureType::Port,
            &[crate::types::TerrainType::Water],
        ),
        (
            TechnologyType::FreeSpirit,
            StructureType::Temple,
            &[crate::types::TerrainType::Field],
        ),
    ];

    for &(tech, struct_type, terrains) in structure_techs {
        if !has_technology(&tribe.tech_vanilla, tech) {
            continue;
        }

        // Check if we have any of this structure built
        let has_structure = tribe.cities.iter().any(|city| {
            city._territory.iter().any(|&idx| {
                if let Some(s) = crate::functions::get_structure_at(state, idx) {
                    s.structure_type == struct_type
                } else {
                    false
                }
            })
        });

        if has_structure {
            continue; // Already using the tech
        }

        // Does the terrain even exist in our territory?
        let has_any_terrain = tribe.cities.iter().any(|city| {
            city._territory.iter().any(|&idx| {
                state.tiles
                    .get(&idx)
                    .map_or(false, |t| terrains.contains(&t.terrain_type))
            })
        });

        if !has_any_terrain {
            penalty += 0.03; // Total waste — terrain doesn't even exist
        } else {
            penalty += 0.02; // Have the terrain but haven't built
        }
    }

    // --- 3. Terrain-less tech chains: tech requires terrain that doesn't exist at all ---
    // Harsher penalty because the entire tech branch is pointless
    use crate::types::TerrainType;
    let terrain_techs: &[(TechnologyType, &[TerrainType])] = &[
        (TechnologyType::Climbing, &[TerrainType::Mountain]),
        (TechnologyType::Mining, &[TerrainType::Mountain]),
        (TechnologyType::Smithery, &[TerrainType::Mountain]),
        (TechnologyType::Meditation, &[TerrainType::Mountain]),
        (TechnologyType::Philosophy, &[TerrainType::Mountain]),
        (TechnologyType::Fishing, &[TerrainType::Water]),
        (
            TechnologyType::Ramming,
            &[TerrainType::Water, TerrainType::Ocean],
        ),
        (
            TechnologyType::Aquatism,
            &[TerrainType::Water, TerrainType::Ocean],
        ),
        (TechnologyType::Hunting, &[TerrainType::Forest]),
        (TechnologyType::Archery, &[TerrainType::Forest]),
        (TechnologyType::Spiritualism, &[TerrainType::Forest]),
    ];

    for &(tech, terrains) in terrain_techs {
        if !has_technology(&tribe.tech_vanilla, tech) {
            continue;
        }

        let has_terrain = tribe.cities.iter().any(|city| {
            city._territory.iter().any(|&idx| {
                state.tiles
                    .get(&idx)
                    .map_or(false, |t| terrains.contains(&t.terrain_type))
            })
        });

        if !has_terrain {
            penalty += 0.025; // Tech chain with no matching terrain — wasted stars
        }
    }

    // --- 4. Adjacency structures: penalize if prerequisite exists but adjacency isn't built ---
    // Sawmill needs adjacent LumberHut, Forge needs Mine, Windmill needs Farm, Market needs Sawmill/Forge/Windmill
    let adjacency_techs: &[(TechnologyType, StructureType, &[StructureType])] = &[
        (
            TechnologyType::Mathematics,
            StructureType::Sawmill,
            &[StructureType::LumberHut],
        ),
        (
            TechnologyType::Smithery,
            StructureType::Forge,
            &[StructureType::Mine],
        ),
        (
            TechnologyType::Construction,
            StructureType::Windmill,
            &[StructureType::Farm],
        ),
        (
            TechnologyType::Trade,
            StructureType::Market,
            &[
                StructureType::Sawmill,
                StructureType::Windmill,
                StructureType::Forge,
            ],
        ),
    ];

    for &(tech, struct_type, prereqs) in adjacency_techs {
        if !has_technology(&tribe.tech_vanilla, tech) {
            continue;
        }

        // Check if the adjacency structure exists already
        let has_adjacency = tribe.cities.iter().any(|city| {
            city._territory.iter().any(|&idx| {
                if let Some(s) = crate::functions::get_structure_at(state, idx) {
                    s.structure_type == struct_type
                } else {
                    false
                }
            })
        });

        if has_adjacency {
            continue;
        }

        // Check if any prerequisite structure exists (meaning we COULD build the adjacency)
        let has_prereq = tribe.cities.iter().any(|city| {
            city._territory.iter().any(|&idx| {
                if let Some(s) = crate::functions::get_structure_at(state, idx) {
                    prereqs.contains(&s.structure_type)
                } else {
                    false
                }
            })
        });

        if has_prereq {
            penalty += 0.015; // Have the tech and prereqs, but haven't built the upgrade
        }
    }

    // Cap total penalty
    penalty.min(0.15)
}

/// Penalty for poorly placed structures.
/// Detects:
/// 1. Lonely adjacency structures (Sawmill/Forge/Windmill/Market with only 1 adjacent prereq)
/// 2. Dead-end roads (roads not connecting to at least 2 roads/cities/ports)
fn penalty_bad_structures(state: &GameState, tribe: &crate::states::TribeState) -> f32 {
    let player_id = tribe.id;
    let mut penalty: f32 = 0.0;

    // Adjacency structures and their prerequisites
    let adjacency_pairs: &[(StructureType, &[StructureType])] = &[
        (StructureType::Sawmill, &[StructureType::LumberHut]),
        (StructureType::Forge, &[StructureType::Mine]),
        (StructureType::Windmill, &[StructureType::Farm]),
        (
            StructureType::Market,
            &[
                StructureType::Sawmill,
                StructureType::Windmill,
                StructureType::Forge,
            ],
        ),
    ];

    for city in &tribe.cities {
        for &tile_idx in &city._territory {
            let structure = match get_structure_at(state, tile_idx) {
                Some(s) => s,
                None => continue,
            };

            // --- 1. Lonely adjacency structures ---
            for &(adj_type, prereqs) in adjacency_pairs {
                if structure.structure_type != adj_type {
                    continue;
                }

                let adj = get_adjacent_indices(state, tile_idx, 1);
                let adj_count = adj
                    .iter()
                    .filter(|&&idx| {
                        if let Some(tile) = state.tiles.get(&idx) {
                            if tile.owner == player_id {
                                if let Some(s) = get_structure_at(state, idx) {
                                    return prereqs.contains(&s.structure_type);
                                }
                            }
                        }
                        false
                    })
                    .count();

                if adj_count <= 1 {
                    penalty += 0.02; // Built alone — wasted potential
                } else if adj_count >= 3 {
                    penalty -= 0.02; // Excellent placement — reward clustering
                }
            }

            // --- 2. Crowded Prerequisite Structures ---
            // Penalize LumberHut/Farm/Mine if they have NO access to a Hub or a Potential Hub (Empty Land Tile).
            let (hub_type, is_prereq) = match structure.structure_type {
                StructureType::LumberHut => (Some(StructureType::Sawmill), true),
                StructureType::Farm => (Some(StructureType::Windmill), true),
                StructureType::Mine => (Some(StructureType::Forge), true),
                _ => (None, false),
            };

            if is_prereq {
                if let Some(hub) = hub_type {
                    let adj = get_adjacent_indices(state, tile_idx, 1);

                    let has_access = adj.iter().any(|&idx| {
                        // 1. Is it an existing Hub?
                        if let Some(s) = get_structure_at(state, idx) {
                            if s.structure_type == hub {
                                return true;
                            }
                        }

                        // 2. Is it a valid empty spot (Potential Hub)?
                        if let Some(tile) = state.tiles.get(&idx) {
                            // Must be land (not water/ocean where we can't build hubs)
                            // Note: WaterTemple/Port are different.
                            if tile.terrain_type == crate::types::TerrainType::Ocean
                                || tile.terrain_type == crate::types::TerrainType::Water
                            {
                                return false;
                            }
                            // Must be empty
                            return get_structure_at(state, idx).is_none();
                        }
                        false
                    });

                    if !has_access {
                        penalty += 0.02; // Crowded / Wasted potential
                    }
                }
            }

            // --- 3. Dead-end roads ---
            if let Some(tile) = state.tiles.get(&tile_idx) {
                if tile.has_road {
                    let adj = get_adjacent_indices(state, tile_idx, 1);
                    let connections = adj
                        .iter()
                        .filter(|&&idx| {
                            if let Some(adj_tile) = state.tiles.get(&idx) {
                                if adj_tile.owner != player_id {
                                    return false;
                                }
                                // Connected to another road
                                if adj_tile.has_road {
                                    return true;
                                }
                                // Connected to a city or port (road endpoint)
                                if let Some(s) = get_structure_at(state, idx) {
                                    return matches!(
                                        s.structure_type,
                                        StructureType::Village | StructureType::Port
                                    );
                                }
                            }
                            false
                        })
                        .count();

                    if connections < 2 {
                        penalty += 0.01; // Dead-end road — connects to nowhere
                    }
                }
            }
        }
    }

    penalty.clamp(-0.06, 0.12)
}

/// Penalty for having low stars (danger zone).
/// Encourages keeping a buffer for emergencies.
fn penalty_low_stars(tribe: &crate::states::TribeState) -> f32 {
    let stars = tribe.stars as f32;
    // Safety buffer: we want at least ~8 stars for emergencies
    let threshold = 8.0;

    if stars >= threshold {
        return 0.0;
    }

    // Quadratic penalty for being under threshold
    // 0 stars -> (1.0 - 0.0)^2 = 1.0
    // 4 stars -> (1.0 - 0.5)^2 = 0.25
    let deficit = 1.0 - (stars / threshold);

    // Max penalty at 0 stars is 0.25 (significant chunk of the 0.0-1.0 range)
    deficit.powi(2) * 0.25
}

/// Penalizes for playing moves, this induces urgency to play and not waste moves
fn penalty_urgency(state: &GameState) -> f32 {
    let mut score = state.settings._recent_moves.len() as f32;
    let current_turn = state.settings.turn;

    if current_turn > 20 {
        score = score / 15.0;
    } else if current_turn > 15 {
        score = score / 10.0;
    } else if current_turn > 10 {
        score = score / 8.0;
    } else if current_turn > 5 {
        score = score / 6.0;
    } else {
        score = score / 4.0;
    }

    score.min(1.0)
}
