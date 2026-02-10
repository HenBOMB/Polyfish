use crate::coords::Coords;
use crate::functions::{calculate_combat_preview, get_adjacent_indices, get_structure_at};
use crate::game::Game;
use crate::moves::Move;
use crate::types::{AbilityType, ModeType, MoveType, RewardType, StructureType};

/// Score a move based on heuristics for move ordering
pub fn score_move(game: &Game, mv: &dyn Move) -> f32 {
    let state = &game.state;
    let move_type = mv.move_type();

    match move_type {
        MoveType::Reward => score_reward(state, mv),

        MoveType::Capture => {
            if let Ok(idx) = mv.source_idx() {
                if let Some(s) = get_structure_at(state, idx as i32) {
                    match s.structure_type {
                        StructureType::Ruin => 100.0,
                        StructureType::Village => 99.8,
                        _ => 100.1, // Capital or City
                    }
                } else {
                    // Likely starfish
                    80.0
                }
            } else {
                80.0
            }
        }

        MoveType::Attack => {
            if let (Ok(src), Ok(target)) = (mv.source_idx(), mv.target_idx()) {
                if let Some(preview) = calculate_combat_preview(state, src as i32, target as i32) {
                    if preview.defender_dies {
                        45.0
                    } else if preview.attacker_dies {
                        1.0 // Suicide is very low priority
                    } else if preview.damage_to_defender > 5 {
                        25.0
                    } else {
                        15.0
                    }
                } else {
                    // Infiltration or other special attack
                    15.0
                }
            } else {
                15.0
            }
        }

        MoveType::Ability => {
            if let Ok(ability) = mv.ability_type() {
                match ability {
                    AbilityType::Promote => 35.0,

                    AbilityType::Explode
                    | AbilityType::Boost
                    | AbilityType::FreezeArea
                    | AbilityType::Convert => 20.0,

                    AbilityType::Recover => {
                        if let Ok(unit_idx) = mv.source_idx() {
                            if let Some(unit) = state
                                .tribes
                                .get(&state.settings.current_player_turn_id)
                                .and_then(|t| {
                                    t.units.iter().find(|u| u.coords.idx == unit_idx as i32)
                                })
                            {
                                let max_hp = crate::functions::get_unit_max_health(unit) as f32;
                                let current_hp = unit.health as f32;
                                let hp_pct = current_hp / max_hp;

                                if hp_pct < 0.4 {
                                    40.0 // Critical heal
                                } else if hp_pct < 0.9 {
                                    // Safe to heal?
                                    // If in city or mountain (defense bonus), encourages healing
                                    let def_bonus =
                                        crate::functions::get_defense_bonus(state, unit);
                                    if def_bonus > 1.0 { 30.0 } else { 20.0 }
                                } else {
                                    5.0 // Waste of a turn if almost full
                                }
                            } else {
                                20.0
                            }
                        } else {
                            20.0
                        }
                    }

                    AbilityType::Disband => -50.0, // Generally avoid unless desperate

                    AbilityType::BurnForest => 5.0, // Low priority unless contextual

                    AbilityType::Destroy => -10.0, // Risky

                    AbilityType::ClearForest => {
                        let stars = state
                            .tribes
                            .get(&state.settings.current_player_turn_id)
                            .map(|t| t.stars)
                            .unwrap_or(0);
                        if stars < 5 {
                            25.0 // Need cash
                        } else {
                            15.0 // Solid eco move
                        }
                    }

                    _ => 10.0,
                }
            } else {
                10.0
            }
        }

        MoveType::Summon => {
            if state.settings.turn < 2 {
                -10.0 // Deprioritize Turn 1 summons to favor development/ending turn
            } else {
                25.0 // Prioritize Turn 2+ summons for expansion
            }
        }

        MoveType::Build | MoveType::Harvest => {
            let mut score = 22.0;

            if let Ok(s_type) = mv.structure_type() {
                // Temple timing bonus in Perfection mode:
                // Temples level up once per turn. To reach max Level 5 by Turn 30,
                // they must be built by Turn 25 at latest. Building by Turn 19 is ideal.
                if state.settings.mode == ModeType::Perfection {
                    match s_type {
                        StructureType::Temple
                        | StructureType::ForestTemple
                        | StructureType::MountainTemple
                        | StructureType::WaterTemple => {
                            let turn = state.settings.turn;
                            if turn <= 19 {
                                score += 15.0 + (19 - turn) as f32;
                            } else if turn <= 25 {
                                score += 8.0;
                            }
                        }
                        _ => {}
                    }
                }

                // Adjacency structure scoring:
                // Sawmill/Forge/Windmill/Market gain value from adjacent prerequisite structures.
                // More adjacencies = more pop/stars, so prioritize clustered placement.
                if let Ok(target) = mv.target_idx() {
                    let player_id = state.settings.current_player_turn_id;
                    let adj = get_adjacent_indices(state, target as i32, 1);

                    let prereqs: &[StructureType] = match s_type {
                        StructureType::Sawmill => &[StructureType::LumberHut],
                        StructureType::Forge => &[StructureType::Mine],
                        StructureType::Windmill => &[StructureType::Farm],
                        StructureType::Market => &[
                            StructureType::Sawmill,
                            StructureType::Windmill,
                            StructureType::Forge,
                        ],
                        _ => &[],
                    };

                    let mut adj_count = 0;

                    if !prereqs.is_empty() {
                        adj_count = adj
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

                        match adj_count {
                            0 => {}             // Shouldn't happen (build gen checks adjacency), but safe
                            1 => score -= 2.0,  // Lonely — discourage building alone
                            2 => score += 5.0,  // Good cluster
                            3 => score += 12.0, // Great cluster
                            _ => score += 18.0, // Exceptional — 4+ adjacencies
                        }
                    }

                    // Road scoring: prioritize roads that connect cities
                    if s_type == StructureType::Road {
                        score += score_road(state, target as i32);
                    }

                    // Population efficiency scoring:
                    // Penalize moves that add population but don't result in a city level up.
                    // This encourages "finishing" cities rather than spreading population thin.
                    let mut pop_gain = 0;
                    match s_type {
                        // Harvests (usually handled via Harvest move type, but StructureType might be set?)
                        // The Move struct has variants. For Build, we check s_type.
                        // For Harvest, we check resource type?
                        // Let's check MoveType::Harvest explicitly below or handle it here if merged.
                        // Actually, Harvest moves often don't have a structure_type set in the move?
                        // Wait, `mv.structure_type()` returns result.
                        
                        StructureType::LumberHut | StructureType::Forge // Forge gives pop per adjacent mine? No, Forge gives pop based on adjacency.
                        | StructureType::Sawmill | StructureType::Windmill | StructureType::Market => {
                            // These give population based on adjacency.
                            // We already calculated `adj_count` above for clustering.
                            // Base pop is usually 0 plus adjacency bonus?
                            // Standard: 
                            // Lumber Hut: +1
                            // Sawmill: 0 base, +1 per adj Lumber Hut
                            // Forge: 0 base, +2 per adj Mine
                            // Windmill: 0 base, +1 per adj Farm
                            // Market: 0 base, +1 per adj Sawmill/Windmill/Forge
                            // Let's approximate.
                            match s_type {
                                StructureType::LumberHut => pop_gain = 1,
                                StructureType::Sawmill => pop_gain = 1 * adj_count as i32, // approx
                                StructureType::Forge => pop_gain = 2 * adj_count as i32,
                                StructureType::Windmill => pop_gain = 1 * adj_count as i32,
                                StructureType::Market => pop_gain = 1 * adj_count as i32,
                                _ => {}
                            }
                        }
                        StructureType::Farm | StructureType::Mine | StructureType::Port => {
                            pop_gain = 2; // Most tier 2 resources/structures give 2 pop
                        }
                        _ => {}
                    }

                    // If this is a Harvest move, we need to check resource type for pop gain
                    if move_type == MoveType::Harvest {
                        // Harvest fruits/fish/animals = +1
                        // We don't have resource type easily accessible here without parsing move or looking at tile 
                        // But typically simple harvest is +1.
                        // Crop harvest (Construction) is +2? No, organization/fishing/hunting is +1.
                        // Farming (Crop) is usually +2? In standard polytopia, yes.
                        // Let's assume +1 minimum.
                        if pop_gain == 0 { pop_gain = 1; }
                    }

                    if pop_gain > 0 {
                        // Find the city this tile belongs to
                        if let Some(tribe) = state.tribes.get(&state.settings.current_player_turn_id) {
                            if let Some(city) = tribe.cities.iter().find(|c| c._territory.contains(&(target as i32))) {
                                let needed = city.level + 1; 
                                let current = city.population;
                                
                                if current + pop_gain < needed {
                                    // Won't level up
                                    score -= 4.0; 
                                } else {
                                    // Will level up! Bonus!
                                    score += 5.0;
                                }
                            }
                        }
                    }
                }
            } // End of if let Ok(target)

            score
        }

        MoveType::Step => {
            let mut score = 50.0;
            if let Ok(target_idx) = mv.target_idx() {
                let player_id = state.settings.current_player_turn_id;

                // Prioritize stepping onto capture targets
                if let Some(tile) = state.tiles.get(&(target_idx as i32)) {
                    if let Some(s) = get_structure_at(state, target_idx as i32) {
                        match s.structure_type {
                            StructureType::Ruin
                            | StructureType::Village
                            | StructureType::Lighthouse => score += 40.0,
                            _ => {
                                // Enemy city potentially
                                // TODO: Doesnt check for peace treaty
                                if tile.owner != player_id && tile.owner != 0 {
                                    score += 45.0;
                                }
                            }
                        }
                    }

                    // Prioritize exploration (stepping into/near fog)
                    let adj = crate::functions::get_adjacent_indices(state, target_idx as i32, 1);
                    for n_idx in adj {
                        if let Some(tile) = state.tiles.get(&n_idx) {
                            if !tile.explorers.contains(&player_id) {
                                score += 2.0; // Cumulative for each fog tile revealed
                            }
                        }
                    }
                }
            }
            score
        }

        MoveType::Research => {
            if let Ok(tech) = mv.tech_type() {
                let player_id = state.settings.current_player_turn_id;
                let utility =
                    crate::ai::evaluator::research::evaluate_tech_utility(state, player_id, tech);

                // Map utility into 8-18 range:
                //   utility of -2 or less -> 8.0 (low priority but still above default 5.0)
                //   utility of +8 or more -> 18.0 (above build/harvest at 22 is too aggressive)
                let base = 8.0 + (utility.clamp(-2.0, 8.0) + 2.0);

                // "Buy before capture" bonus:
                // If any of our units are sitting on an uncaptured village,
                // tech cost will go up after we capture. Research now!
                let has_village_opportunity = if let Some(tribe) = state.tribes.get(&player_id) {
                    tribe.units.iter().any(|unit| {
                        let idx = unit.coords.idx;
                        if let Some(tile) = state.tiles.get(&idx) {
                            if tile.owner != player_id {
                                if let Some(s) = get_structure_at(state, idx) {
                                    return s.structure_type == StructureType::Village;
                                }
                            }
                        }
                        false
                    })
                } else {
                    false
                };

                if has_village_opportunity {
                    base + 5.0 // Push above build/harvest (22.0) to encourage buying first
                } else {
                    base
                }
            } else {
                5.0
            }
        }
        
        MoveType::EndTurn => 0.0,

        _ => 5.0,
    }

    // get only MoveType::EndTurn
    
}

/// Score reward moves contextually based on game situation.
/// Rewards are always highest priority (200+ range), but this
/// differentiates WHICH reward to prefer per slot.
fn score_reward(state: &crate::states::GameState, mv: &dyn Move) -> f32 {
    let base = 200.0;
    let reward = match mv.reward_type() {
        Ok(r) => r,
        Err(_) => return base,
    };

    let player_id = state.settings.current_player_turn_id;
    let is_perfection = state.settings.mode == ModeType::Perfection;

    match reward {
        // --- Slot 1: Explorer vs Workshop ---
        RewardType::Workshop => {
            // Safe best: +1 SPT is always valuable
            base + 10.0
        }
        RewardType::Explorer => {
            if state.settings.turn <= 1 {
                // Not best on first turn — little fog to clear
                base + 3.0
            } else {
                // Decent for multi-tribe maps (discovery stars)
                let tribe_count = state.tribes.len() as f32;
                base + 5.0 + tribe_count // More tribes = more discovery bonus
            }
        }

        // --- Slot 2: Walls vs Resources ---
        RewardType::CityWall => {
            // Preferred if enemies are nearby
            let city_idx = mv.target_idx().unwrap_or(0) as i32;
            let adj = get_adjacent_indices(state, city_idx, 3);
            let enemies_nearby = adj
                .iter()
                .any(|&idx| crate::functions::get_enemy_at(state, idx, player_id).is_some());

            if enemies_nearby {
                base + 12.0 // Strong preference when threatened
            } else {
                base + 4.0 // Low priority if safe
            }
        }
        RewardType::Resources => {
            // +5 stars is great early game when economy is tight
            if state.settings.turn <= 5 {
                base + 9.0
            } else {
                base + 6.0
            }
        }

        // --- Slot 3: PopGrowth vs BorderGrowth ---
        RewardType::PopGrowth => {
            // Generally better — +3 pop is solid and consistent
            base + 8.0
        }
        RewardType::BorderGrowth => {
            // Only worth it if border expansion covers valuable terrain
            // Heuristic: smaller cities benefit more from border growth
            let city_idx = mv.target_idx().unwrap_or(0) as i32;
            if let Some(tribe) = state.tribes.get(&player_id) {
                let city_territory = tribe
                    .cities
                    .iter()
                    .find(|c| c.tile_index == city_idx)
                    .map(|c| c._territory.len())
                    .unwrap_or(0);
                if city_territory < 10 {
                    base + 9.0 // Small city — border growth reveals/claims more
                } else {
                    base + 5.0 // Large city — pop growth is usually better
                }
            } else {
                base + 5.0
            }
        }

        // --- Slot 4+: Park vs SuperUnit ---
        RewardType::Park => {
            if is_perfection {
                // Always choose Park in Perfection — +250 score is massive
                base + 20.0
            } else {
                // In Domination, Park is +1 SPT but no tactical advantage
                base + 5.0
            }
        }
        RewardType::SuperUnit => {
            if is_perfection {
                // In Perfection, super unit matters less than score
                base + 8.0
            } else {
                // In Domination, super unit is game-changing
                base + 18.0
            }
        }

        _ => base,
    }
}

/// Score a road placement based on how well it connects cities.
/// Returns a bonus/penalty relative to the base Build score.
fn score_road(state: &crate::states::GameState, tile_idx: i32) -> f32 {
    let player_id = state.settings.current_player_turn_id;
    let map_size = state.map_size();
    let road_pos = Coords::from_index(tile_idx, map_size);

    // Gather our cities and their positions
    let tribe = match state.tribes.get(&player_id) {
        Some(t) => t,
        None => return 0.0,
    };

    let cities: Vec<(Coords, bool)> = tribe
        .cities
        .iter()
        .map(|c| {
            (
                Coords::from_index(c.tile_index, map_size),
                c.connected_to_capital,
            )
        })
        .collect();

    if cities.len() < 2 {
        return -3.0; // Only 1 city — roads are not useful yet
    }

    let adj = get_adjacent_indices(state, tile_idx, 1);

    // Check adjacency context
    let adj_to_road = adj.iter().any(|&idx| {
        state
            .tiles
            .get(&idx)
            .map_or(false, |t| t.owner == player_id && t.has_road)
    });

    let adj_to_city = adj
        .iter()
        .any(|&idx| tribe.cities.iter().any(|c| c.tile_index == idx));

    // Find the best city pair this road could help connect
    let mut best_score: f32 = -3.0;

    for (i, (city_a, connected_a)) in cities.iter().enumerate() {
        for (city_b, connected_b) in cities.iter().skip(i + 1) {
            // Most valuable: connecting an unconnected city to the capital
            let connection_bonus = if *connected_a != *connected_b {
                8.0 // One connected, one not — this road helps connect them
            } else if !*connected_a && !*connected_b {
                4.0 // Neither connected — still useful
            } else {
                1.0 // Both already connected — lower priority
            };

            let city_dist = city_a.distance_to(city_b);
            if city_dist == 0 {
                continue;
            }

            // Check if this road tile lies roughly on the path between the two cities
            // If dist(A, road) + dist(road, B) is close to dist(A, B), it's on-path
            let dist_a = road_pos.distance_to(city_a);
            let dist_b = road_pos.distance_to(city_b);
            let detour = (dist_a + dist_b) - city_dist;

            if detour <= 1 {
                // On or very near the shortest path
                let path_score = connection_bonus + 5.0 - (city_dist as f32 * 0.2).min(4.0);
                best_score = best_score.max(path_score);
            } else if detour <= 3 {
                // Slightly off path but still reasonable
                let path_score = connection_bonus + 1.0;
                best_score = best_score.max(path_score);
            }
        }
    }

    // Bonus for extending an existing road chain
    if adj_to_road {
        best_score += 2.0;
    }
    // Bonus for being adjacent to a city (starting or ending a connection)
    if adj_to_city {
        best_score += 3.0;
    }

    best_score
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Game;
    use crate::moves::EndTurnMove;

    #[test]
    fn test_basic_ordering() {
        let game = Game::new();
        let end_turn = EndTurnMove;

        // This is just a compilation test of the logic for now
        // Real testing would require a populated game state
        assert!(score_move(&game, &end_turn) == 0.0);
    }
}
