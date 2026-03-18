use crate::ai::genes::AIGenes;
use crate::coords::Coords;
use crate::functions::{calculate_combat_preview, get_adjacent_indices, get_structure_at};
use crate::game::Game;
use crate::moves::Move;
use crate::settings::get_structure_setting;
use crate::types::{AbilityType, ModeType, MoveType, RewardType, StructureType};

/// Score a move based on heuristics for move ordering
pub fn score_move(game: &Game, mv: &dyn Move, genes: &AIGenes) -> f32 {
    let state = &game.state;
    let move_type = mv.move_type();
    let tribe_id = state.settings.current_player_turn_id;
    let tribe = state.tribes.get(&tribe_id).unwrap();

    match move_type {
        MoveType::Reward => score_reward(state, mv, genes),

        MoveType::Capture => {
            if let Ok(idx) = mv.source_idx() {
                if let Some(s) = get_structure_at(state, idx as i32) {
                    match s.structure_type {
                        StructureType::Ruin => genes.ordering.capture_ruin,
                        StructureType::Village => genes.ordering.capture_village,
                        _ => genes.ordering.capture_city, // Capital or City
                    }
                } else {
                    // Likely starfish
                    genes.ordering.capture_starfish
                }
            } else {
                genes.ordering.capture_starfish
            }
        }

        MoveType::Attack => {
            if let (Ok(src), Ok(target)) = (mv.source_idx(), mv.target_idx()) {
                if let Some(preview) = calculate_combat_preview(state, src as i32, target as i32) {
                    if preview.defender_dies {
                        genes.ordering.attack_kill
                    } else if preview.attacker_dies {
                        genes.ordering.attack_suicide
                    } else if preview.damage_to_defender >= genes.ordering.attack_heavy_threshold {
                        genes.ordering.attack_heavy_damage
                    } else {
                        genes.ordering.attack_light_damage
                    }
                } else {
                    // Infiltration or other special attack
                    genes.ordering.attack_light_damage
                }
            } else {
                genes.ordering.attack_light_damage
            }
        }

        MoveType::Ability => {
            if let Ok(ability) = mv.ability_type() {
                match ability {
                    AbilityType::Promote => genes.ordering.ability_promote,

                    AbilityType::Explode
                    | AbilityType::Boost
                    | AbilityType::FreezeArea
                    | AbilityType::Convert => genes.ordering.ability_combat,

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

                                if hp_pct < genes.ordering.ability_recover_critical_threshold {
                                    genes.ordering.ability_recover_critical
                                } else if hp_pct < 0.9 {
                                    // Safe to heal?
                                    // If in city or mountain (defense bonus), encourages healing
                                    let def_bonus = crate::functions::get_defense_bonus(state, unit);
                                    if def_bonus > 1.0 {
                                        genes.ordering.ability_recover_safe
                                    } else {
                                        genes.ordering.ability_recover_waste
                                    }
                                } else {
                                    genes.ordering.ability_recover_waste
                                }
                            } else {
                                genes.ordering.ability_default
                            }
                        } else {
                            genes.ordering.ability_default
                        }
                    }

                    AbilityType::Disband => genes.ordering.ability_disband,

                    AbilityType::BurnForest => genes.ordering.ability_burn_forest,

                    AbilityType::Destroy => genes.ordering.ability_destroy,

                    AbilityType::ClearForest => score_clear_forest(state, tribe, mv, genes),

                    _ => genes.ordering.ability_default,
                }
            } else {
                genes.ordering.ability_default
            }
        }

        MoveType::Summon => score_summon(state, tribe, mv, genes),

        MoveType::Build | MoveType::Harvest => score_build(state, tribe, mv, genes),

        MoveType::Step => score_step(state, mv, genes),

        MoveType::Research => score_research(state, tribe, mv, genes),

        MoveType::EndTurn => 0.0,

        _ => 5.0,
    }
}

fn score_clear_forest(
    state: &crate::states::GameState,
    tribe: &crate::states::TribeState,
    mv: &dyn Move,
    genes: &AIGenes,
) -> f32 {
    let mut score = genes.ordering.clear_forest_base;
    let stars = tribe.stars;
    let has_forestry = crate::settings::technology::has_technology(
        &tribe.tech_vanilla,
        crate::types::TechnologyType::Forestry,
    );

    if let Ok(target) = mv.target_idx() {
        // 1. Penalty for destroying resource
        if let Some(Some(_res)) = state.resources.get(&(target as i32)) {
            score -= genes.ordering.clear_forest_resource_penalty;
        }

        // 2. Penalty for wasting potential Lumber Hut site (if we have the tech)
        let mut growth_useful = true;
        if let Some(city) = tribe
            .cities
            .iter()
            .find(|c| c._territory.contains(&(target as i32)))
        {
            if city.level >= 5 {
                growth_useful = false;
            }
        }

        if has_forestry && growth_useful {
            score -= genes.ordering.clear_forest_forestry_penalty;
        }

        // 3. Clustering Penalty
        let adj = get_adjacent_indices(state, target as i32, 1);
        for empty_idx in adj
            .iter()
            .filter(|&&idx| get_structure_at(state, idx).is_none())
        {
            let neighbors_of_empty = get_adjacent_indices(state, *empty_idx, 1);
            let existing_prereqs = neighbors_of_empty
                .iter()
                .filter(|&&n| {
                    if n == target as i32 {
                        return false;
                    }
                    if crate::functions::get_structure_type_at(state, n)
                        == Some(StructureType::LumberHut)
                    {
                        return true;
                    }
                    if let Some(tile) = state.tiles.get(&n) {
                        if tile.terrain_type == crate::types::TerrainType::Forest {
                            return true;
                        }
                    }
                    false
                })
                .count();

            score -= (existing_prereqs as f32 + 1.0) * genes.ordering.clear_forest_cluster_penalty_per;
        }

        // 4. Strategic bonus
        if stars < 2 {
            let mut enables_level_up = false;
            for city in &tribe.cities {
                if stars <= 1 {
                    let pop_needed = (city.level + 1).saturating_sub(city.population);
                    if pop_needed <= 2 {
                        enables_level_up = true;
                        break;
                    }
                    if pop_needed == 1 && !city.connected_to_capital {
                        enables_level_up = true;
                        break;
                    }
                }
            }
            if enables_level_up {
                score += genes.ordering.clear_forest_enables_levelup_bonus;
            } else {
                score += genes.ordering.clear_forest_desperation_bonus;
            }
        } else if stars >= 5 {
            score -= genes.ordering.clear_forest_healthy_penalty;
        }
    }
    score
}

fn score_summon(
    state: &crate::states::GameState,
    tribe: &crate::states::TribeState,
    mv: &dyn Move,
    genes: &AIGenes,
) -> f32 {
    let tribe_id = tribe.id;
    if state.settings.turn < 2 {
        return genes.ordering.summon_early_penalty;
    }

    let mut score = genes.ordering.summon_base;

    let mut threatened = false;
    for city in &tribe.cities {
        let adj = crate::functions::get_adjacent_indices(state, city.tile_index, 3);
        if adj
            .iter()
            .any(|&idx| crate::functions::get_enemy_at(state, idx, tribe_id).is_some())
        {
            threatened = true;
            break;
        }
    }
    if threatened {
        score += genes.ordering.summon_threat_bonus;
    }

    let unit_count = tribe.units.len();
    let city_count = tribe.cities.len();
    if unit_count < city_count + 1 {
        score += genes.ordering.summon_army_small_bonus;
    } else if unit_count > city_count * 2 && !threatened {
        score -= genes.ordering.summon_army_bloat_penalty;
    }

    if let Ok(u_type) = mv.unit_type() {
        if u_type == crate::types::UnitType::Giant {
            score += genes.ordering.summon_giant_bonus;
        }
    }

    score
}

fn score_build(
    state: &crate::states::GameState,
    tribe: &crate::states::TribeState,
    mv: &dyn Move,
    genes: &AIGenes,
) -> f32 {
    let mut score = genes.ordering.build_base;
    let player_id = tribe.id;

    if let Ok(s_type) = mv.structure_type() {
        // Temple timing bonus
        if state.settings.mode == ModeType::Perfection {
            match s_type {
                StructureType::Temple
                | StructureType::ForestTemple
                | StructureType::MountainTemple
                | StructureType::WaterTemple => {
                    let turn = state.settings.turn;
                    if turn <= 19 {
                        score += genes.ordering.temple_early_bonus + (19 - turn) as f32;
                    } else if turn <= 25 {
                        score += genes.ordering.temple_mid_bonus;
                    }
                }
                StructureType::AltarOfPeace
                | StructureType::TowerOfWisdom
                | StructureType::GrandBazaar
                | StructureType::EmperorsTomb
                | StructureType::GateOfPower
                | StructureType::ParkOfFortune
                | StructureType::EyeOfGod => {
                    score += genes.ordering.monument_bonus;
                }
                _ => {}
            }
        }

        if let Ok(target) = mv.target_idx() {
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

            let matching_hub = match s_type {
                StructureType::LumberHut => Some(StructureType::Sawmill),
                StructureType::Mine => Some(StructureType::Forge),
                StructureType::Farm => Some(StructureType::Windmill),
                _ => None,
            };

            if let Some(_hub) = matching_hub {
                let adj = get_adjacent_indices(state, target as i32, 1);
                for empty_idx in adj
                    .iter()
                    .filter(|&&idx| get_structure_at(state, idx).is_none())
                {
                    let neighbors_of_empty = get_adjacent_indices(state, *empty_idx, 1);
                    let existing_prereqs = neighbors_of_empty
                        .iter()
                        .filter(|&&n| {
                            if n == target as i32 {
                                return false;
                            }
                            crate::functions::get_structure_type_at(state, n) == Some(s_type)
                        })
                        .count();

                    score += (existing_prereqs + 1) as f32 * genes.ordering.clustering_prereq_bonus;
                }
            }

            if !prereqs.is_empty() {
                let adj = get_adjacent_indices(state, target as i32, 1);
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

                match adj_count {
                    0 => score -= genes.ordering.adjacency_lonely_penalty,
                    1 => {} // Neutral
                    2 => score += genes.ordering.adjacency_2_bonus,
                    3 => score += genes.ordering.adjacency_3_bonus,
                    _ => score += genes.ordering.adjacency_4plus_bonus,
                }
            }

            if s_type == StructureType::Road {
                score += score_road(state, target as i32, genes);
            }
        }
    }

    if let Ok(target) = mv.target_idx() {
        let mut pop_gain = 0;

        if mv.move_type() == MoveType::Harvest {
            if let Some(Some(res)) = state.resources.get(&(target as i32)) {
                pop_gain = crate::settings::resources::get_resource_setting(res.resource_type)
                    .reward_pop;
            } else {
                pop_gain = 1;
            }
        } else if let Ok(s_type) = mv.structure_type() {
            pop_gain = get_structure_setting(s_type).reward_pop;

            match s_type {
                StructureType::Sawmill
                | StructureType::Forge
                | StructureType::Windmill
                | StructureType::Market => {
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
                    let adj = get_adjacent_indices(state, target as i32, 1);
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

                    pop_gain = match s_type {
                        StructureType::Forge => adj_count as i32 * 2,
                        _ => adj_count as i32,
                    };
                }
                _ => {}
            }
        }

        if pop_gain > 0 {
            if let Some(city) = tribe
                .cities
                .iter()
                .find(|c| c._territory.contains(&(target as i32)))
            {
                let needed = city.level + 1;
                let current = city.population;

                if current + pop_gain < needed {
                    score -= genes.ordering.levelup_miss_penalty;
                } else {
                    score += genes.ordering.levelup_completion_bonus;
                }
            }
        }
    }

    score
}

fn score_step(state: &crate::states::GameState, mv: &dyn Move, genes: &AIGenes) -> f32 {
    let mut score = genes.ordering.step_base;
    if let Ok(target_idx) = mv.target_idx() {
        let player_id = state.settings.current_player_turn_id;

        if let Some(tile) = state.tiles.get(&(target_idx as i32)) {
            if let Some(s) = get_structure_at(state, target_idx as i32) {
                match s.structure_type {
                    StructureType::Ruin | StructureType::Village | StructureType::Lighthouse => {
                        score += genes.ordering.step_capture_target_bonus
                    }
                    _ => {
                        if tile.owner != player_id && tile.owner != 0 {
                            score += genes.ordering.step_enemy_city_bonus;
                        }
                    }
                }
            }

            let adj = crate::functions::get_adjacent_indices(state, target_idx as i32, 1);
            for n_idx in adj {
                if let Some(tile) = state.tiles.get(&n_idx) {
                    if !tile.explorers.contains(&player_id) {
                        score += genes.ordering.step_fog_reveal_bonus;
                    }
                }
            }
        }
    }
    score
}

fn score_research(
    state: &crate::states::GameState,
    tribe: &crate::states::TribeState,
    mv: &dyn Move,
    genes: &AIGenes,
) -> f32 {
    if let Ok(tech) = mv.tech_type() {
        let player_id = tribe.id;
        let utility =
            crate::ai::evaluator::research::evaluate_tech_utility(state, player_id, tech, genes);

        let base = genes.ordering.research_base + (utility.clamp(-2.0, 8.0) + 2.0);

        let has_village_opportunity = tribe.units.iter().any(|unit| {
            let idx = unit.coords.idx;
            if let Some(tile) = state.tiles.get(&idx) {
                if tile.owner != player_id {
                    if let Some(s) = get_structure_at(state, idx) {
                        return s.structure_type == StructureType::Village;
                    }
                }
            }
            false
        });

        if has_village_opportunity {
            base + genes.ordering.research_buy_before_capture_bonus
        } else {
            base
        }
    } else {
        5.0
    }
}

fn score_reward(state: &crate::states::GameState, mv: &dyn Move, genes: &AIGenes) -> f32 {
    let base = genes.ordering.reward_base;
    let reward = match mv.reward_type() {
        Ok(r) => r,
        Err(_) => return base,
    };

    let player_id = state.settings.current_player_turn_id;
    let is_perfection = state.settings.mode == ModeType::Perfection;

    match reward {
        RewardType::Workshop => base + genes.ordering.reward_workshop_bonus,
        RewardType::Explorer => {
            if state.settings.turn <= 1 {
                base + genes.ordering.reward_explorer_early_penalty
            } else {
                let tribe_count = state.tribes.len() as f32;
                base + genes.ordering.reward_explorer_bonus + tribe_count
            }
        }
        RewardType::CityWall => {
            let city_idx = mv.target_idx().unwrap_or(0) as i32;
            let adj = get_adjacent_indices(state, city_idx, 1);
            let enemies_nearby = adj
                .iter()
                .any(|&idx| crate::functions::get_enemy_at(state, idx, player_id).is_some());

            if enemies_nearby {
                base + genes.ordering.reward_wall_threatened_bonus
            } else {
                base + genes.ordering.reward_wall_safe_bonus
            }
        }
        RewardType::Resources => {
            if state.settings.turn <= 5 {
                base + genes.ordering.reward_resources_early_bonus
            } else {
                base + genes.ordering.reward_resources_late_bonus
            }
        }
        RewardType::PopGrowth => base + genes.ordering.reward_pop_growth_bonus,
        RewardType::BorderGrowth => {
            let city_idx = mv.target_idx().unwrap_or(0) as i32;
            if let Some(tribe) = state.tribes.get(&player_id) {
                let city_territory = tribe
                    .cities
                    .iter()
                    .find(|c| c.tile_index == city_idx)
                    .map(|c| c._territory.len())
                    .unwrap_or(0);
                if city_territory < 10 {
                    base + genes.ordering.reward_border_growth_small_bonus
                } else {
                    base + genes.ordering.reward_border_growth_large_bonus
                }
            } else {
                base + genes.ordering.reward_border_growth_large_bonus
            }
        }
        RewardType::Park => {
            if is_perfection {
                base + genes.ordering.reward_park_perfection_bonus
            } else {
                base + genes.ordering.reward_park_domination_bonus
            }
        }
        RewardType::SuperUnit => {
            if is_perfection {
                base + genes.ordering.reward_super_unit_perfection_bonus
            } else {
                base + genes.ordering.reward_super_unit_domination_bonus
            }
        }
        _ => base,
    }
}

fn score_road(state: &crate::states::GameState, tile_idx: i32, genes: &AIGenes) -> f32 {
    let player_id = state.settings.current_player_turn_id;
    let map_size = state.map_size();
    let road_pos = Coords::from_index(tile_idx, map_size);

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
        return genes.ordering.road_single_city_penalty;
    }

    let adj = get_adjacent_indices(state, tile_idx, 1);

    let adj_to_road = adj.iter().any(|&idx| {
        state
            .tiles
            .get(&idx)
            .map_or(false, |t| t.owner == player_id && t.has_road)
    });

    let adj_to_city = adj
        .iter()
        .any(|&idx| tribe.cities.iter().any(|c| c.tile_index == idx));

    let mut best_score: f32 = genes.ordering.road_single_city_penalty;

    for (i, (city_a, connected_a)) in cities.iter().enumerate() {
        for (city_b, connected_b) in cities.iter().skip(i + 1) {
            let connection_bonus = if *connected_a != *connected_b {
                genes.ordering.road_connection_unconnected_bonus
            } else if !*connected_a && !*connected_b {
                genes.ordering.road_connection_neither_bonus
            } else {
                genes.ordering.road_connection_both_bonus
            };

            let city_dist = city_a.distance_to(city_b);
            if city_dist == 0 {
                continue;
            }

            let dist_a = road_pos.distance_to(city_a);
            let dist_b = road_pos.distance_to(city_b);
            let detour = (dist_a + dist_b) - city_dist;

            if detour <= 1 {
                let path_score = connection_bonus + genes.ordering.road_on_path_bonus - (city_dist as f32 * 0.2).min(4.0);
                best_score = best_score.max(path_score);
            } else if detour <= 3 {
                let path_score = connection_bonus + 1.0;
                best_score = best_score.max(path_score);
            }
        }
    }

    if adj_to_road {
        best_score += genes.ordering.road_adj_road_bonus;
    }
    if adj_to_city {
        best_score += genes.ordering.road_adj_city_bonus;
    }

    best_score
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Game;
    use crate::moves::EndTurnMove;

    #[test]
    fn test_basic_ordering_compiles() {
        let game = Game::new();
        let end_turn = EndTurnMove;
        let genes = AIGenes::default();

        assert!(score_move(&game, &end_turn, &genes) == 0.0);
    }
}
