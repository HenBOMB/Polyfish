use crate::coords::Coords;
use crate::functions::{calculate_combat_preview, get_adjacent_indices, get_structure_at};
use crate::game::Game;
use crate::moves::Move;
use crate::settings::get_structure_setting;
use crate::types::{
    AbilityType, CityRewardType, ModeType, MoveType, StructureType, TerrainType,
};

/// Score a move based on heuristics
pub fn score_move(game: &Game, mv: &dyn Move) -> f32 {
    let state = &game.state;
    let move_type = mv.move_type();
    let tribe_id = state.settings.current_player_turn_id;
    let tribe = match state.tribes.get(&tribe_id) {
        Some(t) => t,
        None => return 0.0,
    };

    match move_type {
        MoveType::Reward => score_reward(state, mv),

        MoveType::Capture => {
            // Above the max Attack score (kill + territory = 110): standing on
            // a capturable, banking it beats any trade — measured on replays,
            // units in contested-village fights attacked every turn instead of
            // capturing, costing 2-3 turns on the first city.
            if let Ok(idx) = mv.source_idx() {
                if let Some(s) = get_structure_at(state, idx as i32) {
                    match s.structure_type {
                        StructureType::Ruin => 115.2,
                        StructureType::Village => 115.0,
                        _ => 115.5, // Capital or City
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
            // Fighting should serve territory (take/hold ground, or deny the
            // opponent that), not be picked for its own sake — so a kill/chip
            // scores meaningfully higher when it's on/near contested or
            // capturable ground than when it's a lone skirmish in the open.
            let territory_bonus = if let Ok(target) = mv.target_idx() {
                if is_territory_relevant(state, target as i32, tribe_id) {
                    15.0
                } else {
                    0.0
                }
            } else {
                0.0
            };

            if let (Ok(src), Ok(target)) = (mv.source_idx(), mv.target_idx()) {
                if let Some(preview) = calculate_combat_preview(state, src as i32, target as i32) {
                    if preview.defender_dies {
                        95.0 + territory_bonus // Rivals a capture: removing the
                        // opponent's ability to contest ground is nearly as
                        // valuable as taking ground.
                    } else if preview.attacker_dies {
                        1.0 // Suicide is very low priority regardless of context
                    } else if preview.damage_to_defender > 5.0 {
                        50.0 + territory_bonus
                    } else {
                        30.0 + territory_bonus
                    }
                } else {
                    // Infiltration or other special attack
                    30.0 + territory_bonus
                }
            } else {
                30.0 + territory_bonus
            }
        }

        MoveType::Ability => {
            if let Ok(ability) = mv.ability_type() {
                match ability {
                    AbilityType::Promote => 35.0,

                    AbilityType::Explode
                    | AbilityType::Swarm
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

                    AbilityType::GrowForest => -5.0, // Generally a waste unless for adjacency

                    AbilityType::ClearForest => {
                        let mut score = -2.0; // Very low base score (lower than EndTurn)
                        let stars = tribe.stars;
                        let has_forestry = crate::settings::technology::has_technology(
                            &tribe.tech_vanilla,
                            crate::types::TechnologyType::Forestry,
                        );

                        if let Ok(target) = mv.target_idx() {
                            // 1. Penalty for destroying resource
                            if let Some(Some(_res)) = state.resources.get(&(target as i32)) {
                                score -= 50.0; // Heavy penalty
                            }

                            // 2. Penalty for wasting potential Lumber Hut site (if we have the tech)
                            // Skip penalty if city is already max targeted level (5)
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
                                score -= 10.0;
                            }

                            // 3. Clustering Penalty: Don't remove forests near potential hub-spots (Sawmills)
                            // If this forest borders an empty tile, see how many other forests/huts also border it.
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
                                        } // Don't count ourselves
                                        if crate::functions::get_structure_type_at(state, n)
                                            == Some(StructureType::LumberHut)
                                        {
                                            return true;
                                        }
                                        if let Some(tile) = state.tiles.get(&n) {
                                            if tile.terrain_type
                                                == crate::types::TerrainType::Forest
                                            {
                                                return true;
                                            }
                                        }
                                        false
                                    })
                                    .count();

                                // Penalty per hub-spot we "weaken"
                                score -= (existing_prereqs as f32 + 1.0) * 2.5;
                            }

                            // 4. Strategic bonus: extra star needed for a level-up
                            // ONLY if we are critically short on stars.
                            if stars < 2 {
                                let mut enables_level_up = false;
                                for city in &tribe.cities {
                                    // If clearing gets us to the cost of a Road (2) or Harvest (2)
                                    if stars <= 1 {
                                        let pop_needed =
                                            (city.level + 1).saturating_sub(city.progress);

                                        // Scenario A: Enables a 2-star building/harvest
                                        if pop_needed <= 2 {
                                            enables_level_up = true;
                                            break;
                                        }

                                        // Scenario B: Enables a 2-star Road to connect capital (+1 pop)
                                        if pop_needed == 1 && !city.connected_to_capital {
                                            enables_level_up = true;
                                            break;
                                        }
                                    }
                                }
                                if enables_level_up {
                                    score += 10.0; // Moderate priority to enable growth
                                } else {
                                    score += 2.0; // Small desperation boost
                                }
                            } else if stars >= 5 {
                                score -= 10.0; // Don't decimate if we have money
                            }
                        }
                        score
                    }

                    _ => 10.0,
                }
            } else {
                10.0
            }
        }

        MoveType::Summon => {
            if state.settings.turn < 2 {
                return -10.0; // Favor development on turn 1
            }

            // Bumped from 10: capturing 2-3 villages in a game needs more
            // than one warrior, and the army-size rule below (unit_count <
            // city_count + 1) underestimated how army-hungry expansion is.
            let mut score = 15.0;

            // 1. Enemy Proximity: High priority if threatened
            let mut threatened = false;
            for city in &tribe.cities {
                let adj = crate::functions::get_adjacent_indices(state, city.idx, 3);
                if adj
                    .iter()
                    .any(|&idx| crate::functions::get_enemy_at(state, idx, tribe_id).is_some())
                {
                    threatened = true;
                    break;
                }
            }
            if threatened {
                score += 15.0;
            }

            // 2. Army Size relative to territory
            let unit_count = tribe.units.len();
            let city_count = tribe.cities.len();
            if unit_count < city_count + 1 {
                score += 8.0; // Need at least one unit per city + explorer (Total 18.0 < 22.0)
            } else if unit_count > city_count * 2 && !threatened {
                score -= 15.0; // Avoid bloat if safe
            }

            // 3. Super Unit preference — per tribe, not hardcoded to Giant
            // (Polaris fields Gaami, Aquarion Crab, Elyrion DragonEgg, Cymanti
            // Centipede).
            if let Ok(u_type) = mv.unit_type() {
                if u_type == crate::settings::units::get_super_unit(tribe.tribe_type) {
                    score += 15.0;
                }
            }

            score
        }

        MoveType::Build | MoveType::Harvest => {
            let mut score = 22.0;
            let player_id = state.settings.current_player_turn_id;

            // 1. Structure specific logic (Temples, Adjacency, Roads, Clustering)
            if let Ok(s_type) = mv.structure_type() {
                // Temple timing bonus in Perfection mode
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

                // Adjacency and Roads
                if let Ok(target) = mv.target_idx() {
                    let prereqs = &get_structure_setting(s_type).adjacent_types;

                    // Future Adjacency Prediction (Clustering Potential)
                    // If building a prereq, value empty tiles that could host the Hub.
                    // If multiple existing prereqs surround the same empty tile, reward it more.
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
                            // How many other prereqs of this type border this empty tile?
                            let neighbors_of_empty = get_adjacent_indices(state, *empty_idx, 1);
                            let existing_prereqs = neighbors_of_empty
                                .iter()
                                .filter(|&&n| {
                                    if n == target as i32 {
                                        return false;
                                    } // Don't count ourselves (we are about to build there)
                                    crate::functions::get_structure_type_at(state, n)
                                        == Some(s_type)
                                })
                                .count();

                            // (Self + others) * weight.
                            // 1 prereq = 2.5 bonus, 2 prereqs = 5.0 bonus, etc.
                            score += (existing_prereqs + 1) as f32 * 2.5;
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
                            0 => {}
                            1 => score -= 2.0,
                            2 => score += 5.0,
                            3 => score += 12.0,
                            _ => score += 18.0,
                        }
                    }

                    if s_type == StructureType::Road {
                        score += score_road(state, target as i32);
                    }
                }
            }

            // 2. Population efficiency scoring (Unified for Build and Harvest)
            if let Ok(target) = mv.target_idx() {
                let mut pop_gain = 0;

                // Determine pop gain for both MoveTypes
                if move_type == MoveType::Harvest {
                    if let Some(Some(res)) = state.resources.get(&(target as i32)) {
                        pop_gain =
                            crate::settings::resources::get_resource_setting(res.resource_type)
                                .reward_pop;
                    } else {
                        pop_gain = 1; // Fallback
                    }
                } else if let Ok(s_type) = mv.structure_type() {
                    pop_gain = get_structure_setting(s_type).reward_pop;

                    match s_type {
                        // Adjacency structures can give population if clustered
                        StructureType::Sawmill
                        | StructureType::Forge
                        | StructureType::Windmill
                        | StructureType::Market => {
                            // Re-calculate adj_count for pop estimation
                            let prereqs = &get_structure_setting(s_type).adjacent_types;
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

                            // Read the yield from the table. Market's reward_pop
                            // is 0 — it pays stars — so crediting it population
                            // invented pop that never arrives.
                            pop_gain = get_structure_setting(s_type).reward_pop
                                * adj_count as i32;
                        }
                        _ => {}
                    }
                }

                if pop_gain > 0 {
                    if let Some(tribe) = state.tribes.get(&player_id) {
                        if let Some(city) = tribe
                            .cities
                            .iter()
                            .find(|c| c._territory.contains(&(target as i32)))
                        {
                            let needed = city.level + 1;
                            let current = city.progress;

                            if current + pop_gain < needed {
                                score -= 4.0; // Doesn't finish level
                            } else {
                                score += 5.0; // Finishes level!
                            }
                        }
                    }
                }
            }

            score
        }

        MoveType::Step => {
            // Lowered from 50: a bare step used to outscore even a killing
            // blow (45) and had no pull toward a visible-but-distant village,
            // so wandering toward unexplored (often peripheral) fog was the
            // only differentiated signal. The gradients below give steps a
            // reason to move purposefully instead.
            let mut score = 35.0;
            if let Ok(target_idx) = mv.target_idx() {
                let player_id = state.settings.current_player_turn_id;

                // Prioritize stepping onto capture targets
                if let Some(tile) = state.tiles.get(&(target_idx as i32)) {
                    if let Some(s) = get_structure_at(state, target_idx as i32) {
                        match s.structure_type {
                            StructureType::Ruin
                            | StructureType::Village
                            | StructureType::Lighthouse => score += 43.0,
                            _ => {
                                // Enemy city potentially
                                // TODO: Doesnt check for peace treaty
                                if tile.owner != player_id && tile.owner != 0 {
                                    score += 50.0;
                                }
                            }
                        }
                    }

                    // `openness` = fraction of a wider (radius-3) window around
                    // the destination that's still unexplored. Off-map tiles
                    // don't count, so a map corner or a dead-end pocket scores
                    // low here even before it's been visited — there's
                    // fundamentally less left to discover there than in an
                    // open interior region.
                    let openness = regional_openness(state, target_idx as i32, player_id, 3);

                    let map_size = state.map_size();
                    let target_coords = Coords::from_index(target_idx as i32, map_size);
                    if let Ok(src_idx) = mv.source_idx() {
                        let src_coords = Coords::from_index(src_idx as i32, map_size);

                        let capturable =
                            nearest_visible_capturable(state, player_id, src_coords);
                        // On final approach (visible capturable within 2), damp the
                        // curiosity terms: measured on replays, reveal-chasing beat
                        // the closing gradient in 85% of d=2 episodes — the unit
                        // veered into fog at the village doorstep. Curiosity can
                        // wait one turn.
                        let approaching = capturable.map_or(false, |(_, d)| d <= 2);
                        let (openness_w, reveal_w) = if approaching {
                            (2.0, 1.0)
                        } else {
                            (6.0, 4.0)
                        };

                        // Exploration still matters (lighthouses, edge ruins), but
                        // scaled by openness rather than a flat per-tile count —
                        // previously uncapped (up to +16) and reliably pulled units
                        // toward unexplored map edges over contested ground.
                        score += openness * openness_w;

                        // Prefer steps that actually reveal fog over sidesteps that
                        // make equal progress toward a village or resource.
                        let newly_revealed = count_newly_revealed_by_step(
                            state,
                            player_id,
                            src_idx as i32,
                            target_idx as i32,
                        );
                        score += newly_revealed as f32 * reveal_w;

                        // Bonus for moving closer to the nearest explored, uncaptured village/ruin.
                        // Flat bonus if closing distance; urgency steep enough near the
                        // target that no curiosity term can outbid the final approach.
                        if let Some((nearest_coords, src_dist)) = capturable {
                            let dest_dist = nearest_coords.chebyshev_distance_to(&target_coords);
                            if dest_dist < src_dist {
                                score += 20.0;
                            }
                            score += (18.0 - 4.0 * dest_dist as f32).max(0.0);
                        } else if let Some((res_coords, res_src_dist)) =
                            nearest_frontier_resource(state, player_id, src_coords)
                        {
                            // No capturable in sight: close on the nearest explored
                            // resource with fog still around it — the spawn-time
                            // "border fruit" hint that a village ring is spilling
                            // into vision. Scaled by the fog left around the
                            // resource, else a unit whose sight can't reach the
                            // village parks on the resource forever; once its area
                            // is swept the pull dies and fog/center take back over.
                            let res_openness =
                                regional_openness(state, res_coords.idx, player_id, 2);
                            let dest_dist = res_coords.chebyshev_distance_to(&target_coords);
                            let mut pull = 0.0;
                            if dest_dist < res_src_dist {
                                pull += 14.0;
                            }
                            pull += (8.0 - 1.5 * dest_dist as f32).max(0.0);
                            score += pull * res_openness;
                        }
                    }

                    // Mild center-of-map pull — controlling the center
                    // controls tempo, and this stops wandering from
                    // defaulting toward the (often less contested) edges.
                    let center_dist = map_center_distance(state, target_coords);
                    score += (6.0 - center_dist as f32).max(0.0);
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
        MoveType::Resign => -100.0,
        _ => 5.0,
    }
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
        CityRewardType::Workshop => {
            // Safe best: +1 SPT is always valuable
            base + 10.0
        }
        CityRewardType::Explorer => {
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
        CityRewardType::CityWall => {
            // Preferred if enemies are nearby
            let city_idx = mv.target_idx().unwrap_or(0) as i32;
            let adj = get_adjacent_indices(state, city_idx, 1);
            let enemies_nearby = adj
                .iter()
                .any(|&idx| crate::functions::get_enemy_at(state, idx, player_id).is_some());

            if enemies_nearby {
                base + 12.0 // Strong preference when threatened
            } else {
                base + 4.0 // Low priority if safe
            }
        }
        CityRewardType::Resources => {
            // +5 stars is great early game when economy is tight
            if state.settings.turn <= 5 {
                base + 9.0
            } else {
                base + 6.0
            }
        }

        // --- Slot 3: PopGrowth vs BorderGrowth ---
        CityRewardType::PopGrowth => {
            // Generally better — +3 pop is solid and consistent
            base + 8.0
        }
        CityRewardType::BorderGrowth => {
            // Only worth it if border expansion covers valuable terrain
            // Heuristic: smaller cities benefit more from border growth
            let city_idx = mv.target_idx().unwrap_or(0) as i32;
            if let Some(tribe) = state.tribes.get(&player_id) {
                let city_territory = tribe
                    .cities
                    .iter()
                    .find(|c| c.idx == city_idx)
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
        CityRewardType::Park => {
            if is_perfection {
                // Always choose Park in Perfection — +250 score is massive
                base + 20.0
            } else {
                // In Domination, Park is +1 SPT but no tactical advantage
                base + 5.0
            }
        }
        CityRewardType::SuperUnit => {
            if is_perfection {
                // In Perfection, super unit matters less than score
                base + 8.0
            } else {
                // In Domination, super unit is game-changing. v10: the gap over
                // Park was 13 points, which through HEURISTIC_TEMP=20 softmaxes
                // to ~2:1 — not the near-always the comment implied. 22 points
                // is 3:1. Perfection is left alone: there a Park's +250 really
                // does win a score race.
                base + 27.0
            }
        }

        _ => base,
    }
}

/// Score a road placement based on how well it connects cities.
/// Returns a bonus/penalty relative to the base Build score.
///
/// EXP_ELO_055: priced off the real capital-network shortest path
/// (`movement::road_relief`, the same BFS `connect_remaining` prices
/// `reward.rs`'s Φ term with) instead of straight-line distance to a city
/// pair — for mobility/map-control, not the population a connection happens
/// to pay. FIRST FIT: dial `RELIEF_PER_TILE` against the measured effect
/// per the project's q-gap method before trusting it.
const RELIEF_PER_TILE: f32 = 5.0;

fn score_road(state: &crate::states::GameState, tile_idx: i32) -> f32 {
    let player_id = state.settings.current_player_turn_id;

    let tribe = match state.tribes.get(&player_id) {
        Some(t) => t,
        None => return 0.0,
    };

    if tribe.cities.len() < 2 {
        return -3.0; // Only 1 city — roads are not useful yet
    }

    let relief = crate::ai::movement::road_relief(state, player_id, tile_idx);
    let mut score = -3.0 + RELIEF_PER_TILE * relief as f32;

    let adj = get_adjacent_indices(state, tile_idx, 1);

    // Bonus for extending an existing road chain
    let adj_to_road = adj.iter().any(|&idx| {
        state
            .tiles
            .get(&idx)
            .map_or(false, |t| t.owner == player_id && t.has_road)
    });
    if adj_to_road {
        score += 2.0;
    }
    // Bonus for being adjacent to a city (starting or ending a connection)
    let adj_to_city = adj
        .iter()
        .any(|&idx| tribe.cities.iter().any(|c| c.idx == idx));
    if adj_to_city {
        score += 3.0;
    }

    score
}

/// True when fighting at `target_idx` serves territory — on/adjacent to an
/// uncaptured village/ruin, or on/adjacent to our own territory — rather than
/// being a skirmish for its own sake on open, uncontested ground.
fn is_territory_relevant(state: &crate::states::GameState, target_idx: i32, tribe_id: i32) -> bool {
    let adj = get_adjacent_indices(state, target_idx, 1);
    for idx in std::iter::once(target_idx).chain(adj) {
        if crate::functions::is_in_own_territory(state, idx, tribe_id) {
            return true;
        }
        if let Some(s) = get_structure_at(state, idx) {
            if matches!(s.structure_type, StructureType::Village | StructureType::Ruin) {
                return true;
            }
        }
    }
    false
}

/// Returns (Coords, dist) to the nearest visible unclaimed Village/Ruin for `tribe_id`,
/// or None if none. Dist is Chebyshev — units move diagonally, so Manhattan
/// overstates reach and misses genuinely-closing diagonal steps.
/// Full scan; called once per Step at MCTS root.
pub(crate) fn nearest_visible_capturable(
    state: &crate::states::GameState,
    tribe_id: i32,
    from: Coords,
) -> Option<(Coords, i32)> {
    let map_size = state.map_size();
    let mut best: Option<(Coords, i32)> = None;
    for (&idx, structure) in state.structures.iter() {
        let Some(s) = structure else { continue };
        let _ = s;
        if !crate::rules::capture::is_capturable(
            state,
            idx,
            tribe_id,
            crate::rules::capture::CaptureKind::NEUTRAL,
            true,
        ) {
            continue;
        }
        let coords = Coords::from_index(idx, map_size);
        let dist = coords.chebyshev_distance_to(&from);
        if best.map_or(true, |(_, b)| dist < b) {
            best = Some((coords, dist));
        }
    }
    best
}

/// Nearest explored, unowned resource that still has unexplored tiles within
/// Chebyshev 2 — mapgen spawns resources only inside a village/capital ring,
/// so a resource at the fog frontier marks where a hidden village may sit
/// (the "border fruit" a human steers toward at spawn). A fully-swept
/// resource carries no information. Starfish excluded (open-water spawns,
/// unrelated to villages). Dist is Chebyshev.
pub(crate) fn nearest_frontier_resource(
    state: &crate::states::GameState,
    tribe_id: i32,
    from: Coords,
) -> Option<(Coords, i32)> {
    let map_size = state.map_size();
    let mut best: Option<(Coords, i32)> = None;
    for (&idx, resource) in state.resources.iter() {
        let Some(r) = resource else { continue };
        if r.resource_type == crate::types::ResourceType::Starfish {
            continue;
        }
        let Some(tile) = state.tiles.get(&idx) else {
            continue;
        };
        if tile.owner != 0 || !tile.explorers.contains(&tribe_id) {
            continue;
        }
        // A hidden generator village must sit on an unexplored tile within 2.
        let has_fog_within_2 = crate::functions::get_square_indices(idx, 2, map_size)
            .into_iter()
            .any(|n| {
                state
                    .tiles
                    .get(&n)
                    .map_or(false, |t| !t.explorers.contains(&tribe_id))
            });
        if !has_fog_within_2 {
            continue;
        }
        let coords = Coords::from_index(idx, map_size);
        let dist = coords.chebyshev_distance_to(&from);
        if best.map_or(true, |(_, b)| dist < b) {
            best = Some((coords, dist));
        }
    }
    best
}

/// Fraction (0.0-1.0) of a `radius`-tile window around `target_idx` that is
/// still unexplored by `player_id`. Off-map tiles are excluded from the
/// unexplored count but NOT from the denominator (a fixed `(2*radius+1)^2 -
/// 1`), so a map corner or dead-end pocket scores lower than an open interior
/// region even when 100% of its (smaller) on-map surroundings are unexplored
/// — there is fundamentally less of the map left to discover there.
fn regional_openness(
    state: &crate::states::GameState,
    target_idx: i32,
    player_id: i32,
    radius: i32,
) -> f32 {
    let map_size = state.map_size();
    let coords = Coords::from_index(target_idx, map_size);
    let window = (2 * radius + 1) * (2 * radius + 1) - 1;
    if window <= 0 {
        return 0.0;
    }
    let mut unexplored = 0;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            if dx == 0 && dy == 0 {
                continue;
            }
            let nx = coords.x + dx;
            let ny = coords.y + dy;
            if nx < 0 || nx >= map_size || ny < 0 || ny >= map_size {
                continue; // Off-map: not discoverable, doesn't count toward openness.
            }
            let n_idx = ny * map_size + nx;
            if let Some(tile) = state.tiles.get(&n_idx) {
                if !tile.explorers.contains(&player_id) {
                    unexplored += 1;
                }
            }
        }
    }
    (unexplored as f32 / window as f32).clamp(0.0, 1.0)
}

fn unit_vision_range(
    state: &crate::states::GameState,
    unit: &crate::states::UnitState,
    at_idx: i32,
) -> i32 {
    // Evaluated at `at_idx`, which may not be where the unit currently stands.
    crate::rules::vision::unit_vision_range_with(
        &crate::settings::units::get_unit_setting(unit.unit_type).skills,
        state
            .tiles
            .get(&at_idx)
            .map_or(false, |t| t.terrain_type == TerrainType::Mountain),
    )
}

/// How many unexplored tiles the unit would reveal by stepping to `dest_idx`:
/// unexplored tiles within its vision range at the destination. Exploration
/// (`tile.explorers`) is permanent, so anything currently seen by other units
/// is already explored — no full-map visibility diff needed. That matters
/// because heuristic MCTS runs this for every Step candidate at every node
/// expansion and rollout ply; the old two-full-visibility-set version
/// dominated the entire search's CPU profile.
fn count_newly_revealed_by_step(
    state: &crate::states::GameState,
    player_id: i32,
    src_idx: i32,
    dest_idx: i32,
) -> i32 {
    if !state.settings._fow {
        return 0;
    }
    let Some(tribe) = state.tribes.get(&player_id) else {
        return 0;
    };
    let Some(unit) = tribe.units.iter().find(|u| u.coords.idx == src_idx) else {
        return 0;
    };
    let range = unit_vision_range(state, unit, dest_idx);

    let mut revealed = 0;
    for idx in std::iter::once(dest_idx).chain(get_adjacent_indices(state, dest_idx, range)) {
        if let Some(tile) = state.tiles.get(&idx) {
            if !tile.explorers.contains(&player_id) {
                revealed += 1;
            }
        }
    }
    revealed
}

/// Manhattan distance from `from` to the map's center tile. Used to give a
/// mild pull toward contested/central ground rather than defaulting toward
/// the map edges (which is where the uncapped fog-reveal bonus used to push
/// units by itself).
fn map_center_distance(state: &crate::states::GameState, from: Coords) -> i32 {
    let map_size = state.map_size();
    let center = Coords::from_xy(map_size / 2, map_size / 2, map_size);
    center.distance_to(&from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actions::units::summon_unit;
    use crate::game::Game;
    use crate::moves::{EndTurnMove, StepMove};
    use crate::states::{ResourceState, TileState, TribeState};
    use crate::types::{ResourceType, TribeType, UnitType};

    fn explored_field_state(player_id: i32) -> crate::states::GameState {
        let mut state = crate::states::GameState::default();
        state.settings.size = 11;
        state.settings._fow = true;
        state.settings.current_player_turn_id = player_id;
        let mut tribe = TribeState::default();
        tribe.id = player_id;
        tribe.tribe_type = TribeType::Imperius;
        state.tribes.insert(player_id, tribe);
        for i in 0..121 {
            let mut tile = TileState::default();
            tile.terrain_type = TerrainType::Field;
            tile.explorers.insert(player_id);
            state.tiles.insert(i, tile);
        }
        state
    }

    #[test]
    fn frontier_resource_needs_fog_within_two() {
        let player_id = 1;
        let mut state = explored_field_state(player_id);
        let from = Coords::from_index(5 * 11 + 5, 11);
        let fruit_idx = 2 * 11 + 3;
        state.resources.insert(
            fruit_idx,
            Some(ResourceState {
                resource_type: ResourceType::Fruit,
            }),
        );

        // Fully-explored surroundings: no hidden village possible, no beacon.
        assert!(nearest_frontier_resource(&state, player_id, from).is_none());

        // Fog within 2 of the fruit: a village could hide there — beacon fires.
        let fog_idx = 0 * 11 + 3;
        state
            .tiles
            .get_mut(&fog_idx)
            .unwrap()
            .explorers
            .remove(&player_id);
        assert_eq!(
            nearest_frontier_resource(&state, player_id, from).map(|(c, _)| c.idx),
            Some(fruit_idx)
        );

        // A resource the player hasn't explored is invisible to the beacon.
        state
            .tiles
            .get_mut(&fruit_idx)
            .unwrap()
            .explorers
            .remove(&player_id);
        assert!(nearest_frontier_resource(&state, player_id, from).is_none());
    }

    #[test]
    fn starfish_and_owned_resources_are_not_beacons() {
        let player_id = 1;
        let mut state = explored_field_state(player_id);
        let from = Coords::from_index(5 * 11 + 5, 11);

        // Both resources have fog within 2, so only type/ownership excludes them.
        state.resources.insert(
            30,
            Some(ResourceState {
                resource_type: ResourceType::Starfish,
            }),
        );
        state.resources.insert(
            40,
            Some(ResourceState {
                resource_type: ResourceType::Game,
            }),
        );
        state.tiles.get_mut(&8).unwrap().explorers.remove(&player_id);
        state.tiles.get_mut(&18).unwrap().explorers.remove(&player_id);
        state.tiles.get_mut(&40).unwrap().owner = 2;
        assert!(nearest_frontier_resource(&state, player_id, from).is_none());
    }

    #[test]
    fn test_basic_ordering() {
        let game = Game::new();
        let end_turn = EndTurnMove;

        // This is just a compilation test of the logic for now
        // Real testing would require a populated game state
        assert!(score_move(&game, &end_turn) == 0.0);
    }

    #[test]
    fn step_toward_fog_scores_above_sidestep_with_equal_resource_progress() {
        let mut state = crate::states::GameState::default();
        state.settings.size = 11;
        state.settings._fow = true;
        let player_id = 1;
        state.settings.current_player_turn_id = player_id;

        let mut tribe = TribeState::default();
        tribe.id = player_id;
        tribe.tribe_type = TribeType::Imperius;
        state.tribes.insert(player_id, tribe);

        for i in 0..121 {
            let mut tile = TileState::default();
            tile.terrain_type = TerrainType::Field;
            tile.explorers.insert(player_id);
            state.tiles.insert(i, tile);
        }

        // Hide a pocket west of (4,5) so stepping there reveals fog; (5,6) does not.
        for x in 0..3 {
            for y in 4..7 {
                let idx = y * 11 + x;
                state.tiles.get_mut(&idx).unwrap().explorers.remove(&player_id);
            }
        }

        let unit_idx = 5 * 11 + 5;
        let toward_fog = 4 * 11 + 5;
        let sideways = 5 * 11 + 6;
        let fruit_idx = 4 * 11 + 6;

        state.resources.insert(
            fruit_idx,
            Some(ResourceState {
                resource_type: ResourceType::Fruit,
            }),
        );

        let _ = summon_unit(&mut state, UnitType::Warrior, unit_idx, false, false);
        let game = Game { state };

        let toward = StepMove::new(unit_idx, toward_fog);
        let side = StepMove::new(unit_idx, sideways);

        let src_coords = Coords::from_index(unit_idx, 11);
        let fruit_coords = Coords::from_index(fruit_idx, 11);
        let src_dist = src_coords.distance_to(&fruit_coords);
        let toward_dist = Coords::from_index(toward_fog, 11).distance_to(&fruit_coords);
        let side_dist = Coords::from_index(sideways, 11).distance_to(&fruit_coords);
        assert_eq!(src_dist - toward_dist, 1);
        assert_eq!(src_dist - side_dist, 1);

        let toward_score = score_move(&game, &toward);
        let side_score = score_move(&game, &side);

        assert!(
            toward_score > side_score,
            "toward fog ({toward_score}) should beat sidestep ({side_score})"
        );
    }

    #[test]
    fn finish_level_uses_progress_not_population() {
        // Level-3 city with lifetime population 7 but progress 1: a 1-pop
        // harvest does NOT finish the level (needs 4). The old code read
        // `population` and always awarded the finish bonus.
        let player_id = 1;
        let mut state = explored_field_state(player_id);
        let target = 3 * 11 + 3;
        state.resources.insert(
            target,
            Some(ResourceState {
                resource_type: ResourceType::Fruit,
            }),
        );
        let mut city = crate::states::CityState::default();
        city.idx = 4 * 11 + 3;
        city.owner = player_id;
        city.level = 3;
        city.population = 7;
        city.progress = 1;
        city._territory.push(target);
        state.tribes.get_mut(&player_id).unwrap().cities.push(city);
        state.tribes.get_mut(&player_id).unwrap().stars = 10;

        let game = Game { state };
        let harvest = crate::moves::harvest::HarvestMove::new(target);
        let stranding = score_move(&game, &harvest);

        // Same city at progress 3: the harvest completes the level.
        let mut game2 = Game { state: game.state.clone() };
        game2.state.tribes.get_mut(&player_id).unwrap().cities[0].progress = 3;
        let finishing = score_move(&game2, &harvest);

        assert!(
            finishing > stranding,
            "finishing harvest ({finishing}) must outscore stranding one ({stranding})"
        );
    }
}
