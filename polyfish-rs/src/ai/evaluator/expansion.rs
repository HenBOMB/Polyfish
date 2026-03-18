use crate::ai::genes::AIGenes;
use crate::states::{GameState, PlayerId};
use crate::types::StructureType;

/// Evaluates the expansion score (Cities, Levels) for a given player.
/// Returns a score roughly between 0.0 and 1.0.
pub fn evaluate_expansion(state: &GameState, player_id: PlayerId, genes: &AIGenes) -> f32 {
    let tribe_opt = state.tribes.get(&player_id);
    if tribe_opt.is_none() {
        return 0.0;
    }
    let tribe = tribe_opt.unwrap();

    // --- 3. Expansion Score (0.0 - 1.0) ---
    // Cities and Levels
    let mut total_city_points = tribe.cities.len() as f32;

    // Potential city bonus (units on neutral/enemy villages or ruins)
    for unit in &tribe.units {
        let idx = unit.coords.idx;
        if let Some(tile) = state.tiles.get(&idx) {
            if tile.owner != player_id {
                if let Some(s) = crate::functions::get_structure_at(state, idx) {
                    match s.structure_type {
                        StructureType::Village | StructureType::Ruin => {
                            total_city_points += genes.expansion.village_capture_bonus;
                        }
                        _ => {
                            // Enemy city
                            if tile.owner != 0 {
                                total_city_points += genes.expansion.enemy_city_capture_bonus;
                            }
                        }
                    }
                }
            }
        }
    }

    let city_count_score = (total_city_points / genes.expansion.city_count_normalizer).clamp(0.0, 1.0);
    let mut total_level = 0.0;
    for city in &tribe.cities {
        total_level += city.level as f32;
    }
    let level_score = (total_level / genes.expansion.level_normalizer).clamp(0.0, 1.0);

    (city_count_score * genes.expansion.city_count_weight) + (level_score * genes.expansion.level_weight)
}
