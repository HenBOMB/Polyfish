use crate::states::{GameState, PlayerId, HEALTH_SCALE};

/// Evaluates a game state for a given player.
/// Returns a score where higher is better for the player.
pub fn evaluate(state: &GameState, player_id: PlayerId) -> f32 {
    let tribe = match state.tribes.get(&player_id) {
        Some(t) => t,
        None => return -1000000.0,
    };

    let mut score = 0.0;

    // 1. Economic Value
    score += tribe.stars as f32 * 0.1;

    let mut income = 0.0;
    for city in &tribe.cities {
        income += city.production as f32 + 1.0;
    }
    score += income * 2.0;

    // 2. City & Territory Value
    score += tribe.cities.len() as f32 * 10.0;
    for city in &tribe.cities {
        score += city.level as f32 * 5.0;
        score += city.population as f32 * 0.5;
        score += city._territory.len() as f32 * 0.2;
    }

    // 3. Military Value
    for unit in &tribe.units {
        score += 2.0;
        score += (unit.health as f32 / HEALTH_SCALE as f32) * 1.5;
        if unit.veteran {
            score += 3.0;
        }
    }

    // 4. Technology Value
    let tech_count = tribe.tech_vanilla.len();
    score += tech_count as f32 * 4.0;

    // 5. Comparison to Opponents (Relative Scoring)
    let mut opponent_score = 0.0;
    for (&id, other_tribe) in &state.tribes {
        if id == player_id {
            continue;
        }

        let mut other_val = 0.0;
        other_val += other_tribe.stars as f32 * 0.1;
        let mut other_income = 0.0;
        for c in &other_tribe.cities {
            other_income += c.production as f32 + 1.0;
        }
        other_val += other_income * 2.0;
        other_val += other_tribe.cities.len() as f32 * 10.0;
        for u in &other_tribe.units {
            other_val += 2.0 + (u.health as f32 / HEALTH_SCALE as f32) * 1.5;
        }

        opponent_score = f32::max(opponent_score, other_val);
    }

    score - (opponent_score * 0.5)
}
