use crate::ai::heuristics;
use crate::states::{GameState, PlayerId};

// Evaluates the power of the army, returns a score between 0.0 and 1.0
pub fn evaluate_army(state: &GameState, player_id: PlayerId) -> f32 {
    let tribe_opt = state.tribes.get(&player_id);
    if tribe_opt.is_none() {
        return 0.0;
    }
    let tribe = tribe_opt.unwrap();

    // --- 2. Military Score (0.0 - 1.0) ---
    // Sum of unit power.
    // A full army of 20 strong units (avg 0.7) = 14.0 score.
    // Soft cap at 20.0 (allows for huge armies to saturate, but typically < 1.0)
    let mut score_army = 0.0;
    for unit in &tribe.units {
        score_army += heuristics::assess_unit_power(state, unit);
    }

    let progress = (state.settings.turn as f32 / state.settings.max_turns as f32).clamp(0.0, 1.0);
    let mut max_units;

    if progress < 0.3 {
        // Early Game: At least 2.0 units per city + 1 extra unit
        max_units = tribe.cities.len() as f32 * 2.0 + 1.0;
    } else if progress < 0.7 {
        // Mid Game: At least 2.0 units per city + 2 extra units
        max_units = tribe.cities.len() as f32 * 2.0 + 2.0;
    } else {
        // Late Game: At least 4.0 units per city + 4 extra units
        max_units = tribe.cities.len() as f32 * 4.0 + 4.0;
    };

    max_units = max_units.min(crate::states::default_max_units() as f32);

    (score_army / max_units).clamp(0.0, 1.0)
}
