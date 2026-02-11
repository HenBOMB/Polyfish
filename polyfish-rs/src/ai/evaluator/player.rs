use super::{army, economy, expansion, exploration};
use crate::states::{GameState, PlayerId};

/// Evaluates a single player's absolute score (0.0 - 1.0).
pub fn evaluate_player(state: &GameState, player_id: PlayerId) -> f32 {
    let tribe_opt = state.tribes.get(&player_id);
    if tribe_opt.is_none() {
        return -1.0;
    }
    let _tribe = tribe_opt.unwrap();

    let eco_score = economy::evaluate_economy(state, player_id);
    let mil_score = army::evaluate_army(state, player_id);
    let exp_score = expansion::evaluate_expansion(state, player_id);
    let fow_score = exploration::evaluate_exploration(state, player_id);

    // --- 4. Game Stage Weighting ---
    let max_turn = state.settings.max_turns.max(1) as f32;
    let current_turn = state.settings.turn as f32;
    let progress = (current_turn / max_turn).clamp(0.0, 1.0);

    // Early Game
    let (w_eco, w_mil, w_exp, w_fow) = if progress < 0.3 {
        // Both Economy and Expansion
        (0.4, 0.05, 0.4, 0.15)
    }
    // Mid Game
    else if progress < 0.7 {
        // Perfection: Balanced Economy and Expansion
        if state.settings.mode == crate::ModeType::Perfection {
            (0.4, 0.1, 0.3, 0.2)
        }
        // Domination: Military and Exploration
        else {
            (0.1, 0.4, 0.1, 0.4)
        }
    }
    // End Game
    else {
        // Perfection: Economy
        if state.settings.mode == crate::ModeType::Perfection {
            (0.4, 0.1, 0.25, 0.25)
        }
        // Domination: Military and Balance
        else {
            (0.2, 0.4, 0.2, 0.2)
        }
    };

    let final_score =
        (eco_score * w_eco) + (exp_score * w_exp) + (mil_score * w_mil) + (fow_score * w_fow);

    final_score.clamp(0.0, 1.0)

    // eco_score
}
