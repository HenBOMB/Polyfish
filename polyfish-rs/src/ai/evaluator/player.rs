use super::{army, economy, expansion, exploration};
use crate::ai::genes::AIGenes;
use crate::states::{GameState, PlayerId};

/// Evaluates a single player's absolute score (0.0 - 1.0).
pub fn evaluate_player(state: &GameState, player_id: PlayerId, genes: &AIGenes) -> f32 {
    let tribe_opt = state.tribes.get(&player_id);
    if tribe_opt.is_none() {
        return -1.0;
    }
    let _tribe = tribe_opt.unwrap();

    let eco_score = economy::evaluate_economy(state, player_id, genes);
    let mil_score = army::evaluate_army(state, player_id, genes);
    let exp_score = expansion::evaluate_expansion(state, player_id, genes);
    let fow_score = exploration::evaluate_exploration(state, player_id, genes);

    // --- 4. Game Stage Weighting ---
    let max_turn = state.settings.max_turns.max(1) as f32;
    let current_turn = state.settings.turn as f32;
    let progress = (current_turn / max_turn).clamp(0.0, 1.0);

    // Early Game
    let (w_eco, w_mil, w_exp, w_fow) = if progress < genes.stages.early_threshold {
        if state.settings.mode == crate::types::ModeType::Perfection {
            (genes.evaluator.early_perf_eco, genes.evaluator.early_perf_mil, genes.evaluator.early_perf_exp, genes.evaluator.early_perf_fow)
        } else {
            (genes.evaluator.early_dom_eco, genes.evaluator.early_dom_mil, genes.evaluator.early_dom_exp, genes.evaluator.early_dom_fow)
        }
    }
    // Mid Game
    else if progress < genes.stages.late_threshold {
        if state.settings.mode == crate::types::ModeType::Perfection {
            (genes.evaluator.mid_perf_eco, genes.evaluator.mid_perf_mil, genes.evaluator.mid_perf_exp, genes.evaluator.mid_perf_fow)
        } else {
            (genes.evaluator.mid_dom_eco, genes.evaluator.mid_dom_mil, genes.evaluator.mid_dom_exp, genes.evaluator.mid_dom_fow)
        }
    }
    // End Game
    else {
        if state.settings.mode == crate::types::ModeType::Perfection {
            (genes.evaluator.end_perf_eco, genes.evaluator.end_perf_mil, genes.evaluator.end_perf_exp, genes.evaluator.end_perf_fow)
        } else {
            (genes.evaluator.end_dom_eco, genes.evaluator.end_dom_mil, genes.evaluator.end_dom_exp, genes.evaluator.end_dom_fow)
        }
    };

    let final_score =
        (eco_score * w_eco) + (exp_score * w_exp) + (mil_score * w_mil) + (fow_score * w_fow);

    final_score.clamp(0.0, 1.0)
}
