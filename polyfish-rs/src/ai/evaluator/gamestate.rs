use super::player;
use crate::ai::genes::AIGenes;
use crate::states::{GameState, PlayerId};

/// Evaluates a game state for a given player using RELATIVE evaluation.
/// Returns (MyScore - MaxOpponentScore). Range roughly -1.0 to 1.0.
pub fn evaluate_gamestate(state: &GameState, player_id: PlayerId, genes: &AIGenes) -> f32 {
    let my_score = player::evaluate_player(state, player_id, genes);

    // If I'm dead or invalid, return lowest possible score
    if my_score < 0.0 {
        return -1.0;
    }

    let mut max_opponent_score = 0.0;
    for &opponent_id in state.tribes.keys() {
        if opponent_id != player_id {
            let opp_score = player::evaluate_player(state, opponent_id, genes);
            if opp_score > max_opponent_score {
                max_opponent_score = opp_score;
            }
        }
    }

    // Return relative score
    (my_score - max_opponent_score).clamp(-1.0, 1.0)
}
