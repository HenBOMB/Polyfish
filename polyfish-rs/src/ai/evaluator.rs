pub mod army;
pub mod economy;
pub mod expansion;
pub mod exploration;
pub mod gamestate;
pub mod player;
pub mod research;

use crate::ai::genes::AIGenes;
use crate::states::{GameState, PlayerId};

/// Evaluates a game state for a given player using RELATIVE evaluation.
/// Returns (MyScore - MaxOpponentScore). Range roughly -1.0 to 1.0.
/// This is a convenience wrapper around `gamestate::evaluate_gamestate`.
pub fn evaluate_state(state: &GameState, player_id: PlayerId, genes: &AIGenes) -> f32 {
    gamestate::evaluate_gamestate(state, player_id, genes)
}
