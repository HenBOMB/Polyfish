use crate::ai::gumbel_mcts::GumbelMctsAgent;
use crate::ai::mcts_types::MoveVisit;
use crate::ai::mcts_zero::ZeroMctsAgent;
use crate::ai::network::PolyZeroNet;
use crate::game::Game;
use crate::moves::{Move, generate_legal_moves};

// class brain
pub struct Brain<'a> {
    pub network: &'a PolyZeroNet,
    pub max_iterations: usize,
}

impl<'a> Brain<'a> {
    pub fn new(network: &'a PolyZeroNet, max_iterations: usize) -> Self {
        Self {
            network,
            max_iterations,
        }
    }

    fn get_iterations(&self, turn: i32, legal_move_count: usize) -> usize {
        let mut iterations = self.max_iterations;

        if legal_move_count == 1 {
            return 0;
        }

        if legal_move_count < 4 {
            iterations = 10;
        } else if turn < 3 || legal_move_count < 10 {
            iterations = 25;
        } else if turn < 6 || legal_move_count < 20 {
            iterations = 50;
        } else if turn < 10 || legal_move_count < 30 {
            iterations = 80;
        }

        iterations
    }

    pub fn think_decomposed(&self, game: &Game) -> (Option<Box<dyn Move>>, Vec<MoveVisit>) {
        let mut moves = generate_legal_moves(&game.state);

        if moves.len() == 1 {
            return (moves.pop(), Vec::new());
        }

        let agent = ZeroMctsAgent::new(
            self.network,
            self.get_iterations(game.state.settings.turn, moves.len()),
        );

        agent.select_move_with_decomposed_visits(&mut game.clone())
    }

    pub fn think_with_stats(&self, game: &Game) -> (Option<Box<dyn Move>>, Vec<f32>) {
        let mut moves = generate_legal_moves(&game.state);

        if moves.len() == 1 {
            return (moves.pop(), Vec::new());
        }

        let agent = ZeroMctsAgent::new(
            self.network,
            self.get_iterations(game.state.settings.turn, moves.len()),
        );

        agent.select_move_with_stats(&mut game.clone())
    }
}

/// Returns the maximum number of game turns the MCTS tree should look ahead
/// from the current turn. This prevents the search from going absurdly deep
/// and getting stuck in long rollouts during mid-game when branching is high.
pub fn max_turns_ahead(current_turn: i32, max_turns: i32) -> i32 {
    let is_last_turn = current_turn >= max_turns;
    // +1 because we want to include the current turn in the lookahead
    1 + match current_turn {
        1 => 2,
        2 => 2,
        3 => 2,
        4 => 2,
        5 => 2,
        6 => 2,
        7 => 1,
        8 => 1,
        9 => 1,
        10 => {
            if is_last_turn {
                0
            } else {
                2
            }
        }
        11 => 1,
        12 => 1,
        13 => 1,
        _ => 1, // 14+
    }
}
