// use crate::ai::gumbel_mcts::GumbelMctsAgent;
use crate::ai::mcts_types::MoveVisit;
use crate::ai::mcts_zero::ZeroMctsAgent;
use crate::ai::network::PolyZeroNet;
use crate::game::Game;
use crate::moves::{Move, generate_legal_moves};
use crate::ai::mcts_zero::ZeroNode;

// class brain
pub struct Brain<'a> {
    pub network: &'a PolyZeroNet,
    pub max_iterations: usize,
    pub root: Option<ZeroNode>,
}

impl<'a> Brain<'a> {
    pub fn new(network: &'a PolyZeroNet, max_iterations: usize) -> Self {
        Self {
            network,
            max_iterations,
            root: None,
        }
    }

    fn _get_iterations(&self, turn: i32, legal_move_count: usize) -> usize {
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

    fn think(&self, game: &Game) -> (Option<ZeroMctsAgent<'a>>, Vec<Box<dyn Move>>) {
        let moves = generate_legal_moves(&game.state);

        if moves.len() == 1 {
            return (None, moves);
        }

        let agent = ZeroMctsAgent::new(
            self.network,
            self.max_iterations, // self.get_iterations(game.state.settings.turn, moves.len()),
        );

        (Some(agent), moves)
    }

    pub fn think_decomposed(&mut self, game: &Game) -> (Option<Box<dyn Move>>, Vec<MoveVisit>) {
        let (agent, mut moves) = self.think(game);

        if agent.is_none() {
            self.root = None;
            return (moves.pop(), Vec::new());
        }

        let root = self.root.take().unwrap_or_else(|| ZeroNode::new(1.0, None));
        let (best_move, visits, new_root) = agent
            .unwrap()
            .select_move_with_decomposed_visits(&mut game.clone_for_mcts(game.state.settings.current_player_turn_id), root);
            
        if let Some(ref m) = best_move {
            use crate::types::MoveType;
            if m.move_type() == MoveType::EndTurn || m.move_type() == MoveType::Resign {
                self.root = None;
            } else {
                self.root = Some(new_root);
            }
        } else {
            self.root = None;
        }

        (best_move, visits)
    }

    pub fn think_with_stats(&mut self, game: &Game) -> (Option<Box<dyn Move>>, Vec<f32>) {
        let (agent, mut moves) = self.think(game);

        if agent.is_none() {
            self.root = None;
            return (moves.pop(), Vec::new());
        }

        let root = self.root.take().unwrap_or_else(|| ZeroNode::new(1.0, None));
        let (best_move, policy, new_root) = agent
            .unwrap()
            .select_move_with_stats(&mut game.clone_for_mcts(game.state.settings.current_player_turn_id), root);
            
        if let Some(ref m) = best_move {
            use crate::types::MoveType;
            if m.move_type() == MoveType::EndTurn || m.move_type() == MoveType::Resign {
                self.root = None;
            } else {
                self.root = Some(new_root);
            }
        } else {
            self.root = None;
        }

        (best_move, policy)
    }
}

/// Returns the maximum number of game turns the MCTS tree should look ahead
/// from the current turn. This prevents the search from going absurdly deep
/// and getting stuck in long rollouts during mid-game when branching is high.
pub fn max_turns_ahead(_current_turn: i32, _max_turns: i32) -> i32 {
    if _current_turn < 8 {
        5
    } else {
        (20 - _current_turn).max(2).min(20) // idea: do not look ahead more than the last turn (20 turns, default)
    }
}
