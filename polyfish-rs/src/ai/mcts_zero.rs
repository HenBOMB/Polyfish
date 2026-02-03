use crate::ai::features::state_to_tensor;
use crate::ai::mapper::ActionMapper;
use crate::ai::network::PolyZeroNet;
use crate::game::Game;
use crate::moves::EndTurnMove;
use crate::moves::Move;
use crate::types::MoveType;

use candle_core::Tensor;
pub struct ZeroMctsAgent<'a> {
    pub network: &'a PolyZeroNet,
    pub iterations: usize,
    pub c_puct: f32,
}

struct ZeroNode {
    pub visits: f32,    // N
    pub value_sum: f32, // W
    pub prior: f32,     // P
    pub children: Vec<ZeroNode>,
    pub move_to_here: Option<Box<dyn Move>>,
    pub is_expanded: bool,
}

impl ZeroNode {
    fn new(prior: f32, move_to_here: Option<Box<dyn Move>>) -> Self {
        Self {
            visits: 0.0,
            value_sum: 0.0,
            prior,
            children: Vec::new(),
            move_to_here,
            is_expanded: false,
        }
    }

    fn value(&self) -> f32 {
        if self.visits == 0.0 {
            0.0
        } else {
            self.value_sum / self.visits
        }
    }

    fn select_child(&mut self, c_puct: f32) -> Option<&mut ZeroNode> {
        let sqrt_n = self.visits.sqrt();

        self.children.iter_mut().max_by(|a, b| {
            let a_score = a.value() + c_puct * a.prior * sqrt_n / (1.0 + a.visits);
            let b_score = b.value() + c_puct * b.prior * sqrt_n / (1.0 + b.visits);
            a_score.partial_cmp(&b_score).unwrap()
        })
    }
}

impl<'a> ZeroMctsAgent<'a> {
    pub fn new(network: &'a PolyZeroNet, iterations: usize) -> Self {
        Self {
            network,
            iterations,
            c_puct: 1.0,
        }
    }

    pub fn select_move(&self, game: &mut Game) -> Option<Box<dyn Move>> {
        let mut root = ZeroNode::new(1.0, None);
        self.expand_node(&mut root, game, false);

        for _ in 0..self.iterations {
            self.search(game, &mut root);
        }

        let best_move = root
            .children
            .into_iter()
            .max_by(|a, b| a.visits.partial_cmp(&b.visits).unwrap())
            .and_then(|n| n.move_to_here);

        move_or_end_turn(best_move)
    }

    pub fn select_move_with_stats(&self, game: &mut Game) -> (Option<Box<dyn Move>>, Vec<f32>) {
        let mut root = ZeroNode::new(1.0, None);
        self.expand_node(&mut root, game, false);

        for _ in 0..self.iterations {
            self.search(game, &mut root);
        }

        // Generate Policy Vector
        let mut policy = vec![0.0; ActionMapper::TOTAL_ACTIONS];
        let map_size = game.state.settings.size;

        let mut best_idx = 0;
        let mut max_visits = -1.0;

        for (i, child) in root.children.iter().enumerate() {
            if let Some(m) = &child.move_to_here {
                if let Some(idx) = ActionMapper::move_to_idx(map_size, m.as_ref()) {
                    if idx < policy.len() {
                        policy[idx] = child.visits;
                    }
                }

                if child.visits > max_visits {
                    max_visits = child.visits;
                    best_idx = i;
                }
            }
        }

        // Normalize policy
        let sum: f32 = policy.iter().sum();
        if sum > 0.0 {
            for p in policy.iter_mut() {
                *p /= sum;
            }
        }

        // Extract best move owned
        let best_move = if !root.children.is_empty() {
            if best_idx < root.children.len() {
                root.children.swap_remove(best_idx).move_to_here
            } else {
                None
            }
        } else {
            None
        };

        (move_or_end_turn(best_move), policy)
    }

    fn search(&self, game: &mut Game, node: &mut ZeroNode) -> f32 {
        if game.state.settings._game_over {
            let _pov = game.state.settings.current_player_turn_id;
            return 0.0;
        }

        if !node.is_expanded {
            let value = self.expand_node(node, game, true);
            return value;
        }

        if node.children.is_empty() {
            return 0.0;
        }

        let child = node.select_child(self.c_puct).unwrap();
        if let Some(m) = &child.move_to_here {
            // do NOT simulate here, cause mcts zero runs with FOW=Disabled
            // TODO after training is complete and we have competent model, SIMULATE MOVE HERE INSTEAD!
            if let Some(undo) = game.play_move(m.as_ref()) {
                let val = -self.search(game, child);
                undo(&mut game.state);

                node.visits += 1.0;
                node.value_sum += val;
                return val;
            }
        }

        0.0
    }

    fn expand_node(&self, node: &mut ZeroNode, game: &Game, allow_end_turn: bool) -> f32 {
        let input = match state_to_tensor(&game.state, game.state.settings.current_player_turn_id) {
            Ok(t) => t,
            Err(_) => return 0.0,
        };

        let (policy_logits, value_tensor): (Tensor, Tensor) =
            match self.network.forward_t(&input, false) {
                Ok(res) => res,
                Err(_) => return 0.0,
            };

        let value = value_tensor
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0];

        let mut legal_moves = game.legal_moves();

        if !allow_end_turn {
            let has_other_moves = legal_moves
                .iter()
                .any(|m| m.move_type() != MoveType::EndTurn);
            if has_other_moves {
                legal_moves.retain(|m| m.move_type() != MoveType::EndTurn);
            }
        }

        if legal_moves.is_empty() {
            node.is_expanded = true;
            return value;
        }

        let mut child_priors = Vec::new();
        let prob_vec = policy_logits
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let map_size = game.state.settings.size;

        for m in legal_moves {
            if let Some(idx) = ActionMapper::move_to_idx(map_size, m.as_ref()) {
                if idx < prob_vec.len() {
                    let logit = prob_vec[idx];
                    child_priors.push((m, logit));
                }
            }
        }

        let max_logit = child_priors
            .iter()
            .map(|(_, l)| *l)
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0;
        let mut blobs = Vec::new();
        for (m, l) in child_priors {
            let p = (l - max_logit).exp();
            sum_exp += p;
            blobs.push((m, p));
        }

        for (m, p_raw) in blobs {
            let prior = p_raw / sum_exp;
            node.children.push(ZeroNode::new(prior, Some(m)));
        }

        node.is_expanded = true;
        value
    }
}

fn move_or_end_turn(best_move: Option<Box<dyn Move>>) -> Option<Box<dyn Move>> {
    if best_move.is_none() {
        // If we filtered out EndTurn but found no other moves, default to End Turn
        // We assume EndTurn is legal.
        Some(Box::new(EndTurnMove))
    } else {
        best_move
    }
}
