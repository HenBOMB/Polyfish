use crate::ai::features::state_to_tensor;
use crate::ai::mapper::ActionMapper;
use crate::ai::network::PolyZeroNet;
use crate::game::Game;
use crate::moves::EndTurnMove;
use crate::moves::Move;
use crate::types::MoveType;

use candle_core::Tensor;
use std::cell::RefCell;

pub struct ZeroMctsAgent<'a> {
    pub network: &'a PolyZeroNet,
    pub iterations: usize,
    pub c_puct: f32,
    pub batch_size: usize,
    pub virtual_loss: f32,
}

struct ZeroNode {
    pub visits: f32,
    pub value_sum: f32,
    pub prior: f32,
    pub children: Vec<ZeroNode>,
    pub move_to_here: Option<Box<dyn Move>>,
    pub is_expanded: bool,
    // Virtual loss for parallel search
    pub virtual_loss: RefCell<f32>,
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
            virtual_loss: RefCell::new(0.0),
        }
    }

    /// Get effective visit count including virtual loss
    fn effective_visits(&self) -> f32 {
        self.visits + *self.virtual_loss.borrow()
    }

    /// Get effective value including virtual loss penalty
    fn effective_value(&self, virtual_loss_value: f32) -> f32 {
        let vl = *self.virtual_loss.borrow();
        if self.visits + vl == 0.0 {
            0.0
        } else {
            (self.value_sum + vl * virtual_loss_value) / (self.visits + vl)
        }
    }

    fn select_child_with_virtual_loss(
        &self,
        c_puct: f32,
        virtual_loss_value: f32,
    ) -> Option<usize> {
        let sqrt_n = self.effective_visits().sqrt();

        self.children
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                let a_visits = a.effective_visits();
                let a_value = a.effective_value(virtual_loss_value);
                let a_score = a_value + c_puct * a.prior * sqrt_n / (1.0 + a_visits);

                let b_visits = b.effective_visits();
                let b_value = b.effective_value(virtual_loss_value);
                let b_score = b_value + c_puct * b.prior * sqrt_n / (1.0 + b_visits);

                a_score.partial_cmp(&b_score).unwrap()
            })
            .map(|(idx, _)| idx)
    }

    /// Add virtual loss to this node
    fn add_virtual_loss(&self, amount: f32) {
        *self.virtual_loss.borrow_mut() += amount;
    }

    /// Remove virtual loss from this node
    fn remove_virtual_loss(&self, amount: f32) {
        *self.virtual_loss.borrow_mut() -= amount;
    }
}

/// Represents a path through the tree with indices
struct SearchPath {
    indices: Vec<usize>,
}

impl SearchPath {
    fn new() -> Self {
        Self {
            indices: Vec::new(),
        }
    }

    fn push(&mut self, idx: usize) {
        self.indices.push(idx);
    }
}

impl<'a> ZeroMctsAgent<'a> {
    pub fn new(network: &'a PolyZeroNet, iterations: usize) -> Self {
        Self {
            network,
            iterations,
            c_puct: 1.0,
            batch_size: 8,
            virtual_loss: 1.0,
        }
    }

    pub fn select_move(&self, game: &mut Game) -> Option<Box<dyn Move>> {
        let mut root = ZeroNode::new(1.0, None);
        self.expand_node(&mut root, game, false);

        // Parallel search with batching
        let mut iteration = 0;
        while iteration < self.iterations {
            let batch_count = (self.iterations - iteration).min(self.batch_size);
            self.parallel_search_batch(game, &mut root, batch_count);
            iteration += batch_count;
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

        // Parallel search with batching
        let mut iteration = 0;
        while iteration < self.iterations {
            let batch_count = (self.iterations - iteration).min(self.batch_size);
            self.parallel_search_batch(game, &mut root, batch_count);
            iteration += batch_count;
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

    /// Perform a batch of parallel searches using virtual loss
    fn parallel_search_batch(&self, game: &mut Game, root: &mut ZeroNode, batch_size: usize) {
        let mut paths = Vec::new();
        let mut leaf_games = Vec::new();
        let mut leaf_nodes_info = Vec::new();

        // Phase 1: Select leaves in parallel using virtual loss
        for _ in 0..batch_size {
            let mut path = SearchPath::new();
            let mut current_game = game.clone();

            if let Some((leaf_info, final_game)) =
                self.select_leaf_with_virtual_loss(root, &mut current_game, &mut path)
            {
                paths.push(path);
                leaf_games.push(final_game);
                leaf_nodes_info.push(leaf_info);
            } else {
                break;
            }
        }

        // Phase 2: Expand all leaves (could be batched in GPU version)
        let mut values = Vec::new();
        for (leaf_info, leaf_game) in leaf_nodes_info.iter().zip(leaf_games.iter()) {
            let value = if leaf_info.needs_expansion {
                self.expand_node_at_path(root, &leaf_info.path_indices, leaf_game, true)
            } else {
                0.0 // Terminal node
            };
            values.push(value);
        }

        // Phase 3: Backpropagate and remove virtual loss
        for (path, value) in paths.iter().zip(values.iter()) {
            self.backpropagate_and_remove_virtual_loss(root, &path.indices, *value);
        }
    }

    /// Select a leaf node using virtual loss, returning path and game state
    fn select_leaf_with_virtual_loss(
        &self,
        root: &ZeroNode,
        game: &mut Game,
        path: &mut SearchPath,
    ) -> Option<(LeafInfo, Game)> {
        let current = root;
        let mut indices_stack = Vec::new();

        loop {
            // Add virtual loss to current node
            current.add_virtual_loss(self.virtual_loss);

            // Check terminal condition
            if game.state.settings._game_over {
                return Some((
                    LeafInfo {
                        needs_expansion: false,
                        path_indices: indices_stack,
                    },
                    game.clone(),
                ));
            }

            // Check if node needs expansion
            if !current.is_expanded {
                return Some((
                    LeafInfo {
                        needs_expansion: true,
                        path_indices: indices_stack,
                    },
                    game.clone(),
                ));
            }

            // Check if terminal node (expanded but no children)
            if current.children.is_empty() {
                return Some((
                    LeafInfo {
                        needs_expansion: false,
                        path_indices: indices_stack,
                    },
                    game.clone(),
                ));
            }

            // Select best child with virtual loss
            let child_idx =
                current.select_child_with_virtual_loss(self.c_puct, -self.virtual_loss)?;

            // Apply move
            if let Some(m) = &current.children[child_idx].move_to_here {
                if let Some(_undo) = game.play_move(m.as_ref()) {
                    // Don't store undo - we'll clone game states instead
                    indices_stack.push(child_idx);
                    path.push(child_idx);

                    // Move to child (careful with borrow checker)
                    // We can't hold a mutable reference, so we'll navigate by index
                    break;
                } else {
                    return None;
                }
            } else {
                return None;
            }
        }

        // Continue navigation using indices
        self.continue_selection_by_indices(root, game, path, &mut indices_stack)
    }

    /// Continue tree traversal using indices instead of references
    fn continue_selection_by_indices(
        &self,
        root: &ZeroNode,
        game: &mut Game,
        path: &mut SearchPath,
        indices_stack: &mut Vec<usize>,
    ) -> Option<(LeafInfo, Game)> {
        loop {
            let current = self.get_node_by_path(root, indices_stack)?;

            current.add_virtual_loss(self.virtual_loss);

            if game.state.settings._game_over {
                return Some((
                    LeafInfo {
                        needs_expansion: false,
                        path_indices: indices_stack.clone(),
                    },
                    game.clone(),
                ));
            }

            if !current.is_expanded {
                return Some((
                    LeafInfo {
                        needs_expansion: true,
                        path_indices: indices_stack.clone(),
                    },
                    game.clone(),
                ));
            }

            if current.children.is_empty() {
                return Some((
                    LeafInfo {
                        needs_expansion: false,
                        path_indices: indices_stack.clone(),
                    },
                    game.clone(),
                ));
            }

            let child_idx =
                current.select_child_with_virtual_loss(self.c_puct, -self.virtual_loss)?;

            if let Some(m) = &current.children[child_idx].move_to_here {
                if game.play_move(m.as_ref()).is_some() {
                    indices_stack.push(child_idx);
                    path.push(child_idx);
                } else {
                    return None;
                }
            } else {
                return None;
            }
        }
    }

    /// Get a node by following a path of indices
    fn get_node_by_path<'b>(&self, root: &'b ZeroNode, indices: &[usize]) -> Option<&'b ZeroNode> {
        let mut current = root;
        for &idx in indices {
            current = current.children.get(idx)?;
        }
        Some(current)
    }

    /// Get a mutable node by following a path of indices
    fn get_node_by_path_mut<'b>(
        &self,
        root: &'b mut ZeroNode,
        indices: &[usize],
    ) -> Option<&'b mut ZeroNode> {
        let mut current = root;
        for &idx in indices {
            current = current.children.get_mut(idx)?;
        }
        Some(current)
    }

    /// Expand a node at a specific path
    fn expand_node_at_path(
        &self,
        root: &mut ZeroNode,
        indices: &[usize],
        game: &Game,
        allow_end_turn: bool,
    ) -> f32 {
        let node = match self.get_node_by_path_mut(root, indices) {
            Some(n) => n,
            None => return 0.0,
        };

        self.expand_node(node, game, allow_end_turn)
    }

    /// Backpropagate value and remove virtual loss along path
    fn backpropagate_and_remove_virtual_loss(
        &self,
        root: &mut ZeroNode,
        indices: &[usize],
        mut value: f32,
    ) {
        // Update root
        root.remove_virtual_loss(self.virtual_loss);
        root.visits += 1.0;
        root.value_sum += value;

        // Update each node along the path
        let mut current = root;
        for &idx in indices {
            value = -value; // Flip value for opponent

            if let Some(child) = current.children.get_mut(idx) {
                child.remove_virtual_loss(self.virtual_loss);
                child.visits += 1.0;
                child.value_sum += value;
                current = child;
            } else {
                break;
            }
        }
    }

    fn expand_node(&self, node: &mut ZeroNode, game: &Game, allow_end_turn: bool) -> f32 {
        let device = self.network.device();
        let input = match state_to_tensor(
            &game.state,
            game.state.settings.current_player_turn_id,
            &device,
        ) {
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

struct LeafInfo {
    needs_expansion: bool,
    path_indices: Vec<usize>,
}

fn move_or_end_turn(best_move: Option<Box<dyn Move>>) -> Option<Box<dyn Move>> {
    if best_move.is_none() {
        Some(Box::new(EndTurnMove))
    } else {
        best_move
    }
}
