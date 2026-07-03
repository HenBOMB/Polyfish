use crate::ai::brain::max_turns_ahead;
use crate::ai::features::{self, GameFeatures, state_to_tensor};
use crate::ai::network::{PolicyOutput, PolyZeroNet};
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

                a_score.partial_cmp(&b_score).unwrap_or_else(|| {
                    panic!(
                        "NaN in UCB score comparison: a_score={} (value={}, prior={}, visits={}), \
                         b_score={} (value={}, prior={}, visits={}) — network is producing NaN, \
                         this must be fixed at the source, not masked here",
                        a_score, a_value, a.prior, a_visits, b_score, b_value, b.prior, b_visits
                    )
                })
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

/// Pre-computed data extracted at a leaf node before undoing moves.
/// This replaces the old approach of cloning the entire `Game` at each leaf.
struct LeafData {
    path_indices: Vec<usize>,
    /// Player ID at each node along the path (same length as path_indices)
    path_players: Vec<i32>,
    /// Feature tensors for NN evaluation (None if leaf is terminal)
    features: Option<GameFeatures>,
    /// Legal moves at this leaf (wrapped in RefCell for interior mutability during take())
    legal_moves: RefCell<Vec<Box<dyn Move>>>,

    /// Map size at leaf state
    map_size: usize,
    /// Terminal value if this is a game-over state (Some = terminal with known outcome)
    terminal_value: Option<f32>,
}

impl<'a> ZeroMctsAgent<'a> {
    pub fn new(network: &'a PolyZeroNet, iterations: usize) -> Self {
        Self {
            network,
            iterations,
            c_puct: 1.0,
            batch_size: 24,
            virtual_loss: 1.0,
        }
    }

    pub fn select_move(&self, game: &mut Game) -> Option<Box<dyn Move>> {
        // 1. Check Opening Book
        use crate::ai::book::Book;
        use rand::seq::SliceRandom;
        // recommend returns Vec<Box<dyn Move>>.
        // We can't use .choose() because that returns &Box<dyn Move> which we can't clone.
        // Instead, we shuffle and pop.
        let mut book_moves = Book::recommend(game);
        if !book_moves.is_empty() {
            let mut rng = rand::thread_rng();
            book_moves.shuffle(&mut rng);
            if let Some(m) = book_moves.pop() {
                return Some(m);
            }
        }

        let start_turn = game.state.settings.turn;
        let mut root = ZeroNode::new(1.0, None);
        // Initial expansion (single)
        self.expand_node_single(&mut root, game, false);

        let mut iteration = 0;
        while iteration < self.iterations {
            let batch_count = (self.iterations - iteration).min(self.batch_size);
            self.parallel_search_batch(game, &mut root, batch_count, start_turn);
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
        // 1. Check Opening Book
        use crate::ai::book::Book;
        use rand::seq::SliceRandom;

        // We need to handle book moves but also return valid stats (policy) matching the legal moves order.
        let mut book_moves = Book::recommend(game);
        if !book_moves.is_empty() {
            let mut rng = rand::thread_rng();
            book_moves.shuffle(&mut rng);
            if let Some(book_move) = book_moves.pop() {
                // To return correct policy vector, we must know the legal moves order.
                // So we expand the root node once.
                let mut root = ZeroNode::new(1.0, None);
                self.expand_node_single(&mut root, game, false);

                // Find the index of the book move in the children
                let num_children = root.children.len();
                let mut policy = vec![0.0f32; num_children.max(1)];

                let mut found_idx = None;
                for (i, child) in root.children.iter().enumerate() {
                    if let Some(m) = &child.move_to_here {
                        // Compare move types and critical fields
                        // Simple equality might not work if Move doesn't implement partialEq correctly for all types,
                        // but typically we can check describe() or similar.
                        // For now, let's assume filtering in `recommend` ensured it's a legal move.
                        // We key off move_type and target for strictness, or just trust `recommend` returned a valid match.
                        // Let's match by description to be safe? Or simple type + indices.
                        if m.describe(&game.state) == book_move.describe(&game.state) {
                            found_idx = Some(i);
                            break;
                        }
                    }
                }

                if let Some(idx) = found_idx {
                    policy[idx] = 1.0;
                    return (Some(book_move), policy);
                }
                // If not found (weird), fall through to normal MCTS
            }
        }

        let start_turn = game.state.settings.turn;
        let mut root = ZeroNode::new(1.0, None);
        self.expand_node_single(&mut root, game, false);

        let mut iteration = 0;
        while iteration < self.iterations {
            let batch_count = (self.iterations - iteration).min(self.batch_size);
            self.parallel_search_batch(game, &mut root, batch_count, start_turn);
            iteration += batch_count;
        }

        // Generate visit count distribution for policy
        let num_children = root.children.len();
        let mut best_idx = 0;
        let mut max_visits = -1.0;

        for (i, child) in root.children.iter().enumerate() {
            if child.visits > max_visits {
                max_visits = child.visits;
                best_idx = i;
            }
        }

        // Create policy from visit counts
        let mut policy = vec![0.0f32; num_children.max(1)];
        for (i, child) in root.children.iter().enumerate() {
            policy[i] = child.visits;
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

    /// Number of plies at the start of a game to sample proportional to
    /// visit counts rather than always taking argmax, for training diversity.
    pub const TEMPERATURE_MOVE_THRESHOLD: usize = 20;

    /// Selects a move for policy training and returns decomposed visit counts; samples by visit for early moves.
    pub fn select_move_with_decomposed_visits(
        &self,
        game: &mut Game,
        move_count: usize,
    ) -> (Option<Box<dyn Move>>, Vec<crate::ai::mcts_types::MoveVisit>) {
        use crate::ai::mcts_types::MoveVisit;

        // 1. Check Opening Book
        use crate::ai::book::Book;
        use rand::seq::SliceRandom;

        let mut book_moves = Book::recommend(game);
        if !book_moves.is_empty() {
            let mut rng = rand::thread_rng();
            book_moves.shuffle(&mut rng);
            if let Some(selected_move) = book_moves.pop() {
                // Create MoveVisit for this move with 100% probability (iterations count)
                let move_info = MoveVisit {
                    move_type: selected_move.move_type(),
                    visits: self.iterations as f32,
                    source_idx: selected_move.source_idx().ok(),
                    target_idx: selected_move.target_idx().ok(),
                    structure_type: selected_move.structure_type().ok(),
                    unit_type: selected_move.unit_type().ok(),
                    tech_type: selected_move.tech_type().ok(),
                    ability_type: selected_move.ability_type().ok(),
                };
                return (Some(selected_move), vec![move_info]);
            }
        }

        let start_turn = game.state.settings.turn;
        let mut root = ZeroNode::new(1.0, None);
        self.expand_node_single(&mut root, game, false);

        // Add Dirichlet noise to root priors for diverse exploration during training
        if root.children.len() > 1 {
            use rand_distr::{Dirichlet, Distribution};
            // Alpha 0.3 is standard for Chess (~30 moves). Polytopia has variable moves but 0.3 is a safe default.
            // In polytopia by move ~7 it ramps upto 80!
            let alpha = 0.3;
            let epsilon = 0.25; // 25% noise
            let dirichlet = Dirichlet::new_with_size(alpha, root.children.len()).unwrap();
            let noise = dirichlet.sample(&mut rand::thread_rng());

            for (child, n) in root.children.iter_mut().zip(noise.iter()) {
                child.prior = (1.0 - epsilon) * child.prior + epsilon * n;
            }
        }

        let mut iteration = 0;
        while iteration < self.iterations {
            let batch_count = (self.iterations - iteration).min(self.batch_size);
            self.parallel_search_batch(game, &mut root, batch_count, start_turn);
            iteration += batch_count;
        }

        // Extract move visit information (decomposed components, no cloning needed)
        let mut move_visits = Vec::new();
        let mut best_idx = 0;
        let mut max_visits = -1.0;

        for (i, child) in root.children.iter().enumerate() {
            if let Some(ref m) = child.move_to_here {
                // Extract decomposed information from the move
                let move_info = MoveVisit {
                    move_type: m.move_type(),
                    visits: child.visits,
                    source_idx: m.source_idx().ok(),
                    target_idx: m.target_idx().ok(),
                    structure_type: m.structure_type().ok(),
                    unit_type: m.unit_type().ok(),
                    tech_type: m.tech_type().ok(),
                    ability_type: m.ability_type().ok(),
                };
                move_visits.push(move_info);

                if child.visits > max_visits {
                    max_visits = child.visits;
                    best_idx = i;
                }
            }
        }

        // in early game, sample proportional to visit counts instead of argmax
        if move_count < Self::TEMPERATURE_MOVE_THRESHOLD && root.children.len() > 1 {
            use rand::distributions::{Distribution, WeightedIndex};
            let weights: Vec<f32> = root.children.iter().map(|c| c.visits.max(0.0)).collect();
            if let Ok(dist) = WeightedIndex::new(&weights) {
                best_idx = dist.sample(&mut rand::thread_rng());
            }
        }

        // Extract best move
        let best_move = if !root.children.is_empty() && best_idx < root.children.len() {
            root.children.swap_remove(best_idx).move_to_here
        } else {
            None
        };

        (move_or_end_turn(best_move), move_visits)
    }

    /// Perform a batch of parallel searches using virtual loss
    fn parallel_search_batch(
        &self,
        game: &mut Game,
        root: &mut ZeroNode,
        batch_size: usize,
        start_turn: i32,
    ) {
        let device = self.network.device();
        let turn_horizon = start_turn + max_turns_ahead(start_turn, game.state.settings.max_turns);
        let mut leaves: Vec<LeafData> = Vec::with_capacity(batch_size);

        // Phase 1: Select leaves sequentially, using undo to restore game state
        for _ in 0..batch_size {
            if let Some(leaf) = self.select_and_extract_leaf(root, game, turn_horizon, &device) {
                leaves.push(leaf);
            } else {
                break;
            }
        }

        if leaves.is_empty() {
            return;
        }

        // Phase 2: Batched NN evaluation for leaves that need expansion
        let mut indices_needing_eval: Vec<usize> = Vec::new();
        let mut spatial_list = Vec::new();
        let mut player_list = Vec::new();

        // Initialize values: use terminal values where available
        let mut values = vec![0.0f32; leaves.len()];
        for (i, leaf) in leaves.iter().enumerate() {
            if let Some(terminal_val) = leaf.terminal_value {
                // Terminal node with known outcome
                values[i] = terminal_val;
            } else if let Some(ref feat) = leaf.features {
                // Non-terminal, needs NN evaluation
                indices_needing_eval.push(i);
                spatial_list.push(feat.spatial_map.clone());
                player_list.push(feat.player_state.clone());
            }
            // else: no features and no terminal value -> stays 0.0 (shouldn't happen)
        }

        if !indices_needing_eval.is_empty() {
            // Stack tensors
            if let (Ok(batch_spatial), Ok(batch_player)) =
                (Tensor::cat(&spatial_list, 0), Tensor::cat(&player_list, 0))
            {
                let batch_spatial = batch_spatial
                    .reshape((
                        indices_needing_eval.len(),
                        features::NUM_CHANNELS,
                        features::MAP_SIZE,
                        features::MAP_SIZE,
                    ))
                    .unwrap();
                let batch_player = batch_player
                    .reshape((indices_needing_eval.len(), 10))
                    .unwrap();

                // Run NN inference
                if let Ok((policy_out, value_out)) =
                    self.network.forward_t(&batch_spatial, &batch_player, false)
                {
                    let win_values = value_out
                        .win_value
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap();

                    for (local_idx, &global_idx) in indices_needing_eval.iter().enumerate() {
                        values[global_idx] = win_values[local_idx];

                        // Slice policy for this leaf
                        let slice_policy = PolicyOutput {
                            action_type: policy_out
                                .action_type
                                .get(local_idx)
                                .unwrap()
                                .unsqueeze(0)
                                .unwrap(),
                            source_spatial: policy_out
                                .source_spatial
                                .get(local_idx)
                                .unwrap()
                                .unsqueeze(0)
                                .unwrap(),
                            target_spatial: policy_out
                                .target_spatial
                                .get(local_idx)
                                .unwrap()
                                .unsqueeze(0)
                                .unwrap(),
                            move_option: policy_out
                                .move_option
                                .get(local_idx)
                                .unwrap()
                                .unsqueeze(0)
                                .unwrap(),
                        };

                        // Expand node using pre-computed data
                        let leaf = &leaves[global_idx];
                        let node = self.get_node_by_path_mut(root, &leaf.path_indices).unwrap();
                        self.expand_node_from_precomputed(
                            node,
                            leaf.legal_moves.take(),
                            leaf.map_size,
                            true,
                            &slice_policy,
                        );
                    }
                }
            } else {
                panic!("BUG: Failed to stack tensors for MCTS batch");
            }
        }

        // Phase 3: Backpropagate and remove virtual loss
        for (leaf, &value) in leaves.iter().zip(values.iter()) {
            self.backpropagate_and_remove_virtual_loss(root, &leaf.path_indices, &leaf.path_players, value);
        }
    }

    /// Select a leaf node, extract all needed data, and undo back to root.
    /// Returns None if no valid leaf can be selected.
    fn select_and_extract_leaf(
        &self,
        root: &ZeroNode,
        game: &mut Game,
        turn_horizon: i32,
        device: &candle_core::Device,
    ) -> Option<LeafData> {
        let mut indices_stack: Vec<usize> = Vec::new();
        let mut path_players: Vec<i32> = Vec::new();
        let mut undos: Vec<crate::actions::UndoCallback> = Vec::new();

        // Capture root player
        let root_player = game.state.settings.current_player_turn_id;
        path_players.push(root_player);

        // Start at root
        let current = root;
        current.add_virtual_loss(self.virtual_loss);

        // First iteration (root → first child) with direct reference
        let leaf_result = loop {
            // Terminal check - separate from horizon
            if game.state.settings._game_over {
                break Some(false); // needs_expansion = false (terminal)
            }
            if game.state.settings.turn > turn_horizon {
                break Some(false); // needs_expansion = false (horizon)
            }
            if !current.is_expanded {
                break Some(true); // needs_expansion = true
            }
            if current.children.is_empty() {
                break Some(false);
            }

            // Select child
            let child_idx =
                current.select_child_with_virtual_loss(self.c_puct, -self.virtual_loss)?;

            // Apply move
            let m = current.children[child_idx].move_to_here.as_ref()?;
            let undo = game.simulate_move(m.as_ref());
            let undo = match undo {
                Some(u) => u,
                None => {
                    let stars = game.current_tribe().map(|t| t.stars).unwrap_or(-1);
                    let desc = m.describe(&game.state);
                    let turn = game.state.settings.turn;
                    let pid = game.state.settings.current_player_turn_id;
                    panic!(
                        "BUG: Legal move failed in MCTS selection.\nMove: {}\nTurn: {}, PID: {}, Stars: {}",
                        desc, turn, pid, stars
                    );
                }
            };
            undos.push(undo);
            indices_stack.push(child_idx);

            // Player after the move (may have changed due to EndTurn)
            path_players.push(game.state.settings.current_player_turn_id);

            // Can't keep `current` borrow, switch to index-based traversal
            break None; // signal: continue via indices
        };

        // Continue traversal by index if needed
        let needs_expansion = if let Some(ne) = leaf_result {
            ne
        } else {
            // Index-based traversal loop
            loop {
                let current = self.get_node_by_path(root, &indices_stack)?;
                current.add_virtual_loss(self.virtual_loss);

                if game.state.settings._game_over {
                    break false; // terminal
                }
                if game.state.settings.turn > turn_horizon {
                    break false; // horizon
                }
                if !current.is_expanded {
                    break true;
                }
                if current.children.is_empty() {
                    break false;
                }

                let child_idx =
                    current.select_child_with_virtual_loss(self.c_puct, -self.virtual_loss)?;

                let m = current.children[child_idx].move_to_here.as_ref()?;
                let result = game.simulate_move(m.as_ref());
                if result.is_none() {
                    let pov_id = game.state.settings.current_player_turn_id;
                    eprintln!("\n=== MOVE EXECUTION FAILED ===");
                    eprintln!("Move: {}", m.describe(&game.state));
                    eprintln!("Turn: {}", game.state.settings.turn);
                    eprintln!("Current player: {}", pov_id);
                    eprintln!("Indices stack: {:?}", indices_stack);
                    eprintln!("=============================\n");
                }
                // // Print all recent moves
                // for mv in game.state.settings._recent_moves.iter() {
                //     eprintln!("[BUG] Recent Move: {:?}", mv);
                // }
                let undo =
                    result.expect("[BUG] Legal move failed to execute in MCTS tree traversal");
                undos.push(undo);
                indices_stack.push(child_idx);

                // Capture player after move
                path_players.push(game.state.settings.current_player_turn_id);
            }
        };

        // --- At the leaf: extract data before undoing ---
        let leaf_data = if game.state.settings._game_over {
            // Terminal state: calculate actual outcome
            let current_player = game.state.settings.current_player_turn_id;
            let my_score = game.state.tribes.get(&current_player)
                .map(|t| t.score)
                .unwrap_or(0);

            // Find opponent's best score
            let opp_best_score = game.state.tribes.iter()
                .filter(|(id, _)| **id != current_player)
                .map(|(_, t)| t.score)
                .max()
                .unwrap_or(0);

            let outcome = if my_score > opp_best_score {
                1.0  // Win
            } else if my_score < opp_best_score {
                -1.0  // Loss
            } else {
                0.0  // Draw
            };

            LeafData {
                path_indices: indices_stack,
                path_players,
                features: None,
                legal_moves: RefCell::new(Vec::new()),
                map_size: game.state.settings.size as usize,
                terminal_value: Some(outcome),
            }
        } else if needs_expansion {
            // Non-terminal, needs expansion: evaluate with NN
            let map_size = game.state.settings.size as usize;

            // Feature tensors for NN evaluation
            let feat = state_to_tensor(
                &game.state,
                game.state.settings.current_player_turn_id,
                device,
            )
            .expect("BUG: Failed to create features at MCTS leaf");

            // Legal moves + heuristic scores
            let mut legal_moves = game.legal_moves();
            // In batched expansion we always allow EndTurn (allow_end_turn=true)
            let has_other = legal_moves
                .iter()
                .any(|m| m.move_type() != MoveType::EndTurn);
            if has_other {
                legal_moves.retain(|m| m.move_type() != MoveType::EndTurn);
            }
            LeafData {
                path_indices: indices_stack,
                path_players,
                features: Some(feat),
                legal_moves: RefCell::new(legal_moves),
                map_size,
                terminal_value: None,
            }
        } else {
            // Horizon reached: evaluate with NN (not terminal!)
            let map_size = game.state.settings.size as usize;

            let feat = state_to_tensor(
                &game.state,
                game.state.settings.current_player_turn_id,
                device,
            )
            .ok();

            LeafData {
                path_indices: indices_stack,
                path_players,
                features: feat,
                legal_moves: RefCell::new(Vec::new()),
                map_size,
                terminal_value: None,
            }
        };

        // --- Undo all moves back to root state ---
        while let Some(undo) = undos.pop() {
            undo(&mut game.state);
        }

        Some(leaf_data)
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

    /// Backpropagate value and remove virtual loss along path
    fn backpropagate_and_remove_virtual_loss(
        &self,
        root: &mut ZeroNode,
        indices: &[usize],
        path_players: &[i32],
        mut value: f32,
    ) {
        // Anchor: the leaf player's perspective
        let leaf_player = path_players.last().copied().unwrap_or(1);

        // Root player is the first in the path
        let root_player = path_players.first().copied().unwrap_or(leaf_player);

        // Root's value from its perspective
        let root_value = if root_player != leaf_player {
            -value
        } else {
            value
        };

        root.remove_virtual_loss(self.virtual_loss);
        root.visits += 1.0;
        root.value_sum += root_value;

        // Update each child node along the path
        let mut current = root;
        let mut prev_player = root_player;

        for (i, &idx) in indices.iter().enumerate() {
            if let Some(child) = current.children.get_mut(idx) {
                // Determine this child's player
                // path_players[i] is the parent's player, path_players[i+1] is the child's
                let child_player = if i + 1 < path_players.len() {
                    path_players[i + 1]
                } else {
                    leaf_player
                };

                // If player changed, flip the sign
                if child_player != prev_player {
                    value = -value;
                }

                child.remove_virtual_loss(self.virtual_loss);
                child.visits += 1.0;
                child.value_sum += value;

                current = child;
                prev_player = child_player;
            } else {
                break;
            }
        }
    }

    fn expand_node_single(&self, node: &mut ZeroNode, game: &Game, allow_end_turn: bool) {
        let device = self.network.device();
        let features = state_to_tensor(
            &game.state,
            game.state.settings.current_player_turn_id,
            &device,
        )
        .expect("BUG: Failed to create features in MCTS expand_node");

        let (policy_output, _value_output) = self
            .network
            .forward_t(&features.spatial_map, &features.player_state, false)
            .expect("BUG: Network forward pass failed in MCTS");

        // Expand
        self.expand_node_from_network_output(node, game, allow_end_turn, &policy_output);
    }

    fn expand_node_from_network_output(
        &self,
        node: &mut ZeroNode,
        game: &Game,
        allow_end_turn: bool,
        policy: &PolicyOutput,
    ) {
        if node.is_expanded {
            return;
        }

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
            return;
        }

        let map_size = game.state.settings.size as usize;
        self.expand_node_from_precomputed(node, legal_moves, map_size, allow_end_turn, policy);
    }

    /// Expand a node using pre-computed legal moves and heuristic scores.
    /// Used by both `expand_node_from_network_output` (initial root expansion)
    /// and batched leaf expansion (where data was pre-computed before undo).
    fn expand_node_from_precomputed(
        &self,
        node: &mut ZeroNode,
        legal_moves: Vec<Box<dyn Move>>,
        map_size: usize,
        allow_end_turn: bool,
        policy: &PolicyOutput,
    ) {
        if node.is_expanded {
            return;
        }

        if legal_moves.is_empty() {
            node.is_expanded = true;
            return;
        }

        let priors = crate::ai::policy_composer::compute_move_priors(
            policy,
            &legal_moves,
            map_size,
            allow_end_turn,
        );

        // Normalize
        let sum: f32 = priors.iter().sum();
        let normalized_priors: Vec<f32> = if sum > 1e-8 {
            priors.iter().map(|p| p / sum).collect()
        } else {
            vec![1.0 / priors.len() as f32; priors.len()]
        };

        for (m, prior) in legal_moves.into_iter().zip(normalized_priors.iter()) {
            node.children.push(ZeroNode::new(*prior, Some(m)));
        }

        node.is_expanded = true;
    }
}

// LeafData is defined above (near SearchPath replacement)

fn move_or_end_turn(best_move: Option<Box<dyn Move>>) -> Option<Box<dyn Move>> {
    if best_move.is_none() {
        Some(Box::new(EndTurnMove))
    } else {
        best_move
    }
}
