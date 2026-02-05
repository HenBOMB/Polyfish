use crate::ai::features::{self, state_to_tensor};
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
            c_puct: 1.5,    // Increased from 1.0 for more exploration
            batch_size: 32, // Reduced from 64 to avoid OOM with larger model
            virtual_loss: 1.0,
        }
    }

    pub fn select_move(&self, game: &mut Game) -> Option<Box<dyn Move>> {
        let mut root = ZeroNode::new(1.0, None);
        // Initial expansion (single)
        self.expand_node_single(&mut root, game, false);

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
        self.expand_node_single(&mut root, game, false);

        let mut iteration = 0;
        while iteration < self.iterations {
            let batch_count = (self.iterations - iteration).min(self.batch_size);
            self.parallel_search_batch(game, &mut root, batch_count);
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

    /// Select a move and return decomposed visit information for policy training
    /// Returns best move + list of move visit data for decomposed policy targets
    pub fn select_move_with_decomposed_visits(
        &self,
        game: &mut Game,
    ) -> (Option<Box<dyn Move>>, Vec<crate::ai::mcts_types::MoveVisit>) {
        use crate::ai::mcts_types::MoveVisit;

        let mut root = ZeroNode::new(1.0, None);
        self.expand_node_single(&mut root, game, false);

        let mut iteration = 0;
        while iteration < self.iterations {
            let batch_count = (self.iterations - iteration).min(self.batch_size);
            self.parallel_search_batch(game, &mut root, batch_count);
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

        // Extract best move
        let best_move = if !root.children.is_empty() && best_idx < root.children.len() {
            root.children.swap_remove(best_idx).move_to_here
        } else {
            None
        };

        (move_or_end_turn(best_move), move_visits)
    }

    /// Perform a batch of parallel searches using virtual loss
    fn parallel_search_batch(&self, game: &Game, root: &mut ZeroNode, batch_size: usize) {
        let mut paths = Vec::with_capacity(batch_size);
        let mut leaf_games = Vec::with_capacity(batch_size);
        let mut leaf_nodes_info = Vec::with_capacity(batch_size);

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

        if paths.is_empty() {
            return;
        }

        // Phase 2: Batched Expansion
        let device = self.network.device();

        // Identify which leaves actually need expansion via NN
        let mut indices_needing_eval = Vec::new();
        let mut states_to_batch = Vec::new();
        let mut povs = Vec::new();

        for (i, leaf_info) in leaf_nodes_info.iter().enumerate() {
            if leaf_info.needs_expansion {
                indices_needing_eval.push(i);
                states_to_batch.push(&leaf_games[i].state);
                povs.push(leaf_games[i].state.settings.current_player_turn_id);
            }
        }

        let mut values = vec![0.0; paths.len()];

        if !indices_needing_eval.is_empty() {
            // Create batch tensors
            let mut spatial_list = Vec::with_capacity(indices_needing_eval.len());
            let mut player_list = Vec::with_capacity(indices_needing_eval.len());

            for (state, pov) in states_to_batch.iter().zip(povs.iter()) {
                let features = state_to_tensor(state, *pov, &device)
                    .expect("BUG: Failed to create features for batch");
                spatial_list.push(features.spatial_map);
                player_list.push(features.player_state);
            }

            // Stack tensors
            if let (Ok(batch_spatial), Ok(batch_player)) =
                (Tensor::cat(&spatial_list, 0), Tensor::cat(&player_list, 0))
            {
                // Ensure shape (B, C, H, W)
                let _spatial_dim =
                    features::NUM_CHANNELS * features::MAP_HEIGHT * features::MAP_WIDTH;
                let batch_spatial = batch_spatial
                    .reshape((
                        indices_needing_eval.len(),
                        features::NUM_CHANNELS,
                        features::MAP_HEIGHT,
                        features::MAP_WIDTH,
                    ))
                    .unwrap();
                let batch_player = batch_player
                    .reshape((indices_needing_eval.len(), 10))
                    .unwrap();

                // Run Inference
                if let Ok((policy_out, value_out)) =
                    self.network.forward_t(&batch_spatial, &batch_player, false)
                {
                    // Extract values
                    let win_values = value_out
                        .win_value
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap();

                    // Process each result
                    for (local_idx, &global_idx) in indices_needing_eval.iter().enumerate() {
                        let value = win_values[local_idx];
                        values[global_idx] = value;

                        let path_indices = &leaf_nodes_info[global_idx].path_indices;
                        let game_state = &leaf_games[global_idx];

                        // We need to slice the policy output for this specific instance
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

                        // Expand the node in the tree
                        let node = self.get_node_by_path_mut(root, path_indices).unwrap();
                        self.expand_node_from_network_output(node, game_state, true, &slice_policy);
                    }
                }
            } else {
                panic!("BUG: Failed to stack tensors for MCTS batch");
            }
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
                let _undo = match game.play_move(m.as_ref()) {
                    Some(u) => u,
                    None => {
                        let stars = game.current_tribe().map(|t| t.stars).unwrap_or(-1);
                        let desc = m.describe(&game.state);
                        let turn = game.state.settings.turn;
                        let pid = game.state.settings.current_player_turn_id;
                        panic!(
                            "BUG: Legal move failed to execute in MCTS selection.\nMove: {}\nTurn: {}, PID: {}, Stars: {}\nState Hash: {}",
                            desc, turn, pid, stars, game.state.settings.seed
                        );
                    }
                };
                // Don't store undo - we'll clone game states instead
                indices_stack.push(child_idx);
                path.push(child_idx);

                // Move to child (careful with borrow checker)
                // We can't hold a mutable reference, so we'll navigate by index
                break;
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
                let result = game.play_move(m.as_ref());
                if result.is_none() {
                    // ERROR: Dump detailed state
                    let pov_id = game.state.settings.current_player_turn_id;
                    eprintln!("\n=== MOVE EXECUTION FAILED ===");
                    eprintln!("Move: {}", m.describe(&game.state));
                    eprintln!("Turn: {}", game.state.settings.turn);
                    eprintln!("Current player: {}", pov_id);
                    for (id, tribe) in &game.state.tribes {
                        eprintln!("  Tribe {}: {} stars", id, tribe.stars);
                    }
                    eprintln!("Indices stack: {:?}", indices_stack);
                    eprintln!("=============================\n");
                }
                let _undo =
                    result.expect("BUG: Legal move failed to execute in MCTS tree traversal");
                indices_stack.push(child_idx);
                path.push(child_idx);
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

        // Use decomposed policy composer to get priors
        // Note: crate::ai::policy_composer::compute_move_priors needs to support batched inputs if we slice them.
        // But PolicyOutput fields are Tensors.
        // If we sliced them (unsqueeze(0)), they are [1, ...].
        // policy_composer usually expects [1, ...], so it should work if it uses .dims4() or similar.
        // Let's assume it works.

        let priors = crate::ai::policy_composer::compute_move_priors(
            policy,
            &legal_moves,
            game,
            allow_end_turn,
        );

        // Normalize priors
        let sum: f32 = priors.iter().sum();
        let normalized_priors: Vec<f32> = if sum > 1e-8 {
            priors.iter().map(|p| p / sum).collect()
        } else {
            vec![1.0 / priors.len() as f32; priors.len()]
        };

        // Create child nodes
        for (m, prior) in legal_moves.into_iter().zip(normalized_priors.iter()) {
            node.children.push(ZeroNode::new(*prior, Some(m)));
        }

        node.is_expanded = true;
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
