use crate::ai::brain::max_turns_ahead;
use crate::ai::eval_server::Evaluator;
use crate::ai::features::{self, RawFeatures};
use crate::ai::mcts_common::{
    self, BackpropNode, LeafData, TreeNode, backpropagate_and_remove_virtual_loss,
    extract_leaf_data, get_node_by_path, get_node_by_path_mut,
};
use crate::ai::network::RawPolicyOutput;
use crate::game::Game;
use crate::moves::EndTurnMove;
use crate::moves::Move;
use crate::types::MoveType;

use std::cell::RefCell;

pub struct ZeroMctsAgent<'a> {
    pub evaluator: &'a Evaluator,
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

    /// Value in this node's PARENT's perspective. `backpropagate_and_remove_virtual_loss`
    /// stores each node's value in its own player's perspective, so a child
    /// reached across a handover must be negated before siblings are compared —
    /// otherwise the parent picks the move that is best for the opponent.
    fn effective_value_for_parent(&self, virtual_loss_value: f32) -> f32 {
        let vl = *self.virtual_loss.borrow();
        let denom = self.visits + vl;
        if denom == 0.0 {
            return 0.0;
        }
        let sum = if mcts_common::edge_hands_over(self.move_to_here.as_deref()) {
            -self.value_sum
        } else {
            self.value_sum
        };
        (sum + vl * virtual_loss_value) / denom
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
                let a_value = a.effective_value_for_parent(virtual_loss_value);
                let a_score = a_value + c_puct * a.prior * sqrt_n / (1.0 + a_visits);

                let b_visits = b.effective_visits();
                let b_value = b.effective_value_for_parent(virtual_loss_value);
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
}

impl TreeNode for ZeroNode {
    fn children(&self) -> &[Self] {
        &self.children
    }
    fn children_mut(&mut self) -> &mut [Self] {
        &mut self.children
    }
}

impl BackpropNode for ZeroNode {
    fn visits_mut(&mut self) -> &mut f32 {
        &mut self.visits
    }
    fn value_sum_mut(&mut self) -> &mut f32 {
        &mut self.value_sum
    }
    fn virtual_loss(&self) -> &RefCell<f32> {
        &self.virtual_loss
    }
}

impl<'a> ZeroMctsAgent<'a> {
    pub fn new(evaluator: &'a Evaluator, iterations: usize) -> Self {
        Self {
            evaluator,
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
            let mut rng = rand::rng();
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
            let mut rng = rand::rng();
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
    /// 0 (current): openings always play the search recommendation — random
    /// map seeds supply enough data diversity, and visit-sampling was adding
    /// ~a turn of noise to opening metrics (was 15).
    pub const TEMPERATURE_MOVE_THRESHOLD: usize = 0;

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
            let mut rng = rand::rng();
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
                    reward_type: selected_move.reward_type().ok(),
                };
                return (Some(selected_move), vec![move_info]);
            }
        }

        let start_turn = game.state.settings.turn;
        let mut root = ZeroNode::new(1.0, None);
        self.expand_node_single(&mut root, game, false);

        // Add Dirichlet noise to root priors for diverse exploration during training
        if root.children.len() > 1 {
            use rand::distr::Distribution;
            use rand_distr::Gamma;
            // Alpha 0.3 is standard for Chess (~30 moves). Polytopia has variable moves but 0.3 is a safe default.
            // In polytopia by move ~7 it ramps upto 80!
            let alpha = 0.3;
            let epsilon = 0.25; // 25% noise
            let gamma = Gamma::new(alpha, 1.0).unwrap();
            let mut noise: Vec<f32> = (0..root.children.len())
                .map(|_| gamma.sample(&mut rand::rng()))
                .collect();
            let sum: f32 = noise.iter().sum();
            if sum > 0.0 {
                for n in &mut noise { *n /= sum; }
            } else {
                for n in &mut noise { *n = 1.0 / root.children.len() as f32; }
            }

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
                    reward_type: m.reward_type().ok(),
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
            use rand::distr::{Distribution, weighted::WeightedIndex};
            let weights: Vec<f32> = root.children.iter().map(|c| c.visits.max(0.0)).collect();
            if let Ok(dist) = WeightedIndex::new(&weights) {
                best_idx = dist.sample(&mut rand::rng());
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
        let turn_horizon = start_turn + max_turns_ahead(start_turn, game.state.settings.max_turns);
        let mut leaves: Vec<LeafData> = Vec::with_capacity(batch_size);

        // Phase 1: Select leaves sequentially, using undo to restore game state
        for _ in 0..batch_size {
            if let Some(leaf) = self.select_and_extract_leaf(root, game, turn_horizon) {
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
        let mut eval_batch: Vec<RawFeatures> = Vec::new();

        // Initialize values: use terminal values where available
        let mut values = vec![0.0f32; leaves.len()];
        for (i, leaf) in leaves.iter().enumerate() {
            if let Some(terminal_val) = leaf.terminal_value {
                // Terminal node with known outcome
                values[i] = terminal_val;
            } else if let Some(ref feat) = leaf.features {
                // Non-terminal, needs NN evaluation.
                indices_needing_eval.push(i);
                eval_batch.push(RawFeatures {
                    spatial: feat.spatial.clone(),
                    player: feat.player.clone(),
                });
            }
            // else: no features and no terminal value -> stays 0.0 (shouldn't happen)
        }

        if !indices_needing_eval.is_empty() {
            let results = self.evaluator.evaluate(eval_batch);

            for (local_idx, &global_idx) in indices_needing_eval.iter().enumerate() {
                let (value, _progress, ref policy_row) = results[local_idx];
                values[global_idx] = value;

                // Expand node using pre-computed data
                let leaf = &leaves[global_idx];
                let node = get_node_by_path_mut(root, &leaf.path_indices).unwrap();
                self.expand_node_from_precomputed(
                    node,
                    leaf.legal_moves.take(),
                    leaf.map_size,
                    true,
                    policy_row,
                );
            }
        }

        // Phase 3: Backpropagate and remove virtual loss
        for (leaf, &value) in leaves.iter().zip(values.iter()) {
            backpropagate_and_remove_virtual_loss(
                root,
                &leaf.path_indices,
                &leaf.path_players,
                self.virtual_loss,
                value,
            );
        }
    }

    /// Select a leaf node, extract all needed data, and undo back to root.
    /// Returns None if no valid leaf can be selected.
    fn select_and_extract_leaf(
        &self,
        root: &ZeroNode,
        game: &mut Game,
        turn_horizon: i32,
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
                let current = get_node_by_path(root, &indices_stack)?;
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
        let leaf_data = extract_leaf_data(game, indices_stack, path_players, needs_expansion);

        // --- Undo all moves back to root state ---
        while let Some(undo) = undos.pop() {
            undo(&mut game.state);
        }

        Some(leaf_data)
    }

    fn expand_node_single(&self, node: &mut ZeroNode, game: &Game, allow_end_turn: bool) {
        let features = features::state_to_cpu_features(
            &game.state,
            game.state.settings.current_player_turn_id,
        )
        .expect("BUG: Failed to create features in MCTS expand_node");

        let results = self.evaluator.evaluate(vec![features]);
        let (_value, _progress, ref policy_row) = results[0];

        // Expand
        self.expand_node_from_network_output(node, game, allow_end_turn, policy_row);
    }

    fn expand_node_from_network_output(
        &self,
        node: &mut ZeroNode,
        game: &Game,
        allow_end_turn: bool,
        policy: &RawPolicyOutput,
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
        policy: &RawPolicyOutput,
    ) {
        if node.is_expanded {
            return;
        }

        if legal_moves.is_empty() {
            node.is_expanded = true;
            return;
        }

        let priors = crate::ai::policy_composer::compute_move_priors_raw(
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
