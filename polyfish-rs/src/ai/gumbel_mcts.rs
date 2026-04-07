use crate::ai::brain::max_turns_ahead;
use crate::ai::features::{self, GameFeatures, state_to_tensor};
use crate::ai::network::PolyZeroNet;
use crate::ai::policy_composer;
use crate::game::Game;
use crate::moves::{EndTurnMove, Move};
use crate::types::MoveType;
use candle_core::Tensor;
use rand::distributions::Distribution;
use rand_distr::Gumbel;
use std::cell::RefCell;

pub struct GumbelMctsAgent<'a> {
    pub network: &'a PolyZeroNet,
    pub iterations: usize,
    pub k: usize, // Number of actions to sample at root
    pub batch_size: usize,
}

struct GumbelNode {
    pub visits: f32,
    pub value_sum: f32,
    pub logit: f32,
    pub gumbel: f32,
    pub children: Vec<GumbelNode>,
    pub move_to_here: Option<Box<dyn Move>>,
    pub is_expanded: bool,
    pub virtual_loss: RefCell<f32>,
}

impl GumbelNode {
    fn new(logit: f32, gumbel: f32, move_to_here: Option<Box<dyn Move>>) -> Self {
        Self {
            visits: 0.0,
            value_sum: 0.0,
            logit,
            gumbel,
            children: Vec::new(),
            move_to_here,
            is_expanded: false,
            virtual_loss: RefCell::new(0.0),
        }
    }

    fn q_value(&self) -> f32 {
        if self.visits == 0.0 {
            0.0
        } else {
            self.value_sum / self.visits
        }
    }

    fn z_score(&self) -> f32 {
        self.logit + self.gumbel + self.q_value()
    }

    fn effective_visits(&self) -> f32 {
        self.visits + *self.virtual_loss.borrow()
    }

    fn effective_value(&self, virtual_loss_value: f32) -> f32 {
        let vl = *self.virtual_loss.borrow();
        if self.visits + vl == 0.0 {
            0.0
        } else {
            (self.value_sum + vl * virtual_loss_value) / (self.visits + vl)
        }
    }

    /// Selection rule for non-root nodes: argmax(logit + Q)
    /// We use a simplified version of the paper's formula.
    fn select_child_in_tree(&self, virtual_loss_value: f32) -> Option<usize> {
        self.children
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                // Simplified selection: logit + Q
                // In practice, adding an exploration term helps.
                let a_score = a.logit + a.effective_value(virtual_loss_value);
                let b_score = b.logit + b.effective_value(virtual_loss_value);
                a_score.partial_cmp(&b_score).unwrap()
            })
            .map(|(idx, _)| idx)
    }
}

struct LeafData {
    path_indices: Vec<usize>,
    features: Option<GameFeatures>,
    legal_moves: RefCell<Vec<Box<dyn Move>>>,
    map_size: usize,
}

impl<'a> GumbelMctsAgent<'a> {
    pub fn new(network: &'a PolyZeroNet, iterations: usize) -> Self {
        Self {
            network,
            iterations,
            k: 16,
            batch_size: 24,
        }
    }

    pub fn select_move(&self, game: &mut Game) -> Option<Box<dyn Move>> {
        let start_turn = game.state.settings.turn;
        let device = self.network.device();

        // 1. Initial expansion at root to get logits for all legal moves
        let features = state_to_tensor(
            &game.state,
            game.state.settings.current_player_turn_id,
            &device,
        )
        .ok()?;
        let (policy_out, _value_out) = self
            .network
            .forward(&features.spatial_map, &features.player_state)
            .ok()?;

        let legal_moves = game.legal_moves();
        if legal_moves.is_empty() {
            return Some(Box::new(EndTurnMove));
        }

        let map_size = game.state.settings.size as usize;
        let logits = policy_composer::compute_move_log_probs(&policy_out, &legal_moves, map_size);

        // 2. Sample Gumbel noise and select top-k actions
        let mut rng = rand::thread_rng();
        let gumbel_dist = Gumbel::new(0.0, 1.0).unwrap();

        let mut root_candidates: Vec<GumbelNode> = legal_moves
            .into_iter()
            .zip(logits.into_iter())
            .map(|(m, l)| {
                let g = gumbel_dist.sample(&mut rng);
                GumbelNode::new(l, g, Some(m))
            })
            .collect();

        // Sort by logit + gumbel and take top k
        root_candidates.sort_by(|a, b| {
            (b.logit + b.gumbel)
                .partial_cmp(&(a.logit + a.gumbel))
                .unwrap()
        });
        if root_candidates.len() > self.k {
            root_candidates.truncate(self.k);
        }

        let mut root = GumbelNode::new(0.0, 0.0, None);
        root.children = root_candidates;
        root.is_expanded = true;

        // 3. Sequential Halving phases
        let num_phases = (self.k as f32).log2().ceil() as usize;
        let iterations_per_phase = self.iterations / num_phases.max(1);

        for phase in 0..num_phases {
            let active_count = root.children.len();
            if active_count <= 1 {
                break;
            }

            // Run iterations for this phase
            let mut phase_iterations = 0;
            while phase_iterations < iterations_per_phase {
                let batch_count = (iterations_per_phase - phase_iterations).min(self.batch_size);
                self.parallel_search_batch(game, &mut root, batch_count, start_turn);
                phase_iterations += batch_count;
            }

            // Prune bottom half
            if phase < num_phases - 1 {
                root.children
                    .sort_by(|a, b| b.z_score().partial_cmp(&a.z_score()).unwrap());
                let kept_count = (active_count + 1) / 2;
                root.children.truncate(kept_count);
            }
        }

        // Final selection: highest z_score
        let best_idx = root
            .children
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.z_score().partial_cmp(&b.z_score()).unwrap())
            .map(|(i, _)| i)?;

        root.children.swap_remove(best_idx).move_to_here
    }

    pub fn select_move_with_decomposed_visits(
        &self,
        game: &mut Game,
    ) -> (Option<Box<dyn Move>>, Vec<crate::ai::mcts_types::MoveVisit>) {
        use crate::ai::mcts_types::MoveVisit;
        let start_turn = game.state.settings.turn;
        let device = self.network.device();

        // 1. Initial expansion at root to get logits for all legal moves
        let features = match state_to_tensor(
            &game.state,
            game.state.settings.current_player_turn_id,
            &device,
        ) {
            Ok(f) => f,
            Err(_) => return (Some(Box::new(EndTurnMove)), Vec::new()),
        };
        let (policy_out, _value_out) = match self
            .network
            .forward(&features.spatial_map, &features.player_state)
        {
            Ok(res) => res,
            Err(_) => return (Some(Box::new(EndTurnMove)), Vec::new()),
        };

        let legal_moves = game.legal_moves();
        if legal_moves.is_empty() {
            return (Some(Box::new(EndTurnMove)), Vec::new());
        }

        let map_size = game.state.settings.size as usize;
        let logits = policy_composer::compute_move_log_probs(&policy_out, &legal_moves, map_size);

        // 2. Sample Gumbel noise and select top-k actions
        let mut rng = rand::thread_rng();
        let gumbel_dist = Gumbel::new(0.0, 1.0).unwrap();

        let mut root_candidates: Vec<GumbelNode> = legal_moves
            .into_iter()
            .zip(logits.into_iter())
            .map(|(m, l)| {
                let g = gumbel_dist.sample(&mut rng);
                GumbelNode::new(l, g, Some(m))
            })
            .collect();

        // Sort by logit + gumbel and take top k
        root_candidates.sort_by(|a, b| {
            (b.logit + b.gumbel)
                .partial_cmp(&(a.logit + a.gumbel))
                .unwrap()
        });
        if root_candidates.len() > self.k {
            root_candidates.truncate(self.k);
        }

        let mut root = GumbelNode::new(0.0, 0.0, None);
        root.children = root_candidates;
        root.is_expanded = true;

        // 3. Sequential Halving phases
        let num_phases = (self.k as f32).log2().ceil() as usize;
        let iterations_per_phase = self.iterations / num_phases.max(1);

        for phase in 0..num_phases {
            let active_count = root.children.len();
            if active_count <= 1 {
                break;
            }

            let mut phase_iterations = 0;
            while phase_iterations < iterations_per_phase {
                let batch_count = (iterations_per_phase - phase_iterations).min(self.batch_size);
                self.parallel_search_batch(game, &mut root, batch_count, start_turn);
                phase_iterations += batch_count;
            }

            if phase < num_phases - 1 {
                root.children
                    .sort_by(|a, b| b.z_score().partial_cmp(&a.z_score()).unwrap());
                let kept_count = (active_count + 1) / 2;
                root.children.truncate(kept_count);
            }
        }

        // Final selection
        let best_idx = root
            .children
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.z_score().partial_cmp(&b.z_score()).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Extract visits for policy training
        let mut move_visits = Vec::new();
        for child in &root.children {
            if let Some(ref m) = child.move_to_here {
                move_visits.push(MoveVisit {
                    move_type: m.move_type(),
                    visits: child.visits,
                    source_idx: m.source_idx().ok(),
                    target_idx: m.target_idx().ok(),
                    structure_type: m.structure_type().ok(),
                    unit_type: m.unit_type().ok(),
                    tech_type: m.tech_type().ok(),
                    ability_type: m.ability_type().ok(),
                });
            }
        }

        let best_move = if !root.children.is_empty() {
            root.children.swap_remove(best_idx).move_to_here
        } else {
            None
        };

        (
            if best_move.is_none() {
                Some(Box::new(EndTurnMove))
            } else {
                best_move
            },
            move_visits,
        )
    }

    pub fn select_move_with_stats(&self, game: &mut Game) -> (Option<Box<dyn Move>>, Vec<f32>) {
        let start_turn = game.state.settings.turn;
        let device = self.network.device();

        // 1. Initial expansion at root to get logits for all legal moves
        let features = match state_to_tensor(
            &game.state,
            game.state.settings.current_player_turn_id,
            &device,
        ) {
            Ok(f) => f,
            Err(_) => return (Some(Box::new(EndTurnMove)), Vec::new()),
        };
        let (policy_out, _value_out) = match self
            .network
            .forward(&features.spatial_map, &features.player_state)
        {
            Ok(res) => res,
            Err(_) => return (Some(Box::new(EndTurnMove)), Vec::new()),
        };

        let legal_moves = game.legal_moves();
        if legal_moves.is_empty() {
            return (Some(Box::new(EndTurnMove)), Vec::new());
        }

        let map_size = game.state.settings.size as usize;
        let logits = policy_composer::compute_move_log_probs(&policy_out, &legal_moves, map_size);

        // 2. Sample Gumbel noise and select top-k actions
        let mut rng = rand::thread_rng();
        let gumbel_dist = Gumbel::new(0.0, 1.0).unwrap();

        let mut root_candidates: Vec<GumbelNode> = legal_moves
            .into_iter()
            .zip(logits.into_iter())
            .map(|(m, l)| {
                let g = gumbel_dist.sample(&mut rng);
                GumbelNode::new(l, g, Some(m))
            })
            .collect();

        // Sort by logit + gumbel and take top k
        root_candidates.sort_by(|a, b| {
            (b.logit + b.gumbel)
                .partial_cmp(&(a.logit + a.gumbel))
                .unwrap()
        });
        if root_candidates.len() > self.k {
            root_candidates.truncate(self.k);
        }

        let mut root = GumbelNode::new(0.0, 0.0, None);
        root.children = root_candidates;
        root.is_expanded = true;

        // 3. Sequential Halving phases
        let num_phases = (self.k as f32).log2().ceil() as usize;
        let iterations_per_phase = self.iterations / num_phases.max(1);

        for phase in 0..num_phases {
            let active_count = root.children.len();
            if active_count <= 1 {
                break;
            }

            let mut phase_iterations = 0;
            while phase_iterations < iterations_per_phase {
                let batch_count = (iterations_per_phase - phase_iterations).min(self.batch_size);
                self.parallel_search_batch(game, &mut root, batch_count, start_turn);
                phase_iterations += batch_count;
            }

            if phase < num_phases - 1 {
                root.children
                    .sort_by(|a, b| b.z_score().partial_cmp(&a.z_score()).unwrap());
                let kept_count = (active_count + 1) / 2;
                root.children.truncate(kept_count);
            }
        }

        // Final selection
        let best_idx = root
            .children
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.z_score().partial_cmp(&b.z_score()).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Extract policy distribution from visits
        let mut policy = vec![0.0f32; root.children.len()];
        let mut sum_visits = 0.0;
        for (i, child) in root.children.iter().enumerate() {
            policy[i] = child.visits;
            sum_visits += child.visits;
        }

        if sum_visits > 0.0 {
            for p in policy.iter_mut() {
                *p /= sum_visits;
            }
        }

        let best_move = if !root.children.is_empty() {
            root.children.swap_remove(best_idx).move_to_here
        } else {
            None
        };

        (
            if best_move.is_none() {
                Some(Box::new(EndTurnMove))
            } else {
                best_move
            },
            policy,
        )
    }

    fn parallel_search_batch(
        &self,
        game: &mut Game,
        root: &mut GumbelNode,
        batch_size: usize,
        start_turn: i32,
    ) {
        let device = self.network.device();
        let turn_horizon = start_turn + max_turns_ahead(start_turn, game.state.settings.max_turns);
        let mut leaves: Vec<LeafData> = Vec::with_capacity(batch_size);

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

        // Batched evaluation
        let mut indices_needing_eval = Vec::new();
        let mut spatial_list = Vec::new();
        let mut player_list = Vec::new();

        for (i, leaf) in leaves.iter().enumerate() {
            if let Some(ref feat) = leaf.features {
                indices_needing_eval.push(i);
                spatial_list.push(feat.spatial_map.clone());
                player_list.push(feat.player_state.clone());
            }
        }

        let mut values = vec![0.0f32; leaves.len()];

        if !indices_needing_eval.is_empty() {
            if let (Ok(batch_spatial), Ok(batch_player)) =
                (Tensor::cat(&spatial_list, 0), Tensor::cat(&player_list, 0))
            {
                let n = indices_needing_eval.len();
                let b_s = batch_spatial
                    .reshape((
                        n,
                        features::NUM_CHANNELS,
                        features::MAP_SIZE,
                        features::MAP_SIZE,
                    ))
                    .unwrap();
                let b_p = batch_player.reshape((n, 10)).unwrap();

                if let Ok((policy_out, value_out)) = self.network.forward(&b_s, &b_p) {
                    let win_values = value_out
                        .win_value
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap();

                    for (local_idx, &global_idx) in indices_needing_eval.iter().enumerate() {
                        values[global_idx] = win_values[local_idx];

                        let node = self
                            .get_node_by_path_mut(root, &leaves[global_idx].path_indices)
                            .unwrap();
                        let leaf = &leaves[global_idx];

                        // Expand node
                        let slice_policy = crate::ai::network::PolicyOutput {
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

                        let legal_moves = leaf.legal_moves.take();
                        if legal_moves.is_empty() {
                            node.is_expanded = true;
                            continue;
                        }

                        let logits = policy_composer::compute_move_log_probs(
                            &slice_policy,
                            &legal_moves,
                            leaf.map_size,
                        );
                        for (m, l) in legal_moves.into_iter().zip(logits.into_iter()) {
                            node.children.push(GumbelNode::new(l, 0.0, Some(m)));
                        }
                        node.is_expanded = true;
                    }
                }
            }
        }

        // Backprop
        for (leaf, &value) in leaves.iter().zip(values.iter()) {
            self.backpropagate_and_remove_virtual_loss(root, &leaf.path_indices, value);
        }
    }

    fn select_and_extract_leaf(
        &self,
        root: &GumbelNode,
        game: &mut Game,
        turn_horizon: i32,
        device: &candle_core::Device,
    ) -> Option<LeafData> {
        let mut indices_stack = Vec::new();
        let mut undos = Vec::new();
        let virtual_loss = 1.0;

        let mut current = root;
        // root is always expanded and has 1.0 virtual loss added by parent if any,
        // but here root is unique.

        loop {
            *current.virtual_loss.borrow_mut() += virtual_loss;

            if game.state.settings._game_over || game.state.settings.turn > turn_horizon {
                break;
            }
            if !current.is_expanded {
                break;
            }
            if current.children.is_empty() {
                break;
            }

            // Selection
            let child_idx = if indices_stack.is_empty() {
                // Root selection: Sequential Halving handles root children.
                // We pick from currently active root children.
                current
                    .children
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| {
                        // At root we use z_score for sequential halving
                        a.z_score().partial_cmp(&b.z_score()).unwrap()
                    })
                    .map(|(idx, _)| idx)?
            } else {
                current.select_child_in_tree(-virtual_loss)?
            };

            let m = current.children[child_idx].move_to_here.as_ref()?;
            if let Some(undo) = game.simulate_move(m.as_ref()) {
                undos.push(undo);
                indices_stack.push(child_idx);
                current = &current.children[child_idx];
            } else {
                break;
            }
        }

        let needs_expansion = !current.is_expanded && !game.state.settings._game_over;
        let leaf_data = if needs_expansion {
            let feat = state_to_tensor(
                &game.state,
                game.state.settings.current_player_turn_id,
                device,
            )
            .ok();
            let mut moves = game.legal_moves();
            if moves.iter().any(|m| m.move_type() != MoveType::EndTurn) {
                moves.retain(|m| m.move_type() != MoveType::EndTurn);
            }
            LeafData {
                path_indices: indices_stack,
                features: feat,
                legal_moves: RefCell::new(moves),
                map_size: game.state.settings.size as usize,
            }
        } else {
            LeafData {
                path_indices: indices_stack,
                features: None,
                legal_moves: RefCell::new(Vec::new()),
                map_size: 0,
            }
        };

        while let Some(undo) = undos.pop() {
            undo(&mut game.state);
        }

        Some(leaf_data)
    }

    fn get_node_by_path_mut<'b>(
        &self,
        root: &'b mut GumbelNode,
        indices: &[usize],
    ) -> Option<&'b mut GumbelNode> {
        let mut current = root;
        for &idx in indices {
            current = current.children.get_mut(idx)?;
        }
        Some(current)
    }

    fn backpropagate_and_remove_virtual_loss(
        &self,
        root: &mut GumbelNode,
        indices: &[usize],
        mut value: f32,
    ) {
        let v_loss = 1.0;
        *root.virtual_loss.borrow_mut() -= v_loss;
        root.visits += 1.0;
        root.value_sum += value;

        let mut current = root;
        for &idx in indices {
            value = -value;
            if let Some(child) = current.children.get_mut(idx) {
                *child.virtual_loss.borrow_mut() -= v_loss;
                child.visits += 1.0;
                child.value_sum += value;
                current = child;
            } else {
                break;
            }
        }
    }
}
