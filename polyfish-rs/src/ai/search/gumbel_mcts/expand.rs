//! Leaf selection and NN-batched expansion (Aug 2026 taxonomy split out of
//! gumbel_mcts.rs to keep every file under ~1000 lines): descending a
//! candidate subtree to a leaf, interior child selection, and turning a wave
//! of collected leaves into NN evaluations plus newly expanded nodes. A
//! second `impl<'a> GumbelMctsAgent<'a>` block — see trace.rs's note.

use super::reuse::blend_heuristic_into_logits;

use crate::ai::features::RawFeatures;
use crate::ai::gumbel_qtransform::{self, softmax};
use crate::ai::mcts_common::{self, extract_leaf_data, get_node_by_path, get_node_by_path_mut};
use crate::ai::network::RawPolicyOutput;
use crate::ai::policy_composer;
use crate::ai::reward;
use crate::game::Game;
use crate::moves::Move;
use crate::types::MoveType;

use super::{GumbelLeaf, GumbelMctsAgent, GumbelNode};

impl<'a> GumbelMctsAgent<'a> {
    /// Descend from the root into candidate `cand_child_idx`'s subtree, then
    /// keep descending via the interior selection rule until a leaf is
    /// reached. Extract leaf data, undo all simulated moves, and return.
    ///
    /// The root-level Gumbel/Sequential-Halving logic only governs the choice
    /// of `cand_child_idx` (made by the caller's round-robin); everything
    /// below depth 1 uses `select_child_interior`.
    pub(super) fn select_and_extract_leaf_under_candidate(
        &self,
        root: &GumbelNode,
        cand_child_idx: usize,
        game: &mut Game,
        turn_horizon: i32,
    ) -> Option<GumbelLeaf> {
        let mut indices_stack: Vec<usize> = Vec::new();
        let mut path_players: Vec<i32> = Vec::new();
        let mut path_rewards: Vec<f32> = Vec::new();
        let mut path_turn_deltas: Vec<i32> = Vec::new();
        let mut undos: Vec<crate::actions::UndoCallback> = Vec::new();
        let mut stale_path: Option<Vec<usize>> = None;

        let root_player = game.state.settings.current_player_turn_id;
        path_players.push(root_player);

        // Virtual loss on the root.
        root.add_virtual_loss(mcts_common::VIRTUAL_LOSS);

        // Apply the candidate's move (root -> candidate), recording the
        // exact score-delta reward this move banked (in the mover's own
        // perspective) and how many turns it crossed.
        let candidate_node = root.children.get(cand_child_idx)?;
        let m = candidate_node.move_to_here.as_ref()?;
        let (my_pre, opp_pre) = self.edge_snapshot(&game.state, root_player, root_player);
        let turn_pre = game.state.settings.turn;
        let undo = if m.move_type() == MoveType::EndTurn {
            Self::cross_end_turn(game, self.unfreeze_opponent)?
        } else {
            game.simulate_move(m.as_ref())?
        };
        undos.push(undo);
        indices_stack.push(cand_child_idx);
        path_players.push(game.state.settings.current_player_turn_id);
        let (my_post, opp_post) = self.edge_snapshot(&game.state, root_player, root_player);
        let r = reward::normalized_reward_wf(my_pre, opp_pre, my_post, opp_post, reward::REL_W);
        candidate_node.edge_reward.set(Some(r));
        path_rewards.push(r);
        path_turn_deltas.push(game.state.settings.turn - turn_pre);

        // Descend below the candidate using the interior selection rule.
        loop {
            let current = match get_node_by_path(root, &indices_stack) {
                Some(c) => c,
                None => break,
            };
            current.add_virtual_loss(mcts_common::VIRTUAL_LOSS);

            if game.state.settings._game_over {
                break;
            }
            if game.state.settings.turn > turn_horizon {
                self.horizon_hits.set(self.horizon_hits.get() + 1);
                break;
            }
            if !current.is_expanded {
                break;
            }
            if current.children.is_empty() {
                break;
            }

            let child_idx = match self.select_child_interior(current) {
                Some(i) => i,
                None => break,
            };
            let child_node = &current.children[child_idx];
            let m = match child_node.move_to_here.as_ref() {
                Some(m) => m,
                None => break,
            };
            let mover = game.state.settings.current_player_turn_id;
            let (my_pre, opp_pre) = self.edge_snapshot(&game.state, mover, root_player);
            let turn_pre = game.state.settings.turn;
            let sim_result = if m.move_type() == MoveType::EndTurn {
                Self::cross_end_turn(game, self.unfreeze_opponent)
            } else {
                game.simulate_move(m.as_ref())
            };
            let undo = match sim_result {
                Some(u) => u,
                None => {
                    // Cached move is illegal in the replayed state (stale
                    // reused subtree). Flag this node for healing and treat
                    // it as the leaf.
                    stale_path = Some(indices_stack.clone());
                    break;
                }
            };
            undos.push(undo);
            indices_stack.push(child_idx);
            path_players.push(game.state.settings.current_player_turn_id);
            let (my_post, opp_post) = self.edge_snapshot(&game.state, mover, root_player);
            let r = reward::normalized_reward_wf(my_pre, opp_pre, my_post, opp_post, reward::REL_W);
            child_node.edge_reward.set(Some(r));
            path_rewards.push(r);
            path_turn_deltas.push(game.state.settings.turn - turn_pre);
        }

        let leaf_depth = indices_stack.len() as u64;
        self.depth_sum.set(self.depth_sum.get() + leaf_depth);
        self.depth_count.set(self.depth_count.get() + 1);
        if leaf_depth as u32 > self.depth_max.get() {
            self.depth_max.set(leaf_depth as u32);
        }

        // A stale node counts as needing expansion so its features and the
        // true state's legal moves are extracted now; the actual re-expansion
        // happens on a later wave, after the wave loop has healed the node.
        let needs_expansion = match get_node_by_path(root, &indices_stack) {
            Some(c) => {
                (!c.is_expanded || stale_path.is_some()) && !game.state.settings._game_over
            }
            None => false,
        };

        let mut leaf_data = extract_leaf_data(
            game,
            indices_stack,
            path_players,
            needs_expansion,
            self.pursuit_focus,
            self.macro_goal.as_ref(),
        );

        // Compute in-tree heuristic scores while `game` is still at the leaf
        // state (Phase B, where priors are actually blended in, has no Game
        // in scope). Aligned 1:1 with `leaf_data.legal_moves`.
        if self.prior_heuristic_weight > 0.0 && leaf_data.terminal_value.is_none() {
            let moves = leaf_data.legal_moves.borrow();
            if !moves.is_empty() {
                leaf_data.heuristic_scores = Some(
                    moves
                        .iter()
                        .map(|m| crate::ai::scoring::score_move(game, m.as_ref()))
                        .collect(),
                );
            }
        }

        // Always undo, regardless of how the descent ended.
        while let Some(undo) = undos.pop() {
            undo(&mut game.state);
        }

        Some(GumbelLeaf {
            data: leaf_data,
            rewards: path_rewards,
            turn_deltas: path_turn_deltas,
            stale_path,
        })
    }
    /// Interior (non-root) child selection: `softmax(logit + sigma(Q))` for
    /// the prior, reduced by `visits / (1 + sum_visits)` to discourage
    /// re-visiting already-explored children. Replaces the old
    /// `argmax(logit + Q)`.
    pub(super) fn select_child_interior(&self, node: &GumbelNode) -> Option<usize> {
        let n = node.children.len();
        if n == 0 {
            return None;
        }
        let child_qvalues: Vec<f32> = node.children.iter().map(|c| c.q_value()).collect();
        let child_visits: Vec<f32> =
            node.children.iter().map(|c| c.effective_visits()).collect();
        let child_priors: Vec<f32> = node.children.iter().map(|c| c.logit).collect();
        let sigma_q = gumbel_qtransform::sigma_completed_q(
            node.own_value,
            &child_priors,
            &child_qvalues,
            &child_visits,
            true,
        );
        let combined: Vec<f32> = child_priors
            .iter()
            .zip(&sigma_q)
            .map(|(l, s)| l + self.tree_q_weight * s)
            .collect();
        let probs = softmax(&combined);
        let sum_visits: f32 = child_visits.iter().sum();

        (0..n).max_by(|&a, &b| {
            let score_a = probs[a] - child_visits[a] / (1.0 + sum_visits);
            let score_b = probs[b] - child_visits[b] / (1.0 + sum_visits);
            score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
        })
    }
    /// Batched NN evaluation + expansion for a wave of leaves. Returns one
    /// value per leaf (terminal outcome or NN value), in leaf order.
    pub(super) fn batched_evaluate_and_expand(
        &self,
        root: &mut GumbelNode,
        leaves: &[GumbelLeaf],
    ) -> Vec<(f32, f32)> {
        let mut values = vec![(0.0f32, 0.0f32); leaves.len()];
        let mut indices_needing_eval: Vec<usize> = Vec::new();
        let mut eval_batch: Vec<RawFeatures> = Vec::new();

        for (i, leaf) in leaves.iter().enumerate() {
            if let Some(tv) = leaf.data.terminal_value {
                values[i] = (tv, 0.0); // Progress is 0.0 at terminal state
            } else if let Some(ref feat) = leaf.data.features {
                indices_needing_eval.push(i);
                eval_batch.push(RawFeatures {
                    spatial: feat.spatial.clone(),
                    player: feat.player.clone(),
                });
            }
        }

        if !indices_needing_eval.is_empty() {
            let results = self.evaluator.evaluate(eval_batch);

            for (local_idx, &global_idx) in indices_needing_eval.iter().enumerate() {
                let (value, progress, ref policy_row) = results[local_idx];
                values[global_idx] = (value, progress);

                let leaf = &leaves[global_idx];
                let node = get_node_by_path_mut(root, &leaf.data.path_indices)
                    .expect("BUG: leaf path not found in tree");

                let legal_moves = leaf.data.legal_moves.take();
                self.expand_gumbel_node_from_precomputed(
                    node,
                    legal_moves,
                    leaf.data.map_size,
                    policy_row,
                    value,
                    progress,
                    leaf.data.heuristic_scores.as_deref(),
                );
            }
        }

        values
    }
    /// Expand a Gumbel node from a pre-computed policy slice. Children are
    /// created with raw logits (no normalization) and `gumbel = 0.0` (non-root).
    /// `own_value` is recorded from the NN value predicted for this node.
    /// `heuristic_scores`, if present (in-tree blending enabled), is blended
    /// into the logits before the children are created, and the node is
    /// flagged so a later root-promotion doesn't blend it again.
    pub(super) fn expand_gumbel_node_from_precomputed(
        &self,
        node: &mut GumbelNode,
        legal_moves: Vec<Box<dyn Move>>,
        map_size: usize,
        policy: &RawPolicyOutput,
        own_value: f32,
        own_progress: f32,
        heuristic_scores: Option<&[f32]>,
    ) {
        if node.is_expanded {
            return;
        }
        node.own_value = own_value;
        node.own_progress = own_progress;

        if legal_moves.is_empty() {
            node.is_expanded = true;
            return;
        }

        let mut logits = policy_composer::compute_move_log_probs_raw(policy, &legal_moves, map_size);
        if self.prior_heuristic_weight > 0.0 {
            if let Some(hs) = heuristic_scores {
                blend_heuristic_into_logits(&mut logits, hs, self.prior_heuristic_weight);
                node.heuristic_blended = true;
            }
        }
        for (m, l) in legal_moves.into_iter().zip(logits.into_iter()) {
            node.children.push(GumbelNode::new(l, 0.0, Some(m)));
        }
        node.is_expanded = true;
    }
}
