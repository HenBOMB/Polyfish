//! Gumbel MuZero search agent.
//!
//! This is a from-scratch implementation of the Gumbel MuZero search from
//! "Learning and Planning in the Space of Convex Utility Functions"
//! (Danihelka et al., 2022). It replaces the earlier broken Gumbel agent,
//! which used the wrong child-selection formula and physically truncated the
//! root's child list after each halving round (collapsing the exported
//! policy target to ~1-2 actions).
//!
//! Key properties of this implementation:
//!   - The root holds **all** legal moves as children (never truncated). A
//!     separate `in_cut` index vector selects the top-`k` by `logit + gumbel`
//!     that actually get searched via Sequential Halving.
//!   - Root selection is round-robin Sequential Halving with equal per-round
//!     visit allocation, batched into one NN call per wave.
//!   - Interior (non-root) selection uses the paper's
//!     `softmax(logit + sigma(completed-Q))` rule with the
//!     `probs - visits/(1+sum visits)` reduction, not the old
//!     `argmax(logit + Q)`.
//!   - The policy target π'(a) ∝ exp(logit(a) + sigma(completed-Q(a))) is
//!     evaluated once over the **full** legal set, so every legal move
//!     receives non-zero support — fixing the policy-collapse bug.
//!   - Value backpropagation (including the player-aware sign flip) is
//!     shared with the AlphaZero agent via `mcts_common`.

use crate::ai::brain::max_turns_ahead;
use crate::ai::eval_server::Evaluator;
use crate::ai::features::{self, RawFeatures};
use crate::ai::gumbel_qtransform::{self, sequence_of_considered_visits, softmax};
use crate::ai::mcts_common::{
    self, BackpropNode, LeafData, TreeNode, backpropagate_and_remove_virtual_loss,
    extract_leaf_data, get_node_by_path, get_node_by_path_mut,
};
use crate::ai::network::RawPolicyOutput;
use crate::ai::policy_composer;
use crate::game::Game;
use crate::moves::{EndTurnMove, Move};
use crate::types::MoveType;
use rand::distributions::Distribution;
use rand_distr::Gumbel;
use std::cell::RefCell;

pub struct GumbelMctsAgent<'a> {
    pub evaluator: &'a Evaluator,
    pub iterations: usize,
    /// Number of root actions to sample into the Sequential-Halving candidate
    /// set (the "top-k" Gumbel cut). The actual candidate count is
    /// `min(k, legal_move_count)`.
    pub k: usize,
    pub batch_size: usize,
    /// Persistent tree across consecutive same-player searches, for
    /// structure-only root-shift reuse. `None` after a fresh build or when
    /// invalidated (book move, terminal root, opponent moved in between).
    tree: Option<GumbelNode>,
    /// Index into `tree`'s children of the move chosen last call. The next
    /// call promotes this child to root iff the new root's feature hash
    /// matches `next_root_hash`.
    last_chosen_idx: Option<usize>,
    /// Feature hash of the state that results from applying the last chosen
    /// move to the previous root. Used as the re-root verification token.
    next_root_hash: Option<u64>,
    /// Diagnostics: number of times a search was served by re-rooting into a
    /// reused subtree rather than building a fresh tree. Exposed for tests
    /// and (future) stats reporting.
    pub tree_reuses: u64,
}

struct GumbelNode {
    visits: f32,
    value_sum: f32,
    logit: f32,
    /// Gumbel(0,1) noise sampled at the root. `0.0` for non-root nodes.
    gumbel: f32,
    /// This node's own NN value prediction, captured at expansion time.
    /// `0.0` until the node is expanded.
    own_value: f32,
    children: Vec<GumbelNode>,
    move_to_here: Option<Box<dyn Move>>,
    is_expanded: bool,
    virtual_loss: RefCell<f32>,
}

impl GumbelNode {
    fn new(logit: f32, gumbel: f32, move_to_here: Option<Box<dyn Move>>) -> Self {
        Self {
            visits: 0.0,
            value_sum: 0.0,
            logit,
            gumbel,
            own_value: 0.0,
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

    fn effective_visits(&self) -> f32 {
        self.visits + *self.virtual_loss.borrow()
    }

    fn add_virtual_loss(&self, amount: f32) {
        *self.virtual_loss.borrow_mut() += amount;
    }
}

impl TreeNode for GumbelNode {
    fn children(&self) -> &[Self] {
        &self.children
    }
    fn children_mut(&mut self) -> &mut [Self] {
        &mut self.children
    }
}

impl BackpropNode for GumbelNode {
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

impl<'a> GumbelMctsAgent<'a> {
    pub fn new(evaluator: &'a Evaluator, iterations: usize, k: usize) -> Self {
        Self {
            evaluator,
            iterations,
            k,
            batch_size: mcts_common::DEFAULT_BATCH_SIZE,
            tree: None,
            last_chosen_idx: None,
            next_root_hash: None,
            tree_reuses: 0,
        }
    }

    /// Drop any cached tree so the next search builds fresh. Called when the
    /// search is bypassed (opening book) or the root is terminal — neither
    /// case leaves a child to promote on the next call.
    fn invalidate_tree(&mut self) {
        self.tree = None;
        self.last_chosen_idx = None;
        self.next_root_hash = None;
    }

    /// Build the root node, either by re-rooting the previous search's tree
    /// (structure-only reuse) or by evaluating the root fresh with the NN.
    ///
    /// **Re-root path.** Within one player's own ~8-ply turn, consecutive
    /// searches are separated by exactly one of that player's own moves, so
    /// the new root is a direct child of the previous root. If the new root's
    /// feature hash matches the hash we recorded for the chosen child last
    /// call, we promote that child: keep its expanded subtree and cached NN
    /// policy/value (skipping the root NN eval and all descendant
    /// expansions), reset visit/value statistics across the subtree so
    /// Sequential Halving runs fresh on a clean slate, re-sample Gumbel noise
    /// on the new root's children, and rebuild `in_cut`. This preserves the
    /// π' policy target's semantics — root-child visit counts come only from
    /// this search's Gumbel-driven allocation, never inherited interior
    /// counts. When the opponent has moved in between (or a book move / forced
    /// move advanced the state), the hash won't match and we build fresh.
    fn search_and_extract(&mut self, game: &mut Game) -> GumbelNode {
        let start_turn = game.state.settings.turn;

        let features = features::state_to_cpu_features(
            &game.state,
            game.state.settings.current_player_turn_id,
        )
        .expect("BUG: Failed to create features at Gumbel root");
        let new_hash = features.hash();

        if let Some(mut prev_root) = self.tree.take() {
            if let Some(chosen_idx) = self.last_chosen_idx
                .filter(|&i| i < prev_root.children.len())
            {
                if self.next_root_hash == Some(new_hash) {
                    let new_root = prev_root.children.swap_remove(chosen_idx);
                    if new_root.is_expanded && !new_root.children.is_empty() {
                        self.tree_reuses += 1;
                        return self.finish_reused_root(game, new_root, start_turn);
                    }
                    // Expanded-but-childless (terminal) reused root: nothing
                    // to search, return as-is.
                    if new_root.is_expanded && new_root.children.is_empty() {
                        return new_root;
                    }
                    // Unexpanded reused root: no cached structure to reuse,
                    // fall through to a fresh build.
                }
            }
            // Mismatch / invalid index: drop the stale tree and build fresh.
        }

        self.build_fresh_root(game, features, start_turn)
    }

    /// Re-root continuation: take the promoted child (already confirmed
    /// expanded with children), reset stats, re-sample Gumbel, suppress
    /// EndTurn, rebuild `in_cut`, and run Sequential Halving.
    fn finish_reused_root(&self, game: &mut Game, mut new_root: GumbelNode, start_turn: i32) -> GumbelNode {
        reset_stats_recursive(&mut new_root);

        // Suppress EndTurn at the new root to mirror the fresh-build path;
        // interior expansion keeps EndTurn, so a reused root may carry one.
        let has_other = new_root.children.iter().any(|c| {
            c.move_to_here
                .as_ref()
                .map_or(false, |m| m.move_type() != MoveType::EndTurn)
        });
        if has_other {
            new_root.children.retain(|c| {
                c.move_to_here
                    .as_ref()
                    .map_or(true, |m| m.move_type() != MoveType::EndTurn)
            });
        }

        // Re-sample Gumbel(0,1) on the new root's children: they were created
        // as non-root nodes with gumbel = 0.0, but root candidates need noise.
        let mut rng = rand::thread_rng();
        let gumbel_dist = Gumbel::new(0.0, 1.0).expect("BUG: Gumbel distribution");
        for c in &mut new_root.children {
            c.gumbel = gumbel_dist.sample(&mut rng);
        }

        let mut in_cut = self.build_in_cut(&new_root);
        self.run_search(game, &mut new_root, &mut in_cut, start_turn);
        new_root
    }

    /// Fresh root: evaluate with the NN, create one child per legal move with
    /// fresh Gumbel draws, build `in_cut`, and run Sequential Halving.
    fn build_fresh_root(&self, game: &mut Game, features: RawFeatures, start_turn: i32) -> GumbelNode {
        let results = self.evaluator.evaluate(vec![features]);
        let (root_value, ref policy_row) = results[0];

        let mut legal_moves = game.legal_moves();
        let map_size = game.state.settings.size as usize;

        let mut root = GumbelNode::new(0.0, 0.0, None);
        root.own_value = root_value;
        root.is_expanded = true;

        if legal_moves.is_empty() {
            return root;
        }

        // Suppress EndTurn at the root when any other move exists to prevent
        // passive play.
        let has_other = legal_moves
            .iter()
            .any(|m| m.move_type() != MoveType::EndTurn);
        if has_other {
            legal_moves.retain(|m| m.move_type() != MoveType::EndTurn);
        }

        let logits = policy_composer::compute_move_log_probs_raw(policy_row, &legal_moves, map_size);

        let mut rng = rand::thread_rng();
        let gumbel_dist = Gumbel::new(0.0, 1.0).expect("BUG: Gumbel distribution");
        root.children = legal_moves
            .into_iter()
            .zip(logits.into_iter())
            .map(|(m, l)| {
                let g = gumbel_dist.sample(&mut rng);
                GumbelNode::new(l, g, Some(m))
            })
            .collect();

        let mut in_cut = self.build_in_cut(&root);
        self.run_search(game, &mut root, &mut in_cut, start_turn);
        root
    }

    /// `in_cut`: indices into `root.children` of the top-`k` by
    /// `(logit + gumbel)`, sorted descending. These are the candidates
    /// actually searched by Sequential Halving.
    fn build_in_cut(&self, root: &GumbelNode) -> Vec<usize> {
        let mut in_cut: Vec<usize> = (0..root.children.len()).collect();
        in_cut.sort_by(|&a, &b| {
            (root.children[b].logit + root.children[b].gumbel)
                .partial_cmp(&(root.children[a].logit + root.children[a].gumbel))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let k = self.k.min(root.children.len());
        in_cut.truncate(k);
        in_cut
    }

    /// Sequential Halving over `in_cut`. Each round considers the top
    /// `round_considered` candidates (by current score) and gives each
    /// exactly `visits_per_candidate` new visits via round-robin batching.
    fn run_search(
        &self,
        game: &mut Game,
        root: &mut GumbelNode,
        in_cut: &mut Vec<usize>,
        start_turn: i32,
    ) {
        if in_cut.is_empty() {
            return;
        }
        let max_considered = in_cut.len();
        let table = sequence_of_considered_visits(max_considered, self.iterations);
        for (round_idx, round_considered, visits_per_candidate) in table {
            let round_considered = round_considered.min(in_cut.len());
            if round_considered <= 1 {
                break;
            }
            // Round 0 keeps the initial (logit + gumbel) order; later rounds
            // re-rank survivors by current score so the best
            // `round_considered` stay in play.
            if round_idx > 0 {
                self.rerank_in_cut(root, in_cut);
            }
            self.run_round_robin_round(
                game,
                root,
                in_cut,
                round_considered,
                visits_per_candidate,
                start_turn,
            );
        }
    }

    /// Re-sort `in_cut` by current score `gumbel + logit + sigma(completed-Q)`
    /// (descending), so the strongest candidates occupy the front positions.
    fn rerank_in_cut(&self, root: &GumbelNode, in_cut: &mut Vec<usize>) {
        let sigma_q = self.sigma_q_for(root, in_cut);
        let mut scored: Vec<(usize, f32)> = in_cut
            .iter()
            .enumerate()
            .map(|(pos, &i)| {
                (
                    i,
                    root.children[i].gumbel + root.children[i].logit + sigma_q[pos],
                )
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        *in_cut = scored.into_iter().map(|(i, _)| i).collect();
    }

    /// sigma(completed-Q) over the candidates referenced by `child_indices`,
    /// returned as a `Vec<f32>` aligned with `child_indices` (i.e. entry `pos`
    /// corresponds to `root.children[child_indices[pos]]`).
    fn sigma_q_for(&self, root: &GumbelNode, child_indices: &[usize]) -> Vec<f32> {
        let q: Vec<f32> = child_indices.iter().map(|&i| root.children[i].q_value()).collect();
        let visits: Vec<f32> = child_indices
            .iter()
            .map(|&i| root.children[i].visits)
            .collect();
        let priors: Vec<f32> = child_indices
            .iter()
            .map(|&i| root.children[i].logit)
            .collect();
        gumbel_qtransform::sigma_completed_q(root.own_value, &priors, &q, &visits, true)
    }

    /// Run one Sequential-Halving round: give each of the first
    /// `round_considered` candidates exactly `visits_per_candidate` new
    /// visits, collected round-robin and batched into one NN call per wave.
    fn run_round_robin_round(
        &self,
        game: &mut Game,
        root: &mut GumbelNode,
        in_cut: &[usize],
        round_considered: usize,
        visits_per_candidate: usize,
        start_turn: i32,
    ) {
        let turn_horizon =
            start_turn + max_turns_ahead(start_turn, game.state.settings.max_turns);

        let total_needed = round_considered * visits_per_candidate;
        let mut collected_per_candidate = vec![0usize; round_considered];
        let mut total_collected = 0;

        while total_collected < total_needed {
            let mut leaves: Vec<LeafData> = Vec::with_capacity(self.batch_size);

            // One wave: cycle through the in-play candidates in order, taking
            // one leaf from each (that hasn't hit its quota) until the batch
            // is full or no candidate can make progress this pass.
            'wave: loop {
                let mut made_progress = false;
                for cand in 0..round_considered {
                    if leaves.len() >= self.batch_size {
                        break 'wave;
                    }
                    if collected_per_candidate[cand] >= visits_per_candidate {
                        continue;
                    }
                    match self.select_and_extract_leaf_under_candidate(
                        root,
                        in_cut[cand],
                        game,
                        turn_horizon,
                    ) {
                        Some(leaf) => {
                            leaves.push(leaf);
                            collected_per_candidate[cand] += 1;
                            made_progress = true;
                        }
                        None => {
                            // Candidate's subtree is terminal / dead-ended.
                            // Mark its quota filled so we don't spin on it.
                            collected_per_candidate[cand] = visits_per_candidate;
                        }
                    }
                }
                if !made_progress {
                    break 'wave;
                }
            }

            if leaves.is_empty() {
                break;
            }
            total_collected += leaves.len();

            let values = self.batched_evaluate_and_expand(root, &leaves);
            for (leaf, &value) in leaves.iter().zip(values.iter()) {
                backpropagate_and_remove_virtual_loss(
                    root,
                    &leaf.path_indices,
                    &leaf.path_players,
                    mcts_common::VIRTUAL_LOSS,
                    value,
                );
            }
        }
    }

    /// Descend from the root into candidate `cand_child_idx`'s subtree, then
    /// keep descending via the interior selection rule until a leaf is
    /// reached. Extract leaf data, undo all simulated moves, and return.
    ///
    /// The root-level Gumbel/Sequential-Halving logic only governs the choice
    /// of `cand_child_idx` (made by the caller's round-robin); everything
    /// below depth 1 uses `select_child_interior`.
    fn select_and_extract_leaf_under_candidate(
        &self,
        root: &GumbelNode,
        cand_child_idx: usize,
        game: &mut Game,
        turn_horizon: i32,
    ) -> Option<LeafData> {
        let mut indices_stack: Vec<usize> = Vec::new();
        let mut path_players: Vec<i32> = Vec::new();
        let mut undos: Vec<crate::actions::UndoCallback> = Vec::new();

        let root_player = game.state.settings.current_player_turn_id;
        path_players.push(root_player);

        // Virtual loss on the root.
        root.add_virtual_loss(mcts_common::VIRTUAL_LOSS);

        // Apply the candidate's move (root -> candidate).
        let m = root.children.get(cand_child_idx)?.move_to_here.as_ref()?;
        let undo = game.simulate_move(m.as_ref())?;
        undos.push(undo);
        indices_stack.push(cand_child_idx);
        path_players.push(game.state.settings.current_player_turn_id);

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
            let m = match current.children[child_idx].move_to_here.as_ref() {
                Some(m) => m,
                None => break,
            };
            let undo = match game.simulate_move(m.as_ref()) {
                Some(u) => u,
                None => break,
            };
            undos.push(undo);
            indices_stack.push(child_idx);
            path_players.push(game.state.settings.current_player_turn_id);
        }

        let needs_expansion = match get_node_by_path(root, &indices_stack) {
            Some(c) => !c.is_expanded && !game.state.settings._game_over,
            None => false,
        };

        let leaf_data = extract_leaf_data(game, indices_stack, path_players, needs_expansion);

        // Always undo, regardless of how the descent ended.
        while let Some(undo) = undos.pop() {
            undo(&mut game.state);
        }

        Some(leaf_data)
    }

    /// Interior (non-root) child selection: `softmax(logit + sigma(Q))` for
    /// the prior, reduced by `visits / (1 + sum_visits)` to discourage
    /// re-visiting already-explored children. Replaces the old
    /// `argmax(logit + Q)`.
    fn select_child_interior(&self, node: &GumbelNode) -> Option<usize> {
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
            .map(|(l, s)| l + s)
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
    fn batched_evaluate_and_expand(
        &self,
        root: &mut GumbelNode,
        leaves: &[LeafData],
    ) -> Vec<f32> {
        let mut values = vec![0.0f32; leaves.len()];
        let mut indices_needing_eval: Vec<usize> = Vec::new();
        let mut eval_batch: Vec<RawFeatures> = Vec::new();

        for (i, leaf) in leaves.iter().enumerate() {
            if let Some(tv) = leaf.terminal_value {
                values[i] = tv;
            } else if let Some(ref feat) = leaf.features {
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
                let (value, ref policy_row) = results[local_idx];
                values[global_idx] = value;

                let leaf = &leaves[global_idx];
                let node = get_node_by_path_mut(root, &leaf.path_indices)
                    .expect("BUG: leaf path not found in tree");

                let legal_moves = leaf.legal_moves.take();
                self.expand_gumbel_node_from_precomputed(
                    node,
                    legal_moves,
                    leaf.map_size,
                    policy_row,
                    value,
                );
            }
        }

        values
    }

    /// Expand a Gumbel node from a pre-computed policy slice. Children are
    /// created with raw logits (no normalization) and `gumbel = 0.0` (non-root).
    /// `own_value` is recorded from the NN value predicted for this node.
    fn expand_gumbel_node_from_precomputed(
        &self,
        node: &mut GumbelNode,
        legal_moves: Vec<Box<dyn Move>>,
        map_size: usize,
        policy: &RawPolicyOutput,
        own_value: f32,
    ) {
        if node.is_expanded {
            return;
        }
        node.own_value = own_value;

        if legal_moves.is_empty() {
            node.is_expanded = true;
            return;
        }

        let logits = policy_composer::compute_move_log_probs_raw(policy, &legal_moves, map_size);
        for (m, l) in legal_moves.into_iter().zip(logits.into_iter()) {
            node.children.push(GumbelNode::new(l, 0.0, Some(m)));
        }
        node.is_expanded = true;
    }

    /// Final move recommendation: among the most-visited root children, pick
    /// the one maximizing `gumbel + logit + sigma(completed-Q)`.
    fn recommend_final_move(&self, root: &GumbelNode) -> usize {
        if root.children.is_empty() {
            return 0;
        }
        let max_visit = root
            .children
            .iter()
            .map(|c| c.visits)
            .fold(0.0f32, f32::max);
        let child_qvalues: Vec<f32> = root.children.iter().map(|c| c.q_value()).collect();
        let child_visits: Vec<f32> = root.children.iter().map(|c| c.visits).collect();
        let child_priors: Vec<f32> = root.children.iter().map(|c| c.logit).collect();
        let sigma_q = gumbel_qtransform::sigma_completed_q(
            root.own_value,
            &child_priors,
            &child_qvalues,
            &child_visits,
            true,
        );

        root.children
            .iter()
            .enumerate()
            .filter(|(_, c)| (c.visits - max_visit).abs() < 0.5)
            .max_by(|(a, ca), (b, cb)| {
                let sa = ca.gumbel + child_priors[*a] + sigma_q[*a];
                let sb = cb.gumbel + child_priors[*b] + sigma_q[*b];
                sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Policy target π'(a) ∝ exp(logit(a) + sigma(completed-Q(a))), evaluated
    /// once over the **full** legal set at the root (all `root.children`).
    /// Returns one `MoveVisit` per legal move, with `visits` carrying the
    /// π' probability mass (not a raw visit count).
    fn extract_policy_targets(&self, root: &GumbelNode) -> Vec<crate::ai::mcts_types::MoveVisit> {
        use crate::ai::mcts_types::MoveVisit;

        if root.children.is_empty() {
            return Vec::new();
        }

        let child_qvalues: Vec<f32> = root.children.iter().map(|c| c.q_value()).collect();
        let child_visits: Vec<f32> = root.children.iter().map(|c| c.visits).collect();
        let child_priors: Vec<f32> = root.children.iter().map(|c| c.logit).collect();

        // v_mix over the full set; only visited (in-cut) children contribute
        // to its weighted-Q sum, so out-of-cut moves do not distort it.
        // Completed-Q is real Q for visited children, v_mix otherwise.
        let sigma_q = gumbel_qtransform::sigma_completed_q(
            root.own_value,
            &child_priors,
            &child_qvalues,
            &child_visits,
            true,
        );
        let raw_scores: Vec<f32> = child_priors
            .iter()
            .zip(&sigma_q)
            .map(|(l, s)| l + s)
            .collect();
        let probs = softmax(&raw_scores); // π'(a)

        let mut targets = Vec::with_capacity(root.children.len());
        for (c, &p) in root.children.iter().zip(probs.iter()) {
            if let Some(m) = &c.move_to_here {
                targets.push(MoveVisit {
                    move_type: m.move_type(),
                    visits: p, // semantically π'(a); see note in plan §2.9
                    source_idx: m.source_idx().ok(),
                    target_idx: m.target_idx().ok(),
                    structure_type: m.structure_type().ok(),
                    unit_type: m.unit_type().ok(),
                    tech_type: m.tech_type().ok(),
                    ability_type: m.ability_type().ok(),
                });
            }
        }
        targets
    }

    pub fn select_move(&mut self, game: &mut Game) -> Option<Box<dyn Move>> {
        use crate::ai::book::Book;
        use rand::seq::SliceRandom;

        let mut book_moves = Book::recommend(game);
        if !book_moves.is_empty() {
            let mut rng = rand::thread_rng();
            book_moves.shuffle(&mut rng);
            if let Some(m) = book_moves.pop() {
                self.invalidate_tree();
                return Some(m);
            }
        }

        let root = self.search_and_extract(game);
        if root.children.is_empty() {
            self.invalidate_tree();
            return Some(Box::new(EndTurnMove));
        }
        let best_idx = self.recommend_final_move(&root);
        let best_move = clone_child_move(&root, best_idx);
        let next_hash = next_root_hash_for(game, best_move.as_deref());
        self.store_tree(root, best_idx, next_hash);
        move_or_end_turn(best_move)
    }

    pub fn select_move_with_decomposed_visits(
        &mut self,
        game: &mut Game,
        move_count: usize,
    ) -> (Option<Box<dyn Move>>, Vec<crate::ai::mcts_types::MoveVisit>) {
        use crate::ai::book::Book;
        use crate::ai::mcts_types::MoveVisit;
        use rand::seq::SliceRandom;

        let mut book_moves = Book::recommend(game);
        if !book_moves.is_empty() {
            let mut rng = rand::thread_rng();
            book_moves.shuffle(&mut rng);
            if let Some(selected_move) = book_moves.pop() {
                self.invalidate_tree();
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

        let root = self.search_and_extract(game);
        if root.children.is_empty() {
            self.invalidate_tree();
            return (Some(Box::new(EndTurnMove)), Vec::new());
        }
        let move_visits = self.extract_policy_targets(&root);

        // Early on, sample based on distribution instead of max visit
        let best_idx = if move_count
            < crate::ai::mcts_zero::ZeroMctsAgent::TEMPERATURE_MOVE_THRESHOLD
            && root.children.len() > 1
        {
            use rand::distributions::WeightedIndex;
            let weights: Vec<f32> = root.children.iter().map(|c| c.visits.max(0.0)).collect();
            match WeightedIndex::new(&weights) {
                Ok(dist) => dist.sample(&mut rand::thread_rng()),
                // All-zero weights (nothing searched) — fall back to the recommendation.
                Err(_) => self.recommend_final_move(&root),
            }
        } else {
            self.recommend_final_move(&root)
        };

        let best_move = clone_child_move(&root, best_idx);
        let next_hash = next_root_hash_for(game, best_move.as_deref());
        self.store_tree(root, best_idx, next_hash);
        (move_or_end_turn(best_move), move_visits)
    }

    pub fn select_move_with_stats(&mut self, game: &mut Game) -> (Option<Box<dyn Move>>, Vec<f32>) {
        use crate::ai::book::Book;
        use rand::seq::SliceRandom;

        let mut book_moves = Book::recommend(game);
        if !book_moves.is_empty() {
            let mut rng = rand::thread_rng();
            book_moves.shuffle(&mut rng);
            if let Some(book_move) = book_moves.pop() {
                // One-hot policy over the legal set; defer to the search for
                // the actual alignment when no book move is found.
                let root = self.search_and_extract(game);
                let mut policy = vec![0.0f32; root.children.len().max(1)];
                for (i, c) in root.children.iter().enumerate() {
                    if let Some(m) = &c.move_to_here {
                        if m.describe(&game.state) == book_move.describe(&game.state) {
                            policy[i] = 1.0;
                            break;
                        }
                    }
                }
                // The book branch discards the searched root; invalidate so
                // the next call does not re-root against a stale tree.
                self.invalidate_tree();
                return (Some(book_move), policy);
            }
        }

        let root = self.search_and_extract(game);
        if root.children.is_empty() {
            self.invalidate_tree();
            return (Some(Box::new(EndTurnMove)), Vec::new());
        }
        let move_visits = self.extract_policy_targets(&root);
        let policy: Vec<f32> = move_visits.iter().map(|mv| mv.visits).collect();
        let best_idx = self.recommend_final_move(&root);
        let best_move = clone_child_move(&root, best_idx);
        let next_hash = next_root_hash_for(game, best_move.as_deref());
        self.store_tree(root, best_idx, next_hash);
        (move_or_end_turn(best_move), policy)
    }

    /// Stash the just-searched root for next-call reuse. `best_idx` must point
    /// at the chosen child, which is kept in the tree (its move was cloned,
    /// not moved out) so the next call can promote it.
    fn store_tree(&mut self, root: GumbelNode, best_idx: usize, next_hash: Option<u64>) {
        self.tree = Some(root);
        self.last_chosen_idx = Some(best_idx);
        self.next_root_hash = next_hash;
    }
}

/// Recursively zero `visits` / `value_sum` / `virtual_loss` across the
/// subtree, keeping `is_expanded`, `children`, `logit`, `own_value`, and
/// `move_to_here` intact. Used by structure-only root-shift reuse so the new
/// search's Sequential Halving runs on a clean statistical slate while the
/// expanded structure and cached NN policy/value are retained.
fn reset_stats_recursive(node: &mut GumbelNode) {
    node.visits = 0.0;
    node.value_sum = 0.0;
    *node.virtual_loss.borrow_mut() = 0.0;
    for c in &mut node.children {
        reset_stats_recursive(c);
    }
}

/// Clone the chosen child's move out of the tree without removing the child,
/// so the subtree below it stays available for next-call root-shift reuse.
fn clone_child_move(root: &GumbelNode, idx: usize) -> Option<Box<dyn Move>> {
    root.children
        .get(idx)
        .and_then(|c| c.move_to_here.as_ref())
        .map(|m| dyn_clone::clone_box(&**m))
}

/// Apply `m` to `game` (assumed to be at the root state, with all search
/// undos applied) via `play_move` — the same path the real game loop uses —
/// and hash the resulting state's features. This is the hash the *next*
/// search's root must match to re-root into this child. Using `play_move`
/// (not `simulate_move`) is load-bearing: the real game loop advances via
/// `play_move`, which updates `_history` and runs FOW discovery that
/// `simulate_move` skips, so a simulate-derived hash would never match the
/// next call's play-derived features. The `game` here is the per-call clone
/// the Brain passes in, which is discarded after we return, so no undo is
/// needed. `None` if the move can't be applied or features can't be built,
/// in which case the next call simply builds fresh.
fn next_root_hash_for(game: &mut Game, m: Option<&dyn Move>) -> Option<u64> {
    let m = m?;
    // The undo callback is intentionally dropped because the game is a per-call clone 
    let _ = game.play_move(m)?;
    let feat = features::state_to_cpu_features(
        &game.state,
        game.state.settings.current_player_turn_id,
    )
    .ok()?;
    Some(feat.hash())
}

fn move_or_end_turn(best_move: Option<Box<dyn Move>>) -> Option<Box<dyn Move>> {
    if best_move.is_none() {
        Some(Box::new(EndTurnMove))
    } else {
        best_move
    }
}
