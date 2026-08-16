//! Root construction methods (Aug 2026 taxonomy split out of gumbel_mcts.rs
//! to keep every file under ~1000 lines): building a fresh root from an NN
//! eval, or re-rooting into a reused subtree from the previous search. A
//! second `impl<'a> GumbelMctsAgent<'a>` block — see trace.rs's note.

use super::gate::{gate_retain, gate_stats, reused_root_gates_enabled};
use super::reuse::{blend_goal_prior, blend_heuristic_prior, reset_stats_recursive, reused_children_match_legal};

use crate::ai::features::{self, RawFeatures};
use crate::ai::policy_composer;
use crate::game::Game;
use crate::types::MoveType;
use rand::distributions::Distribution;
use rand_distr::Gumbel;

use super::{GumbelMctsAgent, GumbelNode};

impl<'a> GumbelMctsAgent<'a> {
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
    /// counts. When the opponent has moved in between (or a forced move
    /// advanced the state), the hash won't match and we build fresh.
    pub(super) fn search_and_extract(&mut self, game: &mut Game) -> GumbelNode {
        let start_turn = game.state.settings.turn;

        let features = features::state_to_cpu_features_goal(
            &game.state,
            game.state.settings.current_player_turn_id,
            self.pursuit_focus,
            self.macro_goal.as_ref(),
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
                        // Revalidate the children to ensure the moves are still legal
                        if reused_children_match_legal(
                            game,
                            &new_root.children,
                            self.star_gate,
                            self.macro_goal.as_ref().map(|g| g.stance),
                            self.goal_aux.as_ref(),
                        ) {
                            self.tree_reuses += 1;
                            return self.finish_reused_root(game, new_root, start_turn);
                        }
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
    pub(super) fn finish_reused_root(&self, game: &mut Game, mut new_root: GumbelNode, start_turn: i32) -> GumbelNode {
        reset_stats_recursive(&mut new_root);

        // v8: the promoted child's children were created as INTERIOR nodes, so
        // they never met the root gates that `build_fresh_root` applies. Tree
        // reuse is the common case mid-turn (~8 plies per game turn), so
        // without this every root gate leaked on all but the first ply of a
        // turn — measured Aug 2: the pop-discipline and road gates were fully
        // inert until this was added. EndTurn is exempt so the root can never
        // be emptied; the suppression below still removes it when it should.
        if (self.star_gate || self.goal_aux.is_some()) && reused_root_gates_enabled() {
            let stance = self.macro_goal.as_ref().map(|g| g.stance);
            let before = new_root.children.len();
            new_root.children.retain(|c| {
                let Some(m) = c.move_to_here.as_ref() else {
                    return true;
                };
                gate_retain(
                    &game.state,
                    m.as_ref(),
                    self.star_gate,
                    stance,
                    self.goal_aux.as_ref(),
                )
            });
            gate_stats::record_ply(before, new_root.children.len());
        }

        // Belt-and-suspenders: `extract_leaf_data` already drops EndTurn from
        // any expansion (root or interior) whenever another move exists, so
        // this is normally a no-op — kept in case a reused root's EndTurn
        // was its sole child at expansion time (then legitimately present)
        // but other moves are available now.
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
            c.gumbel = self.gumbel_scale * gumbel_dist.sample(&mut rng);
        }
        
        // Bootstrap with the priors from the heuristic mcts agent. Skip if
        // this node's children were already blended at in-tree expansion
        // time (avoids double-applying the heuristic on promotion to root).
        if self.prior_heuristic_weight > 0.0 && !new_root.heuristic_blended {
            blend_heuristic_prior(game, &mut new_root.children, self.prior_heuristic_weight);
        }
        blend_goal_prior(game, &mut new_root.children, self.macro_goal.as_ref());

        let mut in_cut = self.build_in_cut(&new_root);
        self.run_search(game, &mut new_root, &mut in_cut, start_turn);
        new_root
    }
    /// Fresh root: evaluate with the NN, create one child per legal move with
    /// fresh Gumbel draws, build `in_cut`, and run Sequential Halving.
    pub(super) fn build_fresh_root(&self, game: &mut Game, features: RawFeatures, start_turn: i32) -> GumbelNode {
        let results = self.evaluator.evaluate(vec![features]);
        let (root_value, root_progress, ref policy_row) = results[0];

        let mut legal_moves = game.legal_moves();
        if self.star_gate || self.goal_aux.is_some() {
            let stance = self.macro_goal.as_ref().map(|g| g.stance);
            let before = legal_moves.len();
            legal_moves.retain(|m| {
                gate_retain(
                    &game.state,
                    m.as_ref(),
                    self.star_gate,
                    stance,
                    self.goal_aux.as_ref(),
                )
            });
            gate_stats::record_ply(before, legal_moves.len());
        }
        let map_size = game.state.settings.size as usize;

        let mut root = GumbelNode::new(0.0, 0.0, None);
        root.own_value = root_value;
        root.own_progress = root_progress;
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

        let logits =
            policy_composer::compute_move_log_probs_raw(policy_row, &legal_moves, map_size);

        let mut rng = rand::thread_rng();
        let gumbel_dist = Gumbel::new(0.0, 1.0).expect("BUG: Gumbel distribution");
        root.children = legal_moves
            .into_iter()
            .zip(logits.into_iter())
            .map(|(m, l)| {
                let g = self.gumbel_scale * gumbel_dist.sample(&mut rng);
                GumbelNode::new(l, g, Some(m))
            })
            .collect();

        // Snapshot pre-blend logits for trace capture below; blend below
        // overwrites child.logit in place, so this is the only chance to see
        // the network's raw (unblended) opinion.
        let raw_logits: Vec<f32> = root.children.iter().map(|c| c.logit).collect();

        // Bootstrap with the priors from the heuristic mcts agent, before the
        // Gumbel top-k cut is built so the cut ranks on blended priors.
        if self.prior_heuristic_weight > 0.0 {
            blend_heuristic_prior(game, &mut root.children, self.prior_heuristic_weight);
        }
        blend_goal_prior(game, &mut root.children, self.macro_goal.as_ref());

        let mut in_cut = self.build_in_cut(&root);
        self.record_root_candidates(game, &root, &in_cut, &raw_logits);

        self.run_search(game, &mut root, &mut in_cut, start_turn);
        root
    }
    /// `in_cut`: indices into `root.children` of the top-`k` by
    /// `(logit + gumbel)`, sorted descending. These are the candidates
    /// actually searched by Sequential Halving.
    pub(super) fn build_in_cut(&self, root: &GumbelNode) -> Vec<usize> {
        let mut in_cut: Vec<usize> = (0..root.children.len()).collect();
        // `total_cmp`, not `partial_cmp().unwrap_or(Equal)`: with a NaN in
        // play the latter is intransitive (NaN reads Equal to everything while
        // the rest keep a real order) and driftsort panics on the violation —
        // observed skipping gauge seeds since before EXP_ELO_051.
        in_cut.sort_by(|&a, &b| {
            (root.children[b].logit + root.children[b].gumbel)
                .total_cmp(&(root.children[a].logit + root.children[a].gumbel))
        });
        let k = self.k.min(root.children.len());
        in_cut.truncate(k);
        in_cut
    }
}
