//! Final move recommendation and the `select_move*` public API (Aug 2026
//! taxonomy split out of gumbel_mcts.rs to keep every file under ~1000
//! lines): picking the root's best child, exporting the full-legal-set
//! policy target, and the three public entry points self_play/arena/brain
//! call. A second `impl<'a> GumbelMctsAgent<'a>` block — see trace.rs's note.

use super::reuse::{clone_child_move, move_or_end_turn, next_root_hash_for};

use crate::ai::gumbel_qtransform::{self, softmax};
use crate::game::Game;
use crate::moves::{EndTurnMove, Move};
use rand::distributions::Distribution;

use super::{GumbelMctsAgent, GumbelNode};

impl<'a> GumbelMctsAgent<'a> {
    /// Final move recommendation: among the most-visited root children, pick
    /// the one maximizing `gumbel + logit + sigma(completed-Q)`.
    pub(super) fn recommend_final_move(&self, root: &GumbelNode) -> usize {
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

        let chosen = root
            .children
            .iter()
            .enumerate()
            .filter(|(_, c)| (c.visits - max_visit).abs() < 0.5)
            .max_by(|(a, ca), (b, cb)| {
                let sa = ca.gumbel + child_priors[*a] + self.tree_q_weight * sigma_q[*a];
                let sb = cb.gumbel + child_priors[*b] + self.tree_q_weight * sigma_q[*b];
                sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Distillation headroom: how often search overrides argmax(prior) over
        // the full legal set. Under GUMBEL_SCALE=0 the prior's top move is
        // always inside the top-k cut, so a disagreement is a genuine override
        // rather than a candidate that search never considered.
        let prior_argmax = child_priors
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        self.decision_count.set(self.decision_count.get() + 1);
        if prior_argmax == chosen {
            self.agree_count.set(self.agree_count.get() + 1);
        }

        chosen
    }
    /// Policy target π'(a) ∝ exp(logit(a) + sigma(completed-Q(a))), evaluated
    /// once over the **full** legal set at the root (all `root.children`).
    /// Returns one `MoveVisit` per legal move, with `visits` carrying the
    /// π' probability mass (not a raw visit count).
    pub(super) fn extract_policy_targets(&self, root: &GumbelNode) -> Vec<crate::ai::mcts_types::MoveVisit> {
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
            .map(|(l, s)| l + self.policy_target_q_weight * s)
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
                    reward_type: m.reward_type().ok(),
                });
            }
        }
        targets
    }
    pub fn select_move(&mut self, game: &mut Game) -> Option<Box<dyn Move>> {
        // Cleared up front; `store_tree` sets it again once a real search
        // (not an empty root) actually accumulates root visits.
        self.last_root_value = None;

        let root = self.search_and_extract(game);
        if root.children.is_empty() {
            self.invalidate_tree();
            return Some(Box::new(EndTurnMove));
        }
        let best_idx = self.recommend_final_move(&root);
        // This path always argmaxes, so pass a move_count past the temperature
        // threshold; without it an armed trace never finalizes here and
        // `take_trace` silently returns None (arena had no traces at all).
        self.record_final(&root, best_idx, usize::MAX);
        let best_move = clone_child_move(&root, best_idx);
        let next_hash = next_root_hash_for(game, best_move.as_deref(), self.pursuit_focus, self.macro_goal.as_ref());
        self.store_tree(root, best_idx, next_hash);
        move_or_end_turn(best_move)
    }
    pub fn select_move_with_decomposed_visits(
        &mut self,
        game: &mut Game,
        move_count: usize,
    ) -> (Option<Box<dyn Move>>, Vec<crate::ai::mcts_types::MoveVisit>) {
        self.last_root_value = None;

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
                Ok(dist) => self.with_rng(|rng| dist.sample(rng)),
                // All-zero weights (nothing searched) — fall back to the recommendation.
                Err(_) => self.recommend_final_move(&root),
            }
        } else {
            self.recommend_final_move(&root)
        };

        self.record_final(&root, best_idx, move_count);

        let best_move = clone_child_move(&root, best_idx);
        let next_hash = next_root_hash_for(game, best_move.as_deref(), self.pursuit_focus, self.macro_goal.as_ref());
        self.store_tree(root, best_idx, next_hash);
        (move_or_end_turn(best_move), move_visits)
    }
    pub fn select_move_with_stats(&mut self, game: &mut Game) -> (Option<Box<dyn Move>>, Vec<f32>) {
        self.last_root_value = None;

        let root = self.search_and_extract(game);
        if root.children.is_empty() {
            self.invalidate_tree();
            return (Some(Box::new(EndTurnMove)), Vec::new());
        }
        let move_visits = self.extract_policy_targets(&root);
        let policy: Vec<f32> = move_visits.iter().map(|mv| mv.visits).collect();
        let best_idx = self.recommend_final_move(&root);
        let best_move = clone_child_move(&root, best_idx);
        let next_hash = next_root_hash_for(game, best_move.as_deref(), self.pursuit_focus, self.macro_goal.as_ref());
        self.store_tree(root, best_idx, next_hash);
        (move_or_end_turn(best_move), policy)
    }
    /// Stash the just-searched root for next-call reuse. `best_idx` must point
    /// at the chosen child, which is kept in the tree (its move was cloned,
    /// not moved out) so the next call can promote it.
    ///
    /// Also records `last_root_value` from this same root: `None` if it
    /// never accumulated a visit (single-legal-move root — `run_search`
    /// short-circuits before any visits land), `Some(root.q_value())`
    /// otherwise. This is the one place all three `select_move*` callers'
    /// non-early-return paths converge, so it's the single spot that needs
    /// to know about `last_root_value` bookkeeping.
    pub(super) fn store_tree(&mut self, root: GumbelNode, best_idx: usize, next_hash: Option<u64>) {
        self.last_root_value = (root.visits > 0.0).then(|| root.q_value());
        self.last_root_own_value = (root.visits > 0.0).then(|| root.own_value);
        self.tree = Some(root);
        self.last_chosen_idx = Some(best_idx);
        self.next_root_hash = next_hash;
    }
}
