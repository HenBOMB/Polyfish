//! Decision-trace recording methods (Aug 2026 taxonomy split out of
//! gumbel_mcts.rs to keep every file under ~1000 lines): the root candidate
//! set, each Sequential-Halving round's survivor ranking, and the final
//! chosen-move record. A second `impl<'a> GumbelMctsAgent<'a>` block — Rust
//! merges impl blocks for the same type across files, so this is a pure
//! file-organization move with no behavior change.

use crate::ai::decision_trace::{CandidateTrace, RoundCandidate, RoundSnapshot, SelectionMode};
use crate::ai::gumbel_qtransform::{self, softmax};
use crate::game::Game;

use super::{GumbelMctsAgent, GumbelNode};

impl<'a> GumbelMctsAgent<'a> {
    /// Record the full legal root move set, called once per fresh root build
    /// (never on the re-root path — `arm_trace` guarantees that path isn't
    /// taken while armed). `raw_logits` must be captured by the caller before
    /// heuristic blending overwrites `child.logit` in place.
    pub(super) fn record_root_candidates(
        &self,
        game: &Game,
        root: &GumbelNode,
        in_cut: &[usize],
        raw_logits: &[f32],
    ) {
        let mut trace_ref = self.trace.borrow_mut();
        let Some(trace) = trace_ref.as_mut() else {
            return;
        };
        trace.root_own_value = root.own_value;
        let raw_probs = softmax(raw_logits);
        let blended_logits: Vec<f32> = root.children.iter().map(|c| c.logit).collect();
        let blended_probs = softmax(&blended_logits);
        for (i, child) in root.children.iter().enumerate() {
            let Some(mv) = child.move_to_here.as_ref() else {
                continue;
            };
            trace.candidates.push(CandidateTrace {
                description: mv.describe(&game.state),
                move_type: format!("{:?}", mv.move_type()),
                source_idx: mv.source_idx().ok(),
                target_idx: mv.target_idx().ok(),
                own_value: None,
                q_value: 0.0,
                visits: 0.0,
                edge_reward: None,
                raw_net_prob: raw_probs[i],
                heuristic_score: crate::ai::scoring::score_move(game, mv.as_ref()),
                search_prior_prob: blended_probs[i],
                gumbel_noise: child.gumbel,
                in_top_k: in_cut.contains(&i),
            });
        }
    }
    /// Record the Sequential-Halving survivor ranking for one round, after
    /// that round's visits have landed.
    pub(super) fn record_round_snapshot(
        &self,
        root: &GumbelNode,
        in_cut: &[usize],
        round_idx: usize,
        round_considered: usize,
        visits_per_candidate: usize,
    ) {
        if self.trace.borrow().is_none() {
            return;
        }
        let survivors_idx = &in_cut[..round_considered.min(in_cut.len())];
        let sigma_q = self.sigma_q_for(root, survivors_idx);

        let mut trace_ref = self.trace.borrow_mut();
        let Some(trace) = trace_ref.as_mut() else {
            return;
        };
        let survivors = survivors_idx
            .iter()
            .zip(sigma_q.iter())
            .map(|(&i, &sq)| RoundCandidate {
                candidate_idx: i,
                score: root.children[i].gumbel + root.children[i].logit + sq,
                visits: root.children[i].visits,
                q_value: root.children[i].q_value(),
            })
            .collect();
        trace.rounds.push(RoundSnapshot {
            round_idx,
            round_considered,
            visits_per_candidate,
            survivors,
        });
    }
    /// Record final per-candidate visits/Q/value and the selected move, once
    /// the search is done and a move has been chosen.
    pub(super) fn record_final(&self, root: &GumbelNode, best_idx: usize, move_count: usize) {
        let mut trace_ref = self.trace.borrow_mut();
        let Some(trace) = trace_ref.as_mut() else {
            return;
        };

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

        for (i, child) in root.children.iter().enumerate() {
            if let Some(c) = trace.candidates.get_mut(i) {
                c.visits = child.visits;
                c.q_value = child.q_value();
                c.own_value = child.is_expanded.then_some(child.own_value);
                c.edge_reward = child.edge_reward.get();
            }
        }
        trace.root_search_value = (root.visits > 0.0).then(|| root.q_value());
        let mode = if move_count < crate::ai::mcts_zero::ZeroMctsAgent::TEMPERATURE_MOVE_THRESHOLD
            && root.children.len() > 1
        {
            SelectionMode::Sampled
        } else {
            SelectionMode::Argmax
        };
        let tiebreak = root
            .children
            .get(best_idx)
            .map(|c| c.gumbel + child_priors[best_idx] + self.tree_q_weight * sigma_q[best_idx])
            .unwrap_or(0.0);
        trace.chosen = Some((mode, best_idx, tiebreak));
    }
}
