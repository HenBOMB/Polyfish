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
use crate::ai::decision_trace::{
    CandidateTrace, DecisionTrace, RoundCandidate, RoundSnapshot, SelectionMode, TraceBuilder,
};
use crate::ai::eval_server::Evaluator;
use crate::ai::features::{self, RawFeatures};
use crate::ai::gumbel_qtransform::{self, sequence_of_considered_visits, softmax};
use crate::ai::mcts_common::{
    self, BackpropNode, LeafData, TreeNode, backpropagate_return_with_rewards, extract_leaf_data,
    get_node_by_path, get_node_by_path_mut,
};
use crate::ai::network::RawPolicyOutput;
use crate::ai::policy_composer;
use crate::ai::reward;
use crate::game::Game;
use crate::moves::{EndTurnMove, Move};
use crate::types::MoveType;
use rand::distributions::Distribution;
use rand_distr::Gumbel;
use std::cell::{Cell, RefCell};

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
    /// invalidated (terminal root, opponent moved in between).
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
    // How much to blend the heuristic prior into the network's root priors.
    // High in the beginning to bootstrap the network but decays over time.
    pub prior_heuristic_weight: f32,
    /// Weight β on σ(completed-Q) in the exported policy TARGET π' =
    /// softmax(logit + β·σ(Q)). This gates how much search re-ranking flows
    /// into training targets — ramp it up as the value head earns trust.
    /// 1.0 = paper behavior; 0.0 = distill the (blended) prior unchanged.
    pub policy_target_q_weight: f32,
    /// Weight β_tree on σ(completed-Q) inside the search itself: interior
    /// selection, the Sequential-Halving re-rank, and the final root
    /// recommendation. min-max rescale normalizes whatever Q spread exists —
    /// signal or noise — to full amplitude (~(C_VISIT+maxvisit)·C_SCALE ≈ 5-6
    /// logits at 64 sims), so an untrusted value head injects ~6 logits of
    /// noise into every selection step and can destroy a correct prior read
    /// (see notes.md, decision-trace section). At 0.0 search degenerates to
    /// prior+gumbel sampling (BC-anchored behavior); 1.0 = paper behavior.
    pub tree_q_weight: f32,
    /// Diagnostic capture for the next search, armed via `arm_trace`. `None`
    /// (the default) costs one `RefCell` borrow-check per call site and
    /// nothing else — see decision_trace.rs.
    trace: RefCell<Option<TraceBuilder>>,
    /// The most recently completed search's root value (`root.q_value()`
    /// after backup — a discounted-return state-value estimate under the
    /// reward-aware backup, `None` if the root never accumulated a visit:
    /// an empty legal set, or a single-legal-move root, which
    /// `run_search` short-circuits before any visits land). Set at the end
    /// of every `select_move*` call, consumed by `self_play`'s TD label
    /// bootstrap via `Brain::last_root_value`.
    last_root_value: Option<f32>,
    /// KL(root visit distribution ‖ root prior softmax) of the most recent
    /// search — how much information the sims added beyond the (blended)
    /// prior. ~0 means search is just echoing the policy and the AlphaZero
    /// improvement operator is idle. `None` when no real search ran.
    last_search_kl: Option<f32>,
}

struct GumbelNode {
    visits: f32,
    value_sum: f32,
    progress_sum: f32,
    logit: f32,
    /// Gumbel(0,1) noise sampled at the root. `0.0` for non-root nodes.
    gumbel: f32,
    /// This node's own NN value prediction, captured at expansion time.
    /// `0.0` until the node is expanded.
    own_value: f32,
    /// Normalized score-delta reward of the edge that produced this node
    /// (parent -> this), cached the first time search traverses the edge.
    /// `None` for the tree root (no incoming edge) or any node never
    /// visited by this or a prior search. Survives re-root (kept out of
    /// `reset_stats_recursive`) like `own_value`/`logit`.
    edge_reward: Cell<Option<f32>>,
    own_progress: f32,
    children: Vec<GumbelNode>,
    move_to_here: Option<Box<dyn Move>>,
    is_expanded: bool,
    virtual_loss: RefCell<f32>,
    /// Set when this node's priors were already heuristic-blended at
    /// expansion time, so `finish_reused_root` doesn't blend it again if it
    /// is later promoted to root.
    heuristic_blended: bool,
}

impl GumbelNode {
    fn new(logit: f32, gumbel: f32, move_to_here: Option<Box<dyn Move>>) -> Self {
        Self {
            visits: 0.0,
            value_sum: 0.0,
            progress_sum: 0.0,
            logit,
            gumbel,
            own_value: 0.0,
            edge_reward: Cell::new(None),
            own_progress: 0.0,
            children: Vec::new(),
            move_to_here,
            is_expanded: false,
            virtual_loss: RefCell::new(0.0),
            heuristic_blended: false,
        }
    }

    fn q_value(&self) -> f32 {
        let q = if self.visits == 0.0 {
            0.0
        } else {
            self.value_sum / self.visits
        };
        q + self.own_progress
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

/// A single collected leaf, wrapping the shared `LeafData` (features, legal
/// moves, terminal value — everything `mcts_zero` also produces) with the
/// per-edge rewards/turn-deltas collected along its path, which only the
/// reward-aware Gumbel backup needs. `rewards[i]`/`turn_deltas[i]` describe
/// the same edge as `data.path_indices[i]` (edge `i`: node `i` -> node
/// `i+1`); `rewards.len() == turn_deltas.len() == data.path_indices.len()`.
struct GumbelLeaf {
    data: LeafData,
    rewards: Vec<f32>,
    turn_deltas: Vec<i32>,
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
            prior_heuristic_weight: 0.0,
            policy_target_q_weight: 1.0,
            tree_q_weight: 1.0,
            trace: RefCell::new(None),
            last_root_value: None,
            last_search_kl: None,
        }
    }

    /// The completed search's root value (see `last_root_value` field docs),
    /// if the most recent `select_move*` call actually ran a search.
    pub fn last_root_value(&self) -> Option<f32> {
        self.last_root_value
    }

    pub fn clear_last_root_value(&mut self) {
        self.last_root_value = None;
        self.last_search_kl = None;
    }

    /// See the `last_search_kl` field docs.
    pub fn last_search_kl(&self) -> Option<f32> {
        self.last_search_kl
    }

    /// Drop any cached tree so the next search builds fresh. Called when the
    /// root is terminal / has no legal moves — no child to promote next call.
    fn invalidate_tree(&mut self) {
        self.tree = None;
        self.last_chosen_idx = None;
        self.next_root_hash = None;
    }

    /// Arm decision-trace capture for the next search. Forces a fresh root
    /// build (see `invalidate_tree`) so raw network logits and heuristic
    /// scores get recomputed instead of reused from `finish_reused_root`'s
    /// already-blended cached subtree, which would otherwise leave most
    /// within-turn traces empty.
    pub fn arm_trace(&mut self) {
        self.invalidate_tree();
        *self.trace.borrow_mut() = Some(TraceBuilder::default());
    }

    /// Drain and finalize the trace captured by the last search. `None` if
    /// never armed, or armed but short-circuited before a selection was made
    /// (empty legal-move root).
    pub fn take_trace(&mut self) -> Option<DecisionTrace> {
        self.trace
            .borrow_mut()
            .take()
            .and_then(|b| b.finish(self.prior_heuristic_weight))
    }

    /// Record the full legal root move set, called once per fresh root build
    /// (never on the re-root path — `arm_trace` guarantees that path isn't
    /// taken while armed). `raw_logits` must be captured by the caller before
    /// heuristic blending overwrites `child.logit` in place.
    fn record_root_candidates(
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
    fn record_round_snapshot(
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
    fn record_final(&self, root: &GumbelNode, best_idx: usize, move_count: usize) {
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
    fn search_and_extract(&mut self, game: &mut Game) -> GumbelNode {
        let start_turn = game.state.settings.turn;

        let features = features::state_to_cpu_features(
            &game.state,
            game.state.settings.current_player_turn_id,
        )
        .expect("BUG: Failed to create features at Gumbel root");
        let new_hash = features.hash();

        if let Some(mut prev_root) = self.tree.take() {
            if let Some(chosen_idx) = self
                .last_chosen_idx
                .filter(|&i| i < prev_root.children.len())
            {
                if self.next_root_hash == Some(new_hash) {
                    let new_root = prev_root.children.swap_remove(chosen_idx);
                    if new_root.is_expanded && !new_root.children.is_empty() {
                        // Revalidate the children to ensure the moves are still legal
                        if reused_children_match_legal(game, &new_root.children) {
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
    fn finish_reused_root(
        &self,
        game: &mut Game,
        mut new_root: GumbelNode,
        start_turn: i32,
    ) -> GumbelNode {
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

        // Bootstrap with the priors from the heuristic mcts agent. Skip if
        // this node's children were already blended at in-tree expansion
        // time (avoids double-applying the heuristic on promotion to root).
        if self.prior_heuristic_weight > 0.0 && !new_root.heuristic_blended {
            blend_heuristic_prior(game, &mut new_root.children, self.prior_heuristic_weight);
        }

        let mut in_cut = self.build_in_cut(&new_root);
        self.run_search(game, &mut new_root, &mut in_cut, start_turn);
        new_root
    }

    /// Fresh root: evaluate with the NN, create one child per legal move with
    /// fresh Gumbel draws, build `in_cut`, and run Sequential Halving.
    fn build_fresh_root(
        &self,
        game: &mut Game,
        features: RawFeatures,
        start_turn: i32,
    ) -> GumbelNode {
        let results = self.evaluator.evaluate(vec![features]);
        let (root_value, root_progress, ref policy_row) = results[0];

        let mut legal_moves = game.legal_moves();
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
                let g = gumbel_dist.sample(&mut rng);
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

        let mut in_cut = self.build_in_cut(&root);
        self.record_root_candidates(game, &root, &in_cut, &raw_logits);

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
            self.record_round_snapshot(
                root,
                in_cut,
                round_idx,
                round_considered,
                visits_per_candidate,
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
    /// corresponds to `root.children[child_indices[pos]]`), scaled by
    /// `tree_q_weight` so trust-gating applies to every selection consumer.
    fn sigma_q_for(&self, root: &GumbelNode, child_indices: &[usize]) -> Vec<f32> {
        let q: Vec<f32> = child_indices
            .iter()
            .map(|&i| root.children[i].q_value())
            .collect();
        let visits: Vec<f32> = child_indices
            .iter()
            .map(|&i| root.children[i].visits)
            .collect();
        let priors: Vec<f32> = child_indices
            .iter()
            .map(|&i| root.children[i].logit)
            .collect();
        let mut sq =
            gumbel_qtransform::sigma_completed_q(root.own_value, &priors, &q, &visits, true);
        for s in &mut sq {
            *s *= self.tree_q_weight;
        }
        sq
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
        let turn_horizon = start_turn + max_turns_ahead(start_turn, game.state.settings.max_turns);

        let total_needed = round_considered * visits_per_candidate;
        let mut collected_per_candidate = vec![0usize; round_considered];
        let mut total_collected = 0;

        while total_collected < total_needed {
            let mut leaves: Vec<GumbelLeaf> = Vec::with_capacity(self.batch_size);

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
            for (leaf, &(value, _progress)) in leaves.iter().zip(values.iter()) {
                backpropagate_return_with_rewards(
                    root,
                    &leaf.data.path_indices,
                    &leaf.data.path_players,
                    &leaf.rewards,
                    &leaf.turn_deltas,
                    mcts_common::VIRTUAL_LOSS,
                    value,
                    reward::GAMMA_TURN,
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
    ) -> Option<GumbelLeaf> {
        let mut indices_stack: Vec<usize> = Vec::new();
        let mut path_players: Vec<i32> = Vec::new();
        let mut path_rewards: Vec<f32> = Vec::new();
        let mut path_turn_deltas: Vec<i32> = Vec::new();
        let mut undos: Vec<crate::actions::UndoCallback> = Vec::new();

        let root_player = game.state.settings.current_player_turn_id;
        path_players.push(root_player);

        // Virtual loss on the root.
        root.add_virtual_loss(mcts_common::VIRTUAL_LOSS);

        // Apply the candidate's move (root -> candidate), recording the
        // exact score-delta reward this move banked (in the mover's own
        // perspective) and how many turns it crossed.
        let candidate_node = root.children.get(cand_child_idx)?;
        let m = candidate_node.move_to_here.as_ref()?;
        let (my_pre, opp_pre) = reward::score_snapshot(&game.state, root_player);
        let turn_pre = game.state.settings.turn;
        let undo = game.simulate_move(m.as_ref())?;
        undos.push(undo);
        indices_stack.push(cand_child_idx);
        path_players.push(game.state.settings.current_player_turn_id);
        let (my_post, opp_post) = reward::score_snapshot(&game.state, root_player);
        let r = reward::normalized_reward(my_pre, opp_pre, my_post, opp_post);
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
            let (my_pre, opp_pre) = reward::score_snapshot(&game.state, mover);
            let turn_pre = game.state.settings.turn;
            let undo = match game.simulate_move(m.as_ref()) {
                Some(u) => u,
                None => break,
            };
            undos.push(undo);
            indices_stack.push(child_idx);
            path_players.push(game.state.settings.current_player_turn_id);
            let (my_post, opp_post) = reward::score_snapshot(&game.state, mover);
            let r = reward::normalized_reward(my_pre, opp_pre, my_post, opp_post);
            child_node.edge_reward.set(Some(r));
            path_rewards.push(r);
            path_turn_deltas.push(game.state.settings.turn - turn_pre);
        }

        let needs_expansion = match get_node_by_path(root, &indices_stack) {
            Some(c) => !c.is_expanded && !game.state.settings._game_over,
            None => false,
        };

        let mut leaf_data = extract_leaf_data(game, indices_stack, path_players, needs_expansion);

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
        })
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
        let child_visits: Vec<f32> = node.children.iter().map(|c| c.effective_visits()).collect();
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
            score_a
                .partial_cmp(&score_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Batched NN evaluation + expansion for a wave of leaves. Returns one
    /// value per leaf (terminal outcome or NN value), in leaf order.
    fn batched_evaluate_and_expand(
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
    fn expand_gumbel_node_from_precomputed(
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

        let mut logits =
            policy_composer::compute_move_log_probs_raw(policy, &legal_moves, map_size);
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
                let sa = ca.gumbel + child_priors[*a] + self.tree_q_weight * sigma_q[*a];
                let sb = cb.gumbel + child_priors[*b] + self.tree_q_weight * sigma_q[*b];
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
        self.last_search_kl = None;

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
        self.last_root_value = None;
        self.last_search_kl = None;

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

        self.record_final(&root, best_idx, move_count);

        let best_move = clone_child_move(&root, best_idx);
        let next_hash = next_root_hash_for(game, best_move.as_deref());
        self.store_tree(root, best_idx, next_hash);
        (move_or_end_turn(best_move), move_visits)
    }

    pub fn select_move_with_stats(&mut self, game: &mut Game) -> (Option<Box<dyn Move>>, Vec<f32>) {
        self.last_root_value = None;
        self.last_search_kl = None;

        let root = self.search_and_extract(game);
        if root.children.is_empty() {
            self.invalidate_tree();
            return (Some(Box::new(EndTurnMove)), Vec::new());
        }
        let move_visits = self.extract_policy_targets(&root);
        let policy: Vec<f32> = move_visits.iter().map(|mv| mv.visits).collect();
        let best_idx = self.recommend_final_move(&root);
        self.record_final(&root, best_idx, 0);
        let best_move = clone_child_move(&root, best_idx);
        let next_hash = next_root_hash_for(game, best_move.as_deref());
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
    fn store_tree(&mut self, root: GumbelNode, best_idx: usize, next_hash: Option<u64>) {
        self.last_root_value = (root.visits > 0.0).then(|| root.q_value());
        self.last_search_kl = root_policy_kl(&root);
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

/// Blend a heuristic prior into a raw logit slice in place.
/// Formula `p' = (1-w)*p_net + w*p_heur`, `p_heur = softmax(heur_scores / TEMP)`.
/// Shared by the root blend (`blend_heuristic_prior`) and in-tree expansion.
fn blend_heuristic_into_logits(logits: &mut [f32], heur_scores: &[f32], weight: f32) {
    const HEURISTIC_TEMP: f32 = 20.0;
    if logits.is_empty() || logits.len() != heur_scores.len() {
        return;
    }

    let p_net = softmax(logits);
    let scaled: Vec<f32> = heur_scores.iter().map(|s| s / HEURISTIC_TEMP).collect();
    let p_heur = softmax(&scaled);

    for (i, l) in logits.iter_mut().enumerate() {
        let p = (1.0 - weight) * p_net[i] + weight * p_heur[i];
        // Add a small epsilon to prevent log(0)
        *l = (p + 1e-9).ln();
    }
}

// Blend the heuristic prior into the network's root priors in place.
fn blend_heuristic_prior(game: &Game, children: &mut [GumbelNode], weight: f32) {
    if children.is_empty() {
        return;
    }
    let mut logits: Vec<f32> = children.iter().map(|c| c.logit).collect();
    let scores: Vec<f32> = children
        .iter()
        .map(|c| {
            c.move_to_here
                .as_ref()
                .map_or(0.0, |m| crate::ai::scoring::score_move(game, m.as_ref()))
        })
        .collect();
    blend_heuristic_into_logits(&mut logits, &scores, weight);
    for (child, l) in children.iter_mut().zip(logits.into_iter()) {
        child.logit = l;
    }
}

/// Multiset-compare a reused root's cached child moves against the real
/// state's legal moves. Any mismatch means the sim-built cache is stale.
fn reused_children_match_legal(game: &Game, children: &[GumbelNode]) -> bool {
    let mut legal = game.legal_moves();
    let has_other = legal.iter().any(|m| m.move_type() != MoveType::EndTurn);
    if has_other {
        legal.retain(|m| m.move_type() != MoveType::EndTurn);
    }

    // Mirror the EndTurn suppression applied to `legal`: interior expansion
    // keeps EndTurn children, but `finish_reused_root` strips them after this
    // check runs, so we must exclude them here too to avoid a count mismatch.
    let filtered_children: Vec<&GumbelNode> = if has_other {
        children
            .iter()
            .filter(|c| {
                c.move_to_here
                    .as_ref()
                    .map_or(true, |m| m.move_type() != MoveType::EndTurn)
            })
            .collect()
    } else {
        children.iter().collect()
    };

    if legal.len() != filtered_children.len() {
        return false;
    }
    let mut remaining: Vec<serde_json::Value> = legal.iter().map(|m| m.serialize()).collect();
    for child in &filtered_children {
        let Some(m) = child.move_to_here.as_ref() else {
            return false;
        };
        let v = m.serialize();
        match remaining.iter().position(|r| *r == v) {
            Some(i) => {
                remaining.swap_remove(i);
            }
            None => return false,
        }
    }
    true
}

/// Apply `m` to a CLONE of `game` (assumed to be at the root state, with all
/// search undos applied) via `play_move` — the same path the real game loop
/// uses — and hash the resulting state's features. This is the hash the
/// *next* search's root must match to re-root into this child. Using
/// `play_move` (not `simulate_move`) is load-bearing: the real game loop
/// advances via `play_move`, which updates `_history` and runs FOW discovery
/// that `simulate_move` skips, so a simulate-derived hash would never match
/// the next call's play-derived features.
///
/// The clone is load-bearing too: some callers (arena, trainer) pass the REAL
/// game, and mutating it here double-applied the chosen move once the caller
/// played it — an EndTurn double-apply silently consumed the opponent's whole
/// turn (arena bots never moved), and double-applied econ moves caused the
/// execute-error spam and star-debt panics.
fn next_root_hash_for(game: &Game, m: Option<&dyn Move>) -> Option<u64> {
    let m = m?;
    let mut preview = game.clone();
    let _ = preview.play_move(m)?;
    let feat = features::state_to_cpu_features(
        &preview.state,
        preview.state.settings.current_player_turn_id,
    )
    .ok()?;
    Some(feat.hash())
}

/// KL(visit distribution ‖ prior softmax) over a searched root's children.
/// `None` when fewer than two children or no visits landed (no real search).
fn root_policy_kl(root: &GumbelNode) -> Option<f32> {
    if root.children.len() < 2 || root.visits <= 0.0 {
        return None;
    }
    let total: f32 = root.children.iter().map(|c| c.visits.max(0.0)).sum();
    if total <= 0.0 {
        return None;
    }
    let max_logit = root
        .children
        .iter()
        .map(|c| c.logit)
        .fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = root
        .children
        .iter()
        .map(|c| (c.logit - max_logit).exp())
        .collect();
    let z: f32 = exps.iter().sum();
    if !z.is_finite() || z <= 0.0 {
        return None;
    }
    let mut kl = 0.0;
    for (c, e) in root.children.iter().zip(&exps) {
        let p = c.visits.max(0.0) / total;
        if p > 0.0 {
            kl += p * (p / (e / z).max(1e-8)).ln();
        }
    }
    Some(kl)
}

fn move_or_end_turn(best_move: Option<Box<dyn Move>>) -> Option<Box<dyn Move>> {
    if best_move.is_none() {
        Some(Box::new(EndTurnMove))
    } else {
        best_move
    }
}
