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
    /// Tree-depth telemetry, accumulated per simulation (leaf descent).
    /// `depth` = plies below the root on the path to the extracted leaf.
    /// `horizon_hits` counts descents cut short by `max_turns_ahead`, which
    /// tells us whether the turn horizon (not the budget) bounds depth.
    pub depth_sum: std::cell::Cell<u64>,
    pub depth_max: std::cell::Cell<u32>,
    pub depth_count: std::cell::Cell<u64>,
    pub horizon_hits: std::cell::Cell<u64>,
    /// Distillation headroom: root decisions where search's final pick equals
    /// `argmax(prior)` (`agree_count`) out of all root decisions
    /// (`decision_count`). A rising override rate with budget means deeper
    /// search finds moves the prior does not know — i.e. signal to distill.
    pub agree_count: std::cell::Cell<u64>,
    pub decision_count: std::cell::Cell<u64>,
    /// Multiplier on the root Gumbel(0,1) exploration noise. 1.0 = normal
    /// self-play exploration; 0.0 = deterministic (argmax) root selection, for
    /// eval-conditions probes. Read once from the GUMBEL_SCALE env var at
    /// construction — diagnostic knob, not a CLI flag.
    pub gumbel_scale: f32,
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
    /// Weight on the development potential Φ in in-tree edge rewards
    /// (EXP_ELO_016): snapshots become `score + w·Φ`. 0.0 = raw score
    /// deltas (bit-exact legacy path).
    pub reward_shape_w: f32,
    /// Weight on the isolated pursuit-progress potential Φ in in-tree edge
    /// rewards (EXP_ELO_018), independent of `reward_shape_w`. 0.0 = off.
    pub pursuit_shape_w: f32,
    /// EXP_ELO_028 Phase 1c: weight on `reward::goal_potential` (stance/order
    /// priced shaping) in in-tree edge rewards, applied on the root player's
    /// edges only (the opponent's goal is unknown). 0.0 = off.
    pub goal_shape_w: f32,
    /// EXP_ELO_017: when crossing an EndTurn edge, give each intervening
    /// opponent a real (deterministic-argmax Greedy) turn instead of the
    /// engine's blind auto-skip — so the tree can see contested villages/
    /// races instead of a frozen opponent. `false` = legacy behavior,
    /// bit-exact. Value backup stays single-player (root-perspective
    /// `win_value`); the opponent's ghost moves are scripted, not searched.
    pub unfreeze_opponent: bool,
    /// EXP_ELO_026 oracle-macro commitment: when set, every feature encode
    /// this agent performs (root, tree leaves, re-root hash) focuses the
    /// pursuit channel on this village alone. Cache-safe: the eval LRU and
    /// tree-reuse checks both key on the encoded feature bytes.
    pub pursuit_focus: Option<i32>,
    /// EXP_ELO_026 star gate: when true, root-level Research moves failing
    /// `oracle_macro::passes_star_gate` are dropped (root only — the tree
    /// below stays unrestricted).
    pub star_gate: bool,
    /// EXP_ELO_028 Stage-1 macro goal: painted into the appended goal
    /// channels of every encode this agent performs (root, leaves, re-root
    /// hash). Cache/tree-reuse safe: both key on feature bytes.
    pub macro_goal: Option<crate::ai::oracle_macro::MacroGoal>,
    /// EXP_ELO_028 v2.3 aux context (NOT painted): environment-fit tech bias
    /// for the goal potential + whole-game tech-purchase caps at the root.
    pub goal_aux: Option<crate::ai::oracle_macro::GoalAux>,
    /// Diagnostic capture for the next search, armed via `arm_trace`. `None`
    /// (the default) costs one `RefCell` borrow-check per call site and
    /// nothing else — see decision_trace.rs.
    trace: RefCell<Option<TraceBuilder>>,
    /// v9: the mirrored `aux_fog` map from the ROOT evaluation of the current
    /// search, reused for every in-tree Φ. Risk is a position property that
    /// moves slowly inside a turn, so a per-ply read is a fair approximation
    /// and costs nothing extra — a per-node read would mean threading each
    /// node's eval into the reward path. `None` when the backend or the
    /// checkpoint cannot produce it; `position_risk` then falls back to its
    /// state-side signals rather than to a silent zero.
    root_fog: RefCell<Option<std::sync::Arc<Vec<f32>>>>,
    /// The most recently completed search's root value (`root.q_value()`
    /// after backup — a discounted-return state-value estimate under the
    /// reward-aware backup, `None` if the root never accumulated a visit:
    /// an empty legal set, or a single-legal-move root, which
    /// `run_search` short-circuits before any visits land). Set at the end
    /// of every `select_move*` call, consumed by `self_play`'s TD label
    /// bootstrap via `Brain::last_root_value`.
    last_root_value: Option<f32>,
    /// The root's RAW NN value prediction (tanh-bounded, pre-search/pre-edge-
    /// reward), captured at the same convergence point as `last_root_value`.
    /// Diagnostic only (value-head calibration) — isolates the head from the
    /// reward-shaping backup that inflates `last_root_value`.
    last_root_own_value: Option<f32>,
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
    /// Path of a node whose cached child move failed to execute on replay.
    /// Reused subtrees are built under simulated dynamics; a real move in
    /// between (which explores tiles, unlike `simulate_move`) can change what
    /// a replayed ruin capture rolls or what is legal, stranding stale
    /// children. The wave loop clears such a node so it re-expands from the
    /// true replayed state.
    stale_path: Option<Vec<usize>>,
}

impl<'a> GumbelMctsAgent<'a> {
    pub fn new(evaluator: &'a Evaluator, iterations: usize, k: usize) -> Self {
        Self {
            evaluator,
            iterations,
            k,
            batch_size: mcts_common::DEFAULT_BATCH_SIZE,
            root_fog: RefCell::new(None),
            tree: None,
            last_chosen_idx: None,
            next_root_hash: None,
            tree_reuses: 0,
            depth_sum: std::cell::Cell::new(0),
            depth_max: std::cell::Cell::new(0),
            depth_count: std::cell::Cell::new(0),
            horizon_hits: std::cell::Cell::new(0),
            agree_count: std::cell::Cell::new(0),
            decision_count: std::cell::Cell::new(0),
            gumbel_scale: std::env::var("GUMBEL_SCALE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(1.0),
            prior_heuristic_weight: 0.0,
            policy_target_q_weight: 1.0,
            tree_q_weight: std::env::var("TREE_Q_WEIGHT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(1.0),
            reward_shape_w: 0.0,
            pursuit_shape_w: 0.0,
            goal_shape_w: 0.0,
            unfreeze_opponent: false,
            pursuit_focus: None,
            star_gate: false,
            macro_goal: None,
            goal_aux: None,
            trace: RefCell::new(None),
            last_root_value: None,
            last_root_own_value: None,
        }
    }

    /// The completed search's root value (see `last_root_value` field docs),
    /// if the most recent `select_move*` call actually ran a search.
    pub fn last_root_value(&self) -> Option<f32> {
        self.last_root_value
    }

    /// Raw NN root value of the most recent search (see field docs).
    pub fn last_root_own_value(&self) -> Option<f32> {
        self.last_root_own_value
    }

    /// Cumulative search telemetry:
    /// (depth_sum, depth_count, depth_max, horizon_hits, agree, decisions).
    /// Mean depth = depth_sum / depth_count; prior-override rate =
    /// 1 - agree/decisions. Accumulates across every search this agent has
    /// run; read once at the end of a match.
    pub fn depth_stats(&self) -> (u64, u64, u32, u64, u64, u64) {
        (
            self.depth_sum.get(),
            self.depth_count.get(),
            self.depth_max.get(),
            self.horizon_hits.get(),
            self.agree_count.get(),
            self.decision_count.get(),
        )
    }

    pub fn clear_last_root_value(&mut self) {
        self.last_root_value = None;
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
    fn finish_reused_root(&self, game: &mut Game, mut new_root: GumbelNode, start_turn: i32) -> GumbelNode {
        reset_stats_recursive(&mut new_root);

        // v8: the promoted child's children were created as INTERIOR nodes, so
        // they never met the root gates that `build_fresh_root` applies. Tree
        // reuse is the common case mid-turn (~8 plies per game turn), so
        // without this every root gate leaked on all but the first ply of a
        // turn — measured Aug 2: the pop-discipline and road gates were fully
        // inert until this was added. EndTurn is exempt so the root can never
        // be emptied; the suppression below still removes it when it should.
        if self.star_gate || self.goal_aux.is_some() {
            let stance = self.macro_goal.as_ref().map(|g| g.stance);
            new_root.children.retain(|c| {
                let Some(m) = c.move_to_here.as_ref() else {
                    return true;
                };
                if m.move_type() == MoveType::EndTurn {
                    return true;
                }
                (!self.star_gate
                    || crate::ai::oracle_macro::passes_star_gate(
                        &game.state,
                        m.as_ref(),
                        stance,
                        self.goal_aux.as_ref(),
                    ))
                    && self.goal_aux.as_ref().map_or(true, |a| {
                        crate::ai::oracle_macro::passes_tech_caps(m.as_ref(), a)
                            && crate::ai::oracle_macro::passes_ability_gate(&game.state, m.as_ref())
                            && crate::ai::oracle_macro::passes_capture_first(
                                &game.state,
                                m.as_ref(),
                            )
                    })
            });
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

        let mut in_cut = self.build_in_cut(&new_root);
        self.run_search(game, &mut new_root, &mut in_cut, start_turn);
        new_root
    }

    /// Fresh root: evaluate with the NN, create one child per legal move with
    /// fresh Gumbel draws, build `in_cut`, and run Sequential Halving.
    fn build_fresh_root(&self, game: &mut Game, features: RawFeatures, start_turn: i32) -> GumbelNode {
        let results = self.evaluator.evaluate(vec![features]);
        let (root_value, root_progress, ref policy_row) = results[0];
        // v9: stash this search's fog map for the reward path (see `root_fog`).
        *self.root_fog.borrow_mut() =
            policy_row.fog.as_ref().map(|f| std::sync::Arc::new(f.clone()));

        let mut legal_moves = game.legal_moves();
        if self.star_gate || self.goal_aux.is_some() {
            let stance = self.macro_goal.as_ref().map(|g| g.stance);
            legal_moves.retain(|m| {
                (!self.star_gate
                    || crate::ai::oracle_macro::passes_star_gate(&game.state, m.as_ref(), stance, self.goal_aux.as_ref()))
                    && self.goal_aux.as_ref().map_or(true, |a| {
                        crate::ai::oracle_macro::passes_tech_caps(m.as_ref(), a)
                            && crate::ai::oracle_macro::passes_ability_gate(&game.state, m.as_ref())
                            && crate::ai::oracle_macro::passes_capture_first(
                                &game.state,
                                m.as_ref(),
                            )
                    })
            });
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
            self.record_round_snapshot(root, in_cut, round_idx, round_considered, visits_per_candidate);
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
        let q: Vec<f32> = child_indices.iter().map(|&i| root.children[i].q_value()).collect();
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
        let turn_horizon =
            start_turn + max_turns_ahead(start_turn, game.state.settings.max_turns);

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

            // Heal nodes whose cached children proved stale on replay (see
            // GumbelLeaf::stale_path): drop the children and mark unexpanded
            // so the next visit re-expands from the true replayed state.
            // Deferred to after backprop because other leaves in this wave may
            // hold path indices through a node being cleared.
            for leaf in &leaves {
                if let Some(path) = &leaf.stale_path {
                    if let Some(node) = get_node_by_path_mut(root, path) {
                        node.children.clear();
                        node.is_expanded = false;
                    }
                }
            }
        }
    }

    /// Cross an `EndTurn` edge. With `unfreeze_opponent` off (default), this
    /// is exactly `game.simulate_move(&EndTurnMove)` — the engine's built-in
    /// auto-skip past every opponent back to the searching player, bit-exact
    /// with all prior behavior. With it on (EXP_ELO_017), the searcher's own
    /// turn ends once, then each intervening opponent plays a REAL turn via
    /// the deterministic (argmax) Greedy heuristic — cheap, NN-free — until
    /// control returns to the searcher. Two-player only: asserts that
    /// happens within `MAX_GHOST_MOVES`, loud rather than silently wrong on
    /// a future 3+-player run.
    /// Edge-reward snapshot: the shared shaped snapshot plus the goal
    /// potential on the root player's own edges (their goal is the only one
    /// this search knows). Pre/post of an edge always use the same (mover,
    /// root) pair, so the added term stays a consistent potential.
    fn edge_snapshot(
        &self,
        state: &crate::states::GameState,
        mover: i32,
        root_player: i32,
    ) -> (f32, f32) {
        let (mut my, opp) =
            reward::shaped_snapshot(state, mover, self.reward_shape_w, self.pursuit_shape_w);
        if self.goal_shape_w != 0.0 && mover == root_player {
            if let Some(goal) = &self.macro_goal {
                let fog = self.root_fog.borrow();
                my += self.goal_shape_w
                    * reward::goal_potential_with_fog(
                        state,
                        mover,
                        goal,
                        self.goal_aux.as_ref(),
                        fog.as_ref().map(|f| f.as_slice()),
                    );
            }
        }
        (my, opp)
    }

    fn cross_end_turn(game: &mut Game, unfreeze_opponent: bool) -> Option<crate::actions::UndoCallback> {
        if !unfreeze_opponent {
            return game.simulate_move(&EndTurnMove);
        }
        // Snapshot the whole state and restore it wholesale on undo, rather
        // than composing per-move undo callbacks across the ghost turn. A ghost
        // turn can include combat, and the kill-undo's index-patching
        // (units.rs) does NOT compose when several moves are bundled and
        // unwound in reverse — it panicked on `Vec::insert` out-of-bounds.
        // Restoring a clone is correct under the wave loop's LIFO unwind: by
        // the time this undo runs, every later edge is already undone, so the
        // live state equals the end-of-ghost-turn state and overwriting it with
        // the pre-ghost-turn snapshot restores exactly this edge. Costs one
        // GameState clone per turn crossing — unfreeze is the expensive-but-
        // robust training path; gauges/arena never cross unfrozen.
        const MAX_GHOST_MOVES: usize = 64;
        let searcher = game.state.settings.current_player_turn_id;
        let snapshot = game.state.clone();
        game.simulate_single_end_turn();
        let mut n = 0usize;
        while game.state.settings.current_player_turn_id != searcher
            && !game.state.settings._game_over
        {
            // Any ghost-turn anomaly (Greedy can't move, an illegal move, or a
            // runaway loop) degrades gracefully to a stale-node signal instead
            // of panicking mid-search and discarding the whole game's data.
            let bail = n >= MAX_GHOST_MOVES
                || match crate::ai::heuristic_mcts::GreedyHeuristicAgent.select_move(game) {
                    None => true,
                    Some(mv) if mv.move_type() == MoveType::EndTurn => {
                        game.simulate_single_end_turn();
                        false
                    }
                    Some(mv) => game.simulate_move(mv.as_ref()).is_none(),
                };
            if bail {
                game.state = snapshot.clone();
                return None;
            }
            n += 1;
        }
        Some(Box::new(move |s: &mut crate::states::GameState| {
            *s = snapshot;
        }))
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

        let root = self.search_and_extract(game);
        if root.children.is_empty() {
            self.invalidate_tree();
            return Some(Box::new(EndTurnMove));
        }
        let best_idx = self.recommend_final_move(&root);
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
                Ok(dist) => dist.sample(&mut rand::thread_rng()),
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
    fn store_tree(&mut self, root: GumbelNode, best_idx: usize, next_hash: Option<u64>) {
        self.last_root_value = (root.visits > 0.0).then(|| root.q_value());
        self.last_root_own_value = (root.visits > 0.0).then(|| root.own_value);
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
    let scores: Vec<f32> = children.iter()
        .map(|c| c.move_to_here.as_ref()
            .map_or(0.0, |m| crate::ai::scoring::score_move(game, m.as_ref())))
        .collect();
    blend_heuristic_into_logits(&mut logits, &scores, weight);
    for (child, l) in children.iter_mut().zip(logits.into_iter()) {
        child.logit = l;
    }
}

/// Multiset-compare a reused root's cached child moves against the real
/// state's legal moves. Any mismatch means the sim-built cache is stale.
///
/// Every expansion path (`build_fresh_root`, and `extract_leaf_data` in
/// `mcts_common.rs` for interior nodes) drops `EndTurn` from a node's
/// children whenever another move is legal, keeping it only when it's the
/// sole option. The cached `children` were built through one of those paths,
/// so the comparison must apply the same normalization to `game.legal_moves()`
/// — otherwise EndTurn's presence in the raw legal set spuriously fails the
/// match on every reuse attempt where any other move exists.
fn reused_children_match_legal(
    game: &Game,
    children: &[GumbelNode],
    star_gate: bool,
    stance: Option<crate::ai::oracle_macro::Stance>,
    goal_aux: Option<&crate::ai::oracle_macro::GoalAux>,
) -> bool {
    let mut legal = game.legal_moves();
    if star_gate || goal_aux.is_some() {
        legal.retain(|m| {
            (!star_gate
                || crate::ai::oracle_macro::passes_star_gate(&game.state, m.as_ref(), stance, goal_aux))
                && goal_aux.map_or(true, |a| {
                    crate::ai::oracle_macro::passes_tech_caps(m.as_ref(), a)
                        && crate::ai::oracle_macro::passes_ability_gate(&game.state, m.as_ref())
                        && crate::ai::oracle_macro::passes_capture_first(&game.state, m.as_ref())
                })
        });
    }
    let has_other = legal.iter().any(|m| m.move_type() != MoveType::EndTurn);
    if has_other {
        legal.retain(|m| m.move_type() != MoveType::EndTurn);
    }
    if legal.len() != children.len() {
        return false;
    }
    let mut remaining: Vec<serde_json::Value> = legal.iter().map(|m| m.serialize()).collect();
    for child in children {
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
fn next_root_hash_for(
    game: &mut Game,
    m: Option<&dyn Move>,
    pursuit_focus: Option<i32>,
    macro_goal: Option<&crate::ai::oracle_macro::MacroGoal>,
) -> Option<u64> {
    let m = m?;
    // The undo callback is intentionally dropped because the game is a per-call clone
    let _ = game.play_move(m)?;
    let feat = features::state_to_cpu_features_goal(
        &game.state,
        game.state.settings.current_player_turn_id,
        pursuit_focus,
        macro_goal,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::eval_server::InlineEvalHandle;
    use crate::ai::network::PolyZeroNet;
    use crate::game::{Game, SIM_MOVE_FAILURES};
    use crate::moves::research::ResearchMove;
    use crate::types::{MapSize, MapType, TechnologyType, TribeType};
    use std::sync::Arc;

    /// A reused subtree can hold moves that are illegal in the replayed state
    /// (real moves explore tiles that `simulate_move` deliberately does not,
    /// which e.g. changes what a replayed ruin capture rolls). The descent
    /// must heal such a node — drop its stale children and re-expand from the
    /// true state — rather than leave it failing forever.
    #[test]
    fn test_search_heals_stale_cached_children() {
        let device = candle_core::Device::Cpu;
        let varmap = candle_nn::VarMap::new();
        let vs =
            candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
        let network = Arc::new(PolyZeroNet::new(vs).unwrap());
        let evaluator = Evaluator::Inline(InlineEvalHandle::new(network));

        let mut game = Game::new();
        game.state = crate::mapgen::generate(crate::mapgen::MapGenSettings {
            size: MapSize::Tiny,
            map_type: MapType::Drylands,
            tribes: vec![TribeType::Imperius, TribeType::Bardur],
            seed: 7,
            ..Default::default()
        });
        game.post_load();

        // Broke: no research is affordable (min cost 4), so the poisoned
        // cached Research below is illegal no matter what the path does.
        let pov = game.state.settings.current_player_turn_id;
        game.state.tribes.get_mut(&pov).unwrap().stars = 0;

        let legal = game.legal_moves();
        assert!(legal.len() >= 2, "need at least two legal moves");
        // A free move for the poisoned branch so the descent reaches it.
        let step_idx = legal
            .iter()
            .position(|m| m.move_type() == MoveType::Step)
            .expect("expected a Step at game start");

        let mut root = GumbelNode::new(0.0, 0.0, None);
        root.is_expanded = true;
        for (i, m) in legal.into_iter().enumerate() {
            let mut child = GumbelNode::new(0.0, 0.0, Some(m));
            if i == step_idx {
                // Simulate a stale reused subtree: cached child that is
                // illegal in the replayed state.
                child.is_expanded = true;
                child.children.push(GumbelNode::new(
                    0.0,
                    0.0,
                    Some(Box::new(ResearchMove::new(TechnologyType::Trade))),
                ));
            }
            root.children.push(child);
        }

        let other_idx = if step_idx == 0 { 1 } else { 0 };
        let mut in_cut = vec![step_idx, other_idx];

        let agent = GumbelMctsAgent::new(&evaluator, 16, 2);
        let failures_before = SIM_MOVE_FAILURES.load(std::sync::atomic::Ordering::Relaxed);
        let start_turn = game.state.settings.turn;
        agent.run_search(&mut game, &mut root, &mut in_cut, start_turn);
        let failures_after = SIM_MOVE_FAILURES.load(std::sync::atomic::Ordering::Relaxed);

        assert!(
            failures_after > failures_before,
            "test setup failed to exercise the stale cached move"
        );

        // The poisoned node must have been healed: its stale Research child
        // gone, replaced by nothing (pending re-expansion) or by the true
        // legal set of the replayed state.
        let node = &root.children[step_idx];
        let still_poisoned = node.children.iter().any(|c| {
            c.move_to_here
                .as_ref()
                .map_or(false, |m| m.move_type() == MoveType::Research)
        });
        assert!(
            !still_poisoned,
            "stale illegal Research child survived the search"
        );
        if node.is_expanded {
            assert!(
                !node.children.is_empty(),
                "healed node re-expanded with no children"
            );
        }
    }

    fn fresh_two_tribe_game(seed: i64) -> Game {
        let mut game = Game::new();
        game.state = crate::mapgen::generate(crate::mapgen::MapGenSettings {
            size: MapSize::Tiny,
            map_type: MapType::Drylands,
            tribes: vec![TribeType::Imperius, TribeType::Bardur],
            seed,
            ..Default::default()
        });
        game.post_load();
        game
    }

    /// `unfreeze_opponent = false` must be bit-exact with the engine's
    /// built-in auto-skip (`simulate_move(&EndTurnMove)`) — no behavior
    /// change for every existing arm/gauge that never sets the flag.
    #[test]
    fn cross_end_turn_off_matches_legacy_auto_skip() {
        let mut via_helper = fresh_two_tribe_game(11);
        let mut via_direct = fresh_two_tribe_game(11);

        let undo_helper = GumbelMctsAgent::cross_end_turn(&mut via_helper, false)
            .expect("legacy path should always succeed from a fresh game");
        let undo_direct = via_direct
            .simulate_move(&EndTurnMove)
            .expect("direct auto-skip should succeed");

        assert_eq!(
            serde_json::to_string(&via_helper.state).unwrap(),
            serde_json::to_string(&via_direct.state).unwrap(),
            "cross_end_turn(false) must match simulate_move(&EndTurnMove) exactly"
        );

        // And undo must cleanly restore both to their (identical) starting states.
        let before = serde_json::to_string(&fresh_two_tribe_game(11).state).unwrap();
        undo_helper(&mut via_helper.state);
        undo_direct(&mut via_direct.state);
        assert_eq!(serde_json::to_string(&via_helper.state).unwrap(), before);
        assert_eq!(serde_json::to_string(&via_direct.state).unwrap(), before);
    }

    /// `unfreeze_opponent = true` must give the opponent a REAL turn (state
    /// actually changes beyond just the turn/player counters) and must still
    /// return control to the original searching player — the 2-player
    /// round-trip the implementation asserts internally.
    #[test]
    fn cross_end_turn_on_gives_opponent_a_real_turn_and_returns_to_searcher() {
        let mut frozen = fresh_two_tribe_game(11);
        let mut unfrozen = fresh_two_tribe_game(11);
        let searcher = frozen.state.settings.current_player_turn_id;

        GumbelMctsAgent::cross_end_turn(&mut frozen, false).unwrap();
        GumbelMctsAgent::cross_end_turn(&mut unfrozen, true).unwrap();

        assert_eq!(
            unfrozen.state.settings.current_player_turn_id, searcher,
            "control must return to the searching player after one crossing"
        );
        assert_ne!(
            serde_json::to_string(&frozen.state).unwrap(),
            serde_json::to_string(&unfrozen.state).unwrap(),
            "unfreeze_opponent=true must leave a different state than a blind skip \
             (the ghost should actually have acted)"
        );

        // Undo must cleanly restore to the true starting state, unwinding
        // every ghost move (not just the searcher's own turn-end).
        let before = serde_json::to_string(&fresh_two_tribe_game(11).state).unwrap();
        let mut replay = fresh_two_tribe_game(11);
        let undo = GumbelMctsAgent::cross_end_turn(&mut replay, true).unwrap();
        undo(&mut replay.state);
        let after = serde_json::to_string(&replay.state).unwrap();
        if after != before {
            std::fs::write("/tmp/before.json", &before).unwrap();
            std::fs::write("/tmp/after.json", &after).unwrap();
        }
        assert_eq!(after, before);
    }

    /// The Greedy ghost must be deterministic (argmax, not the temperature-
    /// sampled early-game path) — otherwise the same tree edge would resolve
    /// to different opponent states on different visits, silently breaking
    /// MCTS (a re-visited node's cached children would no longer match the
    /// state they were expanded from).
    #[test]
    fn cross_end_turn_on_is_deterministic_across_repeated_crossings() {
        let mut a = fresh_two_tribe_game(42);
        let mut b = fresh_two_tribe_game(42);
        GumbelMctsAgent::cross_end_turn(&mut a, true).unwrap();
        GumbelMctsAgent::cross_end_turn(&mut b, true).unwrap();
        assert_eq!(
            serde_json::to_string(&a.state).unwrap(),
            serde_json::to_string(&b.state).unwrap(),
            "identical starting states must resolve to identical ghost turns"
        );
    }
}
