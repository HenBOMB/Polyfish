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
//!
//! Aug 2026: split into gumbel_mcts/{trace,root,rounds,expand,finish,gate,
//! reuse}.rs so no file exceeds ~1000 lines. Each holds a second
//! `impl<'a> GumbelMctsAgent<'a>` block for its group of methods — Rust
//! merges impl blocks for the same type across files, so every
//! `agent.method()` call site keeps resolving unchanged with zero logic
//! moved. `gate_stats` is re-exported so its external call site keeps
//! resolving too.

use crate::ai::decision_trace::{DecisionTrace, TraceBuilder};
use crate::ai::eval_server::Evaluator;
use crate::ai::mcts_common::{self, BackpropNode, LeafData, TreeNode};
use crate::moves::{EndTurnMove, Move};
use crate::types::MoveType;
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
}

mod expand;
mod finish;
mod gate;
mod reuse;
mod root;
mod rounds;
mod trace;

pub use gate::gate_stats;

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
