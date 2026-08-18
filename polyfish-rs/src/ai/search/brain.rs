use crate::ai::eval_server::Evaluator;
use crate::ai::gumbel_mcts::GumbelMctsAgent;
use crate::ai::macro_agent::{MacroLookaheadAgent, MacroParams, MacroScriptAgent};
use crate::ai::mcts_types::MoveVisit;
use crate::ai::mcts_zero::ZeroMctsAgent;
use crate::game::Game;
use crate::moves::{Move, generate_legal_moves};

/// Which search backend `Brain` should use to select moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchBackend {
    Zero,
    Gumbel { k: usize },
    /// Network-free heuristic MCTS (`heuristic_mcts.rs`). Used to generate
    /// imitation/bootstrap corpora — the evaluator is never called.
    Heuristic,
    /// Zero-search softmax over `ordering::score_move` — the fastest teacher
    /// for bulk imitation corpora. No evaluator, no tree.
    Greedy,
    /// EXP_ELO_032: scripted directive + deterministic whole-turn executor
    /// (Stage 0 of the macro bootstrap). Params via `MacroParams`.
    MacroScript,
    /// EXP_ELO_032: shallow macro lookahead over candidate directives
    /// (Stage 1). Params via `MacroParams`.
    MacroLookahead,
    /// EXP_ELO_033: adversarial turn-level MCTS over directives (Stage 2).
    MacroMcts,
}

impl Default for SearchBackend {
    fn default() -> Self {
        SearchBackend::Zero
    }
}

/// Backend choice as parsed from CLI args (clap needs a unit-ish enum; the
/// Gumbel `k` is supplied separately via `--gumbel-k`).
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchBackendArg {
    Zero,
    Gumbel,
    Heuristic,
    Greedy,
    MacroScript,
    MacroLookahead,
    MacroMcts,
}

impl From<SearchBackendArg> for SearchBackend {
    fn from(arg: SearchBackendArg) -> Self {
        match arg {
            SearchBackendArg::Zero => SearchBackend::Zero,
            SearchBackendArg::Gumbel => SearchBackend::Gumbel { k: 16 },
            SearchBackendArg::Heuristic => SearchBackend::Heuristic,
            SearchBackendArg::Greedy => SearchBackend::Greedy,
            SearchBackendArg::MacroScript => SearchBackend::MacroScript,
            SearchBackendArg::MacroLookahead => SearchBackend::MacroLookahead,
            SearchBackendArg::MacroMcts => SearchBackend::MacroMcts,
        }
    }
}

// class brain
pub struct Brain<'a> {
    pub evaluator: &'a Evaluator,
    pub max_iterations: usize,
    pub backend: SearchBackend,
    /// Per-game virtual-loss mini-batch size (leaves coalesced per NN call
    /// within a single game's search). `None` keeps each agent's own
    /// default. Cross-game batching (`EvalServer`) supplies GPU efficiency
    /// independently of this, so self-play can shrink it toward sequential
    /// per-game search without losing throughput.
    pub leaf_batch: Option<usize>,
    /// Lazily-built concrete search agent, held across calls so the agent can
    /// keep its MCTS tree between consecutive same-player searches (structure-
    /// only root-shift reuse; see `gumbel_mcts.rs`). Built once on the first
    /// `think_*` call from `backend` / `evaluator` / `max_iterations` /
    /// `leaf_batch`. The borrow is of the underlying `Evaluator` for lifetime
    /// `'a` (a `Copy` shared reference), not of `self`, so storing it here is
    /// not self-referential.
    agent: Option<SearchAgent<'a>>,
    /// Weight for blending the `ordering::score_move` heuristic prior into the
    /// Gumbel backend's root priors. `None` = pure network policy. Ignored by the Zero backend.
    prior_heuristic_weight: Option<f32>,
    /// β on σ(completed-Q) in the Gumbel backend's exported policy targets.
    /// `None` = 1.0 (full paper behavior). Ignored by other backends.
    policy_target_q_weight: Option<f32>,
    /// β_tree on σ(completed-Q) inside the Gumbel search (selection, halving
    /// re-rank, final recommendation). `None` = 1.0. Ignored by other backends.
    tree_q_weight: Option<f32>,
    /// Weight on the development potential Φ in the Gumbel backend's in-tree
    /// edge rewards (EXP_ELO_016). `None` = 0.0 (raw score deltas).
    reward_shape_w: Option<f32>,
    /// Weight on the isolated pursuit-progress potential Φ in the Gumbel
    /// backend's in-tree edge rewards (EXP_ELO_018). `None` = 0.0.
    pursuit_shape_w: Option<f32>,
    /// EXP_ELO_028 Phase 1c: weight on the goal potential (stance/order
    /// priced in-tree shaping, Gumbel backend only). `None` = 0.0.
    goal_shape_w: Option<f32>,
    /// EXP_ELO_017: unfreeze the opponent during in-tree EndTurn crossings
    /// (Gumbel backend only). `None` = false (legacy blind auto-skip).
    unfreeze_opponent: Option<bool>,
    /// Set by `request_trace`; consumed (and cleared) by the next
    /// `think_decomposed` call, which arms the underlying agent's tracer.
    pending_trace: bool,
    /// EXP_ELO_028 Stage-1 macro goal for the next search(es). Re-applied to
    /// the Gumbel agent on every `think`; ignored by other backends.
    macro_goal: Option<crate::ai::oracle_macro::MacroGoal>,
    /// EXP_ELO_026/028 star gate flag applied alongside `macro_goal`.
    macro_star_gate: bool,
    /// EXP_ELO_028 v2.3 aux context applied alongside `macro_goal`.
    goal_aux: Option<crate::ai::oracle_macro::GoalAux>,
    /// Macro-backend search parameters. `None` = `MacroParams::default()`
    /// (heuristic leaf) — which is what every MACRO_GEN round silently ran
    /// before this was threaded through.
    macro_params: Option<MacroParams>,
}

/// Internal enum wrapping whichever concrete agent the configured backend
/// produced. Matched once per `think_decomposed` / `think_with_stats` call.
///
/// Exposed publicly so `arena.rs` can dispatch over backends without
/// duplicating the enum.
pub enum SearchAgent<'a> {
    Zero(ZeroMctsAgent<'a>),
    Gumbel(GumbelMctsAgent<'a>),
    Heuristic(crate::ai::heuristic_mcts::HeuristicMctsAgent),
    Greedy(crate::ai::heuristic_mcts::GreedyHeuristicAgent),
    MacroScript(MacroScriptAgent),
    MacroLookahead(MacroLookaheadAgent<'a>),
    MacroMcts(crate::ai::macro_mcts::MacroMctsAgent<'a>),
}

impl<'a> SearchAgent<'a> {
    pub fn select_move(&mut self, game: &mut Game) -> Option<Box<dyn Move>> {
        match self {
            SearchAgent::Zero(a) => a.select_move(game),
            SearchAgent::Gumbel(a) => a.select_move(game),
            SearchAgent::Heuristic(a) => a.select_move(game),
            SearchAgent::Greedy(a) => a.select_move(game),
            SearchAgent::MacroScript(a) => a.select_move(game),
            SearchAgent::MacroLookahead(a) => a.select_move(game),
            SearchAgent::MacroMcts(a) => a.select_move(game),
        }
    }

    fn select_move_with_decomposed_visits(
        &mut self,
        game: &mut Game,
        move_count: usize,
    ) -> (Option<Box<dyn Move>>, Vec<MoveVisit>) {
        match self {
            SearchAgent::Zero(a) => a.select_move_with_decomposed_visits(game, move_count),
            SearchAgent::Gumbel(a) => a.select_move_with_decomposed_visits(game, move_count),
            SearchAgent::Heuristic(a) => a.select_move_with_decomposed_visits(game, move_count),
            SearchAgent::Greedy(a) => a.select_move_with_decomposed_visits(game, move_count),
            SearchAgent::MacroScript(a) => (a.select_move(game), Vec::new()),
            SearchAgent::MacroLookahead(a) => (a.select_move(game), Vec::new()),
            // Stage 3: the macro tree's executed move becomes a one-hot
            // policy target — behavior-cloning data for the training loop.
            SearchAgent::MacroMcts(a) => {
                let m = a.select_move(game);
                let visits = m
                    .as_ref()
                    .map(|mv| vec![MoveVisit::one_hot(mv.as_ref())])
                    .unwrap_or_default();
                (m, visits)
            }
        }
    }

    fn select_move_with_stats(&mut self, game: &mut Game) -> (Option<Box<dyn Move>>, Vec<f32>) {
        match self {
            SearchAgent::Zero(a) => a.select_move_with_stats(game),
            SearchAgent::Gumbel(a) => a.select_move_with_stats(game),
            // No NN priors to report stats over; the move is all callers need.
            SearchAgent::Heuristic(a) => (a.select_move(game), Vec::new()),
            SearchAgent::Greedy(a) => (a.select_move(game), Vec::new()),
            SearchAgent::MacroScript(a) => (a.select_move(game), Vec::new()),
            SearchAgent::MacroLookahead(a) => (a.select_move(game), Vec::new()),
            SearchAgent::MacroMcts(a) => (a.select_move(game), Vec::new()),
        }
    }

    /// Arm decision-trace capture for the next search. No-op for backends
    /// other than Gumbel (the only one self-play's diagnostics target).
    fn arm_trace(&mut self) {
        if let SearchAgent::Gumbel(a) = self {
            a.arm_trace();
        }
    }

    fn take_trace(&mut self) -> Option<crate::ai::decision_trace::DecisionTrace> {
        match self {
            SearchAgent::Gumbel(a) => a.take_trace(),
            _ => None,
        }
    }

    /// The most recently completed search's root value (see
    /// `GumbelMctsAgent::last_root_value`). Gumbel reports per-ply; MacroMcts
    /// reports the committed directive's root Q once per turn (net leaf only
    /// — see `MacroMctsAgent::last_root_value`), which lands on the turn
    /// boundary the TD checkpoints key off. `None` elsewhere.
    fn last_root_value(&self) -> Option<f32> {
        match self {
            SearchAgent::Gumbel(a) => a.last_root_value(),
            SearchAgent::MacroMcts(a) => a.last_root_value(),
            _ => None,
        }
    }

    fn last_root_own_value(&self) -> Option<f32> {
        match self {
            SearchAgent::Gumbel(a) => a.last_root_own_value(),
            _ => None,
        }
    }

    /// Tier-1 state when this agent is the macro strategist.
    pub fn macro_playstyle(&self) -> Option<&crate::ai::oracle_macro::LaneState> {
        match self {
            SearchAgent::MacroMcts(a) => Some(a.committed_playstyle()),
            _ => None,
        }
    }

    /// Tier-2: the directive the macro strategist committed for this turn.
    pub fn macro_committed_goal(&self) -> Option<&crate::ai::oracle_macro::MacroGoal> {
        match self {
            SearchAgent::MacroMcts(a) => a.committed_goal(),
            _ => None,
        }
    }

    pub fn depth_stats(&self) -> Option<(u64, u64, u32, u64, u64, u64)> {
        match self {
            SearchAgent::Gumbel(a) => Some(a.depth_stats()),
            _ => None,
        }
    }

    /// Clear the cached root value from the previous search. Useful when
    /// an early return (e.g. forced move) bypasses the search engine but
    /// keeps the agent alive for tree reuse.
    fn clear_last_root_value(&mut self) {
        if let SearchAgent::Gumbel(a) = self {
            a.clear_last_root_value();
        }
    }
}

/// Construct the concrete search agent for a backend, borrowing `evaluator`.
pub fn make_search_agent(
    backend: SearchBackend,
    evaluator: &Evaluator,
    iterations: usize,
    leaf_batch: Option<usize>,
    prior_heuristic_weight: Option<f32>,
    policy_target_q_weight: Option<f32>,
    tree_q_weight: Option<f32>,
    reward_shape_w: Option<f32>,
    pursuit_shape_w: Option<f32>,
    goal_shape_w: Option<f32>,
    unfreeze_opponent: Option<bool>,
    macro_params: Option<MacroParams>,
) -> SearchAgent<'_> {
    match backend {
        SearchBackend::Zero => {
            let mut agent = ZeroMctsAgent::new(evaluator, iterations);
            if let Some(b) = leaf_batch {
                agent.batch_size = b;
            }
            SearchAgent::Zero(agent)
        }
        SearchBackend::Gumbel { k } => {
            let mut agent = GumbelMctsAgent::new(evaluator, iterations, k);
            if let Some(b) = leaf_batch {
                agent.batch_size = b;
            }
            if let Some(w) = prior_heuristic_weight {
                agent.prior_heuristic_weight = w;
            }
            if let Some(b) = policy_target_q_weight {
                agent.policy_target_q_weight = b;
            }
            if let Some(b) = tree_q_weight {
                agent.tree_q_weight = b;
            }
            if let Some(w) = reward_shape_w {
                agent.reward_shape_w = w;
            }
            if let Some(w) = pursuit_shape_w {
                agent.pursuit_shape_w = w;
            }
            if let Some(w) = goal_shape_w {
                agent.goal_shape_w = w;
            }
            if let Some(u) = unfreeze_opponent {
                agent.unfreeze_opponent = u;
            }
            SearchAgent::Gumbel(agent)
        }
        SearchBackend::Heuristic => SearchAgent::Heuristic(
            crate::ai::heuristic_mcts::HeuristicMctsAgent::new(iterations),
        ),
        SearchBackend::Greedy => {
            SearchAgent::Greedy(crate::ai::heuristic_mcts::GreedyHeuristicAgent)
        }
        SearchBackend::MacroScript => SearchAgent::MacroScript(MacroScriptAgent::new(
            macro_params.unwrap_or_default().lambda,
        )),
        SearchBackend::MacroLookahead => SearchAgent::MacroLookahead(
            MacroLookaheadAgent::new(evaluator, macro_params.unwrap_or_default()),
        ),
        SearchBackend::MacroMcts => SearchAgent::MacroMcts(
            crate::ai::macro_mcts::MacroMctsAgent::new(evaluator, macro_params.unwrap_or_default()),
        ),
    }
}

impl<'a> Brain<'a> {
    pub fn new(evaluator: &'a Evaluator, max_iterations: usize) -> Self {
        Self {
            evaluator,
            max_iterations,
            backend: SearchBackend::default(),
            leaf_batch: None,
            agent: None,
            prior_heuristic_weight: None,
            policy_target_q_weight: None,
            tree_q_weight: None,
            reward_shape_w: None,
            pursuit_shape_w: None,
            goal_shape_w: None,
            unfreeze_opponent: None,
            pending_trace: false,
            macro_goal: None,
            macro_star_gate: false,
            goal_aux: None,
            macro_params: None,
        }
    }

    pub fn with_backend(
        evaluator: &'a Evaluator,
        max_iterations: usize,
        backend: SearchBackend,
    ) -> Self {
        Self {
            evaluator,
            max_iterations,
            backend,
            leaf_batch: None,
            agent: None,
            prior_heuristic_weight: None,
            policy_target_q_weight: None,
            tree_q_weight: None,
            reward_shape_w: None,
            pursuit_shape_w: None,
            goal_shape_w: None,
            unfreeze_opponent: None,
            pending_trace: false,
            macro_goal: None,
            macro_star_gate: false,
            goal_aux: None,
            macro_params: None,
        }
    }

    /// EXP_ELO_028: set the macro goal (and star gate) the next searches run
    /// under. Call per ply; cleared by passing `None`/`false`. Applied to the
    /// Gumbel agent on every `think`; ignored by other backends.
    pub fn set_macro_goal(
        &mut self,
        goal: Option<crate::ai::oracle_macro::MacroGoal>,
        star_gate: bool,
    ) {
        self.macro_goal = goal;
        self.macro_star_gate = star_gate;
    }

    /// EXP_ELO_028 v2.3: set the per-ply aux context (environment-fit tech
    /// bias + whole-game purchase caps). Same lifecycle as `set_macro_goal`.
    pub fn set_goal_aux(&mut self, aux: Option<crate::ai::oracle_macro::GoalAux>) {
        self.goal_aux = aux;
    }

    /// Override the per-game virtual-loss mini-batch size (see `--leaf-batch`
    /// in self_play). Builder-style: chain after `with_backend`.
    pub fn with_leaf_batch(mut self, leaf_batch: usize) -> Self {
        self.leaf_batch = Some(leaf_batch);
        self
    }

    /// Tier-1 state of the macro agent (committed lane, tenure, budget,
    /// last per-lane scores). `None` for every other backend.
    pub fn macro_committed_playstyle(
        &self,
    ) -> Option<&crate::ai::oracle_macro::LaneState> {
        match &self.agent {
            Some(SearchAgent::MacroMcts(a)) => Some(a.committed_playstyle()),
            _ => None,
        }
    }

    /// Macro-backend search parameters (leaf kind, sims, k, λ, shaping).
    /// Builder-style: chain after `with_backend`. Without this the macro
    /// backends run `MacroParams::default()` — notably the HEURISTIC leaf,
    /// so the network is never consulted at the leaf.
    pub fn with_macro_params(mut self, macro_params: MacroParams) -> Self {
        self.macro_params = Some(macro_params);
        self
    }

    /// Override the prior heuristic weight. Builder style: chain after `with_backend`.
    pub fn with_prior_heuristic_weight(mut self, prior_heuristic_weight: f32) -> Self {
        self.prior_heuristic_weight = Some(prior_heuristic_weight);
        self
    }

    /// Override β on σ(Q) in exported policy targets. Builder style: chain
    /// after `with_backend`.
    pub fn with_policy_target_q_weight(mut self, policy_target_q_weight: f32) -> Self {
        self.policy_target_q_weight = Some(policy_target_q_weight);
        self
    }

    /// Override β_tree on σ(Q) inside the search. Builder style: chain after
    /// `with_backend`.
    pub fn with_tree_q_weight(mut self, tree_q_weight: f32) -> Self {
        self.tree_q_weight = Some(tree_q_weight);
        self
    }

    /// Override the in-tree development-potential shaping weight
    /// (EXP_ELO_016). Builder style: chain after `with_backend`.
    pub fn with_reward_shape_w(mut self, reward_shape_w: f32) -> Self {
        self.reward_shape_w = Some(reward_shape_w);
        self
    }

    /// Override the in-tree pursuit-progress shaping weight (EXP_ELO_018).
    /// Builder style: chain after `with_backend`.
    pub fn with_goal_shape_w(mut self, goal_shape_w: f32) -> Self {
        self.goal_shape_w = Some(goal_shape_w);
        self
    }

    pub fn with_pursuit_shape_w(mut self, pursuit_shape_w: f32) -> Self {
        self.pursuit_shape_w = Some(pursuit_shape_w);
        self
    }

    /// Override the in-tree opponent-unfreeze flag (EXP_ELO_017). Builder
    /// style: chain after `with_backend`.
    pub fn with_unfreeze_opponent(mut self, unfreeze_opponent: bool) -> Self {
        self.unfreeze_opponent = Some(unfreeze_opponent);
        self
    }

    fn _get_iterations(&self, turn: i32, legal_move_count: usize) -> usize {
        let mut iterations = self.max_iterations;

        if legal_move_count == 1 {
            return 0;
        }

        if legal_move_count < 4 {
            iterations = 10;
        } else if turn < 3 || legal_move_count < 10 {
            iterations = 25;
        } else if turn < 6 || legal_move_count < 20 {
            iterations = 50;
        } else if turn < 10 || legal_move_count < 30 {
            iterations = 80;
        }

        iterations
    }

    /// Build the concrete agent once and reuse it across calls so the agent
    /// can carry its MCTS tree between consecutive same-player searches.
    /// Returns `None` when there is exactly one legal move (no search needed).
    fn think(&mut self, game: &Game) -> (Option<&mut SearchAgent<'a>>, Vec<Box<dyn Move>>) {
        let moves = generate_legal_moves(&game.state);

        if moves.len() == 1 {
            return (None, moves);
        }

        if self.agent.is_none() {
            self.agent = Some(make_search_agent(
                self.backend,
                self.evaluator,
                self.max_iterations,
                self.leaf_batch,
                self.prior_heuristic_weight,
                self.policy_target_q_weight,
                self.tree_q_weight,
                self.reward_shape_w,
                self.pursuit_shape_w,
                self.goal_shape_w,
                self.unfreeze_opponent,
                self.macro_params,
            ));
        }
        if let Some(SearchAgent::Gumbel(a)) = self.agent.as_mut() {
            a.macro_goal = self.macro_goal.clone();
            a.star_gate = self.macro_star_gate;
            a.goal_aux = self.goal_aux.clone();
        }
        (self.agent.as_mut(), moves)
    }

    pub fn think_decomposed(
        &mut self,
        game: &Game,
        move_count: usize,
    ) -> (Option<Box<dyn Move>>, Vec<MoveVisit>) {
        // Read-and-clear before `self.think` below, which mutably borrows
        // `self.agent` for the rest of this call — `self` can't be touched
        // again (e.g. to clear this flag) while that borrow is live.
        let want_trace = self.pending_trace;
        self.pending_trace = false;

        let (agent, mut moves) = self.think(game);

        if agent.is_none() {
            if let Some(a) = &mut self.agent {
                a.clear_last_root_value();
            }
            return (moves.pop(), Vec::new());
        }

        let agent = agent.unwrap();
        if want_trace {
            agent.arm_trace();
        }
        agent.select_move_with_decomposed_visits(&mut game.clone(), move_count)
    }

    /// Request that the next `think_decomposed` call capture a decision
    /// trace (see decision_trace.rs). Consumed by that call whether or not
    /// it actually finds a trace worth taking.
    pub fn request_trace(&mut self) {
        self.pending_trace = true;
    }

    /// Stage 3: the macro agent's committed directive for the current turn
    /// (`None` for every other backend). Recorded features must carry the
    /// goal the agent actually pursued this ply.
    pub fn macro_committed_goal(&self) -> Option<crate::ai::oracle_macro::MacroGoal> {
        match &self.agent {
            Some(SearchAgent::MacroMcts(a)) => a.committed_goal().cloned(),
            _ => None,
        }
    }

    /// Stage 3b (macro policy head, first step): this turn's root ballot —
    /// candidate directives and the tree's own post-search visit count per
    /// candidate. `None` for every non-macro backend.
    pub fn macro_root_ballot(
        &self,
    ) -> Option<(Vec<crate::ai::oracle_macro::MacroGoal>, Vec<f32>)> {
        match &self.agent {
            Some(SearchAgent::MacroMcts(a)) => {
                let (c, v) = a.last_root_ballot();
                Some((c.to_vec(), v.to_vec()))
            }
            _ => None,
        }
    }

    /// Retrieve the trace captured by the most recent `think_decomposed`
    /// call, if `request_trace` was called beforehand and search actually
    /// reached a final selection.
    pub fn take_trace(&mut self) -> Option<crate::ai::decision_trace::DecisionTrace> {
        self.agent.as_mut().and_then(SearchAgent::take_trace)
    }

    /// The most recent `think_decomposed` call's search-backed root value
    /// (a TD bootstrap target under the reward-aware Gumbel backup), if that
    /// call actually ran a search. `None` for non-Gumbel backends or when no
    /// agent has searched yet.
    pub fn last_root_value(&self) -> Option<f32> {
        self.agent.as_ref().and_then(SearchAgent::last_root_value)
    }

    /// Raw NN root value of the most recent search (value-head calibration).
    pub fn last_root_own_value(&self) -> Option<f32> {
        self.agent.as_ref().and_then(SearchAgent::last_root_own_value)
    }

    /// Cumulative search telemetry:
    /// (depth_sum, depth_count, depth_max, horizon_hits, agree, decisions).
    pub fn depth_stats(&self) -> Option<(u64, u64, u32, u64, u64, u64)> {
        self.agent.as_ref().and_then(SearchAgent::depth_stats)
    }

    pub fn think_with_stats(&mut self, game: &Game) -> (Option<Box<dyn Move>>, Vec<f32>) {
        let (agent, mut moves) = self.think(game);

        if agent.is_none() {
            return (moves.pop(), Vec::new());
        }

        let mut mcts_game = game.clone_for_mcts(game.current_player_id());
        agent.unwrap().select_move_with_stats(&mut mcts_game)
    }
}

/// Returns the maximum number of game turns the MCTS tree should look ahead
/// from the current turn. This prevents the search from going absurdly deep
/// and getting stuck in long rollouts during mid-game when branching is high.
pub fn max_turns_ahead(current_turn: i32, max_turns: i32) -> i32 {
    if current_turn < 8 {
        5
    } else {
        // Turns actually remaining, not a hardcoded 20 — games run to
        // max_turns=30, so the old literal pinned the horizon to its 2-turn
        // floor from turn 18 on. Still capped at 20 to bound rollout length.
        (max_turns - current_turn).max(2).min(20)
    }
}
