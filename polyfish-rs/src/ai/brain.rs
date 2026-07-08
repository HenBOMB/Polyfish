use crate::ai::eval_server::Evaluator;
use crate::ai::gumbel_mcts::GumbelMctsAgent;
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
}

impl From<SearchBackendArg> for SearchBackend {
    fn from(arg: SearchBackendArg) -> Self {
        match arg {
            SearchBackendArg::Zero => SearchBackend::Zero,
            SearchBackendArg::Gumbel => SearchBackend::Gumbel { k: 16 },
            SearchBackendArg::Heuristic => SearchBackend::Heuristic,
            SearchBackendArg::Greedy => SearchBackend::Greedy,
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
    /// Set by `request_trace`; consumed (and cleared) by the next
    /// `think_decomposed` call, which arms the underlying agent's tracer.
    pending_trace: bool,
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
}

impl<'a> SearchAgent<'a> {
    pub fn select_move(&mut self, game: &mut Game) -> Option<Box<dyn Move>> {
        match self {
            SearchAgent::Zero(a) => a.select_move(game),
            SearchAgent::Gumbel(a) => a.select_move(game),
            SearchAgent::Heuristic(a) => a.select_move(game),
            SearchAgent::Greedy(a) => a.select_move(game),
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
        }
    }

    fn select_move_with_stats(&mut self, game: &mut Game) -> (Option<Box<dyn Move>>, Vec<f32>) {
        match self {
            SearchAgent::Zero(a) => a.select_move_with_stats(game),
            SearchAgent::Gumbel(a) => a.select_move_with_stats(game),
            // No NN priors to report stats over; the move is all callers need.
            SearchAgent::Heuristic(a) => (a.select_move(game), Vec::new()),
            SearchAgent::Greedy(a) => (a.select_move(game), Vec::new()),
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
            SearchAgent::Gumbel(agent)
        }
        SearchBackend::Heuristic => SearchAgent::Heuristic(
            crate::ai::heuristic_mcts::HeuristicMctsAgent::new(iterations),
        ),
        SearchBackend::Greedy => {
            SearchAgent::Greedy(crate::ai::heuristic_mcts::GreedyHeuristicAgent)
        }
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
            pending_trace: false,
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
            pending_trace: false,
        }
    }

    /// Override the per-game virtual-loss mini-batch size (see `--leaf-batch`
    /// in self_play). Builder-style: chain after `with_backend`.
    pub fn with_leaf_batch(mut self, leaf_batch: usize) -> Self {
        self.leaf_batch = Some(leaf_batch);
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
            ));
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

    /// Retrieve the trace captured by the most recent `think_decomposed`
    /// call, if `request_trace` was called beforehand and search actually
    /// reached a final selection.
    pub fn take_trace(&mut self) -> Option<crate::ai::decision_trace::DecisionTrace> {
        self.agent.as_mut().and_then(SearchAgent::take_trace)
    }

    pub fn think_with_stats(&mut self, game: &Game) -> (Option<Box<dyn Move>>, Vec<f32>) {
        let (agent, mut moves) = self.think(game);

        if agent.is_none() {
            return (moves.pop(), Vec::new());
        }

        agent.unwrap().select_move_with_stats(&mut game.clone())
    }
}

/// Returns the maximum number of game turns the MCTS tree should look ahead
/// from the current turn. This prevents the search from going absurdly deep
/// and getting stuck in long rollouts during mid-game when branching is high.
pub fn max_turns_ahead(_current_turn: i32, _max_turns: i32) -> i32 {
    if _current_turn < 8 {
        5
    } else {
        (20 - _current_turn).max(2).min(20) // idea: do not look ahead more than the last turn (20 turns, default)
    }
}
