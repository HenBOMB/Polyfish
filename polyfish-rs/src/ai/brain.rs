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
}

impl From<SearchBackendArg> for SearchBackend {
    fn from(arg: SearchBackendArg) -> Self {
        match arg {
            SearchBackendArg::Zero => SearchBackend::Zero,
            SearchBackendArg::Gumbel => SearchBackend::Gumbel { k: 16 },
            SearchBackendArg::Heuristic => SearchBackend::Heuristic,
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
}

impl<'a> SearchAgent<'a> {
    pub fn select_move(&mut self, game: &mut Game) -> Option<Box<dyn Move>> {
        match self {
            SearchAgent::Zero(a) => a.select_move(game),
            SearchAgent::Gumbel(a) => a.select_move(game),
            SearchAgent::Heuristic(a) => a.select_move(game),
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
        }
    }

    fn select_move_with_stats(&mut self, game: &mut Game) -> (Option<Box<dyn Move>>, Vec<f32>) {
        match self {
            SearchAgent::Zero(a) => a.select_move_with_stats(game),
            SearchAgent::Gumbel(a) => a.select_move_with_stats(game),
            // No NN priors to report stats over; the move is all callers need.
            SearchAgent::Heuristic(a) => (a.select_move(game), Vec::new()),
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
            SearchAgent::Gumbel(agent)
        }
        SearchBackend::Heuristic => SearchAgent::Heuristic(
            crate::ai::heuristic_mcts::HeuristicMctsAgent::new(iterations),
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
            ));
        }
        (self.agent.as_mut(), moves)
    }

    pub fn think_decomposed(
        &mut self,
        game: &Game,
        move_count: usize,
    ) -> (Option<Box<dyn Move>>, Vec<MoveVisit>) {
        let (agent, mut moves) = self.think(game);

        if agent.is_none() {
            return (moves.pop(), Vec::new());
        }

        agent
            .unwrap()
            .select_move_with_decomposed_visits(&mut game.clone(), move_count)
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
