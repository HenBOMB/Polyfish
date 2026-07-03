use crate::ai::gumbel_mcts::GumbelMctsAgent;
use crate::ai::mcts_types::MoveVisit;
use crate::ai::mcts_zero::ZeroMctsAgent;
use crate::ai::network::PolyZeroNet;
use crate::game::Game;
use crate::moves::{Move, generate_legal_moves};

/// Which search backend `Brain` should use to select moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchBackend {
    Zero,
    Gumbel { k: usize },
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
}

impl From<SearchBackendArg> for SearchBackend {
    fn from(arg: SearchBackendArg) -> Self {
        match arg {
            SearchBackendArg::Zero => SearchBackend::Zero,
            SearchBackendArg::Gumbel => SearchBackend::Gumbel { k: 16 },
        }
    }
}

// class brain
pub struct Brain<'a> {
    pub network: &'a PolyZeroNet,
    pub max_iterations: usize,
    pub backend: SearchBackend,
}

/// Internal enum wrapping whichever concrete agent the configured backend
/// produced. Matched once per `think_decomposed` / `think_with_stats` call.
///
/// Exposed publicly so `arena.rs` can dispatch over backends without
/// duplicating the enum.
pub enum SearchAgent<'a> {
    Zero(ZeroMctsAgent<'a>),
    Gumbel(GumbelMctsAgent<'a>),
}

impl<'a> SearchAgent<'a> {
    pub fn select_move(&self, game: &mut Game) -> Option<Box<dyn Move>> {
        match self {
            SearchAgent::Zero(a) => a.select_move(game),
            SearchAgent::Gumbel(a) => a.select_move(game),
        }
    }

    fn select_move_with_decomposed_visits(
        &self,
        game: &mut Game,
        move_count: usize,
    ) -> (Option<Box<dyn Move>>, Vec<MoveVisit>) {
        match self {
            SearchAgent::Zero(a) => a.select_move_with_decomposed_visits(game, move_count),
            SearchAgent::Gumbel(a) => a.select_move_with_decomposed_visits(game, move_count),
        }
    }

    fn select_move_with_stats(&self, game: &mut Game) -> (Option<Box<dyn Move>>, Vec<f32>) {
        match self {
            SearchAgent::Zero(a) => a.select_move_with_stats(game),
            SearchAgent::Gumbel(a) => a.select_move_with_stats(game),
        }
    }
}

/// Construct the concrete search agent for a backend, borrowing `network`.
pub fn make_search_agent(
    backend: SearchBackend,
    network: &PolyZeroNet,
    iterations: usize,
) -> SearchAgent<'_> {
    match backend {
        SearchBackend::Zero => SearchAgent::Zero(ZeroMctsAgent::new(network, iterations)),
        SearchBackend::Gumbel { k } => {
            SearchAgent::Gumbel(GumbelMctsAgent::new(network, iterations, k))
        }
    }
}

impl<'a> Brain<'a> {
    pub fn new(network: &'a PolyZeroNet, max_iterations: usize) -> Self {
        Self {
            network,
            max_iterations,
            backend: SearchBackend::default(),
        }
    }

    pub fn with_backend(
        network: &'a PolyZeroNet,
        max_iterations: usize,
        backend: SearchBackend,
    ) -> Self {
        Self {
            network,
            max_iterations,
            backend,
        }
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

    fn think(&'_ self, game: &Game) -> (Option<SearchAgent<'_>>, Vec<Box<dyn Move>>) {
        let moves = generate_legal_moves(&game.state);

        if moves.len() == 1 {
            return (None, moves);
        }

        let agent = make_search_agent(self.backend, self.network, self.max_iterations);
        (Some(agent), moves)
    }

    pub fn think_decomposed(
        &self,
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

    pub fn think_with_stats(&self, game: &Game) -> (Option<Box<dyn Move>>, Vec<f32>) {
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
