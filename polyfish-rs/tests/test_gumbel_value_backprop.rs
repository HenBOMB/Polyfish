use candle_core::Device;
use candle_nn::VarMap;
use polyfish::ai::eval_server::{Evaluator, InlineEvalHandle};
use polyfish::ai::gumbel_mcts::GumbelMctsAgent;
use polyfish::ai::network::PolyZeroNet;
use polyfish::game::Game;
use polyfish::mapgen::{generate, MapGenSettings};
use polyfish::types::{MapSize, MapType, TribeType};
use std::sync::Arc;

fn make_game(seed: i64) -> Game {
    let mut game = Game::new();
    let gen_settings = MapGenSettings {
        size: MapSize::Tiny,
        map_type: MapType::Drylands,
        tribes: vec![TribeType::Imperius, TribeType::Imperius],
        seed,
        ..Default::default()
    };
    game.state = generate(gen_settings);
    game.post_load();
    game
}

fn make_network() -> Arc<PolyZeroNet> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vs = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
    Arc::new(PolyZeroNet::new(vs).unwrap())
}

fn make_evaluator(network: &Arc<PolyZeroNet>) -> Evaluator {
    Evaluator::Inline(InlineEvalHandle::new(network.clone()))
}

/// The Gumbel agent must fully undo every simulated move during search, so
/// the game state (current player, turn) is restored after a full
/// `select_move_with_decomposed_visits` call. Mirrors the Zero-side test in
/// `test_value_backprop.rs`, but exercised through Gumbel's round-robin leaf
/// collection path.
#[test]
fn test_gumbel_state_restored_after_search() {
    use polyfish::moves::EndTurnMove;

    let network = make_network();
    let evaluator = make_evaluator(&network);
    let mut game = make_game(42);

    // Advance past the opening book so the search (which simulates and undoes
    // moves) is the code path under test.
    for _ in 0..6 {
        if polyfish::functions::is_game_over(&game.state) {
            break;
        }
        let _ = game.play_move(&EndTurnMove);
    }
    assert!(polyfish::ai::book::Book::recommend(&game).is_empty());

    let initial_player = game.state.settings.current_player_turn_id;
    let initial_turn = game.state.settings.turn;

    let agent = GumbelMctsAgent::new(&evaluator, 32, 8);
    let (best_move, _visits) = agent.select_move_with_decomposed_visits(&mut game, 0);
    assert!(best_move.is_some());

    assert_eq!(
        game.state.settings.current_player_turn_id,
        initial_player,
        "current player must be restored after Gumbel search"
    );
    assert_eq!(
        game.state.settings.turn,
        initial_turn,
        "turn must be restored after Gumbel search"
    );
}

/// Gumbel search must not panic across a range of iteration budgets,
/// including very small budgets (iterations < k) that exercise the
/// degenerate round-robin allocation paths. Advances past the opening book so
/// the actual search (not the book) is exercised for every config.
#[test]
fn test_gumbel_stability_across_iteration_counts() {
    use polyfish::moves::EndTurnMove;

    let network = make_network();
    let evaluator = make_evaluator(&network);
    let mut base = make_game(999);
    for _ in 0..6 {
        if polyfish::functions::is_game_over(&base.state) {
            break;
        }
        let _ = base.play_move(&EndTurnMove);
    }
    assert!(
        polyfish::ai::book::Book::recommend(&base).is_empty(),
        "test setup: expected to be past the opening book"
    );

    for iterations in [1usize, 4, 16] {
        for k in [2usize, 4, 8] {
            let mut game = base.clone();
            let agent = GumbelMctsAgent::new(&evaluator, iterations, k);
            let best_move = agent.select_move(&mut game);
            assert!(
                best_move.is_some(),
                "Gumbel search returned no move for iterations={} k={}",
                iterations,
                k
            );
        }
    }
}

/// Terminal states must be handled gracefully (no panic, still returns a
/// move), mirroring the Zero-side terminal test.
#[test]
fn test_gumbel_terminal_state_handled() {
    let network = make_network();
    let evaluator = make_evaluator(&network);
    let mut game = make_game(123);

    game.state.settings._game_over = true;
    if let Some(tribe1) = game.state.tribes.get_mut(&1) {
        tribe1.score = 1000;
    }
    if let Some(tribe2) = game.state.tribes.get_mut(&2) {
        tribe2.score = 500;
    }

    let agent = GumbelMctsAgent::new(&evaluator, 16, 4);
    // Even with the game flagged over, select_move should not panic and
    // should return an EndTurn fallback (the search root sees no live
    // search to do).
    let best_move = agent.select_move(&mut game);
    assert!(best_move.is_some(), "Gumbel must return a move in terminal state");
}
