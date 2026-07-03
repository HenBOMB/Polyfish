use candle_core::Device;
use candle_nn::VarMap;
use polyfish::ai::gumbel_mcts::GumbelMctsAgent;
use polyfish::ai::network::PolyZeroNet;
use polyfish::game::Game;
use polyfish::mapgen::{generate, MapGenSettings};
use polyfish::types::{MapSize, MapType, TribeType};

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

fn make_network() -> PolyZeroNet {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vs = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
    PolyZeroNet::new(vs).unwrap()
}

#[test]
fn test_gumbel_mcts_basic() {
    let network = make_network();
    let mut game = make_game(42);

    let agent = GumbelMctsAgent::new(&network, 40, 4);

    let best_move = agent.select_move(&mut game);
    assert!(best_move.is_some(), "Gumbel MCTS failed to select a move");
    println!("Selected move: {}", best_move.unwrap().describe(&game.state));
}

#[test]
fn test_gumbel_mcts_sequential_halving() {
    let network = make_network();
    let mut game = make_game(123);

    // k=8 -> at least 3 halving rounds (log2(8)=3).
    let agent = GumbelMctsAgent::new(&network, 64, 8);

    let best_move = agent.select_move(&mut game);
    assert!(best_move.is_some());
}

/// Regression test for bug #4: the old agent truncated root.children after
/// each halving round, collapsing the policy target to ~1-2 actions. The
/// rewrite must emit one MoveVisit per legal move at the root.
#[test]
fn test_gumbel_policy_target_covers_full_legal_set() {
    use polyfish::moves::EndTurnMove;

    let network = make_network();
    let mut game = make_game(7);

    // Fast-forward past the opening book (it covers turns 0-1 only), so the
    // search — not the book — produces the policy target.
    for _ in 0..6 {
        if polyfish::functions::is_game_over(&game.state) {
            break;
        }
        let _ = game.play_move(&EndTurnMove);
    }
    assert!(
        polyfish::ai::book::Book::recommend(&game).is_empty(),
        "test setup: expected to be past the opening book"
    );

    let legal_moves = game.legal_moves();
    let legal_count = legal_moves.len();
    assert!(
        legal_count > 1,
        "test setup: expected >1 legal move, got {}",
        legal_count
    );

    // The Gumbel root suppresses EndTurn when other moves exist (mirroring
    // Zero), so the policy target covers the non-EndTurn legal set.
    let has_other = legal_moves
        .iter()
        .any(|m| m.move_type() != polyfish::types::MoveType::EndTurn);
    let expected_count = if has_other {
        legal_moves
            .iter()
            .filter(|m| m.move_type() != polyfish::types::MoveType::EndTurn)
            .count()
    } else {
        legal_count
    };
    assert!(
        expected_count > 1,
        "test setup: expected >1 non-EndTurn legal move, got {}",
        expected_count
    );

    let agent = GumbelMctsAgent::new(&network, 32, 4);
    let (_best_move, move_visits) = agent.select_move_with_decomposed_visits(&mut game, 0);

    assert_eq!(
        move_visits.len(),
        expected_count,
        "policy target must cover the full (EndTurn-filtered) root legal set \
        (bug #4 regression)"
    );

    // π' is a softmax over all root children, so it must sum to ~1.
    let sum: f32 = move_visits.iter().map(|mv| mv.visits).sum();
    assert!(
        (sum - 1.0).abs() < 1e-4,
        "policy target probabilities should sum to 1.0, got {}",
        sum
    );
}

/// End-to-end smoke test: several consecutive search-and-play steps through a
/// real game loop, exercising the round-robin batching path against the real
/// Game/Move/network stack. Catches integration bugs the pure unit tests
/// can't.
#[test]
fn test_gumbel_multi_step_game_loop_no_panic() {
    use polyfish::moves::EndTurnMove;

    let network = make_network();
    let mut game = make_game(2024);

    // Advance past the opening book so the search path is exercised.
    for _ in 0..6 {
        if polyfish::functions::is_game_over(&game.state) {
            break;
        }
        let _ = game.play_move(&EndTurnMove);
    }

    let agent = GumbelMctsAgent::new(&network, 16, 4);

    for step in 0..4 {
        if polyfish::functions::is_game_over(&game.state) {
            break;
        }
        let (best_move, move_visits) = agent.select_move_with_decomposed_visits(&mut game, step);
        assert!(
            best_move.is_some(),
            "no move returned at step {} (turn {})",
            step,
            game.state.settings.turn
        );
        assert!(
            !move_visits.is_empty(),
            "empty policy target at step {} (turn {})",
            step,
            game.state.settings.turn
        );
        let m = best_move.unwrap();
        let _ = game.play_move(m.as_ref());
    }
}
