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

#[test]
fn test_gumbel_mcts_basic() {
    let network = make_network();
    let evaluator = make_evaluator(&network);
    let mut game = make_game(42);

    let mut agent = GumbelMctsAgent::new(&evaluator, 40, 4);

    let best_move = agent.select_move(&mut game);
    assert!(best_move.is_some(), "Gumbel MCTS failed to select a move");
    println!("Selected move: {}", best_move.unwrap().describe(&game.state));
}

#[test]
fn test_gumbel_mcts_sequential_halving() {
    let network = make_network();
    let evaluator = make_evaluator(&network);
    let mut game = make_game(123);

    // k=8 -> at least 3 halving rounds (log2(8)=3).
    let mut agent = GumbelMctsAgent::new(&evaluator, 64, 8);

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
    let evaluator = make_evaluator(&network);
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

    let mut agent = GumbelMctsAgent::new(&evaluator, 32, 4);
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
    let evaluator = make_evaluator(&network);
    let mut game = make_game(2024);

    // Advance past the opening book so the search path is exercised.
    for _ in 0..6 {
        if polyfish::functions::is_game_over(&game.state) {
            break;
        }
        let _ = game.play_move(&EndTurnMove);
    }

    let mut agent = GumbelMctsAgent::new(&evaluator, 16, 4);

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

/// Tree reuse (structure-only root-shift): a second consecutive same-player
/// search must re-root into the previous tree rather than rebuild from
/// scratch. We advance past the opening book, run one search, apply the
/// chosen move, then run a second search from the same player's seat and
/// assert `tree_reuses` incremented. The returned policy must still be a
/// valid distribution (sums to ~1) — the structure-only reset preserves the
/// π' target's semantics.
#[test]
fn test_gumbel_tree_reuse_on_consecutive_same_player_search() {
    use polyfish::moves::EndTurnMove;

    let network = make_network();
    let evaluator = make_evaluator(&network);
    let mut game = make_game(7);

    // Advance past the opening book so the search path is exercised.
    for _ in 0..6 {
        if polyfish::functions::is_game_over(&game.state) {
            break;
        }
        let _ = game.play_move(&EndTurnMove);
    }

    let mut agent = GumbelMctsAgent::new(&evaluator, 32, 8);

    // First search from the current player's seat.
    let pov = game.state.settings.current_player_turn_id;
    let (m1, mv1) = agent.select_move_with_decomposed_visits(&mut game, 0);
    assert!(m1.is_some(), "first search must return a move");
    assert!(!mv1.is_empty());
    assert_eq!(agent.tree_reuses, 0, "first search builds fresh");

    // Apply the chosen move (still the same player's turn — economy moves
    // don't end the turn). This is the root-shift case.
    let m1 = m1.unwrap();
    let ended_turn = m1.move_type() == polyfish::types::MoveType::EndTurn;
    let _ = game.play_move(m1.as_ref());

    if ended_turn || game.state.settings.current_player_turn_id != pov {
        // The chosen move ended the turn (opponent to move next). Tree reuse
        // is scoped to within one player's own turn, so this scenario can't
        // exercise reuse — skip the reuse assertion rather than flake.
        return;
    }

    // Second search, same player, one move later: must re-root into the
    // cached subtree.
    let (_m2, mv2) = agent.select_move_with_decomposed_visits(&mut game, 1);
    assert!(
        agent.tree_reuses >= 1,
        "second consecutive same-player search must reuse the tree (got {})",
        agent.tree_reuses
    );
    let sum: f32 = mv2.iter().map(|mv| mv.visits).sum();
    assert!(
        (sum - 1.0).abs() < 1e-4,
        "reused-root policy target must still sum to 1.0, got {}",
        sum
    );
}

/// Tree reuse must NOT fire when the opponent moved between two calls of the
/// same agent — the cached tree's chosen child no longer matches the new
/// root, so the agent must fall back to a fresh build.
#[test]
fn test_gumbel_tree_reuse_skipped_after_opponent_move() {
    use polyfish::moves::EndTurnMove;

    let network = make_network();
    let evaluator = make_evaluator(&network);
    let mut game = make_game(99);

    for _ in 0..6 {
        if polyfish::functions::is_game_over(&game.state) {
            break;
        }
        let _ = game.play_move(&EndTurnMove);
    }

    let mut agent = GumbelMctsAgent::new(&evaluator, 32, 8);

    // First search for the current player.
    let (m1, _) = agent.select_move_with_decomposed_visits(&mut game, 0);
    assert!(m1.is_some());

    // Force a full turn cycle: end the current player's turn, then end the
    // opponent's turn, so the same player is to move again but the position
    // has changed by both players' EndTurns.
    let _ = game.play_move(&EndTurnMove);
    let _ = game.play_move(&EndTurnMove);

    // The cached tree's next_root_hash was for the state after the chosen
    // move, not after two EndTurns, so the hash must mismatch → fresh build.
    let before = agent.tree_reuses;
    let (_m2, _) = agent.select_move_with_decomposed_visits(&mut game, 1);
    assert_eq!(
        agent.tree_reuses, before,
        "reuse must not fire after an opponent move changed the root"
    );
}
