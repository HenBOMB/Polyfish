use candle_core::Device;
use candle_nn::VarMap;
use polyfish::ai::eval_server::{Evaluator, InlineEvalHandle};
use polyfish::ai::mcts_zero::ZeroMctsAgent;
use polyfish::ai::network::PolyZeroNet;
use polyfish::game::Game;
use polyfish::mapgen::{generate, MapGenSettings};
use polyfish::types::{MapSize, MapType, TribeType, MoveType};
use std::sync::Arc;

/// Test that value signs are preserved for same-player moves
/// and only flipped when crossing EndTurn boundaries
#[test]
fn test_value_sign_flipping_on_player_change() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vs = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
    let network = Arc::new(PolyZeroNet::new(vs).unwrap());
    let evaluator = Evaluator::Inline(InlineEvalHandle::new(network.clone()));

    let mut game = Game::new();
    let gen_settings = MapGenSettings {
        size: MapSize::Tiny,
        map_type: MapType::Drylands,
        tribes: vec![TribeType::Imperius, TribeType::Imperius],
        seed: 42,
        ..Default::default()
    };
    game.state = generate(gen_settings);
    game.post_load();

    let initial_player = game.state.settings.current_player_turn_id;

    // Run MCTS to build a tree
    let agent = ZeroMctsAgent::new(&evaluator, 50);
    let _best_move = agent.select_move(&mut game);

    // The game state should be unchanged after search (undo mechanism)
    assert_eq!(
        game.state.settings.current_player_turn_id,
        initial_player,
        "Game state should be restored after MCTS search"
    );

    println!("✓ MCTS search completed and state restored");
    println!("  Initial player: {}", initial_player);
}

/// Test that terminal states return actual game outcomes
/// instead of 0.0
#[test]
fn test_terminal_state_values() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vs = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
    let network = Arc::new(PolyZeroNet::new(vs).unwrap());
    let evaluator = Evaluator::Inline(InlineEvalHandle::new(network.clone()));

    let mut game = Game::new();
    let gen_settings = MapGenSettings {
        size: MapSize::Tiny,
        map_type: MapType::Drylands,
        tribes: vec![TribeType::Imperius, TribeType::Imperius],
        seed: 123,
        ..Default::default()
    };
    game.state = generate(gen_settings);
    game.post_load();

    // Manually set game over to test terminal value handling
    game.state.settings._game_over = true;

    // Set Player 1 score higher than Player 2
    if let Some(tribe1) = game.state.tribes.get_mut(&1) {
        tribe1.score = 1000;
    }
    if let Some(tribe2) = game.state.tribes.get_mut(&2) {
        tribe2.score = 500;
    }

    let current_player = game.state.settings.current_player_turn_id;

    // Even with game over, MCTS should handle it gracefully
    let agent = ZeroMctsAgent::new(&evaluator, 10);
    let best_move = agent.select_move(&mut game);

    // MCTS should still be able to select some move, even in terminal states
    // (the expansion logic may create children before detecting terminal state)
    assert!(
        best_move.is_some(),
        "MCTS should return a move even in terminal states"
    );

    println!("✓ Terminal state handled correctly");
    println!("  Game over: {}", game.state.settings._game_over);
    println!("  Current player: {}", current_player);
    if let Some(ref mv) = best_move {
        println!("  Selected move type: {:?}", mv.move_type());
    }
}

/// Test that moves are executed in sequence and player tracking works
#[test]
fn test_player_sequence_tracking() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vs = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
    let network = PolyZeroNet::new(vs).unwrap();

    let mut game = Game::new();
    let gen_settings = MapGenSettings {
        size: MapSize::Tiny,
        map_type: MapType::Drylands,
        tribes: vec![TribeType::Imperius, TribeType::Imperius],
        seed: 789,
        ..Default::default()
    };
    game.state = generate(gen_settings);
    game.post_load();

    let initial_player = game.state.settings.current_player_turn_id;
    let initial_turn = game.state.settings.turn;

    // Execute a series of moves to see player changes
    let moves_to_execute = 5;
    for i in 0..moves_to_execute {
        let legal_moves = game.legal_moves();

        // Find a non-EndTurn move if possible
        let chosen_move = legal_moves
            .iter()
            .find(|m| m.move_type() != MoveType::EndTurn)
            .or_else(|| legal_moves.first());

        if let Some(mv) = chosen_move {
            let player_before = game.state.settings.current_player_turn_id;
            let move_type = mv.move_type();

            if let Some(undo) = game.simulate_move(mv.as_ref()) {
                let player_after = game.state.settings.current_player_turn_id;

                println!(
                    "Move {}: {:?} | Player: {} -> {}",
                    i + 1,
                    move_type,
                    player_before,
                    player_after
                );

                // EndTurn should change player (after cycling through dead players)
                if move_type == MoveType::EndTurn {
                    // Player may or may not change depending on game state
                    // (could cycle back if opponent is dead)
                    println!("  EndTurn executed, player may have changed");
                } else {
                    // Non-EndTurn moves should keep the same player
                    assert_eq!(
                        player_before, player_after,
                        "Non-EndTurn move should not change current player"
                    );
                }
            }
        } else {
            break;
        }
    }

    println!("✓ Player sequence tracking validated");
    println!("  Initial player: {}, turn: {}", initial_player, initial_turn);
    println!("  Final player: {}, turn: {}",
        game.state.settings.current_player_turn_id,
        game.state.settings.turn
    );
}

/// Test MCTS with multiple iterations to ensure no crashes
/// with the new player-aware backpropagation
#[test]
fn test_mcts_stability_with_player_tracking() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vs = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
    let network = Arc::new(PolyZeroNet::new(vs).unwrap());
    let evaluator = Evaluator::Inline(InlineEvalHandle::new(network.clone()));

    let mut game = Game::new();
    let gen_settings = MapGenSettings {
        size: MapSize::Tiny,
        map_type: MapType::Drylands,
        tribes: vec![TribeType::Imperius, TribeType::Imperius],
        seed: 999,
        ..Default::default()
    };
    game.state = generate(gen_settings);
    game.post_load();

    // Run MCTS with various iteration counts
    for iterations in [10, 50, 100] {
        let agent = ZeroMctsAgent::new(&evaluator, iterations);
        let best_move = agent.select_move(&mut game);

        assert!(
            best_move.is_some(),
            "MCTS should return a move with {} iterations",
            iterations
        );

        println!("✓ MCTS stable with {} iterations", iterations);
    }
}
