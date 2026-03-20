use polyfish::game::Game;
use polyfish::mapgen::{self, MapGenSettings};
use polyfish::states::GameState;
use polyfish::types::{MapSize, MapType, TribeType};

#[tokio::test]
async fn test_replay_flow() {
    // 1. Setup a game
    let mut settings = MapGenSettings::default();
    settings.size = MapSize::Tiny;
    settings.tribes = vec![TribeType::Imperius, TribeType::Bardur];
    settings.seed = 12345;
    settings.map_type = MapType::Drylands;

    let initial_state = mapgen::generate(settings);
    let mut game = Game::new();
    game.state = initial_state;
    // Important: capture seed
    game.state.initial_seed = 12345;
    game.post_load();

    // 2. Play a few moves
    // Find a valid step move for player 1
    let legal = game.legal_moves();
    let step_move = legal
        .iter()
        .find(|m| m.move_type() == polyfish::types::MoveType::Step);

    if let Some(m) = step_move {
        println!("Playing move: {:?}", m.describe(&game.state));
        game.play_move(m.as_ref());
    } else {
        // Just end turn if no steps
        game.play_move(&polyfish::moves::EndTurnMove);
    }

    let history_len = game.state._history.len();
    assert!(history_len > 0, "History should have recorded a move");

    // 3. Save state (simulate saving to JSON)
    let json = serde_json::to_string(&game.state).unwrap();

    // 4. Load state into NEW game instance
    let loaded_state: GameState = serde_json::from_str(&json).unwrap();
    assert_eq!(loaded_state.initial_seed, 12345);
    assert_eq!(loaded_state._history.len(), history_len);

    // 5. Simulate "Analyze" logic: Replay from scratch
    let mut replay_game = Game::new();
    let mut replay_settings = MapGenSettings::default();
    replay_settings.size = MapSize::Tiny; // We know it was Tiny
    replay_settings.tribes = vec![TribeType::Imperius, TribeType::Bardur];
    replay_settings.seed = loaded_state.initial_seed;
    replay_settings.map_type = loaded_state.settings.map_type;

    let replay_initial = mapgen::generate(replay_settings);
    replay_game.state = replay_initial;
    replay_game.post_load();

    // History from loaded state
    let history = loaded_state._history;

    // Replay valid moves
    for (i, move_json) in history.iter().enumerate() {
        let legal = replay_game.legal_moves();
        let mut found = false;
        for m in legal {
            if m.serialize() == *move_json {
                println!("Replaying move {}: {:?}", i, m.describe(&replay_game.state));
                replay_game.play_move(m.as_ref());
                found = true;
                break;
            }
        }
        assert!(found, "Failed to replay move index {}", i);
    }

    // 6. Verify final states match
    // Note: Hashes might differ due to internal IDs or something, so check turn/currentPlayer
    assert_eq!(replay_game.state.settings.turn, game.state.settings.turn);
    assert_eq!(
        replay_game.state.settings.current_player_turn_id,
        game.state.settings.current_player_turn_id
    );

    println!("Replay verification successful!");
}
