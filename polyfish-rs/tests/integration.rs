//! Integration tests for loading game states

use polyfish::Game;
use std::path::Path;

#[test]
fn test_load_game_state_from_json() {
    let json_path = Path::new("../data/gamestate.json");

    if !json_path.exists() {
        println!("Skipping test: gamestate.json not found at {:?}", json_path);
        return;
    }

    let result = Game::from_file(json_path);

    match result {
        Ok(game) => {
            println!("Successfully loaded game state!");
            println!("  Map size: {}", game.map_size());
            println!("  Turn: {}", game.turn());
            println!("  Tribes: {}", game.state.tribes.len());
            println!("  Tiles: {}", game.state.tiles.len());

            // Verify basic properties
            assert!(game.map_size() > 0, "Map size should be positive");
            assert!(game.turn() >= 1, "Turn should be at least 1");
            assert!(
                !game.state.tribes.is_empty(),
                "Should have at least one tribe"
            );

            // Count units
            let pov_id = game.current_player_id();
            if let Some(tribe) = game.state.tribes.get(&pov_id) {
                println!(
                    "  Current player {} has {} units",
                    pov_id,
                    tribe.units.len()
                );
                for unit in &tribe.units {
                    println!(
                        "    - Unit {:?} at {} (moved={}, attacked={})",
                        unit.unit_type, unit.coords.idx, unit.moved, unit.attacked
                    );
                }
            }

            // Try generating legal moves
            let moves = game.legal_moves();
            println!("  Legal moves: {}", moves.len());

            // Count by type
            let mut step_count = 0;
            let mut attack_count = 0;
            let mut capture_count = 0;
            let mut harvest_count = 0;
            let mut research_count = 0;
            let mut build_count = 0;
            let mut summon_count = 0;
            let mut ability_count = 0;
            let mut reward_count = 0;
            let mut end_turn_count = 0;
            for m in &moves {
                match m.move_type() {
                    polyfish::MoveType::Step => step_count += 1,
                    polyfish::MoveType::Attack => attack_count += 1,
                    polyfish::MoveType::Capture => capture_count += 1,
                    polyfish::MoveType::Harvest => harvest_count += 1,
                    polyfish::MoveType::Research => research_count += 1,
                    polyfish::MoveType::Build => build_count += 1,
                    polyfish::MoveType::Summon => summon_count += 1,
                    polyfish::MoveType::Ability => ability_count += 1,
                    polyfish::MoveType::Reward => reward_count += 1,
                    polyfish::MoveType::EndTurn => end_turn_count += 1,
                    _ => {}
                }
            }
            println!("    Step moves: {}", step_count);
            println!("    Attack moves: {}", attack_count);
            println!("    Capture moves: {}", capture_count);
            println!("    Harvest moves: {}", harvest_count);
            println!("    Research moves: {}", research_count);
            println!("    Build moves: {}", build_count);
            println!("    Summon/Upgrade moves: {}", summon_count);
            println!("    Ability moves: {}", ability_count);
            println!("    Reward moves: {}", reward_count);
            println!("    EndTurn: {}", end_turn_count);

            assert!(
                !moves.is_empty(),
                "Should have at least one legal move (end turn)"
            );
        }
        Err(e) => {
            panic!("Failed to load game state: {}", e);
        }
    }
}

#[test]
fn test_game_serialization_roundtrip() {
    let game = Game::new();

    // Serialize to JSON
    let json = game.to_json().expect("Should serialize to JSON");

    // Deserialize back
    let game2 = Game::from_json(&json).expect("Should deserialize from JSON");

    // Verify they match
    assert_eq!(game.state.settings.size, game2.state.settings.size);
    assert_eq!(game.state.settings.turn, game2.state.settings.turn);
    assert_eq!(game.state.settings.mode, game2.state.settings.mode);
}
