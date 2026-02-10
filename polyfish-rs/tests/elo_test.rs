use polyfish::ai::heuristic_mcts::HeuristicMctsAgent;
use polyfish::game::Game;
use polyfish::types::MoveType;
use std::path::Path;

#[test]
fn test_mcts_matches_ruin_capture_sequence() {
    let state_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("saved_state.json");
    let initial_game = Game::from_file(state_path).expect("Failed to load saved_state.json");
    let seed = initial_game.state.settings.seed;
    println!("Using seed: {}", seed);

    // Identify tribes in order
    let mut tribe_ids: Vec<_> = initial_game.state.tribes.keys().cloned().collect();
    tribe_ids.sort();
    let tribes: Vec<_> = tribe_ids
        .iter()
        .map(|id| initial_game.state.tribes.get(id).unwrap().tribe_type)
        .collect();

    // Re-generate the game from the same seed to start from the beginning
    let mut game = Game {
        state: polyfish::mapgen::generate(polyfish::mapgen::MapGenSettings {
            seed,
            size: match initial_game.state.settings.size {
                11 => polyfish::types::MapSize::Tiny,
                14 => polyfish::types::MapSize::Small,
                16 => polyfish::types::MapSize::Normal,
                18 => polyfish::types::MapSize::Large,
                20 => polyfish::types::MapSize::Huge,
                30 => polyfish::types::MapSize::Massive,
                _ => polyfish::types::MapSize::Normal,
            },
            map_type: initial_game.state.settings.map_type,
            tribes,
        }),
    };
    game.post_load();

    // The expected sequence of move types
    // Note: User said "step, harvest, harvest, end turn, step on ruins, summon unit, end turn, capture ruin"
    // We will verify the mcts picks these move types in this order.
    let expected_sequence = vec![
        MoveType::Step,
        MoveType::Harvest,
        MoveType::Harvest,
        MoveType::Reward,
        MoveType::EndTurn,
        MoveType::Step, // Step on ruins
        MoveType::Summon,
        MoveType::EndTurn,
        MoveType::Capture, // Capture ruins
    ];

    let agent = HeuristicMctsAgent {
        iterations: 100,
        exploration_constant: 0.4,
    };
    println!("Starting verification sequence...");

    for (i, &expected_type) in expected_sequence.iter().enumerate() {
        println!("Sequence Step {}: Looking for {:?}", i, expected_type);

        let (best_move, analysis) = agent.select_move_with_analysis(&mut game);

        match (best_move.as_ref().map(|m| m.move_type()), expected_type) {
            (Some(actual), expected) if actual == expected => {
                println!(
                    "  OK: MCTS chose {:?} ({})",
                    actual,
                    best_move.as_ref().unwrap().describe(&game.state)
                );
                game.play_move(best_move.as_ref().unwrap().as_ref());

                // IF we ended turn or for some reason it's not P1's turn anymore,
                // skip all other players until it's P1's turn again.
                // We assume P1 is the human player we are testing.
                let mut safety = 0;
                while game.current_player_id() != 1 && safety < 16 {
                    println!("  Skipping Turn for Player {}", game.current_player_id());
                    game.play_move(&polyfish::moves::EndTurnMove);
                    safety += 1;
                }
            }
            (Some(actual), expected) => {
                println!(
                    "  ERROR: Expected {:?}, but MCTS chose {:?}",
                    expected, actual
                );
                println!("  Detailed Analysis (Top 5):");
                for (idx, eval) in analysis.evaluations.iter().take(5).enumerate() {
                    println!(
                        "    {}. {:?} - Visits: {:.1}, Win Rate: {:.2}%",
                        idx + 1,
                        eval.move_type,
                        eval.visits,
                        eval.win_rate * 100.0
                    );
                }

                let legal = game.legal_moves();
                let has_expected = legal.iter().any(|m| m.move_type() == expected);
                println!("  Is expected move {:?} legal? {}", expected, has_expected);

                panic!(
                    "Sequence mismatch at step {}: expected {:?}, got {:?}",
                    i, expected, actual
                );
            }
            (None, expected) => {
                panic!(
                    "MCTS failed to find any move at step {}. Expected {:?}",
                    i, expected
                );
            }
        }
    }

    println!("Sequence matched perfectly!");
}
