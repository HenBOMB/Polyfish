use polyfish::game::Game;
use polyfish::mapgen::MapGenSettings;
use polyfish::types::{MapSize, TribeType};
use std::time::Instant;

fn main() {
    println!("Starting Repro Loop...");

    // Try to simulate random games to catch the loop
    let mut caught = false;
    let iterations = 100;

    for i in 0..iterations {
        if i % 10 == 0 {
            println!("Game {}", i);
        }
        let mut game = Game::new();
        game.state = polyfish::mapgen::generate(MapGenSettings {
            size: MapSize::Tiny, // Tiny to make it faster
            tribes: vec![TribeType::Imperius, TribeType::Imperius],
            seed: i as i64,
            ..Default::default()
        });

        game.state.settings.mode = polyfish::types::ModeType::Domination;
        game.state.settings.max_turns = 30;
        game.post_load();

        let mut move_count = 0;
        let start_time = Instant::now();

        // Run game
        while !polyfish::functions::is_game_over(&game.state) && game.state.settings.turn <= 30 {
            let moves = polyfish::moves::generate_legal_moves(&game.state);
            if moves.is_empty() {
                break;
            }

            // Simple AI: Random Move
            // To better simulate the agent (which picks high score stuff), maybe prioritize some moves?
            // But random should hit logic bugs eventually.
            // We need a deterministic Pseudo-RNG for reproducibility?
            // MapGen used seed `i`. Actions use state seed.
            // But choice of move?

            let move_idx = (game.state.settings.seed as usize) % moves.len();
            let m = &moves[move_idx];

            // Execute
            if let Some(_) = game.play_move(m.as_ref()) {
                move_count += 1;
            } else {
                break;
            }

            if move_count > 2000 {
                println!(
                    "POSSIBLE INFINITE LOOP DETECTED in Game {} at Turn {}",
                    i, game.state.settings.turn
                );
                println!(
                    "Moves exceeding 2000. Last move: {:?}",
                    m.describe(&game.state)
                );
                caught = true;
                break;
            }

            // Hard limit per game
            if start_time.elapsed().as_secs() > 10 {
                println!("Timeout in Game {}", i);
                break;
            }
        }

        if caught {
            break;
        }
    }
}
