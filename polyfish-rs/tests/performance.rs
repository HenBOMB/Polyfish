use polyfish::game::Game;
use polyfish::mapgen::{generate, MapGenSettings};
use polyfish::types::{MapSize, TribeType};
use std::time::Instant;

#[test]
fn benchmark_simulator() {
    // Setup
    let mut settings = MapGenSettings::default();
    settings.size = MapSize::Normal; // Standard size
    settings.tribes = vec![TribeType::Imperius, TribeType::Bardur];
    settings.seed = 42;

    let initial_state = generate(settings);
    let mut game = Game::new();
    game.state = initial_state.clone();

    println!("\n=== Polyfish Rust Simulator Benchmark ===");
    println!("Map Size: 16x16, Tribes: 2");

    // 1. Benchmark Cloning
    let clone_iters = 100_000;
    let start = Instant::now();
    for _ in 0..clone_iters {
        let _ = initial_state.clone();
    }
    let duration = start.elapsed();
    println!(
        "State Cloning: {:.2} ns/op ({:.2} M/s)",
        duration.as_nanos() as f64 / clone_iters as f64,
        clone_iters as f64 / duration.as_secs_f64() / 1_000_000.0
    );

    // 2. Benchmark Move Generation
    // We'll generate moves for the starting state
    let gen_iters = 10_000;
    let start = Instant::now();
    let mut total_moves = 0;
    for _ in 0..gen_iters {
        let moves = game.legal_moves();
        total_moves += moves.len();
    }
    let duration = start.elapsed();
    println!(
        "Move Gen:    {:.2} µs/op ({:.2} k/s) | Avg moves: {:.1}",
        duration.as_micros() as f64 / gen_iters as f64,
        gen_iters as f64 / duration.as_secs_f64() / 1000.0,
        total_moves as f64 / gen_iters as f64
    );

    // 3. Benchmark Move Execution (Random Game)
    // We will play N random games for 30 moves or until end.
    let game_iters = 10000;
    let mut total_turns = 0;
    let start = Instant::now();

    // Pre-allocate RNG to avoid benching RNG creation overhead if possible,
    // but Game might need internal logic. We'll just run loop.

    for i in 0..game_iters {
        // Reset game efficiently?
        // Game::new is cheap if state is ready.
        let mut sim_game = Game::new();
        sim_game.state = initial_state.clone();
        let mut moves_played = 0;

        loop {
            if moves_played > 30 {
                break;
            } // Limit depth

            let moves = sim_game.legal_moves();
            if moves.is_empty() {
                break;
            }

            // Pick first move (deterministic for speed vs rand for coverage)
            // Using index % len
            let idx = i % moves.len();
            let chosen_move = &moves[idx];

            sim_game.play_move(chosen_move.as_ref());
            moves_played += 1;
        }
        total_turns += moves_played;
    }

    let duration = start.elapsed();
    println!(
        "Random Play:   {:.2} µs/move ({:.2} k moves/s)",
        duration.as_micros() as f64 / total_turns as f64,
        total_turns as f64 / duration.as_secs_f64() / 1000.0
    );
    println!(
        "               Simulated {} games, {} total moves in {:.2}s",
        game_iters,
        total_turns,
        duration.as_secs_f64()
    );

    // 4. Benchmark Do/Undo
    // We'll pick one move and repeatedly do/undo it (if valid).
    // Or iterate valid moves.
    let mut undo_game = Game::new();
    undo_game.state = initial_state.clone();
    let moves = undo_game.legal_moves();
    if !moves.is_empty() {
        let chosen_move = &moves[0];
        let undo_iters = 100_000;
        let start = Instant::now();

        for _ in 0..undo_iters {
            if let Some(undo) = undo_game.play_move(chosen_move.as_ref()) {
                undo(&mut undo_game.state);
            } else {
                break; // Should not happen
            }
        }
        let duration = start.elapsed();
        println!(
            "Do/Undo:       {:.2} µs/op ({:.2} k ops/s)",
            duration.as_micros() as f64 / undo_iters as f64,
            undo_iters as f64 / duration.as_secs_f64() / 1000.0
        );
    }

    // 4. Combined TPS (Traits/Transitions Per Second) estimate
    // This effectively measures "Game Updates per Second".
}
