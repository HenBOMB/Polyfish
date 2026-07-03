//! Benchmark script to identify performance bottlenecks
//!
//! Run with: cargo run --release --bin benchmark

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use polyfish::TribeType;
use polyfish::ai::features::state_to_tensor;
use polyfish::ai::mcts_zero::ZeroMctsAgent;
use polyfish::ai::network::PolyZeroNet;
use polyfish::game::Game;
use polyfish::types::{MapSize, MapType, ModeType};
use std::time::{Duration, Instant};

fn main() -> anyhow::Result<()> {
    println!("=== MCTS Zero Performance Benchmark ===\n");

    // Select best available device: Metal (macOS) > CUDA (NVIDIA) > CPU
    let device = Device::metal_if_available(0)
        .or_else(|_| Device::cuda_if_available(0))
        .unwrap_or(Device::Cpu);
    println!("Using device: {:?}\n", device);
    let network = PolyZeroNet::new(VarBuilder::zeros(DType::F32, &device))?;

    // Configuration
    let num_games = 1;
    let max_turns = 10;
    let mcts_iterations = 25;

    // Timing accumulators
    let mut total_game_time = Duration::ZERO;
    let mut total_mcts_time = Duration::ZERO;
    let mut total_feature_time = Duration::ZERO;
    let mut total_nn_time = Duration::ZERO;
    let mut total_move_gen_time = Duration::ZERO;
    let mut total_move_exec_time = Duration::ZERO;

    let mut total_moves = 0;
    let mut total_legal_moves_generated = 0;

    println!(
        "Config: {} games, {} max turns, {} MCTS iterations\n",
        num_games, max_turns, mcts_iterations
    );

    for game_idx in 0..num_games {
        let game_start = Instant::now();
        let seed = (12345 + game_idx as u64) as i64;

        // Create game using mapgen (same as self_play)
        let gen_settings = polyfish::mapgen::MapGenSettings {
            size: MapSize::Small,
            map_type: MapType::Drylands,
            tribes: vec![TribeType::Imperius, TribeType::Bardur],
            seed,
            ..Default::default()
        };

        let mut game = Game::new();
        game.state = polyfish::mapgen::generate(gen_settings);
        game.state.settings.mode = ModeType::Perfection;
        game.state.settings.max_turns = max_turns;
        game.post_load();

        let agent = ZeroMctsAgent::new(&network, mcts_iterations);
        let mut turn = 0;

        while !game.state.settings._game_over && turn < max_turns * 10 {
            // Benchmark: Legal move generation
            let move_gen_start = Instant::now();
            let legal_moves = game.legal_moves();
            total_move_gen_time += move_gen_start.elapsed();
            total_legal_moves_generated += legal_moves.len();

            if legal_moves.is_empty() {
                break;
            }

            // Benchmark: Feature extraction (standalone)
            let pov = game.state.settings.current_player_turn_id;
            let feature_start = Instant::now();
            let _tensor = state_to_tensor(&game.state, pov, &device);
            total_feature_time += feature_start.elapsed();

            // Benchmark: NN inference (standalone)
            let nn_start = Instant::now();
            let features = state_to_tensor(&game.state, pov, &device)?;
            let (_policy, _value) =
                network.forward_t(&features.spatial_map, &features.player_state, false)?;
            total_nn_time += nn_start.elapsed();

            // Benchmark: Full MCTS search
            let mcts_start = Instant::now();
            let best_move = agent.select_move(&mut game);
            total_mcts_time += mcts_start.elapsed();

            // Benchmark: Move execution
            if let Some(m) = best_move {
                let exec_start = Instant::now();
                game.play_move(m.as_ref());
                total_move_exec_time += exec_start.elapsed();
                total_moves += 1;
            }

            turn += 1;
        }

        total_game_time += game_start.elapsed();
        println!(
            "Game {}: {} turns, {:.2}s",
            game_idx + 1,
            turn,
            game_start.elapsed().as_secs_f64()
        );
    }

    println!("\n=== TIMING BREAKDOWN ===\n");

    let total_secs = total_game_time.as_secs_f64();

    println!(
        "{:<25} {:>10.3}s  ({:>5.1}%)",
        "Total time:", total_secs, 100.0
    );
    println!("{:-<45}", "");

    let print_timing = |name: &str, duration: Duration| {
        let secs = duration.as_secs_f64();
        let pct = (secs / total_secs) * 100.0;
        println!("{:<25} {:>10.3}s  ({:>5.1}%)", name, secs, pct);
    };

    print_timing("MCTS search:", total_mcts_time);
    print_timing("NN inference (isolated):", total_nn_time);
    print_timing("Feature extraction:", total_feature_time);
    print_timing("Legal move generation:", total_move_gen_time);
    print_timing("Move execution:", total_move_exec_time);

    println!("\n=== STATISTICS ===\n");

    println!("Total moves played:      {}", total_moves);
    println!("Total legal moves gen:   {}", total_legal_moves_generated);

    if total_moves > 0 {
        println!(
            "Avg MCTS time/move:      {:.1}ms",
            total_mcts_time.as_secs_f64() * 1000.0 / total_moves as f64
        );
        println!(
            "Avg NN time/inference:   {:.1}ms",
            total_nn_time.as_secs_f64() * 1000.0 / total_moves as f64
        );
        println!(
            "Avg move gen time:       {:.3}ms",
            total_move_gen_time.as_secs_f64() * 1000.0 / total_moves as f64
        );
        println!(
            "Avg legal moves/turn:    {:.1}",
            total_legal_moves_generated as f64 / total_moves as f64
        );
    }

    println!("\n=== MCTS BREAKDOWN (estimated) ===\n");

    let mcts_total = total_mcts_time.as_secs_f64();

    // Each MCTS iteration does ~1 NN call during expansion
    // Estimate: NN calls in MCTS ≈ iterations * moves
    let estimated_nn_calls = mcts_iterations * total_moves;
    let avg_nn_per_call = total_nn_time.as_secs_f64() / total_moves as f64;
    let estimated_nn_time = avg_nn_per_call * estimated_nn_calls as f64;

    println!(
        "Estimated NN time in MCTS: {:.1}s ({:.1}% of MCTS)",
        estimated_nn_time.min(mcts_total),
        (estimated_nn_time / mcts_total * 100.0).min(100.0)
    );
    println!(
        "Remaining (selection/backprop): {:.1}s ({:.1}%)",
        (mcts_total - estimated_nn_time).max(0.0),
        ((mcts_total - estimated_nn_time) / mcts_total * 100.0).max(0.0)
    );

    println!("\n=== BOTTLENECK ANALYSIS ===\n");

    let nn_pct = total_nn_time.as_secs_f64() / total_secs * 100.0;
    let mcts_pct = total_mcts_time.as_secs_f64() / total_secs * 100.0;
    let move_gen_pct = total_move_gen_time.as_secs_f64() / total_secs * 100.0;

    if mcts_pct > 80.0 {
        println!(
            "🔴 MCTS search is the dominant bottleneck ({:.1}%)",
            mcts_pct
        );
        if nn_pct > 50.0 {
            println!("   └─ NN inference is likely the main component");
        }
    } else if nn_pct > 50.0 {
        println!(
            "🔴 NN inference is the dominant bottleneck ({:.1}%)",
            nn_pct
        );
    } else if move_gen_pct > 20.0 {
        println!(
            "🟡 Legal move generation is significant ({:.1}%)",
            move_gen_pct
        );
    } else {
        println!("🟢 No single dominant bottleneck detected");
    }

    Ok(())
}
