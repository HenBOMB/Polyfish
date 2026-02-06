use candle_core::Device;
use clap::Parser;
use polyfish::ai::mcts_zero::ZeroMctsAgent;
use polyfish::ai::network::PolyZeroNet;
use polyfish::game::Game;
use polyfish::mapgen::{MapGenSettings, generate};
use polyfish::types::{MapSize, MapType, ModeType, TribeType};
use std::time::Instant;

/// Stats: Calculate average moves available and played
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Number of games to play
    #[arg(long, default_value_t = 10)]
    games: usize,

    /// MCTS Iterations per move (keep low for speed)
    #[arg(long, default_value_t = 25)]
    mcts: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);

    // Initialize dummy network (random weights) for speed if no model found,
    // or load real model if available. Using random is fine for checking legality stats.
    let model_path = "model.safetensors";
    let network = if std::path::Path::new(model_path).exists() {
        println!("Loading model from {}", model_path);
        let mut varmap = candle_nn::VarMap::new();
        varmap.load(model_path)?;
        PolyZeroNet::new(candle_nn::VarBuilder::from_varmap(
            &varmap,
            candle_core::DType::F32,
            &device,
        ))?
    } else {
        println!("Using random weights for stats gathering...");
        PolyZeroNet::new(candle_nn::VarBuilder::zeros(
            candle_core::DType::F32,
            &device,
        ))?
    };

    println!("Starting Stats Collection: {} games...", args.games);

    let mut total_turns_played = 0;
    let mut total_moves_played = 0;
    let mut total_available_sum = 0;
    let mut total_steps_count = 0;

    // Per-turn stats
    let mut moves_per_turn_sum = 0;
    // We need to track moves played in the CURRENT turn to average them at EndTurn
    let mut current_turn_moves = 0;
    let mut current_turn_id = 0;

    for i in 0..args.games {
        let gen_settings = MapGenSettings {
            size: MapSize::Tiny,
            map_type: MapType::Drylands,
            tribes: vec![TribeType::Imperius, TribeType::Imperius], // Mirror match
            seed: rand::random(),
            ..Default::default()
        };

        let mut game = Game::new();
        game.state = generate(gen_settings);
        game.state.settings.mode = ModeType::Perfection;
        game.state.settings.max_turns = 15; // Shorter games
        game.post_load();

        let agent = ZeroMctsAgent::new(&network, args.mcts);

        // Reset per-game trackers
        current_turn_id = game.state.settings.turn;
        current_turn_moves = 0;

        let mut game_steps = 0;

        while !polyfish::functions::is_game_over(&game.state) && game_steps < 300 {
            // Check turn change
            if game.state.settings.turn != current_turn_id {
                // Turn ended
                moves_per_turn_sum += current_turn_moves;
                total_turns_played += 1;

                current_turn_id = game.state.settings.turn;
                current_turn_moves = 0;
            }

            let legal_moves = game.legal_moves();
            if legal_moves.is_empty() {
                break;
            }

            // Record stats BEFORE playing
            let available_count = legal_moves.len();
            total_available_sum += available_count;
            total_steps_count += 1;

            // Pick move
            let best_move = agent.select_move(&mut game);

            if let Some(m) = best_move {
                let is_end_turn = m.move_type() == polyfish::moves::MoveType::EndTurn;
                game.play_move(m.as_ref());

                if !is_end_turn {
                    current_turn_moves += 1;
                    total_moves_played += 1; // Used for global average if needed
                }
            } else {
                break;
            }
            game_steps += 1;
        }
    }

    println!("\n=== GAME STATISTICS ===");
    println!("Games Played:             {}", args.games);
    println!("Total Turns Tracked:      {}", total_turns_played);
    println!("Total Steps Analyzed:     {}", total_steps_count);
    println!("-------------------------");

    if total_steps_count > 0 {
        let avg_available = total_available_sum as f64 / total_steps_count as f64;
        println!("Avg Moves AVAILABLE:      {:.2} (per step)", avg_available);
    }

    if total_turns_played > 0 {
        let avg_played = moves_per_turn_sum as f64 / total_turns_played as f64;
        println!("Avg Moves PLAYED:         {:.2} (per turn)", avg_played);
    }

    Ok(())
}
