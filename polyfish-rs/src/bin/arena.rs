use candle_core::Device;
use clap::Parser;
use polyfish::PlayerId;
use polyfish::ai::mcts_zero::ZeroMctsAgent;
use polyfish::ai::network::PolyZeroNet;
use polyfish::game::Game;
use polyfish::mapgen::{MapGenSettings, generate};
use polyfish::types::{MapSize, MapType, ModeType, TribeType};
use rayon::prelude::*;
use std::time::SystemTime;

/// Arena: Battle two models against each other
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to Player 1's model
    #[arg(long)]
    model1: String,

    /// Path to Player 2's model
    #[arg(long)]
    model2: String,

    /// Number of games to play
    #[arg(long, default_value_t = 10)]
    games: usize,

    /// MCTS Iterations per move
    #[arg(long, default_value_t = 100)]
    mcts: usize,
}

fn load_model(path: &str, device: &Device) -> anyhow::Result<PolyZeroNet> {
    let mut varmap = candle_nn::VarMap::new();
    varmap.load(path)?;
    Ok(PolyZeroNet::new(candle_nn::VarBuilder::from_varmap(
        &varmap,
        candle_core::DType::F32,
        device,
    ))?)
}

fn play_match(
    _game_id: usize,
    net1: &PolyZeroNet,
    net2: &PolyZeroNet,
    mcts: usize,
    seed: u64,
) -> (PlayerId, i32, i32) {
    // (Winner ID, P1 Score, P2 Score)

    // Setup Game
    let gen_settings = MapGenSettings {
        size: MapSize::Tiny,
        map_type: MapType::Drylands,
        tribes: vec![TribeType::Imperius, TribeType::Imperius],
        seed,
        ..Default::default()
    };

    let mut game = Game::new();
    game.state = generate(gen_settings);
    game.state.settings.mode = ModeType::Perfection;
    game.state.settings.max_turns = 10; // 30 turns for a decent match
    game.post_load();

    // Agents
    let agent1 = ZeroMctsAgent::new(net1, mcts);
    let agent2 = ZeroMctsAgent::new(net2, mcts);

    let mut moves = 0;
    while !polyfish::functions::is_game_over(&game.state) && moves < 500 {
        let current_pid = game.state.settings.current_player_turn_id;

        let best_move = if current_pid == 1 {
            // Player 1 uses Net 1
            agent1.select_move(&mut game)
        } else {
            // Player 2 uses Net 2
            agent2.select_move(&mut game)
        };

        if let Some(m) = best_move {
            game.play_move(m.as_ref());
        } else {
            break; // No moves
        }
        moves += 1;
    }

    // Result
    let p1_score = game.state.tribes.get(&1).map(|t| t.score).unwrap_or(0);
    let p2_score = game.state.tribes.get(&2).map(|t| t.score).unwrap_or(0);

    let winner = if p1_score > p2_score {
        1
    } else if p2_score > p1_score {
        2
    } else {
        0 // Draw
    };

    (winner, p1_score, p2_score)
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);

    println!("Loading models...");
    println!("P1: {} (CUDA: {:?})", args.model1, device.is_cuda());
    let net1 = load_model(&args.model1, &device)?;

    println!("P2: {} (CUDA: {:?})", args.model2, device.is_cuda());
    let net2 = load_model(&args.model2, &device)?;

    // We need strict alternating seeds to ensure fairness?
    // Actually, we should probably swap sides half way through?
    // For simplicity, let's just run random seeds.
    // Ideally: Run 2 games per seed (swapping sides).

    let base_seed = SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs();

    println!("Starting Arena: {} games...", args.games);

    let results: Vec<(PlayerId, i32, i32)> = (0..args.games)
        .into_par_iter()
        .map(|i| {
            // We use the same networks (thread safe read only)
            play_match(i, &net1, &net2, args.mcts, base_seed + i as u64)
        })
        .collect();

    let mut p1_wins = 0;
    let mut p2_wins = 0;
    let mut draws = 0;
    let mut p1_total_score = 0;
    let mut p2_total_score = 0;

    for (w, s1, s2) in &results {
        match w {
            1 => p1_wins += 1,
            2 => p2_wins += 1,
            _ => draws += 1,
        }
        p1_total_score += s1;
        p2_total_score += s2;
    }

    println!("\n=== ARENA RESULTS ===");
    println!("Total Games: {}", args.games);
    println!(
        "Model 1 Wins: {} ({:.1}%)",
        p1_wins,
        (p1_wins as f32 / args.games as f32) * 100.0
    );
    println!(
        "Model 2 Wins: {} ({:.1}%)",
        p2_wins,
        (p2_wins as f32 / args.games as f32) * 100.0
    );
    println!("Draws:        {}", draws);
    println!("---------------------");
    println!(
        "Avg Score P1: {:.1}",
        p1_total_score as f32 / args.games as f32
    );
    println!(
        "Avg Score P2: {:.1}",
        p2_total_score as f32 / args.games as f32
    );

    Ok(())
}
