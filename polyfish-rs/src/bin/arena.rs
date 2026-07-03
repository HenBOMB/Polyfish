use candle_core::Device;
use clap::Parser;
use polyfish::ai::brain::{SearchBackend, SearchBackendArg, make_search_agent};
use polyfish::ai::network::PolyZeroNet;
use polyfish::game::Game;
use polyfish::mapgen::{MapGenSettings, generate};
use polyfish::types::{MapSize, MapType, ModeType, TribeType};
use rayon::prelude::*;
use std::sync::Arc;
use std::time::{Instant, SystemTime};

/// Arena: battle two configurations head-to-head.
/// Each seed is played twice with sides swapped; wins are attributed to the
/// configuration, not the seat. Per-move decision time is recorded per config.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to configuration 1's model.
    #[arg(long)]
    model1: String,

    /// Path to configuration 2's model.
    #[arg(long)]
    model2: String,

    /// Number of seeds (each played twice with swapped sides = 2 * games).
    #[arg(long, default_value_t = 10)]
    games: usize,

    /// MCTS iterations per move (override per side with --mcts1 / --mcts2).
    #[arg(long, default_value_t = 100)]
    mcts: usize,

    /// Override MCTS iterations for configuration 1.
    #[arg(long)]
    mcts1: Option<usize>,

    /// Override MCTS iterations for configuration 2.
    #[arg(long)]
    mcts2: Option<usize>,

    /// Search backend for configuration 1.
    #[arg(long, value_enum, default_value_t = SearchBackendArg::Zero)]
    backend1: SearchBackendArg,

    /// Search backend for configuration 2.
    #[arg(long, value_enum, default_value_t = SearchBackendArg::Zero)]
    backend2: SearchBackendArg,

    /// Gumbel top-k at the root (only when a backend is gumbel).
    #[arg(long, default_value_t = 16)]
    gumbel_k: usize,

    /// Max game turns. Higher = more decisive but slower.
    #[arg(long, default_value_t = 30)]
    max_turns: i32,
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

/// Per-match result, attributed to configurations (1 or 2), not seats.
struct MatchResult {
    winner_config: u8,
    score_config1: i32,
    score_config2: i32,
    ns_config1: u64,
    moves_config1: u64,
    ns_config2: u64,
    moves_config2: u64,
}

/// Play one game. `swap` puts config2 in the P1 seat and config1 in P2.
#[allow(clippy::too_many_arguments)]
fn play_match(
    net1: &PolyZeroNet,
    net2: &PolyZeroNet,
    mcts1: usize,
    mcts2: usize,
    backend1: SearchBackend,
    backend2: SearchBackend,
    seed: i64,
    swap: bool,
    max_turns: i32,
) -> MatchResult {
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
    game.state.settings.max_turns = max_turns;
    game.post_load();

    // p1_config / p2_config map each seat to its configuration so timing and
    // scores attribute to the right config when sides are swapped.
    let (agent_p1, p1_config, agent_p2, p2_config) = if swap {
        (make_search_agent(backend2, net2, mcts2), 2u8,
         make_search_agent(backend1, net1, mcts1), 1u8)
    } else {
        (make_search_agent(backend1, net1, mcts1), 1u8,
         make_search_agent(backend2, net2, mcts2), 2u8)
    };

    let mut moves = 0;
    let mut ns_config1: u64 = 0;
    let mut moves_config1: u64 = 0;
    let mut ns_config2: u64 = 0;
    let mut moves_config2: u64 = 0;

    while !polyfish::functions::is_game_over(&game.state) && moves < 500 {
        let current_pid = game.state.settings.current_player_turn_id;

        let t0 = Instant::now();
        let best_move = if current_pid == 1 {
            agent_p1.select_move(&mut game)
        } else {
            agent_p2.select_move(&mut game)
        };
        let dt = t0.elapsed().as_nanos() as u64;

        let cfg = if current_pid == 1 { p1_config } else { p2_config };
        if cfg == 1 {
            ns_config1 += dt;
            moves_config1 += 1;
        } else {
            ns_config2 += dt;
            moves_config2 += 1;
        }

        if let Some(m) = best_move {
            game.play_move(m.as_ref());
        } else {
            break;
        }
        moves += 1;
    }

    let p1_score = game.state.tribes.get(&1).map(|t| t.score).unwrap_or(0);
    let p2_score = game.state.tribes.get(&2).map(|t| t.score).unwrap_or(0);

    let (score_config1, score_config2) = if swap {
        (p2_score, p1_score)
    } else {
        (p1_score, p2_score)
    };

    let winner_config = if score_config1 > score_config2 {
        1
    } else if score_config2 > score_config1 {
        2
    } else {
        0
    };

    MatchResult {
        winner_config,
        score_config1,
        score_config2,
        ns_config1,
        moves_config1,
        ns_config2,
        moves_config2,
    }
}

fn backend_from_arg(arg: SearchBackendArg, k: usize) -> SearchBackend {
    match arg {
        SearchBackendArg::Zero => SearchBackend::Zero,
        SearchBackendArg::Gumbel => SearchBackend::Gumbel { k },
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    // Select best available device: Metal (macOS) > CUDA (NVIDIA) > CPU
    let device = Device::metal_if_available(0)
        .or_else(|_| Device::cuda_if_available(0))
        .unwrap_or(Device::Cpu);

    println!("Loading models...");
    println!("Config 1: {} (GPU: {:?})", args.model1, !matches!(device, Device::Cpu));
    let net1 = Arc::new(load_model(&args.model1, &device)?);

    println!("Config 2: {} (GPU: {:?})", args.model2, !matches!(device, Device::Cpu));
    let net2 = Arc::new(load_model(&args.model2, &device)?);

    let mcts1 = args.mcts1.unwrap_or(args.mcts);
    let mcts2 = args.mcts2.unwrap_or(args.mcts);
    let backend1 = backend_from_arg(args.backend1, args.gumbel_k);
    let backend2 = backend_from_arg(args.backend2, args.gumbel_k);

    println!(
        "Config 1 backend: {:?} (mcts={}), Config 2 backend: {:?} (mcts={}), max_turns={}",
        backend1, mcts1, backend2, mcts2, args.max_turns
    );

    let base_seed = SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs();

    let total_games = args.games * 2;
    println!(
        "Starting Arena: {} seeds x 2 sides = {} games (swapped)...",
        args.games, total_games
    );

    // Each seed produces two games with swapped sides.
    // On Metal, sharing one device across rayon workers races inside candle
    // so each worker builds its own device and network replicas — same pattern as
    // self_play. On CPU/CUDA the loaded networks are shared.
    let per_thread_metal = device.is_metal();
    let results: Vec<MatchResult> = (0..args.games)
        .into_par_iter()
        .map_init(
            || {
                if per_thread_metal {
                    let device = Device::new_metal(0)
                        .expect("failed to create per-thread Metal device");
                    let n1 = Arc::new(
                        load_model(&args.model1, &device)
                            .expect("failed to load per-thread model1"),
                    );
                    let n2 = Arc::new(
                        load_model(&args.model2, &device)
                            .expect("failed to load per-thread model2"),
                    );
                    (n1, n2)
                } else {
                    (net1.clone(), net2.clone())
                }
            },
            |(net1, net2), i| {
                let seed = (base_seed + i as u64) as i64;
                let r1 = play_match(
                    net1, net2, mcts1, mcts2, backend1, backend2, seed, false, args.max_turns,
                );
                let r2 = play_match(
                    net1, net2, mcts1, mcts2, backend1, backend2, seed, true, args.max_turns,
                );
                [r1, r2]
            },
        )
        .flatten_iter()
        .collect();

    let mut config1_wins = 0u32;
    let mut config2_wins = 0u32;
    let mut draws = 0u32;
    let mut score1_total = 0i64;
    let mut score2_total = 0i64;
    let mut ns1_total = 0u128;
    let mut moves1_total = 0u64;
    let mut ns2_total = 0u128;
    let mut moves2_total = 0u64;

    for r in &results {
        match r.winner_config {
            1 => config1_wins += 1,
            2 => config2_wins += 1,
            _ => draws += 1,
        }
        score1_total += r.score_config1 as i64;
        score2_total += r.score_config2 as i64;
        ns1_total += r.ns_config1 as u128;
        moves1_total += r.moves_config1;
        ns2_total += r.ns_config2 as u128;
        moves2_total += r.moves_config2;
    }

    let n = total_games as f32;
    let ms_per_move1 = if moves1_total > 0 {
        (ns1_total as f64) / 1_000_000.0 / (moves1_total as f64)
    } else {
        0.0
    };
    let ms_per_move2 = if moves2_total > 0 {
        (ns2_total as f64) / 1_000_000.0 / (moves2_total as f64)
    } else {
        0.0
    };

    println!("\n=== ARENA RESULTS ===");
    println!("Total Games: {} ({} seeds, sides swapped)", total_games, args.games);
    println!(
        "Config 1 Wins: {} ({:.1}%)",
        config1_wins,
        (config1_wins as f32 / n) * 100.0
    );
    println!(
        "Config 2 Wins: {} ({:.1}%)",
        config2_wins,
        (config2_wins as f32 / n) * 100.0
    );
    println!("Draws:         {}", draws);
    println!("---------------------");
    println!("Avg Score Config 1: {:.1}", score1_total as f32 / n);
    println!("Avg Score Config 2: {:.1}", score2_total as f32 / n);
    println!("---------------------");
    println!(
        "Avg ms/move Config 1: {:.2}  ({} moves, backend={:?}, mcts={})",
        ms_per_move1, moves1_total, backend1, mcts1
    );
    println!(
        "Avg ms/move Config 2: {:.2}  ({} moves, backend={:?}, mcts={})",
        ms_per_move2, moves2_total, backend2, mcts2
    );
    if ms_per_move1 > 0.0 && ms_per_move2 > 0.0 {
        let ratio = ms_per_move1 / ms_per_move2;
        println!(
            "Speed ratio (Config1 / Config2): {:.2}x  (>1 = Config1 is slower)",
            ratio
        );
    }

    Ok(())
}
