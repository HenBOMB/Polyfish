use candle_core::Device;
use clap::Parser;
use polyfish::ai::brain::{SearchBackend, SearchBackendArg, make_search_agent};
use polyfish::ai::eval_server::{Evaluator, InlineEvalHandle};
use polyfish::ai::network::PolyZeroNet;
use polyfish::game::Game;
use polyfish::mapgen::{MapGenSettings, generate};
use polyfish::types::{MapSize, MapType, ModeType, TribeType};
use rayon::ThreadPoolBuilder;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
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

    /// Game mode (ModeType repr; 2 = Domination to match training). In
    /// Domination a game is decisive when one side is eliminated; at the
    /// turn cap the higher score wins (same adjudication as self_play).
    #[arg(long, default_value_t = 2)]
    gamemode: u8,

    /// Cap concurrent rayon workers (= concurrent Metal devices on macOS).
    /// Lower this if you hit Metal GPU errors (command-buffer faults under
    /// memory pressure). 0 = use all cores (rayon default).
    #[arg(long, default_value_t = 0)]
    workers: usize,

    /// Append one JSON line per game to this file (the Elo match ledger
    /// consumed by elo.py). Omit to keep the human-readable summary only.
    #[arg(long)]
    json_out: Option<String>,

    /// Ledger player name for configuration 1 (default: derived from
    /// backend + model file stem + mcts iters).
    #[arg(long)]
    name1: Option<String>,

    /// Ledger player name for configuration 2.
    #[arg(long)]
    name2: Option<String>,

    /// Generate mirrored / symmetric 1v1 maps for Elo evaluation.
    #[arg(long, default_value_t = false)]
    symmetric: bool,
}

/// Stable ledger identity: network-free backends need no model, so their name
/// is backend(+iters) only; network backends are model@backend@iters.
fn player_name(backend: SearchBackend, model: &str, mcts: usize) -> String {
    let stem = std::path::Path::new(model)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| model.to_string());
    match backend {
        SearchBackend::Random => "random".to_string(),
        SearchBackend::Greedy => "greedy".to_string(),
        SearchBackend::StateDiffGreedy => "statediffgreedy".to_string(),
        SearchBackend::Heuristic => format!("heuristic{mcts}"),
        SearchBackend::Zero => format!("{stem}@zero{mcts}"),
        SearchBackend::Gumbel { k } => format!("{stem}@gumbel{mcts}k{k}"),
    }
}

fn load_model(path: &str, device: &Device) -> anyhow::Result<PolyZeroNet> {
    // Load trained weights directly with from_mmaped_safetensors for correctness.
    let vs = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&[path], candle_core::DType::F32, device)?
    };
    Ok(PolyZeroNet::new(vs)?)
}

/// Extract a printable message from a caught panic payload.
fn panic_msg(e: &(dyn std::any::Any + Send)) -> &str {
    e.downcast_ref::<&str>()
        .copied()
        .or_else(|| e.downcast_ref::<String>().map(|s| s.as_str()))
        .unwrap_or("unknown panic")
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
    seed: i64,
    swap: bool,
    /// True when the game ended by elimination rather than score at the cap.
    decisive: bool,
}

/// Play one game. `swap` puts config2 in the P1 seat and config1 in P2.
#[allow(clippy::too_many_arguments)]
fn play_match(
    eval1: &Evaluator,
    eval2: &Evaluator,
    mcts1: usize,
    mcts2: usize,
    backend1: SearchBackend,
    backend2: SearchBackend,
    seed: i64,
    swap: bool,
    max_turns: i32,
    gamemode: u8,
    symmetric: bool,
) -> MatchResult {
    let gen_settings = MapGenSettings {
        size: MapSize::Tiny,
        map_type: MapType::Drylands,
        tribes: vec![TribeType::Imperius, TribeType::Imperius],
        seed,
        symmetric,
        ..Default::default()
    };

    let mut game = Game::new();
    game.state = generate(gen_settings);
    game.state.settings.mode = ModeType::from_repr(gamemode).unwrap_or(ModeType::Perfection);
    game.state.settings.max_turns = max_turns;
    game.post_load();

    // p1_config / p2_config map each seat to its configuration so timing and
    // scores attribute to the right config when sides are swapped.
    let (mut agent_p1, p1_config, mut agent_p2, p2_config) = if swap {
        (
            make_search_agent(backend2, eval2, mcts2, None, None, None, None),
            2u8,
            make_search_agent(backend1, eval1, mcts1, None, None, None, None),
            1u8,
        )
    } else {
        (
            make_search_agent(backend1, eval1, mcts1, None, None, None, None),
            1u8,
            make_search_agent(backend2, eval2, mcts2, None, None, None, None),
            2u8,
        )
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

        let cfg = if current_pid == 1 {
            p1_config
        } else {
            p2_config
        };
        if cfg == 1 {
            ns_config1 += dt;
            moves_config1 += 1;
        } else {
            ns_config2 += dt;
            moves_config2 += 1;
        }

        if std::env::var("ARENA_TRACE").is_ok() {
            let vitals = |pid: i32| {
                game.state
                    .tribes
                    .get(&pid)
                    .map(|t| format!("k{}r{}c{}", t.killed_turn, t.resigned_turn, t.cities.len()))
                    .unwrap_or_else(|| "gone".to_string())
            };
            eprintln!(
                "TRACE turn={} pid={} p1[{}] p2[{}] move={}",
                game.state.settings.turn,
                current_pid,
                vitals(1),
                vitals(2),
                best_move
                    .as_ref()
                    .map(|m| m.describe(&game.state))
                    .unwrap_or_else(|| "<none>".to_string())
            );
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
    let is_alive = |pid: i32| {
        game.state
            .tribes
            .get(&pid)
            .map(|t| t.killed_turn <= 0 && t.resigned_turn <= 0)
            .unwrap_or(false)
    };
    let (p1_alive, p2_alive) = (is_alive(1), is_alive(2));

    let (score_config1, score_config2) = if swap {
        (p2_score, p1_score)
    } else {
        (p1_score, p2_score)
    };

    // Elimination beats score adjudication (mirrors self_play's Domination
    // winner logic); a sole survivor wins decisively regardless of score.
    let decisive = p1_alive != p2_alive;
    let winner_config = if decisive {
        let winner_seat = if p1_alive { 1u8 } else { 2u8 };
        if swap { 3 - winner_seat } else { winner_seat }
    } else if score_config1 > score_config2 {
        1
    } else if score_config2 > score_config1 {
        2
    } else {
        0
    };

    if decisive {
        println!(
            "Game seed {} (swap {}) was decisive at turn {}",
            seed, swap, game.state.settings.turn
        );
    }

    MatchResult {
        winner_config,
        score_config1,
        score_config2,
        ns_config1,
        moves_config1,
        ns_config2,
        moves_config2,
        seed,
        swap,
        decisive,
    }
}

fn backend_from_arg(arg: SearchBackendArg, k: usize) -> SearchBackend {
    match arg {
        SearchBackendArg::Zero => SearchBackend::Zero,
        SearchBackendArg::Gumbel => SearchBackend::Gumbel { k },
        SearchBackendArg::Heuristic => SearchBackend::Heuristic,
        SearchBackendArg::Greedy => SearchBackend::Greedy,
        SearchBackendArg::StateDiffGreedy => SearchBackend::StateDiffGreedy,
        SearchBackendArg::Random => SearchBackend::Random,
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    // Select best available device: CUDA (NVIDIA) > Metal (macOS) > CPU
    let device = match Device::cuda_if_available(0) {
        Ok(Device::Cpu) | Err(_) => Device::metal_if_available(0).unwrap_or(Device::Cpu),
        Ok(d) => d,
    };

    println!("Loading models...");
    println!(
        "Config 1: {} (GPU: {:?})",
        args.model1,
        !matches!(device, Device::Cpu)
    );
    let net1 = Arc::new(load_model(&args.model1, &device)?);

    // When both configs use the same model file, share one GPU copy instead of
    // loading a second — doubles GPU memory otherwise (Metal faults under load).
    let same_model = args.model1 == args.model2;
    println!(
        "Config 2: {} (GPU: {:?})",
        args.model2,
        !matches!(device, Device::Cpu)
    );
    let net2 = if same_model {
        net1.clone()
    } else {
        Arc::new(load_model(&args.model2, &device)?)
    };

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

    let arena_start = Instant::now();
    let completed = AtomicU32::new(0);
    let progress_step = ((total_games / 10) as u32).max(1);

    let per_thread_metal = device.is_metal();

    // Cap concurrent workers (= concurrent Metal devices on macOS) if requested.
    let num_workers = if args.workers > 0 {
        args.workers
    } else if per_thread_metal {
        4 // safe Metal worker count
    } else {
        0
    };

    let pool = if num_workers > 0 {
        Some(
            ThreadPoolBuilder::new()
                .num_threads(num_workers)
                .build()
                .expect("failed to build rayon pool"),
        )
    } else {
        None
    };

    // On Metal, sharing one device across threads races inside candle, so each
    // real worker thread gets its own device and network replicas — same
    // pattern as self_play. On CPU/CUDA the loaded networks are shared.
    //
    // We use rayon::broadcast (not par_iter/map_init) because map_init's init
    // closure can be re-invoked whenever work-stealing rebalances a chunk onto
    // a "new" logical task, which silently created far more than `--workers`
    // Metal devices over a long run (compounding GPU memory pressure over
    // time). broadcast runs its closure exactly once per real pool thread.
    let job_counter = AtomicUsize::new(0);
    let skipped = AtomicU32::new(0);
    let results_mutex: Mutex<Vec<MatchResult>> = Mutex::new(Vec::with_capacity(total_games));

    let use_eval_server = !device.is_metal();
    let (server1, handle1) = if use_eval_server {
        let config = polyfish::ai::eval_server::EvalServerConfig {
            max_batch: 256,
            ..Default::default()
        };
        let spec = polyfish::ai::eval_server::BackendSpec::Candle(net1.clone());
        let (s, h) = polyfish::ai::eval_server::EvalServer::start(spec, config);
        (Some(s), Some(h))
    } else {
        (None, None)
    };

    let (server2, handle2) = if use_eval_server && !same_model {
        let config = polyfish::ai::eval_server::EvalServerConfig {
            max_batch: 256,
            ..Default::default()
        };
        let spec = polyfish::ai::eval_server::BackendSpec::Candle(net2.clone());
        let (s, h) = polyfish::ai::eval_server::EvalServer::start(spec, config);
        (Some(s), Some(h))
    } else {
        (None, None)
    };

    let worker = |_ctx: rayon::BroadcastContext| {
        let (eval1, eval2) = if use_eval_server {
            let h1 = handle1.as_ref().unwrap().clone();
            let h2 = if same_model {
                h1.clone()
            } else {
                handle2.as_ref().unwrap().clone()
            };
            (Evaluator::Server(h1), Evaluator::Server(h2))
        } else {
            let (w_net1, w_net2) = if per_thread_metal {
                let device =
                    Device::new_metal(0).expect("failed to create per-thread Metal device");
                let n1 = Arc::new(
                    load_model(&args.model1, &device).expect("failed to load per-thread model1"),
                );
                let n2 = if same_model {
                    n1.clone()
                } else {
                    Arc::new(
                        load_model(&args.model2, &device)
                            .expect("failed to load per-thread model2"),
                    )
                };
                println!(
                    "  worker ready (Metal device loaded) at {:.1}s",
                    arena_start.elapsed().as_secs_f32()
                );
                (n1, n2)
            } else {
                (net1.clone(), net2.clone())
            };
            (
                Evaluator::Inline(InlineEvalHandle::new(w_net1)),
                Evaluator::Inline(InlineEvalHandle::new(w_net2)),
            )
        };

        loop {
            let idx = job_counter.fetch_add(1, Ordering::Relaxed);
            if idx >= args.games * 2 {
                break;
            }
            let seed_idx = idx / 2;
            let swap = (idx % 2) != 0;
            let seed = (base_seed + seed_idx as u64) as i64;

            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                play_match(
                    &eval1,
                    &eval2,
                    mcts1,
                    mcts2,
                    backend1,
                    backend2,
                    seed,
                    swap,
                    args.max_turns,
                    args.gamemode,
                    args.symmetric,
                )
            }));

            match r {
                Ok(r) => {
                    let winner = match r.winner_config {
                        1 => "config1",
                        2 => "config2",
                        _ => "draw",
                    };
                    println!(
                        "  game seed={} swap={}: {} ({}) scores {}-{}",
                        r.seed,
                        r.swap,
                        winner,
                        if r.decisive {
                            "elimination"
                        } else {
                            "score at cap"
                        },
                        r.score_config1,
                        r.score_config2
                    );
                    results_mutex.lock().unwrap().push(r);
                }
                Err(e) => {
                    skipped.fetch_add(1, Ordering::Relaxed);
                    eprintln!(
                        "  ⚠ seed {} swap {} dropped — game panicked: {}",
                        seed,
                        swap,
                        panic_msg(e.as_ref())
                    );
                }
            }

            let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
            // Print early completions immediately (heartbeat) plus every ~10%.
            if done <= 2 || done % progress_step == 0 || done >= total_games as u32 {
                let elapsed = arena_start.elapsed().as_secs_f32();
                let done = done.min(total_games as u32);
                println!(
                    "  progress: {}/{} games ({:.0}%)  elapsed {:.0}s  ~{:.0}s remaining",
                    done,
                    total_games,
                    100.0 * done as f32 / total_games as f32,
                    elapsed,
                    elapsed * (total_games as f32 / done as f32 - 1.0)
                );
            }
        }
    };

    if let Some(p) = &pool {
        p.broadcast(worker);
    } else {
        rayon::broadcast(worker);
    }

    if let Some(s) = server1 {
        s.shutdown();
    }
    if let Some(s) = server2 {
        s.shutdown();
    }

    let results = results_mutex.into_inner().unwrap();

    let arena_elapsed = arena_start.elapsed();

    if let Some(path) = &args.json_out {
        let name1 = args
            .name1
            .clone()
            .unwrap_or_else(|| player_name(backend1, &args.model1, mcts1));
        let name2 = args
            .name2
            .clone()
            .unwrap_or_else(|| player_name(backend2, &args.model2, mcts2));
        let played_at = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs();
        let mut out = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        use std::io::Write;
        for r in &results {
            let result = match r.winner_config {
                1 => "1",
                2 => "2",
                3 => "dropped",
                _ => "draw",
            };
            let line = serde_json::json!({
                "player1": name1,
                "player2": name2,
                "result": result,
                "score1": r.score_config1,
                "score2": r.score_config2,
                "model1": args.model1,
                "model2": args.model2,
                "backend1": format!("{:?}", backend1),
                "backend2": format!("{:?}", backend2),
                "mcts1": mcts1,
                "mcts2": mcts2,
                "max_turns": args.max_turns,
                "mode": args.gamemode,
                "decisive": r.decisive,
                "seed": r.seed,
                "swap": r.swap,
                "played_at": played_at,
            });
            writeln!(out, "{line}")?;
        }
        println!(
            "Appended {} games to {} ({} vs {})",
            results.len(),
            path,
            name1,
            name2
        );
    }

    let mut config1_wins = 0u32;
    let mut config2_wins = 0u32;
    let mut draws = 0u32;
    let mut decisive_games = 0u32;
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
            3 => {}
            _ => draws += 1,
        }
        if r.decisive {
            decisive_games += 1;
        }
        score1_total += r.score_config1 as i64;
        score2_total += r.score_config2 as i64;
        ns1_total += r.ns_config1 as u128;
        moves1_total += r.moves_config1;
        ns2_total += r.ns_config2 as u128;
        moves2_total += r.moves_config2;
    }

    let n = results.len().max(1) as f32;
    let skipped_count = skipped.load(Ordering::Relaxed);
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
    println!(
        "Total Games: {} completed / {} attempted ({} seeds, sides swapped){}",
        results.len(),
        total_games,
        args.games,
        if skipped_count > 0 {
            format!(", {} seed(s) dropped after in-game panics", skipped_count)
        } else {
            String::new()
        }
    );
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
    println!(
        "Decisive (by elimination): {} ({:.1}%)",
        decisive_games,
        (decisive_games as f32 / n) * 100.0
    );
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
    println!(
        "Wall-clock: {:.1}s total  ({:.2} games/s, {:.1}s/game avg)",
        arena_elapsed.as_secs_f32(),
        total_games as f32 / arena_elapsed.as_secs_f32(),
        arena_elapsed.as_secs_f32() / total_games as f32,
    );

    if skipped_count > 0 {
        eprintln!("\nFailing run: {} game(s) were dropped due to panics.", skipped_count);
        std::process::exit(1);
    }

    Ok(())
}
