use candle_core::{Device, Tensor};
use polyfish::TribeType;
use polyfish::ai::brain::{Brain, SearchBackend, SearchBackendArg};
use polyfish::ai::eval_server::{EvalServer, EvalServerConfig, Evaluator};
use polyfish::ai::features::{self, GameFeatures, state_to_tensor};
use polyfish::ai::mapper::DecomposedMapper;
use polyfish::ai::network::PolyZeroNet;
use polyfish::game::Game;
use polyfish::replayer::{ModReplay, ReplayPlayer, ReplayTurn};
use polyfish::states::PlayerId;
use polyfish::types::MapSize;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Decomposed policy probability distributions for a single step
struct DecomposedPolicyData {
    action_type: Vec<f32>,    // [11]
    source_spatial: Vec<f32>, // [H * W]
    target_spatial: Vec<f32>, // [H * W]
    move_option: Vec<f32>,    // [192]
}

/// Result from a single game - contains all data needed for training
struct GameResult {
    // Each step: features, policy, player_id, my_score_at_step, opponent_score_at_step, move_type
    history: Vec<(
        GameFeatures,
        DecomposedPolicyData,
        PlayerId,
        i32,
        i32,
        polyfish::types::MoveType,
    )>,
    scores: HashMap<i32, i32>,
    moves: usize,
    winner_score: i32,
    recap: ModReplay,
    action_counts: HashMap<polyfish::types::MoveType, usize>,
}

/// Load the main network (and opponent network, defaulting to the main one)
/// onto the given device from `model.safetensors`.
fn load_networks(
    device: &Device,
    opponent: Option<&str>,
) -> anyhow::Result<(Arc<PolyZeroNet>, Arc<PolyZeroNet>)> {
    let model_path = "model.safetensors";
    if !std::path::Path::new(model_path).exists() {
        anyhow::bail!(
            "Model file {} not found! Please run init_model.py first.",
            model_path
        );
    }
    let mut varmap = candle_nn::VarMap::new();
    varmap.load(model_path)?;
    let network1 = Arc::new(PolyZeroNet::new(candle_nn::VarBuilder::from_varmap(
        &varmap,
        candle_core::DType::F32,
        device,
    ))?);

    let network2 = if let Some(opp_path) = opponent {
        let mut varmap2 = candle_nn::VarMap::new();
        varmap2.load(opp_path)?;
        Arc::new(PolyZeroNet::new(candle_nn::VarBuilder::from_varmap(
            &varmap2,
            candle_core::DType::F32,
            device,
        ))?)
    } else {
        network1.clone()
    };
    Ok((network1, network2))
}

/// Play a single game and return the result
#[allow(clippy::too_many_arguments)]
fn play_single_game(
    network1: &PolyZeroNet,
    network2: &PolyZeroNet, // Added network2
    eval1: &Evaluator,
    eval2: &Evaluator,
    mcts_iters: usize,
    game_idx: usize,
    seed: i64,
    tribes: Vec<TribeType>,
    iteration: usize,
    backend: SearchBackend,
    leaf_batch: Option<usize>,
) -> Option<GameResult> {
    // Curriculum logic — Tiny maps only, gradually increase turn count
    let (map_size, max_turns) = if iteration <= 50 {
        (MapSize::Tiny, 10)
    } else if iteration <= 100 {
        (MapSize::Tiny, 15)
    } else if iteration <= 150 {
        (MapSize::Tiny, 20)
    } else {
        (MapSize::Tiny, 30)
    };

    // Init Game using MapGen
    let gen_settings = polyfish::mapgen::MapGenSettings {
        size: map_size,
        map_type: polyfish::types::MapType::Drylands,
        tribes: tribes.clone(),
        seed,
        ..Default::default()
    };
    eprintln!(
        "[Game {}] Started with seed: {} Tribes: {:?} (Curriculum: {:?}, max_turns: {})",
        game_idx, seed, gen_settings.tribes, map_size, max_turns
    );

    let mut game = Game::new();
    game.state = polyfish::mapgen::generate(gen_settings);
    game.state.settings.mode = polyfish::types::ModeType::Perfection;
    game.state.settings.max_turns = max_turns;
    game.post_load();

    // Create two agents (they might share the same network, or be different)
    let mut agent1 = Brain::with_backend(eval1, mcts_iters, backend);
    let mut agent2 = Brain::with_backend(eval2, mcts_iters, backend);
    if let Some(b) = leaf_batch {
        agent1 = agent1.with_leaf_batch(b);
        agent2 = agent2.with_leaf_batch(b);
    }

    let initial_state = game.state.clone();
    let mut flat_recap: Vec<(i32, i32, serde_json::Value)> = Vec::new();

    // Game Loop
    let mut game_history: Vec<(
        GameFeatures,
        DecomposedPolicyData,
        PlayerId,
        i32,
        i32,
        polyfish::types::MoveType,
    )> = Vec::new();
    let mut action_counts: HashMap<polyfish::types::MoveType, usize> = HashMap::new();

    let current_scores: Vec<(PlayerId, i32)> = game
        .state
        .tribes
        .iter()
        .map(|(id, t)| (*id, t.score))
        .collect();

    eprintln!(
        "[Game {}]: Turn: {} Scores: {:?}",
        game_idx, game.state.settings.turn, current_scores
    );

    let mut move_count = 0;
    while !polyfish::functions::is_game_over(&game.state) {
        if move_count > 50000 {
            // Reduced for safety
            eprintln!(
                "[Game {}] Move count exceeded 50000 (Safety Break)",
                game_idx
            );
            break;
        }

        let pov = game.state.settings.current_player_turn_id;

        // Get state tensor
        let current_network = if pov == 1 { network1 } else { network2 };
        let device = current_network.device();
        let state_t = state_to_tensor(&game.state, pov, &device)
            .expect("BUG: Failed to create state tensor - game state is invalid");

        // MCTS Search - use the correct agent
        let current_agent = if pov == 1 { &mut agent1 } else { &mut agent2 };
        let (best_move, move_visits) = current_agent.think_decomposed(&mut game, move_count);

        let map_size = game.state.settings.size as usize;

        // Initialize probability distributions
        let fixed_map_width = features::MAP_SIZE;
        let fixed_spatial_size = features::MAP_SIZE * fixed_map_width;

        let mut p_action = vec![0.0; 11];
        let mut p_source = vec![0.0; fixed_spatial_size];
        let mut p_target = vec![0.0; fixed_spatial_size];
        let mut p_option = vec![0.0; 192]; // Unified option head (Expanded)

        let mut total_visits = 0.0;

        // Aggregate visits into distributions
        for mv in move_visits {
            total_visits += mv.visits;

            // Spatial and Option targets using DecomposedMapper
            let targets = DecomposedMapper::move_visit_to_targets(&mv, map_size);

            let action_idx = targets.action_type;
            if action_idx < p_action.len() {
                p_action[action_idx] += mv.visits;
            }

            if let Some(i) = targets.source_spatial {
                if i < p_source.len() {
                    p_source[i] += mv.visits;
                }
            }

            if let Some(i) = targets.target_spatial {
                if i < p_target.len() {
                    p_target[i] += mv.visits;
                }
            }

            if let Some(i) = targets.target_type {
                if i < p_option.len() {
                    p_option[i] += mv.visits;
                }
            }
        }

        // Normalize
        if total_visits > 0.0 {
            for x in &mut p_action {
                *x /= total_visits;
            }
            for x in &mut p_source {
                *x /= total_visits;
            }
            for x in &mut p_target {
                *x /= total_visits;
            }
            // ... (others)
        }

        let policy_data = DecomposedPolicyData {
            action_type: p_action,
            source_spatial: p_source,
            target_spatial: p_target,
            move_option: p_option,
        };

        if let Some(m) = best_move {
            let m_type = m.move_type();
            *action_counts.entry(m_type).or_insert(0) += 1;

            flat_recap.push((
                game.state.settings.turn,
                game.state.settings.current_player_turn_id,
                m.serialize(),
            ));
            // Snapshot scores at this moment for reward shaping
            let my_score_now = game.state.tribes.get(&pov).map(|t| t.score).unwrap_or(0);
            let opp_score_now = game
                .state
                .tribes
                .iter()
                .filter(|(id, _)| **id != pov)
                .map(|(_, t)| t.score)
                .max()
                .unwrap_or(0);
            game_history.push((
                state_t,
                policy_data,
                pov,
                my_score_now,
                opp_score_now,
                m.move_type(),
            ));
            if move_count > 0 && move_count % 10 == 0 {
                // let current_scores: Vec<(PlayerId, i32)> = game
                //     .state
                //     .tribes
                //     .iter()
                //     .map(|(id, t)| (*id, t.score))
                //     .collect();
                eprintln!(
                    "[Game {}]: Turn: {} Player: {} Move: {}",
                    game_idx,
                    game.state.settings.turn,
                    pov,
                    m.describe(&game.state),
                    // current_scores
                );
            }
            let _ = game.play_move(m.as_ref());
        } else {
            break;
        }
        move_count += 1;
    }

    // Determine scores & winner
    // In Domination, the winner is the last tribe alive.
    // If the game timed out (safety cap), use score as tiebreaker.
    let mut scores: HashMap<i32, i32> = HashMap::new();
    let mut alive: HashMap<i32, bool> = HashMap::new();
    for (id, t) in &game.state.tribes {
        scores.insert(*id, t.score);
        alive.insert(*id, t.killed_turn <= 0 && t.resigned_turn <= 0);
    }

    // Domination winner: the sole survivor, or highest score if timeout
    let alive_tribes: Vec<i32> = alive
        .iter()
        .filter(|(_, is_alive)| **is_alive)
        .map(|(id, _)| *id)
        .collect();

    let (winner_id, winner_score) = if alive_tribes.len() == 1 {
        let wid = alive_tribes[0];
        (wid, *scores.get(&wid).unwrap_or(&0))
    } else {
        // Timeout: use score tiebreaker
        scores
            .iter()
            .max_by_key(|&(_, score)| score)
            .map(|(&id, &score)| (id, score))
            .unwrap_or((0, 0))
    };

    let is_decisive = alive_tribes.len() == 1;
    eprintln!(
        "[Game {}] Finished. Moves: {} | Winner: {} (Score: {}) | Decisive: {}",
        game_idx, move_count, winner_id, winner_score, is_decisive
    );

    Some(GameResult {
        history: game_history,
        scores,
        moves: move_count,
        winner_score,
        recap: ModReplay {
            game_state: initial_state,
            turns: group_recap(flat_recap),
        },
        action_counts,
    })
}

fn group_recap(flat: Vec<(i32, i32, serde_json::Value)>) -> Vec<ReplayTurn> {
    let mut turns: Vec<ReplayTurn> = Vec::new();
    for (turn_num, player_id, cmd) in flat {
        if turns.is_empty() || turns.last().unwrap().turn != turn_num {
            turns.push(ReplayTurn {
                turn: turn_num,
                players: Vec::new(),
            });
        }
        let turn = turns.last_mut().unwrap();
        if turn.players.is_empty() || turn.players.last().unwrap().player_id != player_id {
            turn.players.push(ReplayPlayer {
                player_id,
                commands: Vec::new(),
            });
        }
        turn.players.last_mut().unwrap().commands.push(cmd);
    }
    turns
}

fn main() -> anyhow::Result<()> {
    use clap::Parser;

    let start_time = Instant::now();
    println!("=== Self-Play Started ===");

    #[derive(Parser, Debug)]
    #[command(author, version, about, long_about = None)]
    struct Args {
        /// Number of games to play
        #[arg(long, default_value_t = 10)]
        num_games: usize,

        /// MCTS iterations per move
        #[arg(long, default_value_t = 50)]
        mcts_iters: usize,

        /// Optional opponent model path (if not set, plays against self)
        #[arg(long)]
        opponent: Option<String>,

        /// First tribe (optional, defaults to random)
        #[arg(long)]
        tribe1: Option<String>,

        /// Second tribe (optional, defaults to random)
        #[arg(long)]
        tribe2: Option<String>,

        /// Enable reward shaping (blended per-step score progress + final outcome)
        /// Without this flag, all actions get the same flat final-outcome value.
        #[arg(long, default_value_t = false)]
        reward_shaping: bool,

        /// Current training iteration (for curriculum learning)
        #[arg(long, default_value_t = 1)]
        iteration: usize,

        /// Search backend to use for MCTS.
        #[arg(long, value_enum, default_value_t = SearchBackendArg::Gumbel)]
        search_backend: SearchBackendArg,

        /// Gumbel: number of initial top-k candidates sampled at the root.
        /// Only used when --search-backend gumbel.
        #[arg(long, default_value_t = 16)]
        gumbel_k: usize,

        /// Number of concurrent game actor threads. Each holds a Game clone
        /// + MCTS tree, so RAM (not CPU) is the real ceiling — actors block
        /// (parking, no CPU used) while awaiting eval-server replies, so
        /// oversubscribing past core count is fine. 0 = use core count.
        #[arg(long, default_value_t = 0)]
        actors: usize,

        /// Eval-server batch cap: max leaves coalesced into one forward_t.
        #[arg(long, default_value_t = 256)]
        max_batch: usize,

        /// Eval-server coalescing flush timeout in microseconds.
        #[arg(long, default_value_t = 1000)]
        coalesce_timeout_us: u64,

        /// Per-game virtual-loss mini-batch size (leaves coalesced per NN
        /// call within a single game's search tree). Cross-game batching via
        /// the eval server now supplies GPU efficiency independently, so
        /// this can shrink toward sequential per-game search. None of the
        /// agents' own defaults (24) are overridden unless this is set.
        #[arg(long)]
        leaf_batch: Option<usize>,

        /// Eval-cache LRU capacity (number of cached NN evaluations). 0
        /// disables the cache. Default is 524288 (512K entries, ~900 MB at
        /// ~1.8 KB per row). The cache lives on the eval-server thread and
        /// skips the GPU for any leaf whose RawFeatures hash to a cached
        /// entry — the only lever that reduces GPU work rather than
        /// reshuffling it. Hit rate is reported in EVAL_SERVER_STATS.
        #[arg(long, default_value_t = 524288)]
        cache_cap: usize,
    }

    let args = Args::parse();

    // Default Metal op-flush cadence. candle-Metal flushes its command buffer
    // every `CANDLE_METAL_COMPUTE_PER_BUFFER` queued ops (default 50), which
    // for an 11x11 net is dominated by dispatch overhead, not math. 1000 lets
    // the GPU amortize dispatch across many ops before a `waitUntilCompleted`.
    // Set before `Device::metal_if_available` so candle picks it up at device
    // init; an explicit env var still wins so benchmarks can A/B test.
    if std::env::var("CANDLE_METAL_COMPUTE_PER_BUFFER").is_err() {
        // SAFETY: this runs at the very top of `main` before any other thread
        // is spawned, so there are no concurrent readers of the environment.
        unsafe {
            std::env::set_var("CANDLE_METAL_COMPUTE_PER_BUFFER", "1000");
        }
    }
    let metal_compute_per_buffer =
        std::env::var("CANDLE_METAL_COMPUTE_PER_BUFFER").unwrap_or_else(|_| "1000".to_string());

    let backend = match args.search_backend {
        SearchBackendArg::Zero => SearchBackend::Zero,
        SearchBackendArg::Gumbel => SearchBackend::Gumbel { k: args.gumbel_k },
    };

    // Select device: Metal (macOS) > CUDA (NVIDIA) > CPU, unless overridden via POLYFISH_DEVICE
    let device = match std::env::var("POLYFISH_DEVICE").as_deref() {
        Ok("cpu") => Device::Cpu,
        Ok("metal") => Device::metal_if_available(0)?,
        Ok("cuda") => Device::cuda_if_available(0)?,
        _ => Device::metal_if_available(0)
            .or_else(|_| Device::cuda_if_available(0))
            .unwrap_or(Device::Cpu),
    };
    println!(
        "Using device: {:?} (CANDLE_METAL_COMPUTE_PER_BUFFER={})",
        device, metal_compute_per_buffer
    );

    // Load models (P1, and P2 defaulting to P1 when no opponent is given)
    let load_start = Instant::now();
    println!("Loading main model from model.safetensors");
    match &args.opponent {
        Some(opp_path) => println!("Loading opponent model from {}", opp_path),
        None => println!("No opponent specified. Playing against self."),
    }
    let (network1, network2) = load_networks(&device, args.opponent.as_deref())?;

    let load_duration = load_start.elapsed();
    println!("Model loading took: {:.2}s", load_duration.as_secs_f32());

    let base_seed = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();

    println!(
        "Starting parallel self-play: {} games with {} MCTS iterations",
        args.num_games, args.mcts_iters
    );

    // Pool of tribes to draw from when tribe1/tribe2 aren't pinned via CLI args.
    // Each game in this run independently samples its own pair from this pool
    // (see `pick_tribes` below), rather than the whole run sharing one fixed pair.
    let all_tribes = vec![
        TribeType::Imperius,
        TribeType::Bardur,
        TribeType::Oumaji,
        TribeType::Kickoo,
        TribeType::XinXi,
        TribeType::Zebasi,
        TribeType::AiMo,
        TribeType::Vengir,
        TribeType::Luxidoor,
        TribeType::Quetzali,
        TribeType::Hoodrick,
        TribeType::Yadakk,
    ];

    fn parse_tribe(s: &str, default: TribeType) -> TribeType {
        match s.to_lowercase().as_str() {
            "imperius" => TribeType::Imperius,
            "bardur" => TribeType::Bardur,
            "oumaji" => TribeType::Oumaji,
            "kickoo" => TribeType::Kickoo,
            "xinxi" => TribeType::XinXi,
            "zebasi" => TribeType::Zebasi,
            "aimo" => TribeType::AiMo,
            "vengir" => TribeType::Vengir,
            "luxidoor" => TribeType::Luxidoor,
            "quetzali" => TribeType::Quetzali,
            "hoodrick" => TribeType::Hoodrick,
            "yadakk" => TribeType::Yadakk,
            "aquarion" => TribeType::Aquarion,
            "elyrion" => TribeType::Elyrion,
            "polaris" => TribeType::Polaris,
            "cymanti" => TribeType::Cymanti,
            _ => {
                eprintln!("Unknown tribe {}, using {:?}", s, default);
                default
            }
        }
    }

    // Picks a (t1, t2) pair for one game. If --tribe1/--tribe2 are given they
    // pin that slot for every game; otherwise a distinct pair is sampled from
    // `all_tribes` using `rng`, so each caller with a different rng gets a
    // different pair.
    fn pick_tribes(
        rng: &mut impl rand::Rng,
        all_tribes: &[TribeType],
        tribe1_arg: &Option<String>,
        tribe2_arg: &Option<String>,
    ) -> (TribeType, TribeType) {
        use rand::seq::SliceRandom;
        let t1 = match tribe1_arg {
            Some(s) => parse_tribe(s, TribeType::Imperius),
            None => *all_tribes.choose(rng).unwrap(),
        };
        let t2 = match tribe2_arg {
            Some(s) => parse_tribe(s, TribeType::Oumaji),
            None => loop {
                let t = *all_tribes.choose(rng).unwrap();
                if t != t1 {
                    break t;
                }
            },
        };
        (t1, t2)
    }

    // Game generation: a pool of actor threads pulls game indices off a
    // shared counter. Each actor blocks (parks, no CPU) while awaiting an
    // eval-server reply, so oversubscribing actors past core count is fine —
    // RAM (a Game clone + MCTS tree per actor) is the real ceiling, not CPU.
    // The eval server owns the sole network/device and coalesces requests
    // from every actor into batched forward_t calls (see ai/eval_server.rs
    // for the Metal cross-thread-tensor invariant this design preserves).
    let games_start = Instant::now();
    println!("Starting game generation...");

    let eval_config = EvalServerConfig {
        max_batch: args.max_batch,
        coalesce_timeout: std::time::Duration::from_micros(args.coalesce_timeout_us),
        cache_capacity: if args.cache_cap == 0 {
            None
        } else {
            Some(args.cache_cap)
        },
    };
    let (eval_server1, eval_handle1) = EvalServer::start(network1.clone(), eval_config);
    let (eval_server2, eval_handle2) = if args.opponent.is_some() {
        let (server, handle) = EvalServer::start(network2.clone(), eval_config);
        (Some(server), handle)
    } else {
        // Self-play against the same weights: reuse one server/handle so we
        // don't run two inference threads (and two device contexts) for the
        // same network.
        (None, eval_handle1.clone())
    };
    let eval1 = Evaluator::Server(eval_handle1);
    let eval2 = Evaluator::Server(eval_handle2);

    let num_actors = if args.actors > 0 {
        args.actors
    } else {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    };
    println!(
        "Using {} actor threads, max_batch={}, coalesce_timeout_us={}, leaf_batch={:?}",
        num_actors, args.max_batch, args.coalesce_timeout_us, args.leaf_batch
    );

    let job_counter = Arc::new(AtomicUsize::new(0));
    let results_mutex: Arc<std::sync::Mutex<Vec<GameResult>>> =
        Arc::new(std::sync::Mutex::new(Vec::with_capacity(args.num_games)));

    std::thread::scope(|scope| {
        for _ in 0..num_actors {
            let job_counter = job_counter.clone();
            let results_mutex = results_mutex.clone();
            let network1 = &network1;
            let network2 = &network2;
            let eval1 = &eval1;
            let eval2 = &eval2;
            let args = &args;
            let all_tribes = &all_tribes;
            scope.spawn(move || {
                loop {
                    let i = job_counter.fetch_add(1, Ordering::Relaxed);
                    if i >= args.num_games {
                        break;
                    }

                    let seed = (base_seed + i as u64) as i64;
                    let swap_players = i % 2 == 1; // Swap every other game
                    let (p1_net, p2_net, p1_eval, p2_eval) = if swap_players {
                        (&**network2, &**network1, eval2, eval1)
                    } else {
                        (&**network1, &**network2, eval1, eval2)
                    };

                    // Sample this game's own tribe pair, seeded off its game
                    // seed so runs stay reproducible while each game gets a
                    // distinct matchup.
                    use rand::SeedableRng;
                    let mut tribe_rng = rand::rngs::StdRng::seed_from_u64(seed as u64);
                    let (t1, t2) =
                        pick_tribes(&mut tribe_rng, all_tribes, &args.tribe1, &args.tribe2);
                    let game_tribes = vec![t1, t2];

                    let result = play_single_game(
                        p1_net,
                        p2_net,
                        p1_eval,
                        p2_eval,
                        args.mcts_iters,
                        i,
                        seed,
                        game_tribes,
                        args.iteration,
                        backend,
                        args.leaf_batch,
                    );

                    if let Some(result) = result {
                        results_mutex.lock().unwrap().push(result);
                    }
                }
            });
        }
    });

    let results: Vec<GameResult> = match Arc::try_unwrap(results_mutex) {
        Ok(mutex) => mutex.into_inner().unwrap(),
        Err(_) => panic!("BUG: actor threads still hold a results_mutex reference after scope exit"),
    };

    let games_duration = games_start.elapsed();
    println!("Game generation completed in: {:.2}s ({} games)", games_duration.as_secs_f32(), results.len());
    println!("  Average: {:.2}s per game", games_duration.as_secs_f32() / results.len().max(1) as f32);

    let mut server_stats = vec![(1, eval_server1.stats())];
    if let Some(ref server2) = eval_server2 {
        server_stats.push((2, server2.stats()));
    }
    for (tag, stats) in server_stats {
        let forwards = stats.forwards.load(Ordering::Relaxed);
        let rows = stats.rows.load(Ordering::Relaxed);
        let max_batch = stats.max_batch.load(Ordering::Relaxed);
        let busy_s = stats.busy_us.load(Ordering::Relaxed) as f64 / 1e6;
        let avg_batch = if forwards > 0 {
            rows as f64 / forwards as f64
        } else {
            0.0
        };
        let cache_hits = stats.cache_hits.load(Ordering::Relaxed);
        let cache_misses = stats.cache_misses.load(Ordering::Relaxed);
        let cache_total = cache_hits + cache_misses;
        let cache_hit_rate = if cache_total > 0 {
            cache_hits as f64 / cache_total as f64
        } else {
            0.0
        };
        println!(
            "EVAL_SERVER_STATS: {{\"server\": {}, \"forwards\": {}, \"rows\": {}, \"avg_batch\": {:.2}, \"max_batch\": {}, \"busy_s\": {:.2}, \"busy_frac\": {:.3}, \"cache_hits\": {}, \"cache_misses\": {}, \"cache_hit_rate\": {:.3}}}",
            tag,
            forwards,
            rows,
            avg_batch,
            max_batch,
            busy_s,
            busy_s / games_duration.as_secs_f64().max(1e-9),
            cache_hits,
            cache_misses,
            cache_hit_rate
        );
    }

    // Aggregate results
    let mut collected_spatial_maps: Vec<Tensor> = Vec::new();
    let mut collected_player_states: Vec<Tensor> = Vec::new();

    // Decomposed policy targets (7 heads)
    let mut collected_action_type: Vec<Vec<f32>> = Vec::new();
    let mut collected_source_spatial: Vec<Vec<f32>> = Vec::new();
    let mut collected_target_spatial: Vec<Vec<f32>> = Vec::new();
    let mut collected_option: Vec<Vec<f32>> = Vec::new();

    let mut collected_values: Vec<f32> = Vec::new();

    let mut total_score = 0;
    let mut max_score = 0;
    let mut best_recap: Option<ModReplay> = None;
    let mut total_moves = 0;

    let mut p1_total = 0;
    let mut p2_total = 0;
    let mut p1_count = 0;
    let mut p2_count = 0;

    let mut total_captures = 0;
    let mut total_harvests = 0;
    let mut total_builds = 0;
    let mut total_research = 0;
    let mut total_attacks = 0;

    for result in results {
        total_score += result.winner_score;
        total_moves += result.moves;
        if result.winner_score > max_score {
            max_score = result.winner_score;
            best_recap = Some(result.recap.clone());
        }

        for (id, score) in &result.scores {
            if *id == 1 {
                p1_total += score;
                p1_count += 1;
            } else if *id == 2 {
                p2_total += score;
                p2_count += 1;
            }
        }

        total_captures += result
            .action_counts
            .get(&polyfish::types::MoveType::Capture)
            .copied()
            .unwrap_or(0);
        total_harvests += result
            .action_counts
            .get(&polyfish::types::MoveType::Harvest)
            .copied()
            .unwrap_or(0);
        total_builds += result
            .action_counts
            .get(&polyfish::types::MoveType::Build)
            .copied()
            .unwrap_or(0);
        total_research += result
            .action_counts
            .get(&polyfish::types::MoveType::Research)
            .copied()
            .unwrap_or(0);
        total_attacks += result
            .action_counts
            .get(&polyfish::types::MoveType::Attack)
            .copied()
            .unwrap_or(0);

        // Backpropagate value
        // Domination: Win/Loss is the primary signal.
        // The winner gets +1.0, loser gets -1.0.
        // If timeout, use score differential as a softer signal.
        let final_scores = &result.scores;
        let history_len = result.history.len();

        // Determine the winner_id for this game
        let game_winner_id = {
            // Check who survived (alive = not killed)
            // We stored scores; for decisive win, one player's score is dominant
            // Use the result.winner_score to identify winner
            let mut best_id = 0;
            let mut best_s = i32::MIN;
            for (&id, &s) in &result.scores {
                if s > best_s {
                    best_s = s;
                    best_id = id;
                }
            }
            best_id
        };

        for (step_idx, (features, policy_data, p_id, my_score_now, opp_score_now, move_type)) in
            result.history.into_iter().enumerate()
        {
            let flat_map = features
                .spatial_map
                .flatten_all()
                .expect("BUG: Failed to flatten spatial map tensor");
            collected_spatial_maps.push(flat_map);

            let flat_player = features
                .player_state
                .flatten_all()
                .expect("BUG: Failed to flatten player state tensor");
            collected_player_states.push(flat_player);

            collected_action_type.push(policy_data.action_type);
            collected_source_spatial.push(policy_data.source_spatial);
            collected_target_spatial.push(policy_data.target_spatial);
            collected_option.push(policy_data.move_option);

            // Perfection: Score-based value target
            // Every game produces a meaningful score — use normalized differential
            let my_final = final_scores.get(&p_id).copied().unwrap_or(0) as f32;
            let opp_final = final_scores
                .iter()
                .filter(|(id, _)| **id != p_id)
                .map(|(_, score)| *score as f32)
                .next()
                .unwrap_or(0.0);

            // Asymmetric Reward Shaping to fix P1 advantage
            let (mut my_adjusted, mut opp_adjusted) = (my_final, opp_final);
            if args.reward_shaping {
                let penalty = 0.05; // 5% adjustment
                if p_id == 1 {
                    my_adjusted = my_final * (1.0 - penalty);
                    opp_adjusted = opp_final * (1.0 + penalty);
                } else if p_id == 2 {
                    my_adjusted = my_final * (1.0 + penalty);
                    opp_adjusted = opp_final * (1.0 - penalty);
                }
            }

            // Normalize by combined economic activity with scaling multiplier
            let combined_score = my_adjusted + opp_adjusted;
            // to spread distribution into useful training range
            let scaling_factor = 3.0;
            let final_outcome = if combined_score > 0.0 {
                let ratio = (my_adjusted - opp_adjusted) / combined_score;
                (ratio * scaling_factor).clamp(-1.0, 1.0)
            } else {
                0.0  // Both players scored 0 - treat as draw
            };

            let value = if args.reward_shaping {
                // Blend final score outcome with per-step progress
                let my_advantage_now = (my_score_now - opp_score_now) as f32;
                let combined_now = (my_score_now + opp_score_now) as f32;
                let progress = if combined_now > 0.0 {
                    let ratio = my_advantage_now / combined_now;
                    (ratio * scaling_factor).clamp(-1.0, 1.0)
                } else {
                    0.0
                };

                // Gradually shift from progress signal to final outcome
                let game_progress = step_idx as f32 / (history_len as f32).max(1.0);
                let final_weight = 0.5 + 0.5 * game_progress; // 0.5 early → 1.0 late
                let progress_weight = 1.0 - final_weight;

                (final_weight * final_outcome + progress_weight * progress).clamp(-1.0, 1.0)
            } else {
                final_outcome.clamp(-1.0, 1.0)
            };

            collected_values.push(value);
        }
    }

    // Print Average Metrics
    let avg_score = total_score as f32 / args.num_games as f32;
    let avr_moves = total_moves as f32 / args.num_games as f32;
    let p1_avg = if p1_count > 0 {
        p1_total as f32 / p1_count as f32
    } else {
        0.0
    };
    let p2_avg = if p2_count > 0 {
        p2_total as f32 / p2_count as f32
    } else {
        0.0
    };

    let avg_captures = total_captures as f32 / args.num_games as f32;
    let avg_harvests = total_harvests as f32 / args.num_games as f32;
    let avg_builds = total_builds as f32 / args.num_games as f32;
    let avg_research = total_research as f32 / args.num_games as f32;
    let avg_attacks = total_attacks as f32 / args.num_games as f32;

    println!(
        "METRICS: {{\"avg_score\": {:.2}, \"max_score\": {}, \"avg_moves\": {:.2}, \"p1_avg\": {:.2}, \"p2_avg\": {:.2}, \"avg_captures\": {:.2}, \"avg_harvests\": {:.2}, \"avg_builds\": {:.2}, \"avg_research\": {:.2}, \"avg_attacks\": {:.2}}}",
        avg_score,
        max_score,
        avr_moves,
        p1_avg,
        p2_avg,
        avg_captures,
        avg_harvests,
        avg_builds,
        avg_research,
        avg_attacks
    );

    // Stack and save
    let save_start = Instant::now();
    if !collected_spatial_maps.is_empty() {
        let total_steps = collected_spatial_maps.len();
        println!("Saving {} steps...", total_steps);

        let spatial_dim = features::NUM_CHANNELS * features::MAP_SIZE * features::MAP_SIZE;
        let player_dim = 10;

        let spatial_maps_tensor = Tensor::cat(&collected_spatial_maps, 0)?;
        let spatial_maps_tensor = spatial_maps_tensor.reshape((total_steps, spatial_dim))?;
        println!(
            "Spatial maps shape: {:?} (dim: {})",
            spatial_maps_tensor.shape(),
            spatial_dim
        );

        let player_states_tensor = Tensor::cat(&collected_player_states, 0)?;
        let player_states_tensor = player_states_tensor.reshape((total_steps, player_dim))?;

        // Helper to simple-flatten data
        fn flatten_vec(v: Vec<Vec<f32>>) -> Vec<f32> {
            v.into_iter().flatten().collect()
        }

        let action_tensor = Tensor::from_vec(
            flatten_vec(collected_action_type),
            (total_steps, 11),
            &device,
        )?;

        let spatial_logit_dim = features::MAP_SIZE * features::MAP_SIZE;

        let source_tensor = Tensor::from_vec(
            flatten_vec(collected_source_spatial),
            (total_steps, spatial_logit_dim),
            &device,
        )?;
        let target_tensor = Tensor::from_vec(
            flatten_vec(collected_target_spatial),
            (total_steps, spatial_logit_dim),
            &device,
        )?;
        let option_tensor =
            Tensor::from_vec(flatten_vec(collected_option), (total_steps, 192), &device)?;

        // Values
        let values_tensor = Tensor::from_vec(collected_values, (total_steps, 1), &device)?;

        let mut tensors = HashMap::new();
        tensors.insert("spatial_maps".to_string(), spatial_maps_tensor);
        tensors.insert("player_states".to_string(), player_states_tensor);

        tensors.insert("action_type".to_string(), action_tensor);
        tensors.insert("source_spatial".to_string(), source_tensor);
        tensors.insert("target_spatial".to_string(), target_tensor);
        tensors.insert("move_option".to_string(), option_tensor);

        tensors.insert("values".to_string(), values_tensor);

        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        let filename = format!("games_{}.safetensors", timestamp);
        candle_core::safetensors::save(&tensors, &filename)?;
        println!("Saved to {}", filename);

        // Save BEST game as replay
        if let Some(recap) = best_recap {
            let replay_filename = format!(
                "replays/high_scores/best_game_score_{}_{}.json",
                max_score, timestamp
            );
            if let Ok(json) = serde_json::to_string_pretty(&recap) {
                if let Ok(mut file) = File::create(&replay_filename) {
                    let _ = file.write_all(json.as_bytes());
                    println!(
                        "🏆 Highest score game ({}) saved to {}",
                        max_score, replay_filename
                    );
                }
            }
        }

        let save_duration = save_start.elapsed();
        println!("Data saving took: {:.2}s", save_duration.as_secs_f32());
    }

    let total_duration = start_time.elapsed();
    println!("\n=== Self-Play Complete ===");
    println!("Total time: {:.2}s", total_duration.as_secs_f32());
    println!("Breakdown:");
    println!("  - Model loading: {:.2}s ({:.1}%)", load_duration.as_secs_f32(), 100.0 * load_duration.as_secs_f32() / total_duration.as_secs_f32());
    println!("  - Game generation: {:.2}s ({:.1}%)", games_duration.as_secs_f32(), 100.0 * games_duration.as_secs_f32() / total_duration.as_secs_f32());
    if !collected_spatial_maps.is_empty() {
        let save_duration = save_start.elapsed();
        println!("  - Data saving: {:.2}s ({:.1}%)", save_duration.as_secs_f32(), 100.0 * save_duration.as_secs_f32() / total_duration.as_secs_f32());
    }

    Ok(())
}
