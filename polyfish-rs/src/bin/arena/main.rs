use clap::Parser;
use polyfish::ai::brain::{SearchBackend, SearchBackendArg};
use polyfish::ai::macro_agent::{BeliefMode, MacroLeaf, MacroParams};
use polyfish::ai::eval_backend::{self, EvalBackendKind, PlayerBackend};
use polyfish::ai::eval_server::EvalServerConfig;
use polyfish::ai::network::PolyZeroNet;
use polyfish::eval_seeds::{SeedEntry, load_seed_file, parse_core_tribe, seed_for_game,
                           tribes_for_game};
use polyfish::types::TribeType;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime};
mod cli;
mod dumps;
mod match_play;
mod siege;
use cli::Args;
use match_play::{MatchResult, play_match};

/// Turn at which `--dump-stats-dir` freezes a mid-game board, chosen to sit
/// inside the turn 10-17 window where hubs are actually committed.
pub(crate) const MID_DUMP_TURN: i32 = 12;


fn load_model(path: &str, device: &candle_core::Device) -> anyhow::Result<PolyZeroNet> {
    // Load trained weights directly with from_mmaped_safetensors for correctness.
    let vs = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&[path], candle_core::DType::F32, device)?
    };
    Ok(PolyZeroNet::new(vs)?)
}

/// The map-gen tribe vec is indexed by SEAT, never by config -- it takes
/// no `swap`, so it structurally cannot vary with which config sits where.
pub(crate) fn seat_tribes(tribe1: TribeType, tribe2: TribeType) -> Vec<TribeType> {
    vec![tribe1, tribe2]
}

/// Resolves the CLI backend choice, threading the Gumbel top-k through.
fn backend_from_arg(arg: SearchBackendArg, k: usize) -> SearchBackend {
    match arg {
        SearchBackendArg::Zero => SearchBackend::Zero,
        SearchBackendArg::Gumbel => SearchBackend::Gumbel { k },
        SearchBackendArg::Heuristic => SearchBackend::Heuristic,
        SearchBackendArg::Greedy => SearchBackend::Greedy,
        SearchBackendArg::MacroScript => SearchBackend::MacroScript,
        SearchBackendArg::MacroLookahead => SearchBackend::MacroLookahead,
        SearchBackendArg::MacroMcts => SearchBackend::MacroMcts,
    }
}


fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    if (args.macro_commit || args.macro_star_gate || args.goal_script)
        && !matches!(args.backend1, SearchBackendArg::Gumbel)
    {
        anyhow::bail!(
            "--macro-commit / --macro-star-gate / --goal-script steer config 1's \
             Gumbel agent; pass --backend1 gumbel (EXP_ELO_026/028)"
        );
    }
    if args.goal_w_tree != 0.0 && !args.goal_script {
        anyhow::bail!("--goal-w-tree requires --goal-script (there is no goal to price without it)");
    }
    let is_macro = |b: SearchBackendArg| {
        matches!(
            b,
            SearchBackendArg::MacroScript
                | SearchBackendArg::MacroLookahead
                | SearchBackendArg::MacroMcts
        )
    };
    if (args.macro_k != 4
        || args.macro_horizon != 2
        || args.macro_leaf != MacroLeaf::Heuristic
        || args.macro_lambda != 1.0
        || args.macro_sims != 32)
        && !is_macro(args.backend1)
        && !is_macro(args.backend2)
    {
        anyhow::bail!(
            "--macro-leaf / --macro-k / --macro-horizon / --macro-lambda / --macro-sims \
             configure a macro backend; pass one via --backend1/--backend2 \
             (EXP_ELO_032/033)"
        );
    }
    // EXP_ELO_039 (Stage 3): macro-mcts accepts net leaves — the trained
    // value head challenges the heuristic in the registered paired A/B.
    // Resolve belief modes: the enum flags win; the 035 bool flags alias World.
    let belief_mode1 = if args.macro_belief_mode1 != BeliefMode::Off {
        args.macro_belief_mode1
    } else if args.macro_belief1 {
        BeliefMode::World
    } else {
        BeliefMode::Off
    };
    let belief_mode2 = if args.macro_belief_mode2 != BeliefMode::Off {
        args.macro_belief_mode2
    } else if args.macro_belief2 {
        BeliefMode::World
    } else {
        BeliefMode::Off
    };
    if (belief_mode1 != BeliefMode::Off && !matches!(args.backend1, SearchBackendArg::MacroMcts))
        || (belief_mode2 != BeliefMode::Off
            && !matches!(args.backend2, SearchBackendArg::MacroMcts))
    {
        anyhow::bail!(
            "--macro-belief{{1,2}}/--macro-belief-mode{{1,2}} requires the matching \
             --backend{{1,2}} macro-mcts (EXP_ELO_035/036)"
        );
    }
    let base_params = MacroParams {
        k: args.macro_k,
        horizon: args.macro_horizon,
        leaf: args.macro_leaf,
        lambda: args.macro_lambda,
        // EXP_ELO_061: arena measures strength, not throughput -- keep the
        // rollout/commit split unified (current behavior) here. No CLI
        // override wired in yet; add one if/when quality needs measuring.
        rollout_lambda: args.macro_lambda,
        sims: args.macro_sims,
        belief_mode: BeliefMode::Off,
        shape_w: 0.0,
        root_prior_w: 0.0,
        rollout_nn_w: 0.0,
        rollout_nn_min_depth: usize::MAX,
    };
    let macro_params1 = MacroParams {
        sims: args.macro_sims1.unwrap_or(args.macro_sims),
        k: args.macro_k1.unwrap_or(args.macro_k),
        belief_mode: belief_mode1,
        shape_w: args.macro_shape_w1,
        root_prior_w: args.macro_root_prior_w1,
        rollout_nn_w: args.macro_rollout_nn_w1,
        rollout_nn_min_depth: args.macro_rollout_nn_min_depth1,
        leaf: args.macro_leaf1.unwrap_or(args.macro_leaf),
        ..base_params
    };
    let macro_params2 = MacroParams {
        sims: args.macro_sims2.unwrap_or(args.macro_sims),
        k: args.macro_k2.unwrap_or(args.macro_k),
        belief_mode: belief_mode2,
        shape_w: args.macro_shape_w2,
        root_prior_w: args.macro_root_prior_w2,
        rollout_nn_w: args.macro_rollout_nn_w2,
        rollout_nn_min_depth: args.macro_rollout_nn_min_depth2,
        leaf: args.macro_leaf2.unwrap_or(args.macro_leaf),
        ..base_params
    };
    if is_macro(args.backend1) || is_macro(args.backend2) {
        println!(
            "MACRO BOOTSTRAP (EXP_ELO_032/033): leaf={:?}/{:?} k={}/{} horizon={} lambda={} sims={}/{} belief={:?}/{:?} shape_w={}/{}",
            macro_params1.leaf,
            macro_params2.leaf,
            macro_params1.k,
            macro_params2.k,
            args.macro_horizon,
            args.macro_lambda,
            macro_params1.sims,
            macro_params2.sims,
            macro_params1.belief_mode,
            macro_params2.belief_mode,
            macro_params1.shape_w,
            macro_params2.shape_w,
        );
    }
    if args.goal_script {
        println!(
            "GOAL SCRIPT (EXP_ELO_028): scripted orders+stance drive config 1's goal channels\
             {}",
            if args.goal_w_tree != 0.0 {
                format!(" | goal_w_tree={}", args.goal_w_tree)
            } else {
                String::new()
            }
        );
    }
    if args.macro_commit || args.macro_star_gate {
        println!(
            "ORACLE MACRO (EXP_ELO_026): commit={} star_gate={} (v9: hard block, no reserve)",
            args.macro_commit, args.macro_star_gate,
        );
    }

    // Default Metal op-flush cadence to 1000 for better GPU efficiency on
    // Metal (only exercised by the candle backend; harmless no-op for
    // tch/metal). Mirrors self_play's default.
    if std::env::var("CANDLE_METAL_COMPUTE_PER_BUFFER").is_err() {
        unsafe {
            std::env::set_var("CANDLE_METAL_COMPUTE_PER_BUFFER", "1000");
        }
    }

    let eval_backend_kind = eval_backend::resolve_eval_backend_kind(&args.eval_backend)?;
    let eval_servers = eval_backend::resolve_eval_servers(eval_backend_kind, args.eval_servers)?;

    println!("Loading models...");
    let device1 = eval_backend::select_device()?;
    println!("Config 1: {} (GPU: {:?})", args.model1, !matches!(device1, candle_core::Device::Cpu));
    let net1 = Arc::new(load_model(&args.model1, &device1)?);

    // When both configs use the same model file, share one GPU copy/shard
    // set instead of loading a second — doubles GPU memory otherwise, and
    // under candle, doubles the risk surface for the two-device invariant
    // below.
    let same_model = args.model1 == args.model2;
    println!("Config 2: {} (GPU: {:?})", args.model2, !matches!(device1, candle_core::Device::Cpu));
    let net2 = if same_model {
        net1.clone()
    } else if eval_backend_kind == EvalBackendKind::Candle {
        // Two independent candle EvalServer threads (one per config) will
        // run concurrently — give config 2 its own device so they don't
        // share a command queue (see eval_backend.rs's device-isolation
        // contract; candle's Metal backend corrupts if >1 thread encodes
        // ops against the same Device).
        let device2 = eval_backend::select_device()?;
        Arc::new(load_model(&args.model2, &device2)?)
    } else {
        // tch/metal: each shard loads its own weights from model_path on
        // the eval-server thread; this candle net is unused for inference,
        // only kept to satisfy PlayerBackend's shape.
        Arc::new(load_model(&args.model2, &device1)?)
    };

    let mcts1 = args.mcts1.unwrap_or(args.mcts);
    let mcts2 = args.mcts2.unwrap_or(args.mcts);
    let backend1 = backend_from_arg(args.backend1, args.gumbel_k);
    let backend2 = backend_from_arg(args.backend2, args.gumbel_k);

    println!(
        "Config 1 backend: {:?} (mcts={}), Config 2 backend: {:?} (mcts={}), max_turns={}",
        backend1, mcts1, backend2, mcts2, args.max_turns
    );

    // Each shard sees ~1/N of the working set (hash-routed), so dividing the
    // per-shard cache by N keeps total resident cache ~constant.
    let per_shard_cache = eval_backend::split_cache_capacity(args.cache_cap, eval_servers);
    let eval_config = EvalServerConfig {
        max_batch: args.max_batch,
        coalesce_timeout: std::time::Duration::from_micros(args.coalesce_timeout_us),
        cache_capacity: per_shard_cache,
        pipeline_workers: args.eval_workers,
    };

    // One shared evaluator set per config, backed by dedicated EvalServer
    // threads — the same design self_play uses for cross-match batching.
    // Match-worker threads below never touch a device directly.
    let (p1_servers, p2_servers, eval1, eval2) = eval_backend::build_two_player_evaluators(
        eval_backend_kind,
        eval_servers,
        eval_config,
        PlayerBackend { model_path: &args.model1, candle_net: &net1 },
        (!same_model).then(|| PlayerBackend { model_path: &args.model2, candle_net: &net2 }),
    );

    let backend_label = match eval_backend_kind {
        EvalBackendKind::Tch => "tch (libtorch: MPS/CUDA/CPU)",
        EvalBackendKind::Metal => "metal (MPSGraph)",
        EvalBackendKind::Candle => "candle",
    };

    let base_seed = if args.base_seed > 0 {
        args.base_seed
    } else {
        SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs()
    };

    let seed_entries: Option<Vec<SeedEntry>> = args
        .seed_file
        .as_ref()
        .map(|path| load_seed_file(path, args.games, parse_core_tribe))
        .transpose()?;
    let seed_list: Option<Vec<i64>> = seed_entries
        .as_ref()
        .map(|entries| entries.iter().map(|e| e.seed).collect());

    let total_games = args.games * 2;

    // EXP_ELO_061: the 4x-oversubscription default assumes workers park
    // awaiting eval-server replies. That's true for Zero/Gumbel/net-leaf
    // macro-mcts, but a heuristic-leaf match-worker never touches the eval
    // server (forwards=0) — it's 100% CPU for the whole match, so 4x core
    // count is pure OS-scheduler contention, not extra throughput (see the
    // matched profiling in self_play's --actors doc). If NEITHER config
    // touches the eval server, match to core count instead.
    let touches_eval = |backend: SearchBackendArg, leaf: MacroLeaf| match backend {
        SearchBackendArg::Zero | SearchBackendArg::Gumbel => true,
        // Net, NetAsymPaint, NetAsym all consult the network; only Heuristic
        // is CPU-only.
        SearchBackendArg::MacroMcts => leaf != MacroLeaf::Heuristic,
        SearchBackendArg::Heuristic
        | SearchBackendArg::Greedy
        | SearchBackendArg::MacroScript
        | SearchBackendArg::MacroLookahead => false,
    };
    let either_touches_eval = touches_eval(args.backend1, args.macro_leaf1.unwrap_or(args.macro_leaf))
        || touches_eval(args.backend2, args.macro_leaf2.unwrap_or(args.macro_leaf));

    let concurrency = match args.workers.filter(|&w| w > 0) {
        Some(w) => {
            eprintln!("--workers is deprecated, use --concurrency");
            w
        }
        None if args.concurrency > 0 => args.concurrency,
        // Oversubscribe past core count (workers mostly park awaiting
        // eval-server replies — see the field doc), but never past
        // total_games: a worker with no job left just exits immediately, so
        // more than that is pure overhead, not extra throughput.
        None => {
            let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
            let multiplier = if either_touches_eval { 4 } else { 1 };
            (cores * multiplier).clamp(1, total_games.max(1))
        }
    };
    println!(
        "Starting Arena: {} seeds x 2 sides = {} games (swapped) | eval {backend_label} | \
         {eval_servers} shard(s) cache={per_shard_cache:?} | concurrency={concurrency} \
         max_batch={} coalesce_us={} leaf_batch={:?}",
        args.games, total_games, args.max_batch, args.coalesce_timeout_us, args.leaf_batch
    );

    if let Some(dir) = &args.dump_stats_dir {
        std::fs::create_dir_all(dir)?;
    }
    let dump_stats_dir = args.dump_stats_dir.as_deref();

    if args.belief_calib && dump_stats_dir.is_none() {
        anyhow::bail!("--belief-calib writes into the per-game dumps; pass --dump-stats-dir");
    }

    let trace_tech: Option<polyfish::types::TechnologyType> = match &args.trace_tech {
        None => None,
        Some(name) => {
            use strum::IntoEnumIterator;
            let found = polyfish::types::TechnologyType::iter()
                .find(|t| format!("{t:?}").eq_ignore_ascii_case(name));
            match found {
                Some(t) => {
                    println!("TRACE TECH: capturing root decisions where {t:?} is affordable and unowned");
                    Some(t)
                }
                None => {
                    eprintln!("unknown --trace-tech '{name}'");
                    std::process::exit(2);
                }
            }
        }
    };

    if let Some(dir) = &args.dump_turn_states {
        std::fs::create_dir_all(dir)?;
    }
    let dump_turn_states = args.dump_turn_states.as_deref();
    let tribe_name: &str = &args.tribe;

    let arena_start = Instant::now();
    let completed = AtomicU32::new(0);
    let progress_step = ((total_games / 10) as u32).max(1);
    let job_counter = AtomicUsize::new(0);
    let skipped = AtomicU32::new(0);
    let results_mutex: Mutex<Vec<MatchResult>> = Mutex::new(Vec::with_capacity(total_games));

    // Oversubscribed actor pool: `concurrency` match-worker threads pull
    // independent (seed, swap) jobs off a shared counter and submit into the
    // same eval1/eval2 evaluators, so many concurrent games' leaves coalesce
    // into fat eval-server batches — this cross-match batching (not just the
    // backend swap) is what closes most of the gap to self_play's
    // throughput. Threads borrow eval1/eval2 (not clone+move) so dropping
    // them after this scope closes is enough to unblock EvalServer shutdown
    // below — no clone can outlive the scope and keep a request-channel
    // sender alive.
    std::thread::scope(|scope| {
        for _ in 0..concurrency {
            let eval1 = &eval1;
            let eval2 = &eval2;
            let job_counter = &job_counter;
            let results_mutex = &results_mutex;
            let completed = &completed;
            let skipped = &skipped;
            let seed_list = &seed_list;
            let seed_entries = &seed_entries;
            scope.spawn(move || {
                loop {
                    let idx = job_counter.fetch_add(1, Ordering::Relaxed);
                    if idx >= total_games {
                        break;
                    }
                    let pair_idx = idx / 2;
                    let seed = seed_for_game(pair_idx, base_seed, seed_list.as_deref());
                    // Seed-file per-entry tribes, falling back to --tribe
                    // (applied to both sides) when the entry has none or
                    // --seed-file wasn't used. Same pair for both idx=2i and
                    // idx=2i+1 -- seat_tribes/play_match's swap-invariance
                    // (see there) is what gives both configs both tribes
                    // across the pair, not any flip here.
                    let (tribe1, tribe2) = tribes_for_game(pair_idx, seed_entries.as_deref())
                        .unwrap_or((parse_core_tribe(tribe_name, TribeType::Imperius),
                                    parse_core_tribe(tribe_name, TribeType::Imperius)));
                    let swap = idx % 2 == 1;

                    // Catches ordinary game-logic panics on this thread. A
                    // GPU-driver fault now happens inside the dedicated
                    // EvalServer thread (evaluation no longer runs on this
                    // worker), outside this catch_unwind — see the
                    // eval_server.rs panic-propagation note this mirrors in
                    // self_play. If the eval-server thread itself dies, every
                    // subsequent submit() on it panics (caught here as a
                    // skip), which can cascade into most/all remaining jobs
                    // rather than just the one seed that hit the fault.
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        play_match(
                            eval1, eval2, mcts1, mcts2, backend1, backend2, args.leaf_batch,
                            seed, swap, args.max_turns, args.gamemode, dump_stats_dir,
                            idx, dump_turn_states, args.macro_commit, args.macro_star_gate,
                            args.goal_script, args.goal_w_tree, macro_params1, macro_params2,
                            tribe1, tribe2, trace_tech, args.belief_calib,
                        )
                    }));

                    match result {
                        Ok(r) => {
                            results_mutex.lock().unwrap().push(r);
                        }
                        Err(_) => {
                            skipped.fetch_add(1, Ordering::Relaxed);
                            eprintln!("  ⚠ seed {} (idx {}) skipped after a panic", seed, idx);
                        }
                    }

                    let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
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
            });
        }
    });
    let results = results_mutex.into_inner().unwrap();

    let arena_elapsed = arena_start.elapsed();

    let mut config1_wins = 0u32;
    let mut config2_wins = 0u32;
    let mut draws = 0u32;
    let mut score1_total = 0i64;
    let mut score2_total = 0i64;
    let mut ns1_total = 0u128;
    let mut moves1_total = 0u64;
    let mut ns2_total = 0u128;
    let mut moves2_total = 0u64;

    // Config 1's record split by seat (P1 = first player when !swap).
    let mut c1_wins_p1 = 0u32;
    let mut c1_games_p1 = 0u32;
    let mut c1_wins_p2 = 0u32;
    let mut c1_games_p2 = 0u32;
    let mut depth_sum1: u128 = 0;
    let mut depth_count1: u128 = 0;
    let mut depth_max1: u32 = 0;
    let mut horizon_hits1: u128 = 0;
    let mut agree1: u128 = 0;
    let mut decisions1: u128 = 0;

    for r in &results {
        match r.winner_config {
            1 => config1_wins += 1,
            2 => config2_wins += 1,
            _ => draws += 1,
        }
        if r.swap {
            c1_games_p2 += 1;
            c1_wins_p2 += (r.winner_config == 1) as u32;
        } else {
            c1_games_p1 += 1;
            c1_wins_p1 += (r.winner_config == 1) as u32;
        }
        score1_total += r.score_config1 as i64;
        score2_total += r.score_config2 as i64;
        ns1_total += r.ns_config1 as u128;
        moves1_total += r.moves_config1;
        ns2_total += r.ns_config2 as u128;
        moves2_total += r.moves_config2;
        if let Some((s, c, m, h, ag, dc)) = r.depth_config1 {
            depth_sum1 += s as u128;
            depth_count1 += c as u128;
            depth_max1 = depth_max1.max(m);
            horizon_hits1 += h as u128;
            agree1 += ag as u128;
            decisions1 += dc as u128;
        }
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
            format!(", {} seed(s) skipped after transient errors", skipped_count)
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
    println!("Config 1 Wins as P1: {} (of {})", c1_wins_p1, c1_games_p1);
    println!("Config 1 Wins as P2: {} (of {})", c1_wins_p2, c1_games_p2);
    println!("---------------------");
    println!("Avg Score Config 1: {:.1}", score1_total as f32 / n);
    println!("Avg Score Config 2: {:.1}", score2_total as f32 / n);
    // EXP_ELO_041: siege-defense scoreboard per config.
    for (label, pick) in [
        ("Config 1", &(|r: &MatchResult| r.siege_config1) as &dyn Fn(&MatchResult) -> (u32, u32, u32)),
        ("Config 2", &|r: &MatchResult| r.siege_config2),
    ] {
        let (s, u, l) = results.iter().fold((0u64, 0u64, 0u64), |acc, r| {
            let (s, u, l) = pick(r);
            (acc.0 + s as u64, acc.1 + u as u64, acc.2 + l as u64)
        });
        let rate = if s > 0 { u as f32 / s as f32 * 100.0 } else { 0.0 };
        println!(
            "SIEGE DEFENSE {label}: sieges {s} unsieged {u} ({rate:.0}%) cities_lost {l} | per-game {:.2}/{:.2}/{:.2}",
            s as f32 / n,
            u as f32 / n,
            l as f32 / n
        );
    }
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
    if depth_count1 > 0 {
        println!(
            "TREE DEPTH Config 1: mean {:.2} plies, max {}, over {} sims  |  horizon-capped descents: {} ({:.1}%)",
            depth_sum1 as f64 / depth_count1 as f64,
            depth_max1,
            depth_count1,
            horizon_hits1,
            100.0 * horizon_hits1 as f64 / depth_count1 as f64,
        );
    }
    if decisions1 > 0 {
        let overrides = decisions1 - agree1;
        println!(
            "PRIOR OVERRIDE Config 1: search picked != argmax(prior) on {} / {} root decisions ({:.1}%)",
            overrides,
            decisions1,
            100.0 * overrides as f64 / decisions1 as f64,
        );
    }
    let (div1, planned1) = results.iter().filter_map(|r| r.macro_divergence).fold(
        (0u64, 0u64),
        |(d, p), (dd, pp)| (d + dd as u64, p + pp as u64),
    );
    if planned1 > 0 {
        println!(
            "MACRO DIVERGENCE Config 1: lookahead overrode the scripted base on {} / {} planned turns ({:.1}%)",
            div1,
            planned1,
            100.0 * div1 as f64 / planned1 as f64,
        );
    }
    let (mat_cap, mat_units, mat_planned) = results.iter().filter_map(|r| r.belief_mat).fold(
        (0u64, 0u64, 0u64),
        |(c, u, p), (cc, uu, pp)| (c + cc as u64, u + uu as u64, p + pp as u64),
    );
    if mat_planned > 0 {
        println!(
            "BELIEF MATERIALIZATION Config 1: capital on {} / {} planned turns ({:.1}%), {:.2} units/turn",
            mat_cap,
            mat_planned,
            100.0 * mat_cap as f64 / mat_planned as f64,
            mat_units as f64 / mat_planned as f64,
        );
    }
    let (class_sum, repick_sum, strip_sum) = results.iter().filter_map(|r| r.belief_gen).fold(
        ([0u64; 7], 0u64, 0u64),
        |(mut c, r, s), (cc, rr, ss)| {
            for i in 0..7 {
                c[i] += cc[i] as u64;
            }
            (c, r + rr as u64, s + ss as u64)
        },
    );
    let class_total: u64 = class_sum.iter().sum();
    if class_total > 0 {
        let pct = |i: usize| 100.0 * class_sum[i] as f64 / class_total as f64;
        println!(
            "CANDIDATE CLASS PICKS Config 1: base {:.1}% stance {:.1}% real {:.1}% attack {:.1}% claim {:.1}% contest {:.1}% continuation {:.1}% | repicks {} | mid-turn strips {}",
            pct(0), pct(1), pct(2), pct(3), pct(4), pct(5), pct(6), repick_sum, strip_sum,
        );
    }
    println!(
        "Wall-clock: {:.1}s total  ({:.2} games/s, {:.1}s/game avg)",
        arena_elapsed.as_secs_f32(),
        total_games as f32 / arena_elapsed.as_secs_f32(),
        arena_elapsed.as_secs_f32() / total_games as f32,
    );

    // Deterministic teardown, matching self_play: drop the evaluator handles
    // first (the only remaining request-channel senders) so each eval
    // thread's `recv` errors out and returns, then join the threads before
    // the process starts static/atexit teardown. Without this the tch/
    // libtorch backend can race atexit mutex destruction and abort.
    drop(eval1);
    drop(eval2);
    for server in p1_servers {
        server.shutdown();
    }
    if let Some(p2) = p2_servers {
        for server in p2 {
            server.shutdown();
        }
    }

    Ok(())
}
// NOTE: `arena` has `test = false` in Cargo.toml (like most src/bin/ tools),
// so this module doesn't run under `cargo test`. The seed-selection tests
// that used to sit here moved to src/eval_seeds_tests.rs, where they run.
#[cfg(test)]
mod seat_tests {
    use super::*;


    /// The seat-not-config invariant: `play_match` builds the map-gen tribe
    /// vec from `seat_tribes(tribe1, tribe2)`, which does not take `swap` as
    /// a parameter at all -- structurally, it cannot vary with which config
    /// occupies which seat. This is the property the caller's free
    /// config/tribe fairness across a seed's swapped pair depends on.
    #[test]
    fn seat_tribes_is_indexed_by_seat_not_config() {
        let unswapped = seat_tribes(TribeType::Imperius, TribeType::Bardur);
        let swapped = seat_tribes(TribeType::Imperius, TribeType::Bardur);
        assert_eq!(unswapped, vec![TribeType::Imperius, TribeType::Bardur]);
        // Same tribe1/tribe2 in, same seat-indexed vec out, regardless of
        // which config a caller intends to put in which seat via `swap`.
        assert_eq!(unswapped, swapped);
    }
}
