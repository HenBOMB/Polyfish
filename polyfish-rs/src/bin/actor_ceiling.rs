//! Actor-side ceiling benchmark.
//!
//! Runs the self-play game-generation path with a dummy evaluator that returns
//! a uniform policy and zero value synchronously on the caller's thread — no
//! network, no device, no eval-server thread, no inference latency. The
//! resulting moves/s is the hard ceiling the game-sim + MCTS + leaf-feature
//! encode path can sustain across `N` actor threads, and every GPU/eval
//! optimization is bounded by it.
//!
//! Mirrors `self_play`'s actor pool, curriculum, tribe sampling, and
//! `Brain::with_backend` + `think_decomposed` move loop exactly — it just
//! swaps the `Evaluator` for `Dummy` and drops training-data collection /
//! saving. So the number it prints is directly comparable to self-play's
//! reported `moves/sec`, minus inference.
//!
//! `--eval-latency-us` (with `--eval-lanes`) turns the dummy into a simulated
//! eval server: each batch "processes" for a fixed duration, optionally
//! bounded to N concurrent batches across all actors. Use it to model how
//! throughput degrades under realistic per-batch inference costs before
//! touching the real backends.

use polyfish::TribeType;
use polyfish::ai::brain::{Brain, SearchBackend, SearchBackendArg};
use polyfish::eval_seeds::{CORE_TRIBES, pick_tribes};
use polyfish::ai::eval_server::{DummyEvalHandle, Evaluator};
use polyfish::game::Game;
use polyfish::mapgen::{MapGenSettings, generate};
use polyfish::types::{MapSize, MapType, ModeType};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Per-game result: just the move count. No history, no recap — we're not
/// training, only measuring throughput.
struct GameTally {
    moves: usize,
}

/// Play one game with the dummy evaluator and return its move count.
#[allow(clippy::too_many_arguments)]
fn play_game(
    eval: &Evaluator,
    mcts_iters: usize,
    game_idx: usize,
    seed: i64,
    tribes: Vec<TribeType>,
    backend: SearchBackend,
    leaf_batch: Option<usize>,
) -> GameTally {
    // Curriculum — identical to self_play so the sim workload matches
    // (flat 50-turn cap, no iteration ramp — see self_play.rs).
    let (map_size, max_turns) = (MapSize::Tiny, 50);

    let gen_settings = MapGenSettings {
        size: map_size,
        map_type: MapType::Drylands,
        tribes: tribes.clone(),
        seed,
        ..Default::default()
    };
    let mut game = Game::new();
    game.state = generate(gen_settings);
    game.state.settings.mode = ModeType::Perfection;
    game.state.settings.max_turns = max_turns;
    game.post_load();

    // One agent per player, reused across the game (same as self_play).
    let mut agent1 = Brain::with_backend(eval, mcts_iters, backend);
    let mut agent2 = Brain::with_backend(eval, mcts_iters, backend);
    if let Some(b) = leaf_batch {
        agent1 = agent1.with_leaf_batch(b);
        agent2 = agent2.with_leaf_batch(b);
    }

    let mut move_count = 0usize;
    while !polyfish::functions::is_game_over(&game.state) {
        if move_count > 50000 {
            eprintln!("[Game {}] Move count exceeded 50000 (safety break)", game_idx);
            break;
        }

        let pov = game.state.settings.current_player_turn_id;
        let current_agent = if pov == 1 { &mut agent1 } else { &mut agent2 };
        let (best_move, _visits) = current_agent.think_decomposed(&game, move_count);

        let Some(m) = best_move else { break };
        let _ = game.play_move(m.as_ref());
        move_count += 1;
    }

    GameTally { moves: move_count }
}

fn main() -> anyhow::Result<()> {
    use clap::Parser;

    println!("=== Actor-Ceiling Benchmark (dummy evaluator, no GPU) ===");

    #[derive(Parser, Debug)]
    #[command(author, version, about, long_about = None)]
    struct Args {
        /// Number of games to play.
        #[arg(long, default_value_t = 20)]
        num_games: usize,

        /// MCTS iterations per move (matches self_play --mcts-iters).
        #[arg(long, default_value_t = 50)]
        mcts_iters: usize,

        /// Current training iteration (selects the curriculum bucket, same
        /// thresholds as self_play).
        #[arg(long, default_value_t = 200)]
        iteration: usize,

        /// Search backend (gumbel | zero). Default matches self_play.
        #[arg(long, value_enum, default_value_t = SearchBackendArg::Gumbel)]
        search_backend: SearchBackendArg,

        /// Gumbel top-k. Only used with --search-backend gumbel.
        #[arg(long, default_value_t = 16)]
        gumbel_k: usize,

        /// Concurrent actor threads. 0 = core count. Each holds a Game clone +
        /// MCTS tree, same RAM profile as self_play.
        #[arg(long, default_value_t = 0)]
        actors: usize,

        /// Per-game virtual-loss mini-batch size (leaves coalesced per eval
        /// call within one game's tree). None = agent default (24).
        #[arg(long)]
        leaf_batch: Option<usize>,

        /// Pin tribe1 for every game (otherwise sampled per game).
        #[arg(long)]
        tribe1: Option<String>,

        /// Pin tribe2 for every game (otherwise sampled per game).
        #[arg(long)]
        tribe2: Option<String>,

        /// Simulated eval latency per batch, in microseconds. Models the
        /// eval server's per-batch processing cost (0 = instant, the pure
        /// actor-side ceiling). E.g. 2000 = 2ms per batch.
        #[arg(long, default_value_t = 0)]
        eval_latency_us: u64,

        /// Simulated concurrent-batch capacity across ALL actors. Models
        /// GPU pipeline lanes: 1 = fully serial GPU, 2 = double-buffered
        /// (the metal backend's shape), 0 = unlimited (every actor's batch
        /// "processes" concurrently — latency without contention). Only
        /// meaningful with --eval-latency-us > 0.
        #[arg(long, default_value_t = 0)]
        eval_lanes: usize,
    }

    let args = Args::parse();

    let backend = match args.search_backend {
        SearchBackendArg::Zero => SearchBackend::Zero,
        SearchBackendArg::Gumbel => SearchBackend::Gumbel { k: args.gumbel_k },
        SearchBackendArg::Heuristic => SearchBackend::Heuristic,
        SearchBackendArg::Greedy => SearchBackend::Greedy,
        SearchBackendArg::MacroScript
        | SearchBackendArg::MacroLookahead
        | SearchBackendArg::MacroMcts => {
            anyhow::bail!("macro backends are arena-only (EXP_ELO_032/033)")
        }
    };

    let num_actors = if args.actors > 0 {
        args.actors
    } else {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    };

    // The v1 training pool; special tribes are deliberately excluded.
    let all_tribes = CORE_TRIBES.to_vec();

    // One dummy evaluator shared by both players (self-play). Cloning the
    // handle shares the same uniform row + stats counters (and simulated
    // eval lanes) across actors.
    let sim_latency = std::time::Duration::from_micros(args.eval_latency_us);
    let dummy = DummyEvalHandle::new().with_sim(sim_latency, args.eval_lanes);
    let eval = Evaluator::Dummy(dummy.clone());

    let base_seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_secs() as i64;

    println!(
        "Config: {} games, {} mcts_iters, {:?} backend, {} actors, leaf_batch={:?}, iteration={}",
        args.num_games, args.mcts_iters, backend, num_actors, args.leaf_batch, args.iteration
    );
    if args.eval_latency_us > 0 {
        println!(
            "Simulated eval: {}us per batch, {} lane(s)",
            args.eval_latency_us,
            if args.eval_lanes == 0 {
                "unlimited".to_string()
            } else {
                args.eval_lanes.to_string()
            }
        );
    }
    println!(
        "Available parallelism: {}",
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(0)
    );

    let games_start = Instant::now();
    println!("Starting game generation...");

    let job_counter = Arc::new(AtomicUsize::new(0));
    let total_moves: Arc<AtomicU64> = Arc::new(AtomicU64::new(0));
    let games_done: Arc<AtomicUsize> = Arc::new(AtomicUsize::new(0));

    std::thread::scope(|scope| {
        for _ in 0..num_actors {
            let job_counter = job_counter.clone();
            let total_moves = total_moves.clone();
            let games_done = games_done.clone();
            let eval = &eval;
            let args = &args;
            let all_tribes = &all_tribes;
            scope.spawn(move || {
                loop {
                    let i = job_counter.fetch_add(1, Ordering::Relaxed);
                    if i >= args.num_games {
                        break;
                    }

                    let seed = base_seed.wrapping_add(i as i64);
                    use rand::SeedableRng;
                    let mut tribe_rng = rand::rngs::StdRng::seed_from_u64(seed as u64);
                    let (t1, t2) =
                        pick_tribes(&mut tribe_rng, all_tribes, &args.tribe1, &args.tribe2);

                    let tally = play_game(
                        eval,
                        args.mcts_iters,
                        i,
                        seed,
                        vec![t1, t2],
                        backend,
                        args.leaf_batch,
                    );

                    total_moves.fetch_add(tally.moves as u64, Ordering::Relaxed);
                    games_done.fetch_add(1, Ordering::Relaxed);
                }
            });
        }
    });

    let elapsed = games_start.elapsed();
    let games = games_done.load(Ordering::Relaxed);
    let moves = total_moves.load(Ordering::Relaxed) as f64;
    let secs = elapsed.as_secs_f64().max(1e-9);

    let moves_per_sec = moves / secs;
    let games_per_sec = games as f64 / secs;
    let avg_moves = moves / games.max(1) as f64;

    let evals = dummy.stats().evals.load(Ordering::Relaxed);
    let leaves = dummy.stats().leaves.load(Ordering::Relaxed) as f64;
    let leaves_per_sec = leaves / secs;
    let leaves_per_move = leaves / moves.max(1e-9);

    println!("\n=== Actor-Ceiling Results ===");
    println!("Wall time:           {:.2}s", secs);
    println!("Games:               {}", games);
    println!("Moves:               {}", moves as u64);
    println!("Avg moves / game:    {:.1}", avg_moves);
    println!("Games / s:           {:.2}", games_per_sec);
    println!("Moves / s (ceiling): {:.2}", moves_per_sec);
    println!("Eval calls:          {}", evals);
    println!("Leaves:              {}", leaves as u64);
    println!("Leaves / s:          {:.2}", leaves_per_sec);
    println!("Leaves / move:       {:.2}", leaves_per_move);
    if args.eval_latency_us > 0 {
        println!(
            "\nThis moves/s models an eval server at {}us/batch across {} actors — not the pure ceiling.",
            args.eval_latency_us, num_actors
        );
    } else {
        println!(
            "\nThis moves/s is the actor-side hard ceiling ({} cores, game-sim + MCTS, zero inference).",
            num_actors
        );
        println!(
            "Every GPU/eval optimization is bounded by {:.0} moves/s.",
            moves_per_sec
        );
    }

    Ok(())
}
