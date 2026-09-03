//! Running a batch of games and reporting how the eval servers behaved.
//!
//! Actors are OS threads pulling game indices off one shared counter, not a
//! rayon pool: each blocks while awaiting an eval-server reply, so
//! oversubscribing past core count is fine and the work-stealing keeps long
//! games from stranding a thread.
//!
//! `EVAL_SERVER_STATS_AGG:` and the `Throughput:` line are stdout contracts --
//! bench_eval_sweep.sh regex-matches both.

use candle_core::Device;
use polyfish::TribeType;
use polyfish::ai::eval_backend::{self, EvalBackendKind};
use polyfish::ai::brain::SearchBackend;
use polyfish::ai::eval_server::{EvalServer, EvalServerStats, Evaluator};
use polyfish::ai::macro_agent::MacroParams;
use polyfish::ai::network::PolyZeroNet;
use polyfish::eval_seeds::{SeedEntry, resolve_tribes, seed_for_game, tribes_for_game};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use crate::ProgressMode;
use crate::cli::Args;
use crate::crutches::{ANCHOR_FRAC_DECAY, decay_crutch};
use crate::game::play_single_game;
use crate::result::GameResult;
use crate::stats::finish_milestones;

/// Plays `args.num_games` games across `num_actors` threads and collects the
/// results. Game index parity drives seat swapping, so a seed's two halves
/// see both seats.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_games(
    num_actors: usize,
    args: &Args,
    network1: &Arc<PolyZeroNet>,
    network2: &Arc<PolyZeroNet>,
    eval1: &Evaluator,
    eval2: &Evaluator,
    all_tribes: &[TribeType],
    seed_list: &Option<Vec<i64>>,
    seed_entries: &Option<Vec<SeedEntry>>,
    base_seed: u64,
    backend: SearchBackend,
    macro_params: MacroParams,
    progress_mode: ProgressMode,
    has_opponent: bool,
) -> Vec<GameResult> {
    let job_counter = Arc::new(AtomicUsize::new(0));
    let games_completed = Arc::new(AtomicUsize::new(0));
    let trace_counter = Arc::new(AtomicUsize::new(0));
    let finish_milestones = finish_milestones(args.num_games);
    let results_mutex: Arc<std::sync::Mutex<Vec<GameResult>>> =
        Arc::new(std::sync::Mutex::new(Vec::with_capacity(args.num_games)));

    std::thread::scope(|scope| {
        for _ in 0..num_actors {
            let job_counter = job_counter.clone();
            let results_mutex = results_mutex.clone();
            let games_completed = games_completed.clone();
            let trace_counter = trace_counter.clone();
            let finish_milestones = finish_milestones.clone();
            let network1 = &network1;
            let network2 = &network2;
            let eval1 = &eval1;
            let eval2 = &eval2;
            let args = &args;
            let all_tribes = &all_tribes;
            let seed_list = &seed_list;
            let seed_entries = &seed_entries;
            scope.spawn(move || {
                loop {
                    let i = job_counter.fetch_add(1, Ordering::Relaxed);
                    if i >= args.num_games {
                        break;
                    }

                    let seed = seed_for_game(i, base_seed, seed_list.as_deref());
                    let swap_players = i % 2 == 1; // Swap every other game
                    let (p1_net, p2_net, p1_eval, p2_eval) = if swap_players {
                        (&**network2, &**network1, eval2, eval1)
                    } else {
                        (&**network1, &**network2, eval1, eval2)
                    };

                    // Anchor games: evenly spread across the run at rate
                    // anchor_frac (decayed from its starting value the same
                    // way prior_heuristic_weight is — see decay_crutch);
                    // the anchor's seat alternates by anchor ordinal (game
                    // parity alone would pin it to one seat at e.g. frac
                    // 0.25, where anchor games are all odd-i).
                    let anchor_frac = decay_crutch(
                        args.anchor_frac,
                        ANCHOR_FRAC_DECAY,
                        args.iteration.saturating_sub(args.anchor_decay_start),
                        args.decay_last_iter,
                        args.force_zero_crutches,
                    );
                    let anchor_ordinal = (((i + 1) as f32) * anchor_frac).floor() as usize;
                    let is_anchor = anchor_frac > 0.0
                        && anchor_ordinal > ((i as f32) * anchor_frac).floor() as usize;
                    // Greedy (score_move argmax), not the rollout Heuristic MCTS:
                    // measured first-village capture 1.00/t6.5 vs 0.94/t8.9 — the
                    // rollout noise drowned the ordering gradient. Greedy is also
                    // the exact distribution blend_heuristic_prior injects into the
                    // net's root, so anchor data and search priors agree.
                    // --anchor-seat pins the Greedy seat; otherwise it
                    // alternates so neither seat accumulates a side bias.
                    let anchor_first = match args.anchor_seat {
                        Some(1) => true,
                        Some(2) => false,
                        _ => anchor_ordinal % 2 == 0,
                    };
                    let (backend_seat1, backend_seat2) = if is_anchor {
                        if anchor_first {
                            (SearchBackend::Greedy, backend)
                        } else {
                            (backend, SearchBackend::Greedy)
                        }
                    } else {
                        (backend, backend)
                    };

                    // Seat roles for tempo aggregation: "model" (mirror seat),
                    // "model_vs_anchor" (net seat racing the anchor — the
                    // contested population), "anchor" (Greedy reference
                    // curve), "opponent" (league checkpoint seat).
                    let seat_roles: [&'static str; 2] = if is_anchor {
                        if anchor_first {
                            ["anchor", "model_vs_anchor"]
                        } else {
                            ["model_vs_anchor", "anchor"]
                        }
                    } else if has_opponent {
                        if swap_players {
                            ["opponent", "model"]
                        } else {
                            ["model", "opponent"]
                        }
                    } else {
                        ["model", "model"]
                    };

                    // Sample this game's own tribe pair, seeded off its game
                    // seed so runs stay reproducible while each game gets a
                    // distinct matchup. See `resolve_tribes` for the
                    // CLI > seed-file > random precedence.
                    use rand::SeedableRng;
                    let mut tribe_rng = rand::rngs::StdRng::seed_from_u64(seed as u64);
                    let seed_file_tribes = tribes_for_game(i, seed_entries.as_deref());
                    let (t1, t2) = resolve_tribes(
                        &mut tribe_rng,
                        all_tribes,
                        &args.tribe1,
                        &args.tribe2,
                        seed_file_tribes,
                    );
                    let game_tribes = vec![t1, t2];

                    // Ensure panicking game doesnt kill the whole run
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        play_single_game(
                            p1_net,
                            p2_net,
                            p1_eval,
                            p2_eval,
                            args.mcts_iters,
                            i,
                            seed,
                            game_tribes,
                            args.iteration,
                            args.decay_last_iter,
                            args.force_zero_crutches,
                            args.gamemode,
                            backend_seat1,
                            backend_seat2,
                            args.value_trust,
                            args.leaf_batch,
                            progress_mode,
                            args.trace_villages,
                            args.trace_trigger,
                            args.trace_max,
                            &trace_counter,
                            args.dump_failed_dir.as_deref(),
                            args.dump_games_dir.as_deref(),
                            args.dump_turn_states.as_deref(),
                            args.dump_city_rewards.as_deref(),
                            args.dump_star_spend.as_deref(),
                            args.dump_reward_choices.as_deref(),
                            args.dump_level_completion.as_deref(),
                            args.dump_pop_spend_choices.as_deref(),
                            args.dump_macro_policy.as_deref(),
                            seat_roles,
                            args.shape_w_label,
                            args.shape_w_tree,
                            args.pursuit_w_label,
                            args.pursuit_w_tree,
                            args.unfreeze_opponent,
                            args.dagger_alpha,
                            args.goal_channels,
                            args.goal_w_tree,
                            macro_params,
                            args.max_turns,
                            args.seed_search,
                        )
                    }))
                    .unwrap_or_else(|_| {
                        eprintln!("[ERROR] Game {i} (seed {seed}) panicked — discarding its data");
                        None
                    });

                    if let Some(result) = result {
                        if progress_mode == ProgressMode::SampledFinish {
                            let done = games_completed.fetch_add(1, Ordering::Relaxed) + 1;
                            if finish_milestones.contains(&done) {
                                eprintln!(
                                    "[Progress] {}/{} games complete (game {} — {} moves, winner score {})",
                                    done,
                                    args.num_games,
                                    i,
                                    result.moves,
                                    result.winner_score,
                                );
                            }
                        }
                        results_mutex.lock().unwrap().push(result);
                    }
                }
            });
        }
    });

    match Arc::try_unwrap(results_mutex) {
        Ok(mutex) => mutex.into_inner().unwrap(),
        Err(_) => {
            panic!("BUG: actor threads still hold a results_mutex reference after scope exit")
        }
    }
}


/// Prints the throughput summary and the aggregated per-shard eval-server
/// counters. Both lines are parsed by bench_eval_sweep.sh.
pub(crate) fn report_eval_stats(
    games_duration: Duration,
    results: &[GameResult],
    p1_servers: &[EvalServer],
    p2_servers: Option<&Vec<EvalServer>>,
) {
    println!(
        "Game generation completed in: {:.2}s ({} games)",
        games_duration.as_secs_f32(),
        results.len()
    );
    println!(
        "  Average: {:.2}s per game",
        games_duration.as_secs_f32() / results.len().max(1) as f32
    );
    let total_moves_now: usize = results.iter().map(|r| r.moves).sum();
    let moves_per_sec = total_moves_now as f64 / games_duration.as_secs_f64().max(1e-9);
    println!(
        "  Throughput: {:.2} moves/sec ({} moves over {:.2}s)",
        moves_per_sec,
        total_moves_now,
        games_duration.as_secs_f32()
    );

    // Eval-server stats: aggregate across all shards (the number to compare
    // against the single-server baseline).
    let mut all_shard_stats: Vec<(&str, &EvalServerStats)> = Vec::new();
    for s in p1_servers.iter() {
        all_shard_stats.push(("p1", s.stats()));
    }
    if let Some(p2) = p2_servers {
        for s in p2 {
            all_shard_stats.push(("p2", s.stats()));
        }
    }

    let wall_s = games_duration.as_secs_f64().max(1e-9);

    // Aggregate across shards.
    let (mut agg_forwards, mut agg_rows, mut agg_max_batch, mut agg_busy_us) =
        (0u64, 0u64, 0u64, 0u64);
    let (mut agg_hits, mut agg_misses) = (0u64, 0u64);
    let (mut agg_compiles, mut agg_compile_us) = (0u64, 0u64);
    let (mut agg_prep_us, mut agg_wait_us, mut agg_post_us) = (0u64, 0u64, 0u64);
    for (_, s) in &all_shard_stats {
        agg_forwards += s.forwards.load(Ordering::Relaxed);
        agg_rows += s.rows.load(Ordering::Relaxed);
        agg_max_batch = agg_max_batch.max(s.max_batch.load(Ordering::Relaxed));
        agg_busy_us += s.busy_us.load(Ordering::Relaxed);
        agg_hits += s.cache_hits.load(Ordering::Relaxed);
        agg_misses += s.cache_misses.load(Ordering::Relaxed);
        agg_compiles += s.compiles.load(Ordering::Relaxed);
        agg_compile_us += s.compile_us.load(Ordering::Relaxed);
        agg_prep_us += s.prep_us.load(Ordering::Relaxed);
        agg_wait_us += s.wait_us.load(Ordering::Relaxed);
        agg_post_us += s.post_us.load(Ordering::Relaxed);
    }
    let agg_busy_s = agg_busy_us as f64 / 1e6;
    let agg_compile_s = agg_compile_us as f64 / 1e6;
    let agg_avg_batch = if agg_forwards > 0 {
        agg_rows as f64 / agg_forwards as f64
    } else {
        0.0
    };
    let agg_cache_total = agg_hits + agg_misses;
    let agg_cache_hit_rate = if agg_cache_total > 0 {
        agg_hits as f64 / agg_cache_total as f64
    } else {
        0.0
    };
    println!(
        "EVAL_SERVER_STATS_AGG: {{\"shards\": {}, \"forwards\": {}, \"rows\": {}, \"avg_batch\": {:.2}, \"max_batch\": {}, \"busy_s\": {:.2}, \"busy_frac\": {:.3}, \"prep_s\": {:.2}, \"wait_s\": {:.2}, \"post_s\": {:.2}, \"cache_hits\": {}, \"cache_misses\": {}, \"cache_hit_rate\": {:.3}, \"compiles\": {}, \"compile_s\": {:.3}, \"compile_frac_wall\": {:.4}, \"compile_frac_busy\": {:.4}}}",
        all_shard_stats.len(),
        agg_forwards,
        agg_rows,
        agg_avg_batch,
        agg_max_batch,
        agg_busy_s,
        agg_busy_s / wall_s,
        agg_prep_us as f64 / 1e6,
        agg_wait_us as f64 / 1e6,
        agg_post_us as f64 / 1e6,
        agg_hits,
        agg_misses,
        agg_cache_hit_rate,
        agg_compiles,
        agg_compile_s,
        agg_compile_s / wall_s,
        if agg_busy_s > 0.0 {
            agg_compile_s / agg_busy_s
        } else {
            0.0
        }
    );
}

/// Load the main network (and opponent network, defaulting to the main one)
/// onto the given device from `model.safetensors`.
///
/// When `eval_backend_kind` is `Candle` and a distinct opponent is given, the
/// opponent network is loaded on its own freshly-obtained device rather than
/// `device`: under Candle, player 1 and player 2 each get an independent
/// `EvalServer` thread, and candle's Metal backend corrupts if two threads
/// encode ops (e.g. `forward_t`) against the same `Device` (see
/// `eval_backend.rs`'s device-isolation contract). tch/metal shards load
/// their own weights on the eval-server thread and never touch this candle
/// device for inference, so sharing is harmless for them.
pub(crate) fn load_networks(
    device: &Device,
    opponent: Option<&str>,
    eval_backend_kind: EvalBackendKind,
) -> anyhow::Result<(Arc<PolyZeroNet>, Arc<PolyZeroNet>)> {
    let model_path = "model.safetensors";
    if !std::path::Path::new(model_path).exists() {
        anyhow::bail!(
            "Model file {} not found! Please run init_model.py first.",
            model_path
        );
    }
    // Inference-only load: `VarBuilder::from_mmaped_safetensors` loads by key from file;
    // VarMap::load fills only pre-registered vars.
    let vs1 = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(
            &[model_path],
            candle_core::DType::F32,
            device,
        )?
    };
    let network1 = Arc::new(PolyZeroNet::new(vs1)?);

    let network2 = if let Some(opp_path) = opponent {
        let device2 = if eval_backend_kind == EvalBackendKind::Candle {
            eval_backend::select_device()?
        } else {
            device.clone()
        };
        let vs2 = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                &[opp_path],
                candle_core::DType::F32,
                &device2,
            )?
        };
        Arc::new(PolyZeroNet::new(vs2)?)
    } else {
        network1.clone()
    };
    Ok((network1, network2))
}
