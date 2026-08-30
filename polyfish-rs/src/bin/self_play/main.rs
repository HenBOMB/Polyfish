// The METRICS json! literal outgrew serde_json's default macro recursion.
#![recursion_limit = "256"]

use polyfish::ai::brain::{SearchBackend, SearchBackendArg};
use polyfish::ai::macro_agent::{MacroLeaf, MacroParams};
use polyfish::ai::eval_backend::{self, EvalBackendKind, PlayerBackend};
use polyfish::ai::eval_server::EvalServerConfig;
use polyfish::replayer::ModReplay;
use serde_json::json;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

mod crutches;
mod result;
mod runner;
mod summary;
use summary::print_run_summary;
use runner::{load_networks, report_eval_stats, run_games};
mod labels;
mod stats;
mod tempo;
use stats::is_net_seat;
mod traces;
mod shard;
mod game;
mod cli;
mod dataset;
use dataset::ShardBuffers;
use cli::Args;
mod dumps;
use polyfish::eval_seeds::{SeedEntry, load_seed_file, parse_tribe, CORE_TRIBES};

/// Console verbosity for long self-play runs.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ProgressMode {
    /// Full play-by-play (move every 10 steps, start/finish per game).
    Full,
    /// No move-by-move noise; up to 5 turn-milestone lines per game.
    Periodic,
    /// Silent during games; caller reports ~every 20% on game finish.
    SampledFinish,
}

impl ProgressMode {
    fn from_num_games(num_games: usize) -> Self {
        if num_games >= 64 {
            Self::SampledFinish
        } else if num_games > 32 {
            Self::Periodic
        } else {
            Self::Full
        }
    }
}











fn main() -> anyhow::Result<()> {
    use clap::Parser;

    let start_time = Instant::now();

    let mut args = Args::parse();

    // --dump-games-dir is the "give me everything for this game" flag, but
    // its own dump (.replay.json/.decisions.json) doesn't carry the macro
    // ballot or per-ply candidate scoring for macro-mcts games -- those are
    // separate opt-in mechanisms (--dump-macro-policy, POLYFISH_PLY_TRACE)
    // that silently stay off unless remembered explicitly, leaving
    // .decisions.json's `trace` field null with no companion data anywhere.
    // Default both to the same directory whenever --dump-games-dir is set
    // and they weren't already pointed elsewhere.
    if let Some(dir) = &args.dump_games_dir {
        if args.dump_macro_policy.is_none() {
            args.dump_macro_policy = Some(dir.clone());
        }
        if std::env::var("POLYFISH_PLY_TRACE").is_err() {
            // Safety: still single-threaded here, before any actor threads
            // or the OnceLock in `ply_trace_path()` are touched.
            unsafe {
                std::env::set_var("POLYFISH_PLY_TRACE", format!("{dir}/ply_trace.jsonl"));
            }
        }
    }

    if args.anchor_frac > 0.0 && args.opponent.is_some() {
        anyhow::bail!("--anchor-frac and --opponent are mutually exclusive");
    }
    if !(0.0..=1.0).contains(&args.anchor_frac) {
        anyhow::bail!("--anchor-frac must be in [0, 1]");
    }
    if let Some(t) = args.value_trust {
        if !(0.0..=1.0).contains(&t) {
            anyhow::bail!("--value-trust must be in [0, 1]");
        }
    }
    if !(0.0..=1.0).contains(&args.td_lambda) {
        anyhow::bail!("--td-lambda must be in [0, 1]");
    }
    if args.goal_w_tree != 0.0 && !args.goal_channels {
        anyhow::bail!("--goal-w-tree requires --goal-channels (no goal is set without them)");
    }
    let is_macro_backend = matches!(args.search_backend, SearchBackendArg::MacroMcts);
    // The macro tree commits a directive during think; without goal channels
    // the recorded features carry ZERO goal planes, so the data says nothing
    // about what the teacher was pursuing. Silent before this guard.
    if is_macro_backend && !args.goal_channels {
        anyhow::bail!(
            "--search-backend macro-mcts requires --goal-channels (the tree's committed \
             directive would otherwise be dropped from the recorded features)"
        );
    }
    if !is_macro_backend
        && (args.macro_leaf != MacroLeaf::Heuristic
            || args.macro_sims != 32
            || args.macro_k != 4
            || args.macro_lambda != 1.0
            || args.macro_rollout_lambda.is_some()
            || args.macro_shape_w != 0.0
            || args.macro_root_prior_w != 0.0)
    {
        anyhow::bail!("--macro-* flags require --search-backend macro-mcts");
    }
    let macro_params = MacroParams {
        k: args.macro_k,
        leaf: args.macro_leaf,
        lambda: args.macro_lambda,
        rollout_lambda: args.macro_rollout_lambda.unwrap_or(args.macro_lambda),
        sims: args.macro_sims,
        shape_w: args.macro_shape_w,
        root_prior_w: args.macro_root_prior_w,
        ..MacroParams::default()
    };

    // Default Metal op-flush cadence to 1000 for better GPU efficiency on Metal
    if std::env::var("CANDLE_METAL_COMPUTE_PER_BUFFER").is_err() {
        // This is safe because it runs at the top of `main`, so no concurrent writes.
        unsafe {
            std::env::set_var("CANDLE_METAL_COMPUTE_PER_BUFFER", "1000");
        }
    }
    let metal_compute_per_buffer =
        std::env::var("CANDLE_METAL_COMPUTE_PER_BUFFER").unwrap_or_else(|_| "1000".to_string());

    let backend = match args.search_backend {
        SearchBackendArg::Zero => SearchBackend::Zero,
        SearchBackendArg::Gumbel => SearchBackend::Gumbel { k: args.gumbel_k },
        SearchBackendArg::Heuristic => SearchBackend::Heuristic,
        SearchBackendArg::Greedy => SearchBackend::Greedy,
        // Stage 3: macro-mcts generates training games (behavior-cloning
        // policy targets + on-distribution value labels for the macro leaf).
        SearchBackendArg::MacroMcts => SearchBackend::MacroMcts,
        // EXP_ELO_032: arena-only bootstrap backends.
        SearchBackendArg::MacroScript | SearchBackendArg::MacroLookahead => {
            anyhow::bail!("macro-script/lookahead are arena-only (EXP_ELO_032)")
        }
    };

    let device = eval_backend::select_device()?;

    // Resolve the eval backend up front (explicit --eval-backend, else auto:
    // metal when compiled in, else tch when compiled in, else candle) — the
    // network load below needs it to decide whether player 2 gets an
    // isolated device (see `load_networks`'s doc comment).
    let eval_backend_kind = eval_backend::resolve_eval_backend_kind(&args.eval_backend)?;
    let eval_servers = eval_backend::resolve_eval_servers(eval_backend_kind, args.eval_servers)?;

    // Load models (P1, and P2 defaulting to P1 when no opponent is given)
    let (network1, network2) = load_networks(&device, args.opponent.as_deref(), eval_backend_kind)?;

    let base_seed = if args.base_seed != 0 {
        args.base_seed
    } else {
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs()
    };

    let seed_entries: Option<Vec<SeedEntry>> = args
        .seed_file
        .as_ref()
        .map(|path| load_seed_file(path, args.num_games, parse_tribe))
        .transpose()?;
    let seed_list: Option<Vec<i64>> = seed_entries
        .as_ref()
        .map(|entries| entries.iter().map(|e| e.seed).collect());

    // Pool of tribes to draw from when tribe1/tribe2 aren't pinned via CLI args
    // or a --seed-file entry. Each game in this run independently samples its
    // own pair from this pool (see `pick_tribes`/`resolve_tribes`), rather
    // than the whole run sharing one fixed pair.
    // The v1 training pool; special tribes are deliberately excluded.
    let all_tribes = CORE_TRIBES.to_vec();

    // Game generation: a pool of actor threads pulls game indices off a
    // shared counter. Each actor blocks (parks, no CPU) while awaiting an
    // eval-server reply, so oversubscribing actors past core count is fine —
    // RAM (a Game clone + MCTS tree per actor) is the real ceiling, not CPU.
    // The eval server owns the sole network/device and coalesces requests
    // from every actor into batched forward_t calls (see ai/eval_server.rs
    // for the Metal cross-thread-tensor invariant this design preserves).
    let games_start = Instant::now();

    // Each shard sees ~1/N of the working set (hash-routed), so dividing the
    // per-shard cache by N keeps total resident cache ~constant while
    // preserving the hit rate (cache / working-set ratio is unchanged).
    let per_shard_cache = eval_backend::split_cache_capacity(args.cache_cap, eval_servers);
    let eval_config = EvalServerConfig {
        max_batch: args.max_batch,
        coalesce_timeout: std::time::Duration::from_micros(args.coalesce_timeout_us),
        cache_capacity: per_shard_cache,
        pipeline_workers: args.eval_workers,
    };
    let p1_path = "model.safetensors";
    let p2_path = args.opponent.as_deref().unwrap_or("model.safetensors");
    let has_opponent = args.opponent.is_some();

    // Spawn the shards. Each EvalServer owns its inference thread + device
    // context; the handles are collected into a ShardedEvalHandle that
    // routes leaves by hash so each shard owns its own LRU cache. No
    // opponent => player 2 shares player 1's shard set (one set of
    // inference threads for the same weights).
    let (p1_servers, p2_servers, eval1, eval2) = eval_backend::build_two_player_evaluators(
        eval_backend_kind,
        eval_servers,
        eval_config,
        PlayerBackend {
            model_path: p1_path,
            candle_net: &network1,
        },
        has_opponent.then(|| PlayerBackend {
            model_path: p2_path,
            candle_net: &network2,
        }),
    );

    let num_actors = if args.actors > 0 {
        args.actors
    } else {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    };
    let progress_mode = ProgressMode::from_num_games(args.num_games);

    let tribe_label = match (&args.tribe1, &args.tribe2) {
        (Some(t1), Some(t2)) => format!("{t1} vs {t2}"),
        (Some(t1), None) => format!("{t1} vs random"),
        (None, Some(t2)) => format!("random vs {t2}"),
        (None, None) => "random".to_string(),
    };
    let match_label = match &args.opponent {
        Some(opp) => format!("league vs {opp}"),
        None if args.anchor_frac > 0.0 => {
            format!(
                "self-play + up to {:.0}% heuristic-anchor games (decaying)",
                args.anchor_frac * 100.0
            )
        }
        None => "self-play".to_string(),
    };
    let backend_label = match eval_backend_kind {
        EvalBackendKind::Tch => "tch (libtorch/MPS)",
        EvalBackendKind::Metal => "metal (MPSGraph)",
        EvalBackendKind::Candle => "candle",
    };
    let search_label = match backend {
        SearchBackend::Zero => "Zero MCTS".to_string(),
        SearchBackend::Gumbel { k } => format!("Gumbel k={k}"),
        SearchBackend::Heuristic => "Heuristic MCTS (no NN)".to_string(),
        SearchBackend::Greedy => "Greedy heuristic (no NN, no search)".to_string(),
        SearchBackend::MacroScript => "Macro script (EXP_ELO_032)".to_string(),
        SearchBackend::MacroLookahead => "Macro lookahead (EXP_ELO_032)".to_string(),
        SearchBackend::MacroMcts => "Macro MCTS (EXP_ELO_033)".to_string(),
    };
    println!(
        "[selfplay] {match_label}: {} games, {} mcts-iters, {search_label}, tribes {tribe_label} | eval {backend_label} | {eval_servers} shard(s) cache={per_shard_cache:?} workers={} | {num_actors} actors max_batch={} coalesce_us={} leaf_batch={:?} | device {:?} (CANDLE_METAL_COMPUTE_PER_BUFFER={metal_compute_per_buffer})",
        args.num_games,
        args.mcts_iters,
        args.eval_workers,
        args.max_batch,
        args.coalesce_timeout_us,
        args.leaf_batch,
        device,
    );

    let results = run_games(
        num_actors, &args, &network1, &network2, &eval1, &eval2, &all_tribes,
        &seed_list, &seed_entries, base_seed, backend, macro_params, progress_mode,
        has_opponent,
    );
    let games_duration = games_start.elapsed();

    report_eval_stats(games_duration, &results, &p1_servers, p2_servers.as_ref());


    let mut max_score = 0;
    let mut best_recap: Option<ModReplay> = None;
    let mut total_moves = 0; // both seats — throughput + sim-ratio denominators
    let mut total_net_moves = 0; // net-seat plies — the avg_moves behavior chart

    let mut p1_total = 0;
    let mut p2_total = 0;
    let mut p1_count = 0;
    let mut p2_count = 0;

    let mut total_captures = 0;
    let mut total_cap_ruins = 0;
    let mut total_cap_villages = 0;
    let mut total_cap_cities = 0;
    let mut total_cap_capitals = 0;
    let mut total_harvests = 0;
    let mut total_builds = 0;
    let mut total_research = 0;
    let mut total_attacks = 0;
    let mut total_abilities = 0;
    let mut total_revealed_tiles: i64 = 0;
    let mut total_captured_tiles: i64 = 0;
    let mut hub_totals: HashMap<polyfish::types::StructureType, (u32, i64, u32, u32)> =
        HashMap::new();
    // per type: (games, chosen_sum, best_sum, optimal_games, rank_pct_sum, cand_sum)
    let mut site_totals: HashMap<polyfish::types::StructureType, (u32, i64, i64, u32, f64, u64, i64, i64, u32)> =
        HashMap::new();
    let mut total_t2c = [0.0f64; 6]; // villages p50/p80/all, ruins p50/p80/all
    let (mut first_cap_seats, mut first_cap_captured) = (0u32, 0u32);
    let mut first_cap_turn_sum = 0.0f64;
    let mut first_cap_censored_sum = 0.0f64;
    // Contested anchor games: an embedded per-iteration strength peek vs the
    // Greedy anchor (n is small — ~anchor_frac * num_games — so ±1/sqrt(n)).
    let (mut anchor_games, mut anchor_net_wins) = (0u32, 0u32);
    let mut spt_sums: HashMap<i32, f64> = HashMap::new();
    let mut spt_counts: HashMap<i32, u32> = HashMap::new();
    let mut worth_sums: HashMap<i32, f64> = HashMap::new();
    let mut army_per_city_sums: HashMap<i32, f64> = HashMap::new();

    let mut total_moves_by_turn: HashMap<i32, HashMap<polyfish::types::MoveType, usize>> =
        HashMap::new();

    /// Per-role tempo accumulator across all player-games.
    #[derive(Default)]
    struct TempoAgg {
        /// turn -> ([cities, city_levels, spt, units, army_stars, revealed,
        /// techs, kills, trained_cum, lost_cum, stars_lost_cum] sums, sample count)
        by_turn: HashMap<i32, ([f64; 11], u32)>,
        trained: i64,
        granted: i64,
        lost: i64,
        giants: i64,
        stars_lost: i64,
        kills: i64,
        /// Σ star-cost of units still alive at game end — the "held" counterpart
        /// to `stars_lost`, on the same end-of-game time base.
        army_stars_end: i64,
        player_games: u32,
        /// cities >= 2/3/4: (reached count, turn sum over reached)
        reach: [(u32, f64); 3],
    }
    let mut tempo_aggs: HashMap<&'static str, TempoAgg> = HashMap::new();


    let run_ts = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let mut shard = ShardBuffers::new(&args, run_ts);

    for result in results {
        total_moves += result.moves;
        total_net_moves += result.net_moves;
        total_revealed_tiles += result.revealed_tiles as i64;
        total_captured_tiles += result.captured_tiles as i64;
        for (k, (got, best, n_better, n_cands, cg, cb)) in &result.first_hub_rank {
            let e = site_totals.entry(*k).or_insert((0, 0, 0, 0, 0.0, 0, 0, 0, 0));
            e.0 += 1;
            e.1 += got;
            e.2 += best;
            e.3 += u32::from(got >= best);
            // 1.0 = no legal site would have ended with more partners.
            e.4 += 1.0 - f64::from(*n_better) / f64::from((*n_cands).max(1));
            e.5 += u64::from(*n_cands);
            e.6 += cg;
            e.7 += cb;
            e.8 += u32::from(cg >= cb);
        }
        for (k, (n, sum, starved, lost)) in &result.hub_levels {
            let e = hub_totals.entry(*k).or_insert((0, 0, 0, 0));
            e.0 += n;
            e.1 += sum;
            e.2 += starved;
            e.3 += lost;
        }
        for (&turn, &spt) in &result.spt_at_turn {
            *spt_sums.entry(turn).or_default() += spt as f64;
            *spt_counts.entry(turn).or_default() += 1;
        }
        // Shares spt_counts as its denominator — both are written by the same
        // milestone recorder, so a turn present in one is present in the other.
        for (&turn, &(worth, per_city)) in &result.army_ratios_at_turn {
            *worth_sums.entry(turn).or_default() += worth as f64;
            *army_per_city_sums.entry(turn).or_default() += per_city as f64;
        }
        for (acc, v) in total_t2c.iter_mut().zip([
            result.villages_t2c_p50,
            result.villages_t2c_p80,
            result.villages_t2c_all,
            result.ruins_t2c_p50,
            result.ruins_t2c_p80,
            result.ruins_t2c_all,
        ]) {
            *acc += v as f64;
        }
        first_cap_seats += result.villages_first_seats;
        first_cap_captured += result.villages_first_captured;
        first_cap_turn_sum += result.villages_first_turn_sum;
        first_cap_censored_sum += result.villages_first_censored_sum;
        if result.roles.contains(&"anchor") {
            anchor_games += 1;
            let winner_seat = (result.winner_id - 1) as usize;
            if winner_seat < 2 && result.roles[winner_seat] == "model_vs_anchor" {
                anchor_net_wins += 1;
            }
        }
        // Net-only: mirror games count both seats (both are net); anchor/league
        // games exclude the non-net (Greedy/opponent) seat, so the score metrics
        // reflect the net's play, not the opponent's.
        let mut game_net_max = 0;
        for (id, score) in &result.scores {
            if !is_net_seat(result.roles, *id) {
                continue;
            }
            game_net_max = game_net_max.max(*score);
            if *id == 1 {
                p1_total += score;
                p1_count += 1;
            } else if *id == 2 {
                p2_total += score;
                p2_count += 1;
            }
        }
        // Best net seat rather than `winner_score`: an anchor/league opponent
        // win would otherwise set the reported max and get its replay saved.
        if game_net_max > max_score {
            max_score = game_net_max;
            best_recap = Some(result.recap.clone());
        }

        total_captures += result
            .action_counts
            .get(&polyfish::types::MoveType::Capture)
            .copied()
            .unwrap_or(0);
        total_cap_ruins += result.cap_ruins;
        total_cap_villages += result.cap_villages;
        total_cap_cities += result.cap_cities;
        total_cap_capitals += result.cap_capitals;
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
        total_abilities += result
            .action_counts
            .get(&polyfish::types::MoveType::Ability)
            .copied()
            .unwrap_or(0);

        for (turn, counts) in &result.moves_by_turn {
            let entry = total_moves_by_turn.entry(*turn).or_default();
            for (mt, c) in counts {
                *entry.entry(*mt).or_insert(0) += c;
            }
        }

        for (&pid, track) in &result.tempo {
            let seat = (pid - 1) as usize;
            if seat >= 2 {
                continue;
            }
            let agg = tempo_aggs.entry(result.roles[seat]).or_default();
            agg.player_games += 1;
            agg.trained += track.units_trained as i64;
            agg.granted += track.units_granted as i64;
            agg.lost += track.units_lost as i64;
            agg.giants += track.giants_made as i64;
            agg.stars_lost += track.army_stars_lost as i64;
            // End-of-game state comes from the final forced sample, so games
            // that ended early are still counted at their true final turn.
            if let Some(last) = track.samples.last() {
                agg.kills += last.kills as i64;
                agg.army_stars_end += last.army_stars as i64;
            }
            for s in &track.samples {
                let (sums, n) = agg.by_turn.entry(s.turn).or_default();
                for (acc, v) in sums.iter_mut().zip([
                    s.cities,
                    s.city_levels,
                    s.spt,
                    s.units,
                    s.army_stars,
                    s.revealed,
                    s.techs,
                    s.kills,
                    s.trained_cum,
                    s.lost_cum,
                    s.stars_lost_cum,
                ]) {
                    *acc += v as f64;
                }
                *n += 1;
            }
            for (slot, target) in agg.reach.iter_mut().zip([2, 3, 4]) {
                if let Some(s) = track.samples.iter().find(|s| s.cities >= target) {
                    slot.0 += 1;
                    slot.1 += s.turn as f64;
                }
            }
        }

        // Backpropagate value
        // Domination: Win/Loss is the primary signal.
        // The winner gets +1.0, loser gets -1.0.
        // If timeout, use score differential as a softer signal.
        shard.push_game(result, &args);

        shard.maybe_flush(&device)?;
    }

    let mut net_games = 0u32;
    let (mut net_trained, mut net_granted, mut net_lost, mut net_giants) = (0i64, 0i64, 0i64, 0i64);
    let mut net_kills = 0i64;
    let mut net_reach = [(0u32, 0.0f64); 3];
    for role in ["model", "model_vs_anchor"] {
        if let Some(a) = tempo_aggs.get(role) {
            net_games += a.player_games;
            net_trained += a.trained;
            net_granted += a.granted;
            net_lost += a.lost;
            net_giants += a.giants;
            net_kills += a.kills;
            for (dst, src) in net_reach.iter_mut().zip(a.reach.iter()) {
                dst.0 += src.0;
                dst.1 += src.1;
            }
        }
    }
    let per_net_game = |x: i64| {
        if net_games > 0 {
            x as f64 / f64::from(net_games)
        } else {
            0.0
        }
    };

    // Print Average Metrics. avg_score is net-only (see the score loop): the
    // mean score over net seats across games, so anchor/league games don't
    // blend the opponent's score into the net's performance chart.
    let net_score_count = p1_count + p2_count;
    let avg_score = if net_score_count > 0 {
        (p1_total + p2_total) as f32 / net_score_count as f32
    } else {
        0.0
    };
    let avr_moves = per_net_game(total_net_moves as i64) as f32;
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

    // Per net PLAYER-GAME, not per game: these counters only accrue on net
    // seats, and a mirror game supplies two of them to an anchor game's one.
    // Dividing by games made the whole family drift as anchor_frac decayed.
    let avg_captures = per_net_game(total_captures as i64) as f32;
    let avg_cap_ruins = per_net_game(total_cap_ruins as i64) as f32;
    let avg_cap_villages = per_net_game(total_cap_villages as i64) as f32;
    let avg_cap_cities = per_net_game(total_cap_cities as i64) as f32;
    let avg_cap_capitals = per_net_game(total_cap_capitals as i64) as f32;
    let avg_harvests = per_net_game(total_harvests as i64) as f32;
    let avg_builds = per_net_game(total_builds as i64) as f32;
    let avg_research = per_net_game(total_research as i64) as f32;
    let avg_attacks = per_net_game(total_attacks as i64) as f32;
    let avg_abilities = per_net_game(total_abilities as i64) as f32;
    let avg_revealed_tiles = per_net_game(total_revealed_tiles) as f32;
    let avg_captured_tiles = per_net_game(total_captured_tiles) as f32;

    // -1.0 when the net built no hubs at all: 0.0 is a legal level.
    let (hub_n, hub_sum, hub_starved, hub_lost) = hub_totals.values().fold(
        (0u32, 0i64, 0u32, 0u32),
        |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3),
    );
    let avg_hub_level = if hub_n > 0 {
        hub_sum as f32 / hub_n as f32
    } else {
        -1.0
    };
    let hub_starved_frac = if hub_n > 0 {
        hub_starved as f32 / hub_n as f32
    } else {
        -1.0
    };
    let first_hub_site: serde_json::Value = site_totals
        .iter()
        .map(|(k, (games, got, best, optimal, rank_pct, cands, cg, cb, ceil_opt))| {
            let g = f64::from(*games).max(1.0);
            (
                format!("{k:?}"),
                serde_json::json!({
                    "games": games,
                    "chosen_partners": *got as f64 / g,
                    "best_available_partners": *best as f64 / g,
                    "optimal_frac": f64::from(*optimal) / g,
                    "mean_rank_pct": rank_pct / g,
                    "sites_available": *cands as f64 / g,
                    "ceiling_chosen": *cg as f64 / g,
                    "ceiling_best_available": *cb as f64 / g,
                    "ceiling_optimal_frac": f64::from(*ceil_opt) / g,
                }),
            )
        })
        .collect::<serde_json::Map<_, _>>()
        .into();

    let hub_lost_frac = if hub_n > 0 {
        hub_lost as f32 / hub_n as f32
    } else {
        -1.0
    };
    let avg_hubs_built = per_net_game(i64::from(hub_n)) as f32;
    let hub_levels_by_type: serde_json::Value = hub_totals
        .iter()
        .map(|(k, (n, sum, starved, lost))| {
            (
                format!("{k:?}"),
                serde_json::json!({
                    "built": n,
                    "mean_level": *sum as f32 / *n as f32,
                    "starved_frac": *starved as f32 / *n as f32,
                    "lost_frac": *lost as f32 / *n as f32,
                }),
            )
        })
        .collect::<serde_json::Map<_, _>>()
        .into();
    let avg_kills = per_net_game(net_kills) as f32;

    let avg_spt_at = |turn: i32| -> f32 {
        let c = spt_counts.get(&turn).copied().unwrap_or(0);
        if c == 0 {
            0.0
        } else {
            (spt_sums.get(&turn).copied().unwrap_or(0.0) / c as f64) as f32
        }
    };
    // -1.0 (not 0.0) when a turn was never reached: 0 is a legal value for both
    // ratios, so a sentinel is the only way to distinguish "no army" from "no
    // data" downstream.
    let avg_ratio_at = |sums: &HashMap<i32, f64>, turn: i32| -> f32 {
        let c = spt_counts.get(&turn).copied().unwrap_or(0);
        if c == 0 {
            -1.0
        } else {
            (sums.get(&turn).copied().unwrap_or(0.0) / c as f64) as f32
        }
    };

    // "typical move by turn N" chart data: {"<turn>": {"<MoveType>": count, ...}, ...}
    let moves_by_turn = {
        let mut turns_sorted: Vec<&i32> = total_moves_by_turn.keys().collect();
        turns_sorted.sort();
        let mut turn_map = serde_json::Map::new();
        for turn in turns_sorted {
            let mut counts_map = serde_json::Map::new();
            for (mt, c) in &total_moves_by_turn[turn] {
                counts_map.insert(format!("{mt:?}"), serde_json::Value::from(*c));
            }
            turn_map.insert(turn.to_string(), serde_json::Value::Object(counts_map));
        }
        serde_json::Value::Object(turn_map)
    };

    let shard_files = shard.finish(&device)?;
    // METRICS carries the first shard (the value-distribution reader wants a
    // ~64-game sample, not every file); everything else globs the _p* stem.
    let games_file = shard_files.first().cloned().unwrap_or_default();

    // Save BEST game as replay
    if let Some(recap) = best_recap {
        let replay_filename = format!(
            "replays/high_scores/best_game_score_{}_{}.json",
            max_score, run_ts
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

    // Tempo curves per role + net-seat scalar aggregates ("model" mirror
    // seats + "model_vs_anchor" contested seats combined; "anchor" is the
    // Greedy reference curve and stays out of the scalars).
    let tempo_by_turn = {
        let mut roles_map = serde_json::Map::new();
        for (role, agg) in &tempo_aggs {
            let mut turn_map = serde_json::Map::new();
            let mut turns: Vec<i32> = agg.by_turn.keys().copied().collect();
            turns.sort_unstable();
            for t in turns {
                let (sums, n) = &agg.by_turn[&t];
                let nf = f64::from(*n).max(1.0);
                let mut o = serde_json::Map::new();
                for (name, v) in [
                    "cities",
                    "city_levels",
                    "spt",
                    "units",
                    "army_stars",
                    "revealed",
                    "techs",
                    "kills",
                    "trained_cum",
                    "lost_cum",
                    "stars_lost_cum",
                ]
                .iter()
                .zip(sums.iter())
                {
                    o.insert((*name).to_string(), serde_json::Value::from(v / nf));
                }
                o.insert("n".to_string(), serde_json::Value::from(*n));
                turn_map.insert(t.to_string(), serde_json::Value::Object(o));
            }
            // Unbiased per-player-game totals (the last-turn-key cums under-
            // count games that ended early). "_totals" is non-numeric, so
            // turn-key consumers must filter it.
            let pg = f64::from(agg.player_games).max(1.0);
            let mut totals = serde_json::Map::new();
            for (name, v) in [
                ("trained", agg.trained),
                ("granted", agg.granted),
                ("lost", agg.lost),
                ("giants", agg.giants),
                ("stars_lost", agg.stars_lost),
                ("kills", agg.kills),
                ("army_stars_end", agg.army_stars_end),
            ] {
                totals.insert(name.to_string(), serde_json::Value::from(v as f64 / pg));
            }
            totals.insert(
                "n_games".to_string(),
                serde_json::Value::from(agg.player_games),
            );
            turn_map.insert("_totals".to_string(), serde_json::Value::Object(totals));
            roles_map.insert((*role).to_string(), serde_json::Value::Object(turn_map));
        }
        serde_json::Value::Object(roles_map)
    };
    let reach_rate = |i: usize| {
        if net_games > 0 {
            f64::from(net_reach[i].0) / f64::from(net_games)
        } else {
            0.0
        }
    };
    let reach_turn = |i: usize| {
        if net_reach[i].0 > 0 {
            net_reach[i].1 / f64::from(net_reach[i].0)
        } else {
            -1.0
        }
    };

    let metrics = json!({
        "num_games": args.num_games,
        "avg_score": avg_score,
        "max_score": max_score,
        "avg_moves": avr_moves,
        "p1_avg": p1_avg,
        "p2_avg": p2_avg,
        "avg_captures": avg_captures,
        "avg_cap_ruins": avg_cap_ruins,
        "avg_cap_villages": avg_cap_villages,
        "avg_cap_cities": avg_cap_cities,
        "avg_cap_capitals": avg_cap_capitals,
        "avg_harvests": avg_harvests,
        "avg_builds": avg_builds,
        "avg_research": avg_research,
        "avg_attacks": avg_attacks,
        "avg_abilities": avg_abilities,
        "avg_kills": avg_kills,
        "avg_revealed_tiles": avg_revealed_tiles,
        "avg_captured_tiles": avg_captured_tiles,
        "first_hub_site": first_hub_site,
        "avg_hub_level": avg_hub_level,
        "avg_hubs_built": avg_hubs_built,
        "hub_starved_frac": hub_starved_frac,
        "hub_lost_frac": hub_lost_frac,
        "hub_levels_by_type": hub_levels_by_type,
        "avg_spt_t0": avg_spt_at(0),
        "avg_spt_t5": avg_spt_at(5),
        "avg_spt_t10": avg_spt_at(10),
        "avg_spt_t15": avg_spt_at(15),
        "avg_spt_t20": avg_spt_at(20),
        "avg_spt_t25": avg_spt_at(25),
        "avg_spt_t30": avg_spt_at(30),
        "unit_worth_t15": avg_ratio_at(&worth_sums, 15),
        "unit_worth_t25": avg_ratio_at(&worth_sums, 25),
        "army_stars_per_city_t15": avg_ratio_at(&army_per_city_sums, 15),
        "army_stars_per_city_t25": avg_ratio_at(&army_per_city_sums, 25),
        // Per NET SEAT, not per game — a mirror game contributes two seats and
        // an anchor game one, so a games denominator blended two different
        // per-seat probabilities and drifted with anchor_frac.
        "villages_t2c_first": if first_cap_seats > 0 {
            (first_cap_censored_sum / f64::from(first_cap_seats)) as f32
        } else {
            -1.0
        },
        "villages_first_rate": if first_cap_seats > 0 {
            (f64::from(first_cap_captured) / f64::from(first_cap_seats)) as f32
        } else {
            0.0
        },
        "villages_t2c_first_cond": if first_cap_captured > 0 {
            (first_cap_turn_sum / f64::from(first_cap_captured)) as f32
        } else {
            -1.0
        },
        "tribes": format!(
            "{}+{}",
            args.tribe1.as_deref().unwrap_or("random"),
            args.tribe2.as_deref().unwrap_or("random")
        ),
        "villages_t2c_p50": (total_t2c[0] / args.num_games as f64) as f32,
        "villages_t2c_p80": (total_t2c[1] / args.num_games as f64) as f32,
        "villages_t2c_all": (total_t2c[2] / args.num_games as f64) as f32,
        "ruins_t2c_p50": (total_t2c[3] / args.num_games as f64) as f32,
        "ruins_t2c_p80": (total_t2c[4] / args.num_games as f64) as f32,
        "ruins_t2c_all": (total_t2c[5] / args.num_games as f64) as f32,
        "games_file": games_file,
        "moves_by_turn": moves_by_turn,
        "avg_units_spawned": per_net_game(net_trained),
        "avg_units_granted": per_net_game(net_granted),
        "avg_units_lost": per_net_game(net_lost),
        "avg_giants_made": per_net_game(net_giants),
        "t2c_2nd_rate": reach_rate(0),
        "t2c_2nd_turn": reach_turn(0),
        "t2c_3rd_rate": reach_rate(1),
        "t2c_3rd_turn": reach_turn(1),
        "t2c_4th_rate": reach_rate(2),
        "t2c_4th_turn": reach_turn(2),
        "anchor_games": anchor_games,
        "anchor_net_wr": if anchor_games > 0 {
            f64::from(anchor_net_wins) / f64::from(anchor_games)
        } else {
            -1.0
        },
        "tempo_by_turn": tempo_by_turn,
        "gate_blocks": polyfish::ai::gumbel_mcts::gate_stats::snapshot(),
    });
    std::fs::write(
        ".last_self_play_metrics.json",
        serde_json::to_string(&metrics)?,
    )?;

    print_run_summary(start_time, games_duration, total_moves);

    // Deterministic teardown. Drop the evaluator handles first — these hold the
    // only remaining request-channel senders, so dropping them makes each eval
    // thread's `recv` error out and return, which drops its inference backend
    // (and any MPS/device tensors). Then join the threads so that drop finishes
    // *before* the process starts static/atexit teardown. Without this the
    // detached eval thread races libtorch's atexit mutex destruction and the
    // process aborts with "recursive_mutex lock failed: Invalid argument".
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

