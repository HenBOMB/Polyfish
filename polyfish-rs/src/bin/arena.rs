use clap::Parser;
use polyfish::ai::brain::{SearchAgent, SearchBackend, SearchBackendArg, make_search_agent};
use polyfish::ai::eval_backend::{self, EvalBackendKind, PlayerBackend};
use polyfish::ai::eval_server::{EvalServerConfig, Evaluator};
use polyfish::ai::network::PolyZeroNet;
use polyfish::game::Game;
use polyfish::mapgen::{MapGenSettings, generate};
use polyfish::types::{MapSize, MapType, ModeType, TribeType};
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
    /// Inherits MCTS_ITERS so a reading defaults to the budget the model is
    /// trained at; win rates are only comparable at a fixed (mcts, gumbel_k).
    #[arg(long, env = "MCTS_ITERS", default_value_t = 64)]
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
    #[arg(long, env = "GUMBEL_K", default_value_t = 16)]
    gumbel_k: usize,

    /// Max game turns. Higher = more decisive but slower.
    #[arg(long, default_value_t = 30)]
    max_turns: i32,

    /// Game mode (2 = Domination, the training mode). The mode is a net
    /// input feature and steers the heuristic evaluator — match training.
    #[arg(long, default_value_t = 2)]
    gamemode: u8,

    /// Number of concurrent match-worker threads (games in flight).
    /// Independent of CPU core count — workers park while awaiting
    /// eval-server replies (same eval-serving design as self_play), so
    /// oversubscribing past core count is fine and is what produces the fat
    /// coalesced batches that make the fast eval backends fast. 0 = auto:
    /// 4x core count, clamped to the total game count (2 * --games) — sized
    /// for a single EXP-10-style 32-64 seed reading; raise it by hand for
    /// much larger batches.
    #[arg(long, default_value_t = 0)]
    concurrency: usize,

    /// Deprecated alias for --concurrency (previously capped rayon Metal
    /// devices; eval no longer runs on match-worker threads, so that
    /// rationale no longer applies). Kept for old shell history only.
    #[arg(long)]
    workers: Option<usize>,

    /// NN inference backend: "candle" (Metal/CUDA/CPU), "tch" (libtorch/MPS,
    /// ~19x faster on Metal, requires --features tch-eval), or "metal"
    /// (MPSGraph, bypasses libtorch's serial MPS dispatch queue, requires
    /// --features metal-eval — see metal_network.rs). Empty = auto: "metal"
    /// if the metal-eval feature is compiled in, else "tch" if tch-eval is,
    /// else "candle".
    #[arg(long, default_value = "")]
    eval_backend: String,

    /// Number of concurrent eval-server threads (shards) per config. Each
    /// owns its own weights copy + LRU cache. 0 = auto (3 on metal, 1 on
    /// tch/candle). See self_play's --eval-servers doc for the measured
    /// rationale (tch serializes across shards; candle rejects >1).
    #[arg(long, default_value_t = 0)]
    eval_servers: usize,

    /// Metal backend only: pipelined GPU worker threads per eval server.
    /// Ignored by candle/tch.
    #[arg(long, default_value_t = 2)]
    eval_workers: usize,

    /// Eval-server batch cap: max leaves coalesced into one forward_t.
    #[arg(long, default_value_t = 256)]
    max_batch: usize,

    /// Eval-server coalescing flush timeout in microseconds.
    #[arg(long, default_value_t = 1000)]
    coalesce_timeout_us: u64,

    /// Eval-cache LRU capacity (number of cached NN evaluations), split
    /// across shards. 0 disables the cache.
    #[arg(long, default_value_t = 524288)]
    cache_cap: usize,

    /// Per-game virtual-loss mini-batch size (leaves coalesced per NN call
    /// within a single game's search tree). None keeps each MCTS agent's own
    /// default (24). Once cross-match coalescing exists (this eval-server
    /// setup), self_play measured a larger value as a net throughput loss —
    /// see self_play's --leaf-batch doc. Changing this from the default
    /// alters Gumbel move selection, same as --eval-backend; sweep before
    /// trusting a strength-gauge run against it.
    #[arg(long)]
    leaf_batch: Option<usize>,

    /// Write per-turn stat samples (score/SPT/stars/cities/units/unit-cost/
    /// techs per config) as one JSON per game into this directory — the
    /// EXP_ELO_001 loss-autopsy instrument.
    #[arg(long)]
    dump_stats_dir: Option<String>,

    /// Write one start-of-turn ground-truth snapshot per turn (both players'
    /// cities/units + the model player's FOW-visible neutral villages) as one
    /// JSONL file per game into this directory — the vs-Greedy 3rd-city
    /// pursuit instrument (config1 = the model/gumbel seat).
    #[arg(long)]
    dump_turn_states: Option<String>,

    /// EXP_ELO_026 oracle-macro steer for config 1 (gumbel backend only):
    /// while it holds <3 cities, focus the pursuit channel on one sticky
    /// FOW-visible neutral village (nearest to its units).
    #[arg(long, default_value_t = false)]
    macro_commit: bool,

    /// EXP_ELO_026 oracle-macro steer for config 1 (gumbel backend only):
    /// while a commitment is active, drop root Research moves that would
    /// leave fewer than STAR_GATE_RESERVE stars after purchase.
    #[arg(long, default_value_t = false)]
    macro_star_gate: bool,

    /// Base map seed (seed i = base + i). 0 = derive from the wall clock.
    /// Fix it to play identical maps across separate arena runs (paired
    /// A/B arms).
    #[arg(long, default_value_t = 0)]
    base_seed: u64,

    /// EXP_ELO_028: drive config 1's goal channels with the Stage-1 scripted
    /// goal-setter (orders + stance + star gate) each ply. Gumbel backend1
    /// only. For probing goal-conditioned nets; a net trained without goal
    /// channels ignores the (zero-initialized) planes.
    #[arg(long, default_value_t = false)]
    goal_script: bool,

    /// EXP_ELO_028 Phase 1c: weight on the goal potential in config 1's
    /// in-tree edge rewards (stance/order priced shaping). Requires
    /// --goal-script. 0.0 = off.
    #[arg(long, default_value_t = 0.0)]
    goal_w_tree: f32,
}

fn load_model(path: &str, device: &candle_core::Device) -> anyhow::Result<PolyZeroNet> {
    // Load trained weights directly with from_mmaped_safetensors for correctness.
    let vs = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&[path], candle_core::DType::F32, device)?
    };
    Ok(PolyZeroNet::new(vs)?)
}

/// Append one start-of-turn ground-truth snapshot to <dir>/game<idx>.jsonl for
/// the vs-Greedy 3rd-city pursuit analysis. Both players' cities/units come
/// from ground truth every turn; villages are ground-truth neutral villages
/// (`neutral_villages`) plus the model player's FOW view
/// (`model_visible_villages`). Row-major 11x11 tile indices. One file per game,
/// so concurrent match workers never share a handle.
fn dump_turn_state(
    file: &mut std::fs::File,
    game_idx: usize,
    state: &polyfish::states::GameState,
    model_player: polyfish::states::PlayerId,
    greedy_player: polyfish::states::PlayerId,
) {
    use std::io::Write;
    // Currently-uncaptured (neutral) villages: owner only ever transitions
    // 0 -> nonzero via capture, so `owner == 0` is exactly self_play's
    // incremental open_villages set without intercepting the move loop.
    let neutral_villages: Vec<i32> = state
        .structures
        .iter()
        .filter_map(|(&idx, s)| {
            let s = s.as_ref()?;
            let neutral = s.structure_type == polyfish::types::StructureType::Village
                && state.tiles.get(&idx).map_or(false, |t| t.owner == 0);
            neutral.then_some(idx)
        })
        .collect();
    let model_visible_villages: Vec<i32> = neutral_villages
        .iter()
        .copied()
        .filter(|idx| {
            state
                .tiles
                .get(idx)
                .map_or(false, |t| t.explorers.contains(&model_player))
        })
        .collect();
    let cities_of = |pid: polyfish::states::PlayerId| -> Vec<i32> {
        state
            .tribes
            .get(&pid)
            .map(|t| t.cities.iter().map(|c| c.idx).collect())
            .unwrap_or_default()
    };
    let units_of = |pid: polyfish::states::PlayerId| -> Vec<i32> {
        state
            .tribes
            .get(&pid)
            .map(|t| t.units.iter().map(|u| u.coords.idx).collect())
            .unwrap_or_default()
    };
    let rec = serde_json::json!({
        "game": game_idx,
        "turn": state.settings.turn,
        "acting_player": state.settings.current_player_turn_id,
        "model_player": model_player,
        "model_cities": cities_of(model_player),
        "model_units": units_of(model_player),
        "model_visible_villages": model_visible_villages,
        "greedy_cities": cities_of(greedy_player),
        "greedy_units": units_of(greedy_player),
        "neutral_villages": neutral_villages,
    });
    if let Ok(s) = serde_json::to_string(&rec) {
        let _ = writeln!(file, "{s}");
    }
}

/// One per-turn sample for --dump-stats-dir; arrays index [config1, config2].
#[derive(serde::Serialize)]
struct TurnSample {
    turn: i32,
    score: [i32; 2],
    spt: [i32; 2],
    stars: [i32; 2],
    cities: [usize; 2],
    units: [usize; 2],
    unit_cost: [i32; 2],
    /// Super units alive, per tribe's own type — Giant for Imperius, Gaami for
    /// Polaris. "Did we make giants, and how many by turn N" is a headline
    /// behaviour question and unit COUNT alone cannot answer it.
    super_units: [usize; 2],
    techs: [usize; 2],
}

fn sample_turn(state: &polyfish::states::GameState, swap: bool) -> TurnSample {
    let mut s = TurnSample {
        turn: state.settings.turn,
        score: [0; 2],
        spt: [0; 2],
        stars: [0; 2],
        cities: [0; 2],
        units: [0; 2],
        unit_cost: [0; 2],
        super_units: [0; 2],
        techs: [0; 2],
    };
    for c in 0..2 {
        // Config 1 sits in the P1 seat unless swapped.
        let pid: polyfish::states::PlayerId = if (c == 0) != swap { 1 } else { 2 };
        if let Some(t) = state.tribes.get(&pid) {
            s.score[c] = t.score;
            s.spt[c] = polyfish::functions::get_tribe_spt(state, t);
            s.stars[c] = t.stars;
            s.cities[c] = t.cities.len();
            s.units[c] = t.units.len();
            s.unit_cost[c] = t
                .units
                .iter()
                .map(polyfish::rules::combat::unit_worth)
                .sum();
            let super_type = polyfish::settings::units::get_super_unit(t.tribe_type);
            s.super_units[c] = t.units.iter().filter(|u| u.unit_type == super_type).count();
            s.techs[c] = t.tech_vanilla.len();
        }
    }
    s
}

/// Per-match result, attributed to configurations (1 or 2), not seats.
struct MatchResult {
    winner_config: u8,
    /// true = config 2 sat in the P1 seat this game.
    swap: bool,
    score_config1: i32,
    score_config2: i32,
    ns_config1: u64,
    moves_config1: u64,
    ns_config2: u64,
    moves_config2: u64,
    /// Search telemetry for config1:
    /// (depth_sum, depth_count, depth_max, horizon_hits, agree, decisions).
    depth_config1: Option<(u64, u64, u32, u64, u64, u64)>,
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
    leaf_batch: Option<usize>,
    seed: i64,
    swap: bool,
    max_turns: i32,
    gamemode: u8,
    dump_stats_dir: Option<&str>,
    game_idx: usize,
    dump_turn_states: Option<&str>,
    macro_commit: bool,
    macro_star_gate: bool,
    goal_script: bool,
    goal_w_tree: f32,
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
    game.state.settings.mode =
        ModeType::from_repr(gamemode).unwrap_or(ModeType::Perfection);
    game.state.settings.max_turns = max_turns;
    game.post_load();

    // p1_config / p2_config map each seat to its configuration so timing and
    // scores attribute to the right config when sides are swapped.
    let (mut agent_p1, p1_config, mut agent_p2, p2_config) = if swap {
        (
            make_search_agent(backend2, eval2, mcts2, leaf_batch, None, None, None, None, None, None, Some(false)),
            2u8,
            make_search_agent(backend1, eval1, mcts1, leaf_batch, None, None, None, None, None, None, Some(false)),
            1u8,
        )
    } else {
        (
            make_search_agent(backend1, eval1, mcts1, leaf_batch, None, None, None, None, None, None, Some(false)),
            1u8,
            make_search_agent(backend2, eval2, mcts2, leaf_batch, None, None, None, None, None, None, Some(false)),
            2u8,
        )
    };

    // Config1 (gumbel/model) sits in the P1 seat unless swapped; the model_*
    // dump fields always describe this player, greedy_* the other, regardless
    // of who is acting.
    let model_player: polyfish::states::PlayerId = if swap { 2 } else { 1 };
    let greedy_player: polyfish::states::PlayerId = if swap { 1 } else { 2 };
    let mut turn_dump_file: Option<std::fs::File> = None;
    if let Some(dir) = dump_turn_states {
        match std::fs::File::create(std::path::Path::new(dir).join(format!("game{game_idx}.jsonl")))
        {
            Ok(f) => turn_dump_file = Some(f),
            Err(e) => eprintln!("[dump-turn-states] failed to open game{game_idx} file: {e}"),
        }
    }
    let mut last_dump_key: Option<(i32, polyfish::states::PlayerId)> = None;

    let mut moves = 0;
    let mut ns_config1: u64 = 0;
    let mut moves_config1: u64 = 0;
    let mut ns_config2: u64 = 0;
    let mut moves_config2: u64 = 0;
    let mut samples: Vec<TurnSample> = Vec::new();
    let mut last_sampled_turn = i32::MIN;
    // EXP_ELO_026: config1's sticky expansion commitment (None = retired or
    // no capturable village visible). Tracked even in a gate-only arm, since
    // the gate is defined as active "while committed".
    let mut commitment: Option<i32> = None;
    // v2.3 tech-cap counters for the model seat (goal_script only).
    let mut techs_bought = 0u32;
    let mut tier3_bought = 0u32;
    // v3 archetype doctrine state for the model seat.
    let mut archetype_state = polyfish::ai::oracle_macro::ArchetypeState::default();
    // v7: standing macro commitment for the model seat (mirrors self_play).
    let mut stance_commit = polyfish::ai::oracle_macro::StanceCommit::default();

    while !polyfish::functions::is_game_over(&game.state) && moves < 500 {
        if dump_stats_dir.is_some() && game.state.settings.turn != last_sampled_turn {
            samples.push(sample_turn(&game.state, swap));
            last_sampled_turn = game.state.settings.turn;
        }
        let current_pid = game.state.settings.current_player_turn_id;

        // EXP_ELO_026: refresh the commitment before each of config1's
        // decisions — stars and cities change within a turn, and the target
        // may have been captured since the last ply.
        if (macro_commit || macro_star_gate) && current_pid == model_player {
            commitment =
                polyfish::ai::oracle_macro::update_commitment(&game.state, model_player, commitment);
            let model_agent = if swap { &mut agent_p2 } else { &mut agent_p1 };
            if let SearchAgent::Gumbel(a) = model_agent {
                a.pursuit_focus = if macro_commit { commitment } else { None };
                a.star_gate = macro_star_gate && commitment.is_some();
            }
        }

        // EXP_ELO_028: scripted goal channels for config1.
        if goal_script && current_pid == model_player {
            let goal = polyfish::ai::oracle_macro::update_goal(
                &game.state,
                model_player,
                &mut stance_commit,
                tier3_bought,
            );
            let gate =
                polyfish::ai::oracle_macro::goal_star_gate(&game.state, model_player, &goal);
            polyfish::ai::oracle_macro::update_archetype(
                &game.state,
                model_player,
                &goal,
                &mut archetype_state,
            );
            let aux = polyfish::ai::oracle_macro::scripted_goal_aux(
                &game.state,
                model_player,
                &goal,
                techs_bought,
                tier3_bought,
                Some(&archetype_state),
            );
            let model_agent = if swap { &mut agent_p2 } else { &mut agent_p1 };
            if let SearchAgent::Gumbel(a) = model_agent {
                a.star_gate = gate;
                a.macro_goal = Some(goal);
                a.goal_shape_w = goal_w_tree;
                a.goal_aux = Some(aux);
            }
        }

        // Dump the start-of-turn ground-truth snapshot once per (turn, acting
        // player), before any move that turn mutates the state.
        if let Some(f) = turn_dump_file.as_mut() {
            let key = (game.state.settings.turn, current_pid);
            if last_dump_key != Some(key) {
                dump_turn_state(f, game_idx, &game.state, model_player, greedy_player);
                last_dump_key = Some(key);
            }
        }

        let t0 = Instant::now();
        // Search on a clone: MCTS execute/undo must never touch the scored
        // state (Brain::think_decomposed clones for the same reason).
        let best_move = if current_pid == 1 {
            agent_p1.select_move(&mut game.clone())
        } else {
            agent_p2.select_move(&mut game.clone())
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
            if goal_script
                && current_pid == model_player
                && m.move_type() == polyfish::types::MoveType::Research
            {
                techs_bought += 1;
                if let Ok(tech) = m.tech_type() {
                    if polyfish::settings::technology::get_technology_setting(tech).tier
                        == Some(3)
                    {
                        tier3_bought += 1;
                    }
                }
            }
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

    if let Some(dir) = dump_stats_dir {
        samples.push(sample_turn(&game.state, swap)); // final post-game state
        // End-state build-out for the model seat: what it actually put on the
        // board and at what level, so a game can be held against eco_plan's
        // frontier instead of inferred from SPT alone.
        let model_pid: polyfish::states::PlayerId = if swap { 2 } else { 1 };
        let model_tribe = game.state.tribes.get(&model_pid);
        let mut territory: Vec<i32> = model_tribe
            .map(|t| t.cities.iter().flat_map(|c| c._territory.iter().copied()).collect())
            .unwrap_or_default();
        territory.sort_unstable();
        territory.dedup();
        let model_structures: Vec<serde_json::Value> = territory
            .iter()
            .filter_map(|&idx| {
                let st = polyfish::functions::get_structure_at(&game.state, idx)?;
                Some(serde_json::json!({
                    "idx": idx,
                    "type": format!("{:?}", st.structure_type),
                    "level": st.level,
                }))
            })
            .collect();
        let model_techs: Vec<String> = model_tribe
            .map(|t| {
                t.tech_vanilla
                    .iter()
                    .filter(|x| x.discovered)
                    .map(|x| format!("{:?}", x.tech_type))
                    .collect()
            })
            .unwrap_or_default();
        let model_city_levels: Vec<serde_json::Value> = model_tribe
            .map(|t| {
                t.cities
                    .iter()
                    .map(|c| serde_json::json!({"idx": c.idx, "level": c.level, "pop": c.population}))
                    .collect()
            })
            .unwrap_or_default();
        let dump = serde_json::json!({
            "seed": seed,
            "swap": swap,
            "winner_config": winner_config,
            "score_config1": score_config1,
            "score_config2": score_config2,
            "macro_commit": macro_commit,
            "macro_star_gate": macro_star_gate,
            "samples": samples,
            "model_structures": model_structures,
            "model_techs": model_techs,
            "model_city_levels": model_city_levels,
        });
        let name = format!("game_{}_{}.json", seed, if swap { "b" } else { "a" });
        let path = std::path::Path::new(dir).join(name);
        match serde_json::to_vec_pretty(&dump) {
            Ok(bytes) => {
                if let Err(e) = std::fs::write(&path, bytes) {
                    eprintln!("[dump-stats] failed to write {}: {e}", path.display());
                }
            }
            Err(e) => eprintln!("[dump-stats] failed to serialize seed {seed}: {e}"),
        }
    }

    // config1 sits in the P2 seat when sides are swapped.
    let depth_config1 = if swap {
        agent_p2.depth_stats()
    } else {
        agent_p1.depth_stats()
    };

    MatchResult {
        winner_config,
        swap,
        score_config1,
        score_config2,
        ns_config1,
        moves_config1,
        ns_config2,
        moves_config2,
        depth_config1,
    }
}

fn backend_from_arg(arg: SearchBackendArg, k: usize) -> SearchBackend {
    match arg {
        SearchBackendArg::Zero => SearchBackend::Zero,
        SearchBackendArg::Gumbel => SearchBackend::Gumbel { k },
        SearchBackendArg::Heuristic => SearchBackend::Heuristic,
        SearchBackendArg::Greedy => SearchBackend::Greedy,
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
            "ORACLE MACRO (EXP_ELO_026): commit={} star_gate={} (reserve={})",
            args.macro_commit,
            args.macro_star_gate,
            polyfish::ai::oracle_macro::STAR_GATE_RESERVE,
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

    let total_games = args.games * 2;

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
            (cores * 4).clamp(1, total_games.max(1))
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

    if let Some(dir) = &args.dump_turn_states {
        std::fs::create_dir_all(dir)?;
    }
    let dump_turn_states = args.dump_turn_states.as_deref();

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
            scope.spawn(move || {
                loop {
                    let idx = job_counter.fetch_add(1, Ordering::Relaxed);
                    if idx >= total_games {
                        break;
                    }
                    let seed = (base_seed + (idx / 2) as u64) as i64;
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
                            args.goal_script, args.goal_w_tree,
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
