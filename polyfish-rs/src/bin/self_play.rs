use candle_core::{Device, Tensor};
use polyfish::TribeType;
use polyfish::ai::brain::{Brain, SearchBackend, SearchBackendArg};
use polyfish::ai::eval_server::{
    BackendSpec, EvalHandle, EvalServer, EvalServerConfig, EvalServerStats, Evaluator,
    ShardedEvalHandle,
};
use polyfish::ai::features::{self, GameFeatures, state_to_tensor};
use polyfish::ai::mapper::DecomposedMapper;
use polyfish::ai::network::PolyZeroNet;
use polyfish::game::Game;
use polyfish::replayer::{ModReplay, ReplayPlayer, ReplayTurn};
use polyfish::states::PlayerId;
use polyfish::types::MapSize;
use serde_json::json;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const HEURISTIC_PRIOR_W0: f32 = 0.5; // net & heur blended 50/50 at start
const HEURISTIC_PRIOR_DECAY: f32 = 0.97; // from 0.5 to 0.03 by ~92 iterations

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

/// Up to 5 evenly spaced turn thresholds for periodic in-game progress.
fn turn_milestones(max_turns: i32) -> Vec<i32> {
    const MAX_REPORTS: usize = 5;
    if max_turns <= 0 {
        return Vec::new();
    }
    (1..=MAX_REPORTS)
        .map(|i| (max_turns * i as i32 + MAX_REPORTS as i32 - 1) / MAX_REPORTS as i32)
        .collect()
}

/// Game-count milestones at 20%, 40%, …, 100% for large runs.
fn finish_milestones(num_games: usize) -> Vec<usize> {
    (1..=5).map(|i| num_games * i / 5).collect()
}

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
    /// Move-type counts keyed by turn number, for the "move mix by turn"
    /// training-progress chart (see parse_metrics.py / dashboard).
    moves_by_turn: HashMap<i32, HashMap<polyfish::types::MoveType, usize>>,
    /// Combined (both players) tile-exploration and territory-ownership
    /// counts at game end, for diagnosing where score gains come from.
    revealed_tiles: i32,
    captured_tiles: i32,
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
    progress: ProgressMode,
) -> Option<GameResult> {
    // Curriculum logic — Tiny maps only, gradually increase turn count.
    let (map_size, max_turns) = if iteration <= 25 {
        (MapSize::Tiny, 10)
    } else if iteration <= 50 {
        (MapSize::Tiny, 15)
    } else if iteration <= 75 {
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
    if progress == ProgressMode::Full {
        eprintln!(
            "[Game {}] Started with seed: {} Tribes: {:?} (Curriculum: {:?}, max_turns: {})",
            game_idx, seed, gen_settings.tribes, map_size, max_turns
        );
    }

    let mut game = Game::new();
    game.state = polyfish::mapgen::generate(gen_settings);
    game.state.settings.mode = polyfish::types::ModeType::Perfection;
    game.state.settings.max_turns = max_turns;
    game.post_load();

    let prior_w = HEURISTIC_PRIOR_W0 * HEURISTIC_PRIOR_DECAY.powi(iteration as i32);

    // Create two agents (they might share the same network, or be different)
    let mut agent1 = Brain::with_backend(eval1, mcts_iters, backend)
    .with_prior_heuristic_weight(prior_w);
    let mut agent2 = Brain::with_backend(eval2, mcts_iters, backend)
    .with_prior_heuristic_weight(prior_w);

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
    let mut moves_by_turn: HashMap<i32, HashMap<polyfish::types::MoveType, usize>> =
        HashMap::new();

    let current_scores: Vec<(PlayerId, i32)> = game
        .state
        .tribes
        .iter()
        .map(|(id, t)| (*id, t.score))
        .collect();

    if progress == ProgressMode::Full {
        eprintln!(
            "[Game {}]: Turn: {} Scores: {:?}",
            game_idx, game.state.settings.turn, current_scores
        );
    }

    let milestones = if progress == ProgressMode::Periodic {
        turn_milestones(max_turns)
    } else {
        Vec::new()
    };
    let mut next_milestone = 0usize;

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
            *moves_by_turn
                .entry(game.state.settings.turn)
                .or_default()
                .entry(m_type)
                .or_insert(0) += 1;

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
            if progress == ProgressMode::Full && move_count > 0 && move_count % 10 == 0 {
                eprintln!(
                    "[Game {}]: Turn: {} Player: {} Move: {}",
                    game_idx,
                    game.state.settings.turn,
                    pov,
                    m.describe(&game.state),
                );
            }
            let _ = game.play_move(m.as_ref());

            if progress == ProgressMode::Periodic {
                while next_milestone < milestones.len()
                    && game.state.settings.turn >= milestones[next_milestone]
                {
                    eprintln!(
                        "[Game {}] turn {} reached (move {})",
                        game_idx,
                        milestones[next_milestone],
                        move_count + 1,
                    );
                    next_milestone += 1;
                }
            }
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
    if progress == ProgressMode::Full {
        eprintln!(
            "[Game {}] Finished. Moves: {} | Winner: {} (Score: {}) | Decisive: {}",
            game_idx, move_count, winner_id, winner_score, is_decisive
        );
    }

    let captured_tiles = game.state.tiles.values().filter(|t| t.owner != 0).count() as i32;
    let revealed_tiles: i32 = game
        .state
        .tribes
        .keys()
        .map(|&pid| {
            game.state
                .tiles
                .values()
                .filter(|t| t.explorers.contains(&pid))
                .count() as i32
        })
        .sum();

    Some(GameResult {
        history: game_history,
        scores,
        moves: move_count,
        revealed_tiles,
        captured_tiles,
        winner_score,
        recap: ModReplay {
            game_state: initial_state,
            turns: group_recap(flat_recap),
        },
        action_counts,
        moves_by_turn,
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

    #[derive(Parser, Debug)]
    #[command(author, version, about, long_about = None)]
    struct Args {
        /// Number of games to play
        #[arg(long, default_value_t = 10)]
        num_games: usize,

        /// MCTS iterations per move
        #[arg(long, default_value_t = 64)]
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
        /// this can shrink toward sequential per-game search. Measured
        /// (2026-07-05, 96 actors / 3 metal shards): raising this to 6 DID
        /// fatten coalesced batches (avg 47→60) but was a net ~10%
        /// throughput LOSS — more leaf evals per move, worse cache hit rate
        /// (0.19→0.17), and slower per-forward. Fatter batches via this
        /// knob are added work, not amortization. Keep at 4.
        #[arg(long, default_value = "4")]
        leaf_batch: Option<usize>,

        /// Eval-cache LRU capacity (number of cached NN evaluations). 0
        /// disables the cache. Default is 524288 (512K entries, ~900 MB at
        /// ~1.8 KB per row). The cache lives on the eval-server thread and
        /// skips the GPU for any leaf whose RawFeatures hash to a cached
        /// entry — the only lever that reduces GPU work rather than
        /// reshuffling it. Hit rate is reported in EVAL_SERVER_STATS.
        #[arg(long, default_value_t = 524288)]
        cache_cap: usize,

        /// NN inference backend: "candle" (Metal/CUDA/CPU), "tch"
        /// (libtorch/MPS, ~19x faster on Metal, requires --features
        /// tch-eval), or "metal" (MPSGraph, bypasses libtorch's serial MPS
        /// dispatch queue, requires --features metal-eval — see
        /// metal_network.rs). Empty = auto: "metal" if the metal-eval
        /// feature is compiled in, else "tch" if tch-eval is, else "candle".
        #[arg(long, default_value = "")]
        eval_backend: String,

        /// Number of concurrent eval-server threads (shards). Each owns its
        /// own weights copy + LRU cache; leaves are routed by hash so cache
        /// locality is preserved. 1 by default on every backend right now:
        /// measured (2026-07-05) that 2 tch shards HALVE throughput (156.6
        /// moves/s @ 1 vs 83.3 @ 2) because libtorch's MPS backend
        /// serializes across threads at the C++ level — sharding doesn't
        /// parallelize there. candle also rejects >1 (Metal corrupts when
        /// >1 thread encodes on the same device — see the bug_handoff
        /// invariant in eval_server.rs). metal (MPSGraph) has no known
        /// restriction against >1 but multi-shard concurrency there is
        /// unverified (see the bypass plan's Stage 3) — opt in explicitly.
        /// 0 = auto (currently always 1). Overridable.
        #[arg(long, default_value_t = 0)]
        eval_servers: usize,

        /// Metal backend only: pipelined GPU worker threads per eval server.
        /// Each owns its own MTLCommandQueue, so N coalesced batches can be
        /// in flight on the GPU while the coalescer collects the next one —
        /// unlike --eval-servers sharding, the batch stream and cache stay
        /// unified. Ignored by candle/tch.
        #[arg(long, default_value_t = 2)]
        eval_workers: usize,
    }

    let args = Args::parse();

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
    // Load models (P1, and P2 defaulting to P1 when no opponent is given)
    let (network1, network2) = load_networks(&device, args.opponent.as_deref())?;

    let base_seed = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();

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

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum EvalBackendKind {
        Candle,
        Tch,
        Metal,
    }

    // Resolve the eval backend: explicit --eval-backend, else auto (metal
    // when compiled in, else tch when compiled in, else candle).
    let eval_backend_kind = match args.eval_backend.as_str() {
        "tch" => {
            if !cfg!(feature = "tch-eval") {
                anyhow::bail!(
                    "--eval-backend tch requires building with --features tch-eval"
                );
            }
            EvalBackendKind::Tch
        }
        "metal" => {
            if !cfg!(feature = "metal-eval") {
                anyhow::bail!(
                    "--eval-backend metal requires building with --features metal-eval"
                );
            }
            EvalBackendKind::Metal
        }
        "candle" => EvalBackendKind::Candle,
        "" => {
            if cfg!(feature = "metal-eval") {
                EvalBackendKind::Metal
            } else if cfg!(feature = "tch-eval") {
                EvalBackendKind::Tch
            } else {
                EvalBackendKind::Candle
            }
        }
        other => anyhow::bail!("unknown --eval-backend {other:?} (want candle|tch|metal)"),
    };
    // Resolve shard count. We default to the best measured throughput
    let eval_servers = match args.eval_servers {
        0 => {
            if eval_backend_kind == EvalBackendKind::Metal {
                2
            } else {
                1
            }
        }
        n => {
            if n > 1 && eval_backend_kind == EvalBackendKind::Candle {
                anyhow::bail!(
                    "--eval-servers > 1 requires --eval-backend tch or metal \
                     (candle Metal corrupts when >1 thread encodes on the same device)"
                );
            }
            n
        }
    };
    // Each shard sees ~1/N of the working set (hash-routed), so dividing the
    // per-shard cache by N keeps total resident cache ~constant while
    // preserving the hit rate (cache / working-set ratio is unchanged).
    let per_shard_cache = if args.cache_cap == 0 {
        None
    } else {
        Some(args.cache_cap / eval_servers)
    };
    let eval_config = EvalServerConfig {
        max_batch: args.max_batch,
        coalesce_timeout: std::time::Duration::from_micros(args.coalesce_timeout_us),
        cache_capacity: per_shard_cache,
        pipeline_workers: args.eval_workers,
    };
    // Builds `n` backend specs for one player. For tch: every shard gets its
    // own `BackendSpec::Tch` on the shared MPS device; each shard's thread
    // loads its own `TchPolyZeroNet` (duplicated weights, a few MB). For
    // metal: every shard gets its own `BackendSpec::MetalMps`, each shard's
    // thread loads its own `MetalPolyZeroNet` and owns its own
    // `MTLCommandQueue`. For candle: `n` clones of the passed
    // `Arc<PolyZeroNet>` (player 1 vs player 2 networks differ in opponent
    // mode).
    let make_specs = |kind: EvalBackendKind,
                      n: usize,
                      model_path: &str,
                      candle_net: &Arc<PolyZeroNet>|
     -> Vec<BackendSpec> {
        // `model_path` is only read by the tch/metal branches below; in
        // builds with neither feature it's unused. Touch it so the closure
        // compiles cleanly either way.
        let _ = model_path;
        match kind {
            EvalBackendKind::Tch => {
                #[cfg(feature = "tch-eval")]
                {
                    let dev = if tch::utils::has_mps() {
                        tch::Device::Mps
                    } else {
                        tch::Device::Cpu
                    };
                    return (0..n)
                        .map(|_| BackendSpec::Tch {
                            model_path: model_path.to_string(),
                            device: dev,
                        })
                        .collect();
                }
                #[cfg(not(feature = "tch-eval"))]
                unreachable!("EvalBackendKind::Tch guarded by cfg above");
            }
            EvalBackendKind::Metal => {
                #[cfg(feature = "metal-eval")]
                {
                    return (0..n)
                        .map(|_| BackendSpec::MetalMps {
                            model_path: model_path.to_string(),
                        })
                        .collect();
                }
                #[cfg(not(feature = "metal-eval"))]
                unreachable!("EvalBackendKind::Metal guarded by cfg above");
            }
            EvalBackendKind::Candle => (0..n)
                .map(|_| BackendSpec::Candle(candle_net.clone()))
                .collect(),
        }
    };
    let p1_path = "model.safetensors";
    let p2_path = args
        .opponent
        .as_deref()
        .unwrap_or("model.safetensors");
    let p1_specs = make_specs(eval_backend_kind, eval_servers, p1_path, &network1);
    let has_opponent = args.opponent.is_some();
    let p2_specs = if has_opponent {
        make_specs(eval_backend_kind, eval_servers, p2_path, &network2)
    } else {
        Vec::new()
    };

    // Spawn the shards. Each EvalServer owns its inference thread + device
    // context; the handles are collected into a ShardedEvalHandle that
    // routes leaves by hash so each shard owns its own LRU cache.
    let mut p1_servers: Vec<EvalServer> = Vec::with_capacity(eval_servers);
    let mut p1_handles: Vec<EvalHandle> = Vec::with_capacity(eval_servers);
    for spec in p1_specs {
        let (srv, h) = EvalServer::start(spec, eval_config);
        p1_servers.push(srv);
        p1_handles.push(h);
    }
    let (p2_servers, p2_handles) = if has_opponent {
        // Opponent mode: independent shard set for player 2.
        let mut s = Vec::with_capacity(eval_servers);
        let mut h = Vec::with_capacity(eval_servers);
        for spec in p2_specs {
            let (srv, hh) = EvalServer::start(spec, eval_config);
            s.push(srv);
            h.push(hh);
        }
        (Some(s), h)
    } else {
        // Self-play against the same weights: both players share one shard
        // set so we don't run 2× inference threads for the same network.
        (None, p1_handles.clone())
    };
    let eval1 = Evaluator::Sharded(ShardedEvalHandle::new(p1_handles));
    let eval2 = Evaluator::Sharded(ShardedEvalHandle::new(p2_handles));

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

    let job_counter = Arc::new(AtomicUsize::new(0));
    let games_completed = Arc::new(AtomicUsize::new(0));
    let finish_milestones = finish_milestones(args.num_games);
    let results_mutex: Arc<std::sync::Mutex<Vec<GameResult>>> =
        Arc::new(std::sync::Mutex::new(Vec::with_capacity(args.num_games)));

    std::thread::scope(|scope| {
        for _ in 0..num_actors {
            let job_counter = job_counter.clone();
            let results_mutex = results_mutex.clone();
            let games_completed = games_completed.clone();
            let finish_milestones = finish_milestones.clone();
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
                        progress_mode,
                    );

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

    let results: Vec<GameResult> = match Arc::try_unwrap(results_mutex) {
        Ok(mutex) => mutex.into_inner().unwrap(),
        Err(_) => panic!("BUG: actor threads still hold a results_mutex reference after scope exit"),
    };

    let games_duration = games_start.elapsed();
    println!("Game generation completed in: {:.2}s ({} games)", games_duration.as_secs_f32(), results.len());
    println!("  Average: {:.2}s per game", games_duration.as_secs_f32() / results.len().max(1) as f32);
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
    if let Some(p2) = p2_servers.as_ref() {
        for s in p2 {
            all_shard_stats.push(("p2", s.stats()));
        }
    }

    let wall_s = games_duration.as_secs_f64().max(1e-9);

    // Aggregate across shards.
    let (mut agg_forwards, mut agg_rows, mut agg_max_batch, mut agg_busy_us) = (0u64, 0u64, 0u64, 0u64);
    let (mut agg_hits, mut agg_misses) = (0u64, 0u64);
    let (mut agg_compiles, mut agg_compile_us) = (0u64, 0u64);
    for (_, s) in &all_shard_stats {
        agg_forwards += s.forwards.load(Ordering::Relaxed);
        agg_rows += s.rows.load(Ordering::Relaxed);
        agg_max_batch = agg_max_batch.max(s.max_batch.load(Ordering::Relaxed));
        agg_busy_us += s.busy_us.load(Ordering::Relaxed);
        agg_hits += s.cache_hits.load(Ordering::Relaxed);
        agg_misses += s.cache_misses.load(Ordering::Relaxed);
        agg_compiles += s.compiles.load(Ordering::Relaxed);
        agg_compile_us += s.compile_us.load(Ordering::Relaxed);
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
        "EVAL_SERVER_STATS_AGG: {{\"shards\": {}, \"forwards\": {}, \"rows\": {}, \"avg_batch\": {:.2}, \"max_batch\": {}, \"busy_s\": {:.2}, \"busy_frac\": {:.3}, \"cache_hits\": {}, \"cache_misses\": {}, \"cache_hit_rate\": {:.3}, \"compiles\": {}, \"compile_s\": {:.3}, \"compile_frac_wall\": {:.4}, \"compile_frac_busy\": {:.4}}}",
        all_shard_stats.len(),
        agg_forwards,
        agg_rows,
        agg_avg_batch,
        agg_max_batch,
        agg_busy_s,
        agg_busy_s / wall_s,
        agg_hits,
        agg_misses,
        agg_cache_hit_rate,
        agg_compiles,
        agg_compile_s,
        agg_compile_s / wall_s,
        if agg_busy_s > 0.0 { agg_compile_s / agg_busy_s } else { 0.0 }
    );

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
    let mut total_revealed_tiles: i64 = 0;
    let mut total_captured_tiles: i64 = 0;

    let mut total_moves_by_turn: HashMap<i32, HashMap<polyfish::types::MoveType, usize>> =
        HashMap::new();

    for result in results {
        total_score += result.winner_score;
        total_moves += result.moves;
        total_revealed_tiles += result.revealed_tiles as i64;
        total_captured_tiles += result.captured_tiles as i64;
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

        for (turn, counts) in &result.moves_by_turn {
            let entry = total_moves_by_turn.entry(*turn).or_default();
            for (mt, c) in counts {
                *entry.entry(*mt).or_insert(0) += c;
            }
        }

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
    let avg_revealed_tiles = total_revealed_tiles as f32 / args.num_games as f32;
    let avg_captured_tiles = total_captured_tiles as f32 / args.num_games as f32;

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

    let games_file = if collected_spatial_maps.is_empty() {
        String::new()
    } else {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        format!("games_{timestamp}.safetensors")
    };

    // Stack and save
    if !collected_spatial_maps.is_empty() {
        let total_steps = collected_spatial_maps.len();
        let timestamp = games_file
            .strip_prefix("games_")
            .and_then(|s| s.strip_suffix(".safetensors"))
            .unwrap_or("0");

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

        candle_core::safetensors::save(&tensors, &games_file)?;

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

    }

    let metrics = json!({
        "avg_score": avg_score,
        "max_score": max_score,
        "avg_moves": avr_moves,
        "p1_avg": p1_avg,
        "p2_avg": p2_avg,
        "avg_captures": avg_captures,
        "avg_harvests": avg_harvests,
        "avg_builds": avg_builds,
        "avg_research": avg_research,
        "avg_attacks": avg_attacks,
        "avg_revealed_tiles": avg_revealed_tiles,
        "avg_captured_tiles": avg_captured_tiles,
        "games_file": games_file,
        "moves_by_turn": moves_by_turn,
    });
    std::fs::write(
        ".last_self_play_metrics.json",
        serde_json::to_string(&metrics)?,
    )?;

    let total_duration = start_time.elapsed();
    println!("\n=== Self-Play Complete ===");
    println!("Total time: {:.2}s", total_duration.as_secs_f32());
    println!("Breakdown:");
    println!("  - Game generation: {:.2}s ({:.1}%)", games_duration.as_secs_f32(), 100.0 * games_duration.as_secs_f32() / total_duration.as_secs_f32());
    let final_moves_per_sec = total_moves as f64 / games_duration.as_secs_f64().max(1e-9);
    println!("  - Throughput: {:.2} moves/sec ({} moves)", final_moves_per_sec, total_moves);

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
