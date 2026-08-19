use clap::Parser;
use polyfish::ai::brain::{SearchAgent, SearchBackend, SearchBackendArg, make_search_agent};
use polyfish::ai::macro_agent::{BeliefMode, MacroLeaf, MacroParams};
use polyfish::ai::eval_backend::{self, EvalBackendKind, PlayerBackend};
use polyfish::ai::eval_server::{EvalServerConfig, Evaluator};
use polyfish::ai::network::PolyZeroNet;
use polyfish::game::Game;
use polyfish::mapgen::{MapGenSettings, generate};
use polyfish::types::{MapSize, MapType, ModeType, TribeType};
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime};

/// Turn at which `--dump-stats-dir` freezes a mid-game board, chosen to sit
/// inside the turn 10-17 window where hubs are actually committed.
const MID_DUMP_TURN: i32 = 12;

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

    /// EXP_ELO_034: maintain a per-seat belief state (capital posterior +
    /// score-delta inference) from legal observables only, and log one
    /// belief-vs-truth row per player-turn into each game's dump JSON.
    /// Observation-only — no agent reads it. Requires --dump-stats-dir.
    #[arg(long, default_value_t = false)]
    belief_calib: bool,

    /// EXP_ELO_035: config 1's macro-mcts plans on a belief-materialized
    /// view — believed capital, ghost units, and the inferred residual army
    /// are written into the fogged root before the tree runs. Requires
    /// --backend1 macro-mcts.
    #[arg(long, default_value_t = false)]
    macro_belief1: bool,

    /// EXP_ELO_035: same as --macro-belief1, for config 2.
    #[arg(long, default_value_t = false)]
    macro_belief2: bool,

    /// Capture the full root decision (priors, top-k cut, visits, Q) on every
    /// model ply where this tech is unowned, unlocked by its prerequisite, and
    /// affordable — i.e. the plies where the purchase is a live choice.
    /// Requires --dump-stats-dir. Arming invalidates the reused tree, so a
    /// traced game diverges from an untraced one at the first matching ply.
    #[arg(long)]
    trace_tech: Option<String>,

    /// Tribe for both seats. Spawn terrain is tribe-specific -- Bardur forest,
    /// XinXi mountain/metal, Kickoo water/fruit -- and a hub's ceiling is a
    /// property of the terrain around it, so any statement about hub quality is
    /// tribe-scoped until this is varied.
    #[arg(long, default_value = "imperius")]
    tribe: String,

    /// EXP_ELO_026 oracle-macro steer for config 1 (gumbel backend only):
    /// while it holds <3 cities, focus the pursuit channel on one sticky
    /// FOW-visible neutral village (nearest to its units).
    #[arg(long, default_value_t = false)]
    macro_commit: bool,

    /// EXP_ELO_026 oracle-macro steer for config 1 (gumbel backend only):
    /// while a commitment is active, drop every root Research move. v9 removed
    /// the old 5-star reserve escape, so this arm is now a hard block rather
    /// than the affordability test the original experiment measured.
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

    /// EXP_ELO_032: leaf scorer for a macro-lookahead backend.
    #[arg(long, value_enum, default_value_t = MacroLeaf::Heuristic)]
    macro_leaf: MacroLeaf,

    /// EXP_ELO_032: max candidate directives per turn (base always kept).
    #[arg(long, default_value_t = 4)]
    macro_k: usize,

    /// EXP_ELO_032: own turns simulated per rollout, incl. the candidate turn.
    #[arg(long, default_value_t = 2)]
    macro_horizon: u32,

    /// EXP_ELO_032: λ on Δgoal_potential in the macro executor's ply ranking.
    #[arg(long, default_value_t = 1.0)]
    macro_lambda: f32,

    /// EXP_ELO_033: simulations per turn-level tree search (macro-mcts only).
    #[arg(long, default_value_t = 32)]
    macro_sims: usize,

    /// Override --macro-sims for config 1 (sims-sweep rungs).
    #[arg(long)]
    macro_sims1: Option<usize>,

    /// Override --macro-sims for config 2.
    #[arg(long)]
    macro_sims2: Option<usize>,

    /// Override --macro-k for config 1 (candidate-width rungs, EXP_ELO_036).
    #[arg(long)]
    macro_k1: Option<usize>,

    /// Override --macro-k for config 2.
    #[arg(long)]
    macro_k2: Option<usize>,

    /// EXP_ELO_035/036: config 1's belief consumption (macro-mcts only).
    /// `world` = materialize the plan view (035); `candidates` =
    /// belief-conditioned fog-expansion candidates at the root (036 rung 1);
    /// `both`. Overrides --macro-belief1 when set.
    #[arg(long, value_enum, default_value_t = BeliefMode::Off)]
    macro_belief_mode1: BeliefMode,

    /// Same as --macro-belief-mode1, for config 2.
    #[arg(long, value_enum, default_value_t = BeliefMode::Off)]
    macro_belief_mode2: BeliefMode,

    /// EXP_ELO_039: override --macro-leaf for config 1 (net-vs-heuristic
    /// leaf A/B needs per-side leaves).
    #[arg(long, value_enum)]
    macro_leaf1: Option<MacroLeaf>,

    /// Same as --macro-leaf1, for config 2.
    #[arg(long, value_enum)]
    macro_leaf2: Option<MacroLeaf>,

    /// EXP_ELO_036b: config 1's weight on potential-based Δφ edge rewards in
    /// the macro tree (own edges only; 0 = off). Credits the WORK of advance
    /// toward the active directive inside the search objective — the 028
    /// lesson at the macro layer.
    #[arg(long, default_value_t = 0.0)]
    macro_shape_w1: f32,

    /// Same as --macro-shape-w1, for config 2.
    #[arg(long, default_value_t = 0.0)]
    macro_shape_w2: f32,
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
/// Tribe name to type, mirroring self_play's parser.
fn tribe_of(s: &str) -> TribeType {
    match s.to_lowercase().as_str() {
        "imperius" => TribeType::Imperius,
        "bardur" => TribeType::Bardur,
        "oumaji" => TribeType::Oumaji,
        "kickoo" => TribeType::Kickoo,
        "xinxi" => TribeType::XinXi,
        "zebasi" => TribeType::Zebasi,
        "hoodrick" => TribeType::Hoodrick,
        "vengir" => TribeType::Vengir,
        "luxidoor" => TribeType::Luxidoor,
        "yadakk" => TribeType::Yadakk,
        "aimo" => TribeType::AiMo,
        "quetzali" => TribeType::Quetzali,
        other => {
            eprintln!("unknown tribe {other}, using imperius");
            TribeType::Imperius
        }
    }
}

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
/// EXP_ELO_041: per-seat siege bookkeeping. A "siege" is an enemy unit
/// standing on an owned city tile; each episode resolves as UNSIEGED (enemy
/// gone, city kept) or LOST (ownership flipped). Scanned after every move.
struct SiegeTracker {
    active: std::collections::HashMap<(i32, i32), serde_json::Value>, // (owner pid, city idx) -> open facts
    sieges: [u32; 2],   // per SEAT (P1, P2): episodes started
    unsieged: [u32; 2], // …resolved by clearing the attacker
    lost: [u32; 2],     // …resolved by losing the city
    /// EXP_ELO_049: one closed record per episode, emitted into the game dump.
    episodes: Vec<serde_json::Value>,
    detail: bool,
}

impl SiegeTracker {
    fn new(detail: bool) -> Self {
        Self {
            active: Default::default(),
            sieges: [0; 2],
            unsieged: [0; 2],
            lost: [0; 2],
            episodes: Vec::new(),
            detail,
        }
    }

    /// Facts at the moment the attacker steps onto the city — the ones that
    /// decide whether the defence was POSSIBLE, separately from whether it
    /// happened: who can strike the tile next turn, how far the nearest
    /// unit is, what the bank holds, and whether Tier 2 had even named this
    /// city as something to defend.
    fn open_facts(
        state: &polyfish::states::GameState,
        owner: i32,
        idx: i32,
        goal: Option<&polyfish::ai::oracle_macro::MacroGoal>,
    ) -> serde_json::Value {
        let size = state.settings.size;
        let attacker = polyfish::functions::get_true_unit_at(state, idx);
        let tribe = state.tribes.get(&owner);
        let city_level = tribe
            .and_then(|t| t.cities.iter().find(|c| c.idx == idx).map(|c| c.level))
            .unwrap_or(0);
        // Own units, excluding anything standing on the besieged tile.
        let own: Vec<&polyfish::states::UnitState> = tribe
            .map(|t| t.units.iter().filter(|u| u.coords.idx != idx).collect())
            .unwrap_or_default();
        let nearest = own
            .iter()
            .map(|u| polyfish::functions::get_chebyshev_distance(u.coords.idx, idx, size))
            .min();
        let responders = own
            .iter()
            .filter(|u| polyfish::ai::combat::unit_covers_threat(state, u, idx))
            .count();
        let ordered_defend = goal.map(|g| {
            g.orders.iter().any(|(k, t)| {
                *k == polyfish::ai::oracle_macro::OrderKind::Defend && *t == idx
            })
        });
        serde_json::json!({
            "owner": owner,
            "city": idx,
            "city_level": city_level,
            "turn_open": state.settings.turn,
            "attacker": attacker.map(|u| format!("{:?}", u.unit_type)),
            "attacker_health": attacker.map(|u| u.health),
            "own_units": own.len(),
            "nearest_unit_dist": nearest,
            // Units that could strike the besieging unit next turn — the
            // capability the unsiege actually needs.
            "responders": responders,
            "stars": tribe.map(|t| t.stars),
            "spt": tribe.map(|t| polyfish::functions::get_tribe_spt(state, t)),
            "defend_ordered": ordered_defend,
        })
    }

    fn scan(
        &mut self,
        state: &polyfish::states::GameState,
        goals: [Option<&polyfish::ai::oracle_macro::MacroGoal>; 2],
    ) {
        let (sieges, unsieged, lost) =
            (&mut self.sieges, &mut self.unsieged, &mut self.lost);
        let episodes = &mut self.episodes;
        let detail = self.detail;
        self.active.retain(|&(owner, idx), open| {
            let seat = (owner - 1).clamp(0, 1) as usize;
            let still_owned = state
                .tribes
                .get(&owner)
                .map_or(false, |t| t.cities.iter().any(|c| c.idx == idx));
            let mut close = |outcome: &str| {
                if detail {
                    let mut rec = open.clone();
                    rec["outcome"] = serde_json::json!(outcome);
                    rec["turn_close"] = serde_json::json!(state.settings.turn);
                    episodes.push(rec);
                }
            };
            if !still_owned {
                lost[seat] += 1;
                close("lost");
                return false;
            }
            let enemy_on = polyfish::functions::get_true_unit_at(state, idx)
                .map_or(false, |u| u.owner != owner);
            if !enemy_on {
                unsieged[seat] += 1;
                close("unsieged");
                return false;
            }
            true
        });
        for (pid, t) in &state.tribes {
            let seat = (*pid - 1).clamp(0, 1) as usize;
            for c in &t.cities {
                let enemy_on = polyfish::functions::get_true_unit_at(state, c.idx)
                    .map_or(false, |u| u.owner != *pid);
                if enemy_on && !self.active.contains_key(&(*pid, c.idx)) {
                    let facts = if self.detail {
                        Self::open_facts(state, *pid, c.idx, goals[seat])
                    } else {
                        serde_json::Value::Null
                    };
                    self.active.insert((*pid, c.idx), facts);
                    sieges[seat] += 1;
                }
            }
        }
    }
}

struct MatchResult {
    winner_config: u8,
    /// true = config 2 sat in the P1 seat this game.
    swap: bool,
    score_config1: i32,
    score_config2: i32,
    /// EXP_ELO_041 per config: (sieges suffered, unsieged, cities lost).
    siege_config1: (u32, u32, u32),
    siege_config2: (u32, u32, u32),
    ns_config1: u64,
    moves_config1: u64,
    ns_config2: u64,
    moves_config2: u64,
    /// Search telemetry for config1:
    /// (depth_sum, depth_count, depth_max, horizon_hits, agree, decisions).
    depth_config1: Option<(u64, u64, u32, u64, u64, u64)>,
    /// EXP_ELO_032, config1 macro-lookahead only: (divergent, planned) turns.
    macro_divergence: Option<(u32, u32)>,
    /// EXP_ELO_035, config1 belief-enabled macro-mcts only:
    /// (capital-materialized turns, units materialized, planned turns).
    belief_mat: Option<(u32, u32, u32)>,
    /// EXP_ELO_036/038, config1 macro-mcts: winning-candidate class counts
    /// (base/stance/real/attack/claim/contest/continuation), belief-target
    /// re-picks, mid-turn fog-order strips.
    belief_gen: Option<([u32; 7], u32, u32)>,
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
    macro_params1: MacroParams,
    macro_params2: MacroParams,
    args_tribe: &str,
    trace_tech: Option<polyfish::types::TechnologyType>,
    belief_calib: bool,
) -> MatchResult {
    let gen_settings = MapGenSettings {
        size: MapSize::Tiny,
        map_type: MapType::Drylands,
        tribes: vec![tribe_of(&args_tribe), tribe_of(&args_tribe)],
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
            make_search_agent(backend2, eval2, mcts2, leaf_batch, None, None, None, None, None, None, Some(false), Some(macro_params2)),
            2u8,
            make_search_agent(backend1, eval1, mcts1, leaf_batch, None, None, None, None, None, None, Some(false), Some(macro_params1)),
            1u8,
        )
    } else {
        (
            make_search_agent(backend1, eval1, mcts1, leaf_batch, None, None, None, None, None, None, Some(false), Some(macro_params1)),
            1u8,
            make_search_agent(backend2, eval2, mcts2, leaf_batch, None, None, None, None, None, None, Some(false), Some(macro_params2)),
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
    let mut mid_dumped = false;

    let mut moves = 0;
    let mut ns_config1: u64 = 0;
    let mut moves_config1: u64 = 0;
    let mut ns_config2: u64 = 0;
    let mut moves_config2: u64 = 0;
    let mut samples: Vec<TurnSample> = Vec::new();
    let mut last_sampled_turn = i32::MIN;
    // Hub-placement optimality: for every hub the model builds, the ceiling of
    // the tile it chose against the best ceiling legally available to it at
    // that instant. Measured AT THE DECISION, which end-state ceilings cannot
    // see -- by the last turn every buildable tile is built and chosen and best
    // collapse onto each other.
    let mut placements: Vec<serde_json::Value> = Vec::new();
    // EXP_ELO_026: config1's sticky expansion commitment (None = retired or
    // no capturable village visible). Tracked even in a gate-only arm, since
    // the gate is defined as active "while committed".
    let mut commitment: Option<i32> = None;
    // v2.3 tech-cap counters for the model seat (goal_script only).
    let mut techs_bought = 0u32;
    let mut tier3_bought = 0u32;
    // v3 lane doctrine state for the model seat.
    let mut lane_state = polyfish::ai::oracle_macro::LaneState::default();
    // v7: standing macro commitment for the model seat (mirrors self_play).
    let mut stance_commit = polyfish::ai::oracle_macro::StanceCommit::default();
    // One row per model ply: the goal the script set, and the move that
    // followed it. Separates "the plan was dropped" from "the plan was held
    // and ignored" -- the flip counters alone cannot tell those apart.
    /// Root traces are ~40 candidates plus every halving round each; cap them
    /// so one game's dump stays readable.
    const TRACE_CAP: usize = 12;
    let mut tech_traces: Vec<serde_json::Value> = Vec::new();
    let mut goal_trace: Vec<serde_json::Value> = Vec::new();
    let mut pending_goal: Option<serde_json::Value> = None;
    // EXP_ELO_034/035/036: the belief feed. The harness reads true state
    // solely to stream each observer its legal observables (and, for
    // --belief-calib, to log truth rows). Belief-enabled macro seats consume
    // clones per turn, per their MacroParams::belief_mode.
    let feed_on = |p: &MacroParams| p.belief_mode != polyfish::ai::macro_agent::BeliefMode::Off;
    let mut calib: Option<polyfish::ai::belief::CalibHarness> =
        if belief_calib || feed_on(&macro_params1) || feed_on(&macro_params2) {
            Some(polyfish::ai::belief::CalibHarness::new(&game.state))
        } else {
            None
        };
    let mut last_calib_key: Option<(i32, polyfish::states::PlayerId)> = None;
    // Which SEAT consumes belief (params are seat-swapped like configs).
    let mb_p1 = feed_on(if swap { &macro_params2 } else { &macro_params1 });
    let mb_p2 = feed_on(if swap { &macro_params1 } else { &macro_params2 });

    let mut siege_tracker = SiegeTracker::new(dump_stats_dir.is_some());
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
            let goal = polyfish::ai::oracle_macro::commit_macro_goal(
                &game.state,
                model_player,
                &mut stance_commit,
                tier3_bought,
            );
            let gate =
                polyfish::ai::oracle_macro::tech_discipline_active(&game.state, model_player, &goal);
            polyfish::ai::oracle_macro::update_lane_state(&game.state, model_player, &mut lane_state);
            let aux = polyfish::ai::oracle_macro::compute_goal_aux(
                &game.state,
                model_player,
                &goal,
                techs_bought,
                tier3_bought,
                Some(&lane_state),
            );
            if dump_stats_dir.is_some() {
                // The uncommitted goal too: `commit_macro_goal` returns the stance
                // after hysteresis, so a script that wants to switch and a
                // script that is content look identical in the result alone.
                let fresh = polyfish::ai::oracle_macro::compute_macro_goal(
                    &game.state,
                    model_player,
                    tier3_bought,
                );
                let tribe = game.state.tribes.get(&model_player);
                pending_goal = Some(serde_json::json!({
                    "turn": game.state.settings.turn,
                    "stance": format!("{:?}", goal.stance),
                    "stance_fresh": format!("{:?}", fresh.stance),
                    "save_target": goal.save_target.as_ref().map(|l| l.cost),
                    "save_lane": goal.save_target.as_ref()
                        .map(|l| format!("{:?}+{:?}", l.tech, l.structure)),
                    "save_target_fresh": fresh.save_target.as_ref().map(|l| l.cost),
                    "orders": goal.orders.iter()
                        .map(|(k, i)| serde_json::json!([format!("{k:?}"), i]))
                        .collect::<Vec<_>>(),
                    "star_gate": gate,
                    "stars": tribe.map(|t| t.stars),
                    "spt": tribe.map(|t| polyfish::functions::get_tribe_spt(&game.state, t)),
                    "cities": tribe.map(|t| t.cities.len()),
                    "techs_bought": techs_bought,
                    "tier3_bought": tier3_bought,
                }));
            }
            let model_agent = if swap { &mut agent_p2 } else { &mut agent_p1 };
            if let SearchAgent::Gumbel(a) = model_agent {
                a.star_gate = gate;
                a.macro_goal = Some(goal);
                a.goal_shape_w = goal_w_tree;
                a.goal_aux = Some(aux);
            }
        }

        // A board frozen while the build-out is still live. Hubs are committed
        // turns 10-17, so the final board -- every buildable tile already
        // built -- is the least informative position to plan from.
        if let Some(dir) = dump_stats_dir {
            if !mid_dumped
                && game.state.settings.turn >= MID_DUMP_TURN
                && current_pid == model_player
            {
                mid_dumped = true;
                let p = std::path::Path::new(dir).join(format!(
                    "mid_{}_{}.json",
                    seed,
                    if swap { "b" } else { "a" }
                ));
                if let Ok(j) = serde_json::to_string(&game.state) {
                    let _ = std::fs::write(&p, j);
                }
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

        // EXP_ELO_034: belief-vs-truth row at the start of the acting
        // player's turn — the moment a planner would consume the belief.
        if let Some(c) = calib.as_mut() {
            if belief_calib {
                let key = (game.state.settings.turn, current_pid);
                if last_calib_key != Some(key) {
                    c.turn_row(&game.state, current_pid);
                    last_calib_key = Some(key);
                }
            }
        }

        // EXP_ELO_035: hand the acting belief-enabled macro seat its current
        // belief before it plans this ply's move.
        if (current_pid == 1 && mb_p1) || (current_pid == 2 && mb_p2) {
            if let Some(b) = calib.as_ref().and_then(|c| c.belief_for(current_pid)) {
                let agent = if current_pid == 1 { &mut agent_p1 } else { &mut agent_p2 };
                if let SearchAgent::MacroMcts(a) = agent {
                    a.set_belief(b.clone());
                }
            }
        }

        // Arm the root trace only on plies where `trace_tech` is a live
        // choice: prerequisite owned, tech not yet bought, cost affordable.
        let mut armed_ctx: Option<serde_json::Value> = None;
        if let Some(tech) = trace_tech {
            if current_pid == model_player && tech_traces.len() < TRACE_CAP {
                if let Some(t) = game.state.tribes.get(&model_player) {
                    use polyfish::settings::technology as tech_mod;
                    let owned = tech_mod::has_technology(&t.tech_vanilla, tech);
                    let prereq = tech_mod::get_technology_setting(tech)
                        .requires
                        .map_or(true, |r| tech_mod::has_technology(&t.tech_vanilla, r));
                    let cost = tech_mod::get_tech_cost(
                        t.cities.len() as i32,
                        tech_mod::tech_tier(tech),
                        tech_mod::has_technology(
                            &t.tech_vanilla,
                            polyfish::types::TechnologyType::Philosophy,
                        ),
                    );
                    if !owned && prereq && t.stars >= cost {
                        armed_ctx = Some(serde_json::json!({
                            "turn": game.state.settings.turn,
                            "stars": t.stars,
                            "cost": cost,
                            "cities": t.cities.len(),
                            "spt": polyfish::functions::get_tribe_spt(&game.state, t),
                        }));
                        let a = if swap { &mut agent_p2 } else { &mut agent_p1 };
                        if let SearchAgent::Gumbel(g) = a {
                            g.arm_trace();
                        }
                    }
                }
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

        // Stage 4, the macro path's own trace: `ply <- order <- playstyle`.
        // The scripted emitter above is gated on `--goal-script`, which arena
        // refuses to combine with a non-Gumbel backend — so it never fires
        // here. Same row schema, so downstream analysis stays one parser;
        // filled AFTER the search because the macro agent commits its lane and
        // directive during it.
        if dump_stats_dir.is_some() && current_pid == model_player {
            let model_agent = if swap { &agent_p2 } else { &agent_p1 };
            if let Some(ps) = model_agent.macro_playstyle() {
                let goal = model_agent.macro_committed_goal();
                let tribe = game.state.tribes.get(&model_player);
                pending_goal = Some(serde_json::json!({
                    "turn": game.state.settings.turn,
                    "source": "macro",
                    "playstyle": ps.lane.map(|a| format!("{a:?}")),
                    "playstyle_committed_turn": ps.committed_turn,
                    "playstyle_pivots_used": ps.pivots_used,
                    "lane_blocked_turns": ps.lane_blocked_turns,
                    // oracle_macro::LANE_ORDER: RiderRoads, ArcherLine, SpamGiants.
                    "playstyle_scores": ps.last_scores,
                    "stance": goal.map(|g| format!("{:?}", g.stance)),
                    "save_target": goal.and_then(|g| g.save_target.as_ref().map(|l| l.cost)),
                    "save_lane": goal.and_then(|g| {
                        g.save_target.as_ref().map(|l| format!("{:?}+{:?}", l.tech, l.structure))
                    }),
                    "orders": goal.map(|g| {
                        g.orders
                            .iter()
                            .map(|(k, i)| serde_json::json!([format!("{k:?}"), i]))
                            .collect::<Vec<_>>()
                    }),
                    "stars": tribe.map(|t| t.stars),
                    "spt": tribe.map(|t| polyfish::functions::get_tribe_spt(&game.state, t)),
                    "cities": tribe.map(|t| t.cities.len()),
                    "techs_bought": techs_bought,
                    "tier3_bought": tier3_bought,
                }));
            }
        }

        if let Some(mut ctx) = armed_ctx {
            let a = if swap { &mut agent_p2 } else { &mut agent_p1 };
            if let SearchAgent::Gumbel(g) = a {
                if let Some(tr) = g.take_trace() {
                    ctx["stance"] = pending_goal
                        .as_ref()
                        .and_then(|p| p.get("stance").cloned())
                        .unwrap_or(serde_json::Value::Null);
                    ctx["save_target"] = pending_goal
                        .as_ref()
                        .and_then(|p| p.get("save_target").cloned())
                        .unwrap_or(serde_json::Value::Null);
                    ctx["trace"] = serde_json::to_value(&tr).unwrap_or_default();
                    tech_traces.push(ctx);
                }
            }
        }

        let cfg = if current_pid == 1 { p1_config } else { p2_config };
        if cfg == 1 {
            ns_config1 += dt;
            moves_config1 += 1;
        } else {
            ns_config2 += dt;
            moves_config2 += 1;
        }

        if let Some(mut row) = pending_goal.take() {
            let (kind, desc) = match &best_move {
                Some(m) => (
                    format!("{:?}", m.move_type()),
                    m.describe(&game.state),
                ),
                None => ("None".to_string(), String::new()),
            };
            row["move_type"] = serde_json::json!(kind);
            row["move"] = serde_json::json!(desc);
            row["tech"] = match best_move.as_ref().and_then(|m| m.tech_type().ok()) {
                Some(t) => serde_json::json!(format!("{t:?}")),
                None => serde_json::Value::Null,
            };
            row["structure"] = match best_move.as_ref().and_then(|m| m.structure_type().ok()) {
                Some(s) => serde_json::json!(format!("{s:?}")),
                None => serde_json::Value::Null,
            };
            row["unit"] = match best_move.as_ref().and_then(|m| m.unit_type().ok()) {
                Some(u) => serde_json::json!(format!("{u:?}")),
                None => serde_json::Value::Null,
            };
            goal_trace.push(row);
        }

        if let Some(m) = best_move {
            if current_pid == model_player
                && m.move_type() == polyfish::types::MoveType::Build
            {
                if let (Ok(kind), Ok(tile)) = (m.structure_type(), m.target_idx()) {
                    let setting =
                        polyfish::settings::structures::get_structure_setting(kind);
                    if setting.reward_pop > 0 && !setting.adjacent_types.is_empty() {
                        let chosen = tile as i32;
                        let chosen_ceiling = polyfish::rules::economy::partner_ceiling(
                            &game.state, chosen, kind, model_player,
                        );
                        // Every tile this same hub could legally go on right now.
                        let mut alts: Vec<(i32, i32)> = Vec::new();
                        for cand in polyfish::moves::generate_legal_moves(&game.state) {
                            if cand.move_type() != polyfish::types::MoveType::Build {
                                continue;
                            }
                            if cand.structure_type().ok() != Some(kind) {
                                continue;
                            }
                            if let Ok(t) = cand.target_idx() {
                                let t = t as i32;
                                // limited_per_city: only this city's tiles are
                                // alternatives to this city's placement.
                                if !polyfish::rules::economy::same_city(
                                    &game.state, t, chosen,
                                ) {
                                    continue;
                                }
                                alts.push((
                                    t,
                                    polyfish::rules::economy::partner_ceiling(
                                        &game.state, t, kind, model_player,
                                    ),
                                ));
                            }
                        }
                        let best = alts.iter().map(|&(_, c)| c).max().unwrap_or(chosen_ceiling);
                        let best_tile = alts
                            .iter()
                            .filter(|&&(_, c)| c == best)
                            .map(|&(t, _)| t)
                            .min()
                            .unwrap_or(chosen);
                        // What the model traded away. A tile can be a poor hub
                        // site and still be worth keeping -- it may carry a
                        // resource the hub would crush, or be a partner slot.
                        let describe = |t: i32| {
                            let res = game
                                .state
                                .resources
                                .get(&t)
                                .and_then(|r| r.as_ref())
                                .map(|r| format!("{:?}", r.resource_type));
                            let terr = game
                                .state
                                .tiles
                                .get(&t)
                                .map(|x| format!("{:?}", x.terrain_type));
                            let city = polyfish::functions::get_city_owning_tile(&game.state, t)
                                .map(|c| c.idx);
                            serde_json::json!({
                                "tile": t,
                                "terrain": terr,
                                "resource": res,
                                "city": city,
                                "dist_to_city": city.map(|c| {
                                    polyfish::functions::get_chebyshev_distance(
                                        t, c, game.state.settings.size)
                                }),
                            })
                        };
                        placements.push(serde_json::json!({
                            "turn": game.state.settings.turn,
                            "kind": format!("{kind:?}"),
                            "tile": chosen,
                            "chosen_ceiling": chosen_ceiling,
                            "best_ceiling": best,
                            "n_options": alts.len(),
                            "chosen_detail": describe(chosen),
                            "best_detail": describe(best_tile),
                            "stars": game.state.tribes.get(&model_player).map(|t| t.stars),
                        }));
                    }
                }
            }
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
            {
                // The committed directive of each seat, so an episode records
                // whether Tier 2 had even named this city as one to defend.
                let g1 = agent_p1.macro_committed_goal();
                let g2 = agent_p2.macro_committed_goal();
                siege_tracker.scan(&game.state, [g1, g2]);
            }
            if let Some(c) = calib.as_mut() {
                c.after_move(&game.state, current_pid, m.as_ref());
            }
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

    let seat_siege = |i: usize| {
        (
            siege_tracker.sieges[i],
            siege_tracker.unsieged[i],
            siege_tracker.lost[i],
        )
    };
    let (siege_config1, siege_config2) = if swap {
        (seat_siege(1), seat_siege(0))
    } else {
        (seat_siege(0), seat_siege(1))
    };

    let winner_config = if score_config1 > score_config2 {
        1
    } else if score_config2 > score_config1 {
        2
    } else {
        0
    };

    // EXP_ELO_032: how often lookahead overrode the scripted base directive.
    // A flat lookahead-vs-script result is uninterpretable without this.
    let macro_divergence = {
        let model_agent = if swap { &agent_p2 } else { &agent_p1 };
        match model_agent {
            SearchAgent::MacroLookahead(a) => Some((a.divergent_turns, a.planned_turns)),
            SearchAgent::MacroMcts(a) => Some((a.divergent_turns, a.planned_turns)),
            _ => None,
        }
    };
    // EXP_ELO_035: how often/how much materialization actually ran for
    // config1 — a flat belief-vs-baseline result is uninterpretable without
    // it (the posterior confirms fast, so the window may be turns 0-10 only).
    let belief_mat = {
        let model_agent = if swap { &agent_p2 } else { &agent_p1 };
        match model_agent {
            SearchAgent::MacroMcts(a) if a.belief.is_some() => {
                Some((a.mat_capital_turns, a.mat_units, a.planned_turns))
            }
            _ => None,
        }
    };
    // EXP_ELO_036: which candidate class won each planned turn for config1,
    // plus consecutive-turn re-picks of the same belief fog target.
    let belief_gen = {
        let model_agent = if swap { &agent_p2 } else { &agent_p1 };
        match model_agent {
            SearchAgent::MacroMcts(a) => {
                Some((a.class_picks, a.belief_repicks, a.intra_strips))
            }
            _ => None,
        }
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
                let setting =
                    polyfish::settings::structures::get_structure_setting(st.structure_type);
                // For an adjacency hub: what the site could ever collect vs what
                // it did. Separates a bad tile (low ceiling) from an unfinished
                // one (high ceiling, few partners built).
                let ceiling = if setting.adjacent_types.is_empty() {
                    -1
                } else {
                    polyfish::rules::economy::partner_ceiling(
                        &game.state,
                        idx,
                        st.structure_type,
                        model_pid,
                    )
                };
                let realized = if setting.adjacent_types.is_empty() {
                    -1
                } else {
                    polyfish::rules::economy::partner_count(
                        &game.state,
                        idx,
                        st.structure_type,
                        model_pid,
                    )
                };
                Some(serde_json::json!({
                    "idx": idx,
                    "type": format!("{:?}", st.structure_type),
                    "level": st.level,
                    "ceiling": ceiling,
                    "realized": realized,
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
                    .map(|c| serde_json::json!({
                        "idx": c.idx,
                        "level": c.level,
                        "pop": c.population,
                        // Which side of each level's fork the city took. The
                        // level-3 slot is PopGrowth vs BorderGrowth, and border
                        // is what grows the territory a hub's ceiling lives in.
                        "rewards": c.rewards.iter().map(|r| format!("{r:?}")).collect::<Vec<_>>(),
                    }))
                    .collect()
            })
            .unwrap_or_default();
        // config1 sits in the P2 seat when sides are swapped.
        let macro_playstyle =
            if swap { agent_p2.macro_playstyle() } else { agent_p1.macro_playstyle() }.cloned();
        let dump = serde_json::json!({
            "seed": seed,
            "swap": swap,
            "winner_config": winner_config,
            "score_config1": score_config1,
            "score_config2": score_config2,
            "sieges_config1": siege_config1.0,
            "unsieged_config1": siege_config1.1,
            "cities_lost_config1": siege_config1.2,
            "sieges_config2": siege_config2.0,
            "unsieged_config2": siege_config2.1,
            "cities_lost_config2": siege_config2.2,
            "macro_commit": macro_commit,
            "macro_star_gate": macro_star_gate,
            "samples": samples,
            "model_structures": model_structures,
            "model_techs": model_techs,
            "model_city_levels": model_city_levels,
            "placements": placements,
            "goal_trace": goal_trace,
            // EXP_ELO_049: one record per siege episode — the facts at the
            // moment the attacker stepped on, and how it ended.
            "siege_episodes": siege_tracker.episodes,
            "tech_traces": tech_traces,
            "stance_flips": stance_commit.stance_flips,
            "order_flips": stance_commit.order_flips,
            "goal_turns_seen": stance_commit.turns_seen,
            // EXP_ELO_045a: Tier-1 telemetry for config1's macro seat — the
            // committed lane and how stable it was.
            "playstyle": macro_playstyle
                .as_ref()
                .and_then(|p| p.lane.map(|a| format!("{a:?}"))),
            "playstyle_pivots_used": macro_playstyle.as_ref().map(|p| p.pivots_used),
            "playstyle_committed_turn": macro_playstyle.as_ref().and_then(|p| p.committed_turn),
            "playstyle_scores": macro_playstyle.as_ref().map(|p| p.last_scores.to_vec()),
            "macro_divergent_turns": macro_divergence.map(|(d, _)| d),
            "macro_planned_turns": macro_divergence.map(|(_, p)| p),
            "belief_calib": calib.as_ref().map(|c| c.rows.clone()),
            "mat_capital_turns": belief_mat.map(|(c, _, _)| c),
            "mat_units": belief_mat.map(|(_, u, _)| u),
            "class_picks": belief_gen.map(|(c, _, _)| c.to_vec()),
            "belief_repicks": belief_gen.map(|(_, r, _)| r),
            "intra_strips": belief_gen.map(|(_, _, s)| s),
        });
        // Drop the whole final board next to the summary. The server loads a
        // bare GameState, so the partner count around a hub can be counted off
        // the map rather than trusted. Unconditional: gating this on the model
        // holding a hub conditioned the board sample on the very thing an
        // economy audit is trying to measure.
        let sp = std::path::Path::new(dir)
            .join(format!("state_{}_{}.json", seed, if swap { "b" } else { "a" }));
        if let Ok(j) = serde_json::to_string(&game.state) {
            let _ = std::fs::write(&sp, j);
        }
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
        siege_config1,
        siege_config2,
        ns_config1,
        moves_config1,
        ns_config2,
        moves_config2,
        depth_config1,
        macro_divergence,
        belief_mat,
        belief_gen,
    }
}

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
    };
    let macro_params1 = MacroParams {
        sims: args.macro_sims1.unwrap_or(args.macro_sims),
        k: args.macro_k1.unwrap_or(args.macro_k),
        belief_mode: belief_mode1,
        shape_w: args.macro_shape_w1,
        leaf: args.macro_leaf1.unwrap_or(args.macro_leaf),
        ..base_params
    };
    let macro_params2 = MacroParams {
        sims: args.macro_sims2.unwrap_or(args.macro_sims),
        k: args.macro_k2.unwrap_or(args.macro_k),
        belief_mode: belief_mode2,
        shape_w: args.macro_shape_w2,
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
                            args.goal_script, args.goal_w_tree, macro_params1, macro_params2,
                            tribe_name, trace_tech, args.belief_calib,
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
