// Raised from the default 128: the large METRICS `json!` object expands
// recursively per field and EXP_ARCH_001 added value-composition fields.
#![recursion_limit = "256"]

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
use polyfish::ai::reward;
use polyfish::game::{Game, STARTING_OWNER_ID};
use polyfish::replayer::{ModReplay, ReplayPlayer, ReplayTurn};
use polyfish::states::{GameState, PlayerId};
use polyfish::types::MapSize;
use serde_json::json;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const HEURISTIC_PRIOR_W0: f32 = 0.5; // net & heur blended 50/50 at start
const HEURISTIC_PRIOR_DECAY: f32 = 0.97; // decays 0.5 -> 0.1 floor by ~iteration 47
const HEURISTIC_PRIOR_W_FLOOR: f32 = 0.1; // permanent behavioral floor, root + in-tree

// Absolute yardstick for the value target; 8K points is a good place to end a 30T game
const GOOD_BOT_FINAL_SCORE: f32 = 8000.0;
// How much to weight relative (vs opponent) vs absolute (vs yardstick) final outcome.
// 1.0 = pure relative (zero-sum). The value backup negates across every
// player-turn boundary (mcts_common.rs), which is only valid when
// v(mine) = -v(theirs); an absolute own-progress component is NOT
// antisymmetric — the opponent's progress isn't my loss — so any abs share
// gets systematically corrupted through EndTurn-crossing lines, worse as
// search deepens. The mirror-play "empty relative label" problem is fixed in
// the DATA instead: anchor games vs the greedy backend (--anchor-frac)
// make passivity actually lose, giving the relative label real signal.
const FINAL_OUTCOME_REL_W: f32 = 1.0;

// Weight of the TD(lambda) delta vs the final-outcome tail.
const TD_W: f32 = 0.7;
// Bootstrap/Monte-Carlo blend: center of mass of the geometric weights is
// 1/(1-LAMBDA_RETURN) turns (~5 at 0.8). Chosen to reach the turns-away
// horizon a village approach needs credit across, without drifting back
// toward the high-variance near-pure-MC regime the TD project escaped.
const LAMBDA_RETURN: f32 = 0.8;

// Forward credit window for the near-term value component, in game turns.
const NEAR_DELTA_TURNS: i32 = 4;
// The near-term norm scales with the game's economy: a saturating swing is
// ~15% of combined score over 4 turns, floored for the small opening turns.
// Percentages transfer across map sizes and skill brackets; point totals don't.
const NEAR_DELTA_NORM_FRAC: f32 = 0.15;
const NEAR_DELTA_NORM_FLOOR: f32 = 600.0;
// Weight of the near-term delta vs the final-outcome tail.
const NEAR_DELTA_W: f32 = 0.7;
// Weight of the relative (vs opponent) component within the near-term delta.
// 1.0 = pure relative, for the same negamax-antisymmetry reason as
// FINAL_OUTCOME_REL_W above. The signal against passivity comes from anchor
// games, not from a non-zero-sum label.
const NEAR_DELTA_REL_W: f32 = 1.0;

// Potential-based SPT shaping (economy credit). Φ = (my_spt - opp_spt) /
// default_max_spt, clamped ±1; each step's label gets SPT_SHAPE_W *
// (Φ at next own decision - Φ now). Within a turn the opponent's SPT is
// frozen, so a level-up harvest is credited the instant SPT rises instead
// of waiting for the score tail — dense credit the mirror-play relative
// score label nets to ~0. Antisymmetric (zero-sum backup stays valid) and
// potential-based (Ng et al. 1999), so it telescopes out of full returns
// and cannot change the optimal policy, only the credit assignment.
const SPT_SHAPE_W: f32 = 0.24;

// Ramp (in iterations) for β on σ(Q) in the exported policy targets:
// β = min(1, iteration/20). Early on the value head's Q ordering is noise
// that min-max rescaling amplifies to full strength, so π' corrodes the
// prior; let search re-ranking into the targets only as the head matures.
const POLICY_TARGET_Q_RAMP_ITERS: f32 = 20.0;

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
        Self::SampledFinish
        // if num_games >= 64 {
        //     Self::SampledFinish
        // } else if num_games > 32 {
        //     Self::Periodic
        // } else {
        //     Self::Full
        // }
    }
}

/// Which village-approach moment to capture a decision trace for (see
/// decision_trace.rs / find_village_trigger).
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum TraceTrigger {
    /// Unit 1 tile (Chebyshev) from an open village, deciding whether to Step toward it.
    Adjacent,
    /// Unit already standing on an open village, deciding whether to Capture.
    OnVillage,
}

/// First (unit, village) pair at exactly the distance `trigger` calls for,
/// among units for which the corresponding move (Step/Capture) could
/// actually be legal this ply. `open_villages` is the same incrementally
/// maintained set `play_single_game` already tracks for time-to-capture.
fn find_village_trigger(
    state: &GameState,
    pov: PlayerId,
    open_villages: &std::collections::HashSet<i32>,
    trigger: TraceTrigger,
) -> Option<(i32, i32)> {
    let tribe = state.tribes.get(&pov)?;
    let target_distance = match trigger {
        TraceTrigger::Adjacent => 1,
        TraceTrigger::OnVillage => 0,
    };
    for unit in &tribe.units {
        let eligible = match trigger {
            TraceTrigger::Adjacent => !unit.moved,
            TraceTrigger::OnVillage => !unit.moved && !unit.attacked,
        };
        if !eligible {
            continue;
        }
        for &village_idx in open_villages {
            let Some(village_tile) = state.tiles.get(&village_idx) else {
                continue;
            };
            if unit.coords.chebyshev_distance_to(&village_tile.coords) == target_distance {
                return Some((unit.coords.idx, village_idx));
            }
        }
    }
    None
}

/// Write one captured decision trace to `decision_traces/`, tagged with
/// enough metadata (iteration/game/turn/player/trigger tiles) to sample and
/// compare across games and training iterations. One file per decision —
/// safe under concurrent self-play actors without any shared-file locking.
fn write_decision_trace(
    trace: &polyfish::ai::decision_trace::DecisionTrace,
    iteration: usize,
    game_idx: usize,
    turn: i32,
    move_count: usize,
    player_id: PlayerId,
    trigger_unit_idx: i32,
    trigger_village_idx: i32,
) {
    let wrapped = json!({
        "iteration": iteration,
        "game_idx": game_idx,
        "turn": turn,
        "move_count": move_count,
        "player_id": player_id,
        "trigger_unit_idx": trigger_unit_idx,
        "trigger_village_idx": trigger_village_idx,
        "trace": trace,
    });
    let dir = std::path::Path::new("decision_traces");
    if let Err(e) = std::fs::create_dir_all(dir) {
        eprintln!("[trace] failed to create decision_traces/: {e}");
        return;
    }
    let path = dir.join(format!(
        "iter{iteration}_game{game_idx}_turn{turn}_p{player_id}.json"
    ));
    match serde_json::to_vec_pretty(&wrapped) {
        Ok(bytes) => {
            if let Err(e) = std::fs::write(&path, bytes) {
                eprintln!("[trace] failed to write {}: {e}", path.display());
            }
        }
        Err(e) => eprintln!("[trace] failed to serialize trace: {e}"),
    }
}

/// Curriculum — Tiny maps only, turn cap grows with (effective) iteration.
fn curriculum(iteration: usize) -> (MapSize, i32) {
    if iteration <= 10 {
        (MapSize::Tiny, 10)
    } else if iteration <= 20 {
        (MapSize::Tiny, 15)
    } else if iteration <= 30 {
        (MapSize::Tiny, 20)
    } else {
        (MapSize::Tiny, 45)
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

/// One recorded decision point. `my_score`/`opp_score`/`turn` are snapshotted
/// BEFORE this step's move executes. `root_value` is that same pre-move
/// state's post-search root value (see `GumbelMctsAgent::last_root_value`) —
/// the TD bootstrap target used by whichever *earlier* step's label lands on
/// this step as its "next decision" horizon.
struct HistoryStep {
    features: GameFeatures,
    policy: DecomposedPolicyData,
    player_id: PlayerId,
    my_score: i32,
    opp_score: i32,
    my_spt: i32,
    opp_spt: i32,
    turn: i32,
    root_value: Option<f32>,
}

/// The subset of `HistoryStep` the TD(lambda) label computation needs —
/// split out so `td_lambda_labels` is a pure, directly testable function
/// (no `GameFeatures`/policy tensors to fabricate in a unit test).
#[derive(Clone, Copy)]
struct LabelStep {
    player_id: PlayerId,
    turn: i32,
    my_score: i32,
    opp_score: i32,
    my_spt: i32,
    opp_spt: i32,
    root_value: Option<f32>,
}

impl From<&HistoryStep> for LabelStep {
    fn from(s: &HistoryStep) -> Self {
        LabelStep {
            player_id: s.player_id,
            turn: s.turn,
            my_score: s.my_score,
            opp_score: s.opp_score,
            my_spt: s.my_spt,
            opp_spt: s.opp_spt,
            root_value: s.root_value,
        }
    }
}

/// One player's turn-boundary checkpoint: the first decision of that turn
/// with a recorded root value, else that turn's first decision (root_value
/// stays `None`, so a bootstrap through it contributes 0.0 — matches the
/// original 1-step fallback exactly, just per-horizon instead of once).
struct Checkpoint {
    turn: i32,
    my: i32,
    opp: i32,
    root_value: Option<f32>,
}

fn checkpoints_by_player(history: &[LabelStep]) -> HashMap<PlayerId, Vec<Checkpoint>> {
    let mut out: HashMap<PlayerId, Vec<Checkpoint>> = HashMap::new();
    for step in history {
        let list = out.entry(step.player_id).or_default();
        match list.last_mut() {
            Some(c) if c.turn == step.turn => {
                if c.root_value.is_none() && step.root_value.is_some() {
                    c.my = step.my_score;
                    c.opp = step.opp_score;
                    c.root_value = step.root_value;
                }
            }
            _ => list.push(Checkpoint {
                turn: step.turn,
                my: step.my_score,
                opp: step.opp_score,
                root_value: step.root_value,
            }),
        }
    }
    out
}

/// TD(lambda) forward-view value target for every step in `history` (see
/// the doc comment on `LAMBDA_RETURN`). `final_scores` is keyed by
/// `PlayerId`. Output is aligned 1:1 with `history`.
fn td_lambda_labels(
    history: &[LabelStep],
    final_scores: &HashMap<i32, i32>,
    lambda: f32,
) -> Vec<f32> {
    let checkpoints = checkpoints_by_player(history);

    history
        .iter()
        .map(|step| {
            let my_final = final_scores.get(&step.player_id).copied().unwrap_or(0);
            let opp_final = final_scores
                .iter()
                .filter(|(id, _)| **id != step.player_id)
                .map(|(_, s)| *s)
                .next()
                .unwrap_or(0);
            let terminal_return =
                reward::normalized_reward(step.my_score, step.opp_score, my_final, opp_final);

            let empty = Vec::new();
            let ahead = checkpoints.get(&step.player_id).unwrap_or(&empty);
            let start = ahead.partition_point(|c| c.turn <= step.turn);

            let mut acc = 0.0f32;
            let mut remaining_weight = 1.0f32;
            for cp in &ahead[start..] {
                let r = reward::normalized_reward(step.my_score, step.opp_score, cp.my, cp.opp);
                let dt = (cp.turn - step.turn).max(0);
                let n_step_return = r + reward::GAMMA_TURN.powi(dt) * cp.root_value.unwrap_or(0.0);

                let w = remaining_weight * (1.0 - lambda);
                acc += w * n_step_return;
                remaining_weight *= lambda;
            }
            acc += remaining_weight * terminal_return;

            acc.clamp(-1.0, 1.0)
        })
        .collect()
}

/// Per-step potential-based SPT shaping deltas, aligned 1:1 with `history`
/// (see the doc comment on `SPT_SHAPE_W`). For step i of player p the delta
/// is Φ(p's next recorded decision) - Φ(step i); a player's last recorded
/// step gets 0.0 (terminal potential treated as its own — no phantom credit).
fn spt_shaping_deltas(history: &[LabelStep]) -> Vec<f32> {
    let norm = polyfish::states::default_max_spt() as f32;
    let phi = |s: &LabelStep| ((s.my_spt - s.opp_spt) as f32 / norm).clamp(-1.0, 1.0);

    let mut out = vec![0.0f32; history.len()];
    let mut next_phi: HashMap<PlayerId, f32> = HashMap::new();
    for (i, step) in history.iter().enumerate().rev() {
        let phi_now = phi(step);
        if let Some(&phi_next) = next_phi.get(&step.player_id) {
            out[i] = phi_next - phi_now;
        }
        next_phi.insert(step.player_id, phi_now);
    }
    out
}

/// Result from a single game - contains all data needed for training
struct GameResult {
    /// Index within this run's game batch; games 2k/2k+1 form a mirror pair.
    game_idx: usize,
    history: Vec<HistoryStep>,
    scores: HashMap<i32, i32>,
    final_cities: HashMap<i32, i32>,
    total_cities: i32,
    /// End-of-game tile owner per grid cell (row-major y*MAP_SIZE+x, 0 =
    /// neutral) — the aux ownership-head target, re-signed per sample POV.
    final_tile_owners: Vec<i32>,
    moves: usize,
    /// Mean KL(search visits ‖ policy prior) across this game's searched
    /// decisions — the per-game "search adds information" health metric.
    /// `None` when no decision ran a real search.
    policy_kl: Option<f32>,
    winner_score: i32,
    /// Winning player id (sole survivor, else score adjudication at the cap).
    winner_id: i32,
    /// Ended by elimination rather than score adjudication.
    decisive: bool,
    /// Seat (player id) the greedy anchor occupied; None for non-anchor games.
    anchor_seat: Option<i32>,
    recap: ModReplay,
    cap_ruins: usize,
    cap_villages: usize,
    cap_cities: usize,
    cap_capitals: usize,
    action_counts: HashMap<polyfish::types::MoveType, usize>,
    /// Move-type counts keyed by turn number, for the "move mix by turn"
    /// training-progress chart (see parse_metrics.py / dashboard).
    moves_by_turn: HashMap<i32, HashMap<polyfish::types::MoveType, usize>>,
    /// Combined (both players) tile-exploration and territory-ownership
    /// counts at game end, for diagnosing where score gains come from.
    revealed_tiles: i32,
    captured_tiles: i32,
    /// Turn by which 50%/80%/100% of the map's initial open villages (and
    /// ruins) had been captured by either player — how *directly* the AI
    /// seeks them out. Censored at max_turns when a game never gets there.
    villages_t2c_first: f32,
    villages_t2c_p50: f32,
    villages_t2c_p80: f32,
    villages_t2c_all: f32,
    ruins_t2c_p50: f32,
    ruins_t2c_p80: f32,
    ruins_t2c_all: f32,
    /// Mean tribe SPT sampled at the start of game turns 0, 5, 10, … (player 1
    /// to act, before any moves on that turn).
    spt_at_turn: HashMap<i32, f32>,
}

const SPT_MILESTONES: [i32; 7] = [0, 5, 10, 15, 20, 25, 30];

fn mean_tribe_spt(state: &polyfish::states::GameState) -> f32 {
    let n = state.tribes.len().max(1) as f32;
    state
        .tribes
        .values()
        .map(|t| polyfish::functions::get_tribe_spt(state, t) as f32)
        .sum::<f32>()
        / n
}

fn record_spt_at_turn_start(
    state: &polyfish::states::GameState,
    spt_at_turn: &mut HashMap<i32, f32>,
    next_idx: &mut usize,
) {
    if state.settings.current_player_turn_id != STARTING_OWNER_ID {
        return;
    }
    while *next_idx < SPT_MILESTONES.len() {
        let milestone = SPT_MILESTONES[*next_idx];
        if state.settings.turn < milestone {
            break;
        }
        if state.settings.turn == milestone {
            spt_at_turn.insert(milestone, mean_tribe_spt(state));
        }
        *next_idx += 1;
    }
}

/// Turn by which `frac` of `initial` capturables were taken, given the
/// chronological list of capture turns (`frac` 0.0 = the first capture).
/// `censor` (game length) when the game never reached that fraction or the
/// map had none to begin with.
fn t2c_turn(capture_turns: &[i32], initial: usize, frac: f64, censor: i32) -> f32 {
    if initial == 0 {
        return censor as f32;
    }
    let needed = ((initial as f64 * frac).ceil() as usize).max(1);
    capture_turns
        .get(needed - 1)
        .map(|&t| t as f32)
        .unwrap_or(censor as f32)
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
        let vs2 = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                &[opp_path],
                candle_core::DType::F32,
                device,
            )?
        };
        Arc::new(PolyZeroNet::new(vs2)?)
    } else {
        network1.clone()
    };
    Ok((network1, network2))
}

/// Play a single game and return the result
#[allow(clippy::too_many_arguments)]
/// Final-outcome tail from `p_id`'s POV: ±1 on elimination, else the
/// relative/absolute score blend. Must stay in sync with the value-label
/// semantics (this IS the single source of truth for that computation).
fn outcome_for(result: &GameResult, p_id: i32, reward_shaping: bool) -> f32 {
    if result.decisive {
        return if p_id == result.winner_id { 1.0 } else { -1.0 };
    }
    let my_final = result.scores.get(&p_id).copied().unwrap_or(0) as f32;
    let opp_final = result
        .scores
        .iter()
        .filter(|(id, _)| **id != p_id)
        .map(|(_, score)| *score as f32)
        .next()
        .unwrap_or(0.0);

    // Asymmetric reward shaping to fix P1 advantage
    let (mut my_adjusted, mut opp_adjusted) = (my_final, opp_final);
    if reward_shaping {
        let penalty = 0.05;
        if p_id == 1 {
            my_adjusted = my_final * (1.0 - penalty);
            opp_adjusted = opp_final * (1.0 + penalty);
        } else if p_id == 2 {
            my_adjusted = my_final * (1.0 + penalty);
            opp_adjusted = opp_final * (1.0 - penalty);
        }
    }

    // Normalized score differential, scaled into a useful training range.
    let combined_score = my_adjusted + opp_adjusted;
    let scaling_factor = 3.0;
    let relative_outcome = if combined_score > 0.0 {
        let ratio = (my_adjusted - opp_adjusted) / combined_score;
        (ratio * scaling_factor).clamp(-1.0, 1.0)
    } else {
        0.0
    };

    // Absolute value: final score vs fixed yardstick, not current scoreboard.
    // We use `my_adjusted` so that asymmetric reward shaping (if enabled)
    // applies to this absolute performance yardstick as well.
    let abs_outcome = (my_adjusted / GOOD_BOT_FINAL_SCORE).clamp(0.0, 1.0) * 2.0 - 1.0;
    (FINAL_OUTCOME_REL_W * relative_outcome + (1.0 - FINAL_OUTCOME_REL_W) * abs_outcome)
        .clamp(-1.0, 1.0)
}

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
    gamemode: u8,
    backend1: SearchBackend,
    backend2: SearchBackend,
    value_trust: Option<f32>,
    leaf_batch: Option<usize>,
    progress: ProgressMode,
    trace_villages: bool,
    trace_trigger: TraceTrigger,
    trace_max: usize,
    trace_counter: &AtomicUsize,
) -> Option<GameResult> {
    let (map_size, max_turns) = curriculum(iteration);

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
    game.state.settings.mode = polyfish::types::ModeType::from_repr(gamemode)
        .unwrap_or(polyfish::types::ModeType::Perfection);
    game.state.settings.max_turns = max_turns;
    game.post_load();

    // Time-to-capture tracking: snapshot the map's initial open villages and
    // ruins, then record the turn each one is taken (by either player).
    let mut open_villages: std::collections::HashSet<i32> = std::collections::HashSet::new();
    let mut open_ruins: std::collections::HashSet<i32> = std::collections::HashSet::new();
    for (&idx, s) in game.state.structures.iter() {
        let Some(s) = s else { continue };
        match s.structure_type {
            polyfish::types::StructureType::Village
                if game.state.tiles.get(&idx).map_or(false, |t| t.owner == 0) =>
            {
                open_villages.insert(idx);
            }
            polyfish::types::StructureType::Ruin => {
                open_ruins.insert(idx);
            }
            _ => {}
        }
    }
    let initial_villages = open_villages.len();
    let initial_ruins = open_ruins.len();
    let mut village_capture_turns: Vec<i32> = Vec::new();
    let mut ruin_capture_turns: Vec<i32> = Vec::new();

    let prior_w = (HEURISTIC_PRIOR_W0 * HEURISTIC_PRIOR_DECAY.powi(iteration as i32))
        .max(HEURISTIC_PRIOR_W_FLOOR);
    // One trust scalar drives β on σ(Q) in both the exported targets and the
    // search tree itself. --value-trust overrides the iteration ramp, which
    // saturates immediately on ITER_OFFSET-shifted runs.
    let q_target_w =
        value_trust.unwrap_or_else(|| (iteration as f32 / POLICY_TARGET_Q_RAMP_ITERS).min(1.0));

    // Create two agents (they might share the same network, or be different)
    let mut agent1 = Brain::with_backend(eval1, mcts_iters, backend1)
        .with_prior_heuristic_weight(prior_w)
        .with_policy_target_q_weight(q_target_w)
        .with_tree_q_weight(q_target_w);
    let mut agent2 = Brain::with_backend(eval2, mcts_iters, backend2)
        .with_prior_heuristic_weight(prior_w)
        .with_policy_target_q_weight(q_target_w)
        .with_tree_q_weight(q_target_w);

    if let Some(b) = leaf_batch {
        agent1 = agent1.with_leaf_batch(b);
        agent2 = agent2.with_leaf_batch(b);
    }

    let initial_state = game.state.clone();
    let mut flat_recap: Vec<(i32, i32, serde_json::Value)> = Vec::new();

    let mut cap_ruins = 0;
    let mut cap_villages = 0;
    let mut cap_cities = 0;
    let mut cap_capitals = 0;

    // Game Loop
    let mut game_history: Vec<HistoryStep> = Vec::new();
    let mut action_counts: HashMap<polyfish::types::MoveType, usize> = HashMap::new();
    let mut moves_by_turn: HashMap<i32, HashMap<polyfish::types::MoveType, usize>> = HashMap::new();

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

    let mut spt_at_turn: HashMap<i32, f32> = HashMap::new();
    let mut next_spt_milestone = 0usize;

    let mut move_count = 0;
    let mut traced_in_this_game = false;
    let mut kl_sum = 0.0f32;
    let mut kl_n = 0u32;
    while !polyfish::functions::is_game_over(&game.state) {
        record_spt_at_turn_start(&game.state, &mut spt_at_turn, &mut next_spt_milestone);

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

        let trigger_info = if trace_villages
            && !traced_in_this_game
            && trace_counter.load(Ordering::Relaxed) < trace_max
        {
            find_village_trigger(&game.state, pov, &open_villages, trace_trigger)
        } else {
            None
        };
        if trigger_info.is_some() {
            current_agent.request_trace();
        }

        let (best_move, move_visits) = current_agent.think_decomposed(&mut game, move_count);
        // The search that just ran was for the CURRENT (pre-move) state, so
        // this is that state's own root value — the TD bootstrap target for
        // whichever earlier step's label lands here as its "next decision".
        let root_value = current_agent.last_root_value();
        if let Some(kl) = current_agent.last_search_kl() {
            kl_sum += kl;
            kl_n += 1;
        }

        if let Some((trigger_unit_idx, trigger_village_idx)) = trigger_info {
            if let Some(trace) = current_agent.take_trace() {
                if trace_counter.fetch_add(1, Ordering::Relaxed) < trace_max {
                    write_decision_trace(
                        &trace,
                        iteration,
                        game_idx,
                        game.state.settings.turn,
                        move_count,
                        pov,
                        trigger_unit_idx,
                        trigger_village_idx,
                    );
                }
                traced_in_this_game = true;
            }
        }

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

        let record_sample = !move_visits.is_empty();

        if let Some(m) = best_move {
            let m_type = m.move_type();
            *action_counts.entry(m_type).or_insert(0) += 1;
            *moves_by_turn
                .entry(game.state.settings.turn)
                .or_default()
                .entry(m_type)
                .or_insert(0) += 1;

            if m_type == polyfish::types::MoveType::Capture {
                if let Ok(src) = m.source_idx() {
                    let idx = src as i32;

                    let struct_opt = game.state.structures.get(&idx).and_then(|s| s.as_ref());
                    let is_ruin = struct_opt.map(|s| s.structure_type)
                        == Some(polyfish::types::StructureType::Ruin);
                    let is_village = struct_opt.map(|s| s.structure_type)
                        == Some(polyfish::types::StructureType::Village);
                    let owner = game.state.tiles.get(&idx).map(|t| t.owner).unwrap_or(0);
                    let is_capital = game
                        .state
                        .tiles
                        .get(&idx)
                        .map(|t| t.capital_of > 0)
                        .unwrap_or(false);

                    if is_ruin {
                        cap_ruins += 1;
                    } else if is_village {
                        if owner == 0 {
                            cap_villages += 1;
                        } else if owner != pov {
                            if is_capital {
                                cap_capitals += 1;
                            } else {
                                cap_cities += 1;
                            }
                        }
                    }

                    if open_villages.remove(&idx) {
                        village_capture_turns.push(game.state.settings.turn);
                    } else if open_ruins.remove(&idx) {
                        ruin_capture_turns.push(game.state.settings.turn);
                    }
                }
            }

            flat_recap.push((
                game.state.settings.turn,
                game.state.settings.current_player_turn_id,
                m.serialize(),
            ));
            // Snapshot scores/SPT at this moment (pre-move) for the TD label.
            let (my_score_now, opp_score_now) = reward::score_snapshot(&game.state, pov);
            let (my_spt_now, opp_spt_now) = reward::spt_snapshot(&game.state, pov);
            if record_sample {
                game_history.push(HistoryStep {
                    features: state_t,
                    policy: policy_data,
                    player_id: pov,
                    my_score: my_score_now,
                    opp_score: opp_score_now,
                    my_spt: my_spt_now,
                    opp_spt: opp_spt_now,
                    turn: game.state.settings.turn,
                    root_value,
                });
            }
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

    let mut final_tile_owners = vec![0i32; features::MAP_SIZE * features::MAP_SIZE];
    let size = game.state.settings.size;
    for (idx, tile) in &game.state.tiles {
        let x = (idx % size) as usize;
        let y = (idx / size) as usize;
        if x < features::MAP_SIZE && y < features::MAP_SIZE {
            final_tile_owners[y * features::MAP_SIZE + x] = tile.owner;
        }
    }
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

    let mut final_cities = HashMap::new();
    let mut total_cities = 0;
    for (id, t) in &game.state.tribes {
        final_cities.insert(*id, t.cities.len() as i32);
        total_cities += t.cities.len() as i32;
    }

    Some(GameResult {
        game_idx,
        history: game_history,
        scores,
        final_cities,
        total_cities,
        final_tile_owners,
        moves: move_count,
        revealed_tiles,
        captured_tiles,
        villages_t2c_first: t2c_turn(&village_capture_turns, initial_villages, 0.0, max_turns),
        villages_t2c_p50: t2c_turn(&village_capture_turns, initial_villages, 0.5, max_turns),
        villages_t2c_p80: t2c_turn(&village_capture_turns, initial_villages, 0.8, max_turns),
        villages_t2c_all: t2c_turn(&village_capture_turns, initial_villages, 1.0, max_turns),
        ruins_t2c_p50: t2c_turn(&ruin_capture_turns, initial_ruins, 0.5, max_turns),
        ruins_t2c_p80: t2c_turn(&ruin_capture_turns, initial_ruins, 0.8, max_turns),
        ruins_t2c_all: t2c_turn(&ruin_capture_turns, initial_ruins, 1.0, max_turns),
        spt_at_turn,
        winner_score,
        winner_id,
        decisive: is_decisive,
        policy_kl: (kl_n > 0).then(|| kl_sum / kl_n as f32),
        anchor_seat: match (
            backend1 == SearchBackend::Greedy,
            backend2 == SearchBackend::Greedy,
        ) {
            (true, false) => Some(1),
            (false, true) => Some(2),
            _ => None,
        },
        recap: ModReplay {
            game_state: initial_state,
            turns: group_recap(flat_recap),
        },
        cap_ruins,
        cap_villages,
        cap_cities,
        cap_capitals,
        action_counts,
        moves_by_turn,
    })
}

const ANCHOR_STATE_PATH: &str = ".anchor_state.json";
const ANCHOR_WINRATE_WINDOW: usize = 3;
const ANCHOR_STATE_KEEP: usize = 50;

/// Model-vs-anchor outcome of one self_play run, persisted across
/// invocations so the anchor gate can act on measured strength.
#[derive(serde::Serialize, serde::Deserialize)]
struct AnchorRecord {
    iteration: usize,
    games: usize,
    model_wins: f32,
}

/// Records at or beyond the current iteration are dropped on load, so reruns
/// and fresh --reset campaigns self-heal without manual state cleanup.
fn load_anchor_history(current_iteration: usize) -> Vec<AnchorRecord> {
    let hist = match std::fs::read_to_string(ANCHOR_STATE_PATH) {
        Ok(s) => match serde_json::from_str::<Vec<AnchorRecord>>(&s) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Warning: Failed to parse anchor history: {}", e);
                Vec::new()
            }
        },
        Err(_) => Vec::new(),
    };
    hist.into_iter()
        .filter(|r| r.iteration < current_iteration)
        .collect()
}

fn append_anchor_record(current_iteration: usize, games: usize, model_wins: f32) {
    let mut hist = load_anchor_history(current_iteration + 1);
    hist.retain(|r| r.iteration != current_iteration);
    hist.push(AnchorRecord {
        iteration: current_iteration,
        games,
        model_wins,
    });
    let skip = hist.len().saturating_sub(ANCHOR_STATE_KEEP);
    if let Ok(s) = serde_json::to_string(&hist[skip..]) {
        let temp_path = format!("{}.tmp.{}", ANCHOR_STATE_PATH, std::process::id());
        if let Err(e) = std::fs::write(&temp_path, s) {
            eprintln!("Warning: Failed to write anchor history: {}", e);
        } else if let Err(e) = std::fs::rename(&temp_path, ANCHOR_STATE_PATH) {
            eprintln!("Warning: Failed to rename anchor history: {}", e);
        }
    }
}

/// Rolling model win rate over the last ANCHOR_WINRATE_WINDOW runs that
/// actually played anchor games. None until any measurement exists.
fn rolling_anchor_winrate(current_iteration: usize) -> Option<f32> {
    let hist = load_anchor_history(current_iteration);
    let recent: Vec<&AnchorRecord> = hist
        .iter()
        .rev()
        .filter(|r| r.games > 0)
        .take(ANCHOR_WINRATE_WINDOW)
        .collect();
    let games: usize = recent.iter().map(|r| r.games).sum();
    let wins: f32 = recent.iter().map(|r| r.model_wins).sum();
    (games > 0).then(|| wins / games as f32)
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
        #[arg(long, default_value_t = 2)]
        gamemode: u8,

        /// Number of games to play
        #[arg(long, default_value_t = 10)]
        num_games: usize,

        /// MCTS iterations per move
        #[arg(long, default_value_t = 64)]
        mcts_iters: usize,

        /// Optional opponent model path (if not set, plays against self)
        #[arg(long)]
        opponent: Option<String>,

        /// Fraction of games (0..1) played against the network-free Greedy
        /// backend as an anchor opponent (seat alternates between anchor
        /// games). Anchor games break mirror-play symmetry: greedy attacks
        /// and ELIMINATES a passive net, so the relative value label carries
        /// an anti-passivity gradient and the corpus contains real decisive
        /// wins/losses. The anchor side's data is recorded too (fresh teacher
        /// data, same as the BC corpus). This is a bootstrap: after
        /// --anchor-hold-iters the effective fraction is gated on the
        /// measured win rate vs greedy (see .anchor_state.json), annealing
        /// smoothly to --anchor-probe-frac once the model reliably beats it.
        /// Mutually exclusive with --opponent.
        #[arg(long, default_value_t = 0.0)]
        anchor_frac: f32,

        /// Iterations to hold the anchor at full --anchor-frac before the
        /// win-rate gate takes over.
        #[arg(long, default_value_t = 10)]
        anchor_hold_iters: usize,

        /// Rolling win rate vs greedy at which the anchor counts as
        /// graduated: effective frac anneals linearly from full at 50% down
        /// to --anchor-probe-frac at this value. Must be in (0.5, 1].
        #[arg(long, default_value_t = 0.55)]
        anchor_graduate_winrate: f32,

        /// Residual anchor fraction kept after graduation so the win rate
        /// stays measured and the anchor returns on regression (floored to
        /// >= 1 game per run whenever the gate output is nonzero).
        #[arg(long, default_value_t = 0.05)]
        anchor_probe_frac: f32,

        /// Value-head trust in [0,1]: β on σ(completed-Q) both inside the
        /// search tree and in exported policy targets. Overrides the
        /// iteration-based ramp (min(1, iteration/20)), which saturates
        /// uselessly when ITER_OFFSET-shifted runs start at high effective
        /// iterations. Drive this from the loop script (run-relative ramp or
        /// measured value-head calibration).
        #[arg(long)]
        value_trust: Option<f32>,

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

        /// Mirror-relative value labels: games 2k/2k+1 replay the same seed
        /// with seats swapped; each game's outcome tail is blended toward the
        /// pair mean (fully at game start, not at all by game end), canceling
        /// map/spawn/seat luck from the label. Disable: --mirror-labels=false.
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        mirror_labels: bool,

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
        /// locality is preserved. Never use >1 on tch: measured (2026-07-05)
        /// that 2 tch shards HALVE throughput (156.6 moves/s @ 1 vs 83.3 @ 2)
        /// because libtorch's MPS backend serializes across threads at the
        /// C++ level. candle rejects >1 (Metal corrupts when >1 thread
        /// encodes on the same device — see the bug_handoff invariant in
        /// eval_server.rs). On metal, 3 shards × 2 workers is the measured
        /// best (~610–650 moves/s, see expert_boost_throughput.md).
        /// 0 = auto (3 on metal, 1 on tch/candle). Overridable.
        #[arg(long, default_value_t = 0)]
        eval_servers: usize,

        /// Metal backend only: pipelined GPU worker threads per eval server.
        /// Each owns its own MTLCommandQueue, so N coalesced batches can be
        /// in flight on the GPU while the coalescer collects the next one —
        /// unlike --eval-servers sharding, the batch stream and cache stay
        /// unified. Ignored by candle/tch.
        #[arg(long, default_value_t = 2)]
        eval_workers: usize,

        /// Capture MCTS decision traces at village-approach moments (see
        /// decision_trace.rs) to decision_traces/*.json. Forces a fresh
        /// (non-reused) tree build only for the traced decision, and only
        /// once per game (first trigger), so normal runs are unaffected.
        #[arg(long, default_value_t = false)]
        trace_villages: bool,

        /// Which village-approach moment to trace. Ignored unless
        /// --trace-villages.
        #[arg(long, value_enum, default_value_t = TraceTrigger::Adjacent)]
        trace_trigger: TraceTrigger,

        /// Max decision-trace JSON files written across the whole run.
        /// Ignored unless --trace-villages.
        #[arg(long, default_value_t = 20)]
        trace_max: usize,
    }

    let args = Args::parse();

    if args.anchor_frac > 0.0 && args.opponent.is_some() {
        anyhow::bail!("--anchor-frac and --opponent are mutually exclusive");
    }
    if !(0.0..=1.0).contains(&args.anchor_frac) {
        anyhow::bail!("--anchor-frac must be in [0, 1]");
    }
    if args.anchor_frac > 0.0 {
        if !(args.anchor_graduate_winrate > 0.5 && args.anchor_graduate_winrate <= 1.0) {
            anyhow::bail!("--anchor-graduate-winrate must be in (0.5, 1]");
        }
        if !(0.0..=1.0).contains(&args.anchor_probe_frac) {
            anyhow::bail!("--anchor-probe-frac must be in [0, 1]");
        }
    }
    if let Some(t) = args.value_trust {
        if !(0.0..=1.0).contains(&t) {
            anyhow::bail!("--value-trust must be in [0, 1]");
        }
    }

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
        SearchBackendArg::StateDiffGreedy => SearchBackend::StateDiffGreedy,
        SearchBackendArg::Random => SearchBackend::Random,
    };

    // Select device: CUDA (NVIDIA) > Metal (macOS) > CPU, unless overridden via POLYFISH_DEVICE
    let device = match std::env::var("POLYFISH_DEVICE").as_deref() {
        Ok("cpu") => Device::Cpu,
        Ok("metal") => Device::metal_if_available(0)?,
        Ok("cuda") => Device::cuda_if_available(0)?,
        _ => match Device::cuda_if_available(0) {
            Ok(Device::Cpu) | Err(_) => Device::metal_if_available(0).unwrap_or(Device::Cpu),
            Ok(d) => d,
        },
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

    use polyfish::ai::eval_backend::{resolve_eval_backend_kind, EvalBackendKind};

    // Resolve the eval backend: explicit --eval-backend, else auto (metal
    // when compiled in, else tch when compiled in, else candle).
    let eval_backend_kind = resolve_eval_backend_kind(args.eval_backend.as_str())?;
    // Resolve shard count. We default to the best measured throughput
    let eval_servers = match args.eval_servers {
        0 => {
            if eval_backend_kind == EvalBackendKind::Metal {
                3
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
    use polyfish::ai::eval_backend::build_backend_specs;

    let p1_path = "model.safetensors";
    let p2_path = args.opponent.as_deref().unwrap_or("model.safetensors");
    let p1_specs = build_backend_specs(eval_backend_kind, eval_servers, p1_path, &network1);
    let has_opponent = args.opponent.is_some();
    let p2_specs = if has_opponent {
        build_backend_specs(eval_backend_kind, eval_servers, p2_path, &network2)
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
    // Greedy-anchor gate: hold the full --anchor-frac for the first
    // --anchor-hold-iters, then scale by the measured rolling win rate vs
    // greedy (.anchor_state.json, written by previous runs). Full frac while
    // the model still loses (<= 50%), annealing linearly down to
    // --anchor-probe-frac at --anchor-graduate-winrate; probe games keep the
    // win rate measured so the anchor comes back on regression. Iteration is
    // EFF_ITER, the same clock as the other curriculum gates.
    let anchor_winrate = rolling_anchor_winrate(args.iteration);
    let anchor_frac = if args.anchor_frac <= 0.0
        || args.opponent.is_some()
        || args.iteration <= args.anchor_hold_iters
    {
        args.anchor_frac
    } else {
        match anchor_winrate {
            None => args.anchor_frac,
            Some(wr) => {
                let t = ((wr - 0.5) / (args.anchor_graduate_winrate - 0.5)).clamp(0.0, 1.0);
                let probe = args.anchor_probe_frac.min(args.anchor_frac);
                args.anchor_frac + t * (probe - args.anchor_frac)
            }
        }
    };
    // Floor a nonzero gate output to >= 1 mirror pair this run, else small
    // probe fractions round to zero games and the win rate silently goes stale.
    let anchor_frac = if anchor_frac > 0.0 {
        anchor_frac.max(1.0 / args.num_games.div_ceil(2).max(1) as f32)
    } else {
        anchor_frac
    };
    if args.anchor_frac > 0.0 && args.opponent.is_none() {
        match anchor_winrate {
            Some(wr) => println!(
                "[Self-Play] anchor gate: rolling winrate vs greedy {:.0}% -> effective anchor_frac {:.3} (hold <= iter {}, graduate at {:.0}%, probe {:.3})",
                wr * 100.0,
                anchor_frac,
                args.anchor_hold_iters,
                args.anchor_graduate_winrate * 100.0,
                args.anchor_probe_frac,
            ),
            None => println!(
                "[Self-Play] anchor gate: no measurements yet -> full anchor_frac {:.3}",
                anchor_frac
            ),
        }
    }

    let match_label = match &args.opponent {
        Some(opp) => format!("league vs {opp}"),
        None if anchor_frac > 0.0 => {
            format!(
                "self-play + {:.0}% greedy-anchor games",
                anchor_frac * 100.0
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
        SearchBackend::StateDiffGreedy => "State-diff exp greedy (no NN, no search)".to_string(),
        SearchBackend::Random => "Uniform random (no NN, no search)".to_string(),
    };
    println!(
        "[Self-Play] {match_label}: {} games, {} mcts-iters, {search_label}, tribes {tribe_label} | eval {backend_label} | {eval_servers} shard(s) cache={per_shard_cache:?} workers={} | {num_actors} actors max_batch={} coalesce_us={} leaf_batch={:?} | device {:?} (CANDLE_METAL_COMPUTE_PER_BUFFER={metal_compute_per_buffer})",
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
            scope.spawn(move || {
                loop {
                    let i = job_counter.fetch_add(1, Ordering::Relaxed);
                    if i >= args.num_games {
                        break;
                    }

                    // Mirror pairs: games 2k and 2k+1 share a seed (same map
                    // and tribes) with controllers on swapped seats, so the
                    // value labels can cancel per-map/spawn/seat luck.
                    let seed = (base_seed + (i / 2) as u64) as i64;
                    let swap_players = i % 2 == 1; // Swap every other game
                    let (p1_net, p2_net, p1_eval, p2_eval) = if swap_players {
                        (&**network2, &**network1, eval2, eval1)
                    } else {
                        (&**network1, &**network2, eval1, eval2)
                    };

                    // Anchor games are scheduled per PAIR (evenly spread at
                    // rate anchor_frac) so both halves of a mirror pair are
                    // anchor games, greedy taking opposite seats in each half.
                    let pair_idx = i / 2;
                    let is_anchor = anchor_frac > 0.0
                        && (((pair_idx + 1) as f32) * anchor_frac).floor() as usize
                            > ((pair_idx as f32) * anchor_frac).floor() as usize;
                    let (backend_seat1, backend_seat2) = if is_anchor {
                        if i % 2 == 0 {
                            (SearchBackend::Greedy, backend)
                        } else {
                            (backend, SearchBackend::Greedy)
                        }
                    } else {
                        (backend, backend)
                    };

                    // Sample this game's own tribe pair, seeded off its game
                    // seed so runs stay reproducible while each game gets a
                    // distinct matchup.
                    use rand::SeedableRng;
                    let mut tribe_rng = rand::rngs::StdRng::seed_from_u64(seed as u64);
                    let (t1, t2) =
                        pick_tribes(&mut tribe_rng, all_tribes, &args.tribe1, &args.tribe2);
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
                                    "[Progress] {}/{} games complete (game {} — {} moves, winner score {}, decisive: {}, captures: {}, final spt: {})",
                                    done,
                                    args.num_games,
                                    i,
                                    result.moves,
                                    result.winner_score,
                                    if result.decisive {
                                        "yes"
                                    } else {
                                        "no"
                                    },
                                    result.cap_villages,
                                    result
                                        .spt_at_turn
                                        .iter()
                                        .max_by_key(|(turn, _)| *turn)
                                        .map(|(_, spt)| *spt)
                                        .unwrap_or(0.0),
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
        Err(_) => {
            panic!("BUG: actor threads still hold a results_mutex reference after scope exit")
        }
    };

    let games_duration = games_start.elapsed();
    println!(
        "🚀 Game generation completed in: {:.2}s ({} games)",
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
    if let Some(p2) = p2_servers.as_ref() {
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

    // Aggregate results
    let mut collected_spatial_maps: Vec<Tensor> = Vec::new();
    let mut collected_player_states: Vec<Tensor> = Vec::new();

    // Decomposed policy targets (7 heads)
    let mut collected_action_type: Vec<Vec<f32>> = Vec::new();
    let mut collected_source_spatial: Vec<Vec<f32>> = Vec::new();
    let mut collected_target_spatial: Vec<Vec<f32>> = Vec::new();
    let mut collected_option: Vec<Vec<f32>> = Vec::new();

    let mut collected_values: Vec<f32> = Vec::new();
    let mut collected_progress: Vec<f32> = Vec::new();
    let mut collected_ownership: Vec<Vec<f32>> = Vec::new();

    let mut total_score = 0;
    let mut kl_game_sum = 0.0f64;
    let mut kl_game_n = 0u32;
    let mut max_score = 0;
    let mut best_recap: Option<ModReplay> = None;
    let mut total_moves = 0;

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
    let mut total_revealed_tiles: i64 = 0;
    let mut total_captured_tiles: i64 = 0;
    let mut total_t2c = [0.0f64; 7]; // villages first/p50/p80/all, ruins p50/p80/all
    let mut spt_sums: HashMap<i32, f64> = HashMap::new();
    let mut spt_counts: HashMap<i32, u32> = HashMap::new();

    let mut total_moves_by_turn: HashMap<i32, HashMap<polyfish::types::MoveType, usize>> =
        HashMap::new();

    let mut decisive_games = 0usize;
    let mut anchor_games_n = 0usize;
    let mut anchor_model_wins = 0.0f32;

    // EXP_ARCH_001 drowning meter: absolute contribution of each value-label
    // term, summed over every step, so the logs show how much of the signal is
    // genuine win/loss vs dense score-shaping.
    let mut vlab_td_abs = 0.0f64;
    let mut vlab_wl_abs = 0.0f64; // tail on decisive games = true win/loss
    let mut vlab_scoretail_abs = 0.0f64; // tail on timeouts = score tiebreak
    let mut vlab_spt_abs = 0.0f64;
    let mut vlab_steps = 0u64;

    // P1-POV outcome per game, keyed by game_idx, for the mirror-pair label
    // adjustment (a game's partner is game_idx ^ 1; absent if it panicked).
    let pair_outcomes: HashMap<usize, (f32, f32)> = if args.mirror_labels {
        results
            .iter()
            .map(|r| {
                (
                    r.game_idx,
                    (
                        outcome_for(r, 1, args.reward_shaping),
                        outcome_for(r, 2, args.reward_shaping),
                    ),
                )
            })
            .collect()
    } else {
        HashMap::new()
    };
    if args.mirror_labels {
        let paired = results
            .iter()
            .filter(|r| pair_outcomes.contains_key(&(r.game_idx ^ 1)))
            .count();
        eprintln!(
            "[MirrorLabels] {}/{} games have a live mirror partner",
            paired,
            results.len()
        );
    }

    for result in results {
        if result.decisive {
            decisive_games += 1;
        }
        if let Some(anchor_pid) = result.anchor_seat {
            anchor_games_n += 1;
            if result.winner_id != 0 && result.winner_id != anchor_pid {
                anchor_model_wins += 1.0;
            }
        }
        total_score += result.winner_score;
        total_moves += result.moves;
        if let Some(kl) = result.policy_kl {
            kl_game_sum += kl as f64;
            kl_game_n += 1;
        }
        total_revealed_tiles += result.revealed_tiles as i64;
        total_captured_tiles += result.captured_tiles as i64;
        for (&turn, &spt) in &result.spt_at_turn {
            *spt_sums.entry(turn).or_default() += spt as f64;
            *spt_counts.entry(turn).or_default() += 1;
        }
        for (acc, v) in total_t2c.iter_mut().zip([
            result.villages_t2c_first,
            result.villages_t2c_p50,
            result.villages_t2c_p80,
            result.villages_t2c_all,
            result.ruins_t2c_p50,
            result.ruins_t2c_p80,
            result.ruins_t2c_all,
        ]) {
            *acc += v as f64;
        }
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

        let label_steps: Vec<LabelStep> = result.history.iter().map(LabelStep::from).collect();
        let td_deltas = td_lambda_labels(&label_steps, final_scores, LAMBDA_RETURN);
        let spt_shaping = spt_shaping_deltas(&label_steps);

        // EXP_ARCH_001: reward WINNING, not score-hoarding. Winner = sole
        // survivor (elimination) or, on timeout, the score leader — already
        // adjudicated at game end. `decisive` marks a real elimination; on a
        // timeout we fall back to the score tail below.
        let game_decisive = result.decisive;
        let z_p1 = outcome_for(&result, 1, args.reward_shaping);
        let z_p2 = outcome_for(&result, 2, args.reward_shaping);
        let partner_outcomes = pair_outcomes.get(&(result.game_idx ^ 1)).copied();
        let hist_len = result.history.len();
        // Env-tunable so the win/loss tail vs dense score-shaping mix can be
        // swept without a recompile (default = const TD_W).
        let td_w: f32 = std::env::var("TD_W")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(TD_W);

        for (step_idx, step) in result.history.into_iter().enumerate() {
            let HistoryStep {
                features,
                policy: policy_data,
                player_id: p_id,
                ..
            } = step;
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

            // The long-horizon tail (EXP_ARCH_001): win(+1)/loss(-1) on a
            // decisive game, score differential on a timeout (outcome_for).
            let own_outcome = if p_id == 1 { z_p1 } else { z_p2 };

            // Mirror-relative adjustment: the partner game replays the same
            // map with seats swapped, so the pair mean of this controller's
            // outcomes estimates map/spawn/seat luck-free skill. Early states
            // (where the pair hasn't diverged) take the pair mean; late states
            // keep the true own outcome. Spawn-luck wins thus stop training
            // the early game as if they were earned.
            let final_outcome = match partner_outcomes {
                Some((pz1, pz2)) => {
                    let z_counterpart = if p_id == 1 { pz2 } else { pz1 };
                    let pair_mean = 0.5 * (own_outcome + z_counterpart);
                    let u = if hist_len > 1 {
                        step_idx as f32 / (hist_len - 1) as f32
                    } else {
                        1.0
                    };
                    (1.0 - u) * pair_mean + u * own_outcome
                }
                None => own_outcome,
            };

            let value = if args.reward_shaping {
                // TD delta carries dense per-action credit; the final-outcome
                // tail now carries the WIN/LOSS signal; the SPT potential delta
                // adds dense economy credit (see SPT_SHAPE_W). Lower TD_W (env)
                // to let the win/loss tail dominate the dense score shaping.
                (td_w * td_deltas[step_idx]
                    + (1.0 - td_w) * final_outcome
                    + SPT_SHAPE_W * spt_shaping[step_idx])
                    .clamp(-1.0, 1.0)
            } else {
                final_outcome.clamp(-1.0, 1.0)
            };

            if args.reward_shaping {
                vlab_td_abs += (td_w * td_deltas[step_idx]).abs() as f64;
                let tail = ((1.0 - td_w) * final_outcome).abs() as f64;
                if game_decisive {
                    vlab_wl_abs += tail;
                } else {
                    vlab_scoretail_abs += tail;
                }
                vlab_spt_abs += (SPT_SHAPE_W * spt_shaping[step_idx]).abs() as f64;
                vlab_steps += 1;
            }

            collected_values.push(value);

            let my_final_cities = result.final_cities.get(&p_id).copied().unwrap_or(0) as f32;
            let total_cities = result.total_cities as f32;
            let progress_target = if total_cities > 0.0 {
                (my_final_cities / total_cities).clamp(0.0, 1.0) * 2.0 - 1.0
            } else {
                -1.0
            };
            collected_progress.push(progress_target);

            // Final ownership re-signed to this sample's POV, matching the
            // CH_TILE_OWNER input convention (+1 mine / -1 enemy / 0 neutral).
            collected_ownership.push(
                result
                    .final_tile_owners
                    .iter()
                    .map(|&o| {
                        if o == p_id {
                            1.0
                        } else if o != 0 {
                            -1.0
                        } else {
                            0.0
                        }
                    })
                    .collect(),
            );
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
    let avg_cap_ruins = total_cap_ruins as f32 / args.num_games as f32;
    let avg_cap_villages = total_cap_villages as f32 / args.num_games as f32;
    let avg_cap_cities = total_cap_cities as f32 / args.num_games as f32;
    let avg_cap_capitals = total_cap_capitals as f32 / args.num_games as f32;
    let avg_harvests = total_harvests as f32 / args.num_games as f32;
    let avg_builds = total_builds as f32 / args.num_games as f32;
    let avg_research = total_research as f32 / args.num_games as f32;
    let avg_attacks = total_attacks as f32 / args.num_games as f32;
    let avg_revealed_tiles = total_revealed_tiles as f32 / args.num_games as f32;
    let avg_captured_tiles = total_captured_tiles as f32 / args.num_games as f32;

    let avg_spt_at = |turn: i32| -> f32 {
        let c = spt_counts.get(&turn).copied().unwrap_or(0);
        if c == 0 {
            0.0
        } else {
            (spt_sums.get(&turn).copied().unwrap_or(0.0) / c as f64) as f32
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

    let games_file = if collected_spatial_maps.is_empty() {
        String::new()
    } else {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        // Trace runs are diagnostics, not training data: quarantine their
        // games under a prefix the training loop's games_* glob won't match
        // (stray trace games previously leaked into training — see notes.md).
        if args.trace_villages {
            format!("trace_games_{timestamp}.safetensors")
        } else {
            format!("games_{timestamp}.safetensors")
        }
    };

    // Stack and save
    if !collected_spatial_maps.is_empty() {
        let total_steps = collected_spatial_maps.len();
        let timestamp = games_file
            .strip_prefix("trace_")
            .unwrap_or(&games_file)
            .strip_prefix("games_")
            .and_then(|s| s.strip_suffix(".safetensors"))
            .unwrap_or("0");

        let spatial_dim = features::NUM_CHANNELS * features::MAP_SIZE * features::MAP_SIZE;
        let player_dim = features::RawFeatures::PLAYER_STATE_DIM;

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
        let progress_tensor = Tensor::from_vec(collected_progress, (total_steps, 1), &device)?;
        let ownership_tensor = Tensor::from_vec(
            flatten_vec(collected_ownership),
            (total_steps, spatial_logit_dim),
            &device,
        )?;

        let mut tensors = HashMap::new();
        tensors.insert("spatial_maps".to_string(), spatial_maps_tensor);
        tensors.insert("player_states".to_string(), player_states_tensor);

        tensors.insert("action_type".to_string(), action_tensor);
        tensors.insert("source_spatial".to_string(), source_tensor);
        tensors.insert("target_spatial".to_string(), target_tensor);
        tensors.insert("move_option".to_string(), option_tensor);

        tensors.insert("values".to_string(), values_tensor);
        tensors.insert("progress".to_string(), progress_tensor);
        tensors.insert("ownership".to_string(), ownership_tensor);

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

    // EXP_ARCH_001 drowning meter (computed here, not inline, to keep the big
    // json! macro under its recursion limit). vlab_wl_share is the headline:
    // fraction of the value-label magnitude that is genuine win/loss.
    let vlab_total = vlab_td_abs + vlab_wl_abs + vlab_scoretail_abs + vlab_spt_abs;
    let vlab_wl_share = if vlab_total > 0.0 {
        (vlab_wl_abs / vlab_total) as f32
    } else {
        0.0
    };
    let vlab_td_absmean = if vlab_steps > 0 {
        (vlab_td_abs / vlab_steps as f64) as f32
    } else {
        0.0
    };
    let vlab_wl_absmean = if vlab_steps > 0 {
        (vlab_wl_abs / vlab_steps as f64) as f32
    } else {
        0.0
    };
    let vlab_spt_absmean = if vlab_steps > 0 {
        (vlab_spt_abs / vlab_steps as f64) as f32
    } else {
        0.0
    };

    let metrics = json!({
        "num_games": args.num_games,
        "avg_score": avg_score,
        "max_score": max_score,
        "avg_moves": avr_moves,
        "max_turns": curriculum(args.iteration).1,
        "policy_kl": if kl_game_n > 0 { kl_game_sum / kl_game_n as f64 } else { -1.0 },
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
        "avg_revealed_tiles": avg_revealed_tiles,
        "avg_captured_tiles": avg_captured_tiles,
        "avg_spt_t0": avg_spt_at(0),
        "avg_spt_t5": avg_spt_at(5),
        "avg_spt_t10": avg_spt_at(10),
        "avg_spt_t15": avg_spt_at(15),
        "avg_spt_t20": avg_spt_at(20),
        "avg_spt_t25": avg_spt_at(25),
        "avg_spt_t30": avg_spt_at(30),
        "villages_t2c_first": (total_t2c[0] / args.num_games as f64) as f32,
        "villages_t2c_p50": (total_t2c[1] / args.num_games as f64) as f32,
        "villages_t2c_p80": (total_t2c[2] / args.num_games as f64) as f32,
        "villages_t2c_all": (total_t2c[3] / args.num_games as f64) as f32,
        "ruins_t2c_p50": (total_t2c[4] / args.num_games as f64) as f32,
        "ruins_t2c_p80": (total_t2c[5] / args.num_games as f64) as f32,
        "ruins_t2c_all": (total_t2c[6] / args.num_games as f64) as f32,
        "decisive_frac": decisive_games as f32 / args.num_games.max(1) as f32,
        "vlab_wl_share": vlab_wl_share,
        "vlab_td_absmean": vlab_td_absmean,
        "vlab_wl_absmean": vlab_wl_absmean,
        "vlab_spt_absmean": vlab_spt_absmean,
        "anchor_games": anchor_games_n,
        "anchor_model_wins": anchor_model_wins,
        "games_file": games_file,
        "moves_by_turn": moves_by_turn,
    });
    std::fs::write(
        ".last_self_play_metrics.json",
        serde_json::to_string(&metrics)?,
    )?;

    if args.anchor_frac > 0.0 && args.opponent.is_none() {
        append_anchor_record(args.iteration, anchor_games_n, anchor_model_wins);
        if anchor_games_n > 0 {
            println!(
                "[Self-Play] anchor games: {} | model wins {:.0} ({:.0}%) | decisive games this run: {}",
                anchor_games_n,
                anchor_model_wins,
                100.0 * anchor_model_wins / anchor_games_n as f32,
                decisive_games,
            );
        }
    }

    let total_duration = start_time.elapsed();
    println!("\n=== Self-Play Complete ===");
    println!("Total time: {:.2}s", total_duration.as_secs_f32());
    println!("Breakdown:");
    println!(
        "  - Game generation: {:.2}s ({:.1}%)",
        games_duration.as_secs_f32(),
        100.0 * games_duration.as_secs_f32() / total_duration.as_secs_f32()
    );
    let final_moves_per_sec = total_moves as f64 / games_duration.as_secs_f64().max(1e-9);
    println!(
        "  - Throughput: {:.2} moves/sec ({} moves)",
        final_moves_per_sec, total_moves
    );
    // How often search crossed a turn boundary in-tree (simulated EndTurn
    // edges only; real played moves don't count). ~0/move decision means the
    // tree essentially never sees beyond the current turn.
    let sim_end_turns =
        polyfish::game::SIM_END_TURN_EDGES.load(std::sync::atomic::Ordering::Relaxed);
    println!(
        "  - Sim EndTurn edges: {} total ({:.2} per move decision)",
        sim_end_turns,
        sim_end_turns as f64 / (total_moves as f64).max(1.0)
    );

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

#[cfg(test)]
mod td_lambda_tests {
    use super::*;

    fn step(player_id: PlayerId, turn: i32, my: i32, opp: i32, rv: Option<f32>) -> LabelStep {
        LabelStep {
            player_id,
            turn,
            my_score: my,
            opp_score: opp,
            my_spt: 0,
            opp_spt: 0,
            root_value: rv,
        }
    }

    fn finals(pairs: &[(i32, i32)]) -> HashMap<i32, i32> {
        pairs.iter().copied().collect()
    }

    #[test]
    fn last_decision_of_game_is_pure_terminal_return_at_any_lambda() {
        // Only decision on record for player 1: no checkpoints ahead, so the
        // label must equal the plain (unbootstrapped) reward to final scores
        // regardless of lambda (remaining_weight stays 1.0, loop body never runs).
        let history = vec![step(1, 5, 1000, 800, Some(0.2))];
        let final_scores = finals(&[(1, 1300), (2, 900)]);
        let expected = reward::normalized_reward(1000, 800, 1300, 900).clamp(-1.0, 1.0);

        for lambda in [0.0, 0.5, 0.8, 0.95] {
            let out = td_lambda_labels(&history, &final_scores, lambda);
            assert!(
                (out[0] - expected).abs() < 1e-6,
                "lambda={lambda}: got {}, expected {expected}",
                out[0]
            );
        }
    }

    #[test]
    fn lambda_zero_uses_only_the_first_checkpoint() {
        // Two future checkpoints for player 1 at turn 6 and turn 7. At
        // lambda=0 the label must depend ONLY on the turn-6 checkpoint —
        // this is exactly the original 1-step TD bootstrap, reproduced
        // bit-for-bit as the lambda=0 special case of the new formula.
        let history = vec![
            step(1, 5, 1000, 800, Some(0.2)),  // i
            step(1, 6, 1100, 800, Some(0.9)),  // checkpoint n=1 (this player's next turn)
            step(2, 6, 1000, 850, Some(-0.1)), // other player, ignored
            step(1, 7, 1400, 800, Some(-0.9)), // checkpoint n=2: a wildly different root_value
        ];
        let final_scores = finals(&[(1, 5000), (2, 800)]);

        let out = td_lambda_labels(&history, &final_scores, 0.0);

        let r = reward::normalized_reward(1000, 800, 1100, 800);
        let expected = (r + reward::GAMMA_TURN.powi(1) * 0.9).clamp(-1.0, 1.0);
        assert!(
            (out[0] - expected).abs() < 1e-6,
            "got {}, expected {expected}",
            out[0]
        );

        // Sanity: changing turn 7's root_value must NOT move the lambda=0 label.
        let mut history2 = history.clone();
        history2[3].root_value = Some(12345.0);
        let out2 = td_lambda_labels(&history2, &final_scores, 0.0);
        assert!((out2[0] - out[0]).abs() < 1e-6);
    }

    #[test]
    fn weights_blend_geometrically_and_sum_to_one() {
        // One checkpoint ahead + terminal. At lambda=0.5 the checkpoint gets
        // weight 0.5 and the terminal return gets the residual 0.5 — hand
        // computed, not just asserted-to-sum-to-1, so a weighting bug can't
        // hide behind a normalization step.
        let history = vec![
            step(1, 0, 100, 100, Some(0.4)),
            step(1, 1, 300, 100, Some(0.6)),
        ];
        let final_scores = finals(&[(1, 300), (2, 100)]);

        let out = td_lambda_labels(&history, &final_scores, 0.5);

        let n1 = reward::normalized_reward(100, 100, 300, 100) + reward::GAMMA_TURN.powi(1) * 0.6;
        let terminal = reward::normalized_reward(100, 100, 300, 100);
        let expected = (0.5 * n1 + 0.5 * terminal).clamp(-1.0, 1.0);

        assert!(
            (out[0] - expected).abs() < 1e-6,
            "got {}, expected {expected}",
            out[0]
        );
    }

    #[test]
    fn missing_root_value_at_a_checkpoint_contributes_zero_bootstrap() {
        // Turn 6's only entry has no root value (forced/book/single-legal
        // move) — its n-step return must fall back to pure banked reward
        // (0.0 bootstrap), not skip the checkpoint entirely.
        let history = vec![
            step(1, 5, 1000, 800, Some(0.2)),
            step(1, 6, 1200, 800, None),
        ];
        let final_scores = finals(&[(1, 1200), (2, 800)]);

        let out = td_lambda_labels(&history, &final_scores, 0.0);
        let expected = reward::normalized_reward(1000, 800, 1200, 800).clamp(-1.0, 1.0);
        assert!(
            (out[0] - expected).abs() < 1e-6,
            "got {}, expected {expected}",
            out[0]
        );
    }
}

#[cfg(test)]
mod spt_shaping_tests {
    use super::*;

    fn step_spt(player_id: PlayerId, turn: i32, my_spt: i32, opp_spt: i32) -> LabelStep {
        LabelStep {
            player_id,
            turn,
            my_score: 0,
            opp_score: 0,
            my_spt,
            opp_spt,
            root_value: None,
        }
    }

    fn norm() -> f32 {
        polyfish::states::default_max_spt() as f32
    }

    #[test]
    fn level_up_move_gets_instant_positive_credit() {
        // SPT 2 before the level-up harvest, 3 at the next own decision:
        // the harvesting step itself carries the +1/norm credit.
        let history = vec![step_spt(1, 3, 2, 2), step_spt(1, 3, 3, 2)];
        let out = spt_shaping_deltas(&history);
        assert!((out[0] - 1.0 / norm()).abs() < 1e-6, "got {}", out[0]);
        // A player's last recorded step gets no phantom credit.
        assert_eq!(out[1], 0.0);
    }

    #[test]
    fn opponent_gain_is_charged_at_the_boundary_and_antisymmetric() {
        // P1 ends turn 3 at parity; P2 levels up on their turn; P1's next
        // decision sees the deficit. The drop lands on P1's boundary step,
        // and P2's harvesting step carries the mirror-image positive credit.
        let history = vec![
            step_spt(1, 3, 2, 2), // P1's last move of turn 3
            step_spt(2, 3, 2, 2), // P2's harvest decision (their POV)
            step_spt(2, 3, 3, 2), // P2's next decision: their SPT is up
            step_spt(1, 4, 2, 3), // P1's next turn: opponent SPT up
        ];
        let out = spt_shaping_deltas(&history);
        assert!((out[0] + 1.0 / norm()).abs() < 1e-6, "got {}", out[0]);
        assert!((out[1] - 1.0 / norm()).abs() < 1e-6, "got {}", out[1]);
    }

    #[test]
    fn deltas_telescope_to_last_minus_first_potential_per_player() {
        // Potential-based invariant: a player's summed deltas equal
        // Φ(their last step) - Φ(their first step), so the shaping adds no
        // net free reward over a game — it only relocates credit.
        let history = vec![
            step_spt(1, 0, 0, 0),
            step_spt(2, 0, 0, 0),
            step_spt(1, 1, 2, 1),
            step_spt(2, 1, 1, 2),
            step_spt(1, 2, 5, 3),
            step_spt(2, 2, 3, 5),
        ];
        let out = spt_shaping_deltas(&history);
        let p1_sum: f32 = out[0] + out[2] + out[4];
        let p2_sum: f32 = out[1] + out[3] + out[5];
        assert!((p1_sum - 2.0 / norm()).abs() < 1e-6, "got {p1_sum}");
        assert!((p2_sum + 2.0 / norm()).abs() < 1e-6, "got {p2_sum}");
    }
}
