// The METRICS json! literal outgrew serde_json's default macro recursion.
#![recursion_limit = "256"]

use candle_core::{Device, Tensor};
use polyfish::TribeType;
use polyfish::ai::brain::{Brain, SearchBackend, SearchBackendArg};
use polyfish::ai::eval_backend::{self, EvalBackendKind, PlayerBackend};
use polyfish::ai::eval_server::{EvalServerConfig, EvalServerStats, Evaluator};
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
use strum::IntoEnumIterator;

const HEURISTIC_PRIOR_W0: f32 = 0.5; // net & heur blended 50/50 at start
const HEURISTIC_PRIOR_DECAY: f32 = 0.97; // decays 0.5 -> 0.1 floor by ~iteration 53
const ANCHOR_FRAC_DECAY: f32 = 0.97; // same rate as HEURISTIC_PRIOR_DECAY, own start value
const CRUTCH_FLOOR: f32 = 0.1; // intermediate plateau shared by both crutches below

/// Exponential decay from `w0` toward `CRUTCH_FLOOR`, then a hard cutover to
/// 0 once `iteration >= decay_last_iter` (or immediately if `force_zero`).
/// Shared by `prior_heuristic_weight` (self-play search prior blend) and
/// `anchor_frac` (heuristic-anchor game rate) — both are training-time
/// crutches meant to fully phase out, not asymptote at a permanent floor.
fn decay_crutch(
    w0: f32,
    decay_rate: f32,
    iteration: usize,
    decay_last_iter: usize,
    force_zero: bool,
) -> f32 {
    if force_zero || iteration >= decay_last_iter {
        return 0.0;
    }
    (w0 * decay_rate.powi(iteration as i32)).max(CRUTCH_FLOOR)
}

// Absolute yardstick for the value target; 8K points is a good place to end a 30T game
const GOOD_BOT_FINAL_SCORE: f32 = 8000.0;
// How much to weight relative (vs opponent) vs absolute (vs yardstick) final outcome.
// 1.0 = pure relative (zero-sum). The value backup negates across every
// player-turn boundary (mcts_common.rs), which is only valid when
// v(mine) = -v(theirs); an absolute own-progress component is NOT
// antisymmetric — the opponent's progress isn't my loss — so any abs share
// gets systematically corrupted through EndTurn-crossing lines, worse as
// search deepens. The mirror-play "empty relative label" problem is fixed in
// the DATA instead: anchor games vs the heuristic backend (--anchor-frac)
// make passivity actually lose, giving the relative label real signal.
const FINAL_OUTCOME_REL_W: f32 = 1.0;

// Weight of the TD(lambda) delta vs the final-outcome tail.
const TD_W: f32 = 0.7;
// Bootstrap/Monte-Carlo blend: center of mass of the geometric weights is
// 1/(1-LAMBDA_RETURN) turns (~5 at 0.8). Chosen to reach the turns-away
// horizon a village approach needs credit across, without drifting back
// toward the high-variance near-pure-MC regime the TD project escaped.
const LAMBDA_RETURN: f32 = 0.8;

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
        if num_games >= 64 {
            Self::SampledFinish
        } else if num_games > 32 {
            Self::Periodic
        } else {
            Self::Full
        }
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
    /// Tower window: pov stalled at EXACTLY 2 cities, mid-game (turn >= 15),
    /// with >= 1 unmoved unit. No discovered-village requirement — the fork
    /// where a stalled tribe chooses units/economy/expansion vs. flat tech.
    ThirdCity,
    /// Unambiguous-case control: pov has exactly 2 cities AND a currently
    /// legal (affordable) Harvest move exists for a population-granting
    /// resource. Captures one trace at the trigger turn, then one per turn
    /// for the next 3 turns (window, not single-shot) — see
    /// find_harvest_trigger / decision_traces_harvest/.
    HarvestReady,
    /// Layer-2/Layer-1 discriminator (Jul 21, 2026): pov has exactly 2
    /// cities AND a FOW-discovered, uncaptured village exists more than 1
    /// tile (Chebyshev) from every unit — the same "opportunity" definition
    /// as the FM-1/FM-3 vs-Greedy turn-state measurement, so the trace and
    /// the arena-level stall metric are directly comparable. Captures every
    /// ply of the triggering player for the trigger turn + the next 3 turns.
    /// Purpose: find the "Step toward the village" candidate at these plies
    /// and check whether its raw policy prior is healthy (a reward/value
    /// problem) or crushed like Build/Harvest was (a proposal problem, but
    /// via within-Step-type competition, not cross-type) — see
    /// find_village_pursuit_trigger / decision_traces_pursuit/.
    VillagePursuit,
    /// "Wandering" fork (Jul 24, 2026): pov has NO FOW-discovered open
    /// village in view (nothing to capture) but still has an unmoved unit
    /// deciding where to move — the case where the hand-coded search-prior
    /// nudge (openness/reveal-fog/center pull in scoring.rs) is the only
    /// thing steering movement. Captures every ply of the triggering player
    /// for the trigger turn + the next 3 turns. Purpose: per Step candidate,
    /// compare raw_net_prob (the net's own learned prior) against
    /// heuristic_score to see whether the net internalized the frontier/center
    /// nudge or diverged from it — see find_wander_trigger /
    /// decision_traces_wander/.
    Wander,
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

    if trigger == TraceTrigger::ThirdCity {
        // Tower window: pov stalled at EXACTLY 2 cities, mid-game (turn >= 15),
        // with at least one unit still to move this ply. The 3rd village need
        // NOT be discovered — failing to explore toward it is part of the
        // failure we're hunting. village_idx = -1 if none currently visible.
        if tribe.cities.len() != 2 || state.settings.turn < 15 {
            return None;
        }
        let unit_idx = tribe.units.iter().find(|u| !u.moved).map(|u| u.coords.idx)?;
        let mut village_idx = -1;
        let mut best_d = i32::MAX;
        for &v in open_villages {
            let Some(vt) = state.tiles.get(&v) else { continue };
            if !vt.explorers.contains(&pov) {
                continue; // only villages already discovered by pov
            }
            for unit in &tribe.units {
                let d = unit.coords.chebyshev_distance_to(&vt.coords);
                if d < best_d {
                    best_d = d;
                    village_idx = v;
                }
            }
        }
        return Some((unit_idx, village_idx));
    }

    let target_distance = match trigger {
        TraceTrigger::Adjacent => 1,
        TraceTrigger::OnVillage => 0,
        TraceTrigger::ThirdCity
        | TraceTrigger::HarvestReady
        | TraceTrigger::VillagePursuit
        | TraceTrigger::Wander => {
            unreachable!()
        }
    };
    for unit in &tribe.units {
        let eligible = match trigger {
            TraceTrigger::Adjacent => !unit.moved,
            TraceTrigger::OnVillage => !unit.moved && !unit.attacked,
            TraceTrigger::ThirdCity
            | TraceTrigger::HarvestReady
            | TraceTrigger::VillagePursuit
            | TraceTrigger::Wander => {
            unreachable!()
        }
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

/// Unambiguous-case control for TraceTrigger::HarvestReady: pov has exactly
/// 2 cities AND `generate_legal_moves` currently offers a Harvest move for a
/// tile whose resource grants population (`generate_econ_moves` already
/// gates affordability — a legal Harvest here means the tribe CAN pay for it
/// right now). Returns the target tile index of the best (highest reward_pop)
/// such opportunity, or None if the condition doesn't hold this ply.
fn find_harvest_trigger(state: &GameState, pov: PlayerId) -> Option<i32> {
    let tribe = state.tribes.get(&pov)?;
    if tribe.cities.len() != 2 {
        return None;
    }
    let legal = polyfish::moves::generate_legal_moves(state);
    let mut best: Option<(i32, i32)> = None;
    for m in &legal {
        if m.move_type() != polyfish::types::MoveType::Harvest {
            continue;
        }
        let Ok(idx) = m.target_idx() else { continue };
        let idx = idx as i32;
        if let Some(Some(resource)) = state.resources.get(&idx).as_ref() {
            let pop =
                polyfish::settings::resources::get_resource_setting(resource.resource_type)
                    .reward_pop as i32;
            if pop > 0 && best.map_or(true, |(_, bp)| pop > bp) {
                best = Some((idx, pop));
            }
        }
    }
    best.map(|(idx, _)| idx)
}

/// Trigger for TraceTrigger::VillagePursuit: pov has exactly 2 cities AND a
/// FOW-discovered, uncaptured village more than 1 tile (Chebyshev) from
/// every one of pov's units. Mirrors the "opportunity" definition used by
/// `--dump-turn-states`'s vs-Greedy 3rd-city analysis exactly, so a trace
/// captured here lines up with that measurement's turn/opportunity. Returns
/// the nearest such village's tile index.
fn find_village_pursuit_trigger(
    state: &GameState,
    pov: PlayerId,
    open_villages: &std::collections::HashSet<i32>,
) -> Option<i32> {
    let tribe = state.tribes.get(&pov)?;
    if tribe.cities.len() != 2 {
        return None;
    }
    let mut best: Option<(i32, i32)> = None;
    for &v in open_villages {
        let Some(vt) = state.tiles.get(&v) else {
            continue;
        };
        if !vt.explorers.contains(&pov) {
            continue;
        }
        let d = tribe
            .units
            .iter()
            .map(|u| u.coords.chebyshev_distance_to(&vt.coords))
            .min();
        let Some(d) = d else { continue };
        if d <= 1 {
            continue;
        }
        if best.map_or(true, |(_, bd)| d < bd) {
            best = Some((v, d));
        }
    }
    best.map(|(v, _)| v)
}

/// Trigger for TraceTrigger::Wander: pov has NO FOW-discovered, uncaptured
/// village in view (nothing to pursue) but still has an unmoved unit that
/// must decide where to move. Returns that unit's tile index. The window
/// captures every ply for trigger_turn + 3; each written trace carries the
/// `visible_villages` set so analysis can drop plies where a village later
/// came into view (no longer truly "wandering").
fn find_wander_trigger(
    state: &GameState,
    pov: PlayerId,
    open_villages: &std::collections::HashSet<i32>,
) -> Option<i32> {
    let tribe = state.tribes.get(&pov)?;
    let has_visible_village = open_villages.iter().any(|v| {
        state
            .tiles
            .get(v)
            .map_or(false, |t| t.explorers.contains(&pov))
    });
    if has_visible_village {
        return None;
    }
    tribe.units.iter().find(|u| !u.moved).map(|u| u.coords.idx)
}

/// Write one captured decision trace to `decision_traces/`, tagged with
/// enough metadata (iteration/game/turn/player/trigger tiles) to sample and
/// compare across games and training iterations. One file per decision —
/// safe under concurrent self-play actors without any shared-file locking.
#[allow(clippy::too_many_arguments)]
fn write_decision_trace(
    dir_name: &str,
    trace: &polyfish::ai::decision_trace::DecisionTrace,
    iteration: usize,
    game_idx: usize,
    turn: i32,
    move_count: usize,
    player_id: PlayerId,
    trigger_unit_idx: i32,
    trigger_village_idx: i32,
    visible_villages: &[i32],
    turns_since_trigger: Option<i32>,
) {
    let wrapped = json!({
        "iteration": iteration,
        "game_idx": game_idx,
        "turn": turn,
        "move_count": move_count,
        "player_id": player_id,
        "trigger_unit_idx": trigger_unit_idx,
        "trigger_village_idx": trigger_village_idx,
        "visible_villages": visible_villages,
        "turns_since_trigger": turns_since_trigger,
        "trace": trace,
    });
    let dir = std::path::Path::new(dir_name);
    if let Err(e) = std::fs::create_dir_all(dir) {
        eprintln!("[trace] failed to create {dir_name}/: {e}");
        return;
    }
    let path = dir.join(format!(
        "iter{iteration}_game{game_idx}_turn{turn}_p{player_id}_m{move_count}.json"
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

/// One traced decision from a --dump-failed-dir game: what the search weighed
/// (see decision_trace.rs) and what it picked. `trace` is None for zero-search
/// seats (Greedy/Heuristic anchors have no Gumbel tree).
#[derive(serde::Serialize)]
struct TracedDecision {
    turn: i32,
    move_count: usize,
    player_id: PlayerId,
    chosen: String,
    root_value: Option<f32>,
    trace: Option<polyfish::ai::decision_trace::DecisionTrace>,
}

/// Dump one zero-capture game (no village taken by either player): a
/// watcher-loadable replay plus the full per-decision trace log, tagged with
/// the matchup and seat backends. One file pair per game — actor-safe.
#[allow(clippy::too_many_arguments)]
fn dump_failed_game(
    dir: &str,
    iteration: usize,
    game_idx: usize,
    seed: i64,
    tribes: &[TribeType],
    backend1: SearchBackend,
    backend2: SearchBackend,
    max_turns: i32,
    scores: &HashMap<i32, i32>,
    recap: &ModReplay,
    decisions: &[TracedDecision],
) {
    let dir = std::path::Path::new(dir);
    if let Err(e) = std::fs::create_dir_all(dir) {
        eprintln!("[dump-failed] failed to create {}: {e}", dir.display());
        return;
    }
    let base = format!("failed_iter{iteration}_game{game_idx}_seed{seed}");
    match serde_json::to_vec_pretty(recap) {
        Ok(bytes) => {
            let path = dir.join(format!("{base}.replay.json"));
            if let Err(e) = std::fs::write(&path, bytes) {
                eprintln!("[dump-failed] failed to write {}: {e}", path.display());
            }
        }
        Err(e) => eprintln!("[dump-failed] failed to serialize replay: {e}"),
    }
    let wrapped = json!({
        "iteration": iteration,
        "game_idx": game_idx,
        "seed": seed,
        "tribes": tribes.iter().map(|t| format!("{t:?}")).collect::<Vec<_>>(),
        "backends": [format!("{backend1:?}"), format!("{backend2:?}")],
        "max_turns": max_turns,
        "final_scores": scores,
        "decisions": decisions,
    });
    match serde_json::to_vec(&wrapped) {
        Ok(bytes) => {
            let path = dir.join(format!("{base}.decisions.json"));
            if let Err(e) = std::fs::write(&path, bytes) {
                eprintln!("[dump-failed] failed to write {}: {e}", path.display());
            }
        }
        Err(e) => eprintln!("[dump-failed] failed to serialize decisions: {e}"),
    }
}

/// Append one start-of-player-turn snapshot to <dir>/game<idx>.jsonl for the
/// multi-turn 3rd-city pursuit analysis: the acting player's owned cities,
/// FOW-visible uncaptured villages (open_villages seen by pov — same set the
/// ThirdCity trace uses), and unit tiles. Row-major 11x11 tile indices.
fn dump_turn_state(
    file: &mut File,
    game_idx: usize,
    state: &GameState,
    pov: PlayerId,
    open_villages: &std::collections::HashSet<i32>,
) {
    let Some(tribe) = state.tribes.get(&pov) else {
        return;
    };
    let cities: Vec<i32> = tribe.cities.iter().map(|c| c.idx).collect();
    let visible_villages: Vec<i32> = open_villages
        .iter()
        .copied()
        .filter(|idx| {
            state
                .tiles
                .get(idx)
                .map_or(false, |t| t.explorers.contains(&pov))
        })
        .collect();
    let units: Vec<i32> = tribe.units.iter().map(|u| u.coords.idx).collect();
    let rec = json!({
        "game": game_idx,
        "turn": state.settings.turn,
        "player": pov,
        "cities": cities,
        "city_count": cities.len(),
        "visible_villages": visible_villages,
        "units": units,
    });
    if let Ok(s) = serde_json::to_string(&rec) {
        let _ = writeln!(file, "{s}");
    }
}

/// One per-player development-tempo sample, taken at the start of that
/// player's turn (before any of their moves).
#[derive(Clone)]
struct TempoSample {
    turn: i32,
    cities: i32,
    city_levels: i32,
    spt: i32,
    units: i32,
    /// Σ star-cost of living units — army size weighted by quality.
    army_stars: i32,
    revealed: i32,
    techs: i32,
    /// Enemy units destroyed so far, read straight off `TribeState::kills`
    /// (engine-maintained, undo-safe). Conversions are not kills.
    kills: i32,
    /// Cumulative counters through this sample (mirrors of the TempoTrack
    /// counters, snapshotted so both curve and totals are per-turn/per-role).
    trained_cum: i32,
    lost_cum: i32,
    /// Σ star-cost of units lost so far — a dead giant costs 10, not 1.
    stars_lost_cum: i32,
}

/// One player's tempo curve plus event-accounted unit counters for the game.
/// Counters come from per-move unit-count diffs, so ruin grants, level-up
/// giants, conversions, and retaliation deaths are all captured without
/// hooking the actions layer (a conversion counts as lost+granted).
#[derive(Default, Clone)]
struct TempoTrack {
    samples: Vec<TempoSample>,
    /// Units gained by a Summon move — star-spent production only.
    units_trained: i32,
    /// Units gained any other way (ruins, conversion, level-up rewards).
    units_granted: i32,
    units_lost: i32,
    giants_made: i32,
    /// Σ star-cost of lost units (army VALUE destroyed, not just count).
    army_stars_lost: i32,
}

fn tempo_sample(state: &GameState, pov: PlayerId) -> Option<TempoSample> {
    let tribe = state.tribes.get(&pov)?;
    let army_stars: i32 = tribe
        .units
        .iter()
        .map(|u| polyfish::settings::units::get_unit_setting(u.unit_type).cost)
        .sum();
    Some(TempoSample {
        turn: state.settings.turn,
        cities: tribe.cities.len() as i32,
        city_levels: tribe.cities.iter().map(|c| c.level).sum(),
        spt: polyfish::functions::get_tribe_spt(state, tribe),
        units: tribe.units.len() as i32,
        army_stars,
        revealed: state
            .tiles
            .values()
            .filter(|t| t.explorers.contains(&pov))
            .count() as i32,
        techs: tribe.tech_vanilla.len() as i32,
        kills: tribe.kills,
        // Attached from the TempoTrack counters at the push site.
        trained_cum: 0,
        lost_cum: 0,
        stars_lost_cum: 0,
    })
}

/// `(unit_count, giant_count, army_star_cost)` per player, for post-move
/// diff accounting.
fn unit_tally(state: &GameState) -> HashMap<PlayerId, (i32, i32, i32)> {
    state
        .tribes
        .iter()
        .map(|(id, t)| {
            let giants = t
                .units
                .iter()
                .filter(|u| u.unit_type == polyfish::types::UnitType::Giant)
                .count() as i32;
            let stars: i32 = t
                .units
                .iter()
                .map(|u| polyfish::settings::units::get_unit_setting(u.unit_type).cost)
                .sum();
            (*id, (t.units.len() as i32, giants, stars))
        })
        .collect()
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
    my_score: f32,
    opp_score: f32,
    turn: i32,
    root_value: Option<f32>,
    /// Raw NN root value (tanh-bounded, pre-search) — value-head calibration only.
    root_own_value: Option<f32>,
    /// Ground-truth (unfogged) non-invisible enemy-unit occupancy at decision
    /// time, POV-relative — the aux_fog_units target.
    enemy_units: Vec<f32>,
    my_spt: i32,
    opp_spt: i32,
    /// Pursuit proximity to the nearest capturable village at decision time,
    /// POV-relative, normalized to [0,1] — the aux_pursuit target.
    pursuit: f32,
}

/// The subset of `HistoryStep` the TD(lambda) label computation needs —
/// split out so `td_lambda_labels` is a pure, directly testable function
/// (no `GameFeatures`/policy tensors to fabricate in a unit test).
#[derive(Clone, Copy)]
struct LabelStep {
    player_id: PlayerId,
    turn: i32,
    my_score: f32,
    opp_score: f32,
    root_value: Option<f32>,
}

impl From<&HistoryStep> for LabelStep {
    fn from(s: &HistoryStep) -> Self {
        LabelStep {
            player_id: s.player_id,
            turn: s.turn,
            my_score: s.my_score,
            opp_score: s.opp_score,
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
    my: f32,
    opp: f32,
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
/// `PlayerId`. Output is aligned 1:1 with `history`. `label_rel_w` prices
/// the windows/terminal — may diverge from in-tree REL_W (EXP_ELO_006).
fn td_lambda_labels(
    history: &[LabelStep],
    final_scores: &HashMap<i32, f32>,
    lambda: f32,
    label_rel_w: f32,
) -> Vec<f32> {
    let checkpoints = checkpoints_by_player(history);

    history
        .iter()
        .map(|step| {
            let my_final = final_scores.get(&step.player_id).copied().unwrap_or(0.0);
            let opp_final = final_scores
                .iter()
                .filter(|(id, _)| **id != step.player_id)
                .map(|(_, s)| *s)
                .next()
                .unwrap_or(0.0);
            let terminal_return = reward::normalized_reward_wf(
                step.my_score,
                step.opp_score,
                my_final,
                opp_final,
                label_rel_w,
            );

            let empty = Vec::new();
            let ahead = checkpoints.get(&step.player_id).unwrap_or(&empty);
            let start = ahead.partition_point(|c| c.turn <= step.turn);

            let mut acc = 0.0f32;
            let mut remaining_weight = 1.0f32;
            for cp in &ahead[start..] {
                let r = reward::normalized_reward_wf(
                    step.my_score,
                    step.opp_score,
                    cp.my,
                    cp.opp,
                    label_rel_w,
                );
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

/// Per-decision SPT snapshot for the aux_spt target. Parallel to `LabelStep`
/// rather than riding `Checkpoint`, whose within-turn replace rule is
/// root-value-driven and unrelated to SPT semantics.
#[derive(Clone, Copy)]
struct SptStep {
    player_id: PlayerId,
    turn: i32,
    my_spt: i32,
    opp_spt: i32,
}

/// First decision per (player, turn) — SPT at the start of that player's turn.
fn spt_checkpoints_by_player(steps: &[SptStep]) -> HashMap<PlayerId, Vec<SptStep>> {
    let mut out: HashMap<PlayerId, Vec<SptStep>> = HashMap::new();
    for s in steps {
        let list = out.entry(s.player_id).or_default();
        if list.last().map_or(true, |c| c.turn != s.turn) {
            list.push(*s);
        }
    }
    out
}

/// `[my, opp]` SPT at the first same-player turn >= turn+5, else the final
/// values (game ended inside the horizon).
fn spt_target(cps: Option<&Vec<SptStep>>, turn: i32, final_my: i32, final_opp: i32) -> (i32, i32) {
    cps.and_then(|c| {
        let i = c.partition_point(|s| s.turn < turn + 5);
        c.get(i).map(|s| (s.my_spt, s.opp_spt))
    })
    .unwrap_or((final_my, final_opp))
}

/// Multi-hot over `TechnologyType::iter()` POSITION — discriminants are
/// sparse (-1..121), so raw ids would need a 123-slot vector.
fn tech_multihot(techs: &[polyfish::states::TechnologyState]) -> Vec<f32> {
    let order: Vec<polyfish::types::TechnologyType> =
        polyfish::types::TechnologyType::iter().collect();
    let mut v = vec![0.0f32; order.len()];
    for t in techs.iter().filter(|t| t.discovered) {
        if let Some(p) = order.iter().position(|x| *x == t.tech_type) {
            v[p] = 1.0;
        }
    }
    v
}

/// Ground-truth enemy-unit occupancy from the unfogged master state. Skips
/// `Invisible` units — the engine's single visibility rule.
fn enemy_unit_grid(state: &GameState, pov: PlayerId, len: usize) -> Vec<f32> {
    let mut g = vec![0.0f32; len];
    for (id, t) in &state.tribes {
        if *id == pov {
            continue;
        }
        for u in &t.units {
            if u.effects.contains(&polyfish::types::UnitEffect::Invisible) {
                continue;
            }
            let i = u.coords.idx as usize;
            if i < len {
                g[i] = 1.0;
            }
        }
    }
    g
}

/// End-of-episode tile ownership mapped to the sample's POV: +1 mine,
/// -1 any opponent, 0 unowned.
fn ownership_from_pov(final_owner: &[i32], pov: PlayerId) -> Vec<f32> {
    final_owner
        .iter()
        .map(|&o| {
            if o == pov {
                1.0
            } else if o != 0 {
                -1.0
            } else {
                0.0
            }
        })
        .collect()
}

/// Result from a single game - contains all data needed for training
struct GameResult {
    history: Vec<HistoryStep>,
    scores: HashMap<i32, i32>,
    /// Per-player `score + shape_w_label·Φ` at game end — the terminal
    /// snapshot for TD labels, consistent with the shaped step snapshots.
    /// Equals raw score when shaping is off.
    final_potentials: HashMap<i32, f32>,
    final_cities: HashMap<i32, i32>,
    total_cities: i32,
    moves: usize,
    /// Net-seat plies only (excludes Greedy/opponent seats) — the seat-clean
    /// counterpart of `moves` for the avg_moves behavior chart.
    net_moves: usize,
    winner_score: i32,
    /// Adjudicated winner: sole survivor, else higher final score at timeout.
    winner_id: i32,
    recap: ModReplay,
    cap_ruins: usize,
    cap_villages: usize,
    cap_cities: usize,
    cap_capitals: usize,
    action_counts: HashMap<polyfish::types::MoveType, usize>,
    /// Move-type counts keyed by turn number, for the "move mix by turn"
    /// training-progress chart (see parse_metrics.py / dashboard).
    moves_by_turn: HashMap<i32, HashMap<polyfish::types::MoveType, usize>>,
    /// NET-seat tile-exploration and territory-ownership counts at game end
    /// (anchor/opponent seats excluded since Jul 2026; in mirror self-play
    /// this still sums both seats).
    revealed_tiles: i32,
    captured_tiles: i32,
    /// Turn by which 50%/80%/100% of the map's initial open villages (and
    /// ruins) had been captured by a NET-controlled seat — how *directly*
    /// the net seeks them out. Censored at max_turns when a game never gets
    /// there (incl. when the anchor takes them first — losing the race
    /// reads as censored, not captured).
    villages_t2c_p50: f32,
    villages_t2c_p80: f32,
    villages_t2c_all: f32,
    /// First-village stats, per NET SEAT (2 in a mirror game, 1 in an
    /// anchor/league game) so the aggregator can divide by seats rather than
    /// games — matching the t2c_Nth_rate family. `censored_sum` charges
    /// max_turns to a seat that never captured; `turn_sum` covers only the
    /// seats that did.
    villages_first_seats: u32,
    villages_first_captured: u32,
    villages_first_turn_sum: f64,
    villages_first_censored_sum: f64,
    ruins_t2c_p50: f32,
    ruins_t2c_p80: f32,
    ruins_t2c_all: f32,
    /// Mean tribe SPT sampled at the start of game turns 0, 5, 10, … (player 1
    /// to act, before any moves on that turn).
    spt_at_turn: HashMap<i32, f32>,
    /// (mean unit worth, mean army stars per city) over net seats, at the same
    /// milestones as `spt_at_turn`. Absolute ratios with no opponent term, so
    /// unlike contested counts they can move in mirror self-play; measured
    /// cv ~1.5%/iteration against a Greedy reference of ~3.7 / ~10.0 at t15.
    army_ratios_at_turn: HashMap<i32, (f32, f32)>,
    /// End-of-game ground truth for the aux heads: raw per-tile owner ids,
    /// per-player SPT, and per-player researched-tech multi-hot.
    final_owner: Vec<i32>,
    final_spt: HashMap<PlayerId, i32>,
    final_tech: HashMap<PlayerId, Vec<f32>>,
    /// Per-player tempo curves + unit-accounting counters.
    tempo: HashMap<PlayerId, TempoTrack>,
    /// Seat roles (index = player_id - 1): "model", "model_vs_anchor",
    /// "anchor", or "opponent" — lets the aggregator split tempo curves into
    /// intrinsic (mirror), contested (vs anchor), and reference populations.
    roles: [&'static str; 2],
}

const SPT_MILESTONES: [i32; 7] = [0, 5, 10, 15, 20, 25, 30];

/// True when `pid`'s seat is controlled by the training net ("model" /
/// "model_vs_anchor") — anchor (Greedy) and league-opponent seats are
/// excluded from the aggregate metrics so mixed games report the net only.
fn is_net_seat(seat_roles: [&'static str; 2], pid: PlayerId) -> bool {
    let i = (pid - 1) as usize;
    i < 2 && matches!(seat_roles[i], "model" | "model_vs_anchor")
}

/// Aggregate a move-visit distribution into the four decomposed policy-target
/// arrays (action / source-spatial / target-spatial / option), each normalized
/// to sum 1. Shared by the MCTS visit target and the EXP_ELO_020 DAgger
/// Greedy-teacher target so both are built identically before blending.
fn decompose_visits(
    move_visits: &[polyfish::ai::mcts_types::MoveVisit],
    map_size: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let spatial = features::MAP_SIZE * features::MAP_SIZE;
    let mut p_action = vec![0.0; 11];
    let mut p_source = vec![0.0; spatial];
    let mut p_target = vec![0.0; spatial];
    let mut p_option = vec![0.0; 192];
    let mut total = 0.0;
    for mv in move_visits {
        total += mv.visits;
        let t = DecomposedMapper::move_visit_to_targets(mv, map_size);
        if t.action_type < p_action.len() {
            p_action[t.action_type] += mv.visits;
        }
        if let Some(i) = t.source_spatial {
            if i < p_source.len() {
                p_source[i] += mv.visits;
            }
        }
        if let Some(i) = t.target_spatial {
            if i < p_target.len() {
                p_target[i] += mv.visits;
            }
        }
        if let Some(i) = t.target_type {
            if i < p_option.len() {
                p_option[i] += mv.visits;
            }
        }
    }
    if total > 0.0 {
        for x in &mut p_action {
            *x /= total;
        }
        for x in &mut p_source {
            *x /= total;
        }
        for x in &mut p_target {
            *x /= total;
        }
        for x in &mut p_option {
            *x /= total;
        }
    }
    (p_action, p_source, p_target, p_option)
}

/// Mean SPT over net-controlled tribes only (all tribes as a fallback if
/// none qualify — shouldn't happen with valid seat_roles).
fn mean_net_spt(state: &polyfish::states::GameState, seat_roles: [&'static str; 2]) -> f32 {
    let vals: Vec<f32> = state
        .tribes
        .iter()
        .filter(|(id, _)| is_net_seat(seat_roles, **id))
        .map(|(_, t)| polyfish::functions::get_tribe_spt(state, t) as f32)
        .collect();
    if vals.is_empty() {
        let n = state.tribes.len().max(1) as f32;
        return state
            .tribes
            .values()
            .map(|t| polyfish::functions::get_tribe_spt(state, t) as f32)
            .sum::<f32>()
            / n;
    }
    vals.iter().sum::<f32>() / vals.len() as f32
}

/// Mean over net seats of (Σ unit star cost ÷ unit count, Σ unit star cost ÷
/// city count). A seat with no units (or no cities) contributes 0 to that
/// component rather than being skipped, so the denominator stays the seat count.
fn mean_net_army_ratios(
    state: &polyfish::states::GameState,
    seat_roles: [&'static str; 2],
) -> (f32, f32) {
    let (mut worth, mut per_city, mut seats) = (0.0f32, 0.0f32, 0u32);
    for (_, t) in state
        .tribes
        .iter()
        .filter(|(id, _)| is_net_seat(seat_roles, **id))
    {
        let stars: i32 = t
            .units
            .iter()
            .map(|u| polyfish::settings::units::get_unit_setting(u.unit_type).cost)
            .sum();
        if !t.units.is_empty() {
            worth += stars as f32 / t.units.len() as f32;
        }
        if !t.cities.is_empty() {
            per_city += stars as f32 / t.cities.len() as f32;
        }
        seats += 1;
    }
    if seats == 0 {
        return (0.0, 0.0);
    }
    (worth / seats as f32, per_city / seats as f32)
}

fn record_spt_at_turn_start(
    state: &polyfish::states::GameState,
    spt_at_turn: &mut HashMap<i32, f32>,
    army_ratios_at_turn: &mut HashMap<i32, (f32, f32)>,
    next_idx: &mut usize,
    seat_roles: [&'static str; 2],
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
            spt_at_turn.insert(milestone, mean_net_spt(state, seat_roles));
            army_ratios_at_turn.insert(milestone, mean_net_army_ratios(state, seat_roles));
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
///
/// When `eval_backend_kind` is `Candle` and a distinct opponent is given, the
/// opponent network is loaded on its own freshly-obtained device rather than
/// `device`: under Candle, player 1 and player 2 each get an independent
/// `EvalServer` thread, and candle's Metal backend corrupts if two threads
/// encode ops (e.g. `forward_t`) against the same `Device` (see
/// `eval_backend.rs`'s device-isolation contract). tch/metal shards load
/// their own weights on the eval-server thread and never touch this candle
/// device for inference, so sharing is harmless for them.
fn load_networks(
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
    decay_last_iter: usize,
    force_zero_crutches: bool,
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
    dump_failed_dir: Option<&str>,
    dump_turn_states: Option<&str>,
    dump_city_rewards: Option<&str>,
    dump_star_spend: Option<&str>,
    seat_roles: [&'static str; 2],
    shape_w_label: f32,
    shape_w_tree: f32,
    pursuit_w_label: f32,
    pursuit_w_tree: f32,
    unfreeze_opponent: bool,
    dagger_alpha: f32,
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
    // Turn of each net seat's OWN first village capture. The pooled
    // `village_capture_turns` above cannot answer this: a mirror game puts two
    // net seats in one list, an anchor game one, so any rate built from it is a
    // blend of two different per-seat probabilities.
    let mut first_village_turn: HashMap<PlayerId, i32> = HashMap::new();

    // --dump-failed-dir: trace every decision; the log is written out only
    // if the game ends with zero village captures.
    let trace_all = dump_failed_dir.is_some();
    let mut decision_log: Vec<TracedDecision> = Vec::new();

    // --dump-turn-states: one JSONL file per game, one record per player-turn
    // (written at turn start in the loop below). Distinct game_idx => no
    // cross-actor contention; created/truncated once here.
    let mut turn_dump_file: Option<File> = None;
    if let Some(dir) = dump_turn_states {
        let path = std::path::Path::new(dir);
        if let Err(e) = std::fs::create_dir_all(path) {
            eprintln!("[dump-turn-states] failed to create {}: {e}", path.display());
        } else {
            match File::create(path.join(format!("game{game_idx}.jsonl"))) {
                Ok(f) => turn_dump_file = Some(f),
                Err(e) => eprintln!("[dump-turn-states] failed to open game file: {e}"),
            }
        }
    }
    let mut last_dump_key: Option<(i32, PlayerId)> = None;

    // --dump-city-rewards: one JSONL file per game, one record per city
    // level-up reward choice — (turn, player, city level pre-choice, tribe
    // stars at time of choice, reward type chosen). Reward moves are always
    // forced (generate_reward_moves preempts everything else when a choice
    // is pending — moves/mod.rs), so this is a clean, uncontested read of
    // what the policy actually wants at each level, no Step competition.
    let mut city_reward_file: Option<File> = None;
    if let Some(dir) = dump_city_rewards {
        let path = std::path::Path::new(dir);
        if let Err(e) = std::fs::create_dir_all(path) {
            eprintln!("[dump-city-rewards] failed to create {}: {e}", path.display());
        } else {
            match File::create(path.join(format!("game{game_idx}.jsonl"))) {
                Ok(f) => city_reward_file = Some(f),
                Err(e) => eprintln!("[dump-city-rewards] failed to open game file: {e}"),
            }
        }
    }

    // --dump-star-spend: one JSONL record per Research/Harvest/Build/Summon
    // move actually executed — (turn, player, move type, stars spent =
    // tribe.stars before minus after). Reads the real star delta off game
    // state rather than re-deriving cost formulas, so it's exact even with
    // discounts (e.g. Philosophy's tech discount).
    let mut star_spend_file: Option<File> = None;
    if let Some(dir) = dump_star_spend {
        let path = std::path::Path::new(dir);
        if let Err(e) = std::fs::create_dir_all(path) {
            eprintln!("[dump-star-spend] failed to create {}: {e}", path.display());
        } else {
            match File::create(path.join(format!("game{game_idx}.jsonl"))) {
                Ok(f) => star_spend_file = Some(f),
                Err(e) => eprintln!("[dump-star-spend] failed to open game file: {e}"),
            }
        }
    }
    const STAR_SPEND_TYPES: [polyfish::types::MoveType; 4] = [
        polyfish::types::MoveType::Research,
        polyfish::types::MoveType::Harvest,
        polyfish::types::MoveType::Build,
        polyfish::types::MoveType::Summon,
    ];

    let prior_w = decay_crutch(
        HEURISTIC_PRIOR_W0,
        HEURISTIC_PRIOR_DECAY,
        iteration,
        decay_last_iter,
        force_zero_crutches,
    );
    // One trust scalar drives β on σ(Q) in both the exported targets and the
    // search tree itself. --value-trust overrides the iteration ramp, which
    // saturates immediately on ITER_OFFSET-shifted runs.
    let q_target_w =
        value_trust.unwrap_or_else(|| (iteration as f32 / POLICY_TARGET_Q_RAMP_ITERS).min(1.0));

    // Create two agents (they might share the same network, or be different)
    let mut agent1 = Brain::with_backend(eval1, mcts_iters, backend1)
        .with_prior_heuristic_weight(prior_w)
        .with_policy_target_q_weight(q_target_w)
        .with_tree_q_weight(q_target_w)
        .with_reward_shape_w(shape_w_tree)
        .with_pursuit_shape_w(pursuit_w_tree)
        .with_unfreeze_opponent(unfreeze_opponent);
    let mut agent2 = Brain::with_backend(eval2, mcts_iters, backend2)
        .with_prior_heuristic_weight(prior_w)
        .with_policy_target_q_weight(q_target_w)
        .with_tree_q_weight(q_target_w)
        .with_reward_shape_w(shape_w_tree)
        .with_pursuit_shape_w(pursuit_w_tree)
        .with_unfreeze_opponent(unfreeze_opponent);

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
    let mut army_ratios_at_turn: HashMap<i32, (f32, f32)> = HashMap::new();
    let mut next_spt_milestone = 0usize;

    // Per-player tempo tracking: turn-start samples + move-diff unit counters.
    let mut tempo: HashMap<PlayerId, TempoTrack> = HashMap::new();
    let mut last_tempo_key: Option<(i32, PlayerId)> = None;
    let mut prev_tally = unit_tally(&game.state);

    let mut move_count = 0;
    let mut net_moves = 0; // net-seat plies only (excludes Greedy/opponent seats)
    // Up to 3 traces per game, spaced >= 3 game-turns apart, to sample several
    // mid-game stalled decisions rather than only the turn-15 entry ply.
    let mut traces_this_game = 0usize;
    let mut last_trace_turn = -100i32;
    // HarvestReady window state: fires once per game, then captures every
    // ply belonging to the triggering player for the trigger turn + the
    // next 3 turns (not just the first ply of each turn — Step and
    // Harvest/Build aren't exclusive within a turn; the real question is
    // whether Harvest/Build gets picked up once Step options run dry).
    let mut harvest_trigger_turn: Option<i32> = None;
    let mut harvest_trigger_tile: i32 = -1;
    let mut harvest_trigger_pov: Option<PlayerId> = None;
    // VillagePursuit window state: same shape as HarvestReady's — fires once
    // per game, then captures every ply belonging to the triggering player
    // for the trigger turn + the next 3 turns.
    let mut pursuit_trigger_turn: Option<i32> = None;
    let mut pursuit_trigger_village: i32 = -1;
    let mut pursuit_trigger_pov: Option<PlayerId> = None;
    // Wander window state: same shape as VillagePursuit's.
    let mut wander_trigger_turn: Option<i32> = None;
    let mut wander_trigger_unit: i32 = -1;
    let mut wander_trigger_pov: Option<PlayerId> = None;
    while !polyfish::functions::is_game_over(&game.state) {
        record_spt_at_turn_start(
            &game.state,
            &mut spt_at_turn,
            &mut army_ratios_at_turn,
            &mut next_spt_milestone,
            seat_roles,
        );

        if move_count > 50000 {
            // Reduced for safety
            eprintln!(
                "[Game {}] Move count exceeded 50000 (Safety Break)",
                game_idx
            );
            break;
        }

        let pov = game.state.settings.current_player_turn_id;

        // Dump the start-of-player-turn snapshot once per (turn, pov), before
        // any move that turn mutates the state.
        if let Some(f) = turn_dump_file.as_mut() {
            let key = (game.state.settings.turn, pov);
            if last_dump_key != Some(key) {
                dump_turn_state(f, game_idx, &game.state, pov, &open_villages);
                last_dump_key = Some(key);
            }
        }

        // Tempo curve: sample the acting player once per (turn, pov), pre-move.
        let tempo_key = (game.state.settings.turn, pov);
        if last_tempo_key != Some(tempo_key) {
            if let Some(mut s) = tempo_sample(&game.state, pov) {
                let track = tempo.entry(pov).or_default();
                s.trained_cum = track.units_trained;
                s.lost_cum = track.units_lost;
                s.stars_lost_cum = track.army_stars_lost;
                track.samples.push(s);
            }
            last_tempo_key = Some(tempo_key);
        }

        // Get state tensor
        let current_network = if pov == 1 { network1 } else { network2 };
        let device = current_network.device();
        let state_t = state_to_tensor(&game.state, pov, &device)
            .expect("BUG: Failed to create state tensor - game state is invalid");

        // MCTS Search - use the correct agent
        let current_agent = if pov == 1 { &mut agent1 } else { &mut agent2 };

        let trigger_info = if trace_villages
            && !trace_all
            && !matches!(
                trace_trigger,
                TraceTrigger::HarvestReady | TraceTrigger::VillagePursuit | TraceTrigger::Wander
            )
            && traces_this_game < 3
            && game.state.settings.turn >= last_trace_turn + 3
            && trace_counter.load(Ordering::Relaxed) < trace_max
        {
            find_village_trigger(&game.state, pov, &open_villages, trace_trigger)
        } else {
            None
        };

        // HarvestReady: (trigger_tile, turns_since_trigger) for THIS ply, if
        // it belongs to the triggering player and falls inside the
        // [trigger_turn, trigger_turn+3] window. Captures EVERY such ply
        // (not just the first per turn) so post-hoc analysis can bucket by
        // how many Step candidates were still live at capture time.
        let harvest_capture = if trace_villages
            && !trace_all
            && trace_trigger == TraceTrigger::HarvestReady
            && trace_counter.load(Ordering::Relaxed) < trace_max
        {
            if harvest_trigger_turn.is_none() {
                if let Some(tile) = find_harvest_trigger(&game.state, pov) {
                    harvest_trigger_turn = Some(game.state.settings.turn);
                    harvest_trigger_tile = tile;
                    harvest_trigger_pov = Some(pov);
                }
            }
            harvest_trigger_turn.and_then(|start| {
                let turn = game.state.settings.turn;
                (harvest_trigger_pov == Some(pov) && turn <= start + 3)
                    .then_some((harvest_trigger_tile, turn - start))
            })
        } else {
            None
        };

        // VillagePursuit: (trigger_village, turns_since_trigger) for THIS
        // ply, same window/every-ply shape as HarvestReady above.
        let pursuit_capture = if trace_villages
            && !trace_all
            && trace_trigger == TraceTrigger::VillagePursuit
            && trace_counter.load(Ordering::Relaxed) < trace_max
        {
            if pursuit_trigger_turn.is_none() {
                if let Some(village) =
                    find_village_pursuit_trigger(&game.state, pov, &open_villages)
                {
                    pursuit_trigger_turn = Some(game.state.settings.turn);
                    pursuit_trigger_village = village;
                    pursuit_trigger_pov = Some(pov);
                }
            }
            pursuit_trigger_turn.and_then(|start| {
                let turn = game.state.settings.turn;
                (pursuit_trigger_pov == Some(pov) && turn <= start + 3)
                    .then_some((pursuit_trigger_village, turn - start))
            })
        } else {
            None
        };

        // Wander: (trigger_unit, turns_since_trigger) for THIS ply, same
        // window/every-ply shape as VillagePursuit above.
        let wander_capture = if trace_villages
            && !trace_all
            && trace_trigger == TraceTrigger::Wander
            && trace_counter.load(Ordering::Relaxed) < trace_max
        {
            if wander_trigger_turn.is_none() {
                if let Some(unit) = find_wander_trigger(&game.state, pov, &open_villages) {
                    wander_trigger_turn = Some(game.state.settings.turn);
                    wander_trigger_unit = unit;
                    wander_trigger_pov = Some(pov);
                }
            }
            wander_trigger_turn.and_then(|start| {
                let turn = game.state.settings.turn;
                (wander_trigger_pov == Some(pov) && turn <= start + 3)
                    .then_some((wander_trigger_unit, turn - start))
            })
        } else {
            None
        };

        if trace_all
            || trigger_info.is_some()
            || harvest_capture.is_some()
            || pursuit_capture.is_some()
            || wander_capture.is_some()
        {
            current_agent.request_trace();
        }

        let (best_move, move_visits) = current_agent.think_decomposed(&mut game, move_count);
        // The search that just ran was for the CURRENT (pre-move) state, so
        // this is that state's own root value — the TD bootstrap target for
        // whichever earlier step's label lands here as its "next decision".
        let root_value = current_agent.last_root_value();
        let root_own_value = current_agent.last_root_own_value();
        let step_trace = if trace_all {
            current_agent.take_trace()
        } else {
            None
        };

        if let Some((trigger_unit_idx, trigger_village_idx)) = trigger_info {
            if let Some(trace) = current_agent.take_trace() {
                if trace_counter.fetch_add(1, Ordering::Relaxed) < trace_max {
                    let visible_villages: Vec<i32> = open_villages
                        .iter()
                        .copied()
                        .filter(|idx| {
                            game.state
                                .tiles
                                .get(idx)
                                .map_or(false, |t| t.explorers.contains(&pov))
                        })
                        .collect();
                    write_decision_trace(
                        "decision_traces",
                        &trace,
                        iteration,
                        game_idx,
                        game.state.settings.turn,
                        move_count,
                        pov,
                        trigger_unit_idx,
                        trigger_village_idx,
                        &visible_villages,
                        None,
                    );
                }
                traces_this_game += 1;
                last_trace_turn = game.state.settings.turn;
            }
        }

        if let Some((trigger_tile, turns_since)) = harvest_capture {
            if let Some(trace) = current_agent.take_trace() {
                if trace_counter.fetch_add(1, Ordering::Relaxed) < trace_max {
                    write_decision_trace(
                        "decision_traces_harvest",
                        &trace,
                        iteration,
                        game_idx,
                        game.state.settings.turn,
                        move_count,
                        pov,
                        trigger_tile,
                        -1,
                        &[],
                        Some(turns_since),
                    );
                }
            }
        }

        if let Some((trigger_village, turns_since)) = pursuit_capture {
            if let Some(trace) = current_agent.take_trace() {
                if trace_counter.fetch_add(1, Ordering::Relaxed) < trace_max {
                    write_decision_trace(
                        "decision_traces_pursuit",
                        &trace,
                        iteration,
                        game_idx,
                        game.state.settings.turn,
                        move_count,
                        pov,
                        -1,
                        trigger_village,
                        &[],
                        Some(turns_since),
                    );
                }
            }
        }

        if let Some((trigger_unit, turns_since)) = wander_capture {
            if let Some(trace) = current_agent.take_trace() {
                if trace_counter.fetch_add(1, Ordering::Relaxed) < trace_max {
                    let visible_villages: Vec<i32> = open_villages
                        .iter()
                        .copied()
                        .filter(|idx| {
                            game.state
                                .tiles
                                .get(idx)
                                .map_or(false, |t| t.explorers.contains(&pov))
                        })
                        .collect();
                    write_decision_trace(
                        "decision_traces_wander",
                        &trace,
                        iteration,
                        game_idx,
                        game.state.settings.turn,
                        move_count,
                        pov,
                        trigger_unit,
                        -1,
                        &visible_villages,
                        Some(turns_since),
                    );
                }
            }
        }

        let map_size = game.state.settings.size as usize;

        let (mut p_action, mut p_source, mut p_target, mut p_option) =
            decompose_visits(&move_visits, map_size);

        // EXP_ELO_020: DAgger — blend Greedy's move-ranking AT THE MODEL'S OWN
        // state into the policy target, so the expert corrects the policy
        // exactly where the net actually plays (on-distribution). Net seats
        // only — the Greedy anchor seat already emits Greedy's ranking as its
        // target. This attacks the collapsed capture prior at the model's forks
        // without needing the value head; unlike BC (EXP_ELO_007) it labels the
        // learner's states, not the expert's, which is the erosion fix.
        if dagger_alpha > 0.0 && is_net_seat(seat_roles, pov) {
            let greedy_visits = polyfish::ai::heuristic_mcts::GreedyHeuristicAgent
                .select_move_with_decomposed_visits(&mut game, move_count)
                .1;
            let (ga, gs, gt, go) = decompose_visits(&greedy_visits, map_size);
            // Normalization-preserving blend: mix the target's DIRECTION toward
            // Greedy but keep each head's original total mass, so forced/partial
            // states (empty or sub-normal MCTS heads) aren't distorted — a head
            // with zero MCTS mass stays zero (nothing to correct), a full head
            // stays full.
            let blend = |p: &mut [f32], g: &[f32]| {
                let orig: f32 = p.iter().sum();
                for (x, &gv) in p.iter_mut().zip(g.iter()) {
                    *x = (1.0 - dagger_alpha) * *x + dagger_alpha * gv;
                }
                let mixed: f32 = p.iter().sum();
                if mixed > 0.0 {
                    let k = orig / mixed;
                    for x in p.iter_mut() {
                        *x *= k;
                    }
                }
            };
            blend(&mut p_action, &ga);
            blend(&mut p_source, &gs);
            blend(&mut p_target, &gt);
            blend(&mut p_option, &go);
        }

        let policy_data = DecomposedPolicyData {
            action_type: p_action,
            source_spatial: p_source,
            target_spatial: p_target,
            move_option: p_option,
        };

        if let Some(m) = best_move {
            if trace_all {
                decision_log.push(TracedDecision {
                    turn: game.state.settings.turn,
                    move_count,
                    player_id: pov,
                    chosen: m.describe(&game.state),
                    root_value,
                    trace: step_trace,
                });
            }
            let m_type = m.move_type();
            // Move-mix and capture metrics count NET-controlled seats only —
            // in anchor/league games the Greedy/opponent seat's moves would
            // otherwise blend into "the model's" behavior charts.
            let net_move = is_net_seat(seat_roles, pov);
            if net_move {
                net_moves += 1;
                *action_counts.entry(m_type).or_insert(0) += 1;
                *moves_by_turn
                    .entry(game.state.settings.turn)
                    .or_default()
                    .entry(m_type)
                    .or_insert(0) += 1;
            }

            if m_type == polyfish::types::MoveType::Reward {
                if let Some(f) = city_reward_file.as_mut() {
                    if let (Ok(target), Ok(reward_type)) = (m.target_idx(), m.reward_type()) {
                        let target = target as i32;
                        let city = game
                            .state
                            .tribes
                            .get(&pov)
                            .and_then(|t| t.cities.iter().find(|c| c.idx == target));
                        if let Some(city) = city {
                            let stars = game.state.tribes.get(&pov).map(|t| t.stars).unwrap_or(0);
                            let line = json!({
                                "game_idx": game_idx,
                                "turn": game.state.settings.turn,
                                "player_id": pov,
                                "city_idx": target,
                                "city_level": city.level,
                                "city_population": city.population,
                                "stars": stars,
                                "reward": format!("{:?}", reward_type),
                            });
                            if let Ok(s) = serde_json::to_string(&line) {
                                use std::io::Write;
                                let _ = writeln!(f, "{s}");
                            }
                        }
                    }
                }
            }

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

                    if net_move {
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
                    }

                    // The open-set must shrink regardless of who captured
                    // (it also feeds FOW-visible tracking), but t2c turns
                    // record net captures only — a Greedy grab is the net
                    // LOSING the race, not capturing.
                    if open_villages.remove(&idx) {
                        if net_move {
                            village_capture_turns.push(game.state.settings.turn);
                            first_village_turn
                                .entry(pov)
                                .or_insert(game.state.settings.turn);
                        }
                    } else if open_ruins.remove(&idx) && net_move {
                        ruin_capture_turns.push(game.state.settings.turn);
                    }
                }
            }

            flat_recap.push((
                game.state.settings.turn,
                game.state.settings.current_player_turn_id,
                m.serialize(),
            ));
            // Snapshot (possibly Φ-shaped) scores at this moment (pre-move)
            // for the TD label.
            let (my_score_now, opp_score_now) =
                reward::shaped_snapshot(&game.state, pov, shape_w_label, pursuit_w_label);
            let opp_id = game
                .state
                .tribes
                .keys()
                .copied()
                .find(|id| *id != pov)
                .unwrap_or(pov);
            let spt_of = |id: PlayerId| {
                game.state
                    .tribes
                    .get(&id)
                    .map(|t| polyfish::functions::get_tribe_spt(&game.state, t))
                    .unwrap_or(0)
            };
            game_history.push(HistoryStep {
                features: state_t,
                policy: policy_data,
                player_id: pov,
                my_score: my_score_now,
                opp_score: opp_score_now,
                turn: game.state.settings.turn,
                root_value,
                root_own_value,
                enemy_units: enemy_unit_grid(&game.state, pov, features::MAP_SIZE * features::MAP_SIZE),
                my_spt: spt_of(pov),
                opp_spt: spt_of(opp_id),
                pursuit: reward::pursuit_potential(&game.state, pov)
                    / reward::SHAPE_PURSUIT_PER_TILE
                    / reward::SHAPE_PROX_CAP as f32,
            });
            if progress == ProgressMode::Full && move_count > 0 && move_count % 10 == 0 {
                eprintln!(
                    "[Game {}]: Turn: {} Player: {} Move: {}",
                    game_idx,
                    game.state.settings.turn,
                    pov,
                    m.describe(&game.state),
                );
            }
            let star_spend_pre = star_spend_file.as_ref().and_then(|_| {
                STAR_SPEND_TYPES
                    .contains(&m_type)
                    .then(|| game.state.tribes.get(&pov).map(|t| t.stars).unwrap_or(0))
                    .map(|stars_before| (game.state.settings.turn, stars_before))
            });
            let _ = game.play_move(m.as_ref());
            if let (Some((turn, stars_before)), Some(f)) =
                (star_spend_pre, star_spend_file.as_mut())
            {
                let stars_after = game.state.tribes.get(&pov).map(|t| t.stars).unwrap_or(0);
                let line = json!({
                    "game_idx": game_idx,
                    "turn": turn,
                    "player_id": pov,
                    "move_type": format!("{:?}", m_type),
                    "stars_spent": (stars_before - stars_after).max(0),
                });
                if let Ok(s) = serde_json::to_string(&line) {
                    use std::io::Write;
                    let _ = writeln!(f, "{s}");
                }
            }

            // Unit-accounting diff: attribute gains to Summon (trained) vs
            // anything else (granted), and any decrease as a loss — both in
            // count and in star value (a dead giant is 10, not 1).
            let new_tally = unit_tally(&game.state);
            for (&pid, &(n_units, n_giants, n_stars)) in &new_tally {
                let (p_units, p_giants, p_stars) =
                    prev_tally.get(&pid).copied().unwrap_or((0, 0, 0));
                let track = tempo.entry(pid).or_default();
                if n_units > p_units {
                    if m_type == polyfish::types::MoveType::Summon && pid == pov {
                        track.units_trained += n_units - p_units;
                    } else {
                        track.units_granted += n_units - p_units;
                    }
                } else if n_units < p_units {
                    track.units_lost += p_units - n_units;
                }
                if n_stars < p_stars {
                    track.army_stars_lost += p_stars - n_stars;
                }
                if n_giants > p_giants {
                    track.giants_made += n_giants - p_giants;
                }
            }
            prev_tally = new_tally;

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

    // Final tempo sample per player from the end state (a capture on the last
    // turn would otherwise be invisible); replaces a same-turn start sample.
    let tempo_pids: Vec<PlayerId> = game.state.tribes.keys().copied().collect();
    for pid in tempo_pids {
        if let Some(mut s) = tempo_sample(&game.state, pid) {
            let track = tempo.entry(pid).or_default();
            s.trained_cum = track.units_trained;
            s.lost_cum = track.units_lost;
            s.stars_lost_cum = track.army_stars_lost;
            match track.samples.last_mut() {
                Some(last) if last.turn == s.turn => *last = s,
                _ => track.samples.push(s),
            }
        }
    }

    // Determine scores & winner
    // In Domination, the winner is the last tribe alive.
    // If the game timed out (safety cap), use score as tiebreaker.
    let mut scores: HashMap<i32, i32> = HashMap::new();
    let mut final_potentials: HashMap<i32, f32> = HashMap::new();
    let mut alive: HashMap<i32, bool> = HashMap::new();
    for (id, t) in &game.state.tribes {
        scores.insert(*id, t.score);
        alive.insert(*id, t.killed_turn <= 0 && t.resigned_turn <= 0);
    }
    for id in scores.keys() {
        let mut phi = 0.0;
        if shape_w_label != 0.0 {
            phi += shape_w_label * reward::dev_potential(&game.state, *id);
        }
        if pursuit_w_label != 0.0 {
            phi += pursuit_w_label * reward::pursuit_potential(&game.state, *id);
        }
        final_potentials.insert(*id, scores[id] as f32 + phi);
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

    // Net-seat-only territory/exploration (anchor/opponent seats excluded).
    let captured_tiles = game
        .state
        .tiles
        .values()
        .filter(|t| t.owner != 0 && is_net_seat(seat_roles, t.owner))
        .count() as i32;
    let revealed_tiles: i32 = game
        .state
        .tribes
        .keys()
        .filter(|&&pid| is_net_seat(seat_roles, pid))
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

    // Aux-head ground truth; the final state is dropped when this returns.
    let n_tiles = features::MAP_SIZE * features::MAP_SIZE;
    let mut final_owner = vec![0i32; n_tiles];
    for (&idx, tile) in &game.state.tiles {
        let i = idx as usize;
        if i < n_tiles {
            final_owner[i] = tile.owner;
        }
    }
    let mut final_spt = HashMap::new();
    let mut final_tech = HashMap::new();
    for (id, t) in &game.state.tribes {
        final_spt.insert(*id, polyfish::functions::get_tribe_spt(&game.state, t));
        final_tech.insert(*id, tech_multihot(&t.tech_vanilla));
    }

    let recap = ModReplay {
        game_state: initial_state,
        turns: group_recap(flat_recap),
    };
    if let Some(dir) = dump_failed_dir {
        if village_capture_turns.is_empty() {
            dump_failed_game(
                dir,
                iteration,
                game_idx,
                seed,
                &tribes,
                backend1,
                backend2,
                max_turns,
                &scores,
                &recap,
                &decision_log,
            );
        }
    }

    let net_seats: Vec<PlayerId> = game
        .state
        .tribes
        .keys()
        .copied()
        .filter(|pid| is_net_seat(seat_roles, *pid))
        .collect();
    let (mut vf_captured, mut vf_turn_sum, mut vf_censored_sum) = (0u32, 0.0f64, 0.0f64);
    for pid in &net_seats {
        match first_village_turn.get(pid) {
            Some(&t) => {
                vf_captured += 1;
                vf_turn_sum += f64::from(t);
                vf_censored_sum += f64::from(t);
            }
            None => vf_censored_sum += f64::from(max_turns),
        }
    }

    Some(GameResult {
        history: game_history,
        scores,
        final_potentials,
        final_cities,
        total_cities,
        moves: move_count,
        net_moves,
        revealed_tiles,
        captured_tiles,
        villages_t2c_p50: t2c_turn(&village_capture_turns, initial_villages, 0.5, max_turns),
        villages_t2c_p80: t2c_turn(&village_capture_turns, initial_villages, 0.8, max_turns),
        villages_t2c_all: t2c_turn(&village_capture_turns, initial_villages, 1.0, max_turns),
        villages_first_seats: net_seats.len() as u32,
        villages_first_captured: vf_captured,
        villages_first_turn_sum: vf_turn_sum,
        villages_first_censored_sum: vf_censored_sum,
        ruins_t2c_p50: t2c_turn(&ruin_capture_turns, initial_ruins, 0.5, max_turns),
        ruins_t2c_p80: t2c_turn(&ruin_capture_turns, initial_ruins, 0.8, max_turns),
        ruins_t2c_all: t2c_turn(&ruin_capture_turns, initial_ruins, 1.0, max_turns),
        spt_at_turn,
        army_ratios_at_turn,
        final_owner,
        final_spt,
        final_tech,
        tempo,
        roles: seat_roles,
        winner_score,
        winner_id,
        recap,
        cap_ruins,
        cap_villages,
        cap_cities,
        cap_capitals,
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

        /// STARTING fraction of games (0..1) played against the network-free
        /// Heuristic search backend as an anchor opponent (seat alternates
        /// between anchor games). Anchor games break mirror-play symmetry: a
        /// passive net LOSES them, so the relative value label finally
        /// carries an anti-passivity gradient. The anchor side's data is
        /// recorded too (fresh teacher data, same as the BC corpus). Decays
        /// with `iteration` the same way `prior_heuristic_weight` does (see
        /// `decay_crutch`), then fully to 0 at --decay-last-iter. Mutually
        /// exclusive with --opponent.
        #[arg(long, default_value_t = 0.0)]
        anchor_frac: f32,

        /// Iteration at which both heuristic crutches (the search-prior
        /// blend and anchor-game rate) hard-cut to 0, having spent the
        /// iterations before that decaying down to a 10% floor. Default is
        /// effectively "never" so standalone/benchmark runs aren't
        /// surprised; the training loop passes an explicit value (see
        /// DECAY_LAST_ITER in run_training_loop.sh).
        #[arg(long, default_value_t = usize::MAX)]
        decay_last_iter: usize,

        /// EXP_ELO_004: weight of the TD(lambda) delta vs the final-outcome
        /// tail in the value target (no-op if --no-reward-shaping is set; see
        /// the TD_W const rationale). Default preserves production behavior.
        #[arg(long, default_value_t = TD_W)]
        td_w: f32,

        /// TD(lambda) trace decay in the value label. Sets the credit window's
        /// center of mass to 1/(1-lambda) turns (0.8 -> 5, 0.875 -> 8) and, as
        /// the same parameter, the lambda^n terminal tail INSIDE the TD arm —
        /// the two cannot be dialed apart. The flat 30% outcome share is
        /// `1 - td_w`, independent of this.
        #[arg(long, default_value_t = LAMBDA_RETURN)]
        td_lambda: f32,

        /// EXP_ELO_021: scale on the relative final-outcome ratio before the
        /// [-1,1] clamp in the value LABEL (label-only — not the in-tree
        /// backup, so no EXP_ELO_005 search-disruption risk). Default 3.0
        /// saturates ~32% of outcomes at ±1; lowering it de-saturates so the
        /// value head can learn to distinguish "ahead" from "crushing".
        #[arg(long, default_value_t = 3.0)]
        outcome_scale: f32,

        /// EXP_ELO_006: relative weight used ONLY for TD(lambda) label
        /// windows; the in-tree backup keeps reward::REL_W. Default
        /// preserves production behavior (labels match the backup).
        #[arg(long, default_value_t = reward::REL_W)]
        label_rel_w: f32,

        /// EXP_ELO_002: iteration where the anchor-frac decay clock starts —
        /// the anchor's effective decay iteration is `iteration - this`
        /// (clamped at 0). The loop passes the current iteration to HOLD
        /// anchor_frac at its starting rate until the model crosses 50% vs
        /// Greedy, then pins the crossing iteration so decay runs from
        /// there. The prior-blend decay is unaffected.
        #[arg(long, default_value_t = 0)]
        anchor_decay_start: usize,

        /// Force both heuristic crutches to 0 immediately, regardless of
        /// iteration or --decay-last-iter. Integration point for a future
        /// strength-gated phase-out (model consistently beats the
        /// heuristic-only backend) — not wired to any automatic check yet.
        #[arg(long, default_value_t = false)]
        force_zero_crutches: bool,

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

        /// Opt out of reward shaping (the blended per-step TD(lambda) +
        /// final-outcome value target). On by default — EXP_ELO_004 (Jul 13)
        /// found the flat final-outcome-only fallback trains markedly
        /// slower/weaker at matched budget. Pass this to fall back to a flat
        /// final-outcome value for every action (e.g. to reproduce pre-Jul-13
        /// runs or isolate a regression).
        #[arg(long, default_value_t = false)]
        no_reward_shaping: bool,

        /// EXP_ELO_011: ±1 win/loss value labels from the adjudicated winner,
        /// replacing the score-ratio final outcome — a close loss reads -1,
        /// not ~-0.14. Composes with --td-w blending unless shaping is off.
        #[arg(long, default_value_t = false)]
        wl_labels: bool,

        /// EXP_ELO_016: weight on the development potential Φ in TD-label
        /// snapshots (`score + w·Φ`). 0 = raw score deltas (legacy).
        #[arg(long, default_value_t = 0.0)]
        shape_w_label: f32,

        /// EXP_ELO_016: weight on Φ in the Gumbel in-tree edge rewards.
        /// Threaded separately from the label weight (EXP_ELO_005 lesson:
        /// search reacts violently to reward changes). 0 = legacy.
        #[arg(long, default_value_t = 0.0)]
        shape_w_tree: f32,

        /// EXP_ELO_018: weight on the isolated pursuit-progress potential Φ
        /// in TD-label snapshots (`score + w·Φ_pursuit`), independent of
        /// --shape-w-label. 0 = off.
        #[arg(long, default_value_t = 0.0)]
        pursuit_w_label: f32,

        /// EXP_ELO_018: weight on the pursuit-progress Φ in the Gumbel
        /// in-tree edge rewards, independent of --shape-w-tree. Half-dose
        /// vs the label weight (EXP_ELO_005 lesson). 0 = off.
        #[arg(long, default_value_t = 0.0)]
        pursuit_w_tree: f32,

        /// EXP_ELO_017: unfreeze the opponent during in-tree EndTurn
        /// crossings (Gumbel backend only) — each intervening opponent
        /// plays a real deterministic-Greedy turn instead of the engine's
        /// blind auto-skip. Training-data generation only; arena/gauge
        /// binaries always search frozen so every prior strength reading
        /// stays a valid yardstick.
        #[arg(long, default_value_t = false)]
        unfreeze_opponent: bool,

        /// EXP_ELO_020: DAgger expert dose. At each net-seat decision, blend
        /// Greedy's move-ranking at the MODEL'S OWN state into the policy
        /// target: `(1-a)*mcts + a*greedy`. 0 = off. Corrects the collapsed
        /// capture prior on-distribution (unlike BC, which labels Greedy's
        /// states). Net seats only; frozen search recommended to isolate.
        #[arg(long, default_value_t = 0.0)]
        dagger_alpha: f32,

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

        /// Diagnostics: dump games where NO village was captured by either
        /// player into this dir — <base>.replay.json (watcher-loadable) plus
        /// <base>.decisions.json (search trace for every decision; forces
        /// fresh root builds, so within-turn tree reuse is off).
        #[arg(long)]
        dump_failed_dir: Option<String>,

        /// Trajectory diagnostics: append one JSON record per player-turn
        /// (at turn start, before any moves) to <dir>/game<idx>.jsonl — the
        /// acting player's owned cities, FOW-visible uncaptured villages, and
        /// unit tiles. Ungated; the Python analysis does all filtering.
        #[arg(long)]
        dump_turn_states: Option<String>,

        /// Diagnostics: append one JSON record per city level-up reward
        /// choice (turn, player, city level/population/stars pre-choice,
        /// reward type chosen) to <dir>/game<idx>.jsonl. Ungated, ply-cheap
        /// (no MCTS trace overhead) — ordinary self-play run.
        #[arg(long)]
        dump_city_rewards: Option<String>,

        /// Value-head calibration: append one JSON record per net-seat step to
        /// <file> — {turn, my_score, opp_score, root_value, final_outcome,
        /// value_target}. For measuring whether the value head's prediction
        /// beats a plain current-score-ratio baseline at predicting the game
        /// outcome (does it have foresight, or just read the scoreboard?).
        #[arg(long)]
        dump_value_calib: Option<String>,

        /// Diagnostics: append one JSON record per Research/Harvest/Build/
        /// Summon move executed — (turn, player, move type, stars spent,
        /// read as the real tribe.stars delta) to <dir>/game<idx>.jsonl.
        #[arg(long)]
        dump_star_spend: Option<String>,
    }

    let args = Args::parse();

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

                    let seed = (base_seed + i as u64) as i64;
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
                    let (backend_seat1, backend_seat2) = if is_anchor {
                        if anchor_ordinal % 2 == 0 {
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
                        if anchor_ordinal % 2 == 0 {
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
                            args.dump_turn_states.as_deref(),
                            args.dump_city_rewards.as_deref(),
                            args.dump_star_spend.as_deref(),
                            seat_roles,
                            args.shape_w_label,
                            args.shape_w_tree,
                            args.pursuit_w_label,
                            args.pursuit_w_tree,
                            args.unfreeze_opponent,
                            args.dagger_alpha,
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

    let results: Vec<GameResult> = match Arc::try_unwrap(results_mutex) {
        Ok(mutex) => mutex.into_inner().unwrap(),
        Err(_) => {
            panic!("BUG: actor threads still hold a results_mutex reference after scope exit")
        }
    };

    let games_duration = games_start.elapsed();
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

    // Aux-head targets (see the aux_* helpers above GameResult).
    let num_techs = polyfish::types::TechnologyType::iter().count();
    let mut collected_aux_own: Vec<Vec<f32>> = Vec::new();
    let mut collected_aux_fog: Vec<Vec<f32>> = Vec::new();
    let mut collected_aux_spt: Vec<f32> = Vec::new(); // flat, 2 per step
    let mut collected_aux_tech: Vec<Vec<f32>> = Vec::new();
    let mut collected_aux_pursuit: Vec<f32> = Vec::new(); // scalar per step

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
    let mut total_t2c = [0.0f64; 6]; // villages p50/p80/all, ruins p50/p80/all
    let (mut first_cap_seats, mut first_cap_captured) = (0u32, 0u32);
    let mut first_cap_turn_sum = 0.0f64;
    let mut first_cap_censored_sum = 0.0f64;
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

    // Value-head calibration dump (one JSON line per net-seat step).
    let mut value_calib_file = args
        .dump_value_calib
        .as_ref()
        .and_then(|p| File::create(p).ok());

    for result in results {
        total_moves += result.moves;
        total_net_moves += result.net_moves;
        total_revealed_tiles += result.revealed_tiles as i64;
        total_captured_tiles += result.captured_tiles as i64;
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
        let final_scores = &result.scores;

        let label_steps: Vec<LabelStep> = result.history.iter().map(LabelStep::from).collect();
        let td_deltas = td_lambda_labels(
            &label_steps,
            &result.final_potentials,
            args.td_lambda,
            args.label_rel_w,
        );

        let spt_steps: Vec<SptStep> = result
            .history
            .iter()
            .map(|s| SptStep {
                player_id: s.player_id,
                turn: s.turn,
                my_spt: s.my_spt,
                opp_spt: s.opp_spt,
            })
            .collect();
        let spt_cp = spt_checkpoints_by_player(&spt_steps);

        let game_winner_id = result.winner_id;

        for (step_idx, step) in result.history.into_iter().enumerate() {
            let HistoryStep {
                features,
                policy: policy_data,
                player_id: p_id,
                turn,
                enemy_units,
                pursuit,
                my_score: step_my_score,
                opp_score: step_opp_score,
                root_value: step_root_value,
                root_own_value: step_root_own_value,
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
            if !args.no_reward_shaping {
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
            let scaling_factor = args.outcome_scale;
            let relative_outcome = if combined_score > 0.0 {
                let ratio = (my_adjusted - opp_adjusted) / combined_score;
                (ratio * scaling_factor).clamp(-1.0, 1.0)
            } else {
                0.0 // Both players scored 0 - treat as draw
            };

            // Absolute value: final score vs fixed yardstick, not current scoreboard.
            let abs_outcome = (my_final / GOOD_BOT_FINAL_SCORE).clamp(0.0, 1.0) * 2.0 - 1.0;
            let final_outcome = if args.wl_labels {
                // EXP_ELO_011: the score ratio under-punishes close losses
                // (4257 vs 4676 reads -0.14); win/loss makes them -1.
                if p_id == game_winner_id {
                    1.0
                } else {
                    -1.0
                }
            } else {
                (FINAL_OUTCOME_REL_W * relative_outcome
                    + (1.0 - FINAL_OUTCOME_REL_W) * abs_outcome)
                    .clamp(-1.0, 1.0)
            };

            let value = if !args.no_reward_shaping {
                // TD delta carries per-action credit; the final-outcome tail
                // carries the long-horizon signal.
                (args.td_w * td_deltas[step_idx] + (1.0 - args.td_w) * final_outcome)
                    .clamp(-1.0, 1.0)
            } else {
                final_outcome.clamp(-1.0, 1.0)
            };

            collected_values.push(value);

            // Value-head calibration: NN prediction vs current-score-ratio vs
            // the actual final outcome, for net seats that ran a real search.
            if let (Some(f), Some(rv)) = (value_calib_file.as_mut(), step_root_value) {
                if is_net_seat(result.roles, p_id) {
                    let raw = step_root_own_value.unwrap_or(rv);
                    let _ = writeln!(
                        f,
                        "{{\"turn\":{turn},\"my\":{step_my_score},\"opp\":{step_opp_score},\"root_value\":{rv},\"raw_value\":{raw},\"final_outcome\":{final_outcome},\"value_target\":{value}}}"
                    );
                }
            }

            let my_final_cities = result.final_cities.get(&p_id).copied().unwrap_or(0) as f32;
            let total_cities = result.total_cities as f32;
            let progress_target = if total_cities > 0.0 {
                (my_final_cities / total_cities).clamp(0.0, 1.0) * 2.0 - 1.0
            } else {
                -1.0
            };
            collected_progress.push(progress_target);

            let opp_id = final_scores
                .keys()
                .copied()
                .find(|id| *id != p_id)
                .unwrap_or(p_id);
            collected_aux_own.push(ownership_from_pov(&result.final_owner, p_id));
            collected_aux_fog.push(enemy_units);
            let (spt_my, spt_opp) = spt_target(
                spt_cp.get(&p_id),
                turn,
                result.final_spt.get(&p_id).copied().unwrap_or(0),
                result.final_spt.get(&opp_id).copied().unwrap_or(0),
            );
            collected_aux_spt.push(spt_my as f32 / 20.0);
            collected_aux_spt.push(spt_opp as f32 / 20.0);
            collected_aux_pursuit.push(pursuit);
            collected_aux_tech.push(
                result
                    .final_tech
                    .get(&opp_id)
                    .cloned()
                    .unwrap_or_else(|| vec![0.0; num_techs]),
            );
        }
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

        // Aux-head targets — always emitted together (train.py's per-file
        // presence mask treats them as all-or-nothing).
        let aux_own_tensor = Tensor::from_vec(
            flatten_vec(collected_aux_own),
            (total_steps, spatial_logit_dim),
            &device,
        )?;
        let aux_fog_tensor = Tensor::from_vec(
            flatten_vec(collected_aux_fog),
            (total_steps, spatial_logit_dim),
            &device,
        )?;
        let aux_spt_tensor = Tensor::from_vec(collected_aux_spt, (total_steps, 2), &device)?;
        let aux_pursuit_tensor =
            Tensor::from_vec(collected_aux_pursuit, (total_steps, 1), &device)?;
        let aux_tech_tensor = Tensor::from_vec(
            flatten_vec(collected_aux_tech),
            (total_steps, num_techs),
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

        tensors.insert("aux_ownership".to_string(), aux_own_tensor);
        tensors.insert("aux_fog_units".to_string(), aux_fog_tensor);
        tensors.insert("aux_spt".to_string(), aux_spt_tensor);
        tensors.insert("aux_opp_tech".to_string(), aux_tech_tensor);
        tensors.insert("aux_pursuit".to_string(), aux_pursuit_tensor);

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
        "tempo_by_turn": tempo_by_turn,
    });
    std::fs::write(
        ".last_self_play_metrics.json",
        serde_json::to_string(&metrics)?,
    )?;

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
    // How often a simulated move failed to execute against the replayed state
    // (tree-reuse staleness in Gumbel MCTS — see SIM_MOVE_FAILURES doc comment
    // in game.rs). Set POLYFISH_VERBOSE_SIM_FAILURES=1 for illegal_moves/*.json dumps.
    let sim_move_failures =
        polyfish::game::SIM_MOVE_FAILURES.load(std::sync::atomic::Ordering::Relaxed);
    println!(
        "  - Sim move failures: {} total ({:.2} per move decision)",
        sim_move_failures,
        sim_move_failures as f64 / (total_moves as f64).max(1.0)
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
            my_score: my as f32,
            opp_score: opp as f32,
            root_value: rv,
        }
    }

    fn finals(pairs: &[(i32, i32)]) -> HashMap<i32, f32> {
        pairs.iter().map(|&(id, s)| (id, s as f32)).collect()
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
            let out = td_lambda_labels(&history, &final_scores, lambda, reward::REL_W);
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

        let out = td_lambda_labels(&history, &final_scores, 0.0, reward::REL_W);

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
        let out2 = td_lambda_labels(&history2, &final_scores, 0.0, reward::REL_W);
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

        let out = td_lambda_labels(&history, &final_scores, 0.5, reward::REL_W);

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

        let out = td_lambda_labels(&history, &final_scores, 0.0, reward::REL_W);
        let expected = reward::normalized_reward(1000, 800, 1200, 800).clamp(-1.0, 1.0);
        assert!(
            (out[0] - expected).abs() < 1e-6,
            "got {}, expected {expected}",
            out[0]
        );
    }

    #[test]
    fn label_rel_w_reprices_windows() {
        // I gain 100 while the opponent gains 400 over one window: an
        // abs-only weighting must label it positive, a rel-only one negative
        // — proves the flag actually reaches the window pricing.
        let history = vec![
            step(1, 5, 1000, 800, Some(0.0)),
            step(1, 6, 1100, 1200, Some(0.0)),
        ];
        let final_scores = finals(&[(1, 1100), (2, 1200)]);

        let abs_only = td_lambda_labels(&history, &final_scores, 0.0, 0.0);
        let rel_only = td_lambda_labels(&history, &final_scores, 0.0, 1.0);
        assert!(abs_only[0] > 0.0, "abs-only label should be positive, got {}", abs_only[0]);
        assert!(rel_only[0] < 0.0, "rel-only label should be negative, got {}", rel_only[0]);
    }
}

#[cfg(test)]
mod decay_crutch_tests {
    use super::*;

    #[test]
    fn decays_toward_floor_before_taper() {
        let w0 = decay_crutch(HEURISTIC_PRIOR_W0, HEURISTIC_PRIOR_DECAY, 0, 150, false);
        assert!(
            (w0 - HEURISTIC_PRIOR_W0).abs() < 1e-6,
            "iteration 0 should equal w0, got {w0}"
        );

        let mid = decay_crutch(HEURISTIC_PRIOR_W0, HEURISTIC_PRIOR_DECAY, 23, 150, false);
        assert!(
            (mid - 0.25).abs() < 0.01,
            "iteration 23 should be ~0.25, got {mid}"
        );

        let floored = decay_crutch(HEURISTIC_PRIOR_W0, HEURISTIC_PRIOR_DECAY, 100, 150, false);
        assert!(
            (floored - CRUTCH_FLOOR).abs() < 1e-6,
            "past-decay iteration should sit at the floor, got {floored}"
        );
    }

    #[test]
    fn hard_cuts_to_zero_at_decay_last_iter() {
        let at_cutoff = decay_crutch(HEURISTIC_PRIOR_W0, HEURISTIC_PRIOR_DECAY, 150, 150, false);
        assert_eq!(at_cutoff, 0.0);

        let past_cutoff = decay_crutch(HEURISTIC_PRIOR_W0, HEURISTIC_PRIOR_DECAY, 500, 150, false);
        assert_eq!(past_cutoff, 0.0);

        let just_before = decay_crutch(HEURISTIC_PRIOR_W0, HEURISTIC_PRIOR_DECAY, 149, 150, false);
        assert!((just_before - CRUTCH_FLOOR).abs() < 1e-6);
    }

    #[test]
    fn force_zero_overrides_regardless_of_iteration() {
        let forced = decay_crutch(
            HEURISTIC_PRIOR_W0,
            HEURISTIC_PRIOR_DECAY,
            0,
            usize::MAX,
            true,
        );
        assert_eq!(forced, 0.0);
    }
}

#[cfg(test)]
mod aux_target_tests {
    use super::*;
    use polyfish::coords::Coords;
    use polyfish::states::{TechnologyState, TribeState, UnitState};
    use polyfish::types::{TechnologyType, UnitEffect};

    #[test]
    fn tech_multihot_uses_iter_position_not_discriminant() {
        let mk = |tech_type, discovered| TechnologyState {
            tech_type,
            discovered,
            discovered_turn: 0,
        };
        let techs = vec![
            mk(TechnologyType::Riding, true),
            mk(TechnologyType::ShockTactics, true),
            mk(TechnologyType::Rituals, true),
            mk(TechnologyType::Fishing, false),
        ];
        let v = tech_multihot(&techs);
        let n = TechnologyType::iter().count();
        assert_eq!(v.len(), n);
        assert_eq!(v.iter().filter(|&&x| x == 1.0).count(), 3);
        let rituals_pos = TechnologyType::iter()
            .position(|t| t == TechnologyType::Rituals)
            .unwrap();
        assert_eq!(v[rituals_pos], 1.0);
        // Discriminant-indexed encoding would need a slot at 121 >= n.
        assert!(TechnologyType::Rituals as usize >= n);
    }

    #[test]
    fn ownership_from_pov_maps_signs() {
        let owner = vec![0, 1, 2];
        assert_eq!(ownership_from_pov(&owner, 1), vec![0.0, 1.0, -1.0]);
        assert_eq!(ownership_from_pov(&owner, 2), vec![0.0, -1.0, 1.0]);
    }

    #[test]
    fn enemy_unit_grid_excludes_pov_invisible_and_bounds() {
        let unit = |owner: PlayerId, idx: i32, invisible: bool| {
            let mut u = UnitState {
                owner,
                coords: Coords::from_index(idx, 11),
                ..Default::default()
            };
            if invisible {
                u.effects.insert(UnitEffect::Invisible);
            }
            u
        };
        let mut state = GameState::default();
        let mut t1 = TribeState::default();
        t1.units.push(unit(1, 5, false));
        let mut t2 = TribeState::default();
        t2.units.push(unit(2, 17, false));
        t2.units.push(unit(2, 30, true)); // invisible: excluded
        t2.units.push(unit(2, 500, false)); // out of range: excluded
        state.tribes.insert(1, t1);
        state.tribes.insert(2, t2);

        let g = enemy_unit_grid(&state, 1, 121);
        let set: Vec<usize> = g
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v == 1.0)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(set, vec![17]);
    }

    fn sstep(player_id: PlayerId, turn: i32, my: i32, opp: i32) -> SptStep {
        SptStep {
            player_id,
            turn,
            my_spt: my,
            opp_spt: opp,
        }
    }

    #[test]
    fn spt_checkpoints_keep_first_decision_per_turn() {
        let steps = vec![sstep(1, 3, 5, 4), sstep(1, 3, 9, 9), sstep(1, 4, 6, 5)];
        let cp = spt_checkpoints_by_player(&steps);
        let c1 = &cp[&1];
        assert_eq!(c1.len(), 2);
        assert_eq!((c1[0].turn, c1[0].my_spt), (3, 5));
        assert_eq!((c1[1].turn, c1[1].my_spt), (4, 6));
    }

    #[test]
    fn spt_target_five_turn_lookup_and_final_fallback() {
        let steps = vec![
            sstep(1, 0, 2, 2),
            sstep(1, 3, 4, 3),
            sstep(1, 9, 8, 6),
            sstep(1, 12, 10, 9),
        ];
        let cp = spt_checkpoints_by_player(&steps);
        // T=3: first turn >= 8 is 9.
        assert_eq!(spt_target(cp.get(&1), 3, 99, 99), (8, 6));
        // T=4: exact boundary, turn 9 == 4+5.
        assert_eq!(spt_target(cp.get(&1), 4, 99, 99), (8, 6));
        // T=7: first turn >= 12 is 12 (present exactly).
        assert_eq!(spt_target(cp.get(&1), 7, 0, 0), (10, 9));
        // T=9: nothing at >= 14 -> final fallback.
        assert_eq!(spt_target(cp.get(&1), 9, 99, 98), (99, 98));
        // Unknown player -> final fallback.
        assert_eq!(spt_target(cp.get(&7), 0, 1, 2), (1, 2));
    }
}
