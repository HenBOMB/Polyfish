//! Decision-trace capture: which moment to trace (TraceTrigger and the four
//! finders that detect it), and the two writers that put a captured trace on
//! disk. One file per decision, so concurrent actors never contend.

use polyfish::TribeType;
use polyfish::ai::brain::SearchBackend;
use polyfish::replayer::ModReplay;
use polyfish::ai::brain::Brain;
use polyfish::states::{GameState, PlayerId};
use std::sync::atomic::{AtomicUsize, Ordering};
use serde_json::json;
use std::collections::HashMap;

/// Which village-approach moment to capture a decision trace for (see
/// decision_trace.rs / find_village_trigger).
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TraceTrigger {
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
pub(crate) fn find_village_trigger(
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
pub(crate) fn find_harvest_trigger(state: &GameState, pov: PlayerId) -> Option<i32> {
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
pub(crate) fn find_village_pursuit_trigger(
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
pub(crate) fn find_wander_trigger(
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
pub(crate) fn write_decision_trace(
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
pub(crate) struct TracedDecision {
    pub(crate) turn: i32,
    pub(crate) move_count: usize,
    pub(crate) player_id: PlayerId,
    pub(crate) chosen: String,
    pub(crate) root_value: Option<f32>,
    pub(crate) trace: Option<polyfish::ai::decision_trace::DecisionTrace>,
}

/// Dump one zero-capture game (no village taken by either player): a
/// watcher-loadable replay plus the full per-decision trace log, tagged with
/// the matchup and seat backends. One file pair per game — actor-safe.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dump_failed_game(
    dir: &str,
    // File-name stem: "failed" for the zero-capture filter, "game" for a
    // plain observability dump — the name should not claim a verdict.
    prefix: &str,
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
    let base = format!("{prefix}_iter{iteration}_game{game_idx}_seed{seed}");
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

/// Writes whichever trigger fired on this ply, in a fixed order.
///
/// `take_trace()` CONSUMES the search's captured trace, so at most one of
/// these branches can produce output per ply and the order they are tested in
/// is the priority order. Do not reorder or merge them.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_ply_traces(
    state: &GameState,
    pov: PlayerId,
    agent: &mut Brain<'_>,
    open_villages: &std::collections::HashSet<i32>,
    trace_counter: &AtomicUsize,
    trace_max: usize,
    iteration: usize,
    game_idx: usize,
    move_count: usize,
    trigger_info: Option<(i32, i32)>,
    harvest_capture: Option<(i32, i32)>,
    pursuit_capture: Option<(i32, i32)>,
    wander_capture: Option<(i32, i32)>,
    traces_this_game: &mut usize,
    last_trace_turn: &mut i32,
) {
    if let Some((trigger_unit_idx, trigger_village_idx)) = trigger_info {
        if let Some(trace) = agent.take_trace() {
            if trace_counter.fetch_add(1, Ordering::Relaxed) < trace_max {
                let mut visible_villages: Vec<i32> = open_villages
                    .iter()
                    .copied()
                    .filter(|idx| {
                        state
                            .tiles
                            .get(idx)
                            .map_or(false, |t| t.explorers.contains(&pov))
                    })
                    .collect();
                // Sorted: `open_villages` is a std HashSet, so its iteration order is
                // randomized per process and would otherwise leak into this dump.
                visible_villages.sort_unstable();
                write_decision_trace(
                    "decision_traces",
                    &trace,
                    iteration,
                    game_idx,
                    state.settings.turn,
                    move_count,
                    pov,
                    trigger_unit_idx,
                    trigger_village_idx,
                    &visible_villages,
                    None,
                );
            }
            *traces_this_game += 1;
            *last_trace_turn = state.settings.turn;
        }
    }

    if let Some((trigger_tile, turns_since)) = harvest_capture {
        if let Some(trace) = agent.take_trace() {
            if trace_counter.fetch_add(1, Ordering::Relaxed) < trace_max {
                write_decision_trace(
                    "decision_traces_harvest",
                    &trace,
                    iteration,
                    game_idx,
                    state.settings.turn,
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
        if let Some(trace) = agent.take_trace() {
            if trace_counter.fetch_add(1, Ordering::Relaxed) < trace_max {
                write_decision_trace(
                    "decision_traces_pursuit",
                    &trace,
                    iteration,
                    game_idx,
                    state.settings.turn,
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
        if let Some(trace) = agent.take_trace() {
            if trace_counter.fetch_add(1, Ordering::Relaxed) < trace_max {
                let mut visible_villages: Vec<i32> = open_villages
                    .iter()
                    .copied()
                    .filter(|idx| {
                        state
                            .tiles
                            .get(idx)
                            .map_or(false, |t| t.explorers.contains(&pov))
                    })
                    .collect();
                // Sorted: `open_villages` is a std HashSet, so its iteration order is
                // randomized per process and would otherwise leak into this dump.
                visible_villages.sort_unstable();
                write_decision_trace(
                    "decision_traces_wander",
                    &trace,
                    iteration,
                    game_idx,
                    state.settings.turn,
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
}

/// Per-game trace-window state.
///
/// Three of the six triggers are *windows*: they fire once, then keep
/// capturing every ply of the triggering seat for the next three game turns.
/// The other two are single-shot and rate-limited by `traces_this_game` /
/// `last_trace_turn`. All of it is per-game, so it lives here rather than as
/// eleven loose locals in the ply loop.
#[derive(Default)]
pub(crate) struct TraceWindows {
    pub(crate) traces_this_game: usize,
    pub(crate) last_trace_turn: i32,
    harvest_trigger_turn: Option<i32>,
    harvest_trigger_tile: i32,
    harvest_trigger_pov: Option<PlayerId>,
    pursuit_trigger_turn: Option<i32>,
    pursuit_trigger_village: i32,
    pursuit_trigger_pov: Option<PlayerId>,
    wander_trigger_turn: Option<i32>,
    wander_trigger_unit: i32,
    wander_trigger_pov: Option<PlayerId>,
}

impl TraceWindows {
    /// `last_trace_turn` starts far in the past so the first eligible ply is
    /// not blocked by the >= 3-turn spacing rule.
    pub(crate) fn new() -> Self {
        Self { last_trace_turn: -100, ..Default::default() }
    }

    /// Which triggers, if any, fire on this ply:
    /// `(village, harvest, pursuit, wander)`. Opens a window the first time
    /// its condition holds, then reports every ply inside it.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn for_ply(
        &mut self,
        state: &GameState,
        pov: PlayerId,
        trace_villages: bool,
        trace_all: bool,
        trace_trigger: TraceTrigger,
        trace_counter: &AtomicUsize,
        trace_max: usize,
        open_villages: &std::collections::HashSet<i32>,
    ) -> (Option<(i32, i32)>, Option<(i32, i32)>, Option<(i32, i32)>, Option<(i32, i32)>) {
    let trigger_info = if trace_villages
        && !trace_all
        && !matches!(
            trace_trigger,
            TraceTrigger::HarvestReady | TraceTrigger::VillagePursuit | TraceTrigger::Wander
        )
        && self.traces_this_game < 3
        && state.settings.turn >= self.last_trace_turn + 3
        && trace_counter.load(Ordering::Relaxed) < trace_max
    {
        find_village_trigger(&state, pov, &open_villages, trace_trigger)
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
        if self.harvest_trigger_turn.is_none() {
            if let Some(tile) = find_harvest_trigger(&state, pov) {
                self.harvest_trigger_turn = Some(state.settings.turn);
                self.harvest_trigger_tile = tile;
                self.harvest_trigger_pov = Some(pov);
            }
        }
        self.harvest_trigger_turn.and_then(|start| {
            let turn = state.settings.turn;
            (self.harvest_trigger_pov == Some(pov) && turn <= start + 3)
                .then_some((self.harvest_trigger_tile, turn - start))
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
        if self.pursuit_trigger_turn.is_none() {
            if let Some(village) =
                find_village_pursuit_trigger(&state, pov, &open_villages)
            {
                self.pursuit_trigger_turn = Some(state.settings.turn);
                self.pursuit_trigger_village = village;
                self.pursuit_trigger_pov = Some(pov);
            }
        }
        self.pursuit_trigger_turn.and_then(|start| {
            let turn = state.settings.turn;
            (self.pursuit_trigger_pov == Some(pov) && turn <= start + 3)
                .then_some((self.pursuit_trigger_village, turn - start))
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
        if self.wander_trigger_turn.is_none() {
            if let Some(unit) = find_wander_trigger(&state, pov, &open_villages) {
                self.wander_trigger_turn = Some(state.settings.turn);
                self.wander_trigger_unit = unit;
                self.wander_trigger_pov = Some(pov);
            }
        }
        self.wander_trigger_turn.and_then(|start| {
            let turn = state.settings.turn;
            (self.wander_trigger_pov == Some(pov) && turn <= start + 3)
                .then_some((self.wander_trigger_unit, turn - start))
        })
    } else {
        None
    };
        (trigger_info, harvest_capture, pursuit_capture, wander_capture)
    }
}
