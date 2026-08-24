//! EXP_ELO_032: goal-conditioned deterministic whole-turn executor. Given a
//! fixed `MacroGoal`, plays out one player's turn ply-by-ply on a rollout
//! clone — the micro half of the macro-search bootstrap. Reuses the
//! oracle_macro root gates and prices plies as
//! `score_move + λ·Δgoal_potential` (the edge_snapshot pattern).

use crate::ai::oracle_macro::{
    LaneState, GoalAux, MacroGoal, Stance, tech_discipline_active, passes_ability_gate,
    passes_capture_first, passes_stance_tech_mask, passes_tech_purchase_limits, compute_goal_aux,
    observe_lane_state,
};
use crate::ai::{reward, scoring};
use crate::game::Game;
use crate::moves::{EndTurnMove, Move};
use crate::states::{GameState, PlayerId};
use crate::types::MoveType;

/// Whole-game purchase counters the tech-cap gates read (mirrors the
/// goal_script counting in arena.rs / self_play.rs).
#[derive(Clone, Copy, Debug, Default)]
pub struct TurnCounters {
    pub techs_bought: u32,
    pub tier3_bought: u32,
}

impl TurnCounters {
    /// Count `m` if it is a Research purchase (tier-3 tracked separately).
    pub fn count(&mut self, m: &dyn Move) {
        if m.move_type() != MoveType::Research {
            return;
        }
        self.techs_bought += 1;
        if let Ok(tech) = m.tech_type() {
            if crate::settings::technology::get_technology_setting(tech).tier == Some(3) {
                self.tier3_bought += 1;
            }
        }
    }
}

/// Diagnostic: how many times `rank_plies` has been called (rollout +
/// real-commit sites combined) and how many candidate moves it scored in
/// total, across the process. Ratio to `self_play`'s own "moves" count
/// gives calls-per-real-move — the input to the ply-distillation throughput
/// envelope (EXP_ELO_061 GPU-ply-work plan, Phase 0). Always-on, one atomic
/// add per call/candidate — negligible cost, mirrors `SIM_MOVE_FAILURES`.
pub static RANK_PLIES_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static RANK_PLIES_CANDIDATES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Phase 0 instrumentation for the ply-distillation plan (not a standing
/// feature): when `POLYFISH_DPHI_PROBE=<path>` is set, `rank_plies` appends
/// one JSONL row per scored candidate with its decomposed move coordinates
/// and both the real (aux-aware) Δφ and an aux-free Δφ, so the GoalAux-
/// dependence and coordinate-collision questions can be answered offline
/// without touching the shard format. A no-op (one cached `OnceLock` read)
/// when the env var is unset.
///
/// Sampled 1-in-`DPHI_PROBE_SAMPLE_EVERY` calls and capped at
/// `DPHI_PROBE_MAX_ROWS` total rows, written through one persistent buffered
/// handle instead of open+write+close per row: an unsampled, unbuffered
/// version of this probe wrote ~111k rows/game via ~2.8k file opens, which
/// was enough of a write-storm to trip jetsam on a memory-pressured host
/// (EXP_ELO_065 Phase 0 harvest, Aug 2026) even though the run itself used
/// negligible RSS. Sampling by `call_id` also skips the aux-free
/// recomputation for unsampled calls, so it cuts CPU, not just IO.
const DPHI_PROBE_SAMPLE_EVERY: u64 = 20;
const DPHI_PROBE_MAX_ROWS: u64 = 2_000_000;
static DPHI_PROBE_ROWS_WRITTEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn dphi_probe_path() -> Option<&'static str> {
    static PATH: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    PATH.get_or_init(|| std::env::var("POLYFISH_DPHI_PROBE").ok())
        .as_deref()
}

/// `Some(path)` only for sampled calls; `None` otherwise so callers skip the
/// aux-free Δφ work entirely for the 19-in-20 calls that won't be written.
fn dphi_probe_path_sampled(call_id: u64) -> Option<&'static str> {
    dphi_probe_path().filter(|_| call_id % DPHI_PROBE_SAMPLE_EVERY == 0)
}

fn dphi_probe_writer(
    path: &'static str,
) -> &'static std::sync::Mutex<std::io::BufWriter<std::fs::File>> {
    static WRITER: std::sync::OnceLock<std::sync::Mutex<std::io::BufWriter<std::fs::File>>> =
        std::sync::OnceLock::new();
    WRITER.get_or_init(|| {
        let fh = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .expect("POLYFISH_DPHI_PROBE path must be writable");
        std::sync::Mutex::new(std::io::BufWriter::new(fh))
    })
}

/// Derived path for the Phase 0c state-feature dump (one entry per sampled
/// call, not per candidate — a full `RawFeatures` is ~82KB, so per-candidate
/// would repeat it 20-300x for no reason). Binary, not JSONL: `[call_id: u64
/// LE][n_spatial: u32][n_player: u32][spatial f32...][player f32...]`.
fn dphi_probe_features_path() -> Option<&'static str> {
    static PATH: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    PATH.get_or_init(|| dphi_probe_path().map(|p| format!("{p}.features.bin")))
        .as_deref()
}

fn dphi_probe_features_writer(
    path: &'static str,
) -> &'static std::sync::Mutex<std::io::BufWriter<std::fs::File>> {
    static WRITER: std::sync::OnceLock<std::sync::Mutex<std::io::BufWriter<std::fs::File>>> =
        std::sync::OnceLock::new();
    WRITER.get_or_init(|| {
        let fh = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .expect("dphi probe features path must be writable");
        std::sync::Mutex::new(std::io::BufWriter::new(fh))
    })
}

/// Encodes the same way the real distilled head would see this state at
/// inference (goal channels painted from `goal`); `pursuit_focus` is left
/// `None` — not load-bearing for the Phase 0c agreement check.
fn dphi_probe_write_state(call_id: u64, state: &GameState, player: PlayerId, goal: &MacroGoal) {
    let Some(path) = dphi_probe_features_path() else {
        return;
    };
    let Ok(feats) =
        crate::ai::features::state_to_cpu_features_goal(state, player, None, Some(goal))
    else {
        return;
    };
    use std::io::Write;
    if let Ok(mut w) = dphi_probe_features_writer(path).lock() {
        let _ = w.write_all(&call_id.to_le_bytes());
        let _ = w.write_all(&(feats.spatial.len() as u32).to_le_bytes());
        let _ = w.write_all(&(feats.player.len() as u32).to_le_bytes());
        for v in feats.spatial.iter().chain(feats.player.iter()) {
            let _ = w.write_all(&v.to_le_bytes());
        }
    }
}

/// Flush any buffered probe rows/features. Statics never run `Drop` at
/// process exit, so without this the last (sub-8KB) buffered chunk would be
/// silently lost; call once from `self_play`'s teardown. No-op if the probe
/// was never used.
pub fn dphi_probe_flush() {
    use std::io::Write;
    if let Some(path) = dphi_probe_path() {
        if let Ok(mut w) = dphi_probe_writer(path).lock() {
            let _ = w.flush();
        }
    }
    if let Some(path) = dphi_probe_features_path() {
        if let Ok(mut w) = dphi_probe_features_writer(path).lock() {
            let _ = w.flush();
        }
    }
}

/// `score_move`/`lambda` are logged alongside Δφ because `rank_plies` ranks
/// on `score_move + λ·Δφ`, not Δφ alone — the Phase 0c offline gate has to
/// reconstruct that same sum (with a predicted Δφ) to measure the real
/// top-1 agreement, and `score_move` can't be recomputed from Python.
#[allow(clippy::too_many_arguments)]
fn dphi_probe_row(
    path: &'static str,
    call_id: u64,
    turn: i32,
    player: PlayerId,
    m: &dyn Move,
    map_size: usize,
    score_move: f32,
    lambda: f32,
    dphi_full: f32,
    dphi_no_aux: f32,
) {
    if DPHI_PROBE_ROWS_WRITTEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        >= DPHI_PROBE_MAX_ROWS
    {
        return;
    }
    let t = crate::ai::mapper::DecomposedMapper::move_to_targets(m, map_size);
    let f = |x: Option<usize>| x.map(|v| v.to_string()).unwrap_or_else(|| "null".into());
    let row = format!(
        "{{\"call_id\":{call_id},\"turn\":{turn},\"player\":{player},\"move_type\":\"{:?}\",\"action_type\":{},\"source\":{},\"target\":{},\"option\":{},\"score_move\":{score_move:.6},\"lambda\":{lambda:.6},\"dphi_full\":{dphi_full:.6},\"dphi_no_aux\":{dphi_no_aux:.6}}}\n",
        m.move_type(),
        t.action_type,
        f(t.source_spatial),
        f(t.target_spatial),
        f(t.target_type),
    );
    use std::io::Write;
    if let Ok(mut w) = dphi_probe_writer(path).lock() {
        let _ = w.write_all(row.as_bytes());
    }
}

/// The four oracle_macro root gates, EndTurn always exempt (mirrors
/// gumbel_mcts::gate_retain, which stays private to keep its attribution
/// counters off this path).
pub fn gate_ok(
    state: &GameState,
    m: &dyn Move,
    star_gate: bool,
    stance: Option<Stance>,
    aux: Option<&GoalAux>,
) -> bool {
    if m.move_type() == MoveType::EndTurn {
        return true;
    }
    if star_gate && !passes_stance_tech_mask(state, m, stance, aux) {
        return false;
    }
    if let Some(a) = aux {
        if !passes_tech_purchase_limits(m, a)
            || !passes_ability_gate(state, m)
            || !passes_capture_first(state, m)
        {
            return false;
        }
    }
    true
}

/// Rank the current player's plies under a fixed goal, best first. EndTurn is
/// suppressed while any other move survives the gates (the search backends'
/// root convention); a fully gated-out ply degrades to a lone EndTurn.
/// Δφ probes one move at a time via simulate_move/undo — single-move undo is
/// safe; composed undos across a turn are not (see cross_end_turn).
pub fn rank_plies(
    game: &mut Game,
    player: PlayerId,
    goal: &MacroGoal,
    aux: &GoalAux,
    star_gate: bool,
    lambda: f32,
    unit_goals: Option<&crate::ai::search::unit_goals::UnitGoalStore>,
) -> Vec<(f32, Box<dyn Move>)> {
    // Pre-increment value doubles as a unique per-call ID for the dphi
    // probe below — two rows sharing a call_id came from the same ply
    // decision; two rows sharing only (turn, player) may not (many
    // rollout branches revisit the same turn number).
    let call_id = RANK_PLIES_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut moves = game.legal_moves();
    moves.retain(|m| gate_ok(&game.state, m.as_ref(), star_gate, Some(goal.stance), Some(aux)));
    let has_other = moves.iter().any(|m| m.move_type() != MoveType::EndTurn);
    if has_other {
        moves.retain(|m| m.move_type() != MoveType::EndTurn);
    }
    if moves.is_empty() {
        return vec![(0.0, Box::new(EndTurnMove) as Box<dyn Move>)];
    }
    RANK_PLIES_CANDIDATES.fetch_add(moves.len() as u64, std::sync::atomic::Ordering::Relaxed);

    // EXP_ELO_061 throughput fix: `threat_units` depends only on the
    // OPPONENT's units/ghosts, never on the acting player's own candidate
    // move, so it's computed once per ply here instead of once per
    // candidate inside goal_potential's city_risks call. Profiling found
    // that per-candidate re-scan was 64-86% of actor CPU time under
    // macro-mcts (see combat::city_risks_with_threats's doc comment).
    let threats = if lambda != 0.0 {
        Some(crate::ai::combat::threat_units(&game.state, player))
    } else {
        None
    };
    let phi_pre = if lambda != 0.0 {
        reward::goal_potential_with_unit_goals(&game.state, player, goal, Some(aux), threats.as_deref(), unit_goals)
    } else {
        0.0
    };
    let probe_path = dphi_probe_path_sampled(call_id);
    let phi_pre_no_aux = probe_path
        .map(|_| reward::goal_potential_with_threats(&game.state, player, goal, None, None));
    if probe_path.is_some() {
        dphi_probe_write_state(call_id, &game.state, player, goal);
    }
    let turn = game.state.settings.turn;
    let map_size = game.state.settings.size as usize;
    let mut scored: Vec<(f32, Box<dyn Move>)> = moves
        .into_iter()
        .map(|m| {
            let mut s = scoring::score_move_with_unit_goals(game, m.as_ref(), unit_goals);
            if lambda != 0.0 && m.move_type() != MoveType::EndTurn {
                if let Some(undo) = game.simulate_move(m.as_ref()) {
                    let phi_post = reward::goal_potential_with_unit_goals(
                        &game.state,
                        player,
                        goal,
                        Some(aux),
                        threats.as_deref(),
                        unit_goals,
                    );
                    if let Some(path) = probe_path {
                        let phi_post_no_aux = reward::goal_potential_with_threats(
                            &game.state,
                            player,
                            goal,
                            None,
                            None,
                        );
                        dphi_probe_row(
                            path,
                            call_id,
                            turn,
                            player,
                            m.as_ref(),
                            map_size,
                            s,
                            lambda,
                            phi_post - phi_pre,
                            phi_post_no_aux - phi_pre_no_aux.unwrap_or(0.0),
                        );
                    }
                    undo(&mut game.state);
                    s += lambda * (phi_post - phi_pre);
                }
            }
            (s, m)
        })
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    scored
}

/// Ply cap per executed turn, mirroring cross_end_turn's MAX_GHOST_MOVES.
pub const MAX_EXEC_PLIES: usize = 64;

/// Execute one full turn for `player` on a ROLLOUT CLONE under a fixed goal,
/// ending with `simulate_single_end_turn` (never the blind opponent-skipping
/// `simulate_move(EndTurn)`). Returns false on an anomaly (rejected move) —
/// the caller scores the rollout where it stopped.
pub fn execute_turn(
    game: &mut Game,
    player: PlayerId,
    goal: &MacroGoal,
    lane_state: &mut LaneState,
    counters: &mut TurnCounters,
    lambda: f32,
) -> bool {
    execute_turn_recorded(game, player, goal, lane_state, counters, lambda, None)
}

/// One executed ply, for the Tier-2/Tier-3 boundary probe (EXP_ELO_048).
/// `flip_no_phi` / `flip_no_goal` answer "would this ply have been chosen
/// anyway", with the directive's ranking term removed and with the directive
/// removed entirely — i.e. how much of the executor's choice the directive
/// actually owns.
#[derive(Clone, Debug)]
pub struct PlyRec {
    pub mv: String,
    pub kind: String,
    pub flip_no_phi: bool,
    pub flip_no_goal: bool,
}

/// `execute_turn` with an optional per-ply recorder. One implementation, so a
/// probe can never drift from the executor it is measuring; the recording
/// arms cost two extra `rank_plies` per ply and are OFF unless `rec` is set.
pub fn execute_turn_recorded(
    game: &mut Game,
    player: PlayerId,
    goal: &MacroGoal,
    lane_state: &mut LaneState,
    counters: &mut TurnCounters,
    lambda: f32,
    mut rec: Option<&mut Vec<PlyRec>>,
) -> bool {
    for _ in 0..MAX_EXEC_PLIES {
        if game.state.settings._game_over
            || game.state.settings.current_player_turn_id != player
        {
            return true;
        }
        observe_lane_state(&game.state, player, lane_state);
        let aux = compute_goal_aux(
            &game.state,
            player,
            goal,
            counters.techs_bought,
            counters.tier3_bought,
            Some(lane_state),
        );
        let gate = tech_discipline_active(&game.state, player, goal);
        // Rollout branches never see the real trajectory's UnitGoalStore
        // (Fork 2 of the per-unit-goal design: real-trajectory-only for
        // v1) -- this call stays byte-identical to pre-store behavior.
        let mut ranked = rank_plies(game, player, goal, &aux, gate, lambda, None);
        if ranked.is_empty() {
            break;
        }
        let (_, best) = ranked.swap_remove(0);
        if best.move_type() == MoveType::EndTurn {
            break;
        }
        if let Some(r) = rec.as_deref_mut() {
            let key = best.serialize().to_string();
            // Same gate, no directive PULL: isolates the lambda*dphi term.
            let no_phi = rank_plies(game, player, goal, &aux, gate, 0.0, None);
            // No directive at all: gate open, default goal — the whole Tier-2
            // channel removed, both filter and pull.
            let bare = MacroGoal::default();
            let bare_aux = compute_goal_aux(
                &game.state,
                player,
                &bare,
                counters.techs_bought,
                counters.tier3_bought,
                Some(lane_state),
            );
            let no_goal = rank_plies(game, player, &bare, &bare_aux, false, 0.0, None);
            let top = |v: &Vec<(f32, Box<dyn Move>)>| {
                v.first().map(|(_, m)| m.serialize().to_string()).unwrap_or_default()
            };
            r.push(PlyRec {
                kind: format!("{:?}", best.move_type()),
                flip_no_phi: top(&no_phi) != key,
                flip_no_goal: top(&no_goal) != key,
                mv: key,
            });
        }
        if game.simulate_move(best.as_ref()).is_none() {
            return false;
        }
        counters.count(best.as_ref());
    }
    let _ = game.simulate_single_end_turn();
    true
}

/// Ghost-play every intervening opponent with deterministic Greedy until
/// control returns to `pov` or the game ends (cross_end_turn's loop, minus
/// snapshot/undo — the caller's clone is throwaway). A runaway opponent turn
/// is force-ended so the rollout keeps advancing; returns false only on an
/// anomaly (no move / rejected move), where the caller scores the state as-is.
pub fn ghost_until(game: &mut Game, pov: PlayerId) -> bool {
    while game.state.settings.current_player_turn_id != pov && !game.state.settings._game_over {
        let mut n = 0usize;
        loop {
            if game.state.settings._game_over {
                return true;
            }
            if n >= MAX_EXEC_PLIES {
                let _ = game.simulate_single_end_turn();
                break;
            }
            match crate::ai::heuristic_mcts::GreedyHeuristicAgent.select_move(game) {
                None => return false,
                Some(mv) if mv.move_type() == MoveType::EndTurn => {
                    let _ = game.simulate_single_end_turn();
                    break;
                }
                Some(mv) => {
                    if game.simulate_move(mv.as_ref()).is_none() {
                        return false;
                    }
                }
            }
            n += 1;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::oracle_macro::{compute_macro_goal, commit_macro_goal, StanceCommit};

    fn generated_game(seed: i64) -> Game {
        let mut game = Game::new();
        game.state = crate::mapgen::generate(crate::mapgen::MapGenSettings {
            size: crate::types::MapSize::Tiny,
            map_type: crate::types::MapType::Drylands,
            tribes: vec![crate::types::TribeType::Imperius, crate::types::TribeType::Bardur],
            seed,
            version: 115,
        });
        game.post_load();
        game
    }

    #[test]
    fn executor_terminates_and_hands_over() {
        for seed in 0..8i64 {
            let game = generated_game(seed);
            let pov = game.state.settings.current_player_turn_id;
            let mut sim = game.clone_for_mcts(pov);
            let goal = compute_macro_goal(&sim.state, pov, 0);
            let mut lane_state = LaneState::default();
            let mut counters = TurnCounters::default();
            let ok = execute_turn(&mut sim, pov, &goal, &mut lane_state, &mut counters, 1.0);
            assert!(ok, "seed {seed}: executor bailed on its own turn");
            assert!(
                sim.state.settings.current_player_turn_id != pov
                    || sim.state.settings._game_over,
                "seed {seed}: turn never handed over"
            );
        }
    }

    #[test]
    fn executor_is_deterministic() {
        for seed in 0..4i64 {
            let game = generated_game(seed);
            let pov = game.state.settings.current_player_turn_id;
            let mut history = Vec::new();
            for _ in 0..2 {
                let mut sim = game.clone_for_mcts(pov);
                let goal = compute_macro_goal(&sim.state, pov, 0);
                let mut lane_state = LaneState::default();
                let mut counters = TurnCounters::default();
                execute_turn(&mut sim, pov, &goal, &mut lane_state, &mut counters, 1.0);
                history.push(sim.state._history.clone());
            }
            assert_eq!(history[0], history[1], "seed {seed}: two runs diverged");
        }
    }

    #[test]
    fn gates_never_strand_the_turn() {
        for seed in 0..4i64 {
            let mut game = generated_game(seed);
            let pov = game.state.settings.current_player_turn_id;
            let mut commit = StanceCommit::default();
            let goal = commit_macro_goal(&game.state, pov, &mut commit, 0);
            // Saturated caps: every Research is gated, the turn must still end.
            let aux = compute_goal_aux(
                &game.state,
                pov,
                &goal,
                crate::ai::oracle_macro::TECH_CAP_PER_GAME,
                crate::ai::oracle_macro::TIER3_CAP_PER_GAME,
                None,
            );
            let ranked = rank_plies(&mut game, pov, &goal, &aux, true, 1.0, None);
            assert!(!ranked.is_empty(), "seed {seed}: rank_plies returned empty");
            assert!(
                ranked
                    .iter()
                    .all(|(_, m)| m.move_type() != MoveType::Research),
                "seed {seed}: gated Research survived"
            );
        }
    }

    #[test]
    fn ghost_until_returns_control() {
        for seed in 0..4i64 {
            let game = generated_game(seed);
            let pov = game.state.settings.current_player_turn_id;
            let mut sim = game.clone_for_mcts(pov);
            let goal = compute_macro_goal(&sim.state, pov, 0);
            let mut lane_state = LaneState::default();
            let mut counters = TurnCounters::default();
            assert!(execute_turn(&mut sim, pov, &goal, &mut lane_state, &mut counters, 1.0));
            if sim.state.settings._game_over {
                continue;
            }
            assert!(ghost_until(&mut sim, pov), "seed {seed}: ghost bailed");
            assert!(
                sim.state.settings.current_player_turn_id == pov
                    || sim.state.settings._game_over,
                "seed {seed}: control never returned"
            );
        }
    }
}
