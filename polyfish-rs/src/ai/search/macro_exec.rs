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

/// EXP_ELO_111 diagnostic (temporary, mirrors EXP_ELO_085's counter pair):
/// how often a Step candidate is checked against the fresh-kill-zone gate,
/// and how often it actually fires. A high fire rate on a game/tribe combo
/// the canonical seed0 fixture didn't cover (e.g. the paired-gauge mirror,
/// with borderline-one-shot Knights) is worth eyeballing before trusting
/// that gauge's aggregate win rate.
pub static STEP_LETHAL_ENTRY_CANDIDATES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static STEP_LETHAL_ENTRY_FIRES: std::sync::atomic::AtomicU64 =
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

/// EXP_ELO_093: EndTurn's re-entry price when every other candidate is
/// Φ-negative (see `rank_plies` below). EXP_ELO_075 tried flat 0.0 and
/// regressed the paired gauge (win rate 0.396->0.146) because 0.0 also
/// outcompetes ordinary shallow diminishing-returns plies. EXP_ELO_077
/// (-400) and EXP_ELO_082 (-500) both looked roughly-neutral-to-negative
/// under noisy pre-EXP_ELO_091 gauges; once that non-determinism was fixed,
/// a clean n=128 4-point sweep (-400 < -500 < -700 < hard-gate, all
/// monotonic in win rate) showed the mechanism is net negative at every
/// tested strength, hard-gate (never revive) winning outright. Verdi's
/// call: land at -700 anyway — close enough to hard-gate's measured
/// win rate (0.3203 vs 0.3438, n=128) to capture nearly all of that
/// benefit while keeping the mechanism alive for the rare case it's
/// actually the right call, as a base to build a smarter/contextual
/// EndTurn ranking on later rather than deleting the capability outright.
/// NOTE this is well below the originally-flagged idx179 ply's best real
/// option (-441.240) and even EXP_ELO_082's -500 — this floor does not
/// fire on either of those cases; only much deeper Φ-collapse does.
const ENDTURN_REVIVE_PRICE_DEFAULT: f32 = -700.0;

/// EXP_ELO_111: tiebreaker-scale penalty (NOT value-scale like the reverted
/// EXP_ELO_105/109 attempts) charged when a Step walks a unit from a safe
/// PRE-move position into a one-shot kill zone no existing Φ term prices
/// (`combat::lethal_threat_weight`'s doc comment). Scoped to Step only --
/// EXP_ELO_109 found flat penalties on Attacks suppress genuine kills
/// because flagged-ply margins and ordinary attack values share the same
/// numeric range; a full-game scan for this fix found Step gate-fires were
/// 3/3 true positives (all died) vs Attacks' 1/3, so Attacks are excluded
/// by construction, not by tuning. See EXP_ELO_111 ledger entry.
const STEP_LETHAL_ENTRY_PENALTY: f32 = 3.0;

/// EXP_ELO_111: the penalty a Step candidate pays for walking `unit` (its
/// POST-move state, already relocated by `simulate_move`) into a fresh
/// one-shot kill zone. Zero if `unit` was already lethally exposed
/// PRE-move (`pre_lethal`) -- an already-exposed unit gets no escape
/// gradient from this, only from the underlying Φ terms, matching
/// `lethal_threat_weight`'s own live-vs-frozen semantics.
fn step_lethal_entry_penalty(
    pre_lethal: bool,
    post_state: &GameState,
    unit: &crate::states::UnitState,
    threats: &[(crate::states::UnitState, f32)],
) -> f32 {
    if pre_lethal {
        return 0.0;
    }
    STEP_LETHAL_ENTRY_PENALTY * crate::ai::combat::lethal_threat_weight(post_state, unit, threats)
}

/// EXP_ELO_092 diagnostic: `POLYFISH_ENDTURN_REVIVE_PRICE=<f32>` overrides
/// the compile-time default, so different floors (e.g. -400 vs -500) can
/// be A/B'd from one binary without a rebuild -- mirrors
/// `POLYFISH_ENDTURN_HARD_GATE`'s pattern for the same purpose.
fn endturn_revive_price() -> f32 {
    static PRICE: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *PRICE.get_or_init(|| {
        std::env::var("POLYFISH_ENDTURN_REVIVE_PRICE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(ENDTURN_REVIVE_PRICE_DEFAULT)
    })
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
    eco_plan: Option<&crate::ai::eco_plan_commit::EcoPlanCommit>,
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
    // MapBelief is a pure function of the explored set (see its module doc)
    // — safe to compute once here and reuse across every candidate's
    // phi_post below, same reasoning as `threats` above and the same
    // EXP_ELO_061-class cost this avoids paying per-candidate.
    let belief = if lambda != 0.0 {
        Some(crate::ai::belief::map::MapBelief::observe(&game.state, player))
    } else {
        None
    };
    // EXP_ELO_110: each own unit's health at ply start, so the Defend
    // waterfall (`combat::defend_plan_impl`) can floor a covering unit's
    // contribution against a self-wound shrinking it mid-comparison,
    // without needing a live re-derivation. Same once-per-ply reuse
    // pattern as `threats`/`belief` above; passed to BOTH phi_pre and
    // phi_post below (a no-op on phi_pre, since live == pre there by
    // construction — the floor only ever engages post-move).
    let pre_health: Option<rustc_hash::FxHashMap<u32, f32>> = if lambda != 0.0 {
        game.state
            .tribes
            .get(&player)
            .map(|t| t.units.iter().map(|u| (u.id, u.health)).collect())
    } else {
        None
    };
    // EXP_ELO_111: each own unit's PRE-move lethal-exposure status, reusing
    // the `threats` snapshot above. The gate a Step candidate charges
    // STEP_LETHAL_ENTRY_PENALTY against is POST-move lethal AND PRE-move
    // safe -- an already-exposed unit (e.g. mid-escape) is exempt, matching
    // combat::lethal_threat_weight's own live-vs-frozen semantics.
    let pre_lethal: Option<rustc_hash::FxHashMap<u32, bool>> = if lambda != 0.0 {
        threats.as_deref().map(|th| {
            game.state
                .tribes
                .get(&player)
                .map(|t| {
                    t.units
                        .iter()
                        .map(|u| {
                            (
                                u.id,
                                crate::ai::combat::lethal_threat_weight(&game.state, u, th) > 0.0,
                            )
                        })
                        .collect()
                })
                .unwrap_or_default()
        })
    } else {
        None
    };
    let phi_pre = if lambda != 0.0 {
        reward::goal_potential_with_belief(
            &game.state,
            player,
            goal,
            Some(aux),
            threats.as_deref(),
            unit_goals,
            belief.as_ref(),
            pre_health.as_ref(),
        )
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
            let mut s = scoring::score_move_with_unit_goals(game, m.as_ref(), unit_goals, eco_plan);
            if lambda != 0.0 && m.move_type() != MoveType::EndTurn {
                // EXP_ELO_111: resolve the acting unit's id from its
                // PRE-move coords before simulate_move relocates it.
                // Step-only by construction -- an Attack candidate never
                // sets this, so it structurally can't engage the penalty
                // below.
                let step_unit_id = if m.move_type() == MoveType::Step {
                    m.source_idx().ok().and_then(|src| {
                        game.state.tribes.get(&player).and_then(|t| {
                            t.units
                                .iter()
                                .find(|u| u.coords.idx == src as i32)
                                .map(|u| u.id)
                        })
                    })
                } else {
                    None
                };
                if let Some(undo) = game.simulate_move(m.as_ref()) {
                    let phi_post = reward::goal_potential_with_belief(
                        &game.state,
                        player,
                        goal,
                        Some(aux),
                        threats.as_deref(),
                        unit_goals,
                        belief.as_ref(),
                        pre_health.as_ref(),
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
                    if let Some(uid) = step_unit_id {
                        if let (Some(th), Some(pl)) = (threats.as_deref(), pre_lethal.as_ref()) {
                            let was_pre_lethal = pl.get(&uid).copied().unwrap_or(false);
                            if let Some(u) = game
                                .state
                                .tribes
                                .get(&player)
                                .and_then(|t| t.units.iter().find(|u| u.id == uid))
                            {
                                STEP_LETHAL_ENTRY_CANDIDATES
                                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                let pen = step_lethal_entry_penalty(was_pre_lethal, &game.state, u, th);
                                if pen > 0.0 {
                                    STEP_LETHAL_ENTRY_FIRES
                                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                }
                                s -= pen;
                            }
                        }
                    }
                    undo(&mut game.state);
                    s += lambda * (phi_post - phi_pre);
                }
            }
            (s, m)
        })
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    let scored = revive_endturn_for_lone_doomed_unit(scored, has_other, lambda, &game.state);
    revive_endturn_if_worse_than_floor(scored, has_other, lambda)
}

/// EXP_ELO_077: re-admit EndTurn, priced at `ENDTURN_REVIVE_PRICE_DEFAULT` (not
/// 0.0 — see that constant's doc comment), only when the best surviving
/// candidate is already worse than that floor. Doing nothing beats doing
/// deep active harm; it does not beat an ordinary mediocre move. Pulled out
/// of `rank_plies` as a pure function so the threshold can be unit-tested
/// against synthetic scores without a real game board.
/// EXP_ELO_085 diagnostic (temporary): fire-rate of EndTurn actually
/// WINNING when real alternatives existed, vs the total number of plies
/// where that was even possible (has_other && lambda != 0.0). The
/// "hard-gated" baseline (no revive code at all) is 0/ELIGIBLE by
/// construction -- these two counters alone answer "how often, compared
/// to never."
pub static ENDTURN_ELIGIBLE_PLIES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static ENDTURN_CHOSEN_WITH_ALTERNATIVES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// EXP_ELO_087 diagnostic (temporary): `POLYFISH_ENDTURN_HARD_GATE=1`
/// disables the revive entirely -- the exact pre-EXP_ELO_075 behavior
/// (EndTurn always stripped whenever another move survives gating) --
/// without needing a separate build/checkout, so a hard-gated arm can be
/// A/B'd against the current floor on the identical binary.
fn endturn_hard_gate() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("POLYFISH_ENDTURN_HARD_GATE").as_deref() == Ok("1"))
}

fn revive_endturn_if_worse_than_floor(
    mut scored: Vec<(f32, Box<dyn Move>)>,
    has_other: bool,
    lambda: f32,
) -> Vec<(f32, Box<dyn Move>)> {
    if has_other && lambda != 0.0 && !endturn_hard_gate() {
        ENDTURN_ELIGIBLE_PLIES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let price = endturn_revive_price();
        if scored.first().is_some_and(|(s, _)| *s < price) {
            scored.push((price, Box::new(EndTurnMove) as Box<dyn Move>));
            scored.sort_by(|a, b| b.0.total_cmp(&a.0));
            ENDTURN_CHOSEN_WITH_ALTERNATIVES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
    scored
}

/// EXP_ELO_102: `revive_endturn_if_worse_than_floor`'s flat -700 price is
/// deliberately conservative (EXP_ELO_075/077/082's sweep found any flatter
/// revival net-negative, since it also outcompetes ordinary mediocre plies
/// with real opportunity cost) -- but that conservatism has a real cost of
/// its own when the *entire* surviving candidate set is one already-acted
/// unit's self-lethal attacks: every other unit has nothing left to do this
/// ply, so EndTurn has ZERO opportunity cost and should always win, no
/// matter how the score happens to sit relative to -700 (a real seed0 ply
/// missed the floor by exactly 1.0 point: single candidate at -699.0).
/// Distinct mechanism from the flat floor, not a retuning of it: only fires
/// when there is provably nothing else to do, not merely when a score is
/// low.
fn revive_endturn_for_lone_doomed_unit(
    scored: Vec<(f32, Box<dyn Move>)>,
    has_other: bool,
    lambda: f32,
    state: &GameState,
) -> Vec<(f32, Box<dyn Move>)> {
    if !has_other || lambda == 0.0 || endturn_hard_gate() {
        return scored;
    }
    let all_lethal_same_unit = (|| {
        let (_, first) = scored.first()?;
        let src = first.source_idx().ok()?;
        scored
            .iter()
            .all(|(_, m)| {
                m.move_type() == MoveType::Attack
                    && m.source_idx().ok() == Some(src)
                    && m.target_idx().ok().is_some_and(|t| {
                        crate::functions::calculate_combat_preview(state, src as i32, t as i32)
                            .is_some_and(|p| p.attacker_dies)
                    })
            })
            .then_some(())
    })()
    .is_some();
    if !all_lethal_same_unit {
        return scored;
    }
    ENDTURN_CHOSEN_WITH_ALTERNATIVES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    vec![(0.0, Box::new(EndTurnMove) as Box<dyn Move>)]
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
        let mut ranked = rank_plies(game, player, goal, &aux, gate, lambda, None, None);
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
            let no_phi = rank_plies(game, player, goal, &aux, gate, 0.0, None, None);
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
            let no_goal = rank_plies(game, player, &bare, &bare_aux, false, 0.0, None, None);
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
    use crate::moves::StepMove;

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
            let ranked = rank_plies(&mut game, pov, &goal, &aux, true, 1.0, None, None);
            assert!(!ranked.is_empty(), "seed {seed}: rank_plies returned empty");
            assert!(
                ranked
                    .iter()
                    .all(|(_, m)| m.move_type() != MoveType::Research),
                "seed {seed}: gated Research survived"
            );
        }
    }

    /// EXP_ELO_093: same fixture as EXP_ELO_077's own test (lone
    /// Defend-ordered garrison, 2-tiles-out threat, every Step vacates the
    /// held tile) — confirmed by direct probe to price every Step around
    /// -426..-429. That clears the old -400 floor but NOT the current
    /// -700 one: this documents the calibration tradeoff Verdi explicitly
    /// chose (landing near hard-gate's measured win rate while keeping the
    /// mechanism alive for deeper cases, at the cost of this specific
    /// motivating case no longer being fixed) rather than silently losing
    /// the coverage fact. EndTurn stays gated out here; the best real Step
    /// wins instead, exactly as it did before EXP_ELO_077 shipped.
    #[test]
    fn moderate_forced_harm_no_longer_revives_end_turn_at_the_700_floor() {
        use crate::ai::combat::tests::{board, unit_at};
        use crate::ai::oracle_macro::{MacroGoal, OrderKind, Stance};
        use crate::types::UnitType;

        let mut state = board(60);
        state.tribes.get_mut(&1).unwrap().units.push(unit_at(60, UnitType::Rider, 1));
        state.tribes.get_mut(&2).unwrap().units.push(unit_at(58, UnitType::Swordsman, 2));
        let mut game = Game::new();
        game.state = state;
        game.post_load();

        let goal = MacroGoal {
            orders: vec![(OrderKind::Defend, 60)],
            stance: Stance::Arm,
            save_target: None, prepare: None,
        };
        let aux = compute_goal_aux(&game.state, 1, &goal, 0, 0, None);
        let ranked = rank_plies(&mut game, 1, &goal, &aux, true, 1.0, None, None);

        let steps: Vec<_> = ranked.iter().filter(|(_, m)| m.move_type() == MoveType::Step).collect();
        assert!(!steps.is_empty(), "fixture should offer the Rider Step options");
        assert!(
            steps.iter().all(|(s, _)| *s > ENDTURN_REVIVE_PRICE_DEFAULT),
            "test fixture assumption broken: expected every Step to sit ABOVE the -700 floor \
             (i.e. NOT clear it), so this case stays uncovered by design: {:?}",
            steps.iter().map(|(s, m)| (*s, m.serialize())).collect::<Vec<_>>()
        );

        let (_top_score, top_move) = &ranked[0];
        assert_ne!(
            top_move.move_type(),
            MoveType::EndTurn,
            "at -700 this moderate-depth forced-harm case is uncovered by design; ranked: {:?}",
            ranked.iter().map(|(s, m)| (*s, m.move_type())).collect::<Vec<_>>()
        );
    }

    /// EXP_ELO_093: the regression this threshold exists to prevent. A
    /// mildly-negative best candidate (-50, in EXP_ELO_075's own measured
    /// "75% of fires" band) must NOT revive EndTurn — only genuinely deep
    /// harm (worse than -700) should. Tests the pure threshold function
    /// directly since `rank_plies` itself needs a real game board.
    #[test]
    fn endturn_does_not_revive_for_shallow_negative_plies() {
        let shallow = vec![(-50.0f32, Box::new(StepMove::new(1, 2)) as Box<dyn Move>)];
        let out = revive_endturn_if_worse_than_floor(shallow, true, 1.0);
        assert!(
            out.iter().all(|(_, m)| m.move_type() != MoveType::EndTurn),
            "a -50.0 top score (EXP_ELO_075's dominant regression band) must not revive EndTurn"
        );

        let deep = vec![(-750.0f32, Box::new(StepMove::new(1, 2)) as Box<dyn Move>)];
        let out = revive_endturn_if_worse_than_floor(deep, true, 1.0);
        assert_eq!(
            out[0].1.move_type(),
            MoveType::EndTurn,
            "a -750.0 top score (worse than the revive floor) must revive and win with EndTurn"
        );

        // Boundary: exactly at the floor does not revive (strict `<`).
        let boundary = vec![(ENDTURN_REVIVE_PRICE_DEFAULT, Box::new(StepMove::new(1, 2)) as Box<dyn Move>)];
        let out = revive_endturn_if_worse_than_floor(boundary, true, 1.0);
        assert!(out.iter().all(|(_, m)| m.move_type() != MoveType::EndTurn));

        // lambda == 0.0 (non-goal-shaped callers) never revives.
        let deep_no_lambda = vec![(-750.0f32, Box::new(StepMove::new(1, 2)) as Box<dyn Move>)];
        let out = revive_endturn_if_worse_than_floor(deep_no_lambda, true, 0.0);
        assert!(out.iter().all(|(_, m)| m.move_type() != MoveType::EndTurn));

        // has_other == false (fully gated turn) never revives — that path
        // already degrades to a lone EndTurn earlier in rank_plies.
        let deep_no_other = vec![(-750.0f32, Box::new(StepMove::new(1, 2)) as Box<dyn Move>)];
        let out = revive_endturn_if_worse_than_floor(deep_no_other, false, 1.0);
        assert!(out.iter().all(|(_, m)| m.move_type() != MoveType::EndTurn));
    }

    /// EXP_ELO_102: the case the flat -700 floor misses by design — a real
    /// seed0 ply had exactly one candidate (a self-lethal Attack) scoring
    /// -699.0, one point above the floor. Nothing else can act this ply
    /// (the whole candidate set is one unit's own moves), so EndTurn has
    /// zero opportunity cost and must win regardless of score.
    #[test]
    fn endturn_revives_for_a_lone_units_self_lethal_attacks() {
        use crate::ai::combat::tests::{board, unit_at};
        use crate::moves::AttackMove;
        use crate::types::UnitType;

        let mut state = board(60);
        let mut attacker = unit_at(60, UnitType::Warrior, 1);
        attacker.health = 1.0;
        state.tribes.get_mut(&1).unwrap().units.push(attacker);
        state.tribes.get_mut(&2).unwrap().units.push(unit_at(61, UnitType::Warrior, 2));

        let preview = crate::functions::calculate_combat_preview(&state, 60, 61)
            .expect("fixture attacker/defender must resolve a preview");
        assert!(
            preview.attacker_dies,
            "test fixture assumption broken: attack must be self-lethal: {preview:?}"
        );

        let scored = vec![(-699.0f32, Box::new(AttackMove::new(60, 61)) as Box<dyn Move>)];
        let out = revive_endturn_for_lone_doomed_unit(scored, true, 1.0, &state);
        assert_eq!(
            out.len(),
            1,
            "the sole self-lethal candidate should be replaced, not appended to"
        );
        assert_eq!(
            out[0].1.move_type(),
            MoveType::EndTurn,
            "sole candidate is a self-lethal attack with nothing else to do this ply"
        );
    }

    /// A unit with a SURVIVABLE attack available must not be forced into
    /// EndTurn — this mechanism only fires when every remaining option is
    /// provably fatal, never as a general "score is bad" shortcut (that's
    /// exactly the EXP_ELO_075 regression shape this stays clear of).
    #[test]
    fn endturn_does_not_revive_when_the_attack_is_survivable() {
        use crate::ai::combat::tests::{board, unit_at};
        use crate::moves::AttackMove;
        use crate::types::UnitType;

        let mut state = board(60);
        state.tribes.get_mut(&1).unwrap().units.push(unit_at(60, UnitType::Warrior, 1));
        state.tribes.get_mut(&2).unwrap().units.push(unit_at(61, UnitType::Warrior, 2));

        let preview = crate::functions::calculate_combat_preview(&state, 60, 61)
            .expect("fixture attacker/defender must resolve a preview");
        assert!(
            !preview.attacker_dies,
            "test fixture assumption broken: full-health attacker must survive: {preview:?}"
        );

        let scored = vec![(-50.0f32, Box::new(AttackMove::new(60, 61)) as Box<dyn Move>)];
        let out = revive_endturn_for_lone_doomed_unit(scored, true, 1.0, &state);
        assert!(out.iter().all(|(_, m)| m.move_type() != MoveType::EndTurn));
    }

    /// Two different units' self-lethal attacks must NOT trigger this —
    /// scope stays narrow to what was actually verified (a single doomed
    /// unit with nothing else live this ply), not generalized to "every
    /// candidate happens to be lethal."
    #[test]
    fn endturn_does_not_revive_across_multiple_units_even_if_all_lethal() {
        use crate::ai::combat::tests::{board, unit_at};
        use crate::moves::AttackMove;
        use crate::types::UnitType;

        let mut state = board(60);
        let mut a1 = unit_at(60, UnitType::Warrior, 1);
        a1.health = 1.0;
        let mut a2 = unit_at(62, UnitType::Warrior, 1);
        a2.health = 1.0;
        state.tribes.get_mut(&1).unwrap().units.push(a1);
        state.tribes.get_mut(&1).unwrap().units.push(a2);
        state.tribes.get_mut(&2).unwrap().units.push(unit_at(61, UnitType::Warrior, 2));
        state.tribes.get_mut(&2).unwrap().units.push(unit_at(63, UnitType::Warrior, 2));

        let scored = vec![
            (-699.0f32, Box::new(AttackMove::new(60, 61)) as Box<dyn Move>),
            (-699.0f32, Box::new(AttackMove::new(62, 63)) as Box<dyn Move>),
        ];
        let out = revive_endturn_for_lone_doomed_unit(scored, true, 1.0, &state);
        assert!(
            out.iter().all(|(_, m)| m.move_type() != MoveType::EndTurn),
            "two different units still on the board must not collapse to an unconditional EndTurn"
        );
    }

    /// A non-Attack alternative (even from the same unit) is a real,
    /// survivable option — must not be discarded in favor of EndTurn.
    #[test]
    fn endturn_does_not_revive_when_a_non_attack_option_exists() {
        use crate::ai::combat::tests::{board, unit_at};
        use crate::moves::AttackMove;
        use crate::types::UnitType;

        let mut state = board(60);
        let mut attacker = unit_at(60, UnitType::Warrior, 1);
        attacker.health = 1.0;
        state.tribes.get_mut(&1).unwrap().units.push(attacker);
        state.tribes.get_mut(&2).unwrap().units.push(unit_at(61, UnitType::Warrior, 2));

        let scored = vec![
            (-699.0f32, Box::new(AttackMove::new(60, 61)) as Box<dyn Move>),
            (-620.0f32, Box::new(StepMove::new(60, 71)) as Box<dyn Move>),
        ];
        let out = revive_endturn_for_lone_doomed_unit(scored, true, 1.0, &state);
        assert!(out.iter().all(|(_, m)| m.move_type() != MoveType::EndTurn));
    }

    /// Guards mirror the flat-floor function's: never fires with no other
    /// candidates, no goal-shaping, or the hard-gate diagnostic set.
    #[test]
    fn endturn_lone_doomed_unit_respects_the_same_guards() {
        use crate::ai::combat::tests::{board, unit_at};
        use crate::moves::AttackMove;
        use crate::types::UnitType;

        let mut state = board(60);
        let mut attacker = unit_at(60, UnitType::Warrior, 1);
        attacker.health = 1.0;
        state.tribes.get_mut(&1).unwrap().units.push(attacker);
        state.tribes.get_mut(&2).unwrap().units.push(unit_at(61, UnitType::Warrior, 2));

        let scored = || vec![(-699.0f32, Box::new(AttackMove::new(60, 61)) as Box<dyn Move>)];
        let out = revive_endturn_for_lone_doomed_unit(scored(), false, 1.0, &state);
        assert!(out.iter().all(|(_, m)| m.move_type() != MoveType::EndTurn));
        let out = revive_endturn_for_lone_doomed_unit(scored(), true, 0.0, &state);
        assert!(out.iter().all(|(_, m)| m.move_type() != MoveType::EndTurn));
    }

    /// EXP_ELO_111: same fixture as combat.rs's
    /// `lethal_threat_weight_detects_a_one_shot_but_not_a_safe_hit` (a
    /// Warrior adjacent to a Swordsman is only lethally exposed once
    /// wounded to 1hp) -- confirms the wrapper converts the primitive's
    /// weight into STEP_LETHAL_ENTRY_PENALTY-scaled points.
    #[test]
    fn step_lethal_entry_penalty_fires_on_a_fresh_one_shot() {
        use crate::ai::combat::tests::{board, unit_at};
        use crate::types::UnitType;

        let mut state = board(60);
        let mut wounded = unit_at(70, UnitType::Warrior, 1);
        wounded.health = 1.0;
        state.tribes.get_mut(&1).unwrap().units.push(wounded.clone());
        state.tribes.get_mut(&2).unwrap().units.push(unit_at(69, UnitType::Swordsman, 2));
        let threats = crate::ai::combat::threat_units(&state, 1);

        let penalty = step_lethal_entry_penalty(false, &state, &wounded, &threats);
        assert!(
            penalty >= STEP_LETHAL_ENTRY_PENALTY * 0.99,
            "a 1hp Warrior adjacent to a Swordsman must pay close to the full penalty, got {penalty}"
        );
    }

    /// The safe half of the same fixture: full health, same adjacency --
    /// no penalty at all.
    #[test]
    fn step_lethal_entry_penalty_zero_when_post_move_is_safe() {
        use crate::ai::combat::tests::{board, unit_at};
        use crate::types::UnitType;

        let mut state = board(60);
        let healthy = unit_at(70, UnitType::Warrior, 1);
        state.tribes.get_mut(&1).unwrap().units.push(healthy.clone());
        state.tribes.get_mut(&2).unwrap().units.push(unit_at(69, UnitType::Swordsman, 2));
        let threats = crate::ai::combat::threat_units(&state, 1);

        assert_eq!(step_lethal_entry_penalty(false, &state, &healthy, &threats), 0.0);
    }

    /// The pre-exposed exemption: even a fresh one-shot must pay nothing
    /// if the unit was ALREADY lethally exposed before this Step -- no
    /// escape gradient for units already in danger, matching
    /// `lethal_threat_weight`'s live-vs-frozen semantics and pass-9's
    /// explicit non-goal for this fix.
    #[test]
    fn step_lethal_entry_penalty_exempts_an_already_exposed_unit() {
        use crate::ai::combat::tests::{board, unit_at};
        use crate::types::UnitType;

        let mut state = board(60);
        let mut wounded = unit_at(70, UnitType::Warrior, 1);
        wounded.health = 1.0;
        state.tribes.get_mut(&1).unwrap().units.push(wounded.clone());
        state.tribes.get_mut(&2).unwrap().units.push(unit_at(69, UnitType::Swordsman, 2));
        let threats = crate::ai::combat::threat_units(&state, 1);

        assert_eq!(
            step_lethal_entry_penalty(true, &state, &wounded, &threats),
            0.0,
            "pre_lethal=true must exempt the candidate regardless of post-move exposure"
        );
    }

    /// End-to-end wiring check via rank_plies itself -- a Step that walks
    /// a wounded unit into a fresh one-shot kill zone must rank BELOW an
    /// otherwise-symmetric Step to a tile that stays safe (both 70 and 50
    /// are Chebyshev distance 1 from the source AND from the far-off P1
    /// city, so nothing else on this featureless board should distinguish
    /// them). Insurance against the wiring above regressing silently.
    #[test]
    fn rank_plies_prices_a_step_into_a_fresh_kill_zone_below_a_safe_alternative() {
        use crate::ai::combat::tests::{board, unit_at};
        use crate::ai::oracle_macro::{MacroGoal, Stance};
        use crate::types::UnitType;

        let mut state = board(0);
        let mut wounded = unit_at(50, UnitType::Warrior, 1);
        wounded.health = 1.0;
        state.tribes.get_mut(&1).unwrap().units.push(wounded);
        // Swordsman has Dash (move 1 + range 1 = reach 2), so its
        // Chebyshev-2 danger zone extends past plain adjacency -- 60 sits
        // in that zone, 61 does not, even though both are one Step from
        // the source at 50.
        state.tribes.get_mut(&2).unwrap().units.push(unit_at(69, UnitType::Swordsman, 2));
        let mut game = Game::new();
        game.state = state;
        game.post_load();

        let goal = MacroGoal { orders: vec![], stance: Stance::Save, save_target: None, prepare: None };
        let aux = compute_goal_aux(&game.state, 1, &goal, 0, 0, None);
        let ranked = rank_plies(&mut game, 1, &goal, &aux, true, 1.0, None, None);

        let lethal = ranked
            .iter()
            .find(|(_, m)| m.move_type() == MoveType::Step && m.target_idx().ok() == Some(60));
        let safe = ranked
            .iter()
            .find(|(_, m)| m.move_type() == MoveType::Step && m.target_idx().ok() == Some(61));
        let (lethal_score, _) = lethal.expect("Step 50->60 must be a legal candidate");
        let (safe_score, _) = safe.expect("Step 50->61 must be a legal candidate");
        assert!(
            safe_score > lethal_score,
            "safe={safe_score} lethal={lethal_score}: the fresh-kill-zone Step must not \
             outrank the safe alternative"
        );
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
