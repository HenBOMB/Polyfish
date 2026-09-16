//! `evaluate_state_v2`: a second hand-heuristic evaluator, built to test one
//! question with zero training -- does a richer combination of the same
//! GameState signals `evaluate_state` already uses, plus the one
//! forward-looking signal it lacks (`eco_plan`'s per-city star-efficient
//! ceiling), score better as a macro-mcts search leaf than `evaluate_state`
//! itself? See `hypothesis_driven_improvements.md` for the registered
//! experiment and `current_understanding.md`'s value-head section for the
//! background this is testing against.
//!
//! Deliberately a NEW function, not an edit to `gamestate::evaluate_gamestate`
//! -- that function has half a dozen other call sites (vanilla `mcts.rs`,
//! `heuristic_mcts.rs`, `MacroLeaf::Heuristic`) and `calibrated_heur`'s fitted
//! constants (`self_play/labels.rs`) depend on its exact current formula.

use super::player;
use crate::rules::eco_plan::{self, Goal};
use crate::states::{GameState, PlayerId};

/// `eco_plan`'s Balanced-goal ceiling from `state`, folded into one [0,1]
/// scalar the same way `self_play/game.rs`'s aux-label capture already
/// normalizes it (spt/20, pop/100, giants/10, monuments_used/5), equal-
/// weighted mean of the four. `0.0` (not a missing value) when the player
/// holds no cities or no scenario has a feasible frontier -- "no achievable
/// ceiling from here" is itself a real, low reading.
pub(crate) fn eco_potential(state: &GameState, player_id: PlayerId) -> f32 {
    let cities: Vec<i32> = match state.tribes.get(&player_id) {
        Some(t) if !t.cities.is_empty() => t.cities.iter().map(|c| c.idx).collect(),
        _ => return 0.0,
    };
    let Some(plan) = eco_plan::ceiling_for_goal(state, player_id, &cities, Goal::Balanced) else {
        return 0.0;
    };
    let spt = (plan.spt as f32 / 20.0).clamp(0.0, 1.0);
    let pop = (plan.pop as f32 / 100.0).clamp(0.0, 1.0);
    let giants = (plan.giants as f32 / 10.0).clamp(0.0, 1.0);
    let monuments = (plan.monuments_used() as f32 / 5.0).clamp(0.0, 1.0);
    ((spt + pop + giants + monuments) / 4.0).clamp(0.0, 1.0)
}

/// Weight on `eco_potential` inside the per-player composite, vs. the
/// existing `player::evaluate_player` blend. Placeholder pending the
/// calibration fit against real self-play outcomes (see module doc) -- 0.25
/// puts it on par with a single one of `evaluate_player`'s own four
/// sub-scores at their largest observed weight (0.4).
const ECO_POTENTIAL_W: f32 = 0.25;

/// Per-player absolute score (0.0-1.0), mirroring `player::evaluate_player`'s
/// exact contract (same clamp, same dead-tribe -1.0 sentinel) but folding in
/// `eco_potential` as a fifth term.
fn player_score_v2(state: &GameState, player_id: PlayerId) -> f32 {
    let base = player::evaluate_player(state, player_id);
    if base < 0.0 {
        return base; // dead-tribe sentinel, unchanged
    }
    ((1.0 - ECO_POTENTIAL_W) * base + ECO_POTENTIAL_W * eco_potential(state, player_id))
        .clamp(0.0, 1.0)
}

/// Raw relative margin, mirroring `gamestate::evaluate_gamestate`'s exact
/// structure -- including its dead-POV / all-opponents-dead antisymmetric
/// handling -- but reading `player_score_v2` instead of
/// `player::evaluate_player`.
fn raw_margin_v2(state: &GameState, player_id: PlayerId) -> f32 {
    let my_score = player_score_v2(state, player_id);
    if my_score < 0.0 {
        return -1.0;
    }
    let mut any_opponent_alive = false;
    let mut max_opponent_score = 0.0;
    for &opponent_id in state.tribes.keys() {
        if opponent_id != player_id {
            any_opponent_alive = true;
            let opp_score = player_score_v2(state, opponent_id);
            if opp_score > max_opponent_score {
                max_opponent_score = opp_score;
            }
        }
    }
    if !any_opponent_alive {
        return 1.0;
    }
    (my_score - max_opponent_score).clamp(-1.0, 1.0)
}

/// Turn-band edges for the calibration slope below: bands are [0,10),
/// [10,20), [20,inf). Turns beyond the table pool onto its last entry,
/// mirroring `self_play/labels.rs::ABS_GROWTH_PER_TURN`'s own convention.
const TURN_BAND_EDGES: [i32; 2] = [10, 20];

/// Zero-intercept logistic slope per turn band for `P(win) =
/// sigmoid(b(turn) * raw_margin)`, applied below as `2*sigmoid(bx)-1 =
/// tanh(bx/2)`. Zero intercept is load-bearing, not a simplification:
/// `tanh` is odd in `x`, which is what keeps this transform antisymmetric
/// given an antisymmetric `raw_margin_v2` input -- a fitted nonzero
/// intercept would break that.
///
/// PLACEHOLDER: all three bands currently reuse `evaluate_state`'s own OLS
/// slope (2.3479, `self_play/labels.rs::HEUR_TO_OUTCOME_SLOPE`) as a
/// stand-in pending a real turn-banded logistic fit of `raw_margin_v2`
/// against real self-play win/loss outcomes. Antisymmetry does not depend
/// on these values being correct -- it holds for any real `b`.
const CALIBRATION_SLOPE: [f32; 3] = [2.3479, 2.3479, 2.3479];

fn calibration_slope(turn: i32) -> f32 {
    let band = TURN_BAND_EDGES
        .iter()
        .position(|&edge| turn < edge)
        .unwrap_or(TURN_BAND_EDGES.len());
    CALIBRATION_SLOPE[band]
}

/// `evaluate_state`, but with two changes under test (Phase A of the
/// value-target rework): a richer per-player composite (adds `eco_plan`'s
/// forward-looking ceiling to `evaluate_state`'s existing four sub-scores)
/// and a genuine calibrated win-probability transform (turn-banded,
/// zero-intercept logistic) instead of a raw score-differential clamp.
/// Antisymmetric by construction -- see `evaluate_state_v2_is_antisymmetric`
/// in `search::macro_mcts`'s test module.
pub fn evaluate_state_v2(state: &GameState, player_id: PlayerId) -> f32 {
    let raw = raw_margin_v2(state, player_id);
    let b = calibration_slope(state.settings.turn);
    (b * raw / 2.0).tanh()
}
