//! Shared per-move reward definition for TD value labels (self_play) and
//! reward-aware MCTS backup (gumbel_mcts). One source of truth so a move's
//! score gain is normalized identically whether it's being summed into a
//! training label or backed up through the search tree.

use crate::states::GameState;

/// Turn-boundary discount for TD backup/labels: `γ^Δturn` applied only when
/// an edge crosses into a new game turn (within-turn moves are undiscounted).
/// ~10-turn effective horizon; gives a strict banked-now > pending-later
/// ordering independent of noise, unlike the old fixed forward-window MC
/// label it replaces. See notes.md, decision-trace section.
pub const GAMMA_TURN: f32 = 0.9;

/// Weight of the relative (vs opponent) component in every reward and value
/// label — the single zero-sum convention for both the TD body here and the
/// final-outcome tail in self_play. 1.0 = pure relative. The backup negates
/// across every player-turn boundary (mcts_common.rs), which is only valid
/// when v(mine) = -v(theirs); absolute own-progress is NOT antisymmetric (the
/// opponent's progress isn't my loss), so any abs share is corrupted through
/// EndTurn-crossing lines, worse as search deepens. The measured mirror-play
/// failure that once motivated an abs share (decision traces, Jul 7-8 2026:
/// relative swings net to ~0 in mirror play, so the label was empty) is
/// attacked in the DATA instead — greedy-anchor games (--anchor-frac) make
/// passivity actually lose. See notes.md, "Phase-1 training-signal fixes".
pub const REL_W: f32 = 1.0;

/// Reward normalization scales with the game's economy: a saturating swing
/// is ~15% of combined score, floored for the small opening turns.
pub const NORM_FRAC: f32 = 0.15;
pub const NORM_FLOOR: f32 = 600.0;

/// Normalization denominator for a reward measured from a state where `my`/
/// `opp` are the pre-transition scores.
pub fn score_norm(my: i32, opp: i32) -> f32 {
    (NORM_FRAC * (my + opp) as f32).max(NORM_FLOOR)
}

/// Normalized reward for a transition `(my_pre, opp_pre) -> (my_post,
/// opp_post)`, blending absolute (my own score gain) and relative (my gain
/// vs the opponent's) progress. Not clamped — callers accumulate/discount
/// multiple rewards before clamping the final label.
pub fn normalized_reward(my_pre: i32, opp_pre: i32, my_post: i32, opp_post: i32) -> f32 {
    let norm = score_norm(my_pre, opp_pre);
    let delta_abs = (my_post - my_pre) as f32 / norm;
    let delta_rel = ((my_post - opp_post) - (my_pre - opp_pre)) as f32 / norm;
    REL_W * delta_rel + (1.0 - REL_W) * delta_abs
}

/// `(my_score, best_opponent_score)` for `player` in `state`. Shared snapshot
/// helper for reward computation at both a tree edge (gumbel_mcts) and a
/// self-play history step.
pub fn score_snapshot(state: &GameState, player: i32) -> (i32, i32) {
    let my = state.tribes.get(&player).map(|t| t.score).unwrap_or(0);
    let opp = state
        .tribes
        .iter()
        .filter(|(id, _)| **id != player)
        .map(|(_, t)| t.score)
        .max()
        .unwrap_or(0);
    (my, opp)
}

/// `(my_spt, best_opponent_spt)` for `player` in `state`. Feeds the
/// potential-based SPT shaping term in self_play value labels.
pub fn spt_snapshot(state: &GameState, player: i32) -> (i32, i32) {
    let my = state
        .tribes
        .get(&player)
        .map(|t| crate::functions::get_tribe_spt(state, t))
        .unwrap_or(0);
    let opp = state
        .tribes
        .iter()
        .filter(|(id, _)| **id != player)
        .map(|(_, t)| crate::functions::get_tribe_spt(state, t))
        .max()
        .unwrap_or(0);
    (my, opp)
}
