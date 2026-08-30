//! Shared per-move reward definition for TD value labels (self_play) and
//! reward-aware MCTS backup (gumbel_mcts). One source of truth so a move's
//! score gain is normalized identically whether it's being summed into a
//! training label or backed up through the search tree.
//!
//! Aug 2026: split into reward/{dev_potential,economy_completion,
//! goal_shape_consts,goal_potential}.rs so no file exceeds ~1000 lines. All
//! are re-exported below so every `crate::ai::reward::X` call site keeps
//! resolving unchanged.

use crate::states::GameState;

/// Turn-boundary discount for TD backup/labels: `γ^Δturn` applied only when
/// an edge crosses into a new game turn (within-turn moves are undiscounted).
/// ~10-turn effective horizon; gives a strict banked-now > pending-later
/// ordering independent of noise, unlike the old fixed forward-window MC
/// label it replaces. See notes.md, decision-trace section.
pub const GAMMA_TURN: f32 = 0.9;

/// Weight of the relative (vs opponent) component within a reward. Abs-
/// dominant: in mirror self-play both copies gain roughly in lockstep, so a
/// capture's relative swing nets to ~0 and teaches nothing; an absolute
/// anchor on my own score progress rewards it regardless of the opponent.
/// EXP_ELO_005: raising this to 0.7 broke SEARCH before it could test the
/// label hypothesis (instant hoarding/passivity in self-play) — a label-only
/// rel weight must be threaded separately, not changed here.
pub const REL_W: f32 = 0.4;

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
    normalized_reward_w(my_pre, opp_pre, my_post, opp_post, REL_W)
}

/// `normalized_reward` with an explicit relative weight — lets TD labels
/// price windows independently of the in-tree backup (EXP_ELO_006).
pub fn normalized_reward_w(
    my_pre: i32,
    opp_pre: i32,
    my_post: i32,
    opp_post: i32,
    rel_w: f32,
) -> f32 {
    let norm = score_norm(my_pre, opp_pre);
    let delta_abs = (my_post - my_pre) as f32 / norm;
    let delta_rel = ((my_post - opp_post) - (my_pre - opp_pre)) as f32 / norm;
    rel_w * delta_rel + (1.0 - rel_w) * delta_abs
}

/// `normalized_reward_w` over f32 snapshots — the shaped-potential path
/// (EXP_ELO_016) produces fractional augmented scores.
pub fn normalized_reward_wf(
    my_pre: f32,
    opp_pre: f32,
    my_post: f32,
    opp_post: f32,
    rel_w: f32,
) -> f32 {
    let norm = (NORM_FRAC * (my_pre + opp_pre)).max(NORM_FLOOR);
    let delta_abs = (my_post - my_pre) / norm;
    let delta_rel = ((my_post - opp_post) - (my_pre - opp_pre)) / norm;
    rel_w * delta_rel + (1.0 - rel_w) * delta_abs
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
/// Chebyshev distance between two row-major tile indices.
pub(crate) fn cheb(a: i32, b: i32, width: i32) -> i32 {
    let (ra, ca) = (a / width, a % width);
    let (rb, cb) = (b / width, b % width);
    (ra - rb).abs().max((ca - cb).abs())
}

mod dev_potential;
mod economy_completion;
mod goal_potential;
pub mod goal_shape_consts;

pub use dev_potential::*;
pub use economy_completion::*;
pub use goal_potential::*;
pub use goal_shape_consts::*;

/// Test-only helpers shared across reward/*.rs test modules (split Aug 2026).
#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use crate::coords::Coords;
    use crate::states::{StructureState, TileState, UnitState};
    use crate::types::{StructureType, UnitType};

    pub(crate) fn unit_at(idx: i32, unit_type: UnitType) -> UnitState {
        UnitState {
            unit_type,
            coords: Coords::from_index(idx, 11),
            ..Default::default()
        }
    }
    /// Village structure at `idx`, unowned, explored by player 1.
    pub(crate) fn add_visible_village(state: &mut GameState, idx: i32) {
        state.structures.insert(
            idx,
            Some(StructureState {
                structure_type: StructureType::Village,
                level: 0,
                founded: 0,
            }),
        );
        let mut tile = TileState::default();
        tile.explorers.insert(1);
        state.tiles.insert(idx, tile);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wf_matches_legacy_reward_on_integer_inputs() {
        for (a, b, c, d) in [(1000, 800, 1300, 900), (0, 0, 50, 10), (4000, 4200, 4100, 4900)] {
            let legacy = normalized_reward(a, b, c, d);
            let f = normalized_reward_wf(a as f32, b as f32, c as f32, d as f32, REL_W);
            assert!((legacy - f).abs() < 1e-6, "({a},{b},{c},{d}): {legacy} vs {f}");
        }
    }
}
