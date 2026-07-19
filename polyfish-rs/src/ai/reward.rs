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

// ---- EXP_ELO_016: development-potential shaping ------------------------
// The raw score prices tech at 100·tier (instant, riskless) vs units at
// 5·cost (clawed back on death), making tech-towering approximately the
// greedy-optimal policy under score-delta labels. `dev_potential` reprices
// the label: de-weight tech, pay income/army/village-proximity densely so
// delay itself costs γ (the "pull future credit forward" mechanism). Applied
// as `score + w·Φ` deltas under the existing discounted-delta convention.

/// Fraction of tech's 100·tier score removed from the shaped label.
pub const SHAPE_TECH_DEWEIGHT: f32 = 0.75;
/// Score-equivalents per star-per-turn of income.
pub const SHAPE_SPT: f32 = 20.0;
/// Extra score-equivalents per star of living units (on top of the game's
/// 5·cost) — kept below tech parity so army pays through captures, not count.
pub const SHAPE_ARMY_PER_COST: f32 = 5.0;
/// Score-equivalents per tile of closed distance toward the nearest
/// FOW-visible uncaptured village.
pub const SHAPE_PROX_PER_TILE: f32 = 12.0;
/// Proximity credit saturates beyond this distance.
pub const SHAPE_PROX_CAP: i32 = 7;

/// Chebyshev distance between two row-major tile indices.
fn cheb(a: i32, b: i32, width: i32) -> i32 {
    let (ra, ca) = (a / width, a % width);
    let (rb, cb) = (b / width, b % width);
    (ra - rb).abs().max((ca - cb).abs())
}

/// `max(0, CAP − min dist(own units → nearest visible uncaptured village))`,
/// 0 with no units or no visible village. A step toward the village banks
/// potential now; hovering banks nothing further (potential-based).
fn village_proximity(state: &GameState, player: i32) -> f32 {
    let Some(tribe) = state.tribes.get(&player) else {
        return 0.0;
    };
    if tribe.units.is_empty() {
        return 0.0;
    }
    let width = state.settings.size as i32;
    if width <= 0 {
        return 0.0;
    }
    let mut best: Option<i32> = None;
    for (&idx, s) in state.structures.iter() {
        let Some(s) = s else { continue };
        if s.structure_type != crate::types::StructureType::Village {
            continue;
        }
        let Some(tile) = state.tiles.get(&idx) else {
            continue;
        };
        if tile.owner != 0 || !tile.explorers.contains(&player) {
            continue;
        }
        for u in &tribe.units {
            let d = cheb(u.coords.idx, idx, width);
            if best.map_or(true, |b| d < b) {
                best = Some(d);
            }
        }
    }
    match best {
        Some(d) => (SHAPE_PROX_CAP - d).max(0) as f32 * SHAPE_PROX_PER_TILE,
        None => 0.0,
    }
}

/// The development potential Φ for `player`, in score-equivalent units.
pub fn dev_potential(state: &GameState, player: i32) -> f32 {
    let Some(tribe) = state.tribes.get(&player) else {
        return 0.0;
    };
    let tech_score: f32 = tribe
        .tech_vanilla
        .iter()
        .map(|t| {
            100.0
                * crate::settings::technology::get_technology_setting(t.tech_type)
                    .tier
                    .unwrap_or(1) as f32
        })
        .sum();
    let army_cost: f32 = tribe
        .units
        .iter()
        .map(|u| crate::settings::units::get_unit_setting(u.unit_type).cost as f32)
        .sum();
    let spt = crate::functions::get_tribe_spt(state, tribe) as f32;

    SHAPE_SPT * spt + SHAPE_ARMY_PER_COST * army_cost + village_proximity(state, player)
        - SHAPE_TECH_DEWEIGHT * tech_score
}

/// `score_snapshot` augmented with `w`·Φ per side. `w = 0` short-circuits to
/// the raw snapshot (bit-exact legacy behavior, no Φ cost on the hot path).
pub fn shaped_snapshot(state: &GameState, player: i32, w: f32) -> (f32, f32) {
    if w == 0.0 {
        let (my, opp) = score_snapshot(state, player);
        return (my as f32, opp as f32);
    }
    let my = state
        .tribes
        .get(&player)
        .map(|t| t.score as f32 + w * dev_potential(state, player))
        .unwrap_or(0.0);
    let opp = state
        .tribes
        .iter()
        .filter(|(id, _)| **id != player)
        .map(|(id, t)| t.score as f32 + w * dev_potential(state, *id))
        .max_by(|a, b| a.total_cmp(b))
        .unwrap_or(0.0);
    (my, opp)
}

#[cfg(test)]
mod shaping_tests {
    use super::*;
    use crate::coords::Coords;
    use crate::settings::technology::get_technology_setting;
    use crate::settings::units::get_unit_setting;
    use crate::states::{StructureState, TechnologyState, TileState, TribeState, UnitState};
    use crate::types::{StructureType, TechnologyType, UnitType};

    fn unit_at(idx: i32, unit_type: UnitType) -> UnitState {
        UnitState {
            unit_type,
            coords: Coords::from_index(idx, 11),
            ..Default::default()
        }
    }

    /// Village structure at `idx`, unowned, explored by player 1.
    fn add_visible_village(state: &mut GameState, idx: i32) {
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

    #[test]
    fn wf_matches_legacy_reward_on_integer_inputs() {
        for (a, b, c, d) in [(1000, 800, 1300, 900), (0, 0, 50, 10), (4000, 4200, 4100, 4900)] {
            let legacy = normalized_reward(a, b, c, d);
            let f = normalized_reward_wf(a as f32, b as f32, c as f32, d as f32, REL_W);
            assert!((legacy - f).abs() < 1e-6, "({a},{b},{c},{d}): {legacy} vs {f}");
        }
    }

    #[test]
    fn shaped_snapshot_at_zero_w_is_the_raw_snapshot() {
        let mut state = GameState::default();
        let mut t1 = TribeState::default();
        t1.score = 123;
        let mut t2 = TribeState::default();
        t2.score = 456;
        state.tribes.insert(1, t1);
        state.tribes.insert(2, t2);
        assert_eq!(shaped_snapshot(&state, 1, 0.0), (123.0, 456.0));
        let (my, opp) = shaped_snapshot(&state, 1, 1.0);
        assert_eq!((my, opp), (123.0, 456.0)); // empty tribes: phi = 0
    }

    #[test]
    fn tech_deweight_subtracts_the_scores_towering_subsidy() {
        let mut state = GameState::default();
        let mut t1 = TribeState::default();
        t1.tech_vanilla.push(TechnologyState {
            tech_type: TechnologyType::Riding,
            discovered: true,
            discovered_turn: 0,
        });
        state.tribes.insert(1, t1);
        let tier = get_technology_setting(TechnologyType::Riding).tier.unwrap_or(1) as f32;
        let expected = -SHAPE_TECH_DEWEIGHT * 100.0 * tier;
        assert!((dev_potential(&state, 1) - expected).abs() < 1e-4);
    }

    #[test]
    fn army_term_pays_star_cost_of_living_units() {
        let mut state = GameState::default();
        let mut t1 = TribeState::default();
        t1.units.push(unit_at(60, UnitType::Warrior));
        state.tribes.insert(1, t1);
        let cost = get_unit_setting(UnitType::Warrior).cost as f32;
        assert!((dev_potential(&state, 1) - SHAPE_ARMY_PER_COST * cost).abs() < 1e-4);
    }

    #[test]
    fn proximity_pays_stepping_toward_a_visible_village_and_nothing_for_hovering() {
        let mk = |unit_idx: i32| {
            let mut state = GameState::default();
            add_visible_village(&mut state, 0);
            let mut t1 = TribeState::default();
            t1.units.push(unit_at(unit_idx, UnitType::Warrior));
            state.tribes.insert(1, t1);
            dev_potential(&state, 1)
        };
        // Row 0: idx = column = Chebyshev distance to the village at idx 0.
        let (d4, d2) = (mk(4), mk(2));
        assert!((d2 - d4 - 2.0 * SHAPE_PROX_PER_TILE).abs() < 1e-4);
        // Lateral move at equal distance banks nothing: (0,3) vs (3,3).
        assert!((mk(3) - mk(36)).abs() < 1e-4);
        // Beyond the cap there is no gradient.
        assert!((mk(9) - mk(10)).abs() < 1e-4);
    }

    #[test]
    fn unexplored_or_owned_villages_pay_no_proximity() {
        let mut state = GameState::default();
        add_visible_village(&mut state, 0);
        state.tiles.get_mut(&0).unwrap().explorers.clear(); // fogged
        let mut t1 = TribeState::default();
        t1.units.push(unit_at(2, UnitType::Warrior));
        state.tribes.insert(1, t1);
        let fogged = dev_potential(&state, 1);
        let cost = get_unit_setting(UnitType::Warrior).cost as f32;
        assert!((fogged - SHAPE_ARMY_PER_COST * cost).abs() < 1e-4);

        state.tiles.get_mut(&0).unwrap().explorers.insert(1);
        state.tiles.get_mut(&0).unwrap().owner = 2; // captured by someone
        assert!((dev_potential(&state, 1) - fogged).abs() < 1e-4);
    }
}
