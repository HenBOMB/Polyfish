//! EXP_ELO_016 development-potential shaping (Aug 2026 taxonomy split out of
//! reward.rs to keep every file under ~1000 lines): the raw score prices tech
//! at 100·tier (instant, riskless) vs units at 5·cost (clawed back on death),
//! making tech-towering approximately the greedy-optimal policy under
//! score-delta labels. `dev_potential` reprices the label: de-weight tech,
//! pay income/army/village-proximity densely so delay itself costs γ (the
//! "pull future credit forward" mechanism). Applied as `score + w·Φ` deltas
//! under the existing discounted-delta convention. Re-exported through
//! `reward` so existing `crate::ai::reward::X` call sites keep resolving.

use super::{cheb, score_snapshot};
use crate::states::GameState;

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

/// EXP_ELO_018: score-equivalents per tile of closed distance toward the
/// nearest visible uncaptured village, for the *isolated* pursuit-progress
/// reward (independent weight, see `pursuit_potential`). Sized from the
/// measured chosen−toward Q gap on wrong-move pursuer-turns (median 0.19 /
/// p75 0.42 normalized, ≈150–350 score-equiv through `score_norm≈700`) —
/// ~15× EXP_ELO_016's `SHAPE_PROX_PER_TILE`, which was too weak to flip the
/// decision (FM-3 pursuit metric — see the notes.md pursuit diagnosis;
/// current status in current_understanding.md).
pub const SHAPE_PURSUIT_PER_TILE: f32 = 200.0;

/// `max(0, CAP − min dist(own units → nearest visible uncaptured village))`
/// in TILES, 0 with no units or no visible village — the raw proximity
/// gradient shared by both the EXP_ELO_016 `village_proximity` term and the
/// EXP_ELO_018 `pursuit_potential` term (each applies its own per-tile
/// weight). A step toward the village banks potential now; hovering banks
/// nothing further (potential-based).
fn village_proximity_tiles(state: &GameState, player: i32) -> f32 {
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
        let _ = s;
        if !crate::rules::capture::is_capturable(
            state,
            idx,
            player,
            crate::rules::capture::CaptureKind::OPEN_VILLAGE,
            true,
        ) {
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
        Some(d) => (SHAPE_PROX_CAP - d).max(0) as f32,
        None => 0.0,
    }
}

/// EXP_ELO_016 proximity term (score-equivalent units).
fn village_proximity(state: &GameState, player: i32) -> f32 {
    SHAPE_PROX_PER_TILE * village_proximity_tiles(state, player)
}

/// EXP_ELO_018 isolated pursuit-progress potential Φ (score-equivalent
/// units): the same tile gradient as `village_proximity`, weighted at the
/// data-sized `SHAPE_PURSUIT_PER_TILE` so a step that closes distance to the
/// nearest visible uncaptured village banks enough reward to flip the
/// measured chosen−toward Q gap. Threaded on its own weight, so this arm can
/// run with the tech/SPT/army repricing off (`dev_w = 0`).
pub fn pursuit_potential(state: &GameState, player: i32) -> f32 {
    SHAPE_PURSUIT_PER_TILE * village_proximity_tiles(state, player)
}

pub fn dev_potential(state: &GameState, player: i32) -> f32 {
    let Some(tribe) = state.tribes.get(&player) else {
        return 0.0;
    };
    let tech_score: f32 = tribe
        .tech_vanilla
        .iter()
        .map(|t| 100.0 * crate::settings::technology::tech_tier(t.tech_type) as f32)
        .sum();
    let army_cost: f32 = tribe
        .units
        .iter()
        .map(|u| crate::rules::combat::unit_worth(u) as f32)
        .sum();
    let spt = crate::functions::get_tribe_spt(state, tribe) as f32;

    SHAPE_SPT * spt + SHAPE_ARMY_PER_COST * army_cost + village_proximity(state, player)
        - SHAPE_TECH_DEWEIGHT * tech_score
}

/// `score_snapshot` augmented with `dev_w`·Φ_dev + `pursuit_w`·Φ_pursuit per
/// side (EXP_ELO_016 development shaping + EXP_ELO_018 isolated pursuit-
/// progress shaping, independently weighted). Both weights zero short-circuits
/// to the raw snapshot (bit-exact legacy behavior, no Φ cost on the hot path).
pub fn shaped_snapshot(state: &GameState, player: i32, dev_w: f32, pursuit_w: f32) -> (f32, f32) {
    if dev_w == 0.0 && pursuit_w == 0.0 {
        let (my, opp) = score_snapshot(state, player);
        return (my as f32, opp as f32);
    }
    let phi = |id: i32| dev_w * dev_potential(state, id) + pursuit_w * pursuit_potential(state, id);
    let my = state
        .tribes
        .get(&player)
        .map(|t| t.score as f32 + phi(player))
        .unwrap_or(0.0);
    let opp = state
        .tribes
        .iter()
        .filter(|(id, _)| **id != player)
        .map(|(id, t)| t.score as f32 + phi(*id))
        .max_by(|a, b| a.total_cmp(b))
        .unwrap_or(0.0);
    (my, opp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::test_support::*;
    use crate::settings::units::get_unit_setting;
    use crate::states::{TechnologyState, TribeState};
    use crate::types::{TechnologyType, UnitType};

    #[test]
    fn shaped_snapshot_at_zero_w_is_the_raw_snapshot() {
        let mut state = GameState::default();
        let mut t1 = TribeState::default();
        t1.score = 123;
        let mut t2 = TribeState::default();
        t2.score = 456;
        state.tribes.insert(1, t1);
        state.tribes.insert(2, t2);
        assert_eq!(shaped_snapshot(&state, 1, 0.0, 0.0), (123.0, 456.0));
        let (my, opp) = shaped_snapshot(&state, 1, 1.0, 1.0);
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
        let tier = crate::settings::technology::tech_tier(TechnologyType::Riding) as f32;
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
    fn pursuit_potential_is_the_data_sized_progress_gradient() {
        let mk = |unit_idx: i32| {
            let mut state = GameState::default();
            add_visible_village(&mut state, 0);
            let mut t1 = TribeState::default();
            t1.units.push(unit_at(unit_idx, UnitType::Warrior));
            state.tribes.insert(1, t1);
            pursuit_potential(&state, 1)
        };
        // Row 0: idx = column = Chebyshev distance to the village at idx 0.
        // A one-tile close banks exactly SHAPE_PURSUIT_PER_TILE.
        assert!((mk(2) - mk(3) - SHAPE_PURSUIT_PER_TILE).abs() < 1e-3);
        // Weighted ~15x above the EXP_ELO_016 proximity garnish.
        assert!(SHAPE_PURSUIT_PER_TILE > 10.0 * SHAPE_PROX_PER_TILE);
    }
    #[test]
    fn shaped_snapshot_pursuit_weight_is_independent_of_dev_weight() {
        let mut state = GameState::default();
        add_visible_village(&mut state, 0);
        let mut t1 = TribeState::default();
        t1.score = 100;
        t1.units.push(unit_at(2, UnitType::Warrior)); // 2 tiles from village
        state.tribes.insert(1, t1);
        // pursuit_w only: augments my score by pursuit_potential, dev off.
        let (my_dev_off, _) = shaped_snapshot(&state, 1, 0.0, 1.0);
        let expected = 100.0 + pursuit_potential(&state, 1);
        assert!((my_dev_off - expected).abs() < 1e-3);
        // dev_w does not leak into the pursuit-only run.
        let (my_both, _) = shaped_snapshot(&state, 1, 1.0, 1.0);
        assert!((my_both - my_dev_off - dev_potential(&state, 1)).abs() < 1e-3);
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
