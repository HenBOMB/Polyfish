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

/// EXP_ELO_018: score-equivalents per tile of closed distance toward the
/// nearest visible uncaptured village, for the *isolated* pursuit-progress
/// reward (independent weight, see `pursuit_potential`). Sized from the
/// measured chosen−toward Q gap on wrong-move pursuer-turns (median 0.19 /
/// p75 0.42 normalized, ≈150–350 score-equiv through `score_norm≈700`) —
/// ~15× EXP_ELO_016's `SHAPE_PROX_PER_TILE`, which was too weak to flip the
/// decision (FM-3 pursuit metric — see the notes.md pursuit diagnosis;
/// current status in current_understanding.md).
pub const SHAPE_PURSUIT_PER_TILE: f32 = 200.0;

/// Chebyshev distance between two row-major tile indices.
fn cheb(a: i32, b: i32, width: i32) -> i32 {
    let (ra, ca) = (a / width, a % width);
    let (rb, cb) = (b / width, b % width);
    (ra - rb).abs().max((ca - cb).abs())
}

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

// ---- EXP_ELO_028 Phase 1c: goal-priced in-tree shaping ------------------
// The painted macro goal gets an actuator: each stance/order prices the
// resource conversion it names, as a potential on the goal-holder's side
// only (the opponent's goal is unknown at search time). Sized like
// SHAPE_PURSUIT_PER_TILE — large enough to flip a decisive Q gap (~0.15-0.2
// normalized through score_norm≈600-700).

/// Score-equivalents per star-per-turn of income while stance is GROW.
pub const SHAPE_GOAL_SPT: f32 = 150.0;
/// Extra score-equivalents per star of living army while stance is ARM
/// (on top of the game score's 5·cost).
pub const SHAPE_GOAL_ARM_PER_COST: f32 = 50.0;
/// Score-equivalents per tile of closed distance toward a painted EXPAND
/// target. Summed over targets; an achieved (self-owned) target holds the
/// full cap so the final capture banks its step instead of cliffing -CAP.
pub const SHAPE_GOAL_EXPAND_PER_TILE: f32 = 200.0;
/// Score-equivalents per OWNED environment-recommended tech (GoalAux) —
/// buying the map-fit tech banks this in-tree; off-fit tech banks nothing.
pub const SHAPE_GOAL_TECH_FIT: f32 = 150.0;
/// v2.4 scout term: score-equivalents per explored tile while GROW holds,
/// no EXPAND target is known, and expansion is unfinished — pays the
/// "find your village" step the audit showed nothing was paying for.
pub const SHAPE_GOAL_SCOUT: f32 = 25.0;
/// v2.4: extra proximity-tiles banked when an EXPAND target is achieved
/// (on top of the held cap) — makes the final Capture-vs-Step choice a
/// ~0.4-normalized landslide instead of one more step of gradient.
pub const SHAPE_GOAL_EXPAND_DONE: f32 = 2.0;
/// Score-equivalents per living Rider while the rider push is on (open
/// terrain + active EXPAND).
pub const SHAPE_GOAL_RIDER: f32 = 100.0;

/// Goal potential Φ_goal for `player` under `goal` (score-equivalent units).
pub fn goal_potential(
    state: &GameState,
    player: i32,
    goal: &crate::ai::oracle_macro::MacroGoal,
    aux: Option<&crate::ai::oracle_macro::GoalAux>,
) -> f32 {
    use crate::ai::oracle_macro::{OrderKind, Stance};
    let Some(tribe) = state.tribes.get(&player) else {
        return 0.0;
    };
    let mut phi = match goal.stance {
        Stance::Grow => {
            SHAPE_GOAL_SPT * crate::functions::get_tribe_spt(state, tribe) as f32
        }
        Stance::Arm => {
            SHAPE_GOAL_ARM_PER_COST
                * tribe
                    .units
                    .iter()
                    .map(|u| crate::settings::units::get_unit_setting(u.unit_type).cost as f32)
                    .sum::<f32>()
        }
        Stance::Unlock => 0.0,
    };
    let width = state.settings.size as i32;
    let mut has_expand = false;
    if width > 0 {
        for (kind, idx) in &goal.orders {
            if *kind != OrderKind::Expand {
                continue;
            }
            has_expand = true;
            let Some(tile) = state.tiles.get(idx) else {
                continue;
            };
            // Unexplored (guessed) targets always pay approach — reading their
            // owner would leak FOW. The completion bonus needs a real city
            // capture, not a border-grown empty tile.
            let approach = || {
                let d = tribe
                    .units
                    .iter()
                    .map(|u| cheb(u.coords.idx, *idx, width))
                    .min()
                    .unwrap_or(i32::MAX);
                (SHAPE_PROX_CAP - d).max(0) as f32
            };
            let tiles = if !tile.explorers.contains(&player) {
                approach()
            } else if tile.owner == player {
                if crate::functions::get_city_at(state, *idx).is_some() {
                    SHAPE_PROX_CAP as f32 + SHAPE_GOAL_EXPAND_DONE
                } else {
                    0.0
                }
            } else if tile.owner != 0 {
                0.0
            } else {
                approach()
            };
            phi += SHAPE_GOAL_EXPAND_PER_TILE * tiles;
        }
    }
    // Scout term: with no known village to approach, revealing tiles IS the
    // expansion progress. Retires once a target exists or expansion is done.
    if goal.stance == Stance::Grow
        && !has_expand
        && tribe.cities.len() < crate::ai::oracle_macro::COMMIT_CITY_TARGET
    {
        let explored = state
            .tiles
            .iter()
            .filter(|(_, t)| t.explorers.contains(&player))
            .count();
        phi += SHAPE_GOAL_SCOUT * explored as f32;
    }
    if let Some(aux) = aux {
        let owned = aux
            .recommended_techs
            .iter()
            .filter(|t| {
                crate::settings::technology::is_tech_unlocked(&tribe.tech_vanilla, **t)
            })
            .count();
        phi += SHAPE_GOAL_TECH_FIT * owned as f32;
        if aux.rider_push {
            let riders = tribe
                .units
                .iter()
                .filter(|u| u.unit_type == crate::types::UnitType::Rider)
                .count();
            phi += SHAPE_GOAL_RIDER * riders as f32;
        }
    }
    phi
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
    fn goal_potential_prices_each_stance_and_expand_progress() {
        use crate::ai::oracle_macro::{MacroGoal, OrderKind, Stance};
        let mut state = GameState::default();
        add_visible_village(&mut state, 0);
        let mut t1 = TribeState::default();
        t1.units.push(unit_at(2, UnitType::Warrior)); // 2 tiles from village 0
        state.tribes.insert(1, t1);

        // ARM pays the army's star cost.
        let arm = MacroGoal { orders: vec![], stance: Stance::Arm };
        let cost = get_unit_setting(UnitType::Warrior).cost as f32;
        assert!((goal_potential(&state, 1, &arm, None) - SHAPE_GOAL_ARM_PER_COST * cost).abs() < 1e-4);

        // GROW pays SPT plus the scout term (no EXPAND target known, <3
        // cities, one explored tile in this state).
        let grow = MacroGoal { orders: vec![], stance: Stance::Grow };
        let spt = crate::functions::get_tribe_spt(&state, state.tribes.get(&1).unwrap()) as f32;
        let expected = SHAPE_GOAL_SPT * spt + SHAPE_GOAL_SCOUT;
        assert!((goal_potential(&state, 1, &grow, None) - expected).abs() < 1e-4);

        // EXPAND order: a one-tile close banks one step of the gradient.
        let ex = |orders| MacroGoal { orders, stance: Stance::Arm };
        let base = goal_potential(&state, 1, &ex(vec![(OrderKind::Expand, 0)]), None);
        state.tribes.get_mut(&1).unwrap().units[0] = unit_at(1, UnitType::Warrior);
        let closer = goal_potential(&state, 1, &ex(vec![(OrderKind::Expand, 0)]), None);
        assert!((closer - base - SHAPE_GOAL_EXPAND_PER_TILE).abs() < 1e-3);

        // Achieved target holds cap + completion bonus (no cliff on capture);
        // enemy-owned pays 0. Capture makes the tile an owned CITY.
        state.tiles.get_mut(&0).unwrap().owner = 1;
        state.tribes.get_mut(&1).unwrap().cities.push(crate::states::CityState {
            idx: 0,
            ..Default::default()
        });
        let achieved = goal_potential(&state, 1, &ex(vec![(OrderKind::Expand, 0)]), None);
        let arm_only = goal_potential(&state, 1, &arm, None);
        let done = SHAPE_GOAL_EXPAND_PER_TILE * (SHAPE_PROX_CAP as f32 + SHAPE_GOAL_EXPAND_DONE);
        assert!((achieved - arm_only - done).abs() < 1e-3);
        assert!(achieved >= closer);
        state.tiles.get_mut(&0).unwrap().owner = 2;
        let lost = goal_potential(&state, 1, &ex(vec![(OrderKind::Expand, 0)]), None);
        assert!((lost - arm_only).abs() < 1e-4);
    }

    #[test]
    fn scout_term_pays_reveals_until_a_target_or_third_city() {
        use crate::ai::oracle_macro::{MacroGoal, OrderKind, Stance};
        let mut state = GameState::default();
        let mut t1 = TribeState::default();
        t1.units.push(unit_at(60, UnitType::Warrior));
        state.tribes.insert(1, t1);
        let grow = MacroGoal { orders: vec![], stance: Stance::Grow };

        // Each newly explored tile banks SHAPE_GOAL_SCOUT.
        let base = goal_potential(&state, 1, &grow, None);
        let mut tile = TileState::default();
        tile.explorers.insert(1);
        state.tiles.insert(50, tile);
        let one = goal_potential(&state, 1, &grow, None);
        assert!((one - base - SHAPE_GOAL_SCOUT).abs() < 1e-4);

        // A known EXPAND target retires the scout term: the potential is
        // exactly SPT + approach gradient (unit at 60 is cheb 1 from 50).
        let with_target = MacroGoal {
            orders: vec![(OrderKind::Expand, 50)],
            stance: Stance::Grow,
        };
        let anchored = goal_potential(&state, 1, &with_target, None);
        let spt0 = crate::functions::get_tribe_spt(&state, state.tribes.get(&1).unwrap()) as f32;
        let approach = SHAPE_GOAL_EXPAND_PER_TILE * (SHAPE_PROX_CAP - 1) as f32;
        assert!((anchored - SHAPE_GOAL_SPT * spt0 - approach).abs() < 1e-3);
        // ARM never scouts; neither does a 3-city tribe.
        let arm = MacroGoal { orders: vec![], stance: Stance::Arm };
        let arm_phi = goal_potential(&state, 1, &arm, None);
        let cost = get_unit_setting(UnitType::Warrior).cost as f32;
        assert!((arm_phi - SHAPE_GOAL_ARM_PER_COST * cost).abs() < 1e-4);
        let t1 = state.tribes.get_mut(&1).unwrap();
        for _ in 0..3 {
            t1.cities.push(Default::default());
        }
        let done = goal_potential(&state, 1, &grow, None);
        let spt = crate::functions::get_tribe_spt(&state, state.tribes.get(&1).unwrap()) as f32;
        assert!((done - SHAPE_GOAL_SPT * spt).abs() < 1e-4);
    }

    #[test]
    fn goal_aux_pays_tech_fit_and_riders() {
        use crate::ai::oracle_macro::{GoalAux, MacroGoal, Stance};
        use crate::states::TechnologyState;
        use crate::types::TechnologyType;
        let mut state = GameState::default();
        let mut t1 = TribeState::default();
        t1.units.push(unit_at(60, UnitType::Rider));
        state.tribes.insert(1, t1);
        let goal = MacroGoal { orders: vec![], stance: Stance::Grow };
        let aux = GoalAux {
            recommended_techs: vec![TechnologyType::Mining],
            rider_push: true,
            ..Default::default()
        };
        let base = goal_potential(&state, 1, &goal, None);
        // Rider push pays per living Rider; the unowned recommendation pays 0.
        let with_aux = goal_potential(&state, 1, &goal, Some(&aux));
        assert!((with_aux - base - SHAPE_GOAL_RIDER).abs() < 1e-3);
        // Owning the recommended tech banks the fit bonus.
        state.tribes.get_mut(&1).unwrap().tech_vanilla.push(TechnologyState {
            tech_type: TechnologyType::Mining,
            discovered: true,
            discovered_turn: 0,
        });
        let owned = goal_potential(&state, 1, &goal, Some(&aux));
        assert!((owned - with_aux - SHAPE_GOAL_TECH_FIT).abs() < 1e-3);
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
