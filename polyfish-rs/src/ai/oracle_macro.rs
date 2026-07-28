//! EXP_ELO_026 "oracle macro": a hand-scripted macro layer over the unchanged
//! net, testing whether third-city reach fails at the macro level (commitment
//! and star allocation) rather than micro execution. Two independent steers,
//! both inference-only: an expansion commitment (focus the pursuit channel on
//! one sticky capturable village) and a star gate (drop root tech purchases
//! that would leave the capture unfunded). Nothing here touches training.

use crate::moves::Move;
use crate::states::{GameState, PlayerId};
use crate::types::{MoveType, StructureType};

/// Stars that must remain after a tech purchase for it to pass the gate while
/// a commitment is active — rough price of fielding a capturer.
pub const STAR_GATE_RESERVE: i32 = 5;

/// EXP_ELO_028: order types painted into the goal channels. The discriminant
/// is the channel offset from `features::CH_ORDER_START`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum OrderKind {
    Expand = 0,
    Attack = 1,
    Defend = 2,
}

/// EXP_ELO_028: global spending stance. The discriminant is the channel
/// offset from `features::CH_STANCE_START` (one-hot plane).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Stance {
    #[default]
    Grow = 0,
    Arm = 1,
    Unlock = 2,
}

/// EXP_ELO_028 Stage-1 macro goal: concurrent painted orders (each a target
/// tile) plus one global spending stance. Encoded into the appended goal
/// channels; `orders` must stay sorted so identical goals produce identical
/// feature bytes (the eval cache and tree reuse hash them).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct MacroGoal {
    pub orders: Vec<(OrderKind, i32)>,
    pub stance: Stance,
}

/// Stage-1 scripted goal-setter (the EXP_ELO_026 rules generalized to the
/// orders-field vocabulary): EXPAND painted on every capturable village while
/// under `COMMIT_CITY_TARGET` cities; ATTACK on an enemy city when ≥2 own
/// units stand within Chebyshev 3 of it; DEFEND on an own city when ≥2 enemy
/// units stand within Chebyshev 2 (threat predicate tightened per Phase 0).
/// Stance: ARM while any DEFEND order is active, else GROW.
pub fn scripted_goal(state: &GameState, player: PlayerId) -> MacroGoal {
    let size = state.settings.size as i32;
    let cheb =
        |a: i32, b: i32| ((a / size) - (b / size)).abs().max(((a % size) - (b % size)).abs());
    let Some(tribe) = state.tribes.get(&player) else {
        return MacroGoal::default();
    };
    let own_units: Vec<i32> = tribe.units.iter().map(|u| u.coords.idx).collect();
    let mut orders: Vec<(OrderKind, i32)> = Vec::new();

    if tribe.cities.len() < COMMIT_CITY_TARGET {
        for &idx in state.structures.keys() {
            if still_capturable(state, idx, player) {
                orders.push((OrderKind::Expand, idx));
            }
        }
    }
    for (id, t) in &state.tribes {
        if *id == player {
            continue;
        }
        for c in &t.cities {
            let near = own_units.iter().filter(|&&u| cheb(u, c.idx) <= 3).count();
            if near >= 2 {
                orders.push((OrderKind::Attack, c.idx));
            }
        }
    }
    let enemy_units: Vec<i32> = state
        .tribes
        .iter()
        .filter(|(id, _)| **id != player)
        .flat_map(|(_, t)| t.units.iter().map(|u| u.coords.idx))
        .collect();
    for c in &tribe.cities {
        let near = enemy_units.iter().filter(|&&u| cheb(u, c.idx) <= 2).count();
        if near >= 2 {
            orders.push((OrderKind::Defend, c.idx));
        }
    }

    orders.sort();
    let stance = if orders.iter().any(|(k, _)| *k == OrderKind::Defend) {
        Stance::Arm
    } else {
        Stance::Grow
    };
    MacroGoal { orders, stance }
}

/// Whether the EXP_ELO_026 star gate applies under `goal`: GROW stance with
/// at least one active EXPAND order (the order list already encodes the
/// under-3-cities condition).
pub fn goal_star_gate(goal: &MacroGoal) -> bool {
    goal.stance == Stance::Grow && goal.orders.iter().any(|(k, _)| *k == OrderKind::Expand)
}

/// City count at which the commitment retires (the third-city objective).
pub const COMMIT_CITY_TARGET: usize = 3;

/// True while `idx` still holds a village capturable by `player`: Village
/// structure on an unowned tile that `player` has explored (the pursuit
/// channel's predicate — see features.rs).
pub fn still_capturable(state: &GameState, idx: i32, player: PlayerId) -> bool {
    let is_village = state
        .structures
        .get(&idx)
        .and_then(|s| s.as_ref())
        .map_or(false, |s| s.structure_type == StructureType::Village);
    is_village
        && state
            .tiles
            .get(&idx)
            .map_or(false, |t| t.owner == 0 && t.explorers.contains(&player))
}

/// Nearest capturable village by Chebyshev distance to any of `player`'s
/// units (fallback anchor: its cities), lowest tile index on ties.
pub fn nearest_capturable_village(state: &GameState, player: PlayerId) -> Option<i32> {
    let size = state.settings.size as i32;
    let tribe = state.tribes.get(&player)?;
    let anchors: Vec<i32> = if tribe.units.is_empty() {
        tribe.cities.iter().map(|c| c.idx).collect()
    } else {
        tribe.units.iter().map(|u| u.coords.idx).collect()
    };
    if anchors.is_empty() {
        return None;
    }
    let cheb =
        |a: i32, b: i32| ((a / size) - (b / size)).abs().max(((a % size) - (b % size)).abs());
    state
        .structures
        .keys()
        .filter(|&&idx| still_capturable(state, idx, player))
        .map(|&idx| {
            let d = anchors.iter().map(|&a| cheb(a, idx)).min().unwrap_or(i32::MAX);
            (d, idx)
        })
        .min()
        .map(|(_, idx)| idx)
}

/// Per-decision commitment update: retired at `COMMIT_CITY_TARGET` cities,
/// sticky while the current target stays capturable, else re-picked nearest.
pub fn update_commitment(
    state: &GameState,
    player: PlayerId,
    prev: Option<i32>,
) -> Option<i32> {
    let tribe = state.tribes.get(&player)?;
    if tribe.cities.len() >= COMMIT_CITY_TARGET {
        return None;
    }
    if let Some(idx) = prev {
        if still_capturable(state, idx, player) {
            return Some(idx);
        }
    }
    nearest_capturable_village(state, player)
}

/// Root-only star-gate predicate: keep a Research move only when the buyer
/// retains `STAR_GATE_RESERVE` stars after the purchase (so an affluent tribe
/// can still buy passage tech). Every non-Research move passes.
pub fn passes_star_gate(state: &GameState, m: &dyn Move) -> bool {
    if m.move_type() != MoveType::Research {
        return true;
    }
    let player = state.settings.current_player_turn_id;
    let Some(tribe) = state.tribes.get(&player) else {
        return true;
    };
    let Ok(tech) = m.tech_type() else {
        return true;
    };
    tribe.stars - crate::functions::get_tech_cost(tribe, tech) >= STAR_GATE_RESERVE
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moves::EndTurnMove;
    use crate::moves::research::ResearchMove;
    use crate::Coords;
    use crate::states::{StructureState, TileState, TribeState, UnitState};
    use crate::types::{TechnologyType, UnitType};

    fn unit_at(idx: i32) -> UnitState {
        UnitState {
            unit_type: UnitType::Warrior,
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

    fn state_with_villages(unit_idx: i32, villages: &[i32]) -> GameState {
        let mut state = GameState::default();
        for &v in villages {
            add_visible_village(&mut state, v);
        }
        let mut t1 = TribeState::default();
        t1.units.push(unit_at(unit_idx));
        state.tribes.insert(1, t1);
        state
    }

    #[test]
    fn commitment_picks_nearest_is_sticky_and_retires_at_three_cities() {
        let mut state = state_with_villages(0, &[3, 5]);
        // Fresh pick: village at idx 3 is 3 tiles away vs 5 for idx 5.
        assert_eq!(update_commitment(&state, 1, None), Some(3));
        // Sticky: an existing valid commitment survives a nearer alternative.
        assert_eq!(update_commitment(&state, 1, Some(5)), Some(5));
        // Retires once the third city exists.
        let t1 = state.tribes.get_mut(&1).unwrap();
        for _ in 0..3 {
            t1.cities.push(Default::default());
        }
        assert_eq!(update_commitment(&state, 1, Some(5)), None);
    }

    #[test]
    fn commitment_repicks_when_target_is_captured() {
        let mut state = state_with_villages(0, &[3, 5]);
        state.tiles.get_mut(&5).unwrap().owner = 2;
        assert_eq!(update_commitment(&state, 1, Some(5)), Some(3));
    }

    #[test]
    fn scripted_goal_paints_expand_attack_defend_and_sets_stance() {
        let mut state = state_with_villages(0, &[3, 5]);
        // Under 3 cities with two capturable villages → two EXPAND orders,
        // sorted, GROW stance, star gate active.
        let g = scripted_goal(&state, 1);
        assert_eq!(
            g.orders,
            vec![(OrderKind::Expand, 3), (OrderKind::Expand, 5)]
        );
        assert_eq!(g.stance, Stance::Grow);
        assert!(goal_star_gate(&g));

        // Enemy city at 40 = (3,7) with two own units within Chebyshev 3
        // (39 = (3,6) and 29 = (2,7)) → ATTACK order.
        let mut t2 = TribeState::default();
        t2.cities.push(crate::states::CityState { idx: 40, ..Default::default() });
        state.tribes.insert(2, t2);
        let t1 = state.tribes.get_mut(&1).unwrap();
        t1.units.push(unit_at(39));
        t1.units.push(unit_at(29));
        let g = scripted_goal(&state, 1);
        assert!(g.orders.contains(&(OrderKind::Attack, 40)));
        assert_eq!(g.stance, Stance::Grow);

        // Two enemy units within 2 of an own city → DEFEND + ARM stance.
        state.tribes.get_mut(&1).unwrap().cities.push(crate::states::CityState {
            idx: 0,
            ..Default::default()
        });
        let t2 = state.tribes.get_mut(&2).unwrap();
        t2.units.push(unit_at(1));
        t2.units.push(unit_at(12));
        let g = scripted_goal(&state, 1);
        assert!(g.orders.contains(&(OrderKind::Defend, 0)));
        assert_eq!(g.stance, Stance::Arm);
    }

    #[test]
    fn star_gate_blocks_only_underfunded_research() {
        let mut state = state_with_villages(0, &[3]);
        state.settings.current_player_turn_id = 1;
        let tech = TechnologyType::Organization;
        let cost = crate::functions::get_tech_cost(state.tribes.get(&1).unwrap(), tech);
        let research = ResearchMove::new(tech);

        state.tribes.get_mut(&1).unwrap().stars = cost + STAR_GATE_RESERVE - 1;
        assert!(!passes_star_gate(&state, &research));

        state.tribes.get_mut(&1).unwrap().stars = cost + STAR_GATE_RESERVE;
        assert!(passes_star_gate(&state, &research));

        // Non-research moves always pass, regardless of stars.
        state.tribes.get_mut(&1).unwrap().stars = 0;
        assert!(passes_star_gate(&state, &EndTurnMove));
    }
}
