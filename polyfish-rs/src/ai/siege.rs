//! Shared siege open/close transition detection (EXP_ELO_041's original
//! definition). A siege is an enemy unit standing on an owned city tile; an
//! episode opens on first contact and closes as LOST (ownership flipped) or
//! UNSIEGED (enemy gone, city kept).
//!
//! Extracted from the arena binary's `SiegeTracker` (horizon-compression
//! Stage 2 / EXP_ELO_120's pressure aux head, which needs `self_play` to
//! reuse this EXACT definition rather than inventing a new one) — this is
//! the pure transition-detection core; the arena binary layers its own
//! per-seat counters and episode-detail recording on top, and `self_play`
//! layers its own per-turn event windowing.

use crate::states::GameState;
use std::collections::HashSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SiegeOutcome {
    Lost,
    Unsieged,
}

/// One scan's worth of transitions.
#[derive(Debug, Default)]
pub struct SiegeTransitions {
    /// `(owner, city_idx)` pairs that newly became sieged this scan.
    pub opened: Vec<(i32, i32)>,
    /// `(owner, city_idx, outcome)` for episodes that closed this scan.
    pub closed: Vec<(i32, i32, SiegeOutcome)>,
}

/// Advance `active` by one scan against `state`, returning this scan's
/// open/close transitions. `active` must be the same set threaded across
/// every scan for one game — it is the tracker's only state, keyed by
/// `(owner player_id, city tile idx)`.
pub fn scan_siege_transitions(
    state: &GameState,
    active: &mut HashSet<(i32, i32)>,
) -> SiegeTransitions {
    let mut closed = Vec::new();
    active.retain(|&(owner, idx)| {
        let still_owned = state
            .tribes
            .get(&owner)
            .map_or(false, |t| t.cities.iter().any(|c| c.idx == idx));
        if !still_owned {
            closed.push((owner, idx, SiegeOutcome::Lost));
            return false;
        }
        let enemy_on = crate::functions::get_true_unit_at(state, idx)
            .map_or(false, |u| u.owner != owner);
        if !enemy_on {
            closed.push((owner, idx, SiegeOutcome::Unsieged));
            return false;
        }
        true
    });

    let mut opened = Vec::new();
    for (pid, t) in &state.tribes {
        for c in &t.cities {
            let enemy_on = crate::functions::get_true_unit_at(state, c.idx)
                .map_or(false, |u| u.owner != *pid);
            if enemy_on && !active.contains(&(*pid, c.idx)) {
                active.insert((*pid, c.idx));
                opened.push((*pid, c.idx));
            }
        }
    }

    SiegeTransitions { opened, closed }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::states::{CityState, TribeState, UnitState};

    fn state_with(owner_city: (i32, i32), enemy_on_city: Option<i32>) -> GameState {
        let mut state = GameState::default();
        state.settings.size = 11;
        let (owner, city_idx) = owner_city;
        let mut tribe = TribeState::default();
        tribe.id = owner;
        tribe.cities.push(CityState { idx: city_idx, owner, ..Default::default() });
        state.tribes.insert(owner, tribe);
        if let Some(enemy_owner) = enemy_on_city {
            let mut enemy_tribe = state.tribes.entry(enemy_owner).or_insert_with(|| {
                let mut t = TribeState::default();
                t.id = enemy_owner;
                t
            });
            enemy_tribe.units.push(UnitState {
                coords: crate::coords::Coords::from_index(city_idx, 11),
                owner: enemy_owner,
                ..Default::default()
            });
        }
        state
    }

    #[test]
    fn enemy_landing_on_a_city_opens_a_siege() {
        let state = state_with((1, 50), Some(2));
        let mut active = HashSet::new();
        let t = scan_siege_transitions(&state, &mut active);
        assert_eq!(t.opened, vec![(1, 50)]);
        assert!(t.closed.is_empty());
        assert!(active.contains(&(1, 50)));
    }

    #[test]
    fn same_enemy_still_present_does_not_reopen() {
        let state = state_with((1, 50), Some(2));
        let mut active = HashSet::new();
        scan_siege_transitions(&state, &mut active);
        let t = scan_siege_transitions(&state, &mut active);
        assert!(t.opened.is_empty(), "an already-active siege must not re-open");
    }

    #[test]
    fn enemy_leaving_closes_as_unsieged() {
        let state_sieged = state_with((1, 50), Some(2));
        let mut active = HashSet::new();
        scan_siege_transitions(&state_sieged, &mut active);
        let state_cleared = state_with((1, 50), None);
        let t = scan_siege_transitions(&state_cleared, &mut active);
        assert_eq!(t.closed, vec![(1, 50, SiegeOutcome::Unsieged)]);
        assert!(active.is_empty());
    }

    #[test]
    fn city_captured_closes_as_lost() {
        let state_sieged = state_with((1, 50), Some(2));
        let mut active = HashSet::new();
        scan_siege_transitions(&state_sieged, &mut active);
        // Ownership flips: city 50 moves to tribe 2's list, tribe 1 loses it.
        let mut state_captured = GameState::default();
        state_captured.settings.size = 11;
        let mut tribe2 = TribeState::default();
        tribe2.id = 2;
        tribe2.cities.push(CityState { idx: 50, owner: 2, ..Default::default() });
        state_captured.tribes.insert(2, tribe2);
        let t = scan_siege_transitions(&state_captured, &mut active);
        assert_eq!(t.closed, vec![(1, 50, SiegeOutcome::Lost)]);
    }
}
