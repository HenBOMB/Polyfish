//! Per-unit persistent goals (Aug 2026, per-unit-goal design doc): the
//! turn-level `MacroGoal` says "the tribe is expanding toward these tiles
//! this turn"; this module says "unit 41 is specifically the one going to
//! tile 87, and here's how far along it is." Reuses `OrderKind` as the goal
//! vocabulary rather than inventing a parallel one — Phase 1 only ever
//! mints `Expand` goals, sourced from this turn's already-committed
//! `MacroGoal.orders`.
//!
//! Real-trajectory-only (not threaded into macro-mcts rollouts, see the
//! design doc's Fork 2): lives on `MacroMctsAgent` next to `StanceCommit`,
//! reconciled once per real ply, never cloned into `Node`.

use std::collections::VecDeque;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::ai::movement::assign_expand_targets_by_id;
use crate::ai::oracle_macro::{expand_target_valid, MacroGoal, OrderKind};
use crate::functions::{get_city_at, get_structure_at, get_unit_at};
use crate::states::{GameState, PlayerId};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnitGoal {
    pub kind: OrderKind,
    pub target: i32,
}

/// Outcome of reconciling one unit's goal this ply — the trace-observable
/// status (Step 4 threads this straight into `POLYFISH_PLY_TRACE`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GoalStatus {
    /// No goal this ply (either never assigned, or this turn's `MacroGoal`
    /// carries no Expand order for this unit to pick up).
    Idle,
    /// Actively pursuing a goal assigned on a prior ply.
    Pursuing,
    /// Freshly assigned this ply.
    Assigned,
    /// Reached this ply — own city now sits on the target tile.
    Completed,
    /// Target stopped being a valid Expand target (captured by someone
    /// else's tribe past the point of retake, etc.) before this unit got
    /// there — dropped, no bonus, free for reassignment next ply.
    Invalidated,
}

#[derive(Clone, Debug, Default)]
pub struct UnitGoalStore {
    goals: FxHashMap<u32, VecDeque<UnitGoal>>,
}

impl UnitGoalStore {
    pub fn active(&self, id: u32) -> Option<UnitGoal> {
        self.goals.get(&id).and_then(|q| q.front().copied())
    }

    /// Pop the front goal (completed or invalidated) — the unit is idle
    /// again until reassigned.
    pub fn advance(&mut self, id: u32) {
        if let Some(q) = self.goals.get_mut(&id) {
            q.pop_front();
            if q.is_empty() {
                self.goals.remove(&id);
            }
        }
    }

    pub fn assign(&mut self, id: u32, g: UnitGoal) {
        self.goals.entry(id).or_default().push_back(g);
    }

    /// Drop every stored goal whose unit ID is no longer alive — the
    /// generic handler for death and conversion (see the design doc's
    /// Lifecycle section: both are `UndoCallback`-only mutations that
    /// can't reach this store directly, so pruning against the live set
    /// is the only hook that needs to exist).
    pub fn retain_ids(&mut self, live: &FxHashSet<u32>) {
        self.goals.retain(|id, _| live.contains(id));
    }

    /// Targets already claimed by some unit's active goal — excluded from
    /// fresh assignment so two units never chase the same tile.
    pub fn active_targets(&self) -> FxHashSet<i32> {
        self.goals.values().filter_map(|q| q.front()).map(|g| g.target).collect()
    }
}

/// Per-unit completion predicate for an Expand goal. Deliberately NOT
/// `fog_order_dead`: that whole-tribe check treats any owned-city tile as
/// "not dead" regardless of which unit got there, which would let a unit
/// keep chasing (and get paid for) a target a *different* unit already
/// captured. Returns `Some(true)` = completed (own city on it, pays the
/// bonus), `Some(false)` = invalidated (no longer capturable/retakeable, no
/// bonus), `None` = still live, keep pursuing.
fn goal_outcome(state: &GameState, target: i32, player: PlayerId) -> Option<bool> {
    let Some(tile) = state.tiles.get(&target) else {
        return Some(false);
    };
    if tile.owner == player && get_city_at(state, target).is_some() {
        return Some(true);
    }
    // Ruins are destroyed on capture, never converted into an owned city
    // like a Village is (whose StructureType::Village record never clears —
    // see actions/city.rs), so the check above can't see a Ruin complete.
    // Gated on occupancy: a guessed village site that resolves to empty
    // ground hits the ordinary invalidation branch below well before a unit
    // could physically stand on it (vision outruns approach), so "our own
    // unit is here and nothing's left" reads as a capture, not a bad guess.
    if get_structure_at(state, target).is_none()
        && get_unit_at(state, target).is_some_and(|u| u.owner == player)
    {
        return Some(true);
    }
    if !tile.explorers.contains(&player) {
        return None; // unexplored (a fog guess) -- can't disconfirm what we can't see
    }
    if !expand_target_valid(state, target, player) {
        return Some(false);
    }
    None
}

/// Runs once per real ply (never inside a rollout — see the design doc's
/// Fork 2). Three passes over `tribe.units` in deterministic Vec order:
/// prune dead/converted-away units, advance completed/invalidated goals,
/// assign idle units to unclaimed Expand targets from this turn's
/// `MacroGoal`. Pure bookkeeping — callers read `store.active(id)` for
/// pricing; this function doesn't touch reward itself.
///
/// Returns each live unit's `GoalStatus` this ply, for observability
/// (`POLYFISH_PLY_TRACE`) -- a unit completed/invalidated and immediately
/// reassigned in the same ply reports `Assigned` (the newer fact), not
/// both; the trace's `goal` field still shows the target actually changed.
pub fn reconcile_unit_goals(
    state: &GameState,
    player: PlayerId,
    goal: &MacroGoal,
    store: &mut UnitGoalStore,
) -> FxHashMap<u32, GoalStatus> {
    let mut status: FxHashMap<u32, GoalStatus> = FxHashMap::default();
    let Some(tribe) = state.tribes.get(&player) else {
        store.goals.clear();
        return status;
    };

    let live_ids: FxHashSet<u32> = tribe.units.iter().map(|u| u.id).collect();
    store.retain_ids(&live_ids);

    for unit in &tribe.units {
        let Some(g) = store.active(unit.id) else {
            status.insert(unit.id, GoalStatus::Idle);
            continue;
        };
        if g.kind != OrderKind::Expand {
            status.insert(unit.id, GoalStatus::Pursuing);
            continue;
        }
        match goal_outcome(state, g.target, player) {
            Some(true) => {
                store.advance(unit.id);
                status.insert(unit.id, GoalStatus::Completed);
            }
            Some(false) => {
                store.advance(unit.id);
                status.insert(unit.id, GoalStatus::Invalidated);
            }
            None => {
                status.insert(unit.id, GoalStatus::Pursuing);
            }
        }
    }

    // `goal.orders` legitimately keeps an ACHIEVED target listed all turn
    // (it needs to keep paying the flat completion bonus, target-keyed, in
    // `goal_potential`) -- but that doesn't mean it should be offered for
    // FRESH per-unit assignment. Without this filter, the unit that just
    // captured it gets immediately reassigned right back to the tile it's
    // already standing on (distance 0 always wins the greedy match),
    // stranding it there for the rest of the turn instead of freeing it up
    // for a genuinely new target. Same predicate `goal_outcome` uses to
    // decide whether an ALREADY-assigned goal should advance.
    let expand_targets: Vec<i32> = goal
        .orders
        .iter()
        .filter(|(k, _)| *k == OrderKind::Expand)
        .map(|(_, t)| *t)
        .filter(|t| goal_outcome(state, *t, player).is_none())
        .collect();
    if expand_targets.is_empty() {
        return status;
    }
    let claimed = store.active_targets();
    let unclaimed: Vec<i32> = expand_targets.into_iter().filter(|t| !claimed.contains(t)).collect();
    if unclaimed.is_empty() {
        return status;
    }
    let idle_units: Vec<&crate::states::UnitState> = tribe.units.iter().filter(|u| store.active(u.id).is_none()).collect();
    if idle_units.is_empty() {
        return status;
    }

    for (id, target) in assign_expand_targets_by_id(state, player, &idle_units, &unclaimed) {
        store.assign(id, UnitGoal { kind: OrderKind::Expand, target });
        status.insert(id, GoalStatus::Assigned);
    }
    status
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::oracle_macro::test_support::{add_visible_village, unit_at};
    use crate::states::{CityState, TribeState, UnitState};

    fn empty_goal() -> MacroGoal {
        MacroGoal::default()
    }

    fn expand_goal(target: i32) -> MacroGoal {
        MacroGoal { orders: vec![(OrderKind::Expand, target)], ..Default::default() }
    }

    /// One tribe, one unit at `unit_idx` with the given id, one visible
    /// village at `village_idx`.
    fn state_with_unit(unit_idx: i32, unit_id: u32, village_idx: i32) -> GameState {
        let mut state = GameState::default();
        state.settings.size = 11;
        add_visible_village(&mut state, village_idx);
        let mut tribe = TribeState::default();
        tribe.units.push(UnitState { id: unit_id, ..unit_at(unit_idx) });
        state.tribes.insert(1, tribe);
        state
    }

    #[test]
    fn assign_when_idle() {
        let mut store = UnitGoalStore::default();
        store.assign(1, UnitGoal { kind: OrderKind::Expand, target: 50 });
        assert_eq!(store.active(1), Some(UnitGoal { kind: OrderKind::Expand, target: 50 }));
    }

    #[test]
    fn advance_pops_and_clears() {
        let mut store = UnitGoalStore::default();
        store.assign(1, UnitGoal { kind: OrderKind::Expand, target: 50 });
        store.advance(1);
        assert_eq!(store.active(1), None);
        assert!(!store.goals.contains_key(&1));
    }

    #[test]
    fn retain_ids_prunes_dead_and_converted() {
        let mut store = UnitGoalStore::default();
        store.assign(1, UnitGoal { kind: OrderKind::Expand, target: 50 });
        store.assign(2, UnitGoal { kind: OrderKind::Expand, target: 51 });
        let live: FxHashSet<u32> = [1].into_iter().collect();
        store.retain_ids(&live);
        assert_eq!(store.active(1), Some(UnitGoal { kind: OrderKind::Expand, target: 50 }));
        assert_eq!(store.active(2), None);
    }

    #[test]
    fn active_targets_reflects_only_fronts() {
        let mut store = UnitGoalStore::default();
        store.assign(1, UnitGoal { kind: OrderKind::Expand, target: 50 });
        store.assign(1, UnitGoal { kind: OrderKind::Expand, target: 60 });
        let targets = store.active_targets();
        assert!(targets.contains(&50));
        assert!(!targets.contains(&60));
    }

    #[test]
    fn reconcile_no_expand_order_leaves_units_idle() {
        let state = GameState::default();
        let mut store = UnitGoalStore::default();
        let goal = empty_goal();
        reconcile_unit_goals(&state, 1, &goal, &mut store);
        assert!(store.goals.is_empty());
    }

    #[test]
    fn reconcile_assigns_idle_unit_to_expand_target() {
        let state = state_with_unit(0, 1, 5);
        let mut store = UnitGoalStore::default();
        reconcile_unit_goals(&state, 1, &expand_goal(5), &mut store);
        assert_eq!(store.active(1), Some(UnitGoal { kind: OrderKind::Expand, target: 5 }));
    }

    #[test]
    fn reconcile_progress_keeps_pursuing_same_target() {
        let state = state_with_unit(0, 1, 5);
        let mut store = UnitGoalStore::default();
        let goal = expand_goal(5);
        reconcile_unit_goals(&state, 1, &goal, &mut store);
        // Unit hasn't reached the target yet — a second reconcile on an
        // unchanged state must not reassign or duplicate the goal.
        reconcile_unit_goals(&state, 1, &goal, &mut store);
        assert_eq!(store.active(1), Some(UnitGoal { kind: OrderKind::Expand, target: 5 }));
    }

    #[test]
    fn reconcile_completes_and_pops_own_city_case() {
        let mut state = state_with_unit(0, 1, 5);
        let mut store = UnitGoalStore::default();
        store.assign(1, UnitGoal { kind: OrderKind::Expand, target: 5 });
        // Unit founded a city on the target tile.
        state.tiles.get_mut(&5).unwrap().owner = 1;
        state.tribes.get_mut(&1).unwrap().cities.push(CityState { idx: 5, owner: 1, ..Default::default() });
        reconcile_unit_goals(&state, 1, &empty_goal(), &mut store);
        assert_eq!(store.active(1), None);
    }

    #[test]
    fn reconcile_invalidates_and_pops_enemy_captured_case() {
        let mut state = state_with_unit(0, 1, 5);
        let mut store = UnitGoalStore::default();
        store.assign(1, UnitGoal { kind: OrderKind::Expand, target: 5 });
        // A different tribe's capital now sits on the target: no longer
        // capturable (OPEN_VILLAGE excludes enemy villages) and never
        // retakeable (capital exemption) — this is the corrected
        // predicate's regression case, distinct from `fog_order_dead`'s
        // owned-city exemption which doesn't apply per-unit.
        state.tiles.get_mut(&5).unwrap().owner = 2;
        state.tiles.get_mut(&5).unwrap().capital_of = 2;
        reconcile_unit_goals(&state, 1, &empty_goal(), &mut store);
        assert_eq!(store.active(1), None);
    }

    #[test]
    fn reconcile_prunes_dead_units() {
        let state = state_with_unit(0, 1, 5);
        let mut store = UnitGoalStore::default();
        // Goal stored for a unit id that isn't present in tribe.units —
        // stands in for death (remove_unit shifts/removes the Vec entry).
        store.assign(99, UnitGoal { kind: OrderKind::Expand, target: 5 });
        reconcile_unit_goals(&state, 1, &empty_goal(), &mut store);
        assert_eq!(store.active(99), None);
    }

    #[test]
    fn reconcile_drops_goal_on_conversion_then_reassigns() {
        // Unit 1 (the pre-conversion owner) held a goal; it's no longer in
        // the tribe (converted away). Unit 2 is a fresh idle unit that
        // should pick up the now-unclaimed target instead of unit 1
        // keeping it (conversion must not transfer the goal).
        let state = state_with_unit(0, 2, 5);
        let mut store = UnitGoalStore::default();
        store.assign(1, UnitGoal { kind: OrderKind::Expand, target: 5 });
        reconcile_unit_goals(&state, 1, &expand_goal(5), &mut store);
        assert_eq!(store.active(1), None);
        assert_eq!(store.active(2), Some(UnitGoal { kind: OrderKind::Expand, target: 5 }));
    }
}
