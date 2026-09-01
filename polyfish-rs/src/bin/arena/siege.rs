//! Siege-defense accounting (EXP_ELO_041).
//! 
//! An episode opens when an enemy unit stands on an owned city tile and
//! resolves as UNSIEGED (enemy gone, city kept) or LOST (ownership
//! flipped). Scanned after every move.


/// Per-match result, attributed to configurations (1 or 2), not seats.
/// EXP_ELO_041: per-seat siege bookkeeping. A "siege" is an enemy unit
/// standing on an owned city tile; each episode resolves as UNSIEGED (enemy
/// gone, city kept) or LOST (ownership flipped). Scanned after every move.
pub(crate) struct SiegeTracker {
    pub(crate) active: std::collections::HashMap<(i32, i32), serde_json::Value>, // (owner pid, city idx) -> open facts
    pub(crate) sieges: [u32; 2],   // per SEAT (P1, P2): episodes started
    pub(crate) unsieged: [u32; 2], // …resolved by clearing the attacker
    pub(crate) lost: [u32; 2],     // …resolved by losing the city
    /// EXP_ELO_049: one closed record per episode, emitted into the game dump.
    pub(crate) episodes: Vec<serde_json::Value>,
    pub(crate) detail: bool,
}

impl SiegeTracker {
    pub(crate) fn new(detail: bool) -> Self {
        Self {
            active: Default::default(),
            sieges: [0; 2],
            unsieged: [0; 2],
            lost: [0; 2],
            episodes: Vec::new(),
            detail,
        }
    }

    /// Facts at the moment the attacker steps onto the city — the ones that
    /// decide whether the defence was POSSIBLE, separately from whether it
    /// happened: who can strike the tile next turn, how far the nearest
    /// unit is, what the bank holds, and whether Tier 2 had even named this
    /// city as something to defend.
    pub(crate) fn open_facts(
        state: &polyfish::states::GameState,
        owner: i32,
        idx: i32,
        goal: Option<&polyfish::ai::oracle_macro::MacroGoal>,
    ) -> serde_json::Value {
        let size = state.settings.size;
        let attacker = polyfish::functions::get_true_unit_at(state, idx);
        let tribe = state.tribes.get(&owner);
        let city_level = tribe
            .and_then(|t| t.cities.iter().find(|c| c.idx == idx).map(|c| c.level))
            .unwrap_or(0);
        // Own units, excluding anything standing on the besieged tile.
        let own: Vec<&polyfish::states::UnitState> = tribe
            .map(|t| t.units.iter().filter(|u| u.coords.idx != idx).collect())
            .unwrap_or_default();
        let nearest = own
            .iter()
            .map(|u| polyfish::functions::get_chebyshev_distance(u.coords.idx, idx, size))
            .min();
        let responders = own
            .iter()
            .filter(|u| polyfish::ai::combat::unit_covers_threat(state, u, idx))
            .count();
        let ordered_defend = goal.map(|g| {
            g.orders.iter().any(|(k, t)| {
                *k == polyfish::ai::oracle_macro::OrderKind::Defend && *t == idx
            })
        });
        serde_json::json!({
            "owner": owner,
            "city": idx,
            "city_level": city_level,
            "turn_open": state.settings.turn,
            "attacker": attacker.map(|u| format!("{:?}", u.unit_type)),
            "attacker_health": attacker.map(|u| u.health),
            "own_units": own.len(),
            "nearest_unit_dist": nearest,
            // Units that could strike the besieging unit next turn — the
            // capability the unsiege actually needs.
            "responders": responders,
            "stars": tribe.map(|t| t.stars),
            "spt": tribe.map(|t| polyfish::functions::get_tribe_spt(state, t)),
            "defend_ordered": ordered_defend,
        })
    }

    /// The transition-detection core (open/closed pairs, in the same shape
    /// `polyfish::ai::siege::scan_siege_transitions` returns) is shared with
    /// `self_play`'s pressure aux head (EXP_ELO_120) — extracted so both
    /// consumers agree on exactly what a "siege" is. This method keeps its
    /// own `active: HashMap<_, Value>` (not the shared function's plain
    /// `HashSet`) because it needs to carry `open_facts` payloads per
    /// episode; it mirrors the shared function's own open/close predicates
    /// exactly rather than calling it directly, so the two data structures
    /// (facts-carrying map vs. plain set) don't have to be kept in lockstep
    /// by hand every scan.
    pub(crate) fn scan(
        &mut self,
        state: &polyfish::states::GameState,
        goals: [Option<&polyfish::ai::oracle_macro::MacroGoal>; 2],
    ) {
        let (sieges, unsieged, lost) =
            (&mut self.sieges, &mut self.unsieged, &mut self.lost);
        let episodes = &mut self.episodes;
        let detail = self.detail;
        self.active.retain(|&(owner, idx), open| {
            let seat = (owner - 1).clamp(0, 1) as usize;
            let still_owned = state
                .tribes
                .get(&owner)
                .map_or(false, |t| t.cities.iter().any(|c| c.idx == idx));
            let mut close = |outcome: &str| {
                if detail {
                    let mut rec = open.clone();
                    rec["outcome"] = serde_json::json!(outcome);
                    rec["turn_close"] = serde_json::json!(state.settings.turn);
                    episodes.push(rec);
                }
            };
            if !still_owned {
                lost[seat] += 1;
                close("lost");
                return false;
            }
            let enemy_on = polyfish::functions::get_true_unit_at(state, idx)
                .map_or(false, |u| u.owner != owner);
            if !enemy_on {
                unsieged[seat] += 1;
                close("unsieged");
                return false;
            }
            true
        });
        for (pid, t) in &state.tribes {
            let seat = (*pid - 1).clamp(0, 1) as usize;
            for c in &t.cities {
                let enemy_on = polyfish::functions::get_true_unit_at(state, c.idx)
                    .map_or(false, |u| u.owner != *pid);
                if enemy_on && !self.active.contains_key(&(*pid, c.idx)) {
                    let facts = if self.detail {
                        Self::open_facts(state, *pid, c.idx, goals[seat])
                    } else {
                        serde_json::Value::Null
                    };
                    self.active.insert((*pid, c.idx), facts);
                    sieges[seat] += 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod cross_check {
    //! Byte-for-byte cross-check that this struct's open/close predicates
    //! haven't drifted from the shared `polyfish::ai::siege` core, without
    //! actually routing through it (see the `scan` doc comment for why).
    use super::*;
    use polyfish::ai::siege::{SiegeOutcome, scan_siege_transitions};
    use polyfish::states::{CityState, GameState, TribeState, UnitState};
    use std::collections::HashSet;

    fn state_with(owner_city: (i32, i32), enemy_owner: Option<i32>) -> GameState {
        let mut state = GameState::default();
        state.settings.size = 11;
        let (owner, city_idx) = owner_city;
        let mut tribe = TribeState::default();
        tribe.id = owner;
        tribe.cities.push(CityState { idx: city_idx, owner, ..Default::default() });
        state.tribes.insert(owner, tribe);
        if let Some(eo) = enemy_owner {
            let mut t = TribeState::default();
            t.id = eo;
            t.units.push(UnitState {
                coords: polyfish::coords::Coords::from_index(city_idx, 11),
                owner: eo,
                ..Default::default()
            });
            state.tribes.insert(eo, t);
        }
        state
    }

    #[test]
    fn siege_tracker_agrees_with_the_shared_core_on_open_and_close() {
        let sieged = state_with((1, 50), Some(2));
        let mut arena_tracker = SiegeTracker::new(false);
        let mut shared_active = HashSet::new();

        arena_tracker.scan(&sieged, [None, None]);
        let t = scan_siege_transitions(&sieged, &mut shared_active);
        assert_eq!(t.opened, vec![(1, 50)]);
        assert_eq!(arena_tracker.sieges, [1, 0], "arena's own opened-counter must agree");
        assert!(arena_tracker.active.contains_key(&(1, 50)));

        let cleared = state_with((1, 50), None);
        arena_tracker.scan(&cleared, [None, None]);
        let t2 = scan_siege_transitions(&cleared, &mut shared_active);
        assert_eq!(t2.closed, vec![(1, 50, SiegeOutcome::Unsieged)]);
        assert_eq!(arena_tracker.unsieged, [1, 0]);
        assert!(arena_tracker.active.is_empty());
    }
}
