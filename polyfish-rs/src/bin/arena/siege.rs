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
