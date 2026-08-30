//! Per-turn cached economic plan: which Mine-lane tiles currently partner a
//! committed Forge hub, so real building can follow the SAME plan
//! `bin/eco_plan`'s joint frontier (`rules::eco_plan::enumerate_empire`)
//! already computes correctly, instead of `scoring.rs`'s old reactive
//! "cluster near whatever mine happened to land first" heuristic (the
//! original "gravity" complaint this whole investigation started from).
//!
//! Mine lane only this round (EXP_ELO_100) — the other lanes (Sawmill,
//! Windmill) can follow the same pattern once this one is measured.
//!
//! # Momentum, not a fresh answer every turn
//!
//! Credit accumulates turn to turn instead of being overwritten: once a
//! tile earns credit for partnering a committed hub, it keeps that credit
//! even if a later, more-informed recompute's specific hub choice shifts.
//! `EXP_ELO_100`'s own turn4->turn7 convergence check is the reason this
//! matters — a rough early estimate (planning around a village we don't
//! hold yet) and a precise later one (once it's real) can legitimately
//! disagree on the exact best hub tile without either being wrong, and
//! erasing the early credit on every refinement would waste the very
//! investment this feature exists to protect. Credit is hard-invalidated
//! only when the tile's OWNING city is actually lost.
//!
//! # Prospective cities
//!
//! A village we don't hold yet is planned around (as if already captured,
//! seeded with the same radius-1 square a real capture claims — see
//! `rules::eco_plan::allocate_value_with_prospective`) only when
//! `ai::movement::village_race_confidence` clears `PROSPECTIVE_CONFIDENCE_MIN`
//! — Verdi's own framing: "confidence needs to be high enough for the
//! assumption to hold." A prospective city's own territory is never
//! credited directly (we can't build there yet); only HELD cities' tiles
//! are, against the full committed-hub set (real + prospective), since a
//! currently-buildable mine that will end up adjacent to a not-yet-real
//! neighbour's future hub is still the right move now — the real engine's
//! cross-city partner credit (`city_build_on`) doesn't care about build
//! order, only final adjacency and ownership.

use crate::ai::movement::village_race_confidence;
use crate::ai::oracle_macro::{retakeable_village, still_capturable};
use crate::functions::get_chebyshev_distance;
use crate::rules::eco_plan::{
    allocate_value, allocate_value_with_prospective, enumerate_empire, pick_for_goal, Goal,
    Scenario, SCENARIOS,
};
use crate::states::{GameState, PlayerId};
use crate::types::ResourceType;
use std::collections::{HashMap, HashSet};

/// Minimum `village_race_confidence` for a not-yet-captured village to be
/// planned around as if it were already ours.
pub const PROSPECTIVE_CONFIDENCE_MIN: f32 = 0.75;

/// Forge lane, BorderGrowth on — the scenario `EXP_ELO_099` established as
/// the only one whose territory can even see a neighbour's planned partner
/// (natural scenarios never reach not-yet-owned ground at all).
fn plan_scenario() -> Scenario {
    SCENARIOS[7]
}

fn is_metal_mountain(state: &GameState, idx: i32) -> bool {
    state.resources.get(&idx).and_then(|r| r.as_ref()).is_some_and(|r| r.resource_type == ResourceType::Metal)
}

#[derive(Clone, Debug, Default)]
pub struct EcoPlanCommit {
    key: Option<(i32, PlayerId)>,
    /// Metal tiles that partner some committed Forge hub, accumulated
    /// across turns (see module doc — momentum, not overwritten).
    mine_partners: HashSet<i32>,
    /// Which currently-held city a credited tile's credit is tied to, so
    /// losing that city can drop it again.
    owning_city_of: HashMap<i32, i32>,
}

impl EcoPlanCommit {
    /// Does `tile` currently partner a committed Forge hub?
    pub fn is_mine_partner(&self, tile: i32) -> bool {
        self.mine_partners.contains(&tile)
    }

    /// Recompute if `(turn, player)` has moved on since the last call;
    /// no-op (and cheap — one tuple comparison) otherwise. Call once per
    /// ply from the same place `LaneState`/`TurnCounters` already get
    /// updated, never per candidate move: `rules::eco_plan::enumerate_empire`
    /// costs single-digit milliseconds even on a "+border" scenario, which
    /// is fine once a turn and ruinous evaluated per candidate (rank_plies
    /// scores dozens of candidates, tens of times per real move decision).
    pub fn update(&mut self, state: &GameState, player: PlayerId) {
        let key = (state.settings.turn, player);
        if self.key == Some(key) {
            return;
        }
        self.key = Some(key);

        let Some(tribe) = state.tribes.get(&player) else {
            return;
        };
        let held: Vec<i32> = tribe.cities.iter().map(|c| c.idx).collect();
        let held_set: HashSet<i32> = held.iter().copied().collect();

        // Hard invalidation: a tile whose owning city is no longer ours
        // loses its credit — that investment is genuinely gone, not just
        // stale.
        let owning_city_of = &mut self.owning_city_of;
        let mine_partners = &mut self.mine_partners;
        owning_city_of.retain(|tile, city| {
            let keep = held_set.contains(city);
            if !keep {
                mine_partners.remove(tile);
            }
            keep
        });

        if held.is_empty() {
            return;
        }

        // REAL, confirmed VILLAGES only -- `still_capturable`/
        // `retakeable_village`, not the broader `expand_target_valid` (which
        // also matches capturable Ruins: a one-time reward, not a
        // settlement, with no territory or hub potential — including one
        // here starved `enumerate_empire` of every valid combo, zero plans).
        // Deliberately not the fog-guessed sites `oracle_macro::
        // expand_targets` tops up with below `COMMIT_CITY_TARGET` cities,
        // either: a guess already carries its own, separate "does a village
        // even exist here" confidence (`VillageGuess::confidence`) that
        // `village_race_confidence` was never designed to compose with --
        // Verdi's own example was a village we could already see.
        let prospective: HashSet<i32> = state
            .structures
            .keys()
            .copied()
            .filter(|&idx| still_capturable(state, idx, player) || retakeable_village(state, idx, player))
            .filter(|&t| village_race_confidence(state, player, t) >= PROSPECTIVE_CONFIDENCE_MIN)
            .collect();

        let mut cities = held.clone();
        cities.extend(prospective.iter().copied());
        let sc = plan_scenario();
        let scs: Vec<Scenario> = cities.iter().map(|_| sc).collect();
        let terr = if prospective.is_empty() {
            allocate_value(state, &cities, &scs, 0)
        } else {
            allocate_value_with_prospective(state, &cities, &prospective, &scs, 0)
        };

        let owned: HashSet<crate::types::TechnologyType> = tribe
            .tech_vanilla
            .iter()
            .filter(|t| t.discovered)
            .map(|t| t.tech_type)
            .collect();
        let plans = enumerate_empire(state, &cities, &terr, &scs, &owned, 0, 8, true);
        let Some(best) = pick_for_goal(&plans, Goal::Balanced) else {
            return;
        };

        let hub_sites: Vec<i32> = best.hubs.iter().flatten().copied().collect();
        if hub_sites.is_empty() {
            return;
        }
        let size = state.settings.size;
        for (ci, &city) in cities.iter().enumerate() {
            if !held_set.contains(&city) {
                continue; // only currently-buildable (held) ground can be credited now
            }
            for &t in &terr[ci] {
                if is_metal_mountain(state, t)
                    && hub_sites.iter().any(|&h| get_chebyshev_distance(t, h, size) <= 1)
                {
                    self.mine_partners.insert(t);
                    self.owning_city_of.insert(t, city);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coords::Coords;
    use crate::states::{CityState, ResourceState, StructureState, TileState, TribeState};
    use crate::types::{StructureType, TerrainType};

    fn owned_field(owner: PlayerId, ruling: i32, size: i32) -> TileState {
        let mut t = TileState::default();
        t.terrain_type = TerrainType::Field;
        t.owner = owner;
        t.ruling_city_coords = Some(Coords::from_index(ruling, size));
        t
    }

    /// Minimal real state: one held city with a standing Mine adjacent to a
    /// Field tile that can become its Forge, and enough Crop tiles to clear
    /// the BorderGrowth reachability floor (`POP_FOR_LEVEL_4`).
    fn one_city_state() -> (GameState, PlayerId, i32, i32, i32) {
        let size = 11;
        let mut state = GameState::default();
        state.settings.size = size;
        state.settings.current_player_turn_id = 1;
        let center = 5 * size + 5;
        let hub = 5 * size + 6;
        let mine = 4 * size + 6;
        let crops = [4 * size + 4, 4 * size + 5, 5 * size + 4, 6 * size + 4];

        state.tiles.insert(center, owned_field(1, center, size));
        // A city's own tile keeps `StructureType::Village` in `structures`
        // even long after capture (confirmed against a real captured city's
        // dumped state) -- without this, `tile_options` treats the centre as
        // an ordinary empty Field and a hub can land ON the city itself.
        state.structures.insert(
            center,
            Some(StructureState { structure_type: StructureType::Village, level: 1, founded: 0 }),
        );
        state.tiles.insert(hub, owned_field(1, center, size));
        let mut mine_tile = owned_field(1, center, size);
        mine_tile.terrain_type = TerrainType::Mountain;
        state.tiles.insert(mine, mine_tile);
        state.resources.insert(mine, Some(ResourceState { resource_type: ResourceType::Metal, ..Default::default() }));
        state.structures.insert(mine, Some(StructureState { structure_type: StructureType::Mine, level: 1, founded: 0 }));
        for &c in &crops {
            state.tiles.insert(c, owned_field(1, center, size));
            state.resources.insert(c, Some(ResourceState { resource_type: ResourceType::Crop, ..Default::default() }));
        }

        let mut tribe = TribeState { id: 1, ..Default::default() };
        let mut territory: Vec<i32> = vec![center, hub, mine];
        territory.extend(crops.iter().copied());
        tribe.cities.push(CityState { idx: center, owner: 1, _territory: territory, ..Default::default() });
        state.tribes.insert(1, tribe);
        state.tribes.insert(2, TribeState { id: 2, ..Default::default() });
        (state, 1, center, hub, mine)
    }

    #[test]
    fn credits_a_mine_that_partners_the_committed_hub() {
        let (state, player, ..) = one_city_state();
        let mut commit = EcoPlanCommit::default();
        commit.update(&state, player);
        // The standing mine feeds SOME hub in a 9-tile-plus inner ring; the
        // specific site is `build_out`'s call, this only asserts the wiring
        // (state -> plan -> credited partner set) actually produces credit.
        assert!(!commit.mine_partners.is_empty(), "expected at least one credited partner tile");
    }

    #[test]
    fn recompute_is_gated_on_turn_and_player() {
        let (mut state, player, ..) = one_city_state();
        let mut commit = EcoPlanCommit::default();
        commit.update(&state, player);
        let first_key = commit.key;
        commit.update(&state, player); // same turn -- must be a no-op
        assert_eq!(commit.key, first_key);
        state.settings.turn += 1;
        commit.update(&state, player); // new turn -- must recompute
        assert_eq!(commit.key, Some((state.settings.turn, player)));
    }

    #[test]
    fn losing_a_city_drops_its_credited_tiles() {
        let (mut state, player, center, _hub, mine) = one_city_state();
        let mut commit = EcoPlanCommit::default();
        commit.update(&state, player);
        assert!(commit.owning_city_of.values().any(|&c| c == center));

        state.settings.turn += 1;
        state.tribes.get_mut(&player).unwrap().cities.clear();
        commit.update(&state, player);
        assert!(!commit.is_mine_partner(mine), "credit must not survive losing the owning city");
    }

    #[test]
    fn no_tribe_or_no_cities_does_not_panic_and_credits_nothing() {
        let mut state = GameState::default();
        state.settings.size = 11;
        let mut commit = EcoPlanCommit::default();
        commit.update(&state, 1); // no tribe at all
        assert!(commit.mine_partners.is_empty());
        state.tribes.insert(1, TribeState { id: 1, ..Default::default() });
        commit.update(&state, 1); // tribe with zero cities
        assert!(commit.mine_partners.is_empty());
    }
}
