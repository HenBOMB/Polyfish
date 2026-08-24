//! The observation context: everything the derivation is allowed to look at,
//! resolved once per `observe`. Every fog-sensitive read is funnelled through
//! here, which is what makes the fog discipline auditable in one place.

use crate::functions::get_chebyshev_distance;
use crate::states::{GameState, PlayerId};
use crate::types::{StructureType, TerrainType, TribeType};

use super::belief::Fidelity;
use super::params::{c3_evidence, climate_p_seat2, LIKELIHOOD_FLOOR};
use super::rules::known_sites;

/// Read-only observation context: everything the derivation is allowed to see,
/// resolved once. Every fog-sensitive read is funnelled through here.
pub(super) struct Ctx<'a> {
    pub(super) state: &'a GameState,
    pub(super) size: i32,
    pub(super) player_count: usize,
    pub(super) known: Vec<i32>,
    pub(super) own_capital: Option<i32>,
    pub(super) sighted_capital: Option<i32>,
    pub(super) own_climate: i32,
    pub(super) opp_climate: i32,
    pub(super) own_tribe: TribeType,
    pub(super) opp_tribe: TribeType,
    /// True when the observer holds the LOWER player id; the affinity
    /// flood-fill is a round-robin in seat order and seat 1 wins ties, so the
    /// climate likelihood is not symmetric between the two seats.
    observer_is_seat1: bool,
    /// Explored land fraction, the fallback for P(land) under fog.
    pub(super) land_rate: f32,
    pub(super) fidelity: Fidelity,
    pub(super) has_opponent: bool,
    pub(super) explored: Vec<bool>,
}

impl<'a> Ctx<'a> {
    pub(super) fn new(
        state: &'a GameState,
        observer: PlayerId,
        opponent: PlayerId,
        fidelity: Fidelity,
    ) -> Ctx<'a> {
        let size = state.settings.size;
        let n = (size * size).max(0) as usize;
        let mut explored = vec![false; n];
        for (&i, t) in &state.tiles {
            if i >= 0 && (i as usize) < n && t.explorers.contains(&observer) {
                explored[i as usize] = true;
            }
        }

        let own_tribe = state
            .tribes
            .get(&observer)
            .map(|t| t.tribe_type)
            .unwrap_or(TribeType::Imperius);
        let opp_tribe = state
            .tribes
            .get(&opponent)
            .map(|t| t.tribe_type)
            .unwrap_or(TribeType::Bardur);

        // The observer's own spawn capital. `tile.capital_of` is reassigned to
        // the capturer, so it is not a stable anchor; `starting_tile_coords` is
        // written once by mapgen and never moves.
        let own_capital = state
            .tribes
            .get(&observer)
            .map(|t| t.starting_tile_coords.idx)
            .filter(|&i| i >= 0 && (i as usize) < n);

        // The opponent's spawn capital counts as sighted once the observer has
        // explored it, whoever holds it now. Any explored capital tile that is
        // not the observer's own spawn tile is it (1v1 scoping).
        let sighted_capital = state
            .tiles
            .iter()
            .filter(|(i, t)| {
                **i >= 0
                    && (**i as usize) < n
                    && t.explorers.contains(&observer)
                    && t.capital_of != 0
            })
            .map(|(&i, _)| i)
            .find(|&i| Some(i) != own_capital);

        let land_tiles = state
            .tiles
            .iter()
            .filter(|(i, t)| {
                **i >= 0 && (**i as usize) < n && t.explorers.contains(&observer)
            })
            .filter(|(_, t)| !matches!(t.terrain_type, TerrainType::Water | TerrainType::Ocean))
            .count();
        let explored_count = explored.iter().filter(|e| **e).count();
        let land_rate = if explored_count > 0 {
            land_tiles as f32 / explored_count as f32
        } else {
            1.0
        };

        Ctx {
            state,
            size,
            player_count: state.tribes.len().max(2),
            known: known_sites(state, observer),
            own_capital,
            sighted_capital,
            own_climate: crate::types::classic_climate_id(own_tribe),
            opp_climate: crate::types::classic_climate_id(opp_tribe),
            own_tribe,
            opp_tribe,
            observer_is_seat1: observer < opponent,
            land_rate,
            fidelity,
            has_opponent: opponent != observer,
            explored,
        }
    }

    /// The legacy veto: any cardinal neighbour is Ocean. Kept only to reproduce
    /// the pre-SSOT rules under [`Fidelity::LegacyBugs`].
    pub(super) fn has_ocean_cardinal(&self, idx: i32) -> bool {
        crate::functions::get_plus_sign_indices(idx, self.size)
            .into_iter()
            .any(|n| {
                self.state
                    .tiles
                    .get(&n)
                    .map_or(false, |t| t.terrain_type == TerrainType::Ocean)
            })
    }

    pub(super) fn explored(&self, idx: i32) -> bool {
        idx >= 0
            && (idx as usize) < self.explored.len()
            && self.explored[idx as usize]
    }

    /// Climate is NOT stripped by `obscure_fog`, so this is gated on
    /// exploration and must stay that way.
    pub(super) fn climate_of(&self, idx: i32) -> Option<i32> {
        if !self.explored(idx) {
            return None;
        }
        self.state
            .tiles
            .get(&idx)
            .map(|t| t.climate)
            .filter(|&c| c != 0)
    }

    /// Is a known site within Chebyshev `r` of `idx`? Spacing exclusion and
    /// constraint discharge must agree on this, so it is stated once.
    pub(super) fn known_within(&self, idx: i32, r: i32) -> bool {
        self.known
            .iter()
            .any(|&k| get_chebyshev_distance(idx, k, self.size) <= r)
    }

    pub(super) fn has_village(&self, idx: i32) -> bool {
        matches!(
            self.state.structures.get(&idx),
            Some(Some(s)) if s.structure_type == StructureType::Village
        )
    }

    /// P(tile `idx` carries the OPPONENT's climate | opponent capital at `k`).
    /// Delta is evaluated in SEAT order because the flood-fill's tie-break is.
    pub(super) fn p_opponent_climate(&self, idx: i32, k: i32) -> f32 {
        let Some(own) = self.own_capital else {
            return 0.5;
        };
        let d_own = get_chebyshev_distance(idx, own, self.size);
        let d_opp = get_chebyshev_distance(idx, k, self.size);
        if self.observer_is_seat1 {
            // seat1 = observer, seat2 = opponent; table is P(seat2's climate).
            climate_p_seat2(d_own - d_opp)
        } else {
            // seat1 = opponent, seat2 = observer; the opponent's share is the
            // complement of the table value.
            1.0 - climate_p_seat2(d_opp - d_own)
        }
    }

    /// Bounded log-likelihood of the observed climate field under the
    /// hypothesis "opponent capital at `k`": the MEAN per-tile log-likelihood
    /// scaled by [`C3_EVIDENCE`](super::params::C3_EVIDENCE), never the raw
    /// product (see that constant).
    pub(super) fn climate_log_likelihood(&self, k: i32) -> f32 {
        let w = c3_evidence();
        if w <= 0.0 {
            return 0.0;
        }
        let mut acc = 0.0f32;
        let mut n = 0u32;
        for idx in 0..self.explored.len() as i32 {
            let Some(c) = self.climate_of(idx) else {
                continue;
            };
            let p_opp = self.p_opponent_climate(idx, k);
            let l = if c == self.opp_climate {
                p_opp
            } else if c == self.own_climate {
                1.0 - p_opp
            } else {
                continue;
            };
            acc += l.clamp(LIKELIHOOD_FLOOR, 1.0 - LIKELIHOOD_FLOOR).ln();
            n += 1;
        }
        if n == 0 {
            return 0.0;
        }
        acc / n as f32 * w
    }

    /// P(a fog tile is mountain), mixing the two tribes' generator rates by the
    /// affinity posterior at that tile.
    pub(super) fn mountain_rate(&self, p_opp_affinity: f32) -> f32 {
        let own = crate::mapgen::get_tribe_biome_rates(self.own_tribe).mountain;
        let opp = crate::mapgen::get_tribe_biome_rates(self.opp_tribe).mountain;
        let p = p_opp_affinity.clamp(0.0, 1.0);
        own * (1.0 - p) + opp * p
    }

    /// P(a fog tile is land). Free on fully-dry maps, which is what training
    /// runs; elsewhere fall back to the observer's explored land fraction.
    pub(super) fn p_land(&self, _idx: i32) -> f32 {
        if crate::mapgen::is_fully_dry(self.state.settings.map_type) {
            1.0
        } else {
            self.land_rate.clamp(0.05, 1.0)
        }
    }
}
