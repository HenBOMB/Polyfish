//! Consumer views: how a caller that has to ACT on the belief reads it.
//! Expansion targeting is a walk, so these are ordered by distance; the
//! belief's job is to prune, not to rank.

use crate::ai::belief::prediction::VillageGuess;
use crate::functions::get_chebyshev_distance;
use crate::states::GameState;

use super::belief::MapBelief;
use super::params::BELIEF_POOL;
use super::rules::is_explored;

impl MapBelief {
    /// LEGACY shape, byte-identical to the pre-SSOT `guess_villages` including
    /// its three known fidelity bugs. Consults none of the belief grids: its
    /// only job is the seam, pinned by `tests::parity::guess_villages_parity_holds_*`
    /// against a frozen copy. `state` carries the unit/city anchors.
    pub fn top_village_sites_legacy(
        &self,
        state: &GameState,
        max_sites: usize,
    ) -> Vec<VillageGuess> {
        crate::ai::belief::prediction::legacy_village_sites(state, self.observer, max_sites)
    }

    /// Belief-pruned, distance-ordered sites. The belief picks the
    /// `BELIEF_POOL` most plausible tiles; the nearest of those wins, keeping
    /// the quadrant spread that fixes the "88% of guesses in one bearing" bug.
    /// See `BELIEF_POOL` in `params.rs` for why probability must not order these.
    pub fn top_village_sites(
        &self,
        state: &GameState,
        max_sites: usize,
    ) -> Vec<VillageGuess> {
        let size = self.size;
        let Some(tribe) = state.tribes.get(&self.observer) else {
            return Vec::new();
        };
        let anchors: Vec<i32> = if tribe.units.is_empty() {
            tribe.cities.iter().map(|c| c.idx).collect()
        } else {
            tribe.units.iter().map(|u| u.coords.idx).collect()
        };
        if anchors.is_empty() || size <= 0 {
            return Vec::new();
        }

        let dist = |idx: i32| {
            anchors
                .iter()
                .map(|&a| get_chebyshev_distance(a, idx, size))
                .min()
                .unwrap_or(i32::MAX)
        };

        // Stage 1 — belief PRUNES: keep only the sites the generator's own
        // constraints most support.
        let mut pool: Vec<i32> = (0..size * size)
            .filter(|&idx| !is_explored(state, idx, self.observer) && self.p_village(idx) > 0.0)
            .collect();
        pool.sort_by(|&a, &b| {
            self.p_village(b)
                .total_cmp(&self.p_village(a))
                .then(a.cmp(&b))
        });
        pool.truncate(BELIEF_POOL.max(max_sites));

        // Stage 2 — distance DECIDES: nearest first, exactly as the legacy
        // picker ordered, because the scout has to walk there.
        pool.sort_by_key(|&idx| (dist(idx), idx));

        let (mut cx, mut cy) = (0i32, 0i32);
        for &a in &anchors {
            cx += a % size;
            cy += a / size;
        }
        cx /= anchors.len() as i32;
        cy /= anchors.len() as i32;
        let quadrant = |idx: i32| ((idx % size > cx) as u8) * 2 + ((idx / size > cy) as u8);

        let mut picks: Vec<i32> = Vec::new();
        let mut used_quads = std::collections::HashSet::new();
        for pass in 0..2 {
            for &idx in &pool {
                if picks.len() >= max_sites {
                    break;
                }
                if picks.contains(&idx)
                    || picks
                        .iter()
                        .any(|&p| get_chebyshev_distance(p, idx, size) < 3)
                {
                    continue;
                }
                if pass == 0 && used_quads.contains(&quadrant(idx)) {
                    continue;
                }
                used_quads.insert(quadrant(idx));
                picks.push(idx);
            }
        }

        let opp_tribe = state.tribes.get(&self.opponent).map(|t| t.tribe_type);
        picks
            .into_iter()
            .map(|site| VillageGuess {
                tile: site,
                tribe: if self.p_opponent_affinity(site) > 0.5 {
                    opp_tribe
                } else {
                    None
                },
                confidence: self.p_village(site).clamp(0.05, 1.0),
            })
            .collect()
    }
}
