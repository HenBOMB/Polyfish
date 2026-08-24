//! Identity and reuse: the fingerprint that says two beliefs are the same
//! belief, and the two caches built on it. [`BeliefKey`] is what licenses
//! every form of reuse here — if it is wrong, they all serve stale grids.

use std::sync::Arc;

use crate::states::{GameState, PlayerId};
use crate::types::StructureType;

use super::belief::MapBelief;
use super::rules::is_explored;

/// Cache fingerprint. Excludes `turn` so the belief stays stable across a
/// simulated turn advance and a tree can hold one `Arc` from its root.
/// `villages` and `cities` count separately because a capture moves a site from
/// one to the other, leaving both the sum and the explored count unchanged.
///
/// `hash` identifies the explored/site SETS, not just their sizes. Counts alone
/// are enough to invalidate within one game (the explored set only grows) but
/// collide across games — which silently served a wrong belief from the
/// thread-local memo until `thread_local_memo_agrees_with_a_fresh_derivation`
/// caught it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BeliefKey {
    explored: u32,
    villages: u32,
    cities: u32,
    hash: u64,
}

/// splitmix64 finalizer. Combined by wrapping addition so the fingerprint is
/// independent of tile iteration order.
fn mix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl MapBelief {
    /// Cheap fingerprint for cache invalidation; never allocates the grids.
    pub fn key_of(state: &GameState, observer: PlayerId) -> BeliefKey {
        let mut explored = 0u32;
        let mut hash = 0u64;
        for (&i, t) in &state.tiles {
            if t.explorers.contains(&observer) {
                explored += 1;
                hash = hash.wrapping_add(mix(i as u64));
            }
        }
        let mut villages = 0u32;
        for (&i, st) in &state.structures {
            let is_village = st
                .as_ref()
                .map_or(false, |st| st.structure_type == StructureType::Village);
            if is_village && is_explored(state, i, observer) {
                villages += 1;
                hash = hash.wrapping_add(mix(i as u64 ^ 0xA5A5_0000_0000_0001));
            }
        }
        let mut cities = 0u32;
        for c in state.tribes.values().flat_map(|t| t.cities.iter()) {
            if is_explored(state, c.idx, observer) {
                cities += 1;
                hash = hash.wrapping_add(mix(c.idx as u64 ^ 0x5A5A_0000_0000_0002));
            }
        }
        BeliefKey { explored, villages, cities, hash }
    }
}

/// Turn-scoped memo, mirroring `oracle_macro::GoalCache`.
#[derive(Default, Clone, Debug)]
pub struct MapBeliefCache {
    key: Option<(PlayerId, BeliefKey)>,
    belief: Option<Arc<MapBelief>>,
}

impl MapBeliefCache {
    pub fn get(&mut self, state: &GameState, observer: PlayerId) -> Arc<MapBelief> {
        let key = (observer, MapBelief::key_of(state, observer));
        if self.key == Some(key) {
            if let Some(b) = &self.belief {
                return Arc::clone(b);
            }
        }
        let b = Arc::new(MapBelief::observe(state, observer));
        self.key = Some(key);
        self.belief = Some(Arc::clone(&b));
        b
    }
}

/// Per-thread memo for [`MapBelief::observe`], so repeated `guess_villages`
/// calls inside one ply do not re-derive the same belief.
///
/// Sound because the belief is a pure function of `(explored set, observer)` and
/// [`BeliefKey`] fingerprints exactly that — the same key cannot describe two
/// different beliefs. Thread-local, so no lock and no cross-actor sharing;
/// self-play workers stay on one game at a time, so the hit rate is high. Four
/// entries covers both seats plus churn.
mod memo {
    use super::super::belief::MapBelief;
    use super::BeliefKey;
    use crate::states::{GameState, PlayerId};
    use std::cell::RefCell;
    use std::sync::Arc;

    const CAP: usize = 4;

    thread_local! {
        static CACHE: RefCell<Vec<(PlayerId, BeliefKey, Arc<MapBelief>)>> =
            RefCell::new(Vec::with_capacity(CAP));
    }

    pub fn observe(state: &GameState, observer: PlayerId) -> Arc<MapBelief> {
        let key = MapBelief::key_of(state, observer);
        if let Some(hit) = CACHE.with(|c| {
            c.borrow()
                .iter()
                .find(|(p, k, _)| *p == observer && *k == key)
                .map(|(_, _, b)| Arc::clone(b))
        }) {
            return hit;
        }
        let belief = Arc::new(MapBelief::observe(state, observer));
        CACHE.with(|c| {
            let mut c = c.borrow_mut();
            c.retain(|(p, _, _)| *p != observer);
            if c.len() >= CAP {
                c.pop();
            }
            c.insert(0, (observer, key, Arc::clone(&belief)));
        });
        belief
    }
}

/// Memoized [`MapBelief::observe`] — see the `memo` module below.
///
/// ⚠️ Measured **counterproductive on the search hot path**: simulated captures
/// move [`BeliefKey`] on every tree node, so it misses almost every time and
/// pays two key hashes per miss (-42% self-play throughput vs -17% for a plain
/// derivation). Use only where the state is stable across calls.
pub fn observe_cached(state: &GameState, observer: PlayerId) -> Arc<MapBelief> {
    memo::observe(state, observer)
}
