//! The map generator's placement rules, restated as predicates the belief
//! can evaluate, plus the shared definition of what the observer knows.

use crate::states::{GameState, PlayerId};
use crate::types::{MapType, StructureType};

/// C1 (maximality) only holds where the generator fills villages to saturation:
/// the quadrant maps' post-terrain loop. Lakes/Archipelago also run a
/// *pre*-terrain fixed-count pass and Continents is per-landmass, so neither
/// saturates — but the post-terrain loop still runs for Lakes/Archipelago and
/// tops them up to saturation, which is what the invariant needs.
pub fn c1_applies(map_type: MapType) -> bool {
    matches!(
        map_type,
        MapType::Drylands | MapType::Lakes | MapType::Archipelago | MapType::WaterWorld
    )
}

/// The generator's edge rule: `edge_dist >= 2 && edge_dist != 3`.
pub fn edge_legal(idx: i32, size: i32) -> bool {
    if size <= 0 || idx < 0 || idx >= size * size {
        return false;
    }
    let (x, y) = (idx % size, idx / size);
    let edge_dist = x.min(size - 1 - x).min(y.min(size - 1 - y));
    edge_dist >= 2 && edge_dist != 3
}

/// Has `observer` revealed this tile? The fog gate for the paths that run
/// before a [`Ctx`](super::ctx::Ctx) exists; everything inside the derivation
/// goes through that instead.
pub(super) fn is_explored(state: &GameState, idx: i32, observer: PlayerId) -> bool {
    state
        .tiles
        .get(&idx)
        .map_or(false, |t| t.explorers.contains(&observer))
}

/// Village sites the observer knows about: explored Village structures plus
/// every explored city (capitals and captured villages are spacing sources in
/// the generator too — `village_map[cap] = 2`).
///
/// This is the ONE definition. Exclusion, C1/C2 discharge and spacing must all
/// use the same set; if they diverge, C1 emits constraints the generator never
/// made.
pub fn known_sites(state: &GameState, observer: PlayerId) -> Vec<i32> {
    let mut sites: Vec<i32> = state
        .structures
        .iter()
        .filter(|(idx, s)| {
            s.as_ref()
                .map_or(false, |s| s.structure_type == StructureType::Village)
                && is_explored(state, **idx, observer)
        })
        .map(|(idx, _)| *idx)
        .collect();
    for t in state.tribes.values() {
        sites.extend(
            t.cities
                .iter()
                .map(|c| c.idx)
                .filter(|&i| is_explored(state, i, observer)),
        );
    }
    sites.sort_unstable();
    sites.dedup();
    sites
}
