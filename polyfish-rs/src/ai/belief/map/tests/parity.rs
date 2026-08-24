//! Stage-1a regression gate: the SSOT migration must not move production
//! expansion targeting a single bit until we intend it to.

use std::sync::Arc;

use super::fixtures::corpus;
use crate::ai::belief::map::{observe_cached, MapBelief};
use crate::ai::belief::prediction::VillageGuess;
use crate::functions::{get_adjacent_indices, get_chebyshev_distance, get_plus_sign_indices};
use crate::states::{GameState, PlayerId};
use crate::types::{StructureType, TerrainType, TribeType};

// ---------------------------------------------------------------
// FROZEN pre-migration implementation, copied verbatim from
// `prediction.rs` at commit 658b29c. This is a genuinely independent
// second implementation on purpose: comparing the live path against a
// function it delegates to would pass by construction and pin nothing.
// Do NOT "fix" the bugs in here — Stage 1b measures its delta against
// exactly this behaviour.
// ---------------------------------------------------------------

fn reference_validate(state: &GameState, idx: i32, known: &[i32]) -> bool {
    let size = state.settings.size;

    // 1. Cardinal Neighbor Rule: No Ocean neighbors
    let cardinals = get_plus_sign_indices(idx, size);
    for n_idx in cardinals {
        if let Some(tile) = state.tiles.get(&n_idx) {
            if tile.terrain_type == TerrainType::Ocean {
                return false;
            }
        }
    }

    // 2. Map Edge Rule: edge_dist >= 2 && edge_dist != 3
    let (x, y) = (idx % size, idx / size);
    let dist_x = x.min(size - 1 - x);
    let dist_y = y.min(size - 1 - y);
    let edge_dist = dist_x.min(dist_y);
    if edge_dist < 2 || edge_dist == 3 {
        return false;
    }

    // 3. Distance-3 Rule (Chebyshev) from every known village/capital
    known.iter().all(|&k| get_chebyshev_distance(idx, k, size) >= 3)
}

fn reference_impl(state: &GameState, player: PlayerId, max_sites: usize) -> Vec<VillageGuess> {
    let size = state.settings.size as i32;
    let Some(tribe) = state.tribes.get(&player) else {
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
    let cheb = |a: i32, b: i32| get_chebyshev_distance(a, b, size);
    let explored =
        |idx: i32| state.tiles.get(&idx).map_or(false, |t| t.explorers.contains(&player));

    // Known spacing sources: explored villages + explored cities (capitals
    // and captured villages count as villages in the generator's spacing).
    let mut known: Vec<i32> = state
        .structures
        .iter()
        .filter(|(idx, s)| {
            s.as_ref().map_or(false, |s| s.structure_type == StructureType::Village)
                && explored(**idx)
        })
        .map(|(idx, _)| *idx)
        .collect();
    for t in state.tribes.values() {
        known.extend(t.cities.iter().map(|c| c.idx).filter(|&i| explored(i)));
    }

    // --- Selection: generator geometry alone (unchanged from the old
    // guessed_village_sites) ---
    let mut cands: Vec<(i32, i32)> = (0..size * size)
        .filter(|&idx| {
            !explored(idx) && reference_validate(state, idx, &known)
        })
        .map(|idx| {
            let d = anchors.iter().map(|&a| cheb(a, idx)).min().unwrap_or(i32::MAX);
            (d, idx)
        })
        .collect();
    cands.sort_unstable();
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
        for &(_, idx) in &cands {
            if picks.len() >= max_sites {
                break;
            }
            if picks.contains(&idx) || picks.iter().any(|&p| cheb(p, idx) < 3) {
                continue;
            }
            if pass == 0 && used_quads.contains(&quadrant(idx)) {
                continue;
            }
            used_quads.insert(quadrant(idx));
            picks.push(idx);
        }
    }

    // --- Confidence + tribe: resource/climate evidence near each pick ---
    let pov_climate = crate::ai::belief::prediction::tribe_to_climate(tribe.tribe_type);
    let is_orphan = |res_idx: i32| known.iter().all(|&k| cheb(res_idx, k) > 1);
    picks
        .into_iter()
        .map(|site| {
            let mut score = 0i32;
            let mut climate_evidence = 0i32;
            for n in get_adjacent_indices(state, site, 1) {
                if explored(n)
                    && is_orphan(n)
                    && matches!(state.resources.get(&n), Some(Some(_)))
                {
                    score += 5;
                }
            }
            let res_neighbors = get_adjacent_indices(state, site, 1)
                .into_iter()
                .filter(|&n| matches!(state.resources.get(&n), Some(Some(_))))
                .count();
            if res_neighbors >= 2 {
                score += 10;
            }
            for n in get_adjacent_indices(state, site, 2) {
                if let Some(t) = state.tiles.get(&n) {
                    if explored(n) && t.owner != player && t.climate != pov_climate && t.climate != 0
                    {
                        score += 1;
                        climate_evidence = t.climate;
                    }
                }
            }
            let mut guessed_tribe =
                (climate_evidence != 0).then(|| crate::ai::belief::prediction::climate_to_tribe(climate_evidence));
            if guessed_tribe == Some(TribeType::Bardur) {
                let crop_nearby = get_adjacent_indices(state, site, 1).into_iter().any(|n| {
                    matches!(
                        state.resources.get(&n),
                        Some(Some(r)) if r.resource_type == crate::types::ResourceType::Crop
                    )
                });
                if crop_nearby {
                    score -= 20;
                    guessed_tribe = None;
                }
            }
            VillageGuess {
                tile: site,
                tribe: guessed_tribe,
                // Purely geometric picks still carry a real (if modest) floor —
                // the generator-saturation reasoning alone is solid evidence.
                confidence: (0.3 + score as f32 / 20.0).clamp(0.05, 1.0),
            }
        })
        .collect()
}

#[test]
fn guess_villages_parity_holds_on_a_state_corpus() {
    let states = corpus(7_700_000..7_700_040, 4);
    assert!(states.len() > 40, "thin corpus: {}", states.len());

    let mut compared = 0usize;
    let mut non_empty = 0usize;
    for (n, state) in states.iter().enumerate() {
        for pov in [1, 2] {
            for max_sites in [1usize, 2, 5] {
                let reference = reference_impl(state, pov, max_sites);
                let through_ssot = MapBelief::observe(state, pov)
                    .top_village_sites_legacy(state, max_sites);
                assert_eq!(
                    reference.len(),
                    through_ssot.len(),
                    "state {n} pov {pov} max {max_sites}: length differs"
                );
                for (a, b) in reference.iter().zip(through_ssot.iter()) {
                    assert_eq!(a.tile, b.tile, "state {n} pov {pov}: tile differs");
                    assert_eq!(a.tribe, b.tribe, "state {n} pov {pov}: tribe differs");
                    assert_eq!(
                        a.confidence.to_bits(),
                        b.confidence.to_bits(),
                        "state {n} pov {pov}: confidence differs at tile {}",
                        a.tile
                    );
                }
                compared += 1;
                if !reference.is_empty() {
                    non_empty += 1;
                }
            }
        }
    }
    // A corpus that produced no guesses at all would pass vacuously.
    assert!(
        non_empty * 4 > compared,
        "corpus mostly produced empty guesses ({non_empty}/{compared}) - \
         parity would be vacuous"
    );
    eprintln!("parity: {compared} comparisons, {non_empty} non-empty");
}

/// Routing check: `guess_villages` dispatches to the belief path
/// (EXP_ELO_070). Byte-parity of the frozen legacy path is the test above.
#[test]
fn guess_villages_entry_point_uses_the_belief_path() {
    for state in corpus(7_800_000..7_800_010, 3) {
        for pov in [1, 2] {
            let live = crate::ai::belief::prediction::guess_villages(&state, pov, 2);
            let belief = MapBelief::observe(&state, pov).top_village_sites(&state, 2);
            assert_eq!(live.len(), belief.len(), "entry point is not the belief path");
            for (x, y) in live.iter().zip(belief.iter()) {
                assert_eq!(x.tile, y.tile, "entry point is not the belief path");
                assert_eq!(x.confidence.to_bits(), y.confidence.to_bits());
            }
        }
    }
}

/// The memo must never change an answer — only how fast it arrives.
#[test]
fn thread_local_memo_agrees_with_a_fresh_derivation() {
    for state in corpus(7_810_000..7_810_008, 3) {
        for pov in [1, 2] {
            let fresh = MapBelief::observe(&state, pov);
            // Called twice: first populates the memo, second must hit it.
            let a = observe_cached(&state, pov);
            let b = observe_cached(&state, pov);
            assert!(Arc::ptr_eq(&a, &b), "memo missed on an unchanged state");
            for i in 0..state.settings.size * state.settings.size {
                assert_eq!(fresh.p_village(i).to_bits(), a.p_village(i).to_bits());
                assert_eq!(fresh.p_capital(i).to_bits(), a.p_capital(i).to_bits());
            }
        }
    }
}

/// Picks must be ordered by distance to our units, not by confidence —
/// the whole point of EXP_ELO_070.
#[test]
fn picks_are_ordered_by_distance_not_confidence() {
    let mut checked = 0;
    for state in corpus(7_820_000..7_820_012, 3) {
        for pov in [1, 2] {
            let belief = MapBelief::observe(&state, pov);
            let picks = belief.top_village_sites(&state, 3);
            if picks.len() < 2 {
                continue;
            }
            let Some(tribe) = state.tribes.get(&pov) else { continue };
            let anchors: Vec<i32> = if tribe.units.is_empty() {
                tribe.cities.iter().map(|c| c.idx).collect()
            } else {
                tribe.units.iter().map(|u| u.coords.idx).collect()
            };
            if anchors.is_empty() {
                continue;
            }
            let size = state.settings.size;
            let d = |i: i32| {
                anchors
                    .iter()
                    .map(|&a| get_chebyshev_distance(a, i, size))
                    .min()
                    .unwrap()
            };
            // The quadrant-spread pass can legitimately reorder, so assert
            // the weaker invariant that actually matters: the FIRST pick is
            // the nearest of everything returned.
            let first = d(picks[0].tile);
            assert!(
                picks.iter().all(|g| d(g.tile) >= first),
                "first pick is not the nearest: {:?}",
                picks.iter().map(|g| (g.tile, d(g.tile))).collect::<Vec<_>>()
            );
            checked += 1;
        }
    }
    assert!(checked > 5, "too few multi-pick states to be meaningful: {checked}");
}

/// The belief-ranked path must still DIFFER from legacy — otherwise
/// EXP_ELO_070's planned re-ranking has nothing to work with.
#[test]
fn belief_ranked_path_still_diverges_from_legacy() {
    let mut differed = 0;
    for state in corpus(7_800_000..7_800_010, 3) {
        for pov in [1, 2] {
            let belief = MapBelief::observe(&state, pov).top_village_sites(&state, 2);
            let frozen = reference_impl(&state, pov, 2);
            if belief.iter().map(|g| g.tile).ne(frozen.iter().map(|g| g.tile)) {
                differed += 1;
            }
        }
    }
    assert!(differed > 0, "belief ranking is indistinguishable from legacy");
}
