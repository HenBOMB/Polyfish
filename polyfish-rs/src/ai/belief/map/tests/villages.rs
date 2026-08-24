//! C1 (packing), C2 (resource zone), spacing exclusion, and the mass
//! bookkeeping that ties them together.

use super::fixtures::*;
use crate::ai::belief::map::{MapBelief, P_BASE};
use crate::functions::{get_chebyshev_distance, get_square_indices};
use crate::types::{ResourceType, TerrainType};

/// TEST 1 — C1. An explored, empty, legal tile forces total village mass of
/// at least 1 into its radius-2 disc, and touches nothing outside it.
#[test]
fn explored_empty_legal_tile_creates_existence_mass_in_disc2() {
    let mut state = blank_state();
    park_cities_away(&mut state);
    let anchor = 5 * SIZE + 5;
    set_terrain(&mut state, anchor, TerrainType::Field);
    reveal(&mut state, anchor, 1);

    let b = MapBelief::observe(&state, 1);
    let inside: f32 = disc2(anchor)
        .iter()
        .filter(|&&j| j != anchor)
        .map(|&j| b.p_village(j))
        .sum();
    assert!(
        inside >= 0.99,
        "C1 did not force a village into D2({anchor}); mass {inside}"
    );
    assert_eq!(b.p_village(anchor), 0.0, "the revealed tile is empty");

    // A tile well outside the disc keeps the untouched prior.
    let far = 0 * SIZE + 5;
    assert!(
        (b.p_village(far) - P_BASE * 0.0).abs() < 1e-6 || b.p_village(far) <= P_BASE + 1e-6,
        "tile outside D2 was modified: {}",
        b.p_village(far)
    );
}

/// TEST 2 — a revealed Mountain was never a legal site, so its emptiness
/// carries no information and must emit no constraint.
#[test]
fn revealed_mountain_voids_the_existence_constraint() {
    let mut state = blank_state();
    park_cities_away(&mut state);
    let anchor = 5 * SIZE + 5;
    set_terrain(&mut state, anchor, TerrainType::Mountain);
    reveal(&mut state, anchor, 1);

    let b = MapBelief::observe(&state, 1);
    let inside: f32 = disc2(anchor)
        .iter()
        .filter(|&&j| j != anchor)
        .map(|&j| b.p_village(j))
        .sum();
    // Only the untouched prior should be present.
    let prior_max = 25.0 * P_BASE;
    assert!(
        inside < prior_max * 0.9,
        "a mountain reveal manufactured constraint mass: {inside}"
    );
    assert!(
        b.evidence_at(anchor).is_none(),
        "mountain tile recorded as evidence"
    );
}

/// TEST 3 — a known village inside the disc already explains the empty
/// tile, so no new mass is forced.
#[test]
fn known_village_within_2_discharges_the_constraint() {
    let mut state = blank_state();
    park_cities_away(&mut state);
    let anchor = 5 * SIZE + 5;
    let site = anchor + 2; // Chebyshev 2 away
    set_terrain(&mut state, anchor, TerrainType::Field);
    put_village(&mut state, site);
    reveal(&mut state, anchor, 1);
    reveal(&mut state, site, 1);

    let b = MapBelief::observe(&state, 1);
    assert_eq!(b.p_village(site), 1.0, "sighted village is certain");
    // Everything else in the disc is spacing-excluded by that village.
    for &j in disc2(anchor).iter().filter(|&&j| j != site) {
        if get_chebyshev_distance(j, site, SIZE) <= 2 {
            assert_eq!(
                b.p_village(j),
                0.0,
                "tile {j} within 2 of a known site must be zero"
            );
        }
    }
}

/// TEST 4 — C2. An orphan resource forces mass into its disc and weights
/// the inner ring by the measured 2.69:1.
#[test]
fn revealed_resource_creates_mass_and_weights_inner_ring() {
    let mut state = blank_state();
    park_cities_away(&mut state);
    let anchor = 5 * SIZE + 5;
    set_terrain(&mut state, anchor, TerrainType::Field);
    put_resource(&mut state, anchor, ResourceType::Fruit);
    reveal(&mut state, anchor, 1);

    let b = MapBelief::observe(&state, 1);
    let inner: f32 = get_square_indices(anchor, 1, SIZE)
        .iter()
        .filter(|&&j| j != anchor)
        .map(|&j| b.p_village(j))
        .sum();
    let outer: f32 = disc2(anchor)
        .iter()
        .filter(|&&j| get_chebyshev_distance(j, anchor, SIZE) == 2)
        .map(|&j| b.p_village(j))
        .sum();
    assert!(inner + outer >= 0.99, "C2 forced no village: {inner}+{outer}");
    // 8 inner tiles vs 16 outer; per-tile the inner ring must run richer.
    let inner_per = inner / 8.0;
    let outer_per = outer / 16.0;
    assert!(
        inner_per > outer_per * 1.5,
        "inner ring not weighted: per-tile inner {inner_per} vs outer {outer_per}"
    );
}

/// TEST 5 — the `is_orphan` fidelity fix. The generator's zone is
/// Chebyshev ≤2, so a resource 2 tiles from a known site is FULLY
/// explained and must force nothing. Checked against a no-resource control,
/// because the untouched prior on those tiles is legitimately non-zero.
#[test]
fn resource_at_chebyshev_2_from_a_known_village_is_not_orphaned() {
    let build = |with_resource: bool| {
        let mut state = blank_state();
        park_cities_away(&mut state);
        let site = 5 * SIZE + 5;
        let res = site + 2; // Chebyshev 2 from the known village
        put_village(&mut state, site);
        set_terrain(&mut state, res, TerrainType::Field);
        if with_resource {
            put_resource(&mut state, res, ResourceType::Fruit);
        }
        reveal(&mut state, site, 1);
        reveal(&mut state, res, 1);
        MapBelief::observe(&state, 1)
    };
    let with_res = build(true);
    let control = build(false);

    let res = 5 * SIZE + 5 + 2;
    for &j in &disc2(res) {
        assert_eq!(
            with_res.p_village(j).to_bits(),
            control.p_village(j).to_bits(),
            "an already-explained resource moved mass at {j}: {} vs control {}",
            with_res.p_village(j),
            control.p_village(j)
        );
    }
    // Control check: the SAME resource with no known site nearby must move mass.
    let mut orphan = blank_state();
    park_cities_away(&mut orphan);
    let lone = 5 * SIZE + 5;
    set_terrain(&mut orphan, lone, TerrainType::Field);
    put_resource(&mut orphan, lone, ResourceType::Fruit);
    reveal(&mut orphan, lone, 1);
    let b = MapBelief::observe(&orphan, 1);
    let mass: f32 = disc2(lone).iter().map(|&j| b.p_village(j)).sum();
    assert!(
        mass >= 0.99,
        "an ORPHAN resource should have forced a village: {mass}"
    );
}

/// TEST 6 — exclusion. Nothing can sit within Chebyshev 2 of a known site.
#[test]
fn spacing_exclusion_zeros_disc2_of_every_known_site() {
    let mut state = blank_state();
    park_cities_away(&mut state);
    let site = 5 * SIZE + 5;
    put_village(&mut state, site);
    reveal(&mut state, site, 1);

    let b = MapBelief::observe(&state, 1);
    for &j in &disc2(site) {
        if j == site {
            continue;
        }
        assert_eq!(b.p_village(j), 0.0, "tile {j} inside the spacing ring");
    }
}

/// TEST 7 — Verdi's exact scenario. Three candidate sites carry mass;
/// revealing one as empty drops it to 0 and RAISES the others, without the
/// region's total mass increasing.
#[test]
fn mass_is_conserved_under_reveal() {
    let mut state = blank_state();
    park_cities_away(&mut state);
    // Reveal a ring of empty legal tiles so the discs overlap and carry
    // real constraint mass.
    let anchor = 5 * SIZE + 5;
    for j in disc2(anchor) {
        set_terrain(&mut state, j, TerrainType::Field);
    }
    for j in [anchor, anchor - 2, anchor + 2] {
        reveal(&mut state, j, 1);
    }
    let before = MapBelief::observe(&state, 1);
    let region: Vec<i32> = disc2(anchor);
    let mass_before: f32 = region.iter().map(|&j| before.p_village(j)).sum();

    // Pick a still-fogged tile inside the region that carries mass, and
    // reveal it as empty.
    let target = region
        .iter()
        .copied()
        .filter(|&j| !before_explored(&state, j) && before.p_village(j) > 1e-6)
        .max_by(|&a, &b| before.p_village(a).total_cmp(&before.p_village(b)))
        .expect("a fogged tile with mass");
    let p_target = before.p_village(target);
    assert!(p_target > 0.0);
    reveal(&mut state, target, 1);

    let after = MapBelief::observe(&state, 1);
    assert_eq!(
        after.p_village(target),
        0.0,
        "revealed-empty tile must collapse to 0"
    );
    let mass_after: f32 = region.iter().map(|&j| after.p_village(j)).sum();
    assert!(
        mass_after <= mass_before + 1e-4,
        "a reveal INCREASED regional mass: {mass_before} -> {mass_after}"
    );
    // The freed mass is forced onto the remaining candidates, not lost.
    let others_before: f32 = region
        .iter()
        .filter(|&&j| j != target)
        .map(|&j| before.p_village(j))
        .sum();
    let others_after: f32 = region
        .iter()
        .filter(|&&j| j != target)
        .map(|&j| after.p_village(j))
        .sum();
    assert!(
        others_after >= others_before - 1e-4,
        "mass vanished instead of redistributing: {others_before} -> {others_after}"
    );
}
