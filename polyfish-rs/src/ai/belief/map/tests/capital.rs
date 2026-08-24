//! The capital posterior: C3 climate evidence, sighting collapse, and the
//! two-sided probe that climate is never read through fog.

use super::fixtures::*;
use crate::ai::belief::map::MapBelief;
use crate::ai::belief::{capital_support_by_quad, quad_of};
use crate::functions::{get_chebyshev_distance, get_square_indices};
use crate::types::TribeType;

/// TEST 8 — C3. Seeing the OPPONENT's climate shifts the capital posterior
/// toward the support cell nearest that tile; seeing OUR OWN climate shifts
/// it away (the negative control).
///
/// The witness is a plain fog tile NEXT TO a support cell, never a support
/// cell itself — revealing one of those eliminates it outright and would
/// test elimination rather than the C3 likelihood.
#[test]
fn climate_evidence_moves_capital_posterior_toward_the_nearer_support_cell() {
    let base = blank_state();
    let own_cap = base.tribes[&1].starting_tile_coords.idx;
    let quads = capital_support_by_quad(SIZE, 2);
    let own_quad = quad_of(own_cap, &quads);
    let support: Vec<i32> = quads
        .iter()
        .enumerate()
        .filter(|(q, _)| Some(*q) != own_quad)
        .flat_map(|(_, c)| c.iter().copied())
        .collect();
    assert!(support.len() >= 2, "need a multi-cell support: {support:?}");

    let favored = support[0];
    let disfavored = support
        .iter()
        .copied()
        .skip(1)
        .max_by_key(|&c| get_chebyshev_distance(c, favored, SIZE))
        .expect("a second support cell");

    // A fog tile adjacent to `favored`, and strictly nearer to it than to
    // `disfavored`, that is not itself a hypothesis.
    let witness = get_square_indices(favored, 1, SIZE)
        .into_iter()
        .filter(|&j| j != favored && j != own_cap && !support.contains(&j))
        .find(|&j| {
            get_chebyshev_distance(j, favored, SIZE)
                < get_chebyshev_distance(j, disfavored, SIZE)
        })
        .expect("a witness tile beside the favored support cell");

    let observe_with = |climate: i32| {
        let mut state = base.clone();
        if let Some(t) = state.tiles.get_mut(&witness) {
            t.climate = climate;
            t.explorers.insert(1);
        }
        MapBelief::observe(&state, 1)
    };
    let with_opp = observe_with(crate::types::classic_climate_id(TribeType::Bardur));
    let with_own = observe_with(crate::types::classic_climate_id(TribeType::Imperius));

    assert!(
        with_opp.p_capital(favored) > with_own.p_capital(favored),
        "opponent-climate evidence did not raise the nearer cell {favored}: \
         {} vs {}",
        with_opp.p_capital(favored),
        with_own.p_capital(favored)
    );
    // Negative control: our own climate pushes mass the other way.
    assert!(
        with_own.p_capital(disfavored) > with_opp.p_capital(disfavored),
        "own-climate evidence did not raise the farther cell {disfavored}: \
         {} vs {}",
        with_own.p_capital(disfavored),
        with_opp.p_capital(disfavored)
    );
    // Both stay normalized.
    for b in [&with_opp, &with_own] {
        let total: f32 = (0..SIZE * SIZE).map(|i| b.p_capital(i)).sum();
        assert!((total - 1.0).abs() < 1e-4, "posterior mass {total}");
    }
}

/// TEST 9 — a sighting collapses the posterior to a point mass, preserving
/// `BeliefState::on_explored`'s branch exactly.
#[test]
fn capital_sighting_collapses_to_a_point_mass() {
    let mut state = blank_state();
    let opp_cap = state.tribes[&2].starting_tile_coords.idx;
    reveal(&mut state, opp_cap, 1);

    let b = MapBelief::observe(&state, 1);
    assert_eq!(b.capital_confirmed, Some(opp_cap));
    assert_eq!(b.p_capital(opp_cap), 1.0);
    assert_eq!(b.capital_live(), 1);
    assert_eq!(b.capital_map(), Some(opp_cap));
    assert!((b.capital_confidence() - 1.0).abs() < 1e-6);
}

/// `obscure_fog` does NOT strip `tile.climate` (states.rs), so comparing a
/// true and an obscured view cannot detect a climate leak — both carry the
/// same fog climate. Scramble the climate on every UNEXPLORED tile instead:
/// a derivation that gates every climate read on `explorers` is bit-identical
/// either way, and one that peeks is not.
#[test]
fn climate_is_not_read_through_fog() {
    let mut state = blank_state();
    for t in [2 * SIZE + 2, 5 * SIZE + 5, 7 * SIZE + 4, 4 * SIZE + 8] {
        reveal(&mut state, t, 1);
    }
    let honest = MapBelief::observe(&state, 1);

    let mut scrambled = state.clone();
    let bogus = crate::types::classic_climate_id(TribeType::Vengir);
    let mut touched = 0;
    for (_, tile) in scrambled.tiles.iter_mut() {
        if !tile.explorers.contains(&1) {
            tile.climate = bogus;
            touched += 1;
        }
    }
    assert!(touched > 100, "scrambled too little to be a real probe: {touched}");
    let leaked = MapBelief::observe(&scrambled, 1);

    for i in 0..SIZE * SIZE {
        assert_eq!(
            honest.p_capital(i).to_bits(),
            leaked.p_capital(i).to_bits(),
            "capital posterior at {i} moved when FOGGED climate changed - \
             a climate read is leaking through fog"
        );
        assert_eq!(
            honest.p_opponent_affinity(i).to_bits(),
            leaked.p_opponent_affinity(i).to_bits(),
            "affinity at {i} moved when FOGGED climate changed"
        );
        assert_eq!(
            honest.p_village(i).to_bits(),
            leaked.p_village(i).to_bits(),
            "village mass at {i} moved when FOGGED climate changed"
        );
    }
}

/// The same probe from the other side: changing climate on an EXPLORED tile
/// MUST move the belief, or the test above would pass vacuously.
#[test]
fn explored_climate_does_move_the_belief() {
    let base = blank_state();
    let quads = capital_support_by_quad(SIZE, 2);
    let own_cap = base.tribes[&1].starting_tile_coords.idx;
    let support: Vec<i32> = quads
        .iter()
        .enumerate()
        .filter(|(q, _)| Some(*q) != quad_of(own_cap, &quads))
        .flat_map(|(_, c)| c.iter().copied())
        .collect();
    let witness = get_square_indices(support[0], 1, SIZE)
        .into_iter()
        .find(|&j| j != support[0] && j != own_cap && !support.contains(&j))
        .expect("a witness tile");

    let observe = |climate: i32| {
        let mut s = base.clone();
        if let Some(t) = s.tiles.get_mut(&witness) {
            t.climate = climate;
            t.explorers.insert(1);
        }
        MapBelief::observe(&s, 1)
    };
    let a = observe(crate::types::classic_climate_id(TribeType::Bardur));
    let b = observe(crate::types::classic_climate_id(TribeType::Imperius));
    let moved = (0..SIZE * SIZE).any(|i| a.p_capital(i) != b.p_capital(i));
    assert!(
        moved,
        "explored climate had no effect - the fog test above is vacuous"
    );
}
