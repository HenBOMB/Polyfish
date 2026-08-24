//! The recompute-anywhere contract: same explored set in, same belief out,
//! a key that moves exactly when the belief does, and the map-type gate.

use std::sync::Arc;

use super::fixtures::*;
use crate::ai::belief::map::{c1_applies, MapBelief, MapBeliefCache};
use crate::types::{MapSize, MapType, TribeType};

/// TEST 10 — THE licensing test for "no persistence, no serialization, no
/// `obscure_fog` handling": the belief is a pure function of the explored
/// SET, so two different reveal orders reaching the same set agree exactly.
#[test]
fn belief_is_a_pure_function_of_the_explored_set() {
    let tiles: Vec<i32> = vec![
        2 * SIZE + 2,
        5 * SIZE + 5,
        5 * SIZE + 6,
        7 * SIZE + 4,
        4 * SIZE + 8,
        6 * SIZE + 2,
    ];

    let mut a = blank_state();
    for &t in &tiles {
        reveal(&mut a, t, 1);
    }
    let mut b = blank_state();
    for &t in tiles.iter().rev() {
        reveal(&mut b, t, 1);
    }

    let ba = MapBelief::observe(&a, 1);
    let bb = MapBelief::observe(&b, 1);
    for i in 0..SIZE * SIZE {
        assert_eq!(
            ba.p_village(i).to_bits(),
            bb.p_village(i).to_bits(),
            "village mass differs at {i} by reveal order"
        );
        assert_eq!(
            ba.p_capital(i).to_bits(),
            bb.p_capital(i).to_bits(),
            "capital mass differs at {i} by reveal order"
        );
        assert_eq!(
            ba.p_opponent_affinity(i).to_bits(),
            bb.p_opponent_affinity(i).to_bits(),
            "affinity differs at {i} by reveal order"
        );
    }
    assert_eq!(ba.key(), bb.key());
}

/// TEST 11 — the search freeze. Obscuring a view and advancing the turn the
/// way a rollout does must not move the cache key, or a tree would
/// recompute (and drift) per node.
///
/// Uses a natural post-`post_load` state: re-running `post_load` here would
/// re-derive visibility from scratch and move the key for reasons that have
/// nothing to do with the turn advance.
#[test]
fn belief_key_is_stable_across_a_simulated_turn() {
    let mut game = crate::game::Game::new();
    game.state = crate::mapgen::generate(crate::mapgen::MapGenSettings {
        size: MapSize::Tiny,
        map_type: MapType::Drylands,
        tribes: vec![TribeType::Imperius, TribeType::Bardur],
        seed: 4242,
        ..Default::default()
    });
    game.post_load();
    let before = MapBelief::key_of(&game.state, 1);

    let mut view = game.state.clone();
    view.obscure_fog(1);
    view.settings.turn += 1;
    let after = MapBelief::key_of(&view, 1);
    assert_eq!(before, after, "a simulated turn advance invalidated the key");

    // And the belief itself is unchanged, not merely the key.
    let a = MapBelief::observe(&game.state, 1);
    let b = MapBelief::observe(&view, 1);
    for i in 0..SIZE * SIZE {
        assert_eq!(
            a.p_village(i).to_bits(),
            b.p_village(i).to_bits(),
            "village mass at {i} moved across a simulated turn"
        );
    }
}

/// The cache must return the SAME allocation until the key moves, and a
/// fresh belief once it does.
#[test]
fn cache_reuses_until_the_explored_set_changes() {
    let mut state = blank_state();
    reveal(&mut state, 5 * SIZE + 5, 1);
    let mut cache = MapBeliefCache::default();
    let a = cache.get(&state, 1);
    let b = cache.get(&state, 1);
    assert!(Arc::ptr_eq(&a, &b), "cache recomputed on an unchanged state");

    reveal(&mut state, 5 * SIZE + 7, 1);
    let c = cache.get(&state, 1);
    assert!(!Arc::ptr_eq(&a, &c), "cache served a stale belief");
}

/// Capturing a village turns it into a city: the explored count and the
/// villages+cities SUM are both unchanged, so a naive key would go stale.
#[test]
fn key_separates_villages_from_cities() {
    let mut state = blank_state();
    let site = 5 * SIZE + 5;
    put_village(&mut state, site);
    reveal(&mut state, site, 1);
    let before = MapBelief::key_of(&state, 1);

    // The capture: the Village structure goes away, a city appears there.
    state.structures.remove(&site);
    let mut city = crate::states::CityState::default();
    city.idx = site;
    city.owner = 1;
    state.tribes.get_mut(&1).unwrap().cities.push(city);
    let after = MapBelief::key_of(&state, 1);

    assert_ne!(
        before, after,
        "a village->city capture left the cache key unchanged"
    );
}

/// The map-type gate: C1 must not fire where the generator does not
/// saturate.
#[test]
fn c1_is_gated_on_map_type() {
    assert!(c1_applies(MapType::Drylands));
    assert!(!c1_applies(MapType::Continents));
    assert!(!c1_applies(MapType::Pangea));
}
