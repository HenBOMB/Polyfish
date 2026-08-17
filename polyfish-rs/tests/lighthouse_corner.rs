use polyfish::actions::discovery::discover_tiles;
use polyfish::actions::update_exploration;
use polyfish::mapgen::{MapGenSettings, generate};
use polyfish::states::{CityState, GameState, StructureState, TileState, TribeState};
use polyfish::types::{MapSize, MapType, StructureType, TribeType};

/// Minimal state: one tribe, one capital city at tile 60, 11x11 map,
/// version 114 (the lighthouse-pop gate), a real Lighthouse structure at
/// tile 0 (a real map corner) matching what mapgen always places there,
/// left unexplored so a test can discover it.
fn setup_with_capital() -> GameState {
    let mut state = GameState::default();
    state.settings.current_player_turn_id = 1;
    state.settings.size = 11;
    state.settings.version = 114;
    let mut tribe = TribeState::default();
    tribe.id = 1;
    tribe.cities.push(CityState { idx: 60, owner: 1, ..Default::default() });
    state.tribes.insert(1, tribe);
    let mut capital_tile = TileState::default();
    capital_tile.capital_of = 1;
    state.tiles.insert(60, capital_tile);
    state.tiles.insert(0, TileState::default());
    state.structures.insert(
        0,
        Some(StructureState { structure_type: StructureType::Lighthouse, ..Default::default() }),
    );
    state
}

fn pop(state: &GameState) -> i32 {
    state.tribes.get(&1).unwrap().cities[0].population
}

/// The bug this guards: `discover_tiles`'s FOW-honest simulated branch never
/// looked at `state.structures`, so a lighthouse's +1 pop (real branch,
/// `!simulating`) was invisible to MCTS search no matter how deep it looked
/// — not an exploration-budget problem, a representation gap. Lighthouses
/// are the one exception: mapgen always places one on every map corner
/// (`is_lighthouse_corner`), a known rule, not hidden state, so simulation
/// may credit it without peeking at anything else under fog.
#[test]
fn simulated_discovery_of_a_corner_grants_pop_same_as_real() {
    let mut real = setup_with_capital();
    real.settings._are_you_sure = true;
    let before = pop(&real);
    let _ = discover_tiles(&mut real, 1, None, Some(vec![0]));
    assert_eq!(pop(&real) - before, 1, "real discovery of a corner must grant +1 pop");

    let mut sim = setup_with_capital();
    sim.settings._are_you_sure = false;
    let before = pop(&sim);
    let undo = discover_tiles(&mut sim, 1, None, Some(vec![0]));
    assert_eq!(
        pop(&sim) - before,
        1,
        "simulated discovery of a corner must ALSO grant +1 pop — this is the fix"
    );
    undo(&mut sim);
    assert_eq!(pop(&sim), before, "undo must restore the population");
}

/// A non-corner tile grants no pop even in the real branch (no lighthouse
/// there), so simulation shouldn't either — the corner check must not
/// over-fire.
#[test]
fn simulated_discovery_of_a_non_corner_grants_no_pop() {
    let mut state = setup_with_capital();
    state.settings._are_you_sure = false;
    // Tile 5: mid-edge, not a corner of an 11x11 map.
    state.tiles.insert(5, TileState::default());
    let before = pop(&state);
    let _ = discover_tiles(&mut state, 1, None, Some(vec![5]));
    assert_eq!(pop(&state), before, "a non-corner tile must not grant pop");
}

/// Below version 114 (pre-lighthouse-rule maps), corners aren't lighthouses
/// at all — `is_lighthouse_corner` must not fire regardless of position.
#[test]
fn simulated_discovery_of_a_corner_pre_v114_grants_no_pop() {
    let mut state = setup_with_capital();
    state.settings._are_you_sure = false;
    state.settings.version = 113;
    let before = pop(&state);
    let _ = discover_tiles(&mut state, 1, None, Some(vec![0]));
    assert_eq!(pop(&state), before, "pre-114 maps have no lighthouse-corner rule to credit");
}

#[test]
fn test_lighthouse_corner_placement() {
    let mut settings = MapGenSettings::default();
    settings.size = MapSize::Tiny; // 11x11
    settings.map_type = MapType::Drylands;
    settings.tribes = vec![TribeType::Imperius, TribeType::Bardur];
    settings.seed = 12345;

    let state = generate(settings);
    let size = state.settings.size;
    let corners = [0, size - 1, size * (size - 1), size * size - 1];

    for &idx in &corners {
        let structure = state
            .structures
            .get(&idx)
            .expect("Should have structure at corner");
        let s = structure.as_ref().expect("Should have structure data");
        assert_eq!(
            s.structure_type,
            StructureType::Lighthouse,
            "Corner at {} should be a lighthouse",
            idx
        );
    }
}

#[test]
fn test_lighthouse_corner_visibility_hidden() {
    let mut settings = MapGenSettings::default();
    settings.size = MapSize::Tiny; // 11x11
    settings.map_type = MapType::Drylands;
    settings.tribes = vec![TribeType::Imperius, TribeType::Bardur];
    settings.seed = 6789; // Use specific seed that places capital near edge but not on corner

    let mut state = generate(settings);
    let size = state.settings.size;
    let corners = [0, size - 1, size * (size - 1), size * size - 1];

    // Imperius is Tribe 1
    let _ = update_exploration(&mut state, 1);

    // Find where Tribe 1's city is
    let city_idx = state.tribes[&1].cities[0].idx;

    for &idx in &corners {
        if idx == city_idx {
            continue; // We expect to see our own city
        }

        // If the corner is within distance 2 of our city, it would normally be visible (5x5)
        let cx = city_idx % size;
        let cy = city_idx / size;
        let kx = idx % size;
        let ky = idx / size;
        let dist = (cx - kx).abs().max((cy - ky).abs());

        if dist <= 2 {
            // Check that corners are NOT explored (hidden due to corner rule)
            let is_explored = state
                .tiles
                .get(&idx)
                .map(|t| t.explorers.contains(&1))
                .unwrap_or(false);
            assert!(
                !is_explored,
                "Corner {} at distance {} should be hidden from initial city vision",
                idx, dist
            );
        }
    }
}
