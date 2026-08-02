//! Real-game rule: a Farm built next to an existing Windmill retroactively
//! feeds the windmill's city (same for LumberHut→Sawmill, Mine→Forge).

use polyfish::actions::structure::build_structure;
use polyfish::coords::Coords;
use polyfish::states::{CityState, GameState, StructureState, TileState, TribeState};
use polyfish::types::StructureType;

fn ruled_tile(state: &mut GameState, idx: i32) {
    let tile = state.tiles.entry(idx).or_insert_with(TileState::default);
    tile.owner = 1;
    tile.ruling_city_coords = Some(Coords { x: 5, y: 5, idx: 60 });
}

fn setup() -> GameState {
    let mut state = GameState::default();
    state.settings.current_player_turn_id = 1;
    state.settings.size = 11;
    let mut tribe = TribeState::default();
    tribe.id = 1;
    tribe.stars = 50;
    tribe.cities.push(CityState {
        idx: 60,
        owner: 1,
        ..Default::default()
    });
    state.tribes.insert(1, tribe);
    for idx in [48, 58, 59] {
        ruled_tile(&mut state, idx);
    }
    state
}

fn pop(state: &GameState) -> i32 {
    state.tribes.get(&1).unwrap().cities[0].population
}

#[test]
fn farm_built_after_windmill_feeds_it_retroactively() {
    let mut state = setup();
    state.structures.insert(
        58,
        Some(StructureState {
            structure_type: StructureType::Farm,
            ..Default::default()
        }),
    );

    // Windmill at 59 with one existing farm (58): pays 1 pop at build time.
    let before = pop(&state);
    let _ = build_structure(&mut state, 59, StructureType::Windmill);
    assert_eq!(pop(&state) - before, 1);

    // A second farm at 48 (adjacent to the windmill): its own 2 pop PLUS
    // the windmill's retroactive +1.
    let before = pop(&state);
    let undo = build_structure(&mut state, 48, StructureType::Farm);
    assert_eq!(pop(&state) - before, 3);

    // Undo restores both the farm's own pop and the retroactive share.
    undo(&mut state);
    assert_eq!(pop(&state), before);
}

#[test]
fn non_partner_neighbor_pays_nothing_retroactively() {
    let mut state = setup();
    state.structures.insert(
        59,
        Some(StructureState {
            structure_type: StructureType::Windmill,
            ..Default::default()
        }),
    );
    // A LumberHut next to a windmill is not a windmill partner: only its
    // own 1 pop lands.
    let before = pop(&state);
    let _ = build_structure(&mut state, 48, StructureType::LumberHut);
    assert_eq!(pop(&state) - before, 1);
}
