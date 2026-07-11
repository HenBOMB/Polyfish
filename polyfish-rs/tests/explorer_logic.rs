use polyfish::actions::discovery::predict_explorer;
use polyfish::states::{GameState, StructureState, TileState, TribeState};
use polyfish::types::{StructureType, TechnologyType, TerrainType, TribeType};

fn setup_basic_state() -> GameState {
    let mut state = GameState::default();
    state.settings.size = 11;
    let tribe_id = 1;
    state.settings.current_player_turn_id = tribe_id;

    let mut tribe = TribeState::default();
    tribe.id = tribe_id;
    tribe.tribe_type = TribeType::Imperius;
    state.tribes.insert(tribe_id, tribe);

    for i in 0..121 {
        let mut tile = TileState::default();
        tile.terrain_type = TerrainType::Field;
        state.tiles.insert(i, tile);
    }

    state
}

/// Mark a tile explored and impassible without the given tech.
fn explored_barrier(state: &mut GameState, idx: i32, tribe_id: i32, terrain: TerrainType) {
    let tile = state.tiles.get_mut(&idx).unwrap();
    tile.terrain_type = terrain;
    tile.explorers.insert(tribe_id);
}

/// Force the explorer east from (0,0): south is explored water, east is explored mountain.
fn force_east_corridor(state: &mut GameState, tribe_id: i32) {
    explored_barrier(state, 1, tribe_id, TerrainType::Mountain);
    explored_barrier(state, 11, tribe_id, TerrainType::Water);
}

#[test]
fn test_explorer_tech_mountain() {
    let mut state = setup_basic_state();
    let tribe_id = 1;

    force_east_corridor(&mut state, tribe_id);
    explored_barrier(&mut state, 12, tribe_id, TerrainType::Mountain);

    state.tiles.get_mut(&0).unwrap().explorers.insert(tribe_id);

    let (_, revealed) = predict_explorer(&state, 0);
    assert!(
        !revealed.contains(&2) && !revealed.contains(&3),
        "Explorer should not reveal blocked fog without Climbing"
    );

    state
        .tribes
        .get_mut(&tribe_id)
        .unwrap()
        .tech_vanilla
        .push(polyfish::states::TechnologyState {
            tech_type: TechnologyType::Climbing,
            discovered: true,
            discovered_turn: 0,
        });
    let (_, revealed_with_tech) = predict_explorer(&state, 0);
    assert!(
        revealed_with_tech.contains(&3),
        "Explorer SHOULD reach distance 3 with Climbing"
    );
}

#[test]
fn test_explorer_lighthouse_priority() {
    let mut state = setup_basic_state();
    state.settings.version = 115;

    force_east_corridor(&mut state, 1);
    state.tiles.get_mut(&0).unwrap().explorers.insert(1);

    let mut s = StructureState::default();
    s.structure_type = StructureType::Lighthouse;
    state.structures.insert(2, Some(s));

    state
        .tribes
        .get_mut(&1)
        .unwrap()
        .tech_vanilla
        .push(polyfish::states::TechnologyState {
            tech_type: TechnologyType::Climbing,
            discovered: true,
            discovered_turn: 0,
        });

    let (_, revealed) = predict_explorer(&state, 0);

    assert!(
        revealed.contains(&2),
        "Explorer should prioritize path with lighthouse"
    );
}

#[test]
fn test_explorer_no_backtracking() {
    let mut state = setup_basic_state();

    state.tiles.get_mut(&0).unwrap().explorers.insert(1);
    state.tiles.get_mut(&1).unwrap().explorers.insert(1);

    state.tiles.get_mut(&11).unwrap().terrain_type = TerrainType::Mountain;
    state.tiles.get_mut(&12).unwrap().terrain_type = TerrainType::Mountain;
    state.tiles.get_mut(&13).unwrap().terrain_type = TerrainType::Mountain;

    let (_, revealed) = predict_explorer(&state, 1);
    assert!(
        revealed.contains(&3),
        "Explorer should avoid backtracking (0) and move to 2, revealing 3"
    );
}
