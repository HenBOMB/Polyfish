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

    // Fill map with fields
    for i in 0..121 {
        let mut tile = TileState::default();
        tile.terrain_type = TerrainType::Field;
        state.tiles.insert(i, tile);
    }

    state
}

#[test]
fn test_explorer_tech_mountain() {
    let mut state = setup_basic_state();
    let tribe_id = 1;

    // Put a mountain at (1,0), (1,1), (0,1) to block all paths from (0,0)
    state.tiles.get_mut(&1).unwrap().terrain_type = TerrainType::Mountain;
    state.tiles.get_mut(&11).unwrap().terrain_type = TerrainType::Mountain;
    state.tiles.get_mut(&12).unwrap().terrain_type = TerrainType::Mountain;

    // Explorer at (0,0) - index 0
    // Visible: only (0,0)
    state.tiles.get_mut(&0).unwrap().explorers.insert(tribe_id);

    // Fog at (3,0) - index 3
    // Explorer should want to go to index 3, but all paths are mountains.

    let (_, revealed) = predict_explorer(&state, 0);
    // Without climbing, it should NOT have revealed anything from index 1 (mountain)
    assert!(
        !revealed.contains(&2) && !revealed.contains(&3),
        "Explorer should not reveal blocked fog without Climbing"
    );

    // Now give climbing
    state
        .tribes
        .get_mut(&tribe_id)
        .unwrap()
        .tech_vanilla
        .push(polyfish::states::TechnologyState {
            tech_type: TechnologyType::Climbing,
            discovered: true,
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

    // at (0,0) index 0
    state.tiles.get_mut(&0).unwrap().explorers.insert(1);

    // Fog tiles: (1,0) and (0,1)
    // Put lighthouse at (2,0) - index 2 (revealed by moving to 1,0)

    let mut s = StructureState::default();
    s.structure_type = StructureType::Lighthouse;
    s.tile_index = 2;
    state.structures.insert(2, Some(s));

    let (_, revealed) = predict_explorer(&state, 0);

    assert!(
        revealed.contains(&2),
        "Explorer should prioritize path with lighthouse"
    );
}

#[test]
fn test_explorer_no_backtracking() {
    let mut state = setup_basic_state();

    // at (1,0). (0,0) is visible. (2,0) is fog.
    state.tiles.get_mut(&0).unwrap().explorers.insert(1);
    state.tiles.get_mut(&1).unwrap().explorers.insert(1);

    // explorer at 1. neighbors are 0, 2, 11, 12, 13.
    // Block 11, 12, 13 to make it deterministic
    state.tiles.get_mut(&11).unwrap().terrain_type = TerrainType::Mountain;
    state.tiles.get_mut(&12).unwrap().terrain_type = TerrainType::Mountain;
    state.tiles.get_mut(&13).unwrap().terrain_type = TerrainType::Mountain;

    let (_, revealed) = predict_explorer(&state, 1);
    assert!(
        revealed.contains(&3),
        "Explorer should avoid backtracking (0) and move to 2, revealing 3"
    );
}
