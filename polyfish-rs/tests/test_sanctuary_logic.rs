use polyfish::game::Game;
use polyfish::types::{StructureType, TechnologyType, TribeType, TerrainType, ResourceType};
use polyfish::moves::build::BuildMove;
use polyfish::moves::Move;
use polyfish::states::{TileState, TribeState, TechnologyState, CityState};
use polyfish::actions;

fn setup_elyrion_game() -> Game {
    let mut game = Game::new();
    let pov_id = 1;
    game.state.settings.current_player_turn_id = pov_id;
    game.state.settings.turn = 1;
    game.state.settings._max_tribe_count = 1;
    game.state.settings.size = 16;
    
    // Create tiles
    for i in 0..256 {
        game.state.tiles.insert(i as i32, TileState {
            coords: polyfish::coords::Coords::from_index(i as i32, 16),
            ..Default::default()
        });
    }
    
    // Create Elyrion tribe
    let mut tribe = TribeState::default();
    tribe.id = pov_id;
    tribe.tribe_type = TribeType::Elyrion;
    tribe.stars = 100;
    tribe.tech_vanilla.push(TechnologyState {
        tech_type: TechnologyType::Forestry,
        discovered: true,
        discovered_turn: 0,
    });
    tribe.tech_vanilla.push(TechnologyState {
        tech_type: TechnologyType::Hunting,
        discovered: true,
        discovered_turn: 0,
    });
    
    game.state.tribes.insert(pov_id, tribe);
    game
}

#[test]
fn test_sanctuary_placement_and_limit() {
    let mut game = setup_elyrion_game();
    let pov_id = game.current_player_id();
    
    // Setup a city
    let territory = vec![0, 1, 2, 3];
    if let Some(tribe) = game.state.tribes.get_mut(&pov_id) {
        tribe.cities.push(CityState {
            idx: 0,
            owner: pov_id,
            border_size: 1,
            _territory: territory.clone(),
            ..Default::default()
        });
    }

    for &idx in &territory {
        if let Some(tile) = game.state.tiles.get_mut(&idx) {
            tile.owner = pov_id;
            tile.ruling_city_coords = Some(polyfish::coords::Coords { idx: 0, x: 0, y: 0 });
        }
    }

    // Set terrains
    game.state.tiles.get_mut(&1).unwrap().terrain_type = TerrainType::Field;
    game.state.tiles.get_mut(&2).unwrap().terrain_type = TerrainType::Forest;
    game.state.tiles.get_mut(&3).unwrap().terrain_type = TerrainType::Mountain;

    // 1. Check if Sanctuary can be built on all 3 land types
    let moves = game.legal_moves();
    let can_build_field = moves.iter().any(|m| {
        m.target_idx().unwrap_or(999) == 1 && m.structure_type().unwrap_or(StructureType::None) == StructureType::Sanctuary
    });
    assert!(can_build_field, "Should be able to build Sanctuary on Field");

    let can_build_forest = moves.iter().any(|m| {
        m.target_idx().unwrap_or(999) == 2 && m.structure_type().unwrap_or(StructureType::None) == StructureType::Sanctuary
    });
    assert!(can_build_forest, "Should be able to build Sanctuary on Forest");

    // 2. Build one Sanctuary
    let build_move = BuildMove::new(1, StructureType::Sanctuary);
    game.play_move(&build_move);

    assert_eq!(
        polyfish::functions::get_structure_type_at(&game.state, 1),
        Some(StructureType::Sanctuary)
    );

    // 3. Check limit (should NOT be able to build second Sanctuary in same city)
    let moves_after = game.legal_moves();
    let san_count = moves_after.iter().filter(|m| {
        m.structure_type().unwrap_or(StructureType::None) == StructureType::Sanctuary
    }).count();
    assert_eq!(san_count, 0, "Should NOT be able to build second Sanctuary in same city");
}

#[test]
fn test_sanctuary_income() {
    let mut game = setup_elyrion_game();
    let pov_id = game.current_player_id();
    
    // Setup city at 10
    let territory = vec![10, 11, 9, 10 - 16, 10 + 16];
    if let Some(tribe) = game.state.tribes.get_mut(&pov_id) {
        tribe.cities.push(CityState {
            idx: 10,
            owner: pov_id,
            level: 1,
            production: 1,
            _territory: territory.clone(),
            ..Default::default()
        });
    }

    for &t_idx in &territory {
        if let Some(tile) = game.state.tiles.get_mut(&t_idx) {
            tile.owner = pov_id;
            tile.ruling_city_coords = Some(polyfish::coords::Coords::from_index(10, 16));
            if t_idx == 10 {
                tile.capital_of = pov_id;
            }
        }
    }

    // Place Sanctuary and ANIMAT AT ONCE
    game.state.structures.insert(10, Some(polyfish::states::StructureState {
        structure_type: StructureType::Sanctuary,
        founded: 1,
        level: 1,
    }));

    game.state.resources.insert(11, Some(polyfish::states::ResourceState { resource_type: ResourceType::Game }));
    game.state.resources.insert(9, Some(polyfish::states::ResourceState { resource_type: ResourceType::Game }));

    let tribe = game.state.tribes.get(&pov_id).unwrap();
    let city = &tribe.cities[0];
    let production = polyfish::functions::get_city_production(&game.state, city);
    
    // Base(1) + Capital(1) + Animals(2) = 4
    assert_eq!(production, 4, "Sanctuary should provide +2 stars for 2 adjacent animals");
}

#[test]
fn test_sanctuary_animal_attraction() {
    let mut game = setup_elyrion_game();
    let pov_id = game.current_player_id();
    
    // City at 50
    let territory = vec![50, 51];
    if let Some(tribe) = game.state.tribes.get_mut(&pov_id) {
        tribe.cities.push(CityState {
            idx: 50,
            owner: pov_id,
            _territory: territory.clone(),
            ..Default::default()
        });
    }

    for &t_idx in &territory {
        if let Some(tile) = game.state.tiles.get_mut(&t_idx) {
            tile.owner = pov_id;
            tile.ruling_city_coords = Some(polyfish::coords::Coords::from_index(50, 16));
        }
    }
    
    game.state.tiles.get_mut(&51).unwrap().terrain_type = TerrainType::Forest;

    // Place Sanctuary
    game.state.structures.insert(50, Some(polyfish::states::StructureState {
        structure_type: StructureType::Sanctuary,
        founded: 1, // Created at Turn 1
        level: 1,
    }));

    // Manually progress turns and call effects to bypass EndTurn complexities in solo test
    // Turn 2
    game.state.settings.turn = 2;
    actions::process_start_turn_effects(&mut game.state, pov_id);
    assert!(game.state.resources.get(&51).and_then(|r| r.as_ref()).is_none(), "Should not spawn yet");

    // Turn 3
    game.state.settings.turn = 3;
    actions::process_start_turn_effects(&mut game.state, pov_id);
    assert!(game.state.resources.get(&51).and_then(|r| r.as_ref()).is_none(), "Should not spawn yet");

    // Turn 4 (Age 3)
    game.state.settings.turn = 4;
    actions::process_start_turn_effects(&mut game.state, pov_id);
    
    let res = game.state.resources.get(&51).and_then(|r| r.as_ref());
    assert!(res.is_some(), "Animal should have spawned on Turn 4");
    assert_eq!(res.unwrap().resource_type, ResourceType::Game);
}
