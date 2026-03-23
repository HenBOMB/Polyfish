use polyfish::game::Game;
use polyfish::moves::ResignMove;
use polyfish::states::{GameState, TribeState, CityState, UnitState};
use polyfish::types::{TribeType, UnitType, ModeType};
use polyfish::coords::Coords;
use indexmap::IndexMap;

#[test]
fn test_resignation_removes_units() {
    let mut state = GameState::default();
    state.settings.size = 11;
    state.settings._max_tribe_count = 2; // Need at least 2 to not end game immediately if logic checks for last player
    state.settings.current_player_turn_id = 1;
    state.settings.mode = ModeType::Domination;

    // Initialize tiles
    for idx in 0..(11 * 11) {
        state.tiles.insert(idx, polyfish::states::TileState {
            coords: Coords::from_index(idx, 11),
            ..polyfish::states::TileState::default()
        });
    }

    // Tribe 1
    let mut tribe1 = TribeState::default();
    tribe1.id = 1;
    tribe1.tribe_type = TribeType::Imperius;
    
    let city1 = CityState {
        idx: 0,
        owner: 1,
        level: 1,
        name: "City 1".to_string(),
        ..CityState::default()
    };
    tribe1.cities.push(city1);
    
    let unit1 = UnitState {
        owner: 1,
        unit_type: UnitType::Warrior,
        coords: Coords::from_index(0, 11),
        home_coords: Some(Coords::from_index(0, 11)),
        ..UnitState::default()
    };
    tribe1.units.push(unit1);
    state.tribes.insert(1, tribe1);

    // Tribe 2 (to keep game going)
    let mut tribe2 = TribeState::default();
    tribe2.id = 2;
    tribe2.tribe_type = TribeType::Bardur;
    let city2 = CityState {
        idx: 120,
        owner: 2,
        level: 1,
        name: "City 2".to_string(),
        ..CityState::default()
    };
    tribe2.cities.push(city2);
    state.tribes.insert(2, tribe2);
    
    let mut game = Game { state };
    // Set first player turn BEFORE post_load
    game.state.settings.current_player_turn_id = 1;
    game.state.settings.turn = 1; // Turn must be > 0 for resignation to be tracked properly in this engine
    game.post_load();
    
    // Initial check
    assert_eq!(game.state.tribes[&1].units.len(), 1);
    
    // Resign
    let resign_move = ResignMove;
    game.play_move(&resign_move);
    
    // Verify units are gone for tribe 1
    assert_eq!(game.state.tribes[&1].units.len(), 0);
    assert_eq!(game.state.tribes[&1].resigned_turn, 1);
    
    // Verify city unit count is 0
    let tribe1_after = &game.state.tribes[&1];
    assert_eq!(polyfish::functions::get_city_unit_count(&game.state, &tribe1_after.cities[0]), 0);
}
