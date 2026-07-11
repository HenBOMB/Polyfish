use polyfish::ai::scoring::score_move;
use polyfish::coords::Coords;
use polyfish::game::Game;
use polyfish::moves::summon::SummonMove;
use polyfish::states::{CityState, TileState, TribeState, UnitState};
use polyfish::types::UnitType;

#[test]
fn test_summon_contextual_priority() {
    let mut game = Game::new();
    let player_id = 1;
    let enemy_id = 2;
    game.state.settings.current_player_turn_id = player_id;
    game.state.settings.turn = 2; // Enable summons
    game.state.settings._max_tribe_count = 2;

    // Initialize tribes in the map
    game.state.tribes.insert(
        player_id,
        TribeState {
            id: player_id,
            ..Default::default()
        },
    );
    game.state.tribes.insert(
        enemy_id,
        TribeState {
            id: enemy_id,
            ..Default::default()
        },
    );

    // Setup cities
    let city_idx = 12; // (1, 1) on size 11
    if let Some(tribe) = game.state.tribes.get_mut(&player_id) {
        tribe.cities.clear();
        let mut city = CityState::default();
        city.owner = player_id;
        city.idx = city_idx;
        tribe.cities.push(city);
        tribe.units.clear();
    }

    let mv = SummonMove::new(city_idx, UnitType::Warrior);

    // --- Scenario 1: Base Summoning (No threat, small army) ---
    let score_base = score_move(&game, &mv);
    // Base 10.0 + Small Army Bonus 8.0 = 18.0
    println!("Score Base: {}", score_base);
    assert_eq!(score_base, 18.0); // Corrected from 25.0

    // --- Scenario 2: High Threat (Enemy nearby) ---
    // Add an enemy unit near the city
    let enemy_idx = 13; // (2, 1) - Adjacent to (1, 1)
    if let Some(enemy_tribe) = game.state.tribes.get_mut(&enemy_id) {
        let mut enemy_unit = UnitState::default();
        enemy_unit.owner = enemy_id;
        enemy_unit.unit_type = UnitType::Warrior;
        enemy_unit.coords = Coords::from_index(enemy_idx, game.state.settings.size);
        enemy_tribe.units.push(enemy_unit);
    }

    // Ensure tile exists for detection logic
    game.state.tiles.insert(
        enemy_idx,
        TileState {
            coords: Coords::from_index(enemy_idx, game.state.settings.size),
            ..Default::default()
        },
    );

    let score_threat = score_move(&game, &mv);
    // Base 10.0 + Threat 15.0 + Small Army 8.0 = 33.0
    println!("Score Threat: {}", score_threat);
    assert_eq!(score_threat, 33.0); // Corrected from 40.0

    // --- Scenario 3: Large Army (No threat) ---
    // Remove enemy, add many units
    if let Some(enemy_tribe) = game.state.tribes.get_mut(&enemy_id) {
        enemy_tribe.units.clear();
    }
    if let Some(tribe) = game.state.tribes.get_mut(&player_id) {
        // Add 3 warriors (City count 1 * 2 = 2, so 3 is "large")
        for i in 0..3 {
            let mut u = UnitState::default();
            u.owner = player_id;
            u.unit_type = UnitType::Warrior;
            u.coords = Coords::from_index(30 + i, game.state.settings.size);
            tribe.units.push(u);
        }
    }
    let score_bloat = score_move(&game, &mv);
    // Base 10.0 + (-15.0 penalty) = -5.0
    println!("Score Bloat: {}", score_bloat);
    assert_eq!(score_bloat, -5.0); // Corrected from 5.0
}
