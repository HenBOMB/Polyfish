use polyfish::ai::ordering::score_move;
use polyfish::coords::Coords;
use polyfish::game::Game;
use polyfish::moves::summon::SummonMove;
use polyfish::states::{CityState, UnitState};
use polyfish::types::{MapSize, MapType, UnitType};

#[test]
fn test_summon_contextual_priority() {
    let mut game = Game::new();
    let player_id = 1;
    let enemy_id = 2;
    game.state.settings.current_player_turn_id = player_id;
    game.state.settings.turn = 2; // Enable summons

    // Setup cities
    let city_idx = 10;
    if let Some(tribe) = game.state.tribes.get_mut(&player_id) {
        tribe.cities.clear();
        let mut city = CityState::default();
        city.owner = player_id;
        city.tile_index = city_idx;
        city.id = city_idx;
        tribe.cities.push(city);
        tribe.units.clear();
    }

    let mv = SummonMove::new(city_idx, UnitType::Warrior);

    // --- Scenario 1: Base Summoning (No threat, small army) ---
    let score_base = score_move(&game, &mv);
    // Base 15.0 + Small Army Bonus 10.0 = 25.0
    println!("Score Base: {}", score_base);
    assert_eq!(score_base, 25.0);

    // --- Scenario 2: High Threat (Enemy nearby) ---
    // Add an enemy unit near the city
    if let Some(enemy_tribe) = game.state.tribes.get_mut(&enemy_id) {
        let enemy_idx = 11; // Adjacent
        let mut enemy_unit = UnitState::default();
        enemy_unit.owner = enemy_id;
        enemy_unit.unit_type = UnitType::Warrior;
        enemy_unit.coords = Coords::from_index(enemy_idx, game.state.settings.size);
        enemy_tribe.units.push(enemy_unit);
    }
    let score_threat = score_move(&game, &mv);
    // Base 15.0 + Threat 15.0 + Small Army 10.0 = 40.0
    println!("Score Threat: {}", score_threat);
    assert!(score_threat > score_base);

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
            u.coords = Coords::from_index(20 + i, game.state.settings.size);
            tribe.units.push(u);
        }
    }
    let score_bloat = score_move(&game, &mv);
    // Base 15.0 - Bloat Penalty 10.0 = 5.0
    println!("Score Bloat: {}", score_bloat);
    assert!(score_bloat < score_base);
}
