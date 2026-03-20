use polyfish::ai::ordering::score_move;
use polyfish::game::Game;
use polyfish::moves::harvest::HarvestMove;
use polyfish::states::{CityState, ResourceState, TribeState};
use polyfish::types::ResourceType;

#[test]
fn test_harvest_level_up_prioritization() {
    let mut game = Game::new();
    let player_id = 1;
    game.state.settings.current_player_turn_id = player_id;

    // Initialize tribe in the map
    game.state.tribes.insert(
        player_id,
        TribeState {
            id: player_id,
            ..Default::default()
        },
    );

    // Setup cities
    let city1_idx = 10;
    let city2_idx = 20;

    // Add two cities to the tribe
    if let Some(tribe) = game.state.tribes.get_mut(&player_id) {
        tribe.cities.clear();

        // City 1: Needs 1 pop to level up
        let mut city1 = CityState::default();
        city1.owner = player_id;
        city1.idx = city1_idx;
        city1.level = 1;
        city1.population = 1; // Level 2 needs 2 pop.
        city1._territory = vec![city1_idx, 11];
        tribe.cities.push(city1);

        // City 2: Needs 2 pop to level up
        let mut city2 = CityState::default();
        city2.owner = player_id;
        city2.idx = city2_idx;
        city2.level = 1;
        city2.population = 0; // Level 2 needs 2 pop.
        city2._territory = vec![city2_idx, 21];
        tribe.cities.push(city2);
    }

    // Add Fruit to both cities
    game.state.resources.insert(
        11,
        Some(ResourceState {
            resource_type: ResourceType::Fruit,
        }),
    );
    game.state.resources.insert(
        21,
        Some(ResourceState {
            resource_type: ResourceType::Fruit,
        }),
    );

    // Generate Harvest moves
    let mv1 = HarvestMove::new(11); // Finishes city 1
    let mv2 = HarvestMove::new(21); // Doesn't finish city 2

    let score1 = score_move(&game, &mv1);
    let score2 = score_move(&game, &mv2);

    println!("Score 1 (Level up): {}", score1);
    println!("Score 2 (No level up): {}", score2);

    // Score 1 should have +5.0 bonus (Total 27.0), Score 2 should have -4.0 penalty (Total 18.0)
    assert!(
        score1 > score2,
        "AI should prioritize harvest that finishes city level-up ({} vs {})",
        score1,
        score2
    );
    assert!(score1 >= 27.0, "Score 1 should be at least 27.0");
    assert!(score2 <= 18.0, "Score 2 should be at most 18.0");
}
