use polyfish::ai::ordering::score_move;
use polyfish::game::Game;
use polyfish::moves::ClearForestMove;
use polyfish::states::{CityState, TileState, TribeState};

#[test]
fn test_clear_forest_prevention() {
    let mut game = Game::new();
    let player_id = 1;
    game.state.settings.current_player_turn_id = player_id;

    // Initialize tribe
    let mut tribe = TribeState::default();
    tribe.id = player_id;
    tribe.stars = 5; // Healthy treasury

    let city_idx = 10;
    let mut city = CityState::default();
    city.owner = player_id;
    city.idx = city_idx;
    city._territory = vec![city_idx, 11, 12, 13];
    // Set level/population so it's far from levelling up
    city.level = 3;
    city.population = 0; // Needs 4 pop to level up
    tribe.cities.push(city);
    game.state.tribes.insert(player_id, tribe);

    // Forest on tile 11.
    // On Map Size 11:
    // 10 = (10,0)
    // 11 = (0,1)
    // Actually, in our simplistic test setup, get_adjacent_indices(11, 1) will return neighbors.
    // Let's ensure tiles exist so adj logic works.
    for i in 0..121 {
        game.state.tiles.insert(
            i,
            TileState {
                coords: polyfish::coords::Coords::from_index(i, 11),
                ..Default::default()
            },
        );
    }

    game.state.tiles.get_mut(&11).unwrap().terrain_type = polyfish::types::TerrainType::Forest;
    game.state.tiles.get_mut(&11).unwrap().owner = player_id;

    let mv = ClearForestMove::new(11);

    // Scenario 1: Healthy treasury (stars=5)
    let score_healthy = score_move(&game, &mv);
    // Base 3.0 + Treasury Penalty -10.0 = -7.0
    // Neighbors of 11 (central/edge cases vary, but let's assume it borders empty tiles)
    // The test previously gave -19.5, meaning -12.5 clustering penalty (5 hub-spots * 2.5)
    println!("Score Healthy: {}", score_healthy);
    assert!(score_healthy < -10.0, "Should have clustering penalty");

    // Scenario 2: Broke (stars=0) but doesn't enable level up
    if let Some(t) = game.state.tribes.get_mut(&player_id) {
        t.stars = 0;
    }
    let score_desperate = score_move(&game, &mv);
    // Base 3.0 + Desperation 2.0 = 5.0
    // Then Clustering Penalty (~ -12.5)
    println!("Score Desperate: {}", score_desperate);
    assert!(
        score_desperate < 0.0,
        "Clustering should outweigh simple desperation"
    );

    // Scenario 3: Broke (stars=0) AND enables level up
    if let Some(t) = game.state.tribes.get_mut(&player_id) {
        let city = &mut t.cities[0];
        city.level = 1;
        city.population = 1; // Needs 1 more pop
    }
    // Assume tile 12 has a Fruit (Harvestable)
    game.state.resources.insert(
        12,
        Some(polyfish::states::ResourceState {
            resource_type: polyfish::types::ResourceType::Fruit,
        }),
    );

    let score_enabling = score_move(&game, &mv);
    // Base 3.0 + Enabling 10.0 = 13.0
    // Then Clustering Penalty (~ -12.5)
    println!("Score Enabling: {}", score_enabling);
    assert!(score_enabling > -5.0 && score_enabling < 5.0);
}
