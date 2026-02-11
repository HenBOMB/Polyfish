use polyfish::ai::ordering::score_move;
use polyfish::game::Game;
use polyfish::moves::abilities::forest::ClearForestMove;
use polyfish::types::{
    AbilityType, MapSize, MapType, MoveType, ResourceType, StructureType, TechnologyType,
    TerrainType, TribeType,
};

#[test]
fn test_clear_forest_scoring() {
    let mut game = Game::new();
    // Tiny map for speed
    game.state = polyfish::mapgen::generate(polyfish::mapgen::MapGenSettings {
        size: MapSize::Tiny,
        map_type: MapType::Drylands,
        tribes: vec![TribeType::Imperius, TribeType::Imperius],
        seed: 1234,
        ..Default::default()
    });
    game.post_load();

    let player_id = game.state.settings.current_player_turn_id;
    let city_idx = game.state.tribes[&player_id].cities[0].tile_index;

    // Find a forest tile in our territory
    let forest_idx = *game.state.tribes[&player_id].cities[0]
        ._territory
        .iter()
        .find(|&&idx| game.state.tiles.get(&idx).unwrap().terrain_type == TerrainType::Forest)
        .expect("Should have a forest tile");

    let mv = ClearForestMove::new(forest_idx);

    // --- Scenario 1: Default (low stars) ---
    if let Some(tribe) = game.state.tribes.get_mut(&player_id) {
        tribe.stars = 4;
    }
    let score_default = score_move(&game, &mv);
    // Base 8.0 + Desperation 5.0 = 13.0
    assert!(
        score_default >= 13.0,
        "Score should be at least 13.0 with low stars (got {})",
        score_default
    );

    // --- Scenario 2: Game Resource ---
    game.state.resources.insert(
        forest_idx,
        Some(polyfish::states::ResourceState {
            resource_type: ResourceType::Game,
            ..Default::default()
        }),
    );
    let score_game = score_move(&game, &mv);
    // Base 8.0 + Desperation 5.0 - Game 25.0 = -12.0
    assert!(
        score_game < 0.0,
        "Score should be negative when forest has Game (got {})",
        score_game
    );

    // Remove Game for next tests
    game.state.resources.remove(&forest_idx);

    // --- Scenario 3: Forestry Research ---
    if let Some(tribe) = game.state.tribes.get_mut(&player_id) {
        let forestry_tech = tribe
            .tech_vanilla
            .iter_mut()
            .find(|t| t.tech_type == TechnologyType::Forestry);
        if let Some(t) = forestry_tech {
            t.discovered = true;
        } else {
            tribe.tech_vanilla.push(polyfish::states::TechnologyState {
                tech_type: TechnologyType::Forestry,
                discovered: true,
            });
        }
    }
    let score_forestry = score_move(&game, &mv);
    // Base 8.0 + Desperation 5.0 - Forestry 10.0 = 3.0
    assert!(
        score_forestry < score_default,
        "Score with Forestry ({}) should be lower than default ({})",
        score_forestry,
        score_default
    );

    // --- Scenario 3b: Stuck City (Forestry but level >= 5) ---
    if let Some(tribe) = game.state.tribes.get_mut(&player_id) {
        let city = tribe
            .cities
            .iter_mut()
            .find(|c| c.tile_index == city_idx)
            .unwrap();
        city.level = 5;
    }
    let score_stuck = score_move(&game, &mv);
    // Base 8.0 + Desperation 5.0 = 13.0 (Penalty removed because level >= 5)
    assert!(
        score_stuck > score_forestry,
        "Score when city is level 5 ({}) should be higher than with Forestry penalty ({})",
        score_stuck,
        score_forestry
    );

    // --- Scenario 4: Strategic Level-up ---
    // City needs 1 pop to level. Player has 1 star. Clearing gets to 2 stars.
    // 2 stars = cost of a harvest move (which we'll assume is available and gives 1 pop).
    if let Some(tribe) = game.state.tribes.get_mut(&player_id) {
        // Reset tech discovery for clean test
        if let Some(t) = tribe
            .tech_vanilla
            .iter_mut()
            .find(|t| t.tech_type == TechnologyType::Forestry)
        {
            t.discovered = false;
        }
        tribe.stars = 1;
        let city = tribe
            .cities
            .iter_mut()
            .find(|c| c.tile_index == city_idx)
            .unwrap();
        city.level = 1;
        city.population = 1; // Needs 1 more to reach Level 2 (total 2 needed)
    }
    let score_lvlup = score_move(&game, &mv);
    // Base 8.0 + Levelup Bonus 18.0 = 26.0
    // (Note: desperation boost doesn't trigger if enables_level_up is true in my code)
    assert!(
        score_lvlup > 20.0,
        "Score should be much higher if it enables level up (got {})",
        score_lvlup
    );

    // --- Scenario 5: Road connection to capital ---
    if let Some(tribe) = game.state.tribes.get_mut(&player_id) {
        tribe.stars = 1;
        let city = tribe
            .cities
            .iter_mut()
            .find(|c| c.tile_index == city_idx)
            .unwrap();
        city.level = 1;
        city.population = 1; // Needs 1 more to reach Level 2 (total 2 needed)
        city.connected_to_capital = false;
    }
    let score_road = score_move(&game, &mv);
    // Base 8.0 + Levelup Bonus 18.0 = 26.0
    assert!(
        score_road > 20.0,
        "Score should be high if clearing enables road connection level up (got {})",
        score_road
    );
}
