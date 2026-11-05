use polyfish::PlayerId;
use polyfish::ai::evaluator::research::evaluate_tech_utility;
use polyfish::ai::heuristic_mcts::{GreedyHeuristicAgent, HeuristicMctsAgent};
use polyfish::ai::scoring::score_move;
use polyfish::coords::Coords;
use polyfish::game::Game;
use polyfish::mapgen::{MapGenSettings, generate};
use polyfish::moves::research::ResearchMove;
use polyfish::states::{CityState, TileState, TribeState, UnitState};
use polyfish::types::{
    MapSize, MapType, MoveType, StructureType, TechnologyType, TerrainType, TribeType, UnitType,
};

#[test]
fn test_drylands_sailing_pruned_to_neg_infinity() {
    let gen_settings = MapGenSettings {
        size: MapSize::Small,
        map_type: MapType::Drylands,
        tribes: vec![TribeType::Imperius, TribeType::Oumaji],
        seed: 42,
        ..Default::default()
    };

    let mut game = Game::new();
    game.state = generate(gen_settings);
    game.post_load();

    let player_id: PlayerId = game.state.settings.current_player_turn_id;
    if let Some(tribe) = game.state.tribes.get_mut(&player_id) {
        tribe.stars = 50;
    }

    let sailing_mv = ResearchMove::new(TechnologyType::Sailing);
    let utility = evaluate_tech_utility(&game.state, player_id, TechnologyType::Sailing);
    let score = score_move(&game, &sailing_mv);

    assert_eq!(
        utility,
        f32::NEG_INFINITY,
        "Sailing on Drylands must evaluate to -infinity"
    );
    assert_eq!(
        score,
        f32::NEG_INFINITY,
        "Sailing score on Drylands must be -infinity"
    );
}

#[test]
fn test_farming_and_mining_pruned_when_resources_absent() {
    let mut game = Game::new();
    let player_id: PlayerId = 1;
    game.state.settings.current_player_turn_id = player_id;
    game.state.settings.size = 11;

    let mut tribe = TribeState {
        id: player_id,
        stars: 30,
        ..Default::default()
    };
    // Prerequisites for Farming and Mining
    tribe.tech_vanilla.push(polyfish::states::TechnologyState {
        tech_type: TechnologyType::Organization,
        discovered: true,
        discovered_turn: 0,
    });
    tribe.tech_vanilla.push(polyfish::states::TechnologyState {
        tech_type: TechnologyType::Climbing,
        discovered: true,
        discovered_turn: 0,
    });

    let mut city = CityState::default();
    city.owner = player_id;
    city.idx = 12;
    city._territory = vec![12, 13, 23, 24];
    tribe.cities.push(city);

    // Setup territory as all fields without crops
    for &idx in &[12, 13, 23, 24] {
        let mut tile = TileState::default();
        tile.coords = Coords::from_index(idx, 11);
        tile.terrain_type = TerrainType::Field;
        tile.owner = player_id;
        tile.explorers.insert(player_id);
        game.state.tiles.insert(idx, tile);
    }
    game.state.tribes.insert(player_id, tribe);

    let farm_util = evaluate_tech_utility(&game.state, player_id, TechnologyType::Farming);
    assert_eq!(
        farm_util,
        f32::NEG_INFINITY,
        "Farming with zero crops must evaluate to -infinity"
    );

    let mine_util = evaluate_tech_utility(&game.state, player_id, TechnologyType::Mining);
    assert_eq!(
        mine_util,
        f32::NEG_INFINITY,
        "Mining with zero metal must evaluate to -infinity"
    );
}

#[test]
fn test_hybrid_trade_pruning_and_activation() {
    let mut game = Game::new();
    let player_id: PlayerId = 1;
    game.state.settings.current_player_turn_id = player_id;
    game.state.settings.size = 11;

    let mut tribe = TribeState {
        id: player_id,
        stars: 4, // Insufficient activation capital
        ..Default::default()
    };

    let mut city = CityState::default();
    city.owner = player_id;
    city.idx = 12;
    city._territory = vec![12, 13, 23, 24];
    tribe.cities.push(city);

    // Setup fields and forests
    for &idx in &[12, 13, 23] {
        let mut tile = TileState::default();
        tile.coords = Coords::from_index(idx, 11);
        tile.terrain_type = TerrainType::Field;
        tile.owner = player_id;
        tile.explorers.insert(player_id);
        game.state.tiles.insert(idx, tile);
    }
    let mut forest_tile = TileState::default();
    forest_tile.coords = Coords::from_index(24, 11);
    forest_tile.terrain_type = TerrainType::Forest;
    forest_tile.owner = player_id;
    forest_tile.explorers.insert(player_id);
    game.state.tiles.insert(24, forest_tile);

    game.state.tribes.insert(player_id, tribe);

    // 1. Underfunded (stars = 4): should be hard masked
    let trade_util_underfunded =
        evaluate_tech_utility(&game.state, player_id, TechnologyType::Trade);
    assert_eq!(
        trade_util_underfunded,
        f32::NEG_INFINITY,
        "Trade when underfunded and having 0 hubs must evaluate to -infinity"
    );

    // 2. High liquidity (stars = 40): should be -15.0 (ranks below EndTurn at 0.0)
    if let Some(t) = game.state.tribes.get_mut(&player_id) {
        t.stars = 40;
    }
    let trade_util_liquid = evaluate_tech_utility(&game.state, player_id, TechnologyType::Trade);
    assert_eq!(
        trade_util_liquid, -15.0,
        "Trade with sufficient stars but 0 hubs must evaluate to -15.0"
    );

    let trade_mv = ResearchMove::new(TechnologyType::Trade);
    let trade_score = score_move(&game, &trade_mv);
    assert_eq!(
        trade_score, -15.0,
        "Trade move score must be -15.0 (below EndTurn at 0.0)"
    );

    // 3. Existing Hub (Sawmill placed at index 24): Trade must yield positive utility!
    let mut sawmill = polyfish::states::StructureState::default();
    sawmill.structure_type = StructureType::Sawmill;
    game.state.structures.insert(24, Some(sawmill));

    let trade_util_with_hub =
        evaluate_tech_utility(&game.state, player_id, TechnologyType::Trade);
    assert!(
        trade_util_with_hub > 0.0,
        "Trade with an existing hub should have positive utility, got {}",
        trade_util_with_hub
    );
    let trade_score_with_hub = score_move(&game, &trade_mv);
    assert!(
        trade_score_with_hub >= 10.0,
        "Trade move score with an existing hub should be >= 10.0, got {}",
        trade_score_with_hub
    );
}

#[test]
fn test_climbing_fog_discovery_valuation() {
    let mut game = Game::new();
    let player_id: PlayerId = 1;
    game.state.settings.current_player_turn_id = player_id;
    let size = 11;
    game.state.settings.size = size;

    let mut tribe = TribeState {
        id: player_id,
        stars: 20,
        ..Default::default()
    };

    let mut city = CityState::default();
    city.owner = player_id;
    city.idx = 12;
    city._territory = vec![12, 13];
    tribe.cities.push(city);

    // Unit near a mountain
    let mut warrior = UnitState::default();
    warrior.unit_type = UnitType::Warrior;
    warrior.owner = player_id;
    warrior.coords = Coords::from_index(12, size);
    tribe.units.push(warrior);

    // Setup tiles: index 13 is a mountain, index 14+ are unexplored fog
    let mut mountain_tile = TileState::default();
    mountain_tile.coords = Coords::from_index(13, size);
    mountain_tile.terrain_type = TerrainType::Mountain;
    mountain_tile.explorers.insert(player_id);
    game.state.tiles.insert(13, mountain_tile);

    // Flat tile explored
    let mut flat_tile = TileState::default();
    flat_tile.coords = Coords::from_index(12, size);
    flat_tile.terrain_type = TerrainType::Field;
    flat_tile.explorers.insert(player_id);
    game.state.tiles.insert(12, flat_tile);

    // Unexplored fog tiles around the mountain (e.g. index 14, 25, etc.)
    for &fog_idx in &[14, 15, 24, 25, 26, 35, 36] {
        let mut fog_tile = TileState::default();
        fog_tile.coords = Coords::from_index(fog_idx, size);
        fog_tile.terrain_type = TerrainType::Field;
        // Not explored by player_id
        game.state.tiles.insert(fog_idx, fog_tile);
    }

    game.state.tribes.insert(player_id, tribe);

    let climbing_util = evaluate_tech_utility(&game.state, player_id, TechnologyType::Climbing);
    assert!(
        climbing_util > 0.0,
        "Climbing near unexplored fog should have positive utility from discovery, got {}",
        climbing_util
    );

    let climbing_mv = ResearchMove::new(TechnologyType::Climbing);
    let climbing_score = score_move(&game, &climbing_mv);
    assert!(
        climbing_score >= 10.0,
        "Climbing move score should be >= 10.0 when revealing fog, got {}",
        climbing_score
    );
}

#[test]
fn test_heuristic_mcts_never_selects_neg_infinity_tech() {
    let gen_settings = MapGenSettings {
        size: MapSize::Small,
        map_type: MapType::Drylands,
        tribes: vec![TribeType::Imperius, TribeType::Oumaji],
        seed: 777,
        ..Default::default()
    };

    let mut game = Game::new();
    game.state = generate(gen_settings);
    game.post_load();

    let player_id: PlayerId = game.state.settings.current_player_turn_id;
    if let Some(tribe) = game.state.tribes.get_mut(&player_id) {
        tribe.stars = 100; // Can afford any tech
    }

    let agent = HeuristicMctsAgent::new(32);
    let (chosen_move, analysis) = agent.select_move_with_analysis(&mut game);

    assert!(chosen_move.is_some(), "Agent should choose a move");
    let chosen = chosen_move.unwrap();

    // Verify chosen move is NOT Sailing
    if chosen.move_type() == MoveType::Research {
        let tech = chosen.tech_type().unwrap();
        assert_ne!(
            tech,
            TechnologyType::Sailing,
            "Heuristic MCTS must never select Sailing on Drylands"
        );
    }

    // Verify Sailing never appears among expanded evaluations with positive visits
    for eval in &analysis.evaluations {
        if eval.move_type == MoveType::Research {
            assert!(
                !eval.description.contains("Sailing"),
                "Sailing on Drylands must be pruned from MCTS evaluations"
            );
        }
    }
}

#[test]
fn test_greedy_heuristic_agent_does_not_panic_when_only_pruned_moves_remain() {
    let gen_settings = MapGenSettings {
        size: MapSize::Small,
        map_type: MapType::Drylands,
        tribes: vec![TribeType::Imperius, TribeType::Oumaji],
        seed: 1793849293, // Seed from Game 38 in RunPod log
        ..Default::default()
    };

    let mut game = Game::new();
    game.state = generate(gen_settings);
    game.post_load();

    let player_id: PlayerId = game.state.settings.current_player_turn_id;
    // Clear all harvestable resources
    game.state.resources.clear();
    if let Some(tribe) = game.state.tribes.get_mut(&player_id) {
        tribe.stars = 5;
        // Make all units exhausted
        for unit in &mut tribe.units {
            unit.moved = true;
            unit.attacked = true;
        }
        // Push all tier 1 techs to tech_vanilla except Sailing, so ONLY Sailing is legal to research
        for &t in &[TechnologyType::Climbing, TechnologyType::Riding, TechnologyType::Fishing, TechnologyType::Hunting] {
            if !tribe.tech_vanilla.iter().any(|tech| tech.tech_type == t) {
                tribe.tech_vanilla.push(polyfish::states::TechnologyState {
                    tech_type: t,
                    discovered: true,
                    ..Default::default()
                });
            }
        }
    }

    let agent = GreedyHeuristicAgent::new();
    // This previously panicked at src/ai/heuristic_mcts.rs:205:53 with Option::unwrap() on None (NaN in softmax)
    let (chosen_move, _visits) = agent.select_move_with_decomposed_visits(&mut game, 1);

    assert!(chosen_move.is_some(), "Agent must return a move without panicking");
    let chosen = chosen_move.unwrap();
    // When only pruned tech or end turn exist, it should choose EndTurn
    assert_eq!(
        chosen.move_type(),
        MoveType::EndTurn,
        "GreedyHeuristicAgent should choose EndTurn rather than a pruned negative infinity move"
    );
}
