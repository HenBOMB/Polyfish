use polyfish::PlayerId;
use polyfish::game::Game;
use polyfish::mapgen::{MapGenSettings, generate};
use polyfish::moves::research::ResearchMove;
use polyfish::types::{
    MapSize, MapType, MoveType, ResourceType, TechnologyType, TribeType, UnitType,
};

#[test]
fn test_mcts_tech_unlock_logic() {
    // 1. Setup game
    let gen_settings = MapGenSettings {
        size: MapSize::Small,
        map_type: MapType::Drylands,
        tribes: vec![TribeType::Imperius, TribeType::Imperius],
        seed: 12345,
        ..Default::default()
    };

    let mut game = Game::new();
    game.state = generate(gen_settings);
    game.post_load();

    let player_id: PlayerId = game.state.settings.current_player_turn_id;

    // Give some stars to afford research
    if let Some(tribe) = game.state.tribes.get_mut(&player_id) {
        tribe.stars = 100;
    }

    // 4. Confirm visibility is still hidden for resources
    // Metal requires Climbing (Imperius doesn't start with Climbing)
    let research_climbing = ResearchMove::new(TechnologyType::Climbing);
    let _undo = game
        .simulate_move(&research_climbing)
        .expect("Simulation should succeed");

    let metal_visible = polyfish::functions::is_resource_visible_to_tribe(
        &game.state,
        ResourceType::Metal,
        player_id,
        None,
    );
    assert!(
        !metal_visible,
        "Metal should NOT be visible because Climbing isn't 'discovered'"
    );

    // 5. Move starting unit to free the city
    let city_idx = game.state.tribes[&player_id].cities[0].idx;
    let adj = polyfish::functions::get_adjacent_indices(&game.state, city_idx, 1);
    let target_idx = adj[0];

    if let Some(tribe) = game.state.tribes.get_mut(&player_id) {
        if let Some(unit) = tribe.units.iter_mut().find(|u| u.coords.idx == city_idx) {
            unit.coords.set_at(target_idx, game.state.settings.size);
            game.state.tiles.get_mut(&city_idx).unwrap()._unit_owner_id = None;
            game.state
                .tiles
                .get_mut(&target_idx)
                .unwrap()
                ._unit_owner_id = Some(player_id);
        }
    }

    // 6. Verify we can summon Defender (unlocked by Strategy) after simulated research
    let research_strat = ResearchMove::new(TechnologyType::Strategy);
    let _undo = game
        .simulate_move(&research_strat)
        .expect("Simulation should succeed");

    let tribe = game.state.tribes.get(&player_id).unwrap();
    let strat_state = tribe
        .tech_vanilla
        .iter()
        .find(|t| t.tech_type == TechnologyType::Strategy)
        .expect("Strategy should be in tech list");
    assert!(
        !strat_state.discovered,
        "Strategy should NOT be 'discovered' in simulation"
    );

    let moves = game.legal_moves();
    let has_defender_summon = moves.iter().any(|m| {
        m.move_type() == MoveType::Summon && m.unit_type().ok() == Some(UnitType::Defender)
    });

    assert!(
        has_defender_summon,
        "Should be able to summon Defender during simulation after researching Strategy"
    );

    // 7. Verify Raft Upgrade (Boat/Scout)
    // Research Fishing first (real move)
    let research_fishing = ResearchMove::new(TechnologyType::Fishing);
    let _undo = game
        .play_move(&research_fishing)
        .expect("Fishing should succeed");

    // Add a Raft unit manually
    let raft_tile_idx = target_idx + 1; // adjacent to target_idx
    if let Some(tribe) = game.state.tribes.get_mut(&player_id) {
        let mut raft = polyfish::states::UnitState::default();
        raft.owner = player_id;
        raft.unit_type = UnitType::Raft;
        raft.coords.set_at(raft_tile_idx, game.state.settings.size);
        tribe.units.push(raft);
        game.state
            .tiles
            .get_mut(&raft_tile_idx)
            .unwrap()
            ._unit_owner_id = Some(player_id);
    }

    // Research Sailing via simulate_move
    let research_sailing = ResearchMove::new(TechnologyType::Sailing);
    let _undo = game
        .simulate_move(&research_sailing)
        .expect("Simulation should succeed");

    let moves = game.legal_moves();
    let has_raft_upgrade = moves.iter().any(|m| {
        m.move_type() == MoveType::Summon && m.unit_type().ok() == Some(UnitType::Scoutship)
    });

    assert!(
        has_raft_upgrade,
        "Should be able to upgrade Raft to Scout during simulation after researching Sailing"
    );
}
