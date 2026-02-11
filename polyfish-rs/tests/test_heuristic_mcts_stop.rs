use polyfish::ai::heuristic_mcts::HeuristicMctsAgent;
use polyfish::game::Game;
use polyfish::mapgen::{MapGenSettings, generate};
use polyfish::types::{MapSize, MapType, MoveType, TribeType};

#[test]
fn test_mcts_stops_at_turn_end() {
    let mut game = Game::new();
    let gen_settings = MapGenSettings {
        size: MapSize::Tiny,
        map_type: MapType::Drylands,
        tribes: vec![TribeType::Imperius, TribeType::Bardur],
        seed: 12345,
        ..Default::default()
    };
    game.state = generate(gen_settings);
    game.post_load();

    // 1. Force a state where only EndTurn is possible
    let pid = game.state.settings.current_player_turn_id;
    if let Some(tribe) = game.state.tribes.get_mut(&pid) {
        // Remove units so no movement is possible
        tribe.units.clear();
        // Remove stars so no building/tech is possible
        tribe.stars = 0;
    }

    // Also ensure no cities have "Choose Resource" or other popups waiting?
    // Usually start of game requires picking a tech? No, Imperius starts with 0 stars usually?
    // Actually Imperius starts with Organization.
    // If we have 0 stars, we can't do anything.

    // Verify only EndTurn exists
    let moves = game.legal_moves();
    assert!(!moves.is_empty(), "Should have at least EndTurn");

    let only_end_turn = moves.iter().all(|m| m.move_type() == MoveType::EndTurn);
    if !only_end_turn {
        // Print what other moves exist
        for m in &moves {
            println!("Move available: {:?}", m.move_type());
        }
        panic!("Setup failed: expected only EndTurn, got other moves.");
    }

    // 2. Run MCTS agent
    // 50 iterations is enough to expand the root significantly if allowed
    let agent = HeuristicMctsAgent::new(50);
    let (_best_move, analysis) = agent.select_move_with_analysis(&mut game);

    // 3. Inspect the tree
    // The root should have 1 child (EndTurn).
    // That child should have 0 children (because search stops).

    let tree = analysis.tree.expect("Analysis should contain tree data");

    // We expect the root to have children corresponding to the moves.
    // Since only EndTurn was available, there should be 1 child.
    // Note: HeuristicMcts filters EndTurn if OTHER moves exist. Here no other moves exist.

    assert!(!tree.children.is_empty(), "Root should have children");
    let child = &tree.children[0];

    println!("Child description: {}", child.move_description);

    assert!(
        child.children.is_empty(),
        "EndTurn node should NOT have children. Found {} children. The search continued into opponent's turn!",
        child.children.len()
    );
}
