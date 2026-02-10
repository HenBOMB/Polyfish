use polyfish::ai::book::Book;
use polyfish::game::Game;
use polyfish::mapgen::{MapGenSettings, generate};
use polyfish::states::ResourceState;
use polyfish::types::{MapSize, MoveType, ResourceType, TerrainType, TribeType};

#[test]
fn test_imperius_opening() {
    let mut settings = MapGenSettings::default();
    settings.size = MapSize::Tiny;
    settings.tribes = vec![TribeType::Imperius, TribeType::Oumaji];
    settings.seed = 123; // Deterministic

    let mut game = Game::new();
    game.state = generate(settings);
    game.post_load();

    // Find Imperius Player ID
    let imp_id = game
        .state
        .tribes
        .iter()
        .find(|(_, t)| t.tribe_type == TribeType::Imperius)
        .map(|(&id, _)| id)
        .unwrap();

    // === TURN 1: Expect Harvest or Step ===
    game.state.settings.turn = 1;
    game.state.settings.current_player_turn_id = imp_id;

    // Ensure we have money for Harvest (typically 2 stars)
    game.state.tribes.get_mut(&imp_id).unwrap().stars = 5;

    // Manually place a fruit under the starting unit to guarantee Harvest is legal
    // Imperius starts with Organization, so can harvest Fruit.
    // Find a unit within the tribe's unit list
    let unit_pos = game.state.tribes.get(&imp_id).unwrap().units[0].coords;
    let unit_tile_idx = game
        .state
        .tiles
        .iter()
        .find(|(_, t)| t.coords == unit_pos)
        .map(|(&i, _)| i)
        .unwrap();

    // Modify tile to have Fruit
    // Resources are stored in game.state.resources map
    game.state.resources.insert(
        unit_tile_idx,
        Some(ResourceState {
            resource_type: ResourceType::Fruit,
            tile_index: unit_tile_idx,
        }),
    );

    game.state
        .tiles
        .get_mut(&unit_tile_idx)
        .unwrap()
        .terrain_type = TerrainType::Field; // Fruit on Field

    let rec_moves = Book::recommend(&game);

    println!("Turn 1 Recommended Moves for Imperius:");
    for m in &rec_moves {
        println!("- {:?}", m.move_type());
    }

    assert!(
        !rec_moves.is_empty(),
        "Should have finding book moves for Imperius Turn 1"
    );

    // Check if ALL recommended moves are allowed types (Harvest, Step)
    for m in &rec_moves {
        match m.move_type() {
            MoveType::Harvest | MoveType::Step => {} // OK
            _ => panic!("Unexpected book move for Turn 1: {:?}", m.move_type()),
        }
    }

    // === TURN 2: Expect Summon or Step ===
    game.state.settings.turn = 2;

    // Ensure we have money for Summon (Warrior = 2 stars)
    game.state.tribes.get_mut(&imp_id).unwrap().stars = 5;

    let rec_moves_t2 = Book::recommend(&game);
    println!("Turn 2 Recommended Moves for Imperius:");
    for m in &rec_moves_t2 {
        println!("- {:?}", m.move_type());
    }

    assert!(
        !rec_moves_t2.is_empty(),
        "Should have finding book moves for Imperius Turn 2"
    );

    for m in &rec_moves_t2 {
        match m.move_type() {
            MoveType::Summon | MoveType::Step => {} // OK
            _ => panic!("Unexpected book move for Turn 2: {:?}", m.move_type()),
        }
    }
}

#[test]
fn test_oumaji_opening() {
    let mut settings = MapGenSettings::default();
    settings.size = MapSize::Tiny;
    settings.tribes = vec![TribeType::Imperius, TribeType::Oumaji];
    settings.seed = 456;

    let mut game = Game::new();
    game.state = generate(settings);
    game.post_load();

    let oum_id = game
        .state
        .tribes
        .iter()
        .find(|(_, t)| t.tribe_type == TribeType::Oumaji)
        .map(|(&id, _)| id)
        .unwrap();

    // === TURN 1: Expect Harvest or Step ===
    game.state.settings.turn = 1;
    game.state.settings.current_player_turn_id = oum_id;
    game.state.tribes.get_mut(&oum_id).unwrap().stars = 5;

    let rec_moves = Book::recommend(&game);
    // Oumaji only recommends Harvest/Step on turn 1
    for m in &rec_moves {
        match m.move_type() {
            MoveType::Harvest | MoveType::Step => {}
            _ => panic!("Unexpected book move Oumaji Turn 1: {:?}", m.move_type()),
        }
    }

    // === TURN 2: Expect Step ONLY (Oumaji typically doesn't summon T2 in book?) ===
    // Checking opening.rs:
    // TribeType::Oumaji => match turn { 2 => &[MoveType::Step] }
    game.state.settings.turn = 2;
    let rec_moves_t2 = Book::recommend(&game);

    for m in &rec_moves_t2 {
        match m.move_type() {
            MoveType::Step => {} // OK
            _ => panic!(
                "Unexpected book move Oumaji Turn 2: {:?} (Expected Step only)",
                m.move_type()
            ),
        }
    }
}
