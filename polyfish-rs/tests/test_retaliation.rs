use polyfish::game::Game;
use polyfish::types::UnitType;
use polyfish::{PlayerId, Coords};
use polyfish::moves::AttackMove;

#[test]
fn test_retaliation_logic() {
    let mut game = Game::new();
    // Tiny map is 11x11
    game.state.settings.size = 11;
    
    // Add two tribes
    let p1: PlayerId = 1;
    let p2: PlayerId = 2;
    
    // Setup Tribe 1
    let mut t1 = polyfish::states::TribeState::default();
    t1.id = p1;
    game.state.tribes.insert(p1, t1);
    
    // Setup Tribe 2
    let mut t2 = polyfish::states::TribeState::default();
    t2.id = p2;
    game.state.tribes.insert(p2, t2);

    // Helper to spawn unit
    let spawn_unit = |game: &mut Game, player: PlayerId, unit_type: UnitType, x: i32, y: i32| {
        let size = game.state.settings.size;
        let idx = y * size + x;
        let mut unit = polyfish::states::UnitState::default();
        unit.owner = player;
        unit.unit_type = unit_type;
        unit.health = 10;
        unit.coords = Coords::from_index(idx, size);
        game.state.tribes.get_mut(&player).unwrap().units.push(unit);
        game.state.tiles.entry(idx).or_default()._unit_owner_id = Some(player);
        idx
    };

    // Case 1: Warrior (p1) attacks Warrior (p2) at distance 1
    // Warrior at (5,5), Target at (5,6)
    let w1_idx = spawn_unit(&mut game, p1, UnitType::Warrior, 5, 5);
    let w2_idx = spawn_unit(&mut game, p2, UnitType::Warrior, 5, 6);
    
    let atk_move = AttackMove::new(w1_idx, w2_idx);
    game.play_move(&atk_move);
    
    let w1_after = &game.state.tribes[&p1].units[0];
    let w2_after = &game.state.tribes[&p2].units[0];
    
    println!("Warrior vs Warrior (dist 1):");
    println!("  Attacker HP: {}", w1_after.health);
    println!("  Defender HP: {}", w2_after.health);
    
    // Check for retaliation (Attacker should have lost health)
    assert!(w1_after.health < 10, "Warrior should have been retaliated at distance 1");
    assert!(w2_after.health < 10, "Defender should have taken damage");

    // Reset game state for next test
    game = Game::new();
    game.state.settings.size = 11;
    game.state.tribes.insert(p1, polyfish::states::TribeState { id: p1, ..Default::default() });
    game.state.tribes.insert(p2, polyfish::states::TribeState { id: p2, ..Default::default() });

    // Case 2: Archer (p1) attacks Warrior (p2) at distance 2
    // Archer at (5,5), Warrior at (5,7)
    let a1_idx = spawn_unit(&mut game, p1, UnitType::Archer, 5, 5);
    let w3_idx = spawn_unit(&mut game, p2, UnitType::Warrior, 5, 7);
    
    let atk_move_2 = AttackMove::new(a1_idx, w3_idx);
    game.play_move(&atk_move_2);
    
    let a1_after = &game.state.tribes[&p1].units[0];
    let w3_after = &game.state.tribes[&p2].units[0];
    
    println!("Archer vs Warrior (dist 2):");
    println!("  Attacker HP: {}", a1_after.health);
    println!("  Defender HP: {}", w3_after.health);
    
    assert_eq!(a1_after.health, 10, "Archer should NOT have been retaliated at distance 2 by a Warrior");
    assert!(w3_after.health < 10, "Defender should have taken damage");

    // Reset game state
    game = Game::new();
    game.state.settings.size = 11;
    game.state.tribes.insert(p1, polyfish::states::TribeState { id: p1, ..Default::default() });
    game.state.tribes.insert(p2, polyfish::states::TribeState { id: p2, ..Default::default() });

    // Case 3: Archer (p1) attacks Archer (p2) at distance 2
    // Archer at (5,5), Archer at (5,7)
    let a2_idx = spawn_unit(&mut game, p1, UnitType::Archer, 5, 5);
    let a3_idx = spawn_unit(&mut game, p2, UnitType::Archer, 5, 7);
    
    let atk_move_3 = AttackMove::new(a2_idx, a3_idx);
    game.play_move(&atk_move_3);
    
    let a2_after = &game.state.tribes[&p1].units[0];
    let a3_after = &game.state.tribes[&p2].units[0];
    
    println!("Archer vs Archer (dist 2):");
    println!("  Attacker HP: {}", a2_after.health);
    println!("  Defender HP: {}", a3_after.health);
    
    // This is the CRITICAL one. If user logic "ranged units dont get retaliated when > 1 tile away" is true,
    // then a2_after.health should be 10.
    // If the CURRENT implementation is kept (dist <= def_range), it will be < 10 because a3 has range 2.
    
    if a2_after.health < 10 {
        println!("  RESULT: Archer retaliated against another Archer at distance 2.");
    } else {
        println!("  RESULT: Archer did NOT retaliate at distance 2.");
    }
}
