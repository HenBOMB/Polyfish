use polyfish::coords::Coords;
use polyfish::game::Game;
use polyfish::moves::EndTurnMove;
use polyfish::moves::build::BuildMove;
use polyfish::moves::harvest::HarvestMove;
use polyfish::moves::research::ResearchMove;
use polyfish::moves::summon::SummonMove;
use polyfish::states::{
    CityState, ResourceState, StructureState, TechnologyState, TileState, TribeState,
};
use polyfish::types::{
    ModeType, ResourceType, StructureType, TechnologyType, TerrainType, TribeType, UnitType,
};

fn main() {
    println!("=== Debugging Star Corruption ===");

    test_summon_undo();
    test_research_undo();
    test_turn_cycling_undo();

    test_harvest_undo();
    test_build_undo();
    test_interaction_sequence();

    // Test user's hypothesis: Turn 30 / Game Over boundary
    test_game_over_undo();

    test_capture_ruin_determinism();
    test_capture_ruin_stacking();
}

fn assert_stars(game: &Game, expected: i32, context: &str) {
    let tribe = game.current_tribe().unwrap();
    if tribe.stars != expected {
        panic!(
            "[{}] Expected {} stars, found {}",
            context, expected, tribe.stars
        );
    } else {
        println!("[PASS] {} (Stars: {})", context, tribe.stars);
    }
}

fn test_harvest_undo() {
    println!("\n--- Test Harvest Undo ---");
    let mut game = setup_game();

    // Setup: Imperius needs Organization (Tech) and Fruit (Resource)
    let city_idx = 0; // Using tile 0 as city

    // Create city
    let mut city = CityState::default();
    city.idx = city_idx;
    city.owner = 1;
    city._territory.push(city_idx);

    if let Some(tribe) = game.state.tribes.get_mut(&1) {
        tribe.stars = 10;
        tribe.cities.push(city);
        // Add Organization tech so we can harvest fruit
        tribe.tech_vanilla.push(TechnologyState {
            tech_type: TechnologyType::Organization,
            discovered: true,
            discovered_turn: 0,
        });
    }

    // Add Fruit to tile 0
    game.state.resources.insert(
        city_idx,
        Some(ResourceState {
            resource_type: ResourceType::Fruit,
        }),
    );

    // Ensure tile exists and is owned
    let mut tile = polyfish::states::TileState::default();
    tile.coords.idx = city_idx;
    tile.owner = 1;
    // Fruit needs Field or Forest? Fruit implies Field usually.
    tile.terrain_type = TerrainType::Field;
    // Fix: Set ruling city coords so get_city_owning_tile works
    tile.ruling_city_coords = Some(Coords::from_index(city_idx, game.state.settings.size));

    game.state.tiles.insert(city_idx, tile);

    assert_stars(&game, 10, "Initial");

    // Harvest move: Cost 2
    let move_ = HarvestMove::new(city_idx);
    let cost = 2; // Fruit harvest cost

    let undo = game.play_move(&move_).expect("Harvest failed");

    assert_stars(&game, 10 - cost, "After Harvest");

    // Verify resource consumed (removed from map)
    if game.state.resources.contains_key(&city_idx) {
        panic!("Resource not consumed!");
    }

    undo(&mut game.state);

    assert_stars(&game, 10, "After Undo");

    // Verify resource returned
    if !game.state.resources.contains_key(&city_idx) {
        panic!("Resource not restored!");
    }
}

fn test_build_undo() {
    println!("\n--- Test Build Undo ---");
    let mut game = setup_game();

    // Setup: 10 stars, 1 city
    let city_idx = 0;
    if let Some(tribe) = game.state.tribes.get_mut(&1) {
        tribe.stars = 10;
        let mut city = CityState::default();
        city.idx = city_idx;
        city.owner = 1;
        tribe.cities.push(city);
    }

    // Add tile
    let mut tile = polyfish::states::TileState::default();
    tile.coords.idx = city_idx;
    tile.owner = 1;
    // Set ruling city
    tile.ruling_city_coords = Some(Coords::from_index(city_idx, game.state.settings.size));
    game.state.tiles.insert(city_idx, tile);

    assert_stars(&game, 10, "Initial");

    // Build Road (Cost 2)
    // Need Roads tech? Usually yes. Assume checked by move generator, but `execute` might check too.
    // BuildMove checks `settings::structures::get_structure_setting`.
    // It does not explicitly check checks "owning tech" inside execute, usually.
    // Wait, create_structure checks? No.
    // Assuming we can build if we ask.

    let move_ = BuildMove::new(city_idx, StructureType::Road);
    let cost = 3; // Road cost is 3 in this codebase

    let undo = game.play_move(&move_).expect("Build Road failed");

    assert_stars(&game, 10 - cost, "After Build");

    // Verify structure exists
    if !game.state.structures.contains_key(&city_idx) {
        panic!("Structure not built!");
    }

    undo(&mut game.state);

    assert_stars(&game, 10, "After Undo");

    // Verify structure removed
    if game.state.structures.contains_key(&city_idx) {
        panic!("Structure not removed!");
    }
}

fn test_interaction_sequence() {
    println!("\n--- Test Interaction Sequence (Harvest -> Reward -> EndTurn) ---");
    let mut game = setup_game();

    // Setup: Tribe 1 (Imperius)
    let city_idx = 0;

    // Create city (Pre-primed for Level Up)
    let mut city = CityState::default();
    city.idx = city_idx;
    city.owner = 1;
    city.population = 1; // Needs 1 more for Level 2
    city.progress = 1;
    city.level = 1;
    city.production = 1;
    city._territory.push(city_idx);

    if let Some(tribe) = game.state.tribes.get_mut(&1) {
        tribe.stars = 10;
        tribe.cities.push(city);
        tribe.tech_vanilla.push(TechnologyState {
            tech_type: TechnologyType::Organization,
            discovered: true,
            discovered_turn: 0,
        });
    }

    // Add Fruit
    game.state.resources.insert(
        city_idx,
        Some(ResourceState {
            resource_type: ResourceType::Fruit,
        }),
    );

    // Tile
    let mut tile = polyfish::states::TileState::default();
    tile.coords.idx = city_idx;
    tile.owner = 1;
    tile.capital_of = 1; // Capital Bonus (+1 Prod)
    tile.terrain_type = TerrainType::Field;
    tile.ruling_city_coords = Some(Coords::from_index(city_idx, game.state.settings.size));
    game.state.tiles.insert(city_idx, tile);

    assert_stars(&game, 10, "Initial");

    // 1. Harvest (Cost 2)
    let move1 = HarvestMove::new(city_idx);
    let undo1 = game.play_move(&move1).expect("Harvest failed");

    assert_stars(&game, 8, "After Harvest");

    // Verify Level Up
    {
        let t = game.state.tribes.get(&1).unwrap();
        let c = t.cities.first().unwrap();
        if c.level != 2 {
            panic!("City did not level up! Level: {}", c.level);
        }
        // Check rewards pending
        if c.rewards.len() != 1 {
            // Should have 1 pending reward
            println!(
                "Warning: No pending reward found? (Maybe generate_reward_moves creates them, but city.rewards should track TAKEN rewards?)"
            );
            // Actually, city.rewards tracks TAKEN rewards.
            // generate_reward_moves checks if taken < required.
            // So if we leveled up, taken (0) < required (1).
            // So it's fine.
        }
    }

    // 2. Play Reward (Workshop)
    // We need to manually construct it as MCTS would.
    // RewardMove is in src/moves/reward.rs
    use polyfish::moves::reward::RewardMove;
    use polyfish::types::CityRewardType;

    let move_reward = RewardMove::new(city_idx, CityRewardType::Resources);
    let undo_reward = game.play_move(&move_reward).expect("Reward failed");

    // Resources adds 5 Stars.
    // Stars: 8 (After Harvest) + 5 = 13.
    assert_stars(&game, 13, "After Reward");

    // 3. End Turn (Player 1 -> Player 1)

    // Add Tribe 2
    let mut tribe2 = TribeState::default();
    tribe2.id = 2;
    tribe2.stars = 0;
    game.state.tribes.insert(2, tribe2);
    game.state.settings._max_tribe_count = 2;

    // End Turn (P1 -> P2)
    let move2 = EndTurnMove;
    let undo2 = game.play_move(&move2).expect("EndTurn failed");

    // End Turn (P2 -> P1)
    let move3 = EndTurnMove;
    let undo3 = game.play_move(&move3).expect("EndTurn P2 failed");

    // Now P1's turn again.
    // Production calculation:
    // Base: 1
    // Capital: +1
    // Level Up Bonus: +1
    // Workshop: 0 (Not chosen)
    // Total: 3.
    // Stars: 13 + 3 = 16.

    assert_stars(&game, 16, "After P1 Turn Cycle");

    // UNDO Chain
    undo3(&mut game.state);
    // Should be P2.
    if game.current_player_id() != 2 {
        panic!("Undo 3 failed to restore P2 turn");
    }

    undo2(&mut game.state);
    assert_stars(&game, 13, "After Undo EndTurn");

    undo_reward(&mut game.state);
    assert_stars(&game, 8, "After Undo Reward");

    undo1(&mut game.state);
    // Should be P1. Stars 10.
    assert_stars(&game, 10, "After Undo Harvest");
}

fn setup_game() -> Game {
    let mut game = Game::new();
    // Manually insert Tribe 1 (Imperius)
    let mut tribe1 = polyfish::states::TribeState::default();
    tribe1.tribe_type = TribeType::Imperius;
    game.state.tribes.insert(1, tribe1);

    // Manually insert Tribe 2 (Bardur) for turn cycling
    let mut tribe2 = polyfish::states::TribeState::default();
    tribe2.tribe_type = TribeType::Bardur;
    game.state.tribes.insert(2, tribe2);

    game.state.settings.current_player_turn_id = 1;
    game.state.settings._max_tribe_count = 2;

    game
}

fn test_summon_undo() {
    println!("\n--- Test Summon Undo ---");
    let mut game = setup_game();

    // Setup: Give stars to Imperius (P1)
    if let Some(tribe) = game.state.tribes.get_mut(&1) {
        tribe.stars = 10;
        // Ensure space to spawn
        tribe.units.clear();
    }

    assert_stars(&game, 10, "Initial");

    let move_ = SummonMove::new(0, UnitType::Warrior); // Warrior costs 2
    let cost = 2;

    // We also need a Tile 0 because Summon checks tiles
    let mut tile = polyfish::states::TileState::default();
    tile.coords.idx = 0;
    tile._unit_owner_id = None;
    game.state.tiles.insert(0, tile);

    // Play
    let undo = game.play_move(&move_).expect("Move failed");
    assert_stars(&game, 10 - cost, "After Summon");

    // Undo
    undo(&mut game.state);
    assert_stars(&game, 10, "After Undo");
}

fn test_research_undo() {
    println!("\n--- Test Research Undo ---");
    let mut game = setup_game();

    // Setup
    if let Some(tribe) = game.state.tribes.get_mut(&1) {
        tribe.stars = 10;
        tribe.units.clear();
        tribe.cities.push(polyfish::states::CityState::default()); // 1 city
    }

    // Organization costs 5 (base) for 1 city?
    // Wait, need to check cost formula. 1 city -> Tier 1 cost = 5.

    assert_stars(&game, 10, "Initial");

    let move_ = ResearchMove::new(TechnologyType::Organization);
    let _cost = 5; // Assuming standard

    // Play
    let undo = game.play_move(&move_).expect("Move failed");

    // Check stars decreased
    let tribe = game.current_tribe().unwrap();
    let stars_after = tribe.stars;
    println!("Stars after research: {}", stars_after);
    if stars_after >= 10 {
        panic!("Stars did not decrease!");
    }

    // Undo
    undo(&mut game.state);
    assert_stars(&game, 10, "After Undo");
}

fn test_turn_cycling_undo() {
    println!("\n--- Test Turn Cycling Undo ---");
    let mut game = setup_game();

    // Set stars for next turn verification
    if let Some(tribe) = game.state.tribes.get_mut(&1) {
        tribe.stars = 10;
    }

    assert_stars(&game, 10, "Initial");

    let move_ = EndTurnMove;

    // Play EndTurn -> transitions to Player 2
    let undo = game.play_move(&move_).expect("EndTurn failed");

    if game.current_player_id() == 1 {
        panic!("Did not change player!");
    }

    // Undo
    undo(&mut game.state);

    if game.current_player_id() != 1 {
        panic!("Did not return to Player 1!");
    }
    assert_stars(&game, 10, "After Undo");
}

fn test_game_over_undo() {
    println!("\n--- Test Game Over Undo (Turn 29->30) ---");
    let mut game = setup_game();
    game.state.settings.max_turns = 30;

    // Fast forward to Turn 30
    // Actually, we can just set the turn.
    game.state.settings.turn = 30;
    game.state.settings.mode = ModeType::Perfection;

    // Player 1's turn at Turn 30.
    // If we EndTurn -> Turn 30 ends?
    // In Polytopia, Turn 30 is the last playable turn. After P1 ends, P2 plays.
    // When last player ends turn 30 -> Game Over?
    // Let's set it so we are at the edge.

    // Let's assume 2 players.
    // P1 Turn 30.

    if let Some(tribe) = game.state.tribes.get_mut(&1) {
        tribe.stars = 10;
    }

    assert_stars(&game, 10, "Initial T30");

    let move_ = EndTurnMove;

    // This transitions to Player 2, T30. Not Game Over yet.
    let undo = game.play_move(&move_).expect("P1 EndTurn failed");
    undo(&mut game.state);
    assert_stars(&game, 10, "After P1 Undo");

    // Now setup valid Game Over transition.
    // If P2 (Last player) Ends Turn 30 -> Game Over
    game.state.settings.turn = 30;
    game.state.settings.current_player_turn_id = 2; // P2 (Imperius Mirror?)
    game.state.settings._max_tribe_count = 2;

    // Ensure P2 exists
    if !game.state.tribes.contains_key(&2) {
        // Create dummy p2 if needed for `play_move`
        // Default generator usually makes 2 tribes.
    }

    let undo_p2 = game.play_move(&move_);
    // Should trigger Game Over logic inside end_turn

    if let Some(u) = undo_p2 {
        println!("Move executed. Game Over state: {}", game.is_game_over());
        u(&mut game.state);
        println!("Undone. Game Over state: {}", game.is_game_over());
    } else {
        println!("Game Over immediately prevented move?");
    }

    // Check if simulate_move handles it differently (which MCTS uses)
    // MCTS uses simulate_move which calls end_turn repeatedly until back to POV

    println!("Testing simulate_move EndTurn loop...");
    game.state.settings.current_player_turn_id = 1;
    game.state.settings.turn = 30;
    game.state.settings._game_over = false;

    // reset stars
    if let Some(tribe) = game.state.tribes.get_mut(&1) {
        tribe.stars = 10;
    }

    let move_ = EndTurnMove;
    // simulate_move EndTurn -> should cycle P1 -> P2 -> (Check Game Over? or Cycle?) -> P1

    // If Turn 30 P2 ends turn -> Turn 31? Or Game Over?
    // In Perfection, after T30 ends, game over.
    // So simulate_move loop:
    // 1. P1 ends turn -> P2. (Not original player)
    // 2. Loop continues.
    // 3. P2 ends turn -> Game Over?
    //    If Game Over, loop breaks?

    let undo_sim = game.simulate_move(&move_);

    if let Some(u) = undo_sim {
        println!("Simulate executed.");
        // If it looped back to P1, we expect stars to be potentially higher (production)
        // But wait, simulate_move end_turn logic:
        // "Keep ending turns until we're back at original player"

        // If P2 causes Game Over, the loop breaks because `!self.state.settings._game_over` condition fails.
        // So we are stuck at P2 (or Game Over state).
        // Current player is NOT P1.

        println!("Current Player: {}", game.current_player_id());
        println!("Game Over: {}", game.is_game_over());

        u(&mut game.state);

        assert_stars(&game, 10, "After Simulate Undo");
        println!("Current Player after undo: {}", game.current_player_id());
    } else {
        println!("Simulate failed.");
    }
}

fn test_capture_ruin_determinism() {
    println!("\n--- Test Capture Ruin Determinism ---");
    let mut game = setup_game();

    let ruin_idx = 1;

    // Setup Tribe with unit at ruin
    if let Some(tribe) = game.state.tribes.get_mut(&1) {
        tribe.stars = 0;
        // Add unit at ruin_idx
        let mut unit = polyfish::states::UnitState::default();
        unit.owner = 1;
        unit.coords = Coords::from_index(ruin_idx, game.state.settings.size);
        unit.unit_type = UnitType::Warrior;
        tribe.units.push(unit);

        // Ensure unit on tile
        let mut tile = TileState::default();
        tile.coords = Coords::from_index(ruin_idx, game.state.settings.size);
        tile._unit_owner_id = Some(1);
        game.state.tiles.insert(ruin_idx, tile);

        // Add Ruin Structure
        let structure = StructureState {
            structure_type: StructureType::Ruin,
            level: 1,
            founded: 0,
        };
        game.state.structures.insert(ruin_idx, Some(structure));
    }

    // We will play Capture -> Undo -> Capture -> Undo multiple times.
    // If the results differ (e.g. stars change differently), we have non-determinism.

    let move_ = polyfish::moves::CaptureMove::new(ruin_idx);

    let mut results = Vec::new();

    for _ in 0..10 {
        let undo = game.play_move(&move_).expect("Capture failed");
        let stars = game.current_tribe().unwrap().stars;
        let tech_count = game.current_tribe().unwrap().tech_vanilla.len();

        results.push((stars, tech_count));
        // println!("Run {}: Stars={}, Tech={}", i, stars, tech_count);

        undo(&mut game.state);

        // Verify reset
        if game.current_tribe().unwrap().stars != 0 {
            panic!("Undo failed to reset stars!");
        }
    }

    // Check for variance
    let first = results[0];
    let all_same = results.iter().all(|&r| r == first);

    if !all_same {
        println!("[FAIL] Capture Ruin is NON-DETERMINISTIC!");
        println!("Results: {:?}", results);
    } else {
        println!("[PASS (?)] Capture Ruin appears deterministic (could be luck).");
    }
}

fn test_capture_ruin_stacking() {
    println!("\n--- Test Capture Ruin Stacking ---");
    let mut game = setup_game();
    let ruin_idx = 1;

    // Setup Tribe with unit at ruin
    if let Some(tribe) = game.state.tribes.get_mut(&1) {
        tribe.stars = 0;
        // Add unit at ruin_idx
        let mut unit = polyfish::states::UnitState::default();
        unit.owner = 1;
        unit.coords = Coords::from_index(ruin_idx, game.state.settings.size);
        unit.unit_type = UnitType::Warrior;
        tribe.units.push(unit);

        // Ensure unit on tile
        let mut tile = TileState::default();
        tile.coords = Coords::from_index(ruin_idx, game.state.settings.size);
        tile._unit_owner_id = Some(1);
        game.state.tiles.insert(ruin_idx, tile);

        // Add Ruin Structure
        let structure = StructureState {
            structure_type: StructureType::Ruin,
            level: 1,
            founded: 0,
        };
        game.state.structures.insert(ruin_idx, Some(structure));
    }

    // We iterate seeds to trigger the Unit reward
    let mut unit_reward_triggered = false;
    let move_ = polyfish::moves::CaptureMove::new(ruin_idx);

    for i in 0..1000 {
        game.state.settings.seed = i as i64;

        // We know from logic that if we get a Unit reward, it spawns a Veteran Swordsman.
        // Let's check if the move executes cleanly.
        if let Some(undo) = game.play_move(&move_) {
            let tribe = game.state.tribes.get(&1).unwrap();
            let units_at_ruin = tribe
                .units
                .iter()
                .filter(|u| u.coords.idx == ruin_idx)
                .count();

            // If we got a unit reward, units_at_ruin should be 1.
            // (The original should be pushed).

            if units_at_ruin > 1 {
                panic!(
                    "STACKING DETECTED! Seed {} produced {} units at {}",
                    i, units_at_ruin, ruin_idx
                );
            }

            // Check if we actually got a unit reward (swordsman check)
            let unit_at_ruin = tribe
                .units
                .iter()
                .find(|u| u.coords.idx == ruin_idx)
                .unwrap();
            if unit_at_ruin.unit_type == UnitType::Swordsman {
                if unit_at_ruin.veteran {
                    unit_reward_triggered = true;
                } else {
                    println!("Seed {}: Found Swordsman but NOT veteran?", i);
                }
            } else if unit_at_ruin.unit_type != UnitType::Warrior {
                println!("Seed {}: Found {:?} at ruin.", i, unit_at_ruin.unit_type);
            }

            undo(&mut game.state);
        }
    }

    if unit_reward_triggered {
        println!("[PASS] Stacking check passed (Unit reward triggered & handled).");
    } else {
        println!("[WARNING] Unit reward was NOT triggered in 1000 seeds. Test inconclusive?");
    }
}
