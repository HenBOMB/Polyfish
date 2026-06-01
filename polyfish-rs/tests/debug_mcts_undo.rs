use polyfish::coords::Coords;
use polyfish::game::Game;
use polyfish::moves::EndTurnMove;
use polyfish::moves::harvest::HarvestMove;
use polyfish::states::{CityState, ResourceState, TechnologyState, TileState, TribeState};
use polyfish::types::{CityRewardType, ResourceType, TechnologyType, TerrainType, TribeType};

fn main() {
    fuzz_test_p2_moves();
}

fn fuzz_test_p2_moves() {
    println!("--- Fuzzing P2 Moves after P1 EndTurn ---");

    // Setup
    let mut game = Game::new();

    // P1 (Imperius)
    let p1 = 1;
    let mut tribe1 = TribeState::default();
    tribe1.id = p1;
    tribe1.tribe_type = TribeType::Imperius;
    tribe1.stars = 10;

    // P1 City at 0
    let mut city1 = CityState::default();
    city1.idx = 0;
    city1.owner = p1;
    city1.population = 1;
    city1.level = 1;
    city1.production = 1;
    city1._territory.push(0);
    tribe1.cities.push(city1);

    // P2 (Bardur)
    let p2 = 2;
    let mut tribe2 = TribeState::default();
    tribe2.id = p2;
    tribe2.tribe_type = TribeType::Bardur;
    tribe2.stars = 10;

    // P2 City at 100
    let mut city2 = CityState::default();
    city2.idx = 100;
    city2.owner = p2;
    city2.production = 2;
    city2.level = 2;
    city2.rewards.insert(0, CityRewardType::Workshop); // Resolve pending reward
    city2._territory.push(100);
    city2._territory.push(101); // Add tile with resources
    // city2.capital_of -- removed

    // Give P2 Technology
    tribe2.tech_vanilla.push(TechnologyState {
        tech_type: TechnologyType::Organization,
        discovered: true,
        discovered_turn: 0,
    });
    tribe2.tech_vanilla.push(TechnologyState {
        tech_type: TechnologyType::Forestry,
        discovered: true,
        discovered_turn: 0,
    });

    tribe2.cities.push(city2);

    game.state.tribes.insert(p1, tribe1);
    game.state.tribes.insert(p2, tribe2);

    game.state.settings.current_player_turn_id = p1;
    game.state.settings._max_tribe_count = 2;
    game.state.settings.size = 11;

    // Tiles
    // P1 Tile
    let mut t1 = TileState::default();
    t1.coords.idx = 0;
    t1.owner = p1;
    t1.terrain_type = TerrainType::Field;
    t1.ruling_city_coords = Some(Coords::from_index(0, 11));
    game.state.tiles.insert(0, t1);

    // P2 Tiles (City + Forest for Clear/Harvest)
    let mut t2 = TileState::default();
    t2.coords.idx = 100;
    t2.owner = p2;
    t2.terrain_type = TerrainType::Forest;
    t2.ruling_city_coords = Some(Coords::from_index(100, 11));
    t2.capital_of = p2;
    game.state.tiles.insert(100, t2);

    let mut t3 = TileState::default();
    t3.coords.idx = 101;
    t3.owner = p2;
    t3.terrain_type = TerrainType::Forest;
    t3.ruling_city_coords = Some(Coords::from_index(100, 11));
    game.state.tiles.insert(101, t3);

    // Resource for P2 (Fruit)
    game.state.resources.insert(
        101,
        Some(ResourceState {
            resource_type: ResourceType::Fruit,
        }),
    );

    // P1 Plays Harvest
    let m1 = HarvestMove::new(0);
    game.state.resources.insert(
        0,
        Some(ResourceState {
            resource_type: ResourceType::Fruit,
        }),
    );

    let _ = game.play_move(&m1).expect("P1 Harvest");

    // P1 Ends Turn
    let m2 = EndTurnMove;
    let _ = game.play_move(&m2).expect("P1 EndTurn");

    if game.current_player_id() != p2 {
        panic!("Turn not P2");
    }

    // Check P2 Stars
    let p2_stars = game.current_tribe().unwrap().stars;
    println!("P2 Start Stars: {}", p2_stars);

    // Generate Legal Moves for P2
    println!("Generating P2 moves...");
    let moves = game.legal_moves();
    println!("Found {} moves.", moves.len());

    for (i, m) in moves.iter().enumerate() {
        println!("Testing Move {}: {}", i, m.describe(&game.state));

        let start_stars = game.current_tribe().unwrap().stars;

        // Play
        if let Some(undo) = game.play_move(m.as_ref()) {
            let mid_stars = game.current_tribe().unwrap().stars;
            println!("  Executed. Stars {} -> {}", start_stars, mid_stars);

            if mid_stars < 0 {
                println!("  WARNING: Negative stars!");
            }

            // Undo
            undo(&mut game.state);
            let end_stars = game.current_tribe().unwrap().stars;
            println!("  Undone. Stars -> {}", end_stars);

            if end_stars != start_stars {
                panic!(
                    "Star Corruption detected! Move: {}. Expected {}, Found {}",
                    m.describe(&game.state),
                    start_stars,
                    end_stars
                );
            }
        } else {
            println!("  Failed to execute legal move?");
        }
    }

    println!("Fuzzing Complete. No leaks found in generated moves.");
}
