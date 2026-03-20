use candle_core::Device;
use candle_nn::VarBuilder;
use polyfish::ai::mcts_zero::ZeroMctsAgent;
use polyfish::ai::network::PolyZeroNet;
use polyfish::game::Game;
use polyfish::types::{MoveType, TribeType};

#[test]
fn test_reproduce_summon_panic() {
    let mut game = Game::new();
    // Manual setup for a controlled state
    game.state.settings.current_player_turn_id = 1;
    game.state.settings.size = 11;

    // Add a tribe
    let mut tribe = polyfish::states::TribeState {
        id: 1,
        tribe_type: TribeType::Imperius,
        stars: 1, // ONLY 1 STAR
        ..Default::default()
    };

    // Add a city for that tribe
    let city = polyfish::states::CityState {
        idx: 24,
        level: 2,
        owner: 1,
        ..Default::default()
    };
    tribe.cities.push(city);

    game.state.tribes.insert(1, tribe);

    // Set tile owner
    if let Some(tile) = game.state.tiles.get_mut(&24) {
        tile.owner = 1;
    }

    // Warrior costs 2. Summon Warrior should NOT be legal with 1 star.
    println!("Checking if Warrior is legal with 1 star...");
    let moves = game.legal_moves();
    for m in &moves {
        if m.move_type() == MoveType::Summon {
            let desc = format!("{:?}", m);
            if desc.contains("Warrior") {
                panic!(
                    "BUG FOUND: Training Warrior is legal with 1 star! Move: {}",
                    desc
                );
            }
        }
    }
    println!("Warrior is NOT legal (Good).");

    // Now try MCTS
    let vs = VarBuilder::zeros(candle_core::DType::F32, &Device::Cpu);
    let network = PolyZeroNet::new(vs).unwrap();
    let agent = ZeroMctsAgent::new(&network, 10);

    // This should not panic
    println!("Running MCTS...");
    let (m, _) = agent.select_move_with_stats(&mut game);
    if let Some(mv) = m {
        println!("Selected move: {:?}", mv.move_type());
    }
}
