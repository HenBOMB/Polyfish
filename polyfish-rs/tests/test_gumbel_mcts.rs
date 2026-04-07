use candle_core::Device;
use candle_nn::VarMap;
use polyfish::ai::gumbel_mcts::GumbelMctsAgent;
use polyfish::ai::network::PolyZeroNet;
use polyfish::game::Game;
use polyfish::mapgen::{generate, MapGenSettings};
use polyfish::types::{MapSize, MapType, TribeType};

#[test]
fn test_gumbel_mcts_basic() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vs = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
    let network = PolyZeroNet::new(vs).unwrap();

    let mut game = Game::new();
    let gen_settings = MapGenSettings {
        size: MapSize::Tiny,
        map_type: MapType::Drylands,
        tribes: vec![TribeType::Imperius, TribeType::Imperius],
        seed: 42,
        ..Default::default()
    };
    game.state = generate(gen_settings);
    game.post_load();

    let agent = GumbelMctsAgent {
        network: &network,
        iterations: 40, // Small for fast testing
        k: 4,           // Small for fast testing
        batch_size: 8,
    };

    println!("Running Gumbel MCTS...");
    let best_move = agent.select_move(&mut game);
    
    assert!(best_move.is_some(), "Gumbel MCTS failed to select a move");
    println!("Selected move: {}", best_move.unwrap().describe(&game.state));
}

#[test]
fn test_gumbel_mcts_sequential_halving() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vs = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
    let network = PolyZeroNet::new(vs).unwrap();

    let mut game = Game::new();
    let gen_settings = MapGenSettings {
        size: MapSize::Tiny,
        map_type: MapType::Drylands,
        tribes: vec![TribeType::Imperius, TribeType::Imperius],
        seed: 123,
        ..Default::default()
    };
    game.state = generate(gen_settings);
    game.post_load();

    // Use k=8 to ensure at least 3 phases (log2(8)=3)
    let agent = GumbelMctsAgent {
        network: &network,
        iterations: 64,
        k: 8,
        batch_size: 16,
    };

    let best_move = agent.select_move(&mut game);
    assert!(best_move.is_some());
}
