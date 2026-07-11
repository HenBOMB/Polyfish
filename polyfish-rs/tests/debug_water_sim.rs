use polyfish::ai::eval_server::{Evaluator, InlineEvalHandle};
use polyfish::ai::mcts_zero::ZeroMctsAgent;
use polyfish::ai::network::PolyZeroNet;
use polyfish::game::Game;
use polyfish::types::{MapSize, MapType, TechnologyType, TribeType};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() -> anyhow::Result<()> {
    // Config
    let mcts_iters = 2;
    let seed = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();

    println!("Starting debug water simulation...");
    println!("Seed: {}", seed);

    // Init Game using MapGen - Archipelago for water map
    let gen_settings = polyfish::mapgen::MapGenSettings {
        size: MapSize::Normal, // 16x16
        map_type: MapType::Archipelago,
        tribes: vec![TribeType::Kickoo, TribeType::Kickoo], // Kickoo starts with Fishing
        seed: seed as i64,
        ..Default::default()
    };
    let mut state = polyfish::mapgen::generate(gen_settings);

    // Post-load initialization
    let size = state.settings.size;
    for tile in state.tiles.values_mut() {
        tile.coords.compute_idx(size);
        if let Some(ref mut rc) = tile.ruling_city_coords {
            rc.compute_idx(size);
        }
    }
    for tribe in state.tribes.values_mut() {
        for unit in &mut tribe.units {
            unit.coords.compute_idx(size);
            unit.prev_coords.compute_idx(size);
            if let Some(ref mut hc) = unit.home_coords {
                hc.compute_idx(size);
            }
        }
        tribe.starting_tile_coords.compute_idx(size);

        // Grant Sailing + Fishing to ensure ports are possible
        // Kickoo has Fishing by default, but let's be sure.
        let fishing = polyfish::states::TechnologyState {
            tech_type: TechnologyType::Fishing,
            discovered: true,
            discovered_turn: 0,
        };
        let sailing = polyfish::states::TechnologyState {
            tech_type: TechnologyType::Sailing,
            discovered: true,
            discovered_turn: 0,
        };

        if !polyfish::settings::technology::has_technology(
            &tribe.tech_vanilla,
            TechnologyType::Fishing,
        ) {
            tribe.tech_vanilla.push(fishing);
        }
        if !polyfish::settings::technology::has_technology(
            &tribe.tech_vanilla,
            TechnologyType::Sailing,
        ) {
            tribe.tech_vanilla.push(sailing);
        }

        // Give extra stars to encourage building ports immediately
        tribe.stars = 30; // 10 for port, 5 for warrior, etc.
    }
    polyfish::functions::sync_scores(&mut state);

    let mut game = Game::new();
    game.state = state;
    game.state.settings.mode = polyfish::types::ModeType::Perfection;
    game.state.settings.max_turns = 50; // Long enough to build network

    // Ensure exploration
    let pov_id = game.state.settings.current_player_turn_id;
    let _ = polyfish::actions::update_exploration(&mut game.state, pov_id);

    // Load network (Random for debug)
    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;
    let device = Device::Cpu;
    let vb = VarBuilder::zeros(DType::F32, &device);
    let network = Arc::new(PolyZeroNet::new(vb)?); // 30x30 max size assumed in network anyway
    let evaluator = Evaluator::Inline(InlineEvalHandle::new(network));

    let agent = ZeroMctsAgent::new(&evaluator, mcts_iters);

    let mut turn = 0;
    while !polyfish::functions::is_game_over(&game.state) && turn < 1000 {
        let pov = game.state.settings.current_player_turn_id;

        println!(
            "\n--- Turn {} (Player {}) ---",
            game.state.settings.turn, pov
        );
        if let Some(tribe) = game.state.tribes.get(&pov) {
            println!(
                "Stars: {}, Cities: {}, Score: {}",
                tribe.stars,
                tribe.cities.len(),
                tribe.score
            );
        }

        // MCTS Search
        let (best_move, _) = agent.select_move_with_stats(&mut game);

        if let Some(m) = best_move {
            println!("🚀 Executing: {}", m.describe(&game.state));

            if game.play_move(m.as_ref()).is_some() {
                // Success
            } else {
                println!("FATAL ERROR executing move: {}", m.describe(&game.state));
                // Dump state info
                if let Some(tribe) = game.state.tribes.get(&pov) {
                    println!("Final Tribe State - Stars: {}", tribe.stars);
                }
                break;
            }
        } else {
            println!("No moves available.");
            break;
        }
        turn += 1;
    }

    Ok(())
}
