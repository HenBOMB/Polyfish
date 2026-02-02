use candle_core::{Device, Tensor};
use polyfish::ai::features::{self, state_to_tensor};
use polyfish::ai::mapper::ActionMapper;
use polyfish::ai::mcts_zero::ZeroMctsAgent;
use polyfish::ai::network::PolyZeroNet;
use polyfish::game::Game;
use polyfish::types::MapSize;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() -> anyhow::Result<()> {
    // Config
    let num_games = 10;
    let mcts_iters = 10;
    let device = Device::Cpu;

    // Load existing model if available
    let model_path = "model.safetensors";
    let mut varmap = candle_nn::VarMap::new();

    let network = if std::path::Path::new(model_path).exists() {
        println!("Loading existing model from {}", model_path);
        varmap.load(model_path)?;
        PolyZeroNet::new(candle_nn::VarBuilder::from_varmap(
            &varmap,
            candle_core::DType::F32,
            &device,
        ))?
    } else {
        println!("Starting with new random model.");
        PolyZeroNet::new(candle_nn::VarBuilder::zeros(
            candle_core::DType::F32,
            &device,
        ))?
    };

    let mut collected_states: Vec<Tensor> = Vec::new();
    let mut collected_policies: Vec<f32> = Vec::new(); // Flattened policies
    let mut collected_values: Vec<f32> = Vec::new();

    println!(
        "Starting Self-Play for {} games in PERFECTION mode...",
        num_games
    );

    for i in 0..num_games {
        // Init Game using MapGen
        let gen_settings = polyfish::mapgen::MapGenSettings {
            size: MapSize::Small,
            map_type: polyfish::types::MapType::Drylands,
            seed: i as u64 + SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
            ..Default::default()
        };
        let mut state = polyfish::mapgen::generate(gen_settings);

        // Post-load initialization similar to Game::post_load
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
        }
        polyfish::functions::sync_scores(&mut state);

        let mut game = Game::new();
        game.state = state;
        game.state.settings.mode = polyfish::types::ModeType::Perfection;
        game.state.settings.max_turns = 30;

        // Ensure exploration
        let pov_id = game.state.settings.current_player_turn_id;
        polyfish::actions::update_exploration(&mut game.state, pov_id);

        let agent = ZeroMctsAgent::new(&network, mcts_iters);

        // Store (StateTensor, PolicyVec, PlayerId)
        let mut game_history: Vec<(Tensor, Vec<f32>, polyfish::states::PlayerId)> = Vec::new();

        let mut turn = 0;
        while !polyfish::functions::is_game_over(&game.state) && turn < 500 {
            let pov = game.state.settings.current_player_turn_id;

            // Get state tensor
            let state_t = match state_to_tensor(&game.state, pov) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("Error creating tensor: {:?}", e);
                    break;
                }
            };

            // MCTS Search
            let (best_move, policy) = agent.select_move_with_stats(&mut game);

            if let Some(m) = best_move {
                game_history.push((state_t, policy, pov));
                let _ = game.play_move(m.as_ref());
                print!(".");
                use std::io::Write;
                std::io::stdout().flush()?;
            } else {
                // If MCTS returns None, something is wrong or game is stuck
                println!("Warning: MCTS returned None at turn {}", turn);
                break;
            }
            turn += 1;
        }

        // Determine scores for backprop
        let mut scores: HashMap<i32, i32> = HashMap::new();
        for (id, t) in &game.state.tribes {
            scores.insert(*id, t.score);
        }

        let winner = scores
            .iter()
            .max_by_key(|&(_, score)| score)
            .map(|(&id, _)| id)
            .unwrap_or(0);

        println!(
            "Game {} finished. Turns: {} | Scores: {:?}",
            i, turn, scores
        );

        // Backpropagate value
        for (state_t, policy, p_id) in game_history {
            match state_t.flatten_all() {
                Ok(flat_t) => collected_states.push(flat_t),
                Err(_) => continue,
            }
            collected_policies.extend_from_slice(&policy);

            // Simple win/loss for now
            let value = if p_id == winner { 1.0f32 } else { -1.0f32 };
            collected_values.push(value);
        }
    }

    // stack and save
    if !collected_states.is_empty() {
        let total_steps = collected_states.len();
        println!("Saving {} steps...", total_steps);

        let state_dim = features::NUM_CHANNELS * features::MAP_HEIGHT * features::MAP_WIDTH;

        // Concatenate all state tensors
        let states_tensor = Tensor::cat(&collected_states, 0)?;
        let states_tensor = states_tensor.reshape((total_steps, state_dim))?;

        let policies_tensor = Tensor::from_vec(
            collected_policies,
            (total_steps, ActionMapper::TOTAL_ACTIONS),
            &device,
        )?;
        let values_tensor = Tensor::from_vec(collected_values, (total_steps, 1), &device)?;

        let mut tensors = HashMap::new();
        tensors.insert("states".to_string(), states_tensor);
        tensors.insert("policies".to_string(), policies_tensor);
        tensors.insert("values".to_string(), values_tensor);

        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        let filename = format!("games_{}.safetensors", timestamp);
        candle_core::safetensors::save(&tensors, &filename)?;
        println!("Saved to {}", filename);
    }

    Ok(())
}
