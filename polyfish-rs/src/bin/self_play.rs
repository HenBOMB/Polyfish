use candle_core::{Device, Tensor};
use polyfish::TribeType;
use polyfish::ai::brain::Brain;
use polyfish::ai::features::{self, GameFeatures, state_to_tensor};
use polyfish::ai::mapper::DecomposedMapper;
use polyfish::ai::network::PolyZeroNet;
use polyfish::game::Game;
use polyfish::states::PlayerId;
use polyfish::types::MapSize;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Decomposed policy probability distributions for a single step
struct DecomposedPolicyData {
    action_type: Vec<f32>,    // [11]
    source_spatial: Vec<f32>, // [H * W]
    target_spatial: Vec<f32>, // [H * W]
    move_option: Vec<f32>,    // [192]
}

/// Result from a single game - contains all data needed for training
struct GameResult {
    // Added: current_spt and current_units for each step
    history: Vec<(GameFeatures, DecomposedPolicyData, PlayerId)>,
    scores: HashMap<i32, i32>,
    moves: usize,
    winner_score: i32,
}

/// Play a single game and return the result
fn play_single_game(
    network1: &PolyZeroNet,
    network2: &PolyZeroNet, // Added network2
    mcts_iters: usize,
    game_idx: usize,
    seed: i64,
    tribes: Vec<TribeType>,
) -> Option<GameResult> {
    // Init Game using MapGen
    let gen_settings = polyfish::mapgen::MapGenSettings {
        size: MapSize::Tiny,
        map_type: polyfish::types::MapType::Drylands,
        tribes: tribes.clone(),
        seed,
        ..Default::default()
    };
    eprintln!(
        "[Game {}] Started with seed: {} Tribes: {:?}",
        game_idx, seed, gen_settings.tribes
    );

    let mut game = Game::new();
    game.state = polyfish::mapgen::generate(gen_settings);
    game.state.settings.mode = polyfish::types::ModeType::Perfection;
    game.state.settings.max_turns = 20;
    game.post_load();

    // Create two agents (they might share the same network, or be different)
    let agent1 = Brain::new(network1, mcts_iters);
    let agent2 = Brain::new(network2, mcts_iters);

    // Game Loop
    let mut game_history: Vec<(GameFeatures, DecomposedPolicyData, PlayerId)> = Vec::new();

    let current_scores: Vec<(PlayerId, i32)> = game
        .state
        .tribes
        .iter()
        .map(|(id, t)| (*id, t.score))
        .collect();

    eprintln!(
        "[Game {}]: Turn: {} Scores: {:?}",
        game_idx, game.state.settings.turn, current_scores
    );

    let mut move_count = 0;
    while !polyfish::functions::is_game_over(&game.state) {
        if move_count > 50000 {
            // Reduced for safety
            eprintln!(
                "[Game {}] Move count exceeded 50000 (Safety Break)",
                game_idx
            );
            break;
        }

        let pov = game.state.settings.current_player_turn_id;

        // Get state tensor
        let current_network = if pov == 1 { network1 } else { network2 };
        let device = current_network.device();
        let state_t = state_to_tensor(&game.state, pov, &device)
            .expect("BUG: Failed to create state tensor - game state is invalid");

        // MCTS Search - use the correct agent
        let current_agent = if pov == 1 { &agent1 } else { &agent2 };
        let (best_move, move_visits) = current_agent.think_decomposed(&mut game);

        let map_size = game.state.settings.size as usize;

        // Initialize probability distributions
        let fixed_map_width = features::MAP_SIZE;
        let fixed_spatial_size = features::MAP_SIZE * fixed_map_width;

        let mut p_action = vec![0.0; 11];
        let mut p_source = vec![0.0; fixed_spatial_size];
        let mut p_target = vec![0.0; fixed_spatial_size];
        let mut p_option = vec![0.0; 192]; // Unified option head (Expanded)

        let mut total_visits = 0.0;

        // Aggregate visits into distributions
        for mv in move_visits {
            total_visits += mv.visits;

            // Spatial and Option targets using DecomposedMapper
            let targets = DecomposedMapper::move_visit_to_targets(&mv, map_size);

            let action_idx = targets.action_type;
            if action_idx < p_action.len() {
                p_action[action_idx] += mv.visits;
            }

            if let Some(i) = targets.source_spatial {
                if i < p_source.len() {
                    p_source[i] += mv.visits;
                }
            }

            if let Some(i) = targets.target_spatial {
                if i < p_target.len() {
                    p_target[i] += mv.visits;
                }
            }

            if let Some(i) = targets.target_type {
                if i < p_option.len() {
                    p_option[i] += mv.visits;
                }
            }
        }

        // Normalize
        if total_visits > 0.0 {
            for x in &mut p_action {
                *x /= total_visits;
            }
            for x in &mut p_source {
                *x /= total_visits;
            }
            for x in &mut p_target {
                *x /= total_visits;
            }
            // ... (others)
        }

        let policy_data = DecomposedPolicyData {
            action_type: p_action,
            source_spatial: p_source,
            target_spatial: p_target,
            move_option: p_option,
        };

        if let Some(m) = best_move {
            game_history.push((state_t, policy_data, pov));
            if move_count > 0 && move_count % 10 == 0 {
                // let current_scores: Vec<(PlayerId, i32)> = game
                //     .state
                //     .tribes
                //     .iter()
                //     .map(|(id, t)| (*id, t.score))
                //     .collect();
                eprintln!(
                    "[Game {}]: Turn: {} Player: {} Move: {}",
                    game_idx,
                    game.state.settings.turn,
                    pov,
                    m.describe(&game.state),
                    // current_scores
                );
            }
            let _ = game.play_move(m.as_ref());
        } else {
            break;
        }
        move_count += 1;
    }

    // Determine scores
    let mut scores: HashMap<i32, i32> = HashMap::new();
    for (id, t) in &game.state.tribes {
        scores.insert(*id, t.score);
    }

    let (winner_id, winner_score) = scores
        .iter()
        .max_by_key(|&(_, score)| score)
        .map(|(&id, &score)| (id, score))
        .unwrap_or((0, 0));

    eprintln!(
        "[Game {}] Finished. Moves: {} | Winner: {} (Score: {})",
        game_idx, move_count, winner_id, winner_score
    );

    Some(GameResult {
        history: game_history,
        scores,
        moves: move_count,
        winner_score,
    })
}

fn main() -> anyhow::Result<()> {
    use clap::Parser;

    #[derive(Parser, Debug)]
    #[command(author, version, about, long_about = None)]
    struct Args {
        /// Number of games to play
        #[arg(long, default_value_t = 10)]
        num_games: usize,

        /// MCTS iterations per move
        #[arg(long, default_value_t = 50)]
        mcts_iters: usize,

        /// Optional opponent model path (if not set, plays against self)
        #[arg(long)]
        opponent: Option<String>,

        /// First tribe (optional, defaults to random)
        #[arg(long)]
        tribe1: Option<String>,

        /// Second tribe (optional, defaults to random)
        #[arg(long)]
        tribe2: Option<String>,
    }

    let args = Args::parse();

    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("Using device: {:?}", device);

    // Load Main Model (P1)
    let model_path = "model.safetensors";
    let mut varmap = candle_nn::VarMap::new();

    let network1 = if std::path::Path::new(model_path).exists() {
        println!("Loading main model from {}", model_path);
        varmap.load(model_path)?;
        PolyZeroNet::new(candle_nn::VarBuilder::from_varmap(
            &varmap,
            candle_core::DType::F32,
            &device,
        ))?
    } else {
        panic!(
            "Model file {} not found! Please run init_model.py first.",
            model_path
        );
    };
    let network1 = Arc::new(network1);

    // Load Opponent Model (P2) - Defaults to same as P1
    let network2 = if let Some(opp_path) = args.opponent {
        println!("Loading opponent model from {}", opp_path);
        let mut varmap2 = candle_nn::VarMap::new();
        varmap2.load(&opp_path)?;
        let net = PolyZeroNet::new(candle_nn::VarBuilder::from_varmap(
            &varmap2,
            candle_core::DType::F32,
            &device,
        ))?;
        Arc::new(net)
    } else {
        println!("No opponent specified. Playing against self.");
        network1.clone()
    };

    let base_seed = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();

    println!(
        "Starting parallel self-play: {} games with {} MCTS iterations",
        args.num_games, args.mcts_iters
    );

    // Parse tribes from args or use random if not specified (placeholder, logic moved inside loop or done here)
    // Actually, user wants "per iteration it should pick only 2 and play those 20 games with only those tribes"
    // So we pick them once here.

    // Helper to parse or pick random
    let all_tribes = vec![
        TribeType::Imperius,
        TribeType::Bardur,
        TribeType::Oumaji,
        TribeType::Kickoo,
        TribeType::XinXi,
        TribeType::Zebasi,
        TribeType::AiMo,
        TribeType::Vengir,
        TribeType::Luxidoor, // Luxidoor is valid but maybe check others?
        TribeType::Quetzali,
        TribeType::Hoodrick,
        TribeType::Yadakk,
        // TribeType::Aquarion, TribeType::Elyrion, TribeType::Polaris, TribeType::Cymanti // Special tribes might be too diff for now? user said "strict selection" but "random". Let's stick to standard human tribes first?
        // User didn't specify subset, just "random strict selection".
        // Let's include all standard tribes.
    ];

    let t1 = if let Some(s) = &args.tribe1 {
        // We need a FromStr or manual matching since TribeType might not derive FromStr
        // For now, let's implement a quick helper or match.
        // Actually TribeType usually derives EnumString in other crates, let's assume we can match or defaults.
        // Let's do a simple match for safety as I don't see EnumString derived in view_file(types.rs) - wait I haven't seen types.rs
        // But mapgen used them.
        // Let's rely on standard debug print matching if needed, or better:
        // Let's just hardcode a parser here since we don't have FromStr confirmed.
        match s.to_lowercase().as_str() {
            "imperius" => TribeType::Imperius,
            "bardur" => TribeType::Bardur,
            "oumaji" => TribeType::Oumaji,
            "kickoo" => TribeType::Kickoo,
            "xinxi" => TribeType::XinXi,
            "zebasi" => TribeType::Zebasi,
            "aimo" => TribeType::AiMo,
            "vengir" => TribeType::Vengir,
            "luxidoor" => TribeType::Luxidoor,
            "quetzali" => TribeType::Quetzali,
            "hoodrick" => TribeType::Hoodrick,
            "yadakk" => TribeType::Yadakk,
            "aquarion" => TribeType::Aquarion,
            "elyrion" => TribeType::Elyrion,
            "polaris" => TribeType::Polaris,
            "cymanti" => TribeType::Cymanti,
            _ => {
                eprintln!("Unknown tribe {}, using Imperius", s);
                TribeType::Imperius
            }
        }
    } else {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        *all_tribes.choose(&mut rng).unwrap()
    };

    let t2 = if let Some(s) = &args.tribe2 {
        match s.to_lowercase().as_str() {
            "imperius" => TribeType::Imperius,
            "bardur" => TribeType::Bardur,
            "oumaji" => TribeType::Oumaji,
            "kickoo" => TribeType::Kickoo,
            "xinxi" => TribeType::XinXi,
            "zebasi" => TribeType::Zebasi,
            "aimo" => TribeType::AiMo,
            "vengir" => TribeType::Vengir,
            "luxidoor" => TribeType::Luxidoor,
            "quetzali" => TribeType::Quetzali,
            "hoodrick" => TribeType::Hoodrick,
            "yadakk" => TribeType::Yadakk,
            "aquarion" => TribeType::Aquarion,
            "elyrion" => TribeType::Elyrion,
            "polaris" => TribeType::Polaris,
            "cymanti" => TribeType::Cymanti,
            _ => {
                eprintln!("Unknown tribe {}, using Oumaji", s);
                TribeType::Oumaji
            }
        }
    } else {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        // Pick distinct from t1
        loop {
            let t = *all_tribes.choose(&mut rng).unwrap();
            if t != t1 {
                break t;
            }
        }
    };

    println!("Selected Tribes for this iteration: {:?} vs {:?}", t1, t2);
    let selected_tribes = vec![t1, t2];

    // Parallel game generation using rayon
    let results: Vec<GameResult> = (0..args.num_games)
        .into_par_iter()
        .filter_map(|i| {
            let seed = (base_seed + i as u64) as i64;
            // Play with (Net1, Net2)
            play_single_game(
                &network1,
                &network2,
                args.mcts_iters,
                i,
                seed,
                selected_tribes.clone(),
            )
        })
        .collect();

    // Aggregate results
    let mut collected_spatial_maps: Vec<Tensor> = Vec::new();
    let mut collected_player_states: Vec<Tensor> = Vec::new();

    // Decomposed policy targets (7 heads)
    let mut collected_action_type: Vec<Vec<f32>> = Vec::new();
    let mut collected_source_spatial: Vec<Vec<f32>> = Vec::new();
    let mut collected_target_spatial: Vec<Vec<f32>> = Vec::new();
    let mut collected_option: Vec<Vec<f32>> = Vec::new();

    let mut collected_values: Vec<f32> = Vec::new();

    let mut total_score = 0;
    let mut max_score = 0;
    let mut total_moves = 0;

    let mut p1_total = 0;
    let mut p2_total = 0;
    let mut p1_count = 0;
    let mut p2_count = 0;

    for result in results {
        total_score += result.winner_score;
        total_moves += result.moves;
        if result.winner_score > max_score {
            max_score = result.winner_score;
        }

        for (id, score) in &result.scores {
            if *id == 1 {
                p1_total += score;
                p1_count += 1;
            } else if *id == 2 {
                p2_total += score;
                p2_count += 1;
            }
        }

        // Backpropagate value
        for (features, policy_data, p_id) in result.history {
            let flat_map = features
                .spatial_map
                .flatten_all()
                .expect("BUG: Failed to flatten spatial map tensor");
            collected_spatial_maps.push(flat_map);

            let flat_player = features
                .player_state
                .flatten_all()
                .expect("BUG: Failed to flatten player state tensor");
            collected_player_states.push(flat_player);

            collected_action_type.push(policy_data.action_type);
            collected_source_spatial.push(policy_data.source_spatial);
            collected_target_spatial.push(policy_data.target_spatial);
            collected_option.push(policy_data.move_option);

            // Calculate Value based on FINAL OUTCOME (Win/Loss/Score Diff)
            // This is crucial: Value Head learns "Who Wins", not "What Heuristic Thinks"
            let my_score = result.scores.get(&p_id).copied().unwrap_or(0) as f32;
            let opponent_score = result
                .scores
                .iter()
                .filter(|(id, _)| **id != p_id)
                .map(|(_, score)| *score as f32)
                .next()
                .unwrap_or(0.0);

            let score_diff = my_score - opponent_score;
            let value = (score_diff / polyfish::states::default_max_score() as f32).tanh();

            collected_values.push(value);
        }
    }

    // Print Average Metrics
    let avg_score = total_score as f32 / args.num_games as f32;
    let avr_moves = total_moves as f32 / args.num_games as f32;
    let p1_avg = if p1_count > 0 {
        p1_total as f32 / p1_count as f32
    } else {
        0.0
    };
    let p2_avg = if p2_count > 0 {
        p2_total as f32 / p2_count as f32
    } else {
        0.0
    };

    println!(
        "METRICS: {{\"avg_score\": {:.2}, \"max_score\": {}, \"avg_moves\": {:.2}, \"p1_avg\": {:.2}, \"p2_avg\": {:.2}}}",
        avg_score, max_score, avr_moves, p1_avg, p2_avg
    );

    // Stack and save
    if !collected_spatial_maps.is_empty() {
        let total_steps = collected_spatial_maps.len();
        println!("Saving {} steps...", total_steps);

        let spatial_dim = features::NUM_CHANNELS * features::MAP_SIZE * features::MAP_SIZE;
        let player_dim = 10;

        let spatial_maps_tensor = Tensor::cat(&collected_spatial_maps, 0)?;
        let spatial_maps_tensor = spatial_maps_tensor.reshape((total_steps, spatial_dim))?;
        println!(
            "Spatial maps shape: {:?} (dim: {})",
            spatial_maps_tensor.shape(),
            spatial_dim
        );

        let player_states_tensor = Tensor::cat(&collected_player_states, 0)?;
        let player_states_tensor = player_states_tensor.reshape((total_steps, player_dim))?;

        // Helper to simple-flatten data
        fn flatten_vec(v: Vec<Vec<f32>>) -> Vec<f32> {
            v.into_iter().flatten().collect()
        }

        let action_tensor = Tensor::from_vec(
            flatten_vec(collected_action_type),
            (total_steps, 11),
            &device,
        )?;

        let spatial_logit_dim = features::MAP_SIZE * features::MAP_SIZE;

        let source_tensor = Tensor::from_vec(
            flatten_vec(collected_source_spatial),
            (total_steps, spatial_logit_dim),
            &device,
        )?;
        let target_tensor = Tensor::from_vec(
            flatten_vec(collected_target_spatial),
            (total_steps, spatial_logit_dim),
            &device,
        )?;
        let option_tensor =
            Tensor::from_vec(flatten_vec(collected_option), (total_steps, 192), &device)?;

        // Values
        let values_tensor = Tensor::from_vec(collected_values, (total_steps, 1), &device)?;

        let mut tensors = HashMap::new();
        tensors.insert("spatial_maps".to_string(), spatial_maps_tensor);
        tensors.insert("player_states".to_string(), player_states_tensor);

        tensors.insert("action_type".to_string(), action_tensor);
        tensors.insert("source_spatial".to_string(), source_tensor);
        tensors.insert("target_spatial".to_string(), target_tensor);
        tensors.insert("move_option".to_string(), option_tensor);

        tensors.insert("values".to_string(), values_tensor);

        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        let filename = format!("games_{}.safetensors", timestamp);
        candle_core::safetensors::save(&tensors, &filename)?;
        println!("Saved to {}", filename);
    }

    Ok(())
}
