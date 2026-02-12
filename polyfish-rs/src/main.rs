use axum::{
    Json, Router,
    extract::State,
    routing::{get, post},
};
use polyfish::mapgen::{MapGenSettings, generate};
use polyfish::moves::{
    AttackMove, BuildMove, CaptureMove, DisbandMove, EndTurnMove, HarvestMove, Move, RecoverMove,
    ResearchMove, RewardMove, StepMove, SummonMove, UpgradeMove,
    abilities::{
        boost::BoostMove, convert::ConvertMove, decompose::DecomposeMove, destroy::DestroyMove,
        diplomacy::BreakPeaceMove, enchant_animal::EnchantAnimalMove, explode::ExplodeMove,
        forest::BurnForestMove, forest::ClearForestMove, forest::GrowForestMove,
        freeze_area::FreezeAreaMove, heal_others::HealOthersMove, promote::PromoteMove,
    },
};
use polyfish::types::{AbilityType, MapSize, TribeType};
use polyfish::{MapType, game::Game};
use serde_json::Value;
use std::sync::{Arc, Mutex};
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;

use polyfish::recorder::GameRecorder;

struct AppState {
    game: Mutex<Game>,
    training_status: Mutex<Option<u32>>, // Store PID of running training
    network: Arc<polyfish::ai::network::PolyZeroNet>, // Trained neural network
    recorder: Arc<GameRecorder>,
}

const DEFAULT_TRIBES: &[TribeType] = &[TribeType::Imperius, TribeType::Imperius];
const DEFAULT_SIZE: MapSize = MapSize::Tiny;

#[tokio::main]
async fn main() {
    // Initialize game
    let mut settings = MapGenSettings::default();
    settings.size = DEFAULT_SIZE;
    settings.tribes = DEFAULT_TRIBES.to_vec();
    settings.seed = rand::random();
    settings.map_type = MapType::Drylands;

    let initial_state = generate(settings);
    let mut game = Game::new();
    game.state = initial_state;
    game.state.settings.verbose = true;
    game.state.settings.max_turns = 10;
    game.post_load();

    // Load trained neural network
    use candle_core::Device;
    use candle_nn::VarMap;
    let device = Device::Cpu;

    let model_path = "model.safetensors";
    let mut varmap = VarMap::new();

    let network = if std::path::Path::new(model_path).exists() {
        println!("✅ Loading trained AI model from {}", model_path);
        varmap
            .load(model_path)
            .expect("Failed to load model weights");
        polyfish::ai::network::PolyZeroNet::new(candle_nn::VarBuilder::from_varmap(
            &varmap,
            candle_core::DType::F32,
            &device,
        ))
        .expect("Failed to build neural network")
    } else {
        panic!(
            "Model file {} not found! Please run init_model.py first.",
            model_path
        );
    };

    let recorder = Arc::new(GameRecorder::new());

    let shared_state = Arc::new(AppState {
        game: Mutex::new(game),
        training_status: Mutex::new(None),
        network: Arc::new(network),
        recorder,
    });

    // Build our application with routes
    let app = Router::new()
        .route("/current", get(get_current_state))
        .route("/autostep", post(auto_step))
        .route("/step", post(manual_step))
        .route("/rngstep", post(rng_step))
        .route("/reset", post(reset_game))
        .route("/train", post(trigger_training))
        .route("/train/status", get(get_training_status))
        .route("/save_training_data", post(save_training_data))
        .route("/analyze", get(analyze_game))
        .route("/save", post(save_game))
        .route("/load", post(load_game))
        .route("/simulate/explorer", post(simulate_explorer))
        .route("/simulate/attack", post(simulate_attack))
        .route("/replay/save", post(save_replay_endpoint))
        .route("/replay/load", post(load_replay_endpoint))
        .route("/replay/analyze", post(analyze_replay_step))
        .route("/replay/load_initial", post(load_initial_endpoint))
        .route("/replay/list_initial", get(list_initial_endpoint))
        .route("/trainer/hint", post(get_trainer_hint))
        .nest_service("/", ServeDir::new("../src/public"))
        .layer(CorsLayer::permissive())
        .with_state(shared_state);

    // Run our app
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    println!("Listening on http://localhost:3000");
    axum::serve(listener, app).await.unwrap();
}

fn build_evaluation_json(state: &polyfish::states::GameState) -> Value {
    use polyfish::ai::evaluator::player::evaluate_player;
    let mut players = serde_json::Map::new();
    for &pid in state.tribes.keys() {
        let score = evaluate_player(state, pid);
        players.insert(pid.to_string(), serde_json::json!(score));
    }
    // Advantage from P1's perspective (player 1 minus best opponent)
    let p1_score = players.get("1").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let p2_score = players.get("2").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let advantage = p1_score - p2_score;
    serde_json::json!({
        "players": players,
        "advantage": advantage
    })
}

async fn analyze_game(State(state): State<Arc<AppState>>) -> Json<Value> {
    let mut game = state.game.lock().unwrap();

    // Recalculate predictions on demand
    polyfish::prediction::update_predictions(&mut game.state);

    let current_player = game.state.settings.current_player_turn_id;
    let expansion_analysis = polyfish::functions::analyze_expansion(&game.state, current_player);

    Json(serde_json::json!({
        "tileValues": expansion_analysis.tile_values,
        "threats": expansion_analysis.threats,
        "prediction": game.state._prediction
    }))
}

#[derive(serde::Deserialize)]
struct StepParams {
    #[serde(default = "default_iterations")]
    iterations: usize,
    #[serde(default)]
    dry_run: bool,
}

fn default_iterations() -> usize {
    800
}

async fn get_current_state(State(state): State<Arc<AppState>>) -> Json<Value> {
    let mut game = state.game.lock().unwrap();
    polyfish::prediction::update_predictions(&mut game.state);

    let mut tiles: Vec<_> = game.state.tiles.values().collect();
    tiles.sort_by_key(|t| t.coords.idx);

    let legal_moves: Vec<_> = game.legal_moves().iter().map(|m| m.serialize()).collect();

    let evaluation = build_evaluation_json(&game.state);

    Json(serde_json::json!({
        "state": {
            "settings": game.state.settings,
            "tiles": tiles,
            "structures": game.state.structures,
            "resources": game.state.resources,
            "tribes": game.state.tribes,
            "_hiddenResources": game.state._hidden_resources,
            "_prediction": game.state._prediction,
            "_messages": game.state._messages,
        },
        "legalMoves": legal_moves,
        "evaluation": evaluation
    }))
}

async fn auto_step(
    State(state): State<Arc<AppState>>,
    Json(params): Json<StepParams>,
) -> Json<Value> {
    let mut game = state.game.lock().unwrap();
    // dont spam the front end lol
    game.state.settings.verbose = false;

    // Use trained AI model!
    use polyfish::ai::mcts_zero::ZeroMctsAgent;
    let agent = ZeroMctsAgent::new(&state.network, params.iterations);

    game.state._messages.clear();
    let (chosen_move, policy) = agent.select_move_with_stats(&mut game);
    let mut move_name = "none".to_string();

    // Create serialized best move before potentially consuming it (though we use as_ref so it's fine)
    let best_move_json = chosen_move.as_ref().map(|m| m.serialize());

    if !params.dry_run {
        if let Some(m) = chosen_move {
            move_name = format!("{:?}", m.move_type());
            game.play_move(m.as_ref());
        }
    }

    // Run heuristic MCTS for analysis panel (move descriptions + PV)
    use polyfish::ai::heuristic_mcts::HeuristicMctsAgent;

    let analysis_agent = HeuristicMctsAgent {
        iterations: params.iterations,
        exploration_constant: 0.1,
    };
    let (_, mcts_analysis) = analysis_agent.select_move_with_analysis(&mut game);

    let mut tiles: Vec<_> = game.state.tiles.values().collect();
    tiles.sort_by_key(|t| t.coords.idx);

    let legal_moves: Vec<_> = game.legal_moves().iter().map(|m| m.serialize()).collect();

    let evaluation = build_evaluation_json(&game.state);

    Json(serde_json::json!({
        "state": {
            "settings": game.state.settings,
            "tiles": tiles,
            "structures": game.state.structures,
            "resources": game.state.resources,
            "tribes": game.state.tribes,
            "_hiddenResources": game.state._hidden_resources,
            "_prediction": game.state._prediction,
            "_messages": game.state._messages,
        },
        "movePlayed": move_name,
        "bestMove": best_move_json,
        "legalMoves": legal_moves,
        "policyDistribution": policy,
        "evaluation": evaluation,
        "mctsAnalysis": mcts_analysis
    }))
}

async fn rng_step(State(state): State<Arc<AppState>>) -> Json<Value> {
    let mut game = state.game.lock().unwrap();
    let original_player = game.current_player_id();
    let mut move_name = "none".to_string();

    // Play at least one move
    game.state._messages.clear();
    loop {
        let moves = game.legal_moves();
        if moves.is_empty() {
            break;
        }

        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        let chosen = moves.choose(&mut rng).unwrap();

        move_name = format!("{:?}", chosen.move_type());
        game.play_move(chosen.as_ref());

        // If we just played EndTurn, we need to let the other players play
        // until it is our turn again.
        if game.current_player_id() != original_player {
            // Keep playing random moves until turn comes back to original_player
            // or game over
            let mut steps = 0;
            while game.current_player_id() != original_player && steps < 100 {
                let other_moves = game.legal_moves();
                if other_moves.is_empty() {
                    break;
                }
                let random_move = other_moves.choose(&mut rng).unwrap();
                game.play_move(random_move.as_ref());
                steps += 1;
            }
            // Break the main loop because we have completed a full round
            break;
        } else {
            // We moved but it's still our turn (e.g. moved a unit), break to let UI update
            break;
        }
    }

    let mut tiles: Vec<_> = game.state.tiles.values().collect();
    tiles.sort_by_key(|t| t.coords.idx);

    let legal_moves: Vec<_> = game.legal_moves().iter().map(|m| m.serialize()).collect();

    let evaluation = build_evaluation_json(&game.state);

    Json(serde_json::json!({
        "state": {
            "settings": game.state.settings,
            "tiles": tiles,
            "structures": game.state.structures,
            "resources": game.state.resources,
            "tribes": game.state.tribes,
            "_hiddenResources": game.state._hidden_resources,
            "_prediction": game.state._prediction,
            "_messages": game.state._messages,
        },
        "movePlayed": move_name,
        "legalMoves": legal_moves,
        "evaluation": evaluation
    }))
}

async fn manual_step(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<Value>,
) -> Json<Value> {
    let mut game = state.game.lock().unwrap();
    let move_type = payload
        .get("moveType")
        .and_then(|v| v.as_i64())
        .map(|v| v as i8)
        .unwrap_or(0);

    let move_obj: Box<dyn Move> = match move_type {
        1 => {
            // Step
            let src = payload["src"].as_i64().unwrap() as i32;
            let target = payload["target"].as_i64().unwrap() as i32;
            Box::new(StepMove::new(src, target))
        }
        2 => {
            // Attack
            let src = payload["src"].as_i64().unwrap() as i32;
            let target = payload["target"].as_i64().unwrap() as i32;
            Box::new(AttackMove::new(src, target))
        }
        3 => {
            // Ability
            let src = payload["src"]
                .as_i64()
                .or(payload["target"].as_i64())
                .unwrap_or(0) as i32;
            let ability = payload["type"].as_i64().unwrap_or(0) as i8;
            match unsafe { std::mem::transmute(ability) } {
                AbilityType::Recover => Box::new(RecoverMove::new(src)),
                AbilityType::Promote => Box::new(PromoteMove::new(src)),
                AbilityType::Disband => Box::new(DisbandMove::new(src)),
                AbilityType::BurnForest => Box::new(BurnForestMove::new(src)),
                AbilityType::ClearForest => Box::new(ClearForestMove::new(src)),
                AbilityType::GrowForest => Box::new(GrowForestMove::new(src)),
                AbilityType::Destroy => Box::new(DestroyMove::new(src)),
                AbilityType::Decompose => Box::new(DecomposeMove::new(src)),
                AbilityType::Convert => {
                    let target = payload["target"]
                        .as_i64()
                        .or(payload["src"].as_i64())
                        .unwrap() as i32;
                    Box::new(ConvertMove::new(src, target))
                }
                AbilityType::HealOthers => Box::new(HealOthersMove::new(src)),
                AbilityType::FreezeArea => Box::new(FreezeAreaMove::new(src)),
                AbilityType::Boost => Box::new(BoostMove::new(src)),
                AbilityType::Explode => Box::new(ExplodeMove::new(src)),
                AbilityType::EnchantAnimal => Box::new(EnchantAnimalMove::new(src)),
                AbilityType::BreakPeace => {
                    let target = payload["target"]
                        .as_i64()
                        .or(payload["src"].as_i64())
                        .unwrap() as i32;
                    Box::new(BreakPeaceMove::new(target))
                }
                _ => Box::new(EndTurnMove),
            }
        }
        4 => {
            // Summon or Upgrade
            let tile_index = payload["src"].as_i64().unwrap() as i32;
            let type_val = payload["type"].as_i64().unwrap() as i8;
            if payload
                .get("upgrade")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                Box::new(UpgradeMove::new(tile_index, unsafe {
                    std::mem::transmute(type_val)
                }))
            } else {
                Box::new(SummonMove::new(tile_index, unsafe {
                    std::mem::transmute(type_val)
                }))
            }
        }
        5 => {
            // Harvest
            let idx = payload["target"].as_i64().unwrap() as i32;
            Box::new(HarvestMove::new(idx))
        }
        6 => {
            // Build
            let idx = payload["target"].as_i64().unwrap() as i32;
            let construct = payload["type"].as_i64().unwrap() as i8;
            Box::new(BuildMove::new(idx, unsafe {
                std::mem::transmute(construct)
            }))
        }
        7 => {
            // Research
            let tech = payload["type"].as_i64().unwrap() as i8;
            Box::new(ResearchMove::new(unsafe { std::mem::transmute(tech) }))
        }
        8 => {
            // Capture
            let src = payload["src"].as_i64().unwrap() as i32;
            Box::new(CaptureMove::new(src))
        }
        9 => {
            // Reward
            let idx = payload["target"].as_i64().unwrap() as i32;
            let reward_type = payload["type"].as_i64().unwrap() as i8;
            Box::new(RewardMove::new(idx, unsafe {
                std::mem::transmute(reward_type)
            }))
        }
        10 => Box::new(EndTurnMove),
        _ => Box::new(EndTurnMove),
    };

    // RECORDING: Before we play the move, we capture the state and the move we ARE about to make.
    // We only record moves for non-bots (or whatever the user is controlling).
    // Assuming user controls the current tribe.
    // Calculate simple heuristic values for Eco/Mil.
    // Calculate simple heuristic values for Eco/Mil.
    use polyfish::ai::evaluator::{army, economy};
    let pid = game.state.settings.current_player_turn_id;
    let eco = economy::evaluate_economy(&game.state, pid);
    let mil = army::evaluate_army(&game.state, pid);

    state
        .recorder
        .record_step(&game.state, move_obj.as_ref(), eco, mil);

    let move_name = format!("{:?}", move_obj.move_type());
    game.state._messages.clear();
    game.play_move(move_obj.as_ref());

    let eco1 = economy::evaluate_economy(&game.state, pid);
    let mil1 = army::evaluate_army(&game.state, pid);

    println!(
        "Manual step: player={}, move={:?}, eco={:.4} ({}{:.4}), mil={:.4} ({}{:.4})",
        pid,
        move_obj.move_type(),
        eco,
        if eco1 - eco > 0.0 { "+" } else { "" },
        eco1 - eco,
        mil,
        if mil1 - mil > 0.0 { "+" } else { "" },
        mil1 - mil,
    );

    let mut tiles: Vec<_> = game.state.tiles.values().collect();
    tiles.sort_by_key(|t| t.coords.idx);

    let legal_moves: Vec<_> = game.legal_moves().iter().map(|m| m.serialize()).collect();

    let evaluation = build_evaluation_json(&game.state);

    Json(serde_json::json!({
        "state": {
            "settings": game.state.settings,
            "tiles": tiles,
            "structures": game.state.structures,
            "resources": game.state.resources,
            "tribes": game.state.tribes,
            "_hiddenResources": game.state._hidden_resources,
            "_prediction": game.state._prediction,
            "_messages": game.state._messages,
        },
        "movePlayed": move_name,
        "legalMoves": legal_moves,
        "evaluation": evaluation
    }))
}

async fn save_training_data(State(state): State<Arc<AppState>>) -> Json<Value> {
    match state.recorder.save() {
        Ok(msg) => Json(serde_json::json!({ "status": "success", "message": msg })),
        Err(e) => Json(serde_json::json!({ "status": "error", "message": e.to_string() })),
    }
}

async fn reset_game(State(state): State<Arc<AppState>>) -> Json<Value> {
    let mut game = state.game.lock().unwrap();

    let mut settings = MapGenSettings::default();
    settings.size = DEFAULT_SIZE;
    settings.tribes = DEFAULT_TRIBES.to_vec();
    settings.seed = rand::random();
    settings.map_type = MapType::Drylands;

    let initial_state = generate(settings);
    game.state = initial_state;
    game.state.settings.verbose = true;
    game.post_load();

    let mut tiles: Vec<_> = game.state.tiles.values().collect();
    tiles.sort_by_key(|t| t.coords.idx);

    let legal_moves: Vec<_> = game.legal_moves().iter().map(|m| m.serialize()).collect();

    let evaluation = build_evaluation_json(&game.state);

    Json(serde_json::json!({
        "state": {
            "settings": game.state.settings,
            "tiles": tiles,
            "structures": game.state.structures,
            "resources": game.state.resources,
            "tribes": game.state.tribes,
            "_hiddenResources": game.state._hidden_resources,
            "_prediction": game.state._prediction,
            "_messages": game.state._messages,
        },
        "legalMoves": legal_moves,
        "evaluation": evaluation
    }))
}

async fn trigger_training(State(state): State<Arc<AppState>>) -> Json<Value> {
    use std::fs::File;
    use std::process::{Command, Stdio};

    // Redirect to training.poly.log
    let log_file = match File::create("training.poly.log") {
        Ok(f) => f,
        Err(e) => return Json(serde_json::json!({ "status": "error", "message": e.to_string() })),
    };

    let child = Command::new("cargo")
        .args(["run", "--bin", "self_play", "--release"])
        .current_dir(".")
        .stdout(Stdio::from(log_file.try_clone().unwrap()))
        .stderr(Stdio::from(log_file))
        .spawn();

    match child {
        Ok(c) => {
            let pid = c.id();
            let mut status = state.training_status.lock().unwrap();
            *status = Some(pid);

            Json(serde_json::json!({
                "status": "success",
                "message": format!("Training started (PID: {})", pid)
            }))
        }
        Err(e) => Json(serde_json::json!({
            "status": "error",
            "message": format!("Failed to start training: {}", e)
        })),
    }
}

async fn get_training_status(State(state): State<Arc<AppState>>) -> Json<Value> {
    use std::fs;
    use std::process::Command;

    let pid_opt = *state.training_status.lock().unwrap();
    let mut is_running = false;

    if let Some(pid) = pid_opt {
        // Check if process still exists
        let output = Command::new("ps").arg("-p").arg(pid.to_string()).output();

        if let Ok(out) = output {
            is_running = out.status.success();
        }
    }

    // Read last 15 lines of log for more detail
    let log_content = fs::read_to_string("training.poly.log").unwrap_or_default();
    let lines: Vec<&str> = log_content.lines().collect();
    let last_lines = if lines.len() > 15 {
        lines[lines.len() - 15..].join("\n")
    } else {
        lines.join("\n")
    };

    Json(serde_json::json!({
        "isRunning": is_running,
        "pid": pid_opt,
        "log": last_lines
    }))
}

async fn simulate_explorer(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<Value>,
) -> Json<Value> {
    let game = state.game.lock().unwrap();
    let idx = payload
        .get("idx")
        .and_then(|v| v.as_i64())
        .map(|v| v as i32)
        .unwrap_or(0);

    let (path, revealed) = polyfish::actions::discovery::predict_explorer(&game.state, idx);

    Json(serde_json::json!({
        "path": path,
        "revealed": revealed
    }))
}

async fn simulate_attack(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<Value>,
) -> Json<Value> {
    let game = state.game.lock().unwrap();
    let src = payload
        .get("src")
        .and_then(|v| v.as_i64())
        .map(|v| v as i32)
        .unwrap_or(0);
    let target = payload
        .get("target")
        .and_then(|v| v.as_i64())
        .map(|v| v as i32)
        .unwrap_or(0);

    let preview = polyfish::functions::calculate_combat_preview(&game.state, src, target);

    match preview {
        Some(p) => Json(serde_json::to_value(p).unwrap()),
        None => Json(serde_json::json!({ "error": "No valid attack target found" })),
    }
}

async fn save_game(State(state): State<Arc<AppState>>) -> Json<Value> {
    let game = state.game.lock().unwrap();
    let json = serde_json::to_string_pretty(&game.state).unwrap();
    std::fs::write("saved_state.json", json).expect("Failed to write saved_state.json");

    Json(serde_json::json!({
        "status": "success",
        "message": "Game state saved to saved_state.json"
    }))
}

async fn load_game(State(state): State<Arc<AppState>>) -> Json<Value> {
    let mut game = state.game.lock().unwrap();
    let json =
        std::fs::read_to_string("saved_state.json").expect("Failed to read saved_state.json");
    let loaded_state: polyfish::states::GameState =
        serde_json::from_str(&json).expect("Failed to deserialize game state");

    game.state = loaded_state;
    game.post_load();

    let mut tiles: Vec<_> = game.state.tiles.values().collect();
    tiles.sort_by_key(|t| t.coords.idx);

    let legal_moves: Vec<_> = game.legal_moves().iter().map(|m| m.serialize()).collect();

    let evaluation = build_evaluation_json(&game.state);

    Json(serde_json::json!({
        "state": {
            "settings": game.state.settings,
            "tiles": tiles,
            "structures": game.state.structures,
            "resources": game.state.resources,
            "tribes": game.state.tribes,
            "_hiddenResources": game.state._hidden_resources,
            "_prediction": game.state._prediction,
            "_messages": game.state._messages,
        },
        "legalMoves": legal_moves,
        "evaluation": evaluation
    }))
}

async fn get_trainer_hint(
    State(state): State<Arc<AppState>>,
    Json(params): Json<StepParams>,
) -> Json<Value> {
    let mut game = state.game.lock().unwrap();

    // Use Zero MCTS Agent (Neural Network)
    use polyfish::ai::mcts_zero::ZeroMctsAgent;
    let agent = ZeroMctsAgent::new(&state.network, params.iterations);
    // use polyfish::ai::heuristic_mcts::HeuristicMctsAgent;
    // let agent = HeuristicMctsAgent {
    //     iterations: params.iterations,
    //     exploration_constant: 0.4,
    // };

    // Run MCTS search
    let (best_move, mcts_analysis) = agent.select_move_with_stats(&mut game);

    let move_json = best_move.as_ref().map(|m| m.serialize());
    let move_name = best_move
        .as_ref()
        .map(|m| format!("{:?}", m.move_type()))
        .unwrap_or("None".to_string());
    let move_description = best_move
        .as_ref()
        .map(|m| m.describe(&game.state))
        .unwrap_or("No suggestion".to_string());

    Json(serde_json::json!({
        "proposedMove": move_json,
        "moveName": move_name,
        "moveDescription": move_description,
        "mctsAnalysis": mcts_analysis,
    }))
}

// === Replay System ===

#[derive(serde::Deserialize)]
struct SaveReplayParams {
    name: String,
}

async fn save_replay_endpoint(
    State(state): State<Arc<AppState>>,
    Json(params): Json<SaveReplayParams>,
) -> Json<Value> {
    let game = state.game.lock().unwrap();

    // Create replays directory if not exists
    let _ = std::fs::create_dir_all("replays");

    // Sanitize filename
    let safe_name: String = params
        .name
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '_')
        .collect();

    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let filename = format!("replays/{}_{}.json", safe_name, timestamp);

    // Save full state (which includes history and initial_seed)
    let json = serde_json::to_string_pretty(&game.state).unwrap();
    match std::fs::write(&filename, json) {
        Ok(_) => Json(serde_json::json!({
            "status": "success",
            "message": format!("Replay saved to {}", filename),
            "filename": filename
        })),
        Err(e) => Json(serde_json::json!({
            "status": "error",
            "message": format!("Failed to save replay: {}", e)
        })),
    }
}

#[derive(serde::Deserialize)]
struct LoadReplayParams {
    filename: String,
}

async fn load_replay_endpoint(
    State(state): State<Arc<AppState>>,
    Json(params): Json<LoadReplayParams>,
) -> Json<Value> {
    let mut game = state.game.lock().unwrap();

    // Security check: simple path traversal prevention
    if params.filename.contains("..")
        || params.filename.contains("/")
        || params.filename.contains("\\")
    {
        // allow if it starts with "replays/"
        if !params.filename.starts_with("replays/") {
            return Json(serde_json::json!({ "status": "error", "message": "Invalid filename" }));
        }
    }

    let path = if params.filename.starts_with("replays/") {
        params.filename.clone()
    } else {
        format!("replays/{}", params.filename)
    };

    match std::fs::read_to_string(&path) {
        Ok(json) => match serde_json::from_str::<polyfish::states::GameState>(&json) {
            Ok(loaded_state) => {
                game.state = loaded_state;
                game.post_load();

                let mut tiles: Vec<_> = game.state.tiles.values().collect();
                tiles.sort_by_key(|t| t.coords.idx);

                let legal_moves: Vec<_> =
                    game.legal_moves().iter().map(|m| m.serialize()).collect();

                let evaluation = build_evaluation_json(&game.state);

                Json(serde_json::json!({
                    "status": "success",
                    "filename": path,
                    "state": {
                        "settings": game.state.settings,
                        "tiles": tiles,
                        "structures": game.state.structures,
                        "resources": game.state.resources,
                        "tribes": game.state.tribes,
                        "_hiddenResources": game.state._hidden_resources,
                        "_prediction": game.state._prediction,
                        "_messages": game.state._messages,
                        "history": game.state.history,
                    },
                    "legalMoves": legal_moves,
                    "evaluation": evaluation
                }))
            }
            Err(e) => Json(
                serde_json::json!({ "status": "error", "message": format!("Failed to parse replay: {}", e) }),
            ),
        },
        Err(e) => Json(
            serde_json::json!({ "status": "error", "message": format!("Failed to read replay file: {}", e) }),
        ),
    }
}

#[derive(serde::Deserialize)]
struct LoadInitialParams {
    id: String,
}

async fn load_initial_endpoint(
    State(state): State<Arc<AppState>>,
    Json(params): Json<LoadInitialParams>,
) -> Json<Value> {
    let mut game = state.game.lock().unwrap();

    // Sanitize ID (simple alphanumeric check)
    if !params.id.chars().all(|c| c.is_alphanumeric() || c == '-') {
        return Json(serde_json::json!({ "status": "error", "message": "Invalid Game ID format" }));
    }

    // Construct path: ../src/scraper/data/training-data/{id}.initial.json
    // We are running from polyfish-rs/
    let path = format!(
        "../src/scraper/data/training-data/{}.initial.json",
        params.id
    );

    match std::fs::read_to_string(&path) {
        Ok(json) => {
            let mut root: Value = serde_json::from_str(&json).unwrap_or_default();

            // --- FIXUP LOGIC START ---
            let tile_count = root["tiles"].as_object().map(|o| o.len()).unwrap_or(0);
            let map_size = (tile_count as f64).sqrt() as i32;

            // Missing fields
            if let Some(obj) = root.as_object_mut() {
                if !obj.contains_key("settings") {
                    obj.insert("settings".into(), serde_json::json!({}));
                }
                let settings = obj.get_mut("settings").unwrap().as_object_mut().unwrap();
                // Ensure critical settings exist
                if !settings.contains_key("version") {
                    settings.insert("version".into(), serde_json::json!(0));
                }
                if !settings.contains_key("size") {
                    settings.insert("size".into(), serde_json::json!(map_size));
                }
                //if !settings.contains_key("currentPlayerTurnId") {
                // override to make sure p1 always starts
                settings.insert("currentPlayerTurnId".into(), serde_json::json!(1));
                //}

                // Add fields stripped by scraper
                if !settings.contains_key("_areYouSure") {
                    settings.insert("_areYouSure".into(), serde_json::json!(false));
                }
                if !settings.contains_key("_gameOver") {
                    settings.insert("_gameOver".into(), serde_json::json!(false));
                }
                if !settings.contains_key("_fow") {
                    settings.insert("_fow".into(), serde_json::json!(true));
                }
                if !settings.contains_key("_lastPlayerTurnId") {
                    settings.insert("_lastPlayerTurnId".into(), serde_json::json!(0));
                }
                if !settings.contains_key("_recentMoves") {
                    settings.insert("_recentMoves".into(), serde_json::json!([]));
                }
                if !settings.contains_key("_maxTribeCount") {
                    settings.insert("_maxTribeCount".into(), serde_json::json!(2));
                }
            }

            // Fix cities: inject missing "id" field (equal to tileIndex)
            if let Some(tribes) = root["tribes"].as_object_mut() {
                for (_, tribe) in tribes.iter_mut() {
                    if let Some(cities) = tribe["cities"].as_array_mut() {
                        for city in cities {
                            if let Some(city_obj) = city.as_object_mut() {
                                if !city_obj.contains_key("id") {
                                    if let Some(tid) = city_obj.get("tileIndex") {
                                        city_obj.insert("id".into(), tid.clone());
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Recursive function to fix [x, y] coordinates into [x, y, idx]
            fn fix_coords(val: &mut Value, map_size: i32) {
                match val {
                    Value::Array(arr) => {
                        if arr.len() == 2 && arr[0].is_number() && arr[1].is_number() {
                            let x = arr[0].as_i64().unwrap_or(0) as i32;
                            let y = arr[1].as_i64().unwrap_or(0) as i32;
                            let idx = y * map_size + x;
                            arr.push(serde_json::json!(idx));
                        } else {
                            for item in arr {
                                fix_coords(item, map_size);
                            }
                        }
                    }
                    Value::Object(obj) => {
                        for (_, v) in obj.iter_mut() {
                            fix_coords(v, map_size);
                        }
                    }
                    _ => {}
                }
            }

            fix_coords(&mut root, map_size);

            match Game::from_json(&serde_json::to_string(&root).unwrap()) {
                Ok(new_game) => {
                    game.state = new_game.state;
                    game.post_load();

                    let mut tiles: Vec<_> = game.state.tiles.values().collect();
                    tiles.sort_by_key(|t| t.coords.idx);
                    let legal_moves: Vec<_> =
                        game.legal_moves().iter().map(|m| m.serialize()).collect();
                    let evaluation = build_evaluation_json(&game.state);

                    let moves_path =
                        format!("../src/scraper/data/training-data/{}.moves.json", params.id);
                    let recorded_moves = std::fs::read_to_string(&moves_path)
                        .ok()
                        .and_then(|m| serde_json::from_str::<Value>(&m).ok())
                        .unwrap_or(serde_json::json!([]));

                    Json(serde_json::json!({
                        "status": "success",
                        "recordedMoves": recorded_moves,
                        "state": {
                            "settings": game.state.settings,
                            "tiles": tiles,
                            "structures": game.state.structures,
                            "resources": game.state.resources,
                            "tribes": game.state.tribes,
                            "_hiddenResources": game.state._hidden_resources,
                            "_prediction": game.state._prediction,
                            "_messages": game.state._messages,
                        },
                        "legalMoves": legal_moves,
                        "evaluation": evaluation
                    }))
                }
                Err(e) => Json(serde_json::json!({
                   "status": "error",
                   "message": format!("Failed to parse initial state: {}", e)
                })),
            }
        }
        Err(e) => Json(serde_json::json!({
            "status": "error",
            "message": format!("Failed to read file {}: {}", path, e)
        })),
    }
}

#[derive(serde::Deserialize)]
struct AnalyzeReplayParams {
    filename: String,
    step_index: usize,
    iterations: usize,
}

async fn analyze_replay_step(
    State(state): State<Arc<AppState>>,
    Json(params): Json<AnalyzeReplayParams>,
) -> Json<Value> {
    // 1. Load the replay file to get initial seed and history
    let path = if params.filename.starts_with("replays/") {
        params.filename.clone()
    } else {
        format!("replays/{}", params.filename)
    };

    let replay_state: polyfish::states::GameState = match std::fs::read_to_string(&path) {
        Ok(json) => serde_json::from_str(&json).unwrap_or_default(),
        Err(e) => {
            return Json(serde_json::json!({ "error": format!("Failed to load replay: {}", e) }));
        }
    };

    if replay_state.history.len() <= params.step_index {
        return Json(serde_json::json!({ "error": "Step index out of bounds" }));
    }

    // 2. Initialize a FRESH game with the SAME seed
    let mut map_settings = MapGenSettings::default();
    map_settings.size = match replay_state.settings.size {
        11 => polyfish::types::MapSize::Tiny,
        13 => polyfish::types::MapSize::Small,
        16 => polyfish::types::MapSize::Normal,
        24 => polyfish::types::MapSize::Large,
        32 => polyfish::types::MapSize::Huge,
        90 => polyfish::types::MapSize::Massive,
        _ => polyfish::types::MapSize::Normal,
    };
    map_settings.seed = replay_state.initial_seed;
    map_settings.map_type = replay_state.settings.map_type;

    // Need to extract tribes from replay settings or state?
    // GameSettings doesn't store the tribe list directly as TribeType enum vec, but state.tribes does
    // Extract unique tribe types from replay_state.tribes
    let mut tribes = Vec::new();
    // Sort by ID to ensure consistent order if that matters
    let mut sorted_tribes: Vec<_> = replay_state.tribes.values().collect();
    sorted_tribes.sort_by_key(|t| t.id);
    for t in sorted_tribes {
        tribes.push(t.tribe_type);
    }
    map_settings.tribes = tribes;

    let initial_state = generate(map_settings);
    let mut game = Game::new();
    game.state = initial_state;
    // Restore initial seed again just in case generate() didn't set it (it should have used it)
    game.state.initial_seed = replay_state.initial_seed;
    game.post_load();

    // 3. Replay moves up to step_index (exclusive? or inclusive? let's say we want to analyze the state BEFORE turn step_index is played)
    // Wait, if we want to compare "User Move vs AI Move", we need the state *before* the user made the move at step_index.
    // So we replay moves 0 to step_index - 1.

    for i in 0..params.step_index {
        if i >= replay_state.history.len() {
            break;
        }

        let move_json = &replay_state.history[i];

        // We need to parse this JSON back into a Box<dyn Move>
        // Use a matching logic similar to manual_step... logic duplication is confusing.
        // Better way: Implement Game::deserialize_move?
        // For now, let's copy the deserialization logic or create a helper.
        // Actually, since we only need to play it, we can identify it from legal moves if it matches?
        // But some moves (like Build) have parameters that legal_moves might not fully capture if they are identical?
        // Actually, `legal_moves` generates all distinct moves. We can find the one that matches the JSON.

        // Find matching move in legal moves
        let legal = game.legal_moves();
        let mut found = false;

        // serialized form comparison
        for m in legal {
            if m.serialize() == *move_json {
                game.play_move(m.as_ref());
                found = true;
                break;
            }
        }

        if !found {
            return Json(serde_json::json!({
                "error": format!("Failed to replay move at step {}: Move desync or invalid. JSON: {:?}", i, move_json)
            }));
        }
    }

    // 4. Now game is at the state just before the user played history[step_index]
    // Run MCTS analysis
    use polyfish::ai::mcts_zero::ZeroMctsAgent;
    let agent = ZeroMctsAgent::new(&state.network, params.iterations);
    let (best_move, mcts_analysis) = agent.select_move_with_stats(&mut game);

    let ai_move_json = best_move.as_ref().map(|m| m.serialize());
    let ai_move_desc = best_move
        .as_ref()
        .map(|m| m.describe(&game.state))
        .unwrap_or("None".to_string());

    // User's actual move
    let user_move_json = &replay_state.history[params.step_index];
    // Find desc for user move
    let legal = game.legal_moves();
    let mut user_move_desc = "Unknown Move".to_string();
    for m in legal {
        if m.serialize() == *user_move_json {
            user_move_desc = m.describe(&game.state);
            break;
        }
    }

    // Build state for frontend
    let mut tiles: Vec<_> = game.state.tiles.values().collect();
    tiles.sort_by_key(|t| t.coords.idx);

    Json(serde_json::json!({
        "stepIndex": params.step_index,
        "state": {
            "settings": game.state.settings,
            "tiles": tiles,
            "structures": game.state.structures,
            "resources": game.state.resources,
            "tribes": game.state.tribes,
            "_hiddenResources": game.state._hidden_resources,
            "_prediction": game.state._prediction,
            "_messages": game.state._messages,
        },
        "userMove": {
            "json": user_move_json,
            "description": user_move_desc
        },
        "aiMove": {
            "json": ai_move_json,
            "description": ai_move_desc
        },
        "mctsAnalysis": mcts_analysis,
        "evaluation": build_evaluation_json(&game.state)
    }))
}

async fn list_initial_endpoint() -> Json<Value> {
    let path = "../src/scraper/data/training-data/";
    let mut files = Vec::new();

    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            if let Ok(name) = entry.file_name().into_string() {
                if name.ends_with(".initial.json") {
                    let id = name.replace(".initial.json", "");
                    files.push(id);
                }
            }
        }
    }

    files.sort();

    Json(serde_json::json!({
        "status": "success",
        "files": files
    }))
}
