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
        .route("/simulate/explorer", post(simulate_explorer))
        .route("/simulate/attack", post(simulate_attack))
        .nest_service("/", ServeDir::new("../src/public"))
        .layer(CorsLayer::permissive())
        .with_state(shared_state);

    // Run our app
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    println!("Listening on http://localhost:3000");
    axum::serve(listener, app).await.unwrap();
}

// ... (stats helper functions omitted for brevity if needed) ...

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
    100
}

async fn get_current_state(State(state): State<Arc<AppState>>) -> Json<Value> {
    let mut game = state.game.lock().unwrap();
    polyfish::prediction::update_predictions(&mut game.state);

    let mut tiles: Vec<_> = game.state.tiles.values().collect();
    tiles.sort_by_key(|t| t.coords.idx);

    let legal_moves: Vec<_> = game.legal_moves().iter().map(|m| m.serialize()).collect();

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
        "legalMoves": legal_moves
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

    let mut tiles: Vec<_> = game.state.tiles.values().collect();
    tiles.sort_by_key(|t| t.coords.idx);

    let legal_moves: Vec<_> = game.legal_moves().iter().map(|m| m.serialize()).collect();

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
        "policyDistribution": policy
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
        "legalMoves": legal_moves
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
                        .or(payload["target"].as_i64())
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
                        .or(payload["target"].as_i64())
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
    use polyfish::ai::heuristics::evaluate_state_heuristics;
    let pid = game.state.settings.current_player_turn_id;
    let (eco, mil) = evaluate_state_heuristics(&game.state, pid);

    state
        .recorder
        .record_step(&game.state, move_obj.as_ref(), eco, mil);

    let move_name = format!("{:?}", move_obj.move_type());
    game.state._messages.clear();
    game.play_move(move_obj.as_ref());

    let (eco1, mil1) = evaluate_state_heuristics(&game.state, pid);

    println!(
        "Manual step: player={}, move={:?}, eco={} ({}{}), mil={} ({}{})",
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
        "legalMoves": legal_moves
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

    let initial_state = generate(settings);
    game.state = initial_state;
    game.state.settings.verbose = true;
    game.post_load();

    let mut tiles: Vec<_> = game.state.tiles.values().collect();
    tiles.sort_by_key(|t| t.coords.idx);

    let legal_moves: Vec<_> = game.legal_moves().iter().map(|m| m.serialize()).collect();

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
        "legalMoves": legal_moves
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
