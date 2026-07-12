use axum::{
    Json, Router,
    extract::State,
    handler::HandlerWithoutStateExt,
    routing::{get, post},
};
use polyfish::mapgen::{MapGenSettings, generate};
use polyfish::moves::{
    AttackMove, BuildMove, CaptureMove, DisbandMove, EndTurnMove, HarvestMove, Move, RecoverMove,
    ResearchMove, ResignMove, RewardMove, StepMove, SummonMove, UpgradeMove,
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
    network: Option<Arc<polyfish::ai::network::PolyZeroNet>>, // Trained neural network
    recorder: Arc<GameRecorder>,
}

const DEFAULT_TRIBES: &[TribeType] = &[TribeType::Imperius, TribeType::Imperius];
const DEFAULT_SIZE: MapSize = MapSize::Tiny;

#[tokio::main]
async fn main() {
    let _ = dotenvy::dotenv();

    // Initialize game
    let mut game = Game::new();
    let mut loaded = false;

    // 0. Try to rip live state from running Polytopia process via C++ reader
    // println!("🔍 Searching for Polytopia process...");
    // let pids = std::process::Command::new("pgrep")
    //     .arg("-x")
    //     .arg("Polytopia")
    //     .output()
    //     .ok()
    //     .map(|o| String::from_utf8_lossy(&o.stdout).to_string());

    // 1. Try to load live_game.json or saved_state.json (direct mod save)
    let candidates = ["live_game.json", "saved_state.json"];
    for &filename in &candidates {
        if let Ok(json) = std::fs::read_to_string(filename) {
            if let Ok(state) = serde_json::from_str::<polyfish::states::GameState>(&json) {
                println!("✅ Loading live game from {}", filename);
                game.state = state;
                loaded = true;
                break;
            }
        }
    }

    // 2. Try to load the latest mod replay from replays/ directory
    if !loaded {
        if let Ok(entries) = std::fs::read_dir("replays") {
            let mut mod_replays: Vec<_> = entries
                .flatten()
                .filter(|e| e.file_name().to_string_lossy().starts_with("mod_replay_"))
                .collect();

            // Sort by modification time (latest first)
            mod_replays.sort_by_key(|e| e.metadata().and_then(|m| m.modified()).ok());
            if let Some(latest) = mod_replays.last() {
                if let Ok(json) = std::fs::read_to_string(latest.path()) {
                    // Replay might be GameState OR { turns, gameState }
                    let val: Value = serde_json::from_str(&json).unwrap_or(Value::Null);
                    let state_res = if val["gameState"].is_object() {
                        serde_json::from_value::<polyfish::states::GameState>(
                            val["gameState"].clone(),
                        )
                    } else {
                        serde_json::from_value::<polyfish::states::GameState>(val)
                    };

                    if let Ok(state) = state_res {
                        println!(
                            "✅ Loading live game from latest replay: {:?}",
                            latest.file_name()
                        );
                        game.state = state;
                        loaded = true;
                    }
                }
            }
        }
    }

    if !loaded {
        println!("🎲 No live game found, generating new map...");
        game.state = generate(MapGenSettings {
            size: MapSize::Tiny,
            seed: 12345,
            map_type: MapType::Drylands,
            tribes: vec![TribeType::Imperius, TribeType::Oumaji],
            version: 115,
        });
    }

    game.state.settings._verbose = true;
    game.state.settings.max_turns = 10;
    game.post_load();

    // Load trained neural network. NOTE: must be a file-backed VarBuilder —
    // `VarMap::load` on an empty map is a silent no-op (it only fills
    // pre-registered vars), which used to leave the server on random weights.
    use candle_core::Device;
    let device = Device::Cpu;

    let model_path = "model.safetensors";

    let network = if std::path::Path::new(model_path).exists() {
        println!("✅ Loading trained AI model from {}", model_path);
        let vs = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                &[model_path],
                candle_core::DType::F32,
                &device,
            )
        }
        .expect("Failed to load model weights");
        polyfish::ai::network::PolyZeroNet::new(vs).expect("Failed to build neural network")
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
        network: Some(Arc::new(network)),
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
        .route("/train/halt", post(halt_training))
        .route("/metrics", get(get_metrics))
        .route("/config", post(set_config).get(get_config))
        .route("/save_training_data", post(save_training_data))
        .route("/analyze", get(analyze_game))
        .route("/save", post(save_game))
        .route("/load", post(load_game))
        .route("/simulate/explorer", post(simulate_explorer))
        .route("/simulate/attack", post(simulate_attack))
        .route("/replay/check", post(check_replay_exists))
        .route("/replay/save", post(save_replay_endpoint))
        .route("/replay/save-local", post(save_replay_local_endpoint))
        .route("/replay/load", post(load_replay_endpoint))
        .route("/replay/analyze", post(analyze_replay_step))
        .route("/replay/load_initial", post(load_initial_endpoint))
        .route("/replay/list_initial", get(list_initial_endpoint))
        .route("/trainer/hint", post(get_trainer_hint))
        .route("/system/cpu", get(get_cpu_usage))
        .route("/api/runs", get(polyfish::training_api::api_runs))
        .route(
            "/api/training-metrics",
            get(polyfish::training_api::api_training_metrics),
        )
        .route(
            "/api/moves-by-turn",
            get(polyfish::training_api::api_moves_by_turn),
        )
        .route(
            "/api/value-distribution",
            get(polyfish::training_api::api_value_distribution),
        )
        .route("/api/elo-ladder", get(polyfish::training_api::api_elo_ladder))
        .nest_service("/assets", ServeDir::new("../src/public/assets"))
        .nest_service("/simulator", ServeDir::new("../polyfish-ui/dist/simulator"))
        .nest_service("/static", ServeDir::new("../src/public"))
        // Serve real files (training.html, js/, css/) from public; unmatched
        // paths still fall back to index.html for SPA routing.
        .fallback_service(
            ServeDir::new("../src/public").not_found_service(spa_fallback.into_service()),
        )
        .layer(CorsLayer::permissive())
        .layer(axum::extract::DefaultBodyLimit::max(1024 * 1024 * 50))
        .with_state(shared_state);

    // Run our app
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    println!("Listening on http://localhost:3000");
    axum::serve(listener, app).await.unwrap();
}

/// Serialize `state.tribes` for the client, injecting a computed `maxHealth`
/// into every unit.
fn tribes_json_with_max_health(state: &polyfish::states::GameState) -> Value {
    use polyfish::functions::get_unit_max_health;
    let mut tribes = serde_json::Map::new();
    for (pid, tribe) in &state.tribes {
        let mut tribe_val = serde_json::to_value(tribe).unwrap_or(Value::Null);
        if let Some(units) = tribe_val.get_mut("units").and_then(|v| v.as_array_mut()) {
            for (u_val, u_state) in units.iter_mut().zip(tribe.units.iter()) {
                u_val["maxHealth"] = serde_json::json!(get_unit_max_health(u_state));
            }
        }
        tribes.insert(pid.to_string(), tribe_val);
    }
    Value::Object(tribes)
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
            "tribes": tribes_json_with_max_health(&game.state),
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
    game.state.settings._verbose = false;

    let mut move_name = "none".to_string();
    let mut best_move_json = None;
    let mut policy = serde_json::json!({});

    // 1. Try to get AI move from trained model if available
    if let Some(net) = &state.network {
        use polyfish::ai::brain::Brain;
        use polyfish::ai::eval_server::{Evaluator, InlineEvalHandle};
        let evaluator = Evaluator::Inline(InlineEvalHandle::new(net.clone()));
        let mut brain = Brain::with_backend(&evaluator, params.iterations, polyfish::ai::brain::SearchBackend::Gumbel { k: 16 })
            .with_prior_heuristic_weight(0.1)
            .with_policy_target_q_weight(1.0)
            .with_tree_q_weight(1.0);
        game.state._messages.clear();
        let (chosen_move, brain_policy) = brain.think_with_stats(&mut game);
        policy = brain_policy.into();
        best_move_json = chosen_move.as_ref().map(|m| m.serialize());

        if !params.dry_run {
            if let Some(m) = chosen_move {
                move_name = format!("{:?}", m.move_type());
                game.play_move(m.as_ref());
            }
        }
    }

    // 2. Run heuristic MCTS for analysis panel (move descriptions + PV)
    // This also acts as a fallback for move selection if no network is available
    use polyfish::ai::heuristic_mcts::HeuristicMctsAgent;
    let analysis_agent = HeuristicMctsAgent {
        iterations: params.iterations,
        exploration_constant: 0.1,
    };
    let (h_best_move, mcts_analysis) = analysis_agent.select_move_with_analysis(&mut game);

    // If we don't have a network but we ARE supposed to move, use the heuristic best move
    if state.network.is_none() && !params.dry_run {
        if let Some(m) = h_best_move.as_ref() {
            move_name = format!("{:?}", m.move_type());
            best_move_json = Some(m.serialize());
            game.play_move(m.as_ref());
        }
    } else if state.network.is_none() {
        // Just for dry_run analysis when network is missing
        best_move_json = h_best_move.as_ref().map(|m| m.serialize());
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
            "tribes": tribes_json_with_max_health(&game.state),
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
            "tribes": tribes_json_with_max_health(&game.state),
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
                AbilityType::Swarm => Box::new(BoostMove::new(src)),
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
        11 => Box::new(ResignMove),
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
            "tribes": tribes_json_with_max_health(&game.state),
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
    game.state.settings._verbose = true;
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
            "tribes": tribes_json_with_max_health(&game.state),
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

    let child = Command::new("bash")
        .args(["run_training_loop.sh", "-n"])
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

    let mut pid_opt = *state.training_status.lock().unwrap();
    let mut is_running = false;

    if let Some(pid) = pid_opt {
        // Check if process still exists
        let output = Command::new("ps").arg("-p").arg(pid.to_string()).output();

        if let Ok(out) = output {
            is_running = out.status.success();
        }
    }

    // If not running via internal state, check PID file written by run_training_loop.sh
    if !is_running {
        if let Ok(pid_str) = fs::read_to_string(".training.pid") {
            if let Ok(parsed_pid) = pid_str.trim().parse::<u32>() {
                // Verify the PID is actually alive
                let alive = Command::new("ps").arg("-p").arg(parsed_pid.to_string()).output()
                    .map(|out| out.status.success()).unwrap_or(false);
                if alive {
                    is_running = true;
                    pid_opt = Some(parsed_pid);
                    *state.training_status.lock().unwrap() = Some(parsed_pid);
                } else {
                    // Stale PID file — process died, clean up
                    let _ = fs::remove_file(".training.pid");
                }
            }
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

async fn halt_training(State(state): State<Arc<AppState>>) -> Json<Value> {
    use std::process::Command;

    let pid_opt = *state.training_status.lock().unwrap();
    if let Some(pid) = pid_opt {
        // Kill child processes first to avoid orphans (like python train.py)
        let _ = Command::new("pkill").arg("-P").arg(pid.to_string()).output();
        // Kill the parent bash script
        let _ = Command::new("kill").arg("-9").arg(pid.to_string()).output();
        
        let mut status = state.training_status.lock().unwrap();
        *status = None;

        return Json(serde_json::json!({ "status": "success", "message": format!("Halted PID {}", pid) }));
    }
    Json(serde_json::json!({ "status": "error", "message": "No training process active" }))
}

async fn set_config(Json(payload): Json<Value>) -> Json<Value> {
    if std::fs::write("config.json", serde_json::to_string_pretty(&payload).unwrap_or_default()).is_ok() {
        return Json(serde_json::json!({ "status": "success", "config": payload }));
    }
    Json(serde_json::json!({ "status": "error", "message": "Failed to set config" }))
}

async fn get_config() -> Json<Value> {
    if let Ok(content) = std::fs::read_to_string("config.json") {
        if let Ok(json) = serde_json::from_str::<Value>(&content) {
            return Json(json);
        }
    }
    Json(serde_json::json!({ "cores": 12, "tribes": ["Imperius", "Imperius"] }))
}

async fn get_metrics() -> Json<Value> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    
    let mut metrics = Vec::new();
    if let Ok(file) = File::open("training_log.csv") {
        let reader = BufReader::new(file);
        let mut lines = reader.lines().flatten();
        
        let header_line = lines.next().unwrap_or_default();
        let headers: Vec<&str> = header_line.split(',').collect();
        
        for line in lines {
            let parts: Vec<&str> = line.split(',').collect();
            
            let get_idx = |name: &str| -> Option<usize> {
                headers.iter().position(|&h| h == name)
            };
            
            let parse_f32 = |name: &str| -> f32 {
                get_idx(name).and_then(|i| parts.get(i)).and_then(|s| s.parse().ok()).unwrap_or(0.0)
            };

            let obj = serde_json::json!({
                "iteration": get_idx("iteration").and_then(|i| parts.get(i)).and_then(|s| s.parse::<u32>().ok()).unwrap_or(0),
                "timestamp": get_idx("iter_started_at").and_then(|i| parts.get(i)).unwrap_or(&""),
                "avg_score": parse_f32("avg_score"),
                "max_score": parse_f32("max_score"),
                "p1_avg": parse_f32("p1_avg"),
                "p2_avg": parse_f32("p2_avg"),
                "loss": parse_f32("loss"),
                "policy_loss": parse_f32("policy_loss"),
                "value_loss": parse_f32("value_loss"),
                "value_r2": parse_f32("value_r2"),
                "avg_captures": parse_f32("avg_captures"),
                "avg_harvests": parse_f32("avg_harvests"),
                "avg_builds": parse_f32("avg_builds"),
                "avg_research": parse_f32("avg_research"),
                "avg_attacks": parse_f32("avg_attacks"),
                "avg_ability": parse_f32("avg_abilities"),
                "avg_steps": parse_f32("avg_moves"),
                "avg_spt_t10": parse_f32("avg_spt_t10"),
                "avg_spt_t20": parse_f32("avg_spt_t20"),
                "avg_spt_t30": parse_f32("avg_spt_t30"),
                "villages_t2c_first": parse_f32("villages_t2c_first"),
                "ruins_t2c_p50": parse_f32("ruins_t2c_p50"),
            });
            metrics.push(obj);
        }
    }
    
    let len = metrics.len();
    if len > 100 {
        metrics = metrics.into_iter().skip(len - 100).collect();
    }
    
    Json(serde_json::json!({ "metrics": metrics }))
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

async fn save_game(State(state): State<Arc<AppState>>, body: Option<Json<Value>>) -> Json<Value> {
    let mut game = state.game.lock().unwrap();

    // If we received a state in the body, try to ingest it
    if let Some(Json(_val)) = body {
        match serde_json::from_value::<polyfish::states::GameState>(_val) {
            Ok(new_state) => {
                game.state = new_state;
                game.post_load();
                println!("✅ Ingested GameState from request body");
            }
            Err(e) => {
                return Json(serde_json::json!({
                    "status": "error",
                    "message": format!("Failed to parse GameState: {}", e)
                }));
            }
        }
    }

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
            "tribes": tribes_json_with_max_health(&game.state),
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

    let (best_move, mcts_analysis) = if let Some(net) = &state.network {
        // 1. Use Neural Network (Gumbel MCTS) if available
        use polyfish::ai::brain::Brain;
        use polyfish::ai::eval_server::{Evaluator, InlineEvalHandle};
        let evaluator = Evaluator::Inline(InlineEvalHandle::new(net.clone()));
        let mut brain = Brain::with_backend(&evaluator, params.iterations, polyfish::ai::brain::SearchBackend::Gumbel { k: 16 })
            .with_prior_heuristic_weight(0.1)
            .with_policy_target_q_weight(1.0)
            .with_tree_q_weight(1.0);
        let (bm, _stats) = brain.think_with_stats(&mut game);
        // Brain currently returns stats as Vec<f32>, convert to simpler MCTS analysis or similar
        // For visual consistency, we actually prefer the full analysis from heuristic agent
        // but we'll use the brain's move.
        (
            bm,
            polyfish::ai::mcts::MctsAnalysis {
                evaluations: vec![],
                total_iterations: params.iterations,
                principal_variation: vec![],
                tree: None,
            },
        )
    } else {
        // 2. Fallback to Heuristic MCTS
        use polyfish::ai::heuristic_mcts::HeuristicMctsAgent;
        let agent = HeuristicMctsAgent {
            iterations: params.iterations,
            exploration_constant: 0.4,
        };
        agent.select_move_with_analysis(&mut game)
    };

    // If we used the brain, let's still run a tiny heuristic analysis for the PV/Tree if requested?
    // Actually, let's keep it simple: if Brain is used, we get the move.
    // If the user wants full analysis panels, they use the passive toggle.

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
struct CheckReplayParams {
    seed: Option<u64>,
    game_name: Option<String>,
    uuid: Option<String>,
}

async fn check_replay_exists(Json(params): Json<CheckReplayParams>) -> Json<Value> {
    let supabase_key = std::env::var("SUPABASE_SERVICE_ROLE_KEY")
        .or_else(|_| std::env::var("SUPABASE_PUBLIC_ANON_KEY"))
        .unwrap_or_default();
    let supabase_url = std::env::var("SUPABASE_URL").unwrap_or_default();

    if supabase_url.is_empty() || supabase_key.is_empty() {
        return Json(serde_json::json!({
            "exists": false,
            "proceed": true,
            "message": "Supabase not configured, proceeding by default"
        }));
    }

    let db_url = if let Some(uuid) = &params.uuid {
        format!(
            "{}/rest/v1/games?uuid=eq.{}&select=id",
            supabase_url.trim_end_matches('/'),
            uuid
        )
    } else if let (Some(seed), Some(name)) = (params.seed, &params.game_name) {
        let safe_game_name = name.replace(" ", "%20");
        format!(
            "{}/rest/v1/games?seed=eq.{}&game_name=eq.{}&select=id",
            supabase_url.trim_end_matches('/'),
            seed,
            safe_game_name
        )
    } else {
        return Json(serde_json::json!({
            "exists": false,
            "proceed": true,
            "message": "Neither uuid nor seed/game_name provided"
        }));
    };

    let client = reqwest::Client::new();
    let req = client
        .get(&db_url)
        .header("apikey", &supabase_key)
        .header("Authorization", format!("Bearer {}", supabase_key));

    match req.send().await {
        Ok(res) => match res.json::<serde_json::Value>().await {
            Ok(json_val) => {
                let exists = json_val
                    .as_array()
                    .map(|arr| !arr.is_empty())
                    .unwrap_or(false);
                let check_id = if let Some(u) = &params.uuid {
                    format!("uuid='{}'", u)
                } else {
                    format!("seed={:?}, name={:?}", params.seed, params.game_name)
                };
                println!("🔍 Replay check {}: exists={}", check_id, exists);
                Json(serde_json::json!({
                    "exists": exists,
                    "proceed": !exists
                }))
            }
            Err(e) => {
                println!("❌ Supabase check parse error: {}", e);
                Json(serde_json::json!({
                    "exists": false,
                    "proceed": true,
                    "message": format!("Parse error: {}", e)
                }))
            }
        },
        Err(e) => {
            println!("❌ Supabase check network error: {}", e);
            Json(serde_json::json!({
                "exists": false,
                "proceed": true,
                "message": format!("Network error: {}", e)
            }))
        }
    }
}

fn sanitize_storage_key(name: &str) -> String {
    let mut result = String::new();
    let mut last_was_dash = false;
    for c in name.chars() {
        if c.is_ascii_alphanumeric() || c == '_' {
            result.push(c.to_ascii_lowercase());
            last_was_dash = false;
        } else {
            if !last_was_dash && !result.is_empty() {
                result.push('-');
                last_was_dash = true;
            }
        }
    }
    result.trim_matches('-').to_string()
}

async fn save_replay_endpoint(
    State(state): State<Arc<AppState>>,
    body: Json<Value>,
) -> Json<Value> {
    // Create replays directory if not exists
    let _ = std::fs::create_dir_all("replays");

    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let (filename, content) = if body["turns"].is_array() {
        // get the name from body.gameState.settings.gameName
        let game_name = body["gameState"]["settings"]["gameName"]
            .as_str()
            .unwrap_or("Unknown");
        let seed = body["gameState"]["initial_seed"]
            .as_u64()
            .or_else(|| body["gameState"]["settings"]["seed"].as_u64())
            .unwrap_or(0);

        let supabase_key = std::env::var("SUPABASE_SERVICE_ROLE_KEY")
            .or_else(|_| std::env::var("SUPABASE_PUBLIC_ANON_KEY"))
            .unwrap_or_default();
        let supabase_url = std::env::var("SUPABASE_URL").unwrap_or_default();

        if !supabase_url.is_empty() && !supabase_key.is_empty() {
            let client = reqwest::Client::new();

            let db_url = if let Some(uuid) = body["uuid"].as_str() {
                format!(
                    "{}/rest/v1/games?uuid=eq.{}&select=id",
                    supabase_url.trim_end_matches('/'),
                    uuid
                )
            } else {
                let safe_game_name = game_name.replace(" ", "%20");
                format!(
                    "{}/rest/v1/games?seed=eq.{}&game_name=eq.{}&select=id",
                    supabase_url.trim_end_matches('/'),
                    seed,
                    safe_game_name
                )
            };

            let req = client
                .get(&db_url)
                .header("apikey", &supabase_key)
                .header("Authorization", format!("Bearer {}", supabase_key));

            if let Ok(res) = req.send().await {
                if let Ok(json) = res.json::<serde_json::Value>().await {
                    if let Some(arr) = json.as_array() {
                        if !arr.is_empty() {
                            println!(
                                "⚠️ Rejected duplicate game (UUID or Seed/Name): {}",
                                game_name
                            );
                            return Json(serde_json::json!({
                                "status": "error",
                                "message": "Duplicate game found"
                            }));
                        }
                    }
                }
            }

            // Record does not exist, upload to Supabase Storage directly!
            let bucket_name =
                std::env::var("SUPABASE_STORAGE_BUCKET").unwrap_or_else(|_| "games".to_string());
            let file_name = format!("{}_{}.json", sanitize_storage_key(game_name), timestamp);
            let storage_url = format!(
                "{}/storage/v1/object/{}/{}",
                supabase_url.trim_end_matches('/'),
                bucket_name,
                file_name
            );

            let upload_req = client
                .post(&storage_url)
                .header("apikey", &supabase_key)
                .header("Authorization", format!("Bearer {}", supabase_key))
                .header("Content-Type", "application/json")
                .body(serde_json::to_string(&*body).unwrap_or_default());

            match upload_req.send().await {
                Ok(res) => {
                    if !res.status().is_success() {
                        let err_text = res.text().await.unwrap_or_default();
                        println!("❌ Supabase Storage Upload Failed: {}", err_text);
                        return Json(serde_json::json!({
                            "status": "error",
                            "message": format!("Supabase Storage upload failed: {}", err_text)
                        }));
                    }
                }
                Err(e) => {
                    println!("❌ Supabase Storage Network Error: {}", e);
                    return Json(serde_json::json!({
                        "status": "error",
                        "message": format!("Supabase Storage network error: {}", e)
                    }));
                }
            }

            // Insert record into games table to prevent future duplicates
            let insert_url = format!("{}/rest/v1/games", supabase_url.trim_end_matches('/'));

            // Extract UUID if present (from the root of the serialized replay)
            let uuid_val = body["uuid"].as_str().unwrap_or("").to_string();
            let mut insert_payload = serde_json::json!({
                "seed": seed,
                "game_name": game_name,
                "storage_path": file_name,
                "verified": false
            });
            if !uuid_val.is_empty() {
                insert_payload
                    .as_object_mut()
                    .unwrap()
                    .insert("uuid".into(), serde_json::json!(uuid_val));
            }

            let insert_req = client
                .post(&insert_url)
                .header("apikey", &supabase_key)
                .header("Authorization", format!("Bearer {}", supabase_key))
                .header("Content-Type", "application/json")
                .header("Prefer", "return=minimal")
                .json(&insert_payload);

            match insert_req.send().await {
                Ok(res) => {
                    if !res.status().is_success() {
                        let err_text = res.text().await.unwrap_or_default();
                        println!("❌ Supabase DB Insert Failed: {}", err_text);
                        return Json(serde_json::json!({
                            "status": "error",
                            "message": format!("Supabase DB insert failed: {}", err_text)
                        }));
                    }
                }
                Err(e) => {
                    println!("❌ Supabase DB Network Error: {}", e);
                }
            }

            println!("✅ Successfully uploaded {} to Supabase Storage", game_name);
            return Json(serde_json::json!({
                "status": "success",
                "message": format!("Replay uploaded to Supabase Storage bucket '{}'", bucket_name),
                "filename": file_name
            }));
        }

        println!(
            "✅ Received {} Replay data from mod (Saved Locally)",
            game_name
        );
        (
            format!(
                "replays/{}_{}.json",
                sanitize_storage_key(game_name),
                timestamp
            ),
            serde_json::to_string_pretty(&*body).unwrap(),
        )
    } else if let Some(name) = body["name"].as_str() {
        // Legacy: save current server state under this name
        let game = state.game.lock().unwrap();
        let safe_name: String = name
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '_')
            .collect();
        (
            format!("replays/{}_{}.json", safe_name, timestamp),
            serde_json::to_string_pretty(&game.state).unwrap(),
        )
    } else {
        return Json(
            serde_json::json!({ "status": "error", "message": "Invalid replay data format" }),
        );
    };

    match std::fs::write(&filename, content) {
        Ok(_) => Json(serde_json::json!({
            "status": "success",
            "message": format!("Replay saved locally to {}", filename),
            "filename": filename
        })),
        Err(e) => Json(serde_json::json!({
            "status": "error",
            "message": format!("Failed to save replay locally: {}", e)
        })),
    }
}

// instead of saving to the db, save to /replays
async fn save_replay_local_endpoint(
    State(state): State<Arc<AppState>>,
    body: Json<Value>,
) -> Json<Value> {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let game_name = body["gameState"]["settings"]["gameName"]
        .as_str()
        .unwrap_or("Unknown");

    let (filename, content) = if body["turns"].is_array() {
        // get the name from body.gameState.settings.gameName

        let _seed = body["gameState"]["initial_seed"]
            .as_u64()
            .or_else(|| body["gameState"]["settings"]["seed"].as_u64())
            .unwrap_or(0);

        (
            format!(
                "replays/{}_{}.json",
                sanitize_storage_key(game_name),
                timestamp
            ),
            serde_json::to_string_pretty(&*body).unwrap(),
        )
    } else if let Some(name) = body["name"].as_str() {
        // Legacy: save current server state under this name
        let game = state.game.lock().unwrap();
        let safe_name: String = name
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '_')
            .collect();
        (
            format!("replays/{}_{}.json", safe_name, timestamp),
            serde_json::to_string_pretty(&game.state).unwrap(),
        )
    } else {
        return Json(
            serde_json::json!({ "status": "error", "message": "Invalid replay data format" }),
        );
    };

    println!("✅ Successfully saved {} to Local Storage", game_name);

    match std::fs::write(&filename, content) {
        Ok(_) => Json(serde_json::json!({
            "status": "success",
            "message": format!("Replay saved locally to {}", filename),
            "filename": filename
        })),
        Err(e) => Json(serde_json::json!({
            "status": "error",
            "message": format!("Failed to save replay locally: {}", e)
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
                        "tribes": tribes_json_with_max_health(&game.state),
                        "_prediction": game.state._prediction,
                        "_messages": game.state._messages,
                        "history": game.state._history,
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
            let tribe_count = root["tribes"].as_object().map(|o| o.len()).unwrap_or(2);

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
                if !settings.contains_key("turn") {
                    settings.insert("turn".into(), serde_json::json!(0));
                } else if settings["turn"].as_i64().unwrap_or(0) == 0 {
                    settings.insert("turn".into(), serde_json::json!(0));
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
                    // Tribe ids start at 1
                    settings.insert("_lastPlayerTurnId".into(), serde_json::json!(1));
                }
                if !settings.contains_key("_recentMoves") {
                    settings.insert("_recentMoves".into(), serde_json::json!([]));
                }
                if !settings.contains_key("_maxTribeCount") {
                    settings.insert("_maxTribeCount".into(), serde_json::json!(tribe_count));
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
                            "tribes": tribes_json_with_max_health(&game.state),
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
    if !state.network.is_some() {
        return Json(serde_json::json!({
            "status": "error",
            "message": "No trained network available"
        }));
    }

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

    if replay_state._history.len() <= params.step_index {
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
        if i >= replay_state._history.len() {
            break;
        }

        let move_json = &replay_state._history[i];

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
    use polyfish::ai::eval_server::{Evaluator, InlineEvalHandle};
    use polyfish::ai::mcts_zero::ZeroMctsAgent;
    let evaluator = Evaluator::Inline(InlineEvalHandle::new(state.network.as_ref().unwrap().clone()));
    let agent = ZeroMctsAgent::new(&evaluator, params.iterations);
    let (best_move, mcts_analysis) = agent.select_move_with_stats(&mut game);

    let ai_move_json = best_move.as_ref().map(|m: &Box<dyn polyfish::moves::Move>| m.serialize());
    let ai_move_desc = best_move
        .as_ref()
        .map(|m: &Box<dyn polyfish::moves::Move>| m.describe(&game.state))
        .unwrap_or("None".to_string());

    // User's actual move
    let user_move_json = &replay_state._history[params.step_index];
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
            "tribes": tribes_json_with_max_health(&game.state),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_storage_key() {
        assert_eq!(
            sanitize_storage_key("The Winter of Love"),
            "the-winter-of-love"
        );
        assert_eq!(
            sanitize_storage_key("game-4-(yădakk-qualifiers-"),
            "game-4-y-dakk-qualifiers"
        );
        assert_eq!(sanitize_storage_key("Hello World!!!"), "hello-world");
        assert_eq!(
            sanitize_storage_key("---Multiple---Dashes---"),
            "multiple-dashes"
        );
        assert_eq!(sanitize_storage_key("UPPER_case_123"), "upper_case_123");
        assert_eq!(sanitize_storage_key(""), "");
        assert_eq!(sanitize_storage_key("!@#$%^&*()"), "");
    }
}

async fn get_cpu_usage() -> Json<Value> {
    fn read_cpu() -> Vec<(f64, f64)> {
        let mut cores = Vec::new();
        if let Ok(stat) = std::fs::read_to_string("/proc/stat") {
            for line in stat.lines().filter(|l| l.starts_with("cpu")) {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() > 7 {
                    let user: f64 = parts[1].parse().unwrap_or(0.0);
                    let nice: f64 = parts[2].parse().unwrap_or(0.0);
                    let system: f64 = parts[3].parse().unwrap_or(0.0);
                    let idle: f64 = parts[4].parse().unwrap_or(0.0);
                    let iowait: f64 = parts[5].parse().unwrap_or(0.0);
                    let irq: f64 = parts[6].parse().unwrap_or(0.0);
                    let softirq: f64 = parts[7].parse().unwrap_or(0.0);
                    let total = user + nice + system + idle + iowait + irq + softirq;
                    let active = total - idle - iowait;
                    cores.push((active, total));
                }
            }
        }
        cores
    }

    let mut usages = Vec::new();
    let stats1 = read_cpu();
    if !stats1.is_empty() {
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        let stats2 = read_cpu();
        if stats1.len() == stats2.len() {
            for i in 0..stats1.len() {
                let d_active = stats2[i].0 - stats1[i].0;
                let d_total = stats2[i].1 - stats1[i].1;
                let usage = if d_total > 0.0 { (d_active / d_total) * 100.0 } else { 0.0 };
                usages.push(usage);
            }
        }
    }
    
    Json(serde_json::json!({ "cores": usages }))
}

async fn spa_fallback() -> impl axum::response::IntoResponse {
    use axum::response::IntoResponse;
    let index_path = std::path::Path::new("../polyfish-ui/dist/index.html");
    match std::fs::read_to_string(index_path) {
        Ok(html) => axum::response::Html(html).into_response(),
        Err(_) => (axum::http::StatusCode::NOT_FOUND, "index.html not found").into_response(),
    }
}
