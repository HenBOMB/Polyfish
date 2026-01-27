use axum::{
    extract::State,
    routing::{get, post},
    Json, Router,
};
use polyfish::game::Game;
use polyfish::mapgen::{generate, MapGenSettings};
use polyfish::types::TribeType;
use serde_json::Value;
use std::sync::{Arc, Mutex};
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;

struct AppState {
    game: Mutex<Game>,
}

#[tokio::main]
async fn main() {
    // Initialize game
    let mut settings = MapGenSettings::default();
    settings.size = 16;
    settings.tribes = vec![TribeType::Luxidoor, TribeType::Imperius];
    settings.seed = 42;

    let initial_state = generate(settings);
    let mut game = Game::new();
    game.state = initial_state;
    game.post_load();

    let shared_state = Arc::new(AppState {
        game: Mutex::new(game),
    });

    // Build our application with routes
    let app = Router::new()
        .route("/current", get(get_current_state))
        .route("/autostep", post(auto_step))
        .route("/rngstep", post(rng_step))
        .route("/reset", post(reset_game))
        .nest_service("/", ServeDir::new("../src/public"))
        .layer(CorsLayer::permissive())
        .with_state(shared_state);

    // Run our app
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    println!("Listening on http://localhost:3000");
    axum::serve(listener, app).await.unwrap();
}

#[derive(serde::Deserialize)]
struct StepParams {
    #[serde(default = "default_iterations")]
    iterations: usize,
}

fn default_iterations() -> usize {
    100
}

async fn get_current_state(State(state): State<Arc<AppState>>) -> Json<Value> {
    let game = state.game.lock().unwrap();

    let mut tiles: Vec<_> = game.state.tiles.values().collect();
    tiles.sort_by_key(|t| t.coords.idx);

    let legal_moves: Vec<_> = game
        .legal_moves()
        .iter()
        .map(|m| format!("{:?}", m.move_type()))
        .collect();

    Json(serde_json::json!({
        "state": {
            "settings": game.state.settings,
            "tiles": tiles,
            "structures": game.state.structures,
            "resources": game.state.resources,
            "tribes": game.state.tribes,
            "_visibleTiles": game.state._visible_tiles,
            "_hiddenResources": game.state._hidden_resources,
            "_prediction": game.state._prediction,
        },
        "legalMoves": legal_moves
    }))
}

async fn auto_step(
    State(state): State<Arc<AppState>>,
    Json(params): Json<StepParams>,
) -> Json<Value> {
    let mut game = state.game.lock().unwrap();

    use polyfish::ai::MctsAgent;
    let agent = MctsAgent::new(params.iterations);

    let chosen_move = agent.select_move(&mut game);
    let mut move_name = "none".to_string();
    if let Some(m) = chosen_move {
        move_name = format!("{:?}", m.move_type());
        game.play_move(m.as_ref());
    }

    let mut tiles: Vec<_> = game.state.tiles.values().collect();
    tiles.sort_by_key(|t| t.coords.idx);

    let legal_moves: Vec<_> = game
        .legal_moves()
        .iter()
        .map(|m| format!("{:?}", m.move_type()))
        .collect();

    Json(serde_json::json!({
        "state": {
            "settings": game.state.settings,
            "tiles": tiles,
            "structures": game.state.structures,
            "resources": game.state.resources,
            "tribes": game.state.tribes,
            "_visibleTiles": game.state._visible_tiles,
            "_hiddenResources": game.state._hidden_resources,
            "_prediction": game.state._prediction,
        },
        "movePlayed": move_name,
        "legalMoves": legal_moves
    }))
}

async fn rng_step(State(state): State<Arc<AppState>>) -> Json<Value> {
    let mut game = state.game.lock().unwrap();
    let moves = game.legal_moves();

    let mut move_name = "none".to_string();
    if !moves.is_empty() {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        let chosen = moves.choose(&mut rng).unwrap();
        move_name = format!("{:?}", chosen.move_type());
        game.play_move(chosen.as_ref());
    }

    let mut tiles: Vec<_> = game.state.tiles.values().collect();
    tiles.sort_by_key(|t| t.coords.idx);

    let legal_moves: Vec<_> = game
        .legal_moves()
        .iter()
        .map(|m| format!("{:?}", m.move_type()))
        .collect();

    Json(serde_json::json!({
        "state": {
            "settings": game.state.settings,
            "tiles": tiles,
            "structures": game.state.structures,
            "resources": game.state.resources,
            "tribes": game.state.tribes,
            "_visibleTiles": game.state._visible_tiles,
            "_hiddenResources": game.state._hidden_resources,
            "_prediction": game.state._prediction,
        },
        "movePlayed": move_name,
        "legalMoves": legal_moves
    }))
}

async fn reset_game(State(state): State<Arc<AppState>>) -> Json<Value> {
    let mut game = state.game.lock().unwrap();

    let mut settings = MapGenSettings::default();
    settings.size = 16;
    settings.tribes = vec![TribeType::Luxidoor, TribeType::Imperius];
    settings.seed = rand::random();

    let initial_state = generate(settings);
    game.state = initial_state;
    game.post_load();

    // Use existing get_current_state logic (just calling a helper would be better but let's keep it simple)
    let mut tiles: Vec<_> = game.state.tiles.values().collect();
    tiles.sort_by_key(|t| t.coords.idx);

    let legal_moves: Vec<_> = game
        .legal_moves()
        .iter()
        .map(|m| format!("{:?}", m.move_type()))
        .collect();

    Json(serde_json::json!({
        "state": {
            "settings": game.state.settings,
            "tiles": tiles,
            "structures": game.state.structures,
            "resources": game.state.resources,
            "tribes": game.state.tribes,
            "_visibleTiles": game.state._visible_tiles,
            "_hiddenResources": game.state._hidden_resources,
            "_prediction": game.state._prediction,
        },
        "legalMoves": legal_moves
    }))
}
