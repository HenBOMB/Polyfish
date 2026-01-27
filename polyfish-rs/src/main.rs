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

    let shared_state = Arc::new(AppState {
        game: Mutex::new(game),
    });

    // Build our application with a route
    let app = Router::new()
        .route("/current", get(get_current_state))
        .route("/autostep", post(auto_step))
        .nest_service("/", ServeDir::new("../src/public"))
        .layer(CorsLayer::permissive())
        .with_state(shared_state);

    // Run our app
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    println!("Listening on http://localhost:3000");
    axum::serve(listener, app).await.unwrap();
}

async fn get_current_state(State(state): State<Arc<AppState>>) -> Json<Value> {
    let game = state.game.lock().unwrap();

    // Sort tiles by index to guarantee array order
    let mut tiles: Vec<_> = game.state.tiles.values().collect();
    tiles.sort_by_key(|t| t.coords.idx);

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
        }
    }))
}

async fn auto_step(State(state): State<Arc<AppState>>) -> Json<Value> {
    let mut game = state.game.lock().unwrap();

    use polyfish::ai::MctsAgent;
    let agent = MctsAgent::new(100); // 100 iterations per step

    let chosen_move = agent.select_move(&mut game);
    let move_names = if let Some(m) = chosen_move {
        let name = format!("{:?}", m.move_type());
        game.play_move(m.as_ref());
        vec![name]
    } else {
        vec!["none available".to_string()]
    };

    // Sort tiles for the response as well
    let mut tiles: Vec<_> = game.state.tiles.values().collect();
    tiles.sort_by_key(|t| t.coords.idx);

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
        "moves": move_names
    }))
}
