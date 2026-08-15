//! Standalone training dashboard: serves `training.html` and the metrics API
//! on port 3001 without requiring the main `polyfish` server. Run from
//! `polyfish-rs/` so relative paths (training_log.csv, ../src/public) resolve.
use axum::{Json, Router, routing::get};
use serde_json::Value;
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;

async fn train_status() -> Json<Value> {
    let running = std::fs::read_to_string(".training.pid")
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
        .map(|pid| {
            std::process::Command::new("ps")
                .arg("-p")
                .arg(pid.to_string())
                .output()
                .map(|out| out.status.success())
                .unwrap_or(false)
        })
        .unwrap_or(false);
    Json(serde_json::json!({ "running": running }))
}

#[tokio::main]
async fn main() {
    let app = Router::new()
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
        .route(
            "/api/elo-ladder",
            get(polyfish::training_api::api_elo_ladder),
        )

        .route("/train/status", get(train_status))
        .fallback_service(ServeDir::new("../src/public"))
        .layer(CorsLayer::permissive());

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3001").await.unwrap();
    println!("Training dashboard: http://localhost:3001/training.html");
    axum::serve(listener, app).await.unwrap();
}
