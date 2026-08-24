//! Standalone training dashboard: serves `training.html` and the metrics API
//! on port 3001 without requiring the main `polyfish` server. Run from
//! `polyfish-rs/` so relative paths (training_log.csv, ../src/public) resolve.
use axum::{Json, Router, routing::get};
use serde_json::Value;
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;

/// Columns kept as strings even when they parse as numbers: `run_id` is a unix
/// timestamp the dashboard compares and formats as text.
const CSV_TEXT_COLUMNS: &[&str] = &[
    "run_id",
    "iter_started_at",
    "run_started_at",
    "games_file",
    "match_type",
];

/// Every column of `training_log.csv` verbatim — numbers where the cell parses,
/// null for blanks. Reading the header instead of a fixed struct means a column
/// added to the CSV reaches the dashboard without a change here.
fn training_csv_rows() -> Vec<Value> {
    let content = std::fs::read_to_string("training_log.csv").unwrap_or_default();
    let mut lines = content.lines().filter(|l| !l.trim().is_empty());
    let Some(header) = lines.next() else {
        return Vec::new();
    };
    let headers: Vec<&str> = header.split(',').collect();
    lines
        .filter(|line| line.split(',').count() >= 5)
        .map(|line| {
            let cells: Vec<&str> = line.split(',').collect();
            let row: serde_json::Map<String, Value> = headers
                .iter()
                .enumerate()
                .map(|(i, name)| {
                    let cell = cells.get(i).copied().unwrap_or("").trim();
                    let value = if CSV_TEXT_COLUMNS.contains(name) {
                        Value::from(cell)
                    } else if cell.is_empty() {
                        Value::Null
                    } else {
                        cell.parse::<f64>()
                            .map(Value::from)
                            .unwrap_or_else(|_| Value::from(cell))
                    };
                    ((*name).to_string(), value)
                })
                .collect();
            Value::Object(row)
        })
        .collect()
}

async fn api_training_metrics(
    axum::extract::Query(q): axum::extract::Query<polyfish::training_api::RunFilter>,
) -> Json<Value> {
    let rows: Vec<Value> = training_csv_rows()
        .into_iter()
        .filter(|r| {
            q.run
                .as_ref()
                .is_none_or(|id| r.get("run_id").and_then(Value::as_str) == Some(id.as_str()))
        })
        .collect();
    Json(Value::Array(rows))
}

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
        .route("/api/training-metrics", get(api_training_metrics))
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
