//! Read-only HTTP handlers for the training dashboard.

use axum::{
    Json,
    extract::Query,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use safetensors::SafeTensors;
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const CSV_PATH: &str = "training_log.csv";
const MOVES_PATH: &str = "moves_by_turn.json";
const TEMPO_PATH: &str = "tempo_by_turn.json";
const VALUE_DIST_PATH: &str = "value_distribution.json";
const LADDER_PATH: &str = "ladder.json";

#[derive(Debug, Clone)]
struct MetricRow {
    run_id: String,
    iter_started_at: String,
    iteration: i64,
    games_file: String,
    num_games: i64,
    tribes: String,
    avg_score: f64,
    max_score: f64,
    p1_avg: f64,
    p2_avg: f64,
    loss: f64,
    policy_loss: f64,
    value_loss: f64,
    value_r2: f64,
    aux_own_loss: f64,
    aux_fog_loss: f64,
    aux_spt_loss: f64,
    aux_tech_loss: f64,
    avg_captures: f64,
    avg_cap_ruins: f64,
    avg_cap_villages: f64,
    avg_cap_cities: f64,
    avg_cap_capitals: f64,
    avg_harvests: f64,
    avg_builds: f64,
    avg_research: f64,
    avg_attacks: f64,
    avg_revealed_tiles: f64,
    avg_captured_tiles: f64,
    avg_spt_t0: f64,
    avg_spt_t5: f64,
    avg_spt_t10: f64,
    avg_spt_t15: f64,
    avg_spt_t20: f64,
    avg_spt_t25: f64,
    avg_spt_t30: f64,
    villages_t2c_first: f64,
    villages_first_rate: f64,
    villages_t2c_first_cond: f64,
    villages_t2c_p50: f64,
    villages_t2c_p80: f64,
    villages_t2c_all: f64,
    ruins_t2c_p50: f64,
    ruins_t2c_p80: f64,
    ruins_t2c_all: f64,
    avg_moves: f64,
    match_type: String,
}

fn parse_f64(s: &str) -> f64 {
    s.trim().parse().unwrap_or(0.0)
}

fn parse_i64(s: &str) -> i64 {
    s.trim().parse().unwrap_or(0)
}

fn read_csv_rows() -> Vec<MetricRow> {
    let content = match std::fs::read_to_string(CSV_PATH) {
        Ok(c) => c,
        Err(_) => return vec![],
    };
    let mut lines = content.lines().filter(|l| !l.trim().is_empty()).peekable();
    let Some(header_line) = lines.next() else {
        return vec![];
    };
    let headers: Vec<&str> = header_line.split(',').collect();
    let idx = |name: &str| -> Option<usize> { headers.iter().position(|h| *h == name) };

    let col = |cols: &[&str], name: &str| -> String {
        idx(name)
            .and_then(|i| cols.get(i))
            .map(|s| s.to_string())
            .unwrap_or_default()
    };

    let mut rows = Vec::new();
    for line in lines {
        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() < 5 {
            continue;
        }
        rows.push(MetricRow {
            run_id: col(&cols, "run_id"),
            iter_started_at: {
                let v = col(&cols, "iter_started_at");
                if v.is_empty() {
                    col(&cols, "run_started_at")
                } else {
                    v
                }
            },
            iteration: parse_i64(&col(&cols, "iteration")),
            games_file: col(&cols, "games_file"),
            num_games: parse_i64(&col(&cols, "num_games")),
            tribes: col(&cols, "tribes"),
            avg_score: parse_f64(&col(&cols, "avg_score")),
            max_score: parse_f64(&col(&cols, "max_score")),
            p1_avg: parse_f64(&col(&cols, "p1_avg")),
            p2_avg: parse_f64(&col(&cols, "p2_avg")),
            loss: parse_f64(&col(&cols, "loss")),
            policy_loss: parse_f64(&col(&cols, "policy_loss")),
            value_loss: parse_f64(&col(&cols, "value_loss")),
            value_r2: parse_f64(&col(&cols, "value_r2")),
            aux_own_loss: parse_f64(&col(&cols, "aux_own_loss")),
            aux_fog_loss: parse_f64(&col(&cols, "aux_fog_loss")),
            aux_spt_loss: parse_f64(&col(&cols, "aux_spt_loss")),
            aux_tech_loss: parse_f64(&col(&cols, "aux_tech_loss")),
            avg_captures: parse_f64(&col(&cols, "avg_captures")),
            avg_cap_ruins: parse_f64(&col(&cols, "avg_cap_ruins")),
            avg_cap_villages: parse_f64(&col(&cols, "avg_cap_villages")),
            avg_cap_cities: parse_f64(&col(&cols, "avg_cap_cities")),
            avg_cap_capitals: parse_f64(&col(&cols, "avg_cap_capitals")),
            avg_harvests: parse_f64(&col(&cols, "avg_harvests")),
            avg_builds: parse_f64(&col(&cols, "avg_builds")),
            avg_research: parse_f64(&col(&cols, "avg_research")),
            avg_attacks: parse_f64(&col(&cols, "avg_attacks")),
            avg_revealed_tiles: parse_f64(&col(&cols, "avg_revealed_tiles")),
            avg_captured_tiles: parse_f64(&col(&cols, "avg_captured_tiles")),
            avg_spt_t0: parse_f64(&col(&cols, "avg_spt_t0")),
            avg_spt_t5: parse_f64(&col(&cols, "avg_spt_t5")),
            avg_spt_t10: parse_f64(&col(&cols, "avg_spt_t10")),
            avg_spt_t15: parse_f64(&col(&cols, "avg_spt_t15")),
            avg_spt_t20: parse_f64(&col(&cols, "avg_spt_t20")),
            avg_spt_t25: parse_f64(&col(&cols, "avg_spt_t25")),
            avg_spt_t30: parse_f64(&col(&cols, "avg_spt_t30")),
            villages_t2c_first: parse_f64(&col(&cols, "villages_t2c_first")),
            villages_first_rate: parse_f64(&col(&cols, "villages_first_rate")),
            villages_t2c_first_cond: parse_f64(&col(&cols, "villages_t2c_first_cond")),
            villages_t2c_p50: parse_f64(&col(&cols, "villages_t2c_p50")),
            villages_t2c_p80: parse_f64(&col(&cols, "villages_t2c_p80")),
            villages_t2c_all: parse_f64(&col(&cols, "villages_t2c_all")),
            ruins_t2c_p50: parse_f64(&col(&cols, "ruins_t2c_p50")),
            ruins_t2c_p80: parse_f64(&col(&cols, "ruins_t2c_p80")),
            ruins_t2c_all: parse_f64(&col(&cols, "ruins_t2c_all")),
            avg_moves: parse_f64(&col(&cols, "avg_moves")),
            match_type: col(&cols, "match_type"),
        });
    }
    rows
}

fn row_to_json(r: &MetricRow) -> Value {
    json!({
        "run_id": r.run_id,
        "iter_started_at": r.iter_started_at,
        "iteration": r.iteration,
        "games_file": r.games_file,
        "num_games": r.num_games,
        "tribes": r.tribes,
        "avg_score": r.avg_score,
        "max_score": r.max_score,
        "p1_avg": r.p1_avg,
        "p2_avg": r.p2_avg,
        "loss": r.loss,
        "policy_loss": r.policy_loss,
        "value_loss": r.value_loss,
        "value_r2": r.value_r2,
        "aux_own_loss": r.aux_own_loss,
        "aux_fog_loss": r.aux_fog_loss,
        "aux_spt_loss": r.aux_spt_loss,
        "aux_tech_loss": r.aux_tech_loss,
        "avg_captures": r.avg_captures,
        "avg_cap_ruins": r.avg_cap_ruins,
        "avg_cap_villages": r.avg_cap_villages,
        "avg_cap_cities": r.avg_cap_cities,
        "avg_cap_capitals": r.avg_cap_capitals,
        "avg_harvests": r.avg_harvests,
        "avg_builds": r.avg_builds,
        "avg_research": r.avg_research,
        "avg_attacks": r.avg_attacks,
        "avg_revealed_tiles": r.avg_revealed_tiles,
        "avg_captured_tiles": r.avg_captured_tiles,
        "avg_spt_t0": r.avg_spt_t0,
        "avg_spt_t5": r.avg_spt_t5,
        "avg_spt_t10": r.avg_spt_t10,
        "avg_spt_t15": r.avg_spt_t15,
        "avg_spt_t20": r.avg_spt_t20,
        "avg_spt_t25": r.avg_spt_t25,
        "avg_spt_t30": r.avg_spt_t30,
        "villages_t2c_first": r.villages_t2c_first,
        "villages_first_rate": r.villages_first_rate,
        "villages_t2c_first_cond": r.villages_t2c_first_cond,
        "villages_t2c_p50": r.villages_t2c_p50,
        "villages_t2c_p80": r.villages_t2c_p80,
        "villages_t2c_all": r.villages_t2c_all,
        "ruins_t2c_p50": r.ruins_t2c_p50,
        "ruins_t2c_p80": r.ruins_t2c_p80,
        "ruins_t2c_all": r.ruins_t2c_all,
        "avg_moves": r.avg_moves,
        "match_type": r.match_type,
    })
}

#[derive(Debug, Serialize)]
struct RunSummary {
    run_id: String,
    run_started_at: String,
    iter_count: usize,
    iter_min: i64,
    iter_max: i64,
    last_loss: f64,
    best_score: f64,
}

pub async fn api_runs() -> Json<Value> {
    let rows = read_csv_rows();
    let mut by_run: HashMap<String, Vec<&MetricRow>> = HashMap::new();
    for row in &rows {
        by_run.entry(row.run_id.clone()).or_default().push(row);
    }

    let mut summaries: Vec<RunSummary> = by_run
        .into_iter()
        .map(|(run_id, mut run_rows)| {
            run_rows.sort_by_key(|r| r.iteration);
            let first = run_rows.first().unwrap();
            let last = run_rows.last().unwrap();
            let best_score = run_rows
                .iter()
                .map(|r| r.max_score)
                .fold(0.0_f64, f64::max);
            RunSummary {
                run_id: run_id.clone(),
                run_started_at: first.iter_started_at.clone(),
                iter_count: run_rows.len(),
                iter_min: first.iteration,
                iter_max: last.iteration,
                last_loss: last.loss,
                best_score,
            }
        })
        .collect();

    summaries.sort_by(|a, b| b.run_id.cmp(&a.run_id));
    Json(json!(summaries))
}

#[derive(Debug, serde::Deserialize)]
pub struct RunFilter {
    pub run: Option<String>,
}

pub async fn api_training_metrics(Query(q): Query<RunFilter>) -> Json<Value> {
    let rows = read_csv_rows();
    let filtered: Vec<Value> = rows
        .iter()
        .filter(|r| q.run.as_ref().is_none_or(|id| &r.run_id == id))
        .map(row_to_json)
        .collect();
    Json(json!(filtered))
}

pub async fn api_elo_ladder() -> Json<Value> {
    let content = std::fs::read_to_string(LADDER_PATH).unwrap_or_default();
    let ladder: Value =
        serde_json::from_str(&content).unwrap_or_else(|_| json!({ "anchors": [], "readings": [] }));
    Json(ladder)
}

pub async fn api_moves_by_turn(Query(q): Query<RunFilter>) -> Json<Value> {
    let content = std::fs::read_to_string(MOVES_PATH).unwrap_or_else(|_| "{}".to_string());
    let all: Value = serde_json::from_str(&content).unwrap_or(json!({}));
    if let Some(run_id) = &q.run {
        if let Some(run_data) = all.get(run_id) {
            return Json(run_data.clone());
        }
        return Json(json!({}));
    }
    Json(all)
}

/// Per-role development-tempo curves (`tempo_by_turn.json` sidecar):
/// run_id -> iteration -> role -> turn -> {cities, spt, units, ...}.
pub async fn api_tempo_by_turn(Query(q): Query<RunFilter>) -> Json<Value> {
    let content = std::fs::read_to_string(TEMPO_PATH).unwrap_or_else(|_| "{}".to_string());
    let all: Value = serde_json::from_str(&content).unwrap_or(json!({}));
    if let Some(run_id) = &q.run {
        if let Some(run_data) = all.get(run_id) {
            return Json(run_data.clone());
        }
        return Json(json!({}));
    }
    Json(all)
}

#[derive(Debug, serde::Deserialize)]
pub struct ValueDistQuery {
    pub file: Option<String>,
    pub run: Option<String>,
    pub iteration: Option<i64>,
}

#[derive(Debug, Serialize)]
struct ValueDistribution {
    file: String,
    n: usize,
    stats: ValueStats,
    hist: HistData,
    abs_hist: HistData,
    buckets: BucketData,
    samples: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct ValueStats {
    mean: f64,
    std: f64,
    min: f64,
    max: f64,
    weak_pct: f64,
    moderate_pct: f64,
    strong_pct: f64,
    saturation_pct: f64,
    in_target_range_pct: f64,
}

#[derive(Debug, Serialize)]
struct HistData {
    bins: Vec<f64>,
    counts: Vec<usize>,
}

#[derive(Debug, Serialize)]
struct BucketData {
    weak: f64,
    moderate: f64,
    strong: f64,
    saturation: f64,
}

fn list_games_files() -> Vec<String> {
    let mut files = Vec::new();
    for dir in ["", "archive"] {
        let path = Path::new(dir);
        if !path.exists() {
            continue;
        }
        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with("games_") && name.ends_with(".safetensors") {
                    let rel = if dir.is_empty() {
                        name
                    } else {
                        format!("{dir}/{name}")
                    };
                    files.push(rel);
                }
            }
        }
    }
    files.sort_by(|a, b| b.cmp(a));
    files
}

fn resolve_games_path(file: &str) -> PathBuf {
    let p = PathBuf::from(file);
    if p.exists() {
        return p;
    }
    PathBuf::from("archive").join(file.trim_start_matches("archive/"))
}

fn load_values(path: &Path) -> anyhow::Result<Vec<f32>> {
    let data = std::fs::read(path)?;
    let st = SafeTensors::deserialize(&data)?;
    let view = st
        .tensor("values")
        .map_err(|e| anyhow::anyhow!("no 'values' tensor: {e}"))?;
    let bytes = view.data();
    let mut values = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(values)
}

fn compute_distribution(values: &[f32]) -> ValueDistribution {
    let n = values.len();
    let mean = if n > 0 {
        values.iter().map(|v| *v as f64).sum::<f64>() / n as f64
    } else {
        0.0
    };
    let variance = if n > 1 {
        values
            .iter()
            .map(|v| {
                let d = *v as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / n as f64
    } else {
        0.0
    };
    let std = variance.sqrt();
    let min = values.iter().copied().fold(f32::INFINITY, f32::min) as f64;
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;

    let mut weak = 0usize;
    let mut moderate = 0usize;
    let mut strong = 0usize;
    let mut saturation = 0usize;
    let mut in_target = 0usize;

    for v in values {
        let av = v.abs();
        if av < 0.1 {
            weak += 1;
        } else if av < 0.3 {
            moderate += 1;
        } else if av < 0.5 {
            strong += 1;
        } else {
            saturation += 1;
        }
        if av >= 0.1 && av <= 0.5 {
            in_target += 1;
        }
    }

    let pct = |c: usize| if n > 0 { 100.0 * c as f64 / n as f64 } else { 0.0 };

    const HIST_BINS: usize = 80;
    let mut hist_counts = vec![0usize; HIST_BINS];
    let mut abs_hist_counts = vec![0usize; HIST_BINS];
    for v in values {
        let idx = (((*v + 1.0) / 2.0).clamp(0.0, 0.9999) * HIST_BINS as f32) as usize;
        hist_counts[idx] += 1;
        let av = v.abs().clamp(0.0, 1.0);
        let aidx = ((av * 0.9999) * HIST_BINS as f32) as usize;
        abs_hist_counts[aidx] += 1;
    }

    let hist_bins: Vec<f64> = (0..HIST_BINS)
        .map(|i| -1.0 + (2.0 * (i as f64 + 0.5) / HIST_BINS as f64))
        .collect();
    let abs_bins: Vec<f64> = (0..HIST_BINS)
        .map(|i| (i as f64 + 0.5) / HIST_BINS as f64)
        .collect();

    let max_samples = 8000usize;
    let samples: Vec<f32> = if values.len() <= max_samples {
        values.to_vec()
    } else {
        let step = values.len() / max_samples;
        values.iter().step_by(step.max(1)).copied().collect()
    };

    ValueDistribution {
        file: String::new(),
        n,
        stats: ValueStats {
            mean,
            std,
            min,
            max,
            weak_pct: pct(weak),
            moderate_pct: pct(moderate),
            strong_pct: pct(strong),
            saturation_pct: pct(saturation),
            in_target_range_pct: pct(in_target),
        },
        hist: HistData {
            bins: hist_bins,
            counts: hist_counts,
        },
        abs_hist: HistData {
            bins: abs_bins,
            counts: abs_hist_counts,
        },
        buckets: BucketData {
            weak: pct(weak),
            moderate: pct(moderate),
            strong: pct(strong),
            saturation: pct(saturation),
        },
        samples,
    }
}

fn load_value_dist_cache(run_id: Option<&str>, iteration: Option<i64>) -> Option<Value> {
    let content = std::fs::read_to_string(VALUE_DIST_PATH).ok()?;
    let all: Value = serde_json::from_str(&content).ok()?;
    let run_id = run_id?;
    let iteration = iteration?;
    all.get(run_id)?.get(iteration.to_string()).cloned()
}

pub async fn api_value_distribution(
    Query(q): Query<ValueDistQuery>,
) -> Result<Json<Value>, ApiError> {
    if q.file.is_none() && q.run.is_none() {
        return Ok(Json(json!({ "files": list_games_files() })));
    }

    if let (Some(run_id), Some(iteration)) = (&q.run, q.iteration) {
        if let Some(cached) = load_value_dist_cache(Some(run_id), Some(iteration)) {
            return Ok(Json(cached));
        }
    }

    let file = q.file.as_deref().unwrap_or("");
    if file.is_empty() {
        return Err(ApiError(
            StatusCode::NOT_FOUND,
            "value distribution not found (no cache entry)".into(),
        ));
    }

    let path = resolve_games_path(file);
    if path.exists() {
        let values = load_values(&path).map_err(|e| {
            ApiError(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to read values: {e}"),
            )
        })?;
        let mut dist = compute_distribution(&values);
        dist.file = file.to_string();
        return Ok(Json(serde_json::to_value(dist).unwrap_or(json!({}))));
    }

    if let Some(cached) = load_value_dist_cache(q.run.as_deref(), q.iteration) {
        return Ok(Json(cached));
    }

    Err(ApiError(
        StatusCode::NOT_FOUND,
        format!("file not found: {file}"),
    ))
}

pub struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.0,
            Json(json!({ "error": self.1 })),
        )
            .into_response()
    }
}