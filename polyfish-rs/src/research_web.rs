use std::sync::Arc;

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    routing::{get, post},
};
use polyfish::game::Game;
use polyfish::replay::{Replay, ReplayPlayback};
use polyfish::research::{self, MatchConfig, MatchResult, Opponent};
use polyfish::states::GameState;
use polyfish::types::{MoveType, StructureType, TribeType};
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{AppState, ResetRequest};

type WebError = (StatusCode, String);
type WebResult = Result<Json<Value>, WebError>;

pub(super) fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/research/current", get(current))
        .route("/research/move", post(play))
        .route("/research/bot", post(bot))
        .route("/research/save", get(save))
        .route("/research/load", post(load))
        .route("/research/replay", post(open_replay))
        .route("/research/replay/seek", post(seek_replay))
}

fn bad_request(message: impl ToString) -> WebError {
    (StatusCode::BAD_REQUEST, message.to_string())
}

fn config(game: &Game) -> Result<MatchConfig, WebError> {
    game.state
        .settings
        .research
        .filter(|config| config.preset_version == research::PRESET_VERSION)
        .ok_or_else(|| bad_request("Start a research match first"))
}

pub(super) fn reset(state: &AppState, req: &ResetRequest) -> WebResult {
    let size = research::board_size(req.map_size.unwrap_or(121))
        .ok_or_else(|| bad_request("Choose a map with 121, 196, or 256 tiles"))?;
    let opponent =
        req.opponent_preset
            .unwrap_or(if size.get_size() == 11 && state.network.is_some() {
                Opponent::Learned
            } else {
                Opponent::Heuristic
            });
    if opponent == Opponent::Learned && (size.get_size() != 11 || state.network.is_none()) {
        return Err(bad_request(
            "The learned opponent needs an 11×11 map and a compatible checkpoint",
        ));
    }
    let seed = req.seed.unwrap_or_else(rand::random);
    let mut tribe_rng = StdRng::seed_from_u64(seed as u64);
    let starters = [
        TribeType::XinXi,
        TribeType::Imperius,
        TribeType::Bardur,
        TribeType::Oumaji,
    ];
    let mut tribe = |value: Option<&str>| match value {
        Some(name) if name.eq_ignore_ascii_case("random") => {
            Ok(starters[tribe_rng.random_range(0..starters.len())])
        }
        Some(name) => research::research_tribe(name)
            .ok_or_else(|| bad_request(format!("Unsupported tribe: {name}"))),
        None => Ok(TribeType::Imperius),
    };
    let tribes = [tribe(req.tribe1.as_deref())?, tribe(req.tribe2.as_deref())?];
    if req.max_turns.is_some_and(|turns| turns < 1) {
        return Err(bad_request("The turn limit must be positive"));
    }
    let mut game = research::new_match(
        size,
        tribes,
        seed,
        MatchConfig::new(opponent, 2),
    );
    if let Some(max_turns) = req.max_turns {
        game.state.settings.max_turns = max_turns;
    }
    let mut live = state.game.lock().unwrap();
    *state.replay.lock().unwrap() = None;
    *live = game;
    Ok(Json(snapshot(&live, state.network.is_some(), false)))
}

fn board_view(game: &Game, replaying: bool) -> (Game, bool) {
    let cfg = game.state.settings.research;
    let active = game.current_player_id();
    let bot_turn = !replaying
        && research::result(game) == MatchResult::InProgress
        && cfg
            .is_some_and(|config| config.opponent != Opponent::Human && config.bot_seat == active);
    let pov = cfg
        .filter(|config| config.opponent != Opponent::Human)
        .map_or(active, |config| if config.bot_seat == 1 { 2 } else { 1 });
    let mut observed = research::observation(game, pov);
    observed.state.settings.current_player_turn_id = pov;
    (observed, bot_turn)
}

pub(super) fn graphical_snapshot(game: &Game, has_network: bool) -> Option<Value> {
    let config = game.state.settings.research?;
    let (observed, bot_turn) = board_view(game, false);
    let state = &observed.state;
    let result = research::result(game);
    let mut tiles: Vec<_> = state.tiles.values().collect();
    tiles.sort_by_key(|tile| tile.coords.idx);
    let legal_moves: Vec<_> = if !bot_turn && result == MatchResult::InProgress {
        research::legal_actions(game, &observed)
            .iter()
            .map(|action| action.serialize())
            .collect()
    } else {
        Vec::new()
    };
    Some(json!({
        "state": {
            "settings": state.settings,
            "tiles": tiles,
            "structures": state.structures,
            "resources": state.resources,
            "tribes": crate::tribes_json_with_max_health(state),
        },
        "legalMoves": legal_moves,
        "researchMatch": {
            "opponent": config.opponent,
            "seed": game.state.initial_seed.to_string(),
            "revision": game.state._history.len(),
            "result": result,
            "learnedAvailable": has_network,
            "botTurn": bot_turn,
        },
    }))
}

fn snapshot(game: &Game, has_network: bool, replaying: bool) -> Value {
    let cfg = game.state.settings.research;
    let active = game.current_player_id();
    let result = research::result(game);
    let (observed, bot_turn) = board_view(game, replaying);
    let state = &observed.state;
    let pov = observed.current_player_id();
    let tiles: Vec<_> = state.tiles.values().map(|tile| {
        if !tile.explorers.contains(&pov) && state.settings._fow {
            return json!({"index": tile.coords.idx, "visible": false});
        }
        let index = tile.coords.idx;
        let unit = polyfish::functions::get_unit_at(state, index);
        let city = polyfish::functions::get_city_at(state, index);
        json!({
            "index": index,
            "visible": true,
            "terrain": format!("{:?}", tile.terrain_type),
            "owner": tile.owner,
            "capital": tile.capital_of > 0,
            "road": tile.has_road,
            "city": city.map(|city| json!({"level": city.level, "name": city.name, "population": city.population})),
            "unit": unit.map(|unit| json!({"name": format!("{:?}", unit.unit_type), "owner": unit.owner, "health": unit.health})),
            "structure": state.structures.get(&index).and_then(|s| s.as_ref()).map(|s| format!("{:?}", s.structure_type)),
            "resource": state.resources.get(&index).and_then(|r| r.as_ref()).filter(|resource| {
                polyfish::functions::is_resource_visible_to_tribe(state, resource.resource_type, pov, Some(index))
            }).map(|resource| format!("{:?}", resource.resource_type)),
        })
    }).collect();
    let actions: Vec<_> = if !bot_turn && !replaying && result == MatchResult::InProgress {
        research::legal_actions(game, &observed)
            .into_iter()
            .map(|action| {
                json!({
                    "key": action.serialize(),
                    "label": action.describe(state),
                    "kind": format!("{:?}", action.move_type()),
                    "source": action.source_idx().ok(),
                    "target": action.target_idx().ok(),
                })
            })
            .collect()
    } else {
        Vec::new()
    };
    json!({
        "match": cfg,
        "size": game.map_size(),
        "seed": game.state.initial_seed.to_string(),
        "turn": game.turn(),
        "active": active,
        "pov": pov,
        "stars": state.tribes.get(&pov).map(|tribe| tribe.stars),
        "production": state.tribes.get(&pov).map(|tribe| polyfish::functions::get_total_production(state, &tribe.cities)),
        "revision": game.state._history.len(),
        "result": result,
        "learnedAvailable": has_network,
        "botTurn": bot_turn,
        "actions": actions,
        "tiles": tiles,
    })
}

async fn current(State(app): State<Arc<AppState>>) -> Json<Value> {
    let game = app.game.lock().unwrap();
    if game.state.settings.research.is_none() {
        return Json(json!({"match": null, "learnedAvailable": app.network.is_some()}));
    }
    Json(snapshot(&game, app.network.is_some(), false))
}

#[derive(Deserialize)]
struct ActionRequest {
    revision: usize,
    action: Value,
}

async fn play(State(app): State<Arc<AppState>>, Json(req): Json<ActionRequest>) -> WebResult {
    let mut game = app.game.lock().unwrap();
    let cfg = config(&game)?;
    if research::result(&game) != MatchResult::InProgress
        || (cfg.opponent != Opponent::Human && cfg.bot_seat == game.current_player_id())
        || req.revision != game.state._history.len()
    {
        return Err((
            StatusCode::CONFLICT,
            "The match advanced. Refresh the board.".into(),
        ));
    }
    let view = research::observation(&game, game.current_player_id());
    let action = research::legal_actions(&game, &view)
        .into_iter()
        .find(|candidate| candidate.serialize() == req.action)
        .ok_or_else(|| bad_request("That action is no longer available"))?;
    let excavating = action.move_type() == MoveType::Capture
        && action.source_idx().ok().is_some_and(|idx| {
            polyfish::functions::get_structure_type_at(&game.state, idx as i32)
                == Some(StructureType::Ruin)
        });
    game.state._messages.clear();
    let verbose = std::mem::replace(&mut game.state.settings._verbose, excavating);
    let applied = game.play_move(action.as_ref());
    game.state.settings._verbose = verbose;
    let messages = std::mem::take(&mut game.state._messages);
    let _undo = applied.ok_or_else(|| bad_request("Could not apply the action"))?;
    let mut response = snapshot(&game, app.network.is_some(), false);
    if !messages.is_empty() {
        response["messages"] = json!(messages);
    }
    Ok(Json(response))
}

async fn bot(State(app): State<Arc<AppState>>) -> WebResult {
    tokio::task::spawn_blocking(move || {
        let mut game = app.game.lock().unwrap();
        let cfg = config(&game)?;
        if research::result(&game) == MatchResult::InProgress
            && cfg.opponent != Opponent::Human
            && cfg.bot_seat == game.current_player_id()
        {
            let action = if let Some(preset) = cfg.opponent.preset() {
                research::choose_move(&game, preset)
            } else if cfg.opponent == Opponent::Learned {
                use polyfish::ai::{
                    brain::{Brain, SearchBackend},
                    eval_server::{Evaluator, InlineEvalHandle},
                };
                let net = app
                    .network
                    .as_ref()
                    .filter(|_| game.map_size() == 11)
                    .ok_or_else(|| {
                        bad_request("The learned opponent is unavailable for this match")
                    })?;
                let evaluator = Evaluator::Inline(InlineEvalHandle::new(net.clone()));
                let mut agent =
                    Brain::with_backend(&evaluator, 128, SearchBackend::Gumbel { k: 16 });
                agent.think_with_stats(&mut game).0
            } else {
                research::choose_move(&game, research::OpponentPreset::Normal)
            };
            let action =
                action.ok_or_else(|| bad_request("The bot could not find a legal action"))?;
            let _undo = game
                .play_move(action.as_ref())
                .ok_or_else(|| bad_request("The bot action could not be applied"))?;
        }
        Ok(Json(snapshot(&game, app.network.is_some(), false)))
    })
    .await
    .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))?
}

async fn save(State(app): State<Arc<AppState>>) -> WebResult {
    let game = app.game.lock().unwrap();
    config(&game)?;
    Ok(Json(
        serde_json::to_value(&game.state).map_err(bad_request)?,
    ))
}

async fn load(State(app): State<Arc<AppState>>, Json(state): Json<GameState>) -> WebResult {
    let mut game = Game::new();
    game.state = state;
    let cfg = config(&game)?;
    if !matches!(game.map_size(), 11 | 14 | 16)
        || game.state.tiles.len() != (game.map_size() * game.map_size()) as usize
        || game.state.tribes.len() != 2
        || (cfg.opponent == Opponent::Learned && (game.map_size() != 11 || app.network.is_none()))
    {
        return Err(bad_request("This save is incompatible with the simulator"));
    }
    game.post_load();
    let mut live = app.game.lock().unwrap();
    *live = game;
    *app.replay.lock().unwrap() = None;
    Ok(Json(snapshot(&live, app.network.is_some(), false)))
}

fn playback_snapshot(replay: &ReplayPlayback, learned: bool) -> Value {
    let mut view = snapshot(replay.game(), learned, true);
    view["replay"] = json!({"position": replay.cursor(), "total": replay.total_commands()});
    view
}

async fn open_replay(State(app): State<Arc<AppState>>, Json(replay): Json<Replay>) -> WebResult {
    let playback = ReplayPlayback::new(replay).map_err(bad_request)?;
    let view = playback_snapshot(&playback, app.network.is_some());
    *app.replay.lock().unwrap() = Some(playback);
    Ok(Json(view))
}

#[derive(Deserialize)]
struct ReplaySeek {
    position: usize,
}

async fn seek_replay(State(app): State<Arc<AppState>>, Json(req): Json<ReplaySeek>) -> WebResult {
    let mut replay = app.replay.lock().unwrap();
    let playback = replay
        .as_mut()
        .ok_or_else(|| bad_request("Open a replay first"))?;
    playback.seek(req.position).map_err(bad_request)?;
    Ok(Json(playback_snapshot(playback, app.network.is_some())))
}
