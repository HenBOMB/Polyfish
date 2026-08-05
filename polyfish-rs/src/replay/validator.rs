use crate::ai::features;
use std::collections::HashSet;
use std::path::Path;

use super::{REPLAY_SCHEMA_VERSION, Replay, ReplayError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrainingEligibility {
    pub map_width: usize,
    pub map_height: usize,
}

pub fn validate_replay(replay: &Replay, file: Option<&Path>) -> Result<(), ReplayError> {
    let at = |message: String| {
        ReplayError::validation_at(
            file.map(|p| p.display().to_string()).unwrap_or_default(),
            message,
        )
    };

    if replay.schema_version != REPLAY_SCHEMA_VERSION {
        return Err(at(format!(
            "unsupported schemaVersion {}; supported version is {}",
            replay.schema_version, REPLAY_SCHEMA_VERSION
        )));
    }
    let width = replay.metadata.map_width;
    let height = replay.metadata.map_height;
    if width == 0 || height == 0 {
        return Err(at(format!(
            "map dimensions must be positive, got {width}x{height}"
        )));
    }
    if width != height {
        return Err(at(format!(
            "Polyfish currently requires square engine maps, got {width}x{height}"
        )));
    }
    let state = &replay.initial_state;
    if state.settings.size <= 0 || state.settings.size as usize != width {
        return Err(at(format!(
            "metadata map is {width}x{height}, but initialState.settings.size is {}",
            state.settings.size
        )));
    }
    let expected = width
        .checked_mul(height)
        .ok_or_else(|| at("map area overflow".into()))?;
    if state.tiles.len() != expected {
        return Err(at(format!(
            "initial state has {} tiles; {width}x{height} requires {expected}",
            state.tiles.len()
        )));
    }
    if state.settings.tile_count != expected as i32 {
        return Err(at(format!(
            "initialState.settings.tileCount is {}; expected {expected}",
            state.settings.tile_count
        )));
    }
    for idx in 0..expected as i32 {
        let Some(tile) = state.tiles.get(&idx) else {
            return Err(at(format!("initial state is missing tile index {idx}")));
        };
        if tile.coords.x < 0
            || tile.coords.y < 0
            || tile.coords.x as usize >= width
            || tile.coords.y as usize >= height
        {
            return Err(at(format!(
                "tile {idx} coordinates ({},{}) are outside {width}x{height}",
                tile.coords.x, tile.coords.y
            )));
        }
        let coordinate_index = tile.coords.y as usize * width + tile.coords.x as usize;
        if tile.coords.idx != idx || coordinate_index != idx as usize {
            return Err(at(format!(
                "tile map key {idx} disagrees with coords ({},{}, idx {}), which resolve to {coordinate_index}",
                tile.coords.x, tile.coords.y, tile.coords.idx
            )));
        }
        for (field, player_id) in [
            ("owner", tile.owner),
            ("capitalOf", tile.capital_of),
            ("unitOwner", tile._unit_owner_id.unwrap_or(0)),
        ] {
            if player_id != 0 && !state.tribes.contains_key(&player_id) {
                return Err(at(format!(
                    "tile {idx} {field} references unknown player {player_id}"
                )));
            }
        }
    }
    for &idx in state.structures.keys().chain(state.resources.keys()) {
        if idx < 0 || idx as usize >= expected {
            return Err(at(format!(
                "state side-table index {idx} is outside 0..{expected}"
            )));
        }
    }

    if state.tribes.is_empty() {
        return Err(at("initial state must contain at least one player".into()));
    }
    for (&id, tribe) in &state.tribes {
        if id <= 0 || tribe.id != id {
            return Err(at(format!(
                "tribe map key {id} does not match positive tribe.id {}",
                tribe.id
            )));
        }
        for unit in &tribe.units {
            if unit.owner != id {
                return Err(at(format!(
                    "player {id} contains a unit owned by {} at tile {}",
                    unit.owner, unit.coords.idx
                )));
            }
            if unit.coords.idx < 0 || unit.coords.idx as usize >= expected {
                return Err(at(format!(
                    "player {id} unit tile {} is outside 0..{expected}",
                    unit.coords.idx
                )));
            }
        }
        for city in &tribe.cities {
            if city.owner != id || city.idx < 0 || city.idx as usize >= expected {
                return Err(at(format!(
                    "player {id} has invalid city owner/tile ({}, {})",
                    city.owner, city.idx
                )));
            }
        }
    }
    if !state
        .tribes
        .contains_key(&state.settings.current_player_turn_id)
    {
        return Err(at(format!(
            "initial current player {} is not present in initial state",
            state.settings.current_player_turn_id
        )));
    }
    if replay.metadata.max_turns != state.settings.max_turns
        || replay.metadata.game_mode != state.settings.mode
    {
        return Err(at(format!(
            "metadata settings do not match initial state: maxTurns {}/{} mode {:?}/{:?}",
            replay.metadata.max_turns,
            state.settings.max_turns,
            replay.metadata.game_mode,
            state.settings.mode
        )));
    }

    let mut metadata_ids = HashSet::new();
    for player in &replay.metadata.players {
        if !metadata_ids.insert(player.player_id) {
            return Err(at(format!(
                "duplicate metadata player id {}",
                player.player_id
            )));
        }
        let Some(tribe) = state.tribes.get(&player.player_id) else {
            return Err(at(format!(
                "metadata player {} is absent from initial state",
                player.player_id
            )));
        };
        if tribe.tribe_type != player.tribe {
            return Err(at(format!(
                "metadata player {} tribe {:?} does not match initial state {:?}",
                player.player_id, player.tribe, tribe.tribe_type
            )));
        }
    }
    let state_ids: HashSet<_> = state.tribes.keys().copied().collect();
    if metadata_ids != state_ids {
        return Err(at(format!(
            "metadata player ids {metadata_ids:?} do not exactly match initial-state ids {state_ids:?}"
        )));
    }

    let mut prior_turn = i32::MIN;
    for (turn_index, turn) in replay.turns.iter().enumerate() {
        if turn.commands.is_empty() {
            return Err(at(format!("turn segment {turn_index} has no commands")));
        }
        if !state.tribes.contains_key(&turn.player_id) {
            return Err(at(format!(
                "turn segment {turn_index} declares unknown player {}",
                turn.player_id
            )));
        }
        if turn.turn_number < prior_turn {
            return Err(at(format!(
                "turn segment {turn_index} goes backwards from {prior_turn} to {}",
                turn.turn_number
            )));
        }
        prior_turn = turn.turn_number;
        for (command_index, command) in turn.commands.iter().enumerate() {
            for idx in command.tile_indices() {
                if idx < 0 || idx as usize >= expected {
                    return Err(at(format!(
                        "turn {}, player {}, command {} ({command:?}) references tile {idx} outside 0..{expected}",
                        turn.turn_number, turn.player_id, command_index
                    )));
                }
            }
        }
    }

    if let Some(result) = &replay.result {
        if result.draw && result.winner_player_id.is_some() {
            return Err(at(
                "result cannot declare both draw=true and a winner".into()
            ));
        }
        if let Some(winner) = result.winner_player_id
            && !state.tribes.contains_key(&winner)
        {
            return Err(at(format!("result winner {winner} is not a replay player")));
        }
        for id in result.scores.keys() {
            if !state.tribes.contains_key(id) {
                return Err(at(format!("result contains score for unknown player {id}")));
            }
        }
        if !result.scores.is_empty() {
            let score_ids: HashSet<_> = result.scores.keys().copied().collect();
            if score_ids != state_ids {
                return Err(at(format!(
                    "result score player ids {score_ids:?} do not exactly match replay players {state_ids:?}"
                )));
            }
        }
    }
    Ok(())
}

pub fn validate_training_eligibility(replay: &Replay) -> Result<TrainingEligibility, ReplayError> {
    let width = replay.metadata.map_width;
    let height = replay.metadata.map_height;
    if width != features::MAP_SIZE || height != features::MAP_SIZE {
        return Err(ReplayError::TrainingIneligible {
            message: format!(
                "Replay map is {width}x{height}, but the current training feature encoder supports only {}x{}. The replay can be viewed but cannot be exported for training.",
                features::MAP_SIZE,
                features::MAP_SIZE
            ),
        });
    }
    Ok(TrainingEligibility {
        map_width: width,
        map_height: height,
    })
}
