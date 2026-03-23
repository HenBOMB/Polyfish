use crate::game::Game;
use crate::moves::{CaptureMove, RewardMove};
use crate::states::GameState;
use crate::types::CityRewardType;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ModReplay {
    pub game_state: GameState,
    pub turns: Vec<ReplayTurn>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ReplayTurn {
    pub turn: i32,
    pub players: Vec<ReplayPlayer>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ReplayPlayer {
    pub player_id: i32,
    pub commands: Vec<serde_json::Value>,
}

pub fn replay_game(game: &mut Game, mod_replay: &mut ModReplay) -> Result<(), String> {
    game.state = mod_replay.game_state.clone();

    if game.state.settings.current_player_turn_id == 0 {
        game.state.settings.current_player_turn_id = 1;
    }
    game.post_load();

    // The first turn is inevitable to avoid auto-playing while extracting in the polyfish-mod
    if !mod_replay.turns.is_empty()
        && !mod_replay.turns[0].players.is_empty()
        && mod_replay.turns[0].players[0].commands.len() >= 2
    {
        mod_replay.turns[0].players[0].commands.remove(0); // remove the -1 "start match command"
        mod_replay.turns[0].players[0].commands.remove(0); // remove the first forced auto played move
    }

    for turn_data in &mod_replay.turns {
        for player_data in &turn_data.players {
            // Reset units for this player to make moves legal
            if let Some(tribe) = game.state.tribes.get_mut(&player_data.player_id) {
                for unit in &mut tribe.units {
                    unit.moved = false;
                    unit.attacked = false;
                    unit.attacks_performed = 0;
                }
            }
            game.state.settings.current_player_turn_id = player_data.player_id;

            for cmd_json in &player_data.commands {
                // Skips startmatch / endmatch which are moveType: -1 or 11 (EndTurn is 10, Resign is 11)
                let move_type_opt = cmd_json.get("moveType").and_then(|v| v.as_i64());
                if let Some(move_type) = move_type_opt {
                    if move_type == -1 || move_type == 11 {
                        continue;
                    }
                }

                let mut cmd_stripped = cmd_json
                    .as_object()
                    .ok_or("Command is not an object")?
                    .clone();
                cmd_stripped.remove("_reward");
                cmd_stripped.remove("_revealedTiles");
                let cmd_json_stripped_val = serde_json::Value::Object(cmd_stripped);

                let legal_moves = game.legal_moves();
                let mut found = false;
                for m in &legal_moves {
                    let serialized = m.serialize();

                    if serialized == cmd_json_stripped_val {
                        let move_type = cmd_json.get("moveType").and_then(|v| v.as_i64()).unwrap();

                        if move_type == 9 {
                            let reward_type: CityRewardType =
                                serde_json::from_value(cmd_json.get("type").unwrap().clone())
                                    .map_err(|e| format!("Failed to parse reward type: {}", e))?;
                            let mut m_with_hints = RewardMove::new(
                                cmd_json.get("target").unwrap().as_i64().unwrap() as i32,
                                reward_type,
                            );
                            if let Some(tiles) = cmd_json.get("_revealedTiles") {
                                m_with_hints.revealed_tiles =
                                    Some(serde_json::from_value(tiles.clone()).unwrap());
                            }
                            game.play_move(&m_with_hints);
                        } else if move_type == 8 {
                            let mut m_with_hints = CaptureMove::new(
                                cmd_json.get("src").unwrap().as_i64().unwrap() as i32,
                            );
                            if let Some(r) = cmd_json.get("_reward") {
                                m_with_hints.reward =
                                    Some(serde_json::from_value(r.clone()).unwrap());
                            }
                            if let Some(tiles) = cmd_json.get("_revealedTiles") {
                                m_with_hints.revealed_tiles =
                                    Some(serde_json::from_value(tiles.clone()).unwrap());
                            }
                            game.play_move(&m_with_hints);
                        } else {
                            game.play_move(m.as_ref());
                        }

                        game.state._messages.clear();
                        found = true;
                        break;
                    }
                }

                if !found {
                    return Err(format!(
                        "FAILED TO FIND MOVE IN LEGAL MOVES! Turn {}, Player {}, Command: {}",
                        turn_data.turn,
                        player_data.player_id,
                        serde_json::to_string(&cmd_json).unwrap()
                    ));
                }
            }
        }
    }

    Ok(())
}
