use polyfish::game::Game;
use polyfish::moves::Move;
use polyfish::states::GameState;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct ModReplay {
    pub game_state: GameState,
    pub turns: Vec<ReplayTurn>,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct ReplayTurn {
    pub turn: i32,
    pub players: Vec<ReplayPlayer>,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct ReplayPlayer {
    pub player_id: i32,
    pub commands: Vec<serde_json::Value>,
}

#[tokio::test]
async fn test_mod_replay_ingestion() {
    let json_str = std::fs::read_to_string("replays/mod_replay_1774077520.json").unwrap();
    let mut mod_replay: ModReplay = serde_json::from_str(&json_str).unwrap();

    let mut game = Game::new();
    game.state = mod_replay.game_state;
    {
        let t147 = game.state.tiles.get(&147).unwrap();
        println!("    [DEBUG INIT] Tile 147 Owner: {}", t147.owner);
    }
    if game.state.settings.current_player_turn_id == 0 {
        game.state.settings.current_player_turn_id = 1;
    }
    game.post_load();

    // let mut total_commands = 0;
    let mut success_commands = 0;

    // The first turn is inevitable to avoid auto-playing while extracting in the polyfish-mod
    // So its already baked in to the state, removing it
    mod_replay.turns[0].players[0].commands.remove(0); // remove the -1 "start match command"
    mod_replay.turns[0].players[0].commands.remove(0); // remove the first forced auto played move

    game.state.settings._verbose = true;

    for turn_data in mod_replay.turns {
        println!("--- TURN {} ---", turn_data.turn);
        for player_data in turn_data.players {
            println!("  Player {}", player_data.player_id);
            // Reset units for this player to make moves legal
            if let Some(tribe) = game.state.tribes.get_mut(&player_data.player_id) {
                for unit in &mut tribe.units {
                    unit.moved = false;
                    unit.attacked = false;
                    unit.attacks_performed = 0;
                }
            }
            game.state.settings.current_player_turn_id = player_data.player_id;

            for cmd_json in player_data.commands {
                if let Some(tribe) = game.state.tribes.get(&player_data.player_id) {
                    println!(
                        "[DEBUG LEGAL] Player {} has {} stars",
                        player_data.player_id, tribe.stars
                    );
                }
                // Skips startmatch / endmatch which are moveType: -1
                if let Some(move_type) = cmd_json.get("moveType").and_then(|v| v.as_i64()) {
                    if move_type == -1 {
                        continue;
                    }
                }

                // total_commands += 1;
                let legal_moves = game.legal_moves();
                let mut found = false;
                for m in &legal_moves {
                    let mut serialized = m.serialize();

                    // Standard equality check
                    if serialized == cmd_json {
                        println!("    Executing: {:?}", m.describe(&game.state));
                        
                        // Debug Recover
                        let mut pre_health = 0;
                        if let Some(move_type) = cmd_json.get("moveType").and_then(|v| v.as_i64()) {
                            if move_type == 3 { // Ability
                                let src = cmd_json.get("src").and_then(|v| v.as_i64()).unwrap() as i32;
                                if let Some(u) = polyfish::functions::get_unit_at(&game.state, src) {
                                    pre_health = u.health;
                                    let owner = game.state.tiles.get(&src).map(|t| t.owner).unwrap_or(0);
                                    println!("    [DEBUG ABILITY] Unit at {} has health {}/{} (Tile Owner: {})", src, u.health, polyfish::functions::get_unit_max_health(u), owner);
                                }
                            }
                        }

                        game.play_move(m.as_ref());

                        if let Some(move_type) = cmd_json.get("moveType").and_then(|v| v.as_i64()) {
                            if move_type == 3 {
                                let src = cmd_json.get("src").and_then(|v| v.as_i64()).unwrap() as i32;
                                if let Some(u) = polyfish::functions::get_unit_at(&game.state, src) {
                                    println!("    [DEBUG ABILITY] Unit at {} now has health {} (Healed {})", src, u.health, u.health - pre_health);
                                }
                            }
                        }
                        for msg in &game.state._messages {
                            println!("    {}", msg);
                        }
                        game.state._messages.clear();
                        found = true;
                        success_commands += 1;
                        break;
                    }

                    // Special case for Capture with _reward hint
                    if let (Some(8), Some(8)) = (
                        cmd_json.get("moveType").and_then(|v| v.as_i64()),
                        serialized.get("moveType").and_then(|v| v.as_i64()),
                    ) {
                        if cmd_json.get("src") == serialized.get("src")
                            && cmd_json.get("_reward").is_some()
                        {
                            // It's a capture! Inject the reward hint
                            let reward_val = cmd_json.get("_reward").unwrap();
                            let reward: polyfish::types::RuinsRewardType =
                                serde_json::from_value(reward_val.clone()).unwrap();

                            let m_with_reward = polyfish::moves::CaptureMove::with_reward(
                                cmd_json.get("src").unwrap().as_i64().unwrap() as i32,
                                reward,
                            );

                            println!(
                                "    Executing: {:?} (with reward hint: {:?})",
                                m_with_reward.describe(&game.state),
                                reward
                            );
                            game.play_move(&m_with_reward);
                            for msg in &game.state._messages {
                                println!("    {}", msg);
                            }
                            game.state._messages.clear();
                            found = true;
                            success_commands += 1;
                            break;
                        }
                    }
                }

                if !found {
                    let pov_id = player_data.player_id;
                    let tribe = game.state.tribes.get(&pov_id).unwrap();
                    println!("    FAILED TO FIND MOVE IN LEGAL MOVES!");
                    println!("    Command: {}", serde_json::to_string(&cmd_json).unwrap());
                    println!(
                        "    Tribe {} Stars: {}, Cities: {}",
                        pov_id,
                        tribe.stars,
                        tribe.cities.len()
                    );
                    println!("    Legal Moves (first 50):");
                    for (i, m) in legal_moves.iter().take(50).enumerate() {
                        println!(
                            "      {}: {:?} -> {}",
                            i,
                            m.describe(&game.state),
                            serde_json::to_string(&m.serialize()).unwrap()
                        );
                    }

                    if let Some(tribe) = game.state.tribes.get(&player_data.player_id) {
                        let src_idx = cmd_json.get("src").and_then(|v| v.as_i64()).map(|v| v as i32);
                        if let Some(s_idx) = src_idx {
                            if let Some(u) = tribe.units.iter().find(|u| u.coords.idx == s_idx) {
                                println!("    Unit at SRC ({}): Owner={}, Type={:?}, Moved={}, Attacked={}, Health={}/{}", s_idx, u.owner, u.unit_type, u.moved, u.attacked, u.health, polyfish::functions::get_unit_max_health(u));
                            } else {
                                println!("    NO UNIT at SRC ({})!", s_idx);
                            }
                        }
                    }

                    if let Some(target) = cmd_json.get("target").and_then(|v| v.as_i64()) {
                        let target_idx = target as i32;
                        if let Some(unit) =
                            polyfish::functions::get_unit_at(&game.state, target_idx)
                        {
                            println!(
                                "    Unit already at TARGET ({}): Owner={}, Type={:?}",
                                target_idx, unit.owner, unit.unit_type
                            );
                        }
                    }

                    panic!("Replay parsing failed.");
                }
            }
        }
    }

    println!("Successfully replayed {} commands!", success_commands);
}
