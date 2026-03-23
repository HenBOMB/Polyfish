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
    let replay_base = "replays/kimeusian-monsoon_1774234919";
    let replay_path = format!("{}.json", replay_base);
    let fixed_path = format!("{}_fixed.json", replay_base);

    // check if fixed exists
    let actual_path = if std::path::Path::new(&fixed_path).exists() {
        fixed_path
    } else {
        replay_path
    };

    let json_str = std::fs::read_to_string(&actual_path).unwrap();
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
    game.state.settings._are_you_sure = true;

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
                let move_type_opt = cmd_json.get("moveType").and_then(|v| v.as_i64());
                if let Some(move_type) = move_type_opt {
                    if move_type == -1 || move_type == 11 {
                        continue;
                    }
                }

                let move_type: polyfish::MoveType =
                    polyfish::MoveType::from(cmd_json["moveType"].as_i64().unwrap_or(0) as i32);

                let mut cmd_stripped = cmd_json.as_object().unwrap().clone();
                cmd_stripped.remove("_reward");
                cmd_stripped.remove("_revealedTiles");
                let cmd_json_stripped_val = serde_json::Value::Object(cmd_stripped);

                // --- FIX FOR SANCTUARY/ANIMALS NON-DETERMINISM ---
                // If it's Enchant Animal (moveType 3, type 23) and animal is missing, force spawn it
                if move_type == polyfish::MoveType::Ability
                    && cmd_json.get("type").and_then(|v| v.as_i64()) == Some(23)
                {
                    let target_idx =
                        cmd_json.get("target").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
                    let has_animal = game
                        .state
                        .resources
                        .get(&target_idx)
                        .and_then(|r| r.as_ref())
                        .map(|r| r.resource_type == polyfish::types::ResourceType::Game)
                        .unwrap_or(false);

                    if !has_animal {
                        println!(
                            "    [FIX] Force-spawning animal at {} for EnchantAnimal",
                            target_idx
                        );
                        game.state.resources.insert(
                            target_idx,
                            Some(polyfish::states::ResourceState {
                                resource_type: polyfish::types::ResourceType::Game,
                            }),
                        );
                    }
                }

                let legal_moves = game.legal_moves();
                let mut found = false;
                for m in &legal_moves {
                    let serialized = m.serialize();

                    if serialized == cmd_json_stripped_val {
                        let move_type = cmd_json.get("moveType").and_then(|v| v.as_i64()).unwrap();

                        // Match logic for Reward moves (moveType 9)
                        if move_type == 9 {
                            let reward_type: polyfish::types::CityRewardType =
                                serde_json::from_value(cmd_json.get("type").unwrap().clone())
                                    .unwrap();
                            let mut m_with_hints = polyfish::moves::RewardMove::new(
                                cmd_json.get("target").unwrap().as_i64().unwrap() as i32,
                                reward_type,
                            );
                            if let Some(tiles) = cmd_json.get("_revealedTiles") {
                                m_with_hints.revealed_tiles =
                                    Some(serde_json::from_value(tiles.clone()).unwrap());
                            }

                            println!("    Executing: {:?}", m_with_hints.describe(&game.state));
                            game.play_move(&m_with_hints);
                        }
                        // Match logic for Capture moves (moveType 8)
                        else if move_type == 8 {
                            let mut m_with_hints = polyfish::moves::CaptureMove::new(
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

                            println!("    Executing: {:?}", m_with_hints.describe(&game.state));
                            game.play_move(&m_with_hints);
                        } else {
                            println!("    Executing: {:?}", m.describe(&game.state));

                            // // Debug Recover
                            // let mut pre_health = 0;
                            // if move_type == 3 {
                            //     // Ability
                            //     let src = cmd_json
                            //         .get("src")
                            //         .or(cmd_json.get("target"))
                            //         .and_then(|v| v.as_i64())
                            //         .unwrap() as i32;
                            //     if let Some(u) = polyfish::functions::get_unit_at(&game.state, src)
                            //     {
                            //         pre_health = u.health;
                            //         let owner =
                            //             game.state.tiles.get(&src).map(|t| t.owner).unwrap_or(0);
                            //         println!(
                            //             "    [DEBUG ABILITY] Unit at {} has health {}/{} (Tile Owner: {})",
                            //             src,
                            //             u.health,
                            //             polyfish::functions::get_unit_max_health(u),
                            //             owner
                            //         );
                            //     }
                            // }

                            game.play_move(m.as_ref());

                            // if move_type == 3 {
                            //     let src = cmd_json
                            //         .get("src")
                            //         .or(cmd_json.get("target"))
                            //         .and_then(|v| v.as_i64())
                            //         .unwrap() as i32;
                            //     if let Some(u) = polyfish::functions::get_unit_at(&game.state, src)
                            //     {
                            //         println!(
                            //             "    [DEBUG ABILITY] Unit at {} now has health {} (Healed {})",
                            //             src,
                            //             u.health,
                            //             u.health - pre_health
                            //         );
                            //     }
                            // }
                        }

                        for msg in &game.state._messages {
                            println!("    {}", msg);
                        }
                        game.state._messages.clear();
                        found = true;
                        success_commands += 1;
                        break;
                    }
                }

                if !found {
                    let pov_id = player_data.player_id;
                    let tribe = game.state.tribes.get(&pov_id).unwrap();
                    println!("    FAILED TO FIND MOVE IN LEGAL MOVES!");

                    // if cmd_json["moveType"] == 9? use polyfish::types::CityRewardType

                    if move_type == polyfish::MoveType::Reward {
                        if turn_data.turn == 0 {
                            // replace -> "owner":$ID,"population":0,"production":1|2,"progress":0
                            // with -> "owner":$ID,"population":1,"production":1|2,"progress":1
                            let (new_json, fixed_file_path) = {
                                let content = json_str.replace(
                                    &format!(
                                        "\"owner\":{},\"population\":0,\"production\":1,\"progress\":0",
                                        player_data.player_id
                                    ),
                                    &format!(
                                        "\"owner\":{},\"population\":1,\"production\":1,\"progress\":1",
                                        player_data.player_id
                                    ),
                                ).replace(
                                    &format!(
                                        "\"owner\":{},\"population\":0,\"production\":2,\"progress\":0",
                                        player_data.player_id
                                    ),
                                    &format!(
                                        "\"owner\":{},\"population\":1,\"production\":2,\"progress\":1",
                                        player_data.player_id
                                    ),
                                );
                                let path = format!("{}_fixed.json", replay_base);
                                (content, path)
                            };

                            let _ = std::fs::write(&fixed_file_path, &new_json);
                            println!(
                                "    --- [FIX] Population fixed and saved to {}. Please re-run test.",
                                fixed_file_path
                            );
                        }

                        let reward_type: polyfish::CityRewardType =
                            serde_json::from_value(cmd_json["type"].clone()).unwrap();
                        println!(
                            "    Command: {:?} {:?} {}",
                            move_type,
                            reward_type,
                            cmd_json["target"].as_i64().unwrap(),
                        );
                    } else if move_type == polyfish::MoveType::Ability {
                        let ability_type: polyfish::AbilityType =
                            serde_json::from_value(cmd_json["type"].clone()).unwrap();
                        println!(
                            "    Command: {:?} {:?} {}",
                            move_type,
                            ability_type,
                            cmd_json
                                .get("target")
                                .or_else(|| cmd_json.get("src"))
                                .and_then(|v| v.as_i64())
                                .unwrap_or(0),
                        );
                    } else if move_type == polyfish::MoveType::Build {
                        let build_type: polyfish::StructureType =
                            serde_json::from_value(cmd_json["type"].clone()).unwrap();
                        println!(
                            "    Command: {:?} {:?} {}",
                            move_type,
                            build_type,
                            cmd_json
                                .get("target")
                                .or_else(|| cmd_json.get("src"))
                                .and_then(|v| v.as_i64())
                                .unwrap_or(0),
                        );
                    } else {
                        println!(
                            "    Command: {:?} {}",
                            polyfish::MoveType::from(
                                cmd_json["moveType"].as_i64().unwrap_or(0) as i32
                            ),
                            serde_json::to_string(&cmd_json).unwrap(),
                        );
                    }

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
                        let src_idx = cmd_json
                            .get("src")
                            .and_then(|v| v.as_i64())
                            .map(|v| v as i32);
                        if let Some(s_idx) = src_idx {
                            if let Some(u) = tribe.units.iter().find(|u| u.coords.idx == s_idx) {
                                println!(
                                    "    Unit at SRC ({}): Owner={}, Type={:?}, Moved={}, Attacked={}, Health={}/{}",
                                    s_idx,
                                    u.owner,
                                    u.unit_type,
                                    u.moved,
                                    u.attacked,
                                    u.health,
                                    polyfish::functions::get_unit_max_health(u)
                                );
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

                        // Debug cities for target
                        for (tid, t) in &game.state.tribes {
                            if let Some(c) = t.cities.iter().find(|c| c.idx == target_idx) {
                                println!(
                                    "    TARGET CITY ({}): Tribe={}, Level={}, Progress={}, Population={}, Rewards={:?}",
                                    target_idx, tid, c.level, c.progress, c.population, c.rewards
                                );
                            }
                        }
                    }

                    let failed_json = serde_json::to_string_pretty(&game.state).unwrap();
                    std::fs::write("saved_state.json", failed_json).unwrap();
                    println!("Saved failed state to saved_state.json");
                    panic!("Replay parsing failed.");
                }
            }
        }
    }

    println!("Successfully replayed {} commands!", success_commands);
    let final_json = serde_json::to_string_pretty(&game.state).unwrap();
    std::fs::write("saved_state.json", final_json).unwrap();
    println!("Saved final state to saved_state.json");
}
