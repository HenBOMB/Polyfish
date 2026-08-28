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

    // Steam-mod replays open with a moveType:-1 "start match" marker plus one
    // forced auto-played move; strip them ONLY when the marker is present.
    // Engine-generated recaps (self_play best-game replays) have no padding —
    // unconditional stripping ate their first two real moves and desynced
    // every command after (found via the Stage-3 XinXi observability probe).
    if !mod_replay.turns.is_empty() && !mod_replay.turns[0].players.is_empty() {
        let cmds = &mut mod_replay.turns[0].players[0].commands;
        let has_marker = cmds
            .first()
            .and_then(|c| c.get("moveType"))
            .and_then(|v| v.as_i64())
            == Some(-1);
        if has_marker && cmds.len() >= 2 {
            cmds.remove(0); // the -1 "start match" marker
            cmds.remove(0); // the forced auto-played first move
        }
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
                            if let Some(t) = cmd_json.get("_type").or_else(|| cmd_json.get("type")) {
                                if let Ok(tech) = serde_json::from_value(t.clone()) {
                                    m_with_hints.tech_hint = Some(tech);
                                }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression for a 2026-08-27 replay-desync incident: a stale binary
    /// still keying ruin rewards off drifting `settings.seed` (pre-9a76ca4)
    /// reconstructed a DIFFERENT reward than the real commit granted,
    /// desyncing every later move that turn. Exercises the actual self-play
    /// RECORD (serialize before execute) → REPLAY round trip, with search
    /// noise plus a drifted `settings.seed` standing in for the stale-vs-live
    /// mismatch; `ruin_rng_pick` must ignore both.
    #[test]
    fn ruin_capture_survives_record_replay_round_trip_through_search_noise() {
        use crate::coords::Coords;
        use crate::moves::Move;
        use crate::states::{CityState, StructureState, TechnologyState, TileState, TribeState, UnitState};
        use crate::types::{StructureType, TechnologyType, TerrainType, TribeType, UnitType};

        let tile_idx = 42;
        let cap_idx = 0;

        let mut game = Game::new();
        game.state.initial_seed = 424242;
        game.state.settings.size = 11;
        game.state.settings._max_tribe_count = 2;
        game.state.settings.current_player_turn_id = 1;

        let mut tribe = TribeState::default();
        tribe.id = 1;
        tribe.tribe_type = TribeType::Imperius;
        tribe.stars = 5;
        tribe.tech_vanilla = vec![TechnologyState {
            tech_type: TechnologyType::Basic,
            discovered: true,
            discovered_turn: 0,
        }];
        tribe.cities.push(CityState { owner: 1, idx: cap_idx, ..Default::default() });
        tribe.units.push(UnitState {
            coords: Coords::from_index(tile_idx, 11),
            owner: 1,
            unit_type: UnitType::Warrior,
            ..Default::default()
        });
        game.state.tribes.insert(1, tribe);
        // A second tribe so turn-transition machinery has somewhere to go.
        let mut tribe2 = TribeState::default();
        tribe2.id = 2;
        tribe2.tribe_type = TribeType::Bardur;
        game.state.tribes.insert(2, tribe2);

        game.state.tiles.insert(cap_idx, TileState {
            coords: Coords::from_index(cap_idx, 11),
            terrain_type: TerrainType::Field,
            capital_of: 1,
            owner: 1,
            ..Default::default()
        });
        game.state.tiles.insert(tile_idx, TileState {
            coords: Coords::from_index(tile_idx, 11),
            terrain_type: TerrainType::Field,
            owner: 1,
            ..Default::default()
        });
        game.state.structures.insert(
            tile_idx,
            Some(StructureState { structure_type: StructureType::Ruin, ..Default::default() }),
        );

        game.post_load();
        // The replay's starting snapshot, taken BEFORE `settings.seed` below
        // is drifted — exactly like a real replay file, which stores the
        // map's fixed initial state, not whatever `settings.seed` had
        // drifted to by the time a mid-game ruin gets captured.
        let pristine = game.state.clone();

        // Search noise: speculatively capture-and-undo the same ruin many
        // times before the real commit, mimicking MCTS evaluating the
        // branch. Each call runs with `_are_you_sure = false` (see
        // `simulate_move`), same as a real search rollout.
        for _ in 0..25 {
            if let Some(undo) = game.simulate_move(&CaptureMove::new(tile_idx)) {
                undo(&mut game.state);
            }
        }
        assert_eq!(
            game.state.tribes.get(&1).unwrap().stars,
            5,
            "search noise must fully undo — no leaked stars before the real commit"
        );

        // Simulate unrelated earlier real game history (other captures,
        // combat, etc.) having already advanced the game's running RNG
        // cursor by the time THIS ruin gets captured. This is the drift the
        // pre-fix `next_rng_xxhash` picked its reward index from; the fixed
        // `ruin_rng_pick` must ignore it entirely.
        game.state.settings.seed = 987_654_321;

        // Real commit, recorded self-play-style: serialize BEFORE execute
        // (matches self_play.rs's flat_recap.push(..., m.serialize()) then
        // game.play_move(m.as_ref()) ordering).
        let real_move = CaptureMove::new(tile_idx);
        let recorded_json = real_move.serialize();
        assert!(game.play_move(&real_move).is_some(), "real capture must succeed");

        let real_stars = game.state.tribes.get(&1).unwrap().stars;
        let real_tech: Vec<_> =
            game.state.tribes.get(&1).unwrap().tech_vanilla.iter().map(|t| t.tech_type).collect();

        // Reconstruct purely from the recorded command, from scratch.
        let mut replay = ModReplay {
            game_state: pristine,
            turns: vec![ReplayTurn {
                turn: 0,
                players: vec![ReplayPlayer { player_id: 1, commands: vec![recorded_json] }],
            }],
        };
        let mut reconstructed = Game::new();
        replay_game(&mut reconstructed, &mut replay).expect("replay must reconstruct cleanly");

        assert_eq!(
            reconstructed.state.tribes.get(&1).unwrap().stars,
            real_stars,
            "replay reconstruction must grant the SAME ruin reward as the real commit"
        );
        let reconstructed_tech: Vec<_> = reconstructed
            .state
            .tribes
            .get(&1)
            .unwrap()
            .tech_vanilla
            .iter()
            .map(|t| t.tech_type)
            .collect();
        assert_eq!(reconstructed_tech, real_tech);
    }

    /// Replay-checkpoint probe (manual): board/economy audit of a saved
    /// replay at turn checkpoints, plus giant/hub timelines. Run:
    ///   REPLAY_FILE=replays/<file>.json cargo test --lib replayer -- \
    ///     --ignored --nocapture
    #[test]
    #[ignore]
    fn replay_checkpoint_probe() {
        let path = std::env::var("REPLAY_FILE").expect("set REPLAY_FILE");
        let src: ModReplay =
            serde_json::from_str(&std::fs::read_to_string(&path).expect("read")).expect("parse");
        let total = src.turns.len();
        let mut first_giant: Option<(i32, i32)> = None;
        let mut giant_first_by_player: std::collections::HashMap<i32, i32> = Default::default();

        for n in 1..=total {
            let mut clip = src.clone();
            clip.turns.truncate(n);
            let mut game = Game::new();
            if let Err(e) = replay_game(&mut game, &mut clip) {
                eprintln!("replay error at prefix {n}: {e}");
            }
            let turn = clip.turns.last().map(|t| t.turn).unwrap_or(0);
            for (pid, tr) in &game.state.tribes {
                let g = tr
                    .units
                    .iter()
                    .filter(|u| u.unit_type == crate::types::UnitType::Giant)
                    .count();
                if g > 0 {
                    giant_first_by_player.entry(*pid).or_insert(turn);
                    if first_giant.is_none() {
                        first_giant = Some((turn, *pid));
                    }
                }
            }
            let checkpoint = matches!(turn, 10 | 15 | 20) && {
                // only report each checkpoint once (first prefix reaching it)
                clip.turns.len() == n
                    && (n == total || src.turns.get(n).map(|t| t.turn) != Some(turn))
            };
            if checkpoint || n == total {
                let state = &game.state;
                println!("===== after turn {turn} (prefix {n}/{total}) =====");
                for (pid, tr) in state.tribes.iter() {
                    let spt = crate::functions::get_tribe_spt(state, tr);
                    let giants = tr
                        .units
                        .iter()
                        .filter(|u| u.unit_type == crate::types::UnitType::Giant)
                        .count();
                    println!(
                        "P{pid}: spt={spt} stars={} score={} units={} giants={giants}",
                        tr.stars,
                        tr.score,
                        tr.units.len()
                    );
                    for c in &tr.cities {
                        let w = state.settings.size;
                        println!(
                            "  city@{} ({},{}) lvl {} pop {}/{}:",
                            c.idx,
                            c.idx % w,
                            c.idx / w,
                            c.level,
                            c.population,
                            c.level + 1
                        );
                        for &ti in &c._territory {
                            let tile = match state.tiles.get(&ti) {
                                Some(t) => t,
                                None => continue,
                            };
                            let res = state
                                .resources
                                .get(&ti)
                                .and_then(|r| r.as_ref())
                                .map(|r| format!(" res:{:?}", r.resource_type))
                                .unwrap_or_default();
                            let st = crate::functions::get_structure_at(state, ti)
                                .map(|s| format!(" bld:{:?}(l{})", s.structure_type, s.level))
                                .unwrap_or_default();
                            let un = tile
                                ._unit_owner_id
                                .and_then(|o| {
                                    state.tribes.get(&o).and_then(|t| {
                                        t.units
                                            .iter()
                                            .find(|u| u.coords.idx == ti)
                                            .map(|u| format!(" unit:P{o}:{:?}", u.unit_type))
                                    })
                                })
                                .unwrap_or_default();
                            println!(
                                "    t{ti}({},{}) {:?}{res}{st}{un}",
                                ti % w,
                                ti / w,
                                tile.terrain_type
                            );
                        }
                    }
                    let hubs: Vec<String> = state
                        .structures
                        .iter()
                        .filter_map(|(i, s)| s.as_ref().map(|s| (i, s)))
                        .filter(|(i, s)| {
                            state.tiles.get(*i).map(|t| t.owner) == Some(*pid)
                                && !crate::settings::structures::get_structure_setting(
                                    s.structure_type,
                                )
                                .adjacent_types
                                .is_empty()
                        })
                        .map(|(i, s)| format!("{:?}@{i}(l{})", s.structure_type, s.level))
                        .collect();
                    println!("  hubs[{}]: {}", hubs.len(), hubs.join(", "));
                }
            }
        }
        println!("first giant: {first_giant:?} | per-player first-giant turns: {giant_first_by_player:?}");
    }

    /// Per-turn economy/ownership audit of a saved replay (manual). One line
    /// per player per turn — stars, spt, score, units, and every city with
    /// level and population — plus a CITY CHANGED marker whenever a city
    /// changes hands. Answers "what did it have to spend, and when did it
    /// lose the capital" without stepping the UI. Run:
    ///   REPLAY_FILE=replays/<file>.json cargo test --lib replay_turn_audit -- \
    ///     --ignored --nocapture
    #[test]
    #[ignore]
    fn replay_turn_audit() {
        let path = std::env::var("REPLAY_FILE").expect("set REPLAY_FILE");
        let src: ModReplay =
            serde_json::from_str(&std::fs::read_to_string(&path).expect("read")).expect("parse");
        let total = src.turns.len();
        let mut prev_owner: std::collections::HashMap<i32, i32> = Default::default();
        for n in 1..=total {
            let mut clip = src.clone();
            clip.turns.truncate(n);
            let mut game = Game::new();
            if let Err(e) = replay_game(&mut game, &mut clip) {
                eprintln!("replay error at prefix {n}: {e}");
            }
            let turn = clip.turns.last().map(|t| t.turn).unwrap_or(0);
            let state = &game.state;
            let mut pids: Vec<i32> = state.tribes.keys().copied().collect();
            pids.sort();
            println!("===== after turn {turn} =====");
            for pid in pids {
                let Some(tr) = state.tribes.get(&pid) else { continue };
                let cities: Vec<String> = tr
                    .cities
                    .iter()
                    .map(|c| {
                        format!("@{} lvl{} pop{}/{}", c.idx, c.level, c.population, c.level + 1)
                    })
                    .collect();
                println!(
                    "  P{pid}: stars={} spt={} score={} units={} | cities: {}",
                    tr.stars,
                    crate::functions::get_tribe_spt(state, tr),
                    tr.score,
                    tr.units.len(),
                    cities.join(", ")
                );
                for c in &tr.cities {
                    if prev_owner.insert(c.idx, pid).map_or(false, |o| o != pid) {
                        println!("    ** CITY CHANGED HANDS: @{} -> P{pid} on turn {turn}", c.idx);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod stance_tests {
    use super::*;

    /// Per-turn raw stance signal audit for a replay (manual):
    ///   REPLAY_FILE=... cargo test --lib stance_probe -- --ignored --nocapture
    #[test]
    #[ignore]
    fn stance_probe() {
        let path = std::env::var("REPLAY_FILE").expect("set REPLAY_FILE");
        let src: ModReplay =
            serde_json::from_str(&std::fs::read_to_string(&path).expect("read")).expect("parse");
        println!("turn | P1 stance (raw) orders | enemy units seen <=2 of P1 cities | army stars P1/P2 | P1 stars");
        for n in 1..=src.turns.len() {
            let mut clip = src.clone();
            clip.turns.truncate(n);
            let mut game = Game::new();
            let _ = replay_game(&mut game, &mut clip);
            let turn = clip.turns.last().map(|t| t.turn).unwrap_or(0);
            let state = &game.state;
            let mut commit = crate::ai::oracle_macro::StanceCommit::default();
            let g = crate::ai::oracle_macro::commit_macro_goal(state, 1, &mut commit, 0);
            let w = state.settings.size;
            let cheb = |a: i32, b: i32| ((a % w - b % w).abs()).max((a / w - b / w).abs());
            let near = state
                .tribes
                .get(&2)
                .map(|t| {
                    t.units
                        .iter()
                        .filter(|u| {
                            state
                                .tiles
                                .get(&u.coords.idx)
                                .map(|x| x.explorers.contains(&1))
                                .unwrap_or(false)
                                && state.tribes.get(&1).map_or(false, |p1| {
                                    p1.cities.iter().any(|c| cheb(c.idx, u.coords.idx) <= 2)
                                })
                        })
                        .count()
                })
                .unwrap_or(0);
            let stars_of = |pid: i32| {
                state.tribes.get(&pid).map_or(0, |t| {
                    t.units
                        .iter()
                        .map(|u| crate::settings::units::get_unit_setting(u.unit_type).cost)
                        .sum()
                })
            };
            let arm_i = crate::ai::oracle_macro::stance_pressure(state, 1).arm;
            println!(
                "t{turn:2} | {:?} i={arm_i:.2} ({} orders) | {near} | {}/{} | {}",
                g.stance,
                g.orders.len(),
                stars_of(1),
                stars_of(2),
                state.tribes.get(&1).map_or(0, |t| t.stars)
            );
        }
    }
}
