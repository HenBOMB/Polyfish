//! Village-approach behavior from self-play replays (replays/high_scores/*.json).
//! Steps each game through the engine; at the start of every player turn
//! <= --max-turn, classifies each visible open village by nearest-own-unit
//! Chebyshev distance, then grades the turn's outcome (captured / reached /
//! closed / stalled / moved away). Also classifies every Step as
//! toward/away/neutral w.r.t. the nearest visible open village.
//!
//! Usage: analyze_replays [--max-turn N] <replay.json> [...]

use polyfish::game::Game;
use polyfish::moves::{CaptureMove, RewardMove};
use polyfish::replayer::ModReplay;
use polyfish::states::GameState;
use polyfish::types::CityRewardType;
use std::collections::{BTreeMap, HashSet};

fn open_villages_of(state: &GameState) -> HashSet<i32> {
    let mut set = HashSet::new();
    for (&idx, s) in state.structures.iter() {
        let Some(s) = s else { continue };
        if s.structure_type == polyfish::types::StructureType::Village
            && state.tiles.get(&idx).map_or(false, |t| t.owner == 0)
        {
            set.insert(idx);
        }
    }
    set
}

fn min_dist_to(state: &GameState, pov: i32, village_idx: i32) -> Option<i32> {
    let tile = state.tiles.get(&village_idx)?;
    state
        .tribes
        .get(&pov)?
        .units
        .iter()
        .map(|u| u.coords.chebyshev_distance_to(&tile.coords))
        .min()
}

fn cmd_summary(cmd: &serde_json::Value) -> String {
    let mt = cmd.get("moveType").and_then(|v| v.as_i64()).unwrap_or(-99);
    let src = cmd.get("src").and_then(|v| v.as_i64());
    let tgt = cmd.get("target").and_then(|v| v.as_i64());
    let ty = cmd.get("type").and_then(|v| v.as_i64());
    match (mt, src, tgt) {
        (1, Some(s), Some(t)) => format!("Step {s}->{t}"),
        (8, Some(s), _) => format!("Capture {s}"),
        (5, _, Some(t)) => format!("Harvest {t}"),
        (9, _, Some(t)) => format!("Reward {t}/{}", ty.unwrap_or(-1)),
        (7, _, _) => format!("Research {}", ty.unwrap_or(-1)),
        (10, _, _) => "End".to_string(),
        _ => format!(
            "mt{mt}{}{}",
            src.map(|s| format!(" s{s}")).unwrap_or_default(),
            tgt.map(|t| format!(" t{t}")).unwrap_or_default()
        ),
    }
}

struct Episode {
    village: i32,
    d_start: i32,
}

fn main() {
    let mut max_turn: i32 = 10;
    let mut files: Vec<String> = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == "--max-turn" {
            max_turn = args.next().expect("--max-turn N").parse().expect("N");
        } else {
            files.push(a);
        }
    }
    assert!(!files.is_empty(), "usage: analyze_replays [--max-turn N] <replay.json> ...");

    // (d_class, outcome) -> count; d_class 3 means ">=3"
    let mut outcomes: BTreeMap<(i32, &'static str), usize> = BTreeMap::new();
    // turn -> [toward, away, neutral, no_visible_village]
    let mut step_dirs: BTreeMap<i32, [usize; 4]> = BTreeMap::new();
    let mut misses: Vec<String> = Vec::new();
    let mut capture_turns: Vec<(String, i32)> = Vec::new();
    let mut contested = 0usize;
    let mut invisible_village_turns = 0usize;
    let mut measured_player_turns = 0usize;

    for file in &files {
        let short = file.rsplit('/').next().unwrap_or(file).to_string();
        let raw = std::fs::read_to_string(file).expect("read replay");
        let replay: ModReplay = serde_json::from_str(&raw).expect("parse replay");

        let mut game = Game::new();
        game.state = replay.game_state.clone();
        if game.state.settings.current_player_turn_id == 0 {
            game.state.settings.current_player_turn_id = 1;
        }
        game.post_load();

        let mut open_villages = open_villages_of(&game.state);

        'turns: for turn_data in &replay.turns {
            for player_data in &turn_data.players {
                let pov = player_data.player_id;
                if let Some(tribe) = game.state.tribes.get_mut(&pov) {
                    for unit in &mut tribe.units {
                        unit.moved = false;
                        unit.attacked = false;
                        unit.attacks_performed = 0;
                    }
                }
                game.state.settings.current_player_turn_id = pov;

                let measure = turn_data.turn <= max_turn;
                let mut episodes: Vec<Episode> = Vec::new();
                if measure {
                    measured_player_turns += 1;
                    for &v in &open_villages {
                        let Some(tile) = game.state.tiles.get(&v) else { continue };
                        if !tile.explorers.contains(&pov) {
                            invisible_village_turns += 1;
                            continue;
                        }
                        if tile._unit_owner_id.map_or(false, |o| o != pov) {
                            contested += 1;
                            continue;
                        }
                        if let Some(d) = min_dist_to(&game.state, pov, v) {
                            episodes.push(Episode { village: v, d_start: d });
                        }
                    }
                }

                let mut captured_block: HashSet<i32> = HashSet::new();
                let mut cmds_text: Vec<String> = Vec::new();

                for cmd_json in &player_data.commands {
                    let mt = cmd_json.get("moveType").and_then(|v| v.as_i64());
                    if mt == Some(-1) || mt == Some(11) {
                        continue;
                    }
                    cmds_text.push(cmd_summary(cmd_json));

                    // Classify Step direction vs nearest visible open village (pre-move).
                    if mt == Some(1) && turn_data.turn <= max_turn {
                        let s = cmd_json.get("src").and_then(|v| v.as_i64()).unwrap() as i32;
                        let t = cmd_json.get("target").and_then(|v| v.as_i64()).unwrap() as i32;
                        let entry = step_dirs.entry(turn_data.turn).or_default();
                        let visible_best = open_villages
                            .iter()
                            .filter(|&&v| {
                                game.state
                                    .tiles
                                    .get(&v)
                                    .map_or(false, |tile| tile.explorers.contains(&pov))
                            })
                            .filter_map(|&v| {
                                let vt = game.state.tiles.get(&v)?;
                                let st = game.state.tiles.get(&s)?;
                                Some((v, st.coords.chebyshev_distance_to(&vt.coords)))
                            })
                            .min_by_key(|&(_, d)| d);
                        match visible_best {
                            None => entry[3] += 1,
                            Some((v, d_before)) => {
                                let vt = game.state.tiles.get(&v).unwrap();
                                let tt = game.state.tiles.get(&t).unwrap();
                                let d_after = tt.coords.chebyshev_distance_to(&vt.coords);
                                if d_after < d_before {
                                    entry[0] += 1;
                                } else if d_after > d_before {
                                    entry[1] += 1;
                                } else {
                                    entry[2] += 1;
                                }
                            }
                        }
                    }

                    let mut stripped = cmd_json
                        .as_object()
                        .expect("command is not an object")
                        .clone();
                    stripped.remove("_reward");
                    stripped.remove("_revealedTiles");
                    let stripped_val = serde_json::Value::Object(stripped);

                    let legal_moves = game.legal_moves();
                    let mut found = false;
                    for m in &legal_moves {
                        if m.serialize() != stripped_val {
                            continue;
                        }
                        match mt {
                            Some(9) => {
                                let reward_type: CityRewardType = serde_json::from_value(
                                    cmd_json.get("type").unwrap().clone(),
                                )
                                .expect("reward type");
                                let mut mv = RewardMove::new(
                                    cmd_json.get("target").unwrap().as_i64().unwrap() as i32,
                                    reward_type,
                                );
                                if let Some(tiles) = cmd_json.get("_revealedTiles") {
                                    mv.revealed_tiles =
                                        Some(serde_json::from_value(tiles.clone()).unwrap());
                                }
                                game.play_move(&mv);
                            }
                            Some(8) => {
                                let src =
                                    cmd_json.get("src").unwrap().as_i64().unwrap() as i32;
                                let mut mv = CaptureMove::new(src);
                                if let Some(r) = cmd_json.get("_reward") {
                                    mv.reward = Some(serde_json::from_value(r.clone()).unwrap());
                                }
                                if let Some(tiles) = cmd_json.get("_revealedTiles") {
                                    mv.revealed_tiles =
                                        Some(serde_json::from_value(tiles.clone()).unwrap());
                                }
                                if let Some(t) =
                                    cmd_json.get("_type").or_else(|| cmd_json.get("type"))
                                {
                                    if let Ok(tech) = serde_json::from_value(t.clone()) {
                                        mv.tech_hint = Some(tech);
                                    }
                                }
                                game.play_move(&mv);
                                if open_villages.remove(&src) {
                                    captured_block.insert(src);
                                    capture_turns.push((short.clone(), turn_data.turn));
                                }
                            }
                            _ => {
                                game.play_move(m.as_ref());
                            }
                        }
                        game.state._messages.clear();
                        found = true;
                        break;
                    }
                    if !found {
                        eprintln!(
                            "[{short}] turn {} p{pov}: no legal match for {} — stopping file",
                            turn_data.turn,
                            serde_json::to_string(cmd_json).unwrap()
                        );
                        break 'turns;
                    }
                }

                for ep in &episodes {
                    let d_class = ep.d_start.min(3);
                    let unit_on = game
                        .state
                        .tribes
                        .get(&pov)
                        .map_or(false, |t| t.units.iter().any(|u| u.coords.idx == ep.village));
                    let d_end = min_dist_to(&game.state, pov, ep.village);
                    let outcome: &'static str = if captured_block.contains(&ep.village) {
                        "captured"
                    } else if unit_on {
                        if ep.d_start == 0 { "still-on-uncaptured" } else { "reached" }
                    } else {
                        match d_end {
                            Some(de) if de < ep.d_start => "closed",
                            Some(de) if de == ep.d_start => "stalled",
                            Some(_) => "moved-away",
                            None => "no-units",
                        }
                    };
                    *outcomes.entry((d_class, outcome)).or_default() += 1;
                    let success = matches!(
                        (ep.d_start, outcome),
                        (0, "captured") | (1, "captured") | (1, "reached") | (2, "captured") | (2, "reached") | (2, "closed")
                    );
                    if !success && ep.d_start <= 2 {
                        misses.push(format!(
                            "{short} t{} p{pov} v{} d{}: {} | {}",
                            turn_data.turn,
                            ep.village,
                            ep.d_start,
                            outcome,
                            cmds_text.join(", ")
                        ));
                    }
                }
            }
        }
    }

    println!("=== {} files, {} player-turns measured (turn <= {max_turn}) ===", files.len(), measured_player_turns);
    println!("\n--- village episodes by starting distance (visible, uncontested) ---");
    for d in 0..=3 {
        let total: usize = outcomes.iter().filter(|((dc, _), _)| *dc == d).map(|(_, c)| c).sum();
        if total == 0 {
            continue;
        }
        let label = if d == 3 { ">=3".to_string() } else { d.to_string() };
        print!("[d={label}] n={total}: ");
        let mut parts: Vec<String> = outcomes
            .iter()
            .filter(|((dc, _), _)| *dc == d)
            .map(|((_, o), c)| format!("{o} {c} ({:.0}%)", 100.0 * *c as f64 / total as f64))
            .collect();
        parts.sort();
        println!("{}", parts.join(" | "));
    }
    println!("\ncontested (enemy on village): {contested} | village-not-yet-visible player-turns: {invisible_village_turns}");

    println!("\n--- Step direction vs nearest visible open village, by turn ---");
    println!("turn | toward  away  neutral  no-vis-village");
    for (t, [tw, aw, ne, nv]) in &step_dirs {
        println!("{t:4} | {tw:6} {aw:5} {ne:8} {nv:14}");
    }

    println!("\n--- village capture turns ---");
    for (f, t) in &capture_turns {
        println!("  t{t:2}  {f}");
    }

    println!("\n--- misses (d<=2 episodes that did not progress) ---");
    for m in &misses {
        println!("  {m}");
    }
}
