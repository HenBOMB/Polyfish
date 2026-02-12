//! Loads a raw JSON state dump (like .initial.json) into the Rust simulator
//! and prints the summary. This validates that the state is loadable.

use polyfish::game::Game;
use serde_json::Value;
use std::env;
use std::fs;

// Copy of fixup logic from validate_csv.rs
fn fix_coords_recursive(val: &mut Value, map_size: i32) {
    match val {
        Value::Object(map) => {
            let keys: Vec<String> = map.keys().cloned().collect();
            for key in keys {
                let v = map.get_mut(&key).unwrap();
                if key.contains("oords") {
                    if let Some(arr) = v.as_array() {
                        if arr.len() == 2 {
                            let x = arr[0].as_i64().unwrap_or(0) as i32;
                            let y = arr[1].as_i64().unwrap_or(0) as i32;
                            *v = serde_json::json!({"x": x, "y": y, "idx": y * map_size + x});
                            continue;
                        }
                    }
                }
                fix_coords_recursive(v, map_size);
            }
        }
        Value::Array(arr) => {
            for item in arr.iter_mut() {
                fix_coords_recursive(item, map_size);
            }
        }
        _ => {}
    }
}

fn add_missing_fields(root: &mut Value, map_size: i32) {
    let obj = root.as_object_mut().unwrap();
    if !obj.contains_key("settings") {
        obj.insert("settings".into(), serde_json::json!({}));
    }
    let settings = obj.get_mut("settings").unwrap().as_object_mut().unwrap();
    for (k, v) in [
        ("turn", serde_json::json!(0)),
        ("currentPlayerTurnId", serde_json::json!(1)),
        ("mode", serde_json::json!(0)),
        ("size", serde_json::json!(map_size)),
        ("tileCount", serde_json::json!(map_size * map_size)),
        ("maxTurns", serde_json::json!(30)),
        ("_fow", serde_json::json!(true)),
        ("_areYouSure", serde_json::json!(false)),
        ("_gameOver", serde_json::json!(false)),
        ("_recentMoves", serde_json::json!([])),
        ("_pendingRewards", serde_json::json!([])),
        ("_lastPlayerTurnId", serde_json::json!(-1)),
        ("_maxTribeCount", serde_json::json!(2)),
        ("gameId", serde_json::json!("")),
        ("gameName", serde_json::json!("")),
        ("seed", serde_json::json!(0)),
        ("version", serde_json::json!(0)),
        ("mapType", serde_json::json!(0)),
        ("winByCapital", serde_json::json!(true)),
        ("winByExtermination", serde_json::json!(true)),
    ] {
        if !settings.contains_key(k) {
            settings.insert(k.into(), v);
        }
    }

    if let Some(tribes) = obj.get_mut("tribes").and_then(|t| t.as_object_mut()) {
        for (_, tribe) in tribes.iter_mut() {
            if let Some(cities) = tribe.get_mut("cities").and_then(|c| c.as_array_mut()) {
                for city in cities.iter_mut() {
                    if let Some(co) = city.as_object_mut() {
                        if !co.contains_key("id") {
                            let ti = co.get("tileIndex").and_then(|v| v.as_i64()).unwrap_or(0);
                            co.insert("id".into(), serde_json::json!(ti));
                        }
                    }
                }
            }
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: load_json <json_path>");
        std::process::exit(1);
    }

    let json_path = &args[1];
    println!("Loading state from: {}", json_path);

    let content = fs::read_to_string(json_path).expect("Failed to read JSON file");
    let mut root: Value = serde_json::from_str(&content).expect("Invalid JSON");

    let tile_count = root["tiles"].as_object().map(|o| o.len()).unwrap_or(0);
    // map size is sqrt of tile count
    let map_size = (tile_count as f64).sqrt() as i32;

    add_missing_fields(&mut root, map_size);
    fix_coords_recursive(&mut root, map_size);

    match Game::from_json(&serde_json::to_string(&root).unwrap()) {
        Ok(game) => {
            println!(
                "✅ Successfully loaded game state!\nTurn {}, P{} to play, {}x{} map",
                game.turn(),
                game.current_player_id(),
                map_size,
                map_size
            );

            println!("\nTribes:");
            for (id, tribe) in &game.state.tribes {
                println!(
                    "   P{}: {} (stars={}, units={}, cities={})",
                    id,
                    tribe.username,
                    tribe.stars,
                    tribe.units.len(),
                    tribe.cities.len()
                );
            }

            println!("\nLegal Moves: {}", game.legal_moves().len());
        }
        Err(e) => {
            eprintln!("❌ Failed to load game state: {}", e);
            std::process::exit(1);
        }
    }
}
