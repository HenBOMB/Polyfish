//! Validates scraper CSV data against the Rust simulator.
//!
//! Uses tile ownership changes (tiles.N._unitOwnerID) as the primary signal
//! to identify moves. This avoids unit array index reshuffling problems.
//! The Rust simulator validates each inferred move against legal_moves().

use polyfish::game::Game;
use polyfish::moves::Move;
use polyfish::types::MoveType;
use serde_json::Value;
use std::env;
use std::fs;

// ─── JSON fixup ──────────────────────────────────────────────────────────────

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

// ─── Delta analysis ──────────────────────────────────────────────────────────

fn delta_player(keys: &[String]) -> Option<i32> {
    for k in keys {
        if k.starts_with("tribes.") {
            if let Some(s) = k.split('.').nth(1) {
                if let Ok(id) = s.parse::<i32>() {
                    return Some(id);
                }
            }
        }
    }
    None
}

/// Parsed delta in terms of tile ownership
struct DeltaInfo {
    player: Option<i32>,
    /// Tiles that gained a unit owner: (tile_idx, new_owner)
    tiles_gained: Vec<(i32, i32)>,
    /// Tiles that lost their unit owner (vacated)
    tiles_vacated: Vec<i32>,
    /// Stars change for affected player
    new_stars: Option<i32>,
    /// New tech discovered
    has_tech: bool,
    /// Health changes on enemy units (attack result)
    has_enemy_damage: bool,
    /// Resource removed (harvest)
    resource_removed: Option<i32>,
    /// Structure added
    structure_added: Option<i32>,
    /// Unit type from delta (e.g. 2=Warrior, 3=Rider)
    unit_type: Option<i32>,
    /// Reward detected
    has_reward: bool,
    /// Is this just noise?
    is_noise: bool,
}

fn parse_delta(delta: &Value, keys: &[String]) -> DeltaInfo {
    let player = delta_player(keys);
    let mut tiles_gained = vec![];
    let mut tiles_vacated = vec![];
    let mut new_stars = None;
    let mut has_tech = false;
    let mut has_enemy_damage = false;
    let mut resource_removed = None;
    let mut structure_added = None;
    let mut has_reward = false;

    for k in keys {
        // Tile ownership
        if k.contains("._unitOwnerID") {
            if let Some(idx_str) = k
                .strip_prefix("tiles.")
                .and_then(|s| s.strip_suffix("._unitOwnerID"))
            {
                if let Ok(idx) = idx_str.parse::<i32>() {
                    let val = delta.get(k);
                    if val.map(|v| v.is_null()).unwrap_or(false)
                        || val.map(|v| v == &Value::Null).unwrap_or(false)
                    {
                        tiles_vacated.push(idx);
                    } else if let Some(owner) = val.and_then(|v| v.as_i64()) {
                        tiles_gained.push((idx, owner as i32));
                    }
                }
            }
        }
        // Stars
        if k.ends_with(".stars") && k.starts_with("tribes.") {
            new_stars = delta.get(k).and_then(|v| v.as_i64()).map(|v| v as i32);
        }
        // Tech
        if k.contains("tech_vanilla[") {
            has_tech = true;
        }
        // Health changes (damage)
        if k.contains(".health") && k.contains("units[") {
            // Check if it's a DIFFERENT player's unit
            if let (Some(p), Some(health_tribe)) = (player, k.split('.').nth(1)) {
                if let Ok(ht) = health_tribe.parse::<i32>() {
                    if ht != p {
                        has_enemy_damage = true;
                    }
                }
            }
        }
        // Resource removed
        if k.starts_with("resources.") {
            if delta.get(k).map(|v| v.is_null()).unwrap_or(false) {
                if let Some(idx_str) = k.strip_prefix("resources.") {
                    if let Ok(idx) = idx_str.parse::<i32>() {
                        resource_removed = Some(idx);
                    }
                }
            }
        }
        // Structure added
        if k.starts_with("structures.") {
            if !delta.get(k).map(|v| v.is_null()).unwrap_or(false) {
                if let Some(idx_str) = k
                    .strip_prefix("structures.")
                    .and_then(|s| s.split('.').next())
                {
                    if let Ok(idx) = idx_str.parse::<i32>() {
                        structure_added = Some(idx);
                    }
                }
            }
        }

        // Reward detection
        if k.contains(".explorers")
            || k.contains(".workshop")
            || k.contains(".cityWall")
            || k.contains(".resources") && k.contains("tribes.")
        {
            has_reward = true;
        }
    }

    // Extract unit type from delta (e.g. "tribes.1.units[0].type" = 3)
    let mut unit_type: Option<i32> = None;
    for k in keys {
        if k.contains("units[") && k.ends_with(".type") {
            if let Some(v) = delta.get(k).and_then(|v| v.as_i64()) {
                unit_type = Some(v as i32);
                break; // take the first one
            }
        }
    }

    // Determine if noise
    let has_action = !tiles_gained.is_empty()
        || !tiles_vacated.is_empty()
        || has_tech
        || has_enemy_damage
        || resource_removed.is_some()
        || structure_added.is_some()
        || has_reward;

    let is_noise = !has_action;

    DeltaInfo {
        player,
        tiles_gained,
        tiles_vacated,
        new_stars,
        has_tech,
        has_enemy_damage,
        resource_removed,
        structure_added,
        unit_type,
        has_reward,
        is_noise,
    }
}

// ─── Move matching ───────────────────────────────────────────────────────────

/// Find a Step move where the unit moves TO one of the gained tiles
fn match_step(
    legal: &[Box<dyn Move>],
    game: &Game,
    gained: &[(i32, i32)],
    vacated: &[i32],
) -> Option<usize> {
    let pid = game.current_player_id();

    // A step: unit leaves one tile (vacated), arrives at another (gained)
    for &(gained_tile, owner) in gained {
        if owner != pid {
            continue;
        }
        // Find a legal step that targets this tile
        for (i, m) in legal.iter().enumerate() {
            if m.move_type() == MoveType::Step {
                if m.target_idx().ok() == Some(gained_tile as usize) {
                    // Verify the source is in the vacated list (or at least exists)
                    if let Ok(src) = m.source_idx() {
                        if vacated.contains(&(src as i32)) || vacated.is_empty() {
                            return Some(i);
                        }
                    }
                }
            }
        }
    }

    // Fallback: just match any step that targets a gained tile
    for &(gained_tile, owner) in gained {
        if owner != pid {
            continue;
        }
        for (i, m) in legal.iter().enumerate() {
            if m.move_type() == MoveType::Step && m.target_idx().ok() == Some(gained_tile as usize)
            {
                return Some(i);
            }
        }
    }

    None
}

/// Find an Attack move targeting a tile where enemy damage occurred
fn match_attack(legal: &[Box<dyn Move>], gained: &[(i32, i32)]) -> Option<usize> {
    // The gained tile in an attack is where the enemy is (attacker didn't move there)
    // Actually, after attack the attacker stays in place. Look for Attack moves.
    for (i, m) in legal.iter().enumerate() {
        if m.move_type() == MoveType::Attack {
            return Some(i);
        }
    }
    None
}

/// Find a Research move (any, since the delta tells us one was researched)
fn match_research(
    legal: &[Box<dyn Move>],
    game: &Game,
    delta: &Value,
    keys: &[String],
) -> Option<usize> {
    let pid = game.current_player_id();

    // Try to extract the tech type from the delta
    let mut target_type: Option<i32> = None;
    for k in keys {
        if k.contains("tech_vanilla[") {
            if let Some(val) = delta.get(k) {
                if let Some(t) = val.get("type").and_then(|v| v.as_i64()) {
                    target_type = Some(t as i32);
                }
            }
        }
    }

    // Match by tech type if we have it
    if let Some(tt) = target_type {
        for (i, m) in legal.iter().enumerate() {
            if m.move_type() == MoveType::Research {
                if let Ok(tech) = m.tech_type() {
                    if tech as i32 == tt {
                        return Some(i);
                    }
                }
            }
        }
    }

    // Fallback: any research
    for (i, m) in legal.iter().enumerate() {
        if m.move_type() == MoveType::Research {
            return Some(i);
        }
    }
    None
}

/// Find a Summon move at a gained tile, optionally matching unit type
fn match_summon(
    legal: &[Box<dyn Move>],
    game: &Game,
    gained: &[(i32, i32)],
    _new_stars: Option<i32>,
    delta_unit_type: Option<i32>,
) -> Option<usize> {
    let pid = game.current_player_id();

    // Priority 1: Match by tile + unit type
    for &(tile, owner) in gained {
        if owner != pid {
            continue;
        }
        for (i, m) in legal.iter().enumerate() {
            if m.move_type() == MoveType::Summon && m.source_idx().ok() == Some(tile as usize) {
                if let Some(expected_type) = delta_unit_type {
                    if let Ok(mt) = m.unit_type() {
                        if mt as i32 == expected_type {
                            return Some(i);
                        }
                    }
                } else {
                    return Some(i); // No type filter, take first match at tile
                }
            }
        }
    }

    // Priority 2: Match by tile only (ignore unit type)
    for &(tile, owner) in gained {
        if owner != pid {
            continue;
        }
        for (i, m) in legal.iter().enumerate() {
            if m.move_type() == MoveType::Summon && m.source_idx().ok() == Some(tile as usize) {
                return Some(i);
            }
        }
    }

    // Priority 3: Any summon with matching unit type
    if let Some(expected_type) = delta_unit_type {
        for (i, m) in legal.iter().enumerate() {
            if m.move_type() == MoveType::Summon {
                if let Ok(mt) = m.unit_type() {
                    if mt as i32 == expected_type {
                        return Some(i);
                    }
                }
            }
        }
    }

    // Priority 4: Any summon move
    for (i, m) in legal.iter().enumerate() {
        if m.move_type() == MoveType::Summon {
            return Some(i);
        }
    }

    None
}

/// Find a Harvest move at the resource tile
fn match_harvest(legal: &[Box<dyn Move>], tile: i32) -> Option<usize> {
    for (i, m) in legal.iter().enumerate() {
        if m.move_type() == MoveType::Harvest && m.target_idx().ok() == Some(tile as usize) {
            return Some(i);
        }
    }
    None
}

/// Find a Build move at the structure tile
fn match_build(legal: &[Box<dyn Move>], tile: i32) -> Option<usize> {
    for (i, m) in legal.iter().enumerate() {
        if m.move_type() == MoveType::Build && m.target_idx().ok() == Some(tile as usize) {
            return Some(i);
        }
    }
    None
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: validate_csv <csv_path> [--game2] [--verbose]");
        std::process::exit(1);
    }

    let csv_path = &args[1];
    let use_game2 = args.iter().any(|a| a == "--game2");
    let verbose = args.iter().any(|a| a == "--verbose" || a == "-v");

    let content = fs::read_to_string(csv_path).expect("Failed to read CSV");
    let lines: Vec<&str> = content.lines().collect();

    let mut base_indices = vec![];
    for (i, line) in lines.iter().enumerate().skip(1) {
        let parts: Vec<&str> = line.splitn(3, ',').collect();
        if parts.len() >= 3 && parts[1] == "base" {
            base_indices.push(i);
        }
    }
    let base_idx = if use_game2 && base_indices.len() > 1 {
        base_indices[1]
    } else {
        base_indices[0]
    };
    let end_idx = if use_game2 {
        lines.len()
    } else if base_indices.len() > 1 {
        base_indices[1]
    } else {
        lines.len()
    };

    // Load base state
    let parts: Vec<&str> = lines[base_idx].splitn(3, ',').collect();
    let mut root: Value = serde_json::from_str(parts[2]).expect("Invalid JSON");
    let tile_count = root["tiles"].as_object().map(|o| o.len()).unwrap_or(0);
    let map_size = (tile_count as f64).sqrt() as i32;

    // Detect starting player from first action delta
    let mut starting_player = 1i32;
    for i in (base_idx + 1)..end_idx {
        let dp: Vec<&str> = lines[i].splitn(3, ',').collect();
        if dp.len() < 3 {
            continue;
        }
        let d: Value = serde_json::from_str(dp[2]).unwrap_or_default();
        let keys: Vec<String> = d
            .as_object()
            .map(|o| o.keys().cloned().collect())
            .unwrap_or_default();
        let info = parse_delta(&d, &keys);
        if !info.is_noise {
            if let Some(p) = info.player {
                starting_player = p;
                break;
            }
        }
    }

    add_missing_fields(&mut root, map_size);
    root["settings"]["currentPlayerTurnId"] = serde_json::json!(starting_player);
    // Set turn to 1 so that after the first EndTurn cycle (which increments to 2),
    // production income fires correctly (game.rs checks turn > 1)
    root["settings"]["turn"] = serde_json::json!(1);
    fix_coords_recursive(&mut root, map_size);

    let mut game = match Game::from_json(&serde_json::to_string(&root).unwrap()) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("❌ Failed to load: {}", e);
            std::process::exit(1);
        }
    };

    println!(
        "✅ Loaded: Turn {}, P{} starts, {}x{}",
        game.turn(),
        game.current_player_id(),
        map_size,
        map_size
    );
    for (id, tribe) in &game.state.tribes {
        println!(
            "   P{}: {} ({:?}), stars={}, units={}, cities={}",
            id,
            tribe.username,
            tribe.tribe_type,
            tribe.stars,
            tribe.units.len(),
            tribe.cities.len()
        );
    }

    // ─── Process deltas ──────────────────────────────────────────────────────
    let mut moves_played: Vec<(usize, i32, String, String, Value)> = vec![]; // (row, player, type, desc, json)
    let mut noise_count = 0;
    let mut failed_count = 0;
    // Track last known stars from deltas to correct sim drift
    let mut last_known_stars: std::collections::HashMap<i32, i32> =
        std::collections::HashMap::new();

    for delta_row in (base_idx + 1)..end_idx {
        let dp: Vec<&str> = lines[delta_row].splitn(3, ',').collect();
        if dp.len() < 3 {
            continue;
        }
        let delta: Value = serde_json::from_str(dp[2]).unwrap_or_default();
        let keys: Vec<String> = delta
            .as_object()
            .map(|o| o.keys().cloned().collect())
            .unwrap_or_default();
        let info = parse_delta(&delta, &keys);

        if info.is_noise {
            // Track stars even from noise (income changes, etc)
            if let (Some(p), Some(s)) = (info.player, info.new_stars) {
                last_known_stars.insert(p, s);
            }
            noise_count += 1;
            continue;
        }

        // Ensure correct player's turn
        if let Some(ap) = info.player {
            let cp = game.current_player_id();
            if ap != cp {
                // Auto end-turn to switch
                let legal = game.legal_moves();
                if let Some(et) = legal.iter().find(|m| m.move_type() == MoveType::EndTurn) {
                    let desc = et.describe(&game.state);
                    let ser = et.serialize();
                    game.play_move(et.as_ref());
                    moves_played.push((delta_row, cp, "EndTurn".into(), desc, ser));
                    if verbose {
                        eprintln!("  [auto] EndTurn P{} → P{} at row {}", cp, ap, delta_row);
                    }
                }
                if game.current_player_id() != ap {
                    if verbose {
                        eprintln!(
                            "  ⚠ Row {}: Still P{} after EndTurn",
                            delta_row,
                            game.current_player_id()
                        );
                    }
                    failed_count += 1;
                    continue;
                }
            } else {
                // If the very first action is by P2 (common in "Bullet" games where P1 skips T0),
                // we've already handled the auto-EndTurn above, giving P1 their passive income.
                // The sync logic below will further correct any production mismatch.
            }
        }

        let cp = game.current_player_id();

        // Sync sim's stars from the PREVIOUS delta's reported value
        // This corrects for drift from research cost discrepancies
        if let Some(&known_stars) = last_known_stars.get(&cp) {
            if let Some(tribe) = game.state.tribes.get_mut(&cp) {
                if tribe.stars != known_stars {
                    if verbose {
                        eprintln!(
                            "    [sync] P{} stars {} → {} (from delta)",
                            cp, tribe.stars, known_stars
                        );
                    }
                    tribe.stars = known_stars;
                }
            }
        }

        let legal = game.legal_moves();

        // Match based on delta type (priority order)
        let matched: Option<usize> = if info.has_tech {
            // Research
            match_research(&legal, &game, &delta, &keys)
        } else if !info.tiles_gained.is_empty() && !info.tiles_vacated.is_empty() {
            // Step (tile gained + tile vacated)
            let step = match_step(&legal, &game, &info.tiles_gained, &info.tiles_vacated);
            if step.is_none() {
                // Could be attack (unit attacks into a tile, original tile vacated)
                match_attack(&legal, &info.tiles_gained)
            } else {
                step
            }
        } else if !info.tiles_gained.is_empty()
            && info.tiles_vacated.is_empty()
            && info.new_stars.is_some()
        {
            // Summon (tile gained, no tile vacated, stars decreased)
            match_summon(
                &legal,
                &game,
                &info.tiles_gained,
                info.new_stars,
                info.unit_type,
            )
        } else if info.has_enemy_damage {
            // Attack result (enemy health changed)
            match_attack(&legal, &info.tiles_gained)
        } else if let Some(tile) = info.resource_removed {
            // Harvest
            match_harvest(&legal, tile)
        } else if let Some(tile) = info.structure_added {
            // Build
            match_build(&legal, tile)
        } else if !info.tiles_gained.is_empty() {
            // Step without vacated tile? Try step first, then summon
            let step = match_step(&legal, &game, &info.tiles_gained, &info.tiles_vacated);
            if step.is_none() {
                match_summon(
                    &legal,
                    &game,
                    &info.tiles_gained,
                    info.new_stars,
                    info.unit_type,
                )
            } else {
                step
            }
        } else {
            None
        };

        if let Some(mi) = matched {
            let mt = format!("{:?}", legal[mi].move_type());
            let desc = legal[mi].describe(&game.state);
            let serialized = legal[mi].serialize();
            if verbose {
                eprintln!("  Row {:3}: {} {}", delta_row, mt, desc);
            }
            game.play_move(legal[mi].as_ref());
            // Track post-move stars from delta (for next sync)
            if let (Some(p), Some(s)) = (info.player, info.new_stars) {
                last_known_stars.insert(p, s);
            }
            moves_played.push((delta_row, cp, mt, desc, serialized));
        } else {
            if verbose {
                eprintln!(
                    "  ❌ Row {:3}: No match | gained={:?} vacated={:?} tech={} dmg={} stars={:?}",
                    delta_row,
                    info.tiles_gained,
                    info.tiles_vacated,
                    info.has_tech,
                    info.has_enemy_damage,
                    info.new_stars
                );
                // Debug: show available moves of the expected type
                let summons: Vec<String> = legal
                    .iter()
                    .filter(|m| m.move_type() == MoveType::Summon)
                    .map(|m| {
                        format!(
                            "  Summon src={} {}",
                            m.source_idx().unwrap_or(999),
                            m.describe(&game.state)
                        )
                    })
                    .collect();
                if !summons.is_empty() {
                    for s in &summons {
                        eprintln!("    AVAIL: {}", s);
                    }
                }
                let curr_stars = game.state.tribes.get(&cp).map(|t| t.stars).unwrap_or(-1);
                eprintln!(
                    "    P{} stars={}, units={}",
                    cp,
                    curr_stars,
                    game.state
                        .tribes
                        .get(&cp)
                        .map(|t| t.units.len())
                        .unwrap_or(0)
                );
            }
            // Still track stars from failed deltas
            if let (Some(p), Some(s)) = (info.player, info.new_stars) {
                last_known_stars.insert(p, s);
            }
            failed_count += 1;
        }
    }

    // ─── Results ─────────────────────────────────────────────────────────────
    println!("\n{}", "═".repeat(60));
    println!(
        "  INFERRED MOVES ({} matched, {} noise, {} failed)",
        moves_played.len(),
        noise_count,
        failed_count
    );
    println!("{}\n", "═".repeat(60));

    let mut current_p = 0;
    let mut turn_num = 0;
    for (i, (row, player, mtype, desc, _ser)) in moves_played.iter().enumerate() {
        if *player != current_p {
            if current_p != 0 {
                println!();
            }
            current_p = *player;
            turn_num += 1;
            println!("─── Turn {} (P{}) ───", turn_num, player);
        }
        println!("  {:3}. [{:10}] {} (row {})", i + 1, mtype, desc, row);
    }

    println!("\n{}", "═".repeat(60));
    println!("  P1 MOVES ONLY");
    println!("{}", "═".repeat(60));
    let mut p1_turn = 0;
    for (_, player, mtype, desc, _ser) in &moves_played {
        if *player != 1 {
            continue;
        }
        if mtype == "EndTurn" {
            p1_turn += 1;
            println!("  --- end turn {} ---", p1_turn);
        } else {
            println!("  {} {}", mtype.to_lowercase(), desc.to_lowercase());
        }
    }

    println!(
        "\nFinal: Turn {}, P{}'s turn, stars P1={} P2={}",
        game.turn(),
        game.current_player_id(),
        game.state.tribes.get(&1).map(|t| t.stars).unwrap_or(-1),
        game.state.tribes.get(&2).map(|t| t.stars).unwrap_or(-1)
    );

    // ─── Save validated moves to JSON ────────────────────────────────────────
    let csv_path = std::path::Path::new(&args[1]);
    let out_path = csv_path.with_extension("moves.json");

    let moves_json: Vec<serde_json::Value> = moves_played
        .iter()
        .map(|(row, player, mtype, desc, ser)| {
            serde_json::json!({
                "row": row,
                "player": player,
                "type": mtype,
                "desc": desc,
                "move": ser,
            })
        })
        .collect();

    let json_str = serde_json::to_string_pretty(&moves_json).unwrap();
    std::fs::write(&out_path, &json_str).expect("Failed to write moves JSON");
    println!(
        "\n✅ Saved {} moves to {}",
        moves_json.len(),
        out_path.display()
    );
}
