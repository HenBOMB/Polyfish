//! Map generation module ported from Python
//!
//! Generates a GameState with a procedural map.

use crate::coords::Coords;
use crate::states::GameSettings;
use crate::states::{GameState, TileState, TribeState};
use crate::types::{ClimateType, ModeType, ResourceType, TerrainType, TribeType};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct MapGenSettings {
    pub size: i32,
    pub land_ratio: f32,
    pub smoothing: i32,
    pub relief: i32,
    pub tribes: Vec<TribeType>,
    pub seed: u64,
}

impl Default for MapGenSettings {
    fn default() -> Self {
        Self {
            size: 16,
            land_ratio: 0.5,
            smoothing: 3,
            relief: 4,
            tribes: vec![TribeType::Imperius, TribeType::Bardur],
            seed: 0,
        }
    }
}

// Intermediate tile representation during generation
#[derive(Clone, Debug)]
struct GenTile {
    idx: i32,
    terrain_type: TerrainType, // 'type' in python
    above: Option<String>,     // 'above' in python (resource/structure/ruin tag)
    road: bool,
    tribe_affinity: Option<TribeType>, // 'tribe' in python (owner affinity)
    // 'otribe' seems to be original tribe affinity?
    orig_tribe_affinity: Option<TribeType>,
}

impl GenTile {
    fn new(idx: i32) -> Self {
        Self {
            idx,
            terrain_type: TerrainType::Ocean,
            above: None,
            road: false,
            tribe_affinity: None,
            orig_tribe_affinity: None,
        }
    }
}

// Utils ported from utils.py

fn get_coords(idx: i32, size: i32) -> (i32, i32) {
    (idx % size, idx / size)
}

fn get_idx(x: i32, y: i32, size: i32) -> i32 {
    y * size + x
}

fn distance(a: i32, b: i32, size: i32) -> i32 {
    let (ax, ay) = get_coords(a, size);
    let (bx, by) = get_coords(b, size);
    (ax - bx).abs().max((ay - by).abs())
}

fn circle(center: i32, radius: i32, size: i32) -> Vec<i32> {
    let mut indices = Vec::new();
    let (row, col) = (center / size, center % size); // Python uses (row, col) -> (y, x) logic usually
                                                     // But `distance` used (x,y). Let's stick to (row, col) as (y, x) for grid traversal logic matching python.

    // Python code:
    // row = center // size
    // column = center % size

    // Top edge: i = row - radius
    let i = row - radius;
    if i >= 0 && i < size {
        for j in (col - radius)..(col + radius) {
            if j >= 0 && j < size {
                indices.push(i * size + j);
            }
        }
    }

    // Bottom edge: i = row + radius
    let i = row + radius;
    if i >= 0 && i < size {
        for j in ((col - radius + 1)..=(col + radius)).rev() {
            // python range(col+radius, col-radius, -1) -> inclusive start, exclusive end?
            // Python: range(col + radius, col - radius, -1).
            // Starts at col+radius (exclusive? No python range is start inclusive, stop exclusive).
            // range(5, 2, -1) -> 5, 4, 3.
            // So starts at rightmost, goes to left.
            // My rust range `(col - radius + 1)..=(col + radius)` reversed is `col+radius` down to `col-radius+1`.
            // Wait. Python: range(start, stop, step).
            // Stop is `col - radius`.
            // So it includes `col - radius + 1`.
            // My logic seems correct.
            if j >= 0 && j < size {
                indices.push(i * size + j);
            }
        }
    }

    // Left edge: j = col - radius
    let j = col - radius;
    if j >= 0 && j < size {
        for i in ((row - radius + 1)..=(row + radius)).rev() {
            // Python: range(row + radius, row - radius, -1)
            if i >= 0 && i < size {
                indices.push(i * size + j);
            }
        }
    }

    // Right edge: j = col + radius
    let j = col + radius;
    if j >= 0 && j < size {
        for i in (row - radius)..(row + radius) {
            if i >= 0 && i < size {
                indices.push(i * size + j);
            }
        }
    }

    indices
}

fn get_round(center: i32, radius: i32, size: i32) -> Vec<i32> {
    let mut indices = Vec::new();
    for r in 1..=radius {
        indices.extend(circle(center, r, size));
    }
    indices.push(center);
    indices
}

fn plus_sign(center: i32, size: i32) -> Vec<i32> {
    let mut indices = Vec::new();
    let (row, col) = (center / size, center % size);

    if col > 0 {
        indices.push(center - 1);
    }
    if col < size - 1 {
        indices.push(center + 1);
    }
    if row > 0 {
        indices.push(center - size);
    }
    if row < size - 1 {
        indices.push(center + size);
    }

    indices
}

// Probability Tables
// We return a multiplier (f32) for a given Tribe and Terrain/Resource type.
// Python `terrain_probs` dictionary.

fn get_terrain_prob(tribe: TribeType, key: &str) -> f32 {
    let x0_0 = 0.0;
    let x0_1 = 0.1;
    let x0_2 = 0.2;
    let x0_4 = 0.4;
    let x0_5 = 0.5;
    let x1_0 = 1.0;
    let x1_2 = 1.2;
    let x1_5 = 1.5;
    let x2_0 = 2.0;

    // Default fallbacks if tribe not list (e.g. specialized tribes treated as Xin-Xi or similar?)
    // python dict has keys for all standard tribes.
    // If we have a new tribe, we'll default to 1.0 or Xin-Xi logic.
    // For now I'll map TribeType to the python logic.

    // Match based on key (resource/terrain)
    match key {
        "water" => match tribe {
            TribeType::Kickoo => x0_4,
            TribeType::Aquarion => x1_5,
            TribeType::Cymanti => x1_0,
            _ => x0_0, // Most are 0.0
        },
        "forest" => match tribe {
            TribeType::Oumaji => x0_2,
            TribeType::Hoodrick => x1_5,
            TribeType::Zebasi | TribeType::Yadakk | TribeType::Aquarion => x0_5,
            _ => x1_0,
        },
        "mountain" => match tribe {
            TribeType::XinXi | TribeType::AiMo => x1_5,
            TribeType::Oumaji
            | TribeType::Kickoo
            | TribeType::Hoodrick
            | TribeType::Zebasi
            | TribeType::Yadakk
            | TribeType::Elyrion => x0_5,
            TribeType::Quetzali
            | TribeType::Imperius
            | TribeType::Bardur
            | TribeType::Luxidoor
            | TribeType::Vengir
            | TribeType::Aquarion
            | TribeType::Cymanti => x1_0,
            _ => x1_0,
        },
        "metal" => match tribe {
            TribeType::XinXi => x1_5,
            TribeType::Vengir => x2_0,
            TribeType::Quetzali => x0_1,
            _ => x1_0,
        },
        "fruit" => match tribe {
            TribeType::Imperius | TribeType::Quetzali => x2_0,
            TribeType::Bardur | TribeType::Yadakk => x1_5,
            TribeType::Vengir => x0_1,
            TribeType::Zebasi => x0_5,
            _ => x1_0,
        },
        "crop" => match tribe {
            TribeType::Bardur | TribeType::AiMo | TribeType::Quetzali => x0_1,
            TribeType::Elyrion => x1_5,
            TribeType::Cymanti => x0_0,
            _ => x1_0,
        },
        "spore" => match tribe {
            TribeType::Cymanti => x1_2,
            _ => x0_0,
        },
        "game" => match tribe {
            TribeType::Imperius => x0_5,
            TribeType::Oumaji => x0_2,
            TribeType::Luxidoor => x1_5,
            TribeType::Vengir => x0_1,
            _ => x1_0,
        },
        "fish" => match tribe {
            TribeType::Kickoo => x1_5,
            TribeType::Vengir => x0_1,
            _ => x1_0,
        },
        _ => x1_0,
    }
}

// General probs (from python code)
fn get_general_prob(key: &str) -> f32 {
    match key {
        "mountain" => 0.02, // Reduced for early training as per python comment
        "forest" => 0.38,
        "fruit" => 0.18,
        "crop" => 0.18,
        "fish" => 0.50,
        "game" => 0.19,
        "starfish" => 0.4,
        "metal" => 0.5,
        _ => 0.0,
    }
}

/// The main generation function
pub fn generate(settings: MapGenSettings) -> GameState {
    let mut rng = StdRng::seed_from_u64(settings.seed);
    let size = settings.size;
    let tile_count = size * size;

    // Initialize map
    let mut map: Vec<GenTile> = (0..tile_count).map(|i| GenTile::new(i as i32)).collect();

    // 1. Initial Land Generation
    // Python: while j < ... initial_land: pick random, turn ocean to ground
    let target_land = (tile_count as f32 * settings.land_ratio) as usize;
    let mut land_count = 0;

    while land_count < target_land {
        let idx = rng.gen_range(0..tile_count) as usize;
        if map[idx].terrain_type == TerrainType::Ocean {
            map[idx].terrain_type = TerrainType::Field; // 'ground' -> Field
            land_count += 1;
        }
    }

    let land_coeff = 1.0; // Hardcoded in python

    // 2. Smoothing
    for _ in 0..settings.smoothing {
        let mut road_flags = vec![false; tile_count as usize];

        for i in 0..tile_count {
            let mut water_count = 0;
            let mut total_count = 0;
            let neighbours = get_round(i, 1, size);
            for &n_idx in &neighbours {
                if map[n_idx as usize].terrain_type == TerrainType::Ocean {
                    water_count += 1;
                }
                total_count += 1;
            }

            if (water_count as f32 / total_count as f32) <= land_coeff {
                road_flags[i as usize] = true; // Temporary use of 'road' flag for swapping
            }
        }

        for i in 0..tile_count {
            if road_flags[i as usize] {
                map[i as usize].terrain_type = TerrainType::Field; // Ground
            } else {
                map[i as usize].terrain_type = TerrainType::Ocean;
            }
        }
    }

    // 3. Capital Placement
    let mut capital_cells: Vec<i32> = Vec::new();
    let min_separation = 3;

    for &tribe in &settings.tribes {
        // Build capital map
        let mut candidates: HashMap<i32, i32> = HashMap::new();

        for row in 2..(size - 2) {
            for col in 2..(size - 2) {
                let idx = row * size + col;
                if map[idx as usize].terrain_type != TerrainType::Field {
                    // Must be ground
                    continue;
                }

                let mut too_close = false;
                for &cap in &capital_cells {
                    if distance(idx, cap, size) < min_separation {
                        too_close = true;
                        break;
                    }
                }

                if !too_close {
                    candidates.insert(idx, size); // Initialize score with max possible val
                }
            }
        }

        // Pick furthest
        let mut max_dist = 0;
        let mut final_scores: HashMap<i32, i32> = HashMap::new();

        for (&cell, &_) in &candidates {
            let mut dist_score = candidates[&cell];
            for &cap in &capital_cells {
                dist_score = dist_score.min(distance(cell, cap, size));
            }
            final_scores.insert(cell, dist_score);
            max_dist = max_dist.max(dist_score);
        }

        let best_cells: Vec<i32> = final_scores
            .iter()
            .filter(|&(_, &d)| d == max_dist)
            .map(|(&c, _)| c)
            .collect();

        if !best_cells.is_empty() {
            let chosen_idx = rng.gen_range(0..best_cells.len());
            let chosen = best_cells[chosen_idx];

            capital_cells.push(chosen);
            map[chosen as usize].above = Some("capital".to_string());
            map[chosen as usize].tribe_affinity = Some(tribe);
            map[chosen as usize].orig_tribe_affinity = Some(tribe);
        }
    }

    // 4. Territory Expansion (Biome determination)
    let mut done_tiles: HashSet<i32> = HashSet::new();
    let mut active_tiles: Vec<Vec<i32>> = Vec::new();

    for &cap in &capital_cells {
        done_tiles.insert(cap);
        active_tiles.push(vec![cap]);
    }

    while done_tiles.len() < tile_count as usize {
        for i in 0..settings.tribes.len() {
            if i >= active_tiles.len() {
                continue;
            } // Should match

            // Skip Polaris logic for now (unimplemented in detailed generation in python usually or special case)
            // Python: `if len(active_tiles[i]) and tribes[i] != 'Polaris':`
            if active_tiles[i].is_empty() {
                continue;
            }
            if settings.tribes[i] == TribeType::Polaris {
                continue;
            } // Simplified

            let rand_idx = rng.gen_range(0..active_tiles[i].len());
            let rand_cell = active_tiles[i][rand_idx];

            let neighbours = circle(rand_cell, 1, size);

            // Valid neighbours: not done, not water (Python: type != 'water' but implies ocean too?)
            // Python: `world_map[tile]['type'] != 'water'`. In python generator 'water' is specific?
            // Actually 'ocean' is the base.
            // Let's assume Ocean prevents expansion initially?
            // "valid_neighbours = list(filter(lambda tile: tile not in done_tiles and world_map[tile]['type'] != 'water', neighbours))"
            // Wait, Python generator uses 'ocean' as default water. 'water' is shallow?
            // In python gen `world_map` initializes `type: 'ocean'`.
            // So if it checks `!= 'water'`, it allows 'ocean'?
            // Maybe `water` means "Deep Ocean"? Or logic is to allow expansion into Ocean?
            // Let's assume expansion into Land first.

            let mut valid_neighbours: Vec<i32> = neighbours
                .iter()
                .cloned()
                .filter(|&n| {
                    !done_tiles.contains(&n)
                        && map[n as usize].terrain_type != TerrainType::Water
                        && map[n as usize].terrain_type != TerrainType::Ocean
                })
                .collect();

            if valid_neighbours.is_empty() {
                // Formatting fallback in python: allow water if no land
                valid_neighbours = neighbours
                    .iter()
                    .cloned()
                    .filter(|&n| !done_tiles.contains(&n))
                    .collect();
            }

            if !valid_neighbours.is_empty() {
                let new_rand_idx = rng.gen_range(0..valid_neighbours.len());
                let new_cell = valid_neighbours[new_rand_idx];

                map[new_cell as usize].tribe_affinity = Some(settings.tribes[i]);
                active_tiles[i].push(new_cell);
                done_tiles.insert(new_cell);
            } else {
                active_tiles[i].swap_remove(rand_idx);
            }
        }

        // Break if stuck (all queues empty)
        if active_tiles.iter().all(|q| q.is_empty()) {
            // Check for remaining tiles not done?
            // If Polaris excluded, we might have holes.
            // Just fill remaining with None/Default affinity.
            for idx in 0..tile_count {
                if !done_tiles.contains(&idx) {
                    done_tiles.insert(idx);
                }
            }
            break;
        }
    }

    // 5. Biome Details (Forest/Mountain/Ocean/Water)
    for cell in 0..tile_count {
        if map[cell as usize].terrain_type == TerrainType::Field
            && map[cell as usize].above.is_none()
        {
            let tribe = map[cell as usize]
                .orig_tribe_affinity
                .or(map[cell as usize].tribe_affinity)
                .unwrap_or(TribeType::Imperius); // Fallback

            let rand_val: f32 = rng.gen();

            // Forest?
            if rand_val < get_general_prob("forest") * get_terrain_prob(tribe, "forest") {
                map[cell as usize].terrain_type = TerrainType::Forest;
            } else if rand_val
                > 1.0 - (get_general_prob("mountain") * get_terrain_prob(tribe, "mountain"))
            {
                map[cell as usize].terrain_type = TerrainType::Mountain;
            }

            // Convert to Ocean?
            let rand_val2: f32 = rng.gen();
            if rand_val2 < get_terrain_prob(tribe, "water") {
                map[cell as usize].terrain_type = TerrainType::Ocean;
            }
        }
    }

    // 6. Village Map (Proximity to capitals)
    let mut village_map = vec![-1; tile_count as usize];
    // Init village map (-1 for forbidden, 0 for allowed)
    for cell in 0..tile_count {
        let (row, col) = get_coords(cell, size);
        let t_type = map[cell as usize].terrain_type;

        if t_type == TerrainType::Ocean || t_type == TerrainType::Mountain {
            village_map[cell as usize] = -1;
        } else if row == 0 || row == size - 1 || col == 0 || col == size - 1 {
            // Edges no village
            village_map[cell as usize] = -1;
        } else {
            village_map[cell as usize] = 0;
        }
    }

    // Apply shallow water (around land)
    // Python: "for cell in ocean: if neighbor in land, become water (shallow)"
    // We iterate all ocean cells.
    // Check neighbors.
    // Note: iterating 0..tile_count. modifying map. need to avoid cascade or use copy?
    // Python uses `world_map[cell]` directly. But order matters.
    // Python logic iterates all cells once.
    let old_map = map.clone(); // Snapshot for neighbor checks
    for cell in 0..tile_count {
        if old_map[cell as usize].terrain_type == TerrainType::Ocean {
            let neighbors = plus_sign(cell, size);
            for &n in &neighbors {
                if matches!(
                    old_map[n as usize].terrain_type,
                    TerrainType::Field | TerrainType::Forest | TerrainType::Mountain
                ) {
                    map[cell as usize].terrain_type = TerrainType::Water;
                    break;
                }
            }
        }
    }

    // Villages
    for &cap in &capital_cells {
        village_map[cap as usize] = 3;
        for n in circle(cap, 1, size) {
            village_map[n as usize] = village_map[n as usize].max(2);
        }
        for n in circle(cap, 2, size) {
            village_map[n as usize] = village_map[n as usize].max(1);
        }
    }

    // Add more villages until full
    while village_map.contains(&0) {
        let candidates: Vec<i32> = village_map
            .iter()
            .enumerate()
            .filter(|(_, &v)| v == 0)
            .map(|(i, _)| i as i32)
            .collect();

        if candidates.is_empty() {
            break;
        }

        let new_village = candidates[rng.gen_range(0..candidates.len())];
        village_map[new_village as usize] = 3;

        map[new_village as usize].above = Some("village".to_string());

        for n in circle(new_village, 1, size) {
            village_map[n as usize] = village_map[n as usize].max(2);
        }
        for n in circle(new_village, 2, size) {
            village_map[n as usize] = village_map[n as usize].max(1);
        }
    }

    // 7. Resources
    let border_expansion = 1.0 / 3.0; // Python: 1/3

    for cell in 0..tile_count {
        let tribe = map[cell as usize]
            .orig_tribe_affinity
            .or(map[cell as usize].tribe_affinity)
            .unwrap_or(TribeType::Imperius);

        let proc = |prob: f32, rand_val: f32| -> bool {
            let vm = village_map[cell as usize];
            if vm == 2 {
                rand_val < prob
            } else if vm == 1 {
                rand_val < prob * border_expansion
            } else {
                false
            }
        };

        match map[cell as usize].terrain_type {
            TerrainType::Field => {
                let fruit_prob = get_general_prob("fruit") * get_terrain_prob(tribe, "fruit");
                let crop_prob = get_general_prob("crop") * get_terrain_prob(tribe, "crop");

                if map[cell as usize]
                    .above
                    .as_ref()
                    .map_or(true, |s| s != "capital" && s != "village")
                {
                    if proc(fruit_prob * (1.0 - crop_prob / 2.0), rng.gen()) {
                        map[cell as usize].above = Some("fruit".to_string());
                    } else if proc(crop_prob * (1.0 - fruit_prob / 2.0), rng.gen()) {
                        map[cell as usize].above = Some("crop".to_string());
                    }
                }
            }
            TerrainType::Forest => {
                if map[cell as usize]
                    .above
                    .as_ref()
                    .map_or(true, |s| s != "capital")
                {
                    if village_map[cell as usize] == 3 {
                        map[cell as usize].terrain_type = TerrainType::Field;
                        map[cell as usize].above = Some("village".to_string());
                    } else {
                        let game_prob = get_general_prob("game") * get_terrain_prob(tribe, "game");
                        if proc(game_prob, rng.gen()) {
                            map[cell as usize].above = Some("game".to_string());
                        }
                    }
                }
            }
            TerrainType::Water => {
                let fish_prob = get_general_prob("fish") * get_terrain_prob(tribe, "fish");
                if proc(fish_prob, rng.gen()) {
                    map[cell as usize].above = Some("fish".to_string());
                }
            }
            TerrainType::Ocean => {
                let star_prob = get_general_prob("starfish") * get_terrain_prob(tribe, "starfish");
                if proc(star_prob, rng.gen()) {
                    map[cell as usize].above = Some("starfish".to_string());
                }
            }
            TerrainType::Mountain => {
                let metal_prob = get_general_prob("metal") * get_terrain_prob(tribe, "metal");
                if proc(metal_prob, rng.gen()) {
                    map[cell as usize].above = Some("metal".to_string());
                }
            }
            _ => {}
        }
    }

    // 8. Ruins
    let ruins_number = ((size * size) as f32 / 40.0).round() as i32;
    let water_ruins_number = (ruins_number as f32 / 3.0).round() as i32;
    let mut ruins_count = 0;
    let mut water_ruins_count = 0;

    let mut attempts = 0;
    while ruins_count < ruins_number && attempts < 2000 {
        attempts += 1;

        let candidates: Vec<i32> = village_map
            .iter()
            .enumerate()
            .filter(|(_, &v)| v == -1 || v == 0 || v == 1)
            .map(|(i, _)| i as i32)
            .collect();

        if candidates.is_empty() {
            break;
        }

        let ruin_idx = candidates[rng.gen_range(0..candidates.len())];
        let t_type = map[ruin_idx as usize].terrain_type;

        if t_type != TerrainType::Water
            && (water_ruins_count < water_ruins_number || t_type != TerrainType::Ocean)
        {
            map[ruin_idx as usize].above = Some("ruin".to_string());
            if t_type == TerrainType::Ocean {
                water_ruins_count += 1;
            }
            village_map[ruin_idx as usize] = village_map[ruin_idx as usize].max(2);
            for n in circle(ruin_idx, 1, size) {
                village_map[n as usize] = village_map[n as usize].max(2);
            }
            ruins_count += 1;
        }
    }

    // 9. Post Generate (Guaranteed resources)
    for &capital in &capital_cells {
        let tribe = map[capital as usize]
            .tribe_affinity
            .unwrap_or(TribeType::Imperius);

        let (resource, underneath, quantity) = match tribe {
            TribeType::Imperius => ("fruit", TerrainType::Field, 2),
            TribeType::Bardur => ("game", TerrainType::Forest, 2),
            TribeType::Zebasi => ("crop", TerrainType::Field, 1),
            TribeType::Elyrion => ("game", TerrainType::Forest, 2),
            _ => ("", TerrainType::Field, 0),
        };

        if !resource.is_empty() {
            let mut current_qty = 0;
            for n in circle(capital, 1, size) {
                if map[n as usize].above.as_deref() == Some(resource) {
                    current_qty += 1;
                }
            }

            let circle1 = circle(capital, 1, size);
            if !circle1.is_empty() {
                let mut attempts = 0;
                while current_qty < quantity && attempts < 100 {
                    attempts += 1;
                    let idx = circle1[rng.gen_range(0..circle1.len())] as usize;

                    map[idx].terrain_type = underneath;
                    map[idx].above = Some(resource.to_string());

                    for n in plus_sign(idx as i32, size) {
                        if map[n as usize].terrain_type == TerrainType::Ocean {
                            map[n as usize].terrain_type = TerrainType::Water;
                        }
                    }

                    current_qty = 0;
                    for n in &circle1 {
                        if map[*n as usize].above.as_deref() == Some(resource) {
                            current_qty += 1;
                        }
                    }
                }
            }
        } else if tribe == TribeType::Kickoo {
            let quantity = 2;
            let mut current_qty = 0;
            for n in circle(capital, 1, size) {
                if map[n as usize].above.as_deref() == Some("fish") {
                    current_qty += 1;
                }
            }

            let plus_neighbors = plus_sign(capital, size);
            if !plus_neighbors.is_empty() {
                let mut attempts = 0;
                while current_qty < quantity && attempts < 100 {
                    attempts += 1;
                    let idx = plus_neighbors[rng.gen_range(0..plus_neighbors.len())] as usize;
                    map[idx].terrain_type = TerrainType::Water;
                    map[idx].above = Some("fish".to_string());

                    current_qty = 0;
                    for n in circle(capital, 1, size) {
                        if map[n as usize].above.as_deref() == Some("fish") {
                            current_qty += 1;
                        }
                    }
                    if current_qty >= quantity {
                        break;
                    }
                }
            }
        }

        if tribe == TribeType::Polaris {
            for n in circle(capital, 1, size) {
                map[n as usize].tribe_affinity = Some(TribeType::Polaris);
            }
        }
    }

    // Convert to GameState
    let mut game_state = GameState::default();
    game_state.settings.size = size;
    game_state.settings.tile_count = tile_count;

    let mut tribe_id_map: HashMap<TribeType, i32> = HashMap::new();
    for (i, &tribe) in settings.tribes.iter().enumerate() {
        let id = (i + 1) as i32;
        let mut t_state = TribeState::default();
        t_state.id = id;
        t_state.tribe_type = tribe;
        t_state.score = 0;
        t_state.stars = 5;
        tribe_id_map.insert(tribe, id);
        game_state.tribes.insert(id, t_state);
    }

    for gen_tile in map {
        let mut t_state = TileState::default();
        let (cx, cy) = get_coords(gen_tile.idx, size);
        t_state.coords = Coords {
            x: cx,
            y: cy,
            idx: gen_tile.idx,
        };
        t_state.terrain_type = gen_tile.terrain_type;

        if let Some(s) = gen_tile.above {
            match s.as_str() {
                "village" => {
                    use crate::states::StructureState;
                    use crate::types::StructureType;
                    let mut s_state = StructureState::default();
                    s_state.structure_type = StructureType::Village;
                    s_state.tile_index = gen_tile.idx;
                    game_state.structures.insert(gen_tile.idx, Some(s_state));
                }
                "capital" => {
                    use crate::states::CityState;
                    let mut city = CityState::default();
                    city.tile_index = gen_tile.idx;
                    city.level = 1;
                    city.population = 0;
                    city.border_size = 1;

                    if let Some(tribe) = gen_tile.tribe_affinity {
                        if let Some(&pid) = tribe_id_map.get(&tribe) {
                            city.owner = pid;
                            t_state.owner = pid;
                            t_state.capital_of = pid;

                            if let Some(t) = game_state.tribes.get_mut(&pid) {
                                t.cities.push(city.clone());
                                t.starting_tile_coords = t_state.coords;
                            }
                        }
                    }
                }
                "ruin" => {
                    use crate::states::StructureState;
                    use crate::types::StructureType;
                    let mut s_state = StructureState::default();
                    s_state.structure_type = StructureType::Ruin;
                    s_state.tile_index = gen_tile.idx;
                    game_state.structures.insert(gen_tile.idx, Some(s_state));
                }
                "fruit" => {
                    use crate::states::ResourceState;
                    use crate::types::ResourceType;
                    let mut r_state = ResourceState::default();
                    r_state.resource_type = ResourceType::Fruit;
                    r_state.tile_index = gen_tile.idx;
                    game_state.resources.insert(gen_tile.idx, Some(r_state));
                }
                "crop" => {
                    use crate::states::ResourceState;
                    use crate::types::ResourceType;
                    let mut r_state = ResourceState::default();
                    r_state.resource_type = ResourceType::Crop;
                    r_state.tile_index = gen_tile.idx;
                    game_state.resources.insert(gen_tile.idx, Some(r_state));
                }
                "game" => {
                    use crate::states::ResourceState;
                    use crate::types::ResourceType;
                    let mut r_state = ResourceState::default();
                    r_state.resource_type = ResourceType::Game;
                    r_state.tile_index = gen_tile.idx;
                    game_state.resources.insert(gen_tile.idx, Some(r_state));
                }
                "fish" => {
                    use crate::states::ResourceState;
                    use crate::types::ResourceType;
                    let mut r_state = ResourceState::default();
                    r_state.resource_type = ResourceType::Fish;
                    r_state.tile_index = gen_tile.idx;
                    game_state.resources.insert(gen_tile.idx, Some(r_state));
                }
                "metal" => {
                    use crate::states::ResourceState;
                    use crate::types::ResourceType;
                    let mut r_state = ResourceState::default();
                    r_state.resource_type = ResourceType::Metal;
                    r_state.tile_index = gen_tile.idx;
                    game_state.resources.insert(gen_tile.idx, Some(r_state));
                }
                "starfish" => {
                    use crate::states::ResourceState;
                    use crate::types::ResourceType;
                    let mut r_state = ResourceState::default();
                    r_state.resource_type = ResourceType::Starfish;
                    r_state.tile_index = gen_tile.idx;
                    game_state.resources.insert(gen_tile.idx, Some(r_state));
                }
                "spore" => {
                    use crate::states::ResourceState;
                    use crate::types::ResourceType;
                    let mut r_state = ResourceState::default();
                    r_state.resource_type = ResourceType::Spores;
                    r_state.tile_index = gen_tile.idx;
                    game_state.resources.insert(gen_tile.idx, Some(r_state));
                }
                _ => {}
            }
        }

        game_state.tiles.insert(gen_tile.idx, t_state);
    }

    // Territory and Ruling City Coords
    for tribe in game_state.tribes.values_mut() {
        for city in &mut tribe.cities {
            city._territory = circle(city.tile_index, city.border_size, size);
        }
    }

    let mut territory_updates: Vec<(i32, i32)> = Vec::new(); // (tile_idx, owner_id)
    for tribe in game_state.tribes.values() {
        for city in &tribe.cities {
            for &idx in &city._territory {
                territory_updates.push((idx, tribe.id));
            }
        }
    }

    for (idx, owner) in territory_updates {
        if let Some(t) = game_state.tiles.get_mut(&idx) {
            t.owner = owner;
        }
    }

    for tribe in game_state.tribes.values() {
        for city in &tribe.cities {
            let city_coords = get_coords(city.tile_index, size);
            let city_coords_obj = Coords {
                x: city_coords.0,
                y: city_coords.1,
                idx: city.tile_index,
            };
            for &idx in &city._territory {
                if let Some(t) = game_state.tiles.get_mut(&idx) {
                    t.ruling_city_coords = Some(city_coords_obj);
                }
            }
        }
    }

    game_state
}
