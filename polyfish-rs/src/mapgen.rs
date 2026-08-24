//! Map generation module ported from Python
//!
//! Generates a GameState with a procedural map.

use crate::coords::Coords;
use crate::default_fow;
use crate::functions::{
    get_chebyshev_distance as distance, get_plus_sign_indices as plus_sign,
    get_square_indices as get_square, get_squared_euclidean_distance, idx_to_coords as get_coords,
};
use crate::states::{GameState, TileState, TribeState};
use crate::types::{MapSize, MapType, TerrainType, TribeType};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashSet;

use crate::types::classic_climate_id;

#[derive(Debug, Clone)]
pub struct MapGenSettings {
    pub size: MapSize,
    pub map_type: MapType,
    pub tribes: Vec<TribeType>,
    pub seed: i64,
    pub version: i32,
}

impl Default for MapGenSettings {
    fn default() -> Self {
        Self {
            size: MapSize::Normal,
            map_type: MapType::Continents,
            tribes: vec![TribeType::Imperius, TribeType::Bardur],
            seed: 0,
            version: 115,
        }
    }
}

/// Map types that generate no water at all. Everything water-bound — naval
/// units, ports, fish, the water tech lane — is dead here, so generation must
/// not sneak a tile in through a tribe's starting resources either.
pub fn is_fully_dry(map_type: MapType) -> bool {
    matches!(map_type, MapType::Drylands)
}

// Intermediate tile representation during generation
#[derive(Clone, Debug)]
struct GenTile {
    idx: i32,
    terrain_type: TerrainType,         // 'type' in python
    above: Option<String>,             // 'above' in python (resource/structure/ruin tag)
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
            tribe_affinity: None,
            orig_tribe_affinity: None,
        }
    }
}

// BiomeRates logic moved below or kept here if it doesn't use the deleted utils

#[derive(Debug, Clone, Copy)]
pub struct BiomeRates {
    pub mountain: f32,
    pub forest: f32,
    pub field: f32,
}

pub fn get_tribe_biome_rates(tribe: TribeType) -> BiomeRates {
    let mut rates = BiomeRates {
        mountain: 0.14,
        forest: 0.38,
        field: 0.48,
    };

    let m_mult = match tribe {
        TribeType::XinXi | TribeType::AiMo => 1.5,
        TribeType::Oumaji
        | TribeType::Kickoo
        | TribeType::Zebasi
        | TribeType::Hoodrick
        | TribeType::Yadakk
        | TribeType::Elyrion => 0.5,
        TribeType::Cymanti => 1.2,
        _ => 1.0,
    };

    if m_mult != 1.0 {
        let old_m = rates.mountain;
        rates.mountain *= m_mult;
        let diff = rates.mountain - old_m;
        let non_m_total = rates.forest + rates.field;
        if non_m_total > 0.0 {
            rates.forest -= diff * (rates.forest / non_m_total);
            rates.field -= diff * (rates.field / non_m_total);
        }
    }

    let f_mult = match tribe {
        TribeType::Hoodrick => 1.5,
        TribeType::Bardur => 0.8,
        TribeType::Oumaji => 0.2,
        TribeType::Zebasi | TribeType::Yadakk | TribeType::Aquarion => 0.5,
        _ => 1.0,
    };

    if f_mult != 1.0 {
        let old_f = rates.forest;
        rates.forest *= f_mult;
        let diff = rates.forest - old_f;
        rates.field -= diff;
    }

    rates.mountain = rates.mountain.clamp(0.0, 1.0);
    rates.forest = rates.forest.clamp(0.0, 1.0);
    rates.field = rates.field.clamp(0.0, 1.0);

    rates
}

/// P(resource | eligible terrain tile) inside a village's spawn zone.
/// Bases are the real game's conditionals (mapgen_research.md): the wiki's
/// land-tile fractions divided by the terrain share — NOT the fractions
/// themselves, which an earlier port used directly (~7× too little metal).
/// Outer ring (distance 2) = inner × 1/3, the game's border-expansion factor.
pub fn get_resource_prob(key: &str, tribe: TribeType, inner: bool) -> f32 {
    let base = match key {
        "fruit" | "crop" | "spores" => 0.375,
        "game" => 0.5,
        // Between the 0.85 Moonrise patch constant and the modern 11/14 table
        "metal" => 0.8,
        "fish" => 0.5,
        _ => 0.0,
    };

    let mult = match (key, tribe) {
        // Metal modifiers
        ("metal", TribeType::XinXi) => 1.5,
        ("metal", TribeType::Vengir) => 2.0,
        // Fruit modifiers
        ("fruit", TribeType::Imperius) => 2.0,
        ("fruit", TribeType::Vengir) => 0.1,
        ("fruit", TribeType::Zebasi) => 0.5,
        ("fruit", TribeType::Quetzali) => 2.0,
        ("fruit", TribeType::Yadakk) => 1.5,
        // Game modifiers
        ("game", TribeType::Imperius) => 0.5,
        ("game", TribeType::Oumaji) => 0.2,
        ("game", TribeType::Vengir) => 0.1,
        // Crop modifiers
        ("crop", TribeType::Bardur) => 0.0,
        ("crop", TribeType::AiMo) => 0.1,
        ("crop", TribeType::Quetzali) => 0.1,
        ("crop", TribeType::Elyrion) => 1.5,
        ("crop", TribeType::Cymanti) => 0.0,
        // Fish modifiers
        ("fish", TribeType::Kickoo) => 1.5,
        ("fish", TribeType::Vengir) => 0.1,
        _ => 1.0,
    };

    if inner { base * mult } else { base * mult / 3.0 }
}

/// The main generation function
pub fn generate(settings: MapGenSettings) -> GameState {
    let mut rng = StdRng::seed_from_u64(settings.seed as u64);
    let size = settings.size.get_size();
    let tile_count = size * size;

    // Initialize map
    let mut map: Vec<GenTile> = (0..tile_count).map(|i| GenTile::new(i as i32)).collect();
    let mut is_land = vec![false; tile_count as usize];

    // 1. Capital Placement
    let player_count = settings.tribes.len();
    let mut capital_cells: Vec<i32> = Vec::new();

    let use_quadrants = matches!(
        settings.map_type,
        MapType::Drylands | MapType::Lakes | MapType::Archipelago | MapType::WaterWorld
    );

    if use_quadrants {
        let quad_count = if player_count <= 4 {
            4
        } else if player_count <= 9 {
            9
        } else {
            16
        };
        let quads_per_side = (quad_count as f32).sqrt() as i32;
        let quad_size = size / quads_per_side;

        let mut available_quads: Vec<i32> = (0..quad_count).collect();

        // --- FIX 1: Smart Quadrant Selection ---
        for _ in 0..settings.tribes.len() {
            if available_quads.is_empty() {
                break;
            }

            let q_idx = if capital_cells.is_empty() {
                // First player picks randomly
                rng.gen_range(0..available_quads.len())
            } else {
                // Subsequent players pick a quadrant that is reasonably far from existing capitals.
                // We calculate the center of the available quadrants and compare to existing capitals.
                let mut quads_with_dist = Vec::new();
                let mut max_min_dist = -1;

                for (idx, &quad) in available_quads.iter().enumerate() {
                    let qx = quad % quads_per_side;
                    let qy = quad / quads_per_side;
                    let center_x = qx * quad_size + (quad_size / 2);
                    let center_y = qy * quad_size + (quad_size / 2);
                    let center_idx = center_y * size + center_x;

                    let mut min_dist_to_capitals = i32::MAX;
                    for &cap in &capital_cells {
                        min_dist_to_capitals = min_dist_to_capitals
                            .min(get_squared_euclidean_distance(center_idx, cap, size));
                    }
                    if min_dist_to_capitals > max_min_dist {
                        max_min_dist = min_dist_to_capitals;
                    }
                    quads_with_dist.push((idx, min_dist_to_capitals));
                }

                // Keep quads that are at least 50% of the maximum minimum distance found.
                // In a 2x2 grid, this allows adjacent quadrants (dist 1) as well as opposite (dist 2).
                let threshold = (max_min_dist as f32 * 0.5) as i32;
                let candidates: Vec<usize> = quads_with_dist
                    .into_iter()
                    .filter(|&(_, dist)| dist >= threshold)
                    .map(|(idx, _)| idx)
                    .collect();

                candidates[rng.gen_range(0..candidates.len())]
            };

            let quad = available_quads.remove(q_idx);

            let qx = quad % quads_per_side;
            let qy = quad / quads_per_side;

            let margin = 2;
            let start_x = (qx * quad_size + margin).min(size - 3);
            let end_x = ((qx + 1) * quad_size - margin)
                .max(start_x + 1)
                .min(size - 2);
            let start_y = (qy * quad_size + margin).min(size - 3);
            let end_y = ((qy + 1) * quad_size - margin)
                .max(start_y + 1)
                .min(size - 2);

            let cx = rng.gen_range(start_x..end_x);
            let cy = rng.gen_range(start_y..end_y);
            let chosen = cy * size + cx;

            capital_cells.push(chosen);
            // Assign affinity later when iterating tribes to match index
        }

        // Assign affinities now that positions are chosen
        for (i, &cap) in capital_cells.iter().enumerate() {
            let tribe = settings.tribes[i];
            map[cap as usize].above = Some("capital".to_string());
            map[cap as usize].tribe_affinity = Some(tribe);
            map[cap as usize].orig_tribe_affinity = Some(tribe);
            map[cap as usize].terrain_type = TerrainType::Field;
            is_land[cap as usize] = true;
        }
    }

    // 2. Village Spawning (Pre-terrain / Suburbs)
    let mut village_map = vec![0; tile_count as usize];
    for &cap in &capital_cells {
        village_map[cap as usize] = 2;
    }

    if settings.map_type == MapType::Lakes || settings.map_type == MapType::Archipelago {
        // Suburbs (1-2 per capital, within radius 3, distance >= 3)
        for &cap in &capital_cells {
            let mut sub_count = rng.gen_range(1..=2);
            let mut candidates: Vec<i32> = get_square(cap, 3, size)
                .into_iter()
                .filter(|&idx| {
                    village_map[idx as usize] == 0 && distance(idx, cap, size) >= 3 && {
                        let (x, y) = get_coords(idx, size);
                        x > 0 && x < size - 1 && y > 0 && y < size - 1 // At least 1 tile from edge
                    }
                })
                .collect();

            while sub_count > 0 && !candidates.is_empty() {
                let idx = candidates.remove(rng.gen_range(0..candidates.len()));
                village_map[idx as usize] = 1;
                map[idx as usize].above = Some("village".to_string());
                map[idx as usize].terrain_type = TerrainType::Field;
                is_land[idx as usize] = true;
                sub_count -= 1;
                candidates.retain(|&c| distance(c, idx, size) >= 3);
            }
        }
    }

    if settings.map_type == MapType::Lakes
        || settings.map_type == MapType::Archipelago
        || settings.map_type == MapType::WaterWorld
    {
        // Pre-terrain villages
        let cap_sub_count = village_map.iter().filter(|&&v| v > 0).count() as f32;
        let density = if settings.map_type == MapType::WaterWorld {
            0.1
        } else {
            0.3
        };
        let pre_terrain_count =
            (((size as f32 / 3.0).floor().powi(2) - cap_sub_count) * density) as i32;
        let mut all_candidates: Vec<i32> = (0..tile_count)
            .filter(|&idx| {
                let (x, y) = get_coords(idx, size);
                village_map[idx as usize] == 0
                    && x > 0
                    && x < size - 1
                    && y > 0
                    && y < size - 1 // At least 1 tile from edge
                    && village_map
                        .iter()
                        .enumerate()
                        .filter(|&(_, &v)| v > 0)
                        .all(|(v_idx, _)| distance(idx, v_idx as i32, size) >= 3)
            })
            .collect();

        let mut placed = 0;
        while placed < pre_terrain_count && !all_candidates.is_empty() {
            let idx = all_candidates.remove(rng.gen_range(0..all_candidates.len()));
            village_map[idx as usize] = 1;
            map[idx as usize].above = Some("village".to_string());
            map[idx as usize].terrain_type = TerrainType::Field;
            placed += 1;
            all_candidates.retain(|&c| distance(c, idx, size) >= 3);
        }
    }

    // 3. Terrain Generation
    let land_ratio = match settings.map_type {
        MapType::None => 0.5,
        MapType::Drylands => 1.0,
        MapType::Lakes => 0.72,
        MapType::Continents => 0.45,
        MapType::Pangea => 0.50,
        MapType::Archipelago => 0.30,
        MapType::WaterWorld => 0.05,
    };
    for i in 0..tile_count {
        if village_map[i as usize] > 0 {
            is_land[i as usize] = true;
        }
    }

    let target_land = (tile_count as f32 * land_ratio) as usize;
    let mut current_land = is_land.iter().filter(|&&l| l).count();

    if settings.map_type == MapType::Pangea {
        // Flood-fill growth from center
        let center = (size / 2) * size + (size / 2);
        is_land[center as usize] = true;
        current_land += 1;

        let mut frontier: Vec<i32> = vec![center];
        while current_land < target_land && !frontier.is_empty() {
            let idx = frontier.remove(rng.gen_range(0..frontier.len()));
            for n in plus_sign(idx, size) {
                if !is_land[n as usize] && current_land < target_land {
                    // Probability decreases with distance from center
                    let (nx, ny) = get_coords(n, size);
                    let dist_from_center = ((nx - size / 2).abs() + (ny - size / 2).abs()) as f32;
                    let prob = 1.0 - (dist_from_center / size as f32).min(0.9);
                    if rng.r#gen::<f32>() < prob {
                        is_land[n as usize] = true;
                        current_land += 1;
                        frontier.push(n);
                    }
                }
            }
        }
    } else if settings.map_type == MapType::Continents {
        // Discrete continents generation
        let continent_count = match player_count {
            1..=2 => 2,
            3..=4 => 3,
            _ => 4,
        };
        let min_continent_size = 30.max(target_land / (continent_count * 2));
        let max_continent_size = 200.min(target_land / continent_count + 50);

        let mut seeds: Vec<i32> = Vec::new();
        for _ in 0..continent_count {
            // Find a seed position far from existing continents
            for _ in 0..100 {
                let candidate = rng.gen_range(0..tile_count);
                let (cx, cy) = get_coords(candidate, size);
                // Keep away from edges
                if cx < 2 || cx >= size - 2 || cy < 2 || cy >= size - 2 {
                    continue;
                }
                // Keep at least 6 tiles from other continent seeds
                let far_enough = seeds.iter().all(|&s| distance(candidate, s, size) >= 6);
                if far_enough && !is_land[candidate as usize] {
                    seeds.push(candidate);
                    break;
                }
            }
        }

        // Grow each continent
        for seed in seeds {
            let continent_size = rng.gen_range(min_continent_size..=max_continent_size);
            let mut frontier = vec![seed];
            let mut grown = 0;
            is_land[seed as usize] = true;
            current_land += 1;
            grown += 1;

            while grown < continent_size && !frontier.is_empty() && current_land < target_land {
                let idx = frontier.remove(rng.gen_range(0..frontier.len()));
                for n in plus_sign(idx, size) {
                    if !is_land[n as usize] && grown < continent_size && current_land < target_land
                    {
                        if rng.r#gen::<f32>() < 0.7 {
                            is_land[n as usize] = true;
                            current_land += 1;
                            grown += 1;
                            frontier.push(n);
                        }
                    }
                }
            }
        }
    } else if is_fully_dry(settings.map_type) {
        // Fully-dry map: every tile is land, so no water can survive the
        // shallowing pass below (Aug 2026 — stray puddles made the water tech
        // lane nominally legal on a map where it buys nothing).
        is_land.iter_mut().for_each(|l| *l = true);
    } else {
        // Generic random scatter for other map types
        while current_land < target_land {
            let idx = rng.gen_range(0..tile_count) as usize;
            if !is_land[idx] {
                is_land[idx] = true;
                current_land += 1;
            }
        }
    }

    // Smoothing pass (except Drylands)
    if settings.map_type != MapType::Drylands {
        for _ in 0..3 {
            let mut next_land = is_land.clone();
            for i in 0..tile_count {
                if village_map[i as usize] > 0 {
                    continue;
                }
                let land_neighbors = get_square(i, 1, size)
                    .iter()
                    .filter(|&&n| is_land[n as usize])
                    .count();
                if land_neighbors >= 5 {
                    next_land[i as usize] = true;
                } else if land_neighbors <= 3 {
                    next_land[i as usize] = false;
                }
            }
            is_land = next_land;
        }
    }

    for i in 0..tile_count {
        map[i as usize].terrain_type = if is_land[i as usize] {
            TerrainType::Field
        } else {
            TerrainType::Ocean
        };
    }

    if !use_quadrants {
        if settings.map_type == MapType::Continents {
            // Continents: Identify landmasses and place villages
            // First, identify all distinct landmasses using flood-fill
            let mut landmass_id = vec![-1i32; tile_count as usize];
            let mut current_landmass = 0;

            for start_idx in 0..tile_count {
                if !is_land[start_idx as usize] || landmass_id[start_idx as usize] != -1 {
                    continue;
                }

                // Flood-fill to mark this landmass
                let mut queue = vec![start_idx];
                landmass_id[start_idx as usize] = current_landmass;

                while let Some(idx) = queue.pop() {
                    for n in plus_sign(idx, size) {
                        if is_land[n as usize] && landmass_id[n as usize] == -1 {
                            landmass_id[n as usize] = current_landmass;
                            queue.push(n);
                        }
                    }
                }

                current_landmass += 1;
            }

            let num_landmasses = current_landmass;

            // Place one village per landmass first
            for landmass in 0..num_landmasses {
                let candidates: Vec<i32> = (0..tile_count)
                    .filter(|&i| {
                        landmass_id[i as usize] == landmass
                            && village_map[i as usize] == 0
                            && map[i as usize].terrain_type != TerrainType::Mountain
                            && {
                                let (x, y) = get_coords(i, size);
                                x > 1 && x < size - 2 && y > 1 && y < size - 2
                            }
                            && village_map
                                .iter()
                                .enumerate()
                                .all(|(v_idx, &v)| v == 0 || distance(i, v_idx as i32, size) >= 4)
                    })
                    .collect();

                if let Some(&idx) = candidates.get(rng.gen_range(0..candidates.len().max(1))) {
                    village_map[idx as usize] = 1;
                    if map[idx as usize].terrain_type == TerrainType::Forest {
                        map[idx as usize].terrain_type = TerrainType::Field;
                    }
                    map[idx as usize].above = Some("village".to_string());
                }
            }

            // Then place additional villages randomly (fill phase)
            loop {
                let candidates: Vec<i32> = (0..tile_count)
                    .filter(|&i| {
                        let (x, y) = get_coords(i, size);
                        let dist_x = x.min(size - 1 - x);
                        let dist_y = y.min(size - 1 - y);
                        let edge_dist = dist_x.min(dist_y);

                        is_land[i as usize]
                            && village_map[i as usize] == 0
                            && map[i as usize].terrain_type != TerrainType::Mountain
                            && edge_dist >= 2     // Not within two tiles
                            && edge_dist != 3     // Not three tiles from edge
                            && village_map
                                .iter()
                                .enumerate()
                                .all(|(v_idx, &v)| v == 0 || distance(i, v_idx as i32, size) >= 3)
                    })
                    .collect();

                if candidates.is_empty() {
                    break;
                }

                let idx = candidates[rng.gen_range(0..candidates.len())];
                village_map[idx as usize] = 1;
                if map[idx as usize].terrain_type == TerrainType::Forest {
                    map[idx as usize].terrain_type = TerrainType::Field;
                }
                map[idx as usize].above = Some("village".to_string());
            }

            // Convert villages to capitals (prefer different landmasses, maximize distance, prefer coastal)
            let available_villages: Vec<i32> = (0..tile_count)
                .filter(|&i| village_map[i as usize] == 1)
                .collect();

            let mut used_landmasses: HashSet<i32> = HashSet::new();
            let mut scored_villages: Vec<(i32, i32)> = available_villages
                .iter()
                .map(|&v| {
                    let coastal = plus_sign(v, size).iter().any(|&n| !is_land[n as usize]);
                    let mut dist_score = 100;
                    for &cap in &capital_cells {
                        dist_score = dist_score.min(distance(v, cap, size));
                    }
                    let landmass_bonus = if used_landmasses.contains(&landmass_id[v as usize]) {
                        -20 // Penalty for already used landmass
                    } else {
                        20 // Bonus for new landmass
                    };
                    let coastal_bonus = if coastal { 5 } else { 0 };

                    let mut score = dist_score + coastal_bonus + landmass_bonus;

                    // Strong penalty for being too close in 1v1
                    if settings.tribes.len() == 2 && dist_score < size / 3 {
                        score -= 50;
                    }

                    (v, score)
                })
                .collect();

            for &tribe in &settings.tribes {
                if scored_villages.is_empty() {
                    break;
                }

                // Find max score
                let mut best_idx = 0;
                let mut max_score = i32::MIN;

                for (idx, &(_, score)) in scored_villages.iter().enumerate() {
                    if score > max_score {
                        max_score = score;
                        best_idx = idx;
                    }
                }

                let (best_v, _) = scored_villages.remove(best_idx);

                used_landmasses.insert(landmass_id[best_v as usize]);
                capital_cells.push(best_v);
                village_map[best_v as usize] = 2;
                map[best_v as usize].above = Some("capital".to_string());
                map[best_v as usize].tribe_affinity = Some(tribe);
                map[best_v as usize].orig_tribe_affinity = Some(tribe);

                // Update scores for remaining
                for (v, score) in &mut scored_villages {
                    let coastal_bonus = if plus_sign(*v, size).iter().any(|&n| !is_land[n as usize])
                    {
                        5
                    } else {
                        0
                    };
                    let landmass_bonus = if used_landmasses.contains(&landmass_id[*v as usize]) {
                        -20
                    } else {
                        20
                    };
                    let old_dist = *score - coastal_bonus - landmass_bonus;
                    // Restore potential distance penalty
                    let old_dist = if settings.tribes.len() == 2 && old_dist < -20 {
                        old_dist + 50
                    } else {
                        old_dist
                    };

                    let new_dist = distance(*v, best_v, size);
                    let new_min_dist = old_dist.min(new_dist);

                    let mut new_score = new_min_dist + coastal_bonus + landmass_bonus;
                    if settings.tribes.len() == 2 && new_min_dist < size / 3 {
                        new_score -= 50;
                    }
                    *score = new_score;
                }
            }
        } else {
            // Pangea: Place villages on land (fill phase)
            loop {
                let candidates: Vec<i32> = (0..tile_count)
                    .filter(|&i| {
                        let (x, y) = get_coords(i, size);
                        let dist_x = x.min(size - 1 - x);
                        let dist_y = y.min(size - 1 - y);
                        let edge_dist = dist_x.min(dist_y);

                        is_land[i as usize]
                            && village_map[i as usize] == 0
                            && map[i as usize].terrain_type != TerrainType::Mountain
                            && edge_dist >= 2     // Not within two tiles
                            && edge_dist != 3     // Not three tiles from edge
                            && village_map
                                .iter()
                                .enumerate()
                                .all(|(v_idx, &v)| v == 0 || distance(i, v_idx as i32, size) >= 3)
                    })
                    .collect();

                if candidates.is_empty() {
                    break;
                }

                let idx = candidates[rng.gen_range(0..candidates.len())];
                village_map[idx as usize] = 1;
                if map[idx as usize].terrain_type == TerrainType::Forest {
                    map[idx as usize].terrain_type = TerrainType::Field;
                }
                map[idx as usize].above = Some("village".to_string());
            }

            // Convert some villages to capitals (maximize distance, prefer coastal)
            let available_villages: Vec<i32> = (0..tile_count)
                .filter(|&i| village_map[i as usize] == 1)
                .collect();

            let mut scored_villages: Vec<(i32, i32)> = available_villages
                .iter()
                .map(|&v| {
                    let coastal = plus_sign(v, size).iter().any(|&n| !is_land[n as usize]);
                    let mut dist_score = 100;
                    for &cap in &capital_cells {
                        dist_score = dist_score.min(distance(v, cap, size));
                    }
                    let coastal_bonus = if coastal { 5 } else { 0 };
                    let mut score = dist_score + coastal_bonus;

                    // Strong penalty for being too close in 1v1
                    if settings.tribes.len() == 2 && dist_score < size / 3 {
                        score -= 50;
                    }

                    (v, score)
                })
                .collect();

            for &tribe in &settings.tribes {
                if scored_villages.is_empty() {
                    break;
                }

                // Find max score
                let mut best_idx = 0;
                let mut max_score = -1;
                for (idx, &(_, score)) in scored_villages.iter().enumerate() {
                    if score > max_score {
                        max_score = score;
                        best_idx = idx;
                    }
                }

                let (best_v, _) = scored_villages.remove(best_idx);

                capital_cells.push(best_v);
                village_map[best_v as usize] = 2;
                map[best_v as usize].above = Some("capital".to_string());
                map[best_v as usize].tribe_affinity = Some(tribe);
                map[best_v as usize].orig_tribe_affinity = Some(tribe);

                // Update scores for remaining
                for (v, score) in &mut scored_villages {
                    let coastal_bonus = if plus_sign(*v, size).iter().any(|&n| !is_land[n as usize])
                    {
                        5
                    } else {
                        0
                    };
                    let old_dist = *score - coastal_bonus;
                    // Restore potential distance penalty
                    let old_dist = if settings.tribes.len() == 2 && old_dist < -20 {
                        old_dist + 50
                    } else {
                        old_dist
                    };

                    let new_dist = distance(*v, best_v, size);
                    let new_min_dist = old_dist.min(new_dist);

                    let mut new_score = new_min_dist + coastal_bonus;
                    if settings.tribes.len() == 2 && new_min_dist < size / 3 {
                        new_score -= 50;
                    }
                    *score = new_score;
                }
            }
        }
    }

    // Biomes
    let mut done = HashSet::new();
    let mut active = vec![Vec::new(); settings.tribes.len()];
    for (i, &cap) in capital_cells.iter().enumerate() {
        active[i].push(cap);
        done.insert(cap);
        map[cap as usize].tribe_affinity = Some(settings.tribes[i]);
    }
    loop {
        let mut changed = false;
        for i in 0..settings.tribes.len() {
            if active[i].is_empty() {
                continue;
            }
            let idx = rng.gen_range(0..active[i].len());
            let cell = active[i][idx];
            let neighbors = get_square(cell, 1, size);
            let mut valid: Vec<i32> = neighbors
                .iter()
                .cloned()
                .filter(|&n| !done.contains(&n) && is_land[n as usize])
                .collect();
            if valid.is_empty() {
                valid = neighbors
                    .iter()
                    .cloned()
                    .filter(|&n| !done.contains(&n))
                    .collect();
            }
            if !valid.is_empty() {
                let chosen = valid[rng.gen_range(0..valid.len())];
                map[chosen as usize].tribe_affinity = Some(settings.tribes[i]);
                active[i].push(chosen);
                done.insert(chosen);
                changed = true;
            } else {
                active[i].swap_remove(idx);
            }
        }
        if !changed {
            break;
        }
    }

    // Fill in orphan land tiles (isolated islands) with nearest tribe affinity
    for i in 0..tile_count {
        if is_land[i as usize] && map[i as usize].tribe_affinity.is_none() {
            let mut min_dist = i32::MAX;
            let mut best_tribe = settings.tribes[0]; // Fallback

            for &cap in &capital_cells {
                let d = distance(i as i32, cap, size);
                if d < min_dist {
                    min_dist = d;
                    // Safely unwrap or fallback, though capitals should always have affinity
                    best_tribe = map[cap as usize]
                        .tribe_affinity
                        .unwrap_or(settings.tribes[0]);
                }
            }
            map[i as usize].tribe_affinity = Some(best_tribe);

            // Also assign orig_tribe_affinity if needed
            map[i as usize].orig_tribe_affinity = Some(best_tribe);
        }
    }

    for i in 0..tile_count {
        if !is_land[i as usize] && plus_sign(i, size).iter().any(|&n| is_land[n as usize]) {
            map[i as usize].terrain_type = TerrainType::Water;
        } else if is_land[i as usize] && village_map[i as usize] == 0 {
            let tribe = map[i as usize]
                .tribe_affinity
                .unwrap_or(TribeType::Luxidoor);
            let rates = get_tribe_biome_rates(tribe);
            let r: f32 = rng.r#gen();
            if r < rates.mountain {
                map[i as usize].terrain_type = TerrainType::Mountain;
            } else if r < rates.mountain + rates.forest {
                map[i as usize].terrain_type = TerrainType::Forest;
            }
        }
    }

    // Post-terrain Villages (only for quadrant-based maps: Drylands, Lakes, Archipelago, WaterWorld)
    if matches!(
        settings.map_type,
        MapType::Drylands | MapType::Lakes | MapType::Archipelago | MapType::WaterWorld
    ) {
        loop {
            let candidates: Vec<i32> = (0..tile_count)
                .filter(|&i| {
                    let (x, y) = get_coords(i, size);
                    let dist_x = x.min(size - 1 - x);
                    let dist_y = y.min(size - 1 - y);
                    let edge_dist = dist_x.min(dist_y);

                    is_land[i as usize]
                        && village_map[i as usize] == 0
                        && map[i as usize].terrain_type != TerrainType::Mountain
                        && edge_dist >= 2     // Not within two tiles (0, 1)
                        && edge_dist != 3     // Not three tiles from the edge
                        && village_map
                            .iter()
                            .enumerate()
                            .all(|(v_idx, &v)| v == 0 || distance(i, v_idx as i32, size) >= 3)
                    // Not within two tiles (0, 1, 2)
                })
                .collect();

            if candidates.is_empty() {
                break;
            }

            let idx = candidates[rng.gen_range(0..candidates.len())];
            village_map[idx as usize] = 1;
            // Convert forest to field if needed
            if map[idx as usize].terrain_type == TerrainType::Forest {
                map[idx as usize].terrain_type = TerrainType::Field;
            }
            map[idx as usize].above = Some("village".to_string());
        }
    }

    // Tiny Island Villages (Pangea/Continents/WaterWorld)
    if settings.map_type == MapType::Pangea
        || settings.map_type == MapType::Continents
        || settings.map_type == MapType::WaterWorld
    {
        let island_count = match settings.size {
            MapSize::Tiny => 0,
            MapSize::Small => 1,
            MapSize::Normal => 2,
            MapSize::Large => 3,
            MapSize::Huge => 4,
            MapSize::Massive => 9,
        };

        // Find small isolated land tiles (surrounded mostly by water)
        let mut island_candidates: Vec<i32> = (0..tile_count)
            .filter(|&i| {
                if !is_land[i as usize] || village_map[i as usize] > 0 {
                    return false;
                }
                let neighbors = get_square(i, 1, size);
                let water_count = neighbors.iter().filter(|&&n| !is_land[n as usize]).count();
                // At least 6 of 8 neighbors are water (isolated)
                water_count >= 6
                    && village_map
                        .iter()
                        .enumerate()
                        .all(|(v_idx, &v)| v == 0 || distance(i, v_idx as i32, size) >= 3)
            })
            .collect();

        let mut placed = 0;
        while placed < island_count && !island_candidates.is_empty() {
            let idx = island_candidates.remove(rng.gen_range(0..island_candidates.len()));
            village_map[idx as usize] = 1;
            map[idx as usize].above = Some("village".to_string());
            map[idx as usize].terrain_type = TerrainType::Field;
            placed += 1;
            island_candidates.retain(|&c| distance(c, idx, size) >= 3);
        }
    }

    // Natural resource spawning. Resources exist only within 2 tiles of a
    // village site: full rate at Chebyshev <=1 ("inner city territory"), 1/3
    // of it at distance 2 ("border expansion"), zero beyond. One
    // classification and one roll per tile; inner takes precedence when
    // village zones overlap.
    let village_positions: Vec<i32> = (0..tile_count)
        .filter(|&i| village_map[i as usize] > 0)
        .collect();

    let mut spawn_zone = vec![0u8; tile_count as usize];
    for &v in &village_positions {
        for idx in get_square(v, 2, size) {
            spawn_zone[idx as usize] = spawn_zone[idx as usize].max(1);
        }
    }
    for &v in &village_positions {
        for idx in get_square(v, 1, size) {
            spawn_zone[idx as usize] = 2;
        }
    }

    for i in 0..tile_count {
        let zone = spawn_zone[i as usize];
        if zone == 0 || map[i as usize].above.is_some() {
            continue;
        }
        let inner = zone == 2;
        let tribe = map[i as usize]
            .tribe_affinity
            .unwrap_or(TribeType::Luxidoor);
        match map[i as usize].terrain_type {
            TerrainType::Field => {
                let fp = get_resource_prob("fruit", tribe, inner);
                let (cp, res_name) = if tribe == TribeType::Cymanti {
                    (get_resource_prob("spores", tribe, inner), "spores")
                } else {
                    (get_resource_prob("crop", tribe, inner), "crop")
                };
                let r: f32 = rng.r#gen();
                if r < fp {
                    map[i as usize].above = Some("fruit".to_string());
                } else if r < fp + cp {
                    map[i as usize].above = Some(res_name.to_string());
                }
            }
            TerrainType::Forest => {
                if rng.r#gen::<f32>() < get_resource_prob("game", tribe, inner) {
                    map[i as usize].above = Some("game".to_string());
                }
            }
            TerrainType::Mountain => {
                if rng.r#gen::<f32>() < get_resource_prob("metal", tribe, inner) {
                    map[i as usize].above = Some("metal".to_string());
                }
            }
            TerrainType::Water => {
                if rng.r#gen::<f32>() < get_resource_prob("fish", tribe, inner) {
                    map[i as usize].above = Some("fish".to_string());
                }
            }
            _ => {}
        }
    }

    // Guaranteed starting resources — a top-up AFTER natural spawning (real
    // game semantics: only add what the rolls didn't already provide).
    for &cap in &capital_cells {
        let tribe = map[cap as usize]
            .tribe_affinity
            .unwrap_or(TribeType::Imperius);
        // NB: this block WRITES `target_terrain` onto the chosen tile, so a
        // fish start carves water. On a dry map the water tribes are served by
        // the dedicated capital-pond block below instead, which pins the count
        // at exactly 2 — running both would scatter up to 4.
        let dry = is_fully_dry(settings.map_type);
        let (resource, target_terrain, quantity): (&str, TerrainType, i32) = match tribe {
            TribeType::Imperius => ("fruit", TerrainType::Field, 2),
            TribeType::Bardur => ("game", TerrainType::Forest, 2),
            TribeType::Zebasi => ("crop", TerrainType::Field, 1),
            TribeType::Elyrion => ("game", TerrainType::Forest, 2),
            TribeType::XinXi => ("metal", TerrainType::Mountain, 2),
            TribeType::Kickoo | TribeType::Aquarion if dry => ("", TerrainType::Field, 0),
            TribeType::Kickoo => ("fish", TerrainType::Water, 2),
            TribeType::Aquarion => ("fish", TerrainType::Water, 2),
            TribeType::Cymanti => ("spores", TerrainType::Field, 2),
            _ => ("", TerrainType::Field, 0),
        };

        if resource.is_empty() {
            continue;
        }

        // Count existing resources in radius 1
        let radius1 = get_square(cap, 1, size);
        let existing: i32 = radius1
            .iter()
            .filter(|&&n| map[n as usize].above.as_deref() == Some(resource))
            .count() as i32;

        let needed = quantity - existing;
        if needed <= 0 {
            continue;
        }

        let eligible_terrain = |n: i32| {
            map[n as usize].terrain_type == target_terrain
                || map[n as usize].terrain_type == TerrainType::Field
                || map[n as usize].terrain_type == TerrainType::Forest
                || map[n as usize].terrain_type == TerrainType::Mountain
                || map[n as usize].terrain_type == TerrainType::Water
        };
        // Prefer untouched tiles; if natural spawning saturated the ring,
        // overwrite other resources (never villages/capitals) so the
        // guarantee actually guarantees.
        let mut candidates: Vec<i32> = radius1
            .iter()
            .cloned()
            .filter(|&n| n != cap && map[n as usize].above.is_none() && eligible_terrain(n))
            .collect();
        if (candidates.len() as i32) < needed {
            let overwritable: Vec<i32> = radius1
                .iter()
                .cloned()
                .filter(|&n| {
                    n != cap
                        && matches!(
                            map[n as usize].above.as_deref(),
                            Some("fruit" | "crop" | "game" | "fish" | "spores")
                                if map[n as usize].above.as_deref() != Some(resource)
                        )
                        && eligible_terrain(n)
                })
                .collect();
            candidates.extend(overwritable);
        }

        for _ in 0..needed {
            if candidates.is_empty() {
                break;
            }
            let idx = candidates.remove(rng.gen_range(0..candidates.len()));
            map[idx as usize].terrain_type = target_terrain;
            map[idx as usize].above = Some(resource.to_string());
        }
    }

    // Water tribes are the standing exception to a dry map (Verdi, Aug 2026):
    // Kickoo/Aquarion capitals always get exactly 2 adjacent fish ponds, so the
    // tribe is playable. This is the ONLY water a fully-dry map may hold.
    if is_fully_dry(settings.map_type) {
        for &cap in &capital_cells {
            let tribe = map[cap as usize]
                .tribe_affinity
                .unwrap_or(TribeType::Imperius);
            if !matches!(tribe, TribeType::Kickoo | TribeType::Aquarion) {
                continue;
            }
            let mut placed = 0;
            // Orthogonal neighbours first (a pond behind a diagonal is harder
            // to work), then any free neighbour, so the count is always 2.
            // Pass 1 reclaims resource tiles if spawning saturated the ring.
            let plus = plus_sign(cap, size);
            let ring = get_square(cap, 1, size);
            for pass in 0..2 {
                for n in plus.iter().chain(ring.iter().filter(|n| !plus.contains(n))) {
                    if placed >= 2 {
                        break;
                    }
                    if *n == cap || village_map[*n as usize] > 0 {
                        continue;
                    }
                    let free = map[*n as usize].above.is_none();
                    let overwritable = matches!(
                        map[*n as usize].above.as_deref(),
                        Some("fruit" | "crop" | "game" | "metal" | "spores")
                    );
                    if !(free || (pass == 1 && overwritable)) {
                        continue;
                    }
                    map[*n as usize].terrain_type = TerrainType::Water;
                    map[*n as usize].above = Some("fish".to_string());
                    placed += 1;
                }
            }
        }
    }

    // Ruins & Starfish
    let ruin_count = match settings.size {
        MapSize::Tiny => 4,
        MapSize::Small => 5,
        MapSize::Normal => 7,
        MapSize::Large => 9,
        MapSize::Huge => 11,
        MapSize::Massive => 23,
    };
    // On Lakes, a maximum of one third of these ruins are allowed to spawn on water.
    let max_water_ruins = if settings.map_type == MapType::Lakes {
        ruin_count / 3
    } else {
        0
    };
    let mut placed = 0;
    let mut water_ruins = 0;
    for _ in 0..2000 {
        if placed >= ruin_count {
            break;
        }
        let idx = rng.gen_range(0..tile_count);
        let terrain = map[idx as usize].terrain_type;
        let is_water = terrain == TerrainType::Water || terrain == TerrainType::Ocean;

        if map[idx as usize].above.is_some() || village_map[idx as usize] > 0 {
            continue;
        }

        // Water ruins only on Lakes, and only up to max_water_ruins
        if is_water && water_ruins >= max_water_ruins {
            continue;
        }

        // Adjacency check
        let mut neighbors_ok = true;
        for n in get_square(idx, 1, size) {
            if map[n as usize].above.as_deref() == Some("ruin") || village_map[n as usize] > 0 {
                neighbors_ok = false;
                break;
            }
        }
        if neighbors_ok {
            map[idx as usize].above = Some("ruin".to_string());
            placed += 1;
            if is_water {
                water_ruins += 1;
            }
        }
    }

    let starfish_count = tile_count / 25;
    let mut placed_starfish = 0;
    for _ in 0..1000 {
        if placed_starfish >= starfish_count {
            break;
        }
        let idx = rng.gen_range(0..tile_count);
        if (map[idx as usize].terrain_type == TerrainType::Water
            || map[idx as usize].terrain_type == TerrainType::Ocean)
            && map[idx as usize].above.is_none()
        {
            // Starfish proximity check (cannot be next to other starfish, lighthouse, or city)
            let neighbors = get_square(idx, 1, size);
            let safe = neighbors.iter().all(|&n| {
                let above = map[n as usize].above.as_deref();
                above != Some("starfish")
                    && above != Some("lighthouse")
                    && above != Some("capital")
                    && above != Some("village")
            });

            if safe {
                map[idx as usize].above = Some("starfish".to_string());
                placed_starfish += 1;
            }
        }
    }

    // Place Lighthouses on all 4 corners if version >= 114
    if settings.version >= 114 {
        for &idx in &crate::coords::map_corners(size) {
            map[idx as usize].above = Some("lighthouse".to_string());
        }
    }

    // Conversion to GameState
    let mut game_state = GameState::default();
    game_state.settings.size = size;
    game_state.settings.map_type = settings.map_type;
    game_state.settings.tile_count = tile_count;
    game_state.settings.version = settings.version;
    // Most important rule. Disabled = God mode
    game_state.settings._fow = default_fow();
    game_state.settings._max_tribe_count = settings.tribes.len() as i32;
    game_state.settings.seed = settings.seed;

    for (i, &tribe) in settings.tribes.iter().enumerate() {
        let id = (i + 1) as i32;
        let mut t_state = TribeState::default();
        t_state.id = id;
        t_state.tribe_type = tribe;
        // Initial starting stars
        t_state.stars = match tribe {
            TribeType::Luxidoor => 2,
            TribeType::Oumaji => 6,
            TribeType::Hoodrick | TribeType::XinXi | TribeType::Quetzali | TribeType::Yadakk => 7,
            _ => 5,
        };

        use crate::states::TechnologyState;
        use crate::types::TechnologyType;
        let mut starting_tech = vec![TechnologyState {
            tech_type: TechnologyType::Basic,
            discovered: true,
            discovered_turn: 0,
        }];
        let tech_type = match tribe {
            TribeType::Imperius => Some(TechnologyType::Organization),
            TribeType::Bardur => Some(TechnologyType::Hunting),
            TribeType::Kickoo => Some(TechnologyType::Fishing),
            TribeType::Oumaji => Some(TechnologyType::Riding),
            TribeType::XinXi => Some(TechnologyType::Climbing),
            TribeType::Zebasi => Some(TechnologyType::Farming),
            TribeType::AiMo => Some(TechnologyType::Philosophy),
            TribeType::Hoodrick => Some(TechnologyType::Archery),
            TribeType::Vengir => Some(TechnologyType::Smithery),
            TribeType::Quetzali => Some(TechnologyType::Strategy),
            TribeType::Yadakk => Some(TechnologyType::Roads),
            TribeType::Polaris => Some(TechnologyType::Frostwork),
            TribeType::Cymanti => Some(TechnologyType::Farming),
            TribeType::Elyrion => Some(TechnologyType::ForestMagic),
            TribeType::Aquarion => Some(TechnologyType::Riding),
            _ => None,
        };
        if let Some(t) = tech_type {
            starting_tech.push(TechnologyState {
                tech_type: t,
                discovered: true,
                discovered_turn: 0,
            });
        }
        t_state.tech_vanilla = starting_tech;
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
        if gen_tile.terrain_type == TerrainType::Water
            || gen_tile.terrain_type == TerrainType::Ocean
        {
            t_state.climate = 0;
        } else if let Some(tribe) = gen_tile.tribe_affinity {
            t_state.climate = classic_climate_id(tribe);
        }
        if let Some(ref s) = gen_tile.above {
            match s.as_str() {
                "village" | "capital" => {
                    use crate::states::StructureState;
                    use crate::types::StructureType;
                    let mut s_state = StructureState::default();
                    s_state.structure_type = StructureType::Village;
                    game_state.structures.insert(gen_tile.idx, Some(s_state));
                }
                "lighthouse" => {
                    use crate::states::StructureState;
                    use crate::types::StructureType;
                    let mut s_state = StructureState::default();
                    s_state.structure_type = StructureType::Lighthouse;
                    game_state.structures.insert(gen_tile.idx, Some(s_state));
                }
                "ruin" => {
                    use crate::states::StructureState;
                    use crate::types::StructureType;
                    let mut s_state = StructureState::default();
                    s_state.structure_type = StructureType::Ruin;
                    game_state.structures.insert(gen_tile.idx, Some(s_state));
                }
                "fruit" => {
                    use crate::states::ResourceState;
                    use crate::types::ResourceType;
                    let mut r_state = ResourceState::default();
                    r_state.resource_type = ResourceType::Fruit;
                    game_state.resources.insert(gen_tile.idx, Some(r_state));
                }
                "crop" => {
                    use crate::states::ResourceState;
                    use crate::types::ResourceType;
                    let mut r_state = ResourceState::default();
                    r_state.resource_type = ResourceType::Crop;
                    game_state.resources.insert(gen_tile.idx, Some(r_state));
                }
                "game" => {
                    use crate::states::ResourceState;
                    use crate::types::ResourceType;
                    let mut r_state = ResourceState::default();
                    r_state.resource_type = ResourceType::Game;
                    game_state.resources.insert(gen_tile.idx, Some(r_state));
                }
                "fish" => {
                    use crate::states::ResourceState;
                    use crate::types::ResourceType;
                    let mut r_state = ResourceState::default();
                    r_state.resource_type = ResourceType::Fish;
                    game_state.resources.insert(gen_tile.idx, Some(r_state));
                }
                "metal" => {
                    use crate::states::ResourceState;
                    use crate::types::ResourceType;
                    let mut r_state = ResourceState::default();
                    r_state.resource_type = ResourceType::Metal;
                    game_state.resources.insert(gen_tile.idx, Some(r_state));
                }
                "starfish" => {
                    use crate::states::ResourceState;
                    use crate::types::ResourceType;
                    let mut r_state = ResourceState::default();
                    r_state.resource_type = ResourceType::Starfish;
                    game_state.resources.insert(gen_tile.idx, Some(r_state));
                }
                "spores" => {
                    use crate::states::ResourceState;
                    use crate::types::ResourceType;
                    let mut r_state = ResourceState::default();
                    r_state.resource_type = ResourceType::Spores;
                    game_state.resources.insert(gen_tile.idx, Some(r_state));
                }
                _ => {}
            }
        }
        game_state.tiles.insert(gen_tile.idx, t_state);
    }

    // Assign capital_of to tiles
    for (i, &cap) in capital_cells.iter().enumerate() {
        let pid = (i + 1) as i32;
        if let Some(tile) = game_state.tiles.get_mut(&cap) {
            tile.capital_of = pid;
        }
    }

    // Capital/City Setup
    for (i, &cap) in capital_cells.iter().enumerate() {
        let tribe = settings.tribes[i];
        let pid = (i + 1) as i32;
        use crate::states::CityState;
        let mut city = CityState::default();
        city.idx = cap;
        city.owner = pid;
        city.level = if tribe == TribeType::Luxidoor { 3 } else { 1 };
        city.population = if tribe == TribeType::Luxidoor { 5 } else { 0 };
        city.production = city.level;
        city.border_size = 1;

        let mut territory = Vec::new();
        let (cx, cy) = get_coords(cap, size);
        for dy in -city.border_size..=city.border_size {
            for dx in -city.border_size..=city.border_size {
                let nx = cx + dx;
                let ny = cy + dy;
                if nx >= 0 && nx < size && ny >= 0 && ny < size {
                    territory.push(ny * size + nx);
                }
            }
        }
        city._territory = territory.clone();

        let cap_coords = game_state.tiles[&cap].coords;
        if let Some(t) = game_state.tribes.get_mut(&pid) {
            t.cities.push(city);
            t.starting_tile_coords = cap_coords;
        }
        for idx in territory {
            if let Some(tile) = game_state.tiles.get_mut(&idx) {
                tile.owner = pid;
                tile.ruling_city_coords = Some(cap_coords);
                // Allowing this would be cheating
                if tile.terrain_type != TerrainType::Water
                    && tile.terrain_type != TerrainType::Ocean
                {
                    tile.climate = classic_climate_id(tribe);
                }
            }
        }
    }

    // Starting units
    use crate::types::UnitType;
    for (i, &cap_idx) in capital_cells.iter().enumerate() {
        let tribe = settings.tribes[i];
        let pid = (i + 1) as i32;
        let unit_type = match tribe {
            TribeType::Hoodrick => UnitType::Archer,
            TribeType::Vengir => UnitType::Swordsman,
            TribeType::Oumaji => UnitType::Rider,
            TribeType::Quetzali => UnitType::Defender,
            TribeType::AiMo => UnitType::MindBender,
            TribeType::Aquarion => UnitType::Amphibian,
            TribeType::Polaris => UnitType::Mooni,
            TribeType::Cymanti => UnitType::Shaman,
            _ => UnitType::Warrior,
        };
        use crate::states::UnitState;
        let mut unit = UnitState::default();
        unit.owner = pid;
        unit.unit_type = unit_type;
        unit.coords = game_state.tiles[&cap_idx].coords;
        unit.prev_coords = unit.coords;
        unit.home_coords = Some(unit.coords);
        if let Some(t) = game_state.tribes.get_mut(&pid) {
            t.units.push(unit);
        }
        // Fix: Set tile unit owner
        if let Some(tile) = game_state.tiles.get_mut(&cap_idx) {
            tile._unit_owner_id = Some(pid);
        }
    }

    game_state
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::states::PlayerId;
    use crate::types::{MapSize, MapType, StructureType};

    fn wet_tiles(state: &crate::states::GameState) -> Vec<i32> {
        state
            .tiles
            .iter()
            .filter(|(_, t)| matches!(t.terrain_type, TerrainType::Water | TerrainType::Ocean))
            .map(|(idx, _)| *idx)
            .collect()
    }

    /// Drylands is bone dry for land tribes — the stray puddles it used to
    /// scatter made the (worthless) water tech lane nominally legal.
    #[test]
    fn drylands_generates_no_water_for_land_tribes() {
        let tribe_sets = [
            vec![TribeType::Imperius, TribeType::Bardur],
            vec![TribeType::Zebasi, TribeType::Elyrion],
            vec![TribeType::Bardur, TribeType::Cymanti],
        ];
        for size in [MapSize::Tiny, MapSize::Normal] {
            for tribes in &tribe_sets {
                for seed in 0..40 {
                    let state = generate(MapGenSettings {
                        size,
                        map_type: MapType::Drylands,
                        tribes: tribes.clone(),
                        seed,
                        version: 115,
                    });
                    let wet = wet_tiles(&state);
                    assert!(
                        wet.is_empty(),
                        "seed {seed} {size:?} {tribes:?}: {} water tile(s) at {wet:?}",
                        wet.len()
                    );
                }
            }
        }
    }

    /// Kickoo/Aquarion are the standing exception: exactly 2 fish ponds beside
    /// the capital even on Drylands, and no other water anywhere on the map.
    #[test]
    fn water_tribes_get_exactly_two_capital_ponds_on_drylands() {
        for size in [MapSize::Tiny, MapSize::Normal] {
            for seed in 0..30 {
                let state = generate(MapGenSettings {
                    size,
                    map_type: MapType::Drylands,
                    tribes: vec![TribeType::Kickoo, TribeType::Bardur],
                    seed,
                    version: 115,
                });
                let wet = wet_tiles(&state);
                assert_eq!(
                    wet.len(),
                    2,
                    "seed {seed} {size:?}: expected 2 ponds, got {wet:?}"
                );
                let cap = state
                    .tiles
                    .values()
                    .find(|t| t.capital_of == 1)
                    .expect("Kickoo capital");
                for idx in &wet {
                    assert!(
                        crate::functions::get_square_indices(cap.coords.idx, 1, state.settings.size)
                            .contains(idx),
                        "seed {seed}: pond {idx} is not adjacent to the capital"
                    );
                    assert!(
                        matches!(state.resources.get(idx), Some(Some(_))),
                        "seed {seed}: pond {idx} has no fish"
                    );
                }
            }
        }
    }

    #[test]
    fn test_no_edge_spawns() {
        let map_types = [
            MapType::Drylands,
            MapType::Lakes,
            MapType::Continents,
            MapType::Pangea,
            MapType::Archipelago,
            MapType::WaterWorld,
        ];
        let map_sizes = [MapSize::Tiny, MapSize::Normal];

        for &map_type in &map_types {
            for &size in &map_sizes {
                let settings = MapGenSettings {
                    size,
                    map_type,
                    tribes: vec![TribeType::Imperius, TribeType::Bardur],
                    seed: 42, // Fixed seed for reproducibility
                    version: 115,
                };
                let state = generate(settings);
                let side_size = size.get_size();

                for (idx, tile) in &state.tiles {
                    let (x, y) = (tile.coords.x, tile.coords.y);

                    if let Some(Some(structure)) = state.structures.get(idx) {
                        match structure.structure_type {
                            StructureType::Village => {
                                assert!(
                                    x > 0 && x < side_size - 1 && y > 0 && y < side_size - 1,
                                    "Found Village at ({}, {}) on map type {:?} size {:?}",
                                    x,
                                    y,
                                    map_type,
                                    side_size
                                );
                            }
                            _ => {}
                        }
                    }
                    if tile.capital_of > 0 {
                        assert!(
                            x > 1 && x < side_size - 2 && y > 1 && y < side_size - 2,
                            "Found Capital at ({}, {}) on map type {:?} size {:?}",
                            x,
                            y,
                            map_type,
                            side_size
                        );
                    }
                }
            }
        }
    }
    // 6 map types × 3 sizes × 2000 seeds (~36k generates); fine locally, hours on debug CI.
    #[test]
    #[ignore]
    fn test_min_capital_distance_1v1() {
        let map_types = [
            MapType::Drylands,
            MapType::Lakes,
            MapType::Continents,
            MapType::Pangea,
            MapType::Archipelago,
            MapType::WaterWorld,
        ];
        let map_sizes = [MapSize::Tiny, MapSize::Small, MapSize::Normal];

        for &map_type in &map_types {
            for &size in &map_sizes {
                let mut min_dist = 100;
                for seed in 0..2000 {
                    let settings = MapGenSettings {
                        size,
                        map_type,
                        tribes: vec![TribeType::Imperius, TribeType::Bardur],
                        seed,
                        version: 115,
                    };
                    let state = generate(settings);
                    let mut capitals = Vec::new();
                    // Scan all tribes for their starting cities (capitals)
                    for tribe in state.tribes.values() {
                        for city in &tribe.cities {
                            // In this engine, the first city added is the capital
                            let (x, y) = get_coords(city.idx, size.get_size());
                            capitals.push((x, y));
                        }
                    }

                    if capitals.len() == 2 {
                        let d = (capitals[0].0 - capitals[1].0)
                            .abs()
                            .max((capitals[0].1 - capitals[1].1).abs());
                        if d < min_dist {
                            min_dist = d;
                        }
                        if d <= 3 {
                            println!(
                                "Found capitals too close (dist {}) on map type {:?} size {:?} seed {}",
                                d, map_type, size, seed
                            );
                        }
                    }
                }
                println!("Min distance for {:?} {:?}: {}", map_type, size, min_dist);
            }
        }
    }

    #[test]
    fn test_duplicate_tribes_ownership() {
        let settings = MapGenSettings {
            size: MapSize::Tiny,
            map_type: MapType::Drylands,
            tribes: vec![TribeType::Imperius, TribeType::Imperius],
            seed: 123,
            version: 115,
        };
        let state = generate(settings);

        // Check that we have 2 tribes
        assert_eq!(state.tribes.len(), 2);

        // Check that each tribe has exactly one city and one unit
        for (id, tribe) in &state.tribes {
            assert_eq!(tribe.cities.len(), 1, "Tribe {} should have 1 city", id);
            assert_eq!(tribe.units.len(), 1, "Tribe {} should have 1 unit", id);
        }

        // Check that the cities have different owners
        let owners: HashSet<PlayerId> = state
            .tribes
            .values()
            .flat_map(|t| t.cities.iter().map(|c| c.owner))
            .collect();
        assert_eq!(owners.len(), 2, "There should be 2 unique city owners");

        // Check that units have different owners
        let unit_owners: HashSet<PlayerId> = state
            .tribes
            .values()
            .flat_map(|t| t.units.iter().map(|u| u.owner))
            .collect();
        assert_eq!(unit_owners.len(), 2, "There should be 2 unique unit owners");
    }

    #[test]
    fn test_map_is_perfect_square() {
        let map_types = [
            MapType::Drylands,
            MapType::Lakes,
            MapType::Continents,
            MapType::Pangea,
            MapType::Archipelago,
            MapType::WaterWorld,
        ];
        let map_sizes = [
            MapSize::Tiny,
            MapSize::Small,
            MapSize::Normal,
            MapSize::Large,
            MapSize::Huge,
            MapSize::Massive,
        ];

        for &map_type in &map_types {
            for &size in &map_sizes {
                for seed in 0..10 {
                    let settings = MapGenSettings {
                        size,
                        map_type,
                        tribes: vec![TribeType::Imperius, TribeType::Bardur],
                        seed,
                        version: 115,
                    };
                    let state = generate(settings);
                    let side = size.get_size();
                    let tc = side * side;

                    assert_eq!(
                        state.tiles.len() as i32,
                        tc,
                        "tile count mismatch: map={:?} size={:?} seed={} got={} want={}",
                        map_type,
                        size,
                        seed,
                        state.tiles.len(),
                        tc
                    );

                    let mut seen = std::collections::HashSet::new();
                    for (idx, tile) in &state.tiles {
                        let (x, y) = (tile.coords.x, tile.coords.y);
                        assert!(
                            x >= 0 && x < side && y >= 0 && y < side,
                            "out-of-range coord ({},{}) idx={} map={:?} size={:?} seed={}",
                            x,
                            y,
                            idx,
                            map_type,
                            size,
                            seed
                        );
                        assert_eq!(
                            *idx, tile.coords.idx,
                            "map key idx != coords.idx map={:?} seed={}",
                            map_type,
                            seed
                        );
                        assert_eq!(
                            (x, y),
                            crate::functions::idx_to_coords(*idx, side),
                            "coords <-> idx mismatch map={:?} seed={}",
                            map_type,
                            seed
                        );
                        assert!(
                            seen.insert((x, y)),
                            "duplicate coord ({},{}) map={:?} size={:?} seed={}",
                            x,
                            y,
                            map_type,
                            size,
                            seed
                        );
                    }
                }
            }
        }
    }

    /// Pooled conditional spawn rates near village sites must track the real
    /// game (mapgen_research.md): metal 0.8, game 0.5, fruit 0.375 inner;
    /// outer ring = 1/3 of inner. Luxidoor has no multipliers = pure base.
    #[test]
    fn test_resource_rates_match_real_game() {
        use crate::functions::get_chebyshev_distance;
        use crate::types::{ResourceType, TribeType};

        let mut inner_metal = [0u32; 2];
        let mut outer_metal = [0u32; 2];
        let mut inner_game = [0u32; 2];
        let mut inner_fruit = [0u32; 2];

        for seed in 0..30 {
            let state = generate(MapGenSettings {
                size: MapSize::Normal,
                map_type: MapType::Drylands,
                tribes: vec![TribeType::Luxidoor, TribeType::Luxidoor],
                seed,
                version: 115,
            });
            let size = state.settings.size;
            let sites: Vec<i32> = state
                .structures
                .iter()
                .filter(|(_, s)| {
                    matches!(s, Some(s) if s.structure_type == StructureType::Village)
                })
                .map(|(idx, _)| *idx)
                .collect();

            for (idx, tile) in &state.tiles {
                let d = sites
                    .iter()
                    .map(|&v| get_chebyshev_distance(*idx, v, size))
                    .min()
                    .unwrap_or(99);
                if d > 2 {
                    continue;
                }
                let res = state
                    .resources
                    .get(idx)
                    .and_then(|r| r.as_ref())
                    .map(|r| r.resource_type);
                match tile.terrain_type {
                    TerrainType::Mountain => {
                        let b = if d <= 1 {
                            &mut inner_metal
                        } else {
                            &mut outer_metal
                        };
                        b[0] += 1;
                        if res == Some(ResourceType::Metal) {
                            b[1] += 1;
                        }
                    }
                    TerrainType::Field if d <= 1 => {
                        inner_fruit[0] += 1;
                        if res == Some(ResourceType::Fruit) {
                            inner_fruit[1] += 1;
                        }
                    }
                    TerrainType::Forest if d <= 1 => {
                        inner_game[0] += 1;
                        if res == Some(ResourceType::Game) {
                            inner_game[1] += 1;
                        }
                    }
                    _ => {}
                }
            }
        }

        let rate = |c: &[u32; 2]| c[1] as f64 / c[0].max(1) as f64;
        assert!(
            (0.70..=0.90).contains(&rate(&inner_metal)),
            "inner metal rate {:?}",
            inner_metal
        );
        assert!(
            (0.15..=0.40).contains(&rate(&outer_metal)),
            "outer metal rate {:?}",
            outer_metal
        );
        assert!(
            (0.40..=0.60).contains(&rate(&inner_game)),
            "inner game rate {:?}",
            inner_game
        );
        assert!(
            (0.28..=0.47).contains(&rate(&inner_fruit)),
            "inner fruit rate {:?}",
            inner_fruit
        );
    }

    /// Xin-xi's guaranteed capital resource is metal (Espark's decompiled
    /// Starting Resource table): always >=2 within radius 1.
    #[test]
    fn xinxi_capital_always_has_metal() {
        use crate::functions::get_square_indices;
        use crate::types::{ResourceType, TribeType};

        for seed in 0..30 {
            let state = generate(MapGenSettings {
                size: MapSize::Tiny,
                map_type: MapType::Drylands,
                tribes: vec![TribeType::XinXi, TribeType::Bardur],
                seed,
                version: 115,
            });
            let cap = state
                .tiles
                .values()
                .find(|t| t.capital_of == 1)
                .expect("XinXi capital");
            let metal = get_square_indices(cap.coords.idx, 1, state.settings.size)
                .into_iter()
                .filter(|idx| {
                    matches!(
                        state.resources.get(idx),
                        Some(Some(r)) if r.resource_type == ResourceType::Metal
                    )
                })
                .count();
            assert!(
                metal >= 2,
                "seed {seed}: XinXi capital has {metal} metal in radius 1"
            );
        }
    }

    // -----------------------------------------------------------------
    // Belief-SSOT generator ground-truth probes (belief_grid_ssot_design.md
    // §11). These measure the constraints `ai::belief::map` derives its
    // evidence from; the constants they print are pinned in that module.
    // Heavy (1000+ generates each) — `#[ignore]`, run on demand.
    // -----------------------------------------------------------------

    /// Land, non-mountain, edge-legal — the generator's village-placement
    /// predicate as an OBSERVER can evaluate it on a finished map.
    fn probe_is_legal_site(state: &GameState, idx: i32) -> bool {
        let size = state.settings.size;
        let Some(tile) = state.tiles.get(&idx) else {
            return false;
        };
        if matches!(tile.terrain_type, TerrainType::Water | TerrainType::Ocean) {
            return false;
        }
        if tile.terrain_type == TerrainType::Mountain {
            return false;
        }
        let (x, y) = get_coords(idx, size);
        let edge_dist = x.min(size - 1 - x).min(y.min(size - 1 - y));
        edge_dist >= 2 && edge_dist != 3
    }

    /// Every village site on a finished map: generated villages AND capitals
    /// (mapgen writes `village_map[cap] = 2`, so capitals both block and
    /// satisfy the spacing rule).
    fn probe_village_sites(state: &GameState) -> Vec<i32> {
        let mut sites: Vec<i32> = state
            .structures
            .iter()
            .filter(|(_, s)| {
                s.as_ref()
                    .map_or(false, |s| s.structure_type == StructureType::Village)
            })
            .map(|(&i, _)| i)
            .collect();
        for (&i, t) in &state.tiles {
            if t.capital_of > 0 && !sites.contains(&i) {
                sites.push(i);
            }
        }
        sites.sort_unstable();
        sites
    }

    /// PROBE 12 — C1 (maximality). The post-terrain village pass runs to
    /// saturation, so every legal tile must sit within Chebyshev 2 of a
    /// village or capital. The whole C1 design rests on this.
    /// Also measures `p_base`, the marginal village density on legal tiles.
    #[test]
    #[ignore]
    fn maximality_holds_on_generated_drylands_maps() {
        let tribe_pairs = [
            (TribeType::Imperius, TribeType::Bardur),
            (TribeType::XinXi, TribeType::Oumaji),
            (TribeType::Kickoo, TribeType::Vengir),
        ];
        let mut checked = 0u64;
        let mut violations = 0u64;
        let mut violation_notes: Vec<String> = Vec::new();
        let mut total_legal = 0u64;
        let mut total_sites_on_legal = 0u64;

        for &(t1, t2) in &tribe_pairs {
            for seed in 0..1000i64 {
                let state = generate(MapGenSettings {
                    size: MapSize::Tiny,
                    map_type: MapType::Drylands,
                    tribes: vec![t1, t2],
                    seed,
                    version: 115,
                });
                let size = state.settings.size;
                let sites = probe_village_sites(&state);
                for idx in 0..size * size {
                    if !probe_is_legal_site(&state, idx) {
                        continue;
                    }
                    total_legal += 1;
                    if sites.contains(&idx) {
                        total_sites_on_legal += 1;
                    }
                    checked += 1;
                    let covered = sites
                        .iter()
                        .any(|&v| distance(idx, v, size) <= 2);
                    if !covered {
                        violations += 1;
                        if violation_notes.len() < 20 {
                            let t = &state.tiles[&idx];
                            violation_notes.push(format!(
                                "{:?}/{:?} seed {seed} tile {idx} terrain {:?} climate {}",
                                t1, t2, t.terrain_type, t.climate
                            ));
                        }
                    }
                }
            }
        }
        let p_base = total_sites_on_legal as f64 / total_legal.max(1) as f64;
        println!("PROBE12 checked={checked} violations={violations} rate={:.6}", 
                 violations as f64 / checked.max(1) as f64);
        println!("PROBE12 p_base (villages per legal tile) = {p_base:.6}  \
                  legal={total_legal} sites_on_legal={total_sites_on_legal}");
        for n in &violation_notes {
            println!("PROBE12 violation: {n}");
        }
        assert_eq!(violations, 0, "C1 maximality violated; see notes above");
    }

    /// PROBE 13 — C2 (resource spawn zone). Resources spawn only within
    /// Chebyshev 2 of a village site, nominally 3:1 inner:outer.
    #[test]
    #[ignore]
    fn resources_only_within_2_of_a_village() {
        let mut outside = 0u64;
        let mut inner = 0u64;
        let mut outer = 0u64;
        let mut inner_tiles = 0u64;
        let mut outer_tiles = 0u64;
        let mut notes: Vec<String> = Vec::new();

        for seed in 0..1000i64 {
            let state = generate(MapGenSettings {
                size: MapSize::Tiny,
                map_type: MapType::Drylands,
                tribes: vec![TribeType::Imperius, TribeType::Bardur],
                seed,
                version: 115,
            });
            let size = state.settings.size;
            let sites = probe_village_sites(&state);
            let zone = |idx: i32| -> u8 {
                let d = sites
                    .iter()
                    .map(|&v| distance(idx, v, size))
                    .min()
                    .unwrap_or(99);
                if d <= 1 {
                    2
                } else if d == 2 {
                    1
                } else {
                    0
                }
            };
            for idx in 0..size * size {
                match zone(idx) {
                    2 => inner_tiles += 1,
                    1 => outer_tiles += 1,
                    _ => {}
                }
            }
            for (&idx, r) in &state.resources {
                if r.is_none() {
                    continue;
                }
                match zone(idx) {
                    2 => inner += 1,
                    1 => outer += 1,
                    _ => {
                        outside += 1;
                        if notes.len() < 20 {
                            notes.push(format!(
                                "seed {seed} tile {idx} res {:?}",
                                r.as_ref().map(|r| r.resource_type)
                            ));
                        }
                    }
                }
            }
        }
        let inner_rate = inner as f64 / inner_tiles.max(1) as f64;
        let outer_rate = outer as f64 / outer_tiles.max(1) as f64;
        println!("PROBE13 outside={outside} inner={inner} outer={outer}");
        println!(
            "PROBE13 per-tile rates inner={inner_rate:.4} outer={outer_rate:.4} \
             ratio={:.3} (nominal 3.0)",
            inner_rate / outer_rate.max(1e-9)
        );
        for n in &notes {
            println!("PROBE13 outside-zone: {n}");
        }
        assert_eq!(outside, 0, "resource spawned outside every village zone");
    }

    /// PROBE 14 — C3 (climate Voronoi). Measures P(affinity = seat k) as a
    /// function of the Chebyshev distance difference to the two capitals.
    /// The flood-fill is a round-robin over seats in index order, so seat 1
    /// wins ties: the likelihood is NOT symmetric and a plain logistic is a
    /// bad fit at delta = 0. Emits the empirical table `map.rs` pins.
    #[test]
    #[ignore]
    fn climate_boundary_width() {
        // (tribe order label) -> delta -> (n, # carrying SEAT-2's climate)
        let mut tables: Vec<(String, std::collections::BTreeMap<i32, (u64, u64)>)> = Vec::new();

        for (label, t1, t2) in [
            ("Imperius(s1)/Bardur(s2)", TribeType::Imperius, TribeType::Bardur),
            ("Bardur(s1)/Imperius(s2)", TribeType::Bardur, TribeType::Imperius),
            ("XinXi(s1)/Oumaji(s2)", TribeType::XinXi, TribeType::Oumaji),
        ] {
            let mut buckets: std::collections::BTreeMap<i32, (u64, u64)> =
                std::collections::BTreeMap::new();
            for seed in 0..1000i64 {
                let state = generate(MapGenSettings {
                    size: MapSize::Tiny,
                    map_type: MapType::Drylands,
                    tribes: vec![t1, t2],
                    seed,
                    version: 115,
                });
                let size = state.settings.size;
                let cap1 = state.tiles.values().find(|t| t.capital_of == 1).map(|t| t.coords.idx);
                let cap2 = state.tiles.values().find(|t| t.capital_of == 2).map(|t| t.coords.idx);
                let (Some(cap1), Some(cap2)) = (cap1, cap2) else { continue };
                let c1 = classic_climate_id(t1);
                let c2 = classic_climate_id(t2);
                for (&idx, tile) in &state.tiles {
                    if tile.climate != c1 && tile.climate != c2 {
                        continue;
                    }
                    // delta > 0 means the tile is NEARER seat 2's capital.
                    let delta = distance(idx, cap1, size) - distance(idx, cap2, size);
                    let e = buckets.entry(delta.clamp(-6, 6)).or_insert((0, 0));
                    e.0 += 1;
                    if tile.climate == c2 {
                        e.1 += 1;
                    }
                }
            }
            tables.push((label.to_string(), buckets));
        }

        println!("PROBE14 delta = d(t,cap_seat1) - d(t,cap_seat2); p = P(climate = seat2's)");
        for (label, b) in &tables {
            println!("PROBE14 --- {label}");
            for (d, (n, hits)) in b {
                println!("PROBE14   delta={d:+} n={n} p={:.4}", *hits as f64 / *n as f64);
            }
        }

        // Pooled table across tribe orderings: if the asymmetry is seat-order
        // and not tribe-identity, the three agree closely.
        let mut pooled: std::collections::BTreeMap<i32, (u64, u64)> =
            std::collections::BTreeMap::new();
        for (_, b) in &tables {
            for (d, (n, h)) in b {
                let e = pooled.entry(*d).or_insert((0, 0));
                e.0 += n;
                e.1 += h;
            }
        }
        println!("PROBE14 --- POOLED (the table to pin)");
        let mut max_spread = 0.0f64;
        for (d, (n, hits)) in &pooled {
            let p = *hits as f64 / *n as f64;
            let spread = tables
                .iter()
                .filter_map(|(_, b)| b.get(d).map(|(n, h)| *h as f64 / *n as f64))
                .fold((1.0f64, 0.0f64), |(lo, hi), v| (lo.min(v), hi.max(v)));
            max_spread = max_spread.max(spread.1 - spread.0);
            println!(
                "PROBE14   delta={d:+} n={n} p={p:.4} (per-order range {:.4}..{:.4})",
                spread.0, spread.1
            );
        }
        println!("PROBE14 max per-order spread across tribe orderings = {max_spread:.4}");
        assert!(
            max_spread < 0.05,
            "climate likelihood depends on TRIBE identity, not just seat order \
             (max spread {max_spread:.4}) - the pinned table would be wrong"
        );
    }
}
