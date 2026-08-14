//! Map generation module ported from Python
//!
//! Generates a GameState with a procedural map.

use crate::coords::Coords;
use crate::default_fow;
use crate::dotnet_rng::DotNetRandom as StdRng;
use crate::functions::{
    get_chebyshev_distance as distance, get_plus_sign_indices as plus_sign,
    get_square_indices as get_square, get_squared_euclidean_distance, idx_to_coords as get_coords,
};
use crate::states::{GameState, TileState, TribeState};
use crate::types::{MapSize, MapType, TerrainType, TribeType};
use rand::{Rng, SeedableRng};
use std::collections::HashSet;

fn tribe_to_climate(tribe: TribeType) -> TribeType {
    match tribe {
        TribeType::AiMo => TribeType::AiMo,
        TribeType::Aquarion => TribeType::Aquarion,
        TribeType::Bardur => TribeType::Bardur,
        TribeType::Elyrion => TribeType::Elyrion,
        TribeType::Hoodrick => TribeType::Hoodrick,
        TribeType::Imperius => TribeType::Imperius,
        TribeType::Kickoo => TribeType::Kickoo,
        TribeType::Luxidoor => TribeType::Luxidoor,
        TribeType::Oumaji => TribeType::Oumaji,
        TribeType::Quetzali => TribeType::Quetzali,
        TribeType::Vengir => TribeType::Vengir,
        TribeType::XinXi => TribeType::XinXi,
        TribeType::Yadakk => TribeType::Yadakk,
        TribeType::Zebasi => TribeType::Zebasi,
        TribeType::Polaris => TribeType::Polaris,
        TribeType::Cymanti => TribeType::Cymanti,
        _ => TribeType::Nature,
    }
}

#[derive(Debug, Clone)]
pub struct MapGenSettings {
    pub size: MapSize,
    pub map_type: MapType,
    pub tribes: Vec<TribeType>,
    pub seed: i64,
    pub version: i32,
    pub symmetric: bool,
}

impl Default for MapGenSettings {
    fn default() -> Self {
        Self {
            size: MapSize::Normal,
            map_type: MapType::Continents,
            tribes: vec![TribeType::Imperius, TribeType::Bardur],
            seed: 0,
            version: 115,
            symmetric: false,
        }
    }
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
        TribeType::Bardur => 1.5,
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

pub fn get_resource_prob(key: &str, tribe: TribeType, inner: bool) -> f32 {
    let base = match key {
        "fruit" => {
            if inner {
                0.18
            } else {
                0.06
            }
        }
        "crop" | "spores" => {
            if inner {
                0.18
            } else {
                0.06
            }
        }
        "game" => {
            if inner {
                0.19
            } else {
                0.06
            }
        }
        "metal" => {
            if inner {
                0.11
            } else {
                0.03
            }
        }
        "fish" => 0.50,
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
        ("game", TribeType::Bardur) => 2.0,
        ("game", TribeType::Imperius) => 0.5,
        ("game", TribeType::Oumaji) => 0.2,
        ("game", TribeType::Vengir) => 0.1,
        // Crop modifiers
        ("crop", TribeType::Zebasi) => 2.0,
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

    base * mult
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
        MapType::Drylands => 0.95,
        MapType::Lakes => 0.72,
        MapType::Continents => 0.45,
        MapType::Pangea => 0.78,
        MapType::Archipelago => 0.38,
        MapType::WaterWorld => 0.15,
    };
    for i in 0..tile_count {
        if village_map[i as usize] > 0 {
            is_land[i as usize] = true;
        }
    }

    let target_land = (tile_count as f32 * land_ratio) as usize;
    let mut current_land = is_land.iter().filter(|&&l| l).count();

    if settings.map_type == MapType::Drylands {
        // Drylands is almost entirely land
        let max_attempts = tile_count as usize * 10;
        let mut attempts = 0;
        while current_land < target_land && attempts < max_attempts {
            let idx = rng.gen_range(0..tile_count) as usize;
            if !is_land[idx] {
                is_land[idx] = true;
                current_land += 1;
            }
            attempts += 1;
        }
    } else {
        // Polytopia-style 2D Procedural Noise Pipeline (GenerateNoise -> SmoothNoise -> SetTerrainFromNoise)
        let mut noise = vec![0.0f32; tile_count as usize];

        // 1. GenerateNoise: Harmonic value noise (base frequency + detail octave)
        let grid_size = 4.max(size / 3);
        let mut base_grid = vec![0.0f32; (grid_size * grid_size) as usize];
        for val in &mut base_grid {
            *val = rng.gen_range(-1.0..1.0);
        }

        let detail_size = grid_size * 2;
        let mut detail_grid = vec![0.0f32; (detail_size * detail_size) as usize];
        for val in &mut detail_grid {
            *val = rng.gen_range(-0.5..0.5);
        }

        for y in 0..size {
            for x in 0..size {
                let u = (x as f32 / size as f32) * (grid_size - 1) as f32;
                let v = (y as f32 / size as f32) * (grid_size - 1) as f32;
                let x0 = u.floor() as i32;
                let y0 = v.floor() as i32;
                let x1 = (x0 + 1).min(grid_size - 1);
                let y1 = (y0 + 1).min(grid_size - 1);
                let tx = u - x0 as f32;
                let ty = v - y0 as f32;

                let g00 = base_grid[(y0 * grid_size + x0) as usize];
                let g10 = base_grid[(y0 * grid_size + x1) as usize];
                let g01 = base_grid[(y1 * grid_size + x0) as usize];
                let g11 = base_grid[(y1 * grid_size + x1) as usize];
                let base_val = g00 * (1.0 - tx) * (1.0 - ty)
                    + g10 * tx * (1.0 - ty)
                    + g01 * (1.0 - tx) * ty
                    + g11 * tx * ty;

                // Detail octave
                let du = (x as f32 / size as f32) * (detail_size - 1) as f32;
                let dv = (y as f32 / size as f32) * (detail_size - 1) as f32;
                let dx0 = du.floor() as i32;
                let dy0 = dv.floor() as i32;
                let dx1 = (dx0 + 1).min(detail_size - 1);
                let dy1 = (dy0 + 1).min(detail_size - 1);
                let dtx = du - dx0 as f32;
                let dty = dv - dy0 as f32;

                let dg00 = detail_grid[(dy0 * detail_size + dx0) as usize];
                let dg10 = detail_grid[(dy0 * detail_size + dx1) as usize];
                let dg01 = detail_grid[(dy1 * detail_size + dx0) as usize];
                let dg11 = detail_grid[(dy1 * detail_size + dx1) as usize];
                let detail_val = dg00 * (1.0 - dtx) * (1.0 - dty)
                    + dg10 * dtx * (1.0 - dty)
                    + dg01 * (1.0 - dtx) * dty
                    + dg11 * dtx * dty;

                let idx = (y * size + x) as usize;
                let mut n_val = base_val + detail_val;

                // Capital & Suburb positive elevation boost
                for (v_idx, &v_type) in village_map.iter().enumerate() {
                    if v_type > 0 {
                        let dist = distance(idx as i32, v_idx as i32, size) as f32;
                        if dist <= 3.0 {
                            let boost = (1.0 - dist / 3.5) * 1.8;
                            n_val += boost;
                        }
                    }
                }

                // Pangea edge falloff (ocean border around the supercontinent)
                if settings.map_type == MapType::Pangea {
                    let cx = (x as f32 - size as f32 / 2.0).abs() / (size as f32 / 2.0);
                    let cy = (y as f32 - size as f32 / 2.0).abs() / (size as f32 / 2.0);
                    let edge_dist = cx.max(cy);
                    n_val -= edge_dist.powi(2) * 2.0;
                }

                noise[idx] = n_val;
            }
        }

        // 2. SmoothNoise: Apply smoothing iterations while preserving capital islands
        for _ in 0..2 {
            let mut smoothed = noise.clone();
            for y in 0..size {
                for x in 0..size {
                    let idx = (y * size + x) as usize;
                    let mut sum = 0.0;
                    let mut count = 0.0;
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            let nx = x + dx;
                            let ny = y + dy;
                            if nx >= 0 && nx < size && ny >= 0 && ny < size {
                                sum += noise[(ny * size + nx) as usize];
                                count += 1.0;
                            }
                        }
                    }
                    smoothed[idx] = sum / count;
                }
            }
            noise = smoothed;
        }

        // 3. SetTerrainFromNoise: Compute quantile threshold to exactly match target_land
        let mut sorted_noise = noise.clone();
        sorted_noise.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let threshold_idx = target_land.min(sorted_noise.len().saturating_sub(1));
        let threshold = sorted_noise.get(threshold_idx).copied().unwrap_or(0.0);

        for i in 0..tile_count {
            if village_map[i as usize] > 0 || noise[i as usize] >= threshold {
                is_land[i as usize] = true;
            } else {
                is_land[i as usize] = false;
            }
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
                        && edge_dist >= 2     // Not within two tiles
                        && edge_dist != 3     // Not three tiles from edge
                        && village_map
                            .iter()
                            .enumerate()
                            .all(|(v_idx, &v)| v == 0 || distance(i, v_idx as i32, size) >= 3)
                    // Must be at least 3 tiles away from other villages
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

    // Guaranteed Starting Resources
    for &cap in &capital_cells {
        let tribe = map[cap as usize]
            .tribe_affinity
            .unwrap_or(TribeType::Imperius);
        let (resource, target_terrain, quantity): (&str, TerrainType, i32) = match tribe {
            TribeType::Imperius => ("fruit", TerrainType::Field, 2),
            TribeType::Bardur => ("game", TerrainType::Forest, 2),
            TribeType::Zebasi => ("crop", TerrainType::Field, 1),
            TribeType::Elyrion => ("game", TerrainType::Forest, 2),
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

        // Find eligible tiles in radius 1
        let mut candidates: Vec<i32> = radius1
            .iter()
            .cloned()
            .filter(|&n| {
                n != cap
                    && map[n as usize].above.is_none()
                    && (map[n as usize].terrain_type == target_terrain
                        || map[n as usize].terrain_type == TerrainType::Field
                        || map[n as usize].terrain_type == TerrainType::Forest
                        || map[n as usize].terrain_type == TerrainType::Mountain
                        || map[n as usize].terrain_type == TerrainType::Water)
            })
            .collect();

        for _ in 0..needed {
            if candidates.is_empty() {
                break;
            }
            let idx = candidates.remove(rng.gen_range(0..candidates.len()));
            map[idx as usize].terrain_type = target_terrain;
            map[idx as usize].above = Some(resource.to_string());
        }
    }

    // Resources: Iterate villages and their 2-tile radius
    // Pre-compute village positions for efficiency
    let village_positions: Vec<i32> = (0..tile_count)
        .filter(|&i| village_map[i as usize] > 0)
        .collect();

    for &v in &village_positions {
        let tribe = map[v as usize]
            .tribe_affinity
            .unwrap_or(TribeType::Luxidoor);

        // Determine primary resource caps for initial territory (radius 1)
        // User report: "5 or 6 fruit... is overkill". Guaranteed is 2. Cap at 3 for strictness.
        let (primary_res, max_spawns) = match tribe {
            TribeType::Imperius | TribeType::Quetzali | TribeType::Yadakk => ("fruit", 3),
            TribeType::Bardur | TribeType::Elyrion | TribeType::Hoodrick => ("game", 3),
            TribeType::Kickoo | TribeType::Aquarion => ("fish", 3),
            TribeType::Zebasi => ("crop", 3),
            TribeType::Cymanti => ("spores", 3),
            _ => ("", 99),
        };

        let mut current_res_count = 0;
        // Count existing primary resources in inner territory (radius 1)
        if !primary_res.is_empty() {
            let r1 = get_square(v, 1, size);
            for &idx in &r1 {
                if map[idx as usize].above.as_deref() == Some(primary_res) {
                    current_res_count += 1;
                }
            }
        }

        // Iterate through radius 1 (inner) and radius 2 (outer)
        for radius in 1..=2 {
            let inner = radius == 1;
            let square_tiles = get_square(v, radius, size);

            for tile_idx in square_tiles {
                if map[tile_idx as usize].above.is_some() {
                    continue;
                }

                match map[tile_idx as usize].terrain_type {
                    TerrainType::Field => {
                        let mut fp = get_resource_prob("fruit", tribe, inner);
                        // Apply cap for fruit
                        if primary_res == "fruit" && inner && current_res_count >= max_spawns {
                            fp = 0.0;
                        }

                        let (mut cp, res_name) = if tribe == TribeType::Cymanti {
                            (get_resource_prob("spores", tribe, inner), "spores")
                        } else {
                            (get_resource_prob("crop", tribe, inner), "crop")
                        };

                        // Apply cap for crop/spores
                        if primary_res == res_name && inner && current_res_count >= max_spawns {
                            cp = 0.0;
                        }

                        let r: f32 = rng.r#gen();
                        if r < fp {
                            map[tile_idx as usize].above = Some("fruit".to_string());
                            if primary_res == "fruit" && inner {
                                current_res_count += 1;
                            }
                        } else if r < fp + cp {
                            map[tile_idx as usize].above = Some(res_name.to_string());
                            if primary_res == res_name && inner {
                                current_res_count += 1;
                            }
                        }
                    }
                    TerrainType::Forest => {
                        let mut gp = get_resource_prob("game", tribe, inner);
                        // Apply cap for game
                        if primary_res == "game" && inner && current_res_count >= max_spawns {
                            gp = 0.0;
                        }

                        if rng.r#gen::<f32>() < gp {
                            map[tile_idx as usize].above = Some("game".to_string());
                            if primary_res == "game" && inner {
                                current_res_count += 1;
                            }
                        }
                    }
                    TerrainType::Mountain => {
                        if rng.r#gen::<f32>() < get_resource_prob("metal", tribe, inner) {
                            map[tile_idx as usize].above = Some("metal".to_string());
                        }
                    }
                    TerrainType::Water => {
                        let mut fip = get_resource_prob("fish", tribe, inner);
                        // Apply cap for fish
                        if primary_res == "fish" && inner && current_res_count >= max_spawns {
                            fip = 0.0;
                        }

                        if rng.r#gen::<f32>() < fip {
                            map[tile_idx as usize].above = Some("fish".to_string());
                            if primary_res == "fish" && inner {
                                current_res_count += 1;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // Drylands: Kickoo/Aquarion capitals get 2 water tiles with fish
    if settings.map_type == MapType::Drylands {
        for &cap in &capital_cells {
            let tribe = map[cap as usize]
                .tribe_affinity
                .unwrap_or(TribeType::Imperius);
            if tribe == TribeType::Kickoo || tribe == TribeType::Aquarion {
                let neighbors = plus_sign(cap, size);
                let mut placed = 0;
                for n in neighbors {
                    if placed >= 2 {
                        break;
                    }
                    if map[n as usize].terrain_type != TerrainType::Water {
                        map[n as usize].terrain_type = TerrainType::Water;
                        map[n as usize].above = Some("fish".to_string());
                        placed += 1;
                    }
                }
            }
        }
    }

    // --- Starting Territory Fairness Pass (MapFairnessCalculations / equalityIterations) ---
    // Ensure competitive economic balance across all player capitals within radius 2.
    if capital_cells.len() > 1 {
        let mut scores = Vec::new();
        for &cap in &capital_cells {
            let mut score = 0;
            for n in get_square(cap, 2, size) {
                if let Some(res) = map[n as usize].above.as_deref() {
                    match res {
                        "fruit" | "game" | "crop" | "fish" | "spores" => score += 2,
                        "metal" => score += 3,
                        _ => {}
                    }
                }
                if map[n as usize].terrain_type == TerrainType::Forest {
                    score += 1;
                }
            }
            scores.push((cap, score));
        }

        let max_score = scores.iter().map(|&(_, s)| s).max().unwrap_or(8);
        for &(cap, score) in &scores {
            let mut current_score = score;
            // If a capital has > 4 value points below the highest capital, inject resources in radius 2
            if current_score + 4 < max_score {
                let tribe = map[cap as usize]
                    .tribe_affinity
                    .unwrap_or(TribeType::Imperius);
                let fallback_res = match tribe {
                    TribeType::Bardur | TribeType::Elyrion | TribeType::Hoodrick => {
                        ("game", TerrainType::Forest)
                    }
                    TribeType::Zebasi => ("crop", TerrainType::Field),
                    TribeType::Kickoo | TribeType::Aquarion => ("fish", TerrainType::Water),
                    TribeType::Cymanti => ("spores", TerrainType::Field),
                    _ => ("fruit", TerrainType::Field),
                };

                let mut candidates = get_square(cap, 2, size);
                // Shuffle candidates deterministically using rng
                for _ in 0..candidates.len() {
                    let a = rng.gen_range(0..candidates.len());
                    let b = rng.gen_range(0..candidates.len());
                    candidates.swap(a, b);
                }

                for n in candidates {
                    if current_score + 4 >= max_score {
                        break;
                    }
                    if map[n as usize].above.is_none()
                        && map[n as usize].terrain_type == fallback_res.1
                    {
                        map[n as usize].above = Some(fallback_res.0.to_string());
                        current_score += 2;
                    }
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
    // On non-Drylands maps, a maximum of one third of these ruins are allowed to spawn on water.
    let max_water_ruins = if settings.map_type != MapType::Drylands {
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
        let corners = [0, size - 1, size * (size - 1), size * size - 1];
        for &idx in &corners {
            map[idx as usize].above = Some("lighthouse".to_string());
        }
    }

    // --- Symmetric Map Preset (CreateFromPreset / Competitive 1v1 Mirroring) ---
    // Reflects terrain, resources, forests, villages, and capitals across the center point for 1v1 matches.
    if settings.symmetric && settings.tribes.len() == 2 && capital_cells.len() == 2 {
        let mut cap0 = if capital_cells[0] < tile_count / 2 {
            capital_cells[0]
        } else if capital_cells[1] < tile_count / 2 {
            capital_cells[1]
        } else {
            tile_count - 1 - capital_cells[0]
        };

        if cap0 == tile_count / 2 {
            cap0 = (cap0 - 1).max(0);
        }
        let cap1 = tile_count - 1 - cap0;

        for &old_cap in &capital_cells {
            map[old_cap as usize].above = None;
            village_map[old_cap as usize] = 0;
            map[old_cap as usize].tribe_affinity = None;
            map[old_cap as usize].orig_tribe_affinity = None;
        }

        capital_cells = vec![cap0, cap1];

        for i in 0..(tile_count / 2) {
            let sym_i = tile_count - 1 - i;
            is_land[sym_i as usize] = is_land[i as usize];
            village_map[sym_i as usize] = village_map[i as usize];
            map[sym_i as usize].terrain_type = map[i as usize].terrain_type;
            map[sym_i as usize].above = map[i as usize].above.clone();

            let affinity = match map[i as usize].tribe_affinity {
                Some(t) if t == settings.tribes[0] => Some(settings.tribes[1]),
                Some(t) if t == settings.tribes[1] => Some(settings.tribes[0]),
                other => other,
            };
            map[sym_i as usize].tribe_affinity = affinity;
            map[sym_i as usize].orig_tribe_affinity = affinity;
        }

        // Ensure capital tags and affinities are explicitly set
        map[cap0 as usize].above = Some("capital".to_string());
        map[cap0 as usize].tribe_affinity = Some(settings.tribes[0]);
        map[cap0 as usize].orig_tribe_affinity = Some(settings.tribes[0]);
        map[cap0 as usize].terrain_type = TerrainType::Field;
        is_land[cap0 as usize] = true;

        map[cap1 as usize].above = Some("capital".to_string());
        map[cap1 as usize].tribe_affinity = Some(settings.tribes[1]);
        map[cap1 as usize].orig_tribe_affinity = Some(settings.tribes[1]);
        map[cap1 as usize].terrain_type = TerrainType::Field;
        is_land[cap1 as usize] = true;
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
            t_state.climate = TribeType::Nature;
        } else if let Some(tribe) = gen_tile.tribe_affinity {
            t_state.climate = tribe_to_climate(tribe);
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
                    tile.climate = tribe_to_climate(tribe);
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
                    ..Default::default()
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
    #[test]
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
                        ..Default::default()
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
            ..Default::default()
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
                        ..Default::default()
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
                            map_type, seed
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

    #[test]
    fn test_resource_density() {
        use crate::types::{ResourceType, TribeType};
        let mut settings = MapGenSettings::default();
        settings.tribes = vec![TribeType::Imperius];

        for i in 0..50 {
            settings.seed = i as i64;
            let gamestate = generate(settings.clone());

            let cap_tile = gamestate
                .tiles
                .values()
                .find(|t| t.capital_of == 1) // Imperius is player 1
                .unwrap();

            let size = gamestate.settings.size;
            let mut fruit_count = 0;

            use crate::functions::get_square_indices;
            for idx in get_square_indices(cap_tile.coords.idx, 1, size) {
                if let Some(res) = gamestate.resources.get(&idx).unwrap_or(&None) {
                    if res.resource_type == ResourceType::Fruit {
                        fruit_count += 1;
                    }
                }
            }
            assert!(
                fruit_count <= 3,
                "Seed {}: Found {} fruits, expected <= 3",
                i,
                fruit_count
            );
        }
    }
}
