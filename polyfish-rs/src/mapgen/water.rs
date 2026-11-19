use std::collections::{BinaryHeap, VecDeque};

use crate::dotnet_rng::DotNetRandom;
use crate::types::MapType;

use super::{distance, get_coords, get_square, plus_sign};

pub(super) fn layout(
    kind: MapType,
    size: i32,
    noise: &[f32],
    settlements: &mut [i32],
    capitals: &mut Vec<i32>,
    rng: &mut DotNetRandom,
) -> Vec<bool> {
    let mut land = vec![false; noise.len()];
    let center = (size / 2) * (size + 1);
    let mut island_water = vec![false; noise.len()];
    if matches!(kind, MapType::Pangea | MapType::Continents) {
        reserve_tiny_islands(&mut land, &mut island_water, settlements, size, rng);
    }
    match kind {
        MapType::Lakes => {
            // Connect the existing suburbs and capitals to the land bridges around the lakes.
            add_settlements(settlements, size, 4, rng, |_| true);
            for idx in 0..size * size {
                if edge_distance(idx, size) == 0 {
                    land[idx as usize] = true;
                }
                if settlements[idx as usize] > 0 {
                    let (x, y) = get_coords(idx, size);
                    let (_, step) = [(x, -1), (size - 1 - x, 1), (y, -size), (size - 1 - y, size)]
                        .into_iter()
                        .min()
                        .unwrap();
                    let mut path = idx;
                    while edge_distance(path, size) > 0 {
                        land[path as usize] = true;
                        path += step;
                    }
                }
            }
            let seeds: Vec<_> = (0..size * size).filter(|&idx| land[idx as usize]).collect();
            grow(
                &mut land,
                &seeds,
                noise.len() * 72 / 100,
                noise,
                size,
                |_| true,
            );
        }
        MapType::Pangea => {
            let mainland = grow(&mut land, &[center], noise.len() / 2, noise, size, |idx| {
                edge_distance(idx, size) > 0 && !island_water[idx as usize]
            });
            add_settlements(settlements, size, noise.len(), rng, |idx| {
                land[idx as usize]
            });
            choose_capital(&mainland, settlements, capitals, &land, size, rng);
            choose_capital(&mainland, settlements, capitals, &land, size, rng);
        }
        MapType::Continents => {
            continents(
                &mut land,
                settlements,
                capitals,
                size,
                noise,
                &island_water,
                rng,
            );
        }
        MapType::Archipelago | MapType::WaterWorld => {
            let (tiles_per_settlement, land_percent) = if kind == MapType::Archipelago {
                (32, 38)
            } else {
                (45, 15)
            };
            add_settlements(
                settlements,
                size,
                2 + noise.len() / tiles_per_settlement,
                rng,
                |_| true,
            );
            land = island_chains(settlements, size, noise, noise.len() * land_percent / 100);
        }
        _ => unreachable!("only water map types have a water layout"),
    }
    land
}

fn edge_distance(idx: i32, size: i32) -> i32 {
    let (x, y) = get_coords(idx, size);
    x.min(y).min(size - 1 - x).min(size - 1 - y)
}

fn add_settlements(
    settlements: &mut [i32],
    size: i32,
    desired: usize,
    rng: &mut DotNetRandom,
    allowed: impl Fn(i32) -> bool,
) {
    let mut candidates: Vec<_> = (0..size * size)
        .filter(|&idx| {
            edge_distance(idx, size) >= 2
                && allowed(idx)
                && get_square(idx, 2, size)
                    .into_iter()
                    .all(|near| settlements[near as usize] == 0)
        })
        .collect();
    let existing = settlements.iter().filter(|&&value| value > 0).count();
    for _ in existing..desired {
        if candidates.is_empty() {
            break;
        }
        let idx = candidates.swap_remove(rng.next_index(candidates.len()));
        settlements[idx as usize] = 1;
        candidates.retain(|&other| distance(idx, other, size) >= 3);
    }
}

fn priority(noise: &[f32], idx: i32) -> (i32, i32) {
    ((noise[idx as usize] * 1_000_000.0) as i32, idx)
}

/// Grow from existing land through the contour with the highest available elevation.
fn grow(
    land: &mut [bool],
    seeds: &[i32],
    target: usize,
    noise: &[f32],
    size: i32,
    allowed: impl Fn(i32) -> bool,
) -> Vec<i32> {
    let mut queued = vec![false; land.len()];
    let mut frontier = BinaryHeap::new();
    let mut region = seeds.to_vec();
    for &idx in seeds {
        land[idx as usize] = true;
    }
    let enqueue = |idx, queued: &mut [bool], frontier: &mut BinaryHeap<_>, land: &[bool]| {
        for next in plus_sign(idx, size) {
            if allowed(next) && !land[next as usize] && !queued[next as usize] {
                queued[next as usize] = true;
                frontier.push(priority(noise, next));
            }
        }
    };
    for &idx in seeds {
        enqueue(idx, &mut queued, &mut frontier, land);
    }
    while region.len() < target {
        let Some((_, idx)) = frontier.pop() else {
            break;
        };
        land[idx as usize] = true;
        region.push(idx);
        enqueue(idx, &mut queued, &mut frontier, land);
    }
    region
}

fn choose_capital(
    region: &[i32],
    settlements: &mut [i32],
    capitals: &mut Vec<i32>,
    land: &[bool],
    size: i32,
    rng: &mut DotNetRandom,
) {
    let eligible = |idx| edge_distance(idx, size) >= 2 && settlements[idx as usize] != 2;
    let score = |idx| {
        let separation = capitals
            .iter()
            .map(|&cap| distance(idx, cap, size))
            .min()
            .unwrap_or(size);
        let shore = i32::from(
            plus_sign(idx, size)
                .iter()
                .any(|&near| !land[near as usize]),
        );
        (separation + 3 * shore, shore)
    };
    let mut candidates: Vec<_> = region
        .iter()
        .copied()
        .filter(|&idx| eligible(idx) && settlements[idx as usize] == 1)
        .collect();
    if candidates.is_empty() {
        candidates.extend(region.iter().copied().filter(|&idx| eligible(idx)));
    }
    let best = candidates
        .iter()
        .copied()
        .map(score)
        .max()
        .expect("landmass has capital sites");
    candidates.retain(|&idx| score(idx) == best);
    let idx = candidates[rng.next_index(candidates.len())];
    settlements[idx as usize] = 2;
    capitals.push(idx);
}

fn continents(
    land: &mut [bool],
    settlements: &mut [i32],
    capitals: &mut Vec<i32>,
    size: i32,
    noise: &[f32],
    island_water: &[bool],
    rng: &mut DotNetRandom,
) {
    let half = size / 2;
    let split_both = size >= 30;
    let rotate = rng.next_index(2) == 0;
    let mut regions = Vec::new();
    for row in 0..if split_both { 2 } else { 1 } {
        for column in 0..2 {
            let (x, y) = (
                if column == 0 { half / 2 } else { half * 3 / 2 },
                if split_both {
                    (2 * row + 1) * half / 2
                } else {
                    half
                },
            );
            let idx = |x, y| if rotate { x * size + y } else { y * size + x };
            let eligible = |tile| {
                let (mut tx, mut ty) = get_coords(tile, size);
                if rotate {
                    std::mem::swap(&mut tx, &mut ty);
                }
                // Keep at least one water tile between continents, including diagonally.
                !island_water[tile as usize]
                    && tx != half
                    && (tx > half) == (column == 1)
                    && (!split_both || (ty != half && (ty > half) == (row == 1)))
            };
            let quota = noise.len() / if split_both { 8 } else { 4 };
            regions.push(grow(land, &[idx(x, y)], quota, noise, size, eligible));
        }
    }
    add_settlements(settlements, size, noise.len(), rng, |idx| {
        land[idx as usize]
    });
    let first = rng.next_index(regions.len());
    // For four continents, use opposite corners for the two starts.
    for index in [first, regions.len() - 1 - first] {
        choose_capital(&regions[index], settlements, capitals, land, size, rng);
    }
}

fn island_chains(settlements: &[i32], size: i32, noise: &[f32], target: usize) -> Vec<bool> {
    let mut islands = vec![0; settlements.len()];
    let mut protected_coast = vec![false; settlements.len()];
    let mut frontier = BinaryHeap::new();
    let centers: Vec<_> = (0..size * size)
        .filter(|&idx| settlements[idx as usize] > 0)
        .collect();
    for &idx in &centers {
        let id = idx + 1;
        islands[idx as usize] = id;
        let coast = plus_sign(idx, size)
            .into_iter()
            .min_by_key(|&near| priority(noise, near))
            .unwrap();
        protected_coast[coast as usize] = true;
        if settlements[idx as usize] == 2 {
            protect_harbor(coast, settlements, size, &mut protected_coast);
        }
        for next in plus_sign(idx, size) {
            frontier.push((priority(noise, next), id));
        }
    }
    let mut count = centers.len();
    while count < target {
        let Some(((_, idx), id)) = frontier.pop() else {
            break;
        };
        if islands[idx as usize] != 0 || protected_coast[idx as usize] {
            continue;
        }
        if get_square(idx, 1, size).into_iter().any(|near| {
            let other = islands[near as usize];
            other != 0 && other != id
        }) {
            continue;
        }
        islands[idx as usize] = id;
        count += 1;
        for next in plus_sign(idx, size) {
            if islands[next as usize] == 0 && !protected_coast[next as usize] {
                frontier.push((priority(noise, next), id));
            }
        }
    }
    islands.into_iter().map(|id| id != 0).collect()
}

fn protect_harbor(coast: i32, settlements: &[i32], size: i32, protected: &mut [bool]) {
    // Reserve a route to open water so growth cannot enclose the capital's coast.
    let mut previous = vec![None; settlements.len()];
    let mut queue = VecDeque::from([coast]);
    previous[coast as usize] = Some(coast);
    while let Some(idx) = queue.pop_front() {
        if edge_distance(idx, size) == 0 {
            let mut idx = idx;
            while idx != coast {
                protected[idx as usize] = true;
                idx = previous[idx as usize].unwrap();
            }
            return;
        }
        for next in plus_sign(idx, size) {
            if settlements[next as usize] == 0 && previous[next as usize].is_none() {
                previous[next as usize] = Some(idx);
                queue.push_back(next);
            }
        }
    }
}

fn reserve_tiny_islands(
    land: &mut [bool],
    island_water: &mut [bool],
    settlements: &mut [i32],
    size: i32,
    rng: &mut DotNetRandom,
) {
    let count = match size {
        11 => 0,
        14 => 1,
        16 => 2,
        18 => 3,
        20 => 4,
        30 => 9,
        _ => unreachable!("unsupported board width"),
    };
    // Reserve space offshore before mainland growth, including a navigable water gap.
    let mut candidates: Vec<_> = (0..size * size)
        .filter(|&idx| edge_distance(idx, size) == 1)
        .collect();
    for _ in 0..count {
        let idx = candidates.swap_remove(rng.next_index(candidates.len()));
        land[idx as usize] = true;
        settlements[idx as usize] = 1;
        for near in get_square(idx, 1, size) {
            island_water[near as usize] = true;
        }
        candidates.retain(|&next| distance(idx, next, size) >= 3);
    }
}

pub(super) fn open_fishing_coast(
    land: &mut [bool],
    settlements: &[i32],
    cap: i32,
    size: i32,
    noise: &[f32],
) {
    let neighbors = get_square(cap, 1, size);
    let center = (size / 2) * (size + 1);
    let shallow = neighbors.iter().filter(|&&idx| !land[idx as usize]).count();
    let mut candidates: Vec<_> = neighbors
        .into_iter()
        .filter(|&idx| {
            idx != center
                && edge_distance(idx, size) > 0
                && settlements[idx as usize] == 0
                && land[idx as usize]
        })
        .collect();
    candidates.sort_by_key(|&idx| {
        let inland = plus_sign(idx, size).iter().all(|&near| land[near as usize]);
        (inland, priority(noise, idx))
    });
    for _ in shallow..2 {
        let Some(slot) = candidates
            .iter()
            .position(|&idx| can_carve(land, idx, size))
        else {
            break;
        };
        land[candidates.remove(slot) as usize] = false;
    }
}

fn can_carve(land: &[bool], idx: i32, size: i32) -> bool {
    let neighbors: Vec<_> = plus_sign(idx, size)
        .into_iter()
        .filter(|&near| land[near as usize])
        .collect();
    let Some((&first, rest)) = neighbors.split_first() else {
        return true;
    };
    let mut seen = vec![false; land.len()];
    seen[idx as usize] = true;
    seen[first as usize] = true;
    let mut frontier = vec![first];
    while !rest.iter().all(|&near| seen[near as usize]) {
        let Some(tile) = frontier.pop() else {
            return false;
        };
        for near in plus_sign(tile, size) {
            if land[near as usize] && !seen[near as usize] {
                seen[near as usize] = true;
                frontier.push(near);
            }
        }
    }
    true
}
