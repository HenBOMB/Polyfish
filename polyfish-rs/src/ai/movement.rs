//! Reachability and unit↔target assignment: how many turns it takes a unit
//! (or a whole class of unit) to reach a tile, and which unit should be sent
//! where. Consumed by `oracle_macro`'s T2 orchestration and `reward.rs`'s
//! in-tree shaping; re-exported through `oracle_macro` so existing
//! `crate::ai::oracle_macro::X` call sites keep resolving.

use crate::states::{GameState, PlayerId, UnitState};
use crate::types::TechnologyType;

/// Minimum turns a Rider must save (vs a movement-1 unit) to some EXPAND
/// target for the rider push to fire.
pub const RIDER_PUSH_MIN_TURNS_SAVED: u32 = 1;

/// Simplified land-movement class of a tile: `None` = impassable,
/// `Some(true)` = passable but movement-ending (rough), `Some(false)` = open.
/// FOW-honest: unexplored tiles read as open (optimistic scouting).
fn move_class(state: &GameState, player: PlayerId, idx: i32, climbing: bool) -> Option<bool> {
    use crate::types::TerrainType as T;
    let Some(tile) = state.tiles.get(&idx) else {
        return Some(false);
    };
    if !tile.explorers.contains(&player) {
        return Some(false);
    }
    match tile.terrain_type {
        T::Field | T::None => Some(false),
        T::Forest | T::Wetland | T::Mangrove => Some(true),
        T::Mountain => climbing.then_some(true),
        T::Water | T::Ocean | T::Ice => None,
    }
}

/// Multi-source turns-to-reach for a land unit with `movement` points under
/// simplified Polytopia rules: 8-directional steps, entering rough terrain
/// ends the turn. Returns per-tile turn counts (`u32::MAX` = unreachable).
fn turns_to_reach(
    state: &GameState,
    player: PlayerId,
    anchors: &[i32],
    movement: i32,
    climbing: bool,
) -> Vec<u32> {
    let width = state.settings.size as i32;
    let n = (width * width).max(0) as usize;
    let mut turns = vec![u32::MAX; n];
    let neighbors = |idx: i32| {
        let (r, c) = (idx / width, idx % width);
        let mut out = Vec::with_capacity(8);
        for dr in -1..=1 {
            for dc in -1..=1 {
                if dr == 0 && dc == 0 {
                    continue;
                }
                let (nr, nc) = (r + dr, c + dc);
                if nr >= 0 && nr < width && nc >= 0 && nc < width {
                    out.push(nr * width + nc);
                }
            }
        }
        out
    };
    let mut frontier: Vec<i32> = anchors
        .iter()
        .copied()
        .filter(|&a| (a as usize) < n)
        .collect();
    for &a in &frontier {
        turns[a as usize] = 0;
    }
    let mut t = 0u32;
    while !frontier.is_empty() && t < 64 {
        t += 1;
        let mut next = Vec::new();
        for &p in &frontier {
            for n1 in neighbors(p) {
                let Some(rough1) = move_class(state, player, n1, climbing) else {
                    continue;
                };
                if turns[n1 as usize] > t {
                    turns[n1 as usize] = t;
                    next.push(n1);
                }
                if movement >= 2 && !rough1 {
                    for n2 in neighbors(n1) {
                        if move_class(state, player, n2, climbing).is_some()
                            && turns[n2 as usize] > t
                        {
                            turns[n2 as usize] = t;
                            next.push(n2);
                        }
                    }
                }
            }
        }
        frontier = next;
    }
    turns
}

/// Path-aware rider advantage: max over `targets` of (walker turns − rider
/// turns) along real explored terrain from the player's units (fallback:
/// cities). A forest pocket off the route costs nothing; a forest corridor
/// erases the advantage — exactly the 2-tile-hop question.
pub fn rider_turns_saved(state: &GameState, player: PlayerId, targets: &[i32]) -> u32 {
    let Some(tribe) = state.tribes.get(&player) else {
        return 0;
    };
    let anchors: Vec<i32> = if tribe.units.is_empty() {
        tribe.cities.iter().map(|c| c.idx).collect()
    } else {
        tribe.units.iter().map(|u| u.coords.idx).collect()
    };
    if anchors.is_empty() || targets.is_empty() {
        return 0;
    }
    let climbing = crate::settings::technology::is_tech_unlocked(
        &tribe.tech_vanilla,
        crate::settings::technology::resolve_tech_for_tribe(
            TechnologyType::Climbing,
            tribe.tribe_type,
        ),
    );
    let walk = turns_to_reach(state, player, &anchors, 1, climbing);
    let ride = turns_to_reach(state, player, &anchors, 2, climbing);
    targets
        .iter()
        .filter_map(|&tg| {
            let (w, r) = (walk.get(tg as usize)?, ride.get(tg as usize)?);
            (*w != u32::MAX && *r != u32::MAX).then(|| w.saturating_sub(*r))
        })
        .max()
        .unwrap_or(0)
}

/// Shared greedy nearest-pair core behind `assign_expand_targets`/
/// `assign_expand_targets_by_id`: `units` is (identifying key, current
/// tile) per candidate — the legacy tile-keyed caller uses the same value
/// for both; the id-keyed caller doesn't. Real (explored) targets outrank
/// fog guesses (v6: a scarce unit must never be pinned to a guess while a
/// discovered village waits); ties break on `(key, target)` via the tuple
/// sort. Deterministic, unique per unit AND per target.
fn assign_targets_greedy<K: Ord + Copy + std::hash::Hash>(
    state: &GameState,
    player: PlayerId,
    units: &[(K, i32)],
    targets: &[i32],
) -> Vec<(K, i32)> {
    let size = state.settings.size as i32;
    if size <= 0 {
        return Vec::new();
    }
    let cheb = |a: i32, b: i32| crate::functions::get_chebyshev_distance(a, b, size);
    let is_guess = |t: i32| {
        !state
            .tiles
            .get(&t)
            .map_or(false, |tile| tile.explorers.contains(&player))
    };
    let mut pairs: Vec<(bool, i32, K, i32)> = Vec::new();
    for &(key, pos) in units {
        for &t in targets {
            pairs.push((is_guess(t), cheb(pos, t), key, t));
        }
    }
    pairs.sort_unstable();
    let mut used_u = std::collections::HashSet::new();
    let mut used_t = std::collections::HashSet::new();
    let mut out = Vec::new();
    for (_, _, key, t) in pairs {
        if used_u.contains(&key) || used_t.contains(&t) {
            continue;
        }
        used_u.insert(key);
        used_t.insert(t);
        out.push((key, t));
    }
    out
}

/// Greedy unique unit→EXPAND-target assignment, nearest pair first. Each
/// target's approach term pays only its assigned unit, so two scouts never
/// bank progress on the same fog target (audit: 89% duplicate-sector
/// scouting). Deterministic: ties break on (unit idx, target idx).
pub fn assign_expand_targets(
    state: &GameState,
    player: PlayerId,
    targets: &[i32],
) -> Vec<(i32, i32)> {
    let Some(tribe) = state.tribes.get(&player) else {
        return Vec::new();
    };
    let units: Vec<(i32, i32)> = tribe.units.iter().map(|u| (u.coords.idx, u.coords.idx)).collect();
    assign_targets_greedy(state, player, &units, targets)
}

/// ID-keyed variant for the persistent per-unit goal store (`unit_goals.rs`):
/// same greedy nearest-pair algorithm, restricted to an explicit unit
/// subset (callers pass only units with no active goal) and an explicit
/// target subset (callers pass only unclaimed targets), keyed by the
/// caller's stable `UnitState::id` instead of tile position.
pub fn assign_expand_targets_by_id(
    state: &GameState,
    player: PlayerId,
    units: &[&UnitState],
    targets: &[i32],
) -> Vec<(u32, i32)> {
    let keyed: Vec<(u32, i32)> = units.iter().map(|u| (u.id, u.coords.idx)).collect();
    assign_targets_greedy(state, player, &keyed, targets)
}


/// EXP_ELO_052: road tiles still needed to link each unconnected city into
/// the capital's network, as (city, tiles_remaining).
///
/// The engine connects two adjacent tiles only when BOTH carry a road (city
/// tiles and ports count as road for this purpose) and never through enemy
/// ground — see `actions::connection`. So this is a shortest path from the
/// capital's component where standing road/city tiles are free and buildable
/// ground costs one.
///
/// Why it exists: a connection pays +1 population to the city AND +1 to the
/// capital, but that lands only on the LAST road tile. Every earlier tile on
/// the path earns nothing, so it loses every ballot to a harvest and no city
/// ever connects — measured 0.00 connected cities at t10 across 96 games on
/// three tribes. Handing the remaining-tile count down lets T3 price each
/// tile by the progress it makes.
/// Shared 0-1 BFS from the capital behind `connect_remaining`/`road_relief`:
/// free through standing road, cost 1 per tile that would need building.
/// `force_free`, if given, is treated as already-roaded regardless of its
/// real state — the device `road_relief` uses to ask "what would this one
/// tile change".
fn connect_dist_map(
    state: &GameState,
    player: PlayerId,
    force_free: Option<i32>,
) -> Option<std::collections::HashMap<i32, i32>> {
    use crate::types::StructureType;
    let tribe = state.tribes.get(&player)?;
    // Roads are the only way to build the path; without the tech there is no
    // plan to price, only a constant.
    if !crate::settings::technology::has_technology(&tribe.tech_vanilla, TechnologyType::Roads) {
        return None;
    }
    let cap = crate::functions::get_capital_city(state, player)?;
    let cities: Vec<i32> = tribe.cities.iter().map(|c| c.idx).collect();
    let road_here = |idx: i32| {
        Some(idx) == force_free
            || cities.contains(&idx)
            || crate::functions::get_structure_type_at(state, idx) == Some(StructureType::Road)
    };
    let buildable = |idx: i32| {
        let Some(t) = state.tiles.get(&idx) else { return false };
        if t.owner != 0 && t.owner != player {
            return false;
        }
        if crate::functions::get_structure_at(state, idx).is_some() {
            return false;
        }
        crate::settings::structures::get_structure_setting(StructureType::Road)
            .terrain_types
            .contains(&t.terrain_type)
    };
    let mut dist: std::collections::HashMap<i32, i32> = std::collections::HashMap::new();
    let mut dq: std::collections::VecDeque<i32> = std::collections::VecDeque::new();
    dist.insert(cap.idx, 0);
    dq.push_front(cap.idx);
    while let Some(cur) = dq.pop_front() {
        let d = dist[&cur];
        for n in crate::functions::get_adjacent_indices(state, cur, 1) {
            let free = road_here(n);
            if !free && !buildable(n) {
                continue;
            }
            let nd = d + if free { 0 } else { 1 };
            if dist.get(&n).map_or(true, |&old| nd < old) {
                dist.insert(n, nd);
                if free {
                    dq.push_front(n);
                } else {
                    dq.push_back(n);
                }
            }
        }
    }
    Some(dist)
}

/// EXP_ELO_052: road tiles still needed to link each unconnected city into
/// the capital's network, as (city, tiles_remaining).
///
/// The engine connects two adjacent tiles only when BOTH carry a road (city
/// tiles and ports count as road for this purpose) and never through enemy
/// ground — see `actions::connection`. So this is a shortest path from the
/// capital's component where standing road/city tiles are free and buildable
/// ground costs one.
///
/// Why it exists: a connection pays +1 population to the city AND +1 to the
/// capital, but that lands only on the LAST road tile. Every earlier tile on
/// the path earns nothing, so it loses every ballot to a harvest and no city
/// ever connects — measured 0.00 connected cities at t10 across 96 games on
/// three tribes. Handing the remaining-tile count down lets T3 price each
/// tile by the progress it makes.
pub fn connect_remaining(state: &GameState, player: PlayerId) -> Vec<(i32, i32)> {
    let Some(dist) = connect_dist_map(state, player, None) else {
        return Vec::new();
    };
    let Some(tribe) = state.tribes.get(&player) else {
        return Vec::new();
    };
    let Some(cap) = crate::functions::get_capital_city(state, player) else {
        return Vec::new();
    };
    tribe
        .cities
        .iter()
        .filter(|c| c.idx != cap.idx && !c.connected_to_capital)
        .filter_map(|c| dist.get(&c.idx).map(|&d| (c.idx, d)))
        .collect()
}

/// EXP_ELO_055: real capital-network relief a road built at `tile_idx` would
/// give — total BFS tiles-still-needed removed across every unconnected
/// city, from the SAME shortest-path model `connect_remaining` prices with
/// (terrain/ownership/standing-road aware), not a straight-line distance
/// between a city pair. Priced for mobility/map-control: this measures how
/// much closer the road network gets, independent of what the tile yields.
pub fn road_relief(state: &GameState, player: PlayerId, tile_idx: i32) -> i32 {
    let Some(tribe) = state.tribes.get(&player) else {
        return 0;
    };
    let Some(cap) = crate::functions::get_capital_city(state, player) else {
        return 0;
    };
    let targets: Vec<i32> = tribe
        .cities
        .iter()
        .filter(|c| c.idx != cap.idx && !c.connected_to_capital)
        .map(|c| c.idx)
        .collect();
    if targets.is_empty() {
        return 0;
    }
    let Some(before) = connect_dist_map(state, player, None) else {
        return 0;
    };
    let Some(after) = connect_dist_map(state, player, Some(tile_idx)) else {
        return 0;
    };
    targets
        .iter()
        .filter_map(|t| {
            let b = *before.get(t)?;
            let a = *after.get(t)?;
            Some((b - a).max(0))
        })
        .sum()
}
