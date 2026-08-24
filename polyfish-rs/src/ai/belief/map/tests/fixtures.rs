//! Shared test scaffolding. `blank_state` builds hand-controlled fixtures;
//! `corpus` replays random legal moves for states with genuine explored sets,
//! captured villages and spent resources.

use crate::functions::get_square_indices;
use crate::states::{GameState, PlayerId, StructureState};
use crate::types::{MapSize, MapType, ResourceType, StructureType, TerrainType, TribeType};

pub(super) const SIZE: i32 = 11;

/// A Tiny Drylands 1v1 with NOTHING explored, so a test can reveal exactly
/// the tiles it wants and nothing else moves.
pub(super) fn blank_state() -> GameState {
    let mut state = crate::mapgen::generate(crate::mapgen::MapGenSettings {
        size: MapSize::Tiny,
        map_type: MapType::Drylands,
        tribes: vec![TribeType::Imperius, TribeType::Bardur],
        seed: 4242,
        ..Default::default()
    });
    // Strip every generated village and resource: these tests build their
    // own evidence, and leftover sites would discharge the constraints.
    state.structures.clear();
    state.resources.clear();
    for t in state.tiles.values_mut() {
        t.explorers.clear();
    }
    state
}

pub(super) fn reveal(state: &mut GameState, idx: i32, pov: PlayerId) {
    if let Some(t) = state.tiles.get_mut(&idx) {
        t.explorers.insert(pov);
    }
}

pub(super) fn set_terrain(state: &mut GameState, idx: i32, terrain: TerrainType) {
    if let Some(t) = state.tiles.get_mut(&idx) {
        t.terrain_type = terrain;
    }
}

pub(super) fn put_village(state: &mut GameState, idx: i32) {
    let mut s = StructureState::default();
    s.structure_type = StructureType::Village;
    state.structures.insert(idx, Some(s));
}

pub(super) fn put_resource(state: &mut GameState, idx: i32, r: ResourceType) {
    let mut s = crate::states::ResourceState::default();
    s.resource_type = r;
    state.resources.insert(idx, Some(s));
}

/// Push both players' cities far from `idx` so they never discharge a
/// constraint the test is trying to make.
pub(super) fn park_cities_away(state: &mut GameState) {
    for t in state.tribes.values_mut() {
        t.cities.clear();
    }
}

pub(super) fn disc2(idx: i32) -> Vec<i32> {
    get_square_indices(idx, 2, SIZE)
}

pub(super) fn before_explored(state: &GameState, idx: i32) -> bool {
    state
        .tiles
        .get(&idx)
        .map_or(false, |t| t.explorers.contains(&1))
}

/// Realistic states: play random legal moves from a generated map and
/// snapshot along the way, so the corpus carries genuine explored sets,
/// captured villages, founded cities and spent resources rather than
/// hand-placed fixtures.
pub(super) fn corpus(seeds: std::ops::Range<i64>, snapshots_per_game: usize) -> Vec<GameState> {
    let mut out = Vec::new();
    for seed in seeds {
        let mut game = crate::game::Game::new();
        game.state = crate::mapgen::generate(crate::mapgen::MapGenSettings {
            size: MapSize::Tiny,
            map_type: MapType::Drylands,
            tribes: vec![TribeType::Imperius, TribeType::Bardur],
            seed,
            ..Default::default()
        });
        game.post_load();
        out.push(game.state.clone());

        // A cheap deterministic PRNG; `rand` would pull a dev-dep into the
        // lib tests for one line of work.
        let mut rng = seed as u64 ^ 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for step in 0..(snapshots_per_game * 12) {
            let moves = game.legal_moves();
            if moves.is_empty() || game.is_game_over() {
                break;
            }
            let pick = (next() as usize) % moves.len();
            if game.play_move(moves[pick].as_ref()).is_none() {
                break;
            }
            if step % 12 == 11 {
                out.push(game.state.clone());
            }
        }
    }
    out
}
