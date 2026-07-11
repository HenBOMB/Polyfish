//! Fog memory integration tests: the rider scenario (see notes-memory.md).

use polyfish::ai::features::{
    state_to_cpu_features, CH_MEM_ATTACKED_HERE, CH_MEM_ENEMY_HP, CH_MEM_ENEMY_SEEN, MAP_SIZE,
};
use polyfish::memory;
use polyfish::states::{GameState, TileState, TribeState, UnitState};
use polyfish::types::{TerrainType, UnitType};
use polyfish::Coords;

const SIZE: i32 = 11;

/// 11x11 all-Field map; player 1 has explored the left half (x < 6).
fn build_state() -> GameState {
    let mut state = GameState::default();
    state.settings.size = SIZE;
    state.settings.max_turns = 30;
    state.settings.turn = 5;
    state.settings._fow = true;
    for idx in 0..(SIZE * SIZE) {
        let mut tile = TileState::default();
        tile.coords = Coords::from_index(idx, SIZE);
        tile.terrain_type = TerrainType::Field;
        if idx % SIZE < 6 {
            tile.explorers.insert(1);
        }
        state.tiles.insert(idx, tile);
    }
    for id in [1, 2] {
        state.tribes.insert(
            id,
            TribeState {
                id,
                ..Default::default()
            },
        );
    }
    state
}

fn spatial_at(raw: &polyfish::ai::features::RawFeatures, ch: usize, x: usize, y: usize) -> f32 {
    raw.spatial[ch * (MAP_SIZE * MAP_SIZE) + y * MAP_SIZE + x]
}

fn add_enemy_warrior(state: &mut GameState, idx: i32) {
    state.tribes.get_mut(&2).unwrap().units.push(UnitState {
        owner: 2,
        unit_type: UnitType::Warrior,
        coords: Coords::from_index(idx, SIZE),
        prev_coords: Coords::from_index(idx, SIZE),
        health: 10.0,
        ..Default::default()
    });
}

#[test]
fn test_rider_ghost_persists_and_decays_after_retreat_into_fog() {
    let mut state = build_state();
    let seen_idx = 3; // (3, 0) — explored by player 1
    let fog_idx = 8; // (8, 0) — never explored by player 1
    add_enemy_warrior(&mut state, seen_idx);

    state.settings._are_you_sure = true;

    // Observe: ghost recorded for player 1.
    let undo = memory::observe_all(&mut state);
    assert!(state.tribes.get(&1).unwrap().memory_units.contains_key(&seen_idx));

    // Undo restores the empty memory.
    undo(&mut state);
    assert!(state.tribes.get(&1).unwrap().memory_units.is_empty());

    // Re-observe, then the unit retreats into fog.
    let _ = memory::observe_all(&mut state);
    {
        let u = &mut state.tribes.get_mut(&2).unwrap().units[0];
        u.prev_coords = u.coords;
        u.coords = Coords::from_index(fog_idx, SIZE);
    }
    let _ = memory::observe_all(&mut state);
    state.settings._are_you_sure = false;

    // Ghost stays at the last-seen tile.
    let mem = state.tribes.get(&1).unwrap();
    assert!(mem.memory_units.contains_key(&seen_idx));
    assert_eq!(mem.memory_units.len(), 1);

    // Two turns later the encoder emits a decayed ghost where the unit was...
    state.settings.turn += 2;
    let raw = state_to_cpu_features(&state, 1).unwrap();
    let expected = memory::MEM_DECAY.powi(2);
    assert!((spatial_at(&raw, CH_MEM_ENEMY_SEEN, 3, 0) - expected).abs() < 1e-6);
    assert!((spatial_at(&raw, CH_MEM_ENEMY_HP, 3, 0) - 1.0).abs() < 1e-6);
    // ...and nothing at the (invisible) real position.
    assert_eq!(spatial_at(&raw, CH_MEM_ENEMY_SEEN, 8, 0), 0.0);
}

#[test]
fn test_memory_suppressed_under_visible_enemy() {
    let mut state = build_state();
    let seen_idx = 3;
    add_enemy_warrior(&mut state, seen_idx);

    state.settings._are_you_sure = true;
    let _ = memory::observe_all(&mut state);
    state.settings._are_you_sure = false;

    // Unit still standing there: live channels cover it, memory stays silent.
    let raw = state_to_cpu_features(&state, 1).unwrap();
    assert_eq!(spatial_at(&raw, CH_MEM_ENEMY_SEEN, 3, 0), 0.0);
}

#[test]
fn test_witnessed_death_clears_ghost() {
    let mut state = build_state();
    let seen_idx = 3;
    add_enemy_warrior(&mut state, seen_idx);

    state.settings._are_you_sure = true;
    let _ = memory::observe_all(&mut state);
    assert!(state.tribes.get(&1).unwrap().memory_units.contains_key(&seen_idx));

    let undo = memory::note_unit_removed(&mut state, seen_idx);
    assert!(!state.tribes.get(&1).unwrap().memory_units.contains_key(&seen_idx));

    undo(&mut state);
    assert!(state.tribes.get(&1).unwrap().memory_units.contains_key(&seen_idx));
}

#[test]
fn test_ghost_pruned_past_horizon() {
    let mut state = build_state();
    add_enemy_warrior(&mut state, 3);

    state.settings._are_you_sure = true;
    let _ = memory::observe_all(&mut state);
    // Encoder skips ghosts past MEM_HORIZON even before pruning runs.
    state.settings.turn += memory::MEM_HORIZON + 1;
    state.tribes.get_mut(&2).unwrap().units.clear();
    let raw = state_to_cpu_features(&state, 1).unwrap();
    assert_eq!(spatial_at(&raw, CH_MEM_ENEMY_SEEN, 3, 0), 0.0);
    // And the next observe pass prunes it from the map entirely.
    let _ = memory::observe_all(&mut state);
    assert!(state.tribes.get(&1).unwrap().memory_units.is_empty());
}

#[test]
fn test_attacked_here_marker() {
    let mut state = build_state();
    let hit_idx = 4;

    state.settings._are_you_sure = true;
    let undo = memory::note_attacked(&mut state, 1, hit_idx);
    state.settings._are_you_sure = false;

    state.settings.turn += 1;
    let raw = state_to_cpu_features(&state, 1).unwrap();
    let expected = memory::MEM_DECAY.powi(1);
    assert!((spatial_at(&raw, CH_MEM_ATTACKED_HERE, 4, 0) - expected).abs() < 1e-6);

    undo(&mut state);
    assert!(state.tribes.get(&1).unwrap().memory_attacks.is_empty());
}

#[test]
fn test_memory_serde_roundtrip() {
    let mut state = build_state();
    add_enemy_warrior(&mut state, 3);
    state.settings._are_you_sure = true;
    let _ = memory::observe_all(&mut state);

    let json = serde_json::to_string(&state).unwrap();
    let loaded: GameState = serde_json::from_str(&json).unwrap();
    let mem = &loaded.tribes.get(&1).unwrap().memory_units;
    assert!(mem.contains_key(&3));
    assert_eq!(mem.get(&3).unwrap().unit_type, UnitType::Warrior);

    // Old JSON without memory keys still loads (serde default).
    let t: TribeState = serde_json::from_str(r#"{"id": 1, "type": 2}"#).map_or_else(
        |_| TribeState::default(),
        |t| t,
    );
    assert!(t.memory_units.is_empty());
}
