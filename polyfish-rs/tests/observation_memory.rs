//! Observation-memory (enemy_types_seen / enemy_ghosts) behavior tests.

use polyfish::actions::memory::end_of_turn_sweep;
use polyfish::actions::units::{attack_unit, step_unit};
use polyfish::coords::Coords;
use polyfish::states::{GameState, GhostRecord, TileState, TribeState, UnitState};
use polyfish::types::{TerrainType, TribeType, UnitEffect, UnitType};

const OBSERVER: i32 = 1;
const MOVER: i32 = 2;

fn make_unit(owner: i32, unit_type: UnitType, idx: i32, size: i32) -> UnitState {
    let mut u = UnitState::default();
    u.owner = owner;
    u.unit_type = unit_type;
    u.health = 10.0;
    u.coords = Coords::from_index(idx, size);
    u
}

/// Two tribes on an 11x11 field map. Observer has explored tiles 0..=5
/// (row 0, columns 0-5); everything else is fog to it.
fn setup() -> GameState {
    let mut state = GameState::default();
    state.settings.size = 11;
    state.settings.turn = 5;
    state.settings.current_player_turn_id = MOVER;
    state.settings._are_you_sure = true;

    for id in [OBSERVER, MOVER] {
        let mut tribe = TribeState::default();
        tribe.id = id;
        tribe.tribe_type = TribeType::Imperius;
        state.tribes.insert(id, tribe);
    }
    for i in 0..121 {
        let mut tile = TileState::default();
        tile.terrain_type = TerrainType::Field;
        if i <= 5 {
            tile.explorers.insert(OBSERVER);
        }
        tile.explorers.insert(MOVER);
        state.tiles.insert(i, tile);
    }
    state
}

fn add_unit(state: &mut GameState, owner: i32, unit_type: UnitType, idx: i32) -> usize {
    let size = state.settings.size;
    let unit = make_unit(owner, unit_type, idx, size);
    let tribe = state.tribes.get_mut(&owner).unwrap();
    tribe.units.push(unit);
    let unit_idx = tribe.units.len() - 1;
    state.tiles.get_mut(&idx).unwrap()._unit_owner_id = Some(owner);
    unit_idx
}

#[test]
fn ghost_created_on_step_into_fog() {
    let mut state = setup();
    let u = add_unit(&mut state, MOVER, UnitType::Warrior, 5);

    // Tile 5 is observer-explored, tile 6 is observer fog.
    step_unit(&mut state, MOVER, u, 6, false);

    let obs = state.tribes.get(&OBSERVER).unwrap();
    assert!(obs.enemy_types_seen.contains(&UnitType::Warrior));
    let ghost = obs.enemy_ghosts.get(&6).expect("ghost at fog-side tile");
    assert_eq!(ghost.unit_type, UnitType::Warrior);
    assert_eq!(ghost.owner, MOVER);
    assert_eq!(ghost.turn, 5);
}

#[test]
fn no_ghost_when_move_stays_visible() {
    let mut state = setup();
    let u = add_unit(&mut state, MOVER, UnitType::Warrior, 5);

    step_unit(&mut state, MOVER, u, 4, false);

    let obs = state.tribes.get(&OBSERVER).unwrap();
    assert!(obs.enemy_types_seen.contains(&UnitType::Warrior));
    assert!(obs.enemy_ghosts.is_empty());
}

#[test]
fn no_memory_in_simulation() {
    let mut state = setup();
    state.settings._are_you_sure = false;
    let u = add_unit(&mut state, MOVER, UnitType::Warrior, 5);

    step_unit(&mut state, MOVER, u, 6, false);

    let obs = state.tribes.get(&OBSERVER).unwrap();
    assert!(obs.enemy_types_seen.is_empty());
    assert!(obs.enemy_ghosts.is_empty());
}

#[test]
fn invisible_unit_leaves_no_memory() {
    let mut state = setup();
    let u = add_unit(&mut state, MOVER, UnitType::Cloak, 5);
    state.tribes.get_mut(&MOVER).unwrap().units[u]
        .effects
        .insert(UnitEffect::Invisible);

    step_unit(&mut state, MOVER, u, 6, false);

    let obs = state.tribes.get(&OBSERVER).unwrap();
    assert!(obs.enemy_types_seen.is_empty());
    assert!(obs.enemy_ghosts.is_empty());
}

#[test]
fn attack_records_both_directions_and_fog_ghost() {
    let mut state = setup();
    let def = add_unit(&mut state, OBSERVER, UnitType::Warrior, 5);
    // Attacker strikes from tile 6, which the observer has NOT explored.
    let atk = add_unit(&mut state, MOVER, UnitType::Archer, 6);

    attack_unit(&mut state, MOVER, atk, OBSERVER, Some(def));

    let obs = state.tribes.get(&OBSERVER).unwrap();
    assert!(obs.enemy_types_seen.contains(&UnitType::Archer));
    let ghost = obs.enemy_ghosts.get(&6).expect("ghost at fog attacker tile");
    assert_eq!(ghost.unit_type, UnitType::Archer);

    let mover = state.tribes.get(&MOVER).unwrap();
    assert!(mover.enemy_types_seen.contains(&UnitType::Warrior));
}

#[test]
fn sweep_collects_evidence_and_prunes_ghosts() {
    let mut state = setup();
    add_unit(&mut state, MOVER, UnitType::Rider, 3); // visible to observer

    let turn = state.settings.turn; // 5
    {
        let obs = state.tribes.get_mut(&OBSERVER).unwrap();
        // Ghost on an explored tile: permanent vision supersedes -> dropped.
        obs.enemy_ghosts.insert(
            2,
            GhostRecord { unit_type: UnitType::Knight, owner: MOVER, turn },
        );
        // Stale ghost in fog: pruned by age.
        obs.enemy_ghosts.insert(
            100,
            GhostRecord { unit_type: UnitType::Knight, owner: MOVER, turn: turn - 11 },
        );
        // Fresh ghost in fog: kept.
        obs.enemy_ghosts.insert(
            90,
            GhostRecord { unit_type: UnitType::Knight, owner: MOVER, turn },
        );
    }

    end_of_turn_sweep(&mut state);

    let obs = state.tribes.get(&OBSERVER).unwrap();
    assert!(obs.enemy_types_seen.contains(&UnitType::Rider));
    assert!(!obs.enemy_ghosts.contains_key(&2));
    assert!(!obs.enemy_ghosts.contains_key(&100));
    assert!(obs.enemy_ghosts.contains_key(&90));
}

#[test]
fn obscure_fog_strips_other_tribes_memory() {
    let mut state = setup();
    for id in [OBSERVER, MOVER] {
        let tribe = state.tribes.get_mut(&id).unwrap();
        tribe.enemy_types_seen.insert(UnitType::Knight);
        tribe.enemy_ghosts.insert(
            50,
            GhostRecord { unit_type: UnitType::Knight, owner: 3, turn: 1 },
        );
    }

    state.obscure_fog(OBSERVER);

    let pov = state.tribes.get(&OBSERVER).unwrap();
    assert!(!pov.enemy_types_seen.is_empty());
    assert!(!pov.enemy_ghosts.is_empty());
    let other = state.tribes.get(&MOVER).unwrap();
    assert!(other.enemy_types_seen.is_empty());
    assert!(other.enemy_ghosts.is_empty());
}

#[test]
fn serde_backward_compat_and_round_trip() {
    // Old JSON without the new fields -> defaults to empty.
    let mut v = serde_json::to_value(TribeState::default()).unwrap();
    let obj = v.as_object_mut().unwrap();
    obj.remove("enemyTypesSeen");
    obj.remove("enemyGhosts");
    let old: TribeState = serde_json::from_value(v).unwrap();
    assert!(old.enemy_types_seen.is_empty());
    assert!(old.enemy_ghosts.is_empty());

    // Populated memory survives the JSON round-trip (clone_game path).
    let mut tribe = TribeState::default();
    tribe.enemy_types_seen.insert(UnitType::Catapult);
    tribe.enemy_ghosts.insert(
        7,
        GhostRecord { unit_type: UnitType::Catapult, owner: 2, turn: 9 },
    );
    let json = serde_json::to_string(&tribe).unwrap();
    let back: TribeState = serde_json::from_str(&json).unwrap();
    assert!(back.enemy_types_seen.contains(&UnitType::Catapult));
    assert_eq!(back.enemy_ghosts.get(&7).unwrap().turn, 9);
}
