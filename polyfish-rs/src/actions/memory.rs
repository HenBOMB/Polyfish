//! Observation memory: what each tribe has legitimately seen of its enemies.
//!
//! Follows the `explorers` archetype (see discovery.rs): mutated only on real
//! moves (`_are_you_sure`), permanent, no undo. MCTS simulations never execute
//! enemy actions (the EndTurn fast-forward skips opponent turns), so memory is
//! frozen during search exactly like tile exploration.

use crate::states::{GameState, GhostRecord, PlayerId};
use crate::types::{UnitEffect, UnitType};

/// Ghosts older than this many turns are pruned by the end-of-turn sweep.
const GHOST_MAX_AGE: i32 = 10;

fn tile_explored_by(state: &GameState, idx: i32, observer: PlayerId) -> bool {
    state
        .tiles
        .get(&idx)
        .map(|t| t.explorers.contains(&observer))
        .unwrap_or(false)
}

fn observer_ids(state: &GameState, except: PlayerId) -> Vec<PlayerId> {
    state
        .tribes
        .iter()
        .filter(|(id, t)| **id != except && t.killed_turn == 0 && t.resigned_turn == 0)
        .map(|(id, _)| *id)
        .collect()
}

/// Record what other tribes observed of a unit's completed real move.
/// `full_path` = starting tile + every intermediate/final step tile.
/// Evidence for any observer who saw a path tile; a ghost at the first
/// fog-side tile when the unit's move ended outside the observer's vision.
pub fn observe_unit_step(
    state: &mut GameState,
    unit_owner: PlayerId,
    unit_idx: usize,
    full_path: &[i32],
) {
    if !state.settings._are_you_sure || full_path.is_empty() {
        return;
    }
    // Post-transform type and post-Hide invisibility (call after those blocks).
    let (cur_type, invisible) = match state
        .tribes
        .get(&unit_owner)
        .and_then(|t| t.units.get(unit_idx))
    {
        Some(u) => (u.unit_type, u.effects.contains(&UnitEffect::Invisible)),
        None => return,
    };
    if invisible {
        return;
    }
    let turn = state.settings.turn;
    let final_idx = *full_path.last().unwrap();

    let mut updates: Vec<(PlayerId, Option<i32>)> = Vec::new();
    for o in observer_ids(state, unit_owner) {
        let explored: Vec<bool> = full_path
            .iter()
            .map(|&idx| tile_explored_by(state, idx, o))
            .collect();
        if !explored.iter().any(|&e| e) {
            continue;
        }
        // Ghost only when the move ends in O's fog: the tile right after the
        // last O-visible path tile. Never on an explored tile — permanent
        // vision would immediately supersede it.
        let ghost = if !tile_explored_by(state, final_idx, o) {
            explored
                .iter()
                .rposition(|&e| e)
                .filter(|&i| i + 1 < full_path.len())
                .map(|i| full_path[i + 1])
        } else {
            None
        };
        updates.push((o, ghost));
    }

    for (o, ghost) in updates {
        if let Some(tribe) = state.tribes.get_mut(&o) {
            tribe.enemy_types_seen.insert(cur_type);
            if let Some(g_idx) = ghost {
                tribe.enemy_ghosts.insert(
                    g_idx,
                    GhostRecord {
                        unit_type: cur_type,
                        owner: unit_owner,
                        turn,
                    },
                );
            }
        }
    }
}

/// The defender's tribe observes its attacker (combat is always shown to the
/// victim). If the attacker struck from the defender's fog (e.g. ranged),
/// leave a ghost at the attacker's tile.
pub fn observe_attacker(
    state: &mut GameState,
    attacker_owner: PlayerId,
    attacker_type: UnitType,
    attacker_tile: i32,
    defender_owner: PlayerId,
) {
    if !state.settings._are_you_sure || attacker_owner == defender_owner {
        return;
    }
    let turn = state.settings.turn;
    let from_fog = !tile_explored_by(state, attacker_tile, defender_owner);
    if let Some(tribe) = state.tribes.get_mut(&defender_owner) {
        tribe.enemy_types_seen.insert(attacker_type);
        if from_fog {
            tribe.enemy_ghosts.insert(
                attacker_tile,
                GhostRecord {
                    unit_type: attacker_type,
                    owner: attacker_owner,
                    turn,
                },
            );
        }
    }
}

/// The attacker's tribe observes the defender's type (targets are visible).
pub fn observe_defender(
    state: &mut GameState,
    attacker_owner: PlayerId,
    defender_owner: PlayerId,
    defender_type: UnitType,
) {
    if !state.settings._are_you_sure || attacker_owner == defender_owner {
        return;
    }
    if let Some(tribe) = state.tribes.get_mut(&attacker_owner) {
        tribe.enemy_types_seen.insert(defender_type);
    }
}

/// Evidence for enemy units standing on tiles the observer just explored.
/// `sightings` = (unit owner, tile idx) pairs collected during discovery.
pub fn observe_discovered_units(
    state: &mut GameState,
    observer: PlayerId,
    sightings: &[(PlayerId, i32)],
) {
    if !state.settings._are_you_sure || sightings.is_empty() {
        return;
    }
    let mut types: Vec<UnitType> = Vec::new();
    for &(owner, idx) in sightings {
        if owner == observer {
            continue;
        }
        if let Some(unit) = state
            .tribes
            .get(&owner)
            .and_then(|t| t.units.iter().find(|u| u.coords.idx == idx))
        {
            if !unit.effects.contains(&UnitEffect::Invisible) {
                types.push(unit.unit_type);
            }
        }
    }
    if let Some(tribe) = state.tribes.get_mut(&observer) {
        for ty in types {
            tribe.enemy_types_seen.insert(ty);
        }
    }
}

/// End-of-turn catch-all: evidence for every visible enemy unit (covers
/// spawns, upgrades, growth, conversions that bypass step/attack hooks),
/// drop ghosts whose tile entered permanent vision, prune stale ghosts.
pub fn end_of_turn_sweep(state: &mut GameState) {
    if !state.settings._are_you_sure {
        return;
    }
    let turn_now = state.settings.turn;

    let mut updates: Vec<(PlayerId, Vec<UnitType>, Vec<i32>)> = Vec::new();
    for o in observer_ids(state, 0) {
        let mut evidence = Vec::new();
        for (id, tribe) in &state.tribes {
            if *id == o {
                continue;
            }
            for unit in &tribe.units {
                if !unit.effects.contains(&UnitEffect::Invisible)
                    && tile_explored_by(state, unit.coords.idx, o)
                {
                    evidence.push(unit.unit_type);
                }
            }
        }
        let expired: Vec<i32> = state
            .tribes
            .get(&o)
            .map(|t| {
                t.enemy_ghosts
                    .iter()
                    .filter(|(idx, g)| {
                        tile_explored_by(state, **idx, o) || turn_now - g.turn > GHOST_MAX_AGE
                    })
                    .map(|(idx, _)| *idx)
                    .collect()
            })
            .unwrap_or_default();
        updates.push((o, evidence, expired));
    }

    for (o, evidence, expired) in updates {
        if let Some(tribe) = state.tribes.get_mut(&o) {
            for ty in evidence {
                tribe.enemy_types_seen.insert(ty);
            }
            for idx in expired {
                tribe.enemy_ghosts.remove(&idx);
            }
        }
    }
}
