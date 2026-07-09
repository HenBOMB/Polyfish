//! Fog memory: per-tribe record of the last foreign unit seen on each tile.
//!
//! Exploration is permanent in this engine, so a unit only leaves a tribe's
//! view by moving onto an unexplored tile or dying. These maps remember what
//! was last seen so `ai::features` can emit decayed "ghost" channels
//! (see notes-memory.md). All mutations happen on real moves only
//! (`_are_you_sure`), never inside MCTS simulations, and return undos.

use crate::actions::UndoCallback;
use crate::states::{GameState, MemUnit, PlayerId};

/// Ghosts older than this many turns are pruned (and not encoded).
pub const MEM_HORIZON: i32 = 8;
/// Per-turn decay of the ghost signal in the feature encoding.
pub const MEM_DECAY: f32 = 0.85;

/// Record what every tribe can see after a real move: upsert a ghost for each
/// foreign unit standing on a tile the observer has explored, drop the ghost
/// left behind at the tile the unit moved away from, and prune stale entries.
pub fn observe_all(state: &mut GameState) -> UndoCallback {
    debug_assert!(state.settings._are_you_sure);
    let turn = state.settings.turn;

    // Phase 1 (immutable): gather (prev_idx, idx, ghost) per observer.
    let observers: Vec<PlayerId> = state.tribes.keys().cloned().collect();
    let mut observations: Vec<(PlayerId, Vec<(i32, i32, MemUnit)>)> = Vec::new();
    for &observer in &observers {
        let mut seen = Vec::new();
        for (owner, tribe) in &state.tribes {
            if *owner == observer {
                continue;
            }
            for unit in &tribe.units {
                let idx = unit.coords.idx;
                let explored = state
                    .tiles
                    .get(&idx)
                    .map(|t| t.explorers.contains(&observer))
                    .unwrap_or(false);
                if !explored {
                    continue;
                }
                let max_hp = crate::functions::get_unit_max_health(unit).max(1.0);
                seen.push((
                    unit.prev_coords.idx,
                    idx,
                    MemUnit {
                        unit_type: unit.unit_type,
                        hp_norm: (unit.health / max_hp).clamp(0.0, 1.0),
                        last_seen_turn: turn,
                    },
                ));
            }
        }
        if !seen.is_empty() {
            observations.push((observer, seen));
        }
    }

    // Snapshot for undo (maps stay tiny thanks to MEM_HORIZON pruning).
    let snapshot: Vec<_> = state
        .tribes
        .iter()
        .map(|(id, t)| (*id, t.memory_units.clone(), t.memory_attacks.clone()))
        .collect();

    // Phase 2 (mutable): apply, then prune.
    for (observer, seen) in observations {
        if let Some(tribe) = state.tribes.get_mut(&observer) {
            for (prev_idx, idx, ghost) in seen {
                if prev_idx != idx {
                    tribe.memory_units.shift_remove(&prev_idx);
                }
                tribe.memory_units.insert(idx, ghost);
            }
        }
    }
    for tribe in state.tribes.values_mut() {
        tribe
            .memory_units
            .retain(|_, m| turn - m.last_seen_turn <= MEM_HORIZON);
        tribe.memory_attacks.retain(|_, t| turn - *t <= MEM_HORIZON);
    }

    Box::new(move |s: &mut GameState| {
        for (id, units, attacks) in snapshot {
            if let Some(t) = s.tribes.get_mut(&id) {
                t.memory_units = units;
                t.memory_attacks = attacks;
            }
        }
    })
}

/// A unit died on `tile_idx`: every tribe that explored that tile witnessed
/// it, so their ghost there is no longer a hiding threat — clear it.
pub fn note_unit_removed(state: &mut GameState, tile_idx: i32) -> UndoCallback {
    let witnesses: Vec<PlayerId> = state
        .tiles
        .get(&tile_idx)
        .map(|t| t.explorers.iter().cloned().collect())
        .unwrap_or_default();
    let mut removed: Vec<(PlayerId, MemUnit)> = Vec::new();
    for w in witnesses {
        if let Some(tribe) = state.tribes.get_mut(&w) {
            if let Some(ghost) = tribe.memory_units.shift_remove(&tile_idx) {
                removed.push((w, ghost));
            }
        }
    }
    Box::new(move |s: &mut GameState| {
        for (w, ghost) in removed {
            if let Some(t) = s.tribes.get_mut(&w) {
                t.memory_units.insert(tile_idx, ghost);
            }
        }
    })
}

/// One of `defender_owner`'s units was hit on `tile_idx` — remember combat
/// happened here even if the attacker was never visible.
pub fn note_attacked(
    state: &mut GameState,
    defender_owner: PlayerId,
    tile_idx: i32,
) -> UndoCallback {
    let turn = state.settings.turn;
    let old = state
        .tribes
        .get_mut(&defender_owner)
        .and_then(|t| t.memory_attacks.insert(tile_idx, turn));
    Box::new(move |s: &mut GameState| {
        if let Some(t) = s.tribes.get_mut(&defender_owner) {
            match old {
                Some(v) => {
                    t.memory_attacks.insert(tile_idx, v);
                }
                None => {
                    t.memory_attacks.shift_remove(&tile_idx);
                }
            }
        }
    })
}
