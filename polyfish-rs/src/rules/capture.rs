//! Capture rules — what counts as a capturable target, and for whom.
//!
//! Twelve variants of this predicate existed across the AI and tooling. They
//! disagreed on whether Ruins count, whether an enemy-owned village counts, and
//! whether the tile must be explored — so "how many targets are there" had
//! twelve different answers. The engine's own rule
//! (`moves::generate_capture_moves`) is the reference.

use crate::states::{GameState, PlayerId};
use crate::types::{ResourceType, StructureType, TechnologyType};

/// Which kinds of target a caller cares about.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CaptureKind {
    /// Neutral villages (`owner == 0`).
    pub neutral_villages: bool,
    /// Villages and cities currently held by an enemy — legally capturable, but
    /// several AI consumers deliberately exclude them from "expansion" targets.
    pub enemy_villages: bool,
    pub ruins: bool,
    pub starfish: bool,
}

impl CaptureKind {
    /// Everything the engine would emit a CaptureMove for.
    pub const ANY: Self = Self {
        neutral_villages: true,
        enemy_villages: true,
        ruins: true,
        starfish: true,
    };
    /// Open expansion targets: unclaimed villages only.
    pub const OPEN_VILLAGE: Self = Self {
        neutral_villages: true,
        enemy_villages: false,
        ruins: false,
        starfish: false,
    };
    /// Unclaimed villages and ruins — the "free stuff on the map" set.
    pub const NEUTRAL: Self = Self {
        neutral_villages: true,
        enemy_villages: false,
        ruins: true,
        starfish: false,
    };
}

/// Is `idx` a capture target of the requested kind for `player`?
///
/// `require_explored` applies the fog rule: AI consumers must not target what
/// they cannot see, while the engine's own generator works from a unit already
/// standing on the tile and does not need it.
pub fn is_capturable(
    state: &GameState,
    idx: i32,
    player: PlayerId,
    kind: CaptureKind,
    require_explored: bool,
) -> bool {
    if require_explored && !crate::functions::is_tile_explored(state, idx, player) {
        return false;
    }
    let owner = state.tiles.get(&idx).map_or(0, |t| t.owner);
    if owner == player {
        return false; // already ours
    }
    let structure = crate::functions::get_structure_at(state, idx).map(|s| s.structure_type);

    match structure {
        Some(StructureType::Village) => {
            if owner == 0 {
                kind.neutral_villages
            } else {
                kind.enemy_villages
            }
        }
        Some(StructureType::Ruin) => kind.ruins,
        _ => {
            kind.starfish
                && state
                    .resources
                    .get(&idx)
                    .and_then(|r| r.as_ref())
                    .is_some_and(|r| r.resource_type == ResourceType::Starfish)
                && state.tribes.get(&player).is_some_and(|t| {
                    crate::settings::technology::has_technology(
                        &t.tech_vanilla,
                        TechnologyType::Navigation,
                    )
                })
        }
    }
}
