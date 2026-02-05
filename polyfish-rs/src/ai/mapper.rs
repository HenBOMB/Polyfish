use crate::moves::Move;
use crate::types::*;
use std::collections::HashMap;
use std::sync::LazyLock;
use strum::IntoEnumIterator;

/// Decomposed training targets for a single move
/// Each field represents the target value for one policy head
#[derive(Debug, Clone)]
pub struct DecomposedTargets {
    pub action_type: usize, // 0-10: which action type (10 is EndTurn/None)
    pub source_spatial: Option<usize>, // 0-899: which source tile (if applicable)
    pub target_spatial: Option<usize>, // 0-899: which target tile (if applicable)
    pub target_type: Option<usize>, // 0-191: unified option head
}

impl Default for DecomposedTargets {
    fn default() -> Self {
        Self {
            action_type: 10,
            source_spatial: None,
            target_spatial: None,
            target_type: None,
        }
    }
}

// ============================================================================
// Robust Mapping Lookups
// ============================================================================

const MAX_STRUCTURES: usize = 48;
const MAX_UNITS: usize = 64;
const MAX_TECHS: usize = 48;
const MAX_ABILITIES: usize = 32;

// Offset constants for 192-sized head
pub const OFFSET_STRUCTURES: usize = 0;
pub const OFFSET_UNITS: usize = 48;
pub const OFFSET_TECHS: usize = 112;
pub const OFFSET_ABILITIES: usize = 160;

static STRUCTURE_MAP: LazyLock<HashMap<StructureType, usize>> = LazyLock::new(|| {
    StructureType::iter()
        .filter(|&s| s != StructureType::None)
        .enumerate()
        .map(|(i, s)| (s, i))
        .collect()
});

static UNIT_MAP: LazyLock<HashMap<UnitType, usize>> = LazyLock::new(|| {
    UnitType::iter()
        .filter(|&u| u != UnitType::None)
        .enumerate()
        .map(|(i, u)| (u, i))
        .collect()
});

static TECH_MAP: LazyLock<HashMap<TechnologyType, usize>> = LazyLock::new(|| {
    TechnologyType::iter()
        .filter(|&t| t != TechnologyType::Unrequired && t != TechnologyType::BeyondComprehension)
        .enumerate()
        .map(|(i, t)| (t, i))
        .collect()
});

static ABILITY_MAP: LazyLock<HashMap<AbilityType, usize>> = LazyLock::new(|| {
    AbilityType::iter()
        .filter(|&a| a != AbilityType::None)
        .enumerate()
        .map(|(i, a)| (a, i))
        .collect()
});

pub struct DecomposedMapper;

impl DecomposedMapper {
    pub fn move_type_to_idx(mt: MoveType) -> usize {
        match mt {
            MoveType::None => 0,
            MoveType::Attack => 1,
            MoveType::Step => 2,
            MoveType::Capture => 3,
            MoveType::Ability => 4,
            MoveType::Summon => 5,
            MoveType::Harvest => 6,
            MoveType::Build => 7,
            MoveType::Research => 8,
            MoveType::Reward => 9,
            MoveType::EndTurn => 10,
        }
    }

    /// Convert a move into training targets for each policy head
    pub fn move_to_targets(m: &dyn Move, map_size: usize) -> DecomposedTargets {
        let move_type = m.move_type();
        let move_index = Self::move_type_to_idx(move_type);

        let source_spatial = m
            .source_idx()
            .ok()
            .map(|idx| Self::tile_to_spatial_idx(idx, map_size));

        let target_spatial = m
            .target_idx()
            .ok()
            .map(|idx| Self::tile_to_spatial_idx(idx, map_size));

        let target_type = match move_type {
            MoveType::Build => m.structure_type().ok().and_then(|s| Self::map_structure(s)),
            MoveType::Summon => m.unit_type().ok().and_then(|u| Self::map_unit(u)),
            MoveType::Research => m.tech_type().ok().and_then(|t| Self::map_tech(t)),
            MoveType::Ability => m.ability_type().ok().and_then(|a| Self::map_ability(a)),
            MoveType::Reward => Some(191), // Last slot for generic rewards
            // Harvesting / Capturing does not require a target type
            _ => None,
        };

        DecomposedTargets {
            action_type: move_index,
            source_spatial,
            target_spatial,
            target_type,
        }
    }

    pub fn move_visit_to_targets(
        mv: &crate::ai::mcts_types::MoveVisit,
        map_size: usize,
    ) -> DecomposedTargets {
        let action_type = Self::move_type_to_idx(mv.move_type);

        let source_spatial = mv
            .source_idx
            .map(|idx| Self::tile_to_spatial_idx(idx, map_size));
        let target_spatial = mv
            .target_idx
            .map(|idx| Self::tile_to_spatial_idx(idx, map_size));

        let move_option = match mv.move_type {
            MoveType::Build => mv.structure_type.and_then(|s| Self::map_structure(s)),
            MoveType::Summon => mv.unit_type.and_then(|u| Self::map_unit(u)),
            MoveType::Research => mv.tech_type.and_then(|t| Self::map_tech(t)),
            MoveType::Ability => mv.ability_type.and_then(|a| Self::map_ability(a)),
            MoveType::Reward => Some(191),
            _ => None,
        };

        DecomposedTargets {
            action_type,
            source_spatial,
            target_spatial,
            target_type: move_option,
        }
    }

    #[inline]
    fn tile_to_spatial_idx(tile_idx: usize, map_size: usize) -> usize {
        let y = tile_idx / map_size;
        let x = tile_idx % map_size;
        y * map_size + x
    }

    pub fn map_structure(s: StructureType) -> Option<usize> {
        if s == StructureType::None {
            return None;
        }
        STRUCTURE_MAP.get(&s).map(|&i| {
            if i >= MAX_STRUCTURES {
                eprintln!(
                    "[Mapper Warning] Structure {:?} index {} exceeds MAX_STRUCTURES {}",
                    s, i, MAX_STRUCTURES
                );
            }
            OFFSET_STRUCTURES + i
        })
    }

    pub fn map_unit(u: UnitType) -> Option<usize> {
        if u == UnitType::None {
            return None;
        }
        UNIT_MAP.get(&u).map(|&i| {
            if i >= MAX_UNITS {
                eprintln!(
                    "[Mapper Warning] Unit {:?} index {} exceeds MAX_UNITS {}",
                    u, i, MAX_UNITS
                );
            }
            OFFSET_UNITS + i
        })
    }

    pub fn map_tech(t: TechnologyType) -> Option<usize> {
        if t == TechnologyType::Unrequired || t == TechnologyType::BeyondComprehension {
            return None;
        }
        TECH_MAP.get(&t).map(|&i| {
            if i >= MAX_TECHS {
                eprintln!(
                    "[Mapper Warning] Tech {:?} index {} exceeds MAX_TECHS {}",
                    t, i, MAX_TECHS
                );
            }
            OFFSET_TECHS + i
        })
    }

    pub fn map_ability(a: AbilityType) -> Option<usize> {
        if a == AbilityType::None {
            return None;
        }
        ABILITY_MAP.get(&a).map(|&i| {
            if i >= MAX_ABILITIES {
                eprintln!(
                    "[Mapper Warning] Ability {:?} index {} exceeds MAX_ABILITIES {}",
                    a, i, MAX_ABILITIES
                );
            }
            OFFSET_ABILITIES + i
        })
    }
}
