//! Vision rules — one implementation of "how far can this see".

use crate::settings::units::get_unit_setting;
use crate::states::{GameState, UnitState};
use crate::types::{SkillType, TerrainType};
use rustc_hash::FxHashSet;

/// A unit sees 2 tiles if it is a Scout or standing on a Mountain, else 1.
///
/// Core form: caller supplies the two facts, so the exploration loop — which
/// already has the unit's settings in hand — does no extra lookup.
pub fn unit_vision_range_with(skills: &FxHashSet<SkillType>, on_mountain: bool) -> i32 {
    if skills.contains(&SkillType::Scout) || on_mountain {
        2
    } else {
        1
    }
}

/// A unit's vision range. Five copies of this rule existed and one of them — the
/// post-upgrade reveal — had dropped the Mountain clause entirely, so a unit
/// upgraded on a mountain revealed a radius-1 ring instead of radius 2.
pub fn unit_vision_range(state: &GameState, unit: &UnitState) -> i32 {
    let on_mountain = state
        .tiles
        .get(&unit.coords.idx)
        .is_some_and(|t| t.terrain_type == TerrainType::Mountain);
    unit_vision_range_with(&get_unit_setting(unit.unit_type).skills, on_mountain)
}
