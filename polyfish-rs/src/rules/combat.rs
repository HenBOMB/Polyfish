//! Combat rules — one implementation, shared by the engine and every consumer.

use crate::settings::units::get_unit_setting;
use crate::states::{GameState, UnitState};
use crate::types::SkillType;
use rustc_hash::FxHashSet;

/// Kills required before a unit can be promoted to veteran.
pub const PROMOTION_KILLS: i32 = 3;

/// Does the defender strike back?
///
/// Core form: caller supplies the resolved skills, range and distance, so the
/// engine's attack path (which already holds all of them) pays nothing extra.
pub fn can_retaliate_with(
    atk_skills: &FxHashSet<SkillType>,
    def_skills: &FxHashSet<SkillType>,
    def_range: i32,
    distance: i32,
) -> bool {
    !def_skills.contains(&SkillType::Stiff)
        && !atk_skills.contains(&SkillType::Surprise)
        && distance <= def_range
}

/// Does the defender strike back? Resolves settings and positions itself.
///
/// Retaliation is the single largest prediction gap this module closes:
/// `calculate_combat_preview` used to report full retaliation damage against
/// every attacker, so ranged units and `Surprise` units were predicted to die
/// attacking when in reality they take nothing — and `ai::scoring` then priced
/// those attacks as suicide.
pub fn can_retaliate(state: &GameState, attacker: &UnitState, defender: &UnitState) -> bool {
    let atk = get_unit_setting(attacker.unit_type);
    let def = get_unit_setting(defender.unit_type);
    let distance = crate::functions::get_chebyshev_distance(
        attacker.coords.idx,
        defender.coords.idx,
        state.settings.size,
    );
    can_retaliate_with(&atk.skills, &def.skills, def.range, distance)
}

/// What a unit is worth in stars, the way the engine's own score counts it:
/// its cost plus any passenger's, and **zero if it was converted** — a converted
/// unit changes no score for its new owner.
///
/// Ten AI and tooling sites previously summed bare `get_unit_setting(u).cost`,
/// undervaluing loaded carriers and overvaluing captured units.
pub fn unit_worth(unit: &UnitState) -> i32 {
    if unit.converted {
        return 0;
    }
    get_unit_setting(unit.unit_type).cost
        + unit
            .passenger_type
            .map(|p| get_unit_setting(p).cost)
            .unwrap_or(0)
}

/// Total star worth of a tribe's living units.
pub fn army_worth(units: &[UnitState]) -> i32 {
    units.iter().map(unit_worth).sum()
}

/// Has this unit earned a promotion? `Static` units never promote.
pub fn can_promote(unit: &UnitState) -> bool {
    !unit.veteran
        && unit.kills >= PROMOTION_KILLS
        && !get_unit_setting(unit.unit_type)
            .skills
            .contains(&SkillType::Static)
}
