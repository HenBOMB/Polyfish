//! Technology settings and tech tree

use crate::{
    settings::get_unit_setting,
    types::{
        AbilityType, ResourceType, SkillType, StructureType, TaskType, TechnologyType, TerrainType,
        TribeType, UnitType,
    },
};

/// Technology configuration
#[derive(Debug, Clone, Default)]
pub struct TechnologySetting {
    pub tier: Option<i32>,
    pub requires: Option<TechnologyType>,
    pub replaces_tech: Option<TechnologyType>,
    pub tribe_type: Option<TribeType>,
    pub next: Vec<TechnologyType>,
    pub unlocks_structure: Option<StructureType>,
    pub unlocks_special_structures: Vec<StructureType>,
    pub unlocks_task: Vec<TaskType>,
    pub unlocks_ability: Option<AbilityType>,
    pub unlocks_unit: Option<UnitType>,
    pub unlocks_special_units: Vec<UnitType>,
    /// Terrain this tech grants the 1.5x defense bonus on. Must mirror the
    /// engine's `functions::get_defense_bonus` (pinned by test).
    pub defense_bonus_terrain: Vec<TerrainType>,
    /// Reveals all capital positions (Diplomacy).
    pub unlocks_vision: bool,
    /// Discounts all later tech purchases (Philosophy — see `get_tech_cost`).
    pub tech_discount: bool,
    pub unlocks_terrain: Option<TerrainType>,
}

/// Get technology settings by type — cached, returns a shared `'static`
/// reference built once per tech type (no per-call struct/Vec allocation).
pub fn get_technology_setting(tech_type: TechnologyType) -> &'static TechnologySetting {
    static TABLE: std::sync::LazyLock<rustc_hash::FxHashMap<TechnologyType, TechnologySetting>> =
        std::sync::LazyLock::new(|| {
            use strum::IntoEnumIterator;
            TechnologyType::iter().map(|t| (t, build_technology_setting(t))).collect()
        });
    &TABLE[&tech_type]
}

/// Cost/score tier of a technology.
///
/// Tribe replacement techs carry no `tier` of their own — they inherit the tier
/// of the vanilla tech they replace, so a replacement is never cheaper than the
/// thing it stands in for. Reading `.tier.unwrap_or(1)` directly priced all 13
/// of them as tier 1; go through here instead.
pub fn tech_tier(tech_type: TechnologyType) -> i32 {
    let settings = get_technology_setting(tech_type);
    if let Some(tier) = settings.tier {
        return tier;
    }
    settings
        .replaces_tech
        .and_then(|vanilla| get_technology_setting(vanilla).tier)
        .unwrap_or(1)
}

/// True when this tech prices as tier 3 (see [`tech_tier`]).
pub fn is_tier3(tech_type: TechnologyType) -> bool {
    tech_tier(tech_type) == 3
}

/// Build the settings for one tech type (called once per type at table init).
fn build_technology_setting(tech_type: TechnologyType) -> TechnologySetting {
    use TechnologyType::*;

    match tech_type {
        BeyondComprehension => TechnologySetting::default(),

        Basic => TechnologySetting {
            tier: Some(0),
            next: vec![Riding, Organization, Climbing, Fishing, Hunting],
            unlocks_unit: Some(UnitType::Warrior),
            unlocks_task: vec![
                TaskType::Explorer,
                TaskType::Killer,
                TaskType::Network,
                TaskType::Metropolis,
            ],
            ..Default::default()
        },

        Riding => TechnologySetting {
            tier: Some(1),
            next: vec![FreeSpirit, Roads],
            unlocks_unit: Some(UnitType::Rider),
            unlocks_special_units: vec![UnitType::Hexapod, UnitType::Amphibian],
            ..Default::default()
        },
        Roads => TechnologySetting {
            tier: Some(2),
            requires: Some(Riding),
            next: vec![Trade],
            unlocks_structure: Some(StructureType::Road),
            unlocks_special_structures: vec![StructureType::Bridge, StructureType::Mycelium],
            ..Default::default()
        },
        Trade => TechnologySetting {
            tier: Some(3),
            requires: Some(Roads),
            unlocks_structure: Some(StructureType::Market),
            unlocks_special_structures: vec![StructureType::Clathrus],
            unlocks_task: vec![TaskType::Wealth],
            ..Default::default()
        },
        FreeSpirit => TechnologySetting {
            tier: Some(2),
            requires: Some(Riding),
            unlocks_structure: Some(StructureType::Temple),
            unlocks_ability: Some(AbilityType::Disband),
            next: vec![Chivalry],
            ..Default::default()
        },
        Chivalry => TechnologySetting {
            tier: Some(3),
            requires: Some(FreeSpirit),
            unlocks_unit: Some(UnitType::Knight),
            unlocks_special_units: vec![UnitType::Tridention],
            unlocks_ability: Some(AbilityType::Destroy),
            ..Default::default()
        },

        Organization => TechnologySetting {
            tier: Some(1),
            next: vec![Strategy, Farming],
            ..Default::default()
        },
        Farming => TechnologySetting {
            tier: Some(2),
            requires: Some(Organization),
            next: vec![Construction],
            unlocks_structure: Some(StructureType::Farm),
            unlocks_special_structures: vec![StructureType::Fungi],
            ..Default::default()
        },
        Construction => TechnologySetting {
            tier: Some(3),
            requires: Some(Farming),
            unlocks_structure: Some(StructureType::Windmill),
            unlocks_ability: Some(AbilityType::BurnForest),
            ..Default::default()
        },
        Strategy => TechnologySetting {
            tier: Some(2),
            requires: Some(Organization),
            next: vec![Diplomacy],
            unlocks_unit: Some(UnitType::Defender),
            unlocks_special_units: vec![UnitType::Kiton],
            ..Default::default()
        },
        Diplomacy => TechnologySetting {
            tier: Some(3),
            requires: Some(Strategy),
            unlocks_unit: Some(UnitType::Cloak),
            unlocks_structure: Some(StructureType::Embassy),
            unlocks_vision: true,
            ..Default::default()
        },

        Climbing => TechnologySetting {
            tier: Some(1),
            next: vec![Mining, Meditation],
            unlocks_terrain: Some(TerrainType::Mountain),
            defense_bonus_terrain: vec![TerrainType::Mountain],
            ..Default::default()
        },
        Mining => TechnologySetting {
            tier: Some(2),
            requires: Some(Climbing),
            next: vec![Smithery],
            unlocks_structure: Some(StructureType::Mine),
            ..Default::default()
        },
        Smithery => TechnologySetting {
            tier: Some(3),
            requires: Some(Mining),
            unlocks_structure: Some(StructureType::Forge),
            unlocks_unit: Some(UnitType::Swordsman),
            unlocks_special_units: vec![UnitType::Mantis],
            ..Default::default()
        },
        Meditation => TechnologySetting {
            tier: Some(2),
            requires: Some(Climbing),
            next: vec![Philosophy],
            unlocks_task: vec![TaskType::Pacifist],
            ..Default::default()
        },
        Philosophy => TechnologySetting {
            tier: Some(3),
            requires: Some(Meditation),
            unlocks_unit: Some(UnitType::MindBender),
            unlocks_special_units: vec![UnitType::Shaman],
            tech_discount: true,
            unlocks_task: vec![TaskType::Genius],
            unlocks_structure: Some(StructureType::MountainTemple),
            ..Default::default()
        },

        Fishing => TechnologySetting {
            tier: Some(1),
            next: vec![Sailing, Ramming],
            unlocks_unit: Some(UnitType::Raft),
            unlocks_terrain: Some(TerrainType::Water),
            unlocks_structure: Some(StructureType::Port),
            ..Default::default()
        },
        Sailing => TechnologySetting {
            tier: Some(2),
            requires: Some(Fishing),
            next: vec![Navigation],
            unlocks_unit: Some(UnitType::Scoutship),
            unlocks_terrain: Some(TerrainType::Ocean),
            ..Default::default()
        },
        Navigation => TechnologySetting {
            tier: Some(3),
            requires: Some(Ramming),
            unlocks_unit: Some(UnitType::Bomber),
            ..Default::default()
        },
        Ramming => TechnologySetting {
            tier: Some(2),
            requires: Some(Fishing),
            next: vec![Aquatism],
            unlocks_unit: Some(UnitType::Rammership),
            ..Default::default()
        },
        Aquatism => TechnologySetting {
            tier: Some(3),
            requires: Some(Ramming),
            unlocks_structure: Some(StructureType::WaterTemple),
            defense_bonus_terrain: vec![TerrainType::Water, TerrainType::Ocean],
            ..Default::default()
        },

        Hunting => TechnologySetting {
            tier: Some(1),
            next: vec![Archery, Forestry],
            ..Default::default()
        },
        Archery => TechnologySetting {
            tier: Some(2),
            requires: Some(Hunting),
            next: vec![Spiritualism],
            unlocks_unit: Some(UnitType::Archer),
            defense_bonus_terrain: vec![TerrainType::Forest],
            unlocks_special_units: vec![UnitType::Phychi, UnitType::IceArcher],
            ..Default::default()
        },
        Spiritualism => TechnologySetting {
            tier: Some(3),
            requires: Some(Archery),
            unlocks_structure: Some(StructureType::ForestTemple),
            unlocks_ability: Some(AbilityType::GrowForest),
            ..Default::default()
        },
        Forestry => TechnologySetting {
            tier: Some(2),
            requires: Some(Hunting),
            unlocks_structure: Some(StructureType::LumberHut),
            unlocks_ability: Some(AbilityType::ClearForest),
            unlocks_special_structures: vec![StructureType::Sanctuary],
            next: vec![Mathematics],
            ..Default::default()
        },
        Mathematics => TechnologySetting {
            tier: Some(3),
            requires: Some(Forestry),
            unlocks_structure: Some(StructureType::Sawmill),
            unlocks_unit: Some(UnitType::Catapult),
            unlocks_special_units: vec![UnitType::Exida],
            ..Default::default()
        },

        // Polaris replacements
        Frostwork => TechnologySetting {
            replaces_tech: Some(Fishing),
            tribe_type: Some(TribeType::Polaris),
            next: vec![Polarism],
            unlocks_unit: Some(UnitType::Mooni),
            ..Default::default()
        },
        Sledding => TechnologySetting {
            replaces_tech: Some(Sailing),
            tribe_type: Some(TribeType::Polaris),
            next: vec![PolarWarfare],
            unlocks_unit: Some(UnitType::BattleSled),
            ..Default::default()
        },
        IceFishing => TechnologySetting {
            replaces_tech: Some(Ramming),
            tribe_type: Some(TribeType::Polaris),
            next: vec![Polarism],
            ..Default::default()
        },
        PolarWarfare => TechnologySetting {
            replaces_tech: Some(Navigation),
            tribe_type: Some(TribeType::Polaris),
            unlocks_unit: Some(UnitType::IceFortress),
            ..Default::default()
        },
        Polarism => TechnologySetting {
            replaces_tech: Some(Aquatism),
            tribe_type: Some(TribeType::Polaris),
            unlocks_structure: Some(StructureType::IceTemple),
            defense_bonus_terrain: vec![TerrainType::Water, TerrainType::Ocean],
            ..Default::default()
        },

        // Cymanti replacements
        Recycling => TechnologySetting {
            replaces_tech: Some(Construction),
            tribe_type: Some(TribeType::Cymanti),
            ..Default::default()
        },
        Hydrology => TechnologySetting {
            replaces_tech: Some(Sailing),
            tribe_type: Some(TribeType::Cymanti),
            next: vec![Navigation],
            unlocks_special_units: vec![UnitType::Raychi, UnitType::Boomchi],
            unlocks_special_structures: vec![StructureType::Algae],
            ..Default::default()
        },
        Rituals => TechnologySetting {
            replaces_tech: Some(Meditation),
            tribe_type: Some(TribeType::Cymanti),
            tier: Some(2),
            requires: Some(Climbing),
            next: vec![Philosophy],
            unlocks_task: vec![TaskType::Converter],
            ..Default::default()
        },
        ShockTactics => TechnologySetting {
            replaces_tech: Some(Chivalry),
            tribe_type: Some(TribeType::Cymanti),
            unlocks_unit: Some(UnitType::Doomux),
            ..Default::default()
        },
        Oceantology => TechnologySetting {
            replaces_tech: Some(Navigation),
            tribe_type: Some(TribeType::Cymanti),
            unlocks_unit: Some(UnitType::LivingIsland),
            ..Default::default()
        },
        Synergy => TechnologySetting {
            replaces_tech: Some(Diplomacy),
            tribe_type: Some(TribeType::Cymanti),
            unlocks_unit: Some(UnitType::Moth),
            ..Default::default()
        },

        // Aquarion replacements
        Spearing => TechnologySetting {
            replaces_tech: Some(Chivalry),
            tribe_type: Some(TribeType::Aquarion),
            unlocks_unit: Some(UnitType::Tridention),
            ..Default::default()
        },
        Waterways => TechnologySetting {
            replaces_tech: Some(Roads),
            requires: Some(Riding),
            tribe_type: Some(TribeType::Aquarion),
            ..Default::default()
        },
        FreeDiving => TechnologySetting {
            replaces_tech: Some(FreeSpirit),
            tribe_type: Some(TribeType::Aquarion),
            next: vec![Chivalry],
            unlocks_terrain: Some(TerrainType::Ocean),
            ..Default::default()
        },

        // Elyrion replacement
        ForestMagic => TechnologySetting {
            replaces_tech: Some(Hunting),
            tribe_type: Some(TribeType::Elyrion),
            next: vec![Archery, Forestry],
            ..Default::default()
        },
    }
}

/// Functional annotation of one technology — everything it unlocks, grouped
/// by what it is FOR. Derived from the settings tables (units/structures/
/// resources), so it cannot drift from the engine. EXP_ELO_028 UNLOCK
/// groundwork: the macro script and labelers query this, not raw settings.
#[derive(Debug, Clone, Default)]
pub struct TechEffects {
    /// Unlocked standard units that can fight (attack > 0).
    pub combat_units: Vec<UnitType>,
    /// Unlocked standard non-combat units (transport / healer / converter).
    pub support_units: Vec<UnitType>,
    /// Special-tribe variants (Shaman, Hexapod, …) — listed but deliberately
    /// excluded from the class predicates: vanilla-tribe training never
    /// fields them; use `get_unlocked_units(tech, tribe)` for per-tribe truth.
    pub special_units: Vec<UnitType>,
    /// Terrain granted the 1.5x defense bonus (mirrors `get_defense_bonus`).
    pub defense_bonus_terrain: Vec<TerrainType>,
    /// Resources this tech makes harvestable (`ResourceSetting.tech_required`).
    pub harvests: Vec<ResourceType>,
    /// Unlocked structures paying population or stars.
    pub eco_structures: Vec<StructureType>,
    /// Unlocked structures paying score only (temples).
    pub score_structures: Vec<StructureType>,
    /// Terrain made passable (Climbing/Fishing/Sailing).
    pub mobility_terrain: Option<TerrainType>,
    /// Unlocked connector structures (roads, bridges).
    pub mobility_structures: Vec<StructureType>,
    /// Unlocked structures with no yield in this engine (e.g. Embassy).
    pub other_structures: Vec<StructureType>,
    pub abilities: Vec<AbilityType>,
    pub tasks: Vec<TaskType>,
    pub capital_vision: bool,
    pub tech_discount: bool,
}

/// Get the derived annotation for a tech — cached `'static` like
/// `get_technology_setting`.
pub fn get_tech_effects(tech_type: TechnologyType) -> &'static TechEffects {
    static TABLE: std::sync::LazyLock<rustc_hash::FxHashMap<TechnologyType, TechEffects>> =
        std::sync::LazyLock::new(|| {
            use strum::IntoEnumIterator;
            TechnologyType::iter().map(|t| (t, build_tech_effects(t))).collect()
        });
    &TABLE[&tech_type]
}

fn build_tech_effects(tech_type: TechnologyType) -> TechEffects {
    use strum::IntoEnumIterator;
    let s = get_technology_setting(tech_type);
    let mut e = TechEffects {
        defense_bonus_terrain: s.defense_bonus_terrain.clone(),
        mobility_terrain: s.unlocks_terrain,
        abilities: s.unlocks_ability.iter().copied().collect(),
        tasks: s.unlocks_task.clone(),
        capital_vision: s.unlocks_vision,
        tech_discount: s.tech_discount,
        ..Default::default()
    };
    e.special_units = s.unlocks_special_units.clone();
    if let Some(u) = s.unlocks_unit {
        if get_unit_setting(u).attack > 0.0 {
            e.combat_units.push(u);
        } else {
            e.support_units.push(u);
        }
    }
    for st in s.unlocks_structure.iter().chain(&s.unlocks_special_structures) {
        let ss = crate::settings::structures::get_structure_setting(*st);
        let connector = matches!(
            st,
            StructureType::Road | StructureType::Bridge | StructureType::Mycelium
        );
        if connector {
            e.mobility_structures.push(*st);
        } else if ss.reward_pop > 0 || ss.reward_stars > 0 {
            e.eco_structures.push(*st);
        } else if ss.reward_score > 0 {
            e.score_structures.push(*st);
        } else {
            e.other_structures.push(*st);
        }
    }
    for r in ResourceType::iter() {
        let rs = crate::settings::resources::get_resource_setting(r);
        // Tribe-locked resources (Spores) must not classify a generic tech.
        if r != ResourceType::None && rs.tech_required == tech_type && rs.tribe_type.is_none() {
            e.harvests.push(r);
        }
    }
    e
}

/// Military tech: fields combat units or grants a terrain defense bonus.
pub fn is_military_tech(tech_type: TechnologyType) -> bool {
    let e = get_tech_effects(tech_type);
    !e.combat_units.is_empty() || !e.defense_bonus_terrain.is_empty()
}

/// Economy tech: opens harvests, yield structures, or the tech discount.
pub fn is_eco_tech(tech_type: TechnologyType) -> bool {
    let e = get_tech_effects(tech_type);
    !e.harvests.is_empty() || !e.eco_structures.is_empty() || e.tech_discount
}

/// v7: a tier-3 tech is ECONOMIC when the structure it unlocks yields
/// population or stars — Construction/Mathematics/Smithery/Trade/Philosophy/
/// Aquatism/Spiritualism in, Chivalry/Navigation/Diplomacy out. Derived from
/// the settings tables rather than a hand-written list, so it stays correct if
/// the tables move (the exact discipline `max_affordable_pop` failed at).
pub fn is_eco_tier3(tech_type: TechnologyType) -> bool {
    let s = get_technology_setting(tech_type);
    if !is_tier3(tech_type) {
        return false;
    }
    s.unlocks_structure.map_or(false, |st| {
        let ss = crate::settings::structures::get_structure_setting(st);
        ss.reward_pop > 0 || ss.reward_stars > 0
    })
}

fn terrain_is_water(terrain: TerrainType) -> bool {
    matches!(terrain, TerrainType::Water | TerrainType::Ocean)
}

/// A hull, not an amphibian: `Float`/`Water` units cannot leave the water,
/// whereas `Amphibious`/`Skate` units fight on land too.
fn unit_is_naval(unit_type: UnitType) -> bool {
    let s = get_unit_setting(unit_type);
    s.skills.contains(&SkillType::Float) || s.skills.contains(&SkillType::Water)
}

fn structure_is_water_only(structure_type: StructureType) -> bool {
    let s = crate::settings::structures::get_structure_setting(structure_type);
    !s.terrain_types.is_empty() && s.terrain_types.iter().all(|t| terrain_is_water(*t))
}

/// True when EVERY unlock a tech grants is water-bound — naval hulls,
/// water-only structures, water/ocean passage, a water defense bonus — and
/// none of it is usable on land. Techs that grant nothing of their own (tribe
/// stand-ins such as IceFishing) inherit the class of the tech they replace.
/// Table-derived, pinned by `water_tech_classification_is_table_derived`.
pub fn is_water_tech(tech_type: TechnologyType) -> bool {
    let s = get_technology_setting(tech_type);
    let (mut grants, mut water) = (0usize, 0usize);

    if let Some(t) = s.unlocks_terrain {
        grants += 1;
        water += terrain_is_water(t) as usize;
    }
    for st in s.unlocks_structure.iter().chain(s.unlocks_special_structures.iter()) {
        grants += 1;
        water += structure_is_water_only(*st) as usize;
    }
    for u in s.unlocks_unit.iter().chain(s.unlocks_special_units.iter()) {
        grants += 1;
        water += unit_is_naval(*u) as usize;
    }
    if !s.defense_bonus_terrain.is_empty() {
        grants += 1;
        water += s.defense_bonus_terrain.iter().all(|t| terrain_is_water(*t)) as usize;
    }
    grants += s.unlocks_task.len();
    grants += s.unlocks_ability.is_some() as usize;
    grants += s.unlocks_vision as usize;
    grants += s.tech_discount as usize;

    if grants == 0 {
        return s
            .replaces_tech
            .map_or(false, |base| base != tech_type && is_water_tech(base));
    }
    water == grants
}

/// A water tech with nothing but more water behind it, so masking it out on a
/// map with no water costs the tribe no reachable option. Aquarion's FreeDiving
/// is water-bound but gates Chivalry, so it deliberately survives this check.
pub fn is_water_dead_end(tech_type: TechnologyType) -> bool {
    is_water_tech(tech_type)
        && get_technology_setting(tech_type)
            .next
            .iter()
            .all(|n| is_water_tech(*n))
}

/// Mobility tech: opens terrain passage or connector structures.
pub fn is_mobility_tech(tech_type: TechnologyType) -> bool {
    let e = get_tech_effects(tech_type);
    e.mobility_terrain.is_some() || !e.mobility_structures.is_empty()
}

/// Get the tech cost based on number of cities
pub fn get_tech_cost(num_cities: i32, tier: i32, has_philosophy: bool) -> i32 {
    // Polytopia official: 4 + (cities * tier)
    let cost = 4 + (num_cities * tier);
    if has_philosophy {
        // Philosophy gives ~33% discount
        ((cost as f32) * 0.66).ceil() as i32
    } else {
        cost
    }
}

/// Check if a technology is researched by a tribe
pub fn has_technology(tech_list: &[crate::states::TechnologyState], tech: TechnologyType) -> bool {
    tech_list
        .iter()
        .any(|t| t.tech_type == tech && t.discovered)
}

/// Check if a technology is present in the tribe's tech list, even if not yet "discovered"
/// (Used for unlocking units/actions during MCTS simulations)
pub fn is_tech_unlocked(
    tech_list: &[crate::states::TechnologyState],
    tech: TechnologyType,
) -> bool {
    tech_list.iter().any(|t| t.tech_type == tech)
}

/// Helper to find which technology unlocks a specific unit
pub fn get_tech_unlocking_unit(unit_type: UnitType) -> Option<TechnologyType> {
    use strum::IntoEnumIterator;
    for tech in TechnologyType::iter() {
        let settings = get_technology_setting(tech);
        if settings.unlocks_unit == Some(unit_type)
            || settings.unlocks_special_units.contains(&unit_type)
        {
            return Some(tech);
        }
    }
    None
}

/// Resolve technology replacement for a specific tribe
pub fn resolve_tech_for_tribe(tech: TechnologyType, tribe: TribeType) -> TechnologyType {
    use strum::IntoEnumIterator;
    for candidate_tech in TechnologyType::iter() {
        let settings = get_technology_setting(candidate_tech);
        if settings.replaces_tech == Some(tech) && settings.tribe_type == Some(tribe) {
            return candidate_tech;
        }
    }

    tech
}

/// Get a list of technologies that are currently researchable by a tribe
pub fn get_researchable_techs(
    tech_list: &[crate::states::TechnologyState],
    tribe_type: TribeType,
) -> Vec<TechnologyType> {
    use strum::IntoEnumIterator;
    let mut researchable = Vec::new();

    for tech in TechnologyType::iter() {
        if tech == TechnologyType::Basic || tech == TechnologyType::BeyondComprehension {
            continue;
        }

        // Standard techs are 1-24 range in current implementation (mostly)
        // But we should use the settings to check prerequisites.
        let resolved = resolve_tech_for_tribe(tech, tribe_type);

        if !has_technology(tech_list, resolved) {
            let settings = get_technology_setting(resolved);
            if let Some(req) = settings.requires {
                // To be researchable, prerequisite must be RESEARCHED
                if has_technology(tech_list, req) {
                    researchable.push(resolved);
                }
            } else if settings.tier == Some(1) {
                // Tier 1 techs are always researchable
                researchable.push(resolved);
            }
        }
    }
    researchable
}

/// Get all units unlocked by a technology for a specific tribe
pub fn get_unlocked_units(tech_type: TechnologyType, tribe: TribeType) -> Vec<UnitType> {
    let settings = get_technology_setting(tech_type);
    let mut units = Vec::new();

    // 1. Filter out techs that belong to other special tribes
    if let Some(tech_tribe) = settings.tribe_type {
        if tech_tribe != tribe {
            return units;
        }
    }

    // 2. Add the standard unit if it exists and isn't replaced for this tribe
    if let Some(u) = settings.unlocks_unit {
        if !is_unit_replaced_for_tribe(u, tribe) {
            units.push(u);
        }
    }

    // 3. Add any special units that belong to this tribe
    for &u in &settings.unlocks_special_units {
        if get_unit_setting(u).tribe_type == tribe {
            units.push(u);
        }
    }

    units
}

/// Check if a standard unit type is replaced for a specific tribe
pub fn is_unit_replaced_for_tribe(unit: UnitType, tribe: TribeType) -> bool {
    use TribeType as T;
    use UnitType as U;

    match tribe {
        T::Cymanti => matches!(
            unit,
            U::Rider
                | U::Defender
                | U::Knight
                | U::Swordsman
                | U::Catapult
                | U::MindBender
                | U::Giant
                | U::Archer
        ),
        T::Aquarion => matches!(unit, U::Rider | U::Knight | U::Giant),
        T::Polaris => matches!(unit, U::MindBender | U::Giant | U::Archer | U::Catapult),
        T::Nature => matches!(unit, U::Giant), // Elyrion
        _ => false,
    }
}

#[cfg(test)]
mod effects_tests {
    use super::*;
    use strum::IntoEnumIterator;

    /// The defense annotation and `functions::get_defense_bonus` implement
    /// the same rule table — this pin fails if either side changes alone.
    #[test]
    fn defense_annotation_matches_the_engine_rule_table() {
        use TechnologyType as T;
        use TerrainType as G;
        let def = |t| &get_tech_effects(t).defense_bonus_terrain;
        assert_eq!(def(T::Archery), &vec![G::Forest]);
        assert_eq!(def(T::Aquatism), &vec![G::Water, G::Ocean]);
        assert_eq!(def(T::Climbing), &vec![G::Mountain]);
        for t in TechnologyType::iter() {
            if !matches!(t, T::Archery | T::Aquatism | T::Climbing | T::Polarism) {
                assert!(def(t).is_empty(), "{t:?} claims a defense bonus the engine lacks");
            }
        }
    }

    /// Every vanilla (tribe-agnostic, tiered) tech must annotate at least one
    /// concrete effect — a blank entry means the lookup went stale.
    #[test]
    fn every_vanilla_tech_has_a_nonempty_annotation() {
        for t in TechnologyType::iter() {
            let s = get_technology_setting(t);
            if s.tier.is_none() || s.tribe_type.is_some() {
                continue;
            }
            let e = get_tech_effects(t);
            let nonempty = !e.combat_units.is_empty()
                || !e.support_units.is_empty()
                || !e.defense_bonus_terrain.is_empty()
                || !e.harvests.is_empty()
                || !e.eco_structures.is_empty()
                || !e.score_structures.is_empty()
                || e.mobility_terrain.is_some()
                || !e.mobility_structures.is_empty()
                || !e.other_structures.is_empty()
                || !e.abilities.is_empty()
                || !e.tasks.is_empty()
                || e.capital_vision
                || e.tech_discount;
            assert!(nonempty, "{t:?} has no annotated effects");
        }
    }

    #[test]
    fn class_predicates_match_canonical_examples() {
        use TechnologyType as T;
        // Pure economy: harvest/structure techs.
        assert!(is_eco_tech(T::Farming) && !is_military_tech(T::Farming));
        assert!(is_eco_tech(T::Organization) && !is_military_tech(T::Organization));
        // Pure military: unit techs.
        assert!(is_military_tech(T::Chivalry) && !is_eco_tech(T::Chivalry));
        assert!(is_military_tech(T::Strategy) && !is_eco_tech(T::Strategy));
        // Mixed: Smithery = Swordsman + Forge; Archery = Archer + forest def.
        assert!(is_military_tech(T::Smithery) && is_eco_tech(T::Smithery));
        assert!(is_military_tech(T::Archery));
        // Mobility: terrain passage and roads.
        assert!(is_mobility_tech(T::Climbing) && is_mobility_tech(T::Roads));
        assert!(is_mobility_tech(T::Sailing));
        // Derived harvests come from resources.rs, not hand annotation.
        assert_eq!(get_tech_effects(T::Mining).harvests, vec![ResourceType::Metal]);
        assert_eq!(get_tech_effects(T::Organization).harvests, vec![ResourceType::Fruit]);
        // Support units don't make a tech military.
        assert!(!is_military_tech(T::Philosophy)); // MindBender heals, attack 0
    }

    /// The water lane is masked out wholesale on dry maps, so the exact
    /// membership is pinned rather than trusted — a table edit that widened
    /// this would silently delete a buyable tech from every Drylands game.
    #[test]
    fn water_tech_classification_is_table_derived() {
        use strum::IntoEnumIterator;
        use TechnologyType as T;

        let names = |f: fn(T) -> bool| {
            let mut v: Vec<String> = T::iter().filter(|t| f(*t)).map(|t| format!("{t:?}")).collect();
            v.sort();
            v
        };
        let expect = |list: &[&str]| {
            let mut v: Vec<String> = list.iter().map(|s| s.to_string()).collect();
            v.sort();
            v
        };
        assert_eq!(
            names(is_water_tech),
            expect(&["Fishing", "Sailing", "Navigation", "Ramming", "Aquatism",
                     "IceFishing", "Oceantology", "FreeDiving"]),
            "water-tech set moved"
        );

        // Masking targets only the lane that leads nowhere else. FreeDiving is
        // water-bound but gates Aquarion's Chivalry, so it must survive.
        assert_eq!(
            names(is_water_dead_end),
            expect(&["Fishing", "Sailing", "Navigation", "Ramming", "Aquatism", "Oceantology"]),
            "water dead-end set moved"
        );
        assert!(is_water_tech(T::FreeDiving) && !is_water_dead_end(T::FreeDiving));

        // Land techs that merely touch water must not be swept in.
        for t in [T::Riding, T::Roads, T::Trade, T::Construction, T::Smithery,
                  T::Chivalry, T::Philosophy, T::Climbing, T::Hunting] {
            assert!(!is_water_tech(t), "{t:?} misclassified as water");
        }
        // Amphibious / ice-walking tribe units are land-capable.
        assert!(!is_water_tech(T::Spearing), "Tridention is amphibious");
        assert!(!is_water_tech(T::Frostwork), "Mooni skates on land");
    }
}
