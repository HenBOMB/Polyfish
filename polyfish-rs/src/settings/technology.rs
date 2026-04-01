//! Technology settings and tech tree

use crate::types::{
    AbilityType, StructureType, TaskType, TechnologyType, TerrainType, TribeType, UnitType,
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
    pub unlocks_other: i32,
    pub explicit_cost: Option<i32>,
    pub unlocks_terrain: Option<TerrainType>,
}

/// Get technology settings by type
pub fn get_technology_setting(tech_type: TechnologyType) -> TechnologySetting {
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
            explicit_cost: Some(2),
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
            unlocks_other: 1, // capital vision
            ..Default::default()
        },

        Climbing => TechnologySetting {
            tier: Some(1),
            next: vec![Mining, Meditation],
            unlocks_terrain: Some(TerrainType::Mountain),
            unlocks_other: 1, // pacifist
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
            unlocks_other: 1, // discount
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
            unlocks_other: 2, // water and ocean def
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
            unlocks_other: 1, // forest def
            unlocks_special_units: vec![UnitType::Phychi, UnitType::IceArcher],
            ..Default::default()
        },
        Spiritualism => TechnologySetting {
            tier: Some(3),
            requires: Some(Archery),
            unlocks_structure: Some(StructureType::ForestTemple),
            unlocks_ability: Some(AbilityType::GrowForest),
            explicit_cost: Some(5),
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

/// Get the tech cost based on number of cities
pub fn get_tech_cost(num_cities: i32, tier: i32, has_philosophy: bool) -> i32 {
    // Polytopia official: 4 + (cities * tier)
    let cost = 4 + (num_cities * tier);
    if has_philosophy {
        // Philosophy gives ~23% discount (0.77 multiplier)
        ((cost as f32) * 0.77).ceil() as i32
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
