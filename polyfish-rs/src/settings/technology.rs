//! Technology settings and tech tree

use crate::types::{
    AbilityType, ResourceType, StructureType, TaskType, TechnologyType, TerrainType, TribeType,
    UnitType,
};

/// Technology configuration
#[derive(Debug, Clone, Default)]
pub struct TechnologySetting {
    pub tier: Option<i32>,
    pub requires: Option<TechnologyType>,
    pub replaces_tech: Option<TechnologyType>,
    pub tribe_type: Option<TribeType>,
    pub next: Vec<TechnologyType>,
    pub unlocks_resource: Option<ResourceType>,
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

        Unrequired => TechnologySetting {
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
            unlocks_resource: Some(ResourceType::Fruit),
            ..Default::default()
        },
        Farming => TechnologySetting {
            tier: Some(2),
            requires: Some(Organization),
            next: vec![Construction],
            unlocks_resource: Some(ResourceType::Crop),
            unlocks_structure: Some(StructureType::Farm),
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
            unlocks_resource: Some(ResourceType::Metal),
            unlocks_structure: Some(StructureType::Mine),
            ..Default::default()
        },
        Smithery => TechnologySetting {
            tier: Some(3),
            requires: Some(Mining),
            unlocks_structure: Some(StructureType::Forge),
            unlocks_unit: Some(UnitType::Swordsman),
            ..Default::default()
        },
        Meditation => TechnologySetting {
            tier: Some(2),
            requires: Some(Climbing),
            next: vec![Philosophy],
            unlocks_structure: Some(StructureType::MountainTemple),
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
            ..Default::default()
        },

        Fishing => TechnologySetting {
            tier: Some(1),
            next: vec![Sailing, Ramming],
            unlocks_resource: Some(ResourceType::Fish),
            unlocks_unit: Some(UnitType::Raft),
            unlocks_terrain: Some(TerrainType::Water),
            unlocks_structure: Some(StructureType::Port),
            ..Default::default()
        },
        Sailing => TechnologySetting {
            tier: Some(2),
            requires: Some(Fishing),
            next: vec![Navigation],
            unlocks_unit: Some(UnitType::Scout),
            unlocks_terrain: Some(TerrainType::Ocean),
            ..Default::default()
        },
        Navigation => TechnologySetting {
            tier: Some(3),
            requires: Some(Ramming),
            unlocks_unit: Some(UnitType::Bomber),
            unlocks_resource: Some(ResourceType::Starfish),
            ..Default::default()
        },
        Ramming => TechnologySetting {
            tier: Some(2),
            requires: Some(Fishing),
            next: vec![Aquatism],
            unlocks_unit: Some(UnitType::Rammer),
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
            unlocks_resource: Some(ResourceType::Game),
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
        Pascetism => TechnologySetting {
            replaces_tech: Some(Sailing),
            tribe_type: Some(TribeType::Cymanti),
            next: vec![Navigation],
            unlocks_unit: Some(UnitType::Raychi),
            unlocks_structure: Some(StructureType::Algae),
            ..Default::default()
        },
        ShockTactics => TechnologySetting {
            replaces_tech: Some(Chivalry),
            tribe_type: Some(TribeType::Cymanti),
            unlocks_unit: Some(UnitType::Doomux),
            ..Default::default()
        },
        Hydrology => TechnologySetting {
            replaces_tech: Some(Ramming),
            tribe_type: Some(TribeType::Cymanti),
            next: vec![Aquatism],
            ..Default::default()
        },
        Oceantology => TechnologySetting {
            replaces_tech: Some(Navigation),
            tribe_type: Some(TribeType::Cymanti),
            unlocks_unit: Some(UnitType::LivingIsland),
            unlocks_resource: Some(ResourceType::Starfish),
            ..Default::default()
        },

        // Aquarion replacements
        Spearing => TechnologySetting {
            replaces_tech: Some(Chivalry),
            tribe_type: Some(TribeType::Aquarion),
            ..Default::default()
        },
        Amphibian => TechnologySetting {
            replaces_tech: Some(Riding),
            tribe_type: Some(TribeType::Aquarion),
            unlocks_unit: Some(UnitType::Tridention),
            unlocks_terrain: Some(TerrainType::Water),
            next: vec![Waterways],
            ..Default::default()
        },
        Waterways => TechnologySetting {
            replaces_tech: Some(Roads),
            requires: Some(Amphibian),
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

/// Helper to find which technology unlocks a specific unit
pub fn get_tech_unlocking_unit(unit_type: UnitType) -> Option<TechnologyType> {
    use TechnologyType::*;

    // Instead of iterating (which requires listing all enums or deriving EnumIter),
    // we manually map known units to techs for performance and compile-time checking.
    // This duplicates logic but avoids runtime search.

    match unit_type {
        // Base units
        UnitType::Warrior => Some(Unrequired),

        // Riding branch
        UnitType::Rider => Some(Riding),
        UnitType::Knight => Some(Chivalry),

        // Strategy branch
        UnitType::Defender => Some(Strategy),
        UnitType::Cloak => Some(Diplomacy),

        // Mining branch
        UnitType::Swordsman => Some(Smithery),

        // Philosophy branch
        UnitType::MindBender => Some(Philosophy),

        // Hunting branch
        UnitType::Archer => Some(Archery),
        UnitType::Catapult => Some(Mathematics),

        // Fishing branch (Naval)
        UnitType::Raft => Some(Fishing),
        UnitType::Scout => Some(Sailing),
        UnitType::Bomber => Some(Navigation),
        UnitType::Rammer => Some(Ramming), // Was Ramming

        // Special Units (Tribe Specific)
        UnitType::Hexapod => Some(Riding),   // Cymanti
        UnitType::Amphibian => Some(Riding), // Aquarion? Actually Ride unlocks it in special tree?
        // Wait, Aquarion tech tree: Riding replaced by Amphibian?
        // Look at settings above: Amphibian (tech) unlocks Tridention (unit).
        // What unlocks Amphibian (unit)?
        // Riding (tech) unlocks special units: [Hexapod, Amphibian].
        // So Riding unlocks it.
        UnitType::Tridention => Some(Chivalry), // Or Amphibian tech?
        // Settings above: Chivalry unlocks [Tridention] as special unit IF tribe matches?
        // No, Chivalry above unlocks [Tridention].
        // But Amphibian (Tech) unlocks Tridention too (line 309).
        // If Aquarion uses Amphibian tech (replacing Riding), then Amphibian tech unlocks Tridention?
        // Wait, line 309: unlocks_unit: Some(Tridention).
        // So simple lookup works if I return the tech that unlocks it.
        // If multiple techs unlock it (e.g. Riding for normal tribe vs Amphibian for Aquarion?),
        // I should return the tech relevant for the tribe!
        // But this function doesn't take tribe.
        // It's "get_tech_unlocking_unit".
        // If I assume 1-to-1 mapping, I might be wrong for replacements.
        // E.g. Riding unlocks Rider.
        // Amphibian (Tech) replaces Riding. Does it unlock Rider? No, it unlocks Tridention (line 309)?
        // Wait, line 309 says `unlocks_unit: Some(Tridention)`.
        // So Aquarion gets Tridention via Amphibian tech.
        // Does Aquarion get Rider?
        // Amphibian tech replaces Riding. So they don't get Riding.
        // So they don't get Rider.
        // This implies the unit type itself is unique to the tech path.
        // If so, 1-to-1 works.
        // Tridention -> Amphibian (Tech).
        // Rider -> Riding.

        // Let's verify special units unlocking.
        UnitType::Kiton => Some(Strategy), // Cymanti stuff
        UnitType::Phychi => Some(Archery),
        UnitType::IceArcher => Some(Archery), // Polaris stuff?
        UnitType::Exida => Some(Mathematics),
        UnitType::Shaman => Some(Philosophy),

        // Polaris Techs
        UnitType::Mooni => Some(Frostwork),
        UnitType::BattleSled => Some(Sledding),
        UnitType::IceFortress => Some(PolarWarfare),
        // IceArcher is under Archery (special).

        // Cymanti Techs
        UnitType::Raychi => Some(Pascetism),
        UnitType::Doomux => Some(ShockTactics),
        // Hexapod is under Riding (special).
        // Centipede is super unit?

        // Elyrion
        UnitType::Polytaur => Some(ForestMagic),

        // Aquarion
        // UnitType::Tridention => Some(Amphibian),
        _ => None,
    }
}
