//! Structure settings

use crate::types::{ResourceType, StructureType, TerrainType, TribeType};
use std::collections::HashSet;

/// Structure configuration
#[derive(Debug, Clone, Default)]
pub struct StructureSetting {
    pub cost: Option<i32>,
    pub terrain_types: HashSet<TerrainType>,
    pub adjacent_types: HashSet<StructureType>,
    pub resource_type: Option<ResourceType>,
    pub limited_per_city: bool,
    pub reward_pop: i32,
    pub reward_stars: i32,
    pub tribe_type: Option<TribeType>,
}

/// Helper macro to create terrain sets
macro_rules! terrains {
    ($($t:expr),* $(,)?) => {{
        let mut set = HashSet::new();
        $(set.insert($t);)*
        set
    }};
}

/// Helper macro to create adjacent structure sets
macro_rules! adjacent {
    ($($s:expr),* $(,)?) => {{
        let mut set = HashSet::new();
        $(set.insert($s);)*
        set
    }};
}

/// Get structure settings by type
pub fn get_structure_setting(struct_type: StructureType) -> StructureSetting {
    match struct_type {
        StructureType::None | StructureType::Village | StructureType::Ruin | StructureType::Lighthouse => {
            StructureSetting::default()
        },
        
        StructureType::Spores => StructureSetting {
            resource_type: Some(ResourceType::Spores),
            terrain_types: terrains![TerrainType::Forest],
            tribe_type: Some(TribeType::Cymanti),
            ..Default::default()
        },
        StructureType::Swamp => StructureSetting {
            terrain_types: terrains![TerrainType::Ocean],
            ..Default::default()
        },
        StructureType::Mycelium => StructureSetting {
            terrain_types: terrains![TerrainType::Field, TerrainType::Forest],
            limited_per_city: true,
            tribe_type: Some(TribeType::Cymanti),
            ..Default::default()
        },
        StructureType::Algae => StructureSetting {
            terrain_types: terrains![TerrainType::Water, TerrainType::Ocean],
            tribe_type: Some(TribeType::Cymanti),
            ..Default::default()
        },
        
        StructureType::Road => StructureSetting {
            cost: Some(3),
            terrain_types: terrains![TerrainType::Field, TerrainType::Forest],
            ..Default::default()
        },
        StructureType::Bridge => StructureSetting {
            cost: Some(5),
            terrain_types: terrains![TerrainType::Water],
            ..Default::default()
        },
        StructureType::Temple => StructureSetting {
            cost: Some(20),
            reward_pop: 1,
            terrain_types: terrains![TerrainType::Field],
            ..Default::default()
        },
        StructureType::Market => StructureSetting {
            cost: Some(5),
            reward_stars: 1,
            terrain_types: terrains![TerrainType::Field],
            adjacent_types: adjacent![StructureType::Sawmill, StructureType::Windmill, StructureType::Forge],
            limited_per_city: true,
            ..Default::default()
        },
        
        StructureType::Farm => StructureSetting {
            cost: Some(5),
            reward_pop: 2,
            resource_type: Some(ResourceType::Crop),
            terrain_types: terrains![TerrainType::Field],
            ..Default::default()
        },
        StructureType::Windmill => StructureSetting {
            cost: Some(5),
            terrain_types: terrains![TerrainType::Field],
            adjacent_types: adjacent![StructureType::Farm],
            limited_per_city: true,
            reward_pop: 1,
            ..Default::default()
        },
        StructureType::Embassy => StructureSetting {
            cost: Some(5),
            ..Default::default()
        },
        
        StructureType::Mine => StructureSetting {
            cost: Some(5),
            reward_pop: 2,
            resource_type: Some(ResourceType::Metal),
            terrain_types: terrains![TerrainType::Mountain],
            ..Default::default()
        },
        StructureType::Forge => StructureSetting {
            cost: Some(5),
            reward_pop: 2,
            terrain_types: terrains![TerrainType::Field],
            adjacent_types: adjacent![StructureType::Mine],
            limited_per_city: true,
            ..Default::default()
        },
        StructureType::MountainTemple => StructureSetting {
            cost: Some(20),
            reward_pop: 1,
            terrain_types: terrains![TerrainType::Mountain],
            ..Default::default()
        },
        
        StructureType::Port => StructureSetting {
            cost: Some(7),
            reward_pop: 1,
            terrain_types: terrains![TerrainType::Water],
            ..Default::default()
        },
        StructureType::WaterTemple => StructureSetting {
            cost: Some(20),
            reward_pop: 1,
            terrain_types: terrains![TerrainType::Water, TerrainType::Ocean],
            ..Default::default()
        },
        
        StructureType::LumberHut => StructureSetting {
            cost: Some(3),
            reward_pop: 1,
            terrain_types: terrains![TerrainType::Forest],
            ..Default::default()
        },
        StructureType::Sawmill => StructureSetting {
            cost: Some(5),
            reward_pop: 1,
            terrain_types: terrains![TerrainType::Field],
            adjacent_types: adjacent![StructureType::LumberHut],
            limited_per_city: true,
            ..Default::default()
        },
        StructureType::ForestTemple => StructureSetting {
            cost: Some(15),
            reward_pop: 1,
            terrain_types: terrains![TerrainType::Forest],
            ..Default::default()
        },
        
        StructureType::Outpost => StructureSetting {
            cost: Some(5),
            reward_pop: 1,
            tribe_type: Some(TribeType::Polaris), // TODO: Polaris disabled
            terrain_types: terrains![TerrainType::Ice],
            ..Default::default()
        },
        StructureType::IceTemple => StructureSetting {
            cost: Some(20),
            reward_pop: 1,
            tribe_type: Some(TribeType::Polaris), // TODO: Polaris disabled
            terrain_types: terrains![TerrainType::Ice],
            ..Default::default()
        },
        
        // Monuments
        StructureType::AltarOfPeace | StructureType::TowerOfWisdom | 
        StructureType::GrandBazaar | StructureType::EmperorsTomb | 
        StructureType::GateOfPower | StructureType::ParkOfFortune | 
        StructureType::EyeOfGod => StructureSetting {
            reward_pop: 3,
            terrain_types: terrains![TerrainType::Field, TerrainType::Forest, TerrainType::Water],
            ..Default::default()
        },
    }
}
/// List of structures that can be placed freely (not resource-dependent)
pub const PLACEABLE_STRUCTURES: &[StructureType] = &[
    StructureType::Road,
    StructureType::Port,
    StructureType::Temple,
    StructureType::WaterTemple,
    StructureType::MountainTemple,
    StructureType::ForestTemple,
    StructureType::LumberHut,
    StructureType::Mycelium,
    StructureType::Algae,
];
