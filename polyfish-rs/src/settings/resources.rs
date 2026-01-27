//! Resource settings

use crate::types::{ResourceType, StructureType, TechnologyType, TribeType};

/// Resource configuration
#[derive(Debug, Clone, Default)]
pub struct ResourceSetting {
    pub cost: Option<i32>,
    pub tech_required: TechnologyType,
    pub struct_type: Option<StructureType>,
    pub visible_required: Vec<TechnologyType>,
    pub requires_capture: bool,
    pub reward_pop: i32,
    pub reward_stars: i32,
    pub tribe_type: Option<TribeType>,
}

/// Get resource settings by type
pub fn get_resource_setting(resource_type: ResourceType) -> ResourceSetting {
    use ResourceType::*;
    use TechnologyType::*;
    
    match resource_type {
        None => ResourceSetting::default(),
        
        Game => ResourceSetting {
            cost: Some(2),
            tech_required: Hunting,
            reward_pop: 1,
            ..Default::default()
        },
        Crop => ResourceSetting {
            tech_required: Farming,
            struct_type: Some(StructureType::Farm),
            visible_required: vec![Organization, Farming, Construction],
            ..Default::default()
        },
        Fish => ResourceSetting {
            cost: Some(2),
            tech_required: Fishing,
            reward_pop: 1,
            ..Default::default()
        },
        Metal => ResourceSetting {
            tech_required: Mining,
            struct_type: Some(StructureType::Mine),
            visible_required: vec![Climbing, Mining, Smithery],
            ..Default::default()
        },
        Whale => ResourceSetting {
            tech_required: Unrequired,
            ..Default::default()
        },
        Fruit => ResourceSetting {
            cost: Some(2),
            tech_required: Organization,
            reward_pop: 1,
            ..Default::default()
        },
        Spores => ResourceSetting {
            tech_required: Unrequired,
            struct_type: Some(StructureType::Spores),
            reward_pop: 1,
            tribe_type: Some(TribeType::Cymanti),
            ..Default::default()
        },
        Starfish => ResourceSetting {
            tech_required: Navigation,
            visible_required: vec![Fishing, Sailing, Navigation],
            requires_capture: true,
            reward_stars: 5,
            ..Default::default()
        },
        AquaCrop => ResourceSetting {
            tech_required: BeyondComprehension,
            ..Default::default()
        },
    }
}
