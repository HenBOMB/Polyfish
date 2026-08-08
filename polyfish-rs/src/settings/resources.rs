//! Resource settings

use crate::types::{ResourceType, StructureType, TechnologyType, TribeType};

/// Resource configuration
#[derive(Debug, Clone, Default)]
pub struct ResourceSetting {
    pub cost: Option<i32>,
    pub tech_required: TechnologyType,
    pub struct_required: Option<StructureType>,
    /// Techs that reveal this resource — ANY one suffices. Empty means always
    /// visible (Fruit, Spores). Read by `functions::is_resource_visible_to_tribe`;
    /// it used to be declared here and ignored, with a different rule hardcoded
    /// in that function.
    pub visible_required: Vec<TechnologyType>,
    pub requires_capture: bool,
    pub reward_pop: i32,
    pub reward_stars: i32,
    pub tribe_type: Option<TribeType>,
}

/// Get resource settings by type
pub fn get_resource_setting(resource_type: ResourceType) -> &'static ResourceSetting {
    static TABLE: std::sync::LazyLock<rustc_hash::FxHashMap<ResourceType, ResourceSetting>> =
        std::sync::LazyLock::new(|| {
            use strum::IntoEnumIterator;
            ResourceType::iter()
                .map(|r| (r, build_resource_setting(r)))
                .collect()
        });
    &TABLE[&resource_type]
}

/// Build the settings for one resource type (called once per type at table init).
fn build_resource_setting(resource_type: ResourceType) -> ResourceSetting {
    use ResourceType::*;
    use TechnologyType::*;

    match resource_type {
        None => ResourceSetting::default(),

        Game => ResourceSetting {
            cost: Some(2),
            tech_required: Hunting,
            visible_required: vec![Hunting],
            reward_pop: 1,
            ..Default::default()
        },
        Crop => ResourceSetting {
            tech_required: Farming,
            struct_required: Some(StructureType::Farm),
            visible_required: vec![Organization, Farming, Construction],
            reward_pop: 2,
            ..Default::default()
        },
        Fish => ResourceSetting {
            cost: Some(2),
            tech_required: Fishing,
            visible_required: vec![Fishing],
            reward_pop: 1,
            ..Default::default()
        },
        Metal => ResourceSetting {
            tech_required: Mining,
            struct_required: Some(StructureType::Mine),
            visible_required: vec![Climbing, Mining, Smithery],
            reward_pop: 2,
            ..Default::default()
        },
        Fruit => ResourceSetting {
            cost: Some(2),
            tech_required: Organization,
            reward_pop: 1,
            ..Default::default()
        },
        Spores => ResourceSetting {
            tech_required: Basic,
            struct_required: Some(StructureType::Fungi),
            reward_pop: 1,
            tribe_type: Some(TribeType::Cymanti),
            ..Default::default()
        },
        Starfish => ResourceSetting {
            tech_required: Navigation,
            visible_required: vec![Fishing, Sailing, Navigation],
            requires_capture: true,
            reward_stars: 8,
            ..Default::default()
        },
        AquaCrop => ResourceSetting {
            tech_required: BeyondComprehension,
            visible_required: vec![FreeDiving],
            reward_pop: 2,
            ..Default::default()
        },
    }
}
