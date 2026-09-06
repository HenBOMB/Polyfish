//! Resource settings

use crate::types::{ResourceType, StructureType, TechnologyType, TribeType};

/// Resource configuration
#[derive(Debug, Clone, Default)]
pub struct ResourceSetting {
    pub cost: Option<i32>,
    pub tech_required: TechnologyType,
    pub struct_required: Option<StructureType>,
    /// Techs that reveal this resource — ANY one suffices, so a branch is listed
    /// in full. Empty means always visible: Fruit, Game, Fish, Spores and
    /// AquaCrop are on the map from turn 1, matching the real game, where only
    /// Organization ("reveals crop"), Climbing ("reveals metal") and Navigation
    /// (deep water) uncover anything. Read by
    /// `functions::is_resource_visible_to_tribe`.
    pub visible_required: Vec<TechnologyType>,
    pub requires_capture: bool,
    pub reward_pop: i32,
    pub reward_stars: i32,
    pub tribe_type: Option<TribeType>,
}

/// Get resource settings by type. EXP_ELO_128: plain `Vec` indexed by
/// discriminant, not a hash map — see `get_unit_setting`'s doc comment.
pub fn get_resource_setting(resource_type: ResourceType) -> &'static ResourceSetting {
    static TABLE: std::sync::LazyLock<Vec<ResourceSetting>> = std::sync::LazyLock::new(|| {
        use strum::IntoEnumIterator;
        let max = ResourceType::iter().map(|r| r as i8 as usize).max().unwrap_or(0);
        let mut table: Vec<ResourceSetting> =
            (0..=max).map(|_| build_resource_setting(ResourceType::None)).collect();
        for r in ResourceType::iter() {
            table[r as i8 as usize] = build_resource_setting(r);
        }
        table
    });
    &TABLE[resource_type as i8 as usize]
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
            visible_required: vec![Navigation],
            requires_capture: true,
            reward_stars: 8,
            ..Default::default()
        },
        AquaCrop => ResourceSetting {
            tech_required: BeyondComprehension,
            reward_pop: 2,
            ..Default::default()
        },
    }
}

#[cfg(test)]
mod exp_elo_128_tests {
    use super::*;
    use strum::IntoEnumIterator;

    /// EXP_ELO_128: the `Vec`-by-discriminant table must return exactly
    /// what `build_resource_setting` computes directly, for every real
    /// variant -- pins the array-indexing rewrite against the function it
    /// replaced.
    #[test]
    fn get_resource_setting_matches_a_fresh_build_for_every_variant() {
        for r in ResourceType::iter() {
            assert_eq!(
                format!("{:?}", get_resource_setting(r)),
                format!("{:?}", build_resource_setting(r)),
                "get_resource_setting({r:?}) diverged from a fresh build"
            );
        }
    }
}
