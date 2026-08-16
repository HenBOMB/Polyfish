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
    pub limited_per_tribe: bool,
    pub reward_pop: i32,
    pub reward_stars: i32,
    pub reward_score: i32,
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

/// Get structure settings by type — cached, returns a shared `'static` reference
/// built once per type. Previously this allocated two `HashSet`s on every call,
/// and `moves/build.rs` calls it inside the per-tile x per-structure loop of
/// `generate_build_moves`, so every rollout was allocating per candidate tile.
pub fn get_structure_setting(struct_type: StructureType) -> &'static StructureSetting {
    static TABLE: std::sync::LazyLock<rustc_hash::FxHashMap<StructureType, StructureSetting>> =
        std::sync::LazyLock::new(|| {
            use strum::IntoEnumIterator;
            StructureType::iter()
                .map(|s| (s, build_structure_setting(s)))
                .collect()
        });
    &TABLE[&struct_type]
}

/// The 5 vanilla temple types, which level 1-5 via age-based growth
/// (`actions/mod.rs`'s temple-growth pass) rather than the adjacency
/// mechanic `StructureSetting.adjacent_types` drives for hubs.
pub fn is_temple(struct_type: StructureType) -> bool {
    matches!(
        struct_type,
        StructureType::Temple
            | StructureType::WaterTemple
            | StructureType::ForestTemple
            | StructureType::MountainTemple
            | StructureType::IceTemple
    )
}

/// True when `.level` on this structure type reflects something real — hub
/// adjacency count or temple age-growth — as opposed to the flat `1` every
/// other structure type is built with and keeps forever. Drives whether the
/// UI is worth showing a level for.
pub fn has_meaningful_level(struct_type: StructureType) -> bool {
    is_temple(struct_type) || !get_structure_setting(struct_type).adjacent_types.is_empty()
}

/// Build the settings for one structure type (called once per type at table init).
fn build_structure_setting(struct_type: StructureType) -> StructureSetting {
    match struct_type {
        StructureType::None
        | StructureType::Village
        | StructureType::Ruin
        | StructureType::Lighthouse => StructureSetting::default(),

        // StructureType::Spores => StructureSetting {
        //     resource_type: Some(ResourceType::Spores),
        //     terrain_types: terrains![TerrainType::Forest],
        //     tribe_type: Some(TribeType::Cymanti),
        //     ..Default::default()
        // },
        StructureType::Algae => StructureSetting {
            cost: Some(5),
            terrain_types: terrains![TerrainType::Water, TerrainType::Ocean],
            reward_pop: 1,
            tribe_type: Some(TribeType::Cymanti),
            ..Default::default()
        },
        StructureType::Mycelium => StructureSetting {
            cost: Some(5),
            terrain_types: terrains![
                TerrainType::Field,
                TerrainType::Forest,
                TerrainType::Water,
                TerrainType::Mountain,
                TerrainType::Ocean
            ], // Can be built on algae (which is on water)
            limited_per_city: true,
            tribe_type: Some(TribeType::Cymanti),
            ..Default::default()
        },
        StructureType::Clathrus => StructureSetting {
            cost: Some(5),
            reward_stars: 1,
            terrain_types: terrains![TerrainType::Water, TerrainType::Ocean],
            tribe_type: Some(TribeType::Cymanti),
            ..Default::default()
        },

        StructureType::Road => StructureSetting {
            // 3 is CORRECT and deliberate (Verdi, Aug 2026) — not a training
            // stopgap. The old comment here read as though 2 were the true
            // value and invited a "fix"; it is not.
            //
            // Roads are not an economic build. On a 121-tile map a road net
            // never earns a monument and the connection bonus is a couple of
            // pop at best. What it buys is MOVEMENT: map control, reach, and
            // the ability to overwhelm an opponent — and above all the
            // rider+roads pattern, where a Rider strikes and then withdraws
            // three tiles out of reach. Price roads against that, never
            // against population per star.
            cost: Some(3),
            terrain_types: terrains![
                TerrainType::Field,
                TerrainType::Forest,
                TerrainType::Wetland,
                TerrainType::Mangrove
            ],
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
            adjacent_types: adjacent![
                StructureType::Sawmill,
                StructureType::Windmill,
                StructureType::Forge
            ],
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
            // Update: Forge can now be built on forest tiles (previously required clearing)
            terrain_types: terrains![TerrainType::Field, TerrainType::Forest],
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
        StructureType::IceBank => StructureSetting {
            cost: Some(20),
            tribe_type: Some(TribeType::Polaris),
            terrain_types: terrains![TerrainType::Field, TerrainType::Ice],
            limited_per_tribe: true,
            ..Default::default()
        },

        // Elyrion
        // TODO: https://polytopia.fandom.com/wiki/Sanctuary
        StructureType::Sanctuary => StructureSetting {
            cost: Some(5),
            reward_stars: 1,
            tribe_type: Some(TribeType::Elyrion),
            terrain_types: terrains![
                TerrainType::Field,
                TerrainType::Forest,
                TerrainType::Mountain
            ],
            // required adjacent
            limited_per_city: true,
            ..Default::default()
        },

        StructureType::Fungi => StructureSetting {
            cost: Some(5),
            reward_pop: 1,
            tribe_type: Some(TribeType::Cymanti),
            terrain_types: terrains![TerrainType::Field],
            resource_type: Some(ResourceType::Spores),
            ..Default::default()
        },

        // Monuments
        StructureType::AltarOfPeace
        | StructureType::TowerOfWisdom
        | StructureType::GrandBazaar
        | StructureType::EmperorsTomb
        | StructureType::GateOfPower
        | StructureType::ParkOfFortune
        | StructureType::EyeOfGod
        | StructureType::ChurchOfConverts => StructureSetting {
            reward_pop: 3,
            reward_score: 400,
            terrain_types: terrains![TerrainType::Field, TerrainType::Forest, TerrainType::Water],
            ..Default::default()
        },
    }
}
