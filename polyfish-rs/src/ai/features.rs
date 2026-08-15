//! AI Feature Extraction - Dynamic Channel Mapping
//!
//! This module converts GameState into tensor format for neural network input.
//! Enforces Fog of War and provides full granularity for all game elements.
//!
//! Channel ranges are dynamically computed from enum variants at compile time.

use crate::functions::{get_city_production, get_unit_max_health, is_under_siege};
use crate::states::{GameState, PlayerId};
use crate::types::{
    ModeType, ResourceType, StructureType, TerrainType, TribeType, UnitEffect, UnitType,
};
use candle_core::{Device, Result, Tensor};
use std::sync::LazyLock;
use strum::IntoEnumIterator;

pub const MAP_SIZE: usize = 11;

// ============================================================================
// Dynamic Channel Counts
// These are derived from the enum variant counts in types.rs
// When new variants are added to enums, update these counts accordingly
// ============================================================================

/// Count of terrain types (TerrainType variants)
pub const TERRAIN_COUNT: usize = 8; // None, Water, Ocean, Field, Mountain, Forest, Ice, Algae

/// Count of resource types (ResourceType variants)
pub const RESOURCE_COUNT: usize = 9; // None, Game, Crop, Fish, Metal, Fruit, Spores, Starfish, AquaCrop

/// Count of structure types (StructureType variants)
pub const STRUCTURE_COUNT: usize = 35;

/// Count of unit types (UnitType variants)
pub const UNIT_COUNT: usize = 46;

// ============================================================================
// Channel Range Constants (dynamically positioned)
// ============================================================================

// Terrain channels
pub const CH_TERRAIN_START: usize = 0;
pub const CH_TERRAIN_END: usize = CH_TERRAIN_START + TERRAIN_COUNT;

// Tile flags (fixed count: 8)
pub const CH_TILE_FLAGS_START: usize = CH_TERRAIN_END;
pub const CH_TILE_FLAGS_COUNT: usize = 10;
pub const CH_TILE_FLAGS_END: usize = CH_TILE_FLAGS_START + CH_TILE_FLAGS_COUNT;

// Tile flag offsets within the range
pub const CH_TILE_FROZEN: usize = CH_TILE_FLAGS_START + 0;
pub const CH_TILE_FLOODED: usize = CH_TILE_FLAGS_START + 1;
pub const CH_TILE_HAS_ROAD: usize = CH_TILE_FLAGS_START + 2;
pub const CH_TILE_HAS_ROUTE: usize = CH_TILE_FLAGS_START + 3;
pub const CH_TILE_OWNER: usize = CH_TILE_FLAGS_START + 4;
pub const CH_TILE_CLIMATE: usize = CH_TILE_FLAGS_START + 5;
pub const CH_TILE_IS_EXPLORED: usize = CH_TILE_FLAGS_START + 6;
pub const CH_TILE_VISIBILITY: usize = CH_TILE_FLAGS_START + 7;
pub const CH_TILE_AT_PEACE: usize = CH_TILE_FLAGS_START + 8;
pub const CH_TILE_EMBASSY_LEVEL: usize = CH_TILE_FLAGS_START + 9;

// Resource channels
pub const CH_RESOURCE_START: usize = CH_TILE_FLAGS_END;
pub const CH_RESOURCE_END: usize = CH_RESOURCE_START + RESOURCE_COUNT;

// Structure channels
pub const CH_STRUCTURE_START: usize = CH_RESOURCE_END;
pub const CH_STRUCTURE_END: usize = CH_STRUCTURE_START + STRUCTURE_COUNT;

// Unit type channels
pub const CH_UNIT_START: usize = CH_STRUCTURE_END;
pub const CH_UNIT_END: usize = CH_UNIT_START + UNIT_COUNT;

// Unit stats (fixed count: 16)
pub const CH_UNIT_STATS_START: usize = CH_UNIT_END;
pub const CH_UNIT_STATS_COUNT: usize = 16;
pub const CH_UNIT_STATS_END: usize = CH_UNIT_STATS_START + CH_UNIT_STATS_COUNT;

// Unit stat offsets
pub const CH_UNIT_OWNER: usize = CH_UNIT_STATS_START + 0;
pub const CH_UNIT_HP: usize = CH_UNIT_STATS_START + 1;
pub const CH_UNIT_MAX_HP: usize = CH_UNIT_STATS_START + 2;
pub const CH_UNIT_VETERAN: usize = CH_UNIT_STATS_START + 3;
pub const CH_UNIT_MOVED: usize = CH_UNIT_STATS_START + 4;
pub const CH_UNIT_ATTACKED: usize = CH_UNIT_STATS_START + 5;
pub const CH_UNIT_KILLS: usize = CH_UNIT_STATS_START + 6;
pub const CH_UNIT_EFFECT_POISON: usize = CH_UNIT_STATS_START + 7;
pub const CH_UNIT_EFFECT_BOOST: usize = CH_UNIT_STATS_START + 8;
pub const CH_UNIT_EFFECT_INVISIBLE: usize = CH_UNIT_STATS_START + 9;
pub const CH_UNIT_EFFECT_FROZEN: usize = CH_UNIT_STATS_START + 10;
pub const CH_UNIT_HAS_PASSENGER: usize = CH_UNIT_STATS_START + 11;
pub const CH_UNIT_PASSENGER_TYPE: usize = CH_UNIT_STATS_START + 12;
pub const CH_UNIT_CONVERTED: usize = CH_UNIT_STATS_START + 13;
pub const CH_UNIT_ATTACKS_PERFORMED: usize = CH_UNIT_STATS_START + 14;
// +15 reserved

// City stats (fixed count: 12)
pub const CH_CITY_STATS_START: usize = CH_UNIT_STATS_END;
pub const CH_CITY_STATS_COUNT: usize = 12;
pub const CH_CITY_STATS_END: usize = CH_CITY_STATS_START + CH_CITY_STATS_COUNT;

// City stat offsets
pub const CH_CITY_PRESENT: usize = CH_CITY_STATS_START + 0;
pub const CH_CITY_OWNER: usize = CH_CITY_STATS_START + 1;
pub const CH_CITY_LEVEL: usize = CH_CITY_STATS_START + 2;
pub const CH_CITY_PRODUCTION: usize = CH_CITY_STATS_START + 3;
pub const CH_CITY_IS_CAPITAL: usize = CH_CITY_STATS_START + 4;
pub const CH_CITY_CONNECTED: usize = CH_CITY_STATS_START + 5;
pub const CH_CITY_HAS_WALLS: usize = CH_CITY_STATS_START + 6;
pub const CH_CITY_HAS_RIOT: usize = CH_CITY_STATS_START + 7;
pub const CH_CITY_PENDING_REWARD: usize = CH_CITY_STATS_START + 8;
pub const CH_CITY_BORDER_SIZE: usize = CH_CITY_STATS_START + 9;
pub const CH_CITY_PROGRESS: usize = CH_CITY_STATS_START + 10;
// +11 reserved

pub const CH_MEM_START: usize = CH_CITY_STATS_END;
pub const CH_MEM_COUNT: usize = 6;
pub const CH_MEM_END: usize = CH_MEM_START + CH_MEM_COUNT;

pub const CH_MEM_ENEMY_SEEN: usize = CH_MEM_START + 0;
pub const CH_MEM_ENEMY_HP: usize = CH_MEM_START + 1;
pub const CH_MEM_ENEMY_ATTACK: usize = CH_MEM_START + 2;
pub const CH_MEM_ENEMY_RANGED: usize = CH_MEM_START + 3;
pub const CH_MEM_ENEMY_NAVAL: usize = CH_MEM_START + 4;
pub const CH_MEM_ATTACKED_HERE: usize = CH_MEM_START + 5;

/// Total number of feature channels (dynamically computed)
pub const NUM_CHANNELS: usize = CH_MEM_END;

// ============================================================================
// Runtime Lookup Tables (enum discriminant -> sequential index)
// ============================================================================

/// Maps TerrainType discriminant to sequential index (0, 1, 2, ...)
static TERRAIN_INDEX: LazyLock<std::collections::HashMap<TerrainType, usize>> =
    LazyLock::new(|| {
        TerrainType::iter()
            .enumerate()
            .map(|(idx, t)| (t, idx))
            .collect()
    });

/// Maps ResourceType discriminant to sequential index
static RESOURCE_INDEX: LazyLock<std::collections::HashMap<ResourceType, usize>> =
    LazyLock::new(|| {
        ResourceType::iter()
            .enumerate()
            .map(|(idx, r)| (r, idx))
            .collect()
    });

/// Maps StructureType discriminant to sequential index
static STRUCTURE_INDEX: LazyLock<std::collections::HashMap<StructureType, usize>> =
    LazyLock::new(|| {
        StructureType::iter()
            .enumerate()
            .map(|(idx, s)| (s, idx))
            .collect()
    });

/// Maps UnitType discriminant to sequential index
static UNIT_INDEX: LazyLock<std::collections::HashMap<UnitType, usize>> = LazyLock::new(|| {
    UnitType::iter()
        .enumerate()
        .map(|(idx, u)| (u, idx))
        .collect()
});

// ============================================================================
// Channel Lookup Functions
// ============================================================================

#[inline]
fn terrain_to_channel(terrain: TerrainType) -> usize {
    CH_TERRAIN_START + TERRAIN_INDEX.get(&terrain).copied().unwrap_or(0)
}

#[inline]
fn resource_to_channel(resource: ResourceType) -> usize {
    CH_RESOURCE_START + RESOURCE_INDEX.get(&resource).copied().unwrap_or(0)
}

#[inline]
fn structure_to_channel(structure: StructureType) -> usize {
    CH_STRUCTURE_START + STRUCTURE_INDEX.get(&structure).copied().unwrap_or(0)
}

#[inline]
fn unit_to_channel(unit_type: UnitType) -> usize {
    CH_UNIT_START + UNIT_INDEX.get(&unit_type).copied().unwrap_or(0)
}

// ============================================================================
// Main Feature Extraction
// ============================================================================

/// Features output structure
#[derive(Clone)]
pub struct GameFeatures {
    pub spatial_map: Tensor,  // [1, C, H, W] - tile-based features
    pub player_state: Tensor, // [1, P] - global player features
}

/// Device-free leaf features: owned `Vec<f32>`, safe to send across threads.
///
/// This is the pre-tensorization form of [`GameFeatures`]. Actors that
/// evaluate leaves off the main network thread must produce `RawFeatures`
/// instead of `GameFeatures` — building a device `Tensor` off the owning
/// thread is unsound for the Metal backend (see `bug_handoff.md`).
pub struct RawFeatures {
    pub spatial: Vec<f32>, // len = NUM_CHANNELS * MAP_SIZE * MAP_SIZE
    pub player: Vec<f32>,  // len = PLAYER_STATE_DIM (16)
}

impl RawFeatures {
    pub const PLAYER_STATE_DIM: usize = 16;

    pub fn spatial_len() -> usize {
        NUM_CHANNELS * MAP_SIZE * MAP_SIZE
    }

    pub fn player_len() -> usize {
        Self::PLAYER_STATE_DIM
    }

    /// Tensorize onto `device`, reproducing the exact shapes `state_to_tensor`
    /// used to produce directly.
    pub fn into_game_features(self, device: &Device) -> Result<GameFeatures> {
        let spatial_map =
            Tensor::from_vec(self.spatial, (1, NUM_CHANNELS, MAP_SIZE, MAP_SIZE), device)?;
        let player_state = Tensor::from_vec(self.player, (1, Self::PLAYER_STATE_DIM), device)?;
        Ok(GameFeatures {
            spatial_map,
            player_state,
        })
    }

    /// 64-bit hash over the spatial + player f32 bytes. `f32::to_bits` gives a
    /// deterministic, platform-independent integer for every finite float, so
    /// identical feature vectors always hash identically. Used as the eval
    /// cache key (see `ai/eval_server.rs`) and for tree-reuse root matching
    /// (see `ai/gumbel_mcts.rs`). Collision probability at these scales is
    /// invisible inside MCTS noise; we accept it rather than store full keys.
    pub fn hash(&self) -> u64 {
        use rustc_hash::FxHasher;
        use std::hash::Hasher;
        let mut h = FxHasher::default();
        for &v in &self.spatial {
            h.write_u32(v.to_bits());
        }
        for &v in &self.player {
            h.write_u32(v.to_bits());
        }
        h.finish()
    }
}

/// Convert game state to tensor with decomposed player state
pub fn state_to_tensor(
    state: &GameState,
    perspective: PlayerId,
    device: &Device,
) -> Result<GameFeatures> {
    state_to_cpu_features(state, perspective)?.into_game_features(device)
}

/// CPU-only feature extraction: identical to `state_to_tensor` except it
/// stops short of allocating device tensors, returning owned `Vec<f32>`
/// instead. Safe to call from any thread.
pub fn state_to_cpu_features(state: &GameState, perspective: PlayerId) -> Result<RawFeatures> {
    let mut data = vec![0.0f32; NUM_CHANNELS * MAP_SIZE * MAP_SIZE];
    let map_size = state.settings.size as usize;

    // Get perspective tribe info
    let pov_tribe = state.tribes.get(&perspective);
    let pov_tribe_type = pov_tribe.map(|t| t.tribe_type).unwrap_or(TribeType::None);
    let is_elyrion = pov_tribe_type == TribeType::Elyrion;

    // Handle model versioning for normalization scales
    let (scale_max_turns, scale_max_score, scale_max_stars, scale_max_spt, scale_max_units) = if state.settings.version <= 0 {
        (30.0, 10000.0, 30.0, 30.0, 20.0) // Legacy scales for v0 models
    } else {
        (
            crate::states::default_max_turns() as f32,
            crate::states::default_max_score() as f32,
            crate::states::default_max_stars() as f32,
            crate::states::default_max_spt() as f32,
            crate::states::default_max_units() as f32,
        )
    };

    // Global stats (same for all tiles)
    let turn_norm = (state.settings.turn as f32 / state.settings.max_turns as f32).clamp(0.0, 1.0);
    let max_turns_norm = (state.settings.max_turns as f32 / scale_max_turns).clamp(0.0, 1.0);
    let stars_norm = pov_tribe
        .map(|t| {
            if state.settings.version <= 0 {
                // Restore original log scale for v0 models
                ((t.stars as f32 + 1.0).ln() / 100.0_f32.ln()).clamp(0.0, 1.0)
            } else {
                (t.stars as f32 / scale_max_stars).clamp(0.0, 1.0)
            }
        })
        .unwrap_or(0.0);
    let spt_norm = pov_tribe
        .map(|t| {
            (crate::functions::get_tribe_spt(state, t) as f32 / scale_max_spt).clamp(0.0, 1.0)
        })
        .unwrap_or(0.0);
    let score_norm = pov_tribe
        .map(|t| (t.score as f32 / scale_max_score).clamp(0.0, 1.0))
        .unwrap_or(0.0);
    let tech_count = pov_tribe
        .map(|t| t.tech_vanilla.iter().filter(|tech| tech.discovered).count())
        .unwrap_or(0);
    // maximum n of valid tech in the polytopia tree per tribe is 25
    let tech_norm = (tech_count as f32 / 25.0).clamp(0.0, 1.0);
    // 16 tribes + 1 cause index starts at 1
    let tribe_type_norm = (pov_tribe_type as i8 as f32 / 17.0).clamp(0.0, 1.0);
    let game_mode = match state.settings.mode {
        ModeType::Domination | ModeType::Might => 1.0,
        _ => 0.0,
    };
    let game_over = if state.settings._game_over { 1.0 } else { 0.0 };
    let total_cities =
        (pov_tribe.map(|t| t.cities.len()).unwrap_or(0) as f32 / 5.0).clamp(0.0, 1.0);
    let total_units = (pov_tribe.map(|t| t.units.len()).unwrap_or(0) as f32 / scale_max_units).clamp(0.0, 1.0);
    // at turn 5 it stops tracking
    let pacifist_turns = pov_tribe
        .map(|t| (t.pacifist_turns as f32 / 5.0).clamp(0.0, 1.0))
        .unwrap_or(0.0);
    let tribe_kills = pov_tribe
        .map(|t| (t.kills as f32 / scale_max_units).clamp(0.0, 1.0))
        .unwrap_or(0.0);
    let tribe_casualties = pov_tribe
        .map(|t| (t.casualties as f32 / scale_max_units).clamp(0.0, 1.0))
        .unwrap_or(0.0);
    let tribe_conversions = pov_tribe
        .map(|t| (t.conversions as f32 / scale_max_units).clamp(0.0, 1.0))
        .unwrap_or(0.0);
    let attacked_this_turn = pov_tribe
        .map(|t| if t.attacked_this_turn { 1.0 } else { 0.0 })
        .unwrap_or(0.0);

    // Process each tile
    for y in 0..map_size {
        for x in 0..map_size {
            if x >= MAP_SIZE || y >= MAP_SIZE {
                continue;
            }
            let idx = (y * map_size + x) as i32;

            // Check visibility (now just use explorers - explored = visible)
            let is_explored = state
                .tiles
                .get(&idx)
                .map(|t| t.explorers.contains(&perspective))
                .unwrap_or(false);

            // Set visibility channel (explored tiles are visible)
            let vis_val = if is_explored { 1.0 } else { 0.0 };
            set_feat(&mut data, CH_TILE_VISIBILITY, x, y, vis_val);

            // Skip tile-specific data if not explored
            if !is_explored {
                // Elyrion special ability lets see ruins in fog
                if is_elyrion {
                    if let Some(Some(structure)) = state.structures.get(&idx) {
                        // Ruins are special - Elyrion can see them through fog
                        if structure.structure_type == StructureType::Ruin {
                            let struct_ch = structure_to_channel(structure.structure_type);
                            set_feat(&mut data, struct_ch, x, y, 1.0);
                        }
                    }
                }
                continue;
            }

            if let Some(tile) = state.tiles.get(&idx) {
                // Terrain (always visible if explored)
                let terrain_ch = terrain_to_channel(tile.terrain_type);
                set_feat(&mut data, terrain_ch, x, y, 1.0);

                // Tile flags
                if tile.is_frozen() {
                    set_feat(&mut data, CH_TILE_FROZEN, x, y, 1.0);
                }
                if tile.is_flooded() {
                    set_feat(&mut data, CH_TILE_FLOODED, x, y, 1.0);
                }
                if tile.has_road {
                    set_feat(&mut data, CH_TILE_HAS_ROAD, x, y, 1.0);
                }
                if tile.has_route {
                    set_feat(&mut data, CH_TILE_HAS_ROUTE, x, y, 1.0);
                }

                // Tile owner
                let owner_val = if tile.owner == perspective {
                    1.0
                } else if tile.owner == 0 {
                    0.0
                } else {
                    -1.0
                };
                set_feat(&mut data, CH_TILE_OWNER, x, y, owner_val);

                // Climate
                let climate_norm = tile.climate as i8 as f32 / 17.0;
                set_feat(&mut data, CH_TILE_CLIMATE, x, y, climate_norm);

                // Explored flag
                set_feat(&mut data, CH_TILE_IS_EXPLORED, x, y, 1.0);

                if tile.owner != 0 && tile.owner != perspective {
                    if let Some(pov_t) = pov_tribe {
                        if let Some(rel) = pov_t.relations.get(&tile.owner) {
                            if rel.state == 1 {
                                set_feat(&mut data, CH_TILE_AT_PEACE, x, y, 1.0);
                            }
                            set_feat(
                                &mut data,
                                CH_TILE_EMBASSY_LEVEL,
                                x,
                                y,
                                rel.embassy_level as f32 / 3.0,
                            );
                        }
                    }
                }

                // Resources
                if let Some(Some(resource)) = state.resources.get(&idx) {
                    if crate::functions::is_resource_visible_to_tribe(
                        state,
                        resource.resource_type,
                        perspective,
                        Some(idx),
                    ) {
                        let res_ch = resource_to_channel(resource.resource_type);
                        set_feat(&mut data, res_ch, x, y, 1.0);
                    }
                }

                // Structures
                if let Some(Some(structure)) = state.structures.get(&idx) {
                    // Ruins are special - Elyrion can see them through fog
                    let struct_ch = structure_to_channel(structure.structure_type);
                    set_feat(&mut data, struct_ch, x, y, 1.0);
                }
            }
        }
    }

    // Process units (only visible ones)
    for (player_id, tribe) in &state.tribes {
        for unit in &tribe.units {
            let x = unit.coords.x as usize;
            let y = unit.coords.y as usize;
            if x >= MAP_SIZE || y >= MAP_SIZE {
                continue;
            }

            let idx = unit.coords.idx;

            let unit_explored = state
                .tiles
                .get(&idx)
                .map(|t| t.explorers.contains(&perspective))
                .unwrap_or(false);
            if !unit_explored {
                continue;
            }

            // Enemy invisible units are hidden from the perspective player.
            // Own units are always visible regardless of Invisible effect.
            if *player_id != perspective
                && unit.effects.contains(&UnitEffect::Invisible)
            {
                continue;
            }

            // Unit type channel
            let unit_ch = unit_to_channel(unit.unit_type);
            set_feat(&mut data, unit_ch, x, y, 1.0);

            // Unit stats
            let owner_val = if *player_id == perspective { 1.0 } else { -1.0 };
            set_feat(&mut data, CH_UNIT_OWNER, x, y, owner_val);

            set_feat(
                &mut data,
                CH_UNIT_HP,
                x,
                y,
                unit.health as f32 / get_unit_max_health(unit) as f32,
            );
            // Removed cause it was a bad keyword
            set_feat(&mut data, CH_UNIT_MAX_HP, x, y, 0.0);
            set_feat(
                &mut data,
                CH_UNIT_VETERAN,
                x,
                y,
                if unit.veteran { 1.0 } else { 0.0 },
            );
            set_feat(
                &mut data,
                CH_UNIT_MOVED,
                x,
                y,
                if unit.moved { 1.0 } else { 0.0 },
            );
            set_feat(
                &mut data,
                CH_UNIT_ATTACKED,
                x,
                y,
                if unit.attacked { 1.0 } else { 0.0 },
            );
            set_feat(
                &mut data,
                CH_UNIT_KILLS,
                x,
                y,
                (unit.kills as f32 / 3.0).clamp(0.0, 1.0),
            );
            set_feat(
                &mut data,
                CH_UNIT_CONVERTED,
                x,
                y,
                if unit.converted { 1.0 } else { 0.0 },
            );
            set_feat(
                &mut data,
                CH_UNIT_ATTACKS_PERFORMED,
                x,
                y,
                // Max ~3 for splash/persist units
                (unit.attacks_performed as f32 / 3.0).clamp(0.0, 1.0),
            );

            // Effects
            if unit.effects.contains(&UnitEffect::Poison) {
                set_feat(&mut data, CH_UNIT_EFFECT_POISON, x, y, 1.0);
            }
            if unit.effects.contains(&UnitEffect::Boosted) {
                set_feat(&mut data, CH_UNIT_EFFECT_BOOST, x, y, 1.0);
            }
            if unit.effects.contains(&UnitEffect::Invisible) {
                set_feat(&mut data, CH_UNIT_EFFECT_INVISIBLE, x, y, 1.0);
            }
            if unit.effects.contains(&UnitEffect::Frozen) {
                set_feat(&mut data, CH_UNIT_EFFECT_FROZEN, x, y, 1.0);
            }

            // Passenger
            if unit.passenger_type.is_some() {
                set_feat(&mut data, CH_UNIT_HAS_PASSENGER, x, y, 1.0);
                if let Some(passenger_type) = unit.passenger_type {
                    let passenger_norm = passenger_type as i8 as f32 / UNIT_COUNT as f32;
                    set_feat(&mut data, CH_UNIT_PASSENGER_TYPE, x, y, passenger_norm);
                }
            }
        }

        // Process cities
        for city in &tribe.cities {
            let idx = city.idx;
            let x = (idx % state.settings.size) as usize;
            let y = (idx / state.settings.size) as usize;
            if x >= MAP_SIZE || y >= MAP_SIZE {
                continue;
            }

            // Only show cities we can see
            let city_explored = state
                .tiles
                .get(&idx)
                .map(|t| t.explorers.contains(&perspective))
                .unwrap_or(false);
            if !city_explored {
                continue;
            }

            set_feat(&mut data, CH_CITY_PRESENT, x, y, 1.0);

            let owner_val = if *player_id == perspective { 1.0 } else { -1.0 };
            set_feat(&mut data, CH_CITY_OWNER, x, y, owner_val);
            set_feat(
                &mut data,
                CH_CITY_LEVEL,
                x,
                y,
                (city.level as f32 / 10.0).clamp(0.0, 1.0),
            );
            set_feat(
                &mut data,
                CH_CITY_PRODUCTION,
                x,
                y,
                (get_city_production(state, city) as f32 / 10.0).clamp(0.0, 1.0),
            );
            set_feat(
                &mut data,
                CH_CITY_BORDER_SIZE,
                x,
                y,
                // default 1, + city border reward = 2
                (city.border_size as f32 / 2.0).clamp(0.0, 1.0),
            );

            // Check if capital
            if let Some(tile) = state.tiles.get(&idx) {
                if tile.capital_of == *player_id {
                    set_feat(&mut data, CH_CITY_IS_CAPITAL, x, y, 1.0);
                }
            }

            if city.connected_to_capital {
                set_feat(&mut data, CH_CITY_CONNECTED, x, y, 1.0);
            }
            if city.has_walls() {
                set_feat(&mut data, CH_CITY_HAS_WALLS, x, y, 1.0);
            }
            if is_under_siege(state, city.idx) {
                set_feat(&mut data, CH_CITY_HAS_RIOT, x, y, 1.0);
            }

            // Pending reward (city just leveled up)
            if !city.rewards.is_empty() {
                set_feat(&mut data, CH_CITY_PENDING_REWARD, x, y, 1.0);
            }

            // City progress toward next population
            let progress_norm = (city.progress as f32 / (city.level + 1) as f32).clamp(0.0, 1.0);
            set_feat(&mut data, CH_CITY_PROGRESS, x, y, progress_norm);
        }
    }

    // Process memory units and attacks
    if let Some(tribe) = pov_tribe {
        let mut visible_enemy_tiles: std::collections::HashSet<i32> =
            std::collections::HashSet::new();
        for (player_id, other_tribe) in &state.tribes {
            if *player_id == perspective {
                continue;
            }
            for unit in &other_tribe.units {
                if unit.effects.contains(&UnitEffect::Invisible) {
                    continue;
                }
                let explored = state
                    .tiles
                    .get(&unit.coords.idx)
                    .map(|t| t.explorers.contains(&perspective))
                    .unwrap_or(false);
                if explored {
                    visible_enemy_tiles.insert(unit.coords.idx);
                }
            }
        }

        for (&idx, mem_unit) in &tribe.memory_units {
            let x = (idx % state.settings.size) as usize;
            let y = (idx / state.settings.size) as usize;
            if x >= MAP_SIZE || y >= MAP_SIZE {
                continue;
            }

            if !visible_enemy_tiles.contains(&idx) {
                let age = current_turn - mem_unit.last_seen_turn;
                if age >= 0 {
                    let decay = crate::memory::MEM_DECAY.powi(age);
                    set_feat(&mut data, CH_MEM_ENEMY_SEEN, x, y, decay);
                    set_feat(&mut data, CH_MEM_ENEMY_HP, x, y, mem_unit.hp_norm);
                    let unit_setting = crate::settings::units::get_unit_setting(mem_unit.unit_type);
                    set_feat(&mut data, CH_MEM_ENEMY_ATTACK, x, y, unit_setting.attack / 5.0);
                    if unit_setting.range > 1 {
                        set_feat(&mut data, CH_MEM_ENEMY_RANGED, x, y, 1.0);
                    }
                    if mem_unit.unit_type.is_naval() {
                        set_feat(&mut data, CH_MEM_ENEMY_NAVAL, x, y, 1.0);
                    }
                }
            }
        }

        for (&idx, &attack_turn) in &tribe.memory_attacks {
            let x = (idx % state.settings.size) as usize;
            let y = (idx / state.settings.size) as usize;
            if x >= MAP_SIZE || y >= MAP_SIZE {
                continue;
            }
            let age = state.settings.turn - attack_turn;
            if age >= 0 {
                let decay = crate::memory::MEM_DECAY.powi(age);
                set_feat(&mut data, CH_MEM_ATTACKED_HERE, x, y, decay);
            }
        }
    }

    // Extract player state vector (16 features)
    let player_vec = vec![
        turn_norm,
        max_turns_norm,
        stars_norm,
        spt_norm,
        score_norm,
        tech_norm,
        tribe_type_norm,
        game_mode,
        game_over,
        total_cities,
        total_units,
        pacifist_turns,
        tribe_kills,
        tribe_casualties,
        tribe_conversions,
        attacked_this_turn,
    ];

    Ok(RawFeatures {
        spatial: data,
        player: player_vec,
    })
}

// ============================================================================
// Helper Functions
// ============================================================================

#[inline]
fn set_feat(data: &mut Vec<f32>, channel: usize, x: usize, y: usize, val: f32) {
    let idx = channel * (MAP_SIZE * MAP_SIZE) + (y * MAP_SIZE + x);
    if idx < data.len() {
        data[idx] = val;
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Game;

    #[test]
    fn test_cpu_features_match_state_to_tensor() {
        // Guards the state_to_tensor / state_to_cpu_features split: the two
        // paths must produce element-for-element identical tensors.
        let game = Game::default();
        let device = Device::Cpu;

        let direct = state_to_tensor(&game.state, 1, &device).unwrap();
        let via_raw = state_to_cpu_features(&game.state, 1)
            .unwrap()
            .into_game_features(&device)
            .unwrap();

        let direct_spatial = direct
            .spatial_map
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let via_raw_spatial = via_raw
            .spatial_map
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(direct_spatial, via_raw_spatial);

        let direct_player = direct
            .player_state
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let via_raw_player = via_raw
            .player_state
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(direct_player, via_raw_player);
    }

    #[test]
    fn test_tensor_shape() {
        let game = Game::default();
        let features = state_to_tensor(&game.state, 1, &Device::Cpu).unwrap();
        let dims = features.spatial_map.dims();
        assert_eq!(dims, &[1, NUM_CHANNELS, MAP_SIZE, MAP_SIZE]);
        // Check player state dims
        let player_dims = features.player_state.dims();
        assert_eq!(player_dims, &[1, 16]);
    }

    #[test]
    fn test_channel_ranges_sequential() {
        // Verify channel ranges don't overlap
        assert!(CH_TERRAIN_END <= CH_TILE_FLAGS_START);
        assert!(CH_TILE_FLAGS_END <= CH_RESOURCE_START);
        assert!(CH_RESOURCE_END <= CH_STRUCTURE_START);
        assert!(CH_STRUCTURE_END <= CH_UNIT_START);
        assert!(CH_UNIT_END <= CH_UNIT_STATS_START);
        assert!(CH_UNIT_STATS_END <= CH_CITY_STATS_START);
    }

    #[test]
    fn test_terrain_channels_sequential() {
        let water_idx = TERRAIN_INDEX.get(&TerrainType::Water).unwrap();
        let ocean_idx = TERRAIN_INDEX.get(&TerrainType::Ocean).unwrap();
        // They should be sequential indices, not the raw enum values
        assert!(*water_idx < TERRAIN_COUNT);
        assert!(*ocean_idx < TERRAIN_COUNT);
        assert_ne!(water_idx, ocean_idx);
    }

    #[test]
    fn test_unit_channels_sequential() {
        let warrior_idx = UNIT_INDEX.get(&UnitType::Warrior).unwrap();
        let giant_idx = UNIT_INDEX.get(&UnitType::Giant).unwrap();
        // Raw enum: Warrior=2, Giant=12, but indices should be 0, 1, 2, ...
        assert!(*warrior_idx < UNIT_COUNT);
        assert!(*giant_idx < UNIT_COUNT);
        assert_ne!(warrior_idx, giant_idx);
    }

    #[test]
    fn test_structure_channels_sequential() {
        let village_idx = STRUCTURE_INDEX.get(&StructureType::Village).unwrap();
        let sawmill_idx = STRUCTURE_INDEX.get(&StructureType::Sawmill).unwrap();
        assert!(*village_idx < STRUCTURE_COUNT);
        assert!(*sawmill_idx < STRUCTURE_COUNT);
        assert_ne!(village_idx, sawmill_idx);
    }

    #[test]
    fn test_num_channels() {
        println!("NUM_CHANNELS: {}", NUM_CHANNELS);
        assert_eq!(
            NUM_CHANNELS,
            TERRAIN_COUNT
                + CH_TILE_FLAGS_COUNT
                + RESOURCE_COUNT
                + STRUCTURE_COUNT
                + UNIT_COUNT
                + CH_UNIT_STATS_COUNT
                + CH_CITY_STATS_COUNT
                + CH_MEM_COUNT
        );
    }
}
