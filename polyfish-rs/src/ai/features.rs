//! AI Feature Extraction - Dynamic Channel Mapping
//!
//! This module converts GameState into tensor format for neural network input.
//! Enforces Fog of War and provides full granularity for all game elements.
//!
//! Channel ranges are dynamically computed from enum variants at compile time.

use crate::states::{GameState, PlayerId};
use crate::types::{
    EffectType, ModeType, ResourceType, StructureType, TerrainType, TribeType, UnitType,
};
use candle_core::{Device, Result, Tensor};
use std::sync::LazyLock;
use strum::IntoEnumIterator;

pub const MAP_HEIGHT: usize = 30;
pub const MAP_WIDTH: usize = 30;

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
pub const CH_TILE_FLAGS_COUNT: usize = 8;
pub const CH_TILE_FLAGS_END: usize = CH_TILE_FLAGS_START + CH_TILE_FLAGS_COUNT;

// Tile flag offsets within the range
pub const CH_TILE_FROZEN: usize = CH_TILE_FLAGS_START + 0;
pub const CH_TILE_FLOODED: usize = CH_TILE_FLAGS_START + 1;
pub const CH_TILE_HAS_ROAD: usize = CH_TILE_FLAGS_START + 2;
pub const CH_TILE_HAS_ROUTE: usize = CH_TILE_FLAGS_START + 3;
pub const CH_TILE_OWNER: usize = CH_TILE_FLAGS_START + 4;
pub const CH_TILE_CLIMATE: usize = CH_TILE_FLAGS_START + 5;
pub const CH_TILE_IS_EXPLORED: usize = CH_TILE_FLAGS_START + 6;
// +7 reserved

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

// City stats (fixed count: 13)
pub const CH_CITY_STATS_START: usize = CH_UNIT_STATS_END;
pub const CH_CITY_STATS_COUNT: usize = 13;
pub const CH_CITY_STATS_END: usize = CH_CITY_STATS_START + CH_CITY_STATS_COUNT;

// City stat offsets
pub const CH_CITY_PRESENT: usize = CH_CITY_STATS_START + 0;
pub const CH_CITY_OWNER: usize = CH_CITY_STATS_START + 1;
pub const CH_CITY_LEVEL: usize = CH_CITY_STATS_START + 2;
pub const CH_CITY_POPULATION: usize = CH_CITY_STATS_START + 3;
pub const CH_CITY_PRODUCTION: usize = CH_CITY_STATS_START + 4;
pub const CH_CITY_IS_CAPITAL: usize = CH_CITY_STATS_START + 5;
pub const CH_CITY_CONNECTED: usize = CH_CITY_STATS_START + 6;
pub const CH_CITY_HAS_WALLS: usize = CH_CITY_STATS_START + 7;
pub const CH_CITY_HAS_RIOT: usize = CH_CITY_STATS_START + 8;
pub const CH_CITY_PENDING_REWARD: usize = CH_CITY_STATS_START + 9;
pub const CH_CITY_BORDER_SIZE: usize = CH_CITY_STATS_START + 10;
pub const CH_CITY_PROGRESS: usize = CH_CITY_STATS_START + 11;
// +12 reserved

// Global/Meta (fixed count: 20)
pub const CH_GLOBAL_START: usize = CH_CITY_STATS_END;
pub const CH_GLOBAL_COUNT: usize = 20;
pub const CH_GLOBAL_END: usize = CH_GLOBAL_START + CH_GLOBAL_COUNT;

// Global offsets
pub const CH_VISIBILITY: usize = CH_GLOBAL_START + 0;
pub const CH_TURN: usize = CH_GLOBAL_START + 1;
pub const CH_MAX_TURNS: usize = CH_GLOBAL_START + 2;
pub const CH_STARS: usize = CH_GLOBAL_START + 3;
pub const CH_SCORE: usize = CH_GLOBAL_START + 4;
pub const CH_TECH_COUNT: usize = CH_GLOBAL_START + 5;
pub const CH_MY_TRIBE_TYPE: usize = CH_GLOBAL_START + 6;
pub const CH_GAME_MODE: usize = CH_GLOBAL_START + 7;
pub const CH_GAME_OVER: usize = CH_GLOBAL_START + 8;
pub const CH_AT_PEACE: usize = CH_GLOBAL_START + 9;
pub const CH_EMBASSY_LEVEL: usize = CH_GLOBAL_START + 10;
pub const CH_PACIFIST_TURNS: usize = CH_GLOBAL_START + 11;
pub const CH_TOTAL_CITIES: usize = CH_GLOBAL_START + 12;
pub const CH_TOTAL_UNITS: usize = CH_GLOBAL_START + 13;
pub const CH_TRIBE_KILLS: usize = CH_GLOBAL_START + 14;
pub const CH_TRIBE_CASUALTIES: usize = CH_GLOBAL_START + 15;
pub const CH_TRIBE_CONVERSIONS: usize = CH_GLOBAL_START + 16;
pub const CH_ATTACKED_THIS_TURN: usize = CH_GLOBAL_START + 17;
// +18, +19 reserved

/// Total number of feature channels (dynamically computed)
pub const NUM_CHANNELS: usize = CH_GLOBAL_END;

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

/// Convert game state to tensor with FOW enforcement
pub fn state_to_tensor(state: &GameState, perspective: PlayerId) -> Result<Tensor> {
    let mut data = vec![0.0f32; NUM_CHANNELS * MAP_HEIGHT * MAP_WIDTH];
    let map_size = state.settings.size as usize;

    // Get perspective tribe info
    let pov_tribe = state.tribes.get(&perspective);
    let pov_tribe_type = pov_tribe.map(|t| t.tribe_type).unwrap_or(TribeType::None);
    let is_elyrion = pov_tribe_type == TribeType::Elyrion;

    // Global stats (same for all tiles)
    let turn_norm =
        (state.settings.turn as f32 / state.settings.max_turns.max(1) as f32).clamp(0.0, 1.0);
    let max_turns_norm = (state.settings.max_turns as f32 / 50.0).clamp(0.0, 1.0);
    let stars_norm = pov_tribe
        .map(|t| ((t.stars as f32 + 1.0).ln() / 100.0_f32.ln()).clamp(0.0, 1.0))
        .unwrap_or(0.0);
    let score_norm = pov_tribe
        .map(|t| (t.score as f32 / 10000.0).clamp(0.0, 1.0))
        .unwrap_or(0.0);
    let tech_count = pov_tribe
        .map(|t| t.tech_vanilla.iter().filter(|tech| tech.discovered).count())
        .unwrap_or(0);
    let tech_norm = (tech_count as f32 / 30.0).clamp(0.0, 1.0);
    let tribe_type_norm = (pov_tribe_type as i8 as f32 / 17.0).clamp(0.0, 1.0);
    let game_mode = if state.settings.mode == ModeType::Domination {
        1.0
    } else {
        0.0
    };
    let game_over = if state.settings._game_over { 1.0 } else { 0.0 };
    let total_cities =
        (pov_tribe.map(|t| t.cities.len()).unwrap_or(0) as f32 / 10.0).clamp(0.0, 1.0);
    let total_units = (pov_tribe.map(|t| t.units.len()).unwrap_or(0) as f32 / 20.0).clamp(0.0, 1.0);
    let pacifist_turns = pov_tribe
        .map(|t| (t.pacifist_turns as f32 / 10.0).clamp(0.0, 1.0))
        .unwrap_or(0.0);
    let tribe_kills = pov_tribe
        .map(|t| (t.kills as f32 / 20.0).clamp(0.0, 1.0))
        .unwrap_or(0.0);
    let tribe_casualties = pov_tribe
        .map(|t| (t.casualties as f32 / 20.0).clamp(0.0, 1.0))
        .unwrap_or(0.0);
    let tribe_conversions = pov_tribe
        .map(|t| (t.conversions as f32 / 10.0).clamp(0.0, 1.0))
        .unwrap_or(0.0);
    let attacked_this_turn = pov_tribe
        .map(|t| if t.attacked_this_turn { 1.0 } else { 0.0 })
        .unwrap_or(0.0);

    // Process each tile
    for y in 0..map_size {
        for x in 0..map_size {
            if x >= MAP_WIDTH || y >= MAP_HEIGHT {
                continue;
            }
            let idx = (y * map_size + x) as i32;

            // Check visibility
            let is_visible = state._visible_tiles.contains_key(&idx);
            let is_explored = state
                .tiles
                .get(&idx)
                .map(|t| t.explorers.contains(&perspective))
                .unwrap_or(false);

            // Set visibility channel
            let vis_val = if is_visible {
                1.0
            } else if is_explored {
                0.5
            } else {
                0.0
            };
            set_feat(&mut data, CH_VISIBILITY, x, y, vis_val);

            // Set global channels (same for all tiles)
            set_feat(&mut data, CH_TURN, x, y, turn_norm);
            set_feat(&mut data, CH_MAX_TURNS, x, y, max_turns_norm);
            set_feat(&mut data, CH_STARS, x, y, stars_norm);
            set_feat(&mut data, CH_SCORE, x, y, score_norm);
            set_feat(&mut data, CH_TECH_COUNT, x, y, tech_norm);
            set_feat(&mut data, CH_MY_TRIBE_TYPE, x, y, tribe_type_norm);
            set_feat(&mut data, CH_GAME_MODE, x, y, game_mode);
            set_feat(&mut data, CH_GAME_OVER, x, y, game_over);
            set_feat(&mut data, CH_TOTAL_CITIES, x, y, total_cities);
            set_feat(&mut data, CH_TOTAL_UNITS, x, y, total_units);
            set_feat(&mut data, CH_PACIFIST_TURNS, x, y, pacifist_turns);
            set_feat(&mut data, CH_TRIBE_KILLS, x, y, tribe_kills);
            set_feat(&mut data, CH_TRIBE_CASUALTIES, x, y, tribe_casualties);
            set_feat(&mut data, CH_TRIBE_CONVERSIONS, x, y, tribe_conversions);
            set_feat(&mut data, CH_ATTACKED_THIS_TURN, x, y, attacked_this_turn);

            // Skip tile-specific data if not visible and not explored
            if !is_visible && !is_explored {
                continue;
            }

            if let Some(tile) = state.tiles.get(&idx) {
                // Terrain (always visible if explored)
                let terrain_ch = terrain_to_channel(tile.terrain_type);
                set_feat(&mut data, terrain_ch, x, y, 1.0);

                // Tile flags
                if tile.frozen {
                    set_feat(&mut data, CH_TILE_FROZEN, x, y, 1.0);
                }
                if tile.flooded {
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

                // Peace status with tile owner
                if tile.owner != 0 && tile.owner != perspective {
                    if let Some(pov_t) = pov_tribe {
                        if let Some(rel) = pov_t.relations.get(&tile.owner) {
                            if rel.state == 1 {
                                set_feat(&mut data, CH_AT_PEACE, x, y, 1.0);
                            }
                            set_feat(
                                &mut data,
                                CH_EMBASSY_LEVEL,
                                x,
                                y,
                                rel.embassy_level as f32 / 3.0,
                            );
                        }
                    }
                }

                // Resources (only if visible, not just explored)
                if is_visible {
                    if let Some(Some(resource)) = state.resources.get(&idx) {
                        let res_ch = resource_to_channel(resource.resource_type);
                        set_feat(&mut data, res_ch, x, y, 1.0);
                    }
                }

                // Structures
                if let Some(Some(structure)) = state.structures.get(&idx) {
                    // Ruins are special - Elyrion can see them through fog
                    if is_visible || (is_elyrion && structure.structure_type == StructureType::Ruin)
                    {
                        let struct_ch = structure_to_channel(structure.structure_type);
                        set_feat(&mut data, struct_ch, x, y, 1.0);
                    }
                }
            }
        }
    }

    // Process units (only visible ones)
    for (player_id, tribe) in &state.tribes {
        for unit in &tribe.units {
            let x = unit.coords.x as usize;
            let y = unit.coords.y as usize;
            if x >= MAP_WIDTH || y >= MAP_HEIGHT {
                continue;
            }

            let idx = unit.coords.idx;

            // FOW: Only show units that are visible
            if !state._visible_tiles.contains_key(&idx) {
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
                unit.health as f32 / unit.max_health.max(1) as f32,
            );
            set_feat(
                &mut data,
                CH_UNIT_MAX_HP,
                x,
                y,
                (unit.max_health as f32 / 400.0).clamp(0.0, 1.0),
            );
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
                (unit.kills as f32 / 10.0).clamp(0.0, 1.0),
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
            if unit.effects.contains(&EffectType::Poison) {
                set_feat(&mut data, CH_UNIT_EFFECT_POISON, x, y, 1.0);
            }
            if unit.effects.contains(&EffectType::Boost) {
                set_feat(&mut data, CH_UNIT_EFFECT_BOOST, x, y, 1.0);
            }
            if unit.effects.contains(&EffectType::Invisible) {
                set_feat(&mut data, CH_UNIT_EFFECT_INVISIBLE, x, y, 1.0);
            }
            if unit.effects.contains(&EffectType::Frozen) {
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
            let idx = city.tile_index;
            let x = (idx % state.settings.size) as usize;
            let y = (idx / state.settings.size) as usize;
            if x >= MAP_WIDTH || y >= MAP_HEIGHT {
                continue;
            }

            // FOW: Only show cities we can see
            if !state._visible_tiles.contains_key(&idx) {
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
                CH_CITY_POPULATION,
                x,
                y,
                city.population as f32 / city.level.max(1) as f32,
            );
            set_feat(
                &mut data,
                CH_CITY_PRODUCTION,
                x,
                y,
                (city.production as f32 / 10.0).clamp(0.0, 1.0),
            );
            set_feat(
                &mut data,
                CH_CITY_BORDER_SIZE,
                x,
                y,
                (city.border_size as f32 / 5.0).clamp(0.0, 1.0),
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
            if city._walls {
                set_feat(&mut data, CH_CITY_HAS_WALLS, x, y, 1.0);
            }
            if city._riot {
                set_feat(&mut data, CH_CITY_HAS_RIOT, x, y, 1.0);
            }

            // Pending reward (city just leveled up)
            if !city.rewards.is_empty() {
                set_feat(&mut data, CH_CITY_PENDING_REWARD, x, y, 1.0);
            }

            // City progress toward next population
            let progress_norm = (city.progress as f32 / city.level.max(1) as f32).clamp(0.0, 1.0);
            set_feat(&mut data, CH_CITY_PROGRESS, x, y, progress_norm);
        }
    }

    Tensor::from_vec(data, (1, NUM_CHANNELS, MAP_HEIGHT, MAP_WIDTH), &Device::Cpu)
}

// ============================================================================
// Helper Functions
// ============================================================================

#[inline]
fn set_feat(data: &mut Vec<f32>, channel: usize, x: usize, y: usize, val: f32) {
    let idx = channel * (MAP_HEIGHT * MAP_WIDTH) + (y * MAP_WIDTH + x);
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
    fn test_tensor_shape() {
        let game = Game::default();
        let tensor = state_to_tensor(&game.state, 1).unwrap();
        let dims = tensor.dims();
        assert_eq!(dims, &[1, NUM_CHANNELS, MAP_HEIGHT, MAP_WIDTH]);
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
        assert!(CH_CITY_STATS_END <= CH_GLOBAL_START);
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
        // Should be around 149 based on current counts:
        // 8 terrain + 8 tile flags + 9 resources + 35 structures + 46 units
        // + 15 unit stats + 12 city stats + 16 global = 149
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
                + CH_GLOBAL_COUNT
        );
    }
}
