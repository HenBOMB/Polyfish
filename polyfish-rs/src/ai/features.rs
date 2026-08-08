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

// Global/Meta (fixed count: 24)
pub const CH_GLOBAL_START: usize = CH_CITY_STATS_END;
pub const CH_GLOBAL_COUNT: usize = 24;
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
// Opponent aggregates: public/derivable info the net shouldn't have to mine
// out of the spatial planes (the value label is my-score minus opp-score).
pub const CH_OPP_SCORE: usize = CH_GLOBAL_START + 18;
pub const CH_FRAC_EXPLORED: usize = CH_GLOBAL_START + 19;
pub const CH_VISIBLE_ENEMY_PRODUCTION: usize = CH_GLOBAL_START + 20;
pub const CH_VISIBLE_ENEMY_UNITS: usize = CH_GLOBAL_START + 21;
pub const CH_ENEMY_TYPES_SEEN: usize = CH_GLOBAL_START + 22;
pub const CH_ENEMY_MAX_TIER: usize = CH_GLOBAL_START + 23;

// Ghost sightings (observation memory): last-seen enemy units that departed
// into our fog. See TribeState::enemy_ghosts.
pub const CH_GHOST_START: usize = CH_GLOBAL_END;
pub const CH_GHOST_COUNT: usize = 3;
pub const CH_GHOST_END: usize = CH_GHOST_START + CH_GHOST_COUNT;
pub const CH_GHOST_PRESENT: usize = CH_GHOST_START + 0;
pub const CH_GHOST_TYPE: usize = CH_GHOST_START + 1;
pub const CH_GHOST_AGE: usize = CH_GHOST_START + 2;

// Pursuit: per-tile proximity to the nearest capturable (unowned, explored)
// village — (CAP - Chebyshev dist).max(0)/CAP, CAP = reward::SHAPE_PROX_CAP.
// Appended at the END of the layout for index-stability with older training
// data (which is zero-padded at load). See notes on pursuit-representation.
pub const CH_PURSUIT_START: usize = CH_GHOST_END;
pub const CH_PURSUIT_COUNT: usize = 1;
pub const CH_PURSUIT_END: usize = CH_PURSUIT_START + CH_PURSUIT_COUNT;

// EXP_ELO_028 goal channels (appended; all-zero = "no goal set"). Orders are
// painted as capped proximity blobs (same formula as the pursuit channel),
// max-merged so concurrent same-kind orders coexist; the stance is a one-hot
// constant plane. Offsets within each block follow the `OrderKind` / `Stance`
// enum discriminants in oracle_macro.rs.
pub const CH_ORDER_START: usize = CH_PURSUIT_END;
pub const CH_ORDER_COUNT: usize = 3;
pub const CH_ORDER_END: usize = CH_ORDER_START + CH_ORDER_COUNT;
pub const CH_STANCE_START: usize = CH_ORDER_END;
pub const CH_STANCE_COUNT: usize = 4;
pub const CH_STANCE_END: usize = CH_STANCE_START + CH_STANCE_COUNT;

/// Total number of feature channels (dynamically computed)
pub const NUM_CHANNELS: usize = CH_STANCE_END;

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
    pub player: Vec<f32>,  // len = PLAYER_STATE_DIM (10)
}

impl RawFeatures {
    pub const PLAYER_STATE_DIM: usize = 10;

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
    state_to_cpu_features_focused(state, perspective, None)
}

/// `state_to_cpu_features` with an EXP_ELO_026 oracle-macro commitment: when
/// `pursuit_focus` is a tile that still holds a capturable village for
/// `perspective`, the pursuit channel's proximity field is computed from that
/// village alone; any other value falls back to the all-villages field.
pub fn state_to_cpu_features_focused(
    state: &GameState,
    perspective: PlayerId,
    pursuit_focus: Option<i32>,
) -> Result<RawFeatures> {
    state_to_cpu_features_goal(state, perspective, pursuit_focus, None)
}

/// `state_to_cpu_features_focused` plus the EXP_ELO_028 goal channels: each
/// order paints a capped proximity blob (max-merged) into its kind's plane
/// and the stance fills its one-hot plane. `None` leaves all six planes zero
/// ("no goal set" — the pre-Stage-1 semantics old data zero-pads to).
pub fn state_to_cpu_features_goal(
    state: &GameState,
    perspective: PlayerId,
    pursuit_focus: Option<i32>,
    macro_goal: Option<&crate::ai::oracle_macro::MacroGoal>,
) -> Result<RawFeatures> {
    let mut data = vec![0.0f32; NUM_CHANNELS * MAP_SIZE * MAP_SIZE];
    let map_size = state.settings.size as usize;

    // Get perspective tribe info
    let pov_tribe = state.tribes.get(&perspective);
    let pov_tribe_type = pov_tribe.map(|t| t.tribe_type).unwrap_or(TribeType::None);
    let is_elyrion = pov_tribe_type == TribeType::Elyrion;

    // Global stats (same for all tiles)
    let turn_norm = (state.settings.turn as f32 / state.settings.max_turns as f32).clamp(0.0, 1.0);
    let max_turns_norm = (state.settings.max_turns as f32
        / crate::states::default_max_turns() as f32)
        .clamp(0.0, 1.0);
    let stars_norm = pov_tribe
        .map(|t| (t.stars as f32 / crate::states::default_max_stars() as f32).clamp(0.0, 1.0))
        .unwrap_or(0.0);
    let spt_norm = pov_tribe
        .map(|t| {
            (crate::functions::get_tribe_spt(state, t) as f32
                / crate::states::default_max_spt() as f32)
                .clamp(0.0, 1.0)
        })
        .unwrap_or(0.0);
    let score_norm = pov_tribe
        .map(|t| (t.score as f32 / crate::states::default_max_score() as f32).clamp(0.0, 1.0))
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
    let total_units = (pov_tribe.map(|t| t.units.len()).unwrap_or(0) as f32 / 20.0).clamp(0.0, 1.0);
    // at turn 5 it stops tracking
    let pacifist_turns = pov_tribe
        .map(|t| (t.pacifist_turns as f32 / 5.0).clamp(0.0, 1.0))
        .unwrap_or(0.0);
    let tribe_kills = pov_tribe
        .map(|t| (t.kills as f32 / crate::states::default_max_units() as f32).clamp(0.0, 1.0))
        .unwrap_or(0.0);
    let tribe_casualties = pov_tribe
        .map(|t| (t.casualties as f32 / crate::states::default_max_units() as f32).clamp(0.0, 1.0))
        .unwrap_or(0.0);
    let tribe_conversions = pov_tribe
        .map(|t| (t.conversions as f32 / crate::states::default_max_units() as f32).clamp(0.0, 1.0))
        .unwrap_or(0.0);
    let attacked_this_turn = pov_tribe
        .map(|t| if t.attacked_this_turn { 1.0 } else { 0.0 })
        .unwrap_or(0.0);

    // Opponent aggregates (all FOW-legal: scoreboard is public; the rest is
    // computed strictly from POV-explored tiles / POV observation memory).
    let opp_score_norm = state
        .tribes
        .iter()
        .filter(|(id, _)| **id != perspective)
        .map(|(_, t)| t.score)
        .max()
        .map(|s| (s as f32 / crate::states::default_max_score() as f32).clamp(0.0, 1.0))
        .unwrap_or(0.0);
    let frac_explored = if state.tiles.is_empty() {
        0.0
    } else {
        state
            .tiles
            .values()
            .filter(|t| t.explorers.contains(&perspective))
            .count() as f32
            / state.tiles.len() as f32
    };
    let tile_explored = |idx: i32| {
        state
            .tiles
            .get(&idx)
            .map(|t| t.explorers.contains(&perspective))
            .unwrap_or(false)
    };
    let mut visible_enemy_production = 0.0f32;
    let mut visible_enemy_units = 0.0f32;
    for (id, tribe) in &state.tribes {
        if *id == perspective {
            continue;
        }
        for city in &tribe.cities {
            if tile_explored(city.idx) {
                visible_enemy_production += get_city_production(state, city) as f32;
            }
        }
        for unit in &tribe.units {
            if tile_explored(unit.coords.idx) && !unit.effects.contains(&UnitEffect::Invisible) {
                visible_enemy_units += 1.0;
            }
        }
    }
    let visible_enemy_production = (visible_enemy_production / 30.0).clamp(0.0, 1.0);
    let visible_enemy_units = (visible_enemy_units / 20.0).clamp(0.0, 1.0);
    let enemy_types_seen_norm = pov_tribe
        .map(|t| (t.enemy_types_seen.len() as f32 / 10.0).clamp(0.0, 1.0))
        .unwrap_or(0.0);
    // Star cost as a tier proxy for the strongest enemy unit type evidenced.
    let enemy_max_tier = pov_tribe
        .and_then(|t| {
            t.enemy_types_seen
                .iter()
                .map(|ty| crate::settings::units::get_unit_setting(*ty).cost)
                .max()
        })
        .map(|c| (c as f32 / 10.0).clamp(0.0, 1.0))
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
            set_feat(&mut data, CH_OPP_SCORE, x, y, opp_score_norm);
            set_feat(&mut data, CH_FRAC_EXPLORED, x, y, frac_explored);
            set_feat(&mut data, CH_VISIBLE_ENEMY_PRODUCTION, x, y, visible_enemy_production);
            set_feat(&mut data, CH_VISIBLE_ENEMY_UNITS, x, y, visible_enemy_units);
            set_feat(&mut data, CH_ENEMY_TYPES_SEEN, x, y, enemy_types_seen_norm);
            set_feat(&mut data, CH_ENEMY_MAX_TIER, x, y, enemy_max_tier);

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

            // FOW rule parity with legal-move gen: invisible (Cloak) enemies
            // are hidden even on explored tiles.
            if *player_id != perspective && unit.effects.contains(&UnitEffect::Invisible) {
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
                (unit.kills as f32 / crate::rules::combat::PROMOTION_KILLS as f32).clamp(0.0, 1.0),
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

    // Ghost sightings from observation memory. Suppressed where a live enemy
    // unit is rendered (a ghost tile explored mid-turn, before the sweep drops it).
    if let Some(pov) = pov_tribe {
        let turn_now = state.settings.turn;
        for (&g_idx, ghost) in &pov.enemy_ghosts {
            if g_idx < 0 {
                continue;
            }
            let x = g_idx as usize % map_size;
            let y = g_idx as usize / map_size;
            if x >= MAP_SIZE || y >= MAP_SIZE {
                continue;
            }
            let live_enemy_here = state
                .tiles
                .get(&g_idx)
                .map(|t| {
                    t.explorers.contains(&perspective)
                        && t._unit_owner_id.map_or(false, |o| o != perspective)
                })
                .unwrap_or(false);
            if live_enemy_here {
                continue;
            }
            set_feat(&mut data, CH_GHOST_PRESENT, x, y, 1.0);
            set_feat(
                &mut data,
                CH_GHOST_TYPE,
                x,
                y,
                ghost.unit_type as i8 as f32 / UNIT_COUNT as f32,
            );
            set_feat(
                &mut data,
                CH_GHOST_AGE,
                x,
                y,
                ((turn_now - ghost.turn) as f32 / 10.0).clamp(0.0, 1.0),
            );
        }
    }

    // Pursuit channel: per-tile proximity to the nearest capturable village.
    // Capturable = Village structure on an unowned tile explored by the POV
    // (mirrors reward::village_proximity_tiles' predicate, but tile-centric so
    // the policy target head and value head can read a directional gradient).
    {
        let cap = crate::ai::reward::SHAPE_PROX_CAP as f32;
        let villages: Vec<(i32, i32)> = state
            .structures
            .iter()
            .filter_map(|(&idx, s)| {
                s.as_ref()?;
                if !crate::rules::capture::is_capturable(
                    state,
                    idx,
                    perspective,
                    crate::rules::capture::CaptureKind::OPEN_VILLAGE,
                    true,
                ) {
                    return None;
                }
                Some((idx / map_size as i32, idx % map_size as i32)) // (row, col)
            })
            .collect();
        let villages = match pursuit_focus {
            Some(f) if villages.contains(&(f / map_size as i32, f % map_size as i32)) => {
                vec![(f / map_size as i32, f % map_size as i32)]
            }
            _ => villages,
        };
        if !villages.is_empty() {
            for y in 0..MAP_SIZE {
                for x in 0..MAP_SIZE {
                    let d = villages
                        .iter()
                        .map(|&(vr, vc)| (y as i32 - vr).abs().max((x as i32 - vc).abs()))
                        .min()
                        .unwrap();
                    let prox = (cap - d as f32).max(0.0) / cap;
                    if prox > 0.0 {
                        set_feat(&mut data, CH_PURSUIT_START, x, y, prox);
                    }
                }
            }
        }
    }

    // EXP_ELO_028 goal channels: painted orders + stance one-hot.
    if let Some(goal) = macro_goal {
        let cap = crate::ai::reward::SHAPE_PROX_CAP as f32;
        for &(kind, target) in &goal.orders {
            let ch = CH_ORDER_START + kind as usize;
            let (tr, tc) = (target / map_size as i32, target % map_size as i32);
            for y in 0..MAP_SIZE {
                for x in 0..MAP_SIZE {
                    let d = (y as i32 - tr).abs().max((x as i32 - tc).abs());
                    let prox = (cap - d as f32).max(0.0) / cap;
                    let idx = ch * (MAP_SIZE * MAP_SIZE) + y * MAP_SIZE + x;
                    if prox > data[idx] {
                        data[idx] = prox;
                    }
                }
            }
        }
        let stance_ch = CH_STANCE_START + goal.stance as usize;
        for i in 0..MAP_SIZE * MAP_SIZE {
            data[stance_ch * (MAP_SIZE * MAP_SIZE) + i] = 1.0;
        }
    }

    // Extract player state vector (10 features)
    let player_vec = vec![
        turn_norm,
        stars_norm,
        spt_norm,
        tech_norm,
        total_cities,
        total_units,
        score_norm,
        tribe_kills,
        tribe_casualties,
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

    /// EXP_ELO_026: a valid pursuit focus narrows the pursuit channel to the
    /// committed village's own proximity field; an invalid focus (tile not
    /// capturable) falls back to the normal all-villages field.
    #[test]
    fn test_pursuit_focus_narrows_channel_and_falls_back_when_invalid() {
        use crate::states::{StructureState, TileState};
        use crate::types::StructureType;

        let mut game = Game::default();
        for idx in [0i32, 60] {
            game.state.structures.insert(
                idx,
                Some(StructureState {
                    structure_type: StructureType::Village,
                    level: 0,
                    founded: 0,
                }),
            );
            let mut tile = TileState::default();
            tile.explorers.insert(1);
            game.state.tiles.insert(idx, tile);
        }

        let pursuit = |state: &crate::states::GameState, focus: Option<i32>| -> Vec<f32> {
            let raw = state_to_cpu_features_focused(state, 1, focus).unwrap();
            raw.spatial[CH_PURSUIT_START * MAP_SIZE * MAP_SIZE..CH_PURSUIT_END * MAP_SIZE * MAP_SIZE]
                .to_vec()
        };

        let both = pursuit(&game.state, None);
        let focused = pursuit(&game.state, Some(60));
        assert_ne!(both, focused, "focus on one of two villages must change the field");

        // The focused field is exactly the single-village formula around 60.
        let cap = crate::ai::reward::SHAPE_PROX_CAP as f32;
        for y in 0..MAP_SIZE {
            for x in 0..MAP_SIZE {
                let d = (y as i32 - 60 / 11).abs().max((x as i32 - 60 % 11).abs());
                let expected = (cap - d as f32).max(0.0) / cap;
                assert!((focused[y * MAP_SIZE + x] - expected).abs() < 1e-6);
            }
        }

        // A focus that fails the capturable predicate is ignored.
        game.state.tiles.get_mut(&60).unwrap().owner = 2;
        let after_capture = pursuit(&game.state, Some(60));
        let unfocused_after = pursuit(&game.state, None);
        assert_eq!(after_capture, unfocused_after);
    }

    /// EXP_ELO_028: orders paint max-merged proximity blobs into their kind's
    /// plane, the stance fills its one-hot plane, and no goal leaves all six
    /// appended planes zero.
    #[test]
    fn test_goal_channels_paint_orders_and_stance() {
        use crate::ai::oracle_macro::{MacroGoal, OrderKind, Stance};
        let game = Game::default();
        let goal = MacroGoal {
            orders: vec![
                (OrderKind::Expand, 0),
                (OrderKind::Expand, 60),
                (OrderKind::Attack, 5),
            ],
            stance: Stance::Arm,
            save_target: None,
        };
        let raw = state_to_cpu_features_goal(&game.state, 1, None, Some(&goal)).unwrap();
        let plane =
            |r: &RawFeatures, ch: usize| r.spatial[ch * MAP_SIZE * MAP_SIZE..(ch + 1) * MAP_SIZE * MAP_SIZE].to_vec();
        let cap = crate::ai::reward::SHAPE_PROX_CAP as f32;

        let expand = plane(&raw, CH_ORDER_START);
        for y in 0..MAP_SIZE {
            for x in 0..MAP_SIZE {
                let d0 = (y as i32).max(x as i32);
                let d60 = (y as i32 - 60 / 11).abs().max((x as i32 - 60 % 11).abs());
                let expected = (cap - d0.min(d60) as f32).max(0.0) / cap;
                assert!((expand[y * MAP_SIZE + x] - expected).abs() < 1e-6);
            }
        }
        assert!(plane(&raw, CH_ORDER_START + 1)[5] > 0.99);
        assert!(plane(&raw, CH_ORDER_START + 2).iter().all(|&v| v == 0.0));
        assert!(plane(&raw, CH_STANCE_START + 1).iter().all(|&v| v == 1.0));
        assert!(plane(&raw, CH_STANCE_START).iter().all(|&v| v == 0.0));
        assert!(plane(&raw, CH_STANCE_START + 2).iter().all(|&v| v == 0.0));

        let raw0 = state_to_cpu_features_goal(&game.state, 1, None, None).unwrap();
        for ch in CH_ORDER_START..CH_STANCE_END {
            assert!(plane(&raw0, ch).iter().all(|&v| v == 0.0));
        }
    }

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

        let direct_spatial = direct.spatial_map.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let via_raw_spatial = via_raw.spatial_map.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(direct_spatial, via_raw_spatial);

        let direct_player = direct.player_state.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let via_raw_player = via_raw.player_state.flatten_all().unwrap().to_vec1::<f32>().unwrap();
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
        assert_eq!(player_dims, &[1, 10]);
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
        assert!(CH_GLOBAL_END <= CH_GHOST_START);
        assert!(CH_GHOST_END <= CH_PURSUIT_START);
        assert!(CH_PURSUIT_END <= CH_ORDER_START);
        assert!(CH_ORDER_END <= CH_STANCE_START);
        assert_eq!(CH_STANCE_END, NUM_CHANNELS);
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
                + CH_GLOBAL_COUNT
                + CH_GHOST_COUNT
                + CH_PURSUIT_COUNT
                + CH_ORDER_COUNT
                + CH_STANCE_COUNT
        );
        // Pinned: the trained model's conv1 input width. Bump deliberately
        // (with a weight migration) when adding channels.
        // 168 -> 169: v7 SAVE stance plane (migrate_save_stance.py).
        assert_eq!(NUM_CHANNELS, 169);
    }
}
