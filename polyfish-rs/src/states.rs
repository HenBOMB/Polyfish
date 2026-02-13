//! Game state structures translated from TypeScript

use crate::coords::Coords;
use crate::types::*;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

pub const HEALTH_SCALE: i32 = 10;

/// Helper module for deserializing booleans that may come as integers (0/1)
mod flex_bool {
    use serde::{self, Deserialize, Deserializer};

    pub fn deserialize<'de, D>(deserializer: D) -> Result<bool, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Try to deserialize as various types
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum BoolOrInt {
            Bool(bool),
            Int(i64),
        }

        match BoolOrInt::deserialize(deserializer)? {
            BoolOrInt::Bool(b) => Ok(b),
            BoolOrInt::Int(i) => Ok(i != 0),
        }
    }
}

/// Player ID type (positive integer)
pub type PlayerId = i32;

/// Diplomacy relation state between two players
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DiplomacyRelationState {
    #[serde(default)]
    pub state: i32, // 0 = not at peace, 1 = at peace (serialized as int)
    #[serde(default)]
    pub last_attack_turn: i32,
    #[serde(default)]
    pub embassy_level: i32,
    #[serde(default)]
    pub last_peace_broken_turn: i32,
    #[serde(default)]
    pub first_meet: i32,
    #[serde(default)]
    pub embassy_build_turn: i32,
    #[serde(default)]
    pub previous_attack_turn: i32,
}

/// State of a single tile on the map
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TileState {
    pub coords: Coords,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ruling_city_coords: Option<Coords>,
    #[serde(rename = "type")]
    pub terrain_type: TerrainType,
    #[serde(default)]
    pub explorers: HashSet<i32>,
    #[serde(default)]
    pub has_road: bool,
    #[serde(default)]
    pub has_route: bool,
    #[serde(default)]
    pub had_route: bool,
    #[serde(default)]
    pub capital_of: PlayerId,
    #[serde(default)]
    pub skin_type: i32,
    #[serde(default)]
    pub climate: ClimateType,
    #[serde(default)]
    pub owner: PlayerId,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub _unit_owner_id: Option<PlayerId>,
    #[serde(default, deserialize_with = "flex_bool::deserialize")]
    pub frozen: bool,
    #[serde(default, deserialize_with = "flex_bool::deserialize")]
    pub flooded: bool,
}

impl Default for TileState {
    fn default() -> Self {
        Self {
            coords: Coords::default(),
            ruling_city_coords: None,
            terrain_type: TerrainType::None,
            explorers: HashSet::new(),
            has_road: false,
            has_route: false,
            had_route: false,
            capital_of: 0,
            skin_type: 0,
            climate: ClimateType::Nature,
            owner: 0,
            _unit_owner_id: None,
            frozen: false,
            flooded: false,
        }
    }
}

/// State of a structure on the map
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StructureState {
    #[serde(rename = "type")]
    pub structure_type: StructureType,
    pub level: i32,
    pub founded: i32,
    pub score: i32,
    pub tile_index: i32,
}

/// State of a resource on the map
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourceState {
    #[serde(rename = "type")]
    pub resource_type: ResourceType,
    pub tile_index: i32,
}

/// State of a unit
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UnitState {
    pub owner: PlayerId,
    #[serde(rename = "type")]
    pub unit_type: UnitType,
    pub health: i32,
    #[serde(default)]
    pub max_health: i32,
    #[serde(default, deserialize_with = "flex_bool::deserialize")]
    pub veteran: bool,
    #[serde(default)]
    pub kills: i32,
    pub coords: Coords,
    pub prev_coords: Coords,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub home_coords: Option<Coords>,
    #[serde(default)]
    pub city_id: i32, // Convenience ID for home city (tile index)
    #[serde(default)]
    pub direction: i32,
    #[serde(default, deserialize_with = "flex_bool::deserialize")]
    pub flipped: bool,
    #[serde(default)]
    pub created_turn: i32,
    #[serde(default, deserialize_with = "flex_bool::deserialize")]
    pub moved: bool,
    #[serde(default, deserialize_with = "flex_bool::deserialize")]
    pub attacked: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub passenger_type: Option<UnitType>,
    #[serde(default)]
    pub effects: HashSet<EffectType>,
    #[serde(default, deserialize_with = "flex_bool::deserialize")]
    pub converted: bool,
    #[serde(default)]
    pub attacks_performed: i32,
    /// Index of parent unit in the tribe's unit vector (for segments following parent)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_unit_idx: Option<usize>,
    /// Index of child unit in the tribe's unit vector (for centipede head tracking first segment)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub child_unit_idx: Option<usize>,
}

impl Default for UnitState {
    fn default() -> Self {
        Self {
            owner: 0,
            unit_type: UnitType::None,
            health: 10 * HEALTH_SCALE,
            max_health: 10 * HEALTH_SCALE,
            veteran: false,
            kills: 0,
            coords: Coords::default(),
            prev_coords: Coords::default(),
            home_coords: None,
            city_id: -1,
            direction: 0,
            flipped: false,
            created_turn: 0,
            moved: false,
            attacked: false,
            passenger_type: None,
            effects: HashSet::new(),
            converted: false,
            attacks_performed: 0,
            parent_unit_idx: None,
            child_unit_idx: None,
        }
    }
}

/// State of a city
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CityState {
    pub id: i32, // Equal to tile_index for convenience
    pub name: String,
    pub tile_index: i32,
    #[serde(default)]
    pub population: i32,
    #[serde(default)]
    pub progress: i32,
    #[serde(default)]
    pub border_size: i32,
    #[serde(default)]
    pub connected_to_capital: bool,
    #[serde(default)]
    pub level: i32,
    #[serde(default)]
    pub production: i32,
    pub owner: PlayerId,
    #[serde(default)]
    pub rewards: HashSet<RewardType>,
    #[serde(default)]
    pub _territory: Vec<i32>,
    #[serde(default)]
    pub _walls: bool,
    #[serde(default)]
    pub _riot: bool,
}

impl Default for CityState {
    fn default() -> Self {
        Self {
            id: 0,
            name: String::new(),
            tile_index: 0,
            population: 0,
            progress: 0,
            border_size: 1,
            connected_to_capital: false,
            level: 1,
            production: 0,
            owner: 0,
            rewards: HashSet::new(),
            _territory: Vec::new(),
            _walls: false,
            _riot: false,
        }
    }
}

/// Technology state
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TechnologyState {
    #[serde(rename = "type")]
    pub tech_type: TechnologyType,
    pub discovered: bool,
}

/// State of a tribe/player
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TribeState {
    #[serde(default)]
    pub _hash: u64,
    pub id: PlayerId,
    #[serde(default)]
    pub username: String,
    #[serde(default)]
    pub built_unique_improvements: HashSet<StructureType>,
    #[serde(default)]
    pub known_players: HashSet<PlayerId>,
    #[serde(default)]
    pub bot: bool,
    #[serde(default)]
    pub score: i32,
    #[serde(default)]
    pub stars: i32,
    #[serde(rename = "type")]
    pub tribe_type: TribeType,
    #[serde(default)]
    pub killer_id: PlayerId,
    #[serde(default)]
    pub kills: i32,
    #[serde(default)]
    pub casualties: i32,
    #[serde(default)]
    #[serde(rename = "tech_vanilla")]
    pub tech_vanilla: Vec<TechnologyState>,
    #[serde(default)]
    pub cities: Vec<CityState>,
    #[serde(default)]
    pub units: Vec<UnitState>,
    #[serde(default)]
    pub relations: IndexMap<PlayerId, DiplomacyRelationState>,
    #[serde(default)]
    pub killed_turn: i32,
    #[serde(default)]
    pub resigned_turn: i32,
    #[serde(default)]
    pub starting_tile_coords: Coords,
    #[serde(default)]
    pub attacked_this_turn: bool,
    #[serde(default)]
    pub pacifist_turns: i32,
    #[serde(default)]
    pub conversions: i32,
}

impl Default for TribeState {
    fn default() -> Self {
        Self {
            _hash: 0,
            id: 0,
            username: String::new(),
            built_unique_improvements: HashSet::new(),
            known_players: HashSet::new(),
            bot: false,
            score: 0,
            stars: 0,
            tribe_type: TribeType::None,
            killer_id: 0,
            kills: 0,
            casualties: 0,
            tech_vanilla: Vec::new(),
            cities: Vec::new(),
            units: Vec::new(),
            relations: IndexMap::new(),
            killed_turn: 0,
            resigned_turn: 0,
            starting_tile_coords: Coords::default(),
            attacked_this_turn: false,
            pacifist_turns: 0,
            conversions: 0,
        }
    }
}

/// Game settings
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GameSettings {
    pub mode: ModeType,
    #[serde(default)]
    pub map_type: MapType,
    pub size: i32,
    #[serde(default)]
    pub tile_count: i32,
    #[serde(default = "default_turn")]
    pub turn: i32,
    #[serde(default = "default_max_turns")]
    pub max_turns: i32,
    #[serde(default)]
    pub current_player_turn_id: i32,
    #[serde(default)]
    pub version: i32,
    #[serde(default)]
    pub game_name: String,
    #[serde(default)]
    pub seed: u64,
    #[serde(default)]
    pub win_by_capital: bool,
    #[serde(default)]
    pub win_by_extermination: bool,
    #[serde(default)]
    #[serde(rename = "_lastPlayerTurnId")]
    pub _last_player_turn_id: i32,
    #[serde(rename = "_areYouSure")]
    pub _are_you_sure: bool,
    #[serde(rename = "_gameOver")]
    pub _game_over: bool,
    #[serde(rename = "_recentMoves")]
    pub _recent_moves: Vec<MoveType>,
    // Note: _pending_rewards handled separately as it contains Move objects
    #[serde(rename = "_fow", default = "default_fow")]
    pub _fow: bool,
    #[serde(rename = "_maxTribeCount")]
    pub _max_tribe_count: i32,
    #[serde(default)]
    pub verbose: bool,
}

pub fn default_turn() -> i32 {
    1
}
pub fn default_max_turns() -> i32 {
    10
}
pub fn default_max_score() -> i32 {
    5000
}
pub fn default_max_stars() -> i32 {
    30
}
pub fn default_max_spt() -> i32 {
    25
}
pub fn default_max_units() -> i32 {
    20
}
pub fn default_fow() -> bool {
    true
}

impl Default for GameSettings {
    fn default() -> Self {
        Self {
            mode: ModeType::Domination,
            map_type: MapType::Drylands,
            size: 11,
            tile_count: 11 * 11,
            turn: 0,
            max_turns: default_max_turns(),
            current_player_turn_id: 1,
            version: 0,
            game_name: String::new(),
            seed: 0,
            win_by_capital: false,
            win_by_extermination: false,
            _last_player_turn_id: 0,
            _are_you_sure: false,
            _game_over: false,
            _recent_moves: Vec::new(),
            _fow: true,
            _max_tribe_count: 0,
            verbose: false,
        }
    }
}

/// Prediction state for fog of war
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PredictionState {
    #[serde(default)]
    pub _villages: IndexMap<i32, (TribeType, bool)>,
    #[serde(default)]
    pub _terrain: IndexMap<i32, (TerrainType, ClimateType)>,
    #[serde(default)]
    pub _enemy_capital_suspects: Vec<i32>,
    #[serde(default)]
    pub _city_rewards: Vec<RewardType>,
}

/// Combat result
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CombatResult {
    /// Damage dealt by the attacker
    pub attack_damage: f32,
    /// Damage dealt by the defender as retaliation (0 if defender dies)
    pub defense_damage: f32,
    /// Splash damage from attacker
    pub splash_damage: f32,
}

/// Actions to be performed at the end of a turn
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum EndOfTurnAction {
    Decompose { tile_index: i32, owner_id: PlayerId },
}

/// Full game state
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GameState {
    pub settings: GameSettings,
    pub tiles: IndexMap<i32, TileState>,
    #[serde(default)]
    pub structures: IndexMap<i32, Option<StructureState>>,
    #[serde(default)]
    pub resources: IndexMap<i32, Option<ResourceState>>,
    pub tribes: IndexMap<PlayerId, TribeState>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub _prediction: Option<PredictionState>,
    #[serde(default)]
    pub _end_of_turn_queue: Vec<EndOfTurnAction>,
    #[serde(default)]
    pub _hidden_resources: IndexMap<i32, Option<ResourceState>>,
    #[serde(default)]
    pub _messages: Vec<String>,
    #[serde(default)]
    pub history: Vec<serde_json::Value>,
    #[serde(default)]
    pub initial_seed: u64,
}

impl Default for GameState {
    fn default() -> Self {
        Self {
            settings: GameSettings::default(),
            tiles: IndexMap::new(),
            structures: IndexMap::new(),
            resources: IndexMap::new(),
            tribes: IndexMap::new(),
            _prediction: None,
            _end_of_turn_queue: Vec::new(),
            _hidden_resources: IndexMap::new(),
            _messages: Vec::new(),
            history: Vec::new(),
            initial_seed: 0,
        }
    }
}

impl GameState {
    /// Get the map size
    pub fn map_size(&self) -> i32 {
        self.settings.size
    }

    /// Get tile count
    pub fn tile_count(&self) -> i32 {
        self.settings.size * self.settings.size
    }

    /// Get the current player's tribe
    pub fn current_tribe(&self) -> Option<&TribeState> {
        self.tribes.get(&self.settings.current_player_turn_id)
    }

    /// Get the current player's tribe mutably
    pub fn current_tribe_mut(&mut self) -> Option<&mut TribeState> {
        self.tribes.get_mut(&self.settings.current_player_turn_id)
    }
}
