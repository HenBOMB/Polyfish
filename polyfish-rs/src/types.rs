// WARNING: DO NOT EDIT ANY OF THESE IDS EVER
// EVERY SINGLE ID HAS BEEN TRACKED AND ANALYZED TO MATCH EXACTLY THE REAL VALUES IN THE REAL STEAM GAME ENGINE
// IF YOU CHANGE ANY ID, YOU WILL BREAK THE GAME
// WARNING: DO NOT EDIT ANY OF THESE IDS EVER
// IDS FROM 120 ARE MISSING AND DO NOT MATCH THE REAL IDS AND MUST BE MANUALLY EDITIED BY A HUMAN WITH ACCESS TO THE REAL GAME ENGINE ON STEAM

use serde_repr::{Deserialize_repr, Serialize_repr};

/// Game mode types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize_repr, Deserialize_repr)]
#[repr(u8)]
pub enum ModeType {
    None = 0,
    Perfection = 1,
    Domination = 2,
    Glory = 3,
    Might = 4,
    Custom = 5,
    Sandbox = 6,
    Tutorial = 7,
}

/// Terrain types
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Serialize_repr,
    Deserialize_repr,
    Default,
    strum_macros::EnumIter,
)]
#[repr(u8)]
pub enum TerrainType {
    #[default]
    None = 0,
    Water = 1,
    Ocean = 2,
    Field = 3,
    Mountain = 4,
    Forest = 5,
    Ice = 6,
    Wetland = 7,
    Mangrove = 8,
}

/// Technology types
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Serialize_repr,
    Deserialize_repr,
    Default,
    strum_macros::EnumIter,
    strum_macros::FromRepr,
)]
#[repr(i8)]
pub enum TechnologyType {
    BeyondComprehension = -1,
    #[default]
    Basic = 0,
    Riding = 1,
    FreeSpirit = 2,
    Chivalry = 3,
    Roads = 4,
    Trade = 5,
    Organization = 6,
    Strategy = 7,
    Farming = 8,
    Construction = 9,
    Fishing = 10,
    // Whaling = 11, // removed from the game
    Aquatism = 12,
    Sailing = 13,
    Navigation = 14,
    Hunting = 15,
    Forestry = 16,
    Mathematics = 17,
    Archery = 18,
    Spiritualism = 19,
    Climbing = 20,
    Meditation = 21,
    // Update "Philosophy" -> "Mysticism". not nescesary
    Philosophy = 22,
    Mining = 23,
    Smithery = 24,
    // Aquarion
    FreeDiving = 25,
    Spearing = 26,
    // Amphibian = 27, // Riding2
    // Elyrion
    ForestMagic = 28,
    // WaterMagic = 29,
    // Polaris
    Frostwork = 30,
    PolarWarfare = 31,
    Polarism = 32,
    // Oceanology = 33,
    ShockTactics = 35,
    // Cymanti
    Recycling = 36,
    // Hydrology = 37,
    Diplomacy = 38,
    // Aquaculture = 39,
    // FishingSimple = 40,
    Sledding = 41,
    IceFishing = 42,
    Pascetism = 43,
    Ramming = 44,
    // AquaBase = 45,
    // MarineLife = 46,
    // AquaFishing = 47,
    Waterways = 48,
    Oceantology = 49,
    // Mysticism = 50
    Synergy = 120, // idk, new tech by cymanti
    Rituals = 121, // Cymanti-specific (replaces Meditation)
}

impl From<i32> for TechnologyType {
    fn from(v: i32) -> Self {
        Self::from_repr(v as i8).unwrap_or(TechnologyType::Basic)
    }
}

/// Tribe types
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Serialize_repr,
    Deserialize_repr,
    Default,
    strum_macros::EnumIter,
    strum_macros::FromRepr,
)]
#[repr(i8)]
pub enum TribeType {
    #[default]
    None = 0,
    Nature = 1,
    AiMo = 2,
    Aquarion = 3,
    Bardur = 4,
    Elyrion = 5,
    Hoodrick = 6,
    Imperius = 7,
    Kickoo = 8,
    Luxidoor = 9,
    Oumaji = 10,
    Quetzali = 11,
    Vengir = 12,
    XinXi = 13,
    Yadakk = 14,
    Zebasi = 15,
    Polaris = 16,
    Cymanti = 17,
}

impl From<i32> for TribeType {
    fn from(v: i32) -> Self {
        Self::from_repr(v as i8).unwrap_or(TribeType::None)
    }
}

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Serialize_repr,
    Deserialize_repr,
    Default,
    strum_macros::EnumIter,
    strum_macros::FromRepr,
)]
#[repr(i8)]
pub enum CityRewardType {
    #[default]
    None = 0,
    CityWall = 1,
    Park = 2,
    Workshop = 3,
    Explorer = 4,
    BorderGrowth = 5,
    SuperUnit = 6,
    Resources = 7,
    PopGrowth = 8,
}

impl From<i32> for CityRewardType {
    fn from(v: i32) -> Self {
        Self::from_repr(v as i8).unwrap_or(CityRewardType::None)
    }
}

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Serialize_repr,
    Deserialize_repr,
    Default,
    strum_macros::EnumIter,
    strum_macros::FromRepr,
)]
#[repr(i8)]
pub enum RuinsRewardType {
    #[default]
    None = 0,
    Resources = 1,
    PopGrowth = 2,
    Explorer = 4,
    FreeTech = 5,
    // SuperUnit = 6, // no longer used
    // Battleship = 7, // no longer used
    // Seamonster = 8, // no longer used
    Swordsman = 9,
    City = 10,
}

impl From<i32> for RuinsRewardType {
    fn from(v: i32) -> Self {
        Self::from_repr(v as i8).unwrap_or(RuinsRewardType::None)
    }
}

/// Unit types
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Serialize_repr,
    Deserialize_repr,
    Default,
    strum_macros::EnumIter,
    strum_macros::FromRepr,
)]
#[repr(i8)]
pub enum UnitType {
    #[default]
    None = 0,
    // Scout = 1,
    Warrior = 2,
    Rider = 3,
    Knight = 4,
    Defender = 5,
    // Ship = 6,
    // Battleship = 7,
    Catapult = 8,
    Archer = 9,
    MindBender = 10,
    Swordsman = 11,
    Giant = 12,
    // Bunny = 13,
    // Boat = 14,
    Polytaur = 15,
    // Navalon,
    DragonEgg = 17,
    BabyDragon = 18,
    FireDragon = 19,
    Amphibian = 20,
    Tridention = 21,
    Mooni = 22,
    BattleSled = 23,
    IceFortress = 24,
    IceArcher = 25,
    Crab = 26,
    Gaami = 27,
    Hexapod = 28,
    Doomux = 29,
    Phychi = 30,
    Kiton = 31,
    Exida = 32,
    Centipede = 33,
    Segment = 34,
    Raychi = 35,
    Shaman = 36,
    Dagger = 37,
    Cloak = 38,
    CloakBoat = 39, // Dinghy
    Pirate = 40,
    Bomber = 41,
    Scoutship = 42,
    Transportship = 43, // Raft
    Rammership = 44,
    Juggernaut = 45,
    // MermaidWarrior,
    // MermaidArcher,
    // MermaidSwordsman,
    // MermaidDefender,
    // MermaidCloak,
    // MermaidDagger,
    // Jelly,
    // Shark,
    // Siren,
    // Aquapult,
    Boomchi = 56,
    LivingIsland = 57,
    // Ciru = 58,
    Mantis = 59,
    InsectEgg = 60, // BugEgg
    Moth = 61,
    Larva = 62,
}

impl From<i32> for UnitType {
    fn from(v: i32) -> Self {
        Self::from_repr(v as i8).unwrap_or(UnitType::None)
    }
}

/// Skill types
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Serialize_repr,
    Deserialize_repr,
    Default,
    strum_macros::EnumIter,
)]
#[repr(i8)]
pub enum SkillType {
    #[default]
    None = 0,
    /// Allows a unit to attack or break ice after moving in the same turn.
    Dash = 1,
    /// Allows a unit to move after attacking in the same turn.
    Escape = 2,
    /// Allows a unit to explore a 5x5 area instead of a 3x3 area.
    Scout = 3,
    /// Allows a unit to ignore movement barriers (ie. zone of control) imposed by enemy units. The units themselves still block movement (ie. units with sneak still cannot travel through enemy units).
    Sneak = 4,
    /// Allows a unit to hide itself and become invisible to enemies when it moves.
    Hide = 5,
    // Unknown, not used anywhere
    // Build = 6,
    /// Allows a unit to attack again immediately after killing an enemy unit.
    Persist = 7,
    /// Allows a unit to convert an enemy unit into a friendly unit by attacking it.
    Convert = 8,
    /// Gives a unit the Heal Others unit action, which heals all adjacent friendly units by up to 4 HP.
    HealOthers = 9,
    /// Allows a unit to float on water
    Float = 10, // Swim
    /// Allows a unit to carry another unit inside. A unit with the carry skill can move to a land tile adjacent to water. Doing so releases the unit it was carrying and ends the unit's turn.
    Carry = 11,
    /// Allows a unit to grow into a different unit after a given number of turns.
    Grow = 12,
    /// Not affected by any terrain
    Fly = 13,
    /// Allows a unit to damage or poison enemy units adjacent to the targeted unit.
    Splash = 14,
    // Unknown, not used anywhere
    // Decay = 15,
    /// TODO REMOVE FROM THE GAME AND FIND PROPER REPLACEMENT
    /// Removed: Allowed a unit to move in water or ocean even if no required technology is researched but prevents the unit from moving onto land, except for those with cities and villages.
    Navigate = 16,
    // Unknown, not used anywhere
    // Crush = 17,
    /// Allows a unit to freeze enemy units it attacks.
    Freeze = 18,
    /// Gives a unit the Freeze Area unit action, which freezes adjacent enemy units, freezes adjacent water tiles into ice tiles, and converts adjacent land tiles to the style of the tribe the unit belongs to.
    FreezeArea = 19,
    /// Allows a unit to automatically freeze adjacent enemy units and water tiles (turning them into ice tiles) as it moves.
    AutoFreeze = 20,
    // Doubles movement on ice but limits movement to one and prohibits the use of the dash and escape skills on land.
    Skate = 21,
    /// Allows a unit to receive a defence bonus in a city.
    Fortify = 22,
    /// Allows a unit to ignore movement barriers imposed by terrain (e.g. Forests), but NOT mountains. Note that zone of control cannot be negated.
    Creep = 23,
    // Not sure..
    // Boost = 24,
    /// Units with this skill do not take up a population slot in or belong to any city.
    Independent = 25,
    /// Allows a unit to poison enemy units it attacks.
    Poison = 26,
    /// Allows a unit to grow segments that move in tandem with the unit after killing units via attack (retaliation kills do not produce segments). A unit with segments attached is restricted to 1 movement and cannot enter ports (except those with algae; This is most likely due to a bug) until the segment attached to the unit is removed.
    Eat = 27,
    // Unknown, not used anywhere
    // Unique = 29,
    /// Gives a unit the Explode unit action, which damages using the unit's attack value and poisons all adjacent enemy units, kills the unit itself, and leaves in its place spores (on land) or Algae (on water).
    Explode = 30,
    /// Prevents a unit from triggering retaliation attacks when attacking an enemy unit.
    Surprise = 31,
    // Unknown, not used anywhere
    // Agent = 32,
    /// Allows a unit to incite a revolt and spawn Daggers by entering an enemy city.
    Infiltrate = 35,
    // Unknown, not used anywhere
    // Detect = 36,
    // Unknown, not used anywhere
    // Intercept = 37,
    /// Prevents a unit from being able to do retaliation attacks when attacked by an enemy unit.
    Stiff = 38,
    // Unknown, not used anywhere
    // Protect = 39,
    /// Causes a unit to deal damage to all adjacent enemy units when it moves.
    Stomp = 40,
    // Unknown, not used anywhere
    // AutoHeal = 41,
    /// Prevents a unit from becoming a veteran
    Static = 42,
    // Allows a unit to damage any enemy that moves next to it, as well as when it moves next to an enemy or when it is trained.
    // Tentacles = 44,
    /// Allows a unit to automatically flood any tile it moves onto.
    AutoFlood = 45,
    // Allows a unit to flood any tile it attacks. Units with this skill can also attack tiles without an enemy unit present.
    // Drench = 46,
    /// Allows a unit to move on both land and water tiles but slows movement on land.
    Amphibious = 47,
    /// Restricts a unit to only being able to move on water tiles.
    Water = 48,
    // Land = 49,
    /// Allows a unit to make algae at any tiles it travels.
    Algae = 51,
    // Consumed = 52,
    // Gives a unit the Swarm unit action, which boosts all adjacent friendly units by increasing their movement by 1 until being attacked.
    Swarm = 53,
    /// Allows a unit to attack twice in the same turn. After the first attack is performed, the unit can also still perform certain actions (such as draining or breaking ice, but not capturing cities or excavating ruins)
    DoubleAttack = 54,
}

/// Ability types
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Serialize_repr,
    Deserialize_repr,
    Default,
    strum_macros::EnumIter,
)]
#[repr(i8)]
pub enum AbilityType {
    #[default]
    None = 0,
    BurnForest = 1,
    ClearForest = 2,
    GrowForest = 3,
    Destroy = 4,
    Decompose = 5,
    Convert = 6,
    Recover = 7,
    Disband = 8,
    HealOthers = 9,
    Drain = 11,
    FreezeArea = 13,
    Boost = 14,
    Explode = 15,
    Promote = 16,
    BreakIce = 17,
    BreakPeace = 18,
    PeaceRequestResponse = 19,
    EstablishEmbassy = 20,
    PeaceTreaty = 21,
    DestroyEmbassy = 22,
    EnchantAnimal = 23,
}

/// Structure types
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Serialize_repr,
    Deserialize_repr,
    Default,
    strum_macros::EnumIter,
)]
#[repr(i8)]
pub enum StructureType {
    #[default]
    None = 0,
    Village = 1,
    Ruin = 2,
    Road = 3,
    // CustomsHouse = 4, // replaced with market
    Farm = 5,
    Windmill = 6,
    // Fishing = 7, // no idea what this is
    Port = 8,
    // Hunting = 9, // maybe these are harvests?
    // ClearForest = 10, // same
    // BurnForest = 11, // same
    LumberHut = 12,
    Sawmill = 13,
    // GrowForest = 14, // okay yeah these are abilities, but all chunked in to the same "improvement" data
    // HarvestFruit = 15,
    // WhaleHunting = 16,
    Temple = 17,
    ForestTemple = 18,
    WaterTemple = 19,
    MountainTemple = 20,
    Mine = 21,
    Forge = 22,
    AltarOfPeace = 23,
    TowerOfWisdom = 24,
    GrandBazaar = 25,
    EmperorsTomb = 26,
    GateOfPower = 27,
    ParkOfFortune = 28,
    EyeOfGod = 29,
    // EnchantAnimal = 30, // same here, more abilities
    // EnchantWhale = 31,
    Sanctuary = 32, // TODO add new building (elyrion)
    Outpost = 33,   // TODO new
    IceBank = 34,   // TODO new
    IceTemple = 35, // TODO new
    // PolarisClimate = 36, // no idea
    Fungi = 37, // TODO no idea how its used, not really a structure?
    Algae = 38,
    Mycelium = 39,
    // BurnSpores = 40, // no idea
    Clathrus = 41, // New Cymanti building
    // HiddenSanctuary = 42, // no idea
    // HarvestSpores = 43, // ability to harvest spores by cymanti
    // NullBuilding = 44, // no idea
    // Cultivate = 45, // even less idea
    // StarFishing = 56, // old ability?
    Lighthouse = 47,
    Bridge = 48,
    // Aquafarm = 49, // idk what this is
    Market = 50,
    // Atoll = 51, // idk what this is
    // Canal = 52, // idk what this is
    // Fertilize = 53, // idk what this is
    // LandFill = 54, // idk what this is
    // AlgaeSpawn = 55, // idk what this is
    // and that's all matching the original polytopia src code
    Embassy = 70,

    // This is wrong and prolly uses some other monument with adaptive skin
    ChurchOfConverts = 123, // Cymanti monument for Converter task
}

/// Resource types
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Serialize_repr,
    Deserialize_repr,
    Default,
    strum_macros::EnumIter,
)]
#[repr(i8)]
pub enum ResourceType {
    #[default]
    None = 0,
    Game = 1,
    Crop = 2,
    Fish = 3,
    // Whale = 4, // Removed from the game
    Metal = 5,
    Fruit = 6,
    Spores = 7,
    Starfish = 8,
    AquaCrop = 9,
}

/// Unit effect types
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Serialize_repr,
    Deserialize_repr,
    Default,
    strum_macros::EnumIter,
)]
#[repr(i8)]
pub enum UnitEffect {
    #[default]
    Frozen = 0,
    Poison = 1,
    Boosted = 2,
    Invisible = 3,
    Bubble = 4,
    Petrified = 5,
    Swift = 6,
    DoubleReady = 7,
    Charmed = 8,
}

/// Tile effect types
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Serialize_repr,
    Deserialize_repr,
    Default,
    strum_macros::EnumIter,
)]
#[repr(i8)]
pub enum TileEffect {
    #[default]
    None = 0,
    Flooded = 1,
    Swamped = 2,
    Tentacle = 3,
    Algae = 4,
    Foam = 5,
}

/// Capture types
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Serialize_repr,
    Deserialize_repr,
    Default,
    strum_macros::FromRepr,
)]
#[repr(i8)]
pub enum CaptureType {
    #[default]
    None = 0,
    Ruins = 1,
    Starfish = 2,
    Village = 3,
    City = 4,
}

impl From<i32> for CaptureType {
    fn from(v: i32) -> Self {
        Self::from_repr(v as i8).unwrap_or(CaptureType::None)
    }
}

/// Move types
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Serialize_repr,
    Deserialize_repr,
    Default,
    strum_macros::FromRepr,
)]
#[repr(i8)]
pub enum MoveType {
    #[default]
    None = 0,
    Step = 1,
    Attack = 2,
    Ability = 3,
    Summon = 4,
    Harvest = 5,
    Build = 6,
    Research = 7,
    Capture = 8,
    Reward = 9,
    EndTurn = 10,
    Resign = 11,
}

impl From<i32> for MoveType {
    fn from(v: i32) -> Self {
        Self::from_repr(v as i8).unwrap_or(MoveType::None)
    }
}

/// Task types
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Serialize_repr,
    Deserialize_repr,
    Default,
    strum_macros::EnumIter,
)]
#[repr(i8)]
pub enum TaskType {
    #[default]
    Pacifist = 0,
    Genius = 1,
    Wealth = 2,
    Explorer = 3,
    Killer = 4,
    Network = 5,
    Metropolis = 6,
    Converter = 7, // Cymanti-specific (replaces Pacifist). id is currently unknown but it may be that the pattern continues in this case
}

/// Map sizes
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Serialize_repr,
    Deserialize_repr,
    Default,
    strum_macros::EnumIter,
)]
#[repr(i8)]
pub enum MapSize {
    Tiny = 0,  // 11x11, 121
    Small = 1, // 14x14, 196
    #[default]
    Normal = 2, // 16x16, 256
    Large = 3, // 18x18, 324
    Huge = 4,  // 20x20, 400
    Massive = 5, // 30x30, 900
}

impl MapSize {
    pub fn get_size(&self) -> i32 {
        match self {
            MapSize::Tiny => 11,
            MapSize::Small => 14,
            MapSize::Normal => 16,
            MapSize::Large => 18,
            MapSize::Huge => 20,
            MapSize::Massive => 30,
        }
    }
}

/// Map types (Wetness levels)
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Serialize_repr,
    Deserialize_repr,
    Default,
    strum_macros::EnumIter,
)]
#[repr(i8)]
pub enum MapType {
    #[default]
    None = 0,
    Drylands = 1,
    Lakes = 2,
    Continents = 3,
    Archipelago = 4,
    WaterWorld = 5,
    Pangea = 6,
}
