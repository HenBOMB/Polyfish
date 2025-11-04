#include <string>
#include <cstdint>
#ifndef READER_TYPES_H
#define READER_TYPES_H

struct TileInfo {
    uint32_t index;
    uint16_t tileId;
    uint16_t owner;
    std::string explorers;
    bool hasRoad;
    bool hasRoute;
    bool hadRoute;
    uint16_t capitalOf;
    uint16_t rulingCityX;
    uint16_t rulingCityY;
    uint16_t skinType;
    uint16_t climate;
    uint16_t tileX;
    uint16_t tileY;
};

struct StructureInfo {
    int16_t structureId;
    int16_t structureLevel;
    int16_t structureFounded;
    uint16_t structureBaseScore;
};

struct ResourceInfo {
    int16_t resourceId;
};

struct UnitInfo {
    uint16_t owner;
    uint16_t unitX;
    uint16_t unitY;
    uint16_t unitId;
    uint16_t unitHp;
    uint16_t promoted;
    uint16_t unitKills;
    uint16_t prevTileX;
    uint16_t prevTileY;
    uint16_t homeX;
    uint16_t homeY;
    uint16_t direction;
    bool flipped;
    uint16_t createdTurn;
    bool moved;
    bool attacked;
    uint16_t passengerId;
    std::string unitEffects;
};

struct CityInfo {
    std::string name;
    int16_t population;
    int16_t progress;
    std::string rewards;
    uint16_t production;
    uint16_t borderSize;
    bool connectedToCapital;
    int16_t level;
};

struct PlayerState {
    uint8_t id;
    std::string username;
    int32_t currency;
    int32_t score;
    bool autoplay;
    std::string tech;
    int16_t tribeType;
    uint8_t killerId;
    int32_t kills;
    std::string tasks; 
    std::string builtUniqueImprovements;
    std::string knownPlayers;
    std::string relations;
    int32_t killedTurn;
    int32_t resignedTurn;
    int16_t startingTileX;
    int16_t startingTileY;
};

// Navigation
constexpr size_t _0x_IN_LIST                     = 0x10;
constexpr size_t _0x_IN_LIST_COUNT               = 0x18;
constexpr size_t _0x_IN_LIST_START_SHIFT         = 0x20;
constexpr size_t _0x_IN_DICT_ENTRIES             = 0x18;
constexpr size_t _0x_IN_DICT_COUNT               = 0x20;
constexpr size_t _0x_IN_DICT_KEY_SHIFT           = 0x20;
constexpr size_t _0x_IN_DICT_VAL_SHIFT           = 0x30;
constexpr size_t _0x_GAMEMANAGER_SETTINGS        = 0x20; // GameManager -> GameSettings settings
constexpr size_t _0x_GAMEMANAGER_CLIENT          = 0x28; // GameManager -> ClientBase client
constexpr size_t _0x_GAMEMANAGER_AI_OPPONENTS    = 0x50; // GameManager -> Int32 aiOpponents
constexpr size_t _0x_GAMEMANAGER_HUMAN_OPPONENTS = 0x54; // GameManager -> Int32 playerOpponents
constexpr size_t _0x_GAMEMANAGER_STARTING_TRIBE  = 0x64; // GameManager -> TribeType startingTribe
constexpr size_t _0x_CLIENT_CUR_STATE            = 0x38; // ClientBase -> GameState currentState
constexpr size_t _0x_STATE_CUR_TURN              = 0x18; // GameState -> Int32 currentTurn
constexpr size_t _0x_STATE_PLAYERS               = 0x38; // GameState -> List<PlayerState> players
constexpr size_t _0x_STATE_MAP                   = 0x30; // GameState -> MapData map
constexpr size_t _0x_MAP_TILES                   = 0x18; // MapData -> List<TileData> tiles
constexpr size_t _0x_TRAILING_OFFSET             = 0x10;

// TileData
constexpr size_t _0x_TILE_X             = 0x10; // WorldCoordinates[0] coordinates
constexpr size_t _0x_TILE_Y             = 0x14; // WorldCoordinates[1] coordinates
constexpr size_t _0x_TILE_TERRAIN_TYPE  = 0x18; // TerrainData.Type terrain
constexpr size_t _0x_TILE_CLIMATE_TYPE  = 0x1C; // Int32 climate
constexpr size_t _0x_TILE_SKIN_TYPE     = 0x20; // SkinType _skin
constexpr size_t _0x_TILE_OWNER         = 0x34; // Byte owner
constexpr size_t _0x_TILE_CAPITAL_OF    = 0x35; // Byte capitalOf
constexpr size_t _0x_TILE_EXPLORERS     = 0x38; // List<Byte> explorers
constexpr size_t _0x_TILE_RULING_CITY_X = 0x48; // WorldCoordinates[0] rulingCityCoordinates
constexpr size_t _0x_TILE_RULING_CITY_Y = _0x_TILE_RULING_CITY_X + 0x4; // WorldCoordinates[1] rulingCityCoordinates
constexpr size_t _0x_TILE_IMPROVEMENT     = 0x50; // ImprovementState improvement
constexpr size_t _0x_TILE_RESOURCE      = 0x58; // ResourceState resource
constexpr size_t _0x_TILE_UNIT          = 0x60; // UnitState unit
constexpr size_t _0x_TILE_HAS_ROAD      = 0x68; // Bool hasRoad
constexpr size_t _0x_TILE_HAS_ROUTE     = 0x69; // Bool hasRoute
constexpr size_t _0x_TILE_HAD_ROUTE     = 0x78; // Bool hadRoute
constexpr size_t _0x_TILE_UPGRADE_TECH  = 0x80; // Dictionary<TechdData.Type,Single> upgradeTech
constexpr size_t _0x_TILE_LAST_POP      = 0x88; // Int64 lastPopulationCheck
constexpr size_t _0x_TILE_AVAILABLE_POP = 0x90; // Int32 availablePopulation

// ImprovementState
constexpr size_t _0x_IMPROVEMENT_TYPE        = 0x10; // ImprovementData.Type type
constexpr size_t _0x_IMPROVEMENT_OWNER       = 0x14; // Byte owner
constexpr size_t _0x_IMPROVEMENT_FOUNDER     = 0x15; // Byte founder
constexpr size_t _0x_IMPROVEMENT_LEVEL       = 0x16; // Int16 level
constexpr size_t _0x_IMPROVEMENT_FOUNDED     = 0x18; // Int16f founded
constexpr size_t _0x_IMPROVEMENT_XP          = 0x1A; // Int16 xp
constexpr size_t _0x_IMPROVEMENT_POPULATION  = 0x1C; // Int16 population
constexpr size_t _0x_IMPROVEMENT_PRODUCTION  = 0x1E; // UInt16 production
constexpr size_t _0x_IMPROVEMENT_BASE_SCORE  = 0x20; // UInt16 baseScore
constexpr size_t _0x_IMPROVEMENT_BORDER_SIZE = 0x22; // UInt16 borderSize
constexpr size_t _0x_IMPROVEMENT_UPGRADE     = 0x24; // UInt16 upgrade
constexpr size_t _0x_IMPROVEMENT_CONNECTED_TO_CAPITAL = 0x26; // Byte connectedToCapitalOfPlayer
constexpr size_t _0x_IMPROVEMENT_NAME        = 0x28; // String name
constexpr size_t _0x_IMPROVEMENT_REWARDS     = 0x30; // List<CityReward> rewards
constexpr size_t _0x_IMPROVEMENT_EFFECTS     = 0x38; // List<ImprovementEffect> effects

// UnitState
constexpr size_t _0x_UNIT_LEADER            = 0x14; // UInt32 leader
constexpr size_t _0x_UNIT_FOLLOWER          = 0x18; // UInt32 follower
constexpr size_t _0x_UNIT_OWNER             = 0x1C; // Byte owner
constexpr size_t _0x_UNIT_STYLE             = 0x1E; // Int16 style
constexpr size_t _0x_UNIT_SKIN_TYPE         = 0x20; // SkinType skinType
constexpr size_t _0x_UNIT_TYPE              = 0x24; // UnitData.Type type
constexpr size_t _0x_UNIT_PREV_TURN_END_X   = 0x28; // WorldCoordinates[0] previousTurnEndCoordinates
constexpr size_t _0x_UNIT_PREV_TURN_END_Y   = _0x_UNIT_PREV_TURN_END_X + 0x4; // WorldCoordinates[1] previousTurnEndCoordinates
constexpr size_t _0x_UNIT_X                 = 0x30; // WorldCoordinates[0] coordinates
constexpr size_t _0x_UNIT_Y                 = _0x_UNIT_X + 0x4; // WorldCoordinates[0] coordinates
constexpr size_t _0x_UNIT_HOME_X            = 0x38; // WorldCoordinates[0] home
constexpr size_t _0x_UNIT_HOME_Y            = _0x_UNIT_HOME_X + 0x4; // WorldCoordinates[0] home
constexpr size_t _0x_UNIT_PASSENGER_UNIT    = 0x40; // UnitState passengerUnit
constexpr size_t _0x_UNIT_HEALTH            = 0x48; // Uint16 health
constexpr size_t _0x_UNIT_PROMOTION_LEVEL   = 0x4A; // Uint16 promotionLevel
constexpr size_t _0x_UNIT_XP                = 0x4C; // UInt16 xp
constexpr size_t _0x_UNIT_MOVED             = 0x4E; // Bool moved
constexpr size_t _0x_UNIT_ATTACKED          = 0x4F; // Bool attacked
constexpr size_t _0x_UNIT_DIRECTION         = 0x50; // GridDirection direction
constexpr size_t _0x_UNIT_FLIPPED           = 0x54; // Bool flipped
constexpr size_t _0x_UNIT_CREATED_TURN      = 0x56; // UInt16 createdTurn
constexpr size_t _0x_UNIT_UNITDATA          = 0x58; // UnitData unitData
constexpr size_t _0x_UNIT_EFFECTS           = 0x60; // List<UnitEffect> effects

enum PlayerStateOffsets: size_t {
    OWNER                     = 0x10, // Byte id
    USERNAME                  = 0x18, // String username
    ACCOUNT_ID                = 0x20, // Guid accountId
    AUTOPLAY                  = 0x34, // Bool autoplay
    START_TILE_X              = 0x38, // WorldCoordinates[0] startTile
    START_TILE_Y              = START_TILE_X + 0x4, // WorldCoordinates[1] startTile
    TRIBE_TYPE                = 0x40, // TribeType tribe
    TRIBE_MIX                 = 0x44, // TribeType tribeMix
    RESIGNED_TURN             = 0x50, // Int32 resignedTurn
    RESIGNED_AT_COMMAND_INDEX = 0x50, // Int32 resignedAtCommandIndex
    WIPED_AT_COMMAND_INDEX    = 0x58, // Int32 wipedAtCommandIndex
    AVAILABLE_TECH            = 0x60, // List<TechData.Type> availableTech
    TASKS                     = 0x68, // List<TaskBase> tasks
    PROGRESS                  = 0x70, // Dictionary<Byte,Int32> progressions
    KNOWN_PLAYERS             = 0x78, // List<Byte> knownPlayers
    BUILT_UNIQUE_IMPROVEMENTS = 0x80, // List<ImprovementData> builtUniqueImprovements
    RELATIONS                 = 0x88, // Dictionary<Byte,DiplomacyRelation> relations
    MESSAGES                  = 0x90, // List<Diplomacymessage> messages
    SKIN_TYPE                 = 0x98, // SkinType skinType
    CURRENCY                  = 0x9C, // Int32 currency
    SCORE                     = 0xA0, // Int32 score
    END_SCORE                 = 0xA4, // Int32 endScore
    CITIES                    = 0xA8, // Int32 cities
    KILLS                     = 0xAC, // Int32 kills
    CASUALTIES                = 0xB0, // Int32 casualties
    WIPEOUTS                  = 0xB4, // Int32 wipeOuts
    KILLER_ID                 = 0xB8, // Byte killerId
    KILLED_TURN               = 0xBC, // Int32 killedTurn
    COLOR                     = 0xC0, // Int32 color
    AI_STATE                  = 0xC8, // AIState aiState
    BUFF_SIZE                 = AI_STATE + _0x_TRAILING_OFFSET
};

#endif
