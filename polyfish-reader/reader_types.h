#include <string>
#include <cstdint>
#ifndef READER_TYPES_H
#define READER_TYPES_H
#include <vector>
#include <unordered_map>

struct TileInfo {
    int32_t index;
    int32_t type;
    uint8_t owner;
    std::vector<uint8_t> explorers;
    bool hasRoad;
    bool hasRoute;
    bool hadRoute;
    uint8_t capitalOf;
    int32_t rulingCityX;
    int32_t rulingCityY;
    int32_t skinType;
    int32_t climate;
    int32_t tileX;
    int32_t tileY;
};

struct StructureInfo {
    int16_t type;
    int16_t level;
    int16_t founded;
    uint16_t score;
};



struct CityInfo {
    std::string name;
    int16_t population;
    int16_t progress;
    std::vector<int32_t> rewards;
    uint16_t production;
    uint16_t borderSize;
    bool connectedToCapital;
    int16_t level;
};


// Navigation
constexpr size_t _0x_IN_LIST                     = 0x10;
constexpr size_t _0x_IN_LIST_COUNT               = 0x18;
constexpr size_t _0x_IN_LIST_START_SHIFT         = 0x20;
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
constexpr size_t _0x_TILE_X                 = 0x10; // WorldCoordinates[0] coordinates
constexpr size_t _0x_TILE_Y                 = 0x14; // WorldCoordinates[1] coordinates
constexpr size_t _0x_TILE_TERRAIN_TYPE      = 0x18; // TerrainData.Type terrain
constexpr size_t _0x_TILE_CLIMATE_TYPE      = 0x1C; // Int32 climate
constexpr size_t _0x_TILE_SKIN_TYPE         = 0x20; // SkinType _skin
constexpr size_t _0x_TILE_OWNER             = 0x34; // Byte owner
constexpr size_t _0x_TILE_CAPITAL_OF        = 0x35; // Byte capitalOf
constexpr size_t _0x_TILE_EXPLORERS         = 0x38; // List<Byte> explorers
constexpr size_t _0x_TILE_RULING_CITY_X     = 0x48; // WorldCoordinates[0] rulingCityCoordinates
constexpr size_t _0x_TILE_RULING_CITY_Y     = _0x_TILE_RULING_CITY_X + 0x4; // WorldCoordinates[1] rulingCityCoordinates
constexpr size_t _0x_TILE_IMPROVEMENT         = 0x50; // ImprovementState improvement
constexpr size_t _0x_TILE_RESOURCE          = 0x58; // ResourceState resource
constexpr size_t _0x_TILE_UNIT              = 0x60; // UnitState unit
constexpr size_t _0x_TILE_HAS_ROAD          = 0x68; // Bool hasRoad
constexpr size_t _0x_TILE_HAS_ROUTE         = 0x69; // Bool hasRoute
constexpr size_t _0x_TILE_HAD_ROUTE         = 0x78; // Bool hadRoute
constexpr size_t _0x_TILE_UPGRADE_TECH      = 0x80; // Dictionary<TechdData.Type,Single> upgradeTech
constexpr size_t _0x_TILE_LAST_POP          = 0x88; // Int64 lastPopulationCheck
constexpr size_t _0x_TILE_AVAILABLE_POP     = 0x90; // Int32 availablePopulation
    
enum Offsets: size_t {
    GameSettings_BotDifficulty              = 0x10, // BotDifficulty difficulty
    GameSettings_BaseGameMode               = 0x14, // BaseGameMode baseGameMode
    GameSettings_RulesGameMode              = 0x18, // BaseGameMode rulesGameMode
    GameSettings_SelectedSkins              = 0x30, // List<TribeType> selectedSkins
    GameSettings_Players                    = 0x38, // List<PlayerData> players
    GameSettings_Spectators                 = 0x40, // List<PlayerData> spectators
    GameSettings_MapPreset                  = 0x48, // MapPreset mapPreset
    GameSettings_Rules                      = 0x50, // GameRules rules
    GameSettings_GameName                   = 0x58, //  String gameName
    GameSettings_GameType                   = 0x60, // GameType gameType
    GameSettings_MapSize                    = 0x64, // Int32 mapSize
    GameSettings_TimeLimit                  = 0x68, // Int32 timeLimit
    GameSettings_Size_                      = GameSettings_TimeLimit + _0x_TRAILING_OFFSET,

    
    GameState_Version                       = 0x10, // Int32 version
    GameState_Seed                          = 0x14, // Int32 seed
    GameState_CurrentTurn                   = 0x18, // UInt32 currentTurn
    GameState_CurrentPlayerIndex            = 0x1C, // Byte currentPlayerIndex
    GameState_CurrentUnitID                 = 0x20, // UInt32 currentUnitId
    GameState_CurrentState                  = 0x24, // GameState currentState
    GameState_Settings                      = 0x28, // GameSettings settings
    GameState_Map                           = 0x30, // MapData map
    GameState_PlayerStates                  = 0x38, // List<PlayerState> playerStates
    GameState_Size_                         = GameState_PlayerStates + _0x_TRAILING_OFFSET,


    GameManager_Settings                    = 0x20, // GameManager -> GameSettings settings
    GameManager_ClientBase                  = 0x28, // GameManager -> ClientBase client
    GameManager_AIOpponents                 = 0x50, // GameManager -> Int32 aiOpponents
    GameManager_HumanOpponents              = 0x54, // GameManager -> Int32 playerOpponents
    GameManager_StartingTribe               = 0x64, // GameManager -> TribeType startingTribe

    
    Create_Terrain                          = 0x10, // TerrainData terrain
    Create_Resource                         = 0x18, // ResourceData resource
    Create_Unit                             = 0x20, // UnitData unit
    Create_Effect                           = 0x28, // TileData.EffectType effect
    Create_Size_                            = Create_Effect + _0x_TRAILING_OFFSET,


    ImprovementData_IDX                     = 0x10, // Int32 idx
    ImprovementData_Hidden                  = 0x14, // Bool hidden
    ImprovementData_Cost                    = 0x18, // Int32 cost
    ImprovementData_Work                    = 0x1C, // Int32 work
    ImprovementData_BorderSize              = 0x20, // Int32 borderSize
    ImprovementData_MaxLevel                = 0x24, // Int32 maxLevel
    ImprovementData_IMPROVEMENT_ABILITIES   = 0x28, // List<ImprovementAbility.Type> improvementAbilities
    ImprovementData_CREATES                 = 0x30, // List<Creates> creates
    ImprovementData_REWARDS                 = 0x38, // List<Rewards> rewards
    ImprovementData_TERRAIN_REQUIREMENTS    = 0x40, // List<TerrainRequirements> terrainRequirements
    ImprovementData_ADJACENCY_REQUIREMENTS  = 0x48, // List<AdjacencyRequirements> adjacencyRequirements
    ImprovementData_ADJACENCY_IMPROVEMENTS  = 0x50, // List<AdjacencyImprovements> adjacencyImprovemnts
    ImprovementData_Routes                  = 0x58, // List<TerrainData> routes
    ImprovementData_Range                   = 0x60, // Int32 range
    ImprovementData_Grouth_Rate             = 0x64, // Int32 growthRate
    ImprovementData_Grouth_Rewards          = 0x68, // List<GrowthRewards> growthRewards
    ImprovementData_Size_                   = ImprovementData_Grouth_Rewards + _0x_TRAILING_OFFSET,


    GameRules_TurnLimit                     = 0x10, // Int32 turnLimit
    GameRules_ScoreLimit                    = 0x14, // Int32 scoreLimit
    GameRules_WinByCapital                  = 0x18, // Bool winByCapital
    GameRules_WinByExtermination            = 0x19, // Bool winByExtermination
    GameRules_AllowMirrorPick               = 0x1a, // Bool allowMirrorPick
    GameRules_AllowSpecialTribe             = 0x1b, // Bool allowSpecialTribe
    GameRules_AllowTechSharing              = 0x1c, // Bool allowTechSharing
    GameRules_PlayerDeathCondition          = 0x20, // GameRules.DeathCondition playerDeathCondition
    GameRules_Size_                         = GameRules_PlayerDeathCondition + _0x_TRAILING_OFFSET,


    ImprovementState_Type                   = 0x10, // ImprovementData.Type type
    ImprovementState_Owner                  = 0x14, // Byte owner
    ImprovementState_Founder                = 0x15, // Byte founder
    ImprovementState_Level                  = 0x16, // Int16 level
    ImprovementState_Founded                = 0x18, // Int16f founded
    ImprovementState_XP                     = 0x1A, // Int16 xp
    ImprovementState_Population             = 0x1C, // Int16 population
    ImprovementState_Production             = 0x1E, // UInt16 production
    ImprovementState_BaseScore              = 0x20, // UInt16 baseScore
    ImprovementState_BorderSize             = 0x22, // UInt16 borderSize
    ImprovementState_Upgrade                = 0x24, // UInt16 upgrade
    ImprovementState_ConnectedToCapital     = 0x26, // Byte connectedToCapitalOfPlayer
    ImprovementState_Name                   = 0x28, // String name
    ImprovementState_Rewards                = 0x30, // List<CityReward> rewards
    ImprovementState_Effects                = 0x38, // List<ImprovementEffect> effects
    ImprovementState_Size_                  = ImprovementState_Effects + _0x_TRAILING_OFFSET,


    DiplomacyRelation_State                 = 0x10, // DiplomacyRelationState -> [0x10] -> Int32 state
    DiplomacyRelation_LastAttackTurn        = 0x14, // Int32 lastAttackTurn
    DiplomacyRelation_EmbassyLevel          = 0x18, // Int32 embassyLevel
    DiplomacyRelation_LastPeaceBrokenTurn   = 0x18, // Int32 lastPeaceBrokenTurn
    DiplomacyRelation_FirstMeet             = 0x18, // Int32 firstMeet
    DiplomacyRelation_EmbassyBuildTurn      = 0x24, // Int32 embassyBuildTurn
    DiplomacyRelation_PreviousAttackTurn    = 0x28, // Int32 previousAttackTurn
    DiplomacyRelation_Size_                 = DiplomacyRelation_PreviousAttackTurn + _0x_TRAILING_OFFSET,


    ClientBase_ClientType                   = 0x10, // Clienttype clientType
    ClientBase_ClientActionManager          = 0x18, // ActionManager clientActionManager
    ClientBase_GameID                       = 0x20, //
    ClientBase_InitialGameState             = 0x30, // GameState initialGameState
    ClientBase_CurrentGameState             = 0x38, // GameState currentGameState
};

namespace Data {
    struct DiplomacyRelation {
        int32_t state;
        int32_t lastAttackTurn;
        int32_t embassyLevel;
        int32_t lastPeaceBrokenTurn;
        int32_t firstMeet;
        int32_t embassyBuildTurn;
        int32_t previousAttackTurn;
    };

    struct PlayerState {
        uint8_t id;
        std::string username;
        int32_t currency;
        int32_t score;
        bool autoplay;
        std::vector<int32_t> tech;
        int16_t tribeType;
        uint8_t killerId;
        int32_t kills;
        std::string tasks; 
        std::vector<int32_t> builtUniqueImprovements;
        std::vector<uint8_t> knownPlayers;
        std::unordered_map<uint16_t, Data::DiplomacyRelation> relations;
        int32_t killedTurn;
        int32_t resignedTurn;
        int16_t startingTileX;
        int16_t startingTileY;
        int32_t casualties;
    };

    struct ResourceState {
        int16_t type;
    };

    struct TerrainData {

    };

    struct ResourceData {

    };

    struct UnitData {
        uint16_t owner;
        uint16_t unitX;
        uint16_t unitY;
        uint16_t type;
        uint16_t health;
        uint16_t promoted;
        uint16_t xp;
        uint16_t prevTileX;
        uint16_t prevTileY;
        uint16_t homeX;
        uint16_t homeY;
        uint16_t direction;
        bool flipped;
        uint16_t createdTurn;
        bool moved;
        bool attacked;
        uint16_t passengerType;
        std::vector<int32_t> effects;
    };

    struct Creates {
        TerrainData terrain;
        ResourceData resource;
        UnitData unit;
        int32_t effect;
    };

    struct ImprovementData {
        int32_t idx;
        bool hidden;
        int32_t cost;
        int32_t work;
        int32_t borderSize;
        int32_t maxLevel;
        std::vector<int32_t> improvementAbilities;
        // std::vector<Creates> creates;
        // std::vector<Rewards> rewards;
        // std::vector<TerrainRequirements> terrainRequirements;
        // std::vector<AdjacencyRequirements> adjacencyRequirements;
        // std::vector<AdjacencyImprovements> adjacencyImprovemnts;
        // std::vector<TerrainData> routes;
        int32_t range;
        int32_t growthRate;
        // std::vector<GrowthRewards> growthRewards;
    };
}

// UnitState
constexpr size_t _0x_UNIT_IDX               = 0x10; // UInt32 leader
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

enum DiplomacyRelationOffsets {
    STATE                  = 0x10, // Int32
    LAST_ATTACK_TURN       = 0x14, // Int32
    EMBASSY_LEVEL          = 0x18, // Int32
    LAST_PEACE_BROKEN_TURN = 0x1C, // Int32
    FIRST_MEET             = 0x20, // Int32
    EMBASSY_BUILD_TURN     = 0x24, // Int32
    PREVIOUS_ATTACK_TURN   = 0x28, // Int32
    SIZE__                 = PREVIOUS_ATTACK_TURN + _0x_TRAILING_OFFSET
};

enum PlayerStateOffsets {
    ID                        = 0x10, // Byte id
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
    SIZE_                     = AI_STATE + _0x_TRAILING_OFFSET
};

#endif
