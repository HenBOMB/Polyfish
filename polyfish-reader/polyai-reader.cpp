
#include <sys/uio.h>
#include <sys/ptrace.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <random>
#include <chrono>
#include <cstring>
#include <errno.h>
#include <sys/prctl.h>
#include <sys/auxv.h>
#include "reader_util.h"
#include <random>
#include "json.hpp"
#include <iomanip>
#include <codecvt>
#include <locale>

#define NAME "polyai-reader"

// Anti-debugging macro
#define ANTI_DEBUG() if (ptrace(PTRACE_TRACEME, 0, nullptr, nullptr) == -1) { \
    std::cerr << "Debugger detected, exiting." << std::endl; exit(1); }

using json = nlohmann::json;

extern char** environ;

template<typename... Args>
void appendFields(std::ostringstream& out, char delim, Args&&... args) {
    std::ostringstream temp;
    size_t count = sizeof...(args);
    size_t i = 0;
    ((temp << args << (++i < count ? "," : "")), ...);
    out << temp.str() << delim;
}

void clear_sensitive_env() {
    for (char** env = environ; *env; ++env) {
        if (strstr(*env, "version")) {
            *env = strdup("LANG=en_US.UTF-8");
        }
    }
}

std::string getTargetModule() {
    unsigned char enc[] = {0x12, 0x34, 0x38, 0x30, 0x14, 0x26, 0x26, 0x30,
                           0x38, 0x37, 0x39, 0x2C, 0x7B, 0x31, 0x39, 0x39, 0x55};
    for (size_t i = 0; i < sizeof(enc) - 1; i++) {
        enc[i] ^= 0x55;
    }
    return std::string(reinterpret_cast<char*>(enc));
}

uintptr_t getModuleBase(pid_t pid, std::string modName) {
    std::string mapsPath = "/proc/" + std::to_string(pid) + "/maps";
    std::ifstream maps(mapsPath);
    if (!maps.is_open()) {
        std::cerr << "Maps is closed!\n";
        return 0;
    }
    std::string line;
    while (std::getline(maps, line)) {
        if (line.find(modName) != std::string::npos) {
            uintptr_t base = std::stoull(line.substr(0, line.find('-')), nullptr, 16);
            maps.close();
            return base;
        }
    }
    maps.close();
    return 0;
}

int polyai(uintptr_t modBase, pid_t pid, bool prod) {
    std::unordered_map<uint16_t, TileInfo> tileMap;
    std::unordered_map<uint16_t, StructureInfo> structMap;
    std::unordered_map<uint16_t, Data::ResourceState> resourceMap;
    std::unordered_map<uint16_t, CityInfo> cityMap;
    std::unordered_map<uint16_t, Data::UnitData> unitMap;
    std::unordered_map<uint16_t, Data::PlayerState> tribesMap;

    /**=
     * BotDifficulty / BaseGameMode / UnitEffect
     *   0x10: Int32, value (enum)
     * 
     *   other interesting ones like: current viewing player, curlocalplaerindex
     * 
     * Shoreline
     *   0x10 Bool visible
     *   0x18: String spriteExt
     * 
     * Shorelines
     *   0x10: Bool any
     *   0x18: Shoreline N
     *   0x20: Shoreline S
     *   0x28: Shoreline E
     *   0x30: Shoreline W
     * 
     * WorldContinent
     *   0x10: List<WorldCoordinates> tiles
     *   0x18: Int32 climate
     *   0x1C: SkinType skinType
     *   0x20: Single crop
     *   0x24: Single fish
     *   0x28: Single fruit
     *   0x2C: Single game
     *   0x30: Single metal
     *   0x34: Single whale
     *   0x38: Single spores
     *   0x3C: Single aquacrop
     *   0x40: Single water
     *   0x44: Single ocean
     *   0x48: Single field
     *   0x4C: Single mountiain
     *   0x50: Single forest
     *   0x54: Single ice
     *   0x54: Bool hasAlienClimate
     *   0x54: Int32 LandTileCount
     *   0x54: Int32 NumerOfCapitals
     *   0x54: Int32 MaxSize
     *   ...
     * 
     * PlayerData
     *   0x10: PlayerDataType type
     *   0x14: PlayerDataFriendshipState state
     *   0x18: Bool isSpectating
     *   0x19: Bool knownTribe
     *   0x1C: TribeType tribe
     *   0x20: TribeType tribeMix
     *   0x24: BotDifficulty botDifficulty
     *   0x28: SkinType skinType
     *   0x30: PlayerProfileState profile 
     *   0x38: String defaultName
     * 
     * TerrainData
     *   0x10: Int32 idx
     * 
     * WorldCoords
     *   0x10: Int32 x
     *   0x14: Int32 y
     * 
     * ImprovementData.Type
     *   IceTemple, IceBank, polarisClimate, Algae, Funci, Mycelium, Outpost, EnchantWhale, Monument6, Sanctuary, Monument7, EnchantAnimal, Monument5, BurnSpores, HiddenSanctuary, Market, Aquafarm, Atoll, Fertilize, Canal, Clathrus, Bridge, StarFishing, HarvestSpores, LightHouse, NullBuilding, Cultivate, Landfill, Monument2, Windmill, Farm, Fishing, Hunting, Port, Monument3, CustomsHouse, Ruin, Road, None, City, BurnForest, ClearForest, Sawmill, Mine, MountainTemple, Forge, Monument1, LumberHut, WaterTemple, Temple, GrowForest, ForestTemple, Algae Spawn, Whale Hunting, HarvestFruit
     * 
     * TerrainData.Type
     *   0x10: Int32 value
     *   Mountain, Field, Forest, Ice, Wetland, Ocean, None, Water, Mangrove
     * 
     * UnitData.WeaponType
     *   0x10: Int32 value
     *   Poison, IceArrow, Trident, FireBlow, Sting, Burn, Dagger, Lasr, Pierce, Claw, Gun, None, Rock, Club, Arrow, Sword, Magic, Water
     * 
     * UnitData.UnitEffect:
     *   0x10: Int32 value
     *   Invisible, Bubble, Petrified, Swift, Boosted, Frozen, Poisoned, DoubleReady
     * 
     * UnitData.UnitAbility.Type:
     *   0x10: Int32 value
     *   Infiltrate, Surprise, Detect, Intercept, Explode, Stiff, Eat, Surprise, Unique, Independent, Poison, 
     *   Boost, Protect, AutoHeal, Land, Water, Algae, Consumed, Stomp, Amphibious, AutoFlood, Drench, Static,
     *   Tentacles, Swarm, Creep, Skate, Sneak, Hide, Persist, Convert, Scout, Dash, Escape, None, Fortify,
     *   Heal, Carry, Freeze, FreezeArea, AutoFreeze, Swarm, Crush, Splash, Grow, Navigate, DoubleAttack
     * 
     * UnitData.Type:
     *   0x10: Int32 value
     *   Pirate, Cloak_Boat, Cloak, Bombership, Transportship, Scoutship
     *   Rammership, Dagger, Raychi, Kiton, Shaman, Exida, Segment, Centipede, Phychi, Juggernaut
     *   Island, Boomchi, Aquapult, Ciru, BugEgg, Mantis, MermaidWarrior, Siren, Jelly, MermaidSwordsman, Shark
     *   MermaidDefender, MermaidDagger, MermaidCloak, Moth, Doomux, Gaami, Battleship, Ship, Catapult, MindBender
     *   Archer, Swordsman, Defender, Rider, Knight, None, Warrior, Scout, Hexapod, Giant, Boat, BattleSled,
     *   Mooni, IceFortress, Crab, IceArcer, Bunny, Tridention, FireDragon, Polytaur, Amphibian, Navalon, Larva
     *   DragonEgg, BabyDragon
     * 
     * ImprovementAbility:
     *   Freelance, Manual, Embark, Network, Poison, Slow, Doubled, Picky, Discrete, Flood, Fill, Bridge, Heal, Attract, None, Limited, Unique, Consumed, Expand, Patina, Simple
     *   0x10: Int32 value
     * 
     * Creates:
     *   0x10: TerrainData terrain
     *   0x18: ResourceData resource
     *   0x20: UnitData unit
     *   0x18: TileData.EffectType effect
     * 
     * GrowthRewards:
     *   0x10: Int32 population
     *   0x14: Int32 currency
     *   0x18: Int32 score
     * 
     * GrowthRewards:
     *   0x10: Int32 score
     *   0x14: Int32 population
     * 
     * TerrainRequirements:
     *   0x10: TerrainData terrain
     *   0x18: ResourceData resource
     * 
     * AdjacencyRequirements:
     *   0x10: TerrainData terrain
     *   0x18: ImprovementData improvement
     *   0x20: ResourceData resource
     *   0x28: ImprovementData notImprovement
     * 
     * AdjacencyImprovements:
     *   0x10: ImprovementData improvement
     *   0x18: ResourceData resources
     * 
     * GrowthRewards:
     *   0x10: Int32 score
     *   0x14: Int32 population
     * 
     * ImprovementData:
     *   0x10: Int32 idx
     *   0x14: Bool hidden
     *   0x18: Int32 cost
     *   0x1C: Int32 work
     *   0x20: Int32 borderSize
     *   0x24: Int32 maxLevel
     *   0x28: List<ImprovementAbility.Type> improvementAbilities
     *   0x30: List<Creates> creates
     *   0x38: List<Rewards> rewards
     *   0x40: List<TerrainRequirements> terrainRequirements
     *   0x48: List<AdjacencyRequirements> adjacencyRequirements
     *   0x50: List<AdjacencyImprovements> adjacencyImprovemnts
     *   0x58: List<TerrainData> routes
     *   0x60: Int32 range
     *   0x64: Int32 growthRate
     *   0x68: List<GrowthRewards> growthRewards
     */

    /*
     * ClientBase
     *   0x10: ClientType clientType
     *   0x18: ActionManager clientActionManager
     *   0x20: gameId
     *   0x30: GameState initialGameState
     *   0x38: GameState currentGameState (<--- mapBase[0] !!!)
     *   ...
     *   0x78: Int32 lastTurnGameState (<--- turnBase[0] !!!)
     * 
     * GameManager
     *   0x20: GameSettings settings
     *   0x28: ClientBase client
     *   0x50: Int32 aiOpponents
     *   0x54: Int32 playerOpponents
     *   0x64: TribeType startingTribe
     * 
     *   other interesting ones like: ladderManager, tornamentManager, lobbyMana, replaysMana
     *   ...
     *   0x80: Int32 opponentcount
     */
    
<<<<<<< HEAD
    uintptr_t gameManagerRoot = getPlace(pid, modBase + 0x36700C0, { 0xB8, 0x0 });
    // std::cout << "GM: 0x" << std::hex << gameManagerRoot << std::endl << std::dec;
    uintptr_t playersRoot = getPlace(pid, gameManagerRoot, {Offsets::GameManager_ClientBase, Offsets::ClientBase_CurrentGameState, Offsets::GameState_PlayerStates, _0x_IN_LIST});
    uintptr_t currentTurnRoot = getPlace(pid, gameManagerRoot, {Offsets::GameManager_ClientBase, Offsets::ClientBase_CurrentGameState, Offsets::GameState_CurrentTurn});
    uintptr_t mapRoot = getPlace(pid, gameManagerRoot, {Offsets::GameManager_ClientBase, Offsets::ClientBase_CurrentGameState, Offsets::GameState_Map, _0x_MAP_TILES});
=======
    uintptr_t gameManagerRoot = getPlace(pid, modBase + 0x3674378, { 0xB8, 0x0 });
    uintptr_t playersRoot = getPlace(pid, gameManagerRoot, {_0x_GAMEMANAGER_CLIENT, _0x_CLIENT_CUR_STATE, _0x_STATE_PLAYERS, _0x_IN_LIST});
    uintptr_t currentTurnRoot = getPlace(pid, gameManagerRoot, {_0x_GAMEMANAGER_CLIENT, _0x_CLIENT_CUR_STATE, _0x_STATE_CUR_TURN});
    uintptr_t mapRoot = getPlace(pid, gameManagerRoot, {_0x_GAMEMANAGER_CLIENT, _0x_CLIENT_CUR_STATE, _0x_STATE_MAP, _0x_MAP_TILES});
>>>>>>> 1d69340be55dd05c5986f1f109272c2afef95f24
    std::string _logPlayers = "";

    // std::cout << "Game manager address: " << std::hex << gameManager << std::endl;

    if (!gameManagerRoot || !currentTurnRoot || !mapRoot || !playersRoot) {
        if (!gameManagerRoot) {
            std::cerr << "Failed to get game manager address\n";
        }
        if (!currentTurnRoot) {
            std::cerr << "Failed to get current turn address\n";
        }
        if (!mapRoot) {
            std::cerr << "Failed to get map address\n";
        }
        if (!playersRoot) {
            std::cerr << "Failed to get players address\n";
        }
        return -1;
    }

    int32_t gameId;

    readPiece(pid, getPlace(pid, gameManagerRoot, { Offsets::GameManager_ClientBase, Offsets::ClientBase_GameID }), gameId);

    // ! SETTINGS ! //

    // // ClientInteraction.SelectTile
    // uintptr_t FN_ADDR = modBase + 0x1AD6f90;
    // static const uintptr_t CLIENT_INSTANCE_ADDR = 0x7112D1D0;
    // static const uintptr_t TILE_ADDR = 0x74018280;
    // using FnType = void(*)(void*, void*);
    // FnType fn = reinterpret_cast<FnType>(FN_ADDR);

    // // std::cout << "fn: " << std::hex << FN_ADDR << std::dec << std::endl;
    // // std::cout << "tile: " << std::hex << CLIENT_INSTANCE_ADDR << std::dec << std::endl;
    // // std::cout << "selectTile: " << std::hex << TILE_ADDR << std::dec << std::endl;

    // volatile void* inst = reinterpret_cast<void*>(CLIENT_INSTANCE_ADDR);
    // volatile void* tile = reinterpret_cast<void*>(TILE_ADDR);

    // // std::cout << "invoking function..." << std::endl;

    // // call
    // fn((void*)tile, (void*)inst);

    // std::cout << "function returned (if it returned)." << std::endl;

    // return 1;

    /*
     * GameSettings
     *   0x10: BotDifficulty difficulty
     *   0x14: BaseGameMode baseGameMode
     *   0x18: BaseGameMode rulesGameMode
     *   0x30: List<TribeType> selectedSkins
     *   0x38: List<PlayerData> players
     *   0x40: List<PlayerData> spectators
     *   0x48: MapPreset mapPreset
     *   0x50: GameRules rules
     *   0x58: String gameName
     *   0x60: GameType gameType
     *   0x64: Int32 mapSize
     *   0x68: Int32 timeLimit
     */
    
<<<<<<< HEAD
    uintptr_t gameSettingsRoot = getPlace(pid, gameManagerRoot, {Offsets::GameManager_Settings});
=======
    uintptr_t gameSettingsRoot = getPlace(pid, gameManagerRoot, {_0x_GAMEMANAGER_SETTINGS});
>>>>>>> 1d69340be55dd05c5986f1f109272c2afef95f24
    unsigned char settingsBuffer[Offsets::GameSettings_Size_]; 
    readBlock(pid, getPlace(pid, gameSettingsRoot, {0x0}), settingsBuffer, sizeof(settingsBuffer));

    // std::cout << "Game settings address: " << std::hex << gameSettingsRoot << std::endl;
    int32_t mapSize, baseGameMode, timeLimit;
    std::string gameName;
    bool winByCapital, winByExtermination, allowMirrorPick, allowSpecialTribe, allowTechSharing;
    readPiece(pid, getPlace(pid, gameSettingsRoot, {Offsets::GameSettings_BaseGameMode}), baseGameMode);
    readPiece(pid, getPlace(pid, gameSettingsRoot, {Offsets::GameSettings_TimeLimit}), timeLimit);
    BRUHMOTHERFUCKINGSHIT(pid, getPlace(pid, gameSettingsRoot, {Offsets::GameSettings_GameName, 0x0}), gameName);
    // readPiece(pid, getPlace(pid, gameSettingsRoot, {Offsets::GameSettings_MapSize}), mapSize);

    mapSize = *(int32_t*)&settingsBuffer[Offsets::GameSettings_MapSize];
    // std::cout << "Name: " << gameName << std::endl;

    /*
     * MapData
     *   0x10: UInt16, width
     *   0x12: UInt16, height
     *   0x18: TileData[] tiles
     *   0x20: WorldContinent[] continents
     */

<<<<<<< HEAD
    readPiece(pid, getPlace(pid, gameManagerRoot, {Offsets::GameManager_ClientBase, Offsets::ClientBase_CurrentGameState, Offsets::GameState_Map, 0x12 }), mapSize);
=======
    readPiece(pid, getPlace(pid, gameManagerRoot, {_0x_GAMEMANAGER_CLIENT, _0x_CLIENT_CUR_STATE, Offsets::GameState_Map, 0x12 }), mapSize);
>>>>>>> 1d69340be55dd05c5986f1f109272c2afef95f24
    // std::cout << "Map size: " << mapSize << std::endl;

    /*
     * GameRules
     *   0x10: Int32 turnLimit
     *   0x14: Int32 scoreLimit
     *   0x18: Bool winByCapital
     *   0x19: Bool winByExtermination
     *   0x1a: Bool allowMirrorPick
     *   0x1b: Bool allowSpecialTribe
     *   0x1c: Bool allowTechSharing
     *   0x20: GameRules.DeathCondition playerDeathCondition
     * 
     * GameRules.DeathCondition
     *   0x10: Int32 type
     */

    readPiece(pid, getPlace(pid, gameSettingsRoot, {Offsets::GameSettings_Rules, Offsets::GameRules_WinByCapital}), winByCapital);
    readPiece(pid, getPlace(pid, gameSettingsRoot, {Offsets::GameSettings_Rules, Offsets::GameRules_WinByExtermination}), winByExtermination);
    readPiece(pid, getPlace(pid, gameSettingsRoot, {Offsets::GameSettings_Rules, Offsets::GameRules_AllowMirrorPick}), allowMirrorPick);
    readPiece(pid, getPlace(pid, gameSettingsRoot, {Offsets::GameSettings_Rules, Offsets::GameRules_AllowSpecialTribe}), allowSpecialTribe);
    readPiece(pid, getPlace(pid, gameSettingsRoot, {Offsets::GameSettings_Rules, Offsets::GameRules_AllowTechSharing}), allowTechSharing);
    
    /*
     * GameState
     *   0x24: GameState currentState
     *   0x28: GameSettings settings
     *   0x30: MapData map
     *   0x38: List<PlayerState> playerStates
     */

    int32_t seed, version;
    uint32_t currentTurn, currentUnitID;
    uint8_t currentPlayerIndex;

    uintptr_t gameStateBase = getPlace(pid, gameManagerRoot, {_0x_GAMEMANAGER_CLIENT, _0x_CLIENT_CUR_STATE, 0x0});

    readPiece(pid, gameStateBase + Offsets::GameState_Version, version);
    readPiece(pid, gameStateBase + Offsets::GameState_Seed, seed);
    readPiece(pid, gameStateBase + Offsets::GameState_CurrentTurn, currentTurn);
    readPiece(pid, gameStateBase + Offsets::GameState_CurrentUnitID, currentUnitID);
    readPiece(pid, gameStateBase + Offsets::GameState_CurrentPlayerIndex, currentPlayerIndex);

    // ! TRIBES ! //    
    uint16_t tribeCount = 0;
    readPiece(pid, getPlace(pid, playersRoot, { _0x_IN_LIST_COUNT }), tribeCount);
    // The last one is always "Nature"
    tribeCount -= 1;

    for (uint32_t index = 0; index < tribeCount; ++index) {
        uintptr_t playerRoot = getPlace(pid, playersRoot, { index * 0x8 + _0x_IN_LIST_START_SHIFT });
        uintptr_t playerBase = getPlace(pid, playerRoot, { 0x0 });
        unsigned char playerBuffer[PlayerStateOffsets::SIZE_]; 

        if (playerRoot == 0 || playerBase == 0 || !readBlock(pid, playerBase, playerBuffer, sizeof(playerBuffer))) {
            break;
        }
        
        int32_t currency, resignedTurn, killedTurn, score, kills, casualties;
        int16_t startingTileX, startingTileY, tribeType;
        uint8_t id, killerId;
        std::vector<int32_t> builtUniqueImprovements, tech;
        std::vector<uint8_t> knownPlayers;
        std::unordered_map<uint16_t, Data::DiplomacyRelation> relations;
        std::string tasks, username; 
        bool autoplay; 

        uintptr_t usernameRoot = getPlace(pid, *(uintptr_t*)&playerBuffer[PlayerStateOffsets::USERNAME], {  });
        BRUHMOTHERFUCKINGSHIT(pid, usernameRoot, username);
        
        // std::cout << "[player]: " << username << std::endl;
        _logPlayers += "\n" + username;

        // std::cout << "[address]: 0x" << std::hex << usernameRoot << std::endl << std::dec;
        id            = *(uint8_t*)&playerBuffer[PlayerStateOffsets::ID];
        autoplay      = *(bool*)&playerBuffer[PlayerStateOffsets::AUTOPLAY];
        startingTileX = *(int16_t*)&playerBuffer[PlayerStateOffsets::START_TILE_X];
        startingTileY = *(int16_t*)&playerBuffer[PlayerStateOffsets::START_TILE_Y];
        tribeType     = *(int16_t*)&playerBuffer[PlayerStateOffsets::TRIBE_TYPE];
        resignedTurn  = *(int32_t*)&playerBuffer[PlayerStateOffsets::RESIGNED_TURN];
        casualties    = *(int32_t*)&playerBuffer[PlayerStateOffsets::CASUALTIES];
        readSingleListMagic(pid, getPlace(pid, playerRoot, { PlayerStateOffsets::AVAILABLE_TECH }), tech);
        // TODO tasks
        // TODO aggressions
        readSingleListMagic(pid, getPlace(pid, playerRoot, { PlayerStateOffsets::KNOWN_PLAYERS }), knownPlayers);
        // std::cout << "[address]: 0x" << std::hex << getPlace(pid, playerRoot, { PlayerStateOffsets::RELATIONS }) << std::endl << std::dec;
        readDictionaryMagic(
            pid, 
            getPlace(pid, playerRoot, { PlayerStateOffsets::RELATIONS }), 
            relations, 
            [](uint16_t key, unsigned char *buffer) -> Data::DiplomacyRelation {
                int32_t state               = *(int32_t*)&buffer[DiplomacyRelationOffsets::STATE];
                int32_t lastAttackTurn      = *(int32_t*)&buffer[DiplomacyRelationOffsets::LAST_ATTACK_TURN];
                int32_t embassyLevel        = *(int32_t*)&buffer[DiplomacyRelationOffsets::EMBASSY_LEVEL];
                int32_t lastPeaceBrokenTurn = *(int32_t*)&buffer[DiplomacyRelationOffsets::LAST_PEACE_BROKEN_TURN];
                int32_t firstMeet           = *(int32_t*)&buffer[DiplomacyRelationOffsets::FIRST_MEET];
                int32_t embassyBuildTurn    = *(int32_t*)&buffer[DiplomacyRelationOffsets::EMBASSY_BUILD_TURN];
                int32_t previousAttackTurn  = *(int32_t*)&buffer[DiplomacyRelationOffsets::PREVIOUS_ATTACK_TURN];
                return { state, lastAttackTurn, embassyLevel, lastPeaceBrokenTurn, firstMeet, embassyBuildTurn, previousAttackTurn };
            },
            Offsets::DiplomacyRelation_Size_
        );
        
        // TODO messages
        // PlayerStateOffsets::MESSAGES;
        // TODO skinType
        currency      = *(int32_t*)&playerBuffer[PlayerStateOffsets::CURRENCY];
        score         = *(int32_t*)&playerBuffer[PlayerStateOffsets::SCORE];
        kills         = *(int32_t*)&playerBuffer[PlayerStateOffsets::KILLS];
        // TODO casualities
        // TODO wipeOuts
        killerId      = *(uint8_t*)&playerBuffer[PlayerStateOffsets::KILLER_ID];
        killedTurn    = *(int32_t*)&playerBuffer[PlayerStateOffsets::KILLED_TURN];
        
        readSingleListMagic(pid, getPlace(pid, playerRoot, { PlayerStateOffsets::BUILT_UNIQUE_IMPROVEMENTS }), builtUniqueImprovements);
      
        // TODO apply new startingTileXY to output

        tribesMap[index] = { 
            id, username, currency, score, autoplay, tech, tribeType, 
            killerId, kills, tasks, builtUniqueImprovements, knownPlayers, 
            relations, killedTurn, resignedTurn, startingTileX, startingTileY,
            casualties
        };
    }
    
    // ! MAP ! //
    
    uint16_t tileCount = mapSize * mapSize, unitCount = 0;
    
    for (int32_t index = 0; index < tileCount; ++index) {
        uintptr_t tileRoot = getPlace(pid, mapRoot, { index * 0x8 + _0x_IN_LIST_START_SHIFT });
        uintptr_t tileBase = getPlace(pid, tileRoot, { 0x0 });
        unsigned char tileBuffer[_0x_TILE_HAD_ROUTE + _0x_TRAILING_OFFSET]; 

        if (tileRoot == 0 || tileBase == 0 || !readBlock(pid, tileBase, tileBuffer, sizeof(tileBuffer))) {
            break;
        }

        int32_t type, tileX, tileY, rulingCityX, rulingCityY, 
            skinType, climateType, explorersCount;
        uint8_t owner, capitalOf, byteTribeIndex, byteCapitalOf;
        bool hasRoad, hasRoute, hadRoute; 
        std::vector<uint8_t> explorers;
        uintptr_t unitBase, resourceBase, improvementBase;  
        
        // ! TileData
        /*
         *   0x10: WorldCoordinates coordenates
         *   0x18: TerrainData.Type terrain
         *   0x1C: Int32 climate
         *   0x20: SkinType _skin
         *   0x30: Int32 altitiude
         *   0x34: Byte owner
         *   0x35: Byte capitalOf
         *   0x38: List<Byte> explorers
         *   0x40: TileData.Shorelines shorelines
         *   0x48: WorldCoordinates, rulingCityCoordenates
         *   0x50: ImprovementState improvement
         *   0x58: ResourceState resource
         *   0x60: UnitState unit
         *   0x68: Bool hasRoad
         *   0x69: Bool hasRoute
         *   0x70: WorldContinent continent
         *   0x78: Bool hadRoute
         *   0x80: Dictionary<TechdData.Type,System.Single> upgradeTech
         *   0x88: Int64 lastPopulationCheck
         *   0x90: Int32 availablePopulation
         * 
         * ResourceState
         *   0x10: ResourceType type
         */

        tileX           = *(int32_t*)&tileBuffer[_0x_TILE_X];
        tileY           = *(int32_t*)&tileBuffer[_0x_TILE_Y];
        type            = *(int32_t*)&tileBuffer[_0x_TILE_TERRAIN_TYPE];
        climateType     = *(int32_t*)&tileBuffer[_0x_TILE_CLIMATE_TYPE]; 
        skinType        = *(int32_t*)&tileBuffer[_0x_TILE_SKIN_TYPE];    
        owner           = *(uint8_t*)&tileBuffer[_0x_TILE_OWNER];
        capitalOf       = *(uint8_t*)&tileBuffer[_0x_TILE_CAPITAL_OF];
        rulingCityX     = *(int32_t*)&tileBuffer[_0x_TILE_RULING_CITY_X];
        rulingCityY     = *(int32_t*)&tileBuffer[_0x_TILE_RULING_CITY_Y];
        improvementBase = *(uintptr_t*)&tileBuffer[_0x_TILE_IMPROVEMENT];
        resourceBase    = *(uintptr_t*)&tileBuffer[_0x_TILE_RESOURCE];
        unitBase        = *(uintptr_t*)&tileBuffer[_0x_TILE_UNIT];
        hasRoad         = *(bool*)&tileBuffer[_0x_TILE_HAS_ROAD];                  
        hasRoute        = *(bool*)&tileBuffer[_0x_TILE_HAS_ROUTE];                 
        hadRoute        = *(bool*)&tileBuffer[_0x_TILE_HAD_ROUTE];  

        readSingleListMagic(pid, tileBase + _0x_TILE_EXPLORERS, explorers);

        // ! UnitState
        /*
         *   0x10: Uint32 id
         *   0x14: Uint32 leader
         *   0x18: UInt32 follower
         *   0x1c: Byte owner
         *   0x1e: Int16 sytle
         *   0x20: SkinType skinType
         *   0x24: UnitData.Type type
         *   0x28: WorldCoordinates previousTurnEndCoordinates
         *   0x30: WorldCoordinates coordinates
         *   0x38: WorldCoordinates home
         *   0x40: UnitState passengerUnit
         *   0x48: Uint16 health
         *   0x4a: Uint16 promotionLevel
         *   0x4c: UInt16 xp
         *   0x4e: Bool moved
         *   0x4f: Bool attacked
         *   0x50: GridDirection direction
         *   0x54: Bool flipped
         *   0x56: UInt16 createdTurn
         *   0x58: UnitData unitData
         *   0x60 List<UnitEffect> effects
        */

        unsigned char unitBuffer[_0x_UNIT_EFFECTS + _0x_TRAILING_OFFSET]; 
        if (unitBase != 0 && readBlock(pid, unitBase, unitBuffer, sizeof(unitBuffer))) {
            unitCount += 1;
            // std::cout << "[-] " << std::hex << unitBase << std::endl << std::dec;
            // std::cout << "~unit detected~ 0x" << std::hex << unitBase << std::endl << std::dec;
            
            /*
             * UnitData
             *   0x10: Int32 idx
             *   0x14: Bool hidden
             *   0x18: Int32 cost
             *   0x1c: Int32 health
             *   0x20: Int32 defense
             *   0x24: Int32 movement
             *   0x28: UnitData.WeaponType weapon
             *   0x2c: Int32 range
             *   0x30: Int32 attack
             *   0x38: List<UnitAbility.Type> unitAbilities
             *   0x40: List<TerrainData> movementTerrain
             *   0x48: UnitData upgradesFrom
             *   0x50: UnitData promotesTo
             *   0x58: Int32 growthRate
             */ 

            std::vector<int32_t> effects;
            bool moved, attacked, flipped; 
            uint8_t owner;
            uint16_t promoted, type, tileX, tileY, hp, xp, prevTileX, prevTileY, 
                homeX, homeY, direction, createdTurn;
            // uint16_t classId, classHp, classCost, classDef, classMov, classAtk, classWpn, classRange;
            // bool classHidden;
            
            // ... id
            owner       = *(uint8_t*)&unitBuffer[_0x_UNIT_OWNER];
            // ... style, skinType
            type        = *(uint16_t*)&unitBuffer[_0x_UNIT_TYPE];
            prevTileX   = *(uint16_t*)&unitBuffer[_0x_UNIT_PREV_TURN_END_X];
            prevTileY   = *(uint16_t*)&unitBuffer[_0x_UNIT_PREV_TURN_END_Y];
            tileX       = *(uint16_t*)&unitBuffer[_0x_UNIT_X];
            tileY       = *(uint16_t*)&unitBuffer[_0x_UNIT_Y];
            homeX       = *(uint16_t*)&unitBuffer[_0x_UNIT_HOME_X];
            homeY       = *(uint16_t*)&unitBuffer[_0x_UNIT_HOME_Y];
            hp          = *(uint16_t*)&unitBuffer[_0x_UNIT_HEALTH];
            promoted    = *(uint16_t*)&unitBuffer[_0x_UNIT_PROMOTION_LEVEL]; 
            xp          = *(uint16_t*)&unitBuffer[_0x_UNIT_XP];
            direction   = *(uint16_t*)&unitBuffer[_0x_UNIT_DIRECTION];
            createdTurn = *(uint16_t*)&unitBuffer[_0x_UNIT_CREATED_TURN];
            moved       = *(bool*)&unitBuffer[_0x_UNIT_MOVED];
            attacked    = *(bool*)&unitBuffer[_0x_UNIT_ATTACKED];
            flipped     = *(bool*)&unitBuffer[_0x_UNIT_FLIPPED];
<<<<<<< HEAD
            
            if (homeX > 6553 || homeY > 6553) {
                homeX = -1;
                homeY = -1;
            }
            
=======

>>>>>>> 1d69340be55dd05c5986f1f109272c2afef95f24
            readSingleListMagic(pid, getPlace(pid, tileBase + _0x_TILE_UNIT, {_0x_UNIT_EFFECTS}), effects);

            uint16_t passengerType;
            uintptr_t passengerBase = getPlace(pid, tileBase + _0x_TILE_UNIT, {_0x_UNIT_PASSENGER_UNIT, _0x_UNIT_TYPE});  
<<<<<<< HEAD
            if (passengerBase != 0) {
=======
            if(passengerBase != 0) {
>>>>>>> 1d69340be55dd05c5986f1f109272c2afef95f24
                readPiece(pid, passengerBase, passengerType);

                if (passengerType > 255) {
                    passengerType = 0;
                }
            }

            unitMap[index] = { 
                owner, tileX, tileY, type, hp, promoted, xp, 
                prevTileX, prevTileY, homeX, homeY, direction,
                flipped, createdTurn, moved, attacked, passengerType,
                effects
            };
        }
      
        // ! ImprovementState
        /*
         *   0x10: ImprovementData.Type type
         *   0x14: Byte owner
         *   0x15: Byte founder
         *   0x16: Int16 level
         *   0x18: Int16 founded
         *   0x1a: Int16 xp
         *   0x1c: Int16 population
         *   0x1e: UInt16 production
         *   0x20: UInt16 baseScore
         *   0x22: UInt16 borderSize
         *   0x24: UInt16 upgrade
         *   0x26: Byte connectedToCapitalOfPlayer
         *   0x28: String name
         *   0x30: List<CityReward> rewards
         *   0x38: List<ImprovementEffect> effects 
         */

        unsigned char structureBuffer[Offsets::ImprovementState_Effects + _0x_TRAILING_OFFSET]; 
        if (improvementBase != 0 && readPiece(pid, improvementBase, structureBuffer)) {
            int16_t type, level, founded, progress, population;
            uint16_t production, baseScore, borderSize, upgrade;
            uint8_t owner, founder;
            bool connectedToCapitalOfPlayer;
            std::string name, effects;

            type        = *(int16_t*)&structureBuffer[Offsets::ImprovementState_Type];
            level       = *(int16_t*)&structureBuffer[Offsets::ImprovementState_Level];
            founded     = *(int16_t*)&structureBuffer[Offsets::ImprovementState_Founded];
            progress    = *(int16_t*)&structureBuffer[Offsets::ImprovementState_XP];
            baseScore   = *(uint16_t*)&structureBuffer[Offsets::ImprovementState_BaseScore];
            upgrade     = *(uint16_t*)&structureBuffer[Offsets::ImprovementState_Upgrade];
            owner       = *(uint8_t*)&structureBuffer[Offsets::ImprovementState_Owner];
            founder     = *(uint8_t*)&structureBuffer[Offsets::ImprovementState_Founder];
            // TODO verify == 1
            
            uintptr_t nameBase = getPlace(pid, *(uintptr_t*)&structureBuffer[Offsets::ImprovementState_Name], {  });
            
            if (BRUHMOTHERFUCKINGSHIT(pid, nameBase, name)) {
                std::vector<int32_t> rewards;
                connectedToCapitalOfPlayer = *(uint8_t*)&structureBuffer[Offsets::ImprovementState_ConnectedToCapital] == 1;
                readSingleListMagic(pid, improvementBase + Offsets::ImprovementState_Rewards, rewards);
                population  = *(int16_t*)&structureBuffer[Offsets::ImprovementState_Population];
                production  = *(uint16_t*)&structureBuffer[Offsets::ImprovementState_Production];
                borderSize  = *(uint16_t*)&structureBuffer[Offsets::ImprovementState_BorderSize];
                cityMap[index] = { 
                    name, population, progress, rewards, production,
                    borderSize, connectedToCapitalOfPlayer, level 
                };
            }
            
            structMap[index] = { type, level, founded, baseScore };
        }

        // ! ResourceState
        /*
         *   0x10: ResourceType type
         */
        
        int16_t resourceId = 0;
        if(resourceBase != 0 && readPiece(pid, resourceBase + 0x10, resourceId)) {
        // if(resourceBase != 0 && readPiece(pid, getPlace(pid, tileRoot, {_0x_TILE_RESOURCE, 0x10}), resourceId)) {
        // if(resourceBase != 0 && readPiece(pid, getPlace(pid, tileBase + _0x_TILE_RESOURCE, {0x10}), resourceId)) {
            // std::cout << "[resource] 0x" << std::hex << resourceBase << std::endl << std::dec;
            // std::cout << "resourceId: " << resourceId << std::endl;
            resourceMap[index] = { resourceId };
        }

        tileMap[index] = { 
            index, type, owner, explorers, hasRoad, hasRoute, hadRoute, 
            capitalOf, rulingCityX, rulingCityY, skinType, climateType,
            tileX, tileY,
        };
    }
   
    // ! WRITE OUT ! //

    if(!prod) {
        std::cout << std::dec << "Turn: " << currentTurn 
            << " | Map size: " << mapSize << "x" << mapSize << " (" << tileCount << ")" 
            << " | Units: " << unitCount 
            << " | Tribes: " << _logPlayers 
            << " | GameID: " << gameId 
            // << " | Game: " << gameName
            << std::endl;
        return 0;
    }

    auto coord_to_index = [](uint16_t x, uint16_t y, int max_size) {
        return y * max_size + x;
    };

    json root, tribes, tiles, settings, resources, structures;
    
    // settings
    settings["version"] = version;
    settings["gameId"] = gameId;
    settings["mode"] = baseGameMode;
    settings["size"] = mapSize;
    settings["tileCount"] = mapSize * mapSize;
    settings["turn"] = currentTurn;
    settings["seed"] = seed;
    settings["maxTurns"] = baseGameMode == 1? 30 : baseGameMode == 2? 0 : timeLimit;
    settings["currentPlayerTurnId"] = tribesMap[currentPlayerIndex].id;
    settings["gameName"] = gameName;
    settings["winByCapital"] = winByCapital;
    settings["winByExtermination"] = winByExtermination;
    settings["_lastPlayerTurnId"] = -1;
    settings["_fow"] = true;
    settings["_gameOver"] = false;
    settings["_areYouSure"] = false;
    settings["_recentMoves"] = json::array();
    settings["_pendingRewards"] = json::array();
    settings["_maxTribeCount"] = tribesMap.size();

    // tribes
    for (const auto &kv : tribesMap) {
        const auto &p = kv.second;
        json j;
        j["id"] = p.id;
        j["username"] = p.username;
        j["bot"] = p.autoplay;
        j["score"] = p.score;
        j["stars"] = p.currency;
        j["startingTileCoords"] = json::array({p.startingTileX, p.startingTileY});
        j["type"] = p.tribeType;
        j["killerId"] = p.killerId;
        j["kills"] = p.kills;
        j["tasks"] = p.tasks;
        j["builtUniqueImprovements"] = p.builtUniqueImprovements;
        j["knownPlayers"] = p.knownPlayers;
        j["killedTurn"] = p.killedTurn;
        j["resignedTurn"] = p.resignedTurn;
        j["casualties"] = p.casualties;

        json tech = json::array();
        for (const auto &kv : p.tech) {
            json jm;
            if (kv > 255) break;
            jm["type"] = kv;
            jm["discovered"] = true;
            tech.push_back(std::move(jm));
        }

        json relations = json::object();
        for (const auto &kv : p.relations) {
            const auto &r = kv.second;
            json jr;
            jr["state"] = r.state;
            jr["lastAttackTurn"] = r.lastAttackTurn;
            jr["embassyLevel"] = r.embassyLevel;
            jr["lastPeaceBrokenTurn"] = r.lastPeaceBrokenTurn;
            jr["firstMeet"] = r.firstMeet;
            jr["embassyBuildTurn"] = r.embassyBuildTurn;
            jr["previousAttackTurn"] = r.previousAttackTurn;
            relations[std::to_string(kv.first)] = jr;
        }

        j["tech_vanilla"] = tech;
        j["relations"] = relations;
        j["units"] = json::array();
        j["cities"] = json::array();

        tribes[std::to_string(p.id)] = j;
    }

    // tiles, resources, structures, cities
    for (const auto &kv : tileMap) {
        const auto &t = kv.second;
        const auto idx = coord_to_index(t.tileX, t.tileY, mapSize);
        json jt;
        jt["owner"] = t.owner;
        jt["coords"] = json::array({t.tileX, t.tileY});
        jt["type"] = t.type;
        jt["explorers"] = t.explorers;
        jt["hasRoad"] = (bool)t.hasRoad;
        jt["hasRoute"] = (bool)t.hasRoute;
        jt["hadRoute"] = (bool)t.hadRoute;
        jt["capitalOf"] = t.capitalOf;
        jt["climate"] = t.climate;
        jt["skinType"] = t.skinType;
        if (t.rulingCityX != -1) {
            jt["rulingCityCoords"] = json::array({t.rulingCityX, t.rulingCityY});
        }
        else {
            jt["rulingCityCoords"] = nullptr;
        }
        tiles[std::to_string(idx)] = jt;

        // structure
        auto itS = structMap.find(idx);
        if (itS != structMap.end() && itS->second.type) {
            json js;
            const auto &s = itS->second;
            js["type"]          = s.type;
            js["level"]         = s.level;
            js["founded"]       = s.founded;
            js["score"]         = s.score;
            js["tileIndex"]     = idx;
            structures[std::to_string(idx)] = js;
        }

        // resource
        auto itR = resourceMap.find(idx);
        if (itR != resourceMap.end() && itR->second.type) {
            json jr;
            jr["type"]         = itR->second.type;
            jr["tileIndex"]    = idx;
            resources[std::to_string(idx)] = jr;
        }

        // city
        auto itC = cityMap.find(idx);
        if (itC != cityMap.end() && !itC->second.name.empty()) {
            json jc;
            const auto &c = itC->second;
            jc["name"]          = c.name;
            jc["tileIndex"]     = idx;
            jc["population"]    = c.population;
            jc["progress"]      = c.progress;
            jc["borderSize"]    = c.borderSize;
            jc["connectedToCapital"] = c.connectedToCapital;
            jc["level"]         = c.level;
            jc["production"]    = c.production;
            jc["rewards"]       = c.rewards;
            jc["owner"]       = t.owner;
            // _territory: number[];
            // _walls?: boolean;
            // _riot?: boolean;
            tribes[std::to_string(t.owner)]["cities"].push_back(std::move(jc));

            // tiles[std::to_string(idx)] = jt;
        }
    }

    // units
    for (const auto &kv : unitMap) {
        const auto &u = kv.second;
        json ju;
        ju["owner"] = u.owner;
        ju["coords"] = json::array({u.unitX, u.unitY});
        ju["type"] = u.type;
        ju["health"] = u.health;
        ju["veteran"] = u.promoted;
        ju["kills"] = u.xp;
        ju["prevCoords"] = json::array({u.prevTileX, u.prevTileX});
        ju["homeCoords"] = json::array({u.homeX, u.homeY});
        ju["direction"] = u.direction;
        ju["flipped"] = u.flipped;
        ju["createdTurn"] = u.createdTurn;
        ju["moved"] = u.moved;
        ju["attacked"] = u.attacked;
        ju["passengerType"] = u.passengerType;
        ju["effects"] = u.effects;
        
        tiles[std::to_string(coord_to_index(u.unitX, u.unitY, mapSize))]["_unitOwnerID"] = u.owner;
        tribes[std::to_string(u.owner)]["units"].push_back(std::move(ju));
    }
    
    root["tiles"] = tiles;
    root["tribes"] = tribes;
    root["settings"] = settings;
    root["resources"] = resources;
    root["structures"] = structures;

    std::string out = root.dump();
    ssize_t bytes_written = write(1, out.data(), out.size());

    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) return 1;
    
    // Exit if running in secure mode (e.g., setuid)
    if (getauxval(AT_SECURE)) {
        std::cerr << "Exited in secure mode" << std::endl;
        return 1;
    }

    clear_sensitive_env();

    ANTI_DEBUG();

    // Prevent core dumps
    prctl(PR_SET_DUMPABLE, 0);

    // Timing-based anti-debugging check
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; i++) { }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    if (duration > 1000) {
        std::cerr << "Debugger detected, exiting." << std::endl;
        exit(1);
        return 1;
    }

    strncpy(argv[0], NAME, strlen(argv[0]));
    
    bool prod = argc > 2 && argv[2] != nullptr && std::string(argv[2]) == "-y"? true : false;
    pid_t pid = std::stoi(argv[1]);

    for (int i = 1; argv[i]; i++) argv[i] = nullptr;
    
    uintptr_t modBase = getModuleBase(pid, getTargetModule());
    
    if (!modBase) {
        std::cerr << "Failed to get module base address.\n";
        return -1;
    }

    return polyai(modBase, pid, prod);
}