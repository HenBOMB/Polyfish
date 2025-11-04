
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

#define NAME "polyai-reader"

// Anti-debugging macro
#define ANTI_DEBUG() if (ptrace(PTRACE_TRACEME, 0, nullptr, nullptr) == -1) { \
    std::cerr << "Debugger detected, exiting." << std::endl; exit(1); }



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
    std::unordered_map<uint16_t, ResourceInfo> resourceMap;
    std::unordered_map<uint16_t, CityInfo> cityMap;
    std::unordered_map<uint16_t, UnitInfo> unitMap;
    std::unordered_map<uint16_t, PlayerState> tribesMap;
    
    /**
     * BotDifficulty / BaseGameMode / UnitEffect
     *   0x10: Int32, value (enum)
     * 
     *   other interesting ones like: current viewing player, curlocalplaerindex
     * 
     * 
     *  
     * MapData
     *   0x10: UInt16, width
     *   0x12: UInt16, height
     *   0x18: TileData[] tiles
     *   0x20: WorldContinent[] continents
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
     * 
     * GameState
     *   0x10: Int32 version
     *   0x14: Int32 seed
     *   0x18: UInt32 currentTurn (<--- turnBase[0] !!!)
     *   0x1C: Byte currentPlayerIndex
     *   0x20: UInt32 currentUnitId
     *   0x24: GameState currentState
     *   0x28: GameSettings settings
     *   0x30: MapData map (<--- mapBase[1] !!!)
     *   0x38: List<PlayerState> playerStates (<--- tribesBase[1] !!!)
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
     *   0x20: GameSettings, settings
     *   0x28: ClientBase, client
     *   0x50: Int32, aiOpponents
     *   0x54: Int32, playerOpponents
     *   0x64: TribeType, startingTribe
     * 
     *   other interesting ones like: ladderManager, tornamentManager, lobbyMana, replaysMana
     * 
     * GameSettings
     *   0x10: BotDifficulty, difficulty
     *   0x14: BaseGameMode, baseGameMode
     *   0x18: BaseGameMode, rulesGameMode
     *   0x30: List<TribeType>, selectedSkins
     *   0x38: List<PlayerData>, players
     *   0x40: List<PlayerData>, spectators
     *   0x48: MapPreset, mapPreset
     *   0x50: GameRules, rules
     *   0x58: String, gameName
     *   0x60: GameType, gameType
     *   0x64: Int32, mapSize
     *   0x68: Int32, timeLimit
     *   ...
     *   0x80: Int32, opponentcount
     * 
     * TerrainData
     *   0x10: Int32, idx
     * 
     * WorldCoords
     *   0x10: Int32, x
     *   0x14: Int32, y
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

    uintptr_t gameManager = getPlace(pid, modBase + 0x3674378, { 0xB8, 0x0 });
    // std::cout << "Game manager address: " << std::hex << gameManager << std::endl;
    uintptr_t playersBase = getPlace(pid, gameManager, {_0x_GAMEMANAGER_CLIENT, _0x_CLIENT_CUR_STATE, _0x_STATE_PLAYERS, _0x_IN_LIST});
    uintptr_t currentTurnBase = getPlace(pid, gameManager, {_0x_GAMEMANAGER_CLIENT, _0x_CLIENT_CUR_STATE, _0x_STATE_CUR_TURN});
    // TODO units are now handled per tile, using TileState -> UnitState
    // uintptr_t unitsBase = getPlace(pid, gameManager, {0xB8, 0x0, 0x40});
    uintptr_t mapBase = getPlace(pid, gameManager, {_0x_GAMEMANAGER_CLIENT, _0x_CLIENT_CUR_STATE, _0x_STATE_MAP, _0x_MAP_TILES});

    if (!gameManager || !currentTurnBase || !mapBase || !playersBase) {
        if (!gameManager) {
            std::cerr << "Failed to get game manager address\n";
        }
        if (!currentTurnBase) {
            std::cerr << "Failed to get current turn address\n";
        }
        if (!mapBase) {
            std::cerr << "Failed to get map address\n";
        }
        if (!playersBase) {
            std::cerr << "Failed to get players address\n";
        }
        return -1;
    }

    // ! SETTINGS ! //

    uint16_t turn;

    readPiece(pid, currentTurnBase, turn);

    // ! TRIBES ! //
    
    uint16_t tribeCount = 0;
    
    readPiece(pid, getPlace(pid, playersBase, { _0x_IN_LIST_COUNT }), tribeCount);

    // The last one is always "Nature"
    tribeCount -= 1;

    // TODO Fix
    for (uint32_t index = 0; index < tribeCount; ++index) {
        uintptr_t playerRoot = getPlace(pid, playersBase, { index * 0x8 + _0x_IN_LIST_START_SHIFT });
        uintptr_t playerBase = getPlace(pid, playerRoot, { 0x0 });
        unsigned char playerBuffer[PlayerStateOffsets::BUFF_SIZE]; 

        if (playerRoot == 0 || playerBase == 0 || !readBlock(pid, playerBase, playerBuffer, sizeof(playerBuffer))) {
            break;
        }
        
        int32_t currency, resignedTurn, killedTurn, score, kills;
        int16_t startingTileX, startingTileY, tribeType;
        uint8_t id, killerId;
        std::string tech, tasks, builtUniqueImprovements, knownPlayers, username, relations; 
        bool autoplay; 
        
        // ! PlayerState
        /*
        *   0x10: Byte id
        *   0x18: String username
        *   0x20: Guid accountId
        *   0x34: Bool autoplay
        *   0x38: WorldCoordinates, startTile
        *   0x40: TribeType tribe
        *   0x44: TribeType tribeMix
        *   ...
        *   0x50: Int32 resignedTurn
        *   0x50: Int32 resignedAtCommandIndex
        *   0x58: Int32 wipedAtCommandIndex
        *   0x60: List<TechData.Type> availableTech
        *   0x68: List<TaskBase> tasks
        *   0x70: Dictionary<Byte,Int32> aggressions
        *   0x78: List<Byte> knownPlayers
        *   0x80: List<ImprovementData> builtUniqueImprovements
        *   0x88: Dictionary<Byte,DiplomacyRelation> relations
        *   0x90: List<Diplomacymessage> messages
        *   0x98: SkinType skinType
        *   0x9C: Int32 currency
        *   0xA0: Int32 score
        *   0xA4: Int32 endScore
        *   0xA8: Int32 cities
        *   0xAC: Int32 kills
        *   0xB0: Int32 casualties
        *   0xB4: Int32 wipeOuts
        *   0xB8: Byte killerId
        *   0xBC: Int32 killedTurn
        *   0xC0: Int32 color
        *   0xC8: AIState aiState
        */

        // ! DiplomacyRelation
        /* 
         *  0x10: DiplomacyRelationState state
         *  0x14: Int32 lastAttackTurn
         *  0x18: Int32 embassyLevel
         *  0x1C: Int32 lastPeaceBrokenTurn
         *  0x20: Int32 firstMeet
         *  0x24: Int32 embassyBuildTurn
         *  0x28: Int32 previousAttackTurn
         *
         * DiplomacyRelationState
         *  0x10: Int32 value
         */

        uintptr_t usernameRoot = getPlace(pid, *(uintptr_t*)&playerBuffer[PlayerStateOffsets::USERNAME], {  });
        readWord(pid, usernameRoot, username);
        // std::cout << "[player]: " << username << std::endl;
        // std::cout << "[address]: 0x" << std::hex << usernameRoot << std::endl << std::dec;
        id            = *(uint8_t*)&playerBuffer[PlayerStateOffsets::OWNER];
        autoplay      = *(bool*)&playerBuffer[PlayerStateOffsets::AUTOPLAY];
        startingTileX = *(int16_t*)&playerBuffer[PlayerStateOffsets::START_TILE_X];
        startingTileY = *(int16_t*)&playerBuffer[PlayerStateOffsets::START_TILE_Y];
        tribeType     = *(int16_t*)&playerBuffer[PlayerStateOffsets::TRIBE_TYPE];
        resignedTurn  = *(int32_t*)&playerBuffer[PlayerStateOffsets::RESIGNED_TURN];
        readSingleList(pid, getPlace(pid, playerRoot, { PlayerStateOffsets::AVAILABLE_TECH }), tech);
        // TODO tasks
        // TODO aggressions
        readSingleList(pid, getPlace(pid, playerRoot, { PlayerStateOffsets::KNOWN_PLAYERS }), knownPlayers);
        // std::cout << "[address]: 0x" << std::hex << getPlace(pid, playerRoot, { PlayerStateOffsets::RELATIONS }) << std::endl << std::dec;
        readDictionary(
            pid, 
            getPlace(pid, playerRoot, { PlayerStateOffsets::RELATIONS }), 
            relations, 
            [](uint16_t key, unsigned char *buffer) -> std::string {
                int32_t state               = *(int32_t*)&buffer[0x10];
                int32_t lastAttackTurn      = *(int32_t*)&buffer[0x14];
                int32_t embassyLevel        = *(int32_t*)&buffer[0x18];
                int32_t lastPeaceBrokenTurn = *(int32_t*)&buffer[0x1C];
                int32_t firstMeet           = *(int32_t*)&buffer[0x20];
                int32_t embassyBuildTurn    = *(int32_t*)&buffer[0x24];
                int32_t previousAttackTurn  = *(int32_t*)&buffer[0x28];
                // std::cout << "[" << key << "]: " << state << ", " << lastAttackTurn << ", " << embassyLevel << ", " << lastPeaceBrokenTurn << ", " << firstMeet << ", " << embassyBuildTurn << ", " << previousAttackTurn << std::endl;
                return std::to_string(key) 
                    + '_' + std::to_string(state)
                    + '-' + std::to_string(lastAttackTurn)
                    + '-' + std::to_string(embassyLevel)
                    + '-' + std::to_string(lastPeaceBrokenTurn)
                    + '-' + std::to_string(firstMeet)
                    + '-' + std::to_string(embassyBuildTurn)
                    + '-' + std::to_string(previousAttackTurn);;
            },
            0x28
        );
        // std::cout << std::endl;
        // std::cout << "[relations]: " << relations << std::endl;

        // TODO messages
        // TODO skinType
        currency      = *(int32_t*)&playerBuffer[PlayerStateOffsets::CURRENCY];
        score         = *(int32_t*)&playerBuffer[PlayerStateOffsets::SCORE];
        kills         = *(int32_t*)&playerBuffer[PlayerStateOffsets::KILLS];
        // TODO casualities
        // TODO wipeOuts
        killerId      = *(uint8_t*)&playerBuffer[PlayerStateOffsets::KILLER_ID];
        killedTurn    = *(int32_t*)&playerBuffer[PlayerStateOffsets::KILLED_TURN];

        readSingleList(pid, getPlace(pid, playerRoot, { PlayerStateOffsets::BUILT_UNIQUE_IMPROVEMENTS }), builtUniqueImprovements);

        // TODO apply new startingTileXY to output

        tribesMap[index] = { 
            id, username, currency, score, autoplay, tech, tribeType, 
            killerId, kills, tasks, builtUniqueImprovements, knownPlayers, 
            relations, killedTurn, resignedTurn, startingTileX, startingTileY
        };
    }
    
    // ! MAP ! //
    
    uint16_t tileCount, unitCount = 0;
    
    readPiece(pid, getPlace(pid, mapBase, { _0x_IN_LIST_COUNT }), tileCount);

    uint16_t mapSize = static_cast<uint16_t>(std::sqrt(tileCount));

    for (uint32_t index = 0; index < tileCount; ++index) {
        uintptr_t tileRoot = getPlace(pid, mapBase, { index * 0x8 + _0x_IN_LIST_START_SHIFT });
        uintptr_t tileBase = getPlace(pid, tileRoot, { 0x0 });
        unsigned char tileBuffer[_0x_TILE_HAD_ROUTE + _0x_TRAILING_OFFSET]; 

        if (tileRoot == 0 || tileBase == 0 || !readBlock(pid, tileBase, tileBuffer, sizeof(tileBuffer))) {
            break;
        }

        uint16_t terrainType, tileX, tileY, rulingCityX, rulingCityY, 
            skinType, climateType, owner, capitalOf, explorersCount;
        uint8_t byteTribeIndex, byteCapitalOf;
        bool hasRoad, hasRoute, hadRoute; 
        std::string explorers;
        uintptr_t unitRoot, resourceRoot, improvementRoot;  
        
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

        tileX           = *(uint16_t*)&tileBuffer[_0x_TILE_X];
        tileY           = *(uint16_t*)&tileBuffer[_0x_TILE_Y];
        terrainType     = *(uint16_t*)&tileBuffer[_0x_TILE_TERRAIN_TYPE];
        climateType     = *(uint16_t*)&tileBuffer[_0x_TILE_CLIMATE_TYPE]; 
        skinType        = *(uint16_t*)&tileBuffer[_0x_TILE_SKIN_TYPE];    
        owner           = *(uint8_t*)&tileBuffer[_0x_TILE_OWNER];
        capitalOf       = *(uint8_t*)&tileBuffer[_0x_TILE_CAPITAL_OF];
        rulingCityX     = *(uint16_t*)&tileBuffer[_0x_TILE_RULING_CITY_X];
        rulingCityY     = *(uint16_t*)&tileBuffer[_0x_TILE_RULING_CITY_Y];
        improvementRoot = *(uintptr_t*)&tileBuffer[_0x_TILE_IMPROVEMENT];
        resourceRoot    = *(uintptr_t*)&tileBuffer[_0x_TILE_RESOURCE];
        unitRoot        = *(uintptr_t*)&tileBuffer[_0x_TILE_UNIT];
        hasRoad         = *(bool*)&tileBuffer[_0x_TILE_HAS_ROAD];                  
        hasRoute        = *(bool*)&tileBuffer[_0x_TILE_HAS_ROUTE];                 
        hadRoute        = *(bool*)&tileBuffer[_0x_TILE_HAD_ROUTE];  
        readSingleList(pid, getPlace(pid, tileRoot, { _0x_TILE_EXPLORERS }), explorers);

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
        if (unitRoot != 0 && readBlock(pid, unitRoot, unitBuffer, sizeof(unitBuffer))) {
            unitCount += 1;
            // std::cout << "[-] " << std::hex << unitBase << std::endl << std::dec;
            // std::cout << "~unit detected~ 0x" << std::hex << unitRoot << std::endl << std::dec;
            
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

            std::string effects; 
            bool moved, attacked, flipped; 
            // uint8_t owner;
            uint16_t promoted, owner, type, tileX, tileY, hp, kills, prevTileX, prevTileY, 
                homeX, homeY, direction, createdTurn;
            // uint16_t classId, classHp, classCost, classDef, classMov, classAtk, classWpn, classRange;
            // bool classHidden;
            
            // ... id
            owner       = *(uint16_t*)&unitBuffer[_0x_UNIT_OWNER];
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
            kills       = *(uint16_t*)&unitBuffer[_0x_UNIT_XP];
            direction   = *(uint16_t*)&unitBuffer[_0x_UNIT_DIRECTION];
            createdTurn = *(uint16_t*)&unitBuffer[_0x_UNIT_CREATED_TURN];
            moved       = *(bool*)&unitBuffer[_0x_UNIT_MOVED];
            attacked    = *(bool*)&unitBuffer[_0x_UNIT_ATTACKED];
            flipped     = *(bool*)&unitBuffer[_0x_UNIT_FLIPPED];

            // std::cout << "[pos] " << tileX << "," << *(int32_t*)&unitBuffer[_0x_UNIT_Y] << std::endl;
            // std::cout << "[home] " << homeX << "," << homeY << std::endl;
            // std::cout << "[hp] " << hp << std::endl;
            // std::cout << [status] " << moved << ", " << attacked << std::endl;

            readSingleList(pid, getPlace(pid, unitRoot, { _0x_UNIT_EFFECTS }), effects);

            uint16_t passengerId;
            uintptr_t passengerBase = getPlace(pid, unitRoot, { _0x_UNIT_PASSENGER_UNIT, _0x_UNIT_TYPE });  
            if(passengerBase != 0) {
                readPiece(pid, passengerBase, passengerId);
            }

            unitMap[index] = { 
                owner, tileX, tileY, type, hp, promoted, kills, 
                prevTileX, prevTileY, homeX, homeY, direction,
                flipped, createdTurn, moved, attacked, passengerId,
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

        unsigned char structureBuffer[_0x_IMPROVEMENT_EFFECTS + _0x_TRAILING_OFFSET]; 
        if (improvementRoot != 0 && readPiece(pid, improvementRoot, structureBuffer)) {
            int16_t type, level, founded, progress, population;
            uint16_t production, baseScore, borderSize, upgrade;
            uint8_t owner, founder;
            bool connectedToCapitalOfPlayer;
            std::string name, rewards, effects;

            type        = *(int16_t*)&structureBuffer[_0x_IMPROVEMENT_TYPE];
            level       = *(int16_t*)&structureBuffer[_0x_IMPROVEMENT_LEVEL];
            founded     = *(int16_t*)&structureBuffer[_0x_IMPROVEMENT_FOUNDED];
            progress    = *(int16_t*)&structureBuffer[_0x_IMPROVEMENT_XP];
            population  = *(int16_t*)&structureBuffer[_0x_IMPROVEMENT_POPULATION];
            production  = *(uint16_t*)&structureBuffer[_0x_IMPROVEMENT_PRODUCTION];
            baseScore   = *(uint16_t*)&structureBuffer[_0x_IMPROVEMENT_BASE_SCORE];
            borderSize  = *(uint16_t*)&structureBuffer[_0x_IMPROVEMENT_BORDER_SIZE];
            upgrade     = *(uint16_t*)&structureBuffer[_0x_IMPROVEMENT_UPGRADE];
            owner       = *(uint8_t*)&structureBuffer[_0x_IMPROVEMENT_OWNER];
            founder     = *(uint8_t*)&structureBuffer[_0x_IMPROVEMENT_FOUNDER];
            // TODO verify == 1
            connectedToCapitalOfPlayer = *(uint8_t*)&structureBuffer[_0x_IMPROVEMENT_CONNECTED_TO_CAPITAL] == 1;

            uintptr_t nameRoot = getPlace(pid, *(uintptr_t*)&structureBuffer[_0x_IMPROVEMENT_NAME], {  });

            if (readWord(pid, nameRoot, name)) {
                // std::cout << "[city] 0x" << std::hex << improvementRoot << std::endl << std::dec;
                // std::cout << "name: " << name << std::endl;
                // std::cout << "pop: " << population << std::endl;
                // std::cout << "prg: " << progress << std::endl;
                // std::cout << "pro: " << production << std::endl;
                // std::cout << "lvl: " << level << std::endl;
                cityMap[index] = { 
                    name, population, progress, rewards, production,
                    borderSize, connectedToCapitalOfPlayer, level 
                };
            }

            readSingleList(pid, *(uintptr_t*)&structureBuffer[_0x_IMPROVEMENT_REWARDS], rewards);
            
            structMap[index] = { type, level, founded, baseScore };
        }

        // ! ResourceState
        /*
         *   0x10: ResourceType type
         */
        
        int16_t resourceId = 0;
        if(resourceRoot != 0 && readPiece(pid, resourceRoot + 0x10, resourceId)) {
        // if(resourceRoot != 0 && readPiece(pid, getPlace(pid, tileRoot, {_0x_TILE_RESOURCE, 0x10}), resourceId)) {
        // if(resourceRoot != 0 && readPiece(pid, getPlace(pid, tileBase + _0x_TILE_RESOURCE, {0x10}), resourceId)) {
            // std::cout << "[resource] 0x" << std::hex << resourceRoot << std::endl << std::dec;
            // std::cout << "resourceId: " << resourceId << std::endl;
            resourceMap[index] = { resourceId };
        }

        tileMap[index] = { 
            index, terrainType, owner, explorers, hasRoad, hasRoute, hadRoute, 
            capitalOf, rulingCityX, rulingCityY, skinType, climateType,
            tileX, tileY,
        };
    }
   
    // ! WRITE OUT ! //

    if(!prod) {
        std::cout << std::dec << "Turn: " << turn 
            << " | Map size: " << mapSize << "x" << mapSize << " (" << tileCount << ")" 
            << " | Units: " << unitCount 
            << " | Tribes: " << tribeCount 
            << std::endl;
        return 0;
    }
    
    std::ostringstream out;
    appendFields(out, ',', mapSize, turn);

    out << "\n";
    
    for (const auto& [i, p] : tribesMap) {
        appendFields(out, ',', 
            p.id, p.username, p.autoplay, p.score, p.currency, 
            p.tech, p.tribeType, p.killerId, p.kills, p.tasks, 
            p.builtUniqueImprovements, p.knownPlayers, p.relations,
            p.killedTurn, p.resignedTurn);
        out << ";";
    }
    
    out << "\n";
    
    for (const auto& [index, t] : tileMap) {
        out << t.index << ";";
        
        appendFields(out, ',', t.tileId, t.owner, t.explorers, t.hasRoad, t.hasRoute, t.hadRoute,
            t.capitalOf, t.rulingCityX, t.rulingCityY, t.climate, t.skinType, t.tileX, t.tileY);
        
        out << ";";

        const auto& s = structMap[t.index];
        if(s.structureId) {
            appendFields(out, ',', s.structureId, s.structureLevel, s.structureFounded, s.structureBaseScore);
        }
        
        out << ";";

        const auto& r = resourceMap[t.index];
        if(r.resourceId) {
            out << r.resourceId;
        }
        
        // out << ";";

        // const auto& u = unitMap[t.index];
        // if(u.unitId) { 
        //     appendFields(out, ',', u.owner, u.unitX, u.unitY, u.unitId, u.unitHp, u.unitIsVeteran,
        //         u.unitKills, u.prevTileX, u.prevTileY, u.homeX, u.homeY, u.direction,
        //         u.flipped, u.createdTurn, u.moved, u.attacked, u.passengerId, u.unitEffects);
        // }
        
        out << ";";

        const auto& c = cityMap[t.index];
        if(c.name.size()) {
            appendFields(out, ',', c.name, c.population, c.progress, c.rewards, c.production,
                c.borderSize, c.connectedToCapital, c.level);
        }
        
        out << "+";
    }
    
    ssize_t bytes_written = write(1, out.str().c_str(), out.str().size());

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