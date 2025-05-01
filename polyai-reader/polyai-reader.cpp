
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

uintptr_t getModuleBase(pid_t pid) {
    std::string mapsPath = "/proc/" + std::to_string(pid) + "/maps";
    std::ifstream maps(mapsPath);
    if (!maps.is_open()) return 0;

    std::string line;
    std::string modName = getTargetModule();
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
    
    uintptr_t turnBase = getPlace(pid, modBase + 0x02BE5F28, {0xB8, 0x10, 0xE0});
    uintptr_t mapBase = getPlace(pid, modBase + 0x02BD52B8, {0x20, 0xB8, 0x0, 0x38});
    uintptr_t unitsBase = getPlace(pid, modBase + 0x02C18258, {0xB8, 0x0, 0x40});
    uintptr_t tribesBase = getPlace(pid, modBase + 0x02BE3158, {0xB8, 0x0, 0x78, 0x38, 0x38, 0x10});

    if (!turnBase || !mapBase || !unitsBase || !tribesBase) return -1;

    uint16_t turn, tileCount, unitCount, tribeCount, mapSize;
    
    // ! STATE ! //

    readPiece(pid, turnBase, turn);

    // ! TRIBES ! //
    
    readPiece(pid, getPlace(pid, tribesBase, { 0x18 }), tribeCount);
    for (uint32_t index = 0; index < tribeCount; ++index) {
        uintptr_t tribeBase = getPlace(pid, tribesBase, { index * 0x8 + 0x20 });
        if (tribeBase == 0) break;

        uint16_t owner, tribeId, resignedTurn, killerId, killedTurn, currency, score, kills; 
        std::string tech, tasks, builtUniqueImprovements, knownPlayers, username, relations; 
        bool autoplay; 

        unsigned char tribeBuffer[172 + 4];
        if (readBlock(pid, getPlace(pid, tribeBase, { 0x10 }), tribeBuffer, sizeof(tribeBuffer))) {
            owner = *(uint16_t*)&tribeBuffer[0];         // 0x10
            autoplay = *(bool*)&tribeBuffer[36];         // 0x34
            tribeId = *(uint16_t*)&tribeBuffer[48];      // 0x40
            resignedTurn = *(uint16_t*)&tribeBuffer[64]; // 0x50
            currency = *(uint16_t*)&tribeBuffer[140];    // 0x9C
            score = *(uint16_t*)&tribeBuffer[144];       // 0xA0
            kills = *(uint16_t*)&tribeBuffer[156];       // 0xAC
            killerId = *(uint16_t*)&tribeBuffer[168];    // 0xB8
            killedTurn = *(uint16_t*)&tribeBuffer[172];  // 0xBC
        }

        username = readString(pid, getPlace(pid, tribeBase, { 0x18 }));
        readSingleList(pid, getPlace(pid, tribeBase, { 0x60 }), tech);
        readSingleList(pid, getPlace(pid, tribeBase, { 0x80 }), builtUniqueImprovements);

        // Tasks

        uint16_t tasksSize;
        if (readPiece(pid, getPlace(pid, tribeBase, { 0x68, 0x18 }), tasksSize) && tasksSize > 0) {
            uintptr_t taskBase = getPlace(pid, tribeBase, { 0x68, 0x10 });
            size_t maxTasks = std::min(static_cast<size_t>(tasksSize), size_t(5));
            for (uint32_t i = 0; i < maxTasks; i++) {
                unsigned char taskBuffer[8 + 4];
                if (readBlock(pid, getPlace(pid, taskBase, { 0x20 + i * 0x8, 0x10 }), taskBuffer, sizeof(taskBuffer))) {
                    uint8_t isStarted = taskBuffer[0];      // 0x10
                    if (isStarted != 1) continue;
                    uint8_t isCompleted = taskBuffer[1];    // 0x11
                    uint8_t turns = taskBuffer[8];          // 0x18
                    tasks += std::to_string(isStarted) + '-' + std::to_string(isCompleted);
                    if (turns < turn) {
                        tasks += '-' + std::to_string(turns);
                    }
                    tasks += "&";
                }
            }
            if (!tasks.empty()) tasks.pop_back();
        }
        
        // Known players

        uint16_t knownPlayerCount;
        if (readPiece(pid, getPlace(pid, tribeBase, {0x78, 0x18}), knownPlayerCount) && knownPlayerCount > 0) {
            size_t maxCount = std::min(static_cast<size_t>(knownPlayerCount), size_t(16));
            size_t bufferSize = maxCount;
            std::vector<unsigned char> playersBuffer(bufferSize);
            uintptr_t playersBase = getPlace(pid, tribeBase, { 0x78, 0x10, 0x20 });
            if (playersBase && readBlock(pid, playersBase, playersBuffer.data(), bufferSize)) {
                for (uint32_t i = 0; i < maxCount; i++) {
                    uint8_t byteValue = playersBuffer[i];
                    knownPlayers += std::to_string(byteValue) + "&";
                }
                knownPlayers.pop_back();
            }
        }

        // Relations (dictionary)

        uintptr_t relationsBase = getPlace(pid, tribeBase, { 0x88, 0x18 });
        for (uint32_t i = 0; i < tribeCount; i++) {
            uint16_t keyOwner = 0;
            readBlock(pid, getPlace(pid, relationsBase, { 0x20 + i * 0x18 }), &keyOwner, 1);
            unsigned char relationsBuffer[24 + 4];
            if (readBlock(pid, getPlace(pid, relationsBase, { 0x30 + i * 0x18, 0x10 }), relationsBuffer, sizeof(relationsBuffer))) {
                relations += std::to_string(keyOwner) + '_';
                relations += std::to_string(*((bool*)&relationsBuffer[0]))      // 0x10 (state)
                      + '-' + std::to_string(*(uint16_t*)&relationsBuffer[4])   // 0x14 (lastAttackTurn)
                      + '-' + std::to_string(*(uint16_t*)&relationsBuffer[8])   // 0x18 (embassyLevel)
                      + '-' + std::to_string(*(uint16_t*)&relationsBuffer[12])  // 0x1C (lastPeaceBrokenTurn)
                      + '-' + std::to_string(*(uint16_t*)&relationsBuffer[16])  // 0x20 (firstMeet)
                      + '-' + std::to_string(*(uint16_t*)&relationsBuffer[20])  // 0x24 (embassyBuildTurn)
                      + '-' + std::to_string(*(uint16_t*)&relationsBuffer[24]); // 0x28 (previousAttackTurn)
                relations += "&";
            }
        }
        if (!relations.empty()) relations.pop_back();

        tribesMap[owner] = { 
            owner, username, currency, score, autoplay, tech, tribeId, 
            killerId, kills, tasks, builtUniqueImprovements, knownPlayers, 
            relations, killedTurn, resignedTurn
        };
    }
    
    // ! MAP ! //
    
    readPiece(pid, getPlace(pid, mapBase, { 0x18 }), tileCount);
    mapSize = static_cast<uint16_t>(std::sqrt(tileCount));
    for (uint32_t index = 0; index < tileCount; ++index) {
        uintptr_t tileBase = getPlace(pid, mapBase, { index * 0x8 + 0x20, 0x90 });

        if (tileBase == 0) break;

        uint16_t tileId, tileX, tileY, rulingCityX, rulingCityY, skinType, climate, tribeIndex, capitalOf; 
        uint8_t byteTribeIndex, byteCapitalOf;
        bool hasRoad, hasRoute, hadRoute; 
        std::string explorers;

        // Tile data

        unsigned char tileBuffer[96 + 4]; // 0x10 to 0x70
        if (readBlock(pid, getPlace(pid, tileBase, { 0x10 }), tileBuffer, sizeof(tileBuffer))) {
            tileX = *(uint16_t*)&tileBuffer[0];     // 0x10
            tileY = *(uint16_t*)&tileBuffer[2];     // 0x12
            tileId = *(uint16_t*)&tileBuffer[8];    // 0x18
            climate = *(uint16_t*)&tileBuffer[12];  // 0x1C
            skinType = *(uint16_t*)&tileBuffer[16]; // 0x20
            byteTribeIndex = tileBuffer[24];        // 0x28
            byteCapitalOf = tileBuffer[25];         // 0x29
            rulingCityX = *(uint16_t*)&tileBuffer[44]; // 0x3C
            rulingCityY = *(uint16_t*)&tileBuffer[48]; // 0x40
            tribeIndex = byteTribeIndex;
            capitalOf = byteCapitalOf;
            hasRoad = tileBuffer[80];               // 0x60
            hasRoute = tileBuffer[81];              // 0x61
            hadRoute = tileBuffer[96];              // 0x70
        }

        // Explorers

        uint16_t explorersCount;
        if (readPiece(pid, getPlace(pid, tileBase, { 0x30, 0x18 }), explorersCount) && explorersCount > 0) {
            size_t maxCount = std::min(static_cast<size_t>(explorersCount), static_cast<size_t>(tribeCount));
            std::vector<unsigned char> explorersBuffer(maxCount);

            uintptr_t explorersBase = getPlace(pid, tileBase, { 0x30, 0x10, 0x20 });
            
            if (explorersBase && readBlock(pid, explorersBase, explorersBuffer.data(), maxCount)) {
                for (uint32_t j = 0; j < maxCount; j++) {
                    uint8_t byteValue = explorersBuffer[j];
                    explorers += std::to_string(byteValue) + "&";
                }
                explorers.pop_back();
            }
        }
                
        // Structures
        
        uintptr_t structBase = getPlace(pid, tileBase, { 0x48 });
        uint16_t structureId = 0;
        
        if(readPiece(pid, getPlace(pid, structBase, { 0x10 }), structureId) && structureId > 0) {
            uint16_t founded, reward, level;
            uint16_t population, production, borderSize, progress; 
            uint64_t levelRaw;
            bool connectedToCapital;
            
            unsigned char structureBuffer[18 + 4];
            if (readBlock(pid, getPlace(pid, structBase, { 0x14 }), structureBuffer, sizeof(structureBuffer))) {
                levelRaw = *(uint64_t*)&structureBuffer[0];            // 0x14
                level = *(uint16_t*)&structureBuffer[2];               // 0x16
                founded = *(uint16_t*)&structureBuffer[4];             // 0x18
                population = *(uint16_t*)&structureBuffer[8];          // 0x1C
                production = *(uint16_t*)&structureBuffer[10];         // 0x1E
                reward = *(uint16_t*)&structureBuffer[12];             // 0x20
                borderSize = *(uint16_t*)&structureBuffer[14];         // 0x22
                connectedToCapital = *(uint16_t*)&structureBuffer[18]; // 0x26
            }

            // If structure is a city and is owned
            std::string name = readString(pid, getPlace(pid, structBase, { 0x28 }));
            if (name != "") {
                level = ((levelRaw & 0xFFFFFFFF) >> 16);
                progress = (levelRaw >> 32) / 65536; 
                std::string rewards;
                readSingleList(pid, getPlace(pid, structBase, { 0x30 }), rewards);
                cityMap[index] = { name, population, progress, rewards, production, borderSize, connectedToCapital, level };
            }
            
            structMap[index] = { structureId, level, founded, reward };
        }
        
        // Resources

        uint16_t resourceId = 0;
        if(readPiece(pid, getPlace(pid, tileBase, { 0x50, 0x10 }), resourceId) && resourceId > 0) {
            resourceMap[index] = { resourceId };
        }

        tileMap[index] = { 
            index, tileId, tribeIndex, explorers, hasRoad, hasRoute, hadRoute, 
            capitalOf, rulingCityX, rulingCityY, skinType, climate,
            tileX, tileY,
        };
    }
    
    // ! UNITS ! //
    
    readPiece(pid, getPlace(pid, unitsBase, { 0x18 }), unitCount);
    for (uint32_t index = 0; index < unitCount; ++index) {
        uintptr_t unitBase = getPlace(pid, unitsBase, { 0x10, index * 0x8 + 0x20, 0x40 });
        if (unitBase == 0) break;

        bool promoted, moved, attacked, flipped; 
        uint16_t owner, type, tileX, tileY, hp, kills, prevTileX, prevTileY, homeX, homeY, direction, createdTurn;
        uint16_t classId, classHp, classCost, classDef, classMov, classAtk, classWpn, classRange;
        // bool classHidden;

        unsigned char unitBuffer[58 + 4];
        if (readBlock(pid, getPlace(pid, unitBase, { 0x1C }), unitBuffer, sizeof(unitBuffer))) {
            owner = *(uint16_t*)&unitBuffer[0];        // 0x1C
            type = *(uint16_t*)&unitBuffer[8];         // 0x24
            prevTileX = *(uint16_t*)&unitBuffer[12];   // 0x28
            prevTileY = *(uint16_t*)&unitBuffer[16];   // 0x2C
            tileX = *(uint16_t*)&unitBuffer[20];       // 0x30
            tileY = *(uint16_t*)&unitBuffer[24];       // 0x34
            homeX = *(uint16_t*)&unitBuffer[28];       // 0x38
            homeY = *(uint16_t*)&unitBuffer[32];       // 0x3C
            hp = *(uint16_t*)&unitBuffer[44];          // 0x48
            promoted = *(bool*)&unitBuffer[46];        // 0x4A
            kills = *(uint16_t*)&unitBuffer[48];       // 0x4C
            moved = *(bool*)&unitBuffer[50];           // 0x4E
            attacked = *(bool*)&unitBuffer[51];        // 0x4F
            direction = *(uint16_t*)&unitBuffer[52];   // 0x50
            flipped = *(bool*)&unitBuffer[56];         // 0x54
            createdTurn = *(uint16_t*)&unitBuffer[58]; // 0x56
        }

        // readPiece(pid, getPlace(pid, unitBase, { 0x58, 0x14 }), classHidden);

        std::string effects; 
        readSingleList(pid, getPlace(pid, unitBase, { 0x60 }), effects);

        uint16_t passengerId;
        if(getPlace(pid, unitBase, { 0x40, 0x24 }) != 0) {
            readPiece(pid, getPlace(pid, unitBase, { 0x40, 0x24 }), passengerId);
        }

        unitMap[index] = { 
            owner, tileX, tileY, type, hp, promoted, kills, 
            prevTileX, prevTileY, homeX, homeY, direction,
            flipped, createdTurn, moved, attacked, passengerId,
            effects
        };
    }
    
    // ! WRITE OUT ! //

    if(!prod) {
        std::cout << "Turn: " << turn 
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
            p.index, p.username, p.bot, p.score, p.stars, 
            p.tech, p.tribeId, p.killerId, p.kills, p.tasks, 
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
            appendFields(out, ',', s.structureId, s.structureLevel, s.structureTurn, s.structureReward);
        }
        
        out << ";";

        const auto& r = resourceMap[t.index];
        if(r.resourceId) {
            out << r.resourceId;
        }
        
        out << ";";

        const auto& u = unitMap[t.index];
        if(u.unitId) { 
            appendFields(out, ',', u.owner, u.unitX, u.unitY, u.unitId, u.unitHp, u.unitIsVeteran,
                u.unitKills, u.prevTileX, u.prevTileY, u.homeX, u.homeY, u.direction,
                u.flipped, u.createdTurn, u.moved, u.attacked, u.passengerId, u.unitEffects);
        }
        
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
        // std::cerr << "Debugger detected, exiting." << std::endl;
        exit(1);
        return 1;
    }

    strncpy(argv[0], NAME, strlen(argv[0]));
    
    bool prod = argc > 2 && argv[2] != nullptr && std::string(argv[2]) == "-y"? true : false;
    pid_t pid = std::stoi(argv[1]);

    for (int i = 1; argv[i]; i++) argv[i] = nullptr;
    
    uintptr_t modBase = getModuleBase(pid);
    
    if (!modBase) return -1;

    return polyai(modBase, pid, prod);
}