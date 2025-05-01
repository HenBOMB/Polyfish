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
    uint16_t structureId;
    uint16_t structureLevel;
    uint16_t structureTurn;
    uint16_t structureReward;
};

struct ResourceInfo {
    uint16_t resourceId;
};

struct UnitInfo {
    uint16_t owner;
    uint16_t unitX;
    uint16_t unitY;
    uint16_t unitId;
    uint16_t unitHp;
    bool unitIsVeteran;
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
    uint16_t population;
    uint16_t progress;
    std::string rewards;
    uint16_t production;
    uint16_t borderSize;
    bool connectedToCapital;
    uint16_t level;
};

struct PlayerState {
    uint16_t index;
    std::string username;
    uint16_t stars;
    uint16_t score;
    bool bot;
    std::string tech;
    uint16_t tribeId;
    uint16_t killerId;
    uint16_t kills;
    std::string tasks; 
    std::string builtUniqueImprovements;
    std::string knownPlayers;
    std::string relations;
    uint16_t killedTurn;
    uint16_t resignedTurn;
};

#endif
