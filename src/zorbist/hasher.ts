import { getCityAt, getUnitAt } from "../core/functions";
import { StructureSettings } from "../core/settings/StructureSettings";
import { TechnologySettings } from "../core/settings/TechnologySettings";
import { UnitSettings } from "../core/settings/UnitSettings";
import { GameState, ResourceState, StructureState, UnitState } from "../core/states";
import { StructureType, TechnologyType, UnitType } from "../core/types";
import { ZobristKeys } from "./generateZorbist";

// These are required because some Types do not have a value assigned, 
// the types are used by the live game simulator so they must be normalized to an index

export const TechnologyToID: Record<TechnologyType, number> = Object.keys(TechnologySettings)
    .reduce((a, b, i) => ({ ...a, [Number(b)]: i }), { } as any);

export const StructureToID: Record<StructureType, number> = Object.keys(StructureSettings)
    .reduce((a, b, i) => ({ ...a, [Number(b)]: i }), { } as any);

export const UnitToID: Record<UnitType, number> = Object.keys(UnitSettings)
    .reduce((a, b, i) => ({ ...a, [Number(b)]: i }), { } as any);

export function getEnumID(value: number, map?: Record<number, number>): number {
    if (map) {
        const mappedValue = map[value];
        if (mappedValue === undefined) {
            console.warn(`Zobrist Hashing: Unmapped enum value ${value}. Ensure all enum values used in GameState are in your ID mappers.`);
            return -1; // Or throw error, or handle as per your design
        }
        return mappedValue;
    }
    return value; // Assume value is already a 0-based index
}
    
export function calculateInitialZobristHash(gameState: GameState, zobristKeys: ZobristKeys): bigint {
    const ownerId = gameState.settings._pov;

    let hash: bigint = 0n;

    if (gameState.settings._turn < zobristKeys.turn.length) {
        hash ^= zobristKeys.turn[gameState.settings._turn];
    } else {
        console.warn(`Zobrist: Turn ${gameState.settings._turn} > ${zobristKeys.turn.length} out of bounds for Zobrist keys.`);
    }

    if (ownerId < zobristKeys.pov.length) {
        hash ^= zobristKeys.pov[ownerId];
    } else {
        console.warn(`Zobrist: POV player ${ownerId} > ${zobristKeys.pov.length} out of bounds for Zobrist keys.`);
    }

    hash ^= zobristKeys.gameOver[gameState.settings._gameOver ? 1 : 0];

    for (let i = 0; i < gameState.tiles.length; i++) {
        const tileState = gameState.tiles[i];

        const tileKeys = zobristKeys.tiles[i];

        if (!tileState || !tileKeys) {
            console.warn(`Zobrist: Tile ${i} > ${gameState.tiles.length} out of bounds for Zobrist keys.`);
            continue;
        }

        // Skip unexplored tiles
        if(tileState._explorers.has(ownerId)) {
            continue;    
        }

        // Tile explored
        hash ^= tileKeys.explored[1];

        // Tile Owner
        if (tileState._owner) {
            hash ^= tileKeys.owner[tileState._owner];
        }

        // Tile Terrain
        const terrainId = getEnumID(tileState.terrainType);
        if (terrainId >= 0) {
            hash ^= tileKeys.terrainType[terrainId];
        }

        // Unit at tile
        const unitAt = getUnitAt(gameState, tileState.tileIndex);
        if(unitAt) {
            hash = xorUnit(hash, unitAt, zobristKeys);
        }

        // Structure on Tile
        const structureState = gameState.structures[tileState.tileIndex];
        if (structureState) {
            hash = xorStructure(hash, structureState, zobristKeys);
        }

        // Resource on Tile
        const resourceState = gameState.resources[tileState.tileIndex];
        if (resourceState) {
            xorResource(hash, resourceState, zobristKeys);
        }

        // Cities
        const city = getCityAt(gameState, tileState.tileIndex);
        if (city) {
            const cityKeys = zobristKeys.city[tileState.tileIndex];
    
            if (city._unitCount > 0) {
                if(city._unitCount < cityKeys.unitCount.length) {
                    hash ^= cityKeys.unitCount[city._unitCount];
                }
                else {
                    console.warn(`Zobrist: City with ${city._unitCount} > ${cityKeys.unitCount.length} units out of bounds for Zobrist keys.`);
                }
            }
            
            hash ^= cityKeys.riot[city._riot ? 1 : 0];
            hash ^= cityKeys.owner[city._owner];
        }
    }

    for(const key in gameState.tribes) {
        const tribe = gameState.tribes[key];

        const playerKeys = zobristKeys.player[tribe.owner];
        if (!playerKeys) {
            console.warn(`Zobrist: Player/Tribe (owner: ${tribe.owner} > ${zobristKeys.player.length} out of bounds for Zobrist keys.`);
            break;
        }

        // Tribe Type
        const tribeTypeId = getEnumID(tribe.tribeType);
        if (tribeTypeId < playerKeys.tribeType.length) {
            hash ^= playerKeys.tribeType[tribeTypeId];
        }
        else {
            console.warn(`Zobrist: Tribe Type ${tribeTypeId} > ${playerKeys.tribeType.length} out of bounds for Zobrist keys.`);
        }

        // Stars
        if (tribe._stars < playerKeys.stars.length) {
            hash ^= playerKeys.stars[tribe._stars];
        } else {
            hash ^= playerKeys.stars[playerKeys.stars.length -1];
            console.warn(`Zobrist: Player ${tribe._stars} stars ${tribe._stars} exceed max ${playerKeys.stars.length-1}. Hashing as max.`);
        }

        // Technologies
        tribe._tech.forEach(tech => {
            const techId = getEnumID(tech.techType, TechnologyToID);
            if(playerKeys.hasTech.length) {
                hash ^= playerKeys.hasTech[techId];
            }
            else {
                console.warn(`Zobrist: Tech ${techId} > ${playerKeys.hasTech.length} out of bounds for Zobrist keys.`);
            }
        });

        // Built Unique Structures
        if (tribe._builtUniqueStructures) {
            tribe._builtUniqueStructures.forEach(structType => {
                const structId = getEnumID(structType, StructureToID);
                if (structId < playerKeys.builtUniqueStructures.length) {
                    hash ^= playerKeys.builtUniqueStructures[structId];
                }
                else {
                    console.warn(`Zobrist: Built Unique Structure ${structId} > ${playerKeys.builtUniqueStructures.length} out of bounds for Zobrist keys.`);
                }
            });
        }
    }

    return hash;
}

export function xorUnit(
    hash: bigint,
    unit: UnitState,
    keys: ZobristKeys
): bigint {
    const unitKeys = keys.units[unit._tileIndex];

    hash ^= unitKeys.owner[unit._owner];
    hash ^= unitKeys.type[getEnumID(unit._unitType, UnitToID)];
    hash ^= unitKeys.veteran[unit.veteran ? 1 : 0];
    hash ^= unitKeys.moved[unit._moved ? 1 : 0];
    hash ^= unitKeys.attacked[unit._attacked ? 1 : 0];

    if (unit.kills >= 0 && unit.kills < unitKeys.kills.length) {
        hash ^= unitKeys.kills[unit.kills];
    }

    if (unit._passenger !== undefined) {
        const passengerTypeId = getEnumID(unit._passenger, UnitToID);
        if (passengerTypeId >= 0 && passengerTypeId < unitKeys.passenger.length) {
            hash ^= unitKeys.passenger[passengerTypeId];
        }
    }
    
    unit._effects.forEach(effect => {
        hash ^= unitKeys.effect[getEnumID(effect)];
    });

    return hash;
}

export function xorStructure(
    hash: bigint,
    struct: StructureState,
    keys: ZobristKeys
): bigint {
    hash ^= keys.structure[struct.tileIndex][getEnumID(struct.id, StructureToID)];
    return hash;
}

export function xorResource(
    hash: bigint,
    resource: ResourceState,
    keys: ZobristKeys
): bigint {
    hash ^= keys.resource[resource.tileIndex][getEnumID(resource.id)];
    return hash;
}