import { log } from "node:console";
import { MODEL_CONFIG } from "../aistate";
import { getCityAt, getPovTribe, getResourceAt, getStructureAt, getTrueUnitAt, getUnitAt } from "../core/functions";
import { StructureSettings } from "../core/settings/StructureSettings";
import { TechnologySettings } from "../core/settings/TechnologySettings";
import { UnitSettings } from "../core/settings/UnitSettings";
import { CityState, GameState, TileState, TribeState, UnitState } from "../core/states";
import { EffectType, ResourceType, StructureType, TechnologyType, TerrainType, TribeType, UnitType } from "../core/types";
import { zobristKeys } from "./zobristKeys";

// These are required because some Types do not have a value assigned, 
// the types are used by the live game simulator so they must be normalized to an index

const TechnologyToID: Record<TechnologyType, number> = Object.keys(TechnologySettings)
    .reduce((a, b, i) => ({ ...a, [Number(b)]: i }), { } as any);

const StructureToID: Record<StructureType, number> = Object.keys(StructureSettings)
    .reduce((a, b, i) => ({ ...a, [Number(b)]: i }), { } as any);

const UnitToID: Record<UnitType, number> = Object.keys(UnitSettings)
    .reduce((a, b, i) => ({ ...a, [Number(b)]: i }), { } as any);

function getEnumID(value: number, map?: Record<number, number>): number {
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
    
export function xorState(state: GameState): bigint {
    const pov = getPovTribe(state);

    let hash: bigint = 0n;

    // Settings //

    hash ^= zobristKeys.turn[state.settings._turn];

    hash ^= zobristKeys.pov[pov.owner];

    hash ^= zobristKeys.gameOver[state.settings._gameOver ? 1 : 0];

    // Map //

    for (let tileIndex = 0; tileIndex < state.tiles.length; tileIndex++) {
        const tile = state.tiles[tileIndex];

        hash ^= zobristKeys.tiles[tileIndex].explored[0];

        // Skip unexplored tiles
        if(!tile._explorers.has(pov.owner)) {
            continue;    
        }

        xorTile.discover(state, tile);
    }

    // Player //

    const pKeys = zobristKeys.player[pov.owner];

    if (!pKeys) {
        throw Error(`Zobrist: Player/Tribe (owner: ${pov.owner} > ${zobristKeys.player.length} out of bounds.`);
    }

    const tribeTypeId = getEnumID(pov.tribeType);

    if (tribeTypeId >= pKeys.tribeType.length) {
        throw Error(`Zobrist: Tribe Type ${tribeTypeId} > ${pKeys.tribeType.length} out of bounds.`);
    }

    if (pov._stars >= pKeys.stars.length) {
        throw Error(`Zobrist: Stars ${pov._stars} > ${pov._stars} ${pKeys.stars.length-1} out of bounds.`);
    }

    pov._tech.forEach(tech => {
        const techId = getEnumID(tech.techType, TechnologyToID);
        if (techId >= pKeys.hasTech.length) {
            throw Error(`Zobrist: Tech ${techId} > ${pKeys.hasTech.length} out of bounds.`);
        }
    });

    pov._builtUniqueStructures.forEach(structType => {
        const structId = getEnumID(structType, StructureToID);
        if (structId >= pKeys.unique.length) {
            throw Error(`Zobrist: Built Unique Structure ${structId} > ${pKeys.unique.length} out of bounds.`);
        }
    });

    xorPlayer.set(pov);

    return hash;
}

function xorSetUnit(
    hash: bigint,
    unit: UnitState
): bigint {
    const uKey = zobristKeys.units[unit._tileIndex];

    hash ^= uKey.owner[unit._owner];

    hash ^= uKey.type[getEnumID(unit._unitType, UnitToID)];

    hash ^= uKey.veteran[unit.veteran ? 1 : 0];

    hash ^= uKey.moved[unit._moved ? 1 : 0];

    hash ^= uKey.attacked[unit._attacked ? 1 : 0];

    if(unit._kills <= MODEL_CONFIG.max_unit_kills) {
        hash ^= uKey.kills[unit._kills];
    }

    if (unit._passenger) {
        hash ^= uKey.passenger[getEnumID(unit._passenger, UnitToID)];
    }
    
    // TODO notes promotable
    unit._effects.forEach(effect => {
        hash ^= uKey.effect[getEnumID(effect)];
    });

    return hash;
}

function xorSetStructure(
    hash: bigint,
    structType: StructureType,
    tileIndex: number,
): bigint {
    return hash ^ zobristKeys.structure[tileIndex][getEnumID(structType, StructureToID)];
}

function xorSetResource(
    hash: bigint,
    resourceType: ResourceType,
    tileIndex: number,
): bigint {
    return hash ^ zobristKeys.resource[tileIndex][getEnumID(resourceType)];
}

function xorSetCity(
    hash: bigint,
    city: CityState
): bigint {
    const cKey = zobristKeys.city[city.tileIndex];

    hash ^= cKey.owner[city._owner];

    hash ^= cKey.unitCount[city._unitCount];

    hash ^= cKey.level[city._level];

    hash ^= cKey.riot[city._riot ? 1 : 0];

    return hash;
}

// Assumes tribe already explored the tile
function xorSetTile(
    hash: bigint,
    tile: TileState
): bigint {
    const tKey = zobristKeys.tiles[tile.tileIndex];

    hash ^= tKey.explored[1];

    hash ^= tKey.owner[tile._owner];

    hash ^= tKey.terrainType[getEnumID(tile.terrainType)];

    return hash;
}

export function xorForAll(
    state: GameState,
    tileIndex: number,
    xorCb: (hash: bigint) => bigint,
) {
    state.tiles[tileIndex]._explorers.forEach(x => {
        state.tribes[x].hash = xorCb(state.tribes[x].hash);
    });
}

export class xorPlayer {
    static set(tribe: TribeState) {
        xorPlayer.type(tribe, tribe.tribeType);

        xorPlayer.stars(tribe, tribe._stars);

        for (let i = 0; i < tribe._tech.length; i++) {
            xorPlayer.tech(tribe, tribe._tech[i].techType);
        }

        for(const structType of tribe._builtUniqueStructures) {
            xorPlayer.unique(tribe, structType);
        }
    }

    static type(tribe: TribeState, tribeType: TribeType) {
        tribeType = tribeType ?? tribe.tribeType;
        tribe.hash ^= zobristKeys.player[tribe.owner].tribeType[tribeType];
    }

    static stars(tribe: TribeState, stars: number) {
        if(stars > MODEL_CONFIG.max_stars) return;
        tribe.hash ^= zobristKeys.player[tribe.owner].stars[stars];
    }

    static tech(tribe: TribeState, techType: TechnologyType) {
        tribe.hash ^= zobristKeys.player[tribe.owner].hasTech[getEnumID(techType, TechnologyToID)];
    }

    static unique(tribe: TribeState, structType: StructureType) {
        tribe.hash ^= zobristKeys.player[tribe.owner].unique[getEnumID(structType, StructureToID)];
    }
}

export class xorUnit {
    /** Assumes there was NO unit previously on this tile, or this EXACT unit with its EXACT variables */
    static set(state: GameState, unit: UnitState) {
        let unitHash = 0n;

        unitHash ^= zobristKeys.units[unit._tileIndex].owner[unit._owner];
        unitHash ^= zobristKeys.units[unit._tileIndex].type[getEnumID(unit._unitType, UnitToID)];
        unitHash ^= zobristKeys.units[unit._tileIndex].veteran[unit.veteran? 1 : 0];
        unitHash ^= zobristKeys.units[unit._tileIndex].moved[unit._moved? 1 : 0];
        unitHash ^= zobristKeys.units[unit._tileIndex].attacked[unit._attacked? 1 : 0];
        unitHash ^= zobristKeys.units[unit._tileIndex].kills[unit._kills];
        if(unit._passenger) {
            unitHash ^= zobristKeys.units[unit._tileIndex].passenger[getEnumID(unit._passenger, UnitToID)];
        }
        unit._effects.forEach(x => {
            unitHash ^= zobristKeys.units[unit._tileIndex].effect[getEnumID(x)];
        });

        xorForAll(state, unit._tileIndex, (hash) => hash ^ unitHash);
    }

    static owner(state: GameState, unit: UnitState, owner: number, newOwner: number) {
        xorForAll(state, unit._tileIndex, (hash) => {
            hash ^= zobristKeys.units[unit._tileIndex].owner[owner];
            if(newOwner) {
                hash ^= zobristKeys.units[unit._tileIndex].owner[newOwner];
            }
            return hash;
        });
    }

    static type(state: GameState, unit: UnitState, unitType: UnitType) {
        xorForAll(state, unit._tileIndex, (hash) => {
            hash ^= zobristKeys.units[unit._tileIndex].type[getEnumID(unitType, UnitToID)];
            return hash;
        });
    }

    static veteran(state: GameState, unit: UnitState, veteran: boolean) {
        xorForAll(state, unit._tileIndex, (hash) => {
            hash ^= zobristKeys.units[unit._tileIndex].veteran[veteran? 1 : 0];
            hash ^= zobristKeys.units[unit._tileIndex].veteran[veteran? 0 : 1];
            return hash;
        });
    }

    static moved(state: GameState, unit: UnitState, moved: boolean) {
        xorForAll(state, unit._tileIndex, (hash) => {
            hash ^= zobristKeys.units[unit._tileIndex].moved[moved? 1 : 0];
            hash ^= zobristKeys.units[unit._tileIndex].moved[moved? 0 : 1];
            return hash;
        });
    }

    static attacked(state: GameState, unit: UnitState, attacked: boolean) {
        xorForAll(state, unit._tileIndex, (hash) => {
            hash ^= zobristKeys.units[unit._tileIndex].attacked[attacked? 1 : 0];
            hash ^= zobristKeys.units[unit._tileIndex].attacked[attacked? 0 : 1];
            return hash;
        });
    }   

    static kills(state: GameState, unit: UnitState, curKills: number, newKills?: number) {
        curKills = Math.min(MODEL_CONFIG.max_unit_kills, curKills);
        // kill count can never decrease, so this is fine
        newKills = Math.min(MODEL_CONFIG.max_unit_kills, newKills || 0);
        xorForAll(state, unit._tileIndex, (hash) => {
            hash ^= zobristKeys.units[unit._tileIndex].kills[curKills];
            if(newKills > 0) {
                hash ^= zobristKeys.units[unit._tileIndex].kills[newKills];
            }
            return hash;
        });
    }

    static passenger(state: GameState, unit: UnitState, passenger?: UnitType) {
        if(passenger) {
            xorForAll(state, unit._tileIndex, (hash) => {
                hash ^= zobristKeys.units[unit._tileIndex].passenger[getEnumID(passenger, UnitToID)];
                return hash;
            });
        }
    }

    static effect(state: GameState, unit: UnitState, effect: EffectType) {
        xorForAll(state, unit._tileIndex, (hash) => {
            hash ^= zobristKeys.units[unit._tileIndex].effect[getEnumID(effect)];
            return hash;
        });
    }
}

export class xorCity {
    static set(state: GameState, city: CityState) {
        xorForAll(state, city.tileIndex, (hash) => xorSetCity(hash, city));
    }

    static owner(state: GameState, city: CityState, curOwner: number, newOwner: number) {
        xorForAll(state, city.tileIndex, (hash) => {
            hash ^= zobristKeys.city[city.tileIndex].owner[curOwner];
            hash ^= zobristKeys.city[city.tileIndex].owner[newOwner];
            return hash;
        });
    }

    static unitCount(state: GameState, city: CityState, curCount: number, newCount: number) {
        curCount = Math.min(MODEL_CONFIG.max_structure_level, curCount);
        newCount = Math.min(MODEL_CONFIG.max_structure_level, newCount);
        xorForAll(state, city.tileIndex, (hash) => {
            hash ^= zobristKeys.city[city.tileIndex].unitCount[curCount];
            hash ^= zobristKeys.city[city.tileIndex].unitCount[newCount];
            return hash;
        });
    }

    static level(state: GameState, city: CityState, curLevel: number, newLevel: number) {
        curLevel = Math.min(MODEL_CONFIG.max_structure_level, curLevel);
        newLevel = Math.min(MODEL_CONFIG.max_structure_level, newLevel);
        xorForAll(state, city.tileIndex, (hash) => {
            hash ^= zobristKeys.city[city.tileIndex].level[curLevel];
            hash ^= zobristKeys.city[city.tileIndex].level[newLevel];
            return hash;
        });
    }

    static riot(state: GameState, city: CityState, riot: boolean) {
        xorForAll(state, city.tileIndex, (hash) => {
            hash ^= zobristKeys.city[city.tileIndex].riot[riot? 1 : 0];
            hash ^= zobristKeys.city[city.tileIndex].riot[riot? 0 : 1];
            return hash;
        });
    }
}

export class xorTile {
    // when a new tile is discovered we must update whatever new discovered structs, resources, cities or units are on it
    static discover(state: GameState, tile: TileState) {
        const pov = getPovTribe(state);

        // sneaky: revert unexplored tile
        pov.hash ^= zobristKeys.tiles[tile.tileIndex].explored[0];

        // update normally

        // Tile
        pov.hash = xorSetTile(pov.hash, tile);

        // Unit
        const unitAt = getTrueUnitAt(state, tile.tileIndex);
        if(unitAt) {
            pov.hash = xorSetUnit(pov.hash, unitAt);
        }

        // Structure
        pov.hash = xorSetStructure(pov.hash, state.structures[tile.tileIndex]?.id || StructureType.None, tile.tileIndex);

        // Resource
        pov.hash = xorSetResource(pov.hash, state.resources[tile.tileIndex]?.id || ResourceType.None, tile.tileIndex);

        // Cities
        const city = getCityAt(state, tile.tileIndex);
        if (city) {
            pov.hash = xorSetCity(pov.hash, city);
        }
    }

    static owner(state: GameState, tileIndex: number, oldOwner: number, newOwner: number) {
        xorForAll(state, tileIndex, (hash) => {
            hash ^= zobristKeys.tiles[tileIndex].owner[oldOwner];
            hash ^= zobristKeys.tiles[tileIndex].owner[newOwner];
            return hash;
        });
    }
    
    static terrain(state: GameState, tileIndex: number, oldTerrain: TerrainType, newTerrain: TerrainType) {
        xorForAll(state, tileIndex, (hash) => {
            hash ^= zobristKeys.tiles[tileIndex].terrainType[getEnumID(oldTerrain)];
            hash ^= zobristKeys.tiles[tileIndex].terrainType[getEnumID(newTerrain)];
            return hash;
        });
    }
}

export function xorResource(
    state: GameState,
    tileIndex: number,
    resourceType: ResourceType,
    newResourceType: ResourceType,
): void {
    xorForAll(state, tileIndex, (hash) => {
        hash = xorSetResource(hash, resourceType, tileIndex);
        if(newResourceType) {
            hash = xorSetResource(hash, newResourceType, tileIndex);
        }
        return hash;
    })
}

export function xorStructure(
    state: GameState,
    tileIndex: number,
    structType: StructureType,
    newStructType: StructureType,
): void {
    xorForAll(state, tileIndex, (hash) => {
        hash = xorSetStructure(hash, structType, tileIndex);
        if(newStructType) {
            hash = xorSetStructure(hash, newStructType, tileIndex);
        }
        return hash;
    })
}