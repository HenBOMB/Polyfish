import { log } from "node:console";
import { MODEL_CONFIG } from "../aistate";
import { getCityAt, getPovTribe, getResourceAt, getStructureAt, getTrueUnitAt, getUnitAt } from "../core/functions";
import { StructureSettings } from "../core/settings/StructureSettings";
import { TechnologySettings } from "../core/settings/TechnologySettings";
import { UnitSettings } from "../core/settings/UnitSettings";
import { CityState, GameState, TileState, TribeState, UnitState } from "../core/states";
import { EffectType, ResourceType, StructureType, TechnologyType, TerrainType, TribeType, UnitType } from "../core/types";
import { zobristKeyStrings } from "./zobristKeys";
import { ZobristKeys } from "./generateZorbist";

export function parseZobristKeys(obj: any): any {
    if(typeof obj === 'string' && obj.endsWith('n')) {
        return BigInt(obj.slice(0, -1));
    }
    if(Array.isArray(obj)) {
        return obj.map(parseZobristKeys);
    }
    const newObj: any = { };
    for(const key in obj) {
        if(Object.prototype.hasOwnProperty.call(obj, key)) {
            newObj[key] = parseZobristKeys(obj[key]);
        }
    }
    return newObj;
}

const zobristKeys: ZobristKeys = parseZobristKeys(zobristKeyStrings);

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
    return 0n;
    const pov = getPovTribe(state);

    let hash: bigint = 0n;

    // Settings //

    hash ^= zobristKeys.turn[state.settings.turn];

    hash ^= zobristKeys.pov[pov.id];

    hash ^= zobristKeys.gameOver[state.settings._gameOver ? 1 : 0];

    // Map //

    for (let idx = 0; idx < state.settings.tileCount; idx++) {
        const tile = state.tiles[idx];

        // Skip unexplored tiles
        if(!tile.explorers.has(pov.id)) {
            continue;    
        }

        xorTile.discover(state, tile);
    }

    // Player //

    const pKeys = zobristKeys.player[pov.id];

    if (!pKeys) {
        throw Error(`Zobrist: Player/Tribe (owner: ${pov.id} > ${zobristKeys.player.length} out of bounds.`);
    }

    const tribeTypeId = getEnumID(pov.type);

    if (tribeTypeId >= pKeys.tribeType.length) {
        throw Error(`Zobrist: Tribe Type ${tribeTypeId} > ${pKeys.tribeType.length} out of bounds.`);
    }

    if (pov.stars >= pKeys.stars.length) {
        throw Error(`Zobrist: Stars ${pov.stars} > ${pov.stars} ${pKeys.stars.length-1} out of bounds.`);
    }

    pov.tech_vanilla.forEach(tech => {
        const techId = getEnumID(tech.type, TechnologyToID);
        if (techId >= pKeys.hasTech.length) {
            throw Error(`Zobrist: Tech ${techId} > ${pKeys.hasTech.length} out of bounds.`);
        }
    });

    pov.builtUniqueImprovements.forEach(structType => {
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
    return 0n;
    const uKey = zobristKeys.units[unit.coords.idx];

    hash ^= uKey.owner[unit.owner];
    hash ^= uKey.type[getEnumID(unit.type, UnitToID)];
    hash ^= unit.veteran? uKey.veteran : 0n;
    hash ^= unit.moved? uKey.moved : 0n;
    hash ^= unit.attacked? uKey.attacked : 0n;
    if(unit.kills <= MODEL_CONFIG.max_unit_kills) {
        hash ^= uKey.kills[unit.kills];
    }
    hash ^= uKey.passenger[unit.passengerType? getEnumID(unit.passengerType!, UnitToID) : UnitType.None];
    
    // TODO notes promotable
    unit.effects.forEach(effect => {
        hash ^= uKey.effect[getEnumID(effect)];
    });

    return hash;
}

function xorSetStructure(
    hash: bigint,
    structType: StructureType,
    tileIndex: number,
): bigint {
    return 0n;
    return hash ^ zobristKeys.structure[tileIndex][getEnumID(structType, StructureToID)];
}

function xorSetResource(
    hash: bigint,
    resourceType: ResourceType,
    tileIndex: number,
): bigint {
    return 0n;
    return hash ^ zobristKeys.resource[tileIndex][getEnumID(resourceType)];
}

function xorSetCity(
    hash: bigint,
    city: CityState
): bigint {
    return 0n;
    const cKey = zobristKeys.city[city.tileIndex];

    hash ^= cKey.owner[city.owner];
    hash ^= cKey.level[city.level];
    hash ^= cKey.riot;

    return hash;
}

// Assumes tribe already explored the tile
function xorSetTile(
    hash: bigint,
    tile: TileState
): bigint {
    return 0n;
    const tKey = zobristKeys.tiles[tile.coords.idx];

    hash ^= tKey.explored;
    hash ^= tKey.owner[tile.owner];
    hash ^= tKey.terrainType[getEnumID(tile.type)];

    return hash;
}

export function xorForAll(
    state: GameState,
    tileIndex: number,
    xorCb: (hash: bigint) => bigint,
) {
    return 0n;
    state.tiles[tileIndex].explorers.forEach(x => {
        state.tribes[x]._hash = xorCb(state.tribes[x]._hash);
    });
}

export class xorPlayer {
    static set(tribe: TribeState) {
        return;
        xorPlayer.type(tribe, tribe.type);

        xorPlayer.stars(tribe, tribe.stars);

        for (let i = 0; i < tribe.tech_vanilla.length; i++) {
            xorPlayer.tech(tribe, tribe.tech_vanilla[i].type);
        }

        for(const structType of tribe.builtUniqueImprovements) {
            xorPlayer.unique(tribe, structType);
        }
    }

    static type(tribe: TribeState, tribeType: TribeType) {
        return;
        tribeType = tribeType ?? tribe.type;
        tribe._hash ^= zobristKeys.player[tribe.id].tribeType[tribeType];
    }

    static stars(tribe: TribeState, stars: number) {
        return;
        stars = Math.min(stars, MODEL_CONFIG.max_stars);  
        try {
            tribe._hash ^= zobristKeys.player[tribe.id].stars[stars];
        } catch (error) {
            console.log(error);
            console.log('STARS', stars, tribe.id, tribe._hash);
        }
    }

    static tech(tribe: TribeState, techType: TechnologyType) {
        return;
        tribe._hash ^= zobristKeys.player[tribe.id].hasTech[getEnumID(techType, TechnologyToID)];
    }

    static unique(tribe: TribeState, structType: StructureType) {
        return;
        tribe._hash ^= zobristKeys.player[tribe.id].unique[getEnumID(structType, StructureToID)];
    }
}

export class xorUnit {
    /** Assumes there was NO unit previously on this tile, or this EXACT unit with its EXACT variables */
    static set(state: GameState, unit: UnitState) {
        return;
        const unitHash = xorSetUnit(0n, unit);
        xorForAll(state, unit.coords.idx, (hash) => hash ^ unitHash);
    }

    /**
     * curOwner ^ newOwner || 0n
     * @param state
     * @param unit 
     * @param curOwner
     * @param newOwner
     */
    static owner(state: GameState, unit: UnitState, curOwner: number, newOwner: number) {
        return;
        xorForAll(state, unit.coords.idx, (hash) => {
            hash ^= zobristKeys.units[unit.coords.idx].owner[curOwner];
            if(newOwner) {
                hash ^= zobristKeys.units[unit.coords.idx].owner[newOwner];
            }
            return hash;
        });
    }

    /**
     * curUnitType ^ newUnitType
     * @param state
     * @param unit
     * @param curUnitType
     * @param newUnitType
     */
    static type(state: GameState, unit: UnitState, curUnitType: UnitType, newUnitType: UnitType) {
        return;
        xorForAll(state, unit.coords.idx, (hash) => {
            hash ^= zobristKeys.units[unit.coords.idx].type[getEnumID(curUnitType, UnitToID)];
            hash ^= zobristKeys.units[unit.coords.idx].type[getEnumID(newUnitType, UnitToID)];
            return hash;
        });
    }

    /**
     * xors the _veteran state
     * @param state 
     * @param unit 
     * @param veteran 
     */
    static veteran(state: GameState, unit: UnitState) {
        return;
        xorForAll(state, unit.coords.idx, (hash) => {
            hash ^= zobristKeys.units[unit.coords.idx].veteran;
            return hash;
        });
    }

    /**
     * xors the _moved state
     * @param state 
     * @param unit 
     * @param moved 
     */
    static moved(state: GameState, unit: UnitState) {
        return;
        xorForAll(state, unit.coords.idx, (hash) => {
            hash ^= zobristKeys.units[unit.coords.idx].moved;
            return hash;
        });
    }

    /**
     * xors the _attacked state
     * @param state 
     * @param unit 
     * @param attacked 
     */
    static attacked(state: GameState, unit: UnitState) {
        return;
        xorForAll(state, unit.coords.idx, (hash) => {
            hash ^= zobristKeys.units[unit.coords.idx].attacked;
            return hash;
        });
    }   

    /**
     * curKills ^ newKills
     * @param state 
     * @param unit 
     * @param curKills 
     * @param newKills
     */
    static kills(state: GameState, unit: UnitState, curKills: number, newKills?: number) {
        return;
        curKills = Math.min(MODEL_CONFIG.max_unit_kills, curKills);
        // kill count can never decrease, so this is fine
        newKills = Math.min(MODEL_CONFIG.max_unit_kills, newKills || 0);
        xorForAll(state, unit.coords.idx, (hash) => {
            hash ^= zobristKeys.units[unit.coords.idx].kills[curKills];
            if(newKills! > 0) {
                hash ^= zobristKeys.units[unit.coords.idx].kills[newKills!];
            }
            return hash;
        });
    }

    /**
     * curPassenger ^ newPassenger
     * @param state 
     * @param unit 
     * @param curPassenger
     * @param newPassenger
     */
    static passenger(state: GameState, unit: UnitState, curPassenger: UnitType, newPassenger: UnitType) {
        return;
        xorForAll(state, unit.coords.idx, (hash) => {
            hash ^= zobristKeys.units[unit.coords.idx].passenger[getEnumID(curPassenger, UnitToID)];
            hash ^= zobristKeys.units[unit.coords.idx].passenger[getEnumID(newPassenger, UnitToID)];
            return hash;
        });
    }

    /**
     * 0n ^ effect
     * @param state 
     * @param unit 
     * @param effect
     */
    static effect(state: GameState, unit: UnitState, effect: EffectType) {
        return;
        xorForAll(state, unit.coords.idx, (hash) => {
            hash ^= zobristKeys.units[unit.coords.idx].effect[getEnumID(effect)];
            return hash;
        });
    }
}

export class xorCity {
    static set(state: GameState, city: CityState) {
        return;
        xorForAll(state, city.tileIndex, (hash) => xorSetCity(hash, city));
    }

    static owner(state: GameState, city: CityState, curOwner: number, newOwner: number) {
        return;
        xorForAll(state, city.tileIndex, (hash) => {
            hash ^= zobristKeys.city[city.tileIndex].owner[curOwner];
            hash ^= zobristKeys.city[city.tileIndex].owner[newOwner];
            return hash;
        });
    }

    static level(state: GameState, city: CityState, curLevel: number, newLevel: number) {
        return;
        curLevel = Math.min(MODEL_CONFIG.max_structure_level, curLevel);
        newLevel = Math.min(MODEL_CONFIG.max_structure_level, newLevel);
        xorForAll(state, city.tileIndex, (hash) => {
            hash ^= zobristKeys.city[city.tileIndex].level[curLevel];
            hash ^= zobristKeys.city[city.tileIndex].level[newLevel];
            return hash;
        });
    }

    static riot(state: GameState, city: CityState) {
        return;
        xorForAll(state, city.tileIndex, (hash) => {
            hash ^= zobristKeys.city[city.tileIndex].riot;
            return hash;
        });
    }
}

export class xorTile {
    // when a new tile is discovered we must update whatever new discovered structs, resources, cities or units are on it
    static discover(state: GameState, tile: TileState) {
        return;
        const pov = getPovTribe(state);

        // Tile
        pov._hash = xorSetTile(pov._hash, tile);

        // Unit
        const unitAt = getTrueUnitAt(state, tile.coords.idx);
        if(unitAt) {
            pov._hash = xorSetUnit(pov._hash, unitAt!);
        }

        // Structure
        pov._hash = xorSetStructure(pov._hash, state.structures[tile.coords.idx]?.type || StructureType.None, tile.coords.idx);

        // Resource
        pov._hash = xorSetResource(pov._hash, state.resources[tile.coords.idx]?.type || ResourceType.None, tile.coords.idx);

        // Cities
        const city = getCityAt(state, tile.coords.idx);
        if (city) {
            pov._hash = xorSetCity(pov._hash, city!);
        }
    }

    static owner(state: GameState, idx: number, oldOwner: number, newOwner: number) {
        xorForAll(state, idx, (hash) => {
            hash ^= zobristKeys.tiles[idx].owner[oldOwner];
            hash ^= zobristKeys.tiles[idx].owner[newOwner];
            return hash;
        });
    }
    
    static terrain(state: GameState, idx: number, oldTerrain: TerrainType, newTerrain: TerrainType) {
        xorForAll(state, idx, (hash) => {
            hash ^= zobristKeys.tiles[idx].terrainType[getEnumID(oldTerrain)];
            hash ^= zobristKeys.tiles[idx].terrainType[getEnumID(newTerrain)];
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
        return 0n;
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
        return 0n;
        hash = xorSetStructure(hash, structType, tileIndex);
        if(newStructType) {
            hash = xorSetStructure(hash, newStructType, tileIndex);
        }
        return hash;
    })
}