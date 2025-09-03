import * as fs from 'fs';
import * as crypto from 'crypto';
import { TerrainType, StructureType, ResourceType, UnitType, EffectType, TribeType } from '../core/types';
import { MODEL_CONFIG } from '../aistate';

const terrainTypeCount 	  = Object.values(TerrainType).length / 2;
const structureTypeCount  = Object.values(StructureType).length / 2;
const resourceTypeCount   = Object.values(ResourceType).length / 2;
const unitTypeCount 	  = Object.values(UnitType).length / 2;
const effectTypeCount 	  = Object.values(EffectType).length / 2;
const tribeTypeCount 	  = Object.values(TribeType).length / 2;
const technologyTypeCount = 45;

// +1 are non 0 indexed

const MAX_KILLS = MODEL_CONFIG.max_unit_kills + 1;
const MAX_PLAYERS = MODEL_CONFIG.max_tribes + 1;
const MAX_TILES = MODEL_CONFIG.max_tile_count;
const MAX_STARS_PER_PLAYER = MODEL_CONFIG.max_stars + 1;
const MAX_GAME_TURNS = MODEL_CONFIG.max_turns + 1;
// ? the amount of units a city can own is the same as its level
const MAX_CITY_UNIT_COUNT = MODEL_CONFIG.max_structure_level + 1;
var _taken: Set<bigint> = new Set();

export interface ZobristKeys {
    turn: bigint[];                 // number[]
    pov: bigint[];                  // ownerID[]
    gameOver: bigint[];             // boolean[]

	// *[tileIndex] -> ...

    tiles: {
        terrainType: bigint[];      // TerrainType[]
        explored: bigint;           // bool
        owner: bigint[];            // ownerID[]
    }[];

    // Units
    units: {
        owner: bigint[];            // ownerID[]
		kills: bigint[];            // number[]
        type: bigint[];             // UnitType[]
        passenger: bigint[];        // UnitType[]
		veteran: bigint;            // bool
        moved: bigint;              // bool
        attacked: bigint;           // bool
        effect: bigint[];           // EffectType[]
    }[];

    // Structures
    structure: bigint[][];          // StructureType[]

    // Resources
    resource: bigint[][];           // ResourceType[]    

    // Cities
    city: {
        level: bigint[];            // number[]
        unitCount: bigint[];        // number[]
        riot: bigint;               // bool
        owner: bigint[];            // ownerID[]
    }[];

    // Tribes/Players
    player: {
        unique: bigint[];           // StructureType[]
        stars: bigint[];            // number[]
        tribeType: bigint[];        // TribeType[]
        hasTech: bigint[];          // TechnologyType[]
    }[];
}

function generateRandom64BitBigInt(): bigint {
    let buffer = crypto.randomBytes(8);
    let big = buffer.readBigUInt64BE(0);

    while (_taken.has(big)) {
        buffer = crypto.randomBytes(8);
        big = buffer.readBigUInt64BE(0);
    }

    _taken.add(big);

    return big;
}

function initializeKeys<T>(count1: number, generator: () => T): T[] {
    return Array.from({ length: count1 }, generator);
}

function generateAllZobristKeys(): ZobristKeys {
    const keys: ZobristKeys = { } as ZobristKeys;

    // Settings
    keys.turn = initializeKeys(MAX_GAME_TURNS, generateRandom64BitBigInt);
    keys.pov = initializeKeys(MAX_PLAYERS, generateRandom64BitBigInt);
    keys.gameOver = initializeKeys(2, generateRandom64BitBigInt);

    // Tiles
    keys.tiles = initializeKeys(MAX_TILES, () => ({
        explored: generateRandom64BitBigInt(),
        owner: initializeKeys(MAX_PLAYERS, generateRandom64BitBigInt),
        terrainType: initializeKeys(terrainTypeCount, generateRandom64BitBigInt),
    }));

    // Units
    keys.units = initializeKeys(MAX_TILES, () => ({
        owner: initializeKeys(MAX_PLAYERS, generateRandom64BitBigInt),
        type: initializeKeys(unitTypeCount, generateRandom64BitBigInt),
        passenger: initializeKeys(unitTypeCount, generateRandom64BitBigInt),
        kills: initializeKeys(MAX_KILLS, generateRandom64BitBigInt),
        veteran: generateRandom64BitBigInt(),
        moved: generateRandom64BitBigInt(),
        attacked: generateRandom64BitBigInt(),
        effect: initializeKeys(effectTypeCount, generateRandom64BitBigInt),
    }));

    // Structures
    keys.structure = initializeKeys(MAX_TILES, () => 
       	initializeKeys(structureTypeCount, generateRandom64BitBigInt),
    );

    // Resources
    keys.resource = initializeKeys(MAX_TILES, () => 
        initializeKeys(resourceTypeCount, generateRandom64BitBigInt)
    );

    // Cities
    keys.city = initializeKeys(MAX_TILES, () => ({
        level: initializeKeys(MAX_CITY_UNIT_COUNT, generateRandom64BitBigInt),
        unitCount: initializeKeys(MAX_CITY_UNIT_COUNT, generateRandom64BitBigInt),
        riot: generateRandom64BitBigInt(),
        owner: initializeKeys(MAX_PLAYERS, generateRandom64BitBigInt),
    }));

    // Players
    keys.player = initializeKeys(MAX_PLAYERS, () => ({
        unique: initializeKeys(structureTypeCount, generateRandom64BitBigInt),
        stars: initializeKeys(MAX_STARS_PER_PLAYER, generateRandom64BitBigInt),
        tribeType: initializeKeys(tribeTypeCount, generateRandom64BitBigInt),
        hasTech: initializeKeys(technologyTypeCount, generateRandom64BitBigInt),
    }));

    return keys;
}

export function generateFile() {
    _taken = new Set();
    const out = 'src/zobrist/zobristKeys.ts';
    const data = generateAllZobristKeys();
    const strings = JSON.stringify(data, (key, value) => {
        if(typeof value === 'bigint') {
            return value.toString() + 'n';
        }
        return value;
    }, 2);

    fs.writeFileSync(out, `
// These keys are auto-generated. Do not edit manually.
export const zobristKeyStrings = ${strings};`);
    console.log(`Zobrist keys generated to ${out}`);
}
