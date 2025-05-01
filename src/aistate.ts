import { readFileSync } from "fs";
import { GameState } from "./core/states";
import { cloneState, getPovTribe, getUnitAtTile, isResourceVisible, getMaxHealth, isGameWon, isGameLost, getCity, getCityProduction, getRealUnitSettings, getTechCost, isTempleStructure, getNeighborIndexes } from "./core/functions";
import { CaptureType, Climate2Tribe, EffectType, ModeType, ResourceType, TerrainType, UnitType } from "./core/types";
import Move, { MoveType } from "./core/move";
import { TechnologySettings } from "./core/settings/TechnologySettings";
import { predictBestNextCityReward } from "./eval/prediction";
import { ResourceSettings } from "./core/settings/ResourceSettings";
import { StructureSettings } from "./core/settings/StructureSettings";
import { UnitSettings } from "./core/settings/UnitSettings";

const PRECISION = 2;

/** @type {ModelConfig} */
export const MODEL_CONFIG: {
    max_turns: number,
    max_score: number,
    max_size: number,
    max_stars: number,
    max_tribes: number,
    max_tech: number,
    max_effects: number
    max_actions: number
    tile_channels: number,
    settings_channels: number,
    tile_count: number,
} = JSON.parse(readFileSync('data/model/config.json', 'utf8'));

MODEL_CONFIG.tile_count = MODEL_CONFIG.max_size ** 2;

function safeBool(b: boolean): number { 
    return b? 1 : 0; 
}

function safeArray<T>(a: T[], size: number, _default?: T): T[] {
    if(a.length > size) {
        a = a.slice(0, size);
    }
    else if(a.length < size) {
        a = a.concat(Array(size - a.length).fill(_default || 0));
    }
    return a;
}

function safeFloat(f: number | undefined): number { 
    if(typeof(f) != "number") {
        throw new Error("NaN");
    }
    return !f || f < 0? 0 : Number(f.toFixed(PRECISION)) 
}

function safeID(n: number | undefined): number { return !n? 0 : n > 0? n : 0 }

function extractMap(state: GameState): number[][][] {
    const pov = getPovTribe(state);
    const tileCount = state.settings.size ** 2;
    const allTribes = Object.values(state.tribes);

    if(tileCount > MODEL_CONFIG.max_size ** 2) {
        console.log(MODEL_CONFIG.max_size, state.settings.size)
        throw "Map size too large";
    }

    const offset = Math.floor((MODEL_CONFIG.max_size - state.settings.size) / 2);

    const grid: number[][][] = Array(MODEL_CONFIG.tile_channels).fill(0).map(() =>
        Array(MODEL_CONFIG.max_size).fill(0).map(() => Array(MODEL_CONFIG.max_size).fill(0))
    );

    // console.log('climate', (Object.keys(ClimateType).length / 2 - 1));
    // console.log('resource', (Object.keys(TerrainType).length / 2 - 1));
    // console.log('terrain', (Object.keys(ResourceType).length / 2 - 1));

    for (const i in state.tiles) {
        const tileIndex = Number(i);
        const tile = state.tiles[tileIndex];
        const gridX = tile.x + offset;
        const gridY = tile.y + offset;
        const explored = tile.explorers.includes(state.settings._pov);
        const resource = explored && isResourceVisible(pov, state.resources[tileIndex]?.id)? state.resources[tileIndex] : null;
        const city = getCity(state, tileIndex);
        const unit = getUnitAtTile(state, tileIndex);
        const ownerFromClimate = allTribes.find(x => x.tribeType == Climate2Tribe[tile.climate])?.owner;
        
        const output = explored? [
            // TILE?
            // Tile Index
            tile.tileIndex / tileCount,
            // Terrain Type
            tile.terrainType / (Object.keys(TerrainType).length / 2 - 1),
            // Climate ID -> Representing Tribe Owner ID
            // this is because the Ai cant differenciate what tribe is each owner
            safeID(ownerFromClimate || -1),
            // Resource ID
            city? 0 : safeID(resource?.id) / (Object.keys(ResourceType).length / 2 - 1),
            // Has path
            city? 1 : safeBool(tile.hasRoad || tile.hasRoute),

            // TRIBE?
            // Owner ID
            safeID(tile._owner) / MODEL_CONFIG.max_tribes,

            // City
            !city? 0 : safeBool(tile.capitalOf > 0),
            !city? 0 : (Math.min(city._level, 7) / 7),
            !city? 0 : (city._progress / (city._level + 1)),
            !city? 0 : (((city._walls? 1 : 0) + (city._riot? 2 : 0)) / 3),

            // Unit
            !unit? 0 : (safeID(unit._owner) / MODEL_CONFIG.max_tribes),
            !unit? 0 : (safeID(unit._passenger || 0 > 0? unit._passenger : unit._unitType) / (Object.keys(UnitType).length / 2 - 1)),
            !unit? 0 : (unit._health / getMaxHealth(unit)),
            !unit? 0 : (((unit._moved? 1 : 0) + (unit._attacked? 2 : 0)) / 3),
            !unit? 0 : (Math.min(unit.kills, 3) / 3),
            // poison, boosted, invisible, frozen
            ...(!unit? [0, 0, 0] : safeArray<number>(
                unit._effects
                    .map(x => x == EffectType.None? 0 : (x / MODEL_CONFIG.max_effects))
                    .filter(Boolean)
                    .sort((a, b) => a - b), 
                MODEL_CONFIG.max_effects,
                0
            )),
        ].map(x => safeFloat(x)) : Array(MODEL_CONFIG.tile_channels).fill(0);

        for (let i = 0; i < output.length; i++) {
            grid[i][gridY][gridX] = output[i];
        }
    }

    return grid;
}

function extractPlayer(_state: GameState): number[] {
    const state = cloneState(_state);

    const allTech = Object.keys(TechnologySettings);
    
    if(allTech.length != MODEL_CONFIG.max_tech) {
        throw Error(`Tech count mismatch: ${allTech.length} != ${MODEL_CONFIG.max_tech}`);
    }

    const pov = getPovTribe(state);

    return [
        [
            state.settings._pov / MODEL_CONFIG.max_tribes,
            state.settings.mode == ModeType.Domination? 0 : (pov._score / MODEL_CONFIG.max_score),
            Math.min(pov._stars, MODEL_CONFIG.max_stars) / MODEL_CONFIG.max_stars,
        ],

        allTech.map(x => pov._tech.includes(Number(x))? 1 : 0),
    ].flat();
}

export default class AIState {
    static extract(state: GameState): {
        map: number[][][],
        player: number[],
    } {
        return {
            map: extractMap(state),
            player: extractPlayer(state),
        }
    }

    static executeBestReward(state: GameState, moves: Move[]) {
        const city = getPovTribe(state)._cities.find(x => moves.some(y => y.src == x.tileIndex));
        const rewardType = predictBestNextCityReward(state, city)[0];
        const move = moves.find(x => x.type == rewardType)!;
        move.execute(state);
    }

    static calculateReward(oldState: GameState, newState: GameState, move: Move | null = null): number {
        // ! ASSUMES NO FOW

        let reward = 0;

        const pov = newState.settings._pov;
        const oldTribe = oldState.tribes[pov];
        const newTribe = newState.tribes[pov];

        const calculateProduction = () => {
            return 0.25 * (newTribe._cities.reduce((a, b) => a + getCityProduction(newState, b), 0) - oldTribe._cities.reduce((a, b) => a + getCityProduction(oldState, b), 0));
        }

        if(move) {
            // Technology
            // Penalize if unlocked a new tech and that tech hasnt been used
            // Penalize units that are not doing anything
            if (move.moveType === MoveType.Research) {
                const settings = TechnologySettings[move.type as keyof typeof TechnologySettings];
                
                // Get cost to "use" what this tech has unlocked
                // Order of importance: Resource -> Structure (except temples) -> Ability -> Unit -> Structure (temples)
                // TODO Using this order to reduce bad bias and potentially better moves, not sure tho..

                const cost = (
                    settings.unlocksResource? ResourceSettings[settings.unlocksResource] :
                    settings.unlocksStructure && !isTempleStructure(settings.unlocksStructure)? StructureSettings[settings.unlocksStructure] :
                    settings.unlocksUnit? UnitSettings[settings.unlocksUnit] :
                    settings.unlocksAbility? { cost: 0 } :
                    settings.unlocksStructure && isTempleStructure(settings.unlocksStructure)? StructureSettings[settings.unlocksStructure] : 
                    null
                )?.cost || 0;
    
                // Penalize for not being able to use the tech properly
                if(cost && newTribe._stars < cost) {
                    reward -= 0.4;
                }
                // Stanadard penalty, researching isnt free and comes with consecuenses!
                else {
                    reward -= 0.1;
                }
    
                return reward;
            }
            // Reward well placed structures that require other adjacent structures
            else if(move.moveType == MoveType.Build) {
                const settings = StructureSettings[move.type as keyof typeof StructureSettings];
                if(settings.adjacentTypes) {
                    const rating = newState._visibleTiles.reduce((a: number, x: number) => {
                        if(!x || !newState.structures[x]) return a;
                        const adjacentTypes = StructureSettings[newState.structures[x].id].adjacentTypes;
                        if(!adjacentTypes) return a;
                        const aroundStructures = getNeighborIndexes(newState, Number(x), 1).filter(x => newState.structures[x] && adjacentTypes.includes(newState.structures[x].id));
                        return a + aroundStructures.length;
                    }, 0) / 8; // 8 is max surrounding tiles (ignoring edges)
                    reward += 0.02 * rating;
                }
                return reward + calculateProduction();
            }
            // Reward capturing starfish and ruins
            else if(move.moveType == MoveType.Capture) {
                if(move.type == CaptureType.Starfish || move.type == CaptureType.Ruins) {
                    return 0.5 + calculateProduction();
                }
            }
        }

        // Victory Condition //
        if(isGameWon(newState)) {
            return 5.0;
        }
        else if(isGameLost(newState)) {
            return -5.0;
        }
        
        // Kills //
        reward += 0.6 * (newTribe._kills - oldTribe._kills);

        // Cities //
        reward += 1.0 * (newTribe._cities.length - oldTribe._cities.length);
        // Bonus connecting cities
        reward += 0.05 * (
            newTribe._cities.reduce((a, b) => a + (b._connectedToCapital? 1 : 0), 0) -
            oldTribe._cities.reduce((a, b) => a + (b._connectedToCapital? 1 : 0), 0));
        // Bonus capturing enemy cities
        if(move?.moveType == MoveType.Capture && oldState.tiles[move.src]._owner > 0 && oldState.tiles[move.src]._owner != newState.tiles[move.src]._owner) {
            reward += 1.0;
        }

        // Production //
        reward += calculateProduction();
    
        // Tiny time cost
        reward -= 0.01;

        // clip to [-1,1] to keep returns stable
        return Math.max(-1, Math.min(1, reward));
    }

    static calculatePotential(state: GameState): number {
        // TODO Using 1 enemy pov as reference, for more enemies.. figure it out
        // ! ASSUMES NO FOW

        const pov = getPovTribe(state);
        const [ enemyPov ] = Object.values(state.tribes).filter(x => x.owner != pov.owner);

        // Kills / Army
        // Super units are worth triple!
        const unit_diff = 
            pov._units.reduce((a, b) => a + (getRealUnitSettings(b).cost == 10? 3 : 1), 0) - 
            enemyPov._units.reduce((a, b) => a + (getRealUnitSettings(b).cost == 10? 3 : 1), 0);
        
        // Cities
        const city_diff = pov._cities.length - enemyPov._cities.length;
        
        // Production
        const production_diff = 
            pov._cities.reduce((x, y) => x + getCityProduction(state, y), 0) - 
            enemyPov._cities.reduce((x, y) => x + getCityProduction(state, y), 0);

        return 1.0 * city_diff + 0.1 * unit_diff + production_diff * 0.3;
    }
}