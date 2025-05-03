import { readFileSync } from "fs";
import { GameState } from "./core/states";
import { cloneState, getPovTribe, getUnitAtTile, isResourceVisible, getMaxHealth, isGameWon, isGameLost, getCity, getCityProduction, getRealUnitSettings, getTechCost, isTempleStructure, getNeighborIndexes, getEnemyAtTile } from "./core/functions";
import { AbilityType, CaptureType, Climate2Tribe, EffectType, ModeType, ResourceType, StructureType, TerrainType, UnitType } from "./core/types";
import Move, { CallbackResult, MoveType } from "./core/move";
import { TechnologySettings } from "./core/settings/TechnologySettings";
import { predictBestNextCityReward } from "./eval/prediction";
import { ResourceSettings } from "./core/settings/ResourceSettings";
import { StructureSettings } from "./core/settings/StructureSettings";
import { UnitSettings } from "./core/settings/UnitSettings";
import Game from "./game";

// forgot what this was
const LIVE_TO_CAST_PRECISION = 3;

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
    return !f || f < 0? 0 : Number(f.toFixed(LIVE_TO_CAST_PRECISION)) 
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

export type Observation = {
    map: number[][][],
    player: number[],
};

export default class AIState {
    static extract(state: GameState): Observation {
        return {
            map: extractMap(state),
            player: extractPlayer(state),
        }
    }

    static executeBestReward(state: GameState, moves: Move[]): CallbackResult {
        const city = getPovTribe(state)._cities.find(x => moves.some(y => y.src == x.tileIndex));
        const rewardType = predictBestNextCityReward(state, city)[0];
        const move = moves.find(x => x.type == rewardType)!;
        return move.execute(state);
    }

    // TODO NORMALIZE -1 ... 1
    /**
     * Returns a value between -0.75 and 0.75. Uses game.stateBefore and game.state
     * @param moves Moves recently played
     * @returns 
     */
    static calculateReward(game: Game, ...moves: Move[]): number {
        const move: Move = moves[0];
        // The forced reward move, caused by upgrading a city
        const rewardMove: Move = moves[1];

        // ! ASSUMES NO FOW
        const oldState = game.stateBefore;
        const newState = game.state;

        const pov = newState.settings._pov;
        const oldTribe = oldState.tribes[pov];
        const newTribe = newState.tribes[pov];

        // LOSS = -1
        // ... EVERYTHING ELSE IN BETWEEN ...
        // WIN = 1

        enum HowGood {
            // Defeat  = -1.00,
            Nasty   = -0.75,
            Worse   = -0.5,
            Bad     = -0.25,
            Bleh    = -0.10,
            None    = 0.00,
            Meh     = 0.10,
            Good    = 0.25,
            Great   = 0.50,
            Awesome = 0.75,
            // Victory = 1.00,
        };

        // Capturing Villages and enemy Cities
        const SCORE_CAPTURE_CITY = HowGood.Awesome;
        const SCORE_CAPTURE_VILLAGE = HowGood.Great;
        // Capturing ruins and starfish
        const SCORE_CAPTURE_RUINS_STARFISH = HowGood.Good;
        // Penalize if unlocked a new tech and that tech hasnt been used
        const SCORE_BAD_TECH = HowGood.Worse;
        // Stanadard penalty, researching isnt free and comes with consecuenses!
        const SCORE_COST_TECH = HowGood.Bleh;
        // Reward well placed structures that require other adjacent structures
        const SCORE_GOOD_STRUCT = [HowGood.None, HowGood.Good];
        // Reward killing enemy units, the more the merrier (0 - 3 kills)
        const SCORE_KILLS = [HowGood.None, HowGood.Great, HowGood.Great + 0.1, HowGood.Awesome];
        // Penalize suicide
        const SCORE_SUICIDE = HowGood.Bad;

        // Abilities, based on how many they affect, the higher the score
        // [0] = If it didnt do anything or did worse (worse reward)
        // [1] = The max reward based on how *much* that ability did (good reward)
        const MAX_AFFECTABLE        = 5;
        const SCORE_ABILITY_BOOST   = HowGood.Great;
        const SCORE_ABILITY_EXPLODE = [HowGood.Bad, HowGood.Great];
        // Heal others
        const SCORE_ABILITY_HEAL    = HowGood.Great;
        // When not in territory, reward is halved
        const SCORE_ABILITY_RECOVER = HowGood.Good;
        // Generally not good to disband, unless unit is low on health and its pretty much pointless
        // If health porcentage is less than 40%, then its Meh to disband
        // Also special units yield Bad reward
        const SCORE_ABILITY_DISBAND_TRHRESHOLD = 0.4;
        const SCORE_ABILITY_DISBAND = [HowGood.Bad, HowGood.Meh];
        // Default reward for untracked abilities
        const SCORE_ABILITY_DEFAULT = HowGood.Meh;

        // Penzalize units that are not doing anything??
        // Penalize: capture city -> research?
        // because capturing a (village or city) increases all tech cost dramatically

        const increasedProduction = () => {
            const a = newTribe._cities.reduce((a, b) => a + getCityProduction(newState, b), 0);
            const b = oldTribe._cities.reduce((a, b) => a + getCityProduction(oldState, b), 0);
            return a > b? HowGood.Great : a < b? HowGood.Worse : HowGood.None;
        }

        // Many different moves can trigger pop increase
        let reward = increasedProduction() + (rewardMove? HowGood.Meh : HowGood.None);

        // Tiny turn cost
        reward -= 0.0001;

        // TODO
        switch (move.moveType) {
            case MoveType.Research:
                const settings = TechnologySettings[move.type as keyof typeof TechnologySettings];
                // Order of importance: Resource -> Structure (except temples) -> Ability -> Unit -> Structure (temples)
                // NOTE Using this order to reduce bad bias and potentially better moves, not sure tho..
                const cost = (
                    settings.unlocksResource? ResourceSettings[settings.unlocksResource] :
                    settings.unlocksStructure && !isTempleStructure(settings.unlocksStructure)? StructureSettings[settings.unlocksStructure] :
                    settings.unlocksUnit? UnitSettings[settings.unlocksUnit] :
                    settings.unlocksAbility? { cost: 0 } :
                    settings.unlocksStructure && isTempleStructure(settings.unlocksStructure)? StructureSettings[settings.unlocksStructure] : 
                    null
                )?.cost || 0;
                reward += cost && newTribe._stars < cost? SCORE_BAD_TECH : SCORE_COST_TECH;
                break;

            case MoveType.Build:
                const adjStructTypes = StructureSettings[move.type as StructureType].adjacentTypes;
                if(adjStructTypes) {
                    const aroundStructs = getNeighborIndexes(newState, move.src, 1).filter(
                        x => newState.structures[x] && adjStructTypes.includes(newState.structures[x].id)
                    );
                    const perc = aroundStructs.length / MAX_AFFECTABLE; 
                    reward += SCORE_GOOD_STRUCT[0] + (SCORE_GOOD_STRUCT[1] - SCORE_GOOD_STRUCT[0]) * perc;
                }
                break;

            case MoveType.Capture:
                reward += CaptureType.City? SCORE_CAPTURE_CITY : 
                    CaptureType.Village? SCORE_CAPTURE_VILLAGE : SCORE_CAPTURE_RUINS_STARFISH;
                break;
            
            case MoveType.Attack:
                const casualties = newTribe._casualties - oldTribe._casualties;
                reward += casualties > 0? SCORE_SUICIDE : SCORE_KILLS[Math.min(3, newTribe._kills - oldTribe._kills)];
                break;
            
            case MoveType.Ability:
                switch (move.type as AbilityType) {
                    case AbilityType.Boost:
                        const boosted = getNeighborIndexes(newState, move.src).filter(x => getUnitAtTile(oldState, x));
                        reward += SCORE_ABILITY_BOOST * (boosted.length / MAX_AFFECTABLE);
                        break;
                    case AbilityType.Explode:
                        const poisoned = getNeighborIndexes(newState, move.src).filter(x => {
                            if(!getEnemyAtTile(oldState, x)) return false;
                            // Enemy got killed by the poison
                            if(!getEnemyAtTile(newState, x)) {
                                reward += HowGood.Good;
                            }
                            // Enemy got affected
                            return true;
                        });
                        // Explore worth value
                        reward += poisoned? SCORE_ABILITY_EXPLODE[1] * (poisoned.length / MAX_AFFECTABLE) : SCORE_ABILITY_EXPLODE[0];
                        break
                    case AbilityType.HealOthers:
                        const healed = getNeighborIndexes(newState, move.src).filter(x => getUnitAtTile(oldState, x));
                        reward += SCORE_ABILITY_HEAL * (healed.length / MAX_AFFECTABLE);
                        break;
                    case AbilityType.Recover:
                        const isInTerritory = newState.settings._pov == newState.tiles[move.src]._owner;
                        reward += SCORE_ABILITY_RECOVER * (isInTerritory? 1 : 0.5);
                        break;
                    case AbilityType.Disband:
                        const unit = getUnitAtTile(oldState, move.src)!;
                        const perc = unit._health / getMaxHealth(unit);
                        reward += getRealUnitSettings(unit).cost != 10 && perc > SCORE_ABILITY_DISBAND_TRHRESHOLD? SCORE_ABILITY_DISBAND[0] : SCORE_ABILITY_DISBAND[1];
                        break;
                    default:
                        reward += SCORE_ABILITY_DEFAULT;
                        break;
                }
                break;

            case MoveType.Step:
                break;

            case MoveType.Summon:
                break;

            case MoveType.Harvest:
                break;

            case MoveType.Reward:
                break;

            case MoveType.EndTurn:
                break;
        }

    
        // Bonus connecting cities
        // reward += 0.0001 * (
        //     newTribe._cities.reduce((a, b) => a + (b._connectedToCapital? 1 : 0), 0) -
        //     oldTribe._cities.reduce((a, b) => a + (b._connectedToCapital? 1 : 0), 0));
        // // Bonus capturing enemy cities
        // if(move?.moveType == MoveType.Capture && oldState.tiles[move.src]._owner > 0 && oldState.tiles[move.src]._owner != newState.tiles[move.src]._owner) {
        //     reward += 0.01;
        // }

        return Math.max(-.8, Math.min(.8, reward));
    }

    static calculatePotential(state: GameState, maxReward=1.0): number {
        // ! Assumes 1v1 and FOW is disabled
        // Uses absolutes that will lead to better biases

        const pov = getPovTribe(state);
        const [ enemyPov ] = Object.values(state.tribes).filter(x => x.owner != pov.owner);

        // Army strength
        // Super units are worth x3.0
        // With a maximum advantage strength of 5 = 0.25
        // TODO unit value relative to health
        const unit_diff = 0.05 * Math.min(5, 
            pov._units.reduce((a, b) => a + (getRealUnitSettings(b).cost == 10? 3 : 1), 0) - 
            enemyPov._units.reduce((a, b) => a + (getRealUnitSettings(b).cost == 10? 3 : 1), 0)
        );
        
        // Cities Production
        // With a maximum advantage of 20 SPT (stars per turn) = 0.6
        const production_diff = 0.03 * Math.min(20,
            pov._cities.reduce((x, y) => x + getCityProduction(state, y), 0) - 
            enemyPov._cities.reduce((x, y) => x + getCityProduction(state, y), 0)
        );
        // 

        // MAX = 0.6 + 0.25 = 0.85
        const reward = unit_diff + production_diff;

        return Math.max(-maxReward, Math.min(maxReward, reward)) / 5;
    }
}