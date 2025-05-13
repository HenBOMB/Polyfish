import { readFileSync } from "fs";
import { GameState } from "./core/states";
import { getPovTribe, getUnitAt, isResourceVisible, getMaxHealth, getCityAt, getCityProduction, getRealUnitSettings, isTempleStructure, getNeighborIndexes, getEnemyAt, isTechUnlocked, isAdjacentToEnemy, getRealUnitType, getTribeProduction, getTechSettings, hasEffect } from "./core/functions";
import { AbilityType, CaptureType, EffectType, RewardType, StructureType, TechnologyType, UnitType } from "./core/types";
import Move, { CallbackResult } from "./core/move";
import { MoveType } from "./core/types";
import { TechnologyUnlockable } from "./core/settings/TechnologySettings";
import { predictBestNextCityReward } from "./eval/prediction";
import { ResourceSettings } from "./core/settings/ResourceSettings";
import { StructureSettings } from "./core/settings/StructureSettings";
import { UnitSettings } from "./core/settings/UnitSettings";
import Game from "./game";
import { TribeTypeCount } from "./core/settings/TribeSettings";

const CAST_PRECISION = 3;

export const AbilityTypeSorted: Record<AbilityType, number> = {
    [AbilityType.None]:        -1,
    [AbilityType.Recover]:      0,
    [AbilityType.Promote]:      1,
    [AbilityType.Boost]:        2,

    [AbilityType.HealOthers]:   3,
    [AbilityType.FreezeArea]:   4,
    [AbilityType.BreakIce]:     5,
    [AbilityType.Drain]:        6,

    [AbilityType.Disband]:      7,
    [AbilityType.Explode]:      8,
    [AbilityType.BurnForest]:   9,
    [AbilityType.ClearForest]: 10,
    [AbilityType.GrowForest]:  11,
    [AbilityType.Decompose]:   12,
    [AbilityType.Destroy]:     13,
}

/** @type {ModelConfig} */
export const MODEL_CONFIG: {
    max_tribes: number;
    max_kills: number;
    max_casualties: number;
    max_production: number;
    max_score: number;
    max_cities: number;
    max_units: number;
    max_turns: number;
    max_stars: number;
    max_unit_kills: number;
    max_structure_level: number;
    
    max_tile_count: number;
    dim_map_tile: number;
    dim_map_size: number;
    dim_player: number;
    dim_ability: number;
    dim_moves: number;
    dim_tech: number;
    dim_effects: number;

    hidden_channels: number;
    res_blocks: number;
} = JSON.parse(readFileSync('data/model/config.json', 'utf8'));

MODEL_CONFIG.max_tile_count = MODEL_CONFIG.dim_map_size ** 2;

function safeArray<T>(a: T[], size: number, _default?: T): T[] {
    if(a.length > size) {
        a = a.slice(0, size);
    }
    else if(a.length < size) {
        a = a.concat(Array(size - a.length).fill(_default || 0));
    }
    return a;
}

function safeFloat(f: number | boolean | undefined): number { 
    if(f === undefined) return 0;
    if(typeof(f) === "boolean") return f? 1 : 0;
    if(typeof(f) !== "number") throw new Error("NaN: " + typeof(f));
    if(f > 1) {
        throw new Error(`Out of bounds: ${f} > 1`);
    }
    return f < 0? 0 : Number(f.toFixed(CAST_PRECISION)) 
}

function extractTile(state: GameState, tileIndex: number, force = false): number[] {
    const pov = getPovTribe(state);
    const tile = state.tiles[tileIndex];
    const explored = tile._explorers.has(pov.owner);

    if(!explored && !force) {
        return Array(MODEL_CONFIG.dim_map_tile).fill(0);
    }
    
    const city = getCityAt(state, tileIndex);
    const unit = getUnitAt(state, tileIndex);
    const resource = !city && isResourceVisible(pov, state.resources[tileIndex]?.id)? state.resources[tileIndex] : null;
    const maxUnitTypeCount = Object.keys(UnitSettings).length - 1;
   
    const isOwnedByUs = tile._owner === pov.owner;
    const isOwnedByEnemy = tile._owner && tile._owner !== pov.owner;
    const unitIsOurs = unit && unit._owner === pov.owner? unit : null;
    const unitIsEnemy = unit && unit._owner !== pov.owner? unit : null;

    return [
        // Tile (terrain, owned by us, owned by enemy, above, path)
        safeFloat(tile.terrainType / 6),
        safeFloat(isOwnedByUs),
        safeFloat(isOwnedByEnemy),
        safeFloat(resource? 1 : 0),
        safeFloat(Boolean(city) || tile.hasRoad || tile.hasRoute),

        // City (city tile, is capital, level, progress, walls, riot)
        safeFloat(city? 1 : 0),
        safeFloat(city? tile.capitalOf > 0 : 0),
        safeFloat(city? Math.min(city._level, 7) / 7 : 0),
        safeFloat(city? (city._progress / (city._level + 1)) : 0),
        safeFloat(city? city._walls : 0),
        safeFloat(city? city._riot : 0),

        // Our units
        // unit standing, is enemy, unit type, navy type, health, moved, attacked, kills, invisible, effects, cloak nearby
        safeFloat(Boolean(unitIsOurs)),
        safeFloat(unitIsOurs? getRealUnitType(unitIsOurs) / maxUnitTypeCount : 0),
        safeFloat(unitIsOurs? unitIsOurs._passenger? unitIsOurs._passenger / maxUnitTypeCount : 0 : 0),
        safeFloat(unitIsOurs? unitIsOurs._health / getMaxHealth(unitIsOurs) : 0),
        safeFloat(unitIsOurs? unitIsOurs._moved : 0),
        safeFloat(unitIsOurs? unitIsOurs._attacked : 0),
        safeFloat(unitIsOurs? Math.min(unitIsOurs._kills, 3) / 3 : 0),
        safeFloat(unitIsOurs? hasEffect(unitIsOurs, EffectType.Invisible) : 0),
        safeFloat(unitIsOurs? hasEffect(unitIsOurs, EffectType.Poison) : 0),
        safeFloat(unitIsOurs? hasEffect(unitIsOurs, EffectType.Boost) : 0),
        safeFloat(unitIsOurs? hasEffect(unitIsOurs, EffectType.Frozen) : 0),
        // Hidden enemy[cloak] nearby (its invisible to us but we can still detect it via [Skilltype.Detect])
        safeFloat(unitIsOurs? isAdjacentToEnemy(state, tile, UnitType.Cloak) : 0),

        // Enemy units
        safeFloat(Boolean(unitIsEnemy)),
        safeFloat(unitIsEnemy? getRealUnitType(unitIsEnemy) / maxUnitTypeCount : 0),
        safeFloat(unitIsEnemy? unitIsEnemy._passenger? unitIsEnemy._passenger / maxUnitTypeCount : 0 : 0),
        safeFloat(unitIsEnemy? unitIsEnemy._health / getMaxHealth(unitIsEnemy) : 0),
        safeFloat(unitIsEnemy? unitIsEnemy._moved : 0),
        safeFloat(unitIsEnemy? unitIsEnemy._attacked : 0),
        safeFloat(unitIsEnemy? Math.min(unitIsEnemy._kills, 3) / 3 : 0),
        safeFloat(unitIsEnemy? hasEffect(unitIsEnemy, EffectType.Invisible) : 0),
        safeFloat(unitIsEnemy? hasEffect(unitIsEnemy, EffectType.Poison) : 0),
        safeFloat(unitIsEnemy? hasEffect(unitIsEnemy, EffectType.Boost) : 0),
        safeFloat(unitIsEnemy? hasEffect(unitIsEnemy, EffectType.Frozen) : 0),
    ];
}

function extractMap(state: GameState): number[][][] {
    const offset = Math.floor((MODEL_CONFIG.dim_map_size - state.settings.size) / 2);
    const grid: number[][][] = Array(MODEL_CONFIG.dim_map_tile).fill(0).map(() =>
        Array(MODEL_CONFIG.dim_map_size).fill(0).map(() => Array(MODEL_CONFIG.dim_map_size).fill(0))
    );
    for (const i in state.tiles) {
        const tileIndex = Number(i);
        const tile = state.tiles[tileIndex];
        const gridY = tile.y + offset;
        const gridX = tile.x + offset;
        if (gridY >= 0 && gridY < MODEL_CONFIG.dim_map_size && gridX >= 0 && gridX < MODEL_CONFIG.dim_map_size) {
            const result = extractTile(state, tileIndex);
            for (let c = 0; c < MODEL_CONFIG.dim_map_tile; c++) {
                if (grid[c] && grid[c][gridY]) {
                    grid[c][gridY][gridX] = result[c];
                } else {
                    throw new Error(`Warning: Attempted to write to invalid grid coordinates or channel: [${c}][${gridY}][${gridX}]`);
                }
            }
        } 
        else {
            throw new Error(`Warning: Tile coordinates (${tile.x}, ${tile.y}) are outside the padded grid size ${MODEL_CONFIG.dim_map_size} after applying offset ${offset}.`);
        }
    }
    return grid;
}

function extractPlayer(state: GameState): number[] {
    const pov = getPovTribe(state);
    const [ enemy ] = Object.values(state.tribes).filter(x => x.owner != pov.owner);
    // TODO Assumed no FOW
    return [
        // Us (tribetype, stars, spt, kills, casualties, unit count, city count, ...tech[0/1])
        pov.tribeType / TribeTypeCount,
        Math.min(pov._stars, MODEL_CONFIG.max_stars) / MODEL_CONFIG.max_stars,
        Math.min(getTribeProduction(state, pov), MODEL_CONFIG.max_production) / MODEL_CONFIG.max_production,
        Math.min(pov._kills, MODEL_CONFIG.max_kills) / MODEL_CONFIG.max_kills,
        Math.min(pov._casualties, MODEL_CONFIG.max_casualties) / MODEL_CONFIG.max_casualties,
        Math.min(pov._units.length, MODEL_CONFIG.max_units) / MODEL_CONFIG.max_units,
        Math.min(pov._cities.length, MODEL_CONFIG.max_cities) / MODEL_CONFIG.max_cities,
        ...Object.keys(TechnologyUnlockable).map(x => isTechUnlocked(pov, Number(x))? 1 : 0),
        // TODO Optional later on for enemy:
        // average score, tech
        // Them (tribetype, spt, unit count, city count)
        enemy.tribeType / TribeTypeCount,
        Math.min(getTribeProduction(state, enemy), MODEL_CONFIG.max_production) / MODEL_CONFIG.max_production,
        Math.min(enemy._units.length, MODEL_CONFIG.max_units) / MODEL_CONFIG.max_units,
        Math.min(enemy._cities.length, MODEL_CONFIG.max_cities) / MODEL_CONFIG.max_cities,
    ].map(x => safeFloat(x));
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
        const city = getPovTribe(state)._cities.find(x => moves.some(y => y.getSrc() == x.tileIndex));
        const rewardType = predictBestNextCityReward(state, city)[0];
        const move = moves.find(x => x.getType<RewardType>() == rewardType)!;
        return move.execute(state);
    }

    /**
     * Returns a value between -0.75 and 0.75. Uses game.stateBefore and game.state
     * @param moves Moves recently played
     * @returns 
     */
    static calculateReward(oldState: GameState, game: Game, ...moves: Move[]): number {
        const move: Move = moves[0];
        // The forced reward move, caused by upgrading a city
        const rewardMove: Move = moves[1];

        // ! ASSUMES NO FOW
        const newState = game.state;

        const pov = newState.settings._pov;
        const oldTribe = oldState.tribes[pov];
        const newTribe = newState.tribes[pov];

        // LOSS = -1
        // ... EVERYTHING ELSE IN BETWEEN ...
        // WIN = 1

        enum HowGood {
            // Defeat  = -1.00,
            // Nasty   = -0.75,
            Worse    = -0.5,
            Bad      = -0.25,
            Bleh     = -0.10,
            None     = 0.00,
            Meh      = 0.10,
            Good     = 0.25,
            VeryGood = 0.40,
            Nice     = 0.60,
            Great    = 0.90,
            Awesome  = 2.00,
            // Victory = 1.00,
        };

        // Capturing Villages and enemy Cities
        const SCORE_CAPTURE_CITY = HowGood.Awesome;
        const SCORE_CAPTURE_VILLAGE = HowGood.Great;
        // Capturing ruins and starfish
        const SCORE_CAPTURE_RUINS_STARFISH = HowGood.VeryGood;
        // Penalize if unlocked a new tech and that tech hasnt been used
        const SCORE_BAD_TECH = HowGood.Worse;
        // Stanadard penalty, researching isnt free and comes with consecuenses!
        const SCORE_COST_TECH = HowGood.Bleh;
        // Reward well placed structures that require other adjacent structures
        const SCORE_GOOD_STRUCT = [HowGood.None, HowGood.VeryGood];
        // Reward killing enemy units, the more the merrier (0 - 3 kills)
        const SCORE_KILLS = [HowGood.None, HowGood.Good, HowGood.Nice];
        // Penalize suicide
        const SCORE_SUICIDE = HowGood.Bad;

        // Abilities, based on how many they affect, the higher the score
        // [0] = If it didnt do anything or did worse (worse reward)
        // [1] = The max reward based on how *much* that ability did (good reward)
        const MAX_AFFECTABLE        = 5;
        const SCORE_ABILITY_BOOST   = HowGood.VeryGood;
        const SCORE_ABILITY_EXPLODE = [HowGood.Bad, HowGood.VeryGood];
        // Heal others
        const SCORE_ABILITY_HEAL    = HowGood.VeryGood;
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
            return a > b? HowGood.Nice : a < b? HowGood.Worse : HowGood.None;
        }

        // Many different moves can trigger pop increase
        let reward = increasedProduction() + (rewardMove? HowGood.Meh : HowGood.None);

        // Tiny turn cost
        reward -= 0.0001;

        // TODO
        switch (move.moveType) {
            case MoveType.Research:
                const settings = getTechSettings(move.getType<TechnologyType>());
                // Order of importance: Resource -> Structure (except temples) -> Ability -> Unit -> Structure (temples)
                // NOTE Using this order to reduce bad bias and potentially better moves, not sure tho..
                const cost = (
                    settings.unlocksResource? ResourceSettings[settings.unlocksResource] :
                    settings.unlocksStructure && !isTempleStructure(settings.unlocksStructure)? StructureSettings[settings.unlocksStructure] :
                    settings.unlocksUnit? UnitSettings[settings.unlocksUnit] :
                    settings.unlocksStructure && isTempleStructure(settings.unlocksStructure)? StructureSettings[settings.unlocksStructure] : 
                    null
                )?.cost || 0;
                reward += cost && newTribe._stars < cost? SCORE_BAD_TECH : SCORE_COST_TECH;
                if(settings.unlocksUnit) {
                    reward += HowGood.Meh;
                }
                if(settings.unlocksAbility) {
                    reward += HowGood.Meh;
                }
                if(settings.unlocksResource) {
                    reward += HowGood.Meh;
                }
                if(settings.unlocksStructure) {
                    reward += HowGood.Meh;
                }
                if(settings.unlocksOther) {
                    reward += HowGood.Meh;
                }
                break;

            case MoveType.Build:
                const adjStructTypes = StructureSettings[move.getType<StructureType>()].adjacentTypes;
                if(adjStructTypes) {
                    const aroundStructs = getNeighborIndexes(newState, move.getSrc(), 1).filter(
                        x => newState.structures[x] && adjStructTypes.has(newState.structures[x].id)
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
                switch (move.getType<AbilityType>()) {
                    case AbilityType.Boost:
                        const boosted = getNeighborIndexes(newState, move.getSrc()).filter(x => getUnitAt(oldState, x));
                        reward += SCORE_ABILITY_BOOST * (boosted.length / MAX_AFFECTABLE);
                        break;
                    case AbilityType.Explode:
                        const poisoned = getNeighborIndexes(newState, move.getSrc()).filter(x => {
                            if(!getEnemyAt(oldState, x)) return false;
                            // Enemy got killed by the poison
                            if(!getEnemyAt(newState, x)) {
                                reward += HowGood.Good;
                            }
                            // Enemy got affected
                            return true;
                        });
                        // Explore worth value
                        reward += poisoned? SCORE_ABILITY_EXPLODE[1] * (poisoned.length / MAX_AFFECTABLE) : SCORE_ABILITY_EXPLODE[0];
                        break
                    case AbilityType.HealOthers:
                        const healed = getNeighborIndexes(newState, move.getSrc()).filter(x => getUnitAt(oldState, x));
                        reward += SCORE_ABILITY_HEAL * (healed.length / MAX_AFFECTABLE);
                        break;
                    case AbilityType.Recover:
                        const isInTerritory = newState.settings._pov == newState.tiles[move.getSrc()]._owner;
                        reward += SCORE_ABILITY_RECOVER * (isInTerritory? 1 : 0.5);
                        break;
                    case AbilityType.Disband:
                        const unit = getUnitAt(oldState, move.getSrc())!;
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
            pov._units.reduce((a, b) => a + (getRealUnitSettings(b).super? 3 : 1), 0) - 
            enemyPov._units.reduce((a, b) => a + (getRealUnitSettings(b).super? 3 : 1), 0)
        );
        
        // Cities Production
        // With a maximum advantage of 20 SPT (stars per turn) = 0.6
        const production_diff = 0.03 * Math.min(20,
            pov._cities.reduce((x, y) => x + getCityProduction(state, y), 0) - 
            enemyPov._cities.reduce((x, y) => x + getCityProduction(state, y), 0)
        );

        // MAX = 0.6 + 0.25 = 0.85
        const reward = unit_diff + production_diff;

        return Math.max(-maxReward, Math.min(maxReward, reward)) / 5;
    }

    static assertConfig(state: GameState): void {
        const maxTechTypeCount = Object.keys(TechnologyUnlockable).length;
        const maxAbilityTypeCount = Object.values(TechnologyUnlockable).filter(x => x.unlocksAbility).length;
        const tileCount = state.settings.size ** 2;

        if(maxAbilityTypeCount > MODEL_CONFIG.dim_ability) {
            throw new Error(`Ability count mismatch: ${MODEL_CONFIG.dim_ability} < ${maxAbilityTypeCount}`);
        }

        if(tileCount > MODEL_CONFIG.max_tile_count) {
            throw new Error(`Tile count mismatch: ${MODEL_CONFIG.max_tile_count} < ${tileCount}`);
        }

        if(maxTechTypeCount !== MODEL_CONFIG.dim_tech) {
            throw new Error(`Tech type count mismatch: ${MODEL_CONFIG.dim_tech} -> ${maxTechTypeCount}`);
        }

        const tileChannels = extractTile(state, 0, true);

        if(tileChannels.length !== MODEL_CONFIG.dim_map_tile) {
            console.log(extractTile(state, 0, true));
            throw new Error(`Tile channel count mismatch: ${MODEL_CONFIG.dim_map_tile} -> ${tileChannels.length}`);
        }

        const map = extractMap(state);

        if(map.length !== MODEL_CONFIG.dim_map_tile) {
            throw new Error(`Map channel count mismatch: ${MODEL_CONFIG.dim_map_tile} -> ${map.length}`);
        }

        // why tile channels same as settings channelsdASDAd
        // mega coincidence

        const playerChannels = extractPlayer(state);

        if(playerChannels.length !== MODEL_CONFIG.dim_player) {
            console.log(extractPlayer(state));
            throw new Error(`Player channel count mismatch: ${MODEL_CONFIG.dim_player} -> ${playerChannels.length}`);
        }
    }
}