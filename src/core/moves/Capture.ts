import { Logger } from "../../polyfish/logger";
import { getCityAt, getHomeCity, getNeighborIndexes, getNextTech, getPovTribe, getUnitAt, indexToCoord, isTechUnlocked } from "../functions";
import Move, { CallbackResult, UndoCallback } from "../move";
import { EffectType, MoveType, RewardType, StructureType, TechnologyType, TerrainType, TribeType, UnitType } from "../types";
import { CityState, GameState, TechnologyState } from "../states";
import { addPopulationToCity, discoverTiles, removeUnit, summonUnit } from "../actions";
import { TechnologyUnlockableList } from "../settings/TechnologySettings";
import { predictExplorer } from "../../eval/prediction";

export default class Capture extends Move {
    constructor(src: number) {
        super(MoveType.Capture, src, null, null);
    }
    
    execute(state: GameState): CallbackResult {
        const tile = state.tiles[this.getSrc()];
        const struct = state.structures[this.getSrc()];
        const capturer = getUnitAt(state, this.getSrc())!;
        
        const rewards = [];
        let undo: UndoCallback = () => { };
        
        capturer._moved = capturer._attacked = true;
        
        if(struct) {
            if(struct.id == StructureType.Village) {
                const oldCity = getHomeCity(state, capturer);
                
                if(oldCity) {
                    capturer._homeIndex = capturer._tileIndex;
                    oldCity._unitCount--;
                }
                
                const result = (tile._owner > 0? this.city(state) : this.village(state))!;
                rewards.push(...result.rewards);

                undo = () => {
                    result.undo();
                    if(oldCity) {
                        oldCity._unitCount++;
                        capturer._homeIndex = oldCity.tileIndex;
                    }
                }
            }
            else {
                const result = this.ruins(state)!;
                rewards.push(...result.rewards);
                undo = result.undo;
            }
        }
        else {
            undo = this.starfish(state);
        }
        
        return {
            rewards,
            undo: () => {
                undo();
                capturer._moved = capturer._attacked = false;
            },
        }
    }
    
    village(state: GameState): CallbackResult {
        const pov = getPovTribe(state);        
        const captureIndex = this.getTarget();
        const villageTile = state.tiles[captureIndex];
        const territory = getNeighborIndexes(state, captureIndex, 1, true, true);
        
        const createdCity: CityState = {
            name: `${TribeType[pov.tribeType]} City`,
            _population: 0,
            _progress: 0,
            _borderSize: 1,
            _connectedToCapital: false,
            _level: 1,
            _production: 1,
            _owner: pov.owner,
            tileIndex: captureIndex,
            _rewards: [],
            _territory: territory,
            _unitCount: 1,
        };
        
        pov._cities.push(createdCity);
        villageTile._owner = pov.owner;
        for(const tileIndex of createdCity._territory) {
            const tile = state.tiles[tileIndex];
            tile._owner = pov.owner;
            tile._rulingCityIndex = villageTile.tileIndex;
        }
        
        return {
            rewards: [],
            undo: () => {
                for(const tileIndex of createdCity._territory) {
                    const tile = state.tiles[tileIndex];
                    tile._owner = -1;
                    tile._rulingCityIndex = -1;
                }
                villageTile._owner = -1;
                pov._cities.pop();
            }
        }
    }
    
    city(state: GameState): CallbackResult {
        const capturer = getUnitAt(state, this.getSrc())!;
        const pov = getPovTribe(state);
        const city = getCityAt(state, capturer._tileIndex)!;
        const tile = state.tiles[city.tileIndex];
        const enemy = state.tribes[city._owner];
        const cityName = city.name;
        
        // TODO enemyCity.progress neg population logic (also on unit death it should add if already neg)
        
        const cityListIndex = enemy._cities.indexOf(city);
        
        // Claim the enemy's city
        enemy._cities.splice(cityListIndex, 1)
        pov._cities.push(city);
        city.name = `${TribeType[pov.tribeType]} ${tile.capitalOf > 0? 'Capital' : 'City'}`;
        city._owner = pov.owner;
        tile._owner = pov.owner;
        if(tile.capitalOf > 0) tile.capitalOf = pov.owner;
        
        // Claim the enemy's territory
        for(let i = 0; i < city._territory.length; i++) {
            const tile = state.tiles[city._territory[i]];
            tile._owner = pov.owner;
        }
        
        // If enemy runs out of cities they loose all their units
        const chain: UndoCallback[] = [];
        if(!enemy._cities.length) {
            enemy._killedTurn = state.settings._turn;
            enemy._killerId = pov.owner;
            for(const unit of enemy._units) {
                chain.push(removeUnit(state, unit));
            }
        }
        
        // Update networks
        
        return {
            rewards: [],
            undo: () => {
                if(!enemy._cities.length) {
                    chain.reverse().forEach(x => x());
                    enemy._killerId = -1;
                    enemy._killedTurn = -1;
                }
                
                for(let i = 0; i < city._territory.length; i++) {
                    const tile = state.tiles[city._territory[i]];
                    tile._owner = enemy.owner;
                }
                
                if(tile.capitalOf > 0) tile.capitalOf = enemy.owner;
                tile._owner = enemy.owner;
                city._owner = enemy.owner;
                city.name = cityName;
                pov._cities.pop();
                enemy._cities.splice(cityListIndex, 0, city);
            }
        }
    }
    
    ruins(state: GameState): CallbackResult {
        const capturer = getUnitAt(state, this.getSrc())!;
        const pov = getPovTribe(state);
        const possibleRewards: (() => CallbackResult)[] = [];
        const tileIndex = capturer._tileIndex;
        
        capturer._attacked = true;
        capturer._moved = true;

        // free 5 stars
        possibleRewards.push(() => {
            pov._stars += 5;
            return {
                rewards: [],
                undo: () => {
                    pov._stars -= 5;
                }
            }
        });

        // free tech if tech tree is incomplete
        const scrolls: TechnologyType[] = TechnologyUnlockableList.filter(x => getNextTech(x)?.some(x => !isTechUnlocked(pov, x)))
        
        if (scrolls.length) {
            possibleRewards.push(() => {
                const scroll: TechnologyState = {
                    techType: scrolls[Math.floor(Math.random() * scrolls.length)],
                    discovered: state.settings.areYouSure,
                };
                pov._tech.push(scroll);
                pov._stars += 5;
                return {
                    rewards: [],
                    undo: () => {
                        pov._stars -= 5;
                        pov._tech.pop();
                        if(state.settings.areYouSure) {
                            scroll.discovered = false;
                        }   
                    }
                }
            });
        }

        // 3 free pop to highest level capital
        const city: CityState | null = pov._cities
            .filter(x => state.tiles[x.tileIndex].capitalOf > 0)
            .sort((a, b) => a._production - b._production)[0] || null;
        if(city) {
            possibleRewards.push(() => addPopulationToCity(state, city, 3));
        }

        // free explorer if 5x5 adj area is unexplored
        // note: cymanti cannot get explorers from water tiles
        const terrainType = state.tiles[tileIndex].terrainType;
        if(
            terrainType !== TerrainType.Mountain &&
            (
                pov.tribeType !== TribeType.Cymanti || 
                (terrainType !== TerrainType.Ocean && pov.tribeType === TribeType.Cymanti)
            )
        ) {
            const around = getNeighborIndexes(state, tileIndex, 2, true);
            if(around.some(x => state._visibleTiles.includes(x))) {
                possibleRewards.push(() => discoverTiles(state, null, predictExplorer(state, tileIndex)));
            }
        }

        // free veteran swordsman or free rammer (if on ocean tile)
        possibleRewards.push(() => {
            const summon = summonUnit(
                state, 
                terrainType !== TerrainType.Ocean? UnitType.Swordsman : UnitType.Rammer, 
                tileIndex, 
                false, true
            )!;

            const summoned = pov._units[pov._units.length-1];

            summoned.veteran = true;
            summoned.kills = 3;

            return summon;
        });

        // spawns a level 3 city with a city wall and 4 adjacent shallow water tiles	
        if(pov.tribeType == TribeType.Aquarion && terrainType === TerrainType.Ocean) {
            possibleRewards.push(() => {
                const cityData: CityState = {
                    name: `${TribeType[pov.tribeType]} City`,
                    _population: 2,
                    _progress: 0,
                    _rewards: [RewardType.Explorer, RewardType.CityWall],
                    _borderSize: 1,
                    _connectedToCapital: false,
                    _level: 3,
                    _production: 3,
                    _owner: pov.owner,
                    tileIndex,
                    _territory: getNeighborIndexes(state, tileIndex, 1, true, true),
                    _unitCount: 0,
                };

                const oldStruct = state.structures[tileIndex];
    
                state.structures[tileIndex] = {
                    id: StructureType.Village,
                    _level: cityData._level,
                    turn: state.settings._turn,
                    reward: 0,
                    tileIndex,
                }
    
                const adjWaterTiles = [
                    tileIndex + 1,
                    tileIndex - 1,
                    tileIndex + state.settings.size,
                    tileIndex - state.settings.size,
                ].filter(index => {
                    const [x, y] = indexToCoord(state, index);
                    if (x < 0 || x >= state.settings.size || y < 0 || y >= state.settings.size) {
                        return false;
                    }
                    return true;
                });
    
                adjWaterTiles.forEach((x, i) => {
                    const old = state.tiles[x].terrainType;
                    state.tiles[x].terrainType = TerrainType.Water;
                    adjWaterTiles[i] =- old;
                });
                
                pov._cities.push(cityData);

                // TODO recalculate network connections

                return {
                    rewards: [],
                    undo: () => {
                        pov._cities.pop();
                        adjWaterTiles.forEach((x) => {
                            state.tiles[x].terrainType = x;
                        });
                        state.structures[tileIndex] = oldStruct;
                    }
                }
            });
        }

        const ruins = state.structures[tileIndex];
        delete state.structures[tileIndex];
        
        // Reveal hidden unit
        const effectIndex = capturer._effects.indexOf(EffectType.Invisible);

        if(effectIndex !== -1) {
            capturer._effects.splice(effectIndex, 1);
        }

        const rewardResult = possibleRewards[Math.floor(Math.random() * possibleRewards.length)]();
        
        return {
            rewards: rewardResult?.rewards || [],
            undo: () => {
                rewardResult?.undo();

                if(effectIndex !== -1) {
                    capturer._effects.splice(effectIndex, 0, EffectType.Invisible);
                }
    
                state.structures[tileIndex] = ruins;
                
                capturer._attacked = false;
                capturer._moved = false;
            }
        }
    }
    
    starfish(state: GameState): UndoCallback {
        const capturer = getUnitAt(state, this.getSrc())!;
        
        return () => {

        }
    }
    
    safeguard(state: GameState): 1 | null {
        const unit = getUnitAt(state, this.getSrc());
        
        if(!unit) {
            return Logger.illegal(MoveType.None, `Unit does not exist: ${this.getSrc()}`);
        }
        
        return 1;
    }
}