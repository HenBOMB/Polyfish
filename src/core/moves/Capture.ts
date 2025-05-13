import { getCityAt, getHomeCity, getNeighborIndexes, getNextTech, getPovTribe, getUnitAt, hasEffect, indexToCoord, isTechUnlocked } from "../functions";
import Move, { CallbackResult, UndoCallback } from "../move";
import { EffectType, MoveType, RewardType, StructureType, TechnologyType, TerrainType, TribeType, UnitType } from "../types";
import { CityState, GameState, TechnologyState } from "../states";
import { addPopulationToCity, discoverTiles, spendStars, removeUnit, summonUnit, gainStars, consumeResource, createStructure, destroyStructure, modifyTerrain, tryRemoveEffect } from "../actions";
import { TechnologyUnlockableList } from "../settings/TechnologySettings";
import { predictExplorer } from "../../eval/prediction";
import { xorCity, xorPlayer, xorTile } from "../../zorbist/hasher";

export default class Capture extends Move {
    constructor(src: number) {
        super(MoveType.Capture, src, null, null);
    }
    
    execute(state: GameState): CallbackResult {
        const tile = state.tiles[this.getSrc()];
        const struct = state.structures[this.getSrc()];
        const capturer = getUnitAt(state, this.getSrc())!;
        const pov = getPovTribe(state);

        const rewards = [];
        let undo: UndoCallback = () => { };
        
        capturer._moved = capturer._attacked = true;
        
        if(struct) {
            if(struct.id == StructureType.Village) {
                const oldCity = getHomeCity(state, capturer);
                
                if(oldCity) {
                    xorCity.unitCount(state, oldCity, oldCity._unitCount, oldCity._unitCount - 1);
                    capturer._homeIndex = capturer._tileIndex;
                    oldCity._unitCount--;
                }

                const result = (tile._owner? this.city(state) : this.village(state))!;
                rewards.push(...result.rewards);

                undo = () => {
                    result.undo();

                    if(oldCity) {
                        xorCity.unitCount(state, oldCity, oldCity._unitCount, oldCity._unitCount + 1);
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
        
        const undoBoost = tryRemoveEffect(state, capturer, EffectType.Boost);
        
        return {
            rewards,
            undo: () => {
                undoBoost();
                undo();
                capturer._moved = capturer._attacked = false;
            },
        }
    }
    
    village(state: GameState): CallbackResult {
        const pov = getPovTribe(state);        
        const captureIndex = this.getSrc();
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
            _rewards: new Set(),
            _territory: territory,
            _unitCount: 1,
        };

        xorCity.set(state, createdCity);
        
        pov._cities.push(createdCity);
        villageTile._owner = pov.owner;
        xorTile.owner(state, villageTile.tileIndex, 0, pov.owner);
        
        for(const tileIndex of createdCity._territory) {
            const tile = state.tiles[tileIndex];
            tile._owner = pov.owner;
            tile._rulingCityIndex = villageTile.tileIndex;
            xorTile.owner(state, villageTile.tileIndex, 0, pov.owner);
        }
        
        return {
            rewards: [],
            undo: () => {
                for(const tileIndex of createdCity._territory) {
                    const tile = state.tiles[tileIndex];
                    tile._owner = 0;
                    tile._rulingCityIndex = -1;
                    xorTile.owner(state, tileIndex, pov.owner, 0);
                }

                xorTile.owner(state, villageTile.tileIndex, pov.owner, 0);

                villageTile._owner = 0;
                pov._cities.pop();

                xorCity.set(state, createdCity);
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
        
        xorCity.owner(state, city, enemy.owner, pov.owner);
        xorTile.owner(state, city.tileIndex, enemy.owner, pov.owner);

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
            xorTile.owner(state, tile.tileIndex, enemy.owner, pov.owner);
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
                    xorTile.owner(state, tile.tileIndex, pov.owner, enemy.owner);
                }

                if(tile.capitalOf > 0) tile.capitalOf = enemy.owner;
                tile._owner = enemy.owner;
                city._owner = enemy.owner;
                city.name = cityName;
                pov._cities.pop();
                enemy._cities.splice(cityListIndex, 0, city);

                xorTile.owner(state, city.tileIndex, pov.owner, enemy.owner);
                xorCity.owner(state, city, pov.owner, enemy.owner);
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
            const undoStars = gainStars(state, 5);
            return {
                rewards: [],
                undo: undoStars
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

                xorPlayer.tech(pov, scroll.techType);
                pov._tech.push(scroll);
                const undoPurchase = spendStars(state, -5);

                return {
                    rewards: [],
                    undo: () => {
                        undoPurchase();
                        pov._tech.pop();
                        xorPlayer.tech(pov, scroll.techType);
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
            if(around.some(x => state._visibleTiles[x])) {
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
            summoned._kills = 3;

            return summon;
        });

        // spawns a level 3 city with a city wall and 4 adjacent shallow water tiles	
        if(pov.tribeType == TribeType.Aquarion && terrainType === TerrainType.Ocean) {
            possibleRewards.push(() => {
                const createdCity: CityState = {
                    name: `${TribeType[pov.tribeType]} City`,
                    _population: 2,
                    _progress: 0,
                    _rewards: new Set([RewardType.Explorer, RewardType.CityWall]),
                    _borderSize: 1,
                    _connectedToCapital: false,
                    _level: 3,
                    _production: 3,
                    _owner: pov.owner,
                    tileIndex,
                    _territory: getNeighborIndexes(state, tileIndex, 1, true, true),
                    _unitCount: 0,
                };

                xorCity.set(state, createdCity);

                const undoCreate = createStructure(state, tileIndex, StructureType.Village);

                const chain: UndoCallback[] = [];
                [
                    tileIndex + 1,
                    tileIndex - 1,
                    tileIndex + state.settings.size,
                    tileIndex - state.settings.size,
                ].forEach(index => {
                    const [x, y] = indexToCoord(state, index);
                    if (x < 0 || x >= state.settings.size || y < 0 || y >= state.settings.size) {
                        return;
                    }
                    chain.push(modifyTerrain(state, x, TerrainType.Water));
                });
    
                pov._cities.push(createdCity);

                // TODO recalculate network connections

                return {
                    rewards: [],
                    undo: () => {
                        pov._cities.pop();
                        chain.forEach(x => x());
                        undoCreate();
                        xorCity.set(state, createdCity);
                    }
                }
            });
        }

        const undoDestroyRuins = destroyStructure(state, tileIndex);
        
        // Capturing reveals the hidden unit
        const undoInvis = tryRemoveEffect(state, capturer, EffectType.Invisible);

        const rewardResult = possibleRewards[Math.floor(Math.random() * possibleRewards.length)]();
        
        return {
            rewards: rewardResult?.rewards || [],
            undo: () => {
                rewardResult?.undo();
                undoInvis();
                undoDestroyRuins();
                capturer._attacked = false;
                capturer._moved = false;
            }
        }
    }
    
    starfish(state: GameState): UndoCallback {
        const capturer = getUnitAt(state, this.getSrc())!;
        const undoResource = consumeResource(state, capturer._tileIndex);
        const undoStars = gainStars(state, 8);
        return () => {
            undoStars();
            undoResource();
        }
    }
}