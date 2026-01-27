import { getCityAt, getHomeCity, getAdjacentIndexes, getNextTech, getPovTribe, getUnitAt, isTechUnlocked } from "../functions";
import Move, { Branch, CallbackResult, UndoCallback } from "../move";
import { EffectType, MoveType, RewardType, StructureType, TechnologyType, TerrainType, TribeType, UnitType } from "../types";
import { CityState, Coords, GameState, TechnologyState } from "../states";
import { spendStars, gainStars, modifyTerrain, tryRemoveEffect, endUnitTurn } from "../actions";
import consumeResource from "../actions/resource/Consume";
import addPopulationToCity from "../actions/AddPopulation";
import unlockTechnology from "../actions/UnlockTech";
import claimTerritory from "../actions/ClaimTerritory";
import { destroyStructure } from "../actions/structure/Destroy";
import { createStructure } from "../actions/structure/Create";
import { discoverTiles } from "../actions/DiscoverTiles";
import removeUnit from "../actions/units/Remove";
import summonUnit from "../actions/units/Summon";
import { TechnologyUnlockableList } from "../settings/TechnologySettings";
import { predictExplorer } from "../../ai/prediction";
import { xorCity, xorTile, xorUnit } from "../../zobrist/hasher";
import { GMath } from "../../ai/gmath";

export default class Capture extends Move {
    constructor(src: number) {
        super(MoveType.Capture, src, null, null);
    }
    
    execute(state: GameState): Branch {
        const tile = state.tiles[this.getSrc()];
        const struct = state.structures[this.getSrc()];
        // TODO bug here, capturer is null
        // means the legal gen is not working right or sum is up
        const capturer = getUnitAt(state, this.getSrc())!;

        const rewards = [];
        const undoTurn = endUnitTurn(state, capturer);
        let undoCapture: UndoCallback = () => { };

        if (struct) {
            if (struct.type == StructureType.Village) {
                const oldCity = getHomeCity(state, capturer);
                
                if (oldCity && capturer.homeCoords) {
                    capturer.homeCoords.copy(capturer.coords)
                }

                const result = (tile.owner? this.city(state) : this.village(state))!;
                rewards.push(...result.rewards);

                undoCapture = () => {
                    result.undo();

                    if (oldCity && capturer.homeCoords) {
                        capturer.homeCoords.setAt(oldCity.tileIndex, state)
                    }
                }
            }
            else {
                const result = this.ruins(state);
                rewards.push(...result.rewards);
                undoCapture = result.undo;
            }
        }
        else {
            undoCapture = this.starfish(state);
        }
        
        const undoBoost = tryRemoveEffect(state, capturer, EffectType.Boost);
        
        return {
            rewards,
            undo: () => {
                undoBoost();
                undoCapture();
                undoTurn();
            },
        }
    }
    
    village(state: GameState): CallbackResult {
        const pov = getPovTribe(state);        
        const captureIndex = this.getSrc();
        const territory = getAdjacentIndexes(state, captureIndex, 1, true, true);
        
        const createdCity: CityState = {
            name: `${TribeType[pov.type]} City`,
            population: 0,
            progress: 0,
            borderSize: 1,
            connectedToCapital: false,
            level: 1,
            production: 1,
            owner: pov.id,
            tileIndex: captureIndex,
            rewards: new Set(),
            _territory: territory,
        };

        xorCity.set(state, createdCity);
        pov.cities.push(createdCity);
        const claimBranch = claimTerritory(state, createdCity._territory, createdCity);

        return {
            rewards: claimBranch.rewards,
            undo: () => {
                claimBranch.undo();
                pov.cities.pop();
                xorCity.set(state, createdCity);
            }
        }
    }
    
    city(state: GameState): CallbackResult {
        const capturer = getUnitAt(state, this.getSrc())!;
        const pov = getPovTribe(state);
        const city = getCityAt(state, capturer.coords.idx)!;
        const tile = state.tiles[city.tileIndex];
        const enemy = state.tribes[city.owner];
        const cityName = city.name;
        
        // TODO enemyCity.progress neg population logic (also on unit death it should add if already neg)
        
        const cityListIndex = enemy.cities.indexOf(city);
        
        xorCity.owner(state, city, enemy.id, pov.id);

        // Claim the enemy's city
        enemy.cities.splice(cityListIndex, 1)
        pov.cities.push(city);
        city.name = `${TribeType[pov.type]} ${tile.capitalOf > 0? 'Capital' : 'City'}`;
        city.owner = pov.id;
        tile.owner = pov.id;
        if (tile.capitalOf > 0) tile.capitalOf = pov.id;
        
        // Claim the enemy's territory
        const claimBranch = claimTerritory(state, city._territory, city);
        
        // If enemy runs out of cities they loose all their units
        const chain: UndoCallback[] = [];
        if (!enemy.cities.length) {
            enemy.killedTurn = state.settings.turn;
            enemy.killerId = pov.id;
            for(const unit of enemy.units) {
                chain.push(removeUnit(state, unit));
            }
        }
        
        // TODO recalculate networks

        return {
            rewards: claimBranch.rewards,
            undo: () => {
                if (!enemy.cities.length) {
                    chain.reverse().forEach(x => x());
                    enemy.killerId = -1;
                    enemy.killedTurn = -1;
                }
                
                claimBranch.undo();

                if (tile.capitalOf > 0) tile.capitalOf = enemy.id;
                tile.owner = enemy.id;
                city.owner = enemy.id;
                city.name = cityName;
                pov.cities.pop();
                enemy.cities.splice(cityListIndex, 0, city);

                xorTile.owner(state, city.tileIndex, pov.id, enemy.id);
                xorCity.owner(state, city, pov.id, enemy.id);
            }
        }
    }
    
    ruins(state: GameState): Branch {
        const capturer = getUnitAt(state, this.getSrc())!;
        const pov = getPovTribe(state);
        const possibleRewards: (() => Branch)[] = [];
        const tileIndex = capturer.coords.idx;
        
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
            possibleRewards.push(() => 
                unlockTechnology(state, scrolls[Math.floor(Math.random() * scrolls.length)], true)
            );
        }

        // 3 free pop to highest level capital
        const city: CityState | null = pov.cities
            .filter(x => state.tiles[x.tileIndex].capitalOf > 0)
            .sort((a, b) => a.production - b.production)[0] || null;
        if (city) {
            possibleRewards.push(() => addPopulationToCity(state, city, 3));
        }

        const terrainType = state.tiles[tileIndex].type;
        const isMountain = terrainType === TerrainType.Mountain;
        const isOcean = terrainType === TerrainType.Ocean;

        // free explorer if 5x5 adj area is unexplored
        // note: cymanti cannot get explorers from water tiles
        const isCymanti = pov.type === TribeType.Cymanti

        if (!isMountain && (!isCymanti || (!isOcean && isCymanti))) {
            const around = getAdjacentIndexes(state, tileIndex, 2, true);
            // If there is any neaby unexplored tile
            if (around.some(x => !state._visibleTiles[x])) {
                possibleRewards.push(() => discoverTiles(state, null, predictExplorer(state, tileIndex)));
            }
        }

        // free veteran swordsman or free rammer (if on ocean tile)
        // TODO this is one bug -> (null.__moved)
        // wtf a unit has -1 as homeIndex?? also wtffff
        // fixed it but did it fix the bug?
        possibleRewards.push(() => {
            const summon = summonUnit(
                state, 
                isOcean? UnitType.Rammer : UnitType.Swordsman, 
                tileIndex, 
                false, 
                true
            )!;

            const summoned = pov.units[pov.units.length-1];

            xorUnit.kills(state, summoned, 0, 3);
            xorUnit.veteran(state, summoned);

            summoned.veteran = true;
            summoned.kills = 3;

            return {
                rewards: summon.rewards,
                undo: () => {
                    xorUnit.kills(state, summoned, 3, 0);
                    xorUnit.veteran(state, summoned);
                    summon.undo();
                }
            };
        });
        // TODO: bug causing `No unit at` in step (seed 8)
        possibleRewards.pop();

        // spawns a level 3 city with a city wall and 4 adjacent shallow water tiles	
        if (pov.type == TribeType.Aquarion && isOcean) {
            possibleRewards.push(() => {
                const createdCity: CityState = {
                    name: `${TribeType[pov.type]} City`,
                    population: 2,
                    progress: 0,
                    rewards: new Set([RewardType.Explorer, RewardType.CityWall]),
                    borderSize: 1,
                    connectedToCapital: false,
                    level: 3,
                    production: 3,
                    owner: pov.id,
                    tileIndex,
                    _territory: getAdjacentIndexes(state, tileIndex, 1, true, true),
                };

                xorCity.set(state, createdCity);

                const undoCreate = createStructure(state, tileIndex, StructureType.Village);

                // Transform adjacent tiles from ocean to water
                const chain: UndoCallback[] = [];
                [
                    tileIndex + 1,
                    tileIndex - 1,
                    tileIndex + state.settings.size,
                    tileIndex - state.settings.size,
                ].forEach(index => {
                    const coords = new Coords(index, state);
                    if (coords.x < 0 || coords.x >= state.settings.size || coords.y < 0 || coords.y >= state.settings.size) {
                        return;
                    }
                    chain.push(modifyTerrain(state, coords.idx, TerrainType.Water));
                });
    
                pov.cities.push(createdCity);

                const claimBranch = claimTerritory(state, createdCity._territory, createdCity)

                // TODO recalculate network connections

                return {
                    rewards: claimBranch.rewards,
                    undo: () => {
                        claimBranch.undo();
                        pov.cities.pop();
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

        const rewardBranch = GMath.randArr(possibleRewards)();
        
        return {
            rewards: rewardBranch.rewards,
            undo: () => {
                rewardBranch.undo();
                undoInvis();
                undoDestroyRuins();
            }
        }
    }
    
    starfish(state: GameState): UndoCallback {
        const capturer = getUnitAt(state, this.getSrc())!;
        const undoResource = consumeResource(state, capturer.coords.idx);
        const undoStars = gainStars(state, 8);
        return () => {
            undoStars();
            undoResource();
        }
    }
}