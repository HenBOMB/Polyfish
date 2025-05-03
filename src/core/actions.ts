import AIState from "../aistate";
import { rewardStructure } from "../polyfish/eval";
import { predictBestNextCityReward } from "../eval/prediction";
import { getNeighborTiles, calaulatePushablePosition, getNeighborIndexes, computeReachablePath, isSkilledIn, getPovTribe, getUnitAtTile, getTrueUnitAtTile, getHomeCity, getRulingCity, getMaxHealth, getEnemiesInRange, getEnemiesNearTile, isFrozen, calculateCombat, getUnitRange, getTrueEnemyAtTile, calculateAttack, isSteppable, isWaterTerrain, isPoisoned } from "./functions";
import Move, { MoveType } from "./move";
import UnitMoveGenerator, {  } from "./moves";
import { Branch, UndoCallback } from "./move";
import { ResourceSettings } from "./settings/ResourceSettings";
import { StructureSettings } from "./settings/StructureSettings";
import { TribeSettings } from "./settings/TribeSettings";
import { UnitSettings } from "./settings/UnitSettings";
import { CityState, GameState, StructureState, UnitState } from "./states";
import { RewardType, UnitType, StructureType, ResourceType, SkillType, TerrainType, ModeType, EffectType, ClimateType, TribeType } from "./types";

export function addPopulationToCity(state: GameState, targetCity: CityState, amount: number): Branch {
    if(!amount) {
        return { moves: [], undo: () => {} };
    }
    
    const tribe = state.tribes[state.settings._pov];
    const cityStruct = state.structures[targetCity.tileIndex]!;

    const oldProduction = targetCity._production;
    const oldPopulation = targetCity._population;
    const oldProgress = targetCity._progress;
    
    targetCity._population += amount;
    targetCity._progress += amount;
    
    if(targetCity._progress >= targetCity._level+1) {
        const diff = targetCity._progress - (targetCity._level + 1);
        
        cityStruct._level++;
        targetCity._level++;
        targetCity._progress = diff;
        targetCity._production++;

        if(targetCity._progress >= targetCity._level+1) {
            throw 'WHAT?! CRAZY!!!'
        }
        
        const scoreEconomy = state._scoreEconomy;
        
        state._scoreEconomy += 1;

        const dom = state.settings.mode == ModeType.Domination;
        
        const options: RewardType[] = [
            [ RewardType.Workshop, RewardType.Explorer ],
            [ ...dom? [RewardType.CityWall] : [], RewardType.Resources ],
            [ RewardType.PopulationGrowth, RewardType.BorderGrowth ],
        ][targetCity._level-2] || [ ...dom? [] : [RewardType.Park], RewardType.SuperUnit ];
        
        let rewardType = state._prediction._cityRewards[tribe._cities.indexOf(targetCity)];

        if(!options.includes(rewardType)) {
            rewardType = predictBestNextCityReward(state, targetCity)[0];
        }

        return {
            moves: [],
            chainMoves: [new Move(
                MoveType.Reward,
                targetCity.tileIndex, 0, rewardType,
                (state: GameState) => {
                    let undoReward: UndoCallback = () => { };
                    
                    switch (rewardType) {
                        case RewardType.Workshop:
                            targetCity._production++;
                            break;
                        case RewardType.Explorer:
                            const amount = Math.floor(Math.random() * 11) + 9;
                            state._potentialDiscovery.push(...Array(amount).fill(-1));
                            undoReward = () => { 
                                state._potentialDiscovery.splice(state._potentialDiscovery.length - amount, amount);
                            };
                            break;
                        case RewardType.CityWall:
                            targetCity._walls = true;
                            undoReward = () => {
                                targetCity._walls = false;
                            };
                            break;
                        case RewardType.Resources:
                            tribe._stars += 5;
                            undoReward = () => {
                                tribe._stars -= 5;
                            }
                            break;
                        case RewardType.PopulationGrowth:
                            targetCity._population += 3;
                            targetCity._progress += 3;
                            break;
                        case RewardType.BorderGrowth:
                            targetCity._borderSize++;
                            const tileIndex = targetCity.tileIndex;
                            const undiscovered = getNeighborTiles(state, tileIndex, targetCity._borderSize, false, true)
                                .filter(x => !x.explorers.includes(tribe.owner) && !state._potentialDiscovery.includes(x.tileIndex));
                            state._potentialDiscovery.push(...undiscovered.map(x => x.tileIndex));
                            if(state.settings.live) {
                                if(state.settings.live) {
                                    for(const tile of undiscovered) {
                                        state._visibleTiles.push(tileIndex);
                                        tile.explorers.push(tribe.owner);
                                    }
                                }
                                const unowned = undiscovered.filter(x => x._owner < 1);
                                // TODO Should be potential resources and such, this counts as cheating
                                for(const tile of unowned) {
                                    tile._owner = tribe.owner;
                                    tile._rulingCityIndex = tileIndex;
                                    const struct = state.structures[tileIndex];
                                    const resource = state.resources[tileIndex];
                                    if(struct) {
                                        struct._owner = tribe.owner;
                                    }
                                    if(resource) {
                                        resource._owner = tribe.owner;
                                        tribe._resources.push(tileIndex);
                                    }
                                }
                            }
                            undoReward = () => {
                                state._potentialDiscovery = state._potentialDiscovery.slice(0, -undiscovered.length);
                                targetCity._borderSize--;
                            }
                        break;
                        case RewardType.Park:
                            targetCity._production++;
                            break;
                        case RewardType.SuperUnit:
                            let undoPush: UndoCallback = pushUnit(state, targetCity.tileIndex);
                            
                            const undoSummon = summonUnit(state, 
                                TribeSettings[tribe.tribeType].uniqueSuperUnit || UnitType.Giant, 
                                targetCity.tileIndex
                            );
                            
                            undoReward = () => {
                                undoSummon();
                                undoPush();
                            }
                            break;
                        default:
                            throw `Invalid reward type: ${rewardType}`;
                    }
                    
                    targetCity._rewards.push(rewardType);
                    
                    return {
                        moves: [],
                        undo: () => {
                            targetCity._rewards.pop();
                            undoReward();
                            targetCity._production = oldProduction;
                            targetCity._population = oldPopulation;
                            targetCity._progress = oldProgress;
                        },
                    };
                },
            )],
            undo: () => {
                state._scoreEconomy = scoreEconomy;
                cityStruct._level--;
                targetCity._level--;
                targetCity._production = oldProduction;
                targetCity._population = oldPopulation;
                targetCity._progress = oldProgress;
            },
        }
    }
    
    return {
        moves: [],
        undo: () => {
            targetCity._population = oldPopulation;
            targetCity._progress = oldProgress;
        }
    }
}

export function addMissingConnections(state: GameState, targetCity: CityState, tileIndex: number): UndoCallback[] {
    // If we built a port, we must connect it to any nearby ports and reward connected cities
    const structType = state.structures[tileIndex]?.id;
    if(structType != StructureType.Port) return [];
    
    const tribe = state.tribes[state.settings._pov];
    const undoChain: UndoCallback[] = [];
    const nearbyStructureIndexes = getNeighborIndexes(state, tileIndex, 5);
    const connectedPorts: [number, number[]][] = [];
    const connectedCities: [number, number[]][] = [];
    const potentialNewConnections: [CityState, [number, number[]]][] = [];
    let rewardedCities: CityState[] = [];
    const isTargetCapital = state.tiles[targetCity.tileIndex].capitalOf > 0;
    
    // Not sure if this method is 100% accurate
    
    for (let i = 0; i < nearbyStructureIndexes.length; i++) {
        const structTileIndex = nearbyStructureIndexes[i];
        
        const nearbyStructure = state.structures[structTileIndex];
        
        // If there is any structure and we own it
        if(!nearbyStructure || nearbyStructure._owner != tribe.owner) continue;
        
        const ownerCity = getRulingCity(state, structTileIndex);
        
        if(!ownerCity) continue;
        
        const isOwnerCapital = state.tiles[ownerCity.tileIndex].capitalOf > 0;
        
        // 3 is for types TerrainType.Water (1) and TerrainType.Ocean (2)
        const shortestPathToStruct = computeReachablePath(state, tileIndex, structTileIndex, 
            (state, index) => state.tiles[index].terrainType < 3);
        
        // If struct is not reachable
        if(shortestPathToStruct.length < 1 || shortestPathToStruct.length > 5) continue;
        
        if(nearbyStructure.id == StructureType.Port) {
            // Nearby port is not connected!
            if((targetCity._connectedToCapital || isTargetCapital) && (!ownerCity._connectedToCapital && !isOwnerCapital)) {
                // console.log('reconnect!');
                if(!potentialNewConnections.some(x => x[0] == ownerCity)) {
                    potentialNewConnections.push([ownerCity, [structTileIndex, shortestPathToStruct]]);
                }
            }
            // Connect to the port
            else if((!targetCity._connectedToCapital && !isTargetCapital) 
                && (ownerCity._connectedToCapital || isOwnerCapital)
            ) {
                // console.log('reward!');
                connectedPorts.push([structTileIndex, shortestPathToStruct]);
                if(!rewardedCities.includes(ownerCity)) {
                    rewardedCities.push(ownerCity);
                }
            }
            // If its not a capital
            else {
                // console.log('just route!');
                connectedPorts.push([structTileIndex, shortestPathToStruct]);
            }
        }
        // If we found one of our cities
        else if(nearbyStructure.id == StructureType.Village) {
            // Nearby city is not connected!
            if((targetCity._connectedToCapital || isTargetCapital) && (!ownerCity._connectedToCapital && !isOwnerCapital)) {
                if(!potentialNewConnections.some(x => x[0] == ownerCity)) {
                    potentialNewConnections.push([ownerCity, [structTileIndex, shortestPathToStruct]]);
                }
            }
            // Connect to the city
            if((!targetCity._connectedToCapital && !isTargetCapital) 
                && (ownerCity._connectedToCapital || isOwnerCapital)
            ) {
                connectedCities.push([structTileIndex, shortestPathToStruct]);
                if(!rewardedCities.includes(ownerCity)) {
                    rewardedCities.push(ownerCity);
                }
            }
        }
    }
    
    if(!nearbyStructureIndexes.length || (!connectedPorts.length && !connectedCities.length)) return [];

    // If we were not connected and now we are
    // Capitals go first
    if((!targetCity._connectedToCapital) && rewardedCities.length) {
        if(state.tiles[targetCity.tileIndex].capitalOf > 0) {
            rewardedCities = [targetCity, ...rewardedCities];
        }
        else {
            rewardedCities.push(targetCity);
        }
        for(let i = 0; i < potentialNewConnections.length; i++) {
            const city = potentialNewConnections[i][0];
            if(state.tiles[city.tileIndex].capitalOf > 0) {
                rewardedCities = [city, ...rewardedCities];
            }
            else {
                rewardedCities.push(city);
                connectedPorts.push(potentialNewConnections[i][1]);
            }
        }
    }

    // If more than 1 city was connected, apply rewards
    if(rewardedCities.length > 1) {
        // Calculate it anyway
        for (let i = 0; i < rewardedCities.length; i++) {
            const branch = addPopulationToCity(state, rewardedCities[i], 1);
            undoChain.push(branch.undo);
            if(branch.chainMoves) {
                branch.chainMoves.forEach(x => undoChain.push(x.execute(state)!.undo));
                // console.log('Auto Chosen: ' + RewardType[Number(branch.chainMoves[0].id.split('-')[1])]);
            }
        }
        
        // Modify only unconnected cities
        const unconnectedCities = rewardedCities.filter(x => !x._connectedToCapital);
        for (let i = 0; i < unconnectedCities.length; i++) {
            unconnectedCities[i]._connectedToCapital = true;
        }
        undoChain.push(() => {
            for (let i = 0; i < unconnectedCities.length; i++) {
                unconnectedCities[i]._connectedToCapital = false;
            }
        });
    }

    // Apply missing routes
    const routedTiles: number[] = [];
    for (let i = 0; i < connectedPorts.length; i++) {
        // If the city's path is closer than the port's path, connect the city instead of the port
        const index = connectedCities.findIndex(x => connectedPorts[i][0] == state.tiles[x[0]]._rulingCityIndex)
        if(index > -1) {
            if(connectedCities[index][1].length < connectedPorts[i][1].length) {
                console.log('LMAO CITY PATH IS CLOSER');
                connectedCities[index][1].forEach(x => {
                    const tile = state.tiles[x];
                    if(tile.hasRoute) return;
                    routedTiles.push(x);
                    state.tiles[x].hasRoute = true;
                });
                connectedCities.splice(index, 1);
                continue;
            }
            else {
                console.log('LMAO PORT PATH IS CLOSER');
            }
        }
        connectedPorts[i][1].forEach(x => {
            if(state.tiles[x].hasRoute) return;
            routedTiles.push(x);
            state.tiles[x].hasRoute = true;
        });
    }
    
    // Try to connect any remaining unconnected cities
    for (let i = 0; i < connectedCities.length; i++) {
        connectedCities[i][1].forEach(x => {
            if(state.tiles[x].hasRoute) return;
            routedTiles.push(x);
            state.tiles[x].hasRoute = true;
        });
    }

    // EVAL Reward for having connected a port with a city
    if(rewardedCities.length > 1) {
        state._potentialEconomy += 1;
    }

    state._potentialArmy += routedTiles.length * .8;

    undoChain.push(() => {
        state._potentialArmy -= routedTiles.length * .8;
        if(rewardedCities.length > 1) {
            state._potentialEconomy -= 1;
        }
        for (let i = 0; i < routedTiles.length; i++) {
            state.tiles[routedTiles[i]].hasRoute = false;
        }
    });

    return undoChain.reverse();
}

export function harvestResource(state: GameState, tileIndex: number): Branch {
    const tribe = getPovTribe(state);
    const harvested = state.resources[tileIndex];
    
    if(!harvested) {
        throw Error(`No resource at ${tileIndex}`);
    }
    
    const settings = ResourceSettings[harvested.id];
    const rulingCity = tribe._cities.find(x => x.tileIndex == state.tiles[tileIndex]._rulingCityIndex);
    
    if(!rulingCity) {
        throw Error(`No ruling city for ${ResourceType[harvested.id]}, ${tileIndex} -> ${state.tiles[tileIndex]._rulingCityIndex}`);
    }
    
    const cost = settings.cost || 0;
    
    if(cost > tribe._stars) {
        throw Error(`Not enough stars ${tribe._stars}/${cost}`);
    }
    
    tribe._stars -= cost;
    delete state.resources[tileIndex];
    const resourceArrayIndex = tribe._resources.indexOf(tileIndex);
    tribe._resources.splice(resourceArrayIndex, 1);
    
    const branch = addPopulationToCity(state, rulingCity, settings.rewardPop);
    
    // TODO avoid unnescessary harvest?

    // levelled up
    // if(branch.forcedMoves) {
    //     state._potentialEconomy +=     
    // }

    return {
        moves: branch.moves,
        chainMoves: branch.chainMoves,
        undo: () => {
            branch.undo();
            state.resources[tileIndex] = harvested;
            tribe._resources.splice(resourceArrayIndex, 0, tileIndex);
            tribe._stars += cost;
        }
    };
}

export function buildStructure(state: GameState, strctureType: StructureType, tileIndex: number): Branch {
    const tribe = state.tribes[state.settings._pov];
    const settings = StructureSettings[strctureType];
    const rulingCity = getRulingCity(state, tileIndex);
    
    if(!rulingCity) {
        console.error(`No ruling city for ${StructureType[strctureType]} (${strctureType}), ${state.tiles[tileIndex]._rulingCityIndex}, ${tileIndex}`);
        return { moves: [], undo: () => {} };
    }
    
    const cost = settings.cost || 0;
    
    if(cost > tribe._stars) return { moves: [], undo: () => {} };
    
    tribe._stars -= cost;
    
    const structure: StructureState = {
        id: strctureType,
        _level: 1,
        turn: state.settings._turn,
        tileIndex,
        reward: 0,
        _name: StructureType[strctureType],
        _owner: tribe.owner
    };
    
    state.structures[tileIndex] = structure;
    
    let rewardPopCount = settings.rewardPop || 0;
    let rewardStarCount = settings.rewardStars || 0;
    
    if(settings.adjacentTypes !== undefined) {
        const adjCount = getNeighborTiles(state, tileIndex).filter(x => state.structures[x.tileIndex]? settings.adjacentTypes!.includes(state.structures[x.tileIndex]!.id) : false).length;
        rewardStarCount *= adjCount;
        rewardPopCount *= adjCount;
    }
    
    const rewardBranch = addPopulationToCity(state, rulingCity, rewardPopCount);
    
    if(rewardStarCount) {
        tribe._stars += rewardStarCount;
    }
    
    settings.task && tribe._builtUniqueStructures.push(strctureType);
    
    // TODO This should really return a branch
    const undoConnections = addMissingConnections(state, rulingCity, tileIndex);
    
    // Promote building ports strategically
    // good ports are the ones with more fog, water and no other adjacent ally ports
    
    const potentialEconomy = state._potentialEconomy;

    state._potentialEconomy += rewardStructure(state, structure);

    return {
        moves: [],
        chainMoves: rewardBranch.chainMoves,
        undo: () => {
            state._potentialEconomy = potentialEconomy;
            for (let i = 0; i < undoConnections.length; i++) {
                undoConnections[i]();
            }
            settings.task && tribe._builtUniqueStructures.pop();
            if(rewardStarCount) {
                tribe._stars -= rewardStarCount;
            }
            rewardBranch.undo();
            delete state.structures[tileIndex];
            tribe._stars += cost;
        }
    };
}

export function summonUnit(state: GameState, unitType: UnitType, spawnTileIndex: number, costs = false): UndoCallback {
    const tribe = state.tribes[state.settings._pov];
    const settings = UnitSettings[unitType];
    const inherits = settings.upgradeFrom;
    const health = UnitSettings[unitType].health!;
    
    if(!health || health < 0) {
        console.warn(`Unit ${unitType} has no health (${UnitType[inherits||0]})!`);
        return () => {};
    }
    
    const spawnTile = state.tiles[spawnTileIndex];
    
    // Push occupied unit away
    let undoPush: UndoCallback = () => pushUnit(state, spawnTile.tileIndex);
    
    const oldUnitIdx = spawnTile._unitIdx;
    const oldUnitOwner = spawnTile._unitOwner;
    
    if(costs) tribe._stars -= settings.cost;
    
    const spawnedUnit = {
        idx: state.settings.unitIdx++,
        x: spawnTileIndex % state.settings.size, 
        y: Math.floor(spawnTileIndex / state.settings.size),
        _unitType: unitType,
        _health: health * 10,
        kills: 0,
        prevX: -1,
        prevY: -1,
        direction: 0,
        _owner: tribe.owner,
        createdTurn: state.settings._turn,
        // If its not from a ruin or special unit
        _homeIndex: isSkilledIn(unitType, SkillType.Independent) || !costs? -1 : spawnTileIndex,
        _tileIndex: spawnTileIndex,
        _effects: [],
        _attacked: true,
        _moved: true,
        _passenger: undefined,
    } as UnitState;

    tribe._units.push(spawnedUnit);

    spawnTile._unitIdx = spawnedUnit.idx;
    spawnTile._unitOwner = spawnedUnit._owner;
    
    const cityHome = getHomeCity(state, spawnedUnit);

    if(cityHome) cityHome._unitCount++;
    
	const undoDiscover = discoverTiles(state, spawnedUnit);
    const undoFrozen: UndoCallback = freezeArea(state, spawnedUnit);

    return () => {
        undoFrozen();
        undoDiscover();
        if(cityHome) cityHome._unitCount--;
        spawnTile._unitIdx = oldUnitIdx;
        spawnTile._unitOwner = oldUnitOwner;
        if(costs) tribe._stars += settings.cost;
        state.settings.unitIdx--;
        tribe._units.pop();
        undoPush();
    }
}

export function removeUnit(state: GameState, removed: UnitState, credit?: UnitState): UndoCallback {
    const tribe = state.tribes[removed._owner];
    const tile = state.tiles[removed._tileIndex];
    const oldOwner = removed._owner;
    const oldIdx = tile._unitIdx;
    const cityHome = getHomeCity(state, removed);
    const atIndex = tribe._units.findIndex(x => x.idx == removed.idx);

    tribe._units.splice(atIndex, 1);
    tile._unitIdx = -1;
    tile._unitOwner = -1;
    if(cityHome) cityHome._unitCount--;

    if(credit) {
        credit.kills++;
        state.tribes[credit._owner]._kills++;
    }
    
    return () => {
        if(credit) {
            credit.kills--;
            state.tribes[credit._owner]._kills--;
        }
        if(cityHome) cityHome._unitCount++;
        tile._unitOwner = oldOwner;
        tile._unitIdx = oldIdx;
        tribe._units.splice(atIndex, 0, removed);
    };
}

export function healUnit(unit: UnitState, amount: number): UndoCallback {
    if(isPoisoned(unit))	{
        const index = unit._effects.indexOf(EffectType.Poison);
        unit._effects.splice(index, 1);
        return () => {
            unit._effects.splice(index, 0, EffectType.Poison);
        }
    }
    const oldHealth = unit._health;
    unit._health += amount;
    unit._health = Math.min(unit._health, getMaxHealth(unit));
    return () => {
        unit._health = oldHealth;
    };
}

export function discoverTiles(state: GameState, unit: UnitState): UndoCallback {
    const tile = state.tiles[unit._tileIndex];
    const owner = unit._owner;
    const discoverableTiles = getNeighborIndexes(
        state,
        unit._tileIndex,
        tile.terrainType == TerrainType.Mountain || isSkilledIn(unit, SkillType.Scout)? 2 : 1,
        false,
        true
    ).filter(x => !state.tiles[x].explorers.includes(owner));

    if(!discoverableTiles.length) return () => {};

    const lighthouses = [...state._lighthouses];
    let potential = 0;

    for (const tileIndex of discoverableTiles) {
        if (state.settings.live) {
            if(state._lighthouses.includes(tileIndex)) {
                state._lighthouses.splice(state._lighthouses.indexOf(tileIndex), 1);
            }
            if(!state._visibleTiles.includes(tileIndex)) {
                state._visibleTiles.push(tileIndex);
            }
            state.tiles[tileIndex].explorers.push(owner);
        }
        else {
            if(!state._potentialDiscovery.includes(tileIndex)) {
                if(state._lighthouses.includes(tileIndex)) {
                    state._lighthouses.splice(state._lighthouses.indexOf(tileIndex), 1);
                }
                state._potentialDiscovery.push(tileIndex);
                potential++;
            }
        }
    }

    return () => {
        state._potentialDiscovery = state._potentialDiscovery.slice(0, state._potentialDiscovery.length - potential);
        state._lighthouses = lighthouses;
    };
}

export function freezeArea(state: GameState, freezer: UnitState): UndoCallback {
    if(!isSkilledIn(freezer, SkillType.AutoFreeze, SkillType.FreezeArea)) return () => { };

    const freezeArea = isSkilledIn(freezer, SkillType.FreezeArea);
    
    const undoChain: UndoCallback[] = [];

    // Freeze adjacent water tiles and units
    getNeighborTiles(state, freezer._tileIndex, 1, false, true).filter(tile => {
        // Converts tile to the style of the tribe the unit belongs to
        if(freezeArea && !isWaterTerrain(tile)) {
            const oldClimate = tile.climate;
            // TODO verify this
            tile.climate = Number(ClimateType[TribeType[freezer._owner] as any]);
            undoChain.push(() => {
                tile.climate = oldClimate;
            });
        }
        const occupied = getTrueEnemyAtTile(state, tile, freezer._owner);
        if(occupied) {
            if(isFrozen(occupied)) return;
            occupied._effects.push(EffectType.Frozen);
            undoChain.push(() => {
                occupied._effects.pop();
            });
            return;
        }
        if(tile.terrainType == TerrainType.Water || tile.terrainType == TerrainType.Ocean) {
            const oldTerrain = tile.terrainType;
            tile.terrainType = TerrainType.Ice;
            undoChain.push(() => {
                tile.terrainType = oldTerrain;
            });
        }
        return 
    });

    return () => {
        undoChain.reverse().forEach(x => x());
    }
}

export function splashDamageArea(state: GameState, stomper: UnitState, atk: number): UndoCallback {
    if(!isSkilledIn(stomper, SkillType.Stomp)) return () => { };
    const undoChain = getEnemiesNearTile(state, stomper._tileIndex)
        .map(enemy => attackUnit(state, atk, enemy, stomper))
        .reverse();
    return () => {
        undoChain.forEach(x => x());
    }
}

// TODO should return a branch?
export function pushUnit(state: GameState, tileIndex: number) {
    const pushed = getTrueUnitAtTile(state, tileIndex);

    if (!pushed) return () => { };

    const oldAttacked = pushed._attacked;
    const oldMoved = pushed._moved;
    const movedTo = calaulatePushablePosition(state, pushed);

    let undoPush: UndoCallback = () => { };

    if (movedTo < 0) {
        undoPush = removeUnit(state, pushed);
    }
    else {
        const result = UnitMoveGenerator.stepCallback(state, pushed, movedTo, true);

        if(!result) {
            undoPush = removeUnit(state, pushed);
        }
        else {
            undoPush = result.undo;
        }
    }

    return () => {
        undoPush();
        pushed._moved = oldMoved;
        pushed._attacked = oldAttacked;
    };
}

export function attackUnit(state: GameState, attacker: UnitState | number, defender: UnitState, attackerPov?: UnitState): UndoCallback {
    const undoChain: UndoCallback[] = [];

    if(typeof attacker == 'number') {
        const atk = calculateAttack(state, attacker, defender);

        defender._health -= atk;

        if (defender._health < 1) {
            undoChain.push(removeUnit(state, defender, attackerPov));
        }

        undoChain.push(() => {
            defender._health += atk;
        });
    }
    else {
        const result = calculateCombat(state, attacker, defender);

        defender._health -= result.attackDamage;

        undoChain.push(() => {
            defender._health += result.attackDamage;
        });

        // Deal splash damage
        if(result.splashDamage > 0) {
            undoChain.push(splashDamageArea(state, attacker, result.splashDamage));
        }

        // We killed their unit
        if(defender._health < 1) {
            undoChain.push(removeUnit(state, defender, attacker));
            // Move to the enemy position, if not a ranged unit
            if (getUnitRange(attacker) < 2 && isSteppable(state, attacker, defender._tileIndex)) {
                const result = UnitMoveGenerator.stepCallback(state, attacker, defender._tileIndex, true)!;
                undoChain.push(result.undo);
            }
        }
        // Retaliate
        else {
            // If we have have freeze, then they cant retaliate
            if(isSkilledIn(attacker, SkillType.Freeze)) {
                result.defenseDamage = 0;
                let wasFrozen = false;
                if(!isFrozen(defender)) {
                    defender._effects.push(EffectType.Frozen);
                    wasFrozen = true;
                }
                undoChain.push(() => {
                    if(wasFrozen) {
                        defender._effects.pop();
                    }
                });
            }
            // If we're attacking with range
            else if(getUnitRange(attacker) > 1 && result.defenseDamage > 0) {
                const dist2enemy = Math.floor(Math.hypot(attacker.x - defender.x, attacker.y - defender.y));
                // If they cant reach us, they cant retaliate
                if (dist2enemy > getUnitRange(defender)) {
                    result.defenseDamage = 0;
                }
                // THIS IS CHEATING! (use enemy visibility predictions)
                // If they cant see us, they cant retaliate
                // else if (!enemyTile.explorers.includes(defender._owner)) {
                // 	result.defenseDamage = 0;
                // }
            }

            if(result.defenseDamage > 0) {
                attacker._health -= result.defenseDamage;
    
                undoChain.push(() => {
                    attacker._health += result.defenseDamage;
                });
    
                // Our unit died
                if (attacker._health < 1) {
                    undoChain.push(removeUnit(state, attacker, defender));
                }
            }
        }        
    }

    return () => {
        undoChain.reverse().forEach(x => x());
    };
}