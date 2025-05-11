import { getNeighborTiles, calaulatePushablePosition, getNeighborIndexes, computeReachablePath, isSkilledIn, getPovTribe, getTrueUnitAt, getHomeCity, getRulingCity, getMaxHealth, getEnemiesInRange, getEnemiesNearTile, isFrozen, calculateCombat, getUnitRange, getTrueEnemyAt, calculateAttack, isSteppable, isWaterTerrain, isPoisoned, getCapitalCity, getLighthouses, unPoison, addFreeze } from "./functions";
import { ArmyMovesGenerator, EconMovesGenerator } from "./moves";
import Move, { Branch, CallbackResult, UndoCallback } from "./move";
import { ResourceSettings } from "./settings/ResourceSettings";
import { StructureSettings } from "./settings/StructureSettings";
import { UnitSettings } from "./settings/UnitSettings";
import { CityState, GameState, StructureState, UnitState } from "./states";
import { UnitType, StructureType, SkillType, TerrainType, EffectType, ClimateType, TribeType } from "./types";
import { IsStructureTask } from "./settings/TaskSettings";

export function addPopulationToCity(state: GameState, targetCity: CityState, amount: number): CallbackResult {
    if(!amount) {
        return null;
    }
    
    const cityStruct = state.structures[targetCity.tileIndex]!;
    
    targetCity._population += amount;
    targetCity._progress += amount;

    const next = targetCity._level + 1;
    
    if(targetCity._progress >= next) {
        cityStruct._level++;
        targetCity._level++;
        targetCity._progress -= next;
        targetCity._production++;

        let rewards = EconMovesGenerator.rewards(targetCity);
        let lol = false;

        if(targetCity._progress - next >= (next + 1)) {
            console.warn('LMAOOO');
            lol = true;
            cityStruct._level++;
            targetCity._level++;
            targetCity._progress -= next + 1;
            targetCity._production++;
            rewards.push(...EconMovesGenerator.rewards(targetCity));
        }

        return {
            rewards,
            undo: () => {
                if(lol) {
                    targetCity._production--;
                    targetCity._progress += next + 1;
                    targetCity._level--;
                    cityStruct._level--;
                }

                targetCity._production--;
                targetCity._progress += next;
                targetCity._level--;
                cityStruct._level--;

                targetCity._progress -= amount;
                targetCity._population -= amount;
            },
        }
    }
    
    return {
        rewards: [],
        undo: () => {
            targetCity._progress -= amount;
            targetCity._population -= amount;
        }
    }
}

// Not sure if this method is 100% accurate

export function addMissingConnections(state: GameState, targetCity: CityState, tileIndex: number): CallbackResult {
    // If we built a port, we must connect it to any nearby ports and reward connected cities
    const structType = state.structures[tileIndex]?.id;
    if(structType != StructureType.Port) return null;
    
    const tribe = state.tribes[state.settings._pov];
    const undoChain: UndoCallback[] = [];
    const rewards: Move[] = [];

    // max range 5 tiles
    const nearbyStructureIndexes = getNeighborIndexes(state, tileIndex, 5);
    const connectedPorts: [number, number[]][] = [];
    const connectedCities: [number, number[]][] = [];
    const potentialNewConnections: [CityState, [number, number[]]][] = [];
    let rewardedCities: CityState[] = [];
    const isTargetCapital = state.tiles[targetCity.tileIndex].capitalOf > 0;
    
    for (let i = 0; i < nearbyStructureIndexes.length; i++) {
        const structTileIndex = nearbyStructureIndexes[i];
        const tile = state.tiles[structTileIndex];
        
        const nearbyStructure = state.structures[structTileIndex];
        
        // If there is any structure and we own it
        if(!nearbyStructure || tile._owner != tribe.owner) continue;
        
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
    
    if(!nearbyStructureIndexes.length || (!connectedPorts.length && !connectedCities.length)) {
        return null;
    }

    // If we were not connected and now we are
    // Capitals go first
    if(!targetCity._connectedToCapital && rewardedCities.length) {
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
            if(branch) {
                undoChain.push(branch.undo);
                rewards.push(...branch.rewards);
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

    undoChain.push(() => {
        for (let i = 0; i < routedTiles.length; i++) {
            state.tiles[routedTiles[i]].hasRoute = false;
        }
    });

    return {
        rewards,
        undo: () => {
            undoChain.reverse().forEach(x => x());
        }
    }
}

export function harvestResource(state: GameState, tileIndex: number): Branch {
    const tribe = getPovTribe(state);
    const harvested = state.resources[tileIndex]!;
    const settings = ResourceSettings[harvested.id];
    const rulingCity = tribe._cities.find(x => x.tileIndex == state.tiles[tileIndex]._rulingCityIndex)!;
    
    const cost = settings.cost || 0;
    
    tribe._stars -= cost;
    delete state.resources[tileIndex];
    const popBranch = addPopulationToCity(state, rulingCity, settings.rewardPop);
    
    return {
        rewards: (popBranch?.rewards || []),
        undo: () => {
            popBranch?.undo();
            state.resources[tileIndex] = harvested;
            tribe._stars += cost;
        }
    };
}

export function buildStructure(state: GameState, strctureType: StructureType, tileIndex: number): Branch {
    const tribe = state.tribes[state.settings._pov];
    const settings = StructureSettings[strctureType];
    const rulingCity = getRulingCity(state, tileIndex)!;
    const cost = settings.cost || 0;
    const oldStruct = state.structures[tileIndex];

    tribe._stars -= cost;
    
    const structure: StructureState = {
        id: strctureType,
        _level: 1,
        turn: state.settings._turn,
        tileIndex,
        reward: 0,
    };
    
    state.structures[tileIndex] = structure;
    
    let rewardPopCount = settings.rewardPop || 0;
    let rewardStarCount = settings.rewardStars || 0;
    
    if(settings.adjacentTypes !== undefined) {
        const adjCount = getNeighborTiles(state, tileIndex)
            .filter(x => state.structures[x.tileIndex]? settings.adjacentTypes!.has(state.structures[x.tileIndex]!.id) : false).length;
        rewardStarCount *= adjCount;
        rewardPopCount *= adjCount;
    }
    
    if(rewardStarCount) {
        tribe._stars += rewardStarCount;
    }

    IsStructureTask[strctureType] && tribe._builtUniqueStructures.has(strctureType);

    const popBranch = addPopulationToCity(state, rulingCity, rewardPopCount);
   
    // TODO This should really return a branch
    const portBranch = addMissingConnections(state, rulingCity, tileIndex);
    
    return {
        rewards: [ ...(popBranch?.rewards || []), ...(portBranch?.rewards || []) ],
        undo: () => {
            portBranch?.undo();
            popBranch?.undo();
            IsStructureTask[strctureType] && tribe._builtUniqueStructures.delete(strctureType);
            if(rewardStarCount) {
                tribe._stars -= rewardStarCount;
            }
            state.structures[tileIndex] = oldStruct;
            tribe._stars += cost;
        }
    };
}

export function summonUnit(state: GameState, unitType: UnitType, spawnTileIndex: number, costs = false, forceIndependent = false): CallbackResult {
    const pov = state.tribes[state.settings._pov];
    const settings = UnitSettings[unitType];
    const health = UnitSettings[unitType].health!;
    
    const spawnTile = state.tiles[spawnTileIndex];
    
    // Push occupied unit away (if any)
    let resultPush = pushUnit(state, spawnTile.tileIndex);
    
    const oldUnitOwner = spawnTile._unitOwner;
    
    if(costs) pov._stars -= settings.cost;
    
    const spawnedUnit = {
        x: spawnTileIndex % state.settings.size, 
        y: Math.floor(spawnTileIndex / state.settings.size),
        _unitType: unitType,
        _health: health * 10,
        kills: 0,
        prevX: -1,
        prevY: -1,
        direction: 0,
        _owner: pov.owner,
        createdTurn: state.settings._turn,
        // If its not from a ruin or special unit
        _homeIndex: forceIndependent || isSkilledIn(unitType, SkillType.Independent) || !costs? -1 : spawnTileIndex,
        _tileIndex: spawnTileIndex,
        _effects: new Set(),
        _attacked: true,
        _moved: true,
    } as UnitState;

    pov._units.push(spawnedUnit);

    spawnTile._unitOwner = spawnedUnit._owner;
    
    const cityHome = forceIndependent? null : getHomeCity(state, spawnedUnit);

    if(cityHome) cityHome._unitCount++;
    
	const resultDiscover = discoverTiles(state, spawnedUnit);
    const undoFrozen: UndoCallback = freezeArea(state, spawnedUnit);

    return {
        rewards: [...(resultDiscover?.rewards || []), ...(resultPush?.rewards || [])],
        undo: () => {
            undoFrozen();
            resultDiscover?.undo();
            if(cityHome) cityHome._unitCount--;
            spawnTile._unitOwner = oldUnitOwner;
            if(costs) pov._stars += settings.cost;
            state.settings.unitIdx--;
            pov._units.pop();
            resultPush?.undo();
        }
    }
}

export function removeUnit(state: GameState, removed: UnitState, credit?: UnitState): UndoCallback {
    const tribe = state.tribes[removed._owner];
    const tile = state.tiles[removed._tileIndex];
    const oldOwner = removed._owner;
    const cityHome = getHomeCity(state, removed);
    const atIndex = tribe._units.findIndex(x => x._tileIndex == removed._tileIndex);

    tribe._units.splice(atIndex, 1);
    tile._unitOwner = 0;
    if(cityHome) cityHome._unitCount--;
    if(credit) {
        tribe._casualties++;
        credit.kills++;
        state.tribes[credit._owner]._kills++;
    }
    
    return () => {
        if(credit) {
            tribe._casualties--;
            credit.kills--;
            state.tribes[credit._owner]._kills--;
        }
        if(cityHome) cityHome._unitCount++;
        tile._unitOwner = oldOwner;
        tribe._units.splice(atIndex, 0, removed);
    };
}

export function healUnit(unit: UnitState, amount: number): UndoCallback {
    if(isPoisoned(unit))	{
        return unPoison(unit);
    }
    const oldHealth = unit._health;
    unit._health += amount;
    unit._health = Math.min(unit._health, getMaxHealth(unit));
    return () => {
        unit._health = oldHealth;
    };
}

export function discoverTiles(state: GameState, unit?: UnitState | null, tileIndexes?: number[]): CallbackResult {
    const owner = state.settings._pov;
    const discovered = (tileIndexes || (unit? getNeighborIndexes(
        state,
        unit._tileIndex,
        state.tiles[unit._tileIndex].terrainType == TerrainType.Mountain || isSkilledIn(unit, SkillType.Scout)? 2 : 1,
        false,
        true
    ) : [])).filter(x => !state._visibleTiles[x]);

    // Get undiscovered lightouses
    const lighthouses = getLighthouses(state, false);
    
    let chain: UndoCallback[] = [];
    let rewards: Move[] = [];

    for (const tileIndex of discovered) {
        if(lighthouses.includes(tileIndex)) {
            const city = getCapitalCity(state);
            if(city) {
                const result = addPopulationToCity(state, city, 1);
                if(result) {
                    chain.push(result?.undo);
                    rewards.push(...result.rewards);
                }
            }
        }
        if (state.settings.areYouSure) {
            state.tiles[tileIndex]._explorers.add(owner);
        }
        state._visibleTiles[tileIndex] = true;
    }

    return {
        rewards,
        undo: () => {
            discovered.forEach(x => state._visibleTiles[x] = false);
            if(state.settings.areYouSure) {
                discovered.forEach(x => {
                    state.tiles[x]._explorers.delete(owner);
                });
            }
        }
    }
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
        const occupied = getTrueEnemyAt(state, tile, freezer._owner);
        if(occupied) {
            if(!isFrozen(occupied)) {
                undoChain.push(addFreeze(occupied));
            }
        }
        if(tile.terrainType == TerrainType.Water || tile.terrainType == TerrainType.Ocean) {
            const oldTerrain = tile.terrainType;
            tile.terrainType = TerrainType.Ice;
            undoChain.push(() => {
                tile.terrainType = oldTerrain;
            });
        }
    });

    return () => {
        undoChain.reverse().forEach(x => x());
    }
}

export function splashDamageArea(state: GameState, attacker: UnitState, atk: number): UndoCallback {
    const undoChain = getEnemiesNearTile(state, attacker._tileIndex)
        .map(enemy => attackUnit(state, atk, enemy, attacker));
    return () => {
        undoChain.reverse().forEach(x => x());
    }
}

export function pushUnit(state: GameState, tileIndex: number): CallbackResult {
    const pushed = getTrueUnitAt(state, tileIndex);
    if(!pushed) return null;

    const oldAttacked = pushed._attacked;
    const oldMoved = pushed._moved;
    const movedTo = calaulatePushablePosition(state, pushed);
    const rewards = [];

    let undoPush: UndoCallback = () => { };
    
    if (movedTo < 0) {
        undoPush = removeUnit(state, pushed);
    }
    else {
        const result = ArmyMovesGenerator.computeStep(state, pushed, movedTo, true)!;
        rewards.push(...result.rewards);
    }

    return {
        rewards,
        undo: () => {
            undoPush();
            pushed._moved = oldMoved;
            pushed._attacked = oldAttacked;
        }
    }
}

export function attackUnit(state: GameState, attacker: UnitState | number, defender: UnitState, attackerPov?: UnitState): UndoCallback {
    const undoChain: UndoCallback[] = [];

    if(typeof attacker == 'number') {
        const atk = calculateAttack(state, attacker, defender);

        defender._health -= atk;

        if (defender._health <= 0) {
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
        if(defender._health <= 0) {
            undoChain.push(removeUnit(state, defender, attacker));
            // Move to the enemy position, if not a ranged unit
            if (getUnitRange(attacker) < 2 && isSteppable(state, attacker, defender._tileIndex)) {
                const result = ArmyMovesGenerator.computeStep(state, attacker, defender._tileIndex, true)!;
                undoChain.push(result.undo);
            }
        }
        // Retaliate
        else {
            // If we have have freeze, then they cant retaliate
            if(isSkilledIn(attacker, SkillType.Freeze)) {
                let wasFrozen = false;
                if(!isFrozen(defender)) {
                    defender._effects.add(EffectType.Frozen);
                    wasFrozen = true;
                }
                undoChain.push(() => {
                    if(wasFrozen) {
                        defender._effects.delete(EffectType.Frozen);
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
                if (attacker._health <= 0) {
                    undoChain.push(removeUnit(state, attacker, defender));
                }
            }
        }        
    }

    return () => {
        undoChain.reverse().forEach(x => x());
    };
}