import { getNeighborTiles, calaulatePushablePosition, getNeighborIndexes, isSkilledIn, getPovTribe, getTrueUnitAt, getHomeCity, getRulingCity, getMaxHealth, getEnemiesNearTile, calculateCombat, getUnitRange, getTrueEnemyAt, calculateAttack, isSteppable, isWaterTerrain, getCapitalCity, getLighthouses, getCityOwningTile, hasEffect, getEnemiesInRange, isAquaticOrCanFly, calculateDistance, getStructureAt, getEnemyAt, getStarExchange } from "./functions";
import { EconMovesGenerator } from "./moves";
import Move, { Branch, CallbackResult, UndoCallback } from "./move";
import { ResourceSettings } from "./settings/ResourceSettings";
import { StructureSettings } from "./settings/StructureSettings";
import { UnitSettings } from "./settings/UnitSettings";
import { CityState, GameState, StructureState, UnitState } from "./states";
import { UnitType, StructureType, SkillType, TerrainType, EffectType, ResourceType } from "./types";
import { IsStructureTask } from "./settings/TaskSettings";
import { xorCity, xorPlayer, xorResource, xorStructure, xorTile, xorUnit } from "../zobrist/hasher";

export function addPopulationToCity(state: GameState, city: CityState, amount: number): CallbackResult {
    const pov = getPovTribe(state);

    if(!amount) {
        return null;
    }
    
    const cityStruct = state.structures[city.tileIndex]!;
    
    city._population += amount;
    city._progress += amount;

    const next = city._level + 1;
    
    if(city._progress >= next) {
        const lvl = city._level;

        cityStruct._level++;
        city._level++;
        city._progress -= next;
        city._production++;

        let rewards = EconMovesGenerator.rewards(city);
        let lol = false;

        if(city._progress - next >= (next + 1)) {
            console.warn('MEGA CHAIN!');
            lol = true;
            cityStruct._level++;
            city._level++;
            city._progress -= next + 1;
            city._production++;
            rewards.push(...EconMovesGenerator.rewards(city));
        }

        xorCity.level(state, city, lvl, city._level);

        return {
            rewards,
            undo: () => {
                xorCity.level(state, city, city._level, lvl);

                if(lol) {
                    city._production--;
                    city._progress += next + 1;
                    city._level--;
                    cityStruct._level--;
                }

                city._production--;
                city._progress += next;
                city._level--;
                cityStruct._level--;

                city._progress -= amount;
                city._population -= amount;
            },
        }
    }
    
    return {
        rewards: [],
        undo: () => {
            city._progress -= amount;
            city._population -= amount;
        }
    }
}

/**
 * Attempts to discover any undiscovered tribes and rewards with exchange
 * @param state
 * @returns if the disovery was successfull
 */
export function tryDiscoverRewardOtherTribes(state: GameState): UndoCallback {
	const us = getPovTribe(state);

	// Already discovered all the other tribes
	if (us._knownPlayers.size == state.settings.tribeCount - 1) {
		return () => { };
	}

	const chain: UndoCallback[] = [];

	// Try to meet new tribes, if they they have been seen and not discovered
	for (const x in state._visibleTiles) {
		// If we can see any other tribe's unit, we have met them
		const standing = getEnemyAt(state, Number(x));
		const them = standing?._owner;
		if (them && !us._knownPlayers.has(them)) {
			us._knownPlayers.add(them);
			chain.unshift(gainStars(state, getStarExchange(state, them)));
			chain.unshift(() => {
				us._knownPlayers.delete(them);
			});
		}
	}

	return () => {
		chain.forEach(x => x());
	};
}

export function modifyTerrain(state: GameState, tileIndex: number, terrainType: TerrainType): UndoCallback {
    const tile = state.tiles[tileIndex];
    const oTerrainType = tile.terrainType;
    
    xorTile.terrain(state, tileIndex, oTerrainType, terrainType);
    tile.terrainType = terrainType;

    return () => {
        tile.terrainType = oTerrainType;
        xorTile.terrain(state, tileIndex, terrainType, oTerrainType);
    }
}

export function consumeResource(state: GameState, tileIndex: number, replaceType?: ResourceType): UndoCallback {
    const oldResource = state.resources[tileIndex]!;
    const newResource = replaceType? replaceType : ResourceType.None;

    xorResource(state, tileIndex, oldResource.id, newResource);

    if(replaceType) {
        state.resources[tileIndex] = {
            id: replaceType,
            tileIndex
        }
    }
    else {
        delete state.resources[tileIndex];
    }

    return () => {
        xorResource(state, tileIndex, newResource, oldResource.id);
        state.resources[tileIndex] = oldResource;
    }
}

export function createStructure(state: GameState, tileIndex: number, strctureType: StructureType, level = 1): UndoCallback {
    // specific to ruins -> aquarion free city
    const oldStruct = state.structures[tileIndex];
    
    const structure: StructureState = {
        id: strctureType,
        _level: level,
        turn: state.settings._turn,
        tileIndex,
        reward: 0,
    };
    
    xorStructure(state, tileIndex, oldStruct? oldStruct.id : StructureType.None, strctureType);
    state.structures[tileIndex] = structure;
    
    return () => {
        xorStructure(state, tileIndex, strctureType, oldStruct? oldStruct.id : StructureType.None);
        state.structures[tileIndex] = oldStruct;
    }
}

export function destroyStructure(state: GameState, tileIndex: number): UndoCallback {
    const pov = getPovTribe(state);
    const struct = state.structures[tileIndex]!;

    xorStructure(state, tileIndex, struct.id, StructureType.None);

    if(struct.id === StructureType.Ruin) {
        delete state.structures[tileIndex];
        return () => {
            state.structures[tileIndex] = struct;
            xorStructure(state, tileIndex, StructureType.None, struct.id);
        }
    }

    const city = getCityOwningTile(state, tileIndex)!;
    const settings = StructureSettings[struct.id];

    delete state.structures[tileIndex];

    if(settings.rewardPop) {
        city._population -= settings.rewardPop;
        city._progress -= settings.rewardPop;
        if(city._progress < 0) {
            xorCity.level(state, city, city._level, city._level - 1);
            city._level--;
        }
    }

    // TODO Remove score

    return () => {
        if(settings.rewardPop) {
            if(city._progress < 0) {
                xorCity.level(state, city, city._level, city._level + 1);
                city._level++;
            }
            city._progress += settings.rewardPop;
            city._population += settings.rewardPop;
        }
        state.structures[tileIndex] = struct;
        xorStructure(state, tileIndex, StructureType.None, struct.id);
    }
}

export function gainStars(state: GameState, cost: number): UndoCallback {
    return spendStars(state, -cost);
}

export function spendStars(state: GameState, cost: number): UndoCallback {
    if(!cost) return () => {};
    const pov = getPovTribe(state);

    xorPlayer.stars(pov, pov._stars);
    pov._stars -= cost;
    xorPlayer.stars(pov, pov._stars);

    return () => {
        xorPlayer.stars(pov, pov._stars);
        pov._stars += cost;
        xorPlayer.stars(pov, pov._stars);
    }
}

export function harvestResource(state: GameState, tileIndex: number): Branch {
    const harvested = state.resources[tileIndex]!;
    const settings = ResourceSettings[harvested.id];
    const rulingCity = getCityOwningTile(state, tileIndex)!;
    
    const undoPurchase = spendStars(state, settings.cost || 0);
    const undoResource = consumeResource(state, tileIndex);
    const popBranch = addPopulationToCity(state, rulingCity, settings.rewardPop);
    
    return {
        rewards: (popBranch?.rewards || []),
        undo: () => {
            popBranch?.undo();
            undoResource();
            undoPurchase();
        }
    };
}

export function buildStructure(state: GameState, strctureType: StructureType, tileIndex: number): Branch {
    const pov = state.tribes[state.settings._pov];
    const settings = StructureSettings[strctureType];
    const rulingCity = getRulingCity(state, tileIndex)!;
    const cost = settings.cost || 0;

    const undoPurchase = spendStars(state, cost);
    const undoCreate = createStructure(state, tileIndex, strctureType);
    
    let rewardPopCount = settings.rewardPop || 0;
    
    if(settings.adjacentTypes !== undefined) {
        const adjCount = getNeighborTiles(state, tileIndex)
            .filter(x => state.structures[x.tileIndex]? settings.adjacentTypes!.has(state.structures[x.tileIndex]!.id) : false).length;
        rewardPopCount *= adjCount;
    }

    if(IsStructureTask[strctureType]) {
        pov._builtUniqueStructures.add(strctureType);
        xorPlayer.unique(pov, strctureType);
    }

    const popBranch = addPopulationToCity(state, rulingCity, rewardPopCount);
    // const portBranch = addMissingConnections(state, rulingCity, tileIndex);
    
    return {
        // rewards: [ ...(popBranch?.rewards || []), ...(portBranch?.rewards || []) ],
        rewards: (popBranch?.rewards || []),
        undo: () => {
            // portBranch?.undo();
            popBranch?.undo();
            if(IsStructureTask[strctureType]) {
                pov._builtUniqueStructures.delete(strctureType);
                xorPlayer.unique(pov, strctureType);
            }
            undoCreate();
            undoPurchase();
        }
    };
}

export function summonUnit(state: GameState, unitType: UnitType, spawnTileIndex: number, costs = false, forceIndependent = false): CallbackResult {
    const pov = getPovTribe(state);
    const settings = UnitSettings[unitType];
    const health = UnitSettings[unitType].health!;
    
    const spawnTile = state.tiles[spawnTileIndex];
    
    // Push occupied unit away (if any)
    let resultPush = pushUnit(state, spawnTile.tileIndex);
    
    const oldUnitOwner = spawnTile._unitOwner;
    
    const undoPurchase = costs? spendStars(state, settings.cost) : () => {};
    
    const spawnedUnit = {
        _unitType: unitType,
        _health: health * 10,
        _kills: 0,
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

    xorUnit.set(state, spawnedUnit);

    pov._units.push(spawnedUnit);

    spawnTile._unitOwner = spawnedUnit._owner;
    
    const cityHome = forceIndependent? null : getHomeCity(state, spawnedUnit);

    if(cityHome) {
        xorCity.unitCount(state, cityHome, cityHome._unitCount, cityHome._unitCount + 1);
        cityHome._unitCount++;
    }
    
	const resultDiscover = discoverTiles(state, spawnedUnit);

    const undoFrozen: UndoCallback = isSkilledIn(spawnedUnit, SkillType.AutoFreeze, SkillType.FreezeArea)? 
        freezeArea(state, spawnedUnit) : () => { };

    return {
        rewards: [...(resultDiscover?.rewards || []), ...(resultPush?.rewards || [])],
        undo: () => {
            undoFrozen();
            resultDiscover?.undo();
            if(cityHome) {
                xorCity.unitCount(state, cityHome, cityHome._unitCount, cityHome._unitCount - 1);
                cityHome._unitCount--;
            }
            spawnTile._unitOwner = oldUnitOwner;
            undoPurchase();
            state.settings.unitIdx--;
            pov._units.pop();
            resultPush?.undo();
            xorUnit.set(state, spawnedUnit);
        }
    }
}

export function removeUnit(state: GameState, removed: UnitState, killer?: UnitState): UndoCallback {
    const pov = state.tribes[removed._owner];
    const tile = state.tiles[removed._tileIndex];
    const oldOwner = removed._owner;
    const cityHome = getHomeCity(state, removed);
    const atIndex = pov._units.findIndex(x => x._tileIndex == removed._tileIndex);

    xorUnit.set(state, removed);

    pov._units.splice(atIndex, 1);
    tile._unitOwner = 0;

    if(cityHome) {
        xorCity.unitCount(state, cityHome, cityHome._unitCount, cityHome._unitCount - 1);
        cityHome._unitCount--
    }

    if(killer) {
        xorUnit.kills(state, killer, killer._kills, killer._kills + 1);
        killer._kills++;
        pov._casualties++;
        state.tribes[killer._owner]._kills++;
    }
    
    return () => {
        if(killer) {
            state.tribes[killer._owner]._kills--;
            pov._casualties--;
            xorUnit.kills(state, killer, killer._kills, killer._kills - 1);
            killer._kills--;
        }
        
        if(cityHome) {
            xorCity.unitCount(state, cityHome, cityHome._unitCount, cityHome._unitCount - 1);
            cityHome._unitCount++;
        }

        tile._unitOwner = oldOwner;
        pov._units.splice(atIndex, 0, removed);

        xorUnit.set(state, removed);
    };
}

export function healUnit(state: GameState, unit: UnitState, amount: number): UndoCallback {
    if(hasEffect(unit, EffectType.Poison))	{
        return tryRemoveEffect(state, unit, EffectType.Poison);
    }
    const oldHealth = unit._health;
    unit._health += amount;
    unit._health = Math.min(unit._health, getMaxHealth(unit));
    return () => {
        unit._health = oldHealth;
    };
}

export function tryAddEffect(state: GameState, unit: UnitState, effect: EffectType): UndoCallback {
    if(hasEffect(unit, effect)) {
        return () => { };
    }
    xorUnit.effect(state, unit, effect);
    unit._effects.add(effect);
    return () => {
        unit._effects.delete(effect);
        xorUnit.effect(state, unit, effect);
    }
}

export function tryRemoveEffect(state: GameState, unit: UnitState, effect: EffectType): UndoCallback {
    if(!hasEffect(unit, effect)) {
        return () => { };
    }
    xorUnit.effect(state, unit, effect);
    unit._effects.delete(effect);
    return () => {
        unit._effects.add(effect);
        xorUnit.effect(state, unit, effect);
    }
}

export function discoverTiles(state: GameState, unit?: UnitState | null, tileIndexes?: number[]): CallbackResult {
    const pov = getPovTribe(state);
    const discovered = (tileIndexes || (unit? getNeighborIndexes(
        state,
        unit._tileIndex,
        state.tiles[unit._tileIndex].terrainType == TerrainType.Mountain || isSkilledIn(unit, SkillType.Scout)? 2 : 1,
        false,
        true
    ) : [])).filter(x => !state._visibleTiles[x]);

    const missingLighthouses = getLighthouses(state, false);

    let chain: UndoCallback[] = [];
    let rewards: Move[] = [];

    for(const tileIndex of discovered) {
        xorTile.discover(state, state.tiles[tileIndex]);

        if(missingLighthouses.includes(tileIndex)) {
            const city = getCapitalCity(state);
            if(city) {
                const result = addPopulationToCity(state, city, 1);
                if(result) {
                    chain.push(result?.undo);
                    rewards.push(...result.rewards);
                }
            }
        }

        if(state.settings.areYouSure) {
            state.tiles[tileIndex]._explorers.add(pov.owner);
        }

        state._visibleTiles[tileIndex] = true;
    }

    return {
        rewards,
        undo: () => {
            chain.forEach(x => x());

            discovered.forEach(x => {
                xorTile.discover(state, state.tiles[x]);

                if(state.settings.areYouSure) {
                    state.tiles[x]._explorers.delete(pov.owner);
                }

                state._visibleTiles[x] = false;
            });
        }
    }
}


export function setUnitMove(state: GameState, unit: UnitState): UndoCallback {
    xorUnit.moved(state, unit);

    return () => {
        xorUnit.moved(state, unit);
    }
}

export function setUnitAttack(state: GameState, unit: UnitState): UndoCallback {
    xorUnit.attacked(state, unit);

    return () => {
        xorUnit.attacked(state, unit);
    }
}

export function endUnitTurn(state: GameState, unit: UnitState): UndoCallback {
    const moved = unit._moved;
    const attacked = unit._attacked;

    if(!attacked) {
        xorUnit.attacked(state, unit);
        unit._attacked = true;
    }

    if(!moved) {
        xorUnit.moved(state, unit);
        unit._moved = true;
    }

    return () => {
        if(!attacked) {
            xorUnit.attacked(state, unit);
            unit._attacked = false;
        }

        if(!moved) {
            xorUnit.moved(state, unit);
            unit._moved = false;
        }
    }
}

export function startUnitTurn(state: GameState, unit: UnitState): UndoCallback {
    const moved = unit._moved;
    const attacked = unit._attacked;

    if(attacked) {
        xorUnit.attacked(state, unit);
        unit._attacked = false;
    }

    if(moved) {
        xorUnit.moved(state, unit);
        unit._moved = false;
    }

    return () => {
        if(attacked) {
            xorUnit.attacked(state, unit);
            unit._attacked = true;
        }

        if(moved) {
            xorUnit.moved(state, unit);
            unit._moved = true;
        }
    }
}

export function freezeArea(state: GameState, freezer: UnitState): UndoCallback {
    const chain: UndoCallback[] = [];
    const adjacent = getNeighborIndexes(state, freezer._tileIndex, 1, false, true);

    for (let i = 0; i < adjacent.length; i++) {
        const tile = state.tiles[adjacent[i]];
        const occupied = getTrueEnemyAt(state, tile.tileIndex, freezer._owner);

        // Freeze any adjacent enemy unit
        if(occupied) {
            chain.push(tryAddEffect(state, occupied, EffectType.Frozen));
        }

        // Freeze any adjacent freezable tiles
        if(tile.terrainType == TerrainType.Water || tile.terrainType == TerrainType.Ocean) {
            chain.push(modifyTerrain(state, tile.tileIndex, TerrainType.Ice));
        }
    }

    return () => {
        chain.forEach(x => x());
    }
}

export function splashDamageArea(state: GameState, attacker: UnitState, atk: number): UndoCallback {
    const undoChain = getEnemiesNearTile(state, attacker._tileIndex)
        .map(enemy => attackUnit(state, atk, enemy, attacker)?.undo!);
    return () => {
        undoChain.forEach(x => x());
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
        const result = stepUnit(state, pushed, movedTo, true)!;
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

export function attackUnit(state: GameState, attacker: UnitState | number, defender: UnitState, attackerPov?: UnitState): CallbackResult {
    const undoChain: UndoCallback[] = [];
    const rewards = [];

    if(typeof attacker == 'number') {
        const atk = calculateAttack(state, attacker, defender);

        defender._health -= atk;

        if(defender._health <= 0) {
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
                const result = stepUnit(state, attacker, defender._tileIndex, true)!;
                rewards.push(...result.rewards);
                undoChain.push(result.undo);
            }
        }
        // Retaliate
        else {
            // If we have have freeze, then they cant retaliate
            if(isSkilledIn(attacker, SkillType.Freeze)) {
                undoChain.push(tryAddEffect(state, defender, EffectType.Frozen));
            }
            // If we're attacking with range
            else if(getUnitRange(attacker) > 1 && result.defenseDamage > 0) {
                const dist = calculateDistance(attacker._tileIndex, defender._tileIndex, state.settings.size);
                // if defender cant reach us, they cant retaliate
                if(dist > getUnitRange(defender)) {
                    result.defenseDamage = 0;
                }
                // technically not cheating
                // if defender that cannot see us, they cant retaliate
                else if(state.settings.areYouSure) {
                    if (!state.tiles[defender._tileIndex]._explorers.has(defender._owner)) {
                        result.defenseDamage = 0;
                    }
                }
            }

            if(result.defenseDamage > 0) {
                attacker._health -= result.defenseDamage;
    
                undoChain.push(() => {
                    attacker._health += result.defenseDamage;
                });
    
                // Our unit died
                if(attacker._health <= 0) {
                    undoChain.push(removeUnit(state, attacker, defender));
                }
            }
        }        
    }

    return {
        rewards: rewards,
        undo: () => {
            undoChain.reverse().forEach(x => x());
        }
    }
}

export function stepUnit(state: GameState, stepper: UnitState, toTileIndex: number, involuntary = false): CallbackResult {
	const chain: UndoCallback[] = [];
	const rewards = [];
	const movedBefore = stepper._moved;
	const oldTileIndex = stepper._tileIndex;
	const oldType = stepper._unitType;
	const oldPassenger = stepper._passenger;

	// // TODO; this is not how prev works, it must be applies at the end of the turn
	// stepper.prevX = iX;
	// stepper.prevY = iY;

    // xor out the current unit
    xorUnit.set(state, stepper);

	state.tiles[stepper._tileIndex]._unitOwner = 0;
	stepper._tileIndex = toTileIndex;
	state.tiles[stepper._tileIndex]._unitOwner = stepper._owner;

	// TODO what other skills are missing?

	// Discover terrain
	const resultDiscover = discoverTiles(state, stepper)!;
	rewards.push(...resultDiscover.rewards);
	chain.push(resultDiscover.undo);

    // Units with skate do not loose their turn when pushed
    if(!involuntary || !isSkilledIn(stepper, SkillType.Skate)) {
        // xor to true
        chain.push(endUnitTurn(state, stepper));
    }

    // ! Stomp ! //

	if(isSkilledIn(stepper, SkillType.Stomp)) {
		chain.push(splashDamageArea(state, stepper, 4));
	}

    // ! AutoFreeze //

    if(isSkilledIn(stepper, SkillType.AutoFreeze, SkillType.FreezeArea)) {
	    chain.push(freezeArea(state, stepper));
    }

	// ! Embark ! //
    
	// If a non aquatic unit is moving to our port, place into boat
	if(getStructureAt(state, toTileIndex) && !isAquaticOrCanFly(stepper)) {
		switch (stepper._unitType) {
			case UnitType.Cloak:
				stepper._unitType = UnitType.Dinghy;
				break;
			case UnitType.Dagger:
				stepper._unitType = UnitType.Pirate;
				break;
			case UnitType.Giant:
				stepper._unitType = UnitType.Juggernaut;
				break;
			default:
				stepper._unitType = UnitType.Raft;
				stepper._passenger = oldType;
				break;
		}
	}

    // ! Disembark ! //

	// Carry allows a unit to carry another unit inside
	// A unit with the carry skill can move to a land tile adjacent to water
	// Doing so releases the unit it was carrying and ends the unit's turn
	else if(isSkilledIn(stepper, SkillType.Carry) && !isWaterTerrain(state.tiles[stepper._tileIndex])) {
		stepper._passenger = undefined;
		switch (stepper._unitType) {
			case UnitType.Dinghy:
				stepper._unitType = UnitType.Cloak;
				break;
			case UnitType.Pirate:
				stepper._unitType = UnitType.Dagger;
				break;
			case UnitType.Juggernaut:
				stepper._unitType = UnitType.Giant;
				break;
			default:
				stepper._unitType = oldPassenger!;
				break;
		}
	}

    // ! Hide ! //

	// Going stealth mode uses up our attack
	else if(isSkilledIn(stepper, SkillType.Hide) && !hasEffect(stepper, EffectType.Invisible)) {
        chain.push(tryAddEffect(state, stepper, EffectType.Invisible));
	}

    // ! Dash ! //

	// Allows a unit to attack after moving if there are any enemies in range
	// And if it HAS moved before (this avoids infinite move -> attack loop)
	else if(!involuntary && !movedBefore && isSkilledIn(stepper, SkillType.Dash) && getEnemiesInRange(state, stepper).length > 0) {
		stepper._attacked = false;
	}

    // xor back out true attack
    if(!stepper._attacked) {
        xorUnit.attacked(state, stepper);    
    }

    // xor in the new unit
    xorUnit.set(state, stepper);

	return {
        rewards,
		undo: () => {
            // xor out the new unit
            xorUnit.set(state, stepper);

            if(!stepper._attacked) {
                xorUnit.attacked(state, stepper);    
            }

            stepper._unitType = oldType;
			stepper._passenger = oldPassenger;

			chain.reverse().forEach(x => x());

			state.tiles[stepper._tileIndex]._unitOwner = 0;
			stepper._tileIndex = oldTileIndex;
			state.tiles[stepper._tileIndex]._unitOwner = stepper._owner;

            // xor in the current unit
            xorUnit.set(state, stepper);
		}
	};
}
