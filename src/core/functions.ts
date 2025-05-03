import { CityState, GameState, TileState, TribeState, UnitState } from "./states";
import { TechnologyType, ResourceType, RewardType, SkillType, TerrainType, StructureType, EffectType, CombatResult, UnitType } from "./types";
import { ResourceSettings } from "./settings/ResourceSettings";
import { UnitSettings } from "./settings/UnitSettings";
import { TechnologySettings } from "./settings/TechnologySettings";
import Game from "../game";

/**
 * Attempts to discover any undiscovered tribes and overrides the passed state
 * @param state 
 * @returns if the disovery was successfull
 */
export function tryDiscoverRewardOtherTribes(state: GameState): boolean {
	const us = getPovTribe(state);

	// Already discovered all the tribes or all the tiles
	if(us._knownPlayers.length == state.settings.tribeCount) return false;

	const tribesMet: number[] = [];

	// Try to meet new tribes, if they they have been seen and not discovered

	state._visibleTiles.forEach(x => {
		// If we can see any other tribe's unit, we have met them
		const standing = getUnitAtTile(state, x);
		if(standing 
			&& standing._owner != us.owner
			&& !us._knownPlayers.includes(standing._owner)
			&& !tribesMet.includes(standing._owner)
		) {
			tribesMet.push(standing._owner);
		}
	});

	// Reward stars for met tribes
	tribesMet.forEach(owner => {
		const them = state.tribes[owner];
		us._knownPlayers.push(owner);
		us._stars += getStarExchange(state, them);

		if(them._knownPlayers.includes(us.owner)) {
			return;
		}

		// If they also too just met us
		for(const unit of us._units) {
			if(state.tiles[unit._tileIndex].explorers.includes(them.owner)) {
				them._knownPlayers.push(us.owner);
				them._stars += getStarExchange(state, us);
				break;
			}
		}
	});

	return true;
}

/**
* Uses initial state tech to verify if the tribe can see the resource
* @param tribe 
* @param resourceId 
* @returns 
*/
export function isResourceVisible(tribe: TribeState, resourceId: ResourceType | undefined): boolean {
	if(!resourceId) return false;
	const settings = ResourceSettings[resourceId];
	return !settings.visibleRequired || tribe._trueTech.some(x => settings.visibleRequired!.includes(x));
}

export function isTileVisible(state: GameState, tileIndex: TileState | number, matchOwner?: number): boolean {
	return state.tiles[typeof tileIndex === 'number'? tileIndex : tileIndex.tileIndex].explorers.includes(matchOwner || state.settings._pov);
}

export function isTechLocked(tribe: TribeState, techId: TechnologyType): boolean {
	return techId === TechnologyType.Unbuildable || !((techId === TechnologyType.None || tribe._tech.some(x => (x === techId))));
}

export function isTempleStructure(structType: StructureType) {
	return structType === StructureType.Temple 
		|| structType === StructureType.MountainTemple 
		|| structType === StructureType.ForestTemple 
		|| structType === StructureType.IceTemple 
		|| structType === StructureType.WaterTemple;
}

export function getTechTier(tech: TechnologyType): number {
	const settings = TechnologySettings[tech];
	return settings.replaced? getTechTier(settings.replaced) : settings.tier!;
}

export function getTechCost(tribe: TribeState, techType: TechnologyType): number {
	let cost = getTechTier(techType) * tribe._cities.length + 4;
	if(cost != 0 && !isTechLocked(tribe, TechnologyType.Philosophy)) {
		cost = Math.ceil(cost * .77);
	}
	return cost;
}

/** Returns the correct city production */
export function getCityProduction(state: GameState, ...city: CityState[]): number {
	// If riot or tile is occupied by an enemy unit = 0
	return city.reduce((a, b) => a + (b._riot || getEnemyAtTile(state, b.tileIndex)? 0 : b._production), 0);
}

export function getTerritorryTiles(state: GameState, tribe: TribeState): TileState[] {
	return tribe._cities.reduce((a: number[], b) => ([...a, ...b._territory]), []).map(x => state.tiles[x]);
}

export function getNeighborIndexes(state: GameState, index: number, range = 1, unowned = false, includeUnexplored=false): number[] {
	const width = state.settings.size;
	const neighbors: number[] = [];
	const x = index % width;
	const y = Math.floor(index / width);
	
	for (let dx = -range; dx <= range; dx++) {
		for (let dy = -range; dy <= range; dy++) {
			if (dx === 0 && dy === 0) continue;
			
			const neighborX = x + dx;
			const neighborY = y + dy;
			
			if (neighborX < 0 || neighborX >= width || neighborY < 0 || neighborY >= width) continue;
			
			const neighborIndex = neighborY * width + neighborX;
			
			if(!includeUnexplored || unowned) {
				const explored = state.tiles[neighborIndex].explorers.includes(state.settings._pov);
	
				// Skip unexplored
				if(!includeUnexplored && !explored) continue;
				
				// Optionally filter for unowned tiles.
				if (unowned && (explored? state.tiles[neighborIndex]._owner > 0 : false)) continue;
			}
			
			neighbors.push(neighborIndex);
		}
	}
	
	return neighbors;
}

export function getNeighborTiles(state: GameState, index: number, range = 1, unowned = false, includeUnexplored = false): TileState[] {
	return getNeighborIndexes(state, index, range, unowned, includeUnexplored).map(i => state.tiles[i]);
}

export function getTrueUnitAtTile(state: GameState, tileIndex: TileState | number, matchOwner?: number): UnitState | null {
	const tile = state.tiles[typeof tileIndex === 'number'? tileIndex : tileIndex.tileIndex];
	if(tile._unitIdx < 0) return null;
	const found = state.tribes[tile._unitOwner]._units.find(x => x.idx === tile._unitIdx) || null;
	return found && matchOwner? found._owner === matchOwner? found : null : found;
}

export function getUnitAtTile(state: GameState, tileIndex: TileState | number, matchOwner?: number): UnitState | null {
	if(!isTileVisible(state, tileIndex, matchOwner)) {
		return null;
	}

	const found = getTrueUnitAtTile(state, tileIndex, matchOwner);

	if(!found) return null;

	// If enemy unit is hidden, then we cant see it!
	if(isInvisible(found) && found._owner !== state.settings._pov) {
		return null;
	}

	return matchOwner && found? found._owner === matchOwner? found : null : found;
}

export function getCity(state: GameState, tileIndex: number, matchOwner?: number): CityState | null {
	if(!isTileVisible(state, tileIndex, matchOwner)) {
		return null;
	}
	return Object.values(state.tribes).map(x => x._cities.find(y => y.tileIndex == tileIndex)).filter(Boolean)[0] || null;
}


export function getEnemyAtTile(state: GameState, tileIndex: TileState | number, notMatchOwner?: number): UnitState | null {
	const found = getUnitAtTile(state, tileIndex);
	if(!found) return null;
	return found._owner != (notMatchOwner || state.settings._pov)? found : null;
}

export function getTrueEnemyAtTile(state: GameState, tileIndex: TileState | number, notMatchOwner?: number): UnitState | null {
	const found = getTrueUnitAtTile(state, tileIndex);
	if(!found) return null;
	return found._owner != (notMatchOwner || state.settings._pov)? found : null;
}

export function getAlliesNearTile(state: GameState, tileIndex: number, range = 1): UnitState[] {
	return getNeighborIndexes(state, tileIndex, range)
		.reduce((acc: UnitState[], cur: number) => {
			const ally = getTrueUnitAtTile(state, cur, state.settings._pov);
			return [
				...acc,
				...ally? [ally] : [],
			];
		}, []);
}

export function getEnemiesNearTile(state: GameState, tileIndex: number, range = 1, real = false): UnitState[] {
	return getNeighborIndexes(state, tileIndex, range)
		.reduce((acc: UnitState[], cur: number) => {
			const owner = state.tiles[cur]._unitOwner;
			if(owner < 1 || owner === state.settings._pov) return acc;
			const enemy = (real? getTrueEnemyAtTile  : getEnemyAtTile)(state, cur);
			if(!enemy) return acc;
			if(cur != enemy._tileIndex) {
				throw 'FATAL MISMATCH';
			}
			return [...acc, enemy];
		}, []);
}

export function getEnemiesInRange(state: GameState, unit: UnitState) {
	return getEnemiesNearTile(state, unit._tileIndex, getUnitRange(unit));
}

export function getClosestEnemyCity(state: GameState, tileIndex: number, range = 1): [null | CityState, number] {
	let closestDistance = range;
	let closestCity = null;
	for (let i = 1; i < state.settings.tribeCount; i++) {
		if(i === state.settings._pov) continue;
		for (const city of state.tribes[i]._cities.filter(x => state._visibleTiles.includes(x.tileIndex))) {
			const distance = calculateDistance(tileIndex, city.tileIndex, state.settings.size);
			if (distance < closestDistance || (distance === closestDistance && state.tiles[i].capitalOf)) {
				closestDistance = distance;
				closestCity = city;
			}
		}
	}
	return [closestCity, closestDistance];
}

/** @Obsolete */
export function getClosestReachableEnemyCity(state: GameState, unit: UnitState, range = -1): [CityState | null, number] {
	range = range < 0? state.settings.size + 1 : range;

	let closestDistance = Number.POSITIVE_INFINITY;
	let closestCity = null;

	for (let i = 0; i < state._visibleTiles.length; i++) {
		const tileIndex = state._visibleTiles[i];
		const tile = state.tiles[tileIndex];
			
		if(tile._rulingCityIndex != tileIndex || tile._owner === state.settings._pov) continue;
		
		let ported = false;

		const path = computeReachablePath(state, unit._tileIndex, tileIndex, (state: GameState, index: number) => {
			return isSteppable(state, unit, tile);
		});

		const distance = path.length;

		if(distance < 1) continue;

		// EVAL Deadly bias preferring capital over potentially closer cities
		if (distance < closestDistance || (distance === closestDistance && state.tiles[i].capitalOf)) {
			closestDistance = distance;
			closestCity = state.tribes[tile._owner]._cities.find(x => x.tileIndex === tileIndex)!;
		}
	}

	return [closestCity, closestDistance];
}

export function isWaterTerrain(tile: TileState): boolean {
	return tile.terrainType === TerrainType.Water || tile.terrainType === TerrainType.Ocean;
}

export function isIceTerrain(tile: TileState): boolean {
	// TODO THIS SHOULD BE: tile.frozen?
	return tile.terrainType === TerrainType.Ice;
}

export function isSkilledIn(unit: UnitState | UnitType, ...skills: SkillType[]): boolean {
	const settings = UnitSettings[typeof unit === "number"? unit : unit._unitType];
	const passengerSettings = typeof unit != "number" && unit._passenger? UnitSettings[unit._passenger] : null;
	return skills.some(x => settings.skills.includes(x) || passengerSettings?.skills.includes(x));
}

export function isAquaticOrCanFly(unit: UnitState | UnitType, canfly: boolean = true): boolean {
	return isSkilledIn(unit, 
		...(canfly? [SkillType.Fly] : []),
		SkillType.Carry,
		SkillType.Float,
		SkillType.Navigate,
		SkillType.Splash,
	);
}

export function isNavalUnit(unit: UnitState | UnitType): boolean {
	return isSkilledIn(unit, SkillType.Carry, SkillType.Float, SkillType.Splash);
}

export function isInvisible(unit: UnitState): boolean {
	return unit._effects.includes(EffectType.Invisible);
}

export function isFrozen(unit: UnitState): boolean {
	return unit._effects.includes(EffectType.Frozen);
}

export function isPoisoned(unit: UnitState): boolean {
	return unit._effects.includes(EffectType.Poison);
}

export function isBoosted(unit: UnitState): boolean {
	return unit._effects.includes(EffectType.Boost);
}

export function isInTerritory(state: GameState, unit: UnitState) {
	return state.tiles[unit._tileIndex]._owner == unit._owner;
}


export function isRoadpathAndUsable(state: GameState, unit: UnitState, tileIndex: number) {
	// TODO Friendly terrain
	const tile = state.tiles[tileIndex];
	// Is owned by this unit or is neutral, and has a road or is a city
	return (tile._owner == unit._owner || tile._owner < 1) && (tile.hasRoad || tile._rulingCityIndex == tile.tileIndex);	
}

export function getDefenseBonus(state: GameState, unit: UnitState): number {
	if (unit._effects.includes(EffectType.Poison)) {
		return 0.7;
	}
	
	const tribe = state.tribes[unit._owner];
	
	switch (state.tiles[unit._tileIndex].terrainType) {
		case TerrainType.Water:
		case TerrainType.Ocean:
			if(!isTechLocked(tribe, TechnologyType.Aquatism)) {
				return 1.5;
			}
			break;
		case TerrainType.Forest:
			if(!isTechLocked(tribe, TechnologyType.Archery)) {
				return 1.5;
			}
			break;
		case TerrainType.Mountain:
			if(!isTechLocked(tribe, TechnologyType.Climbing)) {
				return 1.5;
			}
		break;
		default:
			const ownCity = state.tribes[unit._owner]._cities.find(x => x.tileIndex == unit._tileIndex);
			//  City defense
			if(ownCity && isSkilledIn(unit, SkillType.Fortify)) {
				return ownCity._rewards.includes(RewardType.CityWall)? 4 : 1.5;
			}
			break;
	}
	
	return 1;
}

export function canCapture(state: GameState, unit: UnitState): boolean {
	const struct = state.structures[unit._tileIndex];
	return Boolean(struct && struct.id === StructureType.Village && struct._owner != unit._owner && !unit._moved && !unit._attacked);
}

export function isAdjacentToEnemy(state: GameState, tile: TileState): boolean {
	// Get true enemy because invisible units (cloaks) can also control terrain
	return getNeighborIndexes(state, tile.tileIndex).some(x => getTrueEnemyAtTile(state, x));
}

// TODO THIS IS AMBIGUOUS, ONLY WORKS WITH 1v1
export function isGameOver(state: GameState): boolean {
	const tribe = getPovTribe(state);
	return state.settings._gameOver || state.settings._turn > state.settings.maxTurns || isGameLost(state) || isGameWon(state);
}

export function isGameLost(state: GameState): boolean {
	const tribe = getPovTribe(state);
	return tribe._resignedTurn > 0 || tribe._killedTurn > 0;
}

export function isGameWon(state: GameState): boolean {
	return Object.values(state.tribes).every(x => x.owner === state.settings._pov? true : x._resignedTurn > 0 || x._killedTurn > 0);
}

export function getWipeouts(state: GameState, owner?: number): TribeState[] {
	owner = owner || state.settings._pov;
	return Object.values(state.tribes).filter(x => x._killerId === owner);
}

export function isSteppable(state: GameState, unit: UnitState, tileOrIndex: TileState | number, overrideAquatic?: boolean) {
	const tile = typeof tileOrIndex === "number"? state.tiles[tileOrIndex] : tileOrIndex;

	// Unexplored
	if (!tile.explorers.includes(unit._owner)) {
		return false;
	}

	// Occupied
	if(getUnitAtTile(state, tile.tileIndex)) {
		return false;
	}

	// Fly
	if (isSkilledIn(unit, SkillType.Fly)) {
		return true;
	}

	const tribe = state.tribes[unit._owner];

	// Mountain
	if (tile.terrainType === TerrainType.Mountain) {
		return !isTechLocked(tribe, TechnologyType.Climbing);
	}

	const isAquatic = overrideAquatic? true : isAquaticOrCanFly(unit, false);

	// Port
	const isPort = state.structures[tile.tileIndex]?.id === StructureType.Port;
	if (isPort) {
		// Sailing must be unlocked to enter a port
		if(!isTechLocked(tribe, TechnologyType.Sailing)) return false;
		return isAquatic || !isAquatic && tile._owner === unit._owner;
	}
	
	if (tile.terrainType === TerrainType.Water || tile.terrainType === TerrainType.Ocean) {
		if (!isAquatic) return false;

		// If unit has Navigate, it cannot move onto land, except for capturing cities
		if (isSkilledIn(unit, SkillType.Navigate)) {
			if(!isWaterTerrain(tile) && state.structures[tile.tileIndex]?.id !== StructureType.Village) {
				return false;
			}
			return true;
		}

		// Shallow requires fishing, ocean requires sailing
		return !isTechLocked(tribe, tile.terrainType === TerrainType.Water? TechnologyType.Fishing : TechnologyType.Sailing);
	}

	return true;
}

export function isTribeSteppable(state: GameState, tileIndex: number) {
	switch (state.tiles[tileIndex].terrainType) {
		case TerrainType.Water:	
			return !isTechLocked(state.tribes[state.settings._pov], TechnologyType.Fishing);
			
		case TerrainType.Ocean:	
			return !isTechLocked(state.tribes[state.settings._pov], TechnologyType.Sailing);

		case TerrainType.Mountain:	
			return !isTechLocked(state.tribes[state.settings._pov], TechnologyType.Climbing);

		default:
			return true;
	}
}

export function getPovTribe(stateOrState: GameState | Game): TribeState {
	const state = stateOrState instanceof Game? stateOrState.state : stateOrState;
	return state.tribes[state.settings._pov];
}

export function getCapitalCity(state: GameState, owner?: number): CityState | null {
	const pov = owner || state.settings._pov;
	return state.tribes[pov]._cities.find(x => state.tiles[x.tileIndex].capitalOf === pov) || null;
}

export function getTribeCrudeScore(state: GameState, owner?: number): number {
	const tribe = state.tribes[owner || state.settings._pov];

	// ! https://docs.google.com/document/d/1HYiUbT-3RtP4b2SwlMQEZB4bTdAUtN_6K8DOvY6wNsk/edit?tab=t.0

	let score = 0;

	// 100 xp per level, 20 xp per owned territory, 5 xp per population
	for(const city of tribe._cities) {
		score += city._level * 100 
			+ city._territory.length * 20
			+ city._population * 5;

		// Not sure if this is correct
		// 40 for the city itself, 5 for each reward after the first (border growth is not counted)
		// Clamping to a max level of 6 to avoid negative values
		score += city._rewards.length > 1? 40 + Math.max((city._rewards.length - 1), 6) * 5 : 0;

		if(city._rewards.includes(RewardType.Park)) {
			score += 300;
		}
	}

	// 5 xp per revealed tile
	score += Object.values(state.tiles).reduce((x, y) => x + (y.explorers.includes(tribe.owner)? 1 : 0), 0) * 5;

	// 5 xp per star of cost
	for(const unit of tribe._units) {
		score += 5 * UnitSettings[unit._unitType].cost;
	}
	
	// 5 100 per tech tier
	for(const tech of tribe._tech) {
		score += 100 * getTechTier(tech);
	}

	// console.log(TribeType[tribe.tribeType], score);

	return score;
}

/**
 * Returns the real unit's settings, ignoring naval types
 * @param unit 
 * @returns 
 */
export function getRealUnitSettings(unit: UnitState) {
	return getUnitSettings(getRealUnitType(unit));
}

export function getRealUnitType(unit: UnitState): UnitType {
	return unit._passenger || unit._unitType;
}

/**
 * Returns the unit's settings, including naval types
 * @param unit 
 * @returns 
 */
export function getUnitSettings(unit: UnitState | UnitType) {
	return UnitSettings[typeof unit === "number"? unit : unit._unitType];
}

export function getMaxHealth(unit: UnitState) {
	let hp = getRealUnitSettings(unit).health;
	if(!hp) throw Error(`Yo no health bro tf "${unit._unitType}, ${unit._passenger}, ${unit._tileIndex}"`);
	if(unit.veteran) hp += 5;
	return hp * 10;
}

export function getUnitAttack(unit: UnitState) {
	return getUnitSettings(unit).attack;
}

export function getUnitMovement(unit: UnitState) {
	const movement = getUnitSettings(unit).movement;
	if(!movement) throw Error(`Yo no movement bro tf "${unit._unitType}"`);
	return movement;
}

export function getUnitDefense(unit: UnitState) {
	return getRealUnitSettings(unit).defense;
}

export function getUnitRange(unit: UnitState) {
	const range = getUnitSettings(unit).range;
	if(!range) throw Error(`Yo no range bro tf "${unit._unitType}"`);
	return getUnitSettings(unit).range;
}

export function getStarExchange(state: GameState, owner: TribeState | number) {
	const score = state.tribes[typeof owner === "number"? owner : owner.owner]._score;
	if(score < 1000) return 3;
	if(score < 2000) return 6;
	if(score < 3000) return 9;
	return 12;
}

export function getClosestOwnedTile(state: GameState, tileIndex: number, tileType: TerrainType) {
	return state._visibleTiles.reduce((best: any, cur: number) => {
		if(state.tiles[cur]._owner != state.settings._pov || state.tiles[cur].terrainType != tileType) {
			return best;
		}
		const dist = calculateDistance(tileIndex, cur, state.settings.size);
		return best == null || dist < best[1]? [cur, dist] : best;
	}, null);
}

export function getClosestOwnedStructureTile(state: GameState, tileIndex: number, structType: StructureType): [number, number] | null {
	return state._visibleTiles.reduce((best: any, cur: number) => {
		if(cur === tileIndex) return best;
		if(state.tiles[cur]._owner != state.settings._pov || state.structures[cur]?.id != structType) {
			return best;
		}
		const dist = calculateDistance(tileIndex, cur, state.settings.size);
		return best == null || dist < best[1]? [cur, dist] : best;
	}, null);
}

export function getHomeCity(state: GameState, unit: UnitState): CityState | null {
	return isSkilledIn(unit, SkillType.Independent) || unit._homeIndex < 0? null : getPovTribe(state)._cities.find(x => x.tileIndex == unit._homeIndex) || null;
}

export function getRulingCity(state: GameState, tileIndex: number): CityState | null {
	// state.tribes[themOwner]!._cities.find(x => x.tileIndex === targetCityIndex)
	const tile = state.tiles[tileIndex];
	return state.tribes[tile._owner]?._cities.find(x => x.tileIndex === tile._rulingCityIndex) || null;
}


let distanceCache = new Map<number, Map<number, number>>();

export function calculateDistance(tileIndex1: number, tileIndex2: number, size: number, manhattan = false) {
	const cache = distanceCache.get(tileIndex1);
	if (cache) {
		const cachedDistance = cache.get(tileIndex2);
		if (cachedDistance != undefined) return cachedDistance;
	} else {
		distanceCache.set(tileIndex1, new Map());
	}
	const dx = Math.abs((tileIndex1 % size) - (tileIndex2 % size));
	const dy = Math.abs(Math.floor(tileIndex1 / size) - Math.floor(tileIndex2 / size));
	const distance = manhattan? Math.abs(dx + dy) : Math.max(dx, dy);
	distanceCache.get(tileIndex1)!.set(tileIndex2, distance);
	return distance;
}

/**
 * Pushes a unit away from its current tile.
 * @param state
 * @param pushed
 * @returns - The index of the tile where it ended up. -1 if it failed to move the unit.
 */
export function calaulatePushablePosition(state: GameState, pushed: UnitState): number {
	let initialX = pushed.x;
	let initialY = pushed.y;
	let modifiedX = pushed.x;
	let modifiedY = pushed.y;

	const doPush = (dx: number, dy: number) => {
		const newX = modifiedX + dx;
		const newY = modifiedY + dy;
		const tile = state.tiles[newX + newY * state.settings.size];

		if (isSteppable(state, pushed, tile)) {
			modifiedX = newX;
			modifiedY = newY;
			return true;
		}
		return false;
	};

	let dx = 0, dy = 0;
	const centerTile = state.tiles[Math.floor((state.settings.size * state.settings.size) / 2)];

	if (modifiedX !== -1 &&
		pushed.prevY !== -1 &&
		modifiedX != initialX &&
		pushed.prevY != initialY)
	{
		dx = modifiedX === pushed.x ? 0 : modifiedX < pushed.x ? 1 : -1;
		dy = pushed.prevY === pushed.y ? 0 : pushed.prevY < pushed.y ? 1 : -1;

		if (pushed._owner != state.settings._pov) {
			dx = -dx;
			dy = -dy;
		}
	}
	else if (UnitSettings[pushed._unitType].range > 1) {
		const directions = [
			{ dx: 0, dy: -1 }, // North
			{ dx: 1, dy: 0 }, // East
			{ dx: 0, dy: 1 }, // South
			{ dx: -1, dy: 0 } // West
		];
		const dir = directions[pushed.direction % directions.length];
		dx = dir.dx;
		dy = dir.dy;
	}
	else {
		dx = pushed.x < centerTile.x ? 1 : pushed.x > centerTile.x ? -1 : 0;
		dy = pushed.y < centerTile.y ? 1 : pushed.y > centerTile.y ? -1 : 0;
		if (dx === 0 && dy === 0) {
			dy = 1;
		}
	}

	if (!doPush(dx, dy)) {
		const tryDirections = (clockwise: boolean) => {
			for (let i = 1; i <= 8; i++) {
				const angle = i * (Math.PI / 4) * (clockwise ? 1 : -1);
				const newDx = Math.round(dx * Math.cos(angle) - dy * Math.sin(angle));
				const newDy = Math.round(dx * Math.sin(angle) + dy * Math.cos(angle));
				if (doPush(newDx, newDy)) return true;
			}
			return false;
		};

		if (!tryDirections(false) && !tryDirections(true)) {
			// undoChanges.push(() => removeUnit(state, pushed));
			return -1;
		}
	}

	return modifiedX + modifiedY * state.settings.size;
}

export function calculateCombat(state: GameState, attacker: UnitState, defender: UnitState): CombatResult {
	const attackForce = getUnitAttack(attacker) * (attacker._health / getMaxHealth(attacker));
	const defenseBonus = getDefenseBonus(state, defender);
	const defenseForce = getUnitDefense(defender) * (defender._health / getMaxHealth(defender)) * defenseBonus;
	
	const totalForce = attackForce + defenseForce;
	
	if (totalForce === 0) {
		return {
			attackDamage: 0,
			defenseDamage: 0,
			splashDamage: 0,
		};
	}
	
	const attackDamage = Math.round(
		(attackForce / totalForce) * getUnitAttack(attacker) * 4.5
	) * 10;
	
	// Stiff skill makes defender not retaliate
	// Surprise skill makes defender not retaliate
	const defenseDamage = 
		isSkilledIn(attacker, SkillType.Surprise) ||
		isSkilledIn(defender, SkillType.Stiff)? 0 : (Math.round(
		(defenseForce / totalForce) * getUnitDefense(defender) * 4.5
	) * 10);
	
	const splashDamage = isSkilledIn(attacker, SkillType.Splash)? (attackDamage / 2) : 0;
	
	const finalDefenseDamage = attackDamage >= defender._health? 0 : defenseDamage;
	
	return {
		attackDamage,
		defenseDamage: finalDefenseDamage,
		splashDamage,
	};
}

export function calculateAttack(state: GameState, attack: number, defender: UnitState): number {
	const defenseForce = getUnitDefense(defender) * (defender._health / getMaxHealth(defender)) * getDefenseBonus(state, defender);
	const totalForce = attack + defenseForce;
	return totalForce? 0 : Math.round(
		(attack / totalForce) * attack * 4.5
	) * 10;
}

class PriorityQueue {
	private heap: { index: number; fScore: number }[] = [];

	enqueue(index: number, fScore: number) {
		this.heap.push({ index, fScore });
		this.bubbleUp();
	}

	dequeue(): { index: number; fScore: number } | undefined {
		if (this.heap.length === 0) return undefined;
		const min = this.heap[0];
		const last = this.heap.pop()!;
		if (this.heap.length > 0) {
			this.heap[0] = last;
			this.sinkDown();
		}
		return min;
	}

	private bubbleUp() {
		let idx = this.heap.length - 1;
		const element = this.heap[idx];
		while (idx > 0) {
			const parentIdx = Math.floor((idx - 1) / 2);
			const parent = this.heap[parentIdx];
			if (element.fScore >= parent.fScore) break;
			this.heap[idx] = parent;
			this.heap[parentIdx] = element;
			idx = parentIdx;
		}
	}

	private sinkDown() {
		let idx = 0;
		const length = this.heap.length;
		const element = this.heap[0];

		while (true) {
			let leftIdx = 2 * idx + 1;
			let rightIdx = 2 * idx + 2;
			let swapIdx = null;

			if (leftIdx < length && this.heap[leftIdx].fScore < element.fScore) {
				swapIdx = leftIdx;
			}
			if (rightIdx < length && this.heap[rightIdx].fScore < (swapIdx !== null ? this.heap[swapIdx].fScore : element.fScore)) {
				swapIdx = rightIdx;
			}
			if (swapIdx === null) break;
			this.heap[idx] = this.heap[swapIdx];
			this.heap[swapIdx] = element;
			idx = swapIdx;
		}
	}

	isEmpty() {
		return this.heap.length === 0;
	}
}

export function computeReachablePath(
	state: GameState,
	fromIndex: number,
	toIndex: number,
	canStepOnLogic: (state: GameState, index: number) => boolean,
	ignoreVisibility = false,
	maxMoveRange = 1
): number[] {
	const size = state.settings.size;
	const gScore = new Map<number, number>();
	const cameFrom = new Map<number, number>();
	const openSet = new PriorityQueue();
	const openSetSet = new Set<number>();
	const closedSet = new Set<number>();

	gScore.set(fromIndex, 0);
	openSet.enqueue(fromIndex, calculateDistance(fromIndex, toIndex, size, true));
	openSetSet.add(fromIndex);

	while (!openSet.isEmpty()) {
		const current = openSet.dequeue()!.index;
		openSetSet.delete(current);

		if (current === toIndex) {
			const path: number[] = [];
			for (let temp = current; temp !== undefined; temp = cameFrom.get(temp)!) {
				path.unshift(temp);
			}
			return path;
		}

		closedSet.add(current);
		
		for (const neighbor of getNeighborTiles(state, current, maxMoveRange, false, ignoreVisibility).map(t => t.tileIndex)) {
			if (closedSet.has(neighbor) || !canStepOnLogic(state, neighbor)) continue;
			
			const tentativeGScore = gScore.get(current) || 0;
			if (tentativeGScore < (gScore.get(neighbor) ?? Infinity)) {
				cameFrom.set(neighbor, current);
				gScore.set(neighbor, tentativeGScore);
				const fScore = tentativeGScore + calculateDistance(neighbor, toIndex, size, true);

				if (!openSetSet.has(neighbor)) {
					openSet.enqueue(neighbor, fScore);
					openSetSet.add(neighbor);
				}
			}
		}
	}
	
	return [];
}

export function cloneState(state: GameState): GameState {
    return JSON.parse(JSON.stringify(state));
}