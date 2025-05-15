import { CityState, GameState, ResourceState, StructureState, TechnologyState, TileState, TribeState, UnitState } from "./states";
import { TechnologyType, ResourceType, RewardType, SkillType, TerrainType, StructureType, EffectType, UnitType, AbilityType, TribeType } from "./types";
import { CombatResult } from "./states";
import { ResourceSettings } from "./settings/ResourceSettings";
import { UnitSettings } from "./settings/UnitSettings";
import { TechnologyReplacements, TechnologySetting, TechnologySettings } from "./settings/TechnologySettings";
import Game from "../game";

export function indexToCoord(state: GameState, tileIndex: number) {
	return [state.tiles[tileIndex].x, state.tiles[tileIndex].y];
}

type TribeLike = TribeState | TribeType;
type TechLike = TechnologyState | TechnologyType;

export function parseToTribeType(tribe: TribeLike): TribeType {
	return typeof(tribe) == 'number'? tribe as TribeType : tribe.tribeType;
}

export function parseToTechType(tech: TechLike): TechnologyType {
	return typeof(tech) == 'number' || typeof(tech) == 'string'? Number(tech) as TechnologyType : tech.techType;
}

export function getTechSettings(tech: TechLike): TechnologySetting {
	return TechnologySettings[parseToTechType(tech)];
}

export function getReplacedOrTechType(like: TribeLike, tech: TechLike): TechnologyType {
	const techType = parseToTechType(tech);
	return TechnologyReplacements[parseToTribeType(like)]?.find(x => x == techType) || techType;
}

export function getReplacedOrTechSettings(like: TribeLike, tech: TechLike): TechnologySetting {
	return TechnologySettings[getReplacedOrTechType(like, tech)];
}

export function getNextTech(tech: TechLike): TechnologyType[] | null {
	return getTechSettings(tech).next || null;
}

export function getTechUnitType(tribe: TribeLike, tech: TechLike): UnitType | null {
	return getReplacedOrTechSettings(tribe, tech).unlocksUnit || null;
}

export function getTechUpgradeType(tribe: TribeLike, tech: TechLike): UnitType | null {
	const unit = getTechUnitType(tribe, tech);
	if(!unit) return null;
	return UnitSettings[unit].upgradeFrom? unit : null;
}

export function getTechResource(tribe: TribeLike, tech: TechLike): ResourceType | null {
	return getReplacedOrTechSettings(tribe, tech).unlocksResource || null;
}

export function getTechStructure(tribe: TribeLike, tech: TechLike): StructureType | null {
	return getReplacedOrTechSettings(tribe, tech).unlocksStructure || null;
}

export function getTechAbility(like: TribeLike, tierTech: number): AbilityType | null {
	return getReplacedOrTechSettings(like, tierTech).unlocksAbility || null;
}

export function getTechCost(like: TribeState, tierTech: TechLike): number {
	return getTechSettings(tierTech).tier! && (isTechUnlocked(like, TechnologyType.Philosophy)? 0.77 : 1);
}

/** Returns the correct city production */
export function getCityProduction(state: GameState, ...city: CityState[]): number {
	// If riot or tile is occupied by an enemy unit = 0
	return city.reduce((a, b) => a + 
		(b._riot || getEnemyAt(state, b.tileIndex)? 0 : b._production
	), 0);
}

export function getTribeProduction(state: GameState, tribe: TribeState): number {
	return tribe._cities.reduce((a, b) => a + getCityProduction(state, b), 0);
}

export function getPovTerritorry(state: GameState, tribe?: TribeState, cityTarget?: CityState): number[] {
	tribe = tribe || getPovTribe(state);
	return (cityTarget? [cityTarget] : tribe._cities).map(x => x._territory).flat();
}

const neighborCache = new Map<number, number[]>();

export function getNeighborIndexes(state: GameState, index: number, range = 1, unowned = false, includeUnexplored = false): number[] {
	const width = state.settings.size;
	const neighbors: number[] = [];
	const [x, y] = indexToCoord(state, index);
	
	if(!unowned && !includeUnexplored) {
		const neighs = neighborCache.get(index);
		if(neighs) {
			return neighs;
		}
	}

	for (let dx = -range; dx <= range; dx++) {
		for (let dy = -range; dy <= range; dy++) {
			if (dx === 0 && dy === 0) continue;
			
			const neighborX = x + dx;
			const neighborY = y + dy;
			
			if (neighborX < 0 || neighborX >= width || neighborY < 0 || neighborY >= width) continue;
			
			const neighborIndex = neighborY * width + neighborX;
			
			if(!includeUnexplored || unowned) {
				const explored = state._visibleTiles[neighborIndex];
	
				// Skip unexplored
				if(!includeUnexplored && !explored) continue;
				
				// Optionally filter for owned tiles.
				if (unowned && (explored? state.tiles[neighborIndex]._owner > 0 : false)) continue;
			}
			
			neighbors.push(neighborIndex);
		}
	}

	if(!unowned && !includeUnexplored) {
		neighborCache.set(index, neighbors);
	}
	
	return neighbors;
}

export function getNeighborTiles(state: GameState, index: number, range = 1, unowned = false, includeUnexplored = false): TileState[] {
	return getNeighborIndexes(state, index, range, unowned, includeUnexplored).map(i => state.tiles[i]);
}

export function getResourceAt(state: GameState, tileIndex: TileState | number): ResourceType | null {
	return state.resources[typeof tileIndex === 'number'? tileIndex : tileIndex.tileIndex]?.id || null;
}

export function getStructureAt(state: GameState, tileIndex: TileState | number): StructureType | null {
	return state.structures[typeof tileIndex === 'number'? tileIndex : tileIndex.tileIndex]?.id || null;
}

export function getTrueUnitAt(state: GameState, tileIndex: TileState | number, matchOwner?: number): UnitState | null {
	const tile = state.tiles[typeof tileIndex === 'number'? tileIndex : tileIndex.tileIndex];
	if(!tile._unitOwner) return null;
	const found = state.tribes[tile._unitOwner]._units.find(x => x._tileIndex === tile.tileIndex) || null;
	return found && matchOwner? found._owner === matchOwner? found : null : found;
}

export function getUnitAt(state: GameState, tileIndex: TileState | number, matchOwner?: number): UnitState | null {
	if(!isTileExplored(state, tileIndex, matchOwner)) {
		return null;
	}

	const found = getTrueUnitAt(state, tileIndex, matchOwner);

	if(!found) return null;

	// If enemy unit is hidden, then we cant see it!
	if(hasEffect(found, EffectType.Invisible) && found._owner !== state.settings._pov) {
		return null;
	}

	return matchOwner && found? found._owner === matchOwner? found : null : found;
}

export function getCityAt(state: GameState, tileIndex: number, matchOwner?: number): CityState | null {
	if(!isTileExplored(state, tileIndex, matchOwner)) {
		return null;
	}
	const cityOwner = state.tiles[tileIndex]._owner;
	if(cityOwner < 1) {
		return null;
	}
	return state.tribes[cityOwner]._cities.find(x => x.tileIndex == tileIndex) || null;
}

export function getCityOwningTile(state: GameState, tileIndex: number, playerCities?: CityState[]): CityState | null {
	const cityTileIndex = state.tiles[tileIndex]._rulingCityIndex;
	if(cityTileIndex < 0) {
		return null;
	}
	return state.tribes[state.settings._pov]._cities.filter(x => playerCities? playerCities.includes(x) : true).find(x => x.tileIndex == cityTileIndex) || null;
}


export function getEnemyAt(state: GameState, tileIndex: TileState | number, notMatchOwner?: number): UnitState | null {
	const found = getUnitAt(state, tileIndex);
	if(!found) return null;
	return found._owner != (notMatchOwner || state.settings._pov)? found : null;
}

export function getTrueEnemyAt(state: GameState, tileIndex: TileState | number, notMatchOwner?: number): UnitState | null {
	const found = getTrueUnitAt(state, tileIndex);
	if(!found) return null;
	return found._owner != (notMatchOwner || state.settings._pov)? found : null;
}

export function getAlliesNearTile(state: GameState, tileIndex: number, range = 1): UnitState[] {
	return getNeighborIndexes(state, tileIndex, range)
		.reduce((acc: UnitState[], cur: number) => {
			const ally = getTrueUnitAt(state, cur, state.settings._pov);
			return [
				...acc,
				...ally? [ally] : [],
			];
		}, []);
}

export function getEnemiesNearTile(state: GameState, tileIndex: number, range = 1, strict = false): UnitState[] {
	return getNeighborIndexes(state, tileIndex, range)
		.reduce((acc: UnitState[], cur: number) => {
			const owner = state.tiles[cur]._unitOwner;
			if(owner < 1 || owner === state.settings._pov) return acc;
			const enemy = (strict? getTrueEnemyAt  : getEnemyAt)(state, cur);
			if(!enemy) return acc;
			return [...acc, enemy];
		}, []);
}

export function getEnemiesInRange(state: GameState, unit: UnitState) {
	return getEnemiesNearTile(state, unit._tileIndex, getUnitRange(unit));
}

export function getEnemyIndexesInRange(state: GameState, unit: UnitState) {
	return getNeighborIndexes(state, unit._tileIndex, getUnitRange(unit))
		.filter(x => isTileOccupied(state, x, true));
}

export function getClosestEnemyCity(state: GameState, tileIndex: number, range = 1): [null | CityState, number] {
	let closestDistance = range;
	let closestCity = null;
	for (let i = 1; i < state.settings.tribeCount; i++) {
		if(i === state.settings._pov) continue;
		for (const city of state.tribes[i]._cities.filter(x => state._visibleTiles[x.tileIndex])) {
			const distance = calculateDistance(tileIndex, city.tileIndex, state.settings.size);
			if (distance < closestDistance || (distance === closestDistance && state.tiles[i].capitalOf)) {
				closestDistance = distance;
				closestCity = city;
			}
		}
	}
	return [closestCity, closestDistance];
}

/**
 * Returns all tile indexes of lighthouses
 * @param state 
 * @param explored if passed as true or false, will return only the lighthouses that are explored or not
 * @returns 
 */
export function getLighthouses(state: GameState, explored?: boolean) {
	const lighhouses = [
		0,
		state.settings.size - 1,
		state.settings.size * state.settings.size - 1,
		1 + state.settings.size * state.settings.size - state.settings.size
	];
	return explored !== undefined? lighhouses.filter(x => explored === state._visibleTiles[x]) : lighhouses;
}

export function isLighthouse(state: GameState, tileIndex: number) {
	return getLighthouses(state).includes(tileIndex);
}

/**
* Uses initial state tech to verify if the tribe can see the resource
* @param tribe 
* @param resType 
* @returns 
*/
export function isResourceVisible(tribe: TribeState, resType?: ResourceType): boolean {
	if(!resType) return false;
	const settings = ResourceSettings[resType];
	if(settings.visibleRequired) {
		return isTechUnlocked(tribe, settings.techRequired, true);
	}
	return true;
}

/**
 * Checks in tile.explorers
 */
export function isTileExplored(state: GameState, tileIndex: TileState | number, matchOwner?: number): boolean {
	return state.tiles[typeof tileIndex === 'number'? tileIndex : tileIndex.tileIndex]._explorers.has(matchOwner || state.settings._pov);
}

export function isTileOccupied(state: GameState, tileIndex: number, strictEnemy = false): boolean {
	return state.tiles[tileIndex]._unitOwner > 0 && (strictEnemy? state.tiles[tileIndex]._unitOwner != state.settings._pov : true);
}

export function isTileFrozen(state: GameState, tileIndex: number): boolean {
	// TODO should use internal 'frozen' boolean
	return state.tiles[tileIndex].terrainType === TerrainType.Ice;
}


/**
 * Checks if the tribe has unlocked this tech
 * @param tribe
 * @param tech
 * @param strict Wether to check if the move is NOT simulated
 * @returns 
 */
export function isTechUnlocked(tribe: TribeState, tech: TechLike, strict = false): boolean {
	const techType = parseToTechType(tech);
	if(techType == TechnologyType.Unbuildable) return false;
	if(techType == TechnologyType.None) return true;
	const tierTech = getTechSettings(techType).replacesTech || techType;
	return tribe._tech.some(x => x.techType == tierTech && (strict? x.discovered : true));
}

export function isNavigationable(tribe: TribeState, unit: UnitState, tile: TileState): boolean {
	if(isSkilledIn(unit, SkillType.Fly, SkillType.Navigate)) {
		return true;
	}
	switch (tile.terrainType) {
		case TerrainType.Water:
			return tribe._tech.some(x => getReplacedOrTechSettings(tribe, x).unlocksTerrain === TerrainType.Water);
		case TerrainType.Ocean:
			return tribe._tech.some(x => getReplacedOrTechSettings(tribe, x).unlocksTerrain === TerrainType.Ocean);
		case TerrainType.Mountain:
			return tribe._tech.some(x => getReplacedOrTechSettings(tribe, x).unlocksTerrain === TerrainType.Mountain);
		default:
			return true;
	}
}

export function isTempleStructure(structType: StructureType) {
	return structType === StructureType.Temple 
		|| structType === StructureType.MountainTemple 
		|| structType === StructureType.ForestTemple 
		|| structType === StructureType.IceTemple 
		|| structType === StructureType.WaterTemple;
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
	const passengerSettings = typeof unit != "number" && unit._passenger? UnitSettings[unit._passenger].skills : new Set();
	return skills.some(x => settings.skills.has(x) || passengerSettings.has(x));
}

export function isAquaticOrCanFly(unit: UnitState | UnitType, canfly: boolean = true): boolean {
	return isSkilledIn(unit, 
		...(canfly? [SkillType.Fly] : []),
		SkillType.Carry,
		SkillType.Float,
		SkillType.Navigate,
	);
}

export function isNavalUnit(unit: UnitState | UnitType): boolean {
	return isSkilledIn(unit, SkillType.Carry, SkillType.Float, SkillType.Splash);
}

export function hasEffect(unit: UnitState, effect: EffectType): boolean {
	return unit._effects.has(effect);
}

export function isInTerritory(state: GameState, unit: UnitState) {
	return state.tiles[unit._tileIndex]._owner == unit._owner;
}

export function isUnderSiege(state: GameState, city: CityState | number): boolean {
	const tile = state.tiles[typeof city === 'number'? city : city.tileIndex];
	const enemy = getEnemyAt(state, tile.tileIndex, tile._owner);
	return Boolean(enemy);
}

export function isEnemyCity(state: GameState, tileIndex: number): boolean {
	const tile = state.tiles[tileIndex];
	if(tile._rulingCityIndex < 1) return false;
	return tile._owner != state.settings._pov;
}

export function isRoadpathAndUsable(state: GameState, unit: UnitState, tileIndex: number) {
	// TODO Friendly terrain
	const tile = state.tiles[tileIndex];
	// Is owned by this unit or is neutral, and has a road or is a city
	return (tile._owner == unit._owner || tile._owner < 1) && (tile.hasRoad || tile._rulingCityIndex == tile.tileIndex);	
}

export function getDefenseBonus(state: GameState, unit: UnitState): number {
	// Poisoned units cannot recieve defense bonus
	if (hasEffect(unit, EffectType.Poison)) {
		return 1;
	}
	
	const tribe = state.tribes[unit._owner];
	
	switch (state.tiles[unit._tileIndex].terrainType) {
		case TerrainType.Water:
		case TerrainType.Ocean:
			if(isTechUnlocked(tribe, TechnologyType.Aquatism)) {
				return 1.5;
			}
			break;
		case TerrainType.Forest:
			if(isTechUnlocked(tribe, TechnologyType.Archery)) {
				return 1.5;
			}
			break;
		case TerrainType.Mountain:
			if(isTechUnlocked(tribe, TechnologyType.Climbing)) {
				return 1.5;
			}
		break;
		default:
			const ownCity = state.tribes[unit._owner]._cities.find(x => x.tileIndex == unit._tileIndex);
			//  City defense
			if(ownCity && isSkilledIn(unit, SkillType.Fortify)) {
				return ownCity._rewards.has(RewardType.CityWall)? 4 : 1.5;
			}
			break;
	}
	
	return 1;
}

export function isAdjacentToEnemy(state: GameState, tile: TileState, matchUnitType?: UnitType): boolean {
	// Get true enemy because invisible units (cloaks) can also control terrain
	return getNeighborIndexes(state, tile.tileIndex).some(x => {
		const e = getTrueEnemyAt(state, x);
		return e && (!matchUnitType || e._unitType === matchUnitType);
	});
}

// TODO THIS IS AMBIGUOUS, ONLY WORKS WITH 1v1
export function isGameOver(state: GameState): boolean {
	return state.settings._gameOver 
		|| state.settings._turn > state.settings.maxTurns 
		|| isGameLost(state) 
		|| isGameWon(state);
}

export function isGameLost(state: GameState): boolean {
	const tribe = getPovTribe(state);
	return tribe._resignedTurn > 0 || tribe._killedTurn > 0;
}

export function isGameWon(state: GameState): boolean {
	for (let owner = 1; owner <= state.settings.tribeCount; owner++) {
		if(state.tribes[owner]._resignedTurn > 0 || state.tribes[owner]._killedTurn > 0) {
			if(owner === state.settings._pov) {
				return false;
			}
			continue;
		}
		else if(owner === state.settings._pov) {
			continue;
		}

		return false;
	}
	return true;
}

export function getWipeouts(state: GameState, owner?: number): TribeState[] {
	owner = owner || state.settings._pov;
	return Object.values(state.tribes).filter(x => x._killerId === owner);
}

export function isSteppable(state: GameState, unit: UnitState, tileOrIndex: TileState | number) {
	const tile = typeof tileOrIndex === "number"? state.tiles[tileOrIndex] : tileOrIndex;

	// Unexplored
	// Occupied
	if(!state._visibleTiles[tile.tileIndex]
		|| getUnitAt(state, tile.tileIndex)
	) {
		return false;
	}

	// Fly
	if(isSkilledIn(unit, SkillType.Fly)) {
		return true;
	}

	const tribe = state.tribes[unit._owner];
	
	// Checks for: Water, Ocean, Mountain, Fly & Navigation skills
	if(!isNavigationable(tribe, unit, tile)) {
		return false;
	}

	const isAquatic = isAquaticOrCanFly(unit, false);

	// Port and non aquatic units
	if(!isAquatic) {
		const isPort = getStructureAt(state, tile.tileIndex) === StructureType.Port;
		return isPort && tile._owner === unit._owner;
	}
	// Port
	
	// If unit has Navigate, it cannot move onto land, except for capturing cities
	if(isSkilledIn(unit, SkillType.Navigate)) {
		if(!isWaterTerrain(tile) && getStructureAt(state, tile.tileIndex) !== StructureType.Village) {
			return false;
		}
	}

	return true;
}

export function isTribeSteppable(state: GameState, tileIndex: number) {
	switch (state.tiles[tileIndex].terrainType) {
		case TerrainType.Water:	
			return isTechUnlocked(state.tribes[state.settings._pov], TechnologyType.Fishing);
			
		case TerrainType.Ocean:	
			return isTechUnlocked(state.tribes[state.settings._pov], TechnologyType.Sailing);

		case TerrainType.Mountain:	
			return isTechUnlocked(state.tribes[state.settings._pov], TechnologyType.Climbing);

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
	const pov = state.tribes[owner || state.settings._pov];

	// ! https://docs.google.com/document/d/1HYiUbT-3RtP4b2SwlMQEZB4bTdAUtN_6K8DOvY6wNsk/edit?tab=t.0

	let score = 0;

	// 100 xp per level, 20 xp per owned territory, 5 xp per population
	for(const city of pov._cities) {
		score += city._level * 100 
			+ city._territory.length * 20
			+ city._population * 5;

		// Not sure if this is correct
		// 40 for the city itself, 5 for each reward after the first (border growth is not counted)
		// Clamping to a max level of 6 to avoid negative values
		score += city._rewards.size > 1? 40 + Math.max((city._rewards.size - 1), 6) * 5 : 0;

		if(city._rewards.has(RewardType.Park)) {
			score += 300;
		}
	}

	// 5 xp per revealed tile
	score += state.tiles.filter(x => x._explorers.has(pov.owner)).length * 5;

	// 5 xp per star of cost
	for(const unit of pov._units) {
		score += 5 * UnitSettings[unit._unitType].cost;
	}
	
	// 5 100 per tech tier
	for(const tech of pov._tech) {
		score += 100 * getTechSettings(tech).tier!;
	}

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
	if(unit._veteran) hp += 5;
	return hp * 10;
}

export function getUnitAttack(unit: UnitState) {
	let atk = getRealUnitSettings(unit).attack;
	if(hasEffect(unit, EffectType.Boost)) {
		atk += 0.5;
	}
	return atk;
}

export function getUnitMovement(unit: UnitState) {
	let movement = getUnitSettings(unit).movement;
	if(hasEffect(unit, EffectType.Boost)) {
		movement += 1;
	}
	return movement;
}

export function getUnitDefense(unit: UnitState) {
	let def = getRealUnitSettings(unit).defense;
	// 30% damage reduction if poisoned
	if(hasEffect(unit, EffectType.Poison)) {
		def *= 0.7;
	}
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
	const [ initialX, initialY ] = indexToCoord(state, pushed._tileIndex);
	let modifiedX = initialX;
	let modifiedY = initialY;

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

	// TODO

	// Friendly units that previously moved will be pushed in the same direction of their movement
	// Enemy units are pushed the opposite direction

	// Ranged units get pushed in the direction of their last move or last attack

	// Units that were not previously moved will be pushed toward the center of the map

	// If the city where the units spawns is on the exact center of the map,
	// the unit will be pushed south
	
	// If the tile where the unit is supposed to go is occpied or impassable, 
	// it will try counterclockwise and then clockwise one tile at a time, 
	// until it finds a free tile, if none, the unit gets removed, without ganting a kill

	// If there is a direction the unit moved in
	if(initialX !== pushed.prevX || initialY !== pushed.prevY) {
		dx = modifiedX === initialX ? 0 : modifiedX < initialX ? 1 : -1;
		dy = pushed.prevY === initialY ? 0 : pushed.prevY < initialY ? 1 : -1;

		if (pushed._owner != state.settings._pov) {
			dx = -dx;
			dy = -dy;
		}
	}
	else if(UnitSettings[pushed._unitType].range > 1) {
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
		dx = initialX < centerTile.x ? 1 : initialX > centerTile.x ? -1 : 0;
		dy = initialY < centerTile.y ? 1 : initialY > centerTile.y ? -1 : 0;
		if (dx === 0 && dy === 0) {
			dy = 1;
		}
	}

	if(!doPush(dx, dy)) {
		const tryDirections = (clockwise: boolean) => {
			for (let i = 1; i <= 8; i++) {
				const angle = i * (Math.PI / 4) * (clockwise ? 1 : -1);
				const newDx = Math.round(dx * Math.cos(angle) - dy * Math.sin(angle));
				const newDy = Math.round(dx * Math.sin(angle) + dy * Math.cos(angle));
				if (doPush(newDx, newDy)) return true;
			}
			return false;
		};

		if(!tryDirections(false) && !tryDirections(true)) {
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
	
	if(totalForce === 0) {
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
		isSkilledIn(attacker, SkillType.Freeze)? 0 :
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
    return { ...state };
}