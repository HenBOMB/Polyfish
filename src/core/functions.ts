import { CityState, GameState, PlayerID, ResourceState, StructureState, TechnologyState, TileState, TribeState, UnitState } from "./states";
import { TechnologyType, ResourceType, RewardType, SkillType, TerrainType, StructureType, EffectType, UnitType, AbilityType, TribeType } from "./types";
import { CombatResult } from "./states";
import { ResourceSettings } from "./settings/ResourceSettings";
import { UnitSettings } from "./settings/UnitSettings";
import { TechnologyReplacements, TechnologySetting, TechnologySettings } from "./settings/TechnologySettings";
import Game from "../game";
import { calculateDistance } from "../ai/gmath";

type TribeLike = TribeState | TribeType;
type TechLike = TechnologyState | TechnologyType;

export function parseToTribeType(tribe: TribeLike): TribeType {
	return typeof(tribe) == 'number'? tribe as TribeType : tribe.type;
}

export function parseToTechType(techlike: TechLike): TechnologyType {
	const _type = typeof(techlike);
	if (_type == 'object') {
		return (techlike as TechnologyState).type;
	}
	if (_type == 'number' || _type == 'string') {
		return Number(techlike) as TechnologyType;
	}
	throw Error(`Unknown tech=${techlike}`);
}

export function getTechSettings(techlike: TechLike): TechnologySetting {
	return TechnologySettings[parseToTechType(techlike)];
}

export function getReplacedOrTechType(tribelike: TribeLike, techlike: TechLike): TechnologyType {
	const techType = parseToTechType(techlike);
	return TechnologyReplacements[parseToTribeType(tribelike)]?.find(x => x == techType) || techType;
}

export function getReplacedOrTechSettings(like: TribeLike, techlike: TechLike): TechnologySetting {
	const s = TechnologySettings[getReplacedOrTechType(like, techlike)];
	if (!s) {
		console.log(JSON.stringify(techlike));
		throw new Error(`Unknown tech=${techlike}`);
	}
	return s;
}

export function getNextTech(tech: TechLike): TechnologyType[] | null {
	return getTechSettings(tech).next || null;
}

export function getTechUnitType(tribe: TribeLike, tech: TechLike): UnitType | null {
	return getReplacedOrTechSettings(tribe, tech).unlocksUnit || null;
}

export function getTechUpgradeType(tribe: TribeLike, tech: TechLike): UnitType | null {
	const unit = getTechUnitType(tribe, tech);
	if (!unit) return null;
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

// Should not use cause we NEVER use the replaced as a base for the tech list
// export function getOriginalTech(tierTech: TechLike) {
// 	const settings = TechnologySettings[parseToTechType(tierTech)];
// 	return settings.tier? 
// 		settings : TechnologySettings[settings.replacesTech!];
// }

export function getTechCost(tribe: TribeState, tierTech: TechLike): number {
	const baseCost = 4 + getTechSettings(tierTech).tier! * tribe.cities.length;
	if (isTechUnlocked(tribe, TechnologyType.Philosophy)) {
		return Math.ceil(baseCost * 0.77);
	}
	return baseCost;
}

/** 
 * Returns the accurate city production depending on every factor
 * @param state 
 * @param city
 */
export function getCityProduction(state: GameState, ...cities: CityState[]): number {
	// If city is on riot or the tile is occupied by an enemy then production is nullified
	return cities.reduce((acc, city) => acc + 
		(city._riot || getEnemyAt(state, city.tileIndex)? 0 
		: (city.production + Object.values(city.rewards).filter(x => x == RewardType.Park || x == RewardType.Workshop).length)
	), 0);
}

/**
 * Returns the tribe's SPT (stars per turn)
 * @param state 
 * @param tribe
 * @returns 
 */
export function getTribeSPT(state: GameState, tribe?: TribeState): number {
	if (!tribe) tribe = getPovTribe(state);
	return tribe.cities.reduce((a, b) => a + getCityProduction(state, b), 0);
}

export function getPovTerritorry(state: GameState, tribe?: TribeState, city?: CityState): number[] {
	tribe = tribe || getPovTribe(state);
	return city? city._territory : tribe.cities.map(x => x._territory).flat();
}

const neighborCache = new Map<number, number[]>();

export function getAdjacentIndexes(state: GameState, idx: number, range = 1, unowned = false, includeUnexplored = false): number[] {
	const width = state.settings.size;
	const neighbors: number[] = [];
	const tile = state.tiles[idx];
	
	if (!unowned && !includeUnexplored) {
		const neighs = neighborCache.get(idx);
		if (neighs) {
			return neighs;
		}
	}

	for (let dx = -range; dx <= range; dx++) {
		for (let dy = -range; dy <= range; dy++) {
			if (dx === 0 && dy === 0) continue;
			
			const neighborX = tile.coords.x + dx;
			const neighborY = tile.coords.y + dy;
			
			if (neighborX < 0 || neighborX >= width || neighborY < 0 || neighborY >= width) continue;
			
			const neighborIndex = neighborY * width + neighborX;
			
			if (!includeUnexplored || unowned) {
				// const explored = state._visibleTiles[neighborIndex];
				const explored = state.tiles[neighborIndex].explorers;
	
				// Skip unexplored
				if (!includeUnexplored && !explored.size) continue;
				
				// Optionally filter for owned tiles.
				if (unowned && (explored.size? state.tiles[neighborIndex].owner > 0 : false)) continue;
			}
			
			neighbors.push(neighborIndex);
		}
	}

	if (!unowned && !includeUnexplored) {
		neighborCache.set(idx, neighbors);
	}
	
	return neighbors;
}

export function getAdjacentTiles(state: GameState, index: number, range = 1, unowned = false, includeUnexplored = false): TileState[] {
	return getAdjacentIndexes(state, index, range, unowned, includeUnexplored).map(i => state.tiles[i]);
}

export function getResourceAt(state: GameState, tileOrState: TileState | number): ResourceType | null {
	return state.resources[typeof tileOrState === 'number'? tileOrState : tileOrState.coords.idx]?.type || null;
}

export function getStructureAt(state: GameState, tileOrState: TileState | number): StructureType | null {
	return state.structures[typeof tileOrState === 'number'? tileOrState : tileOrState.coords.idx]?.type || null;
}

export function getTrueUnitAt(state: GameState, tileOrState: TileState | number, matchOwner?: number): UnitState | null {
	const tile = state.tiles[typeof tileOrState === 'number'? tileOrState : tileOrState.coords.idx];
	if (!tile._unitOwnerID) return null;
	const found = state.tribes[tile._unitOwnerID].units.find(x => x.coords.idx === tile.coords.idx) || null;
	return found && matchOwner? found.owner === matchOwner? found : null : found;
}

export function getUnitAt(state: GameState, tileOrState: TileState | number, matchOwner?: number): UnitState | null {
	if (!isTileExplored(state, tileOrState, matchOwner)) {
		return null;
	}

	const found = getTrueUnitAt(state, tileOrState, matchOwner);

	if (!found) {
		return null;
	}

	// If enemy unit is hidden, then we cant see it!
	if (hasEffect(found, EffectType.Invisible) && found.owner !== state.settings.currentPlayerTurnId) {
		return null;
	}

	return matchOwner && found? found.owner === matchOwner? found : null : found;
}

export function getCityAt(state: GameState, idx: number, matchOwner?: number): CityState | null {
	if (!isTileExplored(state, idx, matchOwner)) {
		console.log(state.tribes[state.settings.currentPlayerTurnId].username);
		console.log(state.tiles[idx], idx);
		console.log('impossibe!!!');
		return null;
	}
	const cityOwner = state.tiles[idx].owner;
	if (cityOwner < 1) {
		return null;
	}
	return state.tribes[cityOwner].cities.find(x => x.tileIndex == idx) || null;
}

export function getCityOwningTile(state: GameState, idx: number, playerCities?: CityState[]): CityState | null {
	const cityTileIndex = state.tiles[idx].rulingCityCoords?.idx;
	if (!cityTileIndex) {
		return null;
	}
	return state.tribes[state.settings.currentPlayerTurnId].cities.filter(x => playerCities? playerCities.includes(x) : true).find(x => x.tileIndex == cityTileIndex) || null;
}


export function getEnemyAt(state: GameState, idx: TileState | number, notMatchOwner?: number): UnitState | null {
	const found = getUnitAt(state, idx);
	if (!found) {
		return null;
	}
	return found.owner != (notMatchOwner || state.settings.currentPlayerTurnId)? found : null;
}

export function getTrueEnemyAt(state: GameState, idx: TileState | number, notMatchOwner?: number): UnitState | null {
	const found = getTrueUnitAt(state, idx);
	if (!found) {
		return null;
	}
	return found.owner != (notMatchOwner || state.settings.currentPlayerTurnId)? found : null;
}

export function getAlliesNearTile(state: GameState, idx: number, range = 1): UnitState[] {
	return getAdjacentIndexes(state, idx, range)
		.reduce((acc: UnitState[], cur: number) => {
			const ally = getTrueUnitAt(state, cur, state.settings.currentPlayerTurnId);
			return [
				...acc,
				...ally? [ally] : [],
			];
		}, []);
}

export function getEnemiesNearTile(state: GameState, idx: number, range = 1, strict = false): UnitState[] {
	return getAdjacentIndexes(state, idx, range)
		.reduce((acc: UnitState[], cur: number) => {
			const owner = state.tiles[cur]._unitOwnerID;
			if (!owner || owner === state.settings.currentPlayerTurnId) return acc;
			const enemy = (strict? getTrueEnemyAt  : getEnemyAt)(state, cur);
			// cant cheat
			if (!enemy) return acc;
			return [...acc, enemy];
		}, []);
}

export function getEnemiesInRange(state: GameState, unit: UnitState) {
	return getEnemiesNearTile(state, unit.coords.idx, getUnitRange(unit));
}

export function getEnemyIndexesInRange(state: GameState, unit: UnitState) {
	return getAdjacentIndexes(state, unit.coords.idx, getUnitRange(unit))
		.filter(x => isTileOccupied(state, x, true));
}

export function getClosestEnemyCity(state: GameState, tileIndex: number, maxRange = 1): [CityState, number] | null {
	let closestDistance = maxRange;
	let closestCity = null;
	for (let j = 0; j < state.settings._maxTribeCount; j++) {
		const i = j + 1;
		if (i === state.settings.currentPlayerTurnId) continue;
		for (const city of state.tribes[i].cities.filter(x => state._visibleTiles[x.tileIndex])) {
			const distance = calculateDistance(tileIndex, city.tileIndex, state.settings.size);
			if (distance < closestDistance || (distance === closestDistance && state.tiles[i].capitalOf)) {
				closestDistance = distance;
				closestCity = city;
			}
		}
	}
	return closestCity? [closestCity, closestDistance] : null;
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
	if (!resType) return false;
	const settings = ResourceSettings[resType];
	if (settings.visibleRequired) {
		return isTechUnlocked(tribe, settings.techRequired, true);
	}
	return true;
}

/**
 * Checks in tile.explorers
 */
export function isTileExplored(state: GameState, idxOrState: number | TileState, matchOwner?: PlayerID): boolean {
	return !state.settings._fow || state.tiles[typeof idxOrState === 'number'? idxOrState : idxOrState.coords.idx].explorers.has(matchOwner || state.settings.currentPlayerTurnId);
}

export function isTileOccupied(state: GameState, idx: number, strictEnemy = false): boolean {
	return Boolean(state.tiles[idx]._unitOwnerID && (strictEnemy? state.tiles[idx]._unitOwnerID != state.settings.currentPlayerTurnId : true));
}

export function isTileFrozen(state: GameState, idx: number): boolean {
	// TODO should use internal 'frozen' boolean
	return state.tiles[idx].type === TerrainType.Ice;
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
	if (techType == TechnologyType.BeyondComprehension) return false;
	if (techType == TechnologyType.Unrequired) return true;
	const tierTech = getTechSettings(techType).replacesTech || techType;
	return tribe.tech_vanilla.some(x => x.type == tierTech && (strict? x.discovered : true));
}

export function isNavigationable(tribe: TribeState, unit: UnitState, tile: TileState): boolean {
	if (isSkilledIn(unit, SkillType.Fly, SkillType.Navigate)) {
		return true;
	}
	switch (tile.type) {
		case TerrainType.Water:
			return tribe.tech_vanilla.some(x => getReplacedOrTechSettings(tribe, x).unlocksTerrain === TerrainType.Water);
		case TerrainType.Ocean:
			return tribe.tech_vanilla.some(x => getReplacedOrTechSettings(tribe, x).unlocksTerrain === TerrainType.Ocean);
		case TerrainType.Mountain:
			return tribe.tech_vanilla.some(x => getReplacedOrTechSettings(tribe, x).unlocksTerrain === TerrainType.Mountain);
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
	return tile.type === TerrainType.Water || tile.type === TerrainType.Ocean;
}

export function isIceTerrain(tile: TileState): boolean {
	// TODO THIS SHOULD BE: tile.frozen?
	return tile.type === TerrainType.Ice;
}

export function isSkilledIn(unit: UnitState | UnitType, ...skills: SkillType[]): boolean {
	const settings = UnitSettings[typeof unit === "number"? unit : unit.type];
	const passengerSettings = typeof unit != "number" && unit.passengerType? UnitSettings[unit.passengerType].skills : new Set();
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
	return unit.effects.has(effect);
}

export function isInTerritory(state: GameState, unit: UnitState) {
	return state.tiles[unit.coords.idx].owner == unit.owner;
}

export function isUnderSiege(state: GameState, city: CityState | number): boolean {
	const tile = state.tiles[typeof city === 'number'? city : city.tileIndex];
	const enemy = getEnemyAt(state, tile.coords.idx, tile.owner);
	return Boolean(enemy);
}

export function isCity(state: GameState, idx: number): boolean {
	return state.tiles[idx].rulingCityCoords != null && state.tiles[idx].rulingCityCoords.idx == idx;
}

export function isEnemyCity(state: GameState, idx: number): boolean {
	const tile = state.tiles[idx];
	if (!isCity(state, idx)) return false;
	return tile.owner != state.settings.currentPlayerTurnId;
}

export function isRoadpathAndUsable(state: GameState, unit: UnitState, idx: number) {
	// TODO Friendly ally system
	const tile = state.tiles[idx];
	// It must be owned by this unit or is neutral (ally)
	// And the tile must have a road or a city
	return (tile.owner == unit.owner || tile.owner < 1) && (tile.hasRoad || isCity(state, idx));	
}

export function getDefenseBonus(state: GameState, unit: UnitState): number {
	// Poisoned units cannot recieve defense bonus
	if (hasEffect(unit, EffectType.Poison)) {
		return 1;
	}
	
	const tribe = state.tribes[unit.owner];

	switch (state.tiles[unit.coords.idx].type) {
		case TerrainType.Water:
		case TerrainType.Ocean:
			if (isTechUnlocked(tribe, TechnologyType.Aquatism)) {
				return 1.5;
			}
			break;
		case TerrainType.Forest:
			if (isTechUnlocked(tribe, TechnologyType.Archery)) {
				return 1.5;
			}
			break;
		case TerrainType.Mountain:
			if (isTechUnlocked(tribe, TechnologyType.Climbing)) {
				return 1.5;
			}
		break;
		default:
			const ownCity = state.tribes[unit.owner].cities.find(x => x.tileIndex == unit.coords.idx);
			//  City defense
			if (ownCity && isSkilledIn(unit, SkillType.Fortify)) {
				return ownCity.rewards.has(RewardType.CityWall)? 4 : 1.5;
			}
			break;
	}
	
	return 1;
}

export function isAdjacentToEnemy(state: GameState, tile: TileState, matchUnitType?: UnitType, checkForControl=true): boolean {
	// Get true enemy because invisible units (cloaks) can also control terrain
	return getAdjacentIndexes(state, tile.coords.idx).some(x => {
		const e = checkForControl? getTrueEnemyAt(state, x) : getEnemyAt(state, x);
		return e && (!matchUnitType || e.type === matchUnitType);
	});
}

// TODO THIS IS AMBIGUOUS, ONLY WORKS WITH 1v1
export function isGameOver(state: GameState): boolean {
	return state.settings._gameOver 
		|| state.settings.turn > state.settings.maxTurns 
		|| isGameLost(state) 
		|| isGameWon(state);
}

export function isGameLost(state: GameState): boolean {
	const tribe = getPovTribe(state);
	return tribe.resignedTurn > 0 || tribe.killedTurn > 0;
}

export function isGameWon(state: GameState): boolean {
	for (let owner = 1; owner <= state.settings._maxTribeCount; owner++) {
		if (state.tribes[owner].resignedTurn > 0 || state.tribes[owner].killedTurn > 0) {
			if (owner === state.settings.currentPlayerTurnId) {
				return false;
			}
			continue;
		}
		else if (owner === state.settings.currentPlayerTurnId) {
			continue;
		}

		return false;
	}
	return true;
}

export function getWipeouts(state: GameState, owner?: number): TribeState[] {
	owner = owner || state.settings.currentPlayerTurnId;
	return Object.values(state.tribes).filter(x => x.killerId === owner);
}

export function isSteppable(state: GameState, unit: UnitState, tileOrIndex: TileState | number) {
	const tile = typeof tileOrIndex === "number"? state.tiles[tileOrIndex] : tileOrIndex;
	
	// Unexplored
	// Occupied
	if (!state._visibleTiles[tile.coords.idx]
		|| getUnitAt(state, tile.coords.idx)
	) {
		return false;
	}

	// Fly
	if (isSkilledIn(unit, SkillType.Fly)) {
		return true;
	}

	const tribe = state.tribes[unit.owner];
	
	// Checks for: Water, Ocean, Mountain, Fly & Navigation skills
	if (!isNavigationable(tribe, unit, tile)) {
		return false;
	}

	const isAquatic = isAquaticOrCanFly(unit, false);

	// Port and non aquatic units
	if (!isAquatic) {
		const isPort = getStructureAt(state, tile.coords.idx) === StructureType.Port;
		if (isPort) {
			return tile.owner === unit.owner;
		}
	}
	
	// If unit has Navigate, it cannot move onto land, except for capturing cities
	if (isSkilledIn(unit, SkillType.Navigate)) {
		if (!isWaterTerrain(tile) && getStructureAt(state, tile.coords.idx) !== StructureType.Village) {
			return false;
		}
	}

	return true;
}

export function isTribeSteppable(state: GameState, idx: number) {
	switch (state.tiles[idx].type) {
		case TerrainType.Water:	
			return isTechUnlocked(state.tribes[state.settings.currentPlayerTurnId], TechnologyType.Fishing);
			
		case TerrainType.Ocean:	
			return isTechUnlocked(state.tribes[state.settings.currentPlayerTurnId], TechnologyType.Sailing);

		case TerrainType.Mountain:	
			return isTechUnlocked(state.tribes[state.settings.currentPlayerTurnId], TechnologyType.Climbing);

		default:
			return true;
	}
}

export function getPovTribe(stateOrState: GameState | Game): TribeState {
	const state = stateOrState instanceof Game? stateOrState.state : stateOrState;
	const tribe = state.tribes[state.settings.currentPlayerTurnId];
	if(!tribe) {
		console.log(`${state.settings.currentPlayerTurnId}, ${Object.keys(state.tribes)}`);
		throw Error("yo tf");
	}
	return tribe;
}

export function getCapitalCity(state: GameState, owner?: number): CityState | null {
	const pov = owner || state.settings.currentPlayerTurnId;
	return state.tribes[pov].cities.find(x => state.tiles[x.tileIndex].capitalOf === pov) || null;
}

export function calculateTribeScore(state: GameState, owner?: number) {
	const pov = owner? state.tribes[owner] : getPovTribe(state);

	let score = 0;

	score += pov.cities.reduce((total, city) => {
		return total + 20 * city._territory.length;
	}, 0);

	for (const unit of pov.units) {
		score += 5 * UnitSettings[unit.type].cost;
	}

}

export function calculateInitialTribeScore(state: GameState, owner?: number): number {
	const pov = owner? state.tribes[owner] : getPovTribe(state);

	// ! https://docs.google.com/document/d/1HYiUbT-3RtP4b2SwlMQEZB4bTdAUtN_6K8DOvY6wNsk/edit?tab=t.0

	let score = 0;

	// 100 xp per level, 20 xp per owned territory, 5 xp per population
	for (const city of pov.cities) {
		score += city.level * 100 
			+ city._territory.length * 20
			+ city.population * 5;

		// Not sure if this is correct
		// 40 for the city itself, 5 for each reward after the first (border growth is not counted)
		// Clamping to a max level of 6 to avoid negative values
		score += city.rewards.size > 1? 40 + Math.max((city.rewards.size - 1), 6) * 5 : 0;

		if (city.rewards.has(RewardType.Park)) {
			score += 300;
		}
	}

	// 5 xp per revealed tile
	score += Object.values(state.tiles).filter(x => x.explorers.has(pov.id)).length * 5;

	// 5 xp per star of cost
	for (const unit of pov.units) {
		score += 5 * UnitSettings[unit.type].cost;
	}
	
	// 5 100 per tech tier
	for (const tech of pov.tech_vanilla) {
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
	return unit.passengerType || unit.type;
}

/**
 * Returns the unit's settings, including naval types
 * @param unit 
 * @returns 
 */
export function getUnitSettings(unit: UnitState | UnitType) {
	return UnitSettings[typeof unit === "number"? unit : unit.type];
}

export function getMaxHealth(unit: UnitState) {
	let hp = getRealUnitSettings(unit).health;
	if (!hp) throw Error(`Yo no health bro tf "${unit.type}, ${unit.passengerType}, ${unit.coords.idx}"`);
	if (unit.veteran) hp += 5;
	return hp * 10;
}

export function getUnitAttack(unit: UnitState) {
	let atk = getRealUnitSettings(unit).attack;
	if (hasEffect(unit, EffectType.Boost)) {
		atk += 0.5;
	}
	return atk;
}

export function getUnitMovement(unit: UnitState) {
	let movement = getUnitSettings(unit).movement;
	if (hasEffect(unit, EffectType.Boost)) {
		movement += 1;
	}
	return movement;
}

export function getUnitDefense(unit: UnitState) {
	let def = getRealUnitSettings(unit).defense;
	// 30% damage reduction if poisoned
	if (hasEffect(unit, EffectType.Poison)) {
		def *= 0.7;
	}
	return getRealUnitSettings(unit).defense;
}

export function getUnitRange(unit: UnitState) {
	const range = getUnitSettings(unit).range;
	if (!range) throw Error(`Yo no range bro tf "${unit.type}"`);
	return getUnitSettings(unit).range;
}

export function getStarExchange(state: GameState, owner: TribeState | number) {
	const score = state.tribes[typeof owner === "number"? owner : owner.id].score;
	if (score < 1000) return 3;
	if (score < 2000) return 6;
	if (score < 3000) return 9;
	return 12;
}

export function getHomeCity(state: GameState, unit: UnitState): CityState | null {
	return isSkilledIn(unit, SkillType.Independent) || !unit.homeCoords || unit.homeCoords.idx < 0? null : getPovTribe(state).cities.find(x => x.tileIndex == unit.homeCoords!.idx) || null;
}

export function getRulingCity(state: GameState, idx: number): CityState | null {
	if (!state.tiles[idx].rulingCityCoords) return null;
	const cityTile = state.tiles[state.tiles[idx].rulingCityCoords.idx];
	const city = state.tribes[cityTile.owner]!.cities.find(x => x.tileIndex === cityTile.coords.idx)!;
	return city!;
}


export function getTerrainType(state: GameState, idx: number): TerrainType {
	return state.tiles[idx].type;
}

/**
 * Pushes a unit away from its current tile.
 * @param state
 * @param pushed
 * @returns - The index of the tile where it ended up. -1 if it failed to move the unit.
 */
export function calaulatePushablePosition(state: GameState, pushed: UnitState): number {
	const initialX = pushed.coords.x;
	const initialY = pushed.coords.y;
	
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

	// TODO verify

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
	if (initialX !== pushed.prevCoords.x || initialY !== pushed.prevCoords.y) {
		dx = modifiedX === initialX ? 0 : modifiedX < initialX ? 1 : -1;
		dy = pushed.prevCoords.y === initialY ? 0 : pushed.prevCoords.y < initialY ? 1 : -1;

		if (pushed.owner != state.settings.currentPlayerTurnId) {
			dx = -dx;
			dy = -dy;
		}
	}
	else if (UnitSettings[pushed.type].range > 1) {
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
		dx = initialX < centerTile.coords.x ? 1 : initialX > centerTile.coords.x ? -1 : 0;
		dy = initialY < centerTile.coords.y ? 1 : initialY > centerTile.coords.y ? -1 : 0;
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
			return -1;
		}
	}

	return modifiedX + modifiedY * state.settings.size;
}

export function calculateCombat(state: GameState, attacker: UnitState, defender: UnitState): CombatResult {
	const attackForce = getUnitAttack(attacker) * (attacker.health / getMaxHealth(attacker));
	const defenseBonus = getDefenseBonus(state, defender);
	const defenseForce = getUnitDefense(defender) * (defender.health / getMaxHealth(defender)) * defenseBonus;
	
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
		isSkilledIn(attacker, SkillType.Freeze)? 0 :
		isSkilledIn(defender, SkillType.Stiff)? 0 : (Math.round(
		(defenseForce / totalForce) * getUnitDefense(defender) * 4.5
	) * 10);
	
	const splashDamage = isSkilledIn(attacker, SkillType.Splash)? (attackDamage / 2) : 0;
	
	const finalDefenseDamage = attackDamage >= defender.health? 0 : defenseDamage;
	
	return {
		attackDamage,
		defenseDamage: finalDefenseDamage,
		splashDamage,
	};
}

export function calculateAttack(state: GameState, attack: number, defender: UnitState): number {
	const defenseForce = getUnitDefense(defender) * (defender.health / getMaxHealth(defender)) * getDefenseBonus(state, defender);
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

export function getCityUnitCount(state: GameState, city: CityState): number {
	return Object.values(state.tribes).reduce((acc, tribe) => acc + tribe.units.filter(x => x.homeCoords && x.homeCoords.idx == city.tileIndex).length, 0);
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
		
		for (const neighbor of getAdjacentTiles(state, current, maxMoveRange, false, ignoreVisibility).map(t => t.coords.idx)) {
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

// Doesnt work
// export function cloneState(state: GameState): GameState {
// 	const clone = {
// 		...state,
// 		tribes: Object.values(state.tribes).reduce((a, b, i) => ({ ...a, [i+1]: { ...b, hash: String(b.hash) + 'b' } }), {}),
// 	}
//     return JSON.parse(JSON.stringify(clone));
// }


export function _setVisibleTiles(state: GameState, ownerId: PlayerID) {
	state._visibleTiles = { };
    for (let i = 0; i < state.settings.tileCount; i++) {
        state._visibleTiles[i] = state.tiles[i].explorers.has(ownerId);
    }
}