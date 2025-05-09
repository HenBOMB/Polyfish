import { CityState, GameState, TileState, UnitState } from "./states";
import { getNeighborTiles, getPovTerritorry, getNeighborIndexes, isAdjacentToEnemy, isAquaticOrCanFly, isSteppable, isWaterTerrain, getEnemiesInRange, isNavalUnit, getTechCost, getPovTribe, isSkilledIn, getCapitalCity, getRealUnitSettings, getUnitAttack, getUnitMovement, isRoadpathAndUsable, getTrueUnitAt, getUnitAt, getHomeCity, isInvisible, getMaxHealth, getAlliesNearTile, getRealUnitType, isTechUnlocked, isLighthouse, tryDiscoverRewardOtherTribes, isEnemyCity, isUnderSiege, getCityAt, getTechSettings, getTechUnitType, isTileOccupied, isBoosted, getEnemiesNearTile, isFrozen, isTileFrozen, getStarExchange, getTechStructure, getCityOwningTile, isResourceVisible } from './functions';
import { StructureSettings } from "./settings/StructureSettings";
import { SkillType, EffectType, ResourceType, RewardType, StructureType, TechnologyType, TerrainType, UnitType, AbilityType } from "./types";
import { addPopulationToCity, discoverTiles, freezeArea, splashDamageArea } from "./actions";
import { UnitSettings } from "./settings/UnitSettings";
import Move, { Action, CallbackResult, UndoCallback } from "./move";
import { MoveType } from "./types";
import { Logger } from "../polyfish/logger";
import Upgrade from "./moves/Upgrade";
import Summon from "./moves/Summon";
import Step from "./moves/Step";
import Research from "./moves/Research";
import Reward from "./moves/Reward";
import EndTurn from "./moves/EndTurn";
import Harvest from "./moves/Harvest";
import Structure from "./moves/Structure";
import Recover from "./moves/abilities/Recover";
import Disband from "./moves/abilities/Disband";
import HealOthers from "./moves/abilities/HealOthers";
import Promote from "./moves/abilities/Promote";
import Attack from "./moves/Attack";
import Capture from "./moves/Capture";
import Boost from "./moves/abilities/Boost";
import Explode from "./moves/abilities/Explode";
import FreezeArea from "./moves/abilities/FreezeArea";
import Destroy from "./moves/abilities/Destroy";
import Decompose from "./moves/abilities/Decompose";
import BurnForest from "./moves/abilities/BurnForest";
import GrowForest from "./moves/abilities/GrowForest";
import ClearForest from "./moves/abilities/ClearForest";
import { TaskSettings } from "./settings/TaskSettings";
import { ResourceSettings } from "./settings/ResourceSettings";

interface ReachableNode {
	index: number;
	cost: number;
	terminal?: boolean;
}

export class MoveGenerator {
	static legal(state: GameState): Move[] {
		if(state.settings._pendingRewards.length) {
			return state.settings._pendingRewards;
		}

		const moves: Move[] = [new EndTurn()];

		ArmyMovesGenerator.all(state, moves);

		EconMovesGenerator.all(state, moves);

		return moves;
	}

	static legalActions(state: GameState): Action[] {
		return MoveGenerator.legal(state).map(x => x.toAction());
	}
}

export class EconMovesGenerator {
	static all(state: GameState, moves: Move[]) {
		EconMovesGenerator.actions(state, moves);
		EconMovesGenerator.resources(state, moves);
		EconMovesGenerator.structures(state, moves);
		EconMovesGenerator.research(state, moves);
	}

	static actions(state: GameState, moves: Move[]) {
		const pov = getPovTribe(state);
		const territory = getPovTerritorry(state);

		const abilities = getPovTribe(state)._tech.reduce((a: any[], b) => ([
			...a,
			...(getTechSettings(b).unlocksAbility ? [getTechSettings(b).unlocksAbility] : [])
		]), []) as AbilityType[];

		for (let i = 0; i < territory.length; i++) {
			const tile = state.tiles[territory[i]];
			const struct = state.structures[tile.tileIndex];

			if(struct && tile._owner == pov.owner) {
				if(abilities.some(x => x == AbilityType.Destroy)) {
					moves.push(new Decompose(tile.tileIndex));	
				}
				else if(abilities.some(x => x == AbilityType.Decompose)) {
					moves.push(new Destroy(tile.tileIndex));	
				}
			}

			if(tile.terrainType === TerrainType.Forest) {
				if(abilities.some(x => x == AbilityType.ClearForest)) {
					moves.push(new ClearForest(tile.tileIndex));
				}
				if(abilities.some(x => x == AbilityType.GrowForest)) {
					moves.push(new GrowForest(tile.tileIndex));
				}
				if(abilities.some(x => x == AbilityType.BurnForest)) {
					moves.push(new BurnForest(tile.tileIndex));
				}
			}

			// TODO should use internal ice boolean
			// Same with Drain
			// if(tile.terrainType === TerrainType.Ice) {
			// }
		}
	}

	static resources(state: GameState, moves: Move[]) {
		const pov = getPovTribe(state);
		const territory = getPovTerritorry(state);

		for (let i = 0; i < territory.length; i++) {
			const tileIndex = territory[i];
			const resource = state.resources[tileIndex];

			if(!resource) {
				continue;
			}
		
			const settings = ResourceSettings[resource.id];
		
			// Too expensive
			// Resource requires a structure
			// There is a structure
			// Tile blocked by enemy
			// Limited by tech visibility or missing tech
			if((settings.cost || 0) > pov._stars
				|| settings.structType
				|| state.structures[tileIndex]
				|| isTileOccupied(state, tileIndex, true)
				|| !isTechUnlocked(pov, settings.techRequired)
				|| !isResourceVisible(pov, resource.id)
			) {
				continue;
			}
		
			moves.push(new Harvest(tileIndex));
		}
	}
	
	static structures(state: GameState, moves: Move[]) {
		const pov = getPovTribe(state);
		const structTypes:StructureType[] = [];

		for (let i = 0; i < 7; i++) {
			const task = TaskSettings[i as keyof typeof TaskSettings];
			if(task.techType && !isTechUnlocked(pov, task.techType)) {
				continue;
			}
			if(pov._builtUniqueStructures.includes(task.structureType)) {
				continue;
			}
			if(task.task(state)) {
				structTypes.push(task.structureType);
			}
		}

		for (let i = 0; i < pov._tech.length; i++) {
			const structType = getTechStructure(pov, pov._tech[i]);
	
			if(!structType) {
				continue
			}
	
			const settings = StructureSettings[structType];
	
			if(!settings.cost || settings.cost < 1 || pov._stars < settings.cost) {
				continue
			}
	
			structTypes.push(structType);
		}

		const territory = getPovTerritorry(state);
	
		for(let i = 0; i < territory.length; i++) {
			const tile = state.tiles[territory[i]];

			if(state.structures[tile.tileIndex]) {
				continue
			}

			if(tile._unitOwner > 0 && tile._unitOwner != pov.owner) {
				continue
			}

			for(let j = 0; j < structTypes.length; j++) {
				const structType = structTypes[j];
				const settings = StructureSettings[structType];

				if(!settings.terrainType?.includes(tile.terrainType)) {
					if(tile.capitalOf !== pov.owner || structType !== StructureType.Embassy) {
						continue;
					}
				}
				
				if(settings.limitedPerCity) {
					const limited = getCityOwningTile(state, tile.tileIndex)!._territory.some(x => state.structures[x]?.id == structTypes[i]);
					if(limited) {
						continue;
					}
				}

				if(settings.adjacentTypes && !getNeighborTiles(state, tile.tileIndex).some(x => state.structures[x.tileIndex]? 
					settings.adjacentTypes!.includes(state.structures[x.tileIndex]!.id) : false
				)) {
					continue
				}
	
				moves.push(new Structure(
					tile.tileIndex,
					structType
				));
			};
		}
	}
	
	static research(state: GameState, moves: Move[]) {
		const pov = getPovTribe(state);
		const techUnlocks: TechnologyType[] = [];

		for (let i = 0; i < pov._tech.length; i++) {
			const techState = pov._tech[i];
			const next = getTechSettings(techState).next;

			if(!next) {
				continue;
			}

			for (let j = 0; j < next.length; j++) {
				if(isTechUnlocked(pov, next[j])) {
					continue;
				}

				const cost = getTechCost(next[j]);

				if(cost > pov._stars) {
					continue;
				}

				moves.push(new Research(next[j], cost));
			}
		}
	}
	
	static rewards(city: CityState): Move[] {
		const rewards = [
			[ RewardType.Workshop, RewardType.Explorer ],
			[ RewardType.CityWall, RewardType.Resources ],
			[ RewardType.PopulationGrowth, RewardType.BorderGrowth ],
		][city._level-2] || [ RewardType.Park, RewardType.SuperUnit ];
		if(city._rewards.some(x => rewards.includes(x))) {
			return [];
		}
		return rewards.map(rewardType => new Reward(city.tileIndex, rewardType));
	}
}

export class ArmyMovesGenerator {
	static all(state: GameState, moves: Move[]) {
		getPovTribe(state)._units.forEach(x => {
			if(x._health > 0) {
				ArmyMovesGenerator.captures(state, x, moves);
				ArmyMovesGenerator.actions(state, x, moves);
				ArmyMovesGenerator.attacks(state, x, moves);
				ArmyMovesGenerator.steps(state, x, moves);
			}
		});
		ArmyMovesGenerator.summons(state, moves);
	}

	static actions(state: GameState, unit: UnitState, _moves: Move[]) {
		const tileIndex = unit._tileIndex;

		// Promote
		if(!unit.veteran && unit.kills >= 3) {
			_moves.push(new Promote(tileIndex));
		}

		if(unit._moved || unit._attacked) return [];

		// Disband
		if(isTechUnlocked(getPovTribe(state), TechnologyType.FreeSpirit)) {
			_moves.push(new Disband(tileIndex));
		}

		// Recover
		if(unit._health < getMaxHealth(unit)) {
			_moves.push(new Recover(tileIndex));
		}

		// Heal Others
		if(isSkilledIn(unit, SkillType.Heal)) {
			const damagedAround = getAlliesNearTile(state, tileIndex).some(x => x._health < getMaxHealth(x));
			if(damagedAround) {
				_moves.push(new HealOthers(tileIndex));
			}
		}

		// Boost
		if(isSkilledIn(unit, SkillType.Boost)) {
			const unboostedAround = getAlliesNearTile(state, tileIndex).some(x => !isBoosted(x));
			if(unboostedAround) {
				_moves.push(new Boost(tileIndex));
			}
		}
		
		// Explode
		if(isSkilledIn(unit, SkillType.Explode)) {
			const enemiesAround = getEnemiesNearTile(state, tileIndex, 1, true).length;
			if(enemiesAround) {
				_moves.push(new Explode(tileIndex));
			}
		}

		// Freeze Area
		if(isSkilledIn(unit, SkillType.FreezeArea)) {
			const unitsAround = getEnemiesNearTile(state, tileIndex, 1, true).some(x => !isFrozen(x) || !isTileFrozen(state, x._tileIndex));
			if(unitsAround) {
				_moves.push(new FreezeArea(tileIndex));
			}
		}

		return _moves;
	}

	static captures(state: GameState, capturer: UnitState, moves: Move[]) {
		if (capturer._moved || capturer._attacked) return null;

		const pov = state.tribes[capturer._owner];
		const targetCityIndex = capturer._tileIndex;
		const struct = state.structures[targetCityIndex];
		const resource = state.resources[targetCityIndex];
		
		if(struct) {
			if(struct.id == StructureType.Village && state.tiles[targetCityIndex]._owner !== capturer._owner) {
				moves.push(new Capture(capturer._tileIndex));
			}
			if(struct.id == StructureType.Ruin) {
				moves.push(new Capture(capturer._tileIndex));
			}
		}
		else if(resource && resource.id == ResourceType.Starfish && isTechUnlocked(pov, TechnologyType.Navigation)) {
			moves.push(new Capture(capturer._tileIndex));
		}
	}

	static attacks(state: GameState, attacker: UnitState, moves: Move[]) {
		if (attacker._attacked || attacker._health <= 0) return [];

		if (isSkilledIn(attacker, SkillType.Infiltrate)) {
			moves.push(
				...getNeighborIndexes(state, attacker._tileIndex)
					.filter(x => isEnemyCity(state, x) && !isUnderSiege(state, x))
					.map(x => new Attack(attacker._tileIndex, x))
			);
		}
		else {
			// Skip other units that cant attack, eg: raft, mooni
			if(getUnitAttack(attacker) < 0){
				return [];
			}
			moves.push(
				...getEnemiesInRange(state, attacker).map(x => new Attack(attacker._tileIndex, x._tileIndex))
			);
		}

		return moves;
	}

	static summons(state: GameState, moves: Move[]) {
		const tribe = state.tribes[state.settings._pov];
		const upgradables: UnitType[] = [];
		const spawnables: UnitType[] = [];
		
		tribe._tech.map(x => {
			const unitType = getTechUnitType(tribe, x);

			if(!unitType) {
				return null;
			}

			const settings = UnitSettings[unitType];

			if(settings.cost < 1 || tribe._stars < settings.cost) {
				return null;
			}

			if(settings.upgradeFrom) {
				upgradables.push(unitType);
			}
			else {
				spawnables.push(unitType);
			}
		});

		if (spawnables.length) {
			const cities = tribe._cities;
			for (let i = 0; i < cities.length; i++) {
				if(cities[i]._unitCount > cities[i]._level || isTileOccupied(state, cities[i].tileIndex)) {
					continue;
				}
				for (let j = 0; j < spawnables.length; j++) {
					moves.push(new Summon(cities[i].tileIndex, spawnables[j]))
				}
			}
		}

		if(upgradables.length) {
			for(let i = 0; i < tribe._units.length; i++) {
				if(tribe._units[i]._unitType != UnitType.Raft || isTileOccupied(state, tribe._units[i]._tileIndex)) {
					continue;
				}
				for (let j = 0; j < upgradables.length; j++) {
					moves.push(new Upgrade(tribe._units[i]._tileIndex, upgradables[j]))
				}
			}
		}

		return moves;
	}

	static steps(state: GameState, unit: UnitState, moves: Move[]) {
		if (unit._moved) return;

		const steps = Array.from(ArmyMovesGenerator.computeReachableTiles(state, unit).entries());

		for (const [tileIndex, ] of steps) {
			if (unit._tileIndex == tileIndex) {
				continue;
			}
			moves.push(new Step(unit._tileIndex, tileIndex));
		}
	}

	static computeStep(state: GameState, stepper: UnitState, toTileIndex: number, forced = false): CallbackResult {
		if (!forced && (stepper._moved || state.tiles[toTileIndex]._unitOwner > 0 || stepper._tileIndex == toTileIndex)) {
			return Logger.illegal(MoveType.Step, `${stepper._tileIndex} -> ${toTileIndex}, ${getRealUnitType(stepper)} -> ${getRealUnitType(getTrueUnitAt(state, toTileIndex)!)} -${forced}-`);
		}

		const chain: UndoCallback[] = [];
		const rewards = [];

		// const scoreArmy = state._scoreArmy;

		const iX = stepper.x;
		const iY = stepper.y;
		const ipX = stepper.prevX;
		const ipY = stepper.prevY;

		const oldTileIndex = stepper._tileIndex;
		const oldMoved = stepper._moved;
		const oldAttacked = stepper._attacked;
		const oldTile = state.tiles[oldTileIndex];
		const oldType = stepper._unitType;
		const oldPassenger = stepper._passenger;

		const newTile = state.tiles[toTileIndex];
		let newType = oldType;

		const oldNewTileUnitOwner = newTile._unitOwner;
		const oldNewTileUnitIdx = newTile._unitIdx;

		// TODO; this is not how prev works, it must be applies at the end of the turn
		stepper.prevX = iX;
		stepper.prevY = iY;
		stepper.x = toTileIndex % state.settings.size;
		stepper.y = Math.floor(toTileIndex / state.settings.size);
		stepper._tileIndex = toTileIndex;

		oldTile._unitIdx = -1;
		oldTile._unitOwner = -1;
		newTile._unitIdx = stepper.idx;
		newTile._unitOwner = stepper._owner;

		// Apply movement skills

		// TODO what other skills are missing?

		// Stomp
		if(isSkilledIn(stepper, SkillType.Stomp)) {
			chain.push(splashDamageArea(state, stepper, 4));
		}

		// AutoFreeze
		chain.push(freezeArea(state, stepper));

		// Discover terrain
		const preLighthouseCount = state._lighthouses.length;
		const resultDiscover = discoverTiles(state, stepper)!;
		rewards.push(...resultDiscover.rewards);
		chain.push(resultDiscover.undo);

		if(state._lighthouses.length != preLighthouseCount) {
			const capitalCity = getCapitalCity(state);
			if(capitalCity) {
				const result = addPopulationToCity(state, capitalCity, 1)!;
				chain.push(result.undo);
				if(result.rewards) {
					rewards.push(...result.rewards);
				}
			}
		}

		stepper._moved = stepper._attacked = true;

		// Moved to port
		// If ground non floatable or aquatic unit is moving to port, place into boat
		if (state.structures[toTileIndex]?.id == StructureType.Port && !isAquaticOrCanFly(stepper)) {
			// Add embark special units
			switch (stepper._unitType) {
				case UnitType.Cloak:
					newType = UnitType.Dinghy;
					break;
				case UnitType.Dagger:
					newType = UnitType.Pirate;
					break;
				case UnitType.Giant:
					newType = UnitType.Juggernaut;
					break;
				default:
					newType = UnitType.Raft;
					stepper._passenger = oldType;
					break;
			}
		}
		// Carry allows a unit to carry another unit inside
		// A unit with the carry skill can move to a land tile adjacent to water
		// Doing so releases the unit it was carrying and ends the unit's turn
		else if(isSkilledIn(stepper, SkillType.Carry) && !isWaterTerrain(newTile)) {
			stepper._passenger = undefined;
			switch (stepper._unitType) {
				case UnitType.Dinghy:
					newType = UnitType.Cloak;
					break;
				case UnitType.Pirate:
					newType = UnitType.Dagger;
					break;
				case UnitType.Juggernaut:
					newType = UnitType.Giant;
					break;
				default:
					newType = oldPassenger!;
					break;
			}
		}
		// Allows a unit to attack after moving if there are any enemies in range
		// And if it had moved before
		else if(!forced && !oldMoved && isSkilledIn(stepper, SkillType.Dash) && getEnemiesInRange(state, stepper).length > 0) {
			stepper._attacked = false;
		}
		
		// Going stealth mode uses up our attack
		let wasNotInvis = false;
		if(isSkilledIn(stepper, SkillType.Hide) && !isInvisible(stepper)) {
			stepper._attacked = true;
			stepper._effects.push(EffectType.Invisible);
			wasNotInvis = true;
		}

		stepper._unitType = newType;
		
		return {
            rewards,
			undo: () => {
				stepper._unitType = oldType;
				if(wasNotInvis) {
					stepper._effects.pop();
				}
				stepper._passenger = oldPassenger;
				stepper._attacked = oldAttacked;
				stepper._moved = oldMoved;
				chain.reverse().forEach(x => x());
				newTile._unitOwner = oldNewTileUnitOwner;
				newTile._unitIdx = oldNewTileUnitIdx;
				oldTile._unitIdx = stepper.idx;
				oldTile._unitOwner = stepper._owner;
				stepper._tileIndex = oldTileIndex;
				stepper.y = iY;
				stepper.x = iX;
				stepper.prevX = ipX;
				stepper.prevY = ipY;
			}
		};
	}

	/**
	* Returns a map of reachable positions to the cost required to get there.
	*/
	static computeReachableTiles(state: GameState, unit: UnitState): Map<number, number> {
		const effectiveMovement = getUnitMovement(unit);
		
		const reachable = new Map<number, number>();
		const openList: ReachableNode[] = [];

		openList.push({ index: unit._tileIndex, cost: 0 });
		reachable.set(unit._tileIndex, 0);

		while (openList.length > 0) {
			openList.sort((a, b) => a.cost - b.cost);
			const current = openList.shift()!;
	
			if (current.terminal) {
				continue;
			}
	
			const neighbors = getNeighborTiles(state, current.index);
	
			for (let i = 0; i < neighbors.length; i++) {
				const tile = neighbors[i];
				const index = tile.tileIndex;
				if (index == unit._tileIndex) continue;
	
				if (!isSteppable(state, unit, tile)) continue;
	
				const moveCost = this.computeMovementCost(state, unit, current.index, tile);
				if (moveCost < 0) continue;
	
				let newCost = current.cost + moveCost;
	
				if (newCost - effectiveMovement > 1e-6) continue;
	
				const terminal = this.isTerminal(state, unit, tile);
	
				const existingCost = reachable.get(index);
				if (existingCost === undefined || newCost < existingCost) {
					reachable.set(index, newCost);
					openList.push({ index, cost: newCost, terminal });
				}
			}
		}

		return reachable;
	}

	static computeMovementCost(state: GameState, unit: UnitState, fromIndex: number, toTile: TileState): number {
		let cost = 1;

		// If the unit is NOT flying or creeping (which disable road bonuses),
		if (isSkilledIn(unit, SkillType.Fly, SkillType.Creep)) {
			return cost;
		}

		// Road bonus
		if(isRoadpathAndUsable(state, unit, fromIndex) && isRoadpathAndUsable(state, unit, toTile.tileIndex)) {
			cost = 0.5;
		}

		// Skate doubles movement on ice (i.e. halves cost)
		if (toTile.terrainType === TerrainType.Ice && isSkilledIn(unit, SkillType.Skate)) {
			cost *= 0.5; 
		}

		return cost;
	}

	static isTerminal(state: GameState, unit: UnitState, tile: TileState): boolean {
		if(isSkilledIn(unit, SkillType.Fly)) {
			return false;
		}

		// (mountains/forests without bypass skills) blocks further movement.
		// roads allow movement on forest if it has a road
		switch (tile.terrainType) {
			case TerrainType.Forest:
				if(isSkilledIn(unit, SkillType.Creep)) {
					return false;
				}
				else {
					return !isRoadpathAndUsable(state, unit, tile.tileIndex);
				}
			case TerrainType.Mountain:
				if(isSkilledIn(unit, SkillType.Creep)) {
					return false;
				}
				break;
		}

		// Ports
		const isPort = state.structures[tile.tileIndex]?.id == StructureType.Port && tile._owner == unit._owner;
		// Entering a Port for non-fly units turns them into Rafts (ending their turn).
		if (isPort) {
			return !isAquaticOrCanFly(unit, false);
		}

		// Water movement / Disembarking
		if (isNavalUnit(unit)) {
			if(isWaterTerrain(tile)) {
				return false;
			}
			else {
				return true;
			}
		}
		else {
			// If unit has Navigate, it cannot move onto land, except for capturing cities
			if (isSkilledIn(unit, SkillType.Navigate)) {
				return state.structures[tile.tileIndex]?.id == StructureType.Village;
			}
		}

		// ZoC (zone of control)
		// Moving adjacent to an enemy stops further movement (unless Creep).
		// TODO The rule about re-entry is ambiguous
		return isSkilledIn(unit, SkillType.Creep) || isAdjacentToEnemy(state, tile);
	}
}