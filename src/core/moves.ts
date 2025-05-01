import { CityState, GameState, TileState, UnitState } from "./states";
import { getNeighborTiles, getTerritorryTiles, isTechLocked, isResourceVisible, getNeighborIndexes, isAdjacentToEnemy, isAquaticOrCanFly, isSteppable, isWaterTerrain, getEnemiesInRange, isNavalUnit, getTechCost, getPovTribe, isSkilledIn, getCapitalCity, getRealUnitSettings, getUnitSettings, getUnitRange, getUnitAttack, getUnitMovement, isRoadpathAndUsable, getTrueUnitAtTile, getUnitAtTile, getHomeCity, isInvisible, getMaxHealth, isInTerritory, getAlliesNearTile, getRealUnitType } from './functions';
import { StructureByTerrain, StructureSettings } from "./settings/StructureSettings";
import { ResourceSettings } from "./settings/ResourceSettings";
import { SkillType, CaptureType, EffectType, ResourceType, RewardType, StructureType, TechnologyType, TerrainType, TribeType, UnitType, AbilityType } from "./types";
import { TechnologySettings } from "./settings/TechnologySettings";
import { addPopulationToCity, attackUnit, buildStructure, discoverTiles, freezeArea, harvestResource, healUnit, pushUnit, removeUnit, splashDamageArea, summonUnit } from "./actions";
import { rewardCapture, rewardTech, rewardUnitAttack, rewardUnitMove } from "../eval/eval";
import { UnitSettings } from "./settings/UnitSettings";
import { TribeSettings } from "./settings/TribeSettings";
import Move, { CallbackResult, MoveType } from "./move";
import AIState, { MODEL_CONFIG } from "../aistate";
import { Logger } from "../polyfish/logger";
import Upgrade from "./moves/Upgrade";
import Summon from "./moves/Summon";
import Step from "./moves/Step";

export interface ReachableNode {
	index: number;
	cost: number;
	terminal?: boolean;
}

export type UndoCallback = () => void;

export interface Branch {
	moves?: Move[];
	chainMoves?: Move[];
	undo: UndoCallback;
}

/**
 * Format: EndTurn, Eco(Resouce, Struct, Tech)[], Army[]
 * @param state 
 * @returns 
 */
export function generateAllMoves(state: GameState): Move[] {
	const moves = [
		generateEndTurnMove(),
		...generateEcoMoves(state),
		...generateArmyMoves(state),
	];
	if(moves.length > MODEL_CONFIG.max_actions) {
		console.warn(`Too many actions! ${moves.length}/${MODEL_CONFIG.max_actions}`);
	}
	return moves;
}

export function generateArmyMoves(state: GameState): Move[] {
	return getPovTribe(state)._units.map(x => UnitMoveGenerator.all(state, x)).flat();
}

export function generateEcoMoves(state: GameState): Move[] {
	const moves = [
		...generateResourceMoves(state),
		...generateStructureMoves(state),
		...generateTechMoves(state),
	];
	return moves;
}

/** Placeholder for the End Turn move, doesnt do anything */
export function generateEndTurnMove(): Move {
	return new Move(
		MoveType.EndTurn,
		0, 0, 0,
		// id: 'endturn-'+state.settings._turn,
		(state: GameState) => {
			// TODO temples logic
			return {
				moves: [],
				undo: () => { },
			};
		},
	)
}

export function generateResourceMoves(state: GameState, cityTarget?: CityState | null, targetTileIndex?: number): Move[] {
	const moves: Move[] = [];
	const tribe = state.tribes[state.settings._pov];
	
	if(targetTileIndex) {
		const resource = state.resources[targetTileIndex]!;
		return [new Move(
			MoveType.Harvest,
			targetTileIndex, 0, resource.id,
			// id: `harvest-${resource.id}-${targetTileIndex}`,
			(state: GameState) => harvestResource(state, targetTileIndex),
		)];
	}

	for (const tileIndex of cityTarget? cityTarget._territory : tribe._resources) {
		const resource = state.resources[tileIndex];

		if(!resource) continue;

		const settings = ResourceSettings[resource.id];

		// Cant afford it
		if((settings.cost || 0) > tribe._stars) continue;

		// Resource requires a structure
		if(settings.structType) continue;

		// Cant harvest while there is a structure built
		const struct = state.structures[tileIndex];
		if(struct && struct.id != StructureType.None) continue;

		const tile = state.tiles[tileIndex];

		// If enemy is standing on resource, tile is blocked
		if(tile._unitOwner > 0 && tile._unitOwner != tribe.tribeType) continue;

		// Resource is unharvestable
		if(isTechLocked(tribe, settings.techRequired)) continue;

		// Resource is limited by tech visibility
		if(!isResourceVisible(tribe, resource.id)) continue;

		moves.push(new Move(
			MoveType.Harvest,
			tileIndex, 0, resource.id,
			// id: `harvest-${resource.id}-${tileIndex}`,
			(state: GameState) => harvestResource(state, tileIndex),
		));
	}

	return moves;
}

export function generateStructureMoves(state: GameState, cityTarget?: CityState | null, indexAndType?: [number, StructureType]): Move[] {
	const moves: Move[] = [];
	const tribe = state.tribes[state.settings._pov];

	if(indexAndType) {
		const [tileIndex, structId] = indexAndType;
		return [new Move(
			MoveType.Build,
			tileIndex, 0, structId,
			(state: GameState) => buildStructure(state, structId, tileIndex),
		)];
	}

	const territory = cityTarget? cityTarget._territory.map(x => state.tiles[x]) : getTerritorryTiles(state, tribe);
	const placebleTiles = territory.filter(x => !state.structures[x.tileIndex]);

	for(const targetTile of placebleTiles) {
		for(const structId of StructureByTerrain[targetTile.terrainType]) {
			const settings = StructureSettings[structId];

			// ! Struct does not belong to this tribe
			if(settings.tribeType && settings.tribeType != tribe.tribeType) continue;

			if((settings.cost || 0) > tribe._stars) continue;

			if(isTechLocked(tribe, settings.techRequired)) continue;

			// ! Resource is required (and is visible)
			if(settings.resourceType) {
				if(state.resources[targetTile.tileIndex]?.id != settings.resourceType) continue;
				if(!isResourceVisible(tribe, settings.resourceType)) continue;
			}
			// EVAL: If doesnt require a resource but we are building ontop of a resource, assume bad move
			else if(state.resources[targetTile.tileIndex]) {
				continue;
			}

			// ! If enemy is standing on resource, tile is blocked
			if(targetTile._unitOwner > 0 && targetTile._unitOwner != tribe.owner) continue;

			// ! Task already built or not completed
			if(settings.task) {
				if(tribe._builtUniqueStructures.some(x => x == structId) || !settings.task(tribe, state.tiles)) continue;
				// EVAL Skip building it if it doesnt level up the city
				const city = tribe._cities.find(x => targetTile._rulingCityIndex)!;
				const isLvlUp = (city._progress + (settings.rewardPop || 3)) >= city._level + 1;
				if(!isLvlUp) {
					continue;
				}
			}

			// ! Structure has already been built in this city's territory
			if(settings.limitedPerCity) {
				if(territory.some(x => state.structures[x.tileIndex]?.id == structId && x._rulingCityIndex == targetTile._rulingCityIndex)) {
					continue;
				}
			}

			// ! Adjacent tiles do not contain matching structure
			if(settings.adjacentTypes != undefined &&
				!getNeighborTiles(state, targetTile.tileIndex).some(x => state.structures[x.tileIndex]? settings.adjacentTypes!.includes(state.structures[x.tileIndex]!.id) : false)
			) continue;

			return [new Move(
				MoveType.Build,
				targetTile.tileIndex, 0, structId,
				(state: GameState) => 
					buildStructure(state, structId, targetTile.tileIndex),
			)];
		};
	}

	return moves;
}

export function generateTechMoves(state: GameState, techType?: TechnologyType): Move[] {
	const moves: Move[] = [];
	const tribe = state.tribes[state.settings._pov];

	if(techType) {
		const cost = getTechCost(tribe, techType);

		return [new Move(
			MoveType.Research,
			0, 0, techType,
			(state: GameState) => {
				tribe._tech.push(techType);

				if(state.settings.live) {
					tribe._trueTech.push(techType);
				}
				
				tribe._stars -= cost;

				const oldScoreEconomy = state._scoreTech;

				state._scoreTech += rewardTech(state, techType);

				return {
					moves: [],
					undo: () => {
						state._scoreTech = oldScoreEconomy;
						tribe._stars += cost;
						tribe._tech.pop();
					}
				};
			},
		)];
	}

	const dissalowed: TechnologyType[] = [];
	
	for(const techId of Object.keys(TechnologySettings)) {
		const unlockedTech: TechnologyType = Number(techId);

		if(unlockedTech == TechnologyType.Unbuildable) continue;

		if(dissalowed.includes(unlockedTech)) continue;

		let settings = TechnologySettings[unlockedTech];

		if(settings.replaced && settings.tribeType == tribe.tribeType) {
			dissalowed.push(settings.replaced);
		}

		// ! Owner doesnt match
		if(settings.tribeType && settings.tribeType != tribe.tribeType) continue;

		// ! Already unlocked
		if(!isTechLocked(tribe, unlockedTech)) continue;

		// ! Previous tier tech not unlocked
		if(settings.requires && isTechLocked(tribe, settings.requires)) continue;

		const cost = getTechCost(tribe, unlockedTech);

		if(cost > tribe._stars) continue;

		moves.push(new Move(
			MoveType.Research,
			0, 0, unlockedTech,
			(state: GameState) => {
				tribe._tech.push(unlockedTech);
				tribe._stars -= cost;

				const oldScoreEconomy = state._scoreTech;

				state._scoreTech += rewardTech(state, unlockedTech);

				return {
					moves: [],
					undo: () => {
						state._scoreTech = oldScoreEconomy;
						tribe._tech = tribe._tech.filter(x => x != unlockedTech);
						tribe._stars += cost;
					}
				};
			},
		));
	}

	return moves.filter(x => !dissalowed.some(y => y == x.type));
}

// Does not work with generator, deep compare fails
export function generateRewardMoves(state: GameState, rulingCityIndex: number, rewardType?: RewardType): Move[] {
	const tribe = state.tribes[state.settings._pov];
	const city = tribe._cities.find(x => x.tileIndex == rulingCityIndex);
	
	if(!city) return [];

	const options: RewardType[] = rewardType? [rewardType] : ([
		[ RewardType.Workshop, RewardType.Explorer ],
		[ RewardType.CityWall, RewardType.Resources ],
		[ RewardType.PopulationGrowth, RewardType.BorderGrowth ],
	][city._level-2] || [ RewardType.Park, RewardType.SuperUnit ]);

	return options.map(rewardType => new Move(
		MoveType.Reward,
		0, 0, rewardType,
		(state: GameState) => {
			let undoReward: UndoCallback = () => { };
			const oldProduction = city._production;
			const oldPopulation = city._population;
			const oldProgress = city._progress;

			switch (rewardType) {
				case RewardType.Workshop:
					city._production++;
					break;
				case RewardType.Explorer:
					const amount = Math.floor(Math.random() * 11) + 9;
					state._potentialDiscovery.push(...Array(amount).fill(-1));
					undoReward = () => { 
						state._potentialDiscovery.splice(state._potentialDiscovery.length - amount, amount);
					};
					break;
				case RewardType.CityWall:
					city._walls = true;
					undoReward = () => {
						city._walls = false;
					};
					break;
				case RewardType.Resources:
					tribe._stars += 5;
					undoReward = () => {
						tribe._stars -= 5;
					}
					break;
				case RewardType.PopulationGrowth:
					city._population += 3;
					city._progress += 3;
					break;
				case RewardType.BorderGrowth:
					city._borderSize++;
					const tileIndex = city.tileIndex;
					const undiscovered = getNeighborTiles(state, tileIndex, city._borderSize, false, true)
						.filter(x => !x.explorers.includes(tribe.owner) && !state._potentialDiscovery.includes(x.tileIndex));
					state._potentialDiscovery.push(...undiscovered.map(x => x.tileIndex));
					if(state.settings.live) {
						for(const tile of undiscovered) {
							state._visibleTiles.push(tileIndex);
							tile.explorers.push(tribe.owner);
						}
						const unowned = undiscovered.filter(x => x._owner < 1);
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
						state._potentialDiscovery = state._potentialDiscovery.slice(0, state._potentialDiscovery.length - undiscovered.length);
						city._borderSize--;
					}
				break;
				case RewardType.Park:
					city._production++;
					break;
				case RewardType.SuperUnit:
					let undoPush: UndoCallback = pushUnit(state, city.tileIndex);
					
					const undoSummon = summonUnit(state, 
						TribeSettings[tribe.tribeType].uniqueSuperUnit || UnitType.Giant, 
						city.tileIndex
					);
					
					undoReward = () => {
						undoSummon();
						undoPush();
					}
					break;
				default:
					return Logger.illegal(MoveType.Reward, `Invalid type ${rewardType}`);
			}
			
			city._rewards.push(rewardType);
			
			return {
				moves: [],
				undo: () => {
					// state._potentialEconomy -= 0;
					city._rewards.pop();
					undoReward();
					city._production = oldProduction;
					city._population = oldPopulation;
					city._progress = oldProgress;
				},
			};
		}
	)) as Move[];
}

// TODO move to actions.ts

export default class UnitMoveGenerator {
	static all(state: GameState, unitTarget: UnitState, actionsOnly = false): Move[] {
		if (!unitTarget || unitTarget._health < 1) return [];
		const moves: Move[] = [];

		if (!actionsOnly) {
			UnitMoveGenerator.steps(state, unitTarget, moves);
			UnitMoveGenerator.attacks(state, unitTarget, moves);
			moves.push(...UnitMoveGenerator.spawns(state));
		}

		// UnitMoveGenerator.actions(state, unitTarget, moves);

		UnitMoveGenerator.captures(state, unitTarget, moves);

		// ! MCTS EVAL !
		// Prioritize moves
		// Capture -> Ability -> Attack -> Step

		const TABLE: { [moveType: number]: number } = {
			[MoveType.Capture]: 4,
			[MoveType.Ability]: 3,
			[MoveType.Attack]: 2,
			[MoveType.Step]: 1,
		};
		const MAX = 7;

		moves.sort((a, b) => (TABLE[a.moveType] || Math.floor(Math.random() * MAX)) - (TABLE[b.moveType] || Math.floor(Math.random() * MAX)));

		return moves;
	}

	/**
	 * TODO
	 * Generate all unit actions for a unit
	 * @param state 
	 * @param unitTarget 
	 */
	static actions(state: GameState, unitTarget: UnitState, _moves?: Move[]): Move[] {
		const moves = [];

		// Disband
		if(!isTechLocked(getPovTribe(state), TechnologyType.FreeSpirit)) {
			moves.push(new Move(
				MoveType.Action,
				AbilityType.Disband, 0, 0,
				(state: GameState) => {
					const tribe = getPovTribe(state);
					const undoRemove = removeUnit(state, unitTarget);
					const cost = Math.floor(getRealUnitSettings(unitTarget).cost / 2);
					tribe._stars += cost;
					return {
						moves: [],
						undo: () => {
							tribe._stars -= cost;
							undoRemove()
						}
					}
				}
			));
		}

		// Recover
		if(unitTarget._health < getMaxHealth(unitTarget)) {
			moves.push(new Move(
				MoveType.Action,
				AbilityType.Recover, 0, 0,
				(state: GameState) => {
					const undoHeal = healUnit(unitTarget, isInTerritory(state, unitTarget)? 4 : 2)
					return {
						moves: [],
						undo: () => {
							undoHeal();
						}
					};
				}
			));
		}

		// Heal Others
		if(isSkilledIn(unitTarget, SkillType.Heal)) {
			const adjAllies = getAlliesNearTile(state, unitTarget._tileIndex);
			moves.push(new Move(
				MoveType.Action,
				AbilityType.HealOthers, 0, 0,
				(state: GameState) => {
					const chain: UndoCallback[] = [];
					for(const unit of adjAllies) {
						chain.push(healUnit(unit, 4));
					}
					return {
						moves: [],
						undo: () => {
							chain.forEach(x => x());
						}
					}
				},
			));
		}

		// TODO Promotion

		// TODO 
		// ? tile state filed required frozen
		// UnitActionType.BreakIce;
		// ? tile state filed required flodded
		// UnitActionType.Fill;
		// UnitActionType.FloodTile;
		// TODO finish this
		// UnitActionType.FreezeArea;
		// UnitActionType.Boost;
		// UnitActionType.Explode;

		return moves;
	}

	static captures(state: GameState, sieger: UnitState, _moves?: Move[]): Move | null {
		if (sieger._moved || sieger._attacked) return null;

		const us = state.tribes[sieger._owner || 1]!;
		const targetCityIndex = sieger._tileIndex;
		const struct = state.structures[targetCityIndex];
		const resource = state.resources[targetCityIndex];
		const scoreArmy = state._scoreArmy;

		let move: Move | null = null;

		if (struct && struct.id == StructureType.Village && struct._owner != sieger._owner) {
			move = new Move(
				MoveType.Capture,
				sieger._tileIndex, 0, struct._owner < 1 ? CaptureType.Village : CaptureType.City,
				// id: `capture-${sieger.idx}-${sieger._tileIndex}-${targetCityIndex}`,
				(state: GameState) => {
					let undoCapture: UndoCallback = () => { };

					const oldMoved = sieger._moved;
					const oldAttacked = sieger._attacked;
					const oldHome = sieger._homeIndex;
					const usOwner = sieger._owner;
					const cityTile = state.tiles[targetCityIndex];
					const cityStruct = state.structures[targetCityIndex]!;

					if (getHomeCity(state, sieger)) {
						sieger._homeIndex = targetCityIndex;
					}

					sieger._moved = true;
					sieger._attacked = true;

					// Belongs to no tribe
					if (struct._owner < 1) {
						state._potentialArmy++;
						state._potentialEconomy++;
						state._scoreArmy += rewardCapture(state, sieger, CaptureType.Village);

						const createdCity: CityState = {
							name: `${TribeType[us.tribeType]} City`,
							_population: 0,
							_progress: 0,
							_borderSize: 1,
							_connectedToCapital: false,
							_level: 1,
							_production: 1,
							_owner: usOwner,
							tileIndex: cityTile.tileIndex,
							_rewards: [],
							_territory: struct._potentialTerritory!.filter(x => state.tiles[x]._rulingCityIndex < 0),
							_unitCount: 1,
						} as CityState;

						us._cities.push(createdCity);
						// Claim unowned terrirory
						cityStruct!._owner = usOwner;
						cityTile._owner = usOwner;

						for (const tileIndex of createdCity._territory) {
							const tile = state.tiles[tileIndex];
							tile._owner = usOwner;
							tile._rulingCityIndex = cityTile.tileIndex;
							const struct = state.structures[tileIndex];
							const resource = state.resources[tileIndex];
							if (struct) {
								struct._owner = usOwner;
							}
							if (resource) {
								resource._owner = usOwner;
								us._resources.push(tileIndex);
							}
						}

						// console.log(unclaimedTerrirory.length);
						undoCapture = () => {
							// console.log(getNeighborTiles(state, tileIndex, 1).filter(x => x._rulingCityIndex == tileIndex).length);
							for (const tileIndex of createdCity._territory) {
								const tile = state.tiles[tileIndex];
								tile._owner = -1;
								tile._rulingCityIndex = -1;
								const struct = state.structures[tileIndex];
								const resource = state.resources[tileIndex];
								if (struct) {
									struct._owner = -1;
								}
								if (resource) {
									resource._owner = -1;
									us._resources.pop();
								}
							}
							cityStruct!._owner = -1;
							cityTile._owner = -1;
							us._cities.pop();
							state._potentialArmy--;
							state._potentialEconomy--;
						};
					}
					// Belongs to an enemy tribe
					else {
						const them = state.tribes[struct._owner];
						const themOwner = struct._owner;
						const capturedCity = state.tribes[themOwner]!._cities.find(x => x.tileIndex == targetCityIndex)!;
						// TODO SHOULD BE THIS INSTEAD OF ABOVE
						// const capturedCity = getRulingCity(state, targetCityIndex)!;
						const oldName = capturedCity.name;

						capturedCity.name = `${TribeType[us.tribeType]} ${state.tiles[targetCityIndex].capitalOf > 0? 'Capital' : 'City'}`;

						// EVAL Reward for capturing enemy city
						state._potentialArmy += 2;
						state._potentialEconomy += 2;
						state._scoreArmy += rewardCapture(state, sieger, CaptureType.City);

						let totalExplored = 0;

						capturedCity._owner = usOwner;
						// TODO enemyCity.progress neg population logic (also on unit death it should add if already neg)

						const cityListIndex = them._cities.indexOf(capturedCity);

						them._cities.splice(cityListIndex, 1)
						us._cities.push(capturedCity);
						
						cityStruct!._owner = usOwner;
						cityTile._owner = usOwner;

						// Claim the city's territory
						const enemyResources: number[] = [...them._resources];

						for (let i = 0; i < capturedCity._territory.length; i++) {
							const tileIndex = capturedCity._territory[i];
							const tile = state.tiles[tileIndex];
							
							if(tile.explorers.includes(usOwner) || state.settings.live) {
								const struct = state.structures[tileIndex];
								const resource = state.resources[tileIndex];

								tile._owner = usOwner;
								if (tile.capitalOf > 0) tile.capitalOf = usOwner;

								if (struct) {
									struct._owner = usOwner;
								}
	
								if (resource) {
									us._resources.push(tileIndex);
									them._resources.splice(them._resources.indexOf(tileIndex), 1);
								}
							}
							else {
								totalExplored++;
								state._potentialDiscovery.push(tileIndex);
							}
						}

						// If they ran out of cities, then they have lost
						if(state.settings.live) {
							// Tribe ran out of cities
							if(!them._cities.length) {
								// console.log(`${TribeType[them.tribeType]} has been eliminated by ${TribeType[us.tribeType]}!`);

								// Remove all units
								for(const unit of them._units) {
									removeUnit(state, unit);
								}

								them._killedTurn = state.settings._turn;
								them._killerId = us.owner;
							}
						}

						undoCapture = () => {
							for (let i = 0; i < capturedCity._territory.length; i++) {
								const tileIndex = capturedCity._territory[i];
								const tile = state.tiles[tileIndex];
								
								if(tile.explorers.includes(usOwner)) {
									const struct = state.structures[tileIndex];
									const resource = state.resources[tileIndex];
	
									tile._owner = themOwner;
									if (tile.capitalOf > 0) tile.capitalOf = themOwner;
	
									if (struct) {
										struct._owner = themOwner;
									}
		
									if (resource) {
										us._resources.pop();
									}
								}
							}
							// TODO negative population logic
							them._resources = [...enemyResources];
							cityTile._owner = themOwner;
							cityStruct!._owner = themOwner;
							us._cities.pop();
							them._cities.splice(cityListIndex, 0, capturedCity);
							capturedCity._owner = themOwner;
							state._potentialDiscovery = state._potentialDiscovery.slice(0, state._potentialDiscovery.length - totalExplored);
							state._potentialEconomy -= 2;
							state._potentialArmy -= 2;
							capturedCity.name = oldName;
						};
					}

					return {
						moves: [],
						undo: () => {
							state._scoreArmy = scoreArmy;
							undoCapture();
							sieger._moved = oldMoved;
							sieger._attacked = oldAttacked;
							sieger._homeIndex = oldHome;
						}
					};
				},
			);
		}
		else if (struct && struct.id == StructureType.Ruin) {
			move = new Move(
				MoveType.Capture,
				sieger._tileIndex, 0, CaptureType.Ruins,
				// id: `ruins-${sieger.idx}-${targetCityIndex}`,
				(state: GameState) => {
					sieger._attacked = true;
					sieger._moved = true;

					let potentialRewards: number = 1;

					// If Tech tree is not completed
					// +10 stars
					if (us._tech.some(x => TechnologySettings[x].next && TechnologySettings[x].next.some(x => isTechLocked(us, x)))) {
						potentialRewards += 1;
					}
					// If player owns a capital city
					// +3 pop
					if (us._cities.some(x => state.tiles[x.tileIndex].capitalOf > 0)) {
						potentialRewards += 2;
					}
					// Any tile within a 5x5 radius that has not been explored will be that explorers' first move
					if (getNeighborIndexes(state, targetCityIndex, 2, false, true).some(x => !state.tiles[x].explorers.includes(us.owner))) {
						// Cymanti cannot get explorers from water tiles
						if (!(us.tribeType == TribeType.Cymanti && isWaterTerrain(state.tiles[targetCityIndex]))) {
							potentialRewards += 2;
						}
					}
					// If Ruin is not on water
					// Spwans a veteran Swordsman
					if (!isWaterTerrain(state.tiles[targetCityIndex])) {
						potentialRewards += 3;
					}

					// If Ruin is on water
					// Spwans a veteran Rammer (Carries warrior)
					else {
						potentialRewards += 3;
					}
					// Aquarion, if ruin is on ocean tile
					// Spawns a level 3 city with a city wall and 4 adjacent shallow water tiles
					if (us.tribeType == TribeType.Aquarion && state.tiles[targetCityIndex].terrainType == TerrainType.Ocean) {
						potentialRewards += 5;
					}

					potentialRewards *= .5;

					const ruins = state.structures[targetCityIndex];
					state._potentialArmy += potentialRewards;
					delete state.structures[targetCityIndex];

					state._scoreArmy += rewardCapture(state, sieger, CaptureType.Ruins);

					const oldHidden = sieger._hidden;
					sieger._hidden = false;
					sieger._effects.push(EffectType.Invisible);

					return {
						moves: [],
						undo: () => {
							sieger._effects.pop();
							sieger._hidden = oldHidden;
							state._scoreArmy = scoreArmy;
							state.structures[targetCityIndex] = ruins;
							state._potentialArmy -= potentialRewards;
							sieger._attacked = false;
							sieger._moved = false;
						}
					};
				},
			);
		}
		else if (resource && resource.id == ResourceType.Starfish && !isTechLocked(us, TechnologyType.Navigation)) {
			move = new Move(
				MoveType.Capture,
				sieger._tileIndex, 0, CaptureType.Starfish,
				// id: `starfish-${sieger.idx}-${sieger._tileIndex}`,
				(state: GameState) => {
					const tile = state.tiles[sieger._tileIndex];
					us._stars += 8;
					sieger._attacked = true;
					sieger._moved = true;
					const starfish = state.resources[tile.tileIndex];
					delete state.resources[tile.tileIndex];
					let resourceIndex = -1;
					if (tile._owner > 0) {
						resourceIndex = state.tribes[tile._owner]._resources.indexOf(tile.tileIndex);
						state.tribes[tile._owner]._resources.splice(resourceIndex, 1);
					}
					state._scoreArmy += rewardCapture(state, sieger, CaptureType.Starfish);
					const oldHidden = sieger._hidden;
					sieger._hidden = false;
					return {
						moves: [],
						undo: () => {
							sieger._hidden = oldHidden;
							state._scoreArmy = scoreArmy;
							if (resourceIndex > 0) {
								state.tribes[tile._owner]._resources.splice(resourceIndex, 0, tile.tileIndex);
							}
							state.resources[tile.tileIndex] = starfish;
							us._stars -= 8;
							sieger._attacked = false;
							sieger._moved = false;
						}
					};
				},
			);
		}

		if (move && _moves) _moves.push(move);

		return move;
	}

	static riot(state: GameState, unitTarget: UnitState, cityIndex: number, moves?: Move[]): Move[] {
		if (unitTarget._attacked) return [];

		moves = moves || [];

		const tribe = state.tribes[state.settings._pov];
		const cityTile = state.tiles[cityIndex];
		const enemyTribe = state.tribes[cityTile._owner];
		const cityTarget = enemyTribe._cities.find(x => x.tileIndex == cityIndex)!;

		if(!cityTarget || cityTarget._riot) return [];

		moves.push(new Move(
			MoveType.Attack,
			unitTarget._tileIndex, cityIndex, 0,
			(state: GameState) => {
				if(cityTarget._riot) return { moves: [], undo: () => { } };
				
				// It is consumed
				const undoConsume = removeUnit(state, unitTarget);

				// Any enemy unit in the city at the time will be damaged. 
				const enemyTarget = getUnitAtTile(state, cityIndex);

				let undoKillEnemy = () => {};

				// This damage is equivalent to what a unit with an attack of 2 would deal.
				if(enemyTarget) {
					enemyTarget._health -= 2;
					if (enemyTarget._health < 1) {
						const undoRemove = removeUnit(state, enemyTarget);
						tribe._kills++;
						undoKillEnemy = () => {
							tribe._kills--;
							undoRemove();
						};
					}
				}

				// A group of Daggers will spawn in the city's tile. 
				// Daggers will prioritize spawning on terrain whey they can benefit from a defense bonus.
				let defTiles: number[] = [];
				let waterTiles: number[] = [];
				let otherTiles: number[] = [];
				
				cityTarget._territory.forEach(x => {
					const tile = state.tiles[x];
					if(tile._unitOwner > 0 || x == cityIndex) return;
					switch (tile.terrainType) {
						case TerrainType.Mountain:
							if(!isTechLocked(tribe, TechnologyType.Climbing)) {
								defTiles.push(x);
							}
							return
						case TerrainType.Forest:
							if(!isTechLocked(tribe, TechnologyType.Archery)) {
								defTiles.push(x);
							}
							return
						case TerrainType.Water:
							if(!isTechLocked(tribe, TechnologyType.Sailing)) {
								waterTiles.push(x);
							}
							return
					}
					otherTiles.push(x);
					return;
				});
				
				otherTiles.sort(() => Math.random() - 0.5);
				defTiles.sort(() => Math.random() - 0.5);
				waterTiles.sort(() => Math.random() - 0.5);

				// If the enemy died or there wasnt any, then a dagger spawns on the city tile
				if(enemyTarget && enemyTarget._health < 0) {
					// Move to front, guarentee the unit spawns there first
					if(otherTiles.includes(cityIndex)) {
						otherTiles.splice(otherTiles.indexOf(cityIndex), 1);
						otherTiles = [cityIndex, ...otherTiles];
					}
				}

				// They will not be able to perform any actions until the next turn.
				// On water tiles, pirates spawn in stead of daggers
				// A Dagger will only spawn as a Pirate if there are no empty land tiles remaining within the borders of an infiltrated city. 
				// The number of daggers is relative to the city's size, with a max of 5 daggers.

				const daggers: UndoCallback[] = [];

				for (let j = 0; j < Math.min(5, cityTarget._production); j++) {
					let tileIndex = defTiles.pop() || otherTiles.pop();
					let unitType = UnitType.Dagger;
					if(!tileIndex)  {
						// water tile
						tileIndex = waterTiles.pop();
						if(!tileIndex) break;
						unitType = UnitType.Pirate;
					} 
					daggers.push(summonUnit(state, unitType, tileIndex));
				}

				// The infiltrating player will immediately gain a number of stars equal to the income of the city.

				tribe._stars += daggers.length;

				// The city will produce zero stars on their opponent's next turn. 

				cityTarget._riot = true;

				// This does not affect other methods of star production.

				return {
					moves: [],
					undo: () => {
						cityTarget._riot = false;
						tribe._stars -= daggers.length;
						daggers.forEach(x => x());
						undoKillEnemy();
						if(enemyTarget) enemyTarget._health += 2;
						undoConsume();
					}
				};
			}
		));

		return moves;
	}

	static attacks(state: GameState, attacker: UnitState, moves?: Move[]): Move[] {
		if (attacker._attacked) return [];

		moves = moves || [];

		const tribe = state.tribes[attacker?._owner || 1]!;

		// If it can infiltrate, then we can "attack" an enemy city
		if (isSkilledIn(attacker, SkillType.Infiltrate)) {
			// Look for any nearby enemy cities
			const [ targetCityIndex ] = getNeighborIndexes(state, attacker._tileIndex)
				.filter(x => state.tiles[x]._rulingCityIndex > 0 && state.tiles[x]._owner != tribe.owner);

			if(targetCityIndex) {
				UnitMoveGenerator.riot(state, attacker, targetCityIndex, moves);
			}

			return moves;
		}


		const enemiesInRange = getUnitAttack(attacker) === 0? [] : getEnemiesInRange(state, attacker);
		// TODO Allows a unit to convert an enemy unit into a friendly unit by attacking it
		// Converted units take up population in the attacker's city but do not change score for either players
		// const hasConvert = isSkilledIn(unitTarget, SkillType.Convert);

		// Normal attack

		for (let i = 0; i < enemiesInRange.length; i++) {
			const defender = enemiesInRange[i];
			const enemyTribe = state.tribes[defender._owner];
			const enemyTile = state.tiles[defender._tileIndex];

			moves.push(new Move(
				MoveType.Attack,
				attacker._tileIndex, defender._tileIndex, defender._unitType, 
				(state: GameState) => {
					if(!enemyTribe._units.find(x => x.idx == enemyTile._unitIdx)) {
						return Logger.illegal(MoveType.Attack, `Enemy does not exist ${TribeType[enemyTribe.tribeType]}, ${enemyTile._unitIdx}`);
					}

					if(defender._health < 1) {
						return Logger.illegal(MoveType.Attack, `Unit is already dead! ${TribeType[enemyTribe.tribeType]}, ${enemyTile._unitIdx}, ${defender._health}`);
					}

					const oldMoved = attacker._moved;
					const oldAttacked = attacker._attacked;
					const oldScore = state._scoreArmy;

					const undoAttack = attackUnit(state, attacker, defender);

					state._scoreArmy += rewardUnitAttack(state, attacker, defender);

					// Units with Persist skill can keep on killing if they one shot the defender
					if(attacker._health > 0 && isSkilledIn(attacker, SkillType.Persist) && defender._health < 1) {
						attacker._attacked = false;
					}
					else {
						attacker._attacked = true;
					}

					// Units with Escape can move after they attacked
					if (attacker._health > 0 && isSkilledIn(attacker, SkillType.Escape)) {
						attacker._moved = false;
					}
					else {
						attacker._moved = true;
					}

					return {
						moves: [],
						undo: () => {
							attacker._moved = oldMoved;
							attacker._attacked = oldAttacked;
							state._scoreArmy = oldScore;
							undoAttack();
						}
					};
				}
			));
		}

		return moves;
	}

	static spawns(state: GameState, cityTarget?: CityState, unitType?: UnitType): Move[] {
		const moves: Move[] = [];
		const tribe = state.tribes[state.settings._pov];

		if(unitType) {
			const spawn = cityTarget!.tileIndex;
			return [new Move(
				MoveType.Summon,
				spawn, 0, unitType,
				(state: GameState) => {
					const settings = UnitSettings[unitType];
					if (tribe._stars < settings.cost) return { moves: [], undo: () => { } };
					const undo = summonUnit(state, unitType, spawn, true);
					return {
						moves: [],
						undo,
					};
				},
			)];
		}

		const upgradeOptions: UnitType[] = [];

		// Register spawnable units and upgradable units
		const spawnableUnits = (tribe._tech.reduce<UnitType[]>((acc, techType) => {
			const techSettings = TechnologySettings[techType];

			// Its fine to skip cause special units also include a base unlockable unit
			if(!techSettings.unlocksUnit) return acc;
			
			let unitTypes: UnitType[] = [techSettings.unlocksUnit];
			
			// This overrides the base unit
			if(techSettings.unlocksSpecialUnits) {
				const special = techSettings.unlocksSpecialUnits.filter(x => UnitSettings[x].tribeType == tribe.tribeType);
				if(special.length) {
					unitTypes = special;
				}
			}
			
			for(const unitType of unitTypes) {
				// Raft is not purchasable, it requires moving a unit to a port
				if(unitType == UnitType.Raft) continue;

				const settings = UnitSettings[unitType];
				
				// If its purchasable, can afford it and is the same type
				if (!settings.cost 
					|| tribe._stars < settings.cost 
					|| (settings.tribeType && settings.tribeType != tribe.tribeType)
				) {
					continue;
				}
	
				// Is naval upgrade
				if(settings.upgradeFrom || !settings.health) {
					upgradeOptions.push(unitType);
					continue;
				}

				acc.push(unitType);
			}
			
			return acc;
		}, []));

		if (spawnableUnits.length) {
			const cities = cityTarget ? [cityTarget] : tribe._cities;

			for (let i = 0; i < cities.length; i++) {
				// City at max capacity
				// Skip if tile is occupied by some other units
				// Not occupied, double check
				if(cities[i]._unitCount > cities[i]._level
					|| state.tiles[cities[i].tileIndex]._unitOwner > 0
					|| getUnitAtTile(state, cities[i].tileIndex)
				) {
					continue;
				}
	
				for (let j = 0; j < spawnableUnits.length; j++) {
					moves.push(new Summon(cities[i].tileIndex, 0, spawnableUnits[j]))
				}
			}
		}

		if(upgradeOptions.length) {
			// Get rafts ONLY in territory
			const upgradableUnits = tribe._units.reduce<UnitState[]>((acc, cur) => {
				if(cur._unitType != UnitType.Raft) return acc;
				if(state.tiles[cur._tileIndex]._owner != tribe.owner) return acc;
				return [...acc, cur];
			}, []);

			for (let i = 0; i < upgradeOptions.length; i++) {
				const upgradeType = upgradeOptions[i];

				for (let j = 0; j < upgradableUnits.length; j++) {
					const unit = upgradableUnits[j];
					moves.push(new Upgrade(unit._tileIndex, 0, upgradeType))
				}
			}
		}

		return moves;
	}

	static steps(state: GameState, unit: UnitState, moves?: Move[] | null, targetTileIndex?: number): Move[] | null {
		if (unit._moved) return [];
		moves = moves || [];

		const steps = targetTileIndex ? [[targetTileIndex, 1]] : Array.from(UnitMoveGenerator.computeReachableTiles(state, unit).entries());

		for (const [tileIndex, cost] of steps) {
			if (unit._tileIndex == tileIndex) {
				continue;
			}
		
			moves.push(new Step(unit._tileIndex, tileIndex, unit._unitType));
		}

		return moves;
	}

	static stepCallback(state: GameState, unitTarget: UnitState, toTileIndex: number, forced = false): CallbackResult {
		if (!forced && (unitTarget._moved || state.tiles[toTileIndex]._unitOwner > 0 || unitTarget._tileIndex == toTileIndex)) {
			return Logger.illegal(MoveType.Step, `${unitTarget._tileIndex} -> ${toTileIndex}, ${getRealUnitType(unitTarget)} -> ${getRealUnitType(getTrueUnitAtTile(state, toTileIndex)!)} -${forced}-`);
		}

		let chainMoves = undefined;

		// const scoreArmy = state._scoreArmy;

		const iX = unitTarget.x;
		const iY = unitTarget.y;
		const ipX = unitTarget.prevX;
		const ipY = unitTarget.prevY;

		const oldTileIndex = unitTarget._tileIndex;
		const oldMoved = unitTarget._moved;
		const oldAttacked = unitTarget._attacked;
		const oldTile = state.tiles[oldTileIndex];
		const oldType = unitTarget._unitType;
		const oldPassenger = unitTarget._passenger;
		const oldHidden = unitTarget._hidden;

		const newTile = state.tiles[toTileIndex];
		let newType = oldType;

		const oldNewTileUnitOwner = newTile._unitOwner;
		const oldNewTileUnitIdx = newTile._unitIdx;

		// TODO; this is not how prev works, it must be applies at the end of the turn
		unitTarget.prevX = iX;
		unitTarget.prevY = iY;
		unitTarget.x = toTileIndex % state.settings.size;
		unitTarget.y = Math.floor(toTileIndex / state.settings.size);
		unitTarget._tileIndex = toTileIndex;

		oldTile._unitIdx = -1;
		oldTile._unitOwner = -1;
		newTile._unitIdx = unitTarget.idx;
		newTile._unitOwner = unitTarget._owner;

		// Apply movement skills

		// Stomp

		const undoStomp: UndoCallback = splashDamageArea(state, unitTarget, 4);

		// AutoFreeze

		const undoFrozen: UndoCallback = freezeArea(state, unitTarget);

		// Discover terrain

		const preLighthouseCount = state._lighthouses.length;
		const undoDiscover = discoverTiles(state, unitTarget);
		const discoveredLighthouse = state._lighthouses.length != preLighthouseCount;

		let undoDiscoverLighthouse: UndoCallback = () => { };
		if(discoveredLighthouse) {
			const capitalCity = getCapitalCity(state);
			if(capitalCity) {
				const branch = addPopulationToCity(state, capitalCity, 1);
				// TODO Chained moves is disable and algorithm will auto pick
				// chainMoves = branch.chainMoves;
				if(branch.chainMoves) {
					const result2 = AIState.executeBestReward(state, branch.chainMoves)!;
					undoDiscoverLighthouse = () => {
						result2.undo();
						branch.undo();
					};
				}
				else {
					undoDiscoverLighthouse = () => {
						branch.undo();
					};
				}
			}
		}

		unitTarget._moved = unitTarget._attacked = true;

		// Moved to port
		// If ground non floatable or aquatic unit is moving to port, place into boat
		if (state.structures[toTileIndex]?.id == StructureType.Port && !isAquaticOrCanFly(unitTarget)) {
			// Add embark special units
			switch (unitTarget._unitType) {
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
					unitTarget._passenger = oldType;
					break;
			}
		}
		// Carry allows a unit to carry another unit inside
		// A unit with the carry skill can move to a land tile adjacent to water
		// Doing so releases the unit it was carrying and ends the unit's turn
		else if(isSkilledIn(unitTarget, SkillType.Carry) && !isWaterTerrain(newTile)) {
			unitTarget._passenger = undefined;
			// Add disembark special units
			switch (unitTarget._unitType) {
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
		else if(!forced && !oldMoved && isSkilledIn(unitTarget, SkillType.Dash) && getEnemiesInRange(state, unitTarget).length > 0) {
			unitTarget._attacked = false;
		}
		
		// Going stealth mode uses up our attack
		let wasNotInvis = false;
		if(isSkilledIn(unitTarget, SkillType.Hide) && !isInvisible(unitTarget)) {
			unitTarget._hidden = unitTarget._attacked = true;
			unitTarget._effects.push(EffectType.Invisible);
			wasNotInvis = true;
		}

		unitTarget._unitType = newType;
		
		state._scoreArmy += rewardUnitMove(state, unitTarget, ipX, ipY);

		return {
			chainMoves,
			moves: [],
			undo: () => {
				// state._scoreArmy = scoreArmy;
				unitTarget._unitType = oldType;
				if(wasNotInvis) {
					unitTarget._effects.pop();
				}
				unitTarget._hidden = oldHidden;
				unitTarget._passenger = oldPassenger;
				unitTarget._attacked = oldAttacked;
				unitTarget._moved = oldMoved;
				undoDiscoverLighthouse();
				undoDiscover();
				undoFrozen();
				undoStomp();
				newTile._unitOwner = oldNewTileUnitOwner;
				newTile._unitIdx = oldNewTileUnitIdx;
				oldTile._unitIdx = unitTarget.idx;
				oldTile._unitOwner = unitTarget._owner;
				unitTarget._tileIndex = oldTileIndex;
				unitTarget.y = iY;
				unitTarget.x = iX;
				unitTarget.prevX = ipX;
				unitTarget.prevY = ipY;
			}
		};
	}

	/**
	* Returns a map of reachable positions to the cost required to get there.
	* Will return positions with units included standing in the way.
	*/
	static computeReachableTiles(state: GameState, unit: UnitState): Map<number, number> {
		const effectiveMovement = getUnitMovement(unit) + (unit._boosted? 1 : 0);
		
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
	
				for (const tile of neighbors) {
					const index = tile.tileIndex;
					if (index == unit._tileIndex) continue;
	
					if (!isSteppable(state, unit, tile)) continue;
	
					const moveCost = this.computeMovementCost(state, unit, current.index, tile);
					if (moveCost < 0) continue;
	
					let newCost = current.cost + moveCost;
	
					if (newCost - effectiveMovement > 1e-6) continue;
	
					const terminal = this.isTerminalTile(state, unit, tile);
	
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

		// ----- ICE / SKATE -----
		// - Ice: Glide (and/or Skate) skills reduce cost when moving into ice.
		if (toTile.terrainType === TerrainType.Ice && isSkilledIn(unit, SkillType.Skate)) {
			cost *= 0.5; // Skate doubles movement on ice (i.e. halves cost)
		}

		return cost;
	}

	static isTerminalTile(state: GameState, unit: UnitState, tile: TileState): boolean {
		if(isSkilledIn(unit, SkillType.Fly)) {
			return false;
		}

		// ROUGH TERRAIN
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

		// PORTS
		const isPort = state.structures[tile.tileIndex]?.id == StructureType.Port && tile._owner == unit._owner;
		// Entering a Port for non-fly units turns them into Rafts (ending their turn).
		if (isPort) {
			return !isAquaticOrCanFly(unit, false);
		}

		// WATER MOVEMENT / DISEMBARKING
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

		// ZONE-OF-CONTROL
		// Moving adjacent to an enemy stops further movement (unless Creep).
		// The rule about re-entry is ambiguous
		return isSkilledIn(unit, SkillType.Creep) || isAdjacentToEnemy(state, tile);
	}
}