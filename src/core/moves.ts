import { CityState, GameState, TileState, UnitState } from "./states";
import { getNeighborTiles, getPovTerritorry, getNeighborIndexes, isAdjacentToEnemy, isAquaticOrCanFly, isSteppable, isWaterTerrain, isNavalUnit, getTechCost, getPovTribe, isSkilledIn, getUnitAttack, getUnitMovement, isRoadpathAndUsable, getMaxHealth, getAlliesNearTile, isTechUnlocked, isEnemyCity, isUnderSiege, getTechSettings, getTechUnitType, isTileOccupied, getEnemiesNearTile, isTileFrozen, getTechStructure, getCityOwningTile, isResourceVisible, getEnemyIndexesInRange, getReplacedOrTechSettings, isTileExplored, hasEffect, getStructureAt, getUnitSettings, getUnitAt } from './functions';
import { StructureSettings } from "./settings/StructureSettings";
import { SkillType, ResourceType, RewardType, StructureType, TechnologyType, TerrainType, UnitType, AbilityType, EffectType } from "./types";
import { UnitSettings } from "./settings/UnitSettings";
import Move, { Action } from "./move";
import { MoveType } from "./types";
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
import BreakIce from "./moves/abilities/BreakIce";
import Drain from "./moves/abilities/Drain";

export type Prediction = { 
	pi_action: number[]; 
	pi_source: number[]; 
	pi_target: number[]; 
	pi_struct: number[]; 
	pi_skill: number[]; 
	pi_unit: number[]; 
	pi_tech: number[]; 
	pi_reward: number[]; 
	v_win: number 
	// v_eco: number 
	// v_mil: number 
};

interface ReachableNode {
	index: number;
	cost: number;
	terminal?: boolean;
}

const UnitToSpawnableID: Record<UnitType, number> = {
	[UnitType.None]: 		-1,
	[UnitType.Dinghy]: 		-1,
	[UnitType.Pirate]: 		-1,
	[UnitType.Giant]: 		-1,
	[UnitType.DragonEgg]: 	-1,
	[UnitType.BabyDragon]: 	-1,
	[UnitType.FireDragon]: 	-1,
	[UnitType.Crab]: 		-1,
	[UnitType.Centipede]: 	-1,
	[UnitType.Segment]: 	-1,
	[UnitType.Gaami]: 		-1,
	[UnitType.Dagger]: 		-1,
	[UnitType.Raft]: 		-1,
	[UnitType.Juggernaut]: 	-1,

	[UnitType.Warrior]:		0,
	[UnitType.Rider]: 	    1,
	[UnitType.Amphibian]:   2,
	[UnitType.Hexapod]: 	3,

	[UnitType.Knight]: 	    4,
	[UnitType.Tridention]:  5,
	[UnitType.Doomux]: 		6,
	
	[UnitType.Defender]: 	7,
	[UnitType.Cloak]: 	 	8,
	[UnitType.Kiton]: 		9,

	[UnitType.Swordsman]: 	10,

	[UnitType.MindBender]: 	11,
	[UnitType.Shaman]: 		12,

	[UnitType.Archer]: 		13,
	[UnitType.IceArcher]: 	14,
	[UnitType.Phychi]: 		15,
	[UnitType.Catapult]: 	16,
	[UnitType.Exida]: 		17,
	
	[UnitType.Raychi]: 		18,
	[UnitType.Scout]: 		19,
	[UnitType.Bomber]: 		20,
	[UnitType.Rammer]: 		21,
	
	[UnitType.Polytaur]: 	22,
	[UnitType.Mooni]: 		23,
	[UnitType.IceFortress]: 24,
	[UnitType.BattleSled]: 	25,
}

const StructToBuildableID: Record<StructureType, number> = {
	[StructureType.None]: 			-1,
	[StructureType.Village]: 		-1,
	[StructureType.Ruin]: 			-1,
	[StructureType.Lighthouse]: 	-1,
	
	[StructureType.Road]: 			0,
	[StructureType.Bridge]: 		1,
	[StructureType.Market]: 		2,
	[StructureType.Outpost]: 		3,

	[StructureType.Farm]: 			4,
	[StructureType.Windmill]: 		5,
	[StructureType.Embassy]: 		6,
	[StructureType.Temple]: 		7,

	[StructureType.Mine]: 			8,
	[StructureType.Forge]: 			9,
	[StructureType.MountainTemple]: 10,

	[StructureType.Port]: 			11,
	[StructureType.WaterTemple]: 	12,
	[StructureType.IceTemple]: 		13,

	[StructureType.LumberHut]: 		14,
	[StructureType.Sawmill]: 		15,
	[StructureType.ForestTemple]: 	16,

	[StructureType.AltarOfPeace]: 	17,
	[StructureType.TowerOfWisdom]: 	18,
	[StructureType.GrandBazaar]: 	19,
	[StructureType.EmperorsTomb]: 	20,
	[StructureType.GateOfPower]: 	21,
	[StructureType.ParkOfFortune]: 	22,
	[StructureType.EyeOfGod]: 		23,
	
	[StructureType.Spores]: 		24,
	[StructureType.Swamp]: 			25,
	[StructureType.Mycelium]: 		26,
}

const TechToResearchableID: Record<TechnologyType, number> = {
	[TechnologyType.None]:			-1,
	[TechnologyType.Unbuildable]: 	-1,

	[TechnologyType.Riding]: 		0,
	[TechnologyType.Amphibian]: 	0,
	[TechnologyType.Roads]: 		1,
	[TechnologyType.Trade]: 		2,
	[TechnologyType.FreeSpirit]: 	3,
	[TechnologyType.FreeDiving]: 	3,
	[TechnologyType.Chivalry]: 		4,
	[TechnologyType.Spearing]: 		4,
	[TechnologyType.ShockTactics]: 	4,

	[TechnologyType.Organization]: 	5,
	[TechnologyType.Farming]: 		6,
	[TechnologyType.Construction]: 	7,
	[TechnologyType.Recycling]: 	7,
	[TechnologyType.Strategy]: 		8,
	[TechnologyType.Diplomacy]: 	9,
	
	[TechnologyType.Climbing]: 		10,
	[TechnologyType.Mining]: 		11,
	[TechnologyType.Smithery]: 		12,
	[TechnologyType.Meditation]: 	13,
	[TechnologyType.Philosophy]: 	14,

	[TechnologyType.Fishing]: 		15,
	[TechnologyType.IceFishing]: 	15,
	[TechnologyType.Frostwork]: 	15,
	[TechnologyType.Sailing]: 		16,
	[TechnologyType.Sledding]: 		16,
	[TechnologyType.Pascetism]: 	16,
	[TechnologyType.Navigation]: 	17,
	[TechnologyType.PolarWarfare]: 	17,
	[TechnologyType.Oceantology]: 	17,
	[TechnologyType.Ramming]: 		18,
	[TechnologyType.Hydrology]: 	18,
	[TechnologyType.Aquatism]: 		19,
	[TechnologyType.Polarism]: 		19,

	[TechnologyType.Hunting]: 		20,
	[TechnologyType.Archery]: 		21,
	[TechnologyType.Spiritualism]: 	22,
	[TechnologyType.Forestry]: 		23,
	[TechnologyType.ForestMagic]: 	23,
	[TechnologyType.Mathematics]: 	24,
}

const RewardToRewardableID: Record<RewardType, number> = {
	[RewardType.None]: 			-1,
	[RewardType.Workshop]: 		0,
	[RewardType.Explorer]: 		1,
	[RewardType.Resources]: 	0,
	[RewardType.CityWall]: 		1,
	[RewardType.PopGrowth]: 	0,
	[RewardType.BorderGrowth]:	1,
	[RewardType.Park]: 			0,
	[RewardType.SuperUnit]: 	1,
}

export class MoveGenerator {
	static actionToMove(action: Action): Move {
		switch (action.action) {
			case MoveType.Attack:
				return new Attack(action.from!, action.to!);
			case MoveType.Step:
				return new Step(action.from!, action.to!);
			case MoveType.Summon:
				return new Summon(action.from!, action.unit!);
			case MoveType.Research:
				return new Research(action.tech!);
			case MoveType.Harvest:
				return new Harvest(action.to!);
			case MoveType.Build:
				return new Structure(action.to!, action.struct!);
			case MoveType.Reward:
				return new Reward(action.from!, action.reward!);
			case MoveType.Capture:
				return new Capture(action.from!);
			case MoveType.Ability:
				switch (action.ability) {
					case AbilityType.BurnForest:
						return new BurnForest(action.to!);
					case AbilityType.ClearForest:
						return new ClearForest(action.to!);
					case AbilityType.GrowForest:
						return new GrowForest(action.to!);
					case AbilityType.Destroy:
						return new Destroy(action.to!);
					case AbilityType.Decompose:
						return new Decompose(action.to!);
					case AbilityType.Recover:
						return new Recover(action.from!);
					case AbilityType.Disband:
						return new Disband(action.from!);
					case AbilityType.HealOthers:
						return new HealOthers(action.from!);
					case AbilityType.BreakIce:
						return new BreakIce(action.to!);
					case AbilityType.Drain:
						return new Drain(action.to!);
					case AbilityType.FreezeArea:
						return new FreezeArea(action.from!);
					case AbilityType.Boost:
						return new Boost(action.from!);
					case AbilityType.Explode:
						return new Explode(action.from!);
					case AbilityType.Promote:
						return new Promote(action.from!);
				}
			case MoveType.EndTurn:
				return new EndTurn();
			default:
				throw new Error(`MoveType ${action.action} not implemented`);
		}
	}

	static fromActions(actions: Action[]): Move[] {
		return actions.map(x => MoveGenerator.actionToMove(x));
	}

	static transpose: Map<string, Move[]> = new Map();

	static legal(state: GameState): Move[] {
		if(state.settings._pendingRewards.length) {
			return state.settings._pendingRewards;
		}
		const moves: Move[] = [new EndTurn()];

		ArmyMovesGenerator.all(state, moves);

		EconMovesGenerator.all(state, moves);
		
		return moves;
	}

	static serialize(moves: Move[]): string {
		return moves.map(Move.serialize).join('#');
	}

	static fromPrediction(state: GameState, prediction: Prediction, legal?: Move[]): [Move, number, number][] {
		const {
            pi_action, pi_source: pi_actor, pi_target,
            pi_struct, pi_skill, pi_unit,
            pi_tech, pi_reward,
            // v_win
        } = prediction;
		
		return (legal || MoveGenerator.legal(state)).map((move, i) => {
			let prob = pi_action[move.moveType-1];
			
			if(move.hasSrc()) {
				prob *= pi_actor[move.getSrc()];
			}

			if(move.hasTarget()) {
				prob *= pi_target[move.getTarget()];
			}
	
			switch (move.moveType) {
				case MoveType.Ability:
					prob *= pi_skill[move.getType<AbilityType>()];
					break;
				
				case MoveType.Summon:
					prob *= pi_unit[UnitToSpawnableID[move.getType<UnitType>()]];
					break;
	
				case MoveType.Build:
					prob *= pi_struct[StructToBuildableID[move.getType<StructureType>()]];
					break;
	
				case MoveType.Research:
					prob *= pi_tech[TechToResearchableID[move.getType<TechnologyType>()]];
					break;
	
				case MoveType.Reward:
					const rewardChoiceType = RewardToRewardableID[move.getType<RewardType>()];
					if(rewardChoiceType === 0) {
						prob *= (1 - pi_reward[0]);
					} 
					else if (rewardChoiceType === 1) {
						prob *= pi_reward[0];
					}
					break;
	
				default:
					break;
			}

			return [move, prob, i];
		}).sort((a: any, b: any) => b[1] - a[1]) as any;
	}
}

export class EconMovesGenerator {
	static all(state: GameState, moves: Move[]) {
		// EconMovesGenerator.actions(state, moves);
		// EconMovesGenerator.resources(state, moves);
		// EconMovesGenerator.structures(state, moves);
		// EconMovesGenerator.research(state, moves);
		EconMovesGenerator.all_fast(state, moves);
	}

	static all_fast(state: GameState, moves: Move[]) {
		const pov = getPovTribe(state);
		const territory = getPovTerritorry(state);

		const abilities: Set<AbilityType> = new Set();
		const structures: Set<StructureType> = new Set();

		pov._tech.forEach(x => {
			const settings = getTechSettings(x);

			// Tasks
			if(settings.unlocksTask) {
				settings.unlocksTask.forEach(y => {
					const settings = TaskSettings[y];
					if(!pov._builtUniqueStructures.has(settings.structureType)) {
						if(settings.task(state)) {
							structures.add(settings.structureType);
						}
					}
				})
			}

			const realSettings = getReplacedOrTechSettings(pov, x);

			// Actions
			if(realSettings.unlocksAbility) {
				if(!settings.explicitCost || settings.explicitCost <= pov._stars) {
					abilities.add(realSettings.unlocksAbility);
				}
			}

			// Structures
			if(realSettings.unlocksStructure && pov._stars >= (StructureSettings[realSettings.unlocksStructure].cost || 0)) {
				structures.add(realSettings.unlocksStructure);
			}

			// Other tech
			if(settings.next) {
				for (let j = 0; j < settings.next.length; j++) {
					if(isTechUnlocked(pov, settings.next[j])) {
						continue;
					}
	
					const cost = getTechCost(pov, settings.next[j]);
	
					if(cost > pov._stars) {
						continue;
					}
	
					moves.push(new Research(settings.next[j]));
				}
			}
		});

 		for(let i = 0; i < territory.length; i++) {
			const tileIndex = territory[i];
			const enemyOnTile = isTileOccupied(state, tileIndex, true);
			
			if(enemyOnTile) {
				continue;
			}

			if(!isTileExplored(state, tileIndex)) {
				continue
			}

			const tile = state.tiles[tileIndex];
			const resource = state.resources[tileIndex];
			const structure = state.structures[tileIndex];
			
			// ! Harvesting ! //

			if(resource) {
				const settings = ResourceSettings[resource!.id];
				if((settings.cost || 0) <= pov._stars
					&& !settings.structType
					&& !structure
					&& !enemyOnTile
					&& isTechUnlocked(pov, settings.techRequired)
				) {
					moves.push(new Harvest(tileIndex));
				}
			}

			// ! Building ! //

			if(!structure) {
				for(const x in structures) {
					const structType = Number(x) as StructureType;
					const settings = StructureSettings[structType];
	
					if(!settings.terrainType?.has(tile.terrainType)) {
						continue;
					}
					
					if(settings.limitedPerCity) {
						const limited = getCityOwningTile(state, tile.tileIndex)!._territory.some(x => state.structures[x]?.id == structType);
						if(limited) {
							continue;
						}
					}
	
					if(settings.adjacentTypes && !getNeighborIndexes(state, tile.tileIndex).some(x => state.structures[x]? 
						settings.adjacentTypes!.has(state.structures[x]!.id) : false
					)) {
						continue;
					}
		
					moves.push(new Structure(
						tile.tileIndex,
						structType
					));
				}
			}

			// ! Abilities ! //

			if(structure) {
				// TODO Embassy
				// if(tile.capitalOf > 0 && tile.capitalOf !== pov.owner && structType !== StructureType.Embassy) {
				// }
				if(StructureSettings[state.structures[tile.tileIndex]!.id].cost) {
					if(abilities.has(AbilityType.Destroy)) {
						moves.push(new Destroy(tile.tileIndex));	
					}
					else if(abilities.has(AbilityType.Decompose)) {
						moves.push(new Decompose(tile.tileIndex));	
					}
				}
			}
			else if(tile.terrainType === TerrainType.Forest) {
				if(abilities.has(AbilityType.ClearForest)) {
					moves.push(new ClearForest(tile.tileIndex));
				}
				if(abilities.has(AbilityType.BurnForest)) {
					moves.push(new BurnForest(tile.tileIndex));
				}
			}
			else if(tile.terrainType == TerrainType.Field) {
				if(abilities.has(AbilityType.GrowForest)) {
					moves.push(new GrowForest(tile.tileIndex));
				}
			}

			// TODO should use internal ice boolean
			// Same with Drain
			// if(tile.terrainType === TerrainType.Ice) {
			// }
		}
	}

	static actions(state: GameState, moves: Move[]) {
		const pov = getPovTribe(state);
		const territory = getPovTerritorry(state);
		const abilities = new Set(pov._tech.filter(x => {
			const settings = getTechSettings(x);
			if(settings.explicitCost && settings.explicitCost > pov._stars) {
				return false;
			}
			return settings.unlocksAbility;
		}).map(x => getTechSettings(x).unlocksAbility));

		for(let i = 0; i < territory.length; i++) {
			const tile = state.tiles[territory[i]];

			if(state.structures[tile.tileIndex]) {
				if(StructureSettings[state.structures[tile.tileIndex]!.id].cost) {
					if(abilities.has(AbilityType.Destroy)) {
						moves.push(new Destroy(tile.tileIndex));	
					}
					else if(abilities.has(AbilityType.Decompose)) {
						moves.push(new Decompose(tile.tileIndex));	
					}
				}
			}
			else if(tile.terrainType === TerrainType.Forest) {
				if(abilities.has(AbilityType.ClearForest)) {
					moves.push(new ClearForest(tile.tileIndex));
				}
				if(abilities.has(AbilityType.BurnForest)) {
					moves.push(new BurnForest(tile.tileIndex));
				}
			}
			else if(tile.terrainType == TerrainType.Field) {
				if(abilities.has(AbilityType.GrowForest)) {
					moves.push(new GrowForest(tile.tileIndex));
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

		for(let i = 0; i < territory.length; i++) {
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
		const structTypes: Set<StructureType> = new Set();

		for(let i = 0; i < 7; i++) {
			const task = TaskSettings[i as keyof typeof TaskSettings];
			if(task.techType && !isTechUnlocked(pov, task.techType)) {
				continue;
			}
			if(pov._builtUniqueStructures.has(task.structureType)) {
				continue;
			}
			if(task.task(state)) {
				structTypes.add(task.structureType);
			}
		}

		for(let i = 0; i < pov._tech.length; i++) {
			const structType = getTechStructure(pov, pov._tech[i]);
	
			if(!structType || pov._stars < getTechCost(pov, pov._tech[i])) {
				continue
			}
	
			structTypes.add(structType);
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

			for(const x in structTypes) {
				const structType = Number(x) as StructureType;
				const settings = StructureSettings[structType];

				if(!settings.terrainType?.has(tile.terrainType)) {
					if(tile.capitalOf !== pov.owner || structType !== StructureType.Embassy) {
						continue;
					}
				}
				
				if(settings.limitedPerCity) {
					const limited = getCityOwningTile(state, tile.tileIndex)!._territory.some(x => state.structures[x]?.id == structType);
					if(limited) {
						continue;
					}
				}

				if(settings.adjacentTypes && !getNeighborIndexes(state, tile.tileIndex).some(x => state.structures[x]? 
					settings.adjacentTypes!.has(state.structures[x]!.id) : false
				)) {
					continue;
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

				const cost = getTechCost(pov, next[j]);

				if(cost > pov._stars) {
					continue;
				}

				moves.push(new Research(next[j]));
			}
		}
	}
	
	static rewards(city: CityState): Move[] {
		const rewards = [
			[ RewardType.Workshop, RewardType.Explorer ],
			[ RewardType.CityWall, RewardType.Resources ],
			[ RewardType.PopGrowth, RewardType.BorderGrowth ],
		][city._level-2] || [ RewardType.Park, RewardType.SuperUnit ];
		if(rewards.some(x => city._rewards.has(x))) {
			return [];
		}
		return rewards.map(rewardType => new Reward(city.tileIndex, rewardType));
	}
}

export class ArmyMovesGenerator {
	static all(state: GameState, moves: Move[]) {
		getPovTribe(state)._units.forEach(x => {
			ArmyMovesGenerator.captures(state, x, moves);
			ArmyMovesGenerator.actions(state, x, moves);
			ArmyMovesGenerator.attacks(state, x, moves);
			ArmyMovesGenerator.steps(state, x, moves);
		});
		ArmyMovesGenerator.summons(state, moves);
	}

	static actions(state: GameState, unit: UnitState, _moves: Move[]) {
		const tileIndex = unit._tileIndex;

		// Promote
		if(!unit._veteran && unit._kills >= 3) {
			_moves.push(new Promote(tileIndex));
		}

		// Explode
		if(!unit._attacked && isSkilledIn(unit, SkillType.Explode)) {
			const enemiesAround = getEnemiesNearTile(state, tileIndex, 1, true).length;
			if(enemiesAround) {
				_moves.push(new Explode(tileIndex));
			}
		}

		if(unit._moved || unit._attacked) {
			return [];
		}

		// Disband
		if(isTechUnlocked(getPovTribe(state), TechnologyType.FreeSpirit)) {
			_moves.push(new Disband(tileIndex));
		}

		// Recover
		if(unit._health < getMaxHealth(unit)) {
			_moves.push(new Recover(tileIndex));
		}

		if(isSkilledIn(unit, SkillType.Heal, SkillType.Boost, SkillType.FreezeArea)) {
			// Heal Others
			if(isSkilledIn(unit, SkillType.Heal)) {
				const damagedAround = getAlliesNearTile(state, tileIndex).some(x => x._health < getMaxHealth(x));
				if(damagedAround) {
					_moves.push(new HealOthers(tileIndex));
				}
			}
	
			// Boost
			else if(isSkilledIn(unit, SkillType.Boost)) {
				const unboostedAround = getAlliesNearTile(state, tileIndex).some(x => !hasEffect(x, EffectType.Boost));
				if(unboostedAround) {
					_moves.push(new Boost(tileIndex));
				}
			}
	
			// Freeze Area
			else if(isSkilledIn(unit, SkillType.FreezeArea)) {
				const stuffAround = getEnemiesNearTile(state, tileIndex, 1, true).some(x => !hasEffect(x, EffectType.Frozen) || !isTileFrozen(state, x._tileIndex));
				if(stuffAround) {
					_moves.push(new FreezeArea(tileIndex));
				}
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
			switch (struct.id) {
				case StructureType.Village:
					if(!state.tiles[targetCityIndex]._owner) {
						moves.push(new Capture(capturer._tileIndex));
					}
					else if(state.tiles[targetCityIndex]._owner !== capturer._owner) {
						moves.push(new Capture(capturer._tileIndex));
					}
					break;

				case StructureType.Ruin:
					moves.push(new Capture(capturer._tileIndex));
					break;
			
				default:
					break;
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
				...getEnemyIndexesInRange(state, attacker).map(x => new Attack(attacker._tileIndex, x))
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

		if(spawnables.length) {
			const cities = tribe._cities;
			for(let i = 0; i < cities.length; i++) {
				const targetIndex = cities[i].tileIndex;
				if(cities[i]._unitCount > cities[i]._level || isTileOccupied(state, targetIndex)) {
					continue;
				}
				for(let j = 0; j < spawnables.length; j++) {
					const unitType = spawnables[j];
					// If the unit has navigate, it tipically cannot move onto land
					// allow spawning if the unit has at least 1 tile to move to
					if(isSkilledIn(unitType, SkillType.Navigate)) {
						const hasWater = getNeighborIndexes(state, targetIndex).some(x => isWaterTerrain(state.tiles[x]));
						if(!hasWater) {
							continue;
						}
					}
					moves.push(new Summon(targetIndex, unitType))
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
		if(unit._moved) return;

		const steps = Array.from(ArmyMovesGenerator.computeReachableTiles(state, unit).entries());

		for(const [tileIndex, ] of steps) {
			if(unit._tileIndex == tileIndex) {
				continue;
			}
			moves.push(new Step(unit._tileIndex, tileIndex));
		}
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

		while(openList.length > 0) {
			openList.sort((a, b) => a.cost - b.cost);
			const current = openList.shift()!;
	
			if (current.terminal) {
				continue;
			}
	
			const neighbors = getNeighborTiles(state, current.index);
	
			for(let i = 0; i < neighbors.length; i++) {
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
		if(isSkilledIn(unit, SkillType.Fly, SkillType.Creep)) {
			return cost;
		}

		// Road bonus
		if(isRoadpathAndUsable(state, unit, fromIndex) && isRoadpathAndUsable(state, unit, toTile.tileIndex)) {
			cost = 0.5;
		}

		// Skate doubles movement on ice (i.e. halves cost)
		if(toTile.terrainType === TerrainType.Ice && isSkilledIn(unit, SkillType.Skate)) {
			cost *= 0.5; 
		}

		return cost;
	}

	// TODO verify logic
	static isTerminal(state: GameState, unit: UnitState, tile: TileState): boolean {
		if(isSkilledIn(unit, SkillType.Fly)) {
			return false;
		}

		// Embark
		const isPort = getStructureAt(state, tile.tileIndex) == StructureType.Port && tile._owner == unit._owner;
		if(isPort && !isAquaticOrCanFly(unit, false)) {
			return true;
		}

		// All enemy units excert a Zone of Control and stops further movement
		if(isAdjacentToEnemy(state, tile)) {
			return true;
		}
	
		if(isWaterTerrain(tile)) {
			// Navigate on to village
			if(isSkilledIn(unit, SkillType.Navigate) && getStructureAt(state, tile.tileIndex) === StructureType.Village) {
				return true;
			}

			// this should return cause naval or aquatic units dont suffer zoc?
			// plus tehre is no need to check for forest or mountain
		}
		else {
			// Disembark
			if(isNavalUnit(unit)) {
				return true;
			}
		}

		// mountains and forests blocks further movement
		// roads allow movement on forest if it has a road
		switch (tile.terrainType) {
			case TerrainType.Forest:
				if(!isSkilledIn(unit, SkillType.Creep) || !isRoadpathAndUsable(state, unit, tile.tileIndex)) {
					return true;
				}
				break;
			case TerrainType.Mountain:
				if(!isSkilledIn(unit, SkillType.Creep)) {
					return true;
				}
				break;
		}

		return false;
	}
}
