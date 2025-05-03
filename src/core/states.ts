import { MODEL_CONFIG } from "../aistate";
import { MoveType } from "./move";
import { TerrainType, ResourceType, TribeType, StructureType, UnitType, TechnologyType, RewardType, EffectType, ModeType, ClimateType } from "./types";

export interface TileState {
	terrainType: TerrainType;
	explorers: number[];
	hasRoad: boolean;
	hasRoute: boolean;
	hadRoute: boolean;
	capitalOf: number;
	skinType: number;
	climate: ClimateType;
	x: number;
	y: number;
	_rulingCityIndex: number;
	_owner: number;
	tileIndex: number;
	_unitOwner: number;
	_unitIdx: number;
}

export interface StructureState {
	id: StructureType;
	_level: number;
	turn: number;
	reward: number;
	tileIndex: number;
	_name: string;
	_owner: number;
	_potentialTerritory?: number[];
}

export interface ResourceState {
	id: ResourceType;
	tileIndex: number;
	_owner: number;
}

export interface AbilityState {
	id: number;
	name: string;
}

export interface UnitClass {
	name: string;
	id: number;
	health: number;
	defense: number;
	movement: number;
	attack: number;
	cost: number;
	hidden: boolean;
	weapon: number;
	range: number;
	abilities: AbilityState[];
}

export interface UnitState {
	idx: number;
	x: number;
	y: number;
	_unitType: UnitType;
	_health: number;
	veteran?: boolean;
	kills: number;
	// class: UnitClass;
	prevX: number;
	prevY: number;
	direction: number;
	flipped?: boolean;
	createdTurn: number;
	_passenger?: UnitType;
	_owner: number;
	_homeIndex: number;
	_tileIndex: number;
	_boosted?: boolean;
	_hidden?: boolean;
	_moved: boolean;
	_attacked: boolean;
	_effects: EffectType[];
}

export interface RewardState {
	id: RewardType;
	_name?: string;
}

export interface CityState {
	name: string;
	tileIndex: number;
	_population: number;
	_progress: number;
	_borderSize: number;
	_connectedToCapital: boolean;
	_level: number;
	_production: number;
	_owner: number;
	_rewards: RewardType[];
	_territory: number[];
	_walls?: boolean;
	_unitCount: number;
	_riot?: boolean;
}

export interface TaskState {
	started: boolean;
	completed: boolean;
	customData: number;
}

export interface AbilityState {
	id: number;
	name: string;
}

export interface DiplomacyRelationState {
	state: boolean;
	lastAttackTurn: number;
	embassyLevel: number;
	lastPeaceBrokenTurn: number;
	firstMeet: number;
	embassyBuildTurn: number;
	previousAttackTurn: number;
}

export interface TribeState {
	owner: number;
	username: string;
	tasks: TaskState[];
	_builtUniqueStructures: StructureType[];
	_knownPlayers: number[];
	bot: boolean;
	_score: number;
	_stars: number;
	tribeType: TribeType;
	_killerId: number;
	_kills: number;
	_tech: TechnologyType[];
	_cities: CityState[];
	_units: UnitState[];
	_resources: number[],
	_trueTech: TechnologyType[];
	relations: Record<number, DiplomacyRelationState>;
	_killedTurn: number; 
	_resignedTurn: number;
}

export interface GameSettings { 
	size?: number, 
	mode?: ModeType, 
	maxTurns?: number, 
	seed?: number, 
	tribes: TribeType[] 
}

export const DefaultGameSettings: Readonly<GameSettings> = {
	size: MODEL_CONFIG.max_size,
	mode: ModeType.Domination,
	maxTurns: MODEL_CONFIG.max_turns,
	seed: undefined,
	tribes: [TribeType.Imperius, TribeType.Bardur],
}

Object.freeze(DefaultGameSettings);

export interface GameState {
	settings: {
		size: number;
		_turn: number;
		maxTurns: number;
		_pov: number;
		live: boolean;
		unitIdx: number;
		tribeCount: number,
		mode: ModeType,
		_gameOver: boolean,
		_recentMoves: MoveType[],
	};
	tiles: Record<number, TileState>;
	structures: Record<number, StructureState | null>;
	resources: Record<number, ResourceState | null>;
	tribes: Record<number, TribeState>;
	_potentialDiscovery: number[];
	_potentialArmy: number;
	_potentialTech: number;
	_potentialEconomy: number;
	_visibleTiles: number[];
	_scoreArmy: number;
	_scoreTech: number;
	_scoreEconomy: number;
	__: number;
	___: number;
	/** Undiscovered lighthouses, tileIndex[] */
	_lighthouses: number[];
	_prediction: PredictionState;
}

export interface PredictionState {
	_villages?: { [tileIndex: number]: [TribeType, boolean]; };
	_terrain?: { [tileIndex: number]: [TerrainType, ClimateType] };
	_enemyCapitalSuspects?: number[];
	_cityRewards: RewardType[];
}