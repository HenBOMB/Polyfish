import { MODEL_CONFIG } from "../aistate";
import Move from "./move";
import { MoveType } from "./types";
import { TerrainType, ResourceType, TribeType, StructureType, UnitType, TechnologyType, RewardType, EffectType, ModeType, ClimateType } from "./types";

export interface DiplomacyRelationState {
	state: boolean;
	lastAttackTurn: number;
	embassyLevel: number;
	lastPeaceBrokenTurn: number;
	firstMeet: number;
	embassyBuildTurn: number;
	previousAttackTurn: number;
}

export interface TileState {
	terrainType: TerrainType;
	_explorers: Set<number>;
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
}

export interface StructureState {
	id: StructureType;
	_level: number;
	turn: number;
	reward: number;
	tileIndex: number;
}

export interface ResourceState {
	id: ResourceType;
	tileIndex: number;
}

export interface UnitState {
	x: number;
	y: number;
	_unitType: UnitType;
	_health: number;
	veteran?: boolean;
	kills: number;
	prevX: number;
	prevY: number;
	direction: number;
	flipped?: boolean;
	createdTurn: number;
	_passenger?: UnitType;
	_owner: number;
	_homeIndex: number;
	_tileIndex: number;
	_moved: boolean;
	_attacked: boolean;
	_effects: Set<EffectType>;
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
	_rewards: Set<RewardType>;
	_territory: number[];
	_walls?: boolean;
	_unitCount: number;
	_riot?: boolean;
}

export interface TribeState {
	owner: number;
	username: string;
	_builtUniqueStructures: Set<StructureType>;
	_knownPlayers: Set<number>;
	bot: boolean;
	_score: number;
	_stars: number;
	tribeType: TribeType;
	_killerId: number;
	_kills: number;
	_casualties: number;
	/** List of all unlcocked TIER `TechnologyType`, not special tech (eg `TechnologyType.ShockTactics`) */
	_tech: TechnologyState[];
	_cities: CityState[];
	_units: UnitState[];
	relations: Record<number, DiplomacyRelationState>;
	_killedTurn: number; 
	_resignedTurn: number;
}


export interface TechnologyState { 
	techType: TechnologyType,
	discovered: boolean,
}

export interface GameState {
	hash: bigint;
	settings: {
		size: number;
		_turn: number;
		maxTurns: number;
		_pov: number;
		areYouSure: boolean;
		unitIdx: number;
		tribeCount: number;
		mode: ModeType;
		_gameOver: boolean;
		_recentMoves: MoveType[];
		_pendingRewards: Move[];
	};
	tiles: TileState[];
	structures: Record<number, StructureState | null>;
	resources: Record<number, ResourceState | null>;
	tribes: Record<number, TribeState>;
	_visibleTiles: Record<number, boolean>;
	_prediction?: PredictionState;
}

export interface GameSettings { 
	size?: number; 
	mode?: ModeType; 
	maxTurns?: number; 
	seed?: number; 
	tribes: TribeType[];
}

export const DefaultGameSettings: Readonly<GameSettings> = {
	size: MODEL_CONFIG.dim_map_size,
	mode: ModeType.Domination,
	maxTurns: MODEL_CONFIG.max_turns,
	seed: undefined,
	tribes: [TribeType.Imperius, TribeType.Bardur],
}

Object.freeze(DefaultGameSettings);

export interface PredictionState {
	_villages?: { [tileIndex: number]: [TribeType, boolean]; };
	_terrain?: { [tileIndex: number]: [TerrainType, ClimateType] };
	_enemyCapitalSuspects?: number[];
	_cityRewards: RewardType[];
}

export interface CombatResult {
	/** Damage dealt by the attacker */
	attackDamage: number;
	/**
	 * Damage dealt by the defender as retaliation.
	 * When defender dies this is 0.
	 */
	defenseDamage: number;
	/**
	 * The splash damage calculated from the attacker’s damage. (float)
	 */
	splashDamage: number;
}

