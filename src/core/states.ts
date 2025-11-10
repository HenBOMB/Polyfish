import Move from "./move";
import { MoveType } from "./types";
import { TerrainType, ResourceType, TribeType, StructureType, UnitType, TechnologyType, RewardType, EffectType, ModeType, ClimateType } from "./types";

/* UUID Byte that represents a player. > 0 */
export type PlayerID = number;

export class Coords {
	x: number;
	y: number;
	idx: number;
	
	constructor(index: number=-1, state: GameState|null=null, x: number=-1, y: number=-1) {
		this.idx = index;
		this.x = x;
		this.y = y;
		if(index > -1 && state) this.setAt(index, state);
	}

	static from(x: number, y: number, state: GameState) {
		const idx = y * state.settings.size + x;
		return new Coords(idx, state);
	}

	set(x: number, y: number, state: GameState) {
		this.x = x;
		this.y = y;
		this.idx = y * state.settings.size + x;
		return this;
	}

	copy(coords: Coords) {
		this.x = coords.x;
		this.y = coords.y;
		this.idx = coords.idx;
		return this;
	}

	setAt(index: number, state: GameState) {
		this.idx = index;
		this.x = index % state.settings.size;
		this.y = Math.floor(index / state.settings.size);
		return this;
	}
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

export interface TileState {
	coords: Coords;
	rulingCityCoords?: Coords;
	type: TerrainType;
	explorers: Set<number>;
	hasRoad: boolean;
	hasRoute: boolean;
	hadRoute: boolean;
	capitalOf: PlayerID;
	skinType: number;
	climate: ClimateType;
	owner: PlayerID;
	_unitOwnerID?: PlayerID;
}

export interface StructureState {
	type: StructureType;
	level: number;
	founded: number;
	score: number;
	tileIndex: number;
}

export interface ResourceState {
	type: ResourceType;
	tileIndex: number;
}

export interface UnitState {
	owner: PlayerID;
	type: UnitType;
	health: number;
	veteran: boolean;
	kills: number;
	coords: Coords;
	prevCoords: Coords;
	homeCoords?: Coords;
	direction: number;
	flipped?: boolean;
	createdTurn: number;
	moved: boolean;
	attacked: boolean;
	passengerType?: UnitType;
	effects: Set<EffectType>;
	_meta?: {
		// TODO properly implement
		converted: boolean;
	}
}

export interface RewardState {
	type: RewardType;
	_name?: string;
}

export interface CityState {
	name: string;
	tileIndex: number;
	population: number;
	progress: number;
	borderSize: number;
	connectedToCapital: boolean;
	level: number;
	production: number;
	owner: PlayerID;
	rewards: Set<RewardType>;
	/**
	 * The cities OUTER territory, city tileIndex where the city resides is not included
	 */
	_territory: number[];
	_walls?: boolean;
	_riot?: boolean;
}

export interface TribeState {
	_hash: bigint;
	id: PlayerID;
	username: string;
	builtUniqueImprovements: Set<StructureType>;
	knownPlayers: Set<PlayerID>;
	bot: boolean;
	score: number;
	stars: number;
	type: TribeType;
	killerId: PlayerID;
	kills: number;
	casualties: number;
	/** List of all unlcocked TIER `TechnologyType`, not special tech (eg `TechnologyType.ShockTactics`) */
	tech_vanilla: TechnologyState[];
	cities: CityState[];
	units: UnitState[];
	relations: Record<PlayerID, DiplomacyRelationState>;
	killedTurn: number; 
	resignedTurn: number;
	startingTileCoords: Coords;
}

export interface TechnologyState { 
	type: TechnologyType,
	discovered: boolean,
}

export interface GameState {
	settings: {
		mode: ModeType;
		size: number;
		tileCount: number;
		turn: number;
		maxTurns: number;
		currentPlayerTurnId: number;
		version?: number;
		gameName?: string;
		seed?: number;
		winByCapital?: boolean;
		winByExtermination?: boolean;
		_lastPlayerTurnId: number;
		_areYouSure: boolean;
		_gameOver: boolean;
		_recentMoves: MoveType[];
		_pendingRewards: Move[];
		_fow?: boolean;
		_maxTribeCount: number;
	};
	tiles: Record<number, TileState>;
	structures: Record<number, StructureState | null>;
	resources: Record<number, ResourceState | null>;
	tribes: Record<number, TribeState>;
	_visibleTiles: Record<number, boolean>;
	_prediction?: PredictionState;
}

export interface GameSettings { 
	size: number; 
	mode: ModeType; 
	maxTurns: number; 
	tribes: TribeType[];
	fow: boolean;
}

export interface PartialGameSettings { 
	size?: number; 
	mode?: ModeType; 
	maxTurns?: number; 
	fallback?: string; 
	tribes?: TribeType[];
	fow?: boolean;
	seed?: number;
}

export const DefaultGameSettings: Readonly<GameSettings> = {
	size: 11,
	mode: ModeType.Domination,
	maxTurns: 30,
	tribes: [TribeType.Imperius, TribeType.Bardur],
	fow: true
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

