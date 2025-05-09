import { getCityAt, getEnemyAt, getUnitAt } from "./functions";
import { GameState } from "./states";
import { AbilityType, CaptureType, ArmyAbilityTypes, MoveType, ResourceType, RewardType, EconomyAbilityTypes, StructureType, TechnologyType, TerrainType, TribeType, UnitType } from "./types";

export type UndoCallback = () => void;

export interface Branch {
	rewards: Move[];
	undo: UndoCallback;
}

export interface Action {
    action: MoveType;
    to: number | null;
    from: number | null;
    struct: StructureType | null;
    unit: UnitType | null;
    tech: TechnologyType | null;
    ability: AbilityType | null;
    reward: RewardType | null;
}

export type CallbackResult = Branch | null;

export type CallbackExecution = (state: GameState, ...args: (number | null)[]) => CallbackResult;

export default class Move {
    readonly moveType: MoveType;
    private src: number | null;
    private target: number | null;
    private type: number | null;
    readonly costs: number | null;

    constructor(id: MoveType, src: number | null = null, target: number | null = null, type: number | null = null, costs: number | null = null) {
        this.moveType = id;
        this.src = src && src < 0? null : src;
        this.target = target && target < 0? null : target;
        this.type = type && type < 0? null : type;
        this.costs = costs && costs < 0? null : costs;
    }

    execute(_: GameState): CallbackResult {
        throw "execute(state) Not implemented";
    }

    stringify(oldState: GameState, newState: GameState) {
        switch (this.moveType) {
            case MoveType.Step: {
                return `${MoveType[this.moveType]} ${UnitType[(getUnitAt(oldState, this.getSrc()) || getUnitAt(newState, this.getSrc()))!._unitType]}`;
            }
            case MoveType.Attack: {
                const oldAtk = getUnitAt(oldState, this.getSrc())!;
                const oldDef = getEnemyAt(oldState, this.getTarget())!;
                const who = `${UnitType[oldAtk._unitType]} -> ${UnitType[oldDef._unitType]}`;
                const what = !getUnitAt(newState, this.getSrc())? 'suicide' : getEnemyAt(newState, this.getTarget())? 'kill' : 'hit';
                return `${MoveType[this.moveType]} ${what} ${who}`;
            }
            case MoveType.Summon: {
                const city = getCityAt(oldState, this.getSrc())!;
                return `${MoveType[this.moveType]} ${UnitType[this.getType<UnitType>()]} at ${city.name}`;
            }
            case MoveType.Research: {
                return `${MoveType[this.moveType]} ${TechnologyType[this.getType<TechnologyType>()]}`;
            }
            case MoveType.Harvest: {
                const resource = oldState.resources[this.getTarget()]!;
                return `${MoveType[this.moveType]} ${ResourceType[resource.id]}`;
            }
            case MoveType.Build: {
                return `${MoveType[this.moveType]} ${StructureType[this.getType<StructureType>()]}`;
            }
            case MoveType.Reward: {
                return `${MoveType[this.moveType]} pick ${RewardType[this.getType<RewardType>()]}`;
            }
            case MoveType.EndTurn: {
                return `${MoveType[this.moveType]}`;
            }
            case MoveType.Capture: {
                const struct = oldState.structures[this.getSrc()];
                const tile = oldState.tiles[this.getSrc()];
                const capture = struct?
                    struct.id == StructureType.Ruin? CaptureType.Ruins : 
                    tile.terrainType == TerrainType.Ocean? CaptureType.Starfish :
                    tile.capitalOf > 0? CaptureType.City : CaptureType.Village : CaptureType.None;
                return `${MoveType[this.moveType]} ${CaptureType[capture]}`;
            }
            case MoveType.Ability: {
                return `${MoveType[this.moveType]} ${AbilityType[this.getType<AbilityType>()]}`;
            }
            default:
                return `missing: ${MoveType[this.moveType]} - ${this.type}`;
        }
    }

    serialize(): string {
        const action = this.toAction();
        return JSON.stringify([
            action.action,
            this.getSrc(),
            this.getTarget(),
            this.moveType == MoveType.Build? this.getType() : null,
            this.moveType == MoveType.Summon? this.getType() : null,
            this.moveType == MoveType.Research? this.getType() : null,
            this.moveType == MoveType.Ability? this.getType() : null,
            this.moveType == MoveType.Reward? this.getType() : null,
        ].filter(Boolean));
    }

    getSrc(): number {
        return this.src!;
    }

    getTarget(): number {
        return this.target!;
    }

    getType<T>(): T {
        return this.type as T;
    }

    hasSrc(): boolean {
        return this.src !== null;
    }

    hasTarget(): boolean {
        return this.target !== null;
    }

    hasType(): boolean {
        return this.type !== null;
    }

    toAction(): Action {
        const action: Action = {
            action: this.moveType,
            from: null,
            to: null,
            struct: null,
            unit: null,
            tech: null,
            ability: null,
            reward: null
        };

        try {
            switch (this.moveType) {
                case MoveType.Attack:
                case MoveType.Step:
                    action.from = this.getSrc();
                    action.to = this.getTarget();
                    if(!action.from || !action.to) {
                        throw Error(`src: ${action.from}, to: ${action.to}`);
                    }
                    break;
                case MoveType.Summon:
                    action.from = this.getSrc();
                    action.unit = this.getType<UnitType>();
                    if(!action.from || !action.unit) {
                        throw Error(`src: ${action.from}, type: ${action.unit}`);
                    }
                    break;
                case MoveType.Research:
                    action.tech = this.getType<TechnologyType>();
                    if(!action.tech) {
                        throw Error(`type: ${action.tech}`);
                    }
                    break;
                case MoveType.Harvest:
                    action.to = this.getTarget();
                    if(!action.to) {
                        throw Error(`to: ${action.to}`);
                    }
                    break;
                case MoveType.Build:
                    action.to = this.getTarget();
                    action.struct = this.getType<StructureType>();
                    if(!action.to || !action.struct) {
                        throw Error(`src: ${action.to}, type: ${action.struct}`);
                    }
                    break;
                case MoveType.Reward:
                    action.reward = this.getType<RewardType>();
                    if(!action.reward) {
                        throw Error(`type: ${action.reward}`);
                    }
                    break;
                case MoveType.Capture:
                    action.from = this.getSrc();
                    if(!action.from) {
                        throw Error(`src: ${action.from}`);
                    }
                    break;
                case MoveType.Ability:
                    action.ability = this.getType<AbilityType>();
                    if(!action.ability) {
                        throw Error(`type: ${action.ability}`);
                    }
                    if(EconomyAbilityTypes[action.ability]) {
                        action.to = this.getTarget();
                        if(!action.to) {
                            throw Error(`to: ${action.to}`);
                        }
                    }
                    else if(ArmyAbilityTypes.includes(action.ability)) {
                        action.from = this.getSrc();
                        if(!action.from) {
                            throw Error(`from: ${action.from}`);
                        }
                    }
                    else {
                        throw Error(`missing: ${AbilityType[this.getType<AbilityType>()]}`);
                    }
                    throw 'check notes';
                    break;
                case MoveType.EndTurn:
                    break;
                default:
                    throw Error(`wtf`)
                    break;
            }
        } catch (error: any) {
            throw Error(`${MoveType[this.moveType]} invalid, ${error.message || error}`);
        }
        
        return action;
    }
}
