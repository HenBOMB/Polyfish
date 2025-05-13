import { getCityAt, getEnemyAt, getUnitAt } from "./functions";
import { GameState } from "./states";
import { AbilityType, CaptureType, MoveType, ResourceType, RewardType, StructureType, TechnologyType, TerrainType, UnitType } from "./types";

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

    constructor(id: MoveType, src: number | null = null, target: number | null = null, type: number | null = null) {
        this.moveType = id;
        this.src = src && src < 0? null : src;
        this.target = target && target < 0? null : target;
        this.type = type && type < 0? null : type;
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

    static serialize(move: Move): string {
        return JSON.stringify([
            move.moveType,
            move.hasSrc()? move.getSrc() : null,
            move.hasTarget()? move.getTarget() : null,
            move.moveType == MoveType.Build? move.getType() : null,
            move.moveType == MoveType.Summon? move.getType() : null,
            move.moveType == MoveType.Research? move.getType() : null,
            move.moveType == MoveType.Ability? move.getType() : null,
            move.moveType == MoveType.Reward? move.getType() : null,
        ]);
    }

    static deserialize(ser: string): Action {
        const list = JSON.parse(ser) as string[];
        const [ action, src, target, struct, unit, tech, ability, reward ] = list.map(x => Number(x));
        return {
            action,
            from: src,
            to: target,
            struct: struct,
            unit: unit,
            tech: tech,
            ability: ability,
            reward: reward,
        };
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
}
