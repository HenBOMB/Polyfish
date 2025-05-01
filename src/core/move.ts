import { tryDiscoverRewardOtherTribes } from "./functions";
import { Branch } from "./moves";
import { GameState } from "./states";
import { CaptureType, ResourceType, RewardType, StructureType, TechnologyType, UnitType } from "./types";

export enum MoveType {
	None 		= 0,
	Step 		= 1,
	Attack 		= 2,
	Ability		= 3,
	Summon	 	= 4,
	Harvest 	= 5,
	Build 		= 6,
	Research 	= 7,
	Capture 	= 8,
	Reward 	    = 9,
	Action 	    = 10,
	EndTurn 	= 11,
}

export default class Move {
    readonly moveType: MoveType;
    readonly src: number;
    readonly target: number;
    readonly type: number;
    readonly targetName?: string;
    readonly moveName?: string;
    readonly executeCb: (state: GameState) => Branch;

    constructor(id: MoveType, src: number, target: number, type: number, execute: (state: GameState) => Branch, targetName?: string, moveName?: string) {
        if(!MoveType[id]) throw Error(`Invalid move type: ${id}`);
        this.moveType = id;
        this.src = src < 0? 0 : src;
        this.target = target < 0? 0 : target;
        this.type = type < 0? 0 : type;
        this.executeCb = execute;
        this.targetName = targetName;
        this.moveName = moveName;
    }

    execute(state: GameState): Branch {
        try {
            const br = this.executeCb(state);
            // Only if we moved or attacked which causes new disovered tiles
            if(this.moveType == MoveType.Step || this.moveType == MoveType.Attack) {
                tryDiscoverRewardOtherTribes(state);
            }
            return br;
        } catch (error) {
            console.log(`CRITICAL\nMOVE FAILED "${MoveType[this.moveType]}"`);
            console.error(error);
            return {
                undo: () => { }
            };
        }
    }

    stringify() {
        switch (this.moveType) {
            case MoveType.Step: {
                return `step ${UnitType[this.type].toLowerCase()}`;
            }
            case MoveType.Attack: {
                // with type == 0 is riot
                // id: `_owner, _unitType, _tileIndex, _owner, _unitType, _tileIndex`,
                return `attack ${this.type == 0? 'riot' : UnitType[this.type].toLowerCase()}`
            }
            case MoveType.Summon: {
                return `summon ${UnitType[this.type].toLowerCase()}`;
            }
            case MoveType.Research: {
                return `research ${TechnologyType[this.type].toLowerCase()}`;
            }
            case MoveType.Harvest: {
                return `harvest ${ResourceType[this.type].toLowerCase()}`;
            }
            case MoveType.Build: {
                return `build ${StructureType[this.type].toLowerCase()}`;
            }
            case MoveType.Reward: {
                return `reward ${RewardType[this.type].toLowerCase()}`;
            }
            case MoveType.EndTurn: {
                return `end turn`;
            }
            case MoveType.Capture: {
                return `capture ${CaptureType[this.type].toLowerCase()}`;
            }
            default:
                return `missing: ${MoveType[this.moveType]} - ${this.type}`;
        }
    }
}