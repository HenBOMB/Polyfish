import Move, { CallbackResult } from "../move";
import { MoveType, StructureType } from "../types";
import { GameState } from "../states";
import { buildStructure } from "../actions";

export default class Structure extends Move {
    constructor(target: number, type: number) {
        super(MoveType.Build, null, target, type);
    }

    execute(state: GameState): CallbackResult {
        const strucType = this.getType<StructureType>();
        return buildStructure(state, strucType, this.getTarget());
    }
}