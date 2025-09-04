import Move, { CallbackResult } from "../move";
import { MoveType } from "../types";
import { GameState } from "../states";
import harvestResource from "../actions/resource/Harvest";

export default class Harvest extends Move {
    constructor(target: number) {
        super(MoveType.Harvest, null, target, null);
    }

    execute(state: GameState): CallbackResult {
        return harvestResource(state, this.getTarget());
    }
}