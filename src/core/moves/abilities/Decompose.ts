import { CallbackResult } from "../../move";
import { GameState } from "../../states";
import { AbilityType } from "../../types";
import Ability from "../Ability";

export default class Decompose extends Ability {
    constructor(target: number) {
        super(null, target, AbilityType.BurnForest);
    }

    execute(state: GameState): CallbackResult {
        const target = this.getTarget();
        const struct = state.structures[target]!;
        const turn = struct.turn;

        struct.turn = -2;

        return {
            rewards: [],
            undo: () => {
                struct.turn = turn;
            }
        };
    }
}