import { destroyStructure } from "../../actions";
import { CallbackResult } from "../../move";
import { GameState } from "../../states";
import { AbilityType } from "../../types";
import Ability from "../Ability";

export default class Destroy extends Ability {
    constructor(target: number) {
        super(null, target, AbilityType.Destroy);
    }

    execute(state: GameState): CallbackResult {
        return {
            rewards: [],
            undo: destroyStructure(state, this.getTarget()),
        };
    }
}