import { CallbackResult } from "../../move";
import { GameState } from "../../states";
import { AbilityType } from "../../types";
import Ability from "../Ability";

export default class Drain extends Ability {
    constructor(target: number) {
        super(null, target, AbilityType.Drain);
    }

    execute(state: GameState) {
        return {
            rewards: [],
            undo: () => {
            }
        };
    }
}