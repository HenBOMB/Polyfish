import { freezeArea } from "../../actions";
import { getUnitAt } from "../../functions";
import { CallbackResult } from "../../move";
import { GameState } from "../../states";
import { AbilityType } from "../../types";
import Ability from "../Ability";

export default class FreezeArea extends Ability {
    constructor(src: number) {
        super(src, null, AbilityType.FreezeArea);
    }

    execute(state: GameState): CallbackResult {
        return {
            rewards: [],
            undo: freezeArea(state, getUnitAt(state, this.getSrc())!)
        }
    }
}