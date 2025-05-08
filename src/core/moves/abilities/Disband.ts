import { removeUnit } from "../../actions";
import { getPovTribe, getRealUnitSettings, getUnitAt } from "../../functions";
import { CallbackResult } from "../../move";
import { GameState } from "../../states";
import { AbilityType } from "../../types";
import Ability from "../Ability";

export default class Disband extends Ability {
    constructor(src: number) {
        super(src, null, AbilityType.Disband);
    }

    execute(state: GameState): CallbackResult {
        const unit = getUnitAt(state, this.getSrc())!;
        const tribe = getPovTribe(state);
        const undoRemove = removeUnit(state, unit);
        const cost = Math.floor(getRealUnitSettings(unit).cost / 2);
        tribe._stars += cost;
        return {
            rewards: [],
            undo: () => {
                tribe._stars -= cost;
                undoRemove()
            }
        }
    }
}