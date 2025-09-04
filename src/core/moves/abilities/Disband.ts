import { gainStars } from "../../actions";
import removeUnit from "../../actions/units/Remove";
import { getRealUnitSettings, getUnitAt } from "../../functions";
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

        const undoRemove = removeUnit(state, unit);
        const undoStars = gainStars(state, Math.floor(getRealUnitSettings(unit).cost / 2));

        return {
            rewards: [],
            undo: () => {
                undoStars();
                undoRemove()
            }
        }
    }
}