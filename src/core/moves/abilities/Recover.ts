import healUnit from "../../actions/units/Heal";
import { getUnitAt, isInTerritory } from "../../functions";
import { CallbackResult } from "../../move";
import { GameState } from "../../states";
import { AbilityType } from "../../types";
import Ability from "../Ability";

export default class Recover extends Ability {
    constructor(src: number) {
        super(src, null, AbilityType.Recover);
    }

    execute(state: GameState): CallbackResult {
        const unit = getUnitAt(state, this.getSrc())!;
        const undoHeal = healUnit(state, unit, isInTerritory(state, unit)? 4 : 2);
        
        return {
            rewards: [],
            undo: () => {
                undoHeal();
            }
        };
    }
}