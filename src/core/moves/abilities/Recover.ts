import healUnit from "../../actions/units/Heal";
import { getUnitAt, isInTerritory } from "../../functions";
import { GameState } from "../../states";
import { AbilityType } from "../../types";
import Ability from "../Ability";

export default class Recover extends Ability {
    constructor(src: number) {
        super(src, null, AbilityType.Recover);
    }

    execute(state: GameState) {
        const unit = getUnitAt(state, this.getSrc())!;
        const undoHeal = healUnit(state, unit, isInTerritory(state, unit)? 4 : 2);
        
        const oldMoved = unit.moved;
        const oldAttacked = unit.attacked;

        unit.moved = unit.attacked = true;

        return {
            rewards: [],
            undo: () => {
                unit.moved = oldMoved;
                unit.attacked = oldAttacked;
                
                undoHeal();
            }
        };
    }
}