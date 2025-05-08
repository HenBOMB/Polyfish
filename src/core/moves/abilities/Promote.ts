import { getMaxHealth, getUnitAt } from "../../functions";
import { CallbackResult } from "../../move";
import { GameState } from "../../states";
import { AbilityType } from "../../types";
import Ability from "../Ability";

export default class Promote extends Ability {
    constructor(src: number) {
        super(src, null, AbilityType.Promote);
    }

    execute(state: GameState): CallbackResult {
        const unit = getUnitAt(state, this.getSrc())!;
        const hp = unit._health;

        unit.veteran = true;
        unit._health = getMaxHealth(unit);
        
        return {
            rewards: [],
            undo: () => {
                unit._health = hp;
                unit.veteran = false;
            }
        };
    }
}