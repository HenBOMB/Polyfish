import { xorUnit } from "../../../zobrist/hasher";
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

        xorUnit.veteran(state, unit)
        unit._veteran = true;
        unit._health = getMaxHealth(unit);
        
        return {
            rewards: [],
            undo: () => {
                xorUnit.veteran(state, unit)
                unit._health = hp;
                unit._veteran = false;
            }
        };
    }
}