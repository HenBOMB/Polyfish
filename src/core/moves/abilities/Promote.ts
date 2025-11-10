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

    execute(state: GameState) {
        const unit = getUnitAt(state, this.getSrc())!;
        const hp = unit.health;

        xorUnit.veteran(state, unit)
        unit.veteran = true;
        unit.health = getMaxHealth(unit);
        
        return {
            rewards: [],
            undo: () => {
                xorUnit.veteran(state, unit)
                unit.health = hp;
                unit.veteran = false;
            }
        };
    }
}