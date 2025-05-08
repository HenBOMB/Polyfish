import { getAlliesNearTile, getMaxHealth, getUnitAt, isBoosted } from "../../functions";
import { CallbackResult } from "../../move";
import { GameState } from "../../states";
import { AbilityType, EffectType } from "../../types";
import Ability from "../Ability";

export default class BreakIce extends Ability {
    constructor(target: number) {
        super(null, target, AbilityType.BreakIce);
    }

    execute(state: GameState): CallbackResult {
        return {
            rewards: [],
            undo: () => {
            }
        };
    }
}