import { splashDamageArea } from "../../actions";
import { getUnitAt, getUnitAttack } from "../../functions";
import { CallbackResult } from "../../move";
import { GameState } from "../../states";
import { AbilityType } from "../../types";
import Ability from "../Ability";

export default class Explode extends Ability {
    constructor(src: number) {
        super(src, null, AbilityType.Explode);
    }

    execute(state: GameState): CallbackResult {
        const attacker = getUnitAt(state, this.getSrc())!;
        // explode deals half damage the unit deals, i guess its the same as splash logic
        return {
            rewards: [],
            undo: splashDamageArea(state, attacker, getUnitAttack(attacker) / 2),
        };
    }
}