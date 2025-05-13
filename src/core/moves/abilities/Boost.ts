import { tryRemoveEffect } from "../../actions";
import { getAlliesNearTile } from "../../functions";
import { CallbackResult } from "../../move";
import { GameState } from "../../states";
import { AbilityType, EffectType } from "../../types";
import Ability from "../Ability";

export default class Boost extends Ability {
    constructor(src: number) {
        super(src, null, AbilityType.Boost);
    }

    execute(state: GameState): CallbackResult {
        const chain = getAlliesNearTile(state, this.getSrc())
            .map(x => tryRemoveEffect(state, x, EffectType.Boost));
        return {
            rewards: [],
            undo: () => {
                chain.forEach(x => x());
            }
        };
    }
}