import { getAlliesNearTile, unBoost } from "../../functions";
import { CallbackResult } from "../../move";
import { GameState } from "../../states";
import { AbilityType } from "../../types";
import Ability from "../Ability";

export default class Boost extends Ability {
    constructor(src: number) {
        super(src, null, AbilityType.Boost);
    }

    execute(state: GameState): CallbackResult {
        const chain = getAlliesNearTile(state, this.getSrc())
            .map(x => unBoost(x));
        return {
            rewards: [],
            undo: () => {
                chain.forEach(x => x());
            }
        };
    }
}