import { getAlliesNearTile, getMaxHealth, getUnitAt, isBoosted } from "../../functions";
import { CallbackResult } from "../../move";
import { GameState } from "../../states";
import { AbilityType, EffectType } from "../../types";
import Ability from "../Ability";

export default class Boost extends Ability {
    constructor(src: number) {
        super(src, null, AbilityType.Boost);
    }

    execute(state: GameState): CallbackResult {
        const adjAllies = getAlliesNearTile(state, this.getSrc());
        const unboosted: number[] = [];

        for (let i = 0; i < adjAllies.length; i++) {
            if(isBoosted(adjAllies[i])) continue;
            unboosted.push(i);
            adjAllies[i]._effects.push(EffectType.Boost);
        }
        
        return {
            rewards: [],
            undo: () => {
                unboosted.reverse().forEach(x => adjAllies[x]._effects.pop());
            }
        };
    }
}