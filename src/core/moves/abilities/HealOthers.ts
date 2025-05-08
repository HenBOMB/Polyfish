import { healUnit } from "../../actions";
import { getAlliesNearTile } from "../../functions";
import { CallbackResult } from "../../move";
import { GameState } from "../../states";
import { AbilityType } from "../../types";
import Ability from "../Ability";

export default class HealOthers extends Ability {
    constructor(src: number) {
        super(src, null, AbilityType.HealOthers);
    }

    execute(state: GameState): CallbackResult {
        const chain: any[] = [];
        const adjAllies = getAlliesNearTile(state, this.getSrc());

        for(const unit of adjAllies) {
            chain.push(healUnit(unit, 4));
        }

        return {
            rewards: [],
            undo: () => {
                chain.reverse().forEach(x => x());
            }
        }
    }
}