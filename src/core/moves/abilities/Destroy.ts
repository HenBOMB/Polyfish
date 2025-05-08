import { getRulingCity } from "../../functions";
import { CallbackResult } from "../../move";
import { StructureSettings } from "../../settings/StructureSettings";
import { GameState } from "../../states";
import { AbilityType } from "../../types";
import Ability from "../Ability";

export default class Destroy extends Ability {
    constructor(target: number) {
        super(null, target, AbilityType.Destroy);
    }

    execute(state: GameState): CallbackResult {
        const target = this.getTarget();
        const struct = state.structures[target]!;
        const settings = StructureSettings[struct.id];
        const city = getRulingCity(state, target)!;

        delete state.structures[target];
        if(settings.rewardPop) {
            city._population -= settings.rewardPop;
            city._progress -= settings.rewardPop;
            if(city._progress < 0) {
                city._level--;
            }
        }
        // getPovTribe(state)._score -= settings.score;
        // TODO use score to read opponents in FOW
        
        return {
            rewards: [],
            undo: () => {
                if(settings.rewardPop) {
                    if(city._progress < 0) {
                        city._level++;
                    }
                    city._progress += settings.rewardPop;
                    city._population += settings.rewardPop;
                }
                state.structures[target] = struct;
            }
        };
    }
}