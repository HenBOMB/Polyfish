import { getPovTribe, getTechCost } from "../functions";
import Move, { CallbackResult } from "../move";
import { MoveType, TechnologyType } from "../types";
import { GameState, TechnologyState } from "../states";

export default class Research extends Move {
    constructor(type: number) {
        super(MoveType.Research, null, null, type);
    }

    execute(state: GameState): CallbackResult {
        const pov = getPovTribe(state);
        const cost = getTechCost(pov, this.getType<TechnologyType>());
        const tech: TechnologyState = {
            techType: this.getType<TechnologyType>(),
            discovered: state.settings.areYouSure,
        }
        pov._tech.push(tech);
        pov._stars -= cost;
        
        return {
            rewards: [],
            undo: () => {
                pov._stars += cost;
                pov._tech.pop();
                if(state.settings.areYouSure) {
                    tech.discovered = false;
                }
            }
        };
    }
}