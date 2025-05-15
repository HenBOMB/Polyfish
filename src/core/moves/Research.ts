import { getPovTribe, getTechCost } from "../functions";
import Move, { CallbackResult } from "../move";
import { MoveType, TechnologyType } from "../types";
import { GameState, TechnologyState } from "../states";
import { spendStars } from "../actions";
import { xorPlayer } from "../../zobrist/hasher";

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

        xorPlayer.tech(pov, tech.techType);
        pov._tech.push(tech);
        const undoPurchase = spendStars(state, cost);
        
        return {
            rewards: [],
            undo: () => {
                undoPurchase();
                pov._tech.pop();
                xorPlayer.tech(pov, tech.techType);
            }
        };
    }
}