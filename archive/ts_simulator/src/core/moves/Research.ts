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

    execute(state: GameState) {
        const pov = getPovTribe(state);
        const cost = getTechCost(pov, this.getType<TechnologyType>());
        const tech: TechnologyState = {
            type: this.getType<TechnologyType>(),
            discovered: state.settings._areYouSure,
        }

        xorPlayer.tech(pov, tech.type);
        pov.tech_vanilla.push(tech);
        const undoPurchase = spendStars(state, cost);
        
        return {
            rewards: [],
            undo: () => {
                undoPurchase();
                pov.tech_vanilla.pop();
                xorPlayer.tech(pov, tech.type);
            }
        };
    }
}