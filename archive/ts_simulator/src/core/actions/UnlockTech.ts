import { xorPlayer } from "../../zobrist/hasher";
import { spendStars } from "../actions";
import { getPovTribe, getTechCost, getTechSettings } from "../functions";
import { Branch } from "../move";
import { GameState } from "../states";
import { TechnologyType } from "../types";


export default function(state: GameState, type: TechnologyType, free=false): Branch {
    const pov = getPovTribe(state);

    const scroll = {
        type,
        discovered: state.settings._areYouSure,
    };

    xorPlayer.tech(pov, scroll.type);

    pov.tech_vanilla.push(scroll);

    const undoPurchase = free? () => { } : spendStars(state, getTechCost(pov, type));

    // const score = 100 * getOriginalTech(type).tier!;
    const score = 100 * getTechSettings(type).tier!;

    pov.score += score;

    return {
        rewards: [],
        undo: () => {
            pov.score -= score;

            undoPurchase();

            pov.tech_vanilla.pop();

            xorPlayer.tech(pov, scroll.type);
        }
    };
}
