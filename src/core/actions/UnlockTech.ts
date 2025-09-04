import { xorPlayer } from "../../zobrist/hasher";
import { spendStars } from "../actions";
import { getPovTribe, getTechCost, getTechSettings } from "../functions";
import { Branch } from "../move";
import { GameState } from "../states";
import { TechnologyType } from "../types";


export default function(state: GameState, techType: TechnologyType, free = false): Branch {
    const pov = getPovTribe(state);

    const scroll = {
        techType,
        discovered: state.settings.areYouSure,
    };

    xorPlayer.tech(pov, scroll.techType);

    pov._tech.push(scroll);

    const undoPurchase = free ? () => { } : spendStars(state, getTechCost(pov, techType));

    // const score = 100 * getOriginalTech(techType).tier!;
    const score = 100 * getTechSettings(techType).tier!;

    pov._score += score;

    return {
        rewards: [],
        undo: () => {
            pov._score -= score;

            undoPurchase();

            pov._tech.pop();

            xorPlayer.tech(pov, scroll.techType);
        }
    };
}
