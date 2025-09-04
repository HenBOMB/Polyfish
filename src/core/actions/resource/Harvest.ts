import { spendStars } from "../../actions";
import consumeResource from "./Consume";
import addPopulationToCity from "../AddPopulation";
import { getCityOwningTile } from "../../functions";
import { Branch } from "../../move";
import { ResourceSettings } from "../../settings/ResourceSettings";
import { GameState } from "../../states";


export default function(state: GameState, tileIndex: number): Branch {
    const harvested = state.resources[tileIndex]!;
    const settings = ResourceSettings[harvested.id];
    const rulingCity = getCityOwningTile(state, tileIndex)!;

    const undoPurchase = spendStars(state, settings.cost || 0);
    const undoResource = consumeResource(state, tileIndex);
    const popBranch = addPopulationToCity(state, rulingCity, settings.rewardPop);

    return {
        rewards: popBranch.rewards,
        undo: () => {
            popBranch.undo();
            undoResource();
            undoPurchase();
        }
    };
}
