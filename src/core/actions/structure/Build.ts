import { xorPlayer } from "../../../zobrist/hasher";
import { spendStars } from "../../actions";
import addPopulationToCity from "../AddPopulation";
import { createStructure } from "./Create";
import { getRulingCity, getAdjacentTiles } from "../../functions";
import { Branch } from "../../move";
import { StructureSettings } from "../../settings/StructureSettings";
import { IsStructureTask } from "../../settings/TaskSettings";
import { GameState } from "../../states";
import { StructureType } from "../../types";


export default function(state: GameState, strctureType: StructureType, idx: number): Branch {
    const pov = state.tribes[state.settings.currentPlayerTurnId];
    const settings = StructureSettings[strctureType];
    const rulingCity = getRulingCity(state, idx)!;

    const undoPurchase = !settings.cost? () => { } : spendStars(state, settings.cost);
    const undoCreate = createStructure(state, idx, strctureType);

    let rewardPopCount = settings.rewardPop || 0;

    if (settings.adjacentTypes !== undefined) {
        const adjCount = getAdjacentTiles(state, idx)
            .filter(x => state.structures[x.coords.idx] ? settings.adjacentTypes!.has(state.structures[x.coords.idx]!.type) : false).length;
        rewardPopCount *= adjCount;
    }

    if (IsStructureTask[strctureType]) {
        pov.builtUniqueImprovements.add(strctureType);
        xorPlayer.unique(pov, strctureType);
    }

    const popBranch = addPopulationToCity(state, rulingCity, rewardPopCount);
    // const portBranch = addMissingConnections(state, rulingCity, idx);
    return {
        // rewards: [ ...(popBranch?.rewards || []), ...(portBranch?.rewards || []) ],
        rewards: popBranch.rewards,
        undo: () => {
            // portBranch?.undo();
            popBranch.undo();
            if (IsStructureTask[strctureType]) {
                pov.builtUniqueImprovements.delete(strctureType);
                xorPlayer.unique(pov, strctureType);
            }
            undoCreate();
            undoPurchase();
        }
    };
}
