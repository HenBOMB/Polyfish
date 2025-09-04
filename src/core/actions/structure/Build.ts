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


export default function(state: GameState, strctureType: StructureType, tileIndex: number): Branch {
    const pov = state.tribes[state.settings._pov];
    const settings = StructureSettings[strctureType];
    const rulingCity = getRulingCity(state, tileIndex)!;
    const cost = settings.cost || 0;

    const undoPurchase = spendStars(state, cost);
    const undoCreate = createStructure(state, tileIndex, strctureType);

    let rewardPopCount = settings.rewardPop || 0;

    if (settings.adjacentTypes !== undefined) {
        const adjCount = getAdjacentTiles(state, tileIndex)
            .filter(x => state.structures[x.tileIndex] ? settings.adjacentTypes!.has(state.structures[x.tileIndex]!.id) : false).length;
        rewardPopCount *= adjCount;
    }

    if (IsStructureTask[strctureType]) {
        pov._builtUniqueStructures.add(strctureType);
        xorPlayer.unique(pov, strctureType);
    }

    const popBranch = addPopulationToCity(state, rulingCity, rewardPopCount);
    // const portBranch = addMissingConnections(state, rulingCity, tileIndex);
    return {
        // rewards: [ ...(popBranch?.rewards || []), ...(portBranch?.rewards || []) ],
        rewards: popBranch.rewards,
        undo: () => {
            // portBranch?.undo();
            popBranch.undo();
            if (IsStructureTask[strctureType]) {
                pov._builtUniqueStructures.delete(strctureType);
                xorPlayer.unique(pov, strctureType);
            }
            undoCreate();
            undoPurchase();
        }
    };
}
