import { modifyTerrain, spendStars } from "../../actions";
import { getPovTribe } from "../../functions";
import { CallbackResult } from "../../move";
import { GameState } from "../../states";
import { AbilityType, TerrainType } from "../../types";
import Ability from "../Ability";

export default class GrowForest extends Ability {
    constructor(target: number) {
        super(null, target, AbilityType.GrowForest);
    }

    execute(state: GameState): CallbackResult {
        const undoTerrain = modifyTerrain(state, this.getTarget(), TerrainType.Forest);
        const undoStars = spendStars(state, 5);

        return {
            rewards: [],
            undo: () => {
                undoStars();
                undoTerrain();
            }
        };
    }
}