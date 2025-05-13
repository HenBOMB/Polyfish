import { gainStars, modifyTerrain } from "../../actions";
import { CallbackResult } from "../../move";
import { GameState } from "../../states";
import { AbilityType, TerrainType } from "../../types";
import Ability from "../Ability";

export default class ClearForest extends Ability {
    constructor(target: number) {
        super(null, target, AbilityType.ClearForest);
    }

    execute(state: GameState): CallbackResult {
        const undoTerrain = modifyTerrain(state, this.getTarget(), TerrainType.Field);
        const undoStars = gainStars(state, 1);

        return {
            rewards: [],
            undo: () => {
                undoStars();
                undoTerrain();
            }
        };
    }
}