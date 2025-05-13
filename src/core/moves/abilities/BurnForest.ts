import { consumeResource, modifyTerrain, spendStars } from "../../actions";
import { CallbackResult } from "../../move";
import { GameState } from "../../states";
import { AbilityType, ResourceType, TerrainType } from "../../types";
import Ability from "../Ability";

export default class BurnForest extends Ability {
    constructor(target: number) {
        super(null, target, AbilityType.BurnForest);
    }

    execute(state: GameState): CallbackResult {
        const undoTerrain = modifyTerrain(state, this.getTarget(), TerrainType.Field);
        const undoStars = spendStars(state, 2);
        const undoResource = consumeResource(state, this.getTarget(), ResourceType.Crop);
        return {
            rewards: [],
            undo: () => {
                undoResource();
                undoStars();
                undoTerrain();
            }
        };
    }
}