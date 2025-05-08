import { getPovTribe } from "../../functions";
import { CallbackResult } from "../../move";
import { GameState } from "../../states";
import { AbilityType, TerrainType } from "../../types";
import Ability from "../Ability";

export default class ClearForest extends Ability {
    constructor(target: number) {
        super(null, target, AbilityType.ClearForest);
    }

    execute(state: GameState): CallbackResult {
        const pov = getPovTribe(state);
        const tile = state.tiles[this.getTarget()];

        tile.terrainType = TerrainType.Field;
        pov._stars++;

        return {
            rewards: [],
            undo: () => {
                pov._stars--;
                tile.terrainType = TerrainType.Forest;
            }
        };
    }
}