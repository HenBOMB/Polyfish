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
        const target = this.getTarget();
        const pov = getPovTribe(state);
        const tile = state.tiles[target]!;

        tile.terrainType = TerrainType.Forest;
        pov._stars -= 5;

        return {
            rewards: [],
            undo: () => {
                pov._stars += 5;
                tile.terrainType = TerrainType.Field;
            }
        };
    }
}