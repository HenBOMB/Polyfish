import { getPovTribe } from "../../functions";
import { CallbackResult } from "../../move";
import { GameState } from "../../states";
import { AbilityType, ResourceType, TerrainType } from "../../types";
import Ability from "../Ability";

export default class BurnForest extends Ability {
    constructor(target: number) {
        super(null, target, AbilityType.BurnForest);
    }

    execute(state: GameState): CallbackResult {
        const pov = getPovTribe(state);
        const tile = state.tiles[this.getTarget()];
        const resource = state.resources[tile.tileIndex];

        tile.terrainType = TerrainType.Field;
        pov._stars -= 2;

        state.resources[tile.tileIndex] = {
            id: ResourceType.Crop,
            tileIndex: tile.tileIndex
        }

        return {
            rewards: [],
            undo: () => {
                if(!resource) {
                    delete state.resources[tile.tileIndex];
                }
                else {
                    state.resources[tile.tileIndex] = resource;
                }
                pov._stars += 2;
                tile.terrainType = TerrainType.Forest;
            }
        };
    }
}