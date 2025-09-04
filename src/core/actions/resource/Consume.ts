import { xorResource } from "../../../zobrist/hasher";
import { UndoCallback } from "../../move";
import { GameState } from "../../states";
import { ResourceType } from "../../types";


export default function(state: GameState, tileIndex: number, replaceType?: ResourceType): UndoCallback {
    const oldResource = state.resources[tileIndex];
    const newResource = replaceType ? replaceType : ResourceType.None;

    xorResource(state, tileIndex, oldResource ? oldResource.id : ResourceType.None, newResource);

    if (replaceType) {
        state.resources[tileIndex] = {
            id: replaceType,
            tileIndex
        };
    }
    else {
        delete state.resources[tileIndex];
    }

    return () => {
        xorResource(state, tileIndex, newResource, oldResource ? oldResource.id : ResourceType.None);
        state.resources[tileIndex] = oldResource;
    };
}
