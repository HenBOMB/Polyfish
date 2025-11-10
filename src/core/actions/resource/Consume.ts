import { xorResource } from "../../../zobrist/hasher";
import { UndoCallback } from "../../move";
import { GameState } from "../../states";
import { ResourceType } from "../../types";


export default function(state: GameState, idx: number, replaceType?: ResourceType): UndoCallback {
    const oldResource = state.resources[idx];
    const newResource = replaceType ? replaceType : ResourceType.None;

    xorResource(state, idx, oldResource ? oldResource.type : ResourceType.None, newResource);

    if (replaceType) {
        state.resources[idx] = {
            type: replaceType,
            tileIndex: idx
        };
    }
    else {
        delete state.resources[idx];
    }

    return () => {
        xorResource(state, idx, newResource, oldResource ? oldResource.type : ResourceType.None);
        state.resources[idx] = oldResource;
    };
}
