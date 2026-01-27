import { xorStructure } from "../../../zobrist/hasher";
import { UndoCallback } from "../../move";
import { GameState, StructureState } from "../../states";
import { StructureType } from "../../types";


export function createStructure(state: GameState, idx: number, strctureType: StructureType, level = 1): UndoCallback {
    // specific to ruins -> aquarion free city
    const oldStruct = state.structures[idx];

    const structure: StructureState = {
        type: strctureType,
        level: level,
        founded: state.settings.turn,
        tileIndex: idx,
        score: 0,
    };

    xorStructure(state, idx, oldStruct ? oldStruct.type : StructureType.None, strctureType);
    state.structures[idx] = structure;

    return () => {
        xorStructure(state, idx, strctureType, oldStruct ? oldStruct.type : StructureType.None);
        state.structures[idx] = oldStruct;
    };
}
