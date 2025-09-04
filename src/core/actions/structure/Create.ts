import { xorStructure } from "../../../zobrist/hasher";
import { UndoCallback } from "../../move";
import { GameState, StructureState } from "../../states";
import { StructureType } from "../../types";


export function createStructure(state: GameState, tileIndex: number, strctureType: StructureType, level = 1): UndoCallback {
    // specific to ruins -> aquarion free city
    const oldStruct = state.structures[tileIndex];

    const structure: StructureState = {
        id: strctureType,
        _level: level,
        turn: state.settings._turn,
        tileIndex,
        reward: 0,
    };

    xorStructure(state, tileIndex, oldStruct ? oldStruct.id : StructureType.None, strctureType);
    state.structures[tileIndex] = structure;

    return () => {
        xorStructure(state, tileIndex, strctureType, oldStruct ? oldStruct.id : StructureType.None);
        state.structures[tileIndex] = oldStruct;
    };
}
