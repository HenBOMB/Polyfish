import { xorStructure, xorCity } from "../../../zobrist/hasher";
import { getPovTribe, getCityOwningTile } from "../../functions";
import { UndoCallback } from "../../move";
import { StructureSettings } from "../../settings/StructureSettings";
import { GameState } from "../../states";
import { StructureType } from "../../types";


export function destroyStructure(state: GameState, tileIndex: number): UndoCallback {
    const pov = getPovTribe(state);
    const struct = state.structures[tileIndex]!;

    xorStructure(state, tileIndex, struct.id, StructureType.None);

    if (struct.id === StructureType.Ruin) {
        delete state.structures[tileIndex];
        return () => {
            state.structures[tileIndex] = struct;
            xorStructure(state, tileIndex, StructureType.None, struct.id);
        };
    }

    const city = getCityOwningTile(state, tileIndex)!;
    const settings = StructureSettings[struct.id];

    delete state.structures[tileIndex];

    if (settings.rewardPop) {
        city._population -= settings.rewardPop;
        city._progress -= settings.rewardPop;
        if (city._progress < 0) {
            xorCity.level(state, city, city._level, city._level - 1);
            city._level--;
        }
    }

    // TODO Remove score
    return () => {
        if (settings.rewardPop) {
            if (city._progress < 0) {
                xorCity.level(state, city, city._level, city._level + 1);
                city._level++;
            }
            city._progress += settings.rewardPop;
            city._population += settings.rewardPop;
        }
        state.structures[tileIndex] = struct;
        xorStructure(state, tileIndex, StructureType.None, struct.id);
    };
}
