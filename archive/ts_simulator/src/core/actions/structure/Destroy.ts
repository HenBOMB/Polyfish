import { xorStructure, xorCity } from "../../../zobrist/hasher";
import { getPovTribe, getCityOwningTile } from "../../functions";
import { UndoCallback } from "../../move";
import { StructureSettings } from "../../settings/StructureSettings";
import { GameState } from "../../states";
import { StructureType } from "../../types";


export function destroyStructure(state: GameState, idx: number): UndoCallback {
    const pov = getPovTribe(state);
    const struct = state.structures[idx]!;

    xorStructure(state, idx, struct.type, StructureType.None);

    if (struct.type === StructureType.Ruin) {
        delete state.structures[idx];
        return () => {
            state.structures[idx] = struct;
            xorStructure(state, idx, StructureType.None, struct.type);
        };
    }

    const city = getCityOwningTile(state, idx)!;
    const settings = StructureSettings[struct.type];

    delete state.structures[idx];

    if (settings.rewardPop) {
        city.population -= settings.rewardPop;
        city.progress -= settings.rewardPop;
        if (city.progress < 0) {
            xorCity.level(state, city, city.level, city.level - 1);
            city.level--;
        }
    }

    // TODO Remove score
    return () => {
        if (settings.rewardPop) {
            if (city.progress < 0) {
                xorCity.level(state, city, city.level, city.level + 1);
                city.level++;
            }
            city.progress += settings.rewardPop;
            city.population += settings.rewardPop;
        }
        state.structures[idx] = struct;
        xorStructure(state, idx, StructureType.None, struct.type);
    };
}
