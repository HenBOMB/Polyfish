import Move, { CallbackResult, UndoCallback } from "../move";
import { AbilityType, MoveType, TribeType } from "../types";
import { GameState } from "../states";
import { getPovTribe } from "../functions";
import { TechnologySettings } from "../settings/TechnologySettings";
import { StructureSettings } from "../settings/StructureSettings";

export default class EndTurn extends Move {
    constructor() {
        super(MoveType.EndTurn, null, null, null);
    }

    execute(state: GameState): CallbackResult {
        // Special case: decompose, removes the structure at full cost
        const tribe = getPovTribe(state);
        if(tribe.tribeType === TribeType.Cymanti && tribe._tech.some(x => TechnologySettings[x.techType].unlocksAbility === AbilityType.Decompose)) {
            const decomposed: UndoCallback[] = [];

            for (const strTileIndex in state.structures) {
                if(state.structures[strTileIndex]?.turn === -2) {
                    const struct = state.structures[strTileIndex];
                    const cost = StructureSettings[struct.id].cost || 0;
                    delete state.structures[struct.tileIndex];
                    tribe._stars += cost;
                    // Remove score
                    decomposed.push(() => {
                        tribe._stars -= cost;
                        state.structures[struct.tileIndex] = struct;
                    });
                }
            }

            return {
                rewards: [],
                undo: () => {
                    decomposed.reverse().forEach(x => x());
                },
            };
        }

        return {
            rewards: [],
            undo: () => {

            },
        };
    }
}