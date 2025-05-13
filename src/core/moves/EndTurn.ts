import Move, { CallbackResult, UndoCallback } from "../move";
import { AbilityType, MoveType, TribeType } from "../types";
import { GameState } from "../states";
import { getPovTribe } from "../functions";
import { TechnologySettings } from "../settings/TechnologySettings";
import { StructureSettings } from "../settings/StructureSettings";
import { destroyStructure, gainStars } from "../actions";

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
                    const undoDestroy = destroyStructure(state, struct.tileIndex);
                    const undoStars = gainStars(state, StructureSettings[struct.id].cost || 0);
                    decomposed.push(() => {
                        undoStars();
                        undoDestroy();
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