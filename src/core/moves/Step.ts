import { Logger } from "../../polyfish/logger";
import { getTrueUnitAt, getUnitAt } from "../functions";
import Move, { CallbackResult } from "../move";
import { EffectType, MoveType } from "../types";
import { GameState } from "../states";
import { UnitType } from "../types";
import { stepUnit, tryRemoveEffect } from "../actions";

export default class Step extends Move {
    constructor(src: number, target: number) {
        super(MoveType.Step, src, target);
    }

    execute(state: GameState): CallbackResult {
        const unit = getUnitAt(state, this.getSrc())!;
        const target = this.getTarget();

        // If we are stepping over a unit, then it 100% must be an invisble enemy cloak, it must be revealed and the step must be cancelled
        if (state.tiles[target]._unitOwner > 0) {
            if(state.settings.areYouSure) {
                const cloak = getTrueUnitAt(state, target)!;
                const enemy = state.tribes[cloak._owner];

                // reveal the cloak
                const undo = tryRemoveEffect(state, cloak, EffectType.Invisible);

                return {
                    rewards: [],
                    undo: () => {
                        undo();
                    },
                };
            }
            // TODO If not live then some complex setup is needed for allowing two units to be on the same tile
            else {
                return Logger.illegal(MoveType.Step, `${UnitType[unit._unitType]} -> Cloak SUPERPOSITION is required`);
            }
        }

        return stepUnit(state, unit, target);
    }
}