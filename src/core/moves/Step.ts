import { Logger } from "../../polyfish/logger";
import { getTrueUnitAt, getUnitAt } from "../functions";
import Move, { CallbackResult } from "../move";
import { MoveType } from "../types";
import { ArmyMovesGenerator } from "../moves";
import { GameState } from "../states";
import { EffectType, UnitType } from "../types";

export default class Step extends Move {
    constructor(src: number, target: number) {
        super(MoveType.Step, src, target);
    }

    execute(state: GameState): CallbackResult {
        const unit = getUnitAt(state, this.getSrc())!;
        const target = this.getTarget();

        // TODO If a unit moves onto a cloak when in not live mode, then the stepper will override the cloak
        if (state.tiles[target]._unitOwner > 0) {
            const unitType = getTrueUnitAt(state, target)?._unitType!;
            // If cloak is on tile, then it must be revealed
            if(unitType == UnitType.Cloak) {
                if(state.settings.areYouSure) {
                    let effectIndex = unit._effects.indexOf(EffectType.Invisible);
                    const cloak = getTrueUnitAt(state, target)!;
                    cloak._effects.splice(effectIndex, 1);
                    return {
                        rewards: [],
                        undo: () => {
                            cloak._effects.splice(effectIndex, 0, EffectType.Invisible);
                        }
                    };
                }
                else {
                    return Logger.illegal(MoveType.Step, `${UnitType[unit._unitType]} -> Cloak SUPERPOSITION is required`);
                }
            }
            else {
                return Logger.illegal(MoveType.Step, `${unit._tileIndex} -> ${target}, ${UnitType[unit._unitType]} -> ${UnitType[unitType]}`);
            }
        }

        return ArmyMovesGenerator.computeStep(state, unit, target);
    }
}