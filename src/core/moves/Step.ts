import { Logger } from "../../polyfish/logger";
import { getTrueUnitAtTile, getUnitAtTile } from "../functions";
import Move, { CallbackResult, MoveType } from "../move";
import UnitMoveGenerator from "../moves";
import { GameState } from "../states";
import { EffectType, UnitType } from "../types";

export default class Step extends Move {
    constructor(src: number, target: number, type: number) {
        super(MoveType.Step, src, target, type);
    }

    execute(state: GameState): CallbackResult {
        const unitType = this.type as UnitType;
        const unit = getUnitAtTile(state, this.src);

        if(!unit) {
            return Logger.illegal(MoveType.Step, `Unit does not exist: ${UnitType[unitType]}`);
        }

        // TODO If a unit moves onto a cloak when in not live mode, then the stepper will override the cloak

        if (state.tiles[this.target]._unitOwner > 0) {
            const unitType = getTrueUnitAtTile(state, this.target)?._unitType!;
            // If cloak is on tile, then it must be revealed
            if(unitType == UnitType.Cloak) {
                if(state.settings.live) {
                    let effectIndex = unit._effects.indexOf(EffectType.Invisible);
                    const cloak = getTrueUnitAtTile(state, this.target)!;
                    cloak._hidden = false;
                    cloak._effects.splice(effectIndex, 1);
                    return {
                        moves: [],
                        undo: () => {
                            cloak._hidden = true;
                            cloak._effects.splice(effectIndex, 0, EffectType.Invisible);
                        }
                    };
                }
                else {
                    return Logger.illegal(MoveType.Step, `${UnitType[unit._unitType]} -> Cloak SUPERPOSITION is required`);
                }
            }
            else {
                return Logger.illegal(MoveType.Step, `${unit._tileIndex} -> ${this.target}, ${UnitType[unit._unitType]} -> ${UnitType[unitType]}`);
            }
        }

        return UnitMoveGenerator.stepCallback(state, unit, this.target);
    }
}