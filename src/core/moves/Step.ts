import { Logger } from "../../polyfish/logger";
import { getPovTribe, getTrueUnitAtTile, getUnitAtTile } from "../functions";
import Move, { CallbackResult, MoveType } from "../move";
import UnitMoveGenerator from "../moves";
import { UnitSettings } from "../settings/UnitSettings";
import { GameState } from "../states";
import { EffectType, UnitType } from "../types";

export default class Step extends Move {
    constructor(src: number, target: number, type: number) {
        super(MoveType.Step, src, target, type);
    }

    execute(state: GameState): CallbackResult {
        const pov = getPovTribe(state);
        const unitType = this.type as UnitType;

        if (pov._stars < UnitSettings[unitType].cost) {
            return Logger.illegal(MoveType.None, `Oppsie`);
        }

        const unit = getUnitAtTile(state, this.src);

        if(!unit) {
            return Logger.illegal(MoveType.None, `Upgrade - Unit does not exist: ${UnitType[unitType]}`);
        }

        // TODO Cloak is on tile will cause a collision if re-simulating the enemy moves after our simulated move
        // Superposition is required!

        if (state.tiles[this.target]._unitOwner > 0) {
            const cloak = getTrueUnitAtTile(state, this.target)?._unitType!;
            // If cloak is on tile, then it must be revealed
            if(cloak == UnitType.Cloak) {
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
                    return Logger.illegal(MoveType.Step, `Unit -> Cloak SUPERPOSITION is required`);
                }
            }
            else {
                return Logger.illegal(MoveType.Step, `Step ${unit._tileIndex} -> ${this.target}, ${UnitType[unit._unitType]} -> ${UnitType[cloak]}`);
            }
        }

        return UnitMoveGenerator.stepCallback(state, unit, this.target);
    }
}