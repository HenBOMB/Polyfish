import { Logger } from "../../polyfish/logger";
import { getEnemyAt, getTrueEnemyAt, getTrueUnitAt, getUnitAt, unInvisible } from "../functions";
import Move, { CallbackResult } from "../move";
import { EffectType, MoveType } from "../types";
import { ArmyMovesGenerator } from "../moves";
import { GameState } from "../states";
import { UnitType } from "../types";
import { ZobristKeys } from "../../zorbist/generateZorbist";
import { xorUnit } from "../../zorbist/hasher";

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
                    const cloak = getTrueUnitAt(state, target)!;
                    const undo = unInvisible(cloak);
                    return {
                        rewards: [],
                        undo,
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

    hashUndo(state: GameState, keys: ZobristKeys) {
        const source = this.getSrc();
        const stepper = getUnitAt(state, source)!;
        state.hash = xorUnit(state.hash, stepper, keys);
    }

    hashApply(state: GameState, keys: ZobristKeys) {
        const target = this.getTarget();
        const enemyCloak = getEnemyAt(state, target);

        // this means the unit tried to move onto a tile where there was an invisible cloak
        // the cloak gets revealed so the pov now knows and must update the entire unit tile
        if(enemyCloak) {
            const stepper = getUnitAt(state, this.getSrc())!;
            state.hash = xorUnit(state.hash, stepper, keys);
            state.hash = xorUnit(state.hash, enemyCloak, keys);
        }
        else {
            state.hash = xorUnit(state.hash, getUnitAt(state, target)!, keys);
        }
    }
}