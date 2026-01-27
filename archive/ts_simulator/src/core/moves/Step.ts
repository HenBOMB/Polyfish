import { getTrueUnitAt, getUnitAt } from "../functions";
import Move, { Branch } from "../move";
import { EffectType, MoveType } from "../types";
import { GameState } from "../states";
import { tryRemoveEffect } from "../actions";
import stepUnit from "../actions/units/Step";

export default class Step extends Move {
    constructor(src: number, target: number) {
        super(MoveType.Step, src, target);
    }

    execute(state: GameState): Branch {
        const unit = getUnitAt(state, this.getSrc())!;
        const target = this.getTarget();

        // If we are stepping over a unit, then it 100% must be an invisble enemy cloak, it must be revealed and the step must be cancelled
        if (state.tiles[target]._unitOwnerID) {
            if(state.settings._areYouSure) {
                const cloak = getTrueUnitAt(state, target)!;
                const enemy = state.tribes[cloak.owner];

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
                throw "Not implemented lol";
                // return Logger.illegal(MoveType.Step, `${UnitType[unit._unitType]} -> Cloak SUPERPOSITION is required`);
            }
        }

        if (!unit) {
            throw Error(`No unit at: ${this.getSrc()} -> ${target}`);
        }

        return stepUnit(state, unit, target);
    }
}