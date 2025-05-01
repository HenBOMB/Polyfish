import { Logger } from "../../polyfish/logger";
import { summonUnit } from "../actions";
import { getPovTribe, getUnitAtTile } from "../functions";
import Move, { CallbackResult, MoveType } from "../move";
import { UnitSettings } from "../settings/UnitSettings";
import { GameState } from "../states";
import { UnitType } from "../types";

export default class None extends Move {
    constructor(src: number, target: number, type: number) {
        super(MoveType.None, src, target, type as keyof typeof UnitSettings);
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

        return {
            moves: [],
            undo: () => {

            },
        };
    }
}