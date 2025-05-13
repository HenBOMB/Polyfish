import { Logger } from "../../polyfish/logger";
import { getPovTribe, getUnitAt } from "../functions";
import Move, { CallbackResult } from "../move";
import { MoveType } from "../types";
import { UnitSettings } from "../settings/UnitSettings";
import { GameState } from "../states";
import { UnitType } from "../types";
import { spendStars } from "../actions";

export default class Upgrade extends Move {
    constructor(from: number, type: number) {
        super(MoveType.Summon, from, null, type);
    }

    execute(state: GameState): CallbackResult {
        const pov = getPovTribe(state);
        const upgradeType = this.getType<UnitType>();

        if (pov._stars < UnitSettings[upgradeType].cost) {
            return Logger.illegal(MoveType.None, `Upgrade - Cant afford ${UnitType[upgradeType]} ${pov._stars} / ${UnitSettings[upgradeType].cost} stars`);
        }

        const unit = getUnitAt(state, this.getSrc());

        if(!unit) {
            return Logger.illegal(MoveType.None, `Upgrade - Unit does not exist: ${UnitType[upgradeType]}`);
        }

        if(upgradeType != UnitType.Raft) {
            return Logger.illegal(MoveType.None, `Upgrade - Unit must be a Raft, got: ${UnitType[upgradeType]}`);
        }
        
        const oldUnitType = unit._unitType;

        unit._passenger = oldUnitType;
        unit._unitType = upgradeType;
        const undoStars = spendStars(state, UnitSettings[upgradeType].cost);

        return {
            rewards: [],
            undo: () => {
                undoStars();
                unit._passenger = undefined;
                unit._unitType = oldUnitType;
            },
        };
    }
}