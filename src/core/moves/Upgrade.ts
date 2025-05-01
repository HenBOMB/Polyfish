import { Logger } from "../../polyfish/logger";
import { getPovTribe, getUnitAtTile } from "../functions";
import Move, { CallbackResult, MoveType } from "../move";
import { UnitSettings } from "../settings/UnitSettings";
import { GameState } from "../states";
import { UnitType } from "../types";

export default class Upgrade extends Move {
    constructor(src: number, target: number, type: number) {
        super(MoveType.Summon, src, target, type);
    }

    execute(state: GameState): CallbackResult {
        const pov = getPovTribe(state);
        const upgradeType = this.type as UnitType;

        if (pov._stars < UnitSettings[upgradeType].cost) {
            return Logger.illegal(MoveType.None, `Upgrade - Cant afford ${UnitType[upgradeType]} ${pov._stars} / ${UnitSettings[upgradeType].cost} stars`);
        }

        const unit = getUnitAtTile(state, this.src);

        if(!unit) {
            return Logger.illegal(MoveType.None, `Upgrade - Unit does not exist: ${UnitType[upgradeType]}`);
        }

        if(upgradeType != UnitType.Raft) {
            return Logger.illegal(MoveType.None, `Upgrade - Unit does not exist: ${UnitType[upgradeType]}`);
        }
        
        const oldUnitType = unit._unitType;

        unit._passenger = oldUnitType;
        unit._unitType = upgradeType;
        pov._stars -= UnitSettings[upgradeType].cost;

        return {
            moves: [],
            undo: () => {
                unit._passenger = undefined;
                unit._unitType = oldUnitType;
                pov._stars += UnitSettings[upgradeType].cost;
            },
        };
    }
}