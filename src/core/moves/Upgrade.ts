import { getUnitAt } from "../functions";
import Move, { CallbackResult } from "../move";
import { MoveType } from "../types";
import { UnitSettings } from "../settings/UnitSettings";
import { GameState } from "../states";
import { UnitType } from "../types";
import { spendStars } from "../actions";
import { xorUnit } from "../../zobrist/hasher";

export default class Upgrade extends Move {
    constructor(from: number, type: number) {
        super(MoveType.Summon, from, null, type);
    }

    execute(state: GameState) {
        const upgradeType = this.getType<UnitType>();
        const unit = getUnitAt(state, this.getSrc())!;
        const oldUnitType = unit.type;

        xorUnit.passenger(state, unit, UnitType.None, oldUnitType);
        xorUnit.type(state, unit, oldUnitType, upgradeType);

        unit.passengerType = oldUnitType;
        unit.type = upgradeType;

        const undoStars = spendStars(state, UnitSettings[upgradeType].cost);

        return {
            rewards: [],
            undo: () => {
                xorUnit.type(state, unit, upgradeType, oldUnitType);
                xorUnit.passenger(state, unit, oldUnitType, UnitType.None);
                undoStars();
                unit.passengerType = undefined;
                unit.type = oldUnitType;
            },
        };
    }
}