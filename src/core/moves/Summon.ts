import { Logger } from "../../ai/logger";
import { summonUnit } from "../actions";
import { getPovTribe } from "../functions";
import Move, { CallbackResult } from "../move";
import { MoveType, TribeType } from "../types";
import { UnitSettings } from "../settings/UnitSettings";
import { GameState } from "../states";
import { UnitType } from "../types";

export default class Summon extends Move {
    constructor(src: number, type: number) {
        super(MoveType.Summon, src, null, type);
    }

    execute(state: GameState): CallbackResult {
        const pov = getPovTribe(state);
        const unitType = this.getType<UnitType>();

        if(pov._stars < UnitSettings[unitType].cost) {
            return Logger.illegal(MoveType.Summon, `Cant afford ${UnitType[unitType]} ${pov._stars} / ${UnitSettings[unitType].cost} stars`);
        }

        return summonUnit(state, unitType, this.getSrc(), true)
    }
}