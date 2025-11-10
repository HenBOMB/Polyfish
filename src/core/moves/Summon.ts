import { Logger } from "../../ai/logger";
import summonUnit from "../actions/units/Summon";
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

    execute(state: GameState) {
        const pov = getPovTribe(state);
        const unitType = this.getType<UnitType>();

        if(pov.stars < UnitSettings[unitType].cost) {
            console.trace();
            throw new Error(`${TribeType[pov.type]} cant afford ${UnitType[unitType]} ${pov.stars} / ${UnitSettings[unitType].cost} stars`);
            // return Logger.illegal(MoveType.Summon, );
        }

        return summonUnit(state, unitType, this.getSrc(), true)
    }
}