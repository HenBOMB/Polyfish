import { Logger } from "../../polyfish/logger";
import { summonUnit } from "../actions";
import { getPovTribe, getUnitAtTile } from "../functions";
import Move, { CallbackResult, MoveType } from "../move";
import { UnitSettings } from "../settings/UnitSettings";
import { GameState } from "../states";
import { UnitType } from "../types";

export default class Summon extends Move {
    constructor(src: number, target: number, type: number) {
        super(MoveType.Summon, src, target, type);
    }

    execute(state: GameState): CallbackResult {
        const pov = getPovTribe(state);
        const unitType = this.type as UnitType;

        if (pov._stars < UnitSettings[unitType].cost) {
            return Logger.illegal(MoveType.Summon, `Cant afford ${UnitType[unitType]} ${pov._stars} / ${UnitSettings[unitType].cost} stars`);
        }

        return {
            moves: [],
            undo: summonUnit(state, unitType, this.src, true),
        };
    }
}