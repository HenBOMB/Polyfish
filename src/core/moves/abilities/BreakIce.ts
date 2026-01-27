import { GameState } from "../../states";
import { AbilityType, EffectType } from "../../types";
import { getTrueUnitAt, hasEffect } from "../../functions";
import { tryRemoveEffect } from "../../actions";
import Ability from "../Ability";

export default class BreakIce extends Ability {
    constructor(target: number) {
        super(null, target, AbilityType.BreakIce);
    }

    execute(state: GameState) {
        const targetIdx = this.getTarget();
        const tile = state.tiles[targetIdx];

        // Collision detection for invisible units
        const otherUnit = getTrueUnitAt(state, targetIdx);
        if (otherUnit && otherUnit.owner !== state.settings.currentPlayerTurnId && hasEffect(otherUnit, EffectType.Invisible)) {
            // Reveal the cloak
            const undoReveal = tryRemoveEffect(state, otherUnit, EffectType.Invisible);
            return {
                rewards: [],
                undo: () => {
                    undoReveal();
                }
            };
        }

        if (tile.frozen) {
            tile.frozen = false;
            // TODO: zorbist hash?
            return {
                rewards: [],
                undo: () => {
                    tile.frozen = true;
                }
            };
        }

        return { rewards: [], undo: () => { } };
    }
}