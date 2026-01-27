import { xorUnit } from "../../../zobrist/hasher";
import { getRealUnitSettings } from "../../functions";
import { UndoCallback } from "../../move";
import { GameState, UnitState } from "../../states";


export default function(state: GameState, removed: UnitState, killer?: UnitState): UndoCallback {
    const oldOwner = removed.owner;
    const pov = state.tribes[oldOwner];
    const tile = state.tiles[removed.coords.idx];
    const atIndex = pov.units.findIndex(x => x.coords.idx == removed.coords.idx);
    const settings = getRealUnitSettings(removed);

    xorUnit.set(state, removed);

    pov.units.splice(atIndex, 1);
    delete tile._unitOwnerID;

    if (killer) {
        xorUnit.kills(state, killer, killer.kills, killer.kills + 1);
        killer.kills++;
        pov.casualties++;
        state.tribes[killer.owner].kills++;
    }

    if(!removed._meta?.['converted']) {
        pov.score -= 5 * (settings.super ? 10 : settings.cost!);
    }

    return () => {
        if(!removed._meta?.['converted']) {
            pov.score += 5 * (settings.super ? 10 : settings.cost!);
        }
        
        if (killer) {
            state.tribes[killer.owner].kills--;
            pov.casualties--;
            xorUnit.kills(state, killer, killer.kills, killer.kills - 1);
            killer.kills--;
        }

        tile._unitOwnerID = oldOwner;
        pov.units.splice(atIndex, 0, removed);

        xorUnit.set(state, removed);
    };
}
