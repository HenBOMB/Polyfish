import { xorUnit, xorCity } from "../../../zobrist/hasher";
import { getHomeCity, getRealUnitSettings } from "../../functions";
import { UndoCallback } from "../../move";
import { GameState, UnitState } from "../../states";


export default function(state: GameState, removed: UnitState, killer?: UnitState): UndoCallback {
    const pov = state.tribes[removed._owner];
    const tile = state.tiles[removed._tileIndex];
    const oldOwner = removed._owner;
    const cityHome = getHomeCity(state, removed);
    const atIndex = pov._units.findIndex(x => x._tileIndex == removed._tileIndex);
    const settings = getRealUnitSettings(removed);

    xorUnit.set(state, removed);

    pov._units.splice(atIndex, 1);
    tile._unitOwner = 0;

    if (cityHome) {
        xorCity.unitCount(state, cityHome, cityHome._unitCount, cityHome._unitCount - 1);
        cityHome._unitCount--;
    }

    if (killer) {
        xorUnit.kills(state, killer, killer._kills, killer._kills + 1);
        killer._kills++;
        pov._casualties++;
        state.tribes[killer._owner]._kills++;
    }

    if(!removed._meta?.['converted']) {
        pov._score -= 5 * (settings.super ? 10 : settings.cost!);
    }

    return () => {
        if(!removed._meta?.['converted']) {
            pov._score += 5 * (settings.super ? 10 : settings.cost!);
        }
        
        if (killer) {
            state.tribes[killer._owner]._kills--;
            pov._casualties--;
            xorUnit.kills(state, killer, killer._kills, killer._kills - 1);
            killer._kills--;
        }

        if (cityHome) {
            xorCity.unitCount(state, cityHome, cityHome._unitCount, cityHome._unitCount + 1);
            cityHome._unitCount++;
        }

        tile._unitOwner = oldOwner;
        pov._units.splice(atIndex, 0, removed);

        xorUnit.set(state, removed);
    };
}
