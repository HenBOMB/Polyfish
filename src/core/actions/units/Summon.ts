import { xorUnit, xorCity } from "../../../zobrist/hasher";
import { freezeArea, spendStars } from "../../actions";
import { discoverTiles } from "../DiscoverTiles";
import pushUnit from "./Push";
import { getPovTribe, isSkilledIn, getHomeCity } from "../../functions";
import { CallbackResult, UndoCallback } from "../../move";
import { UnitSettings } from "../../settings/UnitSettings";
import { GameState, UnitState } from "../../states";
import { UnitType, SkillType } from "../../types";

export default function(state: GameState, unitType: UnitType, spawnTileIndex: number, costs = false, forceIndependent = false): CallbackResult {
    const pov = getPovTribe(state);
    const settings = UnitSettings[unitType];
    const health = UnitSettings[unitType].health!;

    const spawnTile = state.tiles[spawnTileIndex];

    // Push occupied unit away (if any)
    let resultPush = pushUnit(state, spawnTile.tileIndex);

    const oldUnitOwner = spawnTile._unitOwner;

    const undoPurchase = costs ? spendStars(state, settings.cost) : () => { };

    const spawnedUnit = {
        _unitType: unitType,
        _health: health * 10,
        _kills: 0,
        prevX: -1,
        prevY: -1,
        direction: 0,
        _owner: pov.owner,
        createdTurn: state.settings._turn,
        // If its not from a ruin or special unit
        _homeIndex: forceIndependent || isSkilledIn(unitType, SkillType.Independent) || !costs ? -1 : spawnTileIndex,
        _tileIndex: spawnTileIndex,
        _effects: new Set(),
        _attacked: true,
        _moved: true,
    } as UnitState;

    xorUnit.set(state, spawnedUnit);

    pov._units.push(spawnedUnit);

    spawnTile._unitOwner = spawnedUnit._owner;

    const cityHome = forceIndependent ? null : getHomeCity(state, spawnedUnit);

    if (cityHome) {
        xorCity.unitCount(state, cityHome, cityHome._unitCount, cityHome._unitCount + 1);
        cityHome._unitCount++;
    }

    const resultDiscover = discoverTiles(state, spawnedUnit);

    const undoFrozen: UndoCallback = isSkilledIn(spawnedUnit, SkillType.AutoFreeze, SkillType.FreezeArea) ?
        freezeArea(state, spawnedUnit) : () => { };

    pov._score += 5 * (settings.super ? 10 : settings.cost!);

    return {
        rewards: [...(resultDiscover?.rewards || []), ...(resultPush?.rewards || [])],
        undo: () => {
            pov._score -= 5 * (settings.super ? 10 : settings.cost!);

            undoFrozen();
            resultDiscover?.undo();
            if (cityHome) {
                xorCity.unitCount(state, cityHome, cityHome._unitCount, cityHome._unitCount - 1);
                cityHome._unitCount--;
            }
            spawnTile._unitOwner = oldUnitOwner;
            undoPurchase();
            state.settings.unitIdx--;
            pov._units.pop();
            resultPush?.undo();
            xorUnit.set(state, spawnedUnit);
        }
    };
}

