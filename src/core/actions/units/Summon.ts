import { xorUnit, xorCity } from "../../../zobrist/hasher";
import { freezeArea, spendStars } from "../../actions";
import { discoverTiles } from "../DiscoverTiles";
import pushUnit from "./Push";
import { getPovTribe, isSkilledIn, getHomeCity } from "../../functions";
import { Branch } from "../../move";
import { UnitSettings } from "../../settings/UnitSettings";
import { Coords, GameState, UnitState } from "../../states";
import { UnitType, SkillType } from "../../types";

export default function summonUnit(state: GameState, unitType: UnitType, spawnTileIndex: number, costs = false, forceIndependent = false): Branch {
    const pov = getPovTribe(state);
    const settings = UnitSettings[unitType];
    const health = UnitSettings[unitType].health!;

    const spawnTile = state.tiles[spawnTileIndex];

    // Push occupied unit away (if any)
    const resultPush = pushUnit(state, spawnTile.coords.idx);

    const oldUnitOwner = spawnTile._unitOwnerID;

    const undoPurchase = costs ? spendStars(state, settings.cost) : () => { };
    
    const indepentent = forceIndependent || isSkilledIn(unitType, SkillType.Independent);

    const spawnedUnit: UnitState = {
        type: unitType,
        health: health * 10,
        prevCoords: new Coords(),
        direction: 0,
        veteran: false,
        kills: 0,
        createdTurn: state.settings.turn,
        owner: pov.id,
        // If its not from a ruin or special unit
        homeCoords: indepentent? undefined : new Coords(spawnTileIndex, state),
        coords: new Coords(spawnTileIndex, state),
        moved: true,
        attacked: true,
        effects: new Set(),
    };

    xorUnit.set(state, spawnedUnit);

    pov.units.push(spawnedUnit);

    spawnTile._unitOwnerID = spawnedUnit.owner;

    const resultDiscover = discoverTiles(state, spawnedUnit);

    const undoFrozen = isSkilledIn(spawnedUnit, SkillType.AutoFreeze, SkillType.FreezeArea) ?
        freezeArea(state, spawnedUnit) : () => { };

    pov.score += 5 * (settings.super ? 10 : settings.cost!);

    return {
        rewards: [...resultDiscover.rewards, ...(resultPush?.rewards || [])],
        undo: () => {
            pov.score -= 5 * (settings.super ? 10 : settings.cost!);
            undoFrozen();
            resultDiscover.undo();
            spawnTile._unitOwnerID = oldUnitOwner;
            pov.units.pop();
            undoPurchase();
            resultPush?.undo();
            xorUnit.set(state, spawnedUnit);
        }
    };
}

