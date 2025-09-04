import { xorUnit } from "../../../zobrist/hasher";
import { endUnitTurn, splashDamageArea, tryAddEffect } from "../../actions";
import { discoverTiles } from "../DiscoverTiles";
import { isSkilledIn, getStructureAt, isAquaticOrCanFly, isWaterTerrain, hasEffect, getEnemiesInRange } from "../../functions";
import { CallbackResult, UndoCallback } from "../../move";
import { GameState, UnitState } from "../../states";
import { SkillType, UnitType, EffectType } from "../../types";
import { freezeArea } from "../../actions";


export default function(state: GameState, stepper: UnitState, toTileIndex: number, involuntary = false): CallbackResult {
    const chain: UndoCallback[] = [];
    const rewards = [];
    const movedBefore = stepper._moved;
    const oldTileIndex = stepper._tileIndex;
    const oldType = stepper._unitType;
    const oldPassenger = stepper._passenger;

    // // TODO; this is not how prev works, it must be applies at the end of the turn
    // stepper.prevX = iX;
    // stepper.prevY = iY;
    // xor out the current unit
    xorUnit.set(state, stepper);

    state.tiles[stepper._tileIndex]._unitOwner = 0;
    stepper._tileIndex = toTileIndex;
    state.tiles[stepper._tileIndex]._unitOwner = stepper._owner;

    // TODO what other skills are missing?
    // Discover terrain
    const resultDiscover = discoverTiles(state, stepper)!;
    rewards.push(...resultDiscover.rewards);
    chain.push(resultDiscover.undo);

    // Units with skate do not loose their turn when pushed
    if (!involuntary || !isSkilledIn(stepper, SkillType.Skate)) {
        // xor to true
        chain.push(endUnitTurn(state, stepper));
    }

    // ! Stomp ! //
    if (isSkilledIn(stepper, SkillType.Stomp)) {
        chain.push(splashDamageArea(state, stepper, 4));
    }

    // ! AutoFreeze //
    if (isSkilledIn(stepper, SkillType.AutoFreeze, SkillType.FreezeArea)) {
        chain.push(freezeArea(state, stepper));
    }

    // ! Embark ! //
    // If a non aquatic unit is moving to our port, place into boat
    if (getStructureAt(state, toTileIndex) && !isAquaticOrCanFly(stepper)) {
        switch (stepper._unitType) {
            case UnitType.Cloak:
                stepper._unitType = UnitType.Dinghy;
                break;
            case UnitType.Dagger:
                stepper._unitType = UnitType.Pirate;
                break;
            case UnitType.Giant:
                stepper._unitType = UnitType.Juggernaut;
                break;
            default:
                stepper._unitType = UnitType.Raft;
                stepper._passenger = oldType;
                break;
        }
    }






    // ! Disembark ! //
    // Carry allows a unit to carry another unit inside
    // A unit with the carry skill can move to a land tile adjacent to water
    // Doing so releases the unit it was carrying and ends the unit's turn
    else if (isSkilledIn(stepper, SkillType.Carry) && !isWaterTerrain(state.tiles[stepper._tileIndex])) {
        stepper._passenger = undefined;
        switch (stepper._unitType) {
            case UnitType.Dinghy:
                stepper._unitType = UnitType.Cloak;
                break;
            case UnitType.Pirate:
                stepper._unitType = UnitType.Dagger;
                break;
            case UnitType.Juggernaut:
                stepper._unitType = UnitType.Giant;
                break;
            default:
                stepper._unitType = oldPassenger!;
                break;
        }
    }




    // ! Hide ! //
    // Going stealth mode uses up our attack
    else if (isSkilledIn(stepper, SkillType.Hide) && !hasEffect(stepper, EffectType.Invisible)) {
        chain.push(tryAddEffect(state, stepper, EffectType.Invisible));
    }





    // ! Dash ! //
    // Allows a unit to attack after moving if there are any enemies in range
    // And if it HAS moved before (this avoids infinite move -> attack loop)
    else if (!involuntary && !movedBefore && isSkilledIn(stepper, SkillType.Dash) && getEnemiesInRange(state, stepper).length > 0) {
        stepper._attacked = false;
    }

    // xor back out true attack
    if (!stepper._attacked) {
        xorUnit.attacked(state, stepper);
    }

    // xor in the new unit
    xorUnit.set(state, stepper);

    return {
        rewards,
        undo: () => {
            // xor out the new unit
            xorUnit.set(state, stepper);

            if (!stepper._attacked) {
                xorUnit.attacked(state, stepper);
            }

            stepper._unitType = oldType;
            stepper._passenger = oldPassenger;

            chain.reverse().forEach(x => x());

            state.tiles[stepper._tileIndex]._unitOwner = 0;
            stepper._tileIndex = oldTileIndex;
            state.tiles[stepper._tileIndex]._unitOwner = stepper._owner;

            // xor in the current unit
            xorUnit.set(state, stepper);
        }
    };
}
