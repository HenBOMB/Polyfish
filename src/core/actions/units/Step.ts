import { xorUnit } from "../../../zobrist/hasher";
import { endUnitTurn, splashDamageArea, tryAddEffect } from "../../actions";
import attackUnit from "./Attack";
import removeUnit from "./Remove";
import { discoverTiles } from "../DiscoverTiles";
import { isSkilledIn, getStructureAt, isAquaticOrCanFly, isWaterTerrain, hasEffect, getEnemiesInRange, getTrueUnitAt } from "../../functions";
import { Branch, UndoCallback } from "../../move";
import { GameState, UnitState } from "../../states";
import { SkillType, UnitType, EffectType, StructureType } from "../../types";
import { freezeArea, tryRemoveEffect } from "../../actions";

export default function stepUnit(state: GameState, stepper: UnitState, toTileIndex: number, involuntary = false): Branch {
    if (!stepper) {
        console.trace();
        console.log(toTileIndex, state.tribes[state.settings.currentPlayerTurnId].units, involuntary);
        throw 'no unit';
    }
    const chain: UndoCallback[] = [];
    const rewards = [];
    const movedBefore = stepper.moved;
    const oldTileIndex = stepper.coords.idx;
    const oldType = stepper.type;
    const oldPassenger = stepper.passengerType;
    const oldAttack = stepper.attacked;

    // Collision detection for invisible units
    const otherUnit = getTrueUnitAt(state, toTileIndex);
    if (otherUnit && otherUnit.owner !== stepper.owner && hasEffect(otherUnit, EffectType.Invisible)) {
        // Reveal the cloak
        const undoReveal = tryRemoveEffect(state, otherUnit, EffectType.Invisible);

        return {
            rewards: [],
            undo: () => {
                undoReveal();
            }
        };
    }

    // // TODO; this is not how prev works, it must be applies at the end of the turn
    // stepper.prevX = iX;
    // stepper.prevY = iY;
    // xor out the current unit
    xorUnit.set(state, stepper);

    delete state.tiles[stepper.coords.idx]._unitOwnerID;
    stepper.coords.setAt(toTileIndex, state);
    state.tiles[stepper.coords.idx]._unitOwnerID = stepper.owner;

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

    const struct = getStructureAt(state, toTileIndex);
    const isPort = struct === StructureType.Port;

    // TODO: Doubt - if a non invis cloak moves onto a port, does it become invis?

    // ! Embark ! //
    // If a non aquatic unit is moving to our port, place into boat
    if (isPort && !isAquaticOrCanFly(stepper)) {
        switch (stepper.type) {
            case UnitType.Cloak:
                stepper.type = UnitType.Dinghy;
                break;
            case UnitType.Dagger:
                stepper.type = UnitType.Pirate;
                break;
            case UnitType.Giant:
                stepper.type = UnitType.Juggernaut;
                break;
            default:
                stepper.type = UnitType.Raft;
                stepper.passengerType = oldType;
                break;
        }
    }

    // ! Disembark ! //
    // Carry allows a unit to carry another unit inside
    // A unit with the carry skill can move to a land tile adjacent to water
    // Doing so releases the unit it was carrying and ends the unit's turn
    else if (isSkilledIn(stepper, SkillType.Carry) && !isWaterTerrain(state.tiles[stepper.coords.idx])) {
        stepper.passengerType = undefined;
        switch (stepper.type) {
            case UnitType.Dinghy:
                stepper.type = UnitType.Cloak;
                break;
            case UnitType.Pirate:
                stepper.type = UnitType.Dagger;
                break;
            case UnitType.Juggernaut:
                stepper.type = UnitType.Giant;
                break;
            default:
                stepper.type = oldPassenger!;
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
        stepper.attacked = false;
    }

    // ! Persist ! //
    // Allows a unit to continue attacking if it killed its target
    // if (isSkilledIn(stepper, SkillType.Persist)) {
    //     chain.push(tryAddEffect(state, stepper, EffectType.Persistent));
    // }

    // xor back out true attack
    if (!stepper.attacked) {
        xorUnit.attacked(state, stepper);
    }

    // xor in the new unit
    xorUnit.set(state, stepper);

    // ! Move ???
    stepper.moved = true;

    return {
        rewards,
        undo: () => {
            stepper.moved = false;

            // xor out the new unit
            xorUnit.set(state, stepper);

            if (!stepper.attacked) {
                xorUnit.attacked(state, stepper);
            }

            stepper.attacked = oldAttack;

            stepper.type = oldType;
            stepper.passengerType = oldPassenger;

            chain.reverse().forEach(x => x());

            delete state.tiles[stepper.coords.idx]._unitOwnerID;
            stepper.coords.setAt(oldTileIndex, state);
            state.tiles[stepper.coords.idx]._unitOwnerID = stepper.owner;

            // xor in the current unit
            xorUnit.set(state, stepper);
        }
    };
}
