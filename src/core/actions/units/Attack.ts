import { calculateDistance } from "../../../ai/gmath";
import { splashDamageArea, tryAddEffect } from "../../actions";
import stepUnit from "./Step";
import { calculateAttack, calculateCombat, getUnitRange, isSteppable, isSkilledIn } from "../../functions";
import { CallbackResult, UndoCallback } from "../../move";
import { GameState, UnitState } from "../../states";
import { SkillType, EffectType } from "../../types";
import removeUnit from "./Remove";

export default function attackUnit(state: GameState, attacker: UnitState | number, defender: UnitState, attackerPov?: UnitState): CallbackResult {
    const undoChain: UndoCallback[] = [];
    const rewards = [];

    if (typeof attacker == 'number') {
        const atk = calculateAttack(state, attacker, defender);

        defender.health -= atk;

        if (defender.health <= 0) {
            undoChain.push(removeUnit(state, defender, attackerPov));
        }

        undoChain.push(() => {
            defender.health += atk;
        });
    }
    else {
        const result = calculateCombat(state, attacker, defender);

        defender.health -= result.attackDamage;

        undoChain.push(() => {
            defender.health += result.attackDamage;
        });

        // Deal splash damage
        if (result.splashDamage > 0) {
            undoChain.push(splashDamageArea(state, attacker, result.splashDamage));
        }

        // We killed their unit
        if (defender.health <= 0) {
            undoChain.push(removeUnit(state, defender, attacker));
            // Move to the enemy position, if not a ranged unit
            if (getUnitRange(attacker) < 2 && isSteppable(state, attacker, defender.coords.idx)) {
                // TODO: Bug here with persistent knight
                const result = stepUnit(state, attacker, defender.coords.idx, true)!;
                rewards.push(...result.rewards);
                undoChain.push(result.undo);
            }
        }

        // Retaliate
        else {
            // If we have have freeze, then they cant retaliate
            if (isSkilledIn(attacker, SkillType.Freeze)) {
                undoChain.push(tryAddEffect(state, defender, EffectType.Frozen));
            }

            // If we're attacking with range
            else if (getUnitRange(attacker) > 1 && result.defenseDamage > 0) {
                const dist = calculateDistance(attacker.coords.idx, defender.coords.idx, state.settings.size);
                // if defender cant reach us, they cant retaliate
                if (dist > getUnitRange(defender)) {
                    result.defenseDamage = 0;
                }


                // technically not cheating
                // if defender that cannot see us, they cant retaliate
                else if (state.settings._areYouSure) {
                    if (!state.tiles[defender.coords.idx].explorers.has(defender.owner)) {
                        result.defenseDamage = 0;
                    }
                }
            }

            if (result.defenseDamage > 0) {
                attacker.health -= result.defenseDamage;

                undoChain.push(() => {
                    attacker.health += result.defenseDamage;
                });

                // Our unit died
                if (attacker.health <= 0) {
                    undoChain.push(removeUnit(state, attacker, defender));
                }
            }
        }
    }

    return {
        rewards: rewards,
        undo: () => {
            undoChain.reverse().forEach(x => x());
        }
    };
}
