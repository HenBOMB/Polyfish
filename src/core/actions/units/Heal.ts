import { tryRemoveEffect } from "../../actions";
import { hasEffect, getMaxHealth } from "../../functions";
import { UndoCallback } from "../../move";
import { GameState, UnitState } from "../../states";
import { EffectType } from "../../types";


export default function(state: GameState, unit: UnitState, amount: number): UndoCallback {
    if (hasEffect(unit, EffectType.Poison)) {
        return tryRemoveEffect(state, unit, EffectType.Poison);
    }

    const oldHealth = unit._health;
    
    unit._health += amount;
    unit._health = Math.min(unit._health, getMaxHealth(unit));

    return () => {
        unit._health = oldHealth;
    };
}
