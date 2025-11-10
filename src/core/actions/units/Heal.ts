import { tryRemoveEffect } from "../../actions";
import { hasEffect, getMaxHealth } from "../../functions";
import { UndoCallback } from "../../move";
import { GameState, UnitState } from "../../states";
import { EffectType } from "../../types";


export default function(state: GameState, unit: UnitState, amount: number): UndoCallback {
    if (hasEffect(unit, EffectType.Poison)) {
        return tryRemoveEffect(state, unit, EffectType.Poison);
    }

    const oldHealth = unit.health;
    
    unit.health += amount;
    unit.health = Math.min(unit.health, getMaxHealth(unit));

    return () => {
        unit.health = oldHealth;
    };
}
