import { getPovTribe, getEnemiesNearTile, hasEffect, getEnemyAt, getStarExchange, getAdjacentIndexes, getTrueEnemyAt } from "./functions";
import { UndoCallback } from "./move";
import { GameState, UnitState } from "./states";
import { TerrainType, EffectType } from "./types";
import { xorPlayer, xorTile, xorUnit } from "../zobrist/hasher";
import attackUnit from "./actions/units/Attack";

/**
 * Attempts to discover any undiscovered tribes and rewards with exchange
 * @param state
 * @returns if the disovery was successfull
 */
export function tryDiscoverRewardOtherTribes(state: GameState): UndoCallback {
	const us = getPovTribe(state);

	// Already discovered all the other tribes
	if (us.knownPlayers.size == state.settings._maxTribeCount - 1) {
		return () => { };
	}

	const chain: UndoCallback[] = [];

	// Try to meet new tribes, if they they have been seen and not discovered
	for (const x in state._visibleTiles) {
		// If we can see any other tribe's unit, we have met them
		const standing = getEnemyAt(state, Number(x));
		const them = standing?.owner;
		if (them && !us.knownPlayers.has(them)) {
			us.knownPlayers.add(them);
			chain.unshift(gainStars(state, getStarExchange(state, them)));
			chain.unshift(() => {
				us.knownPlayers.delete(them);
			});
		}
	}

	return () => {
		chain.forEach(x => x());
	};
}

export function modifyTerrain(state: GameState, idx: number, terrainType: TerrainType): UndoCallback {
    const tile = state.tiles[idx];
    const oTerrainType = tile.type;
    
    xorTile.terrain(state, idx, oTerrainType, terrainType);
    tile.type = terrainType;

    return () => {
        tile.type = oTerrainType;
        xorTile.terrain(state, idx, terrainType, oTerrainType);
    }
}

export function gainStars(state: GameState, amount: number): UndoCallback {
    return spendStars(state, -amount);
}

export function spendStars(state: GameState, amount: number): UndoCallback {
    if (!amount) {
        console.trace()
        throw "Cannot spend 0 stars";
        return () => {};
    }
    
    const pov = getPovTribe(state);

    xorPlayer.stars(pov, pov.stars);
    pov.stars -= amount;
    xorPlayer.stars(pov, pov.stars);

    return () => {
        xorPlayer.stars(pov, pov.stars);
        pov.stars += amount;
        xorPlayer.stars(pov, pov.stars);
    }
}

export function tryAddEffect(state: GameState, unit: UnitState, effect: EffectType): UndoCallback {
    if (hasEffect(unit, effect)) {
        return () => { };
    }
    xorUnit.effect(state, unit, effect);
    unit.effects.add(effect);
    return () => {
        unit.effects.delete(effect);
        xorUnit.effect(state, unit, effect);
    }
}

export function tryRemoveEffect(state: GameState, unit: UnitState, effect: EffectType): UndoCallback {
    if (!hasEffect(unit, effect)) {
        return () => { };
    }
    xorUnit.effect(state, unit, effect);
    unit.effects.delete(effect);
    return () => {
        unit.effects.add(effect);
        xorUnit.effect(state, unit, effect);
    }
}

export function endUnitTurn(state: GameState, unit: UnitState): UndoCallback {
    const moved = unit.moved;
    const attacked = unit.attacked;

    if (!attacked) {
        xorUnit.attacked(state, unit);
        unit.attacked = true;
    }

    if (!moved) {
        xorUnit.moved(state, unit);
        unit.moved = true;
    }

    return () => {
        if (!attacked) {
            xorUnit.attacked(state, unit);
            unit.attacked = false;
        }

        if (!moved) {
            xorUnit.moved(state, unit);
            unit.moved = false;
        }
    }
}

export function startUnitTurn(state: GameState, unit: UnitState): UndoCallback {
    const moved = unit.moved;
    const attacked = unit.attacked;

    if (attacked) {
        xorUnit.attacked(state, unit);
        unit.attacked = false;
    }

    if (moved) {
        xorUnit.moved(state, unit);
        unit.moved = false;
    }

    return () => {
        if (attacked) {
            xorUnit.attacked(state, unit);
            unit.attacked = true;
        }

        if (moved) {
            xorUnit.moved(state, unit);
            unit.moved = true;
        }
    }
}

export function splashDamageArea(state: GameState, attacker: UnitState, atk: number): UndoCallback {
    const undoChain = getEnemiesNearTile(state, attacker.coords.idx)
        .map(enemy => attackUnit(state, atk, enemy, attacker)?.undo!);
    return () => {
        undoChain.forEach(x => x());
    }
}

export function freezeArea(state: GameState, freezer: UnitState): UndoCallback {
    const chain: UndoCallback[] = [];
    const adjacent = getAdjacentIndexes(state, freezer.coords.idx, 1, false, true);

    for (let i = 0; i < adjacent.length; i++) {
        const tile = state.tiles[adjacent[i]];
        const occupied = getTrueEnemyAt(state, tile.coords.idx, freezer.owner);

        // Freeze any adjacent enemy unit
        if (occupied) {
            chain.push(tryAddEffect(state, occupied, EffectType.Frozen));
        }

        // Freeze any adjacent freezable tiles
        if (tile.type == TerrainType.Water || tile.type == TerrainType.Ocean) {
            chain.push(modifyTerrain(state, tile.coords.idx, TerrainType.Ice));
        }
    }

    return () => {
        chain.forEach(x => x());
    };
}


