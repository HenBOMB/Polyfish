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
	if (us._knownPlayers.size == state.settings.tribeCount - 1) {
		return () => { };
	}

	const chain: UndoCallback[] = [];

	// Try to meet new tribes, if they they have been seen and not discovered
	for (const x in state._visibleTiles) {
		// If we can see any other tribe's unit, we have met them
		const standing = getEnemyAt(state, Number(x));
		const them = standing?._owner;
		if (them && !us._knownPlayers.has(them)) {
			us._knownPlayers.add(them);
			chain.unshift(gainStars(state, getStarExchange(state, them)));
			chain.unshift(() => {
				us._knownPlayers.delete(them);
			});
		}
	}

	return () => {
		chain.forEach(x => x());
	};
}

export function modifyTerrain(state: GameState, tileIndex: number, terrainType: TerrainType): UndoCallback {
    const tile = state.tiles[tileIndex];
    const oTerrainType = tile.terrainType;
    
    xorTile.terrain(state, tileIndex, oTerrainType, terrainType);
    tile.terrainType = terrainType;

    return () => {
        tile.terrainType = oTerrainType;
        xorTile.terrain(state, tileIndex, terrainType, oTerrainType);
    }
}

export function gainStars(state: GameState, amount: number): UndoCallback {
    return spendStars(state, -amount);
}

export function spendStars(state: GameState, amount: number): UndoCallback {
    if(!amount) return () => {};
    const pov = getPovTribe(state);

    xorPlayer.stars(pov, pov._stars);
    pov._stars -= amount;
    xorPlayer.stars(pov, pov._stars);

    return () => {
        xorPlayer.stars(pov, pov._stars);
        pov._stars += amount;
        xorPlayer.stars(pov, pov._stars);
    }
}

export function tryAddEffect(state: GameState, unit: UnitState, effect: EffectType): UndoCallback {
    if(hasEffect(unit, effect)) {
        return () => { };
    }
    xorUnit.effect(state, unit, effect);
    unit._effects.add(effect);
    return () => {
        unit._effects.delete(effect);
        xorUnit.effect(state, unit, effect);
    }
}

export function tryRemoveEffect(state: GameState, unit: UnitState, effect: EffectType): UndoCallback {
    if(!hasEffect(unit, effect)) {
        return () => { };
    }
    xorUnit.effect(state, unit, effect);
    unit._effects.delete(effect);
    return () => {
        unit._effects.add(effect);
        xorUnit.effect(state, unit, effect);
    }
}

export function setUnitMove(state: GameState, unit: UnitState): UndoCallback {
    xorUnit.moved(state, unit);

    return () => {
        xorUnit.moved(state, unit);
    }
}

export function setUnitAttack(state: GameState, unit: UnitState): UndoCallback {
    xorUnit.attacked(state, unit);

    return () => {
        xorUnit.attacked(state, unit);
    }
}

export function endUnitTurn(state: GameState, unit: UnitState): UndoCallback {
    const moved = unit._moved;
    const attacked = unit._attacked;

    if(!attacked) {
        xorUnit.attacked(state, unit);
        unit._attacked = true;
    }

    if(!moved) {
        xorUnit.moved(state, unit);
        unit._moved = true;
    }

    return () => {
        if(!attacked) {
            xorUnit.attacked(state, unit);
            unit._attacked = false;
        }

        if(!moved) {
            xorUnit.moved(state, unit);
            unit._moved = false;
        }
    }
}

export function startUnitTurn(state: GameState, unit: UnitState): UndoCallback {
    const moved = unit._moved;
    const attacked = unit._attacked;

    if(attacked) {
        xorUnit.attacked(state, unit);
        unit._attacked = false;
    }

    if(moved) {
        xorUnit.moved(state, unit);
        unit._moved = false;
    }

    return () => {
        if(attacked) {
            xorUnit.attacked(state, unit);
            unit._attacked = true;
        }

        if(moved) {
            xorUnit.moved(state, unit);
            unit._moved = true;
        }
    }
}

export function splashDamageArea(state: GameState, attacker: UnitState, atk: number): UndoCallback {
    const undoChain = getEnemiesNearTile(state, attacker._tileIndex)
        .map(enemy => attackUnit(state, atk, enemy, attacker)?.undo!);
    return () => {
        undoChain.forEach(x => x());
    }
}

export function freezeArea(state: GameState, freezer: UnitState): UndoCallback {
    const chain: UndoCallback[] = [];
    const adjacent = getAdjacentIndexes(state, freezer._tileIndex, 1, false, true);

    for (let i = 0; i < adjacent.length; i++) {
        const tile = state.tiles[adjacent[i]];
        const occupied = getTrueEnemyAt(state, tile.tileIndex, freezer._owner);

        // Freeze any adjacent enemy unit
        if (occupied) {
            chain.push(tryAddEffect(state, occupied, EffectType.Frozen));
        }

        // Freeze any adjacent freezable tiles
        if (tile.terrainType == TerrainType.Water || tile.terrainType == TerrainType.Ocean) {
            chain.push(modifyTerrain(state, tile.tileIndex, TerrainType.Ice));
        }
    }

    return () => {
        chain.forEach(x => x());
    };
}


