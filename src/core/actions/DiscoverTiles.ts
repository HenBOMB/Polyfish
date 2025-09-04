import { xorTile } from "../../zobrist/hasher";
import addPopulationToCity from "./AddPopulation";
import { getPovTribe, getAdjacentIndexes, isSkilledIn, getLighthouses, getCapitalCity } from "../functions";
import Move, { CallbackResult, UndoCallback } from "../move";
import { GameState, UnitState } from "../states";
import { TerrainType, SkillType } from "../types";


export function discoverTiles(state: GameState, unit?: UnitState | null, tileIndexes?: number[]): CallbackResult {
    const pov = getPovTribe(state);
    const discovered = (tileIndexes || (unit ? getAdjacentIndexes(
        state,
        unit._tileIndex,
        state.tiles[unit._tileIndex].terrainType == TerrainType.Mountain || isSkilledIn(unit, SkillType.Scout) ? 2 : 1,
        false,
        true
    ) : [])).filter(x => !state._visibleTiles[x]);

    const missingLighthouses = getLighthouses(state, false);

    let chain: UndoCallback[] = [];
    let rewards: Move[] = [];

    for (const tileIndex of discovered) {
        xorTile.discover(state, state.tiles[tileIndex]);

        if (missingLighthouses.includes(tileIndex)) {
            const city = getCapitalCity(state);
            if (city) {
                const result = addPopulationToCity(state, city, 1);
                chain.push(result?.undo);
                rewards.push(...result.rewards);
            }
        }

        if (state.settings.areYouSure) {
            state.tiles[tileIndex]._explorers.add(pov.owner);
        }

        state._visibleTiles[tileIndex] = true;
    }

    pov._score += 5 * discovered.length;

    return {
        rewards,
        undo: () => {
            chain.forEach(x => x());

            pov._score -= 5 * discovered.length;

            discovered.forEach(x => {
                xorTile.discover(state, state.tiles[x]);

                if (state.settings.areYouSure) {
                    state.tiles[x]._explorers.delete(pov.owner);
                }

                state._visibleTiles[x] = false;
            });
        }
    };
}
