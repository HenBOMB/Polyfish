import { xorTile } from "../../zobrist/hasher";
import addPopulationToCity from "./AddPopulation";
import { getPovTribe, getAdjacentIndexes, isSkilledIn, getLighthouses, getCapitalCity } from "../functions";
import Move, { Branch, CallbackResult, UndoCallback } from "../move";
import { GameState, UnitState } from "../states";
import { TerrainType, SkillType } from "../types";


export function discoverTiles(state: GameState, unit?: UnitState | null, tileIndexes?: number[]): Branch {
    const pov = getPovTribe(state);
    const discovered = (tileIndexes || (unit ? getAdjacentIndexes(
        state,
        unit.coords.idx,
        state.tiles[unit.coords.idx].type == TerrainType.Mountain || isSkilledIn(unit, SkillType.Scout) ? 2 : 1,
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

        if (state.settings._areYouSure) {
            state.tiles[tileIndex].explorers.add(pov.id);
        }

        state._visibleTiles[tileIndex] = true;
    }

    pov.score += 5 * discovered.length;

    return {
        rewards,
        undo: () => {
            chain.forEach(x => x());

            pov.score -= 5 * discovered.length;

            discovered.forEach(x => {
                xorTile.discover(state, state.tiles[x]);

                if (state.settings._areYouSure) {
                    state.tiles[x].explorers.delete(pov.id);
                }

                state._visibleTiles[x] = false;
            });
        }
    };
}
