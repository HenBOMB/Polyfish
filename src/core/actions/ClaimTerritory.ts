import { xorTile } from "../../zobrist/hasher";
import { getPovTribe } from "../functions";
import { Branch } from "../move";
import { Coords, GameState } from "../states";
import { discoverTiles } from "./DiscoverTiles";


export default function(state: GameState, territory: number[], force = false, villageTile = -1): Branch {
    const pov = getPovTribe(state);
    const undoDiscover = discoverTiles(state, null, territory)!;

    if (state.settings._areYouSure && !force) {
        territory = territory.filter(tileIndex => state.tiles[tileIndex].owner === 0);
    }

    const oldOwners = new Array(territory.length).fill(0);

    for (let i = 0; i < territory.length; i++) {
        const tile = state.tiles[territory[i]];
        oldOwners[i] = tile.owner;
        xorTile.owner(state, tile.coords.idx, tile.owner, pov.id);
        tile.owner = pov.id;
        if (villageTile != -1) {
            tile.rulingCityCoords = new Coords(villageTile, state);
        }
    }

    pov.score += 20 * territory.length;

    return {
        rewards: undoDiscover.rewards,
        undo: () => {
            pov.score -= 20 * territory.length;

            for (let i = 0; i < territory.length; i++) {
                const tile = state.tiles[territory[i]];
                xorTile.owner(state, tile.coords.idx, tile.owner, pov.id);
                tile.owner = oldOwners[i];
                if (villageTile != -1) {
                    delete tile.rulingCityCoords;
                }
            }

            undoDiscover.undo();
        }
    };
}
