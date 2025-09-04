import { xorTile } from "../../zobrist/hasher";
import { getPovTribe } from "../functions";
import { Branch } from "../move";
import { GameState } from "../states";
import { discoverTiles } from "./DiscoverTiles";


export default function(state: GameState, territory: number[], force = false, villageTile = -1): Branch {
    const pov = getPovTribe(state);
    const undoDiscover = discoverTiles(state, null, territory)!;

    if (state.settings.areYouSure && !force) {
        territory = territory.filter(tileIndex => state.tiles[tileIndex]._owner === 0);
    }

    const oldOwners = new Array(territory.length).fill(0);

    for (let i = 0; i < territory.length; i++) {
        const tile = state.tiles[territory[i]];
        oldOwners[i] = tile._owner;
        xorTile.owner(state, tile.tileIndex, tile._owner, pov.owner);
        tile._owner = pov.owner;
        if (villageTile != -1) {
            tile._rulingCityIndex = villageTile;
        }
    }

    pov._score += 20 * territory.length;

    return {
        rewards: undoDiscover.rewards,
        undo: () => {
            pov._score -= 20 * territory.length;

            for (let i = 0; i < territory.length; i++) {
                const tile = state.tiles[territory[i]];
                xorTile.owner(state, tile.tileIndex, tile._owner, pov.owner);
                tile._owner = oldOwners[i];
                if (villageTile != -1) {
                    tile._rulingCityIndex = 0;
                }
            }

            undoDiscover.undo();
        }
    };
}
