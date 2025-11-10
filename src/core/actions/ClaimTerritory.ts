import { xorTile } from "../../zobrist/hasher";
import { getPovTribe } from "../functions";
import { Branch } from "../move";
import { CityState, Coords, GameState } from "../states";
import { discoverTiles } from "./DiscoverTiles";

export default function(state: GameState, territory: number[], city: CityState, force = false): Branch {
    const pov = getPovTribe(state);
    const undoDiscover = discoverTiles(state, null, territory)!;

    if (state.settings._areYouSure && !force) {
        territory = territory.filter(tileIndex => state.tiles[tileIndex].owner < 1);
    }

    const cityTile = state.tiles[city.tileIndex];
    const oldOwners = new Array(territory.length).fill(0);
    const oldRulingCoords = cityTile.rulingCityCoords;
    const oldOwnerID = cityTile.owner;

    cityTile.owner = pov.id;
    cityTile.rulingCityCoords = new Coords(city.tileIndex, state);

    for (let i = 0; i < territory.length; i++) {
        const tile = state.tiles[territory[i]];
        oldOwners[i] = tile.owner;
        // xorTile.owner(state, tile.coords.idx, tile.owner, pov.id);
        tile.owner = city.owner;
        tile.rulingCityCoords = new Coords(city.tileIndex, state);
    }

    pov.score += 20 * territory.length;

    return {
        rewards: undoDiscover.rewards,
        undo: () => {
            pov.score -= 20 * territory.length;

            for (let i = 0; i < territory.length; i++) {
                const tile = state.tiles[territory[i]];
                // xorTile.owner(state, tile.coords.idx, tile.owner, pov.id);
                tile.owner = oldOwners[i];
                tile.rulingCityCoords = undefined;
            }

            cityTile.owner = oldOwnerID;
            cityTile.rulingCityCoords = oldRulingCoords;

            undoDiscover.undo();
        }
    };
}
