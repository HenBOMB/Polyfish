import { addPopulationToCity } from "./actions";
import { computeReachablePath, getCityAt, getCityOwningTile, getNeighborIndexes, getPovTribe } from "./functions";
import Move, { UndoCallback, CallbackResult } from "./move";
import { GameState, CityState, TribeState } from "./states";
import { StructureType, TerrainType } from "./types";

export default class NetworkManager {
    private state: GameState;
    private pov: TribeState;
    private undoChain: UndoCallback[];
    private rewards: Move[];

    constructor(state: GameState) {
        this.state = state;
        this.undoChain = [];
        this.rewards = [];
        this.pov = getPovTribe(state);
    }

    /**
     * Main function to call after a road or port is built (or any network-affecting change).
     * It recalculates all capital connections for the tribe, applies rewards,
     * and updates routes.
     */
    public updateConnectionsAfterChange(): CallbackResult | null {
        this.pov = getPovTribe(this.state);
        this.undoChain = [];
        this.rewards = [];

        const capital = this.pov._cities.find(c => this.state.tiles[c.tileIndex].capitalOf === this.pov.owner);

        if (!capital) {
            console.warn("NetworkManager: No capital found for tribe owner:", this.pov.owner);
            return null;
        }

        // 1. Snapshot current _connectedToCapital status
        const previousConnectionStatus = new Map<number, boolean>();
        this.pov._cities.forEach(city => {
            if (city !== capital) {
                previousConnectionStatus.set(city.tileIndex, city._connectedToCapital);
            }
        });

        // 2. Identify all player ports and establish port-to-port "jump links"
        const allPlayerPorts = this.getAllPlayerPorts();
        const portJumpLinks = this.establishPortJumpLinks(allPlayerPorts);

        // 3. Recalculate all _connectedToCapital flags using BFS with roads and port jumps
        const connectedCityEntities = this.recalculateCapitalConnections(
            this.pov._cities,
            capital,
            previousConnectionStatus,
            portJumpLinks
        );

        // 4. Determine newly connected cities and apply population rewards
        const newlyConnectedCities: CityState[] = [];
        for (const city of connectedCityEntities) {
            if (city !== capital) {
                const wasConnected = previousConnectionStatus.get(city.tileIndex);
                if (city._connectedToCapital && !wasConnected) {
                    newlyConnectedCities.push(city);
                }
            }
        }
        this.applyPopulationRewards(newlyConnectedCities, capital);

        // 5. Update visual routes (hasRoute on tiles)
        this.updateAllVisualRoutes(this.pov._cities, capital, connectedCityEntities, portJumpLinks, allPlayerPorts);

        if (this.undoChain.length === 0 && this.rewards.length === 0) {
            return null; // No effective changes
        }

        return {
            rewards: this.rewards,
            undo: () => {
                this.undoChain.slice().reverse().forEach(cb => cb());
            }
        };
    }

    private setCityConnected(city: CityState, isConnected: boolean, previousState: boolean): void {
        if (city._connectedToCapital !== isConnected) {
            city._connectedToCapital = isConnected;
            this.undoChain.push(() => { city._connectedToCapital = previousState; });
        }
    }

    /**
     * Finds all tiles that are ports owned by the player.
     */
    private getAllPlayerPorts(): number[] {
        const portTiles: number[] = [];
        for (const city of this.pov._cities) {
            const cityTerritoryTiles = [city.tileIndex, ...(getCityAt(this.state, city.tileIndex)?._territory || [])];
            for (let i = 0; i < cityTerritoryTiles.length; i++) {
                const tileIndex = cityTerritoryTiles[i];
                if (this.state.structures[tileIndex]?.id === StructureType.Port &&
                    this.state.tiles[tileIndex]?._owner === this.pov.owner) {
                    portTiles.push(tileIndex);
                }
            }
        }
        return portTiles;
    }

    /**
     * Establishes direct connections (jump links) between player ports
     * if a valid water path exists between them.
     * Mimics finding port connections in Java's computeTradeNetworkTribe.
     */
    private establishPortJumpLinks(playerPorts: number[]): Map<number, number[]> {
        const jumpLinks = new Map<number, number[]>();
        const MAX_PORT_TRADE_DISTANCE = 5; // From TribesConfig or game settings

        for (const portIdx of playerPorts) {
            jumpLinks.set(portIdx, []); // Initialize for each port
        }

        for (let i = 0; i < playerPorts.length; i++) {
            const port1Idx = playerPorts[i];
            for (let j = i + 1; j < playerPorts.length; j++) {
                const port2Idx = playerPorts[j];

                // Pathfinder for water tiles (like TradeWaterStep)
                const path = computeReachablePath(
                    this.state,
                    port1Idx,
                    port2Idx,
                    (s, tileIdx) => { // Predicate for navigable water
                        const tile = s.tiles[tileIdx];
                        // Java version also checks tribe.isVisible(i,j) and notEnemy
                        // Adapt this predicate based on your game's specific rules for trade routes
                        return (tile.terrainType === TerrainType.Water || tile.terrainType === TerrainType.Ocean) &&
                                (s.tiles[tileIdx]._owner === this.pov.owner || s.tiles[tileIdx]._owner === -1); // Own or neutral water
                                // && s.getVisibility(tileIdx, this.tribeOwner); // If visibility matters
                    },
                    false,
                    MAX_PORT_TRADE_DISTANCE
                );

                // Path length 0 or 1 means adjacent or same tile, > 1 means actual path.
                // If path includes start+end, then path.length-1 is steps.
                // Assuming path.length is number of tiles in path including start. Max steps = MAX_PORT_TRADE_DISTANCE
                if (path && path.length > 0 && (path.length -1) <= MAX_PORT_TRADE_DISTANCE) {
                    jumpLinks.get(port1Idx)!.push(port2Idx);
                    jumpLinks.get(port2Idx)!.push(port1Idx);
                }
            }
        }
        return jumpLinks;
    }

    /**
     * Recalculates which cities are connected to the capital using BFS.
     * The BFS explores via roads and pre-calculated port "jump links".
     */
    private recalculateCapitalConnections(
        playerCities: CityState[],
        capital: CityState,
        previousStatus: Map<number, boolean>,
        portJumpLinks: Map<number, number[]>
    ): Set<CityState> {
        // Reset _connectedToCapital flags for non-capitals and prepare undo
        playerCities.forEach(city => {
            if (city !== capital) {
                const wasConnected = city._connectedToCapital; // State before this function call began
                city._connectedToCapital = false; // Tentatively set to false
                // Undo will revert to its state *before* this recalculation pass
                this.undoChain.push(() => { city._connectedToCapital = wasConnected; });
            }
        });

        const connectedCitiesSet = new Set<CityState>();
        connectedCitiesSet.add(capital); // Capital is always connected

        const queue: number[] = [capital.tileIndex];
        // Add all tiles of the capital to explore roads from any part of it
        (getCityAt(this.state, capital.tileIndex)?._territory || []).forEach(ctIdx => {
            if (!queue.includes(ctIdx)) queue.push(ctIdx);
        });
        const visitedTiles = new Set<number>(queue);

        let head = 0;
        while (head < queue.length) {
            const currentTileIndex = queue[head++];
            // const currentTile = this.state.tiles[currentTileIndex];

            // If currentTileIndex is a city's main tile, ensure its city entity is marked
            const cityAtCurrent = getCityAt(this.state, currentTileIndex);
            if (cityAtCurrent && playerCities.includes(cityAtCurrent) && !cityAtCurrent._connectedToCapital && cityAtCurrent !==capital) {
                this.setCityConnected(cityAtCurrent, true, previousStatus.get(cityAtCurrent.tileIndex) ?? false); // Update and record specific undo for this change
                connectedCitiesSet.add(cityAtCurrent);
            }


            // I. Explore via ROADS (like Java's Pathfinder on `connectedTiles`)
            const roadNeighbors = getNeighborIndexes(this.state, currentTileIndex, 1); // Direct adjacency
            for (const neighborIdx of roadNeighbors) {
                if (visitedTiles.has(neighborIdx)) continue;

                const neighborTile = this.state.tiles[neighborIdx];
                if (neighborTile._owner !== this.pov.owner) continue; // Must be our tile

                // Road Link Condition: (current is road AND neighbor is road) OR
                // (current is road AND neighbor is city) OR (current is city AND neighbor is road)
                // This defines "connectedTiles" implicitly.
                const currentIsRoad = this.state.structures[currentTileIndex]?.id === StructureType.Road;
                const neighborIsRoad = this.state.structures[neighborIdx]?.id === StructureType.Road;

                let canTraverseViaRoad = false;
                if(currentIsRoad && neighborIsRoad) canTraverseViaRoad = true;
                else if(currentIsRoad && getCityOwningTile(this.state, neighborIdx, playerCities)) canTraverseViaRoad = true;
                else if(getCityOwningTile(this.state, currentTileIndex) && neighborIsRoad) canTraverseViaRoad = true;


                if (canTraverseViaRoad) {
                    visitedTiles.add(neighborIdx);
                    queue.push(neighborIdx);

                    const cityOwningNeighbor = getCityOwningTile(this.state, neighborIdx, playerCities);
                    if (cityOwningNeighbor && cityOwningNeighbor !== capital && !cityOwningNeighbor._connectedToCapital) {
                        this.setCityConnected(cityOwningNeighbor, true, previousStatus.get(cityOwningNeighbor.tileIndex) ?? false);
                        connectedCitiesSet.add(cityOwningNeighbor);
                        // If a city becomes connected, explore from all its tiles
                        (getCityAt(this.state, cityOwningNeighbor.tileIndex)?._territory || []).forEach(ctIdx => {
                            if (!visitedTiles.has(ctIdx)) { visitedTiles.add(ctIdx); queue.push(ctIdx); }
                        });
                    }
                }
            }

            // II. Explore via pre-calculated PORT JUMP LINKS
            const isCurrentTileAPort = this.state.structures[currentTileIndex]?.id === StructureType.Port;
            if (isCurrentTileAPort && portJumpLinks.has(currentTileIndex)) {
                const linkedPorts = portJumpLinks.get(currentTileIndex)!;
                for (const remotePortIdx of linkedPorts) {
                    if (visitedTiles.has(remotePortIdx)) continue;

                    visitedTiles.add(remotePortIdx);
                    queue.push(remotePortIdx); // Add the port tile itself to explore from it

                    const cityOwningRemotePort = getCityOwningTile(this.state, remotePortIdx, playerCities);
                    if (cityOwningRemotePort && cityOwningRemotePort !== capital && !cityOwningRemotePort._connectedToCapital) {
                        this.setCityConnected(cityOwningRemotePort, true, previousStatus.get(cityOwningRemotePort.tileIndex) ?? false);
                        connectedCitiesSet.add(cityOwningRemotePort);
                        // If a city becomes connected via this port, explore from all its tiles
                        (getCityAt(this.state, cityOwningRemotePort.tileIndex)?._territory || []).forEach(ctIdx => {
                            if (!visitedTiles.has(ctIdx)) { visitedTiles.add(ctIdx); queue.push(ctIdx); }
                        });
                    }
                }
            }
        }

        return connectedCitiesSet;
    }

    /**
     * Modified Method: Applies population bonuses to newly connected cities and the capital.
     * Uses the processPopulationChange method.
     */
    private applyPopulationRewards(newlyConnectedCities: CityState[], capital: CityState): void {
        if (!capital) return;

        for (const city of newlyConnectedCities) {
            if (city === capital) continue;
            this.processPopulationChange(city, 1);
            this.processPopulationChange(capital, 1);
        }
    }

    /**
     * Calls the external addPopulationToCity function and integrates its
     * undo and rewards into the NetworkManager's collections.
     * This replaces the internal addPopulationToCity.
     */
    private processPopulationChange(city: CityState, amount: number): void {
        const result = addPopulationToCity(this.state, city, amount);
        if (result) {
            this.undoChain.push(result.undo);
            this.rewards.push(...result.rewards);
        }
    }


    /**
     * Updates the `hasRoute` visual flag on tiles.
     */
    private updateAllVisualRoutes(
        playerCities: CityState[],
        capital: CityState,
        connectedCities: Set<CityState>,
        portJumpLinks: Map<number, number[]>,
        allPlayerPorts: number[]
    ): void {
        const pov = getPovTribe(this.state);
        const initiallyRoutedTiles: number[] = [];
        for (let i = 0; i < this.state.tiles.length; i++) { // Assuming tiles is flat array
            if (this.state.tiles[i]._owner === pov.owner && this.state.tiles[i].hasRoute) {
                initiallyRoutedTiles.push(i);
                this.state.tiles[i].hasRoute = false;
            }
        }
        if (initiallyRoutedTiles.length > 0) {
            this.undoChain.push(() => {
                initiallyRoutedTiles.forEach(idx => { this.state.tiles[idx].hasRoute = true; });
            });
        }

        // B. Set routes for all player-owned roads if their city is connected
        for(const i in Object.keys(this.state.structures)) {
            const structure = this.state.structures[i];
            const tile = this.state.tiles[i];
            if (structure?.id === StructureType.Road && tile?._owner === pov.owner) {
                const owningCity = getCityOwningTile(this.state, Number(i), playerCities);
                // Roads are active if they are in a connected city's territory OR connect two connected cities
                // Simpler: if road is on a tile whose ruling city is connected.
                if (owningCity && (connectedCities.has(owningCity) || owningCity === capital) ) {
                    if (!tile.hasRoute) {
                        tile.hasRoute = true;
                        this.undoChain.push(() => { tile.hasRoute = false; });
                    }
                }
            }
        }

        // C. Set routes for water paths between ports of connected cities
        const processedPortPairs = new Set<string>();
        const MAX_PORT_TRADE_DISTANCE = 5; // As defined before

        for (const port1Idx of allPlayerPorts) {
            const city1 = getCityOwningTile(this.state, port1Idx, playerCities);
            // Both ports must belong to cities that are currently connected to the capital
            if (city1 && (connectedCities.has(city1) || city1 === capital)) {
                const linkedPorts = portJumpLinks.get(port1Idx) || [];
                for (const port2Idx of linkedPorts) {
                    const city2 = getCityOwningTile(this.state, port2Idx, playerCities);
                    if (city2 && (connectedCities.has(city2) || city2 === capital)) {
                        const pairKey = `${Math.min(port1Idx, port2Idx)}-${Math.max(port1Idx, port2Idx)}`;
                        if (processedPortPairs.has(pairKey)) continue;

                        const waterPath = computeReachablePath(this.state, port1Idx, port2Idx,
                            (s, tileIdx) => {
                                const t = s.tiles[tileIdx];
                                return (t.terrainType === TerrainType.Water || t.terrainType === TerrainType.Ocean) &&
                                       (s.tiles[tileIdx]._owner === pov.owner || s.tiles[tileIdx]._owner === -1);
                            },
                            false,
                            MAX_PORT_TRADE_DISTANCE);

                        if (waterPath && waterPath.length > 0 && (waterPath.length-1) <= MAX_PORT_TRADE_DISTANCE) {
                            waterPath.forEach(pathTileIndex => {
                                const pathTile = this.state.tiles[pathTileIndex];
                                if (!pathTile.hasRoute) {
                                    pathTile.hasRoute = true;
                                    this.undoChain.push(() => { pathTile.hasRoute = false; });
                                }
                            });
                            processedPortPairs.add(pairKey);
                        }
                    }
                }
            }
        }
    }
}