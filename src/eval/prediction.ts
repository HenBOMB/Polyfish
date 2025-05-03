import { calculateDistance, getCapitalCity, getEnemiesNearTile, getNeighborIndexes, getPovTribe, isResourceVisible, isTribeSteppable } from "../core/functions";
import { CityState, GameState } from "../core/states";
import { SkillType, ClimateType, ModeType, RewardType, StructureType, TechnologyType, TerrainType, TribeType, Climate2Tribe } from "../core/types";

// TODO use adjacent climate tiles to reveal cities

// Mapping from ClimateType to TribeType

// Predict climate for a fogged tile based on visible neighbors
function predictClimate(state: GameState, tileIndex: number): ClimateType {
    const neighbors = getNeighborIndexes(state, tileIndex, 1, false, false); // Only visible neighbors
    const climateCounts: { [key in ClimateType]?: number } = {};

    // Count climates of visible neighbors
    neighbors.forEach(neighbor => {
        if (state._visibleTiles.includes(neighbor)) {
            const climate = state.tiles[neighbor].climate;
            // If we are not the one who has this climate
            // then we decrease our chances of predicting this climate type
            // this is to promote predicting other tribes
            if(Climate2Tribe[state.tiles[neighbor].climate] == getPovTribe(state).tribeType) {
                climateCounts[climate] = (climateCounts[climate] || 0) + 1;
            }
            climateCounts[climate] = (climateCounts[climate] || 0) + 1;
        }
    });

    // If no visible neighbors, default to Nature
    if (Object.keys(climateCounts).length === 0) {
        return ClimateType.Nature;
    }

    // Find the most common climate
    let maxCount = 0;
    let predictedClimate: ClimateType = ClimateType.Nature;
    for (const [climate, count] of Object.entries(climateCounts)) {
        if (count > maxCount) {
            maxCount = count;
            predictedClimate = Number(climate) as ClimateType;
        }
    }
    return predictedClimate;
}

export function predictVillages(state: GameState): { [tileIndex: number]: [TribeType, boolean] } {
    const targetClimate = ClimateType.Oumaji;
    const pov = state.settings._pov;
    const predictedTribe = Climate2Tribe[targetClimate];
    const capital = getCapitalCity(state, pov);

    // EVAL Game over really
    if(!capital) return {};

    // const domainSize = getDomainGrid(state.settings.tribeCount);

    // console.log(domainSize);
    
    // Step 1: Find visible tiles with the target climate, not owned by you
    const candidates: { [tileIndex: number]: number } = [];
    for (const tileIndex of state._visibleTiles) {
        const tile = state.tiles[tileIndex];
        if (tile.climate === targetClimate && tile._owner !== pov) {
            getNeighborIndexes(state, tileIndex, 2, false, true).forEach(neighbor => {
                const tileX = neighbor % state.settings.size;
                const tileY = Math.floor(neighbor / state.settings.size);
                if (tileX <= 1 || tileX >= state.settings.size - 2 || tileY <= 1 || tileY >= state.settings.size - 2) {
                    return;
                }
                // if (calculateDistance(neighbor, capital.tileIndex, state.settings.size) < domainSize) {
                //     return;
                // }
                if (!state.tiles[neighbor].explorers.includes(pov)) {
                    candidates[neighbor] = (candidates[neighbor] || 0) + 1;
                }
            });
        }
    }

    // Step 3: Sort by density (highest first)
    const sortedKeys = Object.keys(candidates).sort((a, b) => candidates[Number(b)] - candidates[Number(a)]);

    // Step 4: Create the prediction map
    const predictionMap: { [tileIndex: number]: [TribeType, boolean] } = {};
    if (sortedKeys.length > 0) {
        predictionMap[Number(sortedKeys[0])] = [predictedTribe, true];
    }

    return predictionMap;
}

/** Predict village locations with tribe type (climate) */
export function predictVillagesOld(state: GameState): { [tileIndex: number]: [TribeType, boolean] } {
    const size = state.settings.size;
    const totalTiles = size * size;
    const density = new Float32Array(totalTiles);
    const tribe = state.tribes[state.settings._pov];
    
    // Step 1: Track known village/city tiles
    const knownVillageTiles = new Set<number>();

    // Step 2: Identify visible resource tiles
    const resourceTiles: number[] = [];

    // Step 3: Track resources explained by known villages/cities (within distance 2)
    const explainedResources = new Set<number>();

    // Analyze visible tiles
    for (let i = 0; i < state._visibleTiles.length; i++) {
        const tileIndex = state._visibleTiles[i];
        const tile = state.tiles[tileIndex];
        const resource = state.resources[tileIndex] && isResourceVisible(tribe, state.resources[tileIndex].id);

        if (resource) {
            resourceTiles.push(tileIndex);
        }

        // Check for villages or cities
        if ((tile._owner > 0 && state.tribes[tile._owner]._cities.some(c => c.tileIndex === tileIndex)) ||
            state.structures[tileIndex]?.id === StructureType.Village) {
            knownVillageTiles.add(tileIndex);
            getNeighborIndexes(state, tileIndex, 2).forEach(neighbor => {
                const neighborTile = state.tiles[neighbor];
                // Skip enemy-owned tiles
                if (neighborTile._rulingCityIndex > 0 && neighborTile._owner !== state.settings._pov) {
                    return;
                }
                if (state.resources[neighbor]) {
                    explainedResources.add(neighbor);
                }
            });
        }
    }

    // Step 4: Find unexplained resources
    const unexplainedResources = resourceTiles.filter(index => !explainedResources.has(index));

    // Step 5: Calculate resource density for each tile
    unexplainedResources.forEach(resourceTile => {
        getNeighborIndexes(state, resourceTile, 2, false, true).forEach(neighbor => {
            density[neighbor]++;
        });
    });

    let suspectedEnemyCities: { [tileIndex: number]: boolean } = {};

    // Step 5.5: Boost density based on enemy territory rows
    const enemyRows = detectEnemyRows(state);
    enemyRows.forEach(([middleTile, direction]) => {
        const x = middleTile % size;
        const y = Math.floor(middleTile / size);
        let boostTiles: number[] = [];

        if (direction === 1) {
            // Boost fog tiles above and below the middle tile
            if (y > 0) boostTiles.push(middleTile - size); // above
            if (y < size - 1) boostTiles.push(middleTile + size); // below
        } else if (direction === 0) {
            // Boost fog tiles left and right of the middle tile
            if (x > 0) boostTiles.push(middleTile - 1); // left
            if (x < size - 1) boostTiles.push(middleTile + 1); // right
        }

        // Apply bonus to fog tiles only
        boostTiles.forEach(tileIndex => {
            if (
                tileIndex >= 0 &&
                tileIndex < totalTiles &&
                !state._visibleTiles.includes(tileIndex)
            ) {
                density[tileIndex] += 1; // Bonus of 1, adjustable
                suspectedEnemyCities[tileIndex] = true;
            }
        });
    });

    // Step 6: Identify potential village locations (fogged tiles with density > 0)
    const potential: [number, number][] = [];
    for (let index = 0; index < totalTiles; index++) {
        if (!state._visibleTiles.includes(index) && density[index] > 0) {
            potential.push([index, density[index]]);
        }
    }

    // Step 7: Sort potential locations by density (descending)
    potential.sort((a, b) => b[1] - a[1]);

    // Step 8: Greedily predict villages and assign tribe types
    const predicted: { [tileIndex: number]: [TribeType, boolean] } = {};
    const invalid = new Set<number>();

    for (const [index, density] of potential) {
        if (!invalid.has(index) && !knownVillageTiles.has(index)) {
            // Predict tribe type based on neighboring tiles (distance 1)
            const neighbors = getNeighborIndexes(state, index, 1, false, true);
            const tribeCounts: Partial<Record<TribeType, number>> = {};

            neighbors.forEach(neighbor => {
                if (state._visibleTiles.includes(neighbor)) {
                    let tribeType = TribeType.None;

                    if(state.tiles[neighbor]._owner > 0) {
                        tribeType = state.tribes[state.tiles[neighbor]._owner].tribeType;
                    }
                    else {
                        tribeType = TribeType[ClimateType[state.tiles[neighbor].climate] as keyof typeof TribeType];
                    }

                    if(tribeType == TribeType.None) {
                        return;
                    }

                    tribeCounts[tribeType] = (tribeCounts[tribeType] || 0) + 1;
                } else if (predicted[neighbor]) {
                    const tribeType = predicted[neighbor][0];
                    tribeCounts[tribeType] = (tribeCounts[tribeType] || 0) + 1;
                }
            });

            // Choose the most common tribe type, default to 0 if none found
            let predictedTribe: TribeType = 0 as TribeType;
            let maxCount = 0;
            for (const [tribeStr, count] of Object.entries(tribeCounts)) {
                const tribeType = Number(tribeStr) as TribeType;
                if (count > maxCount) {
                    maxCount = count;
                    predictedTribe = tribeType;
                }
            }

            // Store the prediction
            predicted[index] = [predictedTribe, suspectedEnemyCities[index] || false];

            // Enforce spacing by marking tiles within distance 2 as invalid
            getNeighborIndexes(state, index, 2, false, true).forEach(neighbor => invalid.add(neighbor));
        }
    }

    return predicted;
}

type RowInfo = [number, 0 | 1];

// TODO
function detectEnemyRows(state: GameState): RowInfo[] {
    const rows: RowInfo[] = [];
    const size = state.settings.size;
    const visibleTiles = new Set(state._visibleTiles);

    for (const tileIndex of state._visibleTiles) {
        const tile = state.tiles[tileIndex];
        // Check if tile is enemy-owned
        if (tile._owner !== state.settings._pov && tile._owner > 0) {
            const x = tileIndex % size;
            const y = Math.floor(tileIndex / size);

            // TODO just one
            // Horizontal row: check this tile and the next two to the right or just two, or just one
            if (x <= size - 3) {
                const tile1 = tileIndex;
                const tile2 = tileIndex + 1;
                const tile3 = tileIndex + 2;
                if (
                    visibleTiles.has(tile2) &&
                    visibleTiles.has(tile3) &&
                    state.tiles[tile2]._owner === tile._owner &&
                    state.tiles[tile3]._owner === tile._owner
                ) {
                    rows.push([tile2, 1]); // tile2 is the middle tile
                }
                else if (
                    visibleTiles.has(tile2) &&
                    !visibleTiles.has(tile3) &&
                    state.tiles[tile2]._owner === tile._owner
                ) {
                    // The tile with most fog is the center tile
                    const count1 = getNeighborIndexes(state, tile1, 1, false, true).filter(i => !visibleTiles.has(i)).length;
                    const count2 = getNeighborIndexes(state, tile2, 1, false, true).filter(i => !visibleTiles.has(i)).length;
                    if (count1 > count2) {
                        rows.push([tile1, 1]);
                    } else {
                        rows.push([tile2, 1]);
                    }
                }
            }

            // TODO just one
            // Vertical row: check this tile and the two below it or just two or just one
            if (y <= size - 3) {
                const tile1 = tileIndex;
                const tile2 = tileIndex + size;
                const tile3 = tileIndex + 2 * size;
                if (
                    visibleTiles.has(tile2) &&
                    visibleTiles.has(tile3) &&
                    state.tiles[tile2]._owner === tile._owner &&
                    state.tiles[tile3]._owner === tile._owner
                ) {
                    rows.push([tile2, 0]); // tile2 is the middle tile
                }
                else if (
                    visibleTiles.has(tile2) &&
                    !visibleTiles.has(tile3) &&
                    state.tiles[tile2]._owner === tile._owner
                ) {
                    // The tile with most fog is the center tile
                    const count1 = getNeighborIndexes(state, tile1, 1, false, true).filter(i => !visibleTiles.has(i)).length;
                    const count2 = getNeighborIndexes(state, tile2, 1, false, true).filter(i => !visibleTiles.has(i)).length;
                    if (count1 > count2) {
                        rows.push([tile1, 0]);
                        
                    } else {
                        rows.push([tile2, 0]);
                    }
                }
            }
        }
    }

    return rows;
}

// Get the border clouds (fogged tiles adjacent to visible tiles)
export function getBorderClouds(state: GameState): number[] {
    const size = state.settings.size;
    const totalTiles = size * size;
    const visibleTiles = new Set(state._visibleTiles);
    const borderClouds = new Set<number>();

    // Step 1: Iterate over all visible tiles
    state._visibleTiles.forEach(tileIndex => {
        // Step 2: Get neighboring tiles (range = 1, including unexplored)
        const neighbors = getNeighborIndexes(state, tileIndex, 1, false, true);

        // Step 3: Filter for fogged tiles
        neighbors.forEach(neighbor => {
            // Ensure the neighbor is within bounds and not visible
            if (neighbor >= 0 && neighbor < totalTiles && !visibleTiles.has(neighbor)) {
                borderClouds.add(neighbor);
            }
        });
    });

    // Step 4: Convert Set to array and return
    return Array.from(borderClouds);
}

// Predict terrain and tribe type of tiles adjacent to fogged tiles
export function predictOuterFogTerrain(
    state: GameState,
    fogPredictions: { [tileIndex: number]: [TerrainType, ClimateType, boolean] }
): { [tileIndex: number]: [TerrainType, ClimateType] } {
    const visibleTiles = new Set(state._visibleTiles);
    // Update outerPredictions to store [TerrainType, ClimateType] tuples
    const outerPredictions: { [tileIndex: number]: [TerrainType, ClimateType] } = {};
    const fogTiles = Object.keys(fogPredictions).map(x => Number(x));

    // Step 1: Identify outer tiles (adjacent to fog tiles, not visible or fog)
    const outerTiles: { [tileIndex: number]: [TerrainType | null, ClimateType | null] } = {};
    fogTiles.forEach(fogIndex => {
        outerTiles[fogIndex] = [fogPredictions[fogIndex][0], fogPredictions[fogIndex][1]];
        const neighbors = getNeighborIndexes(state, fogIndex, 1, false, true);
        neighbors.forEach(neighbor => {
            if (!visibleTiles.has(neighbor) && !fogTiles.includes(neighbor)) {
                outerTiles[neighbor] = [null, null];
            }
        });
    });

    // Step 2: Predict terrain and climate type for each outer tile
    for(const outerIndex in outerTiles) {
        const [outerTerrain, outerClimate] = outerTiles[outerIndex];
        const tileIndex = Number(outerIndex);

        let neighbors = getNeighborIndexes(state, tileIndex, 1, false, true);
        if(neighbors.length == 0) {
            neighbors = getNeighborIndexes(state, tileIndex, 2, false, true);
            if(neighbors.length == 0) {
                neighbors = getNeighborIndexes(state, tileIndex, 3, false, true);
            }
        }

        const terrainCounts: Partial<Record<TerrainType, number>> = {};
        const climateCounts: Partial<Record<ClimateType, number>> = {};

        // Count terrain and climate types from visible and predicted fog tiles
        neighbors.forEach(neighbor => {
            if (visibleTiles.has(neighbor)) {
                const tile = state.tiles[neighbor];
                const terrain = tile.terrainType;
                terrainCounts[terrain] = (terrainCounts[terrain] || 0) + 1;
                climateCounts[tile.climate] = (climateCounts[tile.climate] || 0) + 1;
            } else if (fogPredictions[neighbor]) {
                // Correctly destructure the tuple from fogPredictions
                const [terrain, climate] = fogPredictions[neighbor];
                terrainCounts[terrain] = (terrainCounts[terrain] || 0) + 1;
                climateCounts[climate] = (climateCounts[climate] || 0) + 1;
            }
        });

        // Step 3: Find the most common terrain type
        let predictedTerrain = outerTerrain;
        if(!outerTerrain) {
            let maxTerrainCount = 0;
            for (const [terrainStr, count] of Object.entries(terrainCounts)) {
                const terrain = Number(terrainStr) as TerrainType;
                if (count > maxTerrainCount) {
                    maxTerrainCount = count;
                    predictedTerrain = terrain;
                }
            }
        }

        // Step 4: Find the most common climate type
        let predictedClimate = outerClimate;
        if(!predictedClimate) {
            let maxClimateCount = 0;
            for (const [climateStr, count] of Object.entries(climateCounts)) {
                const climate = Number(climateStr) as ClimateType;
                if (count > maxClimateCount) {
                    maxClimateCount = count;
                    predictedClimate = climate;
                }
            }
        }

        if (!predictedTerrain) {
            predictedTerrain = TerrainType.Field;
        }

        if (!predictedClimate || (predictedTerrain == TerrainType.Water || predictedTerrain == TerrainType.Ocean)) {
            predictedClimate = ClimateType.Nature;
        }

        outerPredictions[tileIndex] = [predictedTerrain, predictedClimate];
    };

    return outerPredictions;
}

/**
 * Filters the neighboring tiles to those the explorer can move onto.
 */
function getAllowedNeighbors(state: GameState, tileIndex: number, includeUnexplored: boolean = true): number[] {
    const tribe = state.tribes[state.settings._pov];
    const odds = 
        0.45 +
        (tribe._tech.includes(TechnologyType.Fishing)? .25 : 0) +
        (tribe._tech.includes(TechnologyType.Sailing)? .1 : 0) +
        (tribe._tech.includes(TechnologyType.Climbing)? .1 : 0);
    return shuffleArray(
        getNeighborIndexes(state, tileIndex, 1, false, includeUnexplored)
        // Only if its absolutely visible and steppable, if not then we dont really know what is there
        .filter(x => 
            // isTribeSteppable(state, x) // (cheating)
            state.tiles[x].explorers.includes(tribe.owner)? isTribeSteppable(state, x) : 
            Math.random() < odds
        )
    );
}

/**
 * Shuffles an array in-place using the Fisher-Yates algorithm.
 */
function shuffleArray<T>(array: T[]): T[] {
    const arr = array.slice();
    for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
}

/**
 * Performs a BFS from the start tile to find a "cloud" tile (i.e. an unrevealed tile not in state._visibleTiles)
 * that is reachable within maxDistance moves. If multiple cloud tiles are found at the same distance, one is chosen at random.
 * Returns the path (as an array of tile indices) from start to target or null if none found.
 */
function findNearestCloud(state: GameState, startIndex: number, maxDistance: number): number[] | null {
    // Each queue entry holds the current tile and the path (list of tileIndices) from the start to that tile.
    const queue: { tileIndex: number; path: number[] }[] = [];
    const visited = new Set<number>();
    queue.push({ tileIndex: startIndex, path: [startIndex] });
    visited.add(startIndex);

    let candidates: { path: number[] }[] = [];
    let currentDistance = 0;
    
    // Standard BFS by level.
    while (queue.length > 0 && currentDistance <= maxDistance) {
        const levelSize = queue.length;
        for (let i = 0; i < levelSize; i++) {
            const { tileIndex: tile, path } = queue.shift()!;
            // Check: if this tile is unrevealed (cloud) and not the starting tile,
            // then add it to candidates.
            if (path.length > 1 && !state._visibleTiles.includes(tile)) {
                candidates.push({ path });
            }
            // If we have not exceeded maxDistance, add allowed neighbors.
            if (path.length - 1 < maxDistance) {
                const neighbors = getAllowedNeighbors(state, tile);
                for (const neighbor of neighbors) {
                    if (!visited.has(neighbor)) {
                        visited.add(neighbor);
                        queue.push({ tileIndex: neighbor, path: [...path, neighbor] });
                    }
                }
            }
        }
        // If any candidate was found on this level, stop searching.
        if (candidates.length > 0) {
            break;
        }
        currentDistance++;
    }
    if (candidates.length === 0) {
        return null;
    }
    // Randomly pick one of the candidate paths.
    const candidate = candidates[Math.floor(Math.random() * candidates.length)];
    return candidate.path;
}

/**
 * Predicts where an explorer will go.
 * The explorer moves 15 times. On each move, it will:
 *   - Try to find an unrevealed ("cloud") tile reachable within 4 moves.
 *   - If found, move one step along a shortest path toward that tile.
 *   - If not found, move randomly to one of the allowed neighboring tiles.
 * 
 * @param state - The game state.
 * @param tileIndex - The starting tile of the explorer.
 * @returns tiles explored (an array of tile indices for each move).
 */
export function predictExplorer(state: GameState, tileIndex: number): number[][] {
    const visible = [...state._visibleTiles];
    const path: number[] = [];
    let currentTile = tileIndex;

    for (let move = 0; move < 15; move++) {
        // Look for a cloud tile within 4 moves.
        const cloudPath = findNearestCloud(state, currentTile, 4);
        let nextTile: number;
        if (cloudPath && cloudPath.length > 1) {
            // Move one step along the path toward the cloud.
            nextTile = cloudPath[1];
        } else {
            // No cloud within 4 moves: choose a random allowed neighboring tile.
            const allowed = getAllowedNeighbors(state, currentTile, false);
            if (allowed.length > 0) {
                nextTile = allowed[Math.floor(Math.random() * allowed.length)];
            } else {
                // If no allowed move exists, the explorer stays in place.
                nextTile = currentTile;
            }
        }
		state._visibleTiles = Array.from(new Set([
            ...state._visibleTiles, 
            nextTile, 
            ...getNeighborIndexes(state, nextTile, 1, false, true)
        ]));
        path.push(nextTile);
        currentTile = nextTile;
    }

    const explored = state._visibleTiles.filter(x => !visible.includes(x));

    state._visibleTiles = [...visible];

    return [path, explored];
}


// Predicts enemy capitals and surrounding tiles //

export function predictEnemyCapitalsAndSurroundings(state: GameState): number[] {
    const mapSize = state.settings.size;
    const gridSize = getDomainGrid(state.settings.tribeCount); // 2x2 grid
    const totalDomains = gridSize * gridSize; // 4 domains

    // Step 1: Find your domain based on your capital
    const tribe = state.tribes[state.settings._pov];
    const yourDomains = new Set<number>();
    tribe._cities.forEach(city => {
        if (state.tiles[city.tileIndex].capitalOf > 0) {
            const x = city.tileIndex % mapSize;
            const y = Math.floor(city.tileIndex / mapSize);
            yourDomains.add(getDomainIndex(x, y, gridSize, mapSize));
        }
    });

    // Step 2: Collect visible enemy tiles by domain
    const enemyTilesByDomain: { [domainIndex: number]: number[] } = {};
    state._visibleTiles.forEach(tileIndex => {
        const tile = state.tiles[tileIndex];
        const ownedByEnemy = tile._owner > 0 && tile._owner !== tribe.owner;
        // TODO recently spawned so climate can reveal where their city might be at
        // const recentlySpawned = state.settings.turn < 7 && state.tribes[] (convert climate to tribe? find?)
        if (ownedByEnemy) {
            const x = tileIndex % mapSize;
            const y = Math.floor(tileIndex / mapSize);
            const domainIndex = getDomainIndex(x, y, gridSize, mapSize);
            if (!enemyTilesByDomain[domainIndex]) {
                enemyTilesByDomain[domainIndex] = [];
            }
            enemyTilesByDomain[domainIndex].push(tileIndex);
        }
    });

    // Step 3: Predict capitals in all domains except ours
    const predictedCapitals: number[] = [];
    let predictedCount = 0;
    for (let domainIndex = 0; domainIndex < totalDomains - 1; domainIndex++) {
        if (yourDomains.has(domainIndex)) continue; // Skip domains with our capitals

        const domainBounds = getDomainBounds(domainIndex, gridSize, mapSize);
        let predictedTiles: number[] = []; // Change to array to hold multiple tiles

        if (enemyTilesByDomain[domainIndex] && enemyTilesByDomain[domainIndex].length > 0) {
            // If enemy tiles are visible, predict near their centroid
            predictedTiles = [calculateCentroid(enemyTilesByDomain[domainIndex], mapSize)];
        } else {
            // No enemy tiles, predict the center and its neighbors
            const centerTile = getDomainCenter(domainBounds, mapSize);
            // const neighbors = getNeighborIndexes(state, centerTile, 1, false, true);
            predictedTiles = [centerTile];
        }

        predictedCapitals.push(...predictedTiles);
        predictedCount++;
    }

    // Step 4: Add surrounding tiles (e.g., range 1 around each capital)
    const adjacentTiles: number[] = [];
    predictedCapitals.forEach(capitalTile => {
        const neighbors = getNeighborIndexes(state, capitalTile, 1, false, true);
        adjacentTiles.push(...neighbors);
    });

    // Step 5: Return unique fogged tiles
    const allPredictedTiles = [...new Set([...predictedCapitals, ...adjacentTiles])];
    return allPredictedTiles.filter(tile => !state._visibleTiles.includes(tile));
}

// Determines grid size based on player count (e.g., 2x2 for 4 players)
function getDomainGrid(playerCount: number): number {
    if (playerCount <= 4) return 2; // 2x2 grid
    if (playerCount <= 9) return 3; // 3x3 grid
    return 4; // 4x4 grid for 10-16 players
}

// Calculates which domain a tile belongs to
function getDomainIndex(x: number, y: number, gridSize: number, mapSize: number): number {
    const domainWidth = Math.ceil(mapSize / gridSize);
    const domainX = Math.floor(x / domainWidth);
    const domainY = Math.floor(y / domainWidth);
    return domainY * gridSize + domainX;
}

// Gets the boundaries of a domain
function getDomainBounds(domainIndex: number, gridSize: number, mapSize: number): [number, number, number, number] {
    const domainWidth = Math.ceil(mapSize / gridSize);
    const domainX = domainIndex % gridSize;
    const domainY = Math.floor(domainIndex / gridSize);
    const minX = domainX * domainWidth;
    const minY = domainY * domainWidth;
    const maxX = Math.min(minX + domainWidth - 1, mapSize - 1);
    const maxY = Math.min(minY + domainWidth - 1, mapSize - 1);
    return [minX, maxX, minY, maxY];
}

// Gets the inner area of a domain (where capitals typically spawn)
function getInnerDomainBounds(domainBounds: [number, number, number, number]): [number, number, number, number] {
    const [minX, maxX, minY, maxY] = domainBounds;
    const innerMinX = minX + 2;
    const innerMaxX = maxX - 2;
    const innerMinY = minY + 2;
    const innerMaxY = maxY - 2;
    return [
        Math.max(innerMinX, minX),
        Math.min(innerMaxX, maxX),
        Math.max(innerMinY, minY),
        Math.min(innerMaxY, maxY)
    ];
}

// Calculates the centroid of a list of tile indices
function calculateCentroid(tileIndices: number[], mapSize: number): number {
    if (tileIndices.length === 0) return -1;
    let sumX = 0, sumY = 0;
    tileIndices.forEach(tileIndex => {
        const x = tileIndex % mapSize;
        const y = Math.floor(tileIndex / mapSize);
        sumX += x;
        sumY += y;
    });
    const avgX = Math.round(sumX / tileIndices.length);
    const avgY = Math.round(sumY / tileIndices.length);
    return getTileIndex(avgX, avgY, mapSize);
}

// Gets the center tile of the inner domain
function getDomainCenter(domainBounds: [number, number, number, number], mapSize: number): number {
    const [innerMinX, innerMaxX, innerMinY, innerMaxY] = getInnerDomainBounds(domainBounds);
    const centerX = Math.floor((innerMinX + innerMaxX) / 2);
    const centerY = Math.floor((innerMinY + innerMaxY) / 2);
    return getTileIndex(centerX, centerY, mapSize);
}

// Converts x, y coordinates to a tile index
function getTileIndex(x: number, y: number, mapSize: number): number {
    return y * mapSize + x;
}




// Predict best next city reward to choose //

export function predictBestNextCityReward(state: GameState, targetCity?: CityState): RewardType[] {
    const tribe = getPovTribe(state);
    const rewards: RewardType[] = [];
    const minAmountOfSus = 3;
    const minPotentialPop = 4;

    for(const city of targetCity? [targetCity] : state.tribes[state.settings._pov]._cities) {
        const tileIndex = city.tileIndex;
        switch (city._level) {
            case 2:
                // EVAL Dont waste explore with the capital
                // Workshop or Explorer
                if(state.tiles[tileIndex].capitalOf < 1 && state._prediction._enemyCapitalSuspects) {
                    const maxIter = 20;
                    let pass = 0;
                    for (let i = 0; i < maxIter; i++) {
                        const explorer = predictExplorer(state, tileIndex);
                        // If the explorer explored any of the predicted enemy capitals tiles
                        if(explorer.flat().filter(x => state._prediction._enemyCapitalSuspects?.includes(x)).length >= minAmountOfSus) {
                            pass++;
                        }
                    }
                    // console.log('pass', pass);
                    // At least half passes
                    if(pass >= maxIter / 2) {
                        rewards.push(RewardType.Explorer);
                        break;
                    }
                }
                rewards.push(RewardType.Workshop);
                break;
                
            case 3:
                // Walls or Resources
                if(getEnemiesNearTile(state, city.tileIndex, 2).length > 1) {
                    rewards.push(RewardType.CityWall);
                }
                else {
                    rewards.push(RewardType.Resources);
                }
                break;

            case 4:
                let potentialPop = 0;
                for(const tileIndex of city._territory) {
                    if(state.resources[tileIndex] && isResourceVisible(tribe, state.resources[tileIndex].id)) potentialPop++;
                }
                if(potentialPop >= minPotentialPop) {
                    rewards.push(RewardType.BorderGrowth);
                }
                else {
                    rewards.push(RewardType.PopulationGrowth);
                }
                break;
            default:
                if(state.settings.mode == ModeType.Domination){// || !tribe._units.some(x => isSkilledIn(x, AbilityType.Independent))) {
                    rewards.push(RewardType.SuperUnit);
                }
                else {
                    rewards.push(RewardType.Park);
                }
                break;
        }
    }

    return rewards;
}