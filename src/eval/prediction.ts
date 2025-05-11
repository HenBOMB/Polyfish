import { calculateDistance, getCapitalCity, getEnemiesNearTile, getNeighborIndexes, getPovTribe, isResourceVisible, isTechUnlocked, isTribeSteppable } from "../core/functions";
import { CityState, GameState } from "../core/states";
import { SkillType, ClimateType, ModeType, RewardType, StructureType, TechnologyType, TerrainType, TribeType } from "../core/types";

// TODO use adjacent climate tiles to reveal cities

// Mapping from ClimateType to TribeType

export const Climate2Tribe: { [key in ClimateType]: TribeType } = {
	[ClimateType.XinXi]: TribeType.XinXi,
	[ClimateType.Imperius]: TribeType.Imperius,
	[ClimateType.Bardur]: TribeType.Bardur,
	[ClimateType.Oumaji]: TribeType.Oumaji,
	[ClimateType.Kickoo]: TribeType.Kickoo,
	[ClimateType.Hoodrick]: TribeType.Hoodrick,
	[ClimateType.Luxidoor]: TribeType.Luxidoor,
	[ClimateType.Vengir]: TribeType.Vengir,
	[ClimateType.Zebasi]: TribeType.Zebasi,
	[ClimateType.AiMo]: TribeType.AiMo,
	[ClimateType.Aquarion]: TribeType.Aquarion,
	[ClimateType.Quetzali]: TribeType.Quetzali,
	[ClimateType.Elyrion]: TribeType.Elyrion,
	[ClimateType.Yadakk]: TribeType.Yadakk,
	[ClimateType.Polaris]: TribeType.Polaris,
	[ClimateType.Cymanti]: TribeType.Cymanti,
	[ClimateType.Nature]: TribeType.Nature
};

export const Tribe2Climate: { [key in TribeType]: ClimateType } = {
	[TribeType.XinXi]: ClimateType.XinXi,
	[TribeType.Imperius]: ClimateType.Imperius,
	[TribeType.Bardur]: ClimateType.Bardur,
	[TribeType.Oumaji]: ClimateType.Oumaji,
	[TribeType.Kickoo]: ClimateType.Kickoo,
	[TribeType.Hoodrick]: ClimateType.Hoodrick,
	[TribeType.Luxidoor]: ClimateType.Luxidoor,
	[TribeType.Vengir]: ClimateType.Vengir,
	[TribeType.Zebasi]: ClimateType.Zebasi,
	[TribeType.AiMo]: ClimateType.AiMo,
	[TribeType.Aquarion]: ClimateType.Aquarion,
	[TribeType.Quetzali]: ClimateType.Quetzali,
	[TribeType.Elyrion]: ClimateType.Elyrion,
	[TribeType.Yadakk]: ClimateType.Yadakk,
	[TribeType.Polaris]: ClimateType.Polaris,
	[TribeType.Cymanti]: ClimateType.Cymanti,
	[TribeType.None]: ClimateType.Nature,
	[TribeType.Nature]: ClimateType.Nature
};


Object.freeze(Climate2Tribe);
Object.freeze(Tribe2Climate);

// Predict climate for a fogged tile based on visible neighbors
function predictClimate(state: GameState, tileIndex: number): ClimateType {
    const neighbors = getNeighborIndexes(state, tileIndex, 1, false, false); // Only visible neighbors
    const climateCounts: { [key in ClimateType]?: number } = {};

    // Count climates of visible neighbors
    neighbors.forEach(neighbor => {
        if (state._visibleTiles[neighbor]) {
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
    for (const x in state._visibleTiles) {
        const tileIndex = Number(x);
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
                if (!state.tiles[neighbor]._explorers.has(pov)) {
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

type RowInfo = [number, 0 | 1];

// TODO
function detectEnemyRows(state: GameState): RowInfo[] {
    const rows: RowInfo[] = [];
    const size = state.settings.size;

    for (const x in state._visibleTiles) {
        const tileIndex = Number(x); 
        const tile = state.tiles[tileIndex];
        // Check if tile is enemy-owned
        if (tile._owner !== state.settings._pov && tile._owner) {
            const x = tileIndex % size;
            const y = Math.floor(tileIndex / size);

            // TODO just one
            // Horizontal row: check this tile and the next two to the right or just two, or just one
            if (x <= size - 3) {
                const tile1 = tileIndex;
                const tile2 = tileIndex + 1;
                const tile3 = tileIndex + 2;
                if (
                    state._visibleTiles[tile2] &&
                    state._visibleTiles[tile3] &&
                    state.tiles[tile2]._owner === tile._owner &&
                    state.tiles[tile3]._owner === tile._owner
                ) {
                    rows.push([tile2, 1]); // tile2 is the middle tile
                }
                else if (
                    state._visibleTiles[tile2] &&
                    !state._visibleTiles[tile3] &&
                    state.tiles[tile2]._owner === tile._owner
                ) {
                    // The tile with most fog is the center tile
                    const count1 = getNeighborIndexes(state, tile1, 1, false, true).filter(i => !state._visibleTiles[i]).length;
                    const count2 = getNeighborIndexes(state, tile2, 1, false, true).filter(i => !state._visibleTiles[i]).length;
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
                    state._visibleTiles[tile2] &&
                    state._visibleTiles[tile3] &&
                    state.tiles[tile2]._owner === tile._owner &&
                    state.tiles[tile3]._owner === tile._owner
                ) {
                    rows.push([tile2, 0]); // tile2 is the middle tile
                }
                else if (
                    state._visibleTiles[tile2] &&
                    !state._visibleTiles[tile3] &&
                    state.tiles[tile2]._owner === tile._owner
                ) {
                    // The tile with most fog is the center tile
                    const count1 = getNeighborIndexes(state, tile1, 1, false, true).filter(i => !state._visibleTiles[i]).length;
                    const count2 = getNeighborIndexes(state, tile2, 1, false, true).filter(i => !state._visibleTiles[i]).length;
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
    const visibleTiles = {...state._visibleTiles};
    const borderClouds = new Set<number>();

    for(const x in state._visibleTiles) {
        const neighbors = getNeighborIndexes(state, Number(x), 1, false, true);
        neighbors.forEach(neighbor => {
            // Ensure the neighbor is within bounds and not visible
            if (neighbor >= 0 && neighbor < totalTiles && !visibleTiles[neighbor]) {
                borderClouds.add(neighbor);
            }
        });
    }

    // Step 4: Convert Set to array and return
    return Array.from(borderClouds);
}

// Predict terrain and tribe type of tiles adjacent to fogged tiles
export function predictOuterFogTerrain(
    state: GameState,
    fogPredictions: { [tileIndex: number]: [TerrainType, ClimateType, boolean] }
): { [tileIndex: number]: [TerrainType, ClimateType] } {
    // Update outerPredictions to store [TerrainType, ClimateType] tuples
    const outerPredictions: { [tileIndex: number]: [TerrainType, ClimateType] } = {};
    const fogTiles = Object.keys(fogPredictions).map(x => Number(x));

    // Step 1: Identify outer tiles (adjacent to fog tiles, not visible or fog)
    const outerTiles: { [tileIndex: number]: [TerrainType | null, ClimateType | null] } = {};
    fogTiles.forEach(fogIndex => {
        outerTiles[fogIndex] = [fogPredictions[fogIndex][0], fogPredictions[fogIndex][1]];
        const neighbors = getNeighborIndexes(state, fogIndex, 1, false, true);
        neighbors.forEach(neighbor => {
            if (!state._visibleTiles[neighbor] && !fogTiles.includes(neighbor)) {
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
            if (state._visibleTiles[neighbor]) {
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
    const pov = getPovTribe(state);
    const odds = 
        0.45 +
        (isTechUnlocked(pov, TechnologyType.Fishing)?  0.25 : 0) +
        (isTechUnlocked(pov, TechnologyType.Sailing)?  0.10 : 0) +
        (isTechUnlocked(pov, TechnologyType.Climbing)? 0.10 : 0);
    return shuffleArray(
        getNeighborIndexes(state, tileIndex, 1, false, includeUnexplored)
        // Only if its absolutely visible and steppable, if not then we dont really know what is there
        .filter(x => 
            // isTribeSteppable(state, x) // (cheating)
            state.tiles[x]._explorers.has(pov.owner)? isTribeSteppable(state, x) : 
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
            if (path.length > 1 && !state._visibleTiles[tile]) {
                candidates.push({ path });
            }
            // If we have not exceeded maxDistance, add allowed neighbors.
            if (path.length - 1 < maxDistance) {
                const neighbors = shuffleArray(getAllowedNeighbors(state, tile));
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
 * @returns [path taken, tiles explored]
 */
export function predictExplorer(state: GameState, tileIndex: number): number[] {
    const visible = { ...state._visibleTiles };
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
        state._visibleTiles[nextTile] = true;
        getNeighborIndexes(state, nextTile, 1, false, true).forEach(x => state._visibleTiles[x] = true);
        path.push(nextTile);
        currentTile = nextTile;
    }

    const explored = [];
    
    for(const x in visible) {
        if(!state._visibleTiles[Number(x)]) {
            explored.push(Number(x));
        }
    }

    state._visibleTiles = visible;

    // return [path, explored];
    return explored;
}


// Predicts enemy capitals and surrounding tiles //

export function predictEnemyCapitalsAndSurroundings(state: GameState): number[] {
    const mapSize = state.settings.size;
    const gridSize = getDomainGrid(state.settings.tribeCount); // 2x2 grid
    const totalDomains = gridSize * gridSize; // 4 domains

    // Step 1: Find your domain based on your capital
    const tribe = getPovTribe(state);
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
    for(const x in state._visibleTiles) {
        const tileIndex = Number(x);
        const tile = state.tiles[tileIndex];
        const ownedByEnemy = tile._owner && tile._owner !== tribe.owner;
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
    };

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
    return allPredictedTiles.filter(tile => !state._visibleTiles[tile]);
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
    if(state._prediction === undefined) return [];

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
                        if(explorer.flat().filter(x => state._prediction!._enemyCapitalSuspects?.includes(x)).length >= minAmountOfSus) {
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