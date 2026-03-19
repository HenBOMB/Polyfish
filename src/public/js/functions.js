/**
 * Helper functions for game logic calculation in the frontend.
 */

/**
 * Get adjacent tile indices within a square range.
 */
function getAdjacentIndices(idx, range, size) {
    const cx = idx % size;
    const cy = Math.floor(idx / size);
    const result = [];

    for (let dy = -range; dy <= range; dy++) {
        for (let dx = -range; dx <= range; dx++) {
            if (dx === 0 && dy === 0) continue;
            const nx = cx + dx;
            const ny = cy + dy;
            if (nx >= 0 && nx < size && ny >= 0 && ny < size) {
                result.push(ny * size + nx);
            }
        }
    }
    return result;
}

/**
 * Get the structure at a tile index.
 */
function getStructureAt(state, idx) {
    return state.structures[idx] || null;
}

/**
 * Check if there is an enemy unit at a tile index.
 */
function getEnemyAt(state, idx, owner) {
    // Tribe units are stored in state.tribes[tribeId].units
    for (const tribeId in state.tribes) {
        const tribe = state.tribes[tribeId];
        if (parseInt(tribeId) === owner) continue;

        // Relations check (if not at peace)
        const relations = state.tribes[owner]?.relations || {};
        const isPeace = relations[tribeId]?.state === 1;
        if (isPeace) continue;

        const unit = (tribe.units || []).find(u => u.coords.idx === idx);
        if (unit) {
            // Check for invisibility (EffectType::Invisible = 3)
            if (unit.effects && unit.effects.includes(3)) continue;
            return unit;
        }
    }
    return null;
}

/**
 * Calculate final city production including adjacency bonuses.
 */
function getCityProduction(state, city) {
    if (!city) return 0;

    // If city is on riot or the tile is occupied by an enemy then production is nullified
    if (city._riot || getEnemyAt(state, city.tileIndex, city.owner)) {
        return 0;
    }

    let prod = city.level || 0;

    // Capitals get a +1 star bonus
    const centerTile = state.map.tiles[city.tileIndex];
    if (centerTile && centerTile.capitalOf === city.owner && centerTile.capitalOf !== 0) {
        prod += 1;
    }

    const rewards = (city.rewards || []).filter(r => r === RewardTypes.Park || r === 2);
    prod += rewards.length;

    const size = state.settings.size;

    // Adjacency Bonuses
    if (city._territory) {
        for (const idx of city._territory) {
            const structure = getStructureAt(state, idx);
            if (!structure) continue;

            const type = structure.structureType || structure.type;

            if (type === 120) { // Clathrus (Cymanti)
                // +1 star for each adjacent Algae in friendly territory
                const adj = getAdjacentIndices(idx, 1, size);
                for (const nIdx of adj) {
                    const nTile = state.map.tiles[nIdx];
                    if (nTile && nTile.terrainType === 120 && nTile.owner === city.owner) { // Algae terrain = 120
                        prod += 1;
                    }
                }
            } else if (type === 50) { // Market
                // +1 star for each adjacent "production" building (Sawmill=13, Windmill=6, Forge=22)
                const adj = getAdjacentIndices(idx, 1, size);
                for (const nIdx of adj) {
                    const nStruct = getStructureAt(state, nIdx);
                    if (nStruct) {
                        const nType = nStruct.structureType || nStruct.type;
                        if ([13, 6, 22].includes(nType)) {
                            prod += 1;
                        }
                    }
                }
            } else if (type === 121) { // Sanctuary (Elyrion)
                // +1 star for each adjacent animal (ResourceTypes: Game=1)
                const adj = getAdjacentIndices(idx, 1, size);
                for (const nIdx of adj) {
                    const nRes = state.resources[nIdx];
                    if (nRes && (nRes.resourceType === 1 || nRes.type === 1)) {
                        prod += 1;
                    }
                }
            }
        }
    }

    return prod;
}

// Export for use in other scripts if needed, though most are just loaded via <script>
window.getCityProduction = getCityProduction;
window.getAdjacentIndices = getAdjacentIndices;
window.getStructureAt = getStructureAt;
window.getEnemyAt = getEnemyAt;