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
 * Final city production, computed server-side by `functions::get_city_production`
 * and serialized onto the city. The rule (siege, capital bonus, Park/Workshop,
 * Market income by hub level, territory overlap) lives in Rust only — the
 * re-implementation that used to sit here drifted on every one of those points.
 */
function getCityProduction(state, city) {
    return city?.production ?? 0;
}

// Export for use in other scripts if needed, though most are just loaded via <script>
window.getCityProduction = getCityProduction;
window.getAdjacentIndices = getAdjacentIndices;
window.getStructureAt = getStructureAt;
window.getEnemyAt = getEnemyAt;