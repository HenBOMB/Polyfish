const tileSize = 128;
const TILE_OFFSET = 4;
const mapContainer = document.getElementById("map");

let GAME_STATE = {};
let currentLegalMoves = [];
let selectedUnitIdx = null; // Currently selected unit's tile index
let ENABLE_FOW = true; // Fog of War toggle

// UI Elements
const turnVal = document.getElementById('turn-val');
const tribeNameLabel = document.getElementById('tribe-name');
const starsVal = document.getElementById('stars-val');
const scoreVal = document.getElementById('score-val');
const incomeVal = document.getElementById('income-val');
const techList = document.getElementById('tech-list');
const movesList = document.getElementById('moves-list');
const lastMoveVal = document.getElementById('last-move-val');
const mctsDepth = document.getElementById('mcts-depth');
const mctsDepthVal = document.getElementById('mcts-depth-val');

class MapRenderer {
    constructor(container) {
        this.container = container;
        this.elements = new Map(); // idx -> { ground, unit, city, structures, resources, fog }
        this.selectedIdx = null;
        this.visibleMap = {};
    }

    clear() {
        this.container.innerHTML = '';
        this.elements.clear();
    }

    getPos(x, y) {
        const posX = (x - y) * (tileSize / 2 - TILE_OFFSET);
        const posY = (x + y) * (tileSize / 4 + TILE_OFFSET);
        return { x: posX, y: posY };
    }

    render(state, legalMoves) {
        const currentTribeId = state.settings.currentPlayerTurnId;
        this.visibleMap = state._visibleTiles || state.visibleTiles || state._visible_tiles || {};

        const unitsByIndex = {};
        const citiesByIndex = {};
        Object.values(state.tribes).forEach(tribe => {
            (tribe.units || []).forEach(u => unitsByIndex[u.coords.idx] = { ...u, tribe });
            (tribe.cities || []).forEach(c => citiesByIndex[c.tileIndex] = { ...c, tribe });
        });

        const allTiles = Object.values(state.tiles).sort((a, b) => a.coords.idx - b.coords.idx);

        allTiles.forEach(tile => {
            this.renderTile(tile, state, unitsByIndex, citiesByIndex, currentTribeId);
        });

        this.renderMoveOverlays(legalMoves);
    }

    renderTile(tile, state, unitsByIndex, citiesByIndex, currentTribeId) {
        const idx = tile.coords.idx;
        const { x, y } = tile.coords;
        const pos = this.getPos(x, y);

        let data = this.elements.get(idx);
        if (!data) {
            data = { layers: {} };
            this.elements.set(idx, data);
        }

        // FOW Logic
        let isExplored = true;
        let isVisible = true;
        if (ENABLE_FOW) {
            if (tile.explorers && !tile.explorers.includes(currentTribeId)) isExplored = false;
            if (!this.visibleMap[idx] && !this.visibleMap[idx.toString()]) isVisible = false;
        }

        if (!isExplored) {
            this.updateLayer(idx, 'ground', 'terrain/tiles/undiscovered', pos, 10, ['ground', 'undiscovered']);
            this.removeLayer(idx, 'unit');
            this.removeLayer(idx, 'city');
            this.removeLayer(idx, 'resource');
            this.removeLayer(idx, 'structure');
            this.removeLayer(idx, 'ambient');
            return;
        }

        // Ground
        const tilefile = [null, 'terrain/water/water', 'terrain/water/ocean', null, null, null, 'terrain/tiles/ice'][tile.type]
            || `terrain/tiles/ground_${tile.climate}`;

        const groundEl = this.updateLayer(idx, 'ground', tilefile, pos, 0, ['ground']);
        groundEl.dataset.tileIdx = idx;

        if (!isVisible) groundEl.classList.add('fog');
        else groundEl.classList.remove('fog');

        // Ambient (Mountains/Forests)
        if (tile.type === 4) { // Mountain
            this.updateLayer(idx, 'ambient', `terrain/mountains/mountain_${tile.climate}`, pos, 2000, ['mountain', !isVisible ? 'fog' : '']);
        } else if (tile.type === 5) { // Forest
            this.updateLayer(idx, 'ambient', `terrain/forests/Forest_${tile.climate}`, pos, 2000, ['forest', !isVisible ? 'fog' : '']);
        } else {
            this.removeLayer(idx, 'ambient');
        }

        // Resources & Structures
        const struct = state.structures[idx];
        const res = state.resources[idx];

        if (struct && struct.type === 71) { // Road
            this.updateLayer(idx, 'road', 'misc/Road', pos, 1500, ['structure', 'road', !isVisible ? 'fog' : '']);
        } else {
            this.removeLayer(idx, 'road');
        }

        if (res) {
            const resFile = getResourceFile(res.type, tile.climate);
            const resClass = { 1: 'animal', 2: 'crop', 3: 'fish', 5: 'metal', 6: 'fruit' }[res.type];
            this.updateLayer(idx, 'resource', resFile, pos, 2500, ['resource', resClass, !isVisible ? 'fog' : '']);
        } else {
            this.removeLayer(idx, 'resource');
        }

        if (struct && struct.type !== 71) {
            // Skip rendering village if there's a city here
            if (struct.type === 1 && citiesByIndex[idx]) {
                this.removeLayer(idx, 'structure');
            } else {
                const structFile = getStructureFile(struct.type, tile.climate);
                const classes = ['structure'];
                if (struct.type === 1) classes.push('village');
                if (struct.type === 2) classes.push('ruins');
                if (struct.type === 29) classes.push('monument');
                if (!isVisible) classes.push('fog');
                this.updateLayer(idx, 'structure', structFile, pos, 3000, classes);
            }
        } else {
            this.removeLayer(idx, 'structure');
        }

        // Cities
        const city = citiesByIndex[idx];
        if (city) {
            const tribeName = TRIBE_ID_2_NAME[city.tribe.type];
            const climateIndex = CLIMATE_IDS.indexOf(tribeName);
            const cityEl = this.updateLayer(idx, 'city', `buildings/${tribeName}/Default/Houses/House_${climateIndex}_5`, pos, 4000, ['city', !isVisible ? 'fog' : '']);
            const rewards = Object.keys(city.rewards).map(r => RewardEmojis[r]).join('');
            // RewardEmojis[move.reward]
            const unitCount = Object.values(GAME_STATE.tribes).flatMap(t => t.units).filter(u => u.cityId === city.id).length;
            cityEl.innerHTML = `<div class="city-stats">
                <span>${tile.capitalOf > 0 ? '👑 ' : ''}${city.name || 'City'} Lvl ${city.level}</span>
                <span>${city.connectedToCapital ? '🔗' : ''}${rewards}</span>
                <span>${new Array(unitCount).fill('🪖').join('')}</span>
                <span>+${city.production} 💰</span>
                <span>${city.population} 😀</span>
            </div>`;
        } else {
            this.removeLayer(idx, 'city');
        }

        // Units
        const unit = (isVisible || !ENABLE_FOW) ? unitsByIndex[idx] : null;
        if (unit) {
            const tribeName = TRIBE_ID_2_NAME[unit.tribe.type];
            const className = ClassNameToId[unit.unitType || unit.type];
            if (className) {
                const classes = ['unit'];
                if (unit.moved || unit.attacked) classes.push('exausted');
                if (unit.flipped) classes.push('flipped');
                if (this.selectedIdx === idx) classes.push('selected-unit-highlight');

                const unitEl = this.updateLayer(idx, 'unit', `units/${tribeName}/default/${tribeName}_default_${className}`, pos, 5000, classes);
                unitEl.innerHTML = `<div class="health">${Math.floor(unit.health / 10)}</div>`;
            }
        } else {
            this.removeLayer(idx, 'unit');
        }
    }

    updateLayer(idx, layerName, filename, pos, zIndex, classes = []) {
        let data = this.elements.get(idx);
        let el = data.layers[layerName];

        if (!el) {
            el = document.createElement('div');
            el.classList.add('tile');
            this.container.appendChild(el);
            data.layers[layerName] = el;
        }

        el.style.backgroundImage = `url('textures/${filename}.png')`;
        el.style.left = `${pos.x}px`;
        el.style.top = `${pos.y}px`;
        el.style.zIndex = Math.floor(pos.y + zIndex);

        // Reset classes and apply new ones
        el.className = 'tile';
        classes.forEach(c => { if (c) el.classList.add(c) });

        // If this is the ground layer, ensure it has a click mask
        if (layerName === 'ground') {
            this.updateClickMask(idx, pos);
        }

        return el;
    }

    updateClickMask(idx, pos) {
        let data = this.elements.get(idx);
        let mask = data.layers['click-mask'];

        if (!mask) {
            mask = document.createElement('div');
            mask.classList.add('tile', 'click-mask');
            this.container.appendChild(mask);
            data.layers['click-mask'] = mask;
        }

        mask.style.left = `${pos.x}px`;
        mask.style.top = `${pos.y}px`;
        mask.style.zIndex = Math.floor(pos.y + 10000); // Topmost for interaction

        mask.onclick = (e) => {
            const unit = this.getUnitAt(idx);
            this.handleTileClick(e, idx, unit, this.isTileVisible(idx));
        };
        this.setupHover(mask, idx, this.isTileVisible(idx));
    }

    isTileVisible(idx) {
        if (!ENABLE_FOW) return true;
        return !!(this.visibleMap[idx] || this.visibleMap[idx.toString()]);
    }

    removeLayer(idx, layerName) {
        const data = this.elements.get(idx);
        if (data && data.layers[layerName]) {
            data.layers[layerName].remove();
            delete data.layers[layerName];
        }
    }

    handleTileClick(e, idx, unit, isVisible) {
        e.stopPropagation();
        if (ENABLE_FOW && !isVisible) {
            if (this.selectedIdx !== null) {
                this.selectedIdx = null;
                this.render(GAME_STATE, currentLegalMoves);
            }
            return;
        }

        if (unit && (isVisible || !ENABLE_FOW)) {
            // Priority selection if unit exists
            this.selectedIdx = (this.selectedIdx === idx) ? null : idx;
            selectedUnitIdx = this.selectedIdx;
        } else {
            this.selectedIdx = null;
            selectedUnitIdx = null;
        }
        this.render(GAME_STATE, currentLegalMoves);
    }

    setupHover(el, idx, isVisible) {
        el.onmouseenter = (e) => {
            if (!this.isTileVisible(idx)) return;
            const tile = GAME_STATE.tiles[idx];
            if (!tile) return;

            hoverEl.classList.remove('hidden');
            const unit = (isVisible || !ENABLE_FOW) ? this.getUnitAt(idx) : null;
            const struct = GAME_STATE.structures[idx];
            const resource = GAME_STATE.resources[idx];

            let html = `<strong>Tile ${idx} (${tile.coords.x}, ${tile.coords.y})</strong><br>`;
            html += `⛰️ ${TerrainType[tile.type] || tile.type} (${tile.climate})<br>`;
            if (unit) html += `🪖 ${TRIBE_ID_2_NAME[unit.tribe.type]} ${ClassNameToId[unit.type || unit.unitType]} (${unit.health / 10}/${unit.maxHealth / 10})<br>`;
            if (struct) html += `🗼 ${StructureNames[struct.type] || struct.type}<br>`;
            if (resource) html += `🥝 ${ResourceTypes[resource.type] || resource.type}<br>`;

            hoverEl.innerHTML = html;

            // Highlight the visual tile
            const data = this.elements.get(idx);
            if (data && data.layers['ground']) {
                data.layers['ground'].classList.add('tile-hover-highlight');
            }
        };

        el.onmousemove = (e) => {
            hoverEl.style.left = `${e.clientX + 15}px`;
            hoverEl.style.top = `${e.clientY + 15}px`;
        };

        el.onmouseleave = () => {
            hoverEl.classList.add('hidden');
            const data = this.elements.get(idx);
            if (data && data.layers['ground']) {
                data.layers['ground'].classList.remove('tile-hover-highlight');
            }
        };
    }

    getUnitAt(idx) {
        let found = null;
        Object.values(GAME_STATE.tribes).forEach(tribe => {
            const u = (tribe.units || []).find(u => u.coords.idx === idx);
            if (u) found = { ...u, tribe };
        });
        return found;
    }

    renderMoveOverlays(legalMoves) {
        document.querySelectorAll('.move-overlay').forEach(el => el.remove());
        if (this.selectedIdx === null) return;

        const unitMoves = legalMoves.filter(m => m && typeof m === 'object' && m.src === this.selectedIdx);

        unitMoves.forEach(move => {
            const targetIdx = move.target;
            if (targetIdx === undefined || targetIdx === null) return;

            const targetTile = GAME_STATE.tiles[targetIdx];
            if (!targetTile) return;

            const pos = this.getPos(targetTile.coords.x, targetTile.coords.y);
            const overlay = document.createElement('div');
            overlay.classList.add('tile', 'move-overlay');
            overlay.style.left = `${pos.x}px`;
            overlay.style.top = `${pos.y}px`;
            overlay.style.zIndex = Math.floor(pos.y + 20000); // Higher than click-mask (10000)

            const img = document.createElement('img');
            img.src = move.moveType === 2 ? 'textures/misc/attackTarget.png' : 'textures/misc/moveTarget.png';
            img.style.width = '128px';
            overlay.appendChild(img);

            overlay.onclick = (e) => {
                e.stopPropagation();
                playMove(move);
            };
            this.container.appendChild(overlay);
        });
    }
}

const renderer = new MapRenderer(mapContainer);
const hoverEl = document.getElementById('hovertile');

async function apiAction(endpoint, body) {
    document.querySelectorAll('.btn').forEach(b => b.disabled = true);
    try {
        const res = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        const data = await res.json();
        updateUI(data);
    } catch (e) {
        console.error("API Error:", e);
    } finally {
        document.querySelectorAll('.btn').forEach(b => b.disabled = false);
    }
}

function updateUI(data) {
    const oldTribeId = (GAME_STATE.settings) ? GAME_STATE.settings.currentPlayerTurnId : null;
    if (data.state) GAME_STATE = data.state;
    if (data.legalMoves) currentLegalMoves = data.legalMoves;
    if (data.movePlayed) lastMoveVal.textContent = data.movePlayed;

    const currentTribeId = GAME_STATE.settings.currentPlayerTurnId;
    const currentTribe = GAME_STATE.tribes[currentTribeId.toString()] || GAME_STATE.tribes[currentTribeId];

    if (!currentTribe) return;

    // Update Stats
    turnVal.textContent = GAME_STATE.settings.turn;
    const currentTribeName = TRIBE_ID_2_NAME[currentTribe.type];
    tribeNameLabel.textContent = currentTribeName || 'Unknown';
    starsVal.textContent = currentTribe.stars;
    scoreVal.textContent = currentTribe.score;

    const income = currentTribe.cities.reduce((acc, cur) => {
        let prod = (cur.production || 0);
        const cityTile = GAME_STATE.tiles[cur.tileIndex];
        if (cityTile && cityTile.capitalOf > 0) prod += 1;
        return acc + prod;
    }, 0);
    incomeVal.textContent = `+${income}`;

    // Tech Tree update (kept similar but using helper if needed)
    renderTechTree(currentTribe);

    // Moves List update
    renderMovesList(currentLegalMoves);

    renderer.render(GAME_STATE, currentLegalMoves);

    // If turn changed, pan smoothly to next player's capital
    if (oldTribeId !== null && oldTribeId !== currentTribeId) {
        setTimeout(() => focusCamera(true), 100);
    }
}

function renderTechTree(tribe) {
    techList.innerHTML = '';
    const unlockedTechs = (tribe.tech_vanilla || []).map(t => t.type);
    const researchableTechs = new Set();

    if (TechTree[0]) TechTree[0].forEach(t => { if (!unlockedTechs.includes(t)) researchableTechs.add(t); });
    unlockedTechs.forEach(techId => {
        (TechTree[techId] || []).forEach(t => { if (!unlockedTechs.includes(t)) researchableTechs.add(t); });
    });

    Object.entries(TechnologyNames).forEach(([id, name]) => {
        const techId = parseInt(id);
        if (unlockedTechs.includes(techId)) {
            const badge = document.createElement('div');
            badge.className = 'tech-badge unlocked';
            badge.textContent = name;
            techList.appendChild(badge);
        }
    });

    researchableTechs.forEach(techId => {
        const name = TechnologyNames[techId];
        const badge = document.createElement('div');
        badge.className = 'tech-badge';
        badge.style.border = '1px dashed var(--gold)';
        badge.style.color = 'var(--gold)';
        badge.textContent = `→ ${name}`;

        const move = currentLegalMoves.find(m => m && (m.moveType === 7 || m.tech !== undefined) && m.tech === techId);
        if (move) {
            badge.style.cursor = 'pointer';
            badge.onclick = () => playMove(move);
        } else {
            badge.style.opacity = '0.5';
            badge.style.cursor = 'not-allowed';
        }
        techList.appendChild(badge);
    });
}

function renderMovesList(moves) {
    movesList.innerHTML = '';
    const MoveTypeNames = {
        0: 'None', 1: 'Step', 2: 'Attack', 3: 'Ability', 4: 'Summon',
        5: 'Harvest', 6: 'Build', 7: 'Research', 8: 'Capture', 9: 'Reward', 10: 'EndTurn'
    };

    const uniqueMoves = [];
    const moveStrings = new Set();
    (moves || []).forEach(move => {
        if (!move) return;
        const s = JSON.stringify(move);
        if (!moveStrings.has(s)) {
            moveStrings.add(s);
            uniqueMoves.push(move);
        }
    });

    uniqueMoves.slice(0, 50).forEach(move => {
        const li = document.createElement('li');
        li.style.cursor = 'pointer';
        li.classList.add('move-item');

        const moveType = move.moveType !== undefined ? move.moveType : (move.tech !== undefined ? 7 : (move.structure !== undefined ? 6 : (move.ability !== undefined ? 3 : (move.reward !== undefined ? 9 : 0))));

        let text = '';
        const typeName = MoveTypeNames[moveType] || moveType;
        const resource = GAME_STATE.resources[move.target];
        const tile = GAME_STATE.tiles[move.target];
        const structure = GAME_STATE.structures[move.target || move.src];

        if (moveType === 4) {
            const isUpgrade = move.upgrade === true;
            text = `${isUpgrade ? 'Upgrade' : 'Summon'} ${UnitTypes[move.type] || move.type}`;
        }
        else if (moveType === 7) text = `Research ${TechnologyNames[move.tech] || move.tech}`;
        else if (moveType === 6) text = `🔨 ${StructureNames[move.structure]} @ ${move.tileIndex}`;
        else if (moveType === 5) text = `🥝 ${resource ? ResourceTypes[resource.type] : 'Resource'}`;
        else if (moveType === 8) text = `Capture ${tile && tile.owner > 0 ? 'City' : StructureNames[structure.type]}`;
        else if (moveType === 9) text = `${RewardEmojis[move.reward]} ${RewardTypes[move.reward]}`;
        else if (moveType === 10) text = 'End Turn';
        else text = `${typeName} (${moveType}) ${move.src ?? move.target ?? ''} → ${move.target ?? ''}`;

        li.textContent = text;
        li.onclick = () => playMove(move);
        movesList.appendChild(li);
    });
}

function playMove(move) {
    fetch('/step', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(move)
    })
        .then(r => r.json())
        .then(data => {
            renderer.selectedIdx = null;
            selectedUnitIdx = null;
            updateUI(data);
        })
        .catch(err => console.error('Error playing move:', err));
}

function getStructureFile(type, climate) {
    const map = {
        1: `buildings/common/Tribe`,
        2: `terrain/misc/ResourceGFX_ruin`,
        5: `buildings/common/Farm`,
        6: `buildings/common/Windmill`,
        8: `buildings/common/Port`,
        12: `buildings/common/Lumber Hut`,
        13: `buildings/common/Sawmill`,
        21: `buildings/common/Mine`,
        22: `buildings/common/Forge`,
        29: `buildings/${CLIMATE_IDS[climate]}/Default/Monuments/Monument7_${climate}`,
        71: `misc/Road`
    };
    return map[type] || 'misc/missing';
}

function getResourceFile(type, climate) {
    const map = {
        1: `animals/${CLIMATE_TO_ANIMAL[climate]}`,
        2: `terrain/misc/ResourceGFX_crop`,
        3: `animals/fish`,
        5: `terrain/misc/ResourceGFX_metal`,
        6: `fruits/ResourceGFX_fruit_${climate}`,
        8: `terrain/misc/ResourceGFX_starfish`
    };
    return map[type] || 'misc/missing';
}

// Camera and Zoom logic (keeping existing as it works well)
let scale = 0.5;
let translateX = 0;
let translateY = 0;
const mapViewport = document.getElementById('map-viewport');

function updateTransform() {
    mapContainer.style.transform = `translate(${translateX}px, ${translateY}px) scale(${scale})`;
}

function centerOnCoordinates(tX, tY, smooth = false) {
    const pos = renderer.getPos(tX, tY);
    const viewportRect = mapViewport.getBoundingClientRect();
    if (viewportRect.width === 0) return;
    translateX = (viewportRect.width / 2) - (pos.x * scale);
    translateY = (viewportRect.height / 2) - (pos.y * scale);

    if (smooth) {
        mapContainer.style.transition = 'transform 0.8s cubic-bezier(0.4, 0, 0.2, 1)';
        updateTransform();
        setTimeout(() => {
            mapContainer.style.transition = '';
        }, 850);
    } else {
        updateTransform();
    }
}

function focusCamera(smooth = false) {
    if (!GAME_STATE.settings) return;
    const currentTribeId = GAME_STATE.settings.currentPlayerTurnId;
    const tribe = GAME_STATE.tribes[currentTribeId.toString()] || GAME_STATE.tribes[currentTribeId] || Object.values(GAME_STATE.tribes)[0];

    if (tribe && tribe.cities && tribe.cities.length > 0) {
        // Find capital if possible
        let cityToFocus = tribe.cities[0];
        const capital = tribe.cities.find(c => {
            const tile = GAME_STATE.tiles[c.tileIndex];
            return tile && tile.capitalOf > 0;
        });
        if (capital) cityToFocus = capital;

        const cityTile = GAME_STATE.tiles[cityToFocus.tileIndex];
        centerOnCoordinates(cityTile.coords.x, cityTile.coords.y, smooth);
    } else {
        centerOnCoordinates(8, 8, smooth);
    }
}

window.addEventListener('load', () => {
    fetch('/current').then(r => r.json()).then(data => {
        updateUI(data);
        setTimeout(focusCamera, 100);
    });

    // Drag, Zoom, Event listeners...
    let dragging = false, lx, ly;
    mapViewport.addEventListener('mousedown', e => {
        if (e.button !== 0) return;
        dragging = true; lx = e.clientX; ly = e.clientY;
        mapViewport.style.cursor = 'grabbing';
    });
    window.addEventListener('mousemove', e => {
        if (!dragging) return;
        translateX += (e.clientX - lx);
        translateY += (e.clientY - ly);
        lx = e.clientX; ly = e.clientY;
        updateTransform();
    });
    window.addEventListener('mouseup', () => {
        dragging = false;
        mapViewport.style.cursor = 'default';
    });
    mapViewport.addEventListener('wheel', e => {
        e.preventDefault();
        const rect = mapViewport.getBoundingClientRect();
        const oldScale = scale;
        scale = Math.min(2, Math.max(0.2, scale * (e.deltaY > 0 ? 0.9 : 1.1)));
        translateX = (rect.width / 2) - ((rect.width / 2) - translateX) * (scale / oldScale);
        translateY = (rect.height / 2) - ((rect.height / 2) - translateY) * (scale / oldScale);
        updateTransform();
    }, { passive: false });
});

// Event Listeners for buttons
const trainingUI = document.getElementById('training-ui');
const trainingIndicator = document.getElementById('training-indicator');
const trainingLog = document.getElementById('training-log');

async function pollTrainingStatus() {
    try {
        const res = await fetch('/train/status');
        if (!res.ok) return;
        const data = await res.json();

        if (data.pid) {
            trainingUI.classList.remove('hidden');
            if (data.isRunning) {
                trainingIndicator.textContent = "Running (PID: " + data.pid + ")";
                trainingIndicator.className = "stat-value small green";
            } else {
                trainingIndicator.textContent = "Finished / Stopped";
                trainingIndicator.className = "stat-value small";
            }
            if (data.log) {
                trainingLog.textContent = data.log;
                trainingLog.scrollTop = trainingLog.scrollHeight;
            }
        } else {
            trainingUI.classList.add('hidden');
        }
    } catch (e) {
        console.error("Status check failed", e);
    }
}

setInterval(pollTrainingStatus, 2000);

document.getElementById('btn-reset').onclick = () => apiAction('/reset', {});
document.getElementById('btn-fow').onclick = () => { ENABLE_FOW = !ENABLE_FOW; renderer.render(GAME_STATE, currentLegalMoves); };
document.getElementById('btn-train').onclick = () => {
    if (confirm("Start a background training session? This will run 'cargo run --bin self_play' on the server.")) {
        apiAction('/train', {}).then(data => {
            if (data && data.message) alert(data.message);
        });
    }
};
document.getElementById('btn-rng').onclick = () => apiAction('/rngstep', {});
document.getElementById('btn-step').onclick = () => apiAction('/autostep', { iterations: parseInt(mctsDepth.value) });
mctsDepth.oninput = (e) => mctsDepthVal.textContent = e.target.value;
