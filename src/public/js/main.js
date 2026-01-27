const tileSize = 128;
const someOffset = 4;
const mapContainer = document.getElementById("map");

let GAME_STATE = {};
let TILE_ELEMENTS = {};
let lastLegalMoves = [];
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

// Event Listeners
document.getElementById('btn-reset').onclick = () => apiAction('/reset', {});
document.getElementById('btn-fow').onclick = () => {
    ENABLE_FOW = !ENABLE_FOW;
    renderMap();
};
document.getElementById('btn-rng').onclick = () => apiAction('/rngstep', {});
document.getElementById('btn-step').onclick = () => {
    const iterations = parseInt(mctsDepth.value);
    apiAction('/autostep', { iterations });
};
mctsDepth.oninput = (e) => mctsDepthVal.textContent = e.target.value;

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
    if (data.state) GAME_STATE = data.state;
    if (data.legalMoves) lastLegalMoves = data.legalMoves;
    if (data.movePlayed) lastMoveVal.textContent = data.movePlayed;

    const currentTribeId = GAME_STATE.settings.currentPlayerTurnId;
    const currentTribe = GAME_STATE.tribes[currentTribeId.toString()] || GAME_STATE.tribes[currentTribeId];

    if (!currentTribe) {
        console.error("Current tribe not found for ID:", currentTribeId);
        return;
    }

    // Update Stats
    turnVal.textContent = GAME_STATE.settings.turn;
    // Mapgen.rs uses #[serde(rename = "type")] for tribe_type
    const currentTribeName = TRIBE_ID_2_NAME[currentTribe.type];
    if (tribeNameLabel) tribeNameLabel.textContent = currentTribeName || 'Unknown';
    starsVal.textContent = currentTribe.stars;
    scoreVal.textContent = currentTribe.score;

    // Calculate income: base production + 1 bonus only for capitals
    const income = currentTribe.cities.reduce((acc, cur) => {
        let prod = (cur.production || 0);
        const cityTile = GAME_STATE.tiles[cur.tileIndex];
        if (cityTile && cityTile.capitalOf > 0) prod += 1;
        return acc + prod;
    }, 0);
    incomeVal.textContent = `+${income}`;

    // Update Tech Tree
    techList.innerHTML = '';
    // TechnologyState has #[serde(rename = "type")] for tech_type
    const unlockedTechs = (currentTribe.tech_vanilla || []).map(t => t.type);

    // Find researchable techs (techs whose prerequisites are met)
    const researchableTechs = new Set();
    // Start with tier 1 techs (always available via implicit Unrequired)
    if (TechTree[0]) {
        TechTree[0].forEach(t => {
            if (!unlockedTechs.includes(t)) researchableTechs.add(t);
        });
    }
    // Add techs unlocked by discovered techs
    unlockedTechs.forEach(techId => {
        const nextTechs = TechTree[techId] || [];
        nextTechs.forEach(t => {
            if (!unlockedTechs.includes(t)) researchableTechs.add(t);
        });
    });

    // Show unlocked techs first
    Object.entries(TechnologyNames).forEach(([id, name]) => {
        const techId = parseInt(id);
        const isUnlocked = unlockedTechs.includes(techId);
        if (!isUnlocked) return;

        const badge = document.createElement('div');
        badge.className = 'tech-badge unlocked';
        badge.textContent = name;
        techList.appendChild(badge);
    });

    // Show researchable techs
    researchableTechs.forEach(techId => {
        const name = TechnologyNames[techId];
        if (!name) return;

        const badge = document.createElement('div');
        badge.className = 'tech-badge';
        badge.style.border = '1px dashed var(--gold)';
        badge.style.color = 'var(--gold)';
        badge.textContent = `→ ${name}`;

        // Find if this is a legal move
        const researchMove = lastLegalMoves.find(m =>
            typeof m === 'object' && m.moveType === 7 && m.tech === techId
        );

        if (researchMove) {
            badge.style.cursor = 'pointer';
            badge.onclick = () => playMove(researchMove);
            badge.title = "Click to research";
            badge.onmouseover = () => badge.style.backgroundColor = 'rgba(255, 215, 0, 0.1)';
            badge.onmouseout = () => badge.style.backgroundColor = 'transparent';
        } else {
            badge.style.opacity = '0.5';
            badge.style.cursor = 'not-allowed';
            badge.title = "Not enough stars";
        }

        techList.appendChild(badge);
    });

    if (unlockedTechs.length === 0 && researchableTechs.size === 0) {
        const emptyMsg = document.createElement('div');
        emptyMsg.className = 'tech-badge';
        emptyMsg.style.background = 'transparent';
        emptyMsg.style.border = '1px dashed var(--text-dim)';
        emptyMsg.style.color = 'var(--text-dim)';
        emptyMsg.textContent = 'No technologies';
        techList.appendChild(emptyMsg);
    }

    // Update Moves List
    movesList.innerHTML = '';
    const MoveTypeNames = {
        0: 'None', 1: 'Step', 2: 'Attack', 3: 'Ability', 4: 'Summon',
        5: 'Harvest', 6: 'Build', 7: 'Research', 8: 'Capture', 9: 'Reward', 10: 'EndTurn'
    };
    lastLegalMoves.slice(0, 50).forEach(move => {
        const li = document.createElement('li');
        li.style.cursor = 'pointer';
        li.classList.add('move-item'); // Add style for hover if needed

        let moveText = '';
        if (typeof move === 'object') {
            const typeName = MoveTypeNames[move.moveType] || move.moveType;
            if (move.moveType === 7) { // Research
                const techName = TechnologyNames[move.tech] || move.tech;
                moveText = `Research ${techName}`;
            } else if (move.moveType === 6) { // Build
                moveText = `Build ${StructureNames[move.structure]} @ ${move.tileIndex}`;
            } else if (move.moveType === 5) { // Harvest
                moveText = `Harvest @ ${ResourceType[move.target]}`;
            } else if (move.moveType === 8) { // Capture
                moveText = `Capture @ ${move.src}`;
            } else if (move.moveType === 10) { // End Turn
                moveText = 'End Turn';
            } else {
                moveText = `${typeName} ${move.src ?? move.target ?? ''} → ${move.target ?? ''}`;
            }
        } else {
            moveText = String(move);
        }
        li.textContent = moveText;

        li.onclick = () => playMove(move);
        movesList.appendChild(li);
    });
    if (lastLegalMoves.length > 50) {
        const li = document.createElement('li');
        li.textContent = `... and ${lastLegalMoves.length - 50} more`;
        movesList.appendChild(li);
    }

    renderMap();
}

function renderMap() {
    mapContainer.innerHTML = '';
    TILE_ELEMENTS = {};

    const mapSize = GAME_STATE.settings.size;
    const currentTribeId = GAME_STATE.settings.currentPlayerTurnId;

    // Determine visibility maps
    // _visible_tiles renamed to _visibleTiles via camelCase?
    // Let's handle likely cases
    const visibleMap = GAME_STATE._visibleTiles || GAME_STATE.visibleTiles || GAME_STATE._visible_tiles || {};

    function createTile(x, y, filename, z = 0) {
        const tile = document.createElement("div");
        tile.classList.add("tile");
        tile.style.backgroundImage = `url('textures/${filename}.png')`;

        const posX = (x - y) * (tileSize / 2 - someOffset);
        const posY = (x + y) * (tileSize / 4 + someOffset);

        tile.style.left = `${posX}px`;
        tile.style.top = `${posY}px`;
        tile.style.zIndex = Math.floor(posY + z);
        mapContainer.appendChild(tile);
        return tile;
    }

    const unitsByIndex = {};
    const citiesByIndex = {};
    Object.values(GAME_STATE.tribes).forEach(tribe => {
        (tribe.units || []).forEach(u => unitsByIndex[u.coords.idx] = { ...u, tribe });
        (tribe.cities || []).forEach(c => citiesByIndex[c.tileIndex] = { ...c, tribe });
    });

    // GAME_STATE.tiles is a HashMap (object)
    const allTiles = Object.values(GAME_STATE.tiles).sort((a, b) => a.coords.idx - b.coords.idx);

    allTiles.forEach(tile => {
        const { x, y } = tile.coords;
        const idx = tile.coords.idx;

        // FOW Logic
        let isExplored = true;
        let isVisible = true;

        if (ENABLE_FOW) {
            // Check exploration
            if (tile.explorers && !tile.explorers.includes(currentTribeId)) {
                isExplored = false;
            }
            // Check visibility
            // Map keys might be strings in JS
            if (!visibleMap[idx] && !visibleMap[idx.toString()]) {
                isVisible = false;
            }
        }

        if (!isExplored) {
            // Render Clouds / Undiscovered
            const cloud = createTile(x, y, 'terrain/tiles/undiscovered', 10);
            return; // Skip contents
        }

        // Ground
        // TileState.terrain_type is renamed to "type"
        const tilefile = [null, 'terrain/water/water', 'terrain/water/ocean', null, null, null, 'terrain/tiles/ice'][tile.type]
            || `terrain/tiles/ground_${tile.climate}`;
        const ground = createTile(x, y, tilefile);
        ground.classList.add('ground');
        ground.dataset.tileIdx = tile.coords.idx;

        if (!isVisible) {
            ground.classList.add('fog');
        }

        // Click handler for unit selection
        ground.addEventListener('click', (e) => {
            console.log('click', e);
            e.stopPropagation();
            const idx = parseInt(ground.dataset.tileIdx);

            // Allow selecting only visible units? Or any if FOW off?
            // If FOW is on and tile is in fog, we shouldn't see units, so no selection.
            if (ENABLE_FOW && !isVisible) {
                // Deselect if clicking into fog
                if (selectedUnitIdx !== null) {
                    selectedUnitIdx = null;
                    renderMoveOverlays();
                }
                return;
            }

            const hasUnit = unitsByIndex[idx];
            if (hasUnit && (isVisible || !ENABLE_FOW)) {
                // Toggle selection
                if (selectedUnitIdx === idx) {
                    selectedUnitIdx = null;
                } else {
                    selectedUnitIdx = idx;
                }
            } else {
                // Clicked on empty tile, deselect
                selectedUnitIdx = null;
            }
            renderMoveOverlays();
        });

        // Ambient (Mountains/Forests)
        if (tile.type === 4) {
            const m = createTile(x, y, `terrain/mountains/mountain_${tile.climate}`, 3);
            m.classList.add('mountain');
            if (!isVisible) m.classList.add('fog');
        }
        if (tile.type === 5) {
            const f = createTile(x, y, `terrain/forests/Forest_${tile.climate}`, 1);
            f.classList.add('forest');
            if (!isVisible) f.classList.add('fog');
        }

        // Structures
        const struct = GAME_STATE.structures[tile.coords.idx];
        if (struct) {
            const file = getStructureFile(struct.type, tile.climate);
            if (file) {
                const e = createTile(x, y, file, 3);
                e.classList.add('structure');
                if (struct.type === 1) e.classList.add('village');
                if (struct.type === 2) e.classList.add('ruins');
                if (!isVisible) e.classList.add('fog');
            }
        }

        // Resources
        const res = GAME_STATE.resources[tile.coords.idx];
        if (res) {
            const file = getResourceFile(res.type, tile.climate);
            if (file) {
                const e = createTile(x, y, file, 3);
                e.classList.add('resource');
                const resClass = { 1: 'animal', 2: 'crop', 3: 'fish', 5: 'metal', 6: 'fruit' }[res.type];
                if (resClass) e.classList.add(resClass);
                if (!isVisible) e.classList.add('fog');
            }
        }

        // Cities
        const city = citiesByIndex[tile.coords.idx];
        if (city) {
            const tribeName = TRIBE_ID_2_NAME[city.tribe.type];
            const climateIndex = CLIMATE_IDS.indexOf(tribeName);
            const e = createTile(x, y, `buildings/${tribeName}/Default/Houses/House_${climateIndex}_5`, 50);
            e.classList.add('city');
            e.innerHTML = `<div><p class="${tile.capitalOf > 0 ? 'capital' : ''}">${city.name || 'City'} (${city.level})</p></div>`;
            if (!isVisible) {
                e.classList.add('fog');
                // Maybe hide name?
            }
        }

        // Units - Only render if visible (or FOW off)
        if (isVisible || !ENABLE_FOW) {
            const unit = unitsByIndex[tile.coords.idx];
            if (unit) {
                const tribeName = TRIBE_ID_2_NAME[unit.tribe.type];
                const className = ClassNameToId[unit.type];
                if (className) {
                    const e = createTile(x, y, `units/${tribeName}/default/${tribeName}_default_${className}`, 500);
                    e.classList.add('unit');
                    if (unit.moved || unit.attacked) e.classList.add('exausted');
                    if (unit.flipped) e.classList.add('flipped');
                    if (selectedUnitIdx === tile.coords.idx) e.classList.add('selected-unit-highlight');
                    e.innerHTML = `<div class="health">${Math.floor(unit.health / 10)}</div>`;
                }
            }
        }

        // Store tile ref for overlays
        TILE_ELEMENTS[tile.coords.idx] = { x, y, ground };
    });

    // Add move overlays for selected unit
    renderMoveOverlays();
}

function renderMoveOverlays() {
    // Remove old overlays
    document.querySelectorAll('.move-overlay').forEach(el => el.remove());

    if (selectedUnitIdx === null) return;

    // Find moves for this unit (src matches selected)
    // Move move: src is starting position
    // Attack move: src is attacker position
    const unitMoves = lastLegalMoves.filter(m => {
        if (!m || typeof m !== 'object') return false;
        return m.src === selectedUnitIdx;
    });

    unitMoves.forEach(move => {
        const targetIdx = move.target;
        if (targetIdx === undefined || targetIdx === null) return;

        const tileRef = TILE_ELEMENTS[targetIdx];
        if (!tileRef) return;

        const overlay = document.createElement('div');
        overlay.classList.add('tile', 'move-overlay');

        const posX = (tileRef.x - tileRef.y) * (tileSize / 2 - someOffset);
        const posY = (tileRef.x + tileRef.y) * (tileSize / 4 + someOffset);
        overlay.style.left = `${posX}px`;
        overlay.style.top = `${posY}px`;
        overlay.style.zIndex = 999;
        overlay.style.pointerEvents = 'none';

        // Use sprite image based on move type
        const img = document.createElement('img');
        img.style.width = '128px'; // Same as tile size
        img.style.height = '64px'; // Half tile size usually for iso

        if (move.moveType === 2) {
            // Attack
            img.src = 'textures/misc/attackTarget.png';
        } else {
            // Move/Step
            img.src = 'textures/misc/moveTarget.png';
        }

        overlay.appendChild(img);

        // Click to execute move
        overlay.style.pointerEvents = 'auto'; // Enable clicks
        overlay.style.cursor = 'pointer';
        overlay.onclick = (e) => {
            e.stopPropagation();
            playMove(move);
        };

        mapContainer.appendChild(overlay);
    });
}

function playMove(move) {
    console.log('Playing move:', move);
    fetch('/step', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(move)
    })
        .then(r => r.json())
        .then(data => {
            // Clear selection after move
            // Unless it was a unit action that allows further actions? 
            // For simplicity, clear it.
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
        12: `buildings/common/Lumber Hut`,
        21: `buildings/common/Mine`
    };
    return map[type];
}

function getResourceFile(type, climate) {
    const map = {
        1: `animals/${CLIMATE_TO_ANIMAL[climate]}`,
        2: `terrain/misc/ResourceGFX_crop`,
        3: `animals/fish`,
        5: `terrain/misc/ResourceGFX_metal`,
        6: `fruits/ResourceGFX_fruit_${climate}`
    };
    return map[type];
}

let scale = 0.5;
let translateX = 0;
let translateY = 0;

const mapViewport = document.getElementById('map-viewport');

function updateTransform() {
    mapContainer.style.transform = `translate(${translateX}px, ${translateY}px) scale(${scale})`;
}

function centerOnMapMiddle() {
    const mapSize = GAME_STATE.settings.size || 16;
    // Center of the map is at approximately (size/2, size/2)
    const centerTileX = mapSize / 2;
    const centerTileY = mapSize / 2;

    const posX = (centerTileX - centerTileY) * (tileSize / 2 - someOffset);
    const posY = (centerTileX + centerTileY) * (tileSize / 4 + someOffset);

    // Get viewport dimensions (the main map area, not full window)
    const viewportRect = mapViewport.getBoundingClientRect();
    const viewportCenterX = viewportRect.width / 2;
    const viewportCenterY = viewportRect.height / 2;

    translateX = viewportCenterX - (posX * scale);
    translateY = viewportCenterY - (posY * scale);
    updateTransform();
}

window.addEventListener('load', () => {
    fetch('/current').then(r => r.json()).then(data => {
        updateUI(data);
        centerOnMapMiddle();
    });

    let dragging = false, lx, ly;
    document.addEventListener('mousedown', e => {
        if (e.target.closest('.glass')) return;
        dragging = true; lx = e.clientX; ly = e.clientY;
    });
    document.addEventListener('mousemove', e => {
        if (!dragging) return;
        translateX += e.clientX - lx;
        translateY += e.clientY - ly;
        lx = e.clientX; ly = e.clientY;
        updateTransform();
    });
    document.addEventListener('mouseup', () => dragging = false);

    // Zoom towards center of viewport
    mapViewport.addEventListener('wheel', e => {
        e.preventDefault();
        const rect = mapViewport.getBoundingClientRect();
        const centerX = rect.width / 2;
        const centerY = rect.height / 2;

        const oldScale = scale;
        const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
        scale = Math.min(2, Math.max(0.2, scale * zoomFactor));

        // Adjust translate to zoom towards center
        translateX = centerX - (centerX - translateX) * (scale / oldScale);
        translateY = centerY - (centerY - translateY) * (scale / oldScale);

        updateTransform();
    }, { passive: false });
});
