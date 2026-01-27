const tileSize = 128;
const someOffset = 4;
const mapContainer = document.getElementById("map");

let GAME_STATE = {};
let TILE_ELEMENTS = {};
let lastLegalMoves = [];

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
    Object.entries(TechnologyNames).forEach(([id, name]) => {
        const badge = document.createElement('div');
        badge.className = 'tech-badge';
        if (unlockedTechs.includes(parseInt(id))) badge.classList.add('unlocked');
        badge.textContent = name;
        techList.appendChild(badge);
    });

    // Update Moves List
    movesList.innerHTML = '';
    lastLegalMoves.slice(0, 50).forEach(move => {
        const li = document.createElement('li');
        li.textContent = move;
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

        // Ground
        // TileState.terrain_type is renamed to "type"
        const tilefile = [null, 'terrain/water/water', 'terrain/water/ocean', null, null, null, 'terrain/tiles/ice'][tile.type]
            || `terrain/tiles/ground_${tile.climate}`;
        const ground = createTile(x, y, tilefile);
        ground.classList.add('ground');

        // Ambient (Mountains/Forests)
        if (tile.type === 4) createTile(x, y, `terrain/mountains/mountain_${tile.climate}`, 3).classList.add('mountain');
        if (tile.type === 5) createTile(x, y, `terrain/forests/Forest_${tile.climate}`, 1).classList.add('forest');

        // Structures
        const struct = GAME_STATE.structures[tile.coords.idx];
        if (struct) {
            const file = getStructureFile(struct.type, tile.climate);
            if (file) {
                const e = createTile(x, y, file, 3);
                e.classList.add('structure');
                if (struct.type === 1) e.classList.add('village');
                if (struct.type === 2) e.classList.add('ruins');
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
        }

        // Units
        const unit = unitsByIndex[tile.coords.idx];
        if (unit) {
            const tribeName = TRIBE_ID_2_NAME[unit.tribe.type];
            const className = ClassNameToId[unit.type];
            if (className) {
                const e = createTile(x, y, `units/${tribeName}/default/${tribeName}_default_${className}`, 500);
                e.classList.add('unit');
                if (unit.moved || unit.attacked) e.classList.add('exausted');
                if (unit.flipped) e.classList.add('flipped');
                e.innerHTML = `<div class="health">${Math.floor(unit.health / 10)}</div>`;
            }
        }
    });
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

let scale = 0.6;
let translateX = window.innerWidth / 2;
let translateY = window.innerHeight / 3;

function updateTransform() {
    mapContainer.style.transform = `translate(${translateX}px, ${translateY}px) scale(${scale})`;
}

function centerOnCapital() {
    const currentTribeId = GAME_STATE.settings.currentPlayerTurnId;
    const currentTribe = GAME_STATE.tribes[currentTribeId.toString()] || GAME_STATE.tribes[currentTribeId];
    if (currentTribe && currentTribe.startingTileCoords) {
        const { x, y } = currentTribe.startingTileCoords;
        const posX = (x - y) * (tileSize / 2 - someOffset);
        const posY = (x + y) * (tileSize / 4 + someOffset);

        translateX = (window.innerWidth / 2) - (posX * scale);
        translateY = (window.innerHeight / 2) - (posY * scale);
        updateTransform();
    }
}

window.addEventListener('load', () => {
    fetch('/current').then(r => r.json()).then(data => {
        updateUI(data);
        centerOnCapital();
    });
    updateTransform();

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
    document.addEventListener('wheel', e => {
        scale = Math.min(2, Math.max(0.1, scale - e.deltaY * 0.001));
        updateTransform();
    });
});
