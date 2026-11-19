const tileSize = 128; // @ts-check
const TILE_OFFSET = 4;
const mapContainer = document.getElementById("map");

// Initial State Preview with Modal
window.previewInitialState = async function (id) {
    if (id) {
        // Direct load (from confirm or arg)
        await loadInitialState(id);
        return;
    }

    // Show modal and fetch list
    const modal = document.getElementById('initial-state-modal');
    modal.classList.remove('hidden');

    const select = document.getElementById('initial-state-select');
    select.innerHTML = '<option>Loading...</option>';

    try {
        const res = await fetch('/replay/list_initial');
        const data = await res.json();

        if (data.status === 'success') {
            select.innerHTML = '<option value="">Select a Game ID...</option>';
            data.files.forEach(f => {
                const opt = document.createElement('option');
                opt.value = f;
                opt.textContent = f;
                select.appendChild(opt);
            });
        } else {
            select.innerHTML = '<option>Error loading list</option>';
        }
    } catch (e) {
        console.error(e);
        select.innerHTML = '<option>Network Error</option>';
    }
};

window.closeInitialStateModal = function () {
    document.getElementById('initial-state-modal').classList.add('hidden');
};

window.confirmInitialStateSelection = function () {
    const select = document.getElementById('initial-state-select');
    const id = select.value;
    if (id) {
        closeInitialStateModal();
        loadInitialState(id);
    } else {
        showToast("⚠️ Please select a game ID");
    }
};

async function loadInitialState(id) {
    showToast("⏳ Loading initial state...");
    try {
        const res = await fetch('/replay/load_initial', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ id })
        });
        const data = await res.json();

        if (data.status === 'error') {
            showToast("❌ " + data.message);
            console.error(data.message);
        } else {
            try {
                updateUI(data);
                showToast("✅ Loaded Initial State!");
            } catch (e) {
                console.error("UI Update failed:", e);
                showToast("❌ UI Update Error");
            }
            // Force camera reset
            setTimeout(() => {
                centerOnCoordinates(6, 6, true);
            }, 200);
        }
    } catch (e) {
        showToast("❌ Network Error");
        console.error(e);
    }
}



var GAME_STATE = {};
var currentLegalMoves = [];
let selectedUnitIdx = null; // Currently selected unit's tile index
var ENABLE_FOW = true; // Fog of War toggle
const SHOW_PREDICTIONS = false;

// UI Elements
const turnVal = document.getElementById('turn-val');
const turnTotalVal = document.getElementById('turn-total');
const tribeNameLabel = document.getElementById('tribe-name');
const starsVal = document.getElementById('stars-val');
const scoreVal = document.getElementById('score-val');
const incomeVal = document.getElementById('income-val');
const aiTurnIndicator = document.getElementById('ai-turn-indicator');
const toastContainer = document.getElementById('toast-container');
const tileActionPopup = document.getElementById('tile-action-popup');

function showToast(message) {
    if (!toastContainer) return;
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    toastContainer.appendChild(toast);

    // Auto-remove after animation
    setTimeout(() => {
        toast.remove();
    }, 3000);
}

// --- Move Interaction ---

const renderer = new MapRenderer(mapContainer);
const hoverEl = document.getElementById('hovertile');

function updateUI(data) {
    const prevSettings = GAME_STATE.settings || null;
    const prevSize = prevSettings ? prevSettings.size : null;
    const prevTileCount = prevSettings ? prevSettings.tile_count : null;
    const oldTribeId = prevSettings ? prevSettings.currentPlayerTurnId : null;
    if (data.state) {
        // If the map itself changed (new game / load / different size), purge
        // stale tile DOM first. Otherwise leftover tiles from the previous
        // map remain in the container and render outside the new square.
        const newSettings = data.state.settings;
        const newSize = newSettings ? newSettings.size : null;
        const newTileCount = newSettings ? newSettings.tile_count : null;
        if (prevSize !== null && (newSize !== prevSize || newTileCount !== prevTileCount)) {
            renderer.clear();
        }
        GAME_STATE = data.state;
    }
    if (data.legalMoves) currentLegalMoves = data.legalMoves;
    const currentTribeId = GAME_STATE.settings.currentPlayerTurnId;
    const currentTribe = GAME_STATE.tribes[currentTribeId.toString()] || GAME_STATE.tribes[currentTribeId];

    if (!currentTribe) return;

    // Update Stats
    turnVal.textContent = GAME_STATE.settings.turn;
    turnTotalVal.textContent = GAME_STATE.settings.maxTurns;
    const currentTribeName = TRIBE_ID_2_NAME[currentTribe.type];
    tribeNameLabel.textContent = currentTribeName || 'Unknown';
    starsVal.textContent = currentTribe.stars;

    scoreVal.textContent = currentTribe.score;

    const income = currentTribe.cities.reduce((acc, cur) => {
        return acc + getCityProduction(GAME_STATE, cur);
    }, 0);
    incomeVal.textContent = `+${income}`;

    // Perspective rendering:
    // We use the current player's POV for rendering to ensure vision/FOW updates immediately.
    const renderPov = currentTribeId;
    renderer.render(GAME_STATE, currentLegalMoves, renderPov);
    updateSelectionInfo();

    checkRewardPopup();
    if (oldTribeId !== null && oldTribeId !== currentTribeId) {
        setTimeout(() => focusCamera(true), 100);
    }
    updateTurnIndicator(currentTribe);

    // Show messages as toasts
    if (GAME_STATE._messages && GAME_STATE._messages.length > 0) {
        GAME_STATE._messages.forEach(msg => showToast(msg));
        // Clear them so they don't reappear on every UI sync if we don't have a fresh state
        GAME_STATE._messages = [];
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
        if (!name) console.log(techId);

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

        const moveType = move.moveType !== undefined ? move.moveType : (move.techType !== undefined ? 7 : (move.structure !== undefined ? 6 : (move.reward !== undefined ? 9 : 0)));

        let text = '';
        const typeName = MoveTypeNames[moveType] || moveType;
        const resource = GAME_STATE.resources[move.target];
        const structure = GAME_STATE.structures[move.target];
        const tile = GAME_STATE.tiles[move.target || move.src];

        if (moveType === 4) {
            const isUpgrade = move.upgrade === true;
            text = `${isUpgrade ? 'Upgrade' : 'Summon'} ${UnitTypes[move.type] || move.type}`;
        }
        else if (moveType === 7) text = `Research ${TechnologyNames[move.type] || move.type}`;
        else if (moveType === 6) text = `🔨 ${StructureTypes[move.type]} @ ${move.src}`;
        else if (moveType === 5) text = `🥝 ${resource ? ResourceTypes[resource.type] : 'Resource'}`;
        else if (moveType === 8) text = `Capture ${tile && tile.owner > 0 ? 'City' : StructureTypes[structure.type]} (${move.src})`;
        else if (moveType === 9) text = `${RewardEmojis[move.type]} ${RewardTypes[move.type]}`;
        else if (moveType === 10) text = 'End Turn';
        else text = `${typeName} (${moveType}) ${move.src ?? move.target ?? ''} → ${move.target ?? ''}`;

        li.textContent = text;
        li.onclick = () => playMove(move);
        li.onmouseenter = () => renderer.highlightMove(move);
        li.onmouseleave = () => renderer.clearMoveHighlight();

        if (moveType === 9 && RewardTypes[move.type] === 'Explorer') {
            const simBtn = document.createElement('span');
            simBtn.textContent = ' 👁️';
            simBtn.title = 'Simulate Explorer Path';
            simBtn.className = 'sim-btn';
            simBtn.style.marginLeft = '10px';
            simBtn.style.fontSize = '1.2em';
            simBtn.onclick = (e) => {
                e.stopPropagation();
                simulateExplorer(move.target);
            };
            li.appendChild(simBtn);
        }

        movesList.appendChild(li);
    });
}

// Camera and Zoom logic (keeping existing as it works well)
let scale = 1.4;
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
    // Calculate the visual center of the tile (128x128 box) in map space.
    // Horizontally it's centered (64), vertically the diamond top-surface center is approx 32px down.
    const tileCenterX = pos.x + 64;
    const tileCenterY = pos.y + 32;

    translateX = (viewportRect.width / 2) - (tileCenterX * scale);
    translateY = (viewportRect.height / 2) - (tileCenterY * scale);

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
    // Camera should pan to CURRENT player's capital
    const currentTribeId = GAME_STATE.settings.currentPlayerTurnId;
    const tribe = GAME_STATE.tribes[currentTribeId.toString()] || GAME_STATE.tribes[currentTribeId] || Object.values(GAME_STATE.tribes)[0];

    if (tribe && tribe.cities && tribe.cities.length > 0) {
        // Find capital if possible
        let cityToFocus = tribe.cities[0];
        const capital = tribe.cities.find(c => {
            const tile = GAME_STATE.tiles[c.idx];
            return tile && tile.capitalOf > 0;
        });
        if (capital) cityToFocus = capital;

        const cityTile = GAME_STATE.tiles[cityToFocus.idx];
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

function updateSelectionInfo(clickX = null, clickY = null) {
    const idx = renderer.selectedIdx;
    const richContainer = document.getElementById('selection-rich-content');
    const floatingPopup = document.getElementById('tile-action-popup');

    if (!richContainer) return;

    // Reset floating popup by default
    if (floatingPopup) {
        floatingPopup.innerHTML = '';
        floatingPopup.classList.add('hidden');
    }

    if (idx === null) {
        richContainer.innerHTML = '<div id="selection-info">Select a unit or tile</div>';
        return;
    }

    const tile = GAME_STATE.tiles[idx];
    const unit = getUnitAt(idx);
    const resource = GAME_STATE.resources[idx];

    // 1. UPDATE BOTTOM CONSOLE (Rich Info)
    const isUnitMode = selectedUnitIdx === idx;
    // Consistently use Current Player's POV for selection info / Tech visibility
    const povId = GAME_STATE.settings.currentPlayerTurnId;
    const povTribe = GAME_STATE.tribes[povId.toString()] || GAME_STATE.tribes[povId];
    const terrainName = TerrainType[tile.type] || `ID=${tile.type}`;
    const rawResourceName = resource ? ResourceTypes[resource.type] : "";
    const resourceVisible = resource && isResourceVisible(resource.type, povTribe);
    const resourceName = resourceVisible ? rawResourceName : "";

    let title, subtitle, thumbFile;

    if (unit && isUnitMode) {
        const tribeName = TRIBE_ID_2_NAME[unit.tribe.type];
        const className = UnitTypes[unit.unitType || unit.type] || `ID=${unit.type}`;
        title = `${tribeName} ${className}`;
        subtitle = `Health: ${unit.health}/${unit.maxHealth}`;
        thumbFile = `units/${tribeName}/default/${tribeName}_default_${className}`;
    } else {
        const city = Object.values(GAME_STATE.tribes).flatMap(t => t.cities || []).find(c => (c.idx) === idx);
        if (city) {
            const tribeType = city.tribe?.type || GAME_STATE.tribes[city.owner]?.type;
            const tribeName = TRIBE_ID_2_NAME[tribeType];
            title = city.name || (tile.capitalOf > 0 ? "Capital" : "City");
            subtitle = `${tribeName} Territory`;
            thumbFile = getCityFile(tribeType, tile.capitalOf > 0);
        } else {
            title = resourceName ? `${terrainName}, ${resourceName}` : terrainName;
            subtitle = resourceName ? "Extract this resource to upgrade your city" : "A sturdy foundation for your empire";
            thumbFile = resourceVisible ? getResourceFile(resource.type, tile.climate) : getGroundFile(tile.type, tile.climate);
        }
    }

    richContainer.innerHTML = `
        <div class="tile-thumbnail-container">
            <div class="tile-thumbnail-sprite" style="background-image: url('textures/${thumbFile}.png')"></div>
        </div>
        <div class="selection-text-group">
            <div class="selection-title">${title}</div>
            <div class="selection-subtitle">${subtitle}</div>
        </div>
    `;

    // 2. UPDATE FLOATING ACTIONS (Polytopia Style)
    // Only show city actions if we are NOT in unit selection mode for this tile
    const harvestMove = !isUnitMode ? currentLegalMoves.find(m => m.moveType === 5 && (m.target === idx || m.target_index === idx)) : null;
    const buildMovesCheck = !isUnitMode ? currentLegalMoves.filter(m => m.moveType === 6 && (m.target === idx || m.target_index === idx)) : [];
    const summonMoves = !isUnitMode ? currentLegalMoves.filter(m => m.moveType === 4 && (m.src === idx || m.src_index === idx)) : [];
    const abilityMoves = currentLegalMoves.filter(m =>
        m.moveType === 3 && (m.target === idx || m.target_index === idx || m.src === idx || m.src_index === idx)
    );

    if ((harvestMove || buildMovesCheck.length > 0 || summonMoves.length > 0 || abilityMoves.length > 0) && floatingPopup && clickX !== null && clickY !== null) {
        floatingPopup.classList.remove('hidden');
        floatingPopup.style.left = `${clickX}px`;
        floatingPopup.style.top = `${clickY - 20}px`; // Slightly above the click point

        let actionsHtml = '';

        const currentStars = GAME_STATE.tribes[GAME_STATE.settings.currentPlayerTurnId].stars;

        // Harvest
        if (harvestMove) {
            const cost = 2; // Default harvest cost
            const resEmoji = ResourceEmojis[resource.type] || "🥝";
            const resName = ResourceTypes[resource.type] || `ID=${resource.type}`;
            const isDisabled = currentStars < cost;

            actionsHtml += `
                <div class="poly-action-container">
                    <button class="poly-action-btn ${isDisabled ? 'disabled' : ''}" data-move-id="${currentLegalMoves.indexOf(harvestMove)}">
                        <span class="icon">${resEmoji}</span>
                        <div class="cost-badge">⭐ ${cost}</div>
                    </button>
                    <div class="poly-action-label">Harvest ${resName}</div>
                </div>
            `;
        }

        // Build (Metal Mine, Farm, etc.)
        const buildMoves = currentLegalMoves.filter(m => m.moveType === 6 && (m.target === idx || m.target_index === idx));

        buildMoves.forEach(bm => {
            const structType = bm.type;
            const emoji = StructureEmojis[structType] || "🏗️";
            const name = StructureTypes[structType] || `ID=${structType}`;
            const cost = StructureCosts[structType] || 5;
            const isDisabled = currentStars < cost;

            actionsHtml += `
                <div class="poly-action-container">
                    <button class="poly-action-btn ${isDisabled ? 'disabled' : ''}" data-move-id="${currentLegalMoves.indexOf(bm)}">
                        <span class="icon">${emoji}</span>
                        <div class="cost-badge">⭐ ${cost}</div>
                    </button>
                    <div class="poly-action-label">Build ${name}</div>
                </div>
            `;
        });

        const UnitCosts = {
            2: 2, // Warrior
            3: 3, // Rider
            4: 8, // Knight
            5: 3, // Defender
            8: 8, // Catapult
            9: 3, // Archer
            10: 5, // MindBender
            11: 5, // Swordsman
            12: 10, // Giant (Super)
            28: 3, // Hexapod
            31: 3, // Kiton
            32: 8, // Exida
        };
        // Summons
        summonMoves.forEach(m => {
            const unitType = m.type || m.unitType;
            const emoji = UnitEmojis[unitType] || "🪖";
            const name = UnitTypes[unitType] || "Unknown";
            const cost = UnitCosts[unitType] || 3;
            const isDisabled = currentStars < cost;

            actionsHtml += `
                <div class="poly-action-container">
                    <button class="poly-action-btn ${isDisabled ? 'disabled' : ''}" data-move-id="${currentLegalMoves.indexOf(m)}">
                        <span class="icon">${emoji}</span>
                        <div class="cost-badge">⭐ ${cost}</div>
                    </button>
                    <div class="poly-action-label">Train ${name}</div>
                </div>
            `;
        });

        // Abilities (Clear Forest, Grow Forest, Recover, etc.)
        abilityMoves.forEach(am => {
            const abilityType = (am.type && typeof am.type === 'object') ? am.type.Ok : am.type;
            const emoji = AbilityEmojis[abilityType] || "🪄";
            const name = AbilityNames[abilityType] || `ID=${abilityType}`;
            const costValue = AbilityCosts[abilityType] || 0;
            const isDisabled = costValue > 0 && currentStars < costValue;

            let costHtml = `<div class="cost-badge">⭐ ${costValue}</div>`;
            if (costValue < 0) {
                costHtml = `<div class="cost-badge reward-badge">⭐ +${Math.abs(costValue)}</div>`;
            } else if (costValue === 0) {
                costHtml = ''; // No badge for free actions
            }

            actionsHtml += `
                <div class="poly-action-container">
                    <button class="poly-action-btn ${isDisabled ? 'disabled' : ''}" data-move-id="${currentLegalMoves.indexOf(am)}">
                        <span class="icon">${emoji}</span>
                        ${costHtml}
                    </button>
                    <div class="poly-action-label">${name}</div>
                </div>
            `;
        });

        floatingPopup.innerHTML = `<div style="display: flex; gap: 15px; flex-wrap: wrap; justify-content: center;">${actionsHtml}</div>`;

        // Bind clicks
        floatingPopup.querySelectorAll('.poly-action-btn').forEach(btn => {
            btn.onclick = (e) => {
                e.stopPropagation();
                if (btn.classList.contains('disabled')) return;
                const moveIdx = parseInt(btn.getAttribute('data-move-id'));
                const move = currentLegalMoves[moveIdx];
                if (move) {
                    playMove(move);
                    floatingPopup.classList.add('hidden');
                }
            };
        });
    }
}

function updateTechModal(tribe, modalBody, overlayEl) {
    modalBody.innerHTML = `
        <div class="tech-starfield"></div>
        <div class="tech-tree-view">
            <svg class="tech-edges-svg" id="tech-edges"></svg>
            <div id="tech-nodes-container"></div>
        </div>
    `;

    const nodesContainer = document.getElementById('tech-nodes-container');
    const edgesSvg = document.getElementById('tech-edges');
    const unlockedTechs = (tribe.tech_vanilla || []).map(t => t.type);

    // Central Hub
    const hub = document.createElement('div');
    hub.className = 'tech-node tech-hub';
    const tribeName = TRIBE_ID_2_NAME[tribe.type];
    const hubIcon = `textures/units/${tribeName}/default/${tribeName}_default_Warrior.png`;
    hub.style.left = `calc(50% - 55px)`;
    hub.style.top = `calc(50% - 55px)`;
    hub.title = tribeName;
    hub.innerHTML = `<img src="${hubIcon}" alt="">`;
    nodesContainer.appendChild(hub);

    const centerX = 0;
    const centerY = 0;

    // Helper to get coordinates
    const getCoords = (techId) => {
        const node = TechNodes[techId];
        return node ? { x: node.x, y: node.y } : { x: 0, y: 0 };
    };

    // Render Nodes and Edges
    Object.keys(TechNodes).forEach(id => {
        const techId = parseInt(id);
        const meta = TechNodes[techId];
        const { x, y } = getCoords(techId);

        // Find parent for edge
        let parentId = 0;
        for (const [pId, children] of Object.entries(TechTree)) {
            if (children.includes(techId)) {
                parentId = parseInt(pId);
                break;
            }
        }

        const pCoords = parentId === 0 ? { x: 0, y: 0 } : getCoords(parentId);

        // Draw edge
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        // We need the coordinates relative to the SVG center.
        // Our nodes are positioned via transform: translate(x, y) relative to center.
        // SVG coordinates are absolute. We'll set viewBox to -600 -500 1200 1000
        line.setAttribute('x1', pCoords.x);
        line.setAttribute('y1', pCoords.y);
        line.setAttribute('x2', x);
        line.setAttribute('y2', y);
        line.setAttribute('class', 'tech-line' + (unlockedTechs.includes(techId) ? ' active' : ''));
        edgesSvg.appendChild(line);

        // Draw node
        const node = document.createElement('div');
        const isUnlocked = unlockedTechs.includes(techId);
        const canResearch = currentLegalMoves.some(m => m && (m.moveType === 7 || m.tech !== undefined) && m.type === techId);

        node.className = `tech-node ${isUnlocked ? 'researched' : canResearch ? 'available' : 'locked'}`;
        node.style.left = `calc(50% + ${x}px - 45px)`;
        node.style.top = `calc(50% + ${y}px - 45px)`;

        // Accurate Tech Cost Calculation
        const numCities = (tribe.cities || []).length;
        // Determine tier via BFS or hardcode?
        // For visual simplicity, let's assume standard costs for now or fetch from backend if available.
        // Actually, we can get tier from the tree depth.
        // Or just generic:
        const tier = 1; // Simplify for now as we don't have tier data in TechNodes yet, or we can look it up.
        // Better:
        let depth = 1;
        // ... (We can improve tier calc later, or add it to TechNodes)

        const hasPhilosophy = unlockedTechs.includes(22);
        // Fallback cost logic
        const baseCost = 4 + (numCities * depth);
        const cost = hasPhilosophy ? Math.ceil(baseCost * 0.77) : baseCost;

        node.innerHTML = `
            <img class="tech-icon" src="textures/${meta.icon}.png" alt="">
            <div class="label">${TechnologyNames[techId]}</div>
            ${canResearch ? `<div class="cost"><img src="textures/misc/star.png" alt="">${cost}</div>` : ''}
        `;

        if (canResearch) {
            node.onclick = () => {
                const move = currentLegalMoves.find(m => m && (m.moveType === 7 || m.tech !== undefined) && m.type === techId);
                if (move) {
                    playMove(move);
                    overlayEl.remove();
                }
            };
        }
        nodesContainer.appendChild(node);
    });

    edgesSvg.setAttribute('viewBox', `-600 -450 1200 900`);

    // --- ZOOM & PAN LOGIC ---
    let zoom = 0.9;
    let panX = 0;
    let panY = 0;
    let isPanning = false;
    let startX, startY;

    const view = document.querySelector('.tech-tree-view');
    const updateTransform = () => {
        view.style.transform = `translate(${panX}px, ${panY}px) scale(${zoom})`;
    };
    updateTransform();

    // Zoom (Wheel)
    view.addEventListener('wheel', (e) => {
        e.preventDefault();
        const zoomSpeed = 0.1;
        const delta = e.deltaY > 0 ? 1 - zoomSpeed : 1 + zoomSpeed;
        zoom *= delta;
        zoom = Math.min(Math.max(zoom, 0.4), 2); // Constraints
        updateTransform();
    }, { passive: false });

    // Pan (Drag)
    view.addEventListener('mousedown', (e) => {
        // Only pan if background or lines clicked (optional: any drag starts pan)
        isPanning = true;
        startX = e.clientX - panX;
        startY = e.clientY - panY;
    });

    overlayEl.addEventListener('mousemove', (e) => {
        if (!isPanning) return;
        panX = e.clientX - startX;
        panY = e.clientY - startY;
        updateTransform();
    });

    window.addEventListener('mouseup', () => {
        isPanning = false;
    }, { once: false });
}

function checkRewardPopup() {
    try {
        // Log move type distribution for debugging
        const moveTypes = currentLegalMoves.map(m => m.moveType);
        const counts = moveTypes.reduce((acc, t) => { acc[t] = (acc[t] || 0) + 1; return acc; }, {});
        if (counts[9]) console.log("🎁 Reward moves detected in legal moves:", counts[9]);

        // Collect all unique reward moves (type 9)
        const rewardMoves = currentLegalMoves.filter(m => m && m.moveType === 9);

        // Polytopia usually forces selection if there are multiple rewards (e.g. Workshop vs Explorer)
        if (rewardMoves.length >= 1) {
            // Prevent multiple popups if one is already open
            if (document.getElementById('level-up-reward-overlay')) return;

            const overlay = document.createElement('div');
            overlay.id = 'level-up-reward-overlay';
            overlay.className = 'reward-overlay';

            let cityTitle = "City level up!";
            // Try to find city name from the first reward move's target index
            if (rewardMoves[0].target !== undefined) {
                const city = Object.values(GAME_STATE.tribes)
                    .flatMap(t => t.cities)
                    .find(c => (c.idx) === rewardMoves[0].target);
                if (city) cityTitle = `${city.name} level up!`;
            }

            overlay.innerHTML = `
                <div class="reward-modal">
                    <h2>${cityTitle}</h2>
                    <div class="reward-options"></div>
                </div>
            `;
            document.body.appendChild(overlay);

            const optionsContainer = overlay.querySelector('.reward-options');
            rewardMoves.forEach(move => {
                const btn = document.createElement('button');
                btn.className = 'reward-btn';
                const name = RewardTypes[move.type];
                const emoji = RewardEmojis[move.type];

                btn.innerHTML = `
                    <div class="reward-icon-container">
                        <span class="icon">${emoji || '❓'}</span>
                    </div>
                    <div class="label">${name || 'Unknown'}</div>
                `;

                btn.onclick = () => {
                    playMove(move);
                    overlay.remove();
                };
                optionsContainer.appendChild(btn);
            });
        }
    } catch (e) {
        console.error("Error in checkRewardPopup:", e);
    }
}

function isResourceVisible(resourceType, tribe) {
    if (!resourceType || resourceType === 0) return true;
    if (!tribe || !tribe.tech_vanilla) return true;

    const hasTech = (techId) => tribe.tech_vanilla.some(t => t.type === techId && t.discovered);

    switch (resourceType) {
        case 6: // Fruit
        case 7: // Spores
            return true;
        case 8: // Starfish
            return hasTech(14); // Navigation (14)
        case 2: // Crop
            return hasTech(6) || hasTech(8); // Organization (6) or Farming (8)
        case 5: // Metal
            return hasTech(20); // Climbing (20)
        case 1: // Game (Animal)
            return hasTech(15); // Hunting (15)
        case 3: // Fish
            return hasTech(10); // Fishing (10)
        case 9: // AquaCrop
            return hasTech(25); // FreeDiving (25)
        default:
            return true;
    }
}

function getUnitAt(idx) {
    if (!GAME_STATE.tribes) return null;
    for (const tribe of Object.values(GAME_STATE.tribes)) {
        const unit = (tribe.units || []).find(u => u.coords.idx === idx);
        if (unit) return { ...unit, tribe };
    }
    return null;
}

function getStructureFile(type, climate) {
    const tribeName = TRIBE_ID_2_NAME[climate];
    const textureIndex = terrainTextureIndex(climate);
    const map = {
        1: `buildings/common/Tribe`,
        2: `terrain/misc/ResourceGFX_ruin`,
        3: `misc/Road`,
        5: `buildings/common/Farm`,
        6: 'buildings/common/Windmill_0',
        8: `buildings/common/Port`,
        12: `buildings/common/Lumber Hut`,
        13: 'buildings/common/Sawmill_1',
        21: `buildings/common/Mine`,
        22: 'buildings/common/Forge_1',
        27: `buildings/${tribeName}/Default/Monuments/Monument5_${textureIndex}`,
        28: `buildings/${tribeName}/Default/Monuments/Monument6_${textureIndex}`,
        29: `buildings/${tribeName}/Default/Monuments/Monument7_${textureIndex}`,
        37: `buildings/Cymanti/Default/Unique/spores_4`,
        47: 'misc/lighthouse',
    };
    !map[type] && console.log(`MISSING STRUCTURE: ${type}`);

    return map[type] || 'misc/missing';
}

function getResourceFile(type, climate) {
    const textureIndex = terrainTextureIndex(climate);
    const map = {
        1: `animals/${TEXTURE_INDEX_TO_ANIMAL[textureIndex]}`,
        2: `terrain/misc/ResourceGFX_crop`,
        3: `animals/fish`,
        5: `terrain/misc/ResourceGFX_metal`,
        6: `fruits/ResourceGFX_fruit_${textureIndex}`,
        // 7: `buildings/Cymanti/Default/Unique/spores_4`, // requires different texture
        8: `terrain/misc/ResourceGFX_starfish`
    };
    !map[type] && console.log(`MISSING RESOURCE: ${type}`);
    return map[type] || 'misc/missing';
}

function updateTurnIndicator(tribe) {
    document.getElementById('btn-end-turn').disabled =
        !currentLegalMoves.some(move => move?.moveType === 10);
    const name = tribe.username || TRIBE_ID_2_NAME[tribe.type];
    aiTurnIndicator.textContent = name ? `${name}'s turn` : 'Your turn';
    aiTurnIndicator.classList.remove('hidden');
    aiTurnIndicator.classList.add('your-turn');
}

document.getElementById('btn-fow').onclick = event => {
    if (!GAME_STATE.settings) return;
    ENABLE_FOW = !ENABLE_FOW;
    event.currentTarget.setAttribute('aria-pressed', ENABLE_FOW);
    renderer.render(GAME_STATE, currentLegalMoves, GAME_STATE.settings.currentPlayerTurnId);
};
document.getElementById('btn-end-turn').onclick = () => {
    const endTurnMove = currentLegalMoves.find(m => m && m.moveType === 10);
    if (endTurnMove) playMove(endTurnMove);
    else showToast("No End Turn move available yet.");
};

document.getElementById('btn-tech').onclick = () => {
    if (!GAME_STATE.settings) return;
    const techOverlay = document.createElement('div');
    techOverlay.className = 'tech-overlay-modal';
    techOverlay.innerHTML = `
        <div class="tech-modal-content">
            <button class="close-btn" style="position: absolute; top: 20px; right: 30px; z-index: 100;">×</button>
            <div id="tech-modal-root" style="width: 100%; height: 100%;"></div>
        </div>
    `;
    document.body.appendChild(techOverlay);

    const closeBtn = techOverlay.querySelector('.close-btn');
    closeBtn.onclick = () => techOverlay.remove();
    techOverlay.onclick = (e) => { if (e.target === techOverlay) techOverlay.remove(); };

    const tribe = GAME_STATE.tribes[GAME_STATE.settings.currentPlayerTurnId.toString()];
    const modalRoot = techOverlay.querySelector('#tech-modal-root');

    updateTechModal(tribe, modalRoot, techOverlay);
};

const gameFiles = document.getElementById('game-files');
gameFiles.addEventListener('click', event => {
    if (event.target.closest('button')) gameFiles.open = false;
});
window.addEventListener('click', event => {
    if (!gameFiles.contains(event.target)) gameFiles.open = false;
    tileActionPopup.classList.add('hidden');
});
gameFiles.addEventListener('keydown', event => {
    if (event.key === 'Escape') {
        gameFiles.open = false;
        gameFiles.querySelector('summary').focus();
    }
});

document.getElementById('btn-reset').onclick = async () => {
    try {
        const response = await fetch('/reset', { method: 'POST' });
        if (!response.ok) throw new Error(await response.text());
        renderer.clear();
        updateUI(await response.json());
        focusCamera();
    } catch (error) {
        showToast(error.message);
    }
};
document.getElementById('btn-save-state').onclick = async () => {
    try {
        const res = await fetch('/save', { method: 'POST' });
        const data = await res.json();
        if (data.status === 'success') {
            showToast("💾 " + data.message);
        }
    } catch (e) {
        showToast("❌ Save failed");
    }
};

document.getElementById('btn-load-state').onclick = async () => {
    try {
        const res = await fetch('/load', { method: 'POST' });
        const data = await res.json();
        if (data.state) {
            updateUI(data);
            showToast("📂 Game State Loaded!");
        }
    } catch (e) {
        showToast("❌ Load failed");
    }
};
