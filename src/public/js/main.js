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
        if (recordedMovesList) recordedMovesList.innerHTML = '';
        recordedMoves = [];
        REPLAY_HISTORY = [];
        REPLAY_INDEX = -1;

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
var SHOW_PREDICTIONS = false; // AI Predictions overlay toggle
var TRAIN_MODE_ACTIVE = false; // Train Mode Toggle
var INTERACTIVE_MODE = false; // AI Assistant Toggle
var PENDING_AI_MOVE = null;
const teacherHud = document.getElementById('teacher-hud');

const MOVE_TYPE_ICONS = {
    'Step': '👟',
    'Attack': '⚔️',
    'Research': '🧪',
    'Summon': '🧚',
    'Build': '🏗️',
    'Harvest': '🍎',
    'Capture': '🚩',
    'EndTurn': '⌛',
    'None': '🎓'
};

const TECH_ICONS = {
    0: '❓', 1: '🏇', 2: '🍃', 3: '🛡️', 4: '🛤️', 5: '⚖️',
    6: '🍎', 7: '📜', 8: '🚜', 9: '🏗️', 10: '🎣',
    12: '⚓', 13: '⛵', 14: '🔭', 15: '🏹', 16: '🌲',
    17: '📐', 18: '🎯', 19: '✨', 20: '🧗', 21: '🧘',
    22: '🧠', 23: '⛏️', 24: '🔨'
};

// Event Listener for AI VIS Toggle
document.getElementById('btn-predictions').addEventListener('click', () => {
    SHOW_PREDICTIONS = !SHOW_PREDICTIONS;
    const btn = document.getElementById('btn-predictions');
    btn.classList.toggle('active', SHOW_PREDICTIONS);

    if (SHOW_PREDICTIONS) {
        showToast("🔮 AI Vision: Enabled");
        // Force refresh of overlays
        if (GAME_STATE._prediction) renderer.renderFOWPredictions(GAME_STATE._prediction);
    } else {
        showToast("🔮 AI Vision: Disabled");
        renderer.renderFOWPredictions(null); // Clear
    }
});

// UI Elements
const turnVal = document.getElementById('turn-val');
const turnTotalVal = document.getElementById('turn-total');
const tribeNameLabel = document.getElementById('tribe-name');
const starsVal = document.getElementById('stars-val');
const scoreVal = document.getElementById('score-val');
const incomeVal = document.getElementById('income-val');
const recordedMovesList = document.getElementById('recorded-moves-list');
let recordedMoves = [];
const trainingStatusHUD = document.getElementById('training-status');
const aiTurnIndicator = document.getElementById('ai-turn-indicator');
const lastMoveVal = document.getElementById('last-move-val');
const mctsDepth = document.getElementById('mcts-depth');
const mctsDepthVal = document.getElementById('mcts-depth-val');
const toastContainer = document.getElementById('toast-container');
const selectionInfo = document.getElementById('selection-info');
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

// --- Evaluation Bar ---
function updateEvalBar(evaluation) {
    if (!evaluation) return;
    const fillP1 = document.getElementById('eval-fill-p1');
    const fillP2 = document.getElementById('eval-fill-p2');
    const indicator = document.getElementById('eval-indicator');
    if (!fillP1 || !fillP2 || !indicator) return;

    const adv = evaluation.advantage || 0; // -1 to 1
    // Map advantage to fill ratio: adv=1 means P1 fills 100%, adv=-1 means P2 fills 100%
    const p1Pct = ((1 + adv) / 2) * 100; // 0-100
    const p2Pct = 100 - p1Pct;

    fillP1.style.flex = `${p1Pct} 0 0`;
    fillP2.style.flex = `${p2Pct} 0 0`;

    // Update indicator text
    const sign = adv > 0 ? '+' : '';
    indicator.textContent = `${sign}${adv.toFixed(2)}`;

    // Style the indicator based on who's winning
    indicator.classList.remove('p1-winning', 'p2-winning', 'even');
    if (adv > 0.05) {
        indicator.classList.add('p1-winning');
    } else if (adv < -0.05) {
        indicator.classList.add('p2-winning');
    } else {
        indicator.classList.add('even');
    }
}

// --- MCTS Analysis Panel ---
let mctsParams = {
    passive: false,
};

async function runPassiveAnalysis(force = false) {
    if (!mctsParams.passive && !force) return;
    if (pendingMctsMove || PENDING_AI_MOVE) {
        if (!force) setTimeout(runPassiveAnalysis, 1000);
        return;
    }

    try {
        const res = await fetch('/autostep', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ iterations: Number(mctsDepth.value), dry_run: true })
        });
        const data = await res.json();

        // Update ONLY the analysis panel and heatmap to avoid full UI redraw stutter
        if (data.mctsAnalysis) {
            updateMctsPanel(data.mctsAnalysis);
            renderer.renderMCTSHeatmap(data.mctsAnalysis);
        }

        // Update predictions if AI VIS is on
        if (SHOW_PREDICTIONS && data.state && data.state._prediction) {
            renderer.renderFOWPredictions(data.state._prediction);
        }
    } catch (e) {
        console.error("Analysis Error:", e);
    }

    // Schedule next run if still enabled (and not just a forced one-shot)
    if (mctsParams.passive && !force) {
        setTimeout(runPassiveAnalysis, 3000);
    }
}

function updateMctsPanel_disabled(analysis) {
    const panels = document.querySelectorAll('.mcts-panel');
    const moveList = document.getElementById('mcts-moves-list');
    const pvContainer = document.getElementById('mcts-pv');

    if (!analysis || !moveList || !pvContainer) {
        return;
    }

    // 1. Top Moves
    moveList.innerHTML = '';
    const evals = analysis.evaluations || [];
    const maxVisits = Math.max(...evals.map(e => e.visits || 0), 1);

    if (evals.length === 0) {
        moveList.innerHTML = '<div class="mcts-empty">No analysis available</div>';
    } else {
        // Show top 15
        evals.slice(0, 15).forEach((ev, idx) => {
            const row = document.createElement('div');
            row.className = 'mcts-move-row' + (idx === 0 ? ' best' : '');

            const isPositive = ev.win_rate > 50;
            const isNegative = ev.win_rate < -50;
            const winRateClass = isPositive ? 'positive' : (isNegative ? 'negative' : 'neutral');

            // Normalize visits for bar width
            const barWidth = Math.max(0, Math.min(100, (ev.visits / maxVisits) * 100));

            row.innerHTML = `
                <div class="mcts-move-rank">${idx + 1}</div>
                <div class="mcts-move-desc" title="${ev.description || 'Unknown Move'}">${ev.description || 'Move ' + ev.target}</div>
                <div class="mcts-move-bar-wrap">
                    <div class="mcts-move-bar" style="width: ${barWidth}%"></div>
                </div>
                <div class="mcts-move-winrate ${winRateClass}">${ev.win_rate > 0 ? '+' : ''}${ev.win_rate.toFixed(0)}</div>
            `;
            moveList.appendChild(row);
        });
    }

    // 2. Principal Variation
    pvContainer.innerHTML = '';
    const pv = analysis.principal_variation || [];
    if (pv.length === 0) {
        pvContainer.innerHTML = '<div class="mcts-empty">Line not explored</div>';
    } else {
        pv.forEach((moveDesc, i) => {
            if (i > 0) {
                const arrow = document.createElement('div');
                arrow.className = 'pv-arrow';
                arrow.textContent = '▶';
                pvContainer.appendChild(arrow);
            }
            const chip = document.createElement('div');
            chip.className = 'pv-chip';
            chip.textContent = moveDesc;
            pvContainer.appendChild(chip);
        });
    }
}

// --- Move Interaction ---

const renderer = new MapRenderer(mapContainer);
const hoverEl = document.getElementById('hovertile');

let pendingMctsMove = null;

async function apiAction(endpoint, body) {
    try {
        const res = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        const data = await res.json();
        updateUI(data);
        return data;
    } catch (e) {
        console.error("API Error:", e);
    } finally {
        document.querySelectorAll('.rts-btn').forEach(b => b.disabled = false);
    }
}

function updateUI(data) {
    const oldTribeId = (GAME_STATE.settings) ? GAME_STATE.settings.currentPlayerTurnId : null;
    if (data.state) GAME_STATE = data.state;
    if (data.legalMoves) currentLegalMoves = data.legalMoves;
    if (data.movePlayed) lastMoveVal.textContent = data.movePlayed;

    let lastMctsAnalysis = null;

    // Clear pending AI move if a move was played (manual or AI)
    if (data.movePlayed) {
        PENDING_AI_MOVE = null;
        const hud = document.getElementById('teacher-hud');
        if (hud) hud.classList.add('hidden');
        renderer.clearMoveHighlight();
    }

    // Reset MCTS analysis if not provided in this update (e.g. manual move)
    if (data.mctsAnalysis) {
        lastMctsAnalysis = data.mctsAnalysis;
    } else if (data.movePlayed && data.movePlayed !== 'none') {
        lastMctsAnalysis = null;
    }

    const currentTribeId = GAME_STATE.settings.currentPlayerTurnId;
    const currentTribe = GAME_STATE.tribes[currentTribeId.toString()] || GAME_STATE.tribes[currentTribeId];

    if (!currentTribe) return;

    let lastScore = null;

    // Clear predictions if turn changed or if move played (optional, but safer)
    if (oldTribeId !== null && oldTribeId !== currentTribeId) {
        document.querySelectorAll('.village-marker').forEach(el => el.remove());
        document.querySelectorAll('.predicted-terrain').forEach(el => el.remove());

        // Also clear pending MCTS move on turn change
        resetMctsButton();
        lastScore = null;
    }

    // Update Stats
    turnVal.textContent = GAME_STATE.settings.turn;
    turnTotalVal.textContent = GAME_STATE.settings.maxTurns;
    const currentTribeName = TRIBE_ID_2_NAME[currentTribe.type];
    tribeNameLabel.textContent = currentTribeName || 'Unknown';
    starsVal.textContent = currentTribe.stars;

    // Score Toast
    if (lastScore !== null && currentTribe.score !== lastScore) {
        const diff = currentTribe.score - lastScore;
        const sign = diff > 0 ? '+' : '';
        const emoji = diff > 0 ? '📈' : '📉';
        showToast(`${emoji} Score ${sign}${diff} (${currentTribe.score})`);
    }
    lastScore = currentTribe.score;
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

    // Highlight End Turn if it's the only move left
    const endTurnBtn = document.getElementById('btn-end-turn');
    if (endTurnBtn) {
        if (currentLegalMoves.length === 1) {
            endTurnBtn.classList.add('primary');
        } else {
            endTurnBtn.classList.remove('primary');
        }
    }

    // Check for forced rewards
    checkRewardPopup();

    // Render MCTS heatmap if we have analysis data
    if (lastMctsAnalysis) {
        renderer.renderMCTSHeatmap(lastMctsAnalysis);
    }

    // Render FOW predictions (villages, enemy capitals)
    if (GAME_STATE._prediction) {
        renderer.renderFOWPredictions(GAME_STATE._prediction);
    }

    // If turn changed, pan smoothly to next player's capital
    if (oldTribeId !== null && oldTribeId !== currentTribeId) {
        // In Train Mode, do NOT pan to AI. Only pan if it's the human's turn (ID 1) or if Train Mode is OFF.
        if (!TRAIN_MODE_ACTIVE || currentTribeId === 1) {
            setTimeout(() => focusCamera(true), 100);
        }
    }

    // Auto-Play for Train Mode
    checkTrainMode();

    // Show messages as toasts
    if (GAME_STATE._messages && GAME_STATE._messages.length > 0) {
        GAME_STATE._messages.forEach(msg => showToast(msg));
        // Clear them so they don't reappear on every UI sync if we don't have a fresh state
        GAME_STATE._messages = [];
    }

    // Update Evaluation Bar
    if (data.evaluation) {
        updateEvalBar(data.evaluation);
    }

    // Update MCTS Analysis Panel
    if (data.mctsAnalysis) {
        updateMctsPanel(data.mctsAnalysis);
    }

    if (data.recordedMoves) {
        recordedMoves = data.recordedMoves;
        updateRecordedMovesList(recordedMoves);
    }
}

function updateRecordedMovesList(moves) {
    if (!recordedMovesList) return;
    recordedMovesList.innerHTML = '';

    if (!moves || moves.length === 0) {
        recordedMovesList.innerHTML = '<div class="mcts-empty">No moves recorded</div>';
        return;
    }

    moves.forEach((m, i) => {
        const el = document.createElement('div');
        el.className = 'mcts-move-row';

        const rank = document.createElement('span');
        rank.className = 'mcts-move-rank';
        rank.textContent = i + 1;

        const label = document.createElement('span');
        label.className = 'mcts-move-desc';
        label.textContent = m.desc || m.type || 'Move';

        const player = document.createElement('span');
        player.className = 'mcts-move-winrate neutral';
        player.textContent = `P${m.player}`;
        player.style.color = m.player === 1 ? 'var(--p1-color)' : 'var(--p2-color)';

        el.appendChild(rank);
        el.appendChild(label);
        el.appendChild(player);

        el.onclick = () => {
            if (m.move) {
                playMove(m.move);
            }
        };

        recordedMovesList.appendChild(el);
    });
}

// --- MCTS Analysis Panel ---
function updateMctsPanel(analysis) {
    const panels = document.querySelectorAll('.mcts-panel');
    const moveList = document.getElementById('mcts-moves-list');
    const pvContainer = document.getElementById('mcts-pv');

    if (!analysis || !moveList || !pvContainer) {
        return;
    }

    // 1. Top Moves
    moveList.innerHTML = '';
    const evals = analysis.evaluations || [];
    const maxVisits = Math.max(...evals.map(e => e.visits || 0), 1);

    if (evals.length === 0) {
        moveList.innerHTML = '<div class="mcts-empty">No analysis available</div>';
    } else {
        // Show top 15
        evals.slice(0, 15).forEach((ev, idx) => {
            const row = document.createElement('div');
            row.className = 'mcts-move-row' + (idx === 0 ? ' best' : '');

            const isPositive = ev.win_rate > 50;
            const isNegative = ev.win_rate < -50;
            const winRateClass = isPositive ? 'positive' : (isNegative ? 'negative' : 'neutral');

            // Normalize visits for bar width
            const barWidth = Math.max(0, Math.min(100, (ev.visits / maxVisits) * 100));

            row.innerHTML = `
                <div class="mcts-move-rank">${idx + 1}</div>
                <div class="mcts-move-desc" title="${ev.description || 'Unknown Move'}">${ev.description || 'Move ' + ev.target}</div>
                <div class="mcts-move-bar-wrap">
                    <div class="mcts-move-bar" style="width: ${barWidth}%"></div>
                </div>
                <div class="mcts-move-winrate ${winRateClass}">${ev.win_rate > 0 ? '+' : ''}${(ev.win_rate).toFixed(5)}</div>
            `;
            moveList.appendChild(row);
        });
    }

    // 2. Principal Variation
    pvContainer.innerHTML = '';
    const pv = analysis.principal_variation || [];
    if (pv.length === 0) {
        pvContainer.innerHTML = '<div class="mcts-empty">Line not explored</div>';
    } else {
        pv.forEach((moveDesc, i) => {
            if (i > 0) {
                const arrow = document.createElement('div');
                arrow.className = 'pv-arrow';
                arrow.textContent = '▶';
                pvContainer.appendChild(arrow);
            }
            const chip = document.createElement('div');
            chip.className = 'pv-chip';
            chip.textContent = moveDesc;
            pvContainer.appendChild(chip);
        });
    }

    // 3. Decision Tree
    const treeContainer = document.getElementById('mcts-tree');
    if (treeContainer) {
        treeContainer.innerHTML = '';
        if (!analysis.tree) {
            treeContainer.innerHTML = '<div class="mcts-empty">No tree data</div>';
        } else {
            function renderNode(node, depth) {
                const el = document.createElement('div');
                el.className = 'mcts-tree-node';

                const row = document.createElement('div');
                row.className = 'mcts-tree-row';

                const label = document.createElement('span');
                label.className = 'node-label';
                // Clean up description if slightly repetitive
                label.textContent = (node.move_description || 'Root').replace('MoveType::', '');

                const stats = document.createElement('span');
                stats.className = 'node-stats';
                // Assuming value is sum of scores 0..1
                const winRate = node.visits > 0 ? ((node.value / node.visits) * 100).toFixed(1) + '%' : '0.0%';
                stats.textContent = `${node.visits}v ${winRate}`;

                row.appendChild(label);
                row.appendChild(stats);
                el.appendChild(row);

                if (node.children && node.children.length > 0) {
                    const childrenContainer = document.createElement('div');
                    childrenContainer.className = 'mcts-tree-children';
                    node.children.forEach(child => {
                        childrenContainer.appendChild(renderNode(child, depth + 1));
                    });
                    el.appendChild(childrenContainer);
                }
                return el;
            }

            treeContainer.appendChild(renderNode(analysis.tree, 0));
        }
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

        const moveType = move.moveType !== undefined ? move.moveType : (move.type !== undefined ? 7 : (move.type !== undefined ? 6 : (move.type !== undefined ? 3 : (move.type !== undefined ? 9 : 0))));

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

function resetMctsButton() {
    pendingMctsMove = null;
    const btn = document.getElementById('btn-step');
    if (btn) {
        btn.innerHTML = '<span class="icon">🤖</span> MCTS Step';
        btn.classList.remove('pulsing-btn');
    }
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

    // Panel Toggle
    runPassiveAnalysis(true);

    // Passive Analysis Toggle
    const passiveToggle = document.getElementById('mcts-passive-toggle');
    if (passiveToggle) {
        passiveToggle.onchange = (e) => {
            mctsParams.passive = e.target.checked;
            if (mctsParams.passive) {
                runPassiveAnalysis();
                showToast("Passive Analysis Enabled");
            } else {
                showToast("Passive Analysis Disabled");
            }
        };
    }
});

// Event Listeners for buttons
// trainingStatusHUD already declared at line 18
// No more trainingLog element in index.html to avoid UI clutter
// const trainingLog = document.getElementById('training-log');

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
    const terrainName = TerrainType[tile.type] || "Unknown";
    const rawResourceName = resource ? ResourceTypes[resource.type] : "";
    const resourceVisible = resource && isResourceVisible(resource.type, povTribe);
    const resourceName = resourceVisible ? rawResourceName : "";

    let title, subtitle, thumbFile;

    if (unit && isUnitMode) {
        const tribeName = TRIBE_ID_2_NAME[unit.tribe.type];
        const className = UnitTypes[unit.unitType || unit.type] || "Unknown";
        title = `${tribeName} ${className}`;
        subtitle = `Health: ${Math.floor(unit.health / 10)}/${Math.floor(unit.maxHealth / 10)}`;
        thumbFile = `units/${tribeName}/default/${tribeName}_default_${className}`;
    } else {
        const city = Object.values(GAME_STATE.tribes).flatMap(t => t.cities || []).find(c => c.tileIndex === idx);
        if (city) {
            const tribeName = TRIBE_ID_2_NAME[city.tribe?.type || GAME_STATE.tribes[city.owner]?.type];
            title = city.name || (tile.capitalOf > 0 ? "Capital" : "City");
            subtitle = `${tribeName} Territory`;
            const climateIndex = CLIMATE_IDS.indexOf(tribeName);
            thumbFile = `buildings/${tribeName}/Default/Houses/House_${climateIndex}_5`;
        } else {
            title = resourceName ? `${terrainName}, ${resourceName}` : terrainName;
            subtitle = resourceName ? "Extract this resource to upgrade your city" : "A sturdy foundation for your empire";
            thumbFile = `terrain/tiles/ground_${tile.climate}`;
            if (tile.type === 1) thumbFile = 'terrain/water/water';
            if (tile.type === 2) thumbFile = 'terrain/water/ocean';
            if (resourceVisible) thumbFile = getResourceFile(resource.type, tile.climate);
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
            const resName = ResourceTypes[resource.type] || "Unknown";
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
            const name = StructureTypes[structType] || "Unknown";
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
            const name = AbilityNames[abilityType] || "Unknown";
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
    hub.innerHTML = `<img src="${hubIcon}" onerror="this.src='https://api.dicebear.com/7.x/bottts/svg?seed=${tribeName}'">`;
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
            <span class="icon">${meta.icon}</span>
            <div class="label">${TechnologyNames[techId]}</div>
            ${canResearch ? `<div class="cost">⭐ ${cost}</div>` : ''}
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
    // Collect all unique reward moves (type 9)
    const rewardMoves = currentLegalMoves.filter(m => m && m.moveType === 9);

    // Polytopia usually forces selection if there are multiple rewards (e.g. Workshop vs Explorer)
    if (rewardMoves.length > 0) {
        // Debugging Teacher Mode Popup Prevention
        if (INTERACTIVE_MODE) {
            console.log("Teacher Mode Reward Check:", {
                turn: GAME_STATE.settings.currentPlayerTurnId,
                isHuman: GAME_STATE.settings.currentPlayerTurnId === 1
            });
        }

        // In Teacher Mode, if it's AI's turn, do NOT show the popup. Let the AI propose.
        if (INTERACTIVE_MODE && GAME_STATE.settings.currentPlayerTurnId !== 1) {
            console.log("Skipping Reward Popup for AI Teacher Mode");
            return;
        }

        // Prevent multiple popups if one is already open
        if (document.querySelector('.dynamic-reward-overlay')) return;

        const overlay = document.createElement('div');
        overlay.className = 'reward-overlay dynamic-reward-overlay';

        let cityTitle = "City level up!";
        // Try to find city name from the first reward move's target index
        if (rewardMoves[0].target !== undefined) {
            const city = Object.values(GAME_STATE.tribes)
                .flatMap(t => t.cities)
                .find(c => c.tileIndex === rewardMoves[0].target);
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
                    <span class="icon">${emoji}</span>
                </div>
                <div class="label">${name}</div>
            `;

            btn.onclick = () => {
                playMove(move);
                overlay.remove();
            };
            optionsContainer.appendChild(btn);
        });
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

setInterval(pollTrainingStatus, 2000);

function checkTrainMode() {
    if (!GAME_STATE.settings) return;

    const currentTribeId = GAME_STATE.settings.currentPlayerTurnId;

    // 1. Handle Human Turn (Player 1)
    if (currentTribeId === 1) {
        if (INTERACTIVE_MODE) {
            // Request AI hint if we haven't already
            if (!PENDING_AI_MOVE && teacherHud.classList.contains('hidden')) {
                setTimeout(requestAiHint, 200);
            }
        }

        if (aiTurnIndicator) {
            aiTurnIndicator.textContent = `⌛ ${GAME_STATE.tribes[currentTribeId].username || 'Your'}`;
            aiTurnIndicator.classList.remove('hidden');
            aiTurnIndicator.classList.add('your-turn');
        }
    } else {
        // 2. Handle AI Turn (Player > 1)
        if (TRAIN_MODE_ACTIVE) {
            if (aiTurnIndicator) {
                aiTurnIndicator.textContent = "🤖 AI Thinking...";
                aiTurnIndicator.classList.remove('hidden');
                aiTurnIndicator.classList.remove('your-turn');
            }

            // Auto-play the AI turn
            setTimeout(() => {
                if (!TRAIN_MODE_ACTIVE) return;
                apiAction('/autostep', { iterations: parseInt(mctsDepth.value), dry_run: false });
            }, 100);
        } else {
            if (aiTurnIndicator) {
                aiTurnIndicator.textContent = `⌛ ${GAME_STATE.tribes[currentTribeId].username || 'AI'}`;
                aiTurnIndicator.classList.remove('hidden');
                aiTurnIndicator.classList.remove('your-turn');
            }
        }
    }
}

document.getElementById('btn-train-mode').onclick = () => {
    TRAIN_MODE_ACTIVE = !TRAIN_MODE_ACTIVE;
    const btn = document.getElementById('btn-train-mode');

    if (TRAIN_MODE_ACTIVE) {
        btn.classList.add('active');
        // btn.textContent = "Train Mode: ON";
        showToast("Train Mode ACTIVE: Auto-playing enemies");
    } else {
        btn.classList.remove('active');
        // btn.textContent = "Train Mode";
        showToast("Train Mode STOPPED");
    }
};

document.getElementById('btn-interactive').onclick = () => {
    INTERACTIVE_MODE = !INTERACTIVE_MODE;
    const btn = document.getElementById('btn-interactive');
    btn.classList.toggle('active', INTERACTIVE_MODE);

    if (INTERACTIVE_MODE) {
        showToast("🎓 AI Assistant: Enabled");
        checkTrainMode();
    } else {
        showToast("🎓 AI Assistant: Disabled");
        if (teacherHud) teacherHud.classList.add('hidden');
        PENDING_AI_MOVE = null;
        renderer.clearMoveHighlight();
    }
};

document.getElementById('btn-reset').onclick = () => apiAction('/reset', {});
document.getElementById('btn-fow').onclick = () => {
    ENABLE_FOW = !ENABLE_FOW;
    renderer.render(GAME_STATE, currentLegalMoves, 1);
};
document.getElementById('btn-train').onclick = () => {
    if (confirm("Start a background training session? This will run 'cargo run --bin self_play' on the server.")) {
        apiAction('/train', {}).then(data => {
            if (data && data.message) showToast(data.message);
        });
    }
};

document.getElementById('btn-end-turn').onclick = () => {
    const endTurnMove = currentLegalMoves.find(m => m && m.moveType === 10);
    if (endTurnMove) playMove(endTurnMove);
    else showToast("No End Turn move available yet.");
};

document.getElementById('btn-tech').onclick = () => {
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

document.getElementById('btn-save').onclick = () => {
    if (confirm("Save captured human gameplay data to disk?")) {
        apiAction('/save_training_data', {}).then(data => {
            if (data && data.message) alert(data.message);
        });
    }
};
document.getElementById('btn-predictions').onclick = async () => {
    SHOW_PREDICTIONS = !SHOW_PREDICTIONS;
    const btn = document.getElementById('btn-predictions');
    btn.classList.toggle('prediction-active', SHOW_PREDICTIONS);

    if (SHOW_PREDICTIONS) {
        showToast("AI Visibility Enabled: Requesting Analysis...");
        runPassiveAnalysis(true); // Trigger immediate analysis for the heatmap/predictions
    } else {
        // Clear all prediction overlays if disabled
        document.querySelectorAll('.mcts-overlay').forEach(el => el.remove());
        document.querySelectorAll('.village-marker').forEach(el => el.remove());
        document.querySelectorAll('.analysis-overlay').forEach(el => el.remove());
        document.querySelectorAll('.predicted-terrain').forEach(el => el.remove());
    }

    // Re-render move overlays to show/hide combat previews
    renderer.renderMoveOverlays(currentLegalMoves);
};
document.getElementById('btn-rng').onclick = () => apiAction('/rngstep', {});
document.getElementById('btn-step').onclick = async () => {
    const btn = document.getElementById('btn-step');

    if (pendingMctsMove) {
        // Play the pending move
        playMove(pendingMctsMove);
    } else {
        // Run MCTS (Dry Run)
        const data = await apiAction('/autostep', { iterations: parseInt(mctsDepth.value), dry_run: true });

        if (data && data.bestMove) {
            pendingMctsMove = data.bestMove;
            btn.innerHTML = '<span class="icon">▶️</span> Play Best';
            btn.classList.add('pulsing-btn');

            // Highlight the move
            renderer.highlightMove(data.bestMove);
            console.log(data);
            showToast(`MCTS picked: ${data.movePlayed || 'Move'}`);
        }
    }
};
mctsDepth.oninput = (e) => mctsDepthVal.textContent = e.target.value;

// Clear the tile action popup if clicking anywhere else
window.addEventListener('click', () => {
    if (tileActionPopup) tileActionPopup.classList.add('hidden');
});

// AI Assistant / Interactive logic


// Assistant functions

// Teacher/Assistant logic
async function requestAiHint() {
    if (!INTERACTIVE_MODE || PENDING_AI_MOVE) return;

    const depth = parseInt(mctsDepth.value) || 100;

    // Call backend
    const res = await fetch('/trainer/hint', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ iterations: depth })
    });
    const data = await res.json();

    if (data.proposedMove && INTERACTIVE_MODE) {
        PENDING_AI_MOVE = data.proposedMove;

        // Determine Icon
        let icon = MOVE_TYPE_ICONS[data.moveName] || '🎓';
        if (data.moveName === 'Research' && data.proposedMove.type !== undefined) {
            icon = TECH_ICONS[data.proposedMove.type] || '🧪';
        }

        // Show HUD
        const moveNameEl = document.getElementById('teacher-move-name');
        const moveIconEl = document.getElementById('teacher-move-icon');

        if (moveNameEl) moveNameEl.textContent = data.moveDescription || data.moveName;
        if (moveIconEl) moveIconEl.textContent = icon;

        if (teacherHud) teacherHud.classList.remove('hidden');

        // Visualize on map
        renderer.highlightMove(PENDING_AI_MOVE);
    }
}

function acceptAiMove() {
    if (!PENDING_AI_MOVE) return;
    playMove(PENDING_AI_MOVE);
    PENDING_AI_MOVE = null;
    if (teacherHud) teacherHud.classList.add('hidden');
}

function enableOverride() {
    PENDING_AI_MOVE = null;
    if (teacherHud) teacherHud.classList.add('hidden');
    renderer.clearMoveHighlight();
    showToast("👉 Play your move now");
}

// Keyboard shortcuts for Assistant
window.addEventListener('keydown', (e) => {
    if (!INTERACTIVE_MODE || teacherHud.classList.contains('hidden')) return;

    if (e.key.toLowerCase() === 'y') {
        acceptAiMove();
    } else if (e.key.toLowerCase() === 'n') {
        enableOverride();
    }
});

document.getElementById('btn-reset').onclick = () => apiAction('/reset', {});
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

document.getElementById('btn-save').onclick = async () => {
    try {
        const res = await fetch('/save_training_data', { method: 'POST' });
        const data = await res.json();
        showToast("📊 " + (data.message || data.status));
    } catch (e) {
        showToast("❌ Dataset save failed");
    }
};

document.getElementById('btn-train').onclick = async () => {
    try {
        const res = await fetch('/train', { method: 'POST' });
        const data = await res.json();
        showToast("🚀 " + (data.message || data.status));
    } catch (e) {
        showToast("❌ Training start failed");
    }
};
