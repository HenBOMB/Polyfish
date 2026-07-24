/**
 * Replay System Logic
 */

let REPLAY_MODE = false;
let REPLAY_DATA = null;
let REPLAY_STEP_INDEX = 0;
let REPLAY_HISTORY = []; // Array of serialized moves
let REPLAY_INITIAL_STATE = null; // Snapshot of turn 0

// Track current simulation state relative to replay
let REPLAY_CURRENT_GAME_STATE = null;
let REPLAY_REQUEST_SEQ = 0; // guards against out-of-order /replay/analyze responses

// Autoplay: a self-pacing loop that advances one step, waits, repeats. Manual
// navigation pauses it. REPLAY_PLAY_TOKEN invalidates an in-flight loop so a
// pause/seek can't be overtaken by a stale iteration still awaiting its fetch.
let REPLAY_PLAYING = false;
let REPLAY_PLAY_TOKEN = 0;
let REPLAY_INTERVAL_MS = 1500;

async function openReplayMenu() {
    // For now, just prompt for a filename or list them
    // Ideally we'd have a modal. Let's use a simple prompt for MVP.
    const filename = prompt("Enter replay filename (e.g. 'game_123.json'):");
    if (filename) {
        await loadReplay(filename);
    }
}

async function loadReplay(filename) {
    showToast("Loading Replay...");
    try {
        const res = await fetch('/replay/load', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename })
        });
        const data = await res.json();

        if (data.status === 'error') {
            alert(data.message);
            return;
        }

        window.CURRENT_REPLAY_FILENAME = data.filename; // Store for analysis calls
        enterReplayMode(data);
    } catch (e) {
        console.error("Failed to load replay:", e);
        showToast("Error loading replay");
    }
}

function enterReplayMode(data) {
    REPLAY_MODE = true;
    REPLAY_DATA = data;
    // Prefer the engine-recorded history; fall back to flattening the mod's
    // turns[].players[].commands[] (in turn, then playerId order) so wrapped
    // replays can be stepped too.
    const hist = (data.state && data.state.history && data.state.history.length)
        ? data.state.history
        : flattenTurns(data.turns);
    REPLAY_HISTORY = hist;
    // The loaded state is usually the FINAL state. 
    // To replay, we actually need the INITIAL state.
    // However, the backend /load endpoint returns the SAVE state.
    // If we want to step through, we have to start from Turn 1.
    // The backend /analyze endpoint handles re-simulation from seed.
    // Frontend-side simple replay:
    // We can't easily "undo" moves in frontend without backend support for full state at each step.
    // SO: We will rely on the backend to provide state at step X? 
    // OR we just use /analyze endpoint which computes state at step X.
    // Let's rely on /replay/analyze to jump to steps, 
    // OR we just use the final state to show "Game Over" and then "Restart" to go to step 0?

    // Actually, 'analyze_replay_step' endpoint replays from scratch up to index.
    // So to "View Step X", we call /replay/analyze with step_index=X.
    // This returns the User Move and AI Analysis.
    // But it ALSO returns 'evaluation' (current state score).
    // It doesn't return the full GameState to render... wait.
    // We NEED the full GameState to render the map at that step.
    // The current `analyze_replay_step` returns `evaluation` but NOT `state`.
    // I should probably update `analyze_replay_step` to return the `state` as well, 
    // or create a separate `get_replay_state` endpoint.

    // For now, let's update the backend to return `state` in `analyze_replay_step`.
    // I will do that in a separate step.

    // Initialize UI
    document.getElementById('app-container').classList.add('replay-mode');
    document.getElementById('replay-controls').classList.remove('hidden');
    updatePlayButton();

    // Show the starting board IMMEDIATELY from the /replay/load payload (which
    // already carries the full initial state) instead of blocking on the step-0
    // /replay/analyze round-trip. That analyze call runs the network for the AI
    // overlay, and the first eval after startup can be slow (Metal kernel
    // warmup); decoupling the board render makes load feel instant.
    if (data.state) {
        REPLAY_STEP_INDEX = 0;
        updateReplayControls();
        try {
            applyReplayState(data.state, null);
        } catch (e) {
            reportReplayError('initial-render', e.message || String(e), e);
        }
    }

    // Then fetch step 0 to populate the AI-suggestion overlay.
    jumpToStep(0);
}

function exitReplayMode() {
    stopReplayPlayback();
    REPLAY_MODE = false;
    document.getElementById('app-container').classList.remove('replay-mode');
    document.getElementById('replay-controls').classList.add('hidden');
    // Reload page or re-fetch current live state
    location.reload();
}

// Surface replay failures durably: browser console (with a [replay] prefix so
// they are greppable) AND a persistent panel in the replay controls, so a
// failed step is diagnosable without leaving the step counter silently
// climbing over a frozen map.
function reportReplayError(context, message, extra) {
    const filename = (REPLAY_DATA && REPLAY_DATA.filename) || extractFilename() || '?';
    console.error(`[replay] step ${REPLAY_STEP_INDEX} (${filename}) — ${context}: ${message}`, extra || '');
    const el = document.getElementById('replay-error');
    if (el) {
        el.textContent = `⚠️ Step ${REPLAY_STEP_INDEX} ${context}:\n${message}`;
        el.classList.remove('hidden');
    }
    showToast(`Replay ${context} error (see panel/console)`);
}

function clearReplayError() {
    const el = document.getElementById('replay-error');
    if (el) {
        el.textContent = '';
        el.classList.add('hidden');
    }
}

// Returns true if the step rendered, false on any error (so autoplay can pause
// on failure) or if the request was superseded by a newer one.
async function jumpToStep(index) {
    if (index < 0) index = 0;
    // The analyze endpoint replays moves 0..index and reports the state BEFORE
    // history[index], so the last reachable step is length-1 (index == length
    // is rejected as out-of-bounds server-side).
    const maxIndex = Math.max(0, REPLAY_HISTORY.length - 1);
    if (index > maxIndex) index = maxIndex;

    REPLAY_STEP_INDEX = index;
    updateReplayControls();

    // Rapid next/prev clicks can fire overlapping requests that resolve out
    // of order; a stale response landing last would freeze the Turn/stat
    // display on an old step even though REPLAY_STEP_INDEX has moved on. Only
    // apply the response if this is still the most recent request in flight.
    const requestSeq = ++REPLAY_REQUEST_SEQ;

    let res, data;
    try {
        res = await fetch('/replay/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                filename: REPLAY_DATA.filename || extractFilename(), // We need to ensure we have the filename
                step_index: index,
                iterations: 8 // Light MCTS overlay so stepping stays responsive (100 = ~6s/step on candle-CPU)
            })
        });
    } catch (e) {
        if (requestSeq !== REPLAY_REQUEST_SEQ) return false;
        reportReplayError('network', e.message || String(e), e);
        return false;
    }

    // A non-2xx (e.g. a server panic → 500) would otherwise make res.json()
    // throw with no clue why; capture the raw body so the reason is visible.
    if (!res.ok) {
        if (requestSeq !== REPLAY_REQUEST_SEQ) return false;
        const body = await res.text().catch(() => '(unreadable body)');
        reportReplayError('server', `HTTP ${res.status} ${res.statusText}\n${body.slice(0, 800)}`);
        return false;
    }

    try {
        data = await res.json();
    } catch (e) {
        if (requestSeq !== REPLAY_REQUEST_SEQ) return false;
        reportReplayError('bad-response', 'Response was not valid JSON: ' + (e.message || e), e);
        return false;
    }

    if (requestSeq !== REPLAY_REQUEST_SEQ) return false; // superseded by a newer step

    // Backend logical errors (move desync, out-of-bounds, parse failure) come
    // back as {error} / {status:'error'}. This is the primary diagnostic — the
    // message names the exact step and move that failed to replay.
    if (data.error || data.status === 'error') {
        reportReplayError('replay', data.error || data.message || 'Unknown backend error', data);
        return false;
    }

    if (!data.state) {
        reportReplayError('missing-state', 'Backend returned no game state for this step', data);
        return false;
    }

    clearReplayError();

    // Render Analysis
    renderReplayAnalysis(data);

    // Render throws must not masquerade as fetch failures: give them their own
    // guard so the panel/console says "render" and names the step, instead of
    // the counter climbing over a frozen map with nothing logged.
    try {
        applyReplayState(data.state, data.mctsAnalysis);
    } catch (e) {
        reportReplayError('render', e.message || String(e), e);
        return false;
    }
    return true;
}

// Apply a replay game state to the board + stat bar. Shared by step-through
// (jumpToStep) and the instant initial render in enterReplayMode.
function applyReplayState(state, mctsAnalysis) {
    // Replays can use a different map size than the live game; purge stale
    // tiles so they don't render outside the replay's square.
    const prevSettings = GAME_STATE.settings || null;
    const newSize = state.settings ? state.settings.size : null;
    const newTileCount = state.settings ? state.settings.tile_count : null;
    if (prevSettings && (newSize !== prevSettings.size || newTileCount !== prevSettings.tile_count)) {
        renderer.clear();
    }
    GAME_STATE = state;

    // The stat bar (Turn/Score/Stars/Income) is otherwise only kept in sync by
    // updateUI(), which the replay step path bypasses — update it here too so it
    // reflects this step instead of staying frozen at its pre-replay value.
    const stepTribeId = GAME_STATE.settings.currentPlayerTurnId;
    const stepTribe = GAME_STATE.tribes[stepTribeId.toString()] || GAME_STATE.tribes[stepTribeId];
    turnVal.textContent = GAME_STATE.settings.turn;
    turnTotalVal.textContent = GAME_STATE.settings.maxTurns;
    if (stepTribe) {
        tribeNameLabel.textContent = TRIBE_ID_2_NAME[stepTribe.type] || 'Unknown';
        starsVal.textContent = stepTribe.stars;
        scoreVal.textContent = stepTribe.score;
        const income = stepTribe.cities.reduce((acc, cur) => acc + getCityProduction(GAME_STATE, cur), 0);
        incomeVal.textContent = `+${income}`;
    }

    // Render from the POV of whoever's turn it is at this step, so the view
    // (and FOW) follows the active player as you step through.
    renderer.render(GAME_STATE, []);
    renderer.renderMCTSHeatmap(mctsAnalysis);
}

// ---- Autoplay ----------------------------------------------------------

function replaySleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function updatePlayButton() {
    const btn = document.getElementById('replay-play-btn');
    if (btn) btn.textContent = REPLAY_PLAYING ? '⏸ Pause' : '▶ Play';
}

function stopReplayPlayback() {
    REPLAY_PLAYING = false;
    REPLAY_PLAY_TOKEN++; // invalidate any loop currently awaiting a fetch
    updatePlayButton();
}

async function toggleReplayPlay() {
    if (REPLAY_PLAYING) {
        stopReplayPlayback();
        return;
    }
    if (REPLAY_HISTORY.length === 0) return;
    // Restart from the top if we're parked at the end.
    if (REPLAY_STEP_INDEX >= REPLAY_HISTORY.length - 1) {
        await jumpToStep(0);
    }
    REPLAY_PLAYING = true;
    updatePlayButton();
    const token = ++REPLAY_PLAY_TOKEN;
    playReplayLoop(token);
}

// Self-pacing loop: advance one step, then wait out the remainder of the
// interval (so slow steps never overlap and the pace stays ~REPLAY_INTERVAL_MS
// regardless of how long each analyze call takes). Pauses on error or at end.
async function playReplayLoop(token) {
    while (REPLAY_PLAYING && token === REPLAY_PLAY_TOKEN) {
        if (REPLAY_STEP_INDEX >= REPLAY_HISTORY.length - 1) {
            stopReplayPlayback();
            return;
        }
        const t0 = (typeof performance !== 'undefined') ? performance.now() : Date.now();
        const ok = await jumpToStep(REPLAY_STEP_INDEX + 1);
        if (!REPLAY_PLAYING || token !== REPLAY_PLAY_TOKEN) return; // paused/seeked mid-fetch
        if (!ok) { stopReplayPlayback(); return; }                  // pause so the error is visible
        const elapsed = ((typeof performance !== 'undefined') ? performance.now() : Date.now()) - t0;
        await replaySleep(Math.max(250, REPLAY_INTERVAL_MS - elapsed));
    }
}

function setReplaySpeed(ms) {
    REPLAY_INTERVAL_MS = Number(ms) || 1500;
}

function extractFilename() {
    // Hack: we didn't store filename in REPLAY_DATA in load_replay_endpoint response? 
    // Check load_replay_endpoint... it doesn't return filename in the root.
    // I should fix that too.
    // For now, store it in global when loading.
    return window.CURRENT_REPLAY_FILENAME;
}

// Flatten a mod-format replay's top-level `turns` array into a single
// play-ordered list of command objects (turn, then playerId order). Mirrors
// the same flattening done server-side in analyze_replay_step.
function flattenTurns(turns) {
    const moves = [];
    if (!Array.isArray(turns)) return moves;
    for (const turn of turns) {
        const players = Array.isArray(turn && turn.players) ? [...turn.players] : [];
        players.sort((a, b) => (a && (a.playerId ?? 0)) - (b && (b.playerId ?? 0)));
        for (const player of players) {
            if (player && Array.isArray(player.commands)) {
                for (const cmd of player.commands) moves.push(cmd);
            }
        }
    }
    return moves;
}

function renderReplayAnalysis(data) {
    const p1 = document.getElementById('replay-user-move');
    const p2 = document.getElementById('replay-ai-move');
    const score = document.getElementById('replay-score-diff');

    if (p1) p1.textContent = data.userMove ? (data.userMove.description || "None") : "None";
    if (p2) p2.textContent = (data.aiMove && data.aiMove.json)
        ? (data.aiMove.description || "None")
        : "AI disabled (no network)";

    // Compare
    // scores? data.mctsAnalysis.evaluations has the AI moves.
    // We can try to find the user move in evaluations to compare win rates.
}

function updateReplayControls() {
    document.getElementById('replay-step-val').textContent = `${REPLAY_STEP_INDEX} / ${REPLAY_HISTORY.length}`;
}

// Global exposure. Manual navigation pauses autoplay so it can't fight the
// user's clicks; Play/Pause is the only thing that (re)starts the loop.
window.openReplayMenu = openReplayMenu;
window.exitReplayMode = exitReplayMode;
window.toggleReplayPlay = toggleReplayPlay;
window.setReplaySpeed = setReplaySpeed;
window.nextReplayStep = () => { stopReplayPlayback(); return jumpToStep(REPLAY_STEP_INDEX + 1); };
window.prevReplayStep = () => { stopReplayPlayback(); return jumpToStep(REPLAY_STEP_INDEX - 1); };
window.replayFirst = () => { stopReplayPlayback(); return jumpToStep(0); };
window.replayLast = () => { stopReplayPlayback(); return jumpToStep(REPLAY_HISTORY.length - 1); };
