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
    REPLAY_HISTORY = data.state.history || [];
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

    // Start at the end? Or beginning?
    // Let's start at the beginning (Step 0)
    jumpToStep(0);
}

function exitReplayMode() {
    REPLAY_MODE = false;
    document.getElementById('app-container').classList.remove('replay-mode');
    document.getElementById('replay-controls').classList.add('hidden');
    // Reload page or re-fetch current live state
    location.reload();
}

async function jumpToStep(index) {
    if (index < 0) index = 0;
    if (index > REPLAY_HISTORY.length) index = REPLAY_HISTORY.length;

    REPLAY_STEP_INDEX = index;
    updateReplayControls();

    showToast(`Seeking to step ${index}...`);

    try {
        const res = await fetch('/replay/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                filename: REPLAY_DATA.filename || extractFilename(), // We need to ensure we have the filename
                step_index: index,
                iterations: 100 // Default quick analysis
            })
        });
        const data = await res.json();

        if (data.error) {
            showToast("Error: " + data.error);
            return;
        }

        // Render Analysis
        renderReplayAnalysis(data);

        // If the backend returns the state (I need to add this), we update the map.
        if (data.state) {
            GAME_STATE = data.state;
            renderer.render(GAME_STATE, [], 1); // Render with no moves valid (it's replay)
            renderer.renderMCTSHeatmap(data.mctsAnalysis);
        } else {
            // Fallback if I haven't updated backend yet (I will next)
            showToast("State update missing - Backend update required");
        }

    } catch (e) {
        console.error(e);
    }
}

function extractFilename() {
    // Hack: we didn't store filename in REPLAY_DATA in load_replay_endpoint response? 
    // Check load_replay_endpoint... it doesn't return filename in the root.
    // I should fix that too.
    // For now, store it in global when loading.
    return window.CURRENT_REPLAY_FILENAME;
}

function renderReplayAnalysis(data) {
    const p1 = document.getElementById('replay-user-move');
    const p2 = document.getElementById('replay-ai-move');
    const score = document.getElementById('replay-score-diff');

    if (p1) p1.textContent = data.userMove ? data.userMove.description : "None";
    if (p2) p2.textContent = data.aiMove ? data.aiMove.description : "Thinking...";

    // Compare
    // scores? data.mctsAnalysis.evaluations has the AI moves.
    // We can try to find the user move in evaluations to compare win rates.
}

function updateReplayControls() {
    document.getElementById('replay-step-val').textContent = `${REPLAY_STEP_INDEX} / ${REPLAY_HISTORY.length}`;
}

// Global exposure
window.openReplayMenu = openReplayMenu;
window.exitReplayMode = exitReplayMode;
window.nextReplayStep = () => jumpToStep(REPLAY_STEP_INDEX + 1);
window.prevReplayStep = () => jumpToStep(REPLAY_STEP_INDEX - 1);
