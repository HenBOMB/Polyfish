/** Canonical replay v1 playback. Rust owns validation and every state transition. */

let REPLAY_MODE = false;
let REPLAY_STEP_INDEX = 0;
let REPLAY_TOTAL_COMMANDS = 0;
let REPLAY_REQUEST_SEQ = 0;
let REPLAY_PLAY_TIMER = null;

function openReplayMenu() {
    document.getElementById('replay-file-input').click();
}

async function openReplayFile(input) {
    const file = input.files && input.files[0];
    if (!file) return;
    showToast(`Loading ${file.name}...`);
    try {
        const replay = JSON.parse(await file.text());
        const response = await fetch('/replay/open', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(replay),
        });
        const data = await response.json();
        if (!response.ok || data.status === 'error') {
            throw new Error(data.message || `HTTP ${response.status}`);
        }
        enterReplayMode(data, file.name);
    } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        showToast(`Replay error: ${message}`);
        window.alert(`Could not load replay:\n\n${message}`);
    } finally {
        input.value = '';
    }
}

async function loadReplay(filename) {
    showToast('Loading replay...');
    try {
        const response = await fetch('/replay/load', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename }),
        });
        const data = await response.json();
        if (!response.ok || data.status === 'error') throw new Error(data.message);
        enterReplayMode(data, filename);
    } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        showToast(`Replay error: ${message}`);
    }
}

function enterReplayMode(data, filename) {
    REPLAY_MODE = true;
    REPLAY_STEP_INDEX = data.cursor || 0;
    REPLAY_TOTAL_COMMANDS = data.totalCommands || 0;
    document.getElementById('app-container').classList.add('replay-mode');
    document.getElementById('replay-controls').classList.remove('hidden');
    document.getElementById('replay-title').textContent =
        data.metadata?.gameId || filename || 'Replay';
    applyReplayState(data);
}

function exitReplayMode() {
    stopReplay();
    REPLAY_MODE = false;
    document.getElementById('app-container').classList.remove('replay-mode');
    document.getElementById('replay-controls').classList.add('hidden');
    location.reload();
}

async function jumpToStep(index) {
    if (!REPLAY_MODE) return;
    const target = Math.max(0, Math.min(Number(index), REPLAY_TOTAL_COMMANDS));
    const requestSeq = ++REPLAY_REQUEST_SEQ;
    try {
        const response = await fetch('/replay/state', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ commandIndex: target }),
        });
        const data = await response.json();
        if (requestSeq !== REPLAY_REQUEST_SEQ) return;
        if (!response.ok || data.status === 'error') throw new Error(data.message);
        applyReplayState(data);
    } catch (error) {
        stopReplay();
        const message = error instanceof Error ? error.message : String(error);
        showToast(`Replay step failed: ${message}`);
    }
}

function applyReplayState(data) {
    REPLAY_STEP_INDEX = data.cursor;
    REPLAY_TOTAL_COMMANDS = data.totalCommands;
    updateReplayControls(data);

    const previousSize = GAME_STATE.settings?.size;
    const nextSize = data.state.settings?.size;
    if (previousSize && nextSize !== previousSize) renderer.clear();
    GAME_STATE = data.state;

    const playerId = GAME_STATE.settings.currentPlayerTurnId;
    const tribe = GAME_STATE.tribes[String(playerId)] || GAME_STATE.tribes[playerId];
    turnVal.textContent = GAME_STATE.settings.turn;
    turnTotalVal.textContent = GAME_STATE.settings.maxTurns;
    if (tribe) {
        tribeNameLabel.textContent = TRIBE_ID_2_NAME[tribe.type] || `Player ${playerId}`;
        starsVal.textContent = tribe.stars;
        scoreVal.textContent = tribe.score;
        const income = tribe.cities.reduce(
            (sum, city) => sum + getCityProduction(GAME_STATE, city), 0,
        );
        incomeVal.textContent = `+${income}`;
    }
    renderer.render(GAME_STATE, []);
}

function updateReplayControls(data = {}) {
    document.getElementById('replay-step-val').textContent =
        `${REPLAY_STEP_INDEX} / ${REPLAY_TOTAL_COMMANDS}`;
    const context = data.context;
    document.getElementById('replay-turn-player').textContent = context
        ? `Turn ${context.turnNumber} · Player ${context.playerId}`
        : 'Replay complete';
    document.getElementById('replay-command').textContent = data.currentCommand
        ? JSON.stringify(data.currentCommand)
        : '—';
    const result = data.result;
    document.getElementById('replay-result').textContent = !result
        ? 'Result not recorded'
        : result.draw
            ? 'Draw'
            : result.winnerPlayerId != null
                ? `Winner: player ${result.winnerPlayerId}`
                : 'Result recorded (no winner)';
    document.getElementById('replay-play').textContent = REPLAY_PLAY_TIMER ? 'Pause' : 'Play';
}

function toggleReplayPlay() {
    if (REPLAY_PLAY_TIMER) {
        stopReplay();
        return;
    }
    REPLAY_PLAY_TIMER = window.setInterval(() => {
        if (REPLAY_STEP_INDEX >= REPLAY_TOTAL_COMMANDS) {
            stopReplay();
        } else {
            jumpToStep(REPLAY_STEP_INDEX + 1);
        }
    }, 700);
    updateReplayControls();
}

function stopReplay() {
    if (REPLAY_PLAY_TIMER) window.clearInterval(REPLAY_PLAY_TIMER);
    REPLAY_PLAY_TIMER = null;
    const button = document.getElementById('replay-play');
    if (button) button.textContent = 'Play';
}

window.openReplayMenu = openReplayMenu;
window.openReplayFile = openReplayFile;
window.loadReplay = loadReplay;
window.exitReplayMode = exitReplayMode;
window.jumpToStep = jumpToStep;
window.firstReplayStep = () => jumpToStep(0);
window.lastReplayStep = () => jumpToStep(REPLAY_TOTAL_COMMANDS);
window.nextReplayStep = () => jumpToStep(REPLAY_STEP_INDEX + 1);
window.prevReplayStep = () => jumpToStep(REPLAY_STEP_INDEX - 1);
window.toggleReplayPlay = toggleReplayPlay;
