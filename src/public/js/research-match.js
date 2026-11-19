let graphicalMatch = null;
let graphicalBusy = false;

const researchSetup = document.getElementById('research-setup');
const researchForm = researchSetup.querySelector('form');
const researchSetupError = document.getElementById('research-setup-error');
const fogControl = document.getElementById('btn-fow');
let learnedAvailable = false;

async function researchRequest(path, body) {
    const response = await fetch(path, body === undefined ? undefined : {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body
    });
    if (!response.ok) throw new Error(await response.text());
    return response.json();
}

function showGraphicalTurn() {
    fogControl.classList.toggle('hidden', Boolean(graphicalMatch));
    fogControl.setAttribute('aria-pressed', ENABLE_FOW);
    if (!graphicalMatch) return false;

    const { botTurn, opponent, result } = graphicalMatch;
    const finished = result.status !== 'inProgress';
    document.getElementById('btn-end-turn').disabled = graphicalBusy || botTurn || finished
        || !currentLegalMoves.some(move => move?.moveType === 10);
    aiTurnIndicator.classList.remove('hidden');
    aiTurnIndicator.classList.toggle('your-turn', !botTurn && !finished);
    if (result.status === 'won') {
        aiTurnIndicator.textContent = `${GAME_STATE.tribes[result.winner]?.username || `Player ${result.winner}`} wins`;
    } else if (finished) {
        aiTurnIndicator.textContent = 'Turn limit reached';
    } else if (botTurn) {
        aiTurnIndicator.textContent = `${opponent[0].toUpperCase()}${opponent.slice(1)} bot thinking…`;
    } else {
        const tribe = GAME_STATE.tribes[GAME_STATE.settings.currentPlayerTurnId];
        aiTurnIndicator.textContent = opponent === 'human' ? `${tribe.username}'s turn` : 'Your turn';
    }
    return true;
}

async function refreshGraphicalBoard() {
    updateUI(await researchRequest('/current'));
}

async function runGraphicalTurn(action) {
    if (graphicalBusy || !graphicalMatch) return;
    graphicalBusy = true;
    showGraphicalTurn();
    try {
        let match = action === undefined
            ? graphicalMatch
            : await researchRequest('/research/move', JSON.stringify({ revision: graphicalMatch.revision, action }));
        for (const message of match.messages || []) showToast(message);
        if (match.botTurn && !graphicalMatch.botTurn) await refreshGraphicalBoard();
        while (match.botTurn) {
            match = await researchRequest('/research/bot', '{}');
        }
        await refreshGraphicalBoard();
    } catch (error) {
        showToast(error.message || 'Could not play the turn');
        await refreshGraphicalBoard().catch(console.error);
    } finally {
        graphicalBusy = false;
        showGraphicalTurn();
    }
}

function playGraphicalMove(action) {
    if (graphicalBusy || graphicalMatch.botTurn || graphicalMatch.result.status !== 'inProgress') return;
    renderer.selectedIdx = null;
    selectedUnitIdx = null;
    runGraphicalTurn(action);
}

function fitGraphicalBoard() {
    if (!GAME_STATE.settings) return;
    const last = GAME_STATE.settings.size - 1;
    const { x, y } = renderer.getPos(last, 0);
    const width = 2 * x + tileSize;
    const height = 2 * y + tileSize;
    scale = Math.min(2, 0.9 * mapViewport.clientWidth / width, 0.9 * mapViewport.clientHeight / height);
    centerOnCoordinates(last / 2, last / 2);
}

function updateLearnedOption() {
    const opponent = researchForm.elements.opponentPreset;
    const learned = opponent.querySelector('[value="learned"]');
    learned.disabled = !learnedAvailable || researchForm.elements.mapSize.value !== '121';
    if (learned.disabled && opponent.value === 'learned') opponent.value = 'normal';
}

async function openResearchSetup() {
    if (graphicalBusy) return;
    researchSetupError.hidden = true;
    researchSetup.showModal();
    try {
        learnedAvailable = graphicalMatch?.learnedAvailable ?? (await researchRequest('/research/current')).learnedAvailable;
        updateLearnedOption();
    } catch (error) {
        console.error(error);
    }
}

researchForm.elements.mapSize.addEventListener('change', updateLearnedOption);
document.getElementById('btn-cancel-setup').onclick = () => researchSetup.close();
researchForm.addEventListener('submit', async event => {
    event.preventDefault();
    if (graphicalBusy) return;
    graphicalBusy = true;
    researchSetupError.hidden = true;
    const submit = event.submitter;
    submit.disabled = true;
    try {
        const { seed, mapSize, ...options } = Object.fromEntries(new FormData(researchForm));
        const settings = JSON.stringify({ ...options, mapSize: Number(mapSize) });
        const body = seed ? `${settings.slice(0, -1)},"seed":${BigInt(seed)}}` : settings;
        await researchRequest('/reset', body);
        renderer.clear();
        await refreshGraphicalBoard();
        researchSetup.close();
        fitGraphicalBoard();
    } catch (error) {
        researchSetupError.textContent = error.message;
        researchSetupError.hidden = false;
    } finally {
        submit.disabled = false;
        graphicalBusy = false;
        showGraphicalTurn();
    }
});
