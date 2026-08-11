
function playMove(move) {
    // Single-flight: clicks during an in-flight request would queue on the
    // server's game lock and execute as duplicate moves.
    if (window.MOVE_IN_FLIGHT) {
        showToast && showToast('⏳ Waiting for previous move…');
        return;
    }
    window.MOVE_IN_FLIGHT = true;

    renderer.clearMoveHighlight();
    document.querySelectorAll('.move-overlay').forEach(el => el.remove());
    document.querySelectorAll('.combat-preview').forEach(el => el.remove());

    // Reset pending state whenever a move is played
    resetMctsButton();

    document.querySelectorAll('.rts-btn').forEach(b => b.disabled = true);
    fetch('/step', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(move)
    })
        .then(r => r.json())
        .then(data => {
            renderer.selectedIdx = null;
            selectedUnitIdx = null;
            try {
                updateUI(data);
            } catch (e) {
                console.error('UI Update failed:', e);
                showToast && showToast('❌ UI Update Error');
            }
        })
        .catch(err => {
            console.error('Error playing move:', err);
            showToast && showToast('❌ Move failed');
        })
        .finally(() => {
            window.MOVE_IN_FLIGHT = false;
            document.querySelectorAll('.rts-btn').forEach(b => b.disabled = false);
        });
}


async function pollTrainingStatus() {
    try {
        const res = await fetch('/train/status');
        if (!res.ok) return;
        const data = await res.json();

        if (data.pid) {
            trainingStatusHUD.classList.remove('hidden');
            if (data.isRunning) {
                trainingStatusHUD.textContent = "TRAINING...";
                trainingStatusHUD.className = "training-led active";
            } else {
                trainingStatusHUD.textContent = "FINISHED";
                trainingStatusHUD.className = "training-led";
            }
        } else {
            trainingStatusHUD.classList.add('hidden');
        }
    } catch (e) {
        console.error("Status check failed", e);
    }
}

async function simulateExplorer(startIdx) {
    try {
        const res = await fetch('/simulate/explorer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ idx: startIdx })
        });
        const data = await res.json();
        console.log("Simulation:", data);
        renderer.renderExplorerPath(data.path, data.revealed);
    } catch (e) { console.error(e); }
}