
function playMove(move) {
    if (REPLAY_MODE) return;
    renderer.clearMoveHighlight();
    document.querySelectorAll('.move-overlay').forEach(el => el.remove());
    document.querySelectorAll('.combat-preview').forEach(el => el.remove());

    if (graphicalMatch) {
        playGraphicalMove(move);
        return;
    }

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
            document.querySelectorAll('.rts-btn').forEach(b => b.disabled = false);
            const tribe = GAME_STATE.tribes[GAME_STATE.settings.currentPlayerTurnId];
            if (tribe) updateTurnIndicator(tribe);
        });
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
