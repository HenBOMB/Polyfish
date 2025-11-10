function updateScores(scores = [0, 0]) {
    const [myScore, enemyScore] = scores;
    const totalScore = myScore + enemyScore;

    // let myScorePercentage = 50;
    // let enemyScorePercentage = 50;
    // if (totalScore > 0) {
    // }
    myScorePercentage = (myScore / totalScore) * 100;
    enemyScorePercentage = (enemyScore / totalScore) * 100;

    const scoreMeterElement = document.getElementById('score-meter');
    if (scoreMeterElement) {
        scoreMeterElement.style.setProperty('--my-score-height', `${myScorePercentage}%`);
        scoreMeterElement.style.setProperty('--enemy-score-height', `${enemyScorePercentage}%`);
    }
}

async function evaluateAll() {
    const response = await fetch('/eval');
    const data = await response.json();
    return data;
}

function updateCo() {
    return setInterval(async () => {
        const scores = await evaluateAll();
        updateScores(scores);
        if (controlState) {
            const index = remaining.shift();
            if (index === undefined) {
                controlState = 0;
                controlsAutostep();
            }
            else {
                state = await runSequence([index]);
                generateMap(state)
            }
        }
    }, 3000);
}

async function runSequence(sequence=[]) {
    return (await fetch('/sequence', { method: 'POST', body: JSON.stringify({ 
        ids: sequence, 
    }), headers: { 'Content-Type': 'application/json' } }).then(x => x.json())).state;
}

let controlState = 0;
let remaining = [];

async function controlsBck() {
    console.log('bck');
}

async function controlsFwd() {
    console.log('fwd');
    const { bestMoves } = await fetch('/bestmoves', { method: 'POST', headers: { 'Content-Type': 'application/json' } }).then(x => x.json());
    state = await runSequence([bestMoves.shift()]);
    generateMap(state);
    remaining = bestMoves;
    controlState = 1;
}

async function controlsAutostep() {
    console.log('autostepping');
    if (!controlState) {
        const { bestMoves } = await fetch('/bestmoves', { method: 'POST', headers: { 'Content-Type': 'application/json' } }).then(x => x.json());
        remaining = bestMoves;
        controlState = 1;
    }
    else {
        controlState = 0;
    }
}

let update = updateCo();

window.addEventListener('load', async () => {
    const scores = await evaluateAll();
    updateScores(scores);
});

window.addEventListener('unload', () => {
    clearInterval(updateCo);
    const scoreMeterElement = document.getElementById('score-meter');
    if(scoreMeterElement) {
        scoreMeterElement.remove();
    }
});
