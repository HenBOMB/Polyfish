import AIState from "./aistate";
import { isGameOver } from "./core/functions";
import GameLoader from "./core/gameloader";
import { UndoCallback } from "./core/move";
import { MoveGenerator } from "./core/moves";
import Game from "./game";

let finishedComputing = true;

export function deepCompare<T>(a: T, b: T, key: string, ignoreObjKeyLength?: boolean) {
    let success = true;
    if (Array.isArray(a) && Array.isArray(b)) {
        if (a.length !== b.length) {
            console.log('Count', key, 'is', b.length, '!=', a.length);
            success = false;
        } else {
            for (let i = 0; i < a.length; i++) {
                if(!deepCompare(a[i], b[i], key+'-'+i.toString(), ignoreObjKeyLength)) success = false;
            }
        }
    } else if (typeof a === 'object' && typeof b === 'object') {
        if (Object.keys(a as any).length !== Object.keys(b as any).length && !ignoreObjKeyLength) {
            console.log('Count', key, 'is', Object.keys(b as any), '!=', Object.keys(a as any));
            success = false;
        } else {
            for (const k in a) {
                if(!deepCompare(a[k], b![k], k, ignoreObjKeyLength)) success = false;
            }
        }
    } else if (typeof a === 'number' && typeof b === 'number') {
        const epsilon = 1e-10;
        if (Math.abs(a - b) > epsilon) {
            console.log('Mismatch number:', key, b, '!=', a);
            success = false;
        }
    } else if (a !== b) {
        console.log('Mismatch any:', key, b, '!=', a);
        success = false;
    }
    return success;
}

export default async function main() {
    const loader = new GameLoader();
    await loader.loadRandom();
    // console.time('test');
    // console.log(await predict(loader.currentState));
    // console.timeEnd('test');
    // Inside your game logic or a test script where you have 'state'

    AIState.assertConfig(loader.currentState);
    
    const game = new Game();
    game.load(loader.currentState);

    let depth = 2;

    const superchain: UndoCallback[] = [];

    while(!isGameOver(game.state)) {
        const moves = MoveGenerator.legal(game.state);
        const move = moves[Math.floor(Math.random() * moves.length)];
        const result = game.playMove(move);

        if(result) {
            const [ played, undo ] = result;
            console.log(played.stringify(game.stateBefore, game.state));
            superchain.push(undo);            
        }

        depth--;
        
        if(depth === 0) {
            console.log(moves.map(x => x.stringify(game.stateBefore, game.state)));
            break;
        }
    }

    superchain.reverse().forEach(x => x());

    deepCompare(
        { state: { 
            l: loader.currentState._lighthouses, 
            v: loader.currentState._visibleTiles, 
            r: loader.currentState.resources, 
            s: loader.currentState.structures, 
            t: loader.currentState.tiles, 
            T: loader.currentState.tribes
        } },
        { state: { 
            l: game.state._lighthouses, 
            v: game.state._visibleTiles, 
            r: game.state.resources, 
            s: game.state.structures, 
            t: game.state.tiles, 
            T: game.state.tribes
        } },
        'state'
    );

    console.log(superchain.length);

    // loader.loadFromSpawnNotation(
    //     'domination,0,30,1'
    //     + ';' + 
    //     'yaaielvelukizeaqbapo'
    //     + ';' + 
    //     [   // Replace water tiles with 0
    //         'aqaqaqaqbabababababalulululu',
    //         'aqaqaqaqaqbaaqbababalulululu',
    //         'aqaqaqaqaqaqbabababalulululu',
    //         'yaaqaqaqaqaqzebababalulululu',
    //         'yayayayaaqbabazezezeaiaiailu',
    //         'yayayayaaqbazezezezeaiaiailu',
    //         'yayayayayaaqzezezezeaiaiaiai',
    //         'yapopopokikikizezezeaiaiaiai',
    //         'yapopopokikikizezezeaiveaiai',
    //         'yapopopokikikielzeaiveveaive',
    //         'yayakikikikielelelelveveveai',
    //         'yakikikikikielelelelveveveve',
    //         'aielaiaiaiaielelelelveveveve',
    //         'elaielaiaielelelelveveveveve',
    //     ].join('')
    //     + ';' +
    //     [
    //         '----f-f-fff-m-',
    //         'f---f--m-mw-f-',
    //         'mw-m---f-mf-wf',
    //         'fw--m-ffwwwww-',
    //         '-ww-f-f-mwoww-',
    //         '-w-wf----ww-w-',
    //         '-wwwwf-w-fffw-',
    //         '--wf-w-f-m-fw-',
    //         '-f-f-----f--f-',
    //         'm-----wf-ffmf-',
    //         'm--mwwwm-fffw-',
    //         'fff-wow---w-w-',
    //         'fff-wwwff-wwwf',
    //         '-mm--f-f--f-f-',
    //     ].join('')
    //     + ';' +
    //     [
    //         '------yyy-----',
    //         '--yy----y-y-yy',
    //         '-y-----y--y-y-',
    //         '-yy---yy-y-y-y',
    //         '---y--yy--y--y',
    //         '-y-y-y------y-',
    //         '-yyy---y-yyy--',
    //         '-y--yy-y------',
    //         '-y-y--y-------',
    //         '--y---y--y----',
    //         '----y----yyy--',
    //         '-y--y----y----',
    //         '---y---yy-----',
    //         '--------------',
    //     ].join('')
    //     + ';' +
    //     [
    //         '----------------------------',
    //         '----------vv----------------',
    //         '----aq----------ba----lu----',
    //         '----------------------------',
    //         '----------vv----------------',
    //         '----ya----------ze----ai----',
    //         '----------------------------',
    //         '----------------------------',
    //         '----po----ki----rs--vv----rs',
    //         '----------------------------',
    //         '----------------------------',
    //         'rs----vv--------el----ve----',
    //         '----------------------------',
    //         '--------rs------------------',
    //     ].join('')
    // );
    
    // loader.saveState(loader.currentState, 'challenges/3-16-yadaak');
    
    // await loader.loadLive();

    // loader.saveTo('spawns/imperius-05');


    // const game = new Game(loader.currentState);

    // console.log(await game.simulate(false, 30));

    // loader.currentState = game.state;
    
    // loader.currentState = cloneState(game.state);

    // const state = loader.loadFromSave(`challenges/3-16-yadaak`);

    // const oState: GameState = cloneState(state);
 
    // const undoChain: UndoCallback[] = [];

    // Find the moves and apply them
    // const besties = bestMoves(state).moves;
    // logAndUndoMoves(besties, state, true);

    // playMove(state, ...besties);
    // endTurn(loader, state); // end bardur
    
    // playRandomMoves(state);
    // endTurn(loader, state); // end kickoo

    // console.log('\n');
    // logAndUndoMoves(bestMoves(state).moves, state, true);
    // playRandomMoves(state);
    // endTurn(loader, state); // end hoodrick

    // playRandomMoves(state);
    // endTurn(loader, state); // end yaadak

    // playRandomMoves(state);
    // endTurn(loader, state); // end oumaji

    // // move oumaji rider
    // playMove(
    //     state, 
    //     ...UnitMoveGenerator.steps(state, getPovTribe(state)._units[0]!, null, 60)
    // );
    // endTurn(state); // end oumaji

    // new turn starts

    // loader.saveState(state, `challenges/3-15-bardur-${state.settings.turn}`);

    // console.log('\n');
    // logAndUndoMoves(bestMoves(state).moves, state, true);//, []);

    // loader.currentState = cloneState(state);

    // undoChain.reverse().forEach(undo => undo());

    // console.log('\n');
    // if(!deepCompare(oState, state, 'state', true)) {
    //     console.log('\n---DEEP COMPARE FAILED---');
    // }
}

// const interval = setInterval(() => {
//     main();
// }, 3 * 1000);

// process.on('SIGINT', () => {
//     clearInterval(interval);
//     process.exit(2);
// });