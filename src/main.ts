import AIState from "./aistate";
import { isGameOver } from "./core/functions";
import GameLoader from "./core/gameloader";
import { UndoCallback } from "./core/move";
import { MoveGenerator } from "./core/moves";
import { MoveType, TechnologyType } from "./core/types";
import Game from "./game";
// import { generateFile } from "./zorbist/generateZorbist";

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

export default async function main(loader: GameLoader) {
    // generateFile();
    // return;

    console.clear();

    await loader.loadRandom(6);
    
    return;

    const game = new Game();
    game.load(loader.currentState);

    // console.log(game.state.hash.toString() === '16459832259115826859');


    // console.time('test');
    // console.log(await predict(loader.currentState));
    // console.timeEnd('test');
    // Inside your game logic or a test script where you have 'state'

    AIState.assertConfig(loader.currentState);

    let depth = 100_000;

    const superchain: UndoCallback[] = [];
    const label = `play ${depth} depth`;
    console.time(label);

    const map = { };

    try {
        while(depth > 0) {
        // while(!isGameOver(game.state)) {
            // const oldState = game.cloneState();
            // const str = keyify(game.state);
            // check there
            // if(state.settings._pendingRewards.length) {
            //     return state.settings._pendingRewards;
            // }
            const moves = MoveGenerator.legal(game.state);
            // const moves = game.poser.get(game);
            const move = moves[Math.floor(Math.random() * moves.length)];
            let result;
            try {
                result = game.
                playMove(move);
            } catch (error) {
                console.log(move);
                console.log(error);
                break;
            }
    
            if(result) {
                const [ played, undo ] = result;
                // console.log(played.stringify(oldState, game.state));
                superchain.push(undo);
            }
    
            depth--;
            
            if(isGameOver(game.state)) {
                game.reset();
            }

            // if(depth === 0) {
            //     // console.log(moves.map(x => x.stringify(oldState, game.state)));
            //     break;
            // }
        }
    } catch (error) {
        console.log(error);
    }

    console.log(MoveGenerator.transpose.size);

    game.poser.close();

    console.timeEnd(label);
    // game.poser.save();

    // MoveGenerator.legal(game.state).map(x => x.stringify(superOldState, game.state));

    const modified = game.cloneState();

    // const chain = playSequence(
    //     game,
    //     MoveType.EndTurn,
    //     MoveType.EndTurn,
    // );
    
    superchain.reverse().forEach(x => x());

    deepCompare(
        { state: {
            r: loader.currentState.resources,
            s: loader.currentState.structures,
            t: loader.currentState.tiles,
            T: loader.currentState.tribes,
            S: loader.currentState.settings,
        } },
        { state: {
            r: game.state.resources,
            s: game.state.structures,
            t: game.state.tiles,
            T: game.state.tribes,
            S: game.state.settings,
        } },
        'state',
        true
    );

    loader.currentState = modified;

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

function playSequence(game: Game, ...moves: MoveType[]) {
    const chain = [];
    const p1 = game.state.tribes[1];
    const p2 = game.state.tribes[2];

    p1._tech.push({ discovered: true, techType: TechnologyType.Fishing});
    p2._tech.push({ discovered: true, techType: TechnologyType.Fishing});

    for(const moveType of moves) {
        try {
            const oldState = game.cloneState();
            const moves = MoveGenerator.legal(game.state).filter(x => x.moveType === moveType);
            const move = moves[Math.floor(Math.random() * moves.length)];
            const result = game.playMove(move);

            if(result) {
                const [ played, undo ] = result;
                console.log('+', played.stringify(oldState, game.state));
                chain.push(undo);
            }

            // const nets = game.network.updateConnectionsAfterChange();
            // if(nets) {
            //     console.log('NET!');
            //     console.log(nets);
            // }
        } catch (error) {
            console.log(error);
            if(error == 'Move is undefiend!') continue;
            console.log(MoveGenerator.legal(game.state));
            return
        }
    }

    console.log('--- moves --');
    console.log(game.state.settings._pov);
    MoveGenerator.legal(game.state).forEach(x =>
        console.log('-', x.stringify(game.state, game.state))
    );
    console.log('--- moves --');

    return chain;
}
