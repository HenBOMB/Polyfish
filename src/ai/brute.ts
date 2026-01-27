import { getPovTribe, isGameOver } from "../core/functions";
import Move, { UndoCallback } from "../core/move";
import { GameState } from "../core/states";
import { MoveType, TribeType } from "../core/types";
import { Prediction } from "../core/moves";
import Game from "../game";
import { evaluateState } from "./eval";
import { MCTS } from "./mcts/mcts";

type PartialMCTSSEttings = {
    depth?: number;
    deterministic?: boolean;
    dirichlet?: boolean;
    cPuct?: number;
    nThreads?: number;
    maxTurnsAhead?: number;
    legalFn?: (state: GameState) => Move[] | undefined;
    predict?: (state: GameState) => Promise<Prediction>;
}

type MCTSSettings = {
    depth: number;
    deterministic: boolean;
    dirichlet: boolean;
    cPuct: number;
    nThreads: number;
    maxTurnsAhead: number;
    legalFn?: (state: GameState) => Move[] | undefined;
    predict?: (state: GameState) => Promise<Prediction>;
}

function parseSettings(settings: PartialMCTSSEttings | null = null): MCTSSettings {
    return {
        depth: 1000,
        deterministic: false,
        dirichlet: false,
        cPuct: 1.5,
        nThreads: 2,
        maxTurnsAhead: 3,
        ...(settings || { })
    };
}

/**
 * @param game The game class in use
 * @param turnsAhead Amount of turns (i*tribeCount) to return moves for
 * @param settings Settings for the MCTS solver
 * @returns 
 */
export async function CalculateBestMoves(
    game: Game,
    turnsAhead=3,
    settings: PartialMCTSSEttings | null = null
): Promise<[Move[], number, number, number, number[]]> {
    const { depth, cPuct, dirichlet, deterministic, nThreads, maxTurnsAhead, legalFn, predict } = parseSettings(settings);
    const mcts = new MCTS(game, cPuct, dirichlet, nThreads, maxTurnsAhead, legalFn as any, predict);
    await mcts.prepare();
    
    const state = game.state;
    const maxTurn = Math.min(state.settings.turn + turnsAhead, state.settings.maxTurns);
    const undoChain: UndoCallback[] = [];
    const bestMoves: Move[] = [];

    console.log('[BRUTE] Started loop');
    let _remaining = turnsAhead;
    let _prevPov = 0;
    let _result = 0;
    let _sequence = [];
    const oPov = state.settings.currentPlayerTurnId;

    while (!isGameOver(state) && state.settings.turn <= maxTurn) {
        if (_prevPov != state.settings.currentPlayerTurnId) {
            _remaining--;
            if(_remaining < 0) {
                break;
            }
            console.log(`  (${_result > 0? '+' : ''}${_result.toFixed(8)})`);
            console.log(`\n${TribeType[getPovTribe(state).type]}'s turn`);
            _prevPov = state.settings.currentPlayerTurnId;
            _result = 0;
        }

        // Play moves until a stop function, for now case the end turn move
        // TODO: Add consistent stop function
        
        const root = await mcts.search(
            depth, 
            true,
        );
        const probs = root.distribution(deterministic? 0 : 1); 
        const bestMoveIndex = probs.indexOf(Math.max(...probs));

        if (bestMoveIndex == -1) {
            console.log(probs);
            throw Error("YO WTF");
        }

        _sequence.push(bestMoveIndex);
        
        // careful, cloning is expensive
        const oldState = game.cloneState();
        const oldEval = evaluateState(game);
        const playData = game.playMove(bestMoveIndex);

        if (!playData) {
            console.log('dead');
            break;
        }
        
        const [ playedMove, undo ] = playData;
        
        
        let diff = 0;
        
        if (playedMove.moveType !== MoveType.EndTurn) {
            const newEval = evaluateState(game);
            diff = newEval[2] - oldEval[2];

            if (diff == 0) {
                diff = newEval[1] - oldEval[1];
            }
        }

        _result += diff;

        if (diff != 0) {
            if (playedMove.moveType == MoveType.Step) {
                console.log('  ', playedMove.stringifyNow(state));
            }
            else {
                console.log(`  ${diff > 0? '+' : ''}${diff.toFixed(4)} ${playData[0].stringify(oldState, state)}`);
            }
        }
        else {
            console.log(`  ${playData[0].stringify(oldState, state)}`);
        }

        if (state.settings.currentPlayerTurnId === oPov) {
            bestMoves.push(playedMove);
        }

        if (playedMove.moveType === MoveType.EndTurn) {
            // TODO -->
            // We cant play the enemies turns because that would give us access to their entire POV and kill the FOW
            // BUT since we're not playing with FOW, we can!

            if (state.settings._fow) {
                console.log(state.settings);
                console.error("FOW not supported... yet!");
                break
            }
            else {
                // do nothing because the game class already handles the turn switch!
                // track and print it out for clarity
               _prevPov = -1;
            }
        }

        // Backwards so it undoes properly forward
        undoChain.unshift(undo);
    }

    mcts.destroy();

    if (isGameOver(state)) {
        console.log('[BRUTE] Halted by game end')
    }
    else if (state.settings.turn > maxTurn) {
        console.log('[BRUTE] Halted by game limit')
    }
    else if (_remaining < 0) {
        console.log('[BRUTE] Halted by stop function')
    }
    else {
        console.error("Fatal, something internally went south! :(");
    }

    console.log('--SEQUENCE--');
    console.log(`  [${_sequence.join(', ')}]`);

    for(const undo of undoChain) {
        undo();
    }

    return [bestMoves, ...evaluateState(game), _sequence];
}
