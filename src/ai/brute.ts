import { getPovTerritorry, getPovTribe, isGameOver } from "../core/functions";
import Move, { UndoCallback } from "../core/move";
import { EconMovesGenerator, MoveGenerator } from "../core/moves";
import EndTurn from "../core/moves/EndTurn";
import { MoveType, TribeType } from "../core/types";
import Game from "../game";
import { evaluateState } from "./eval";
import { MCTS } from "./mcts/mcts";

type StopFunction = 'lol';

type PartialMCTSSEttings = {
    depth?: number;
    deterministic?: boolean;
    dirichlet?: boolean;
    cPuct?: number;
    nThreads?: number;
}

type MCTSSettings = {
    depth: number;
    deterministic: boolean;
    dirichlet: boolean;
    cPuct: number;
    nThreads: number;
}

function parseSettings(settings: PartialMCTSSEttings | null = null): MCTSSettings {
    return {
        depth: 1000,
        deterministic: false,
        dirichlet: false,
        cPuct: 1.5,
        nThreads: 2,
        ...(settings || { })
    };
}

// export function CalculateBestMove(
//     game: Game, 
//     settings?: PartialMCTSSEttings
// ): [number, number, number, number] {
//     const { depth, cPuct, dirichlet, deterministic } = parseSettings(settings);

//     const mcts = new MCTS(game, cPuct, dirichlet);
//     const root = mcts.search(depth);

//     const probs = root.distribution(deterministic? 0 : 1); 

//     const bestMoveIndex = probs.indexOf(
//         Math.max(...probs)
//     );
    
//     // const recommended = Opening.recommend(state, legal);
//     // if (recommended.length) {
//     //     return [recommended, ...evaluateState(game)];
//     // }

//     return [bestMoveIndex, ...evaluateState(game)];
// }

/**
 * @param game The game class in use
 * @param turnsAhead Amount of turns (i*tribeCount) to return moves for
 * @param settings Settings for the MCTS solver
 * @returns 
 */
export async function CalculateBestMoves(
    game: Game,
    turnsAhead=1,
    // stopFn: StopFunction = 'limit',
    settings: PartialMCTSSEttings | null = null
): Promise<[Move[], number, number, number]> {
    const { depth, cPuct, dirichlet, deterministic, nThreads } = parseSettings(settings);
    const mcts = new MCTS(game, cPuct, dirichlet, nThreads);
    await mcts.prepare();
    
    const state = game.state;
    const maxTurn = Math.min(state.settings._turn + turnsAhead, state.settings.maxTurns);
    const undoChain: UndoCallback[] = [];
    const bestMoves: Move[] = [];

    console.log('[BRUTE] Started loop');
    let _remaining = turnsAhead + 1;
    let _prevPov = 0;

    while (!isGameOver(state) && state.settings._turn <= maxTurn) {
        if (_prevPov != state.settings._pov) {
            _remaining--;
            if(_remaining < 0) {
                break;
            }
            console.log(`\n${TribeType[getPovTribe(state).tribeType]}'s turn`);
            _prevPov = state.settings._pov;
        }

        // Play moves until a stop function, for now case the end turn move
        // TODO: Add consistent stop function
        
        const root = await mcts.search(
            depth, 
            true/*,
            // ! Not supported because Game.playMove doesnt support custom legal move generation
            (state: any) => {
                if(state.settings._pendingRewards.length) {
                    return state.settings._pendingRewards.slice();
                }
                const moves: any = [new EndTurn()];
                EconMovesGenerator.all_fast(state, moves);
                return moves;
            }*/
        );
        const probs = root.distribution(deterministic? 0 : 1); 
        const bestMoveIndex = probs.indexOf(Math.max(...probs));
        
        // careful, cloning is expensive
        const oldState = game.cloneState();
        const oldEval = evaluateState(game);
        const playData = game.playMove(bestMoveIndex);

        if (!playData) {
            break;
        }
        
        const [ playedMove, undo ] = playData;
        
        const newEval = evaluateState(game);
        const diff = newEval[2] - oldEval[2];
        if (diff != 0) {
            console.log(`> ${diff > 0? '+' : ''}${diff.toFixed(3)} ${playData[0].stringify(oldState, state)}`);
        }
        else {
            console.log(`> ${playData[0].stringify(oldState, state)}`);
        }

        if (playedMove.moveType === MoveType.EndTurn) {
            // TODO -->
            // We cant play the enemies turns because that would give us access to their entire POV and kill the FOW
            // BUT since we're not playing with FOW, we can!

            if (state.settings.fow) {
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

        bestMoves.push(playedMove);

        // Backwards so it undoes properly forward
        undoChain.unshift(undo);
    }

    mcts.destroy();

    if (isGameOver(state)) {
        console.log('[BRUTE] Halted by game end')
    }
    else if (state.settings._turn > maxTurn) {
        console.log('[BRUTE] Halted by game limit')
    }
    else if (_remaining < 1) {
        console.log('[BRUTE] Halted by stop function')
    }
    else {
        console.error("Fatal, something internally went south! :(");
    }

    for(const undo of undoChain) {
        undo();
    }

    return [bestMoves, ...evaluateState(game)];
}
