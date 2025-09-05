import { getPovTerritorry, getPovTribe, isGameOver } from "../core/functions";
import Move, { UndoCallback } from "../core/move";
import { MoveType, TribeType } from "../core/types";
import Game from "../game";
import { evaluateState } from "./eval";
import { MCTS } from "./mctsCpu";

type StopFunction = 'lol';

type SettingsMCTSOptional = {
    depth?: number;
    deterministic?: boolean;
    dirichlet?: boolean;
    cPuct?: number;
}

type SettingsMCTS = {
    depth: number;
    deterministic: boolean;
    dirichlet: boolean;
    cPuct: number;
}

function parseSettings(settings?: SettingsMCTSOptional): SettingsMCTS {
    return {
        depth: 1000,
        deterministic: false,
        dirichlet: false,
        cPuct: 1.5,
        ...(settings || { })
    };
}

export function CalculateBestMove(
    game: Game, 
    settings?: SettingsMCTSOptional
): [number, number, number, number] {
    const { depth, cPuct, dirichlet, deterministic } = parseSettings(settings);

    const mcts = new MCTS(game, cPuct, dirichlet);
    const root = mcts.search(depth);

    const probs = root.distribution(deterministic? 0 : 1); 

    const bestMoveIndex = probs.indexOf(
        Math.max(...probs)
    );
    
    // const recommended = Opening.recommend(state, legal);
    // if (recommended.length) {
    //     return [recommended, ...evaluateState(game)];
    // }

    return [bestMoveIndex, ...evaluateState(game)];
}

/**
 * 
 * @param game The game class in use
 * @param turnsAhead Amount of turns (i*tribeCount) to search and return, set to -1 to let the AI decide based on the current game stage
 * @param settings Settings for the MCTS solver
 * @returns 
 */
export function CalculateBestMoves(
    game: Game,
    turnsAhead: number,
    // stopFn: StopFunction = 'limit',
    settings?: SettingsMCTS
): [Move[], number, number, number] {
    const { depth, cPuct, dirichlet, deterministic } = parseSettings(settings);
    const mcts = new MCTS(game, cPuct, dirichlet);
    const state = game.state;
    const maxTurn = Math.min(state.settings._turn + turnsAhead, state.settings.maxTurns);
    const undoChain: UndoCallback[] = [];
    const bestMoves: Move[] = [];

    console.log('[BRUTE] Started loop');
    let _remaining = turnsAhead;
    let _prevPov = 0;

    while (!isGameOver(state) && state.settings._turn <= maxTurn && _remaining > 0) {
        if (_prevPov != state.settings._pov) {
            console.log(`[BRUTE-loop] ${TribeType[getPovTribe(state).tribeType]}'s turn`);
            _prevPov = state.settings._pov;
        }

        // Play moves until a stop function, for now case the end turn move
        // TODO: Add consistent stop function

        const root = mcts.search(depth);
        const probs = root.distribution(deterministic? 0 : 1); 
        const bestMoveIndex = probs.indexOf(Math.max(...probs));
        
        // careful, cloning is expensive
        const oldState = game.cloneState();
        const playData = game.playMove(bestMoveIndex);

        if(!playData) {
            break;
        }
        
        const [ playedMove, undo ] = playData;
        
        console.log('[BRUTE-loop] played', playData[0].stringify(oldState, state));

        if (playedMove.moveType === MoveType.EndTurn) {
            _remaining--;
            // TODO -->
            // We cant play the enemies turns because that would give us access to their entire POV and kill the FOW
            // BUT since we're not playing with FOW, we can!

            if (state.settings.fow) {
                console.log(state.settings);
                console.error("[BRUTE-loop] FOW not supported... yet!");
                break
            }
            else {
                // do nothing because the game class already handles the turn switch!
                // track and print it out for clarity
               _prevPov = -1;
            }
        }

        bestMoves.push(playedMove);

        undoChain.push(undo);
    }

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
