import { cloneState } from "../core/functions";
import { generateTechMoves, generateResourceMoves, generateStructureMoves, generateEndTurnMove } from "../core/moves";
import { GameState } from "../core/states";
import GameLoader from "../core/gameloader";
import { evaluateArmy, evaluateEconomy } from "./eval";
import { BestMoves } from "./evalMoves";
import { findBestMoves } from "./mctsgpt";

const loader = new GameLoader();

export function benchEconomy(state: GameState, iterations: number) {
    const newState: GameState = cloneState(state);
    const initialScore = evaluateEconomy(newState);
    
    const generator = (state: GameState) => {
        const score = evaluateEconomy(state);
        
        let bestTechMove: null | BestMoves = findBestMoves(state, 10, 5, undefined, (state: GameState) => {
            return [
                ...generateTechMoves(state), 
                ...generateResourceMoves(state),
                ...generateStructureMoves(state),
            ];
        });
        
        if(bestTechMove && score <= bestTechMove.score) {
            bestTechMove = null;
        }
        
        return [
            // ...(bestUnitMove? [bestUnitMove.moves[0]] : []), 
            ...(bestTechMove? [bestTechMove.moves[0]] : []),
            ...generateStructureMoves(state),
            ...generateResourceMoves(state),
            generateEndTurnMove(),
            // ...generateUnitMoves(state, state.tribes[pov]._units[0]),
        ];
    };

    for (let i = 0; i < iterations; i++) {
        const DEPTH = [4, 6, 10][state.settings._turn < 5? 0 : state.settings._turn < 10? 1 : 2];
        const bestEconomyMoves = findBestMoves(newState, 1000, DEPTH, undefined, generator);
        console.log((bestEconomyMoves.score - initialScore).toFixed(2));
    }
}