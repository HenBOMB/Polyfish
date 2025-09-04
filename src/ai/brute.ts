import Game from "../game";
import { evaluateState } from "./eval";
import { MCTS } from "./mctsCpu";
import { Opening } from "./opening";

export function CalculateBestMoves(
    game: Game, 
    depth=1000, 
    deterministic=false,
    dirichlet=false,
    cPuct=1.5
): [number[], number, number, number] {
    console.log(`[BRUTE] Calculating for turn ${game.state.settings._turn}`);

    const state = game.state;
    
    const mcts = new MCTS(game, cPuct, dirichlet);
    const rootNode = mcts.search(depth);

    // T=0 picks the best move
    // T=1 picks based on visit counts
    const moveProbabilities = rootNode.distribution(deterministic? 0 : 1); 

    // 4. Find the index of the best move
    const bestMoveIndex = moveProbabilities.indexOf(Math.max(...moveProbabilities));
    
    
    // const recommended = Opening.recommend(state, legal);
    // if (recommended.length) {
    //     return [recommended, ...evaluateState(game)];
    // }

    console.warn("[BRUTE] finished")

    return [[bestMoveIndex], ...evaluateState(game)];
}