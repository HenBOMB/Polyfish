import Move from "../core/move";
import { MoveGenerator } from "../core/moves";
import { GameState } from "../core/states";
import Game from "../game";
import { evaluateEconomy, evaluateState } from "./eval";
import { MCTS } from "./mcts";
import { Opening } from "./opening";

export function CalculateBestMoves(game: Game): [Move[], number, number, number] {
    console.log(`[BRUTE] Calculating for turn ${game.state.settings._turn}`);

    const state = game.state;
    
    const legal = MoveGenerator.legal(state);

    const recommended = Opening.recommend(state, legal);

    if (recommended.length) {
        return [recommended, ...evaluateState(state)];
    }

    console.warn("BRUTE NOT FULLY IMPLEMENTED")

    return [[], ...evaluateState(state)];
}