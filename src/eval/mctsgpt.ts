import { generateEcoMoves } from "../core/moves";
import { GameState } from "../core/states";
import { evaluateEconomy } from "./eval";
import { UndoCallback } from "../core/moves";
import { BestMoves } from "./evalMoves";
import Move from "../core/move";

class MCTSNode {
    parent: MCTSNode | null;
    move: Move | null;
    children: MCTSNode[];
    visits: number;
    totalScore: number;
    untriedMoves: Move[];
    rewardMoves: Move[];

    constructor(parent: MCTSNode | null, move: Move | null, state: GameState, generator?: (state: GameState) => Move[]) {
        this.parent = parent;
        this.move = move;
        this.children = [];
        this.visits = 0;
        this.totalScore = 0;
        this.rewardMoves = [];
        this.untriedMoves = generator ? generator(state) : generateEcoMoves(state);
    }
}

function uctValue(child: MCTSNode, parentVisits: number, explorationConstant: number): number {
    if (child.visits === 0) return Infinity;
    return (child.totalScore / child.visits) +
           explorationConstant * Math.sqrt(Math.log(parentVisits) / child.visits);
}

function runIteration(
    rootNode: MCTSNode,
    state: GameState,
    maxDepth: number,
    explorationConstant: number,
    generator?: (state: GameState) => Move[],
    evaluator?: (state: GameState) => number,
) {
    let node = rootNode;
    const undoStack: UndoCallback[] = [];
    let depth = 0;

    evaluator = evaluator || evaluateEconomy;
    generator = generator || generateEcoMoves;

    // SELECTION
    while (node.untriedMoves.length === 0 && node.children.length > 0 && depth < maxDepth) {
        const bestChild = node.children.reduce((prev, curr) =>
            uctValue(curr, node.visits, explorationConstant) > uctValue(prev, node.visits, explorationConstant) ? curr : prev
        );
        if (bestChild.move) {
            const branch = bestChild.move.execute(state);
            undoStack.push(branch.undo);
            if(bestChild.rewardMoves.length) {
                for(const br of  bestChild.rewardMoves) {
                    undoStack.push(br.execute(state).undo);
                }
            }
        }
        node = bestChild;
        depth++;
    }

    // EXPANSION
    if (depth < maxDepth && node.untriedMoves.length > 0) {
        const moveIndex = Math.floor(Math.random() * node.untriedMoves.length);
        const move = node.untriedMoves.splice(moveIndex, 1)[0];
        const branch = move.execute(state);
        undoStack.push(branch.undo);
        const childNode = new MCTSNode(node, move, state, generator);
        const chained = branch.chainMoves? branch.chainMoves[0] : null;
        if(chained) {
            undoStack.push(chained.execute(state).undo);
            childNode.rewardMoves = [chained];
        }
        node.children.push(childNode);
        node = childNode;
        depth++;
    }

    // SIMULATION
    let simulationDepth = maxDepth - depth;
    while (simulationDepth > 0) {
        const moves = generator(state);
        if (moves.length === 0) break;
        const randomMove = moves[Math.floor(Math.random() * moves.length)];
        const branch = randomMove.execute(state);
        undoStack.push(branch.undo);
        const chained = branch.chainMoves? branch.chainMoves[0] : null;
        if(chained) {
            undoStack.push(chained.execute(state).undo);
        }
        simulationDepth--;
        // EVAL Stop when reward is found
        // if(branch.chainMoves?.length) {
        //     break;
        // }
    }

    // PROPAGATION
    const result = evaluator(state);

    let currentNode: MCTSNode | null = node;
    while (currentNode !== null) {
        currentNode.visits++;
        currentNode.totalScore += result;
        currentNode = currentNode.parent;
    }

    while (undoStack.length > 0) {
        undoStack.pop()!();
    }
}

/**
 * Performs MCTS starting from the given state.
 *
 * @param state The game state.
 * @param iterations Number of MCTS iterations.
 * @param maxDepth Maximum search depth.
 * @param explorationConstant The exploration constant (default √2).
 * @param generator Optional move generator.
 * @returns BestMoves with the full move sequence (including any chosen forced moves) and score.
 */
function mctsSearch(
    state: GameState,
    iterations: number,
    maxDepth: number,
    explorationConstant: number = Math.SQRT2,
    generator?: (state: GameState) => Move[],
    evaluator?: (state: GameState) => number
): BestMoves {
    const rootNode = new MCTSNode(null, null, state, generator);

    evaluator = evaluator || evaluateEconomy;

    for (let i = 0; i < iterations; i++) {
        runIteration(rootNode, state, maxDepth, explorationConstant, generator, evaluator);
    }

    // Choose the best child from the root by average score.
    const bestChild = rootNode.children.reduce((best, child) => {
        const bestAvg = best.visits > 0 ? best.totalScore / best.visits : -Infinity;
        const childAvg = child.visits > 0 ? child.totalScore / child.visits : -Infinity;
        return childAvg > bestAvg ? child : best;
    }, rootNode.children[0]);

    if(!bestChild) {
        return { moves: [], score: evaluator(state) };
    }

    // Build the full move sequence, appending each node's primary move and its single forced move.
    let movesSequence: Move[] = [];
    let current: MCTSNode | null = bestChild;
    while (current && current.move) {
        movesSequence = [...movesSequence, current.move, ...current.rewardMoves];
        if (current.children.length > 0) {
            current = current.children.reduce((best, child) => {
                const bestAvg = best.visits > 0 ? best.totalScore / best.visits : -Infinity;
                const childAvg = child.visits > 0 ? child.totalScore / child.visits : -Infinity;
                return childAvg > bestAvg ? child : best;
            }, current.children[0]);
        } else {
            break;
        }
    }

    const avgScore = bestChild.visits > 0 ? bestChild.totalScore / bestChild.visits : evaluator(state);
    return { moves: movesSequence, score: avgScore };
}

/**
 * Entry point for finding the best move sequence using MCTS.
 *
 * @param state The current game state.
 * @param iterations Number of MCTS iterations.
 * @param maxDepth Maximum search depth.
 * @param explorationConstant Optional exploration constant (default √2).
 * @param generator Optional move generator.
 * @param evaluator Optional state evaluator.
 * @returns BestMoves containing the move sequence (with chosen forced moves) and evaluation score.
 */
export function findBestMoves(
    state: GameState,
    iterations: number,
    maxDepth: number,
    explorationConstant?: number,
    generator?: (state: GameState) => Move[],
    evaluator? : (state: GameState) => number,
): BestMoves {
    const constant = explorationConstant !== undefined ? explorationConstant : Math.SQRT2;
    return mctsSearch(state, iterations, maxDepth, constant, generator, evaluator);
}
