import { parentPort, workerData, isMainThread } from 'worker_threads';

if (isMainThread) {
    throw new Error("Wrong thread bud");
}

import Game from "../../game";
import { isGameOver, isGameWon, isGameLost } from "../../core/functions";
import { MoveGenerator } from "../../core/moves";
import { GameState } from "../../core/states";
import { evaluateState } from "../eval";
import { MCTSNode } from './mcts';
import { UndoCallback } from '../../core/move';

(BigInt.prototype as any).toJSON = function() {
  return this.toString();
};

(Set.prototype as any).toJSON = function() {
  return Array.from(this);
};

function simulate(root: MCTSNode, game: Game, cPuct: number, stopTurn: number): {
	path: Array<[MCTSNode, number]>;
	value: number;
} {
	const path: [MCTSNode, number][] = [];
	let currentNode = root;
	let value: number = 0;
	let undos: UndoCallback[] = [];
	
	while (currentNode.isExpanded() && !isGameOver(game.state) && stopTurn > game.state.settings._turn) {
		let moveIndex = currentNode.select(cPuct);
		
		const [played, undo] = game.playMove(moveIndex)!;
		
		undos.unshift(undo);

		const child = currentNode.getOrCreateChild(moveIndex, game.state);
		
		path.push([currentNode, moveIndex]);
		
		currentNode = child;
	}
	
	if (isGameOver(game.state)) {
		value = isGameWon(game.state)? 1 : isGameLost(game.state)? -1 : 0;
	}
	else {
		currentNode.expand();
		const [ eco, army, finalScore ] = evaluateState(game);
		value = finalScore;
	}
	
	undos.forEach(x => x());
	
	return { path, value };
}

parentPort!.on('message', ({ nSims, gameJson, cPuct, stopTurn, ok }) => {
	if(ok) {
		parentPort!.postMessage({
			ok: true,
		});
		return;
	}
    const game = Game.deserialize(gameJson);
    const legalGen = (state: GameState) => MoveGenerator.legal(state);
    const root = new MCTSNode(game.state, legalGen);
    root.expand();

    for (let i = 0; i < nSims; i++) {
        const { path, value } = simulate(root, game, cPuct, stopTurn);
        root.backpropagate(path, value, root.pov);
    }
    
    parentPort!.postMessage({
        N: root.N,
        W: root.W,
    });
});