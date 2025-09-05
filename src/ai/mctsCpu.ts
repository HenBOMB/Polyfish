import AIState from "../aistate";
import { getPovTribe, isGameLost, isGameOver, isGameWon } from "../core/functions";
import { MoveType, RewardType } from "../core/types";
import { MoveGenerator, Prediction } from "../core/moves";
import { GameState } from "../core/states";
import { calculateStageValue, evaluateState, scoreMovePriority } from "./eval";
import Game from "../game";
import Move, { UndoCallback } from "../core/move";
import { GMath } from "./gmath";

export class MCTSNode {
	readonly legal: Move[];
	readonly count: number;
	readonly state: GameState;
	readonly pov: number;
	readonly children: Map<number, MCTSNode> = new Map();
	private expanded: boolean;
	public P: number[];
	public N: number[];
	public W: number[];
	public Q: number[];

	constructor(state: GameState) {
		this.expanded = false;
		this.pov = getPovTribe(state).owner;
		this.legal = MoveGenerator.legal(state);
		this.count = this.legal.length;
		this.state = state;
		this.P = new Array(this.count).fill(0);
		this.N = Array(this.count).fill(0);
        this.W = Array(this.count).fill(0);
        this.Q = Array(this.count).fill(0);
	}

	getOrCreateChild(a: number, state: GameState): MCTSNode {
		let child = this.children.get(a);
		if(!child) {
			child = new MCTSNode(state);
		  	this.children.set(a, child);
		}
		return child;
	}
	
	isExpanded(): boolean {
		return this.expanded;
	}
	
	expand() {
		if(this.expanded) return;
		this.expanded = true;
		
		const epsFloor = 1e-6;
		const movePriorities = this.legal.map(move => scoreMovePriority(move));
        const sumPriorities = movePriorities.reduce((sum, priority) => sum + priority, 0);

		// This should never happen but just in case then the priorities are setup wrong
		if (sumPriorities === 0) {
			throw new Error("Sum of move priorities is 0");
            // // If all priorities are 0, use a uniform distribution
            // for(let a = 0; a < this.count; a++) {
            //     this.P[a] = 1 / this.count;
            // }
		}

		for (let a = 0; a < this.count; a++) {
			this.P[a] = Math.max(movePriorities[a] / sumPriorities, epsFloor);
		}

		const s = this.P.reduce((x, y) => x + y, 0);

		for (let a = 0; a < this.count; a++) this.P[a] /= s;
	}

	select(cPuct: number): number {
		const sumN = this.N.reduce((s, x) => s + x, 0);
		const sqrtSumN = Math.sqrt(Math.max(1, sumN)); // avoid 0
		// const epsilon = 1e-9;
		let bestScore = -Infinity;
		let bestAction = -1;
		for (let a = 0; a < this.count; a++) {
			const move = this.legal[a];
			// Skip the end turn since its 100% a complete waste to end the turn after it started
			if (move.moveType === MoveType.EndTurn && this.state.settings._recentMoves.length === 0) {
				continue;
			}

			const u = this.Q[a] + cPuct * this.P[a] * (sqrtSumN / (1 + this.N[a]));
			if (
				u > bestScore || 
				(u === bestScore && this.N[a] === 0 && bestAction !== -1 && this.N[bestAction] !== 0)
			) {
				bestScore = u;
				bestAction = a;
			}
		}
		if (bestAction === -1) {
			// if all failed return the move with the highest Q
			// return this.Q.indexOf(Math.max(...this.Q));
			// or a random move
			return Math.random() * this.count | 0;
		}
		return bestAction;
	}
	
	backpropagate(path: [MCTSNode, number][], value: number, rootPov: number) {
		// TODO assuming now its always a 1v1
		for (let i = path.length - 1; i >= 0; i--) {
			const [node, a] = path[i];
			node.N[a] += 1;
			node.W[a] += node.pov === rootPov? value : -value;
			// node.W[a] += this.state.settings._pov === rootPov? value : -value;
			node.Q[a] = node.W[a] / node.N[a];
		}
	}
	
	distribution(T: number = 1.0): number[] {
		let counts = [];
		for(let a = 0; a < this.count; a++) {
			counts.push(this.N[a] ?? 0)
		}
		if(T === 0) {
			const best = counts.indexOf(Math.max(...counts));
			return counts.map((_, i) => (i === best ? 1 : 0));
		}
		counts = counts.map(c => Math.pow(c, 1 / T));
		const sum = counts.reduce((s, x) => s + x, 0);
		if(sum === 0) return Array(this.count).fill(1 / this.count);
		return counts.map(x => x / sum);
	}
}

let largestTurn = 0;

export class MCTS {
	private game: Game;
	private cPuct: number;
	private dirichlet: boolean;
	private stopTurn: number;
	
	/**
	 * @param game The game class in use
	 * @param cPuct
	 * @param dirichlet 
	 * @param turnsAhead Stops after X turns have been played by the pov tribe
	 */
	constructor(game: Game, cPuct: number, dirichlet = false, maxTurnsAhead=3) {
		this.game = game;
		this.cPuct = cPuct;
		this.dirichlet = dirichlet;
		this.stopTurn = maxTurnsAhead + game.state.settings._turn;
	}

	private simulate(root: MCTSNode, game: Game): { 
		path: Array<[MCTSNode, number]>;
		value: number;
	} {
		const path: [MCTSNode, number][] = [];
		let currentNode = root;
		let value: number = 0;
		let undos: UndoCallback[] = [];
		
		while (currentNode.isExpanded() && !isGameOver(game.state) && this.stopTurn > game.state.settings._turn) {
			// console.log(`[MCTS] legal: ${(currentNode as any).legal.length}`);
			let moveIndex = currentNode.select(this.cPuct);
			// console.log(`[MCTS] playing: ${moveIndex} -> ${(currentNode as any).legal[moveIndex].stringify(game.state, game.state)}`);
			
			const old = game.cloneState();
			const [played, undo] = game.playMove(moveIndex)!;
			// console.log('played', played.stringify(old, game.state));
			
			// if (played.getType<RewardType>() === RewardType.Workshop) {
			// 	console.log('workshop!');
			// }

			undos.unshift(undo);
			
			const child = currentNode.getOrCreateChild(moveIndex, game.state);

			path.push([currentNode, moveIndex]);

			currentNode = child;
		}
		
		if (isGameOver(game.state)) {
			isGameWon(game.state) && console.log("[MCTS] victory detected! woah!");
			value = isGameWon(game.state)? 1 : isGameLost(game.state)? -1 : 0;
		}
		else {
			currentNode.expand();
			const [ eco, army, finalScore ] = evaluateState(game);
			value = finalScore;
		}

		// largestTurn = game.state.settings._turn;

		undos.forEach(x => x());

		// magnify the tiny evalutaion score
		// value = Math.max(-1, Math.min(1, value / 0.005));

		return { path, value };
	}

	public search(nSims: number): MCTSNode {
        const root = new MCTSNode(this.game.state);
		root.expand();

		// let largestTurns = [];

		const count = MoveGenerator.legal(this.game.state).length;
		
        if(this.dirichlet) {
			let stageMult = calculateStageValue(this.game.state);
			// TODO Use the strongest multiplier?
			const alpha = 0.3 * (count / count * 2) * (1 + (stageMult[1] > stageMult[0]? stageMult[1] : stageMult[0]));
			const eps = 0.25;
            const noise = GMath.dirichlet(alpha, count);
            for (let a = 0; a < count; a++) {
                root.P[a] = (1 - eps) * root.P[a] + eps * noise[a];
            }
        }

		for(let i = 0; i < nSims; i++) {
			// console.log(`\n[MCTS] simulating: ${i}/${nSims}`);
			const { path, value } = this.simulate(root, this.game);
			// largestTurns.push(largestTurn);
			root.backpropagate(path, value, root.pov);
		}

		console.log("[MCTS] Root stats:");
		for (let a = 0; a < root.count; a++) {
			const move = root.legal[a];
			console.log(`a=${a} move=${move.stringify(this.game.state)} P=${root.P[a].toFixed(4)} N=${root.N[a]} Q=${root.Q[a].toFixed(4)} U=${root.Q[a].toFixed(6)}`);
		}

		// console.log(`[MCTS] minmax turn: ${Math.min(...largestTurns)}/${Math.max(...largestTurns)}`);
		
		return root;
	}
}
