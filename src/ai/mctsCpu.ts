import AIState from "../aistate";
import { getPovTribe, isGameLost, isGameOver, isGameWon } from "../core/functions";
import { MoveType } from "../core/types";
import { MoveGenerator, Prediction } from "../core/moves";
import { GameSettings, GameState } from "../core/states";
import { MIN_PLAYED_MOVES, calculateStageValue, evaluateState, lerpViaGameStage, scoreMovePriority } from "./eval";
import Game from "../game";
import Move, { UndoCallback } from "../core/move";
import { GMath } from "./gmath";

export class MCTSNode {
	private legal: Move[];
	private recent: number;
	readonly pov: number;
	public P: number[];
	public N: number[];
	public W: number[];
	public Q: number[];
	private expanded: boolean;
	private children: Map<number, MCTSNode> = new Map();
	readonly count: number;

	constructor(state: GameState) {
		this.expanded = false;
		this.pov = getPovTribe(state).owner;
		this.legal = MoveGenerator.legal(state);
		this.count = this.legal.length;
		this.recent = state.settings._recentMoves.length;
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

		const movePriorities = this.legal.map(move => scoreMovePriority(move));
        const sumPriorities = movePriorities.reduce((sum, priority) => sum + priority, 0);

		if (sumPriorities === 0) {
            // If all priorities are 0, use a uniform distribution
            for(let a = 0; a < this.count; a++) {
                this.P[a] = 1 / this.count;
            }
        } else {
            // Normalize the priorities to create probabilities
            for(let a = 0; a < this.count; a++) {
                this.P[a] = movePriorities[a] / sumPriorities;
            }
        }

		for(let a = 0; a < this.count; a++) {
			this.N[a] = this.W[a] = this.Q[a] = 0;
		}
	}

	select(cPuct: number): number {
		let totalN = 0;
		for(let a = 0; a < this.count; a++) {
			totalN += this.N[a];
		}
		totalN = Math.sqrt(totalN + 1);
		let bestScore = -Infinity;
		let bestAction = 0;
		for(let a = 0; a < this.count; a++) {
			// const min = lerpViaGameStage(state, MIN_PLAYED_MOVES);
			// EndTurn is always at move index 0
			// Prevent losing too early, at least 1 turn
			if(!a && !this.recent && this.count) {
				continue;
			}
			const u = this.Q[a] + cPuct * this.P[a] * totalN / (1 + this.N[a]);
			if(u > bestScore) {
				bestScore = u;
				bestAction = a;
			}
		}
		return bestAction;
	}
	
	backpropagate(path: [MCTSNode, number][], value: number, rootPov: number) {
		// TODO assuming now its always a 1v1
		for (let i = path.length - 1; i >= 0; i--) {
			const [node, a] = path[i];
			node.N[a] += 1;
			node.W[a] += node.pov === rootPov? value : -value;
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

export class MCTS {
	private game: Game;
	private cPuct: number;
	private dirichlet: boolean;
	
	constructor(game: Game, cPuct: number, dirichlet = false) {
		this.game = game;
		this.cPuct = cPuct;
		this.dirichlet = dirichlet;
	}

	private simulate(root: MCTSNode, game: Game): { 
		path: Array<[MCTSNode, number]>;
		value: number;
	} {
		const path: [MCTSNode, number][] = [];
		let currentNode = root;
		let value: number = 0;
		let undos: UndoCallback[] = [];
		
		while (currentNode.isExpanded() && !isGameOver(game.state)) {
			// console.log(`[MCTS] legal: ${(currentNode as any).legal.length}`);
			let moveIndex = currentNode.select(this.cPuct);
			// console.log(`[MCTS] playing: ${moveIndex} -> ${(currentNode as any).legal[moveIndex].stringify(game.state, game.state)}`);
			
			const [_, undo] = game.playMove(moveIndex)!;
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

		undos.forEach(x => x());

		return { path, value };
	}

	public search(nSims: number): MCTSNode {
        const root = new MCTSNode(this.game.state);
		root.expand();

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
			root.backpropagate(path, value, root.pov);
		}
	
		return root;
	}
}
