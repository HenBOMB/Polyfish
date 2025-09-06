import path from "path";
import { getPovTribe, isGameOver, isGameWon, isGameLost } from "../../core/functions";
import Move, { UndoCallback } from "../../core/move";
import { MoveGenerator } from "../../core/moves";
import { GameState } from "../../core/states";
import { MoveType } from "../../core/types";
import Game from "../../game";
import { scoreMovePriority, evaluateState, calculateStageValue } from "../eval";
import { GMath } from "../gmath";
import { Worker } from 'worker_threads';

export class MCTSNode {
	readonly legal: Move[];
	readonly count: number;
	readonly state: GameState;
	readonly pov: number;
	readonly legalGen: (state: GameState) => Move[];
	readonly children: Map<number, MCTSNode> = new Map();
	private expanded: boolean;
	public P: number[];
	public N: number[];
	public W: number[];
	public Q: number[];
	
	constructor(state: GameState, legalGen: (state: GameState) => Move[]) {
		this.expanded = false;
		this.pov = getPovTribe(state).owner;
		this.legal = legalGen(state);
		this.count = this.legal.length;
		this.state = state;
		this.legalGen = legalGen;
		this.P = new Array(this.count).fill(0);
		this.N = Array(this.count).fill(0);
		this.W = Array(this.count).fill(0);
		this.Q = Array(this.count).fill(0);
	}
	
	getOrCreateChild(a: number, state: GameState): MCTSNode {
		let child = this.children.get(a);
		if(!child) {
			child = new MCTSNode(state, this.legalGen);
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

export class MCTS {
	readonly cPuct: number;
	readonly dirichlet: boolean;
	readonly stopTurn: number;
	readonly numThreads: number = 0;
	private game: Game;
	private workers: Worker[] = [];
	
	constructor(
		game: Game, 
		cPuct=1.0, 
		dirichlet=false, 
		maxTurnsAhead=3,
		numThreads=1
	) {
		this.game = game;
		this.cPuct = cPuct;
		this.dirichlet = dirichlet;
		this.stopTurn = maxTurnsAhead + game.state.settings._turn;
		this.numThreads = numThreads;
		this.workers = [];
        
	}
	
	private simulate_old(root: MCTSNode, game: Game): { 
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
			
			// const old = game.cloneState();
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
		
		undos.forEach(x => x());
		
		// magnify the evalutaion score
		// value = Math.max(-1, Math.min(1, value / 0.01));
		
		return { path, value };
	}
	
	public search_old(
		nSims: number, 
		debug=false,
		legalGen=(state: GameState) => MoveGenerator.legal(state)
	): MCTSNode {
		const root = new MCTSNode(this.game.state, legalGen);
		root.expand();
		
		if(this.dirichlet) {
			const count = legalGen(this.game.state).length;
			
			let stageMult = calculateStageValue(this.game.state);
			
			// TODO Use the strongest multiplier?
			const alpha = 0.3 * (count / count * 2) * (1 + (stageMult[1] > stageMult[0]? stageMult[1] : stageMult[0]));
			const eps = 0.25;
			const noise = GMath.dirichlet(alpha, count);
			
			for (let a = 0; a < count; a++) {
				root.P[a] = (1 - eps) * root.P[a] + eps * noise[a];
			}
		}
		
		for (let i = 0; i < nSims; i++) {
			const { path, value } = this.simulate_old(root, this.game);
			root.backpropagate(path, value, root.pov);
		}
		
		if (debug) {
			console.log("[MCTS] Root stats:");
			for (let a = 0; a < root.count; a++) {
				const move = root.legal[a];
				console.log(`N=${root.N[a]}\tQ=${root.Q[a].toFixed(4)}   [${move.stringify(this.game.state)}]`);
			}
		}
		
		return root;
	}

	public async prepare(): Promise<void> {
		const workerPromises = [];

		for (let i = 0; i < this.numThreads; i++) {
            const worker = new Worker(path.resolve(__dirname, 'mcts-worker.ts'), {
				execArgv: ['-r', 'ts-node/register']
			});

            this.workers.push(worker);

            const promise = new Promise<{ N: number[]; W: number[] }>((resolve, reject) => {
                worker.once('message', resolve);
                worker.postMessage({
                    ok: true,
                });
            });
			
            workerPromises.push(promise);
        }

        await Promise.all(workerPromises);
	}

	public destroy(): void {
        for (const worker of this.workers) {
            worker.terminate();
        }
    }

	public async search(
		nSims: number, 
		debug=false,
		legalGen=(state: GameState) => MoveGenerator.legal(state)
	): Promise<MCTSNode> {
		const root = new MCTSNode(this.game.state, legalGen);
		root.expand();
		
		if(this.dirichlet) {
			const count = legalGen(this.game.state).length;
			let stageMult = calculateStageValue(this.game.state);
			const alpha = 0.3 * (count / count * 2) * (1 + (stageMult[1] > stageMult[0]? stageMult[1] : stageMult[0]));
			const eps = 0.25;
			const noise = GMath.dirichlet(alpha, count);
			
			for (let a = 0; a < count; a++) {
				root.P[a] = (1 - eps) * root.P[a] + eps * noise[a];
			}
		}

        const simsPerWorker = Math.ceil(nSims / this.numThreads);
        const workerPromises: Promise<{ N: number[]; W: number[] }>[] = [];
		const gameJson = Game.serialize(this.game);

		for (const worker of this.workers) {
            const promise = new Promise<{ N: number[]; W: number[] }>((resolve, reject) => {
                worker.once('message', (res) => {
					resolve(res);
					// to prevent memory leakage and event listener overflow (+14/15?)
					worker.removeAllListeners('error')
					worker.removeAllListeners('exit')
				});
                worker.once('error', reject);
                worker.once('exit', (code) => {
                    if (code !== 0) {
                        reject(new Error(`Worker stopped with exit code ${code}`));
                    }
                });
                worker.postMessage({
                    nSims: simsPerWorker,
                    gameJson: gameJson,
                    cPuct: this.cPuct,
                    stopTurn: this.stopTurn,
                });
            });
            workerPromises.push(promise);
        }

        const results = await Promise.all(workerPromises);
		
        for (const result of results) {
            for (let i = 0; i < root.N.length; i++) {
                root.N[i] += result.N[i];
                root.W[i] += result.W[i];
                root.Q[i] = root.N[i] > 0 ? root.W[i] / root.N[i] : 0;
            }
        }
		
		return root;
	}
}