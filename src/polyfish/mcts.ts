import { appendFileSync } from "fs";
import AIState, { MODEL_CONFIG } from "../aistate";
import { cloneState, getPovTribe, isGameOver } from "../core/functions";
import GameLoader from "../core/gameloader";
import Move, { MoveType } from "../core/move";
import { generateAllMoves } from "../core/moves";
import { GameSettings, GameState } from "../core/states";
import { TribeType } from "../core/types";
import Game from "../game";

export type Prediction = { pi: number[]; v: number };

// var transpositionCache: Map<GameState, Prediction> = new Map();

export function sampleFromDistribution(dist: number[]): number {
	const r = Math.random();
	let cum = 0;
	for (let i = 0; i < dist.length; i++) {
		cum += dist[i];
		if (r < cum) {
			return i;
		}
	}
	return dist.length - 1;
}

export class MCTS {
	private iState: GameState;
	private cPuct: number;
	private rootPov: number;
	private oldPot: number;
	private gamma: number;
	private predict: (state: GameState) => Promise<Prediction>;
	
	constructor(state: GameState, predict: (state: GameState) => Promise<Prediction>, cPuct: number, gamma: number) {
		this.iState = state;
		this.cPuct = cPuct;
		this.predict = predict;
		this.gamma = gamma;
		this.rootPov = getPovTribe(state).owner;
		this.oldPot = AIState.calculatePotential(state);
	}

	private async simulate(root: MCTSNode): Promise<{ path: Array<[MCTSNode, number]>; value: number }> {
		const game = new Game();
		game.load(this.iState);
		const path: [MCTSNode, number][] = [];

		let node = root;
		
		while (node.isExpanded()) {
			const moveIndex = node.select(this.cPuct);
			path.push([node, moveIndex]);
			game.playMove(moveIndex);
			node = new MCTSNode(game.state);
		}
		
		let value = 0;

		if(isGameOver(game.state)) {
			const moveReward = AIState.calculateReward(this.iState, game.state);
			const newPot = AIState.calculatePotential(game.state)
			value += moveReward + this.gamma * newPot - this.oldPot;
		}
		else {
			value = await node.expand(this.predict);
		}

		return { path, value };
	}
	
	async search(nSims: number): Promise<MCTSNode> {
		const batchSize = 5;
		const root = new MCTSNode(this.iState);

		await root.expand(this.predict);
	
		for (let start = 0; start < nSims; start += batchSize) {
			const end = Math.min(start + batchSize, nSims);
			const batchPromises = [];
			for (let i = start; i < end; i++) {
				batchPromises.push(this.simulate(root));
			}
			const results = await Promise.all(batchPromises);
			for (const { path, value } of results) {
				root.backpropagate(path, value, this.rootPov);
			}
		}
	
		return root;
	}
}

export class MCTSNode {
	public count: number;
	private pov: number;
	private state: GameState;
	private P: Record<number, number> = {};
	private N: Record<number, number> = {};
	private W: Record<number, number> = {};
	private Q: Record<number, number> = {};
	
	constructor(state: GameState) {
		this.pov = getPovTribe(state).owner;
		this.state = cloneState(state);
		this.count = generateAllMoves(state).length;
	}
	
	isExpanded(): boolean {
		return Object.keys(this.P).length > 0;
	}
	
	async expand(predict: (state: GameState) => Promise<Prediction>): Promise<number> {
		const out = await predict(this.state);
		out.pi.forEach((p, a) => {
			if(a < this.count) {
				this.P[a] = p;
				this.N[a] = 0;
				this.W[a] = 0;
				this.Q[a] = 0;
			}
		});
		return out.v;
	}

	select(cPuct: number): number {
		const totalN = Object.values(this.N).reduce((acc, v) => acc + v, 0) + 1;
		let bestScore = -Infinity;
		let bestAction = 0;
		for (const [aStr, p] of Object.entries(this.P)) {
			const a = Number(aStr);
			// Prevent ending the turn when no moves are played (end turn is always action at index 0)
			if(a == 0 && this.state.settings._playedMoves.length < 3 && this.count > 3) {
				// console.log('no!');
				// Skips this turn
				continue;
			}
			else {
				// console.log('ok!');
			}
			const q = this.Q[a];
			const u = q + cPuct * p * Math.sqrt(totalN) / (1 + this.N[a]);
			if (u > bestScore) {
				bestScore = u;
				bestAction = a;
			}
		}
		return bestAction;
	}
	
	backpropagate(
		path: [MCTSNode, number][],
		value: number,
		rootPov: number
	): void {
		for (let i = path.length - 1; i >= 0; i--) {
			const [node, action] = path[i];
			node.N[action] += 1;
			// If current player is not root player, then its some other opponent
			// If its negative then keep it since it doesnt matter if its an opponent or not
			node.W[action] += value < 0? value : node.pov == rootPov? value : -value;
			node.Q[action] = node.W[action] / node.N[action];
		}
	}
	
	distribution(temperature: number = 1.0): number[] {
		const moves = Object.keys(this.P).sort();
		let counts = moves.map(move => this.N[Number(move)] ?? 0);
		if (temperature === 0) {
			const best = counts.indexOf(Math.max(...counts));
			const probs = new Array(counts.length).fill(0);
			probs[best] = 1.0;
			return probs;
		}
		counts = counts.map(count => Math.pow(count, 1 / temperature));
		const sum = counts.reduce((a, b) => a + b, 0);
		return counts.map(count => count / sum);
	}
}

/**
 * Run nGames of self-play, using MCTS to produce π at each step,
 * then return a flattened array of training triples: [obs, π, G].
 */
export async function SelfPlay(
	predict: (state: GameState) => Promise<Prediction>,
	nGames: number,
	nSims: number,
	temperature: number,
	cPuct: number,
	gamma: number,
	deterministic = false,
	settings: GameSettings,
): Promise<Array<[any, number[], number]>> {
	const trainingData: Array<[any, number[], number]> = [];
	const loader = new GameLoader(settings);

	for (let g = 0; g < nGames; g++) {
		// load a fresh random game
		const game = new Game();
		game.load(await loader.loadRandom());

		// For this one game, we store triples of (obs, π, r)
		const episode: Array<{ obs: any; pi: number[]; r: number }> = [];
		const povTribe = getPovTribe(game);
		console.log('\n' + TribeType[povTribe.tribeType].toLowerCase(), povTribe._stars);

		// Keep going until terminal
		while (!isGameOver(game.state)) {
			const obs = AIState.extract(game.state);

			let moveIndex: number = 0;
			let fullProbs: number[] = [];
			let pseudoLegalMoves = generateAllMoves(game.state);

			// If there's literally only one legal move (EndTurn), skip simulating pointless turns
			if(pseudoLegalMoves.length == 1) {
				moveIndex = 0;
				fullProbs = new Array<number>(MODEL_CONFIG.max_actions).fill(0);
				fullProbs[0] = 1;
			}
			else {
				// let start = Date.now();
				const root = await new MCTS(game.state, predict, cPuct, gamma).search(nSims);
				const probs = root.distribution(temperature);
				// console.log(`took: ${Date.now() - start}ms`);
				fullProbs = new Array<number>(MODEL_CONFIG.max_actions).fill(0);
				probs.forEach((p, idx) => { fullProbs[idx] = p; });
				moveIndex = deterministic
					? probs.indexOf(Math.max(...probs))
					: sampleFromDistribution(probs);
			}

			// Step the game and compute the *immediate* shaped reward
			const oldPot = AIState.calculatePotential(game.state);
			const movesPlayed = game.playMove(moveIndex)!;

			// Make sure move was legal
			if(movesPlayed) {
				const isEnd = movesPlayed[0].moveType === MoveType.EndTurn;
				const moveReward = isEnd ? 0 : AIState.calculateReward(game.stateBefore, game.state, movesPlayed[0]);
				const newPot     = isEnd ? 0 : AIState.calculatePotential(game.state);
				const r_t        = isEnd ? 0 : (moveReward + gamma * newPot - oldPot);
				// Verbose output for debugging
				if(!isEnd) {
					for(const move of movesPlayed) {
						console.log(move.stringify());
					}
					if(moveReward || newPot) {
						console.log(Number(moveReward.toFixed(3)), '', Number(newPot.toFixed(3)), '', Number(r_t.toFixed(3)));
					}
				}
				else {
					const povTribe = getPovTribe(game);
					console.log('\n' + TribeType[povTribe.tribeType].toLowerCase(), povTribe._stars);
				}
				episode.push({ obs, pi: fullProbs, r: r_t });
			}
			else {
				episode.push({ obs, pi: fullProbs, r: -0.1 });
			}
		}

		// Game is over, now compute discounted returns G_t for each step
		// let G = 0;
		// for (let t = episode.length - 1; t >= 0; t--) {
		// 	G = episode[t].r + gamma * G;
		// 	// Optionally normalize or clip G here, e.g. G = Math.max(-1, Math.min(1, G));
		// 	trainingData.push([episode[t].obs, episode[t].pi, G]);
		// }

		// Compute raw G
		let G = 0;
		const returns = new Array(episode.length);
		for (let t = episode.length - 1; t >= 0; t--) {
			G = episode[t].r + gamma * G;
			returns[t] = G;
		}

		// Normalize
		const R = 20;
		for (let t = 0; t < returns.length; t++) {
			returns[t] = Math.max(-R, Math.min(R, returns[t])) / R;
			trainingData.push([ episode[t].obs, episode[t].pi, returns[t] ]);
		}

		const maxG = Math.max(...returns);
		const minG = Math.min(...returns);
		const avrG = returns.reduce((acc, r) => acc + r, 0) / returns.length;
		// console.log(Number(G.toFixed(3)), Number(minG.toFixed(3)), Number(maxG.toFixed(3)), Number(avrG.toFixed(3)));
		appendFileSync('training.log', `G: ${minG.toFixed(3)}, ${maxG.toFixed(3)}, ${avrG.toFixed(3)}\n`);
	}

	return trainingData;
}
