import AIState, { MODEL_CONFIG } from "../aistate";
import { getPovTribe, isGameLost, isGameOver, isGameWon } from "../core/functions";
import GameLoader, { STARTING_OWNER_ID } from "../core/gameloader";
import { MoveType } from "../core/move";
import { generateAllMoves } from "../core/moves";
import { GameSettings, GameState } from "../core/states";
import { TribeType } from "../core/types";
import { MIN_PLAYED_MOVES, lerpViaGameStage, SCORE_MOVE_PRIORITY } from "./eval";
import Game from "../game";
import { Logger } from "./logger";

export type Prediction = { pi: number[]; v: number };

// obs -> Command[]

// var transpositionCache: Map<Observation, Command[]> = new Map();

/**
 * Sample from a symmetric Dirichlet(α) distribution of given size.
 */
export function sampleDirichlet(alpha: number, size: number): number[] {
	function randGamma(k: number): number {
		if (k < 1) {
			const u = Math.random();
			return randGamma(k + 1) * Math.pow(u, 1 / k);
		}
		const d = k - 1 / 3;
		const c = 1 / Math.sqrt(9 * d);
		while (true) {
			let x: number, v: number;
			do {
				const u1 = Math.random(), u2 = Math.random();
				x = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
				v = 1 + c * x;
			} while (v <= 0);
			v = v * v * v;
			const u = Math.random();
			if (u < 1 - 0.0331 * x * x * x * x) return d * v;
			if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
		}
	}
  
	const xs = Array.from({ length: size }, () => randGamma(alpha));
	const sum = xs.reduce((a, b) => a + b, 0);
	return xs.map(x => x / sum);
}

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

export class MCTSNode {
	count: number;
	pov: number;
	P: Record<number, number> = {};
	N: Record<number, number> = {};
	W: Record<number, number> = {};
	Q: Record<number, number> = {};
	expanded = false;
	children: Map<number, MCTSNode> = new Map();

	// Keep track of which moves we’ve pruned (illegal)
	private pruned = new Set<number>();

	constructor(state: GameState) {
		this.pov = getPovTribe(state).owner;
		this.count = generateAllMoves(state).length;
	}

	pruneMove(a: number) {
		this.pruned.add(a);
		// Optionally shrink count or mark P[a]=0, etc.
		this.P[a] = 0;
	}

	getOrCreateChild(a: number, state: GameState): MCTSNode {
		if (!this.children.has(a)) {
		  this.children.set(a, new MCTSNode(state));
		}
		return this.children.get(a)!;
	}
	
	isExpanded(): boolean {
		return this.expanded;
	}
	
	expand(pred: Prediction) {
		if(this.expanded) return;
		this.expanded = true;
		// Slice off only the legal-prefix of pi, then renormalize
		const raw = pred.pi.slice(0, this.count);
		const sum = raw.reduce((s, x) => s + x, 0) || 1;
		for (let a = 0; a < this.count; a++) {
			this.P[a] = raw[a] / sum;
			this.N[a] = this.W[a] = this.Q[a] = 0;
		}
	}

	select(cPuct: number, state: GameState): number {
		let totalN = 0;
		this.count = generateAllMoves(state).length;
		for (let a = 0; a < this.count; a++) {
			totalN += this.N[a];
		}
		totalN = Math.sqrt(totalN + 1);
		let bestScore = -Infinity;
		let bestAction = 0;
		for (let a = 0; a < this.count; a++) {
			if(this.pruned.has(a)) continue;

			const min = lerpViaGameStage(state, MIN_PLAYED_MOVES);

			// Prevent losing too early
			if(a === 0 && state.settings._recentMoves.length < min && this.count > min) {
				continue;
			}

			const u = this.Q[a] + cPuct * this.P[a] * totalN / (1 + this.N[a]);
			if (u > bestScore) {
				bestScore = u;
				bestAction = a;
			}
		}
		return bestAction;
	}
	
	backpropagate(path: [MCTSNode, number][], value: number, rootPov: number) {
		// For now its just 1v1
		for (let i = path.length - 1; i >= 0; i--) {
			const [node, a] = path[i];
			node.N[a] += 1;
			node.W[a] += (node.pov === rootPov? 1 : -1) * value;
			node.Q[a] = node.W[a] / node.N[a];
		}
	}
	
	distribution(temperature: number = 1.0): number[] {
		let counts = [];
		for (let a = 0; a < this.count; a++) {
			counts.push(this.N[a] ?? 0)
		}
		if (temperature === 0) {
			const best = counts.indexOf(Math.max(...counts));
			return counts.map((_, i) => (i === best ? 1 : 0));
		}
		counts = counts.map(c => Math.pow(c, 1 / temperature));
		const sum = counts.reduce((s, x) => s + x, 0);
		if (sum === 0) return Array(this.count).fill(1 / this.count);
		return counts.map(x => x / sum);
	}
}

export class MCTS {
	private iState: GameState;
	private cPuct: number;
	private gamma: number;
	private dirichlet: boolean;
	private predict: (state: GameState) => Promise<Prediction>;
	private rollouts: number;
	
	constructor(state: GameState, predict: (state: GameState) => Promise<Prediction>, cPuct: number, gamma: number, dirichlet: boolean, rollouts: number) {
		this.iState = state;
		this.cPuct = cPuct;
		this.predict = predict;
		this.gamma = gamma;
		this.dirichlet = dirichlet;
		this.rollouts = rollouts;
	}

	private async simulate(root: MCTSNode, game: Game): Promise<{ path: Array<[MCTSNode, number]>; value: number }> {
		const path: [MCTSNode, number][] = [];
		let currentNode = root;
		let value: number = 0;
		let prevPotential = AIState.calculatePotential(game.state);
		
		while (currentNode.isExpanded() && !isGameOver(game.state)) {
			let moveIndex = currentNode.select(this.cPuct, game.state);

			// If illegal, prune it permanently from this node
			if (!game.playMove(moveIndex)) {
				currentNode.pruneMove(moveIndex);
				continue;
			}

			const child = currentNode.getOrCreateChild(moveIndex, game.state);
			path.push([currentNode, moveIndex]);
			currentNode = child;
		}
		
		if(isGameOver(game.state)) {
			value = isGameWon(game.state)? 1.0 : isGameLost(game.state)? -1.0 : 0.0;
		}
		else if (currentNode.isExpanded()) {
			// If we somehow arrived at an already-expanded leaf, do a quick rollout
			value = this.rollout(game.deepClone(), prevPotential);
		} 
		else {
			const prediction = await this.predict(game.state);
			currentNode.expand(prediction);
			value = prediction.v;
		}

		return { path, value };
	}

	// Plays random heuristic moves until either the turn ends or hit a max depth
	private rollout(game: Game, prevPot: number): number {
		let depth = 0;
	  
		while (!isGameOver(game.state) && depth < this.rollouts) {
			const min = lerpViaGameStage(game.state, MIN_PLAYED_MOVES);
			const weights = generateAllMoves(game.state).map(x => {
				switch (x.moveType) {
					case MoveType.EndTurn:
					  	return game.state.settings._recentMoves.length >= min? 10 : 0.1;
					default:
					  	return SCORE_MOVE_PRIORITY(x);
				}
			});
		
			// Sample a move index according to these weights
			const totalW = weights.reduce((a, b) => a + b, 0);
			let pick = Math.random() * totalW;
			let moveIdx = 0;
			for (; moveIdx < weights.length; moveIdx++) {
				pick -= weights[moveIdx];
				if (pick <= 0) break;
		  	}
	  
			const moves = game.playMove(moveIdx);

		  	if (!moves) continue;
	  
		  	depth++;
		}
	  
		// Terminal override
		if (isGameOver(game.state)) {
			return isGameWon(game.state) ? 1 : isGameLost(game.state) ? -1 : 0;
		}

		return 0;

		// // Shaped reward from potential delta
		// const newPot = AIState.calculatePotential(game.state);
		// const shapedR = newPot - prevPot;

		// return shapedR;
	}
	
	async search(nSims: number): Promise<MCTSNode> {
		const batchSize = 3;
        const root = new MCTSNode(this.iState);
        const rootGame = new Game();
        rootGame.load(this.iState);
		root.expand(await this.predict(this.iState));

		root.count;
        if (this.dirichlet) {
			// Derive a custom alpha table based on how many moves there are
			const table = [0, 0.2, 0.3];
			const alpha = 0.3 * (root.count / MODEL_CONFIG.max_actions) * (1 + lerpViaGameStage(this.iState, table));
			const eps = 0.25;
            const moves = Array.from({ length: root.count }, (_, i) => i);
            const priors = moves.map(a => root.P[a]);
            const noise = sampleDirichlet(alpha, moves.length);
            for (let i = 0; i < moves.length; i++) {
                const a = moves[i];
                root.P[a] = (1 - eps) * priors[i] + eps * noise[i];
            }
        }

        for (let start = 0; start < nSims; start += batchSize) {
            const end = Math.min(start + batchSize, nSims);
            const batchPromises = [];
            for (let i = start; i < end; i++) {
                batchPromises.push(this.simulate(root, rootGame.deepClone()));
            }
            const results = await Promise.all(batchPromises);
            for (const { path, value } of results) {
                root.backpropagate(path, value, root.pov);
            }
        }
	
		return root;
	}
}

let winRate: number[] = [];

export async function SelfPlay(
	predict: (state: GameState) => Promise<Prediction>,
	nGames: number,
	nSims: number,
	temperature: number,
	cPuct: number,
	gamma: number,
	deterministic: boolean,
	dirichlet: boolean,
	rollouts: number,
	settings: GameSettings,
): Promise<Array<[any, number[], number]>> {
	const trainingData: Array<[any, number[], number]> = [];
	const loader = new GameLoader(settings);

	for (let i = 0; i < nGames; i++) {
		// load a fresh random game
		const game = new Game();
		game.load(await loader.loadRandom());

		// For this one game, we store triples of (obs, π, r)
		const episode: Array<{ obs: any; pi: number[]; r: number; pov: number }> = [];
		// Logger.logPlay(getPovTribe(game), 0, ...[{ stringify: () => 'start turn' }] as any);

		// Keep going until terminal
		while (!isGameOver(game.state)) {
			const obs = AIState.extract(game.state);
			const pi: number[] = new Array<number>(MODEL_CONFIG.max_actions).fill(0);
			const pseudoLegalMoves = generateAllMoves(game.state);
			const povTribe = getPovTribe(game);

			let moveIndex: number = 0;
			
			// Skip simulating if there are no more turns (only EndTurn remains)
			if(pseudoLegalMoves.length == 1) {
				moveIndex = 0;
				pi[0] = 1;
			}
			else {
				// console.time('search');
				const root = await new MCTS(game.state, predict, cPuct, gamma, dirichlet, rollouts).search(nSims);
				// console.timeEnd('search');
				const probs = root.distribution(temperature);
				probs.forEach((p, idx) => { pi[idx] = p; });
				moveIndex = deterministic
					? probs.indexOf(Math.max(...probs))
					: sampleFromDistribution(probs);
			}
			
			const movesPlayed = game.playMove(moveIndex)!;
			
			// Make sure move was legal
			if(movesPlayed) {
				// const shaped = gamma * AIState.calculatePotential(game.state) - AIState.calculatePotential(game.stateBefore); 
				// const r_t = AIState.calculateReward(game, ...movesPlayed) + shaped;
				const r_t = isGameWon(game.state)? 1 : isGameLost(game.state)? -1 : 0;

				Logger.logPlay(povTribe, r_t, ...movesPlayed);
				// if(movesPlayed[0].moveType === MoveType.EndTurn) {
				// 	Logger.log(`${TribeType[povTribe.tribeType]}`);
				// }
				// else {
				// 	const str = movesPlayed.map(x => x.stringify()).join(', ');
				// 	// Logger.log(`${str} ${(await predict(game.state)).v.toFixed(4)}, ${r_t.toFixed(4)}, ${shaped.toFixed(4)}`);
				// 	Logger.log(`${str} (${(await predict(game.state)).v.toFixed(4)})`);
				// }
				episode.push({ obs, pi, r: r_t, pov: povTribe.owner });
			}
			else {
				Logger.illegal(pseudoLegalMoves[moveIndex].moveType, `FATAL - Move wasnt legal`);
			}
		}

		let G = 0;
		for (let t = episode.length - 1; t >= 0; t--) {
			const { obs, pi, pov, r } = episode[t];
			G = r + gamma * G;
			// In case the previous state's tribe's turn is not the current tribe's turn
			// STARTING_OWNER_ID = Hack for now, since entire simulator uses this and it never changes
			const signedG = pov === STARTING_OWNER_ID? G : -G;
			trainingData.push([obs, pi, signedG]);
		}	

		winRate.push(isGameWon(game.state) || isGameLost(game.state)? 1 : 0);

		Logger.log(`Finished game (${(i+1)}/${nGames})`);
		Logger.log(`state: ${isGameWon(game.state)? 'Won' :isGameLost(game.state)? 'Lost' : 'Truncated'}`);
		Logger.log(`turn: ${game.state.settings._turn}`);
	}
	
	Logger.log(`\nSelf play ended`);
	Logger.log(`collected: ${trainingData.length}`);
	Logger.log(`win rate: ${Math.floor((winRate.reduce((a, b) => a + b, 0) / winRate.length) * 100)}%\n`);

	return trainingData;
}
