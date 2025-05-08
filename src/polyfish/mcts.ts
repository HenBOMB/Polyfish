import AIState, { MODEL_CONFIG } from "../aistate";
import { getPovTribe, isGameLost, isGameOver, isGameWon } from "../core/functions";
import GameLoader, { STARTING_OWNER_ID } from "../core/gameloader";
import { MoveType } from "../core/types";
import { MoveGenerator } from "../core/moves";
import { GameSettings, GameState } from "../core/states";
import { MIN_PLAYED_MOVES, lerpViaGameStage, SCORE_MOVE_PRIORITY } from "./eval";
import Game from "../game";
import { Logger } from "./logger";
import { Opening } from "./opening";
import { sampleFromDistribution, sampleDirichlet } from "./util";
import { CaptureType } from "../core/types";

export type Prediction = { pi: number[]; v: number };

export class MCTSNode {
	count: number;
	recent: number;
	pov: number;
	P: number[] = new Array(/*MODEL_CONFIG.max_actions*/1).fill(0);
	N: number[] = new Array(/*MODEL_CONFIG.max_actions*/1).fill(0);
	W: number[] = new Array(/*MODEL_CONFIG.max_actions*/1).fill(0);
	Q: number[] = new Array(/*MODEL_CONFIG.max_actions*/1).fill(0);
	expanded: boolean;
	children: Map<number, MCTSNode> = new Map();
	pruned = new Set<number>();

	constructor(state: GameState) {
		this.expanded = false;
		this.pov = getPovTribe(state).owner;
		this.count = MoveGenerator.legal(state).length;
		this.recent = state.settings._recentMoves.length;
	}

	pruneMove(a: number) {
		this.pruned.add(a);
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

	select(cPuct: number): number {
		let totalN = 0;
		for (let a = 0; a < this.count; a++) {
			totalN += this.N[a];
		}
		totalN = Math.sqrt(totalN + 1);
		let bestScore = -Infinity;
		let bestAction = 0;
		for (let a = 0; a < this.count; a++) {
			if(this.pruned.has(a)) continue;

			// const min = lerpViaGameStage(state, MIN_PLAYED_MOVES);
			// Prevent losing too early, at least 1 turn
			if(a === 0 && this.recent < 1 && this.count > 1) {
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
	private state: GameState;
	private cPuct: number;
	private gamma: number;
	private dirichlet: boolean;
	private predict: (state: GameState) => Promise<Prediction>;
	private rollouts: number;
	
	constructor(state: GameState, predict: (state: GameState) => Promise<Prediction>, cPuct: number, gamma: number, dirichlet: boolean, rollouts: number) {
		this.state = state;
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
		
		while(currentNode.isExpanded() && !isGameOver(game.state)) {
			let moveIndex = currentNode.select(this.cPuct);
			const moves = game.playMove(moveIndex);
			if(!moves) {
				currentNode.pruneMove(moveIndex);
				break;
			}
			const child = currentNode.getOrCreateChild(moveIndex, game.state);
			path.push([currentNode, moveIndex]);
			currentNode = child;
		}
		
		if(isGameOver(game.state)) {
			isGameWon(game.state) && console.log("Simulator victory detected!");
			value = isGameWon(game.state)? 1.0 : isGameLost(game.state)? -1.0 : 0.0;
		}
		else {
			const prediction = await this.predict(game.state);
			currentNode.expand(prediction);
			value = prediction.v;
		}

		return { path, value };
	}

	private rollout(game: Game, prevPot: number): number {
		let depth = 0;
	  
		while (!isGameOver(game.state) && depth < this.rollouts) {
			const min = lerpViaGameStage(game.state, MIN_PLAYED_MOVES);
			const weights = MoveGenerator.legal(game.state).map(x => {
				switch (x.moveType) {
					case MoveType.EndTurn:
					  	return game.state.settings._recentMoves.length >= min? 10 : 0.1;
					default:
					  	return SCORE_MOVE_PRIORITY(x);
				}
			});
		
			const totalW = weights.reduce((a, b) => a + b, 0);
			let pick = Math.random() * totalW;
			let moveIdx = 0;
			for (; moveIdx < weights.length; moveIdx++) {
				pick -= weights[moveIdx];
				if (pick <= 0) break;
		  	}
	  
			const moves = game.playMove(moveIdx);

			depth++;

		  	if(!moves) continue;
		}
	  
		if (isGameOver(game.state)) {
			isGameWon(game.state) && console.log("Rollout victory detected!");
			return isGameWon(game.state) ? 1 : isGameLost(game.state) ? -1 : 0;
		}

		const newPot = AIState.calculatePotential(game.state);
		const shapedR = newPot - prevPot;
		return shapedR;
	}
	
	async search(nSims: number): Promise<MCTSNode> {
        const root = new MCTSNode(this.state);
        const rootGame = new Game();
        rootGame.load(this.state);
		root.expand(await this.predict(this.state));
		const count = MoveGenerator.legal(this.state).length;
		
        // if (this.dirichlet) {
		// 	const table = [0, 0.2, 0.3];
		// 	const alpha = 0.3 * (count / MODEL_CONFIG.max_actions) * (1 + lerpViaGameStage(this.state, table));
		// 	const eps = 0.25;
        //     const noise = sampleDirichlet(alpha, count);
        //     for (let a = 0; a < count; a++) {
        //         root.P[a] = (1 - eps) * root.P[a] + eps * noise[a];
        //     }
        // }

		// for (let i = 0; i < nSims; i++) {
		// 	const { path, value } = await this.simulate(root, rootGame.deepClone());
		// 	root.backpropagate(path, value, root.pov);
		// }
	
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
		const game = new Game();
		game.load(await loader.loadRandom());

		const episode: Array<{ obs: any; pi: number[]; r: number; pov: number }> = [];

		// Keep going until terminal
		while (!isGameOver(game.state)) {
			// TODO update
			// const obs = AIState.extract(game.state);
			// const pi: number[] = new Array<number>(MODEL_CONFIG.max_actions).fill(0);
			// const pseudoLegalMoves = MoveGenerator.legal(game.state);
			// const povTribe = getPovTribe(game);

			// let moveIndex: number = Opening.recommend(game.state, pseudoLegalMoves) || 0;
			
			// if(!moveIndex) {
			// 	// Skip simulating if there are no more turns (only EndTurn remains)
			// 	if(pseudoLegalMoves.length == 1) {
			// 		moveIndex = 0;
			// 		pi[0] = 1;
			// 	}
			// 	else {
			// 		// console.time('search');
			// 		const root = await new MCTS(game.state, predict, cPuct, gamma, dirichlet, rollouts).search(nSims);
			// 		// console.timeEnd('search');
			// 		const probs = root.distribution(temperature);
			// 		probs.forEach((p, idx) => { pi[idx] = p; });
			// 		moveIndex = deterministic
			// 			? probs.indexOf(Math.max(...probs))
			// 			: sampleFromDistribution(probs);
			// 	}
			// }
			
			// const result = game.playMove(moveIndex);
			
			// // Make sure move was legal
			// if(result) {
			// 	const [move, undo] = result;
			// 	const shaped = gamma * AIState.calculatePotential(game.state) - AIState.calculatePotential(game.stateBefore); 
			// 	const reward = AIState.calculateReward(game, move) + shaped;
			// 	const r_t = isGameWon(game.state)? 1 : 
			// 		isGameLost(game.state)? -1 : 
			// 		(reward * 0.25);
			// 	Logger.logPlay(game.stateBefore, game.state, [move], [(await predict(game.state)).v]);
			// 	episode.push({ obs, pi, r: r_t, pov: povTribe.owner });
			// }
			// else {
			// 	Logger.illegal(pseudoLegalMoves[moveIndex].moveType, `FATAL - Move wasnt legal`);
			// }
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

	Logger.log(`Self play ended`);
	Logger.log(`collected: ${trainingData.length}`);
	Logger.log(`win rate: ${Math.floor((winRate.reduce((a, b) => a + b, 0) / winRate.length) * 100)}%\n`);

	winRate = [];

	return trainingData;
}
