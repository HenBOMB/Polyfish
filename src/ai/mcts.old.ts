import AIState, { MODEL_CONFIG } from "../aistate";
import { getPovTribe, isGameLost, isGameOver, isGameWon } from "../core/functions";
import GameLoader, { STARTING_OWNER_ID } from "../core/gameloader";
import { MoveType } from "../core/types";
import { MoveGenerator, Prediction } from "../core/moves";
import { GameSettings, GameState } from "../core/states";
import { MIN_PLAYED_MOVES, lerpViaGameStage, scoreMovePriority } from "./eval";
import Game from "../game";
import { Logger } from "./logger";
import { Opening } from "./opening";
import { sampleFromDistribution, sampleDirichlet } from "./util";
import { CaptureType } from "../core/types";
import Move, { UndoCallback } from "../core/move";

interface ObsDict { 
	map: number[][][]; 
	player: number[];
}

interface TargetPoliciesDict {
    pi_action?: number[];
    pi_source?: number[];
    pi_target?: number[];
    pi_struct?: number[];
    pi_unit?: number[];
    pi_skill?: number[];
    pi_tech?: number[];
    pi_reward?: number;
}

interface TargetValuesDict {
    v_win: number;
    v_econ?: number;
    v_mil?: number;
}

type TrainingSample = [ObsDict, TargetPoliciesDict, TargetValuesDict];

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
	
	expand(state: GameState, pred: Prediction) {
		if(this.expanded) return;
		this.expanded = true;
		// [move, probability, moveIndex]
		const result = MoveGenerator.fromPrediction(state, pred, this.legal);
		const sum = result.reduce((s: any, x: any) => s[1] + x[1], [null, 0]) || 1;
		for(let a = 0; a < this.count; a++) {
			this.P[a] = result[a][1] / sum;
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
		// ? For now its just 1v1
		for (let i = path.length - 1; i >= 0; i--) {
			const [node, a] = path[i];
			node.N[a] += 1;
			node.W[a] += (node.pov === rootPov? 1 : -1) * value;
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

	private async simulate(root: MCTSNode, game: Game): Promise<{ 
		path: Array<[MCTSNode, number]>;
		v_win: number;
	}> {
		const path: [MCTSNode, number][] = [];
		let currentNode = root;
		let v_win: number = 0;
		let undos: UndoCallback[] = [];
		
		while(currentNode.isExpanded() && !isGameOver(game.state)) {
			let moveIndex = currentNode.select(this.cPuct);
			const [_, undo] = game.playMove(moveIndex)!;
			undos.unshift(undo);
			const child = currentNode.getOrCreateChild(moveIndex, game.state);
			path.push([currentNode, moveIndex]);
			currentNode = child;
		}
		
		if(isGameOver(game.state)) {
			isGameWon(game.state) && console.log("Simulator victory detected!");
			v_win = isGameWon(game.state)? 1 : isGameLost(game.state)? -1 : 0;
		}
		else {
			const prediction = await this.predict(game.state);
			currentNode.expand(game.state, prediction);
			v_win = prediction.v_win;
		}

		undos.forEach(x => x());

		return { 
			path, 
			v_win,
		};
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
					  	return scoreMovePriority(x);
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
		root.expand(this.state, await this.predict(this.state));
		const count = MoveGenerator.legal(this.state).length;
		
        if(this.dirichlet) {
			const table = [0, 0.2, 0.3];
			const alpha = -1;
			// const alpha = 0.3 * (count / count * 2) * (1 + lerpViaGameStage(this.state, table));
			throw Error("Disabled temporarily")
			const eps = 0.25;
            const noise = sampleDirichlet(alpha, count);
            for (let a = 0; a < count; a++) {
                root.P[a] = (1 - eps) * root.P[a] + eps * noise[a];
            }
        }

		for(let i = 0; i < nSims; i++) {
			const { path, v_win } = await this.simulate(root, rootGame);
			root.backpropagate(path, v_win, root.pov);
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
	game_settings: GameSettings
) {
	throw new Error("Not implemented");
}

// export async function SelfPlay(
// 	predict: (state: GameState) => Promise<Prediction>,
// 	nGames: number,
// 	nSims: number,
// 	temperature: number,
// 	cPuct: number,
// 	gamma: number,
// 	deterministic: boolean,
// 	dirichlet: boolean,
// 	rollouts: number,
// 	game_settings: GameSettings,
// ): Promise<Array<TrainingSample>> {
// 	const allTrainingData: Array<TrainingSample> = [];
// 	const loader = new GameLoader(game_settings);
	
// 	for (let i = 0; i < nGames; i++) {
// 		await loader.loadRandom();
// 		const game = new Game();
// 		const undos: UndoCallback[] = [];

// 		game.load(loader.currentState);

// 		const episode_buffer: Array<{
//             obs_dict: ObsDict;
//             chosen_move_object: Move;
//             current_player_pov: number;
//         }> = [];

// 		// Keep going until terminal
// 		while (!isGameOver(game.state)) {
// 			// TODO update
// 			const obs = AIState.extract(game.state);
// 			// max_actions doesnt exist anymore, old
// 			const pi: number[] = new Array<number>(MODEL_CONFIG.max_actions).fill(0);
// 			const legal = MoveGenerator.legal(game.state);
// 			const pov = getPovTribe(game);

// 			// instead of forcing the move, reward it for being a book move.
// 			// returns always good move indexes from the legal moves list
// 			let book: number[] = Opening.recommend(game.state, legal);
// 			let moveIndex: number;

// 			// Skip simulating if there are no more turns (only EndTurn remains)
// 			if(legal.length == 1) {
// 				moveIndex = 0;
// 				pi[0] = 1;
// 			}
// 			else {
// 				// console.time('search');
// 				const root = await new MCTS(game.state, predict, cPuct, gamma, dirichlet, rollouts).search(nSims);
// 				// console.timeEnd('search');
// 				const probs = root.distribution(temperature);
// 				probs.forEach((p, idx) => { pi[idx] = p; });
// 				moveIndex = deterministic
// 					? probs.indexOf(Math.max(...probs))
// 					: sampleFromDistribution(probs);
// 			}
			
// 			const [ move, undo ] = game.playMove(moveIndex)!;
// 			undos.unshift(undo);

// 			const shaped = gamma * AIState.calculatePotential(game.state) - AIState.calculatePotential(game.stateBefore); 
// 			const reward = AIState.calculateReward(game, move) + shaped;
// 			const v_win = isGameWon(game.state)? 1 : 
// 				isGameLost(game.state)? -1 : 
// 				(reward * 0.25);
// 			Logger.logPlay(game.stateBefore, game.state, [move], [(await predict(game.state)).v]);
// 			episode.push({ 
// 				obs, 
// 				pi,
// 				v_win, 
// 				pov: pov.owner 
// 			});
// 		}

// 		let G = 0;
// 		for (let t = episode.length - 1; t >= 0; t--) {
// 			const { obs, pi, pov, v_win } = episode[t];
// 			G = v_win + gamma * G;
// 			// In case the previous state's tribe's turn is not the current tribe's turn
// 			// STARTING_OWNER_ID = Hack for now, since entire simulator uses this and it never changes
// 			const signedG = pov === STARTING_OWNER_ID? G : -G;
// 			trainingData.push([obs, pi, signedG]);
// 		}	

// 		winRate.push(isGameWon(game.state) || isGameLost(game.state)? 1 : 0);

// 		Logger.log(`Finished game (${(i+1)}/${nGames})`);
// 		Logger.log(`state: ${isGameWon(game.state)? 'Won' :isGameLost(game.state)? 'Lost' : 'Truncated'}`);
// 		Logger.log(`turn: ${game.state.settings._turn}`);

// 		undos.forEach(x => x());
// 	}

// 	Logger.log(`Self play ended`);
// 	Logger.log(`collected: ${trainingData.length}`);
// 	Logger.log(`win rate: ${Number((winRate.reduce((a, b) => a + b, 0) / winRate.length).toFixed(2))}%\n`);

// 	winRate = [];

// 	return trainingData;
// }
