import express, { Request, Response } from "express";
import { join } from "path";
import GameLoader from "./src/core/gameloader";
import AIState, { MODEL_CONFIG } from "./src/aistate";
import { ModeType, TribeType } from "./src/core/types";
import { spawn } from "child_process";
import { CityState, DefaultGameSettings, GameSettings, GameState } from "./src/core/states";
import Game from "./src/game";
import Move from "./src/core/move";
// import { sampleFromDistribution } from "./src/polyfish/util";
import { ArmyMovesGenerator, EconMovesGenerator, MoveGenerator, Prediction } from "./src/core/moves";
import main, { deepCompare } from "./src/main";
import { getPovTribe } from "./src/core/functions";
import { Logger } from "./src/ai/logger";
import { SelfPlay } from "./src/ai/mcts.old";
import { evaluateState } from "./src/ai/eval";
import { CalculateBestMoves } from "./src/ai/brute";
import { MCTS } from "./src/ai/mcts/mcts";

const app = express();
const py = spawn(".venv/bin/python3", ["polyfish/main.py"]);
type Task = { data: string, resolve: (value: Prediction) => void };
const queue: Task[] = [];
let current: Task | null = null;
let currentGame = new Game();

(BigInt.prototype as any).toJSON = function() {
  return this.toString();
};

(Set.prototype as any).toJSON = function() {
  return Array.from(this);
};

py.stderr.on("data", (data: any) => {
    console.log(data.toString());
})

const next = () => {
    if(current) {
        return;
    }
    
    current = queue.shift() || null;
    
    if(!current) {
        return;
    }
    
    py.stdout.once("data", (data: any) => {
        try {
            current!.resolve(JSON.parse(data.toString()));
        } catch (error) {
            console.log(error);
            console.log('CONTENT');
            console.log(data.toString());
            current!.resolve({ } as any);
        } finally {
            current = null;
            next();
        }
    });
    
    py.stdin.write(current!.data + '\n');
}

async function predict(state: GameState): Promise<Prediction> {
    return new Promise((resolve) => {
        queue.push({ data: JSON.stringify({
            ...AIState.extract(state),
            cmd: 'predict'
        }), resolve });
        next();
    });
}

app.use(express.static(join(process.cwd(), "public")));
app.use(express.json({ limit: '1mb' }));

app.get('/', (req: Request, res: Response) => {
    res.sendFile(join(process.cwd(), "public", "index.html"));
});

app.get('/current', async (req: Request, res: Response) => {
    const state = currentGame.state;
    res.json({
        state,
        reward: 0,
        done: false,
        truncated: false,
    });
});

app.get('/live', async (req: Request, res: Response) => {
    const fow = req.query.fow == 'true' || req.query.fow == undefined? true : false;
    await currentGame.loadLive({ fow });
    const state = currentGame.state;
    Logger.clear();
    // main(loader);
    res.json({
        state,
        // obs: AIState.extract(state),
        // info: currentGame.loader.getSettings(),
        reward: 0,
        done: false,
        truncated: false,
    });
});

app.get('/random', async (req: Request, res: Response) => {
    const fow = req.query.fow == 'true' || req.query.fow == undefined? true : false;
    const settings: GameSettings = req.query as any;
    settings.fow = fow;
    if(req.query.size && Number(req.query.size) < 8) {
        res.status(400).json({ error: "Size must be at least 8." });
        return
    }
    if(req.query.tribes) {
        settings.tribes = String(req.query.tribes || "Imperius,Bardur").split(',').map(x => TribeType[x.trim() as keyof typeof TribeType]) as TribeType[];
    }
    await currentGame.loadRandom(settings);
    const state = currentGame.state;    
    Logger.clear();
    // main(loader);
    res.json({
        state,
        // obs: AIState.extract(state),
        info: { 
            tribes: settings.tribes,
            mode: ModeType[state.settings.mode],
            turns: state.settings.maxTurns,
            size: state.settings.size,
        },
        reward: 0,
        done: false,
        truncated: false,
    });
});

app.post('/predict', async (req: Request, res: Response) => {
    const state: GameState = req.body.state;
    if (!state) {
        res.status(400).json({ error: "Missing 'state' in request body." });
        return;
    }
    
    try {
        const moves = MoveGenerator.legal(state);

        if (moves.length === 0) {
            res.status(200).json({ move: null, reason: "No available moves." });
            return;
        }
        
        const prediction = await predict(state);
        throw "TODO CONVERSION"
        // const bestIndex = prediction.pi.indexOf(Math.max(...prediction.pi));
        
        // const move = moves[bestIndex] ?? null;
        // res.json({ move, value: prediction.v });
        
    } catch (err) {
        console.error("Error in /predict:", err);
        res.status(500).json({ error: "Prediction failed." });
    }
});

app.post("/mcts", async (req: Request, res: Response) => {
    try {
        // const prevState: GameState = req.body.state;
        // const game = new Game();
        // // const oldState = game.cloneState();
        // game.load(prevState);
        // const moves = MoveGenerator.legal(prevState);
        // const root = await new MCTS(
        //     game.state, predict, 
        //     req.body.cpuct || 1.0, 
        //     req.body.gamma || 0.997, 
        //     req.body.dirichlet || true, 
        //     req.body.rollouts || 50, 
        // ).search(req.body.iterations || 100);
        // const probs = root.distribution(req.body.temperature || 0.7);
        // // const moveIndex = (req.body.deterministic || false)
        // //     ? probs.indexOf(Math.max(...probs))
        // //     : sampleFromDistribution(probs);
        res.json({
            // probs: probs,
            move: 'not working',
            // move: moves[moveIndex].stringify(oldState, game.state).toLowerCase(),
        });
    } catch (err: any) {
        console.error("autostep error:", err);
        res.status(500).send({
            move: null,
            state: req.body.state,
            value: 0,
            potential: 0,
            reward: 0,
            error: err.message || err
        });
    }
});

app.post("/autostep", async (req: Request, res: Response) => {
    try {
        const prevState: GameState = req.body.state;
        const game = new Game();
        game.load(prevState);
        
        // const oldState = game.cloneState();
        const movez = MoveGenerator.legal(prevState);
        let moves: Move[] = [];
        throw "TODO CONVERSION"
        // const { pi, v } = await predict(prevState);
        
        // if(req.body.mcts) {
        //     let probs: number[] = [];
        //     // let start = Date.now();
        //     const root = await new MCTS(
        //         game.state, predict, 
        //         req.body.cpuct || 1.0, 
        //         req.body.gamma || 0.997, 
        //         req.body.dirichlet || true, 
        //         req.body.rollouts || 50, 
        //     ).search(req.body.iterations || 100);
        //     probs = root.distribution(req.body.temperature || 0.7);
        //     // console.log(`took: ${Date.now() - start}ms`);
        //     const moveIndex = (req.body.deterministic || false)
        //         ? probs.indexOf(Math.max(...probs))
        //         : sampleFromDistribution(probs);
        //     moves = [movez[moveIndex]];
        //     game.playMove(moveIndex);
        // }
        // else {
        //     const action = pi.indexOf(Math.max(...pi));
        //     const result = game.playMove(action < 0 || action >= movez.length? pi.indexOf(Math.max(...pi.slice(0, movez.length))) : action);
        //     if(!result) {
        //         throw 'Illegal Move';
        //     }
        //     moves = [result![0]];
        // }

        // res.json({
        //     moves: ['not working'],
        //     // moves: moves.map(x => x.stringify(oldState, game.state).toLowerCase()),
        //     state: game.state,
        //     potential:  AIState.calculatePotential(prevState) - AIState.calculatePotential(game.state),
        //     // reward: AIState.calculateReward(oldState, game, ...moves),
        //     reward: -1,
        //     value: v,
        // });
    } catch (err: any) {
        console.error("autostep error:", err);
        res.status(500).send({
            moves: [],
            state: req.body.state,
            value: 0,
            potential: 0,
            reward: 0,
            error: err.message || err
        });
    }
});

app.post('/selfplay', async (req: Request, res: Response) => {
    const settings = req.body.settings || DefaultGameSettings;
    const tribes = settings.tribes;
    if(typeof tribes == 'string') {
        settings.tribes = tribes.split(',').map(x => TribeType[x.trim() as keyof typeof TribeType]) as TribeType[];
    }
    res.json(await SelfPlay(
        predict,
        req.body.n_games || 3, 
        req.body.n_sims || 100, 
        req.body.temperature || 0.7, 
        req.body.cPuct || 1.0,
        req.body.gamma || 0.997,
        req.body.deterministic || false,
        req.body.dirichlet || true,
        req.body.rollouts || 50,
        settings,
    ));
})

app.post('/train', async (req: Request, res: Response) => {
    res.json(new Promise((resolve) => {
        queue.push({ data: JSON.stringify({
            ...req.body,
            cmd: 'train'
        }), resolve });
        next();
    }));
})

async function benchmarkThreadPerformance(
    currentGame: Game,
    simulations: number = 2000,
    maxThreads: number = 6,
    runsPerSetting: number = 3
) {
    console.log(`Starting MCTS benchmark...`);
    console.log(`- Simulations per search: ${simulations}`);
    console.log(`- Runs per thread setting: ${runsPerSetting}`);
    console.log(`- Max threads to test: ${maxThreads}\n`);

    const results: { [threads: number]: number } = {};

    for (let threadCount = 1; threadCount <= maxThreads; threadCount++) {
        const timings: number[] = [];
        console.log(`--- Testing with ${threadCount} thread(s) ---`);

        const mcts = new MCTS(currentGame, 1.0, false, 3, threadCount);
        await mcts.prepare();

        for (let run = 1; run <= runsPerSetting; run++) {
            const startTime = performance.now();
            const root = await mcts.search(simulations, false);
            const endTime = performance.now();
            const duration = endTime - startTime;
            timings.push(duration);
            console.log(`  Run #${run}: ${duration.toFixed(2)} ms`);

            if (run === 1) {
                const probs = root.distribution(1.0);
                const bestMoveIndex = probs.indexOf(Math.max(...probs));
                const legalMoves = MoveGenerator.legal(currentGame.state);
                const bestMove = legalMoves[bestMoveIndex];
                if (bestMove) {
                    console.log(`  -> Best move found: ${bestMove.stringify(currentGame.state, currentGame.state)}`);
                } else {
                    console.log(`  -> No best move found.`);
                }
            }
        }
        
        mcts.destroy();

        // Calculate and store the average time for the current thread count
        const averageTime = timings.reduce((sum, time) => sum + time, 0) / timings.length;
        results[threadCount] = averageTime;
        console.log(`-----------------------------------------`);
        console.log(`Average for ${threadCount} thread(s): ${averageTime.toFixed(2)} ms`);
        console.log(`-----------------------------------------\n`);
    }

    console.log("======== MCTS Benchmark Summary ========");
    let bestTime = Infinity;
    let optimalThreads = 0;

    for (const threads in results) {
        const time = results[threads];
        if (time < bestTime) {
            bestTime = time;
            optimalThreads = parseInt(threads);
        }
        console.log(`- ${threads} Thread(s): ${time.toFixed(2)} ms`);
    }

    console.log("\n========================================");
    console.log(`Optimal thread count found: ${optimalThreads} threads (${bestTime.toFixed(2)} ms)`);
    console.log("========================================");
}

app.listen(3000, async () => {
    Logger.clear();
    console.log(`INITIALIZED ON PORT 3000\n`);
    
    // RUN SOME TESTS
    await currentGame.loadRandom({ fow: false, tribes: [TribeType.Imperius, TribeType.Bardur] });

    // console.log(evaluateState(currentGame));
    // benchmarkThreadPerformance(currentGame, 1500, 16, 100);
    
    // const [ bestMoves ] = await CalculateBestMoves(
    //     currentGame,
    //     1,
    //     { depth: 1500, cPuct: 1.0, nThreads: 6 }
    // );

    MoveGenerator.legal(currentGame.state).forEach(x => {
        console.log(x.stringify(currentGame.state, currentGame.state));
    });

    const move: any[] = [];
    console.log(ArmyMovesGenerator.all(currentGame.state, move));
    console.log(move);

    // const mcts = new MCTS(currentGame, 1.0, false, 3, 16);
    // console.time('prepare');
    // await mcts.prepare();
    // console.timeEnd('prepare');

    // const l = `took ${mcts.numThreads} threads`
    // console.time(l);
    // await mcts.search(
    //     1500, 
    //     true/*,
    //     // ! Not supported because Game.playMove doesnt support custom legal move generation
    //     (state: any) => {
    //         if(state.settings._pendingRewards.length) {
    //             return state.settings._pendingRewards.slice();
    //         }
    //         const moves: any = [new EndTurn()];
    //         EconMovesGenerator.all_fast(state, moves);
    //         return moves;
    //     }*/
    // );
    // console.timeEnd(l);
    // console.time(l);
    // await mcts.search(1500, true);
    // console.timeEnd(l);
    // mcts.destroy();

    // console.log(currentGame.state.settings._pendingRewards.push(...EconMovesGenerator.rewards(
    //     { _level: 3, _rewards: new Set([]), tileIndex: 0 } as any as CityState,
    // )));
    // console.log(currentGame.state.settings);

    // TODO why is it picking explorer over workshop?
    // workshop gives +0.015 guarenteed!
    // explorer gives nothing cause all tiles have already been explored!

    // console.log(moves[bestMove].stringify(currentGame.state, currentGame.state));

    // main(loader);
    // await loader.loadRandom();
    // const prediction = await predict(loader.currentState);
    // console.log(MoveGenerator.fromPrediction(loader.currentState, prediction));

    // const rebuiltGame = Game.deserialize(Game.serialize(currentGame));
    // console.log(deepCompare(currentGame, rebuiltGame, 'state', true)? "success" : "failed");

    // const rebuiltState = Game.deserializeState(Game.serializeState(currentGame.state));
    // console.log(deepCompare(currentGame.state, rebuiltState, 'state', true)? "success" : "failed");
});