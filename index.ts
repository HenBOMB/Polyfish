import express, { Request, Response } from "express";
import { join } from "path";
import AIState from "./src/aistate";
import { ModeType, TribeType } from "./src/core/types";
import { spawn } from "child_process";
import { GameSettings, GameState, UnitState, DefaultGameSettings } from "./src/core/states";
import Game from "./src/game";
import {  ArmyMovesGenerator, EconMovesGenerator, MoveGenerator, Prediction } from "./src/core/moves";
import { deepCompare } from "./src/main";
import { Logger } from "./src/ai/logger";
import { evaluateAllStates, evaluateState } from "./src/ai/eval";
import { CalculateBestMoves } from "./src/ai/brute";
import { MCTS, SelfPlay } from "./src/ai/mcts/mcts";
import Move from "./src/core/move";
import EndTurn from "./src/core/moves/EndTurn";

const app = express();
const py = spawn(".venv/bin/python3", ["polyfish/main.py"]);
type Task = { data: string, resolve: (value: Prediction) => void };
const queue: Task[] = [];
let current: Task | null = null;
let game = new Game();

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
            const result = JSON.parse(data.toString());
            current!.resolve(result);
        } catch (error) {
            console.error('JSON Parse Error:', error);
            console.error('Raw Python output:', data.toString());
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

app.get('/eval', async (req: Request, res: Response) => {
    res.json(evaluateAllStates(game));    
});

app.get('/current', async (req: Request, res: Response) => {
    const state = game.state;
    res.json({
        state,
        reward: 0,
        done: false,
        truncated: false,
    });
});

app.get('/live', async (req: Request, res: Response) => {
    const fow = req.query.fow == 'true' || req.query.fow == undefined? true : false;
    await game.loadLive({ fow });
    const state = game.state;
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

app.post('/sequence', async (req: Request, res: Response) => {
    game.playSequence(...((req.body?.ids || [0]) as number[]));
    res.json({
        state: game.state
    });
});

app.post('/bestmoves', (req: Request, res: Response) => {
    try {
        CalculateBestMoves(
            game,
            6,
            { 
                depth: 1500,
                cPuct: 1.0,
                nThreads: 6,
                maxTurnsAhead: 5,
                deterministic: true,
                predict: predict  // Pass the neural network predict function
            }
        ).then(async ([ ,,,, bestMoves ]) => {
            if (bestMoves.length === 0) {
                res.status(200).json({ move: null, reason: "No available moves." });
                return;
            }

            // currentGame.playSequence(...bestMoves);

            res.json({
                bestMoves
            });
        }).catch((e) => {
            throw Error(e);
        });
    } catch (err) {
        console.error("Error in /predict:", err);
        res.status(500).json({ error: "Prediction failed." });
    }
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
    await game.loadRandom(settings);
    const state = game.state;    
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
    const rawState: GameState = req.body.state;
    if (!rawState) {
        res.status(400).json({ error: "Missing 'state' in request body." });
        return;
    }
    
    try {
        // Normalize state: convert JSON arrays back to Sets, etc.
        // Use Game.deserializeState to properly restore Sets from JSON
        const normalizedState = Game.deserializeState(JSON.stringify(rawState));
        
        console.log('[PREDICT] Extracting observation...');
        const obs = AIState.extract(normalizedState);
        console.log('[PREDICT] Observation extracted, map shape:', obs.map.length, obs.map[0]?.length, obs.map[0]?.[0]?.length);
        console.log('[PREDICT] Player vector length:', obs.player.length);
        
        const prediction = await predict(normalizedState);
        console.log('[PREDICT] Got prediction, keys:', Object.keys(prediction));
        
        if (!prediction || Object.keys(prediction).length === 0) {
            throw new Error("Empty prediction received from Python");
        }
        
        res.json(prediction);
    } catch (err: any) {
        console.error("Error in /predict:", err);
        console.error("Stack:", err.stack);
        res.status(500).json({ error: "Prediction failed.", details: err.message });
    }
});

app.post("/play", async (req: Request, res: Response) => {
    try {
        const rawState: GameState = req.body.state || game.state;
        // Normalize state: convert JSON arrays back to Sets, etc.
        const normalizedState = rawState === game.state 
            ? rawState 
            : Game.deserializeState(JSON.stringify(rawState));
        const tempGame = new Game();
        tempGame.load(normalizedState);
        
        const moves = MoveGenerator.legal(tempGame.state);
        if (moves.length === 0) {
            res.json({ move: null, reason: "No available moves." });
            return;
        }
        
        const mcts = new MCTS(
            tempGame,
            req.body.cPuct || 1.0,
            req.body.dirichlet || false,
            req.body.maxTurnsAhead || 3,
            req.body.nThreads || 1,
            undefined,
            predict
        );
        await mcts.prepare();
        
        const root = await mcts.search(req.body.iterations || 100);
        mcts.destroy();
        
        const probs = root.distribution(req.body.temperature || 0.7);
        const moveIndex = (req.body.deterministic || false)
            ? probs.indexOf(Math.max(...probs))
            : (() => {
                const rand = Math.random();
                let cumsum = 0;
                for (let i = 0; i < probs.length; i++) {
                    cumsum += probs[i];
                    if (rand < cumsum) return i;
                }
                return probs.length - 1;
            })();
        
        const selectedMove = moves[moveIndex];
        
        res.json({
            move: selectedMove.serialize('array'),
            moveIndex: moveIndex,
            probabilities: probs,
            value: root.Q[moveIndex] || 0,
        });
    } catch (err: any) {
        console.error("play error:", err);
        res.status(500).send({
            move: null,
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

app.post('/unitmoves', async (req: Request, res: Response) => {
    const unit = req.body.unit as UnitState;

    const legalMoves = MoveGenerator.legal(game.state);

    res.json(legalMoves.map(x => x.serialize('array')));
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
    console.log(`\nReady on: http://localhost:3000/\n`);
    
    // await game.loadLive({ fow: true, fallback: 'data/gamestate.json' });

    // game.state.settings._fow = false;

    // RUN SOME TESTS
    // await currentGame.loadRandom({ 
    //     fow: false,
    //     tribes: [TribeType.Imperius, TribeType.Bardur],
    //     seed: 8
    // });

    // console.log(evaluateState(currentGame));
    // benchmarkThreadPerformance(currentGame, 1500, 16, 100);
    
    // currentGame.playSequence(...[6, 8, 5, 0, 1, 0, 0, 4, 9, 0, 1, 1, 0, 3, 0, 1, 1, 0, 1, 1, 1, 0, 2, 1, 1, 0]);
    // currentGame.playSequence(...[1, 5, 0, 1, 1, 0, 1, 5, 0, 1, 1, 0, 4, 6, 0, 4, 3, 0, 1, 1, 4, 1, 7, 7, 0, 0, 5, 1, 5, 0, 0, 5, 9, 8, 1, 1, 0, 12, 1, 1, 1, 0, 6, 7, 1, 1, 1, 1, 0, 7, 3, 1, 1, 0, 15, 5, 5, 1, 1, 0, 6, 6, 1, 9, 1]);

    // const iState = game.cloneState();
    // let moves: Move[] = [];
    // console.log(moves.map(x => {
    //     const evalBefore = evaluateState(game)[2];
    //     const play = x.execute(game.state);
    //     const evalNow = evaluateState(game)[2];
    //     const evalDiff = evalNow - evalBefore;
    //     const str = x.stringify(iState, game.state);
    //     play.undo();
    //     return `[${str}] ${evalDiff>0?'+':'-'}${evalDiff.toFixed(3)}`;
    // }).length);
    
    // await CalculateBestMoves(
    //     game,
    //     1,
    //     { 
    //         depth: 2000,
    //         cPuct: 1.0,
    //         nThreads: 6,
    //         maxTurnsAhead: 2,
    //         deterministic: false,
    //         legalFn: (state) => {
    //             const moves: Move[] = [];
    //             ArmyMovesGenerator.legal(state, moves);
    //             moves.unshift(new EndTurn());
    //             return moves;
    //         },
    //     }
    // ).then(x => {
    //     // console.log(x[0].map(x => x.stringify(game.state, game.state)));
    // }).catch((e) => {
    //     console.error(e);
    // });

    // deepCompare(
    //     { state: {
    //         r: iState.resources,
    //         s: iState.structures,
    //         t: iState.tiles,
    //         T: iState.tribes,
    //         S: iState.settings,
    //     } },
    //     { state: {
    //         r: game.state.resources,
    //         s: game.state.structures,
    //         t: game.state.tiles,
    //         T: game.state.tribes,
    //         S: game.state.settings,
    //     } },
    //     'state',
    //     true
    // );

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