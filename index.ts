import express, { Request, Response } from "express";
import { join } from "path";
import GameLoader from "./src/core/gameloader";
import AIState, { MODEL_CONFIG } from "./src/aistate";
import { ModeType, TribeType } from "./src/core/types";
import { MCTS, Prediction, sampleFromDistribution, SelfPlay } from "./src/polyfish/mcts";
import { spawn } from "child_process";
import { DefaultGameSettings, GameSettings, GameState } from "./src/core/states";
import { generateAllMoves, generateEndTurnMove } from "./src/core/moves";
import Game from "./src/game";
import Move, { MoveType } from "./src/core/move";

const app = express();
const py = spawn(".venv/bin/python3", ["polyfish/main.py"]);
type Task = { data: string, resolve: (value: Prediction) => void };
const queue: Task[] = [];
let current: Task | null = null;

py.stderr.on("data", (data) => {
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
    
    py.stdout.once("data", (data) => {
        try {
            current!.resolve(JSON.parse(data.toString()));
        } catch (error) {
            console.log(error);
            console.log('CONTENT');
            console.log(data.toString());
            current!.resolve({ pi: [], v: 0 });
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

app.get('/random', async (req: Request, res: Response) => {
    const settings: GameSettings = req.query as any;
    if(req.query.tribes) {
        settings.tribes = String(req.query.tribes || "Imperius,Bardur").split(',').map(x => TribeType[x.trim() as keyof typeof TribeType]) as TribeType[];
    }
    const loader = new GameLoader(settings);
    const state = await loader.loadRandom();
    res.json({
        state,
        obs: AIState.extract(state),
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
        const moves = generateAllMoves(state);
        if (moves.length === 0) {
            res.status(200).json({ move: null, reason: "No available moves." });
            return;
        }
        
        const prediction = await predict(state);
        const bestIndex = prediction.pi.indexOf(Math.max(...prediction.pi));
        
        const move = moves[bestIndex] ?? null;
        res.json({ move, value: prediction.v });
        
    } catch (err) {
        console.error("Error in /predict:", err);
        res.status(500).json({ error: "Prediction failed." });
    }
});

app.post("/autostep", async (req: Request, res: Response) => {
    try {
        const prevState: GameState = req.body.state;
        const game = new Game();
        game.load(prevState);
        
        const movez = generateAllMoves(prevState);
        const { pi, v } = await predict(prevState);
        const action = pi.indexOf(Math.max(...pi));
        let moves: Move[] = [];

        if(req.body.mcts) {
            if(movez.length == 1) {
                // Used just to translate to "end turn" (Move.stringify)
                moves = [generateEndTurnMove()];
                game.playMove(0);
            }
            else {
                let probs: number[] = [];
                // let start = Date.now();
                const root = await new MCTS(game.state, predict, req.body.cpuct || 1.0, req.body.gamma || 0.997).search(req.body.iterations || 200);
                probs = root.distribution(req.body.temperature || 0.7);
                // console.log(`took: ${Date.now() - start}ms`);
                const fullProbs = new Array<number>(MODEL_CONFIG.max_actions).fill(0);
                probs.forEach((p, idx) => { fullProbs[idx] = p; });
                const moveIndex = false
                    ? probs.indexOf(Math.max(...probs))
                    : sampleFromDistribution(probs);
                moves = [movez[moveIndex]];
                game.playMove(moveIndex);
            }
        }
        else {
            moves = game.playMove(action < 0 || action >= movez.length? pi.indexOf(Math.max(...pi.slice(0, movez.length))) : action);
        }

        res.json({
            moves: moves.map(x => x.stringify()),
            value: v,
            state: game.state,
            potential: AIState.calculatePotential(game.state),
            reward: AIState.calculateReward(prevState, game.state),
        });
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
    res.json(await SelfPlay(
        predict,
        req.body.n_games || 3, 
        req.body.n_sims || 100, 
        req.body.temperature || 0.7, 
        req.body.cPuct || 1.0,
        req.body.gamma || 0.997,
        req.body.deterministic || false,
        req.body.settings || DefaultGameSettings
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

app.listen(3000, async () => {
    console.log(`INITIALIZED ON PORT 3000\n`);
    console.log('FOW DISABLED\n');
    const loader = new GameLoader();
    await loader.loadRandom();
    await predict(loader.currentState);
});