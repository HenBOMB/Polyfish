import GameLoader from "./src/core/gameloader";
import Game from "./src/game";
import { MoveGenerator } from "./src/core/moves";
import Move from "./src/core/move";
import { writeFileSync } from "fs";

(BigInt.prototype as any).toJSON = function () {
    return this.toString();
};

(Set.prototype as any).toJSON = function () {
    return Array.from(this);
};

async function run() {
    const seed = 12345;
    const loader = new GameLoader();
    console.log(`Loading game with seed ${seed}...`);

    // Load random game with specific seed
    // We use a small map for faster testing and ensuring interactions
    await loader.loadRandom({ seed, size: 11, maxTurns: 50 }, false);

    const game = new Game();
    game.load(loader.currentState);

    const initialState = JSON.parse(Game.serializeState(game.state));

    const moves: any[] = [];
    const states: any[] = []; // Optional: store full states or hashes for debugging

    // Play 50 moves or until game over
    const MAX_MOVES = 100;

    console.log("Playing moves...");
    for (let i = 0; i < MAX_MOVES; i++) {
        if (game.state.settings._gameOver) {
            console.log("Game Over reached.");
            break;
        }

        const legalMoves = MoveGenerator.legal(game.state);
        if (legalMoves.length === 0) {
            console.log("No legal moves!");
            break;
        }

        // Pick a random move
        // Use a simple pseudo-random generator to be deterministic if needed, 
        // but since we record the moves, standard random is fine for generation.
        const randomIdx = Math.floor(Math.random() * legalMoves.length);
        const move = legalMoves[randomIdx];

        // Execute move
        game.playMove(move);

        // Serialize move
        // We use the JSON serialization format designed in Move.serialize
        const moveJson = JSON.parse(Move.serialize(move, 'json'));

        // Store
        moves.push(moveJson);

        // Store state hash or summary if needed for debugging
        // For now, we rely on the Rust test to run the moves and compare end state or step-by-step
    }

    const finalState = JSON.parse(Game.serializeState(game.state));

    const output = {
        initialState,
        moves,
        finalState
    };

    writeFileSync("comparison_data.json", JSON.stringify(output, null, 2));
    console.log(`Saved ${moves.length} moves to comparison_data.json`);
}

run().catch(console.error);
