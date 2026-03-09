# Polyfish

An `AI model` and `Rust` game simulator based on the Polytopia strategy game.

I built the original version in TS then worked with AI to translate it to Rust.

## Quick Start

1. Install Rust
2. Run `./run-server.sh`
3. Go to http://localhost:3000

## Training

- **`polyfish-rs/local_setup.sh`**: Install Python requirements.
- **`polyfish-rs/run_training_loop.sh`**: Launch the training loop.

## Game

- **`polyfish-rs/`**: Exact (i hope) replica of the Polytopia game logic with an AI/MCTS agent.
- **`src/public/`**: Web UI for playing with the simulator and AI.
- **`run-server.sh`**: Helper script to launch the backend server UI to play the simulated game.
- **`notes.md`**: Research and architecture info.

## AI

- **`polyfish-rs/src/ai/mcts_zero.rs`**: Alpha-Zero style MCTS.
- **`polyfish-rs/src/ai/network.rs`**: Neural network architecture.
- **`polyfish-rs/src/ai/features.rs`**: Gamestate feature extraction.
- **`polyfish-rs/src/ai/heuristics.rs`**: Economy and Military based heuristic evaluation.
- **`polyfish-rs/src/ai/evaluation.rs`**: Winning position evaluation function.
- **`polyfish-rs/src/ai/opening.rs`**: Standard opening moves book for different tribes.

## GameState Ripper

- **`polyfish-reader/polyai-reader`**: C++ script that injects into the running Steam game and extracts the real game state. Ready to be parsed by the Rust simulator.
- **`polyfish-reader/inputer`**: Failed attempt to manipulate the game's live state into playing AI moves.