# Polyfish

An `AI NN` + MCTS `Rust` capable of playing the award winning `Polytopia` strategy game. 

## Features
- **Replica Simulation**: I painstakingly rebuilt the entire Polytopia game logic in Typescript, then translated it to Rust for performance and AI training.
- **Web UI**: Mimicry of the original game's UI, fully interactive and served by the Rust backend.
- **AI Engine**: A hybrid MCTS (Monte Carlo Tree Search) + Neural Network (Alpha-Zero style) approach.
- **Game RIPPER**: C++ injection script that extracts live game states from the Steam version of Polytopia.

## TODOs
- Need monster compute to train the NN.
- Elo rating system.

## Quick Start

1. Install **Rust**, **Python 3**, and **CMake**.
2. Run `./run-server.sh` (Initializes backend on port 3000).
3. Open `http://localhost:3000` in your browser.

## Training

- **`polyfish-rs/local_setup.sh`**: Set up the Python virtual environment and dependencies (`candle`, `numpy`, `safetensors`).
- **`polyfish-rs/run_training_loop.sh`**: Launch the self-play and training cycle.
- **`polyfish-rs/train.py`**: The PyTorch/Candle-compatible training script.

## Core Modules

- **`polyfish-rs/`**: The Rust-based game engine and AI agent.
- **`src/public/`**: The modern Web UI frontend (JS/HTML/CSS).
- **`polyfish-scraper/`**: Utilities for gathering game data and assets.
- **`polyfish-reader/`**: The C++ game state extraction suite.
- **`current_understanding.md`**: Single source of current truth on how the AI plays and where it's weak — **read this first.**
- **`notes.md` / `notes-heuristics.md`**: Historical research journal and evaluator-design spec (defer to `current_understanding.md` for what's still current).

## AI Architecture

- **`polyfish-rs/src/ai/mcts_zero.rs`**: Core Alpha-Zero style MCTS implementation.
- **`polyfish-rs/src/ai/network.rs`**: Candle-powered Neural Network architecture.
- **`polyfish-rs/src/ai/evaluator/`**: Modular logic for Economy, Military, and Expansion evaluation.
- **`polyfish-rs/src/ai/heuristic_mcts.rs`**: Lightweight MCTS using heuristics for UI analysis.
- **`polyfish-rs/src/ai/book.rs`**: Opening move library for standardized tribe starts.
- **`polyfish-rs/src/ai/features.rs`**: Logic for encoding GameState into NN tensors.

## GameState Ripper (Steam)

- **`polyfish-reader/polyfish-reader.cpp`**: C++ memory reader that extracts live game state for the simulator.
- **`polyfish-reader/inputerv2`**: Interactive tool for manipulating live game memory.
- **`polyfish-reader/polyfish-scanner.cpp`**: Utility for finding memory offsets in new game versions.