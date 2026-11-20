# AGENTS.md

Polyfish is a Rust game engine for *The Battle of Polytopia*, with a browser simulator and toolkits for bot training and evaluation. Open PRs against `main`.

## Where to work

- `polyfish-rs/` contains the engine, web server, Rust agents, and training pipeline. The public simulator entry points are in `src/game/decision.rs` and `src/research.rs`. [Simulator API](research/simulator-api.md)
- `src/public/` is the primary graphical simulator served at `/simulator/`. `polyfish-ui/public/research/` is the plain board at `/simulator/plain/`. `polyfish-ui/public/simulator/` is a separate copy. Check the served path before editing UI files.
- `research/` contains headless examples, parity fixtures, and benchmarks. [Simulator and benchmark guide](research/README.md)
- `polyfish-mod/` is the Steam replay capture mod, `polyfish-scraper/` contains data utilities, and `scripts/` contains launcher and reporting tools.

## Build and verify

Run these commands from `polyfish-rs/`.

```sh
POLYFISH_LISTEN=127.0.0.1:3000 cargo run --locked --bin polyfish
cargo test --locked --no-default-features --lib --tests --bin self_play
```

Use the checks relevant to your change.

- **Engine rules and search.** Run the Rust suite. For simulation or scoring changes, also consider the ignored cases in `tests/undo_integrity.rs` and `tests/score_parity.rs`.
- **Training CLI or driver.** Run `python3 scripts/check_cli_contract.py` and `./scripts/smoke_train_loop.sh`. The smoke runs one iteration in a scratch directory and exercises the anchor freeze and audit paths. Review shell callers when changing a Rust or Python flag, including `run_training_loop.sh` and `../scripts/auto_train.sh`.
- **Python training or ladder.** Run `./scripts/run_python_tests.sh`. Use the training environment to include the PyTorch-dependent cases.
- **Model or features.** Run `cargo test --no-default-features --test parity_widths` and `./scripts/run_forward_parity.sh`. Run the applicable `examples/tch_parity.rs` and `examples/metal_parity.rs` checks on macOS when changing those backends.

CI treats the Clippy correctness check as blocking and the full Clippy and formatting reports as advisory. [Rust workflow](.github/workflows/rust.yml) · [Training smoke workflow](.github/workflows/smoke.yml)

## Engine behavior

- Use `Game::play` to commit a validated action and `Game::simulate` for search. Use `Game::decision` when inspecting and applying from the same legal-action list. Simulation takes an explicit `TurnRotation`. Apply undo callbacks in reverse order on the same game. Committed undo keeps discoveries visible, and simulation undo reverses them.
- Keep fog-of-war policies on the acting player's observation and filtered legal actions. Research observations represent unexplored terrain as `TerrainType::Fog`. Training runs with fog enabled. `POLYFISH_ADVERSARIAL_SEARCH=1` changes internal MCTS turn rotation and opponent visibility. It defaults to off. Serialize tests that toggle it.
- Keep incremental scores and canonical recomputation in agreement. Both use `score.rs`. Tile ownership determines territory and structure score, while cities contribute their level, population, and parks. See `tests/score_parity.rs`.
- Preserve save and replay compatibility when changing `states.rs`, `types.rs`, or action identities. The canonical replay implementation lives in `src/replay/`. The top-level `src/recorder.rs` records human and mod steps for imitation data.

## Search and model compatibility

- Self-play uses `gumbel_mcts.rs`. `mcts_zero.rs` is the PUCT variant, and `heuristic_mcts.rs` implements model-free search. PUCT stores values in each node's player perspective. Gumbel stores edge values in the parent's perspective. Keep backup and child selection consistent with the agent's convention and use `mcts_common::edge_hands_over` for turn changes.
- Keep value labels and reward-aware backup aligned through `reward.rs`. The default relative reward is antisymmetric across seats. `v_progress` is trained as a separate output. Keep it out of search Q values and policy targets. See `tests/test_progress_head_not_in_search.rs`.
- Keep `train.py`, `src/ai/network.rs`, `tch_network.rs`, and `metal_network.rs` compatible with the shared `model.safetensors` format. Update `features.rs` and `mapper.rs` together with any shape or policy change. Current neural training uses 11×11 boards. Policy targets encode action type, source, target, and option independently of legal-list order.
- Keep the `aux_*` training heads in PyTorch. The Rust Candle trainer writes `model_candle.safetensors` to preserve the primary checkpoint and its extra heads. Preserve the legacy padding and migration paths when changing model formats. [Migration script](polyfish-rs/migrate_model.py)
- Make transposed tensors contiguous before feeding them to Candle linear layers or matrix multiplication. Cross-framework forward parity is the check for batched numerical agreement. Cargo documents the optional backend features and libtorch setup. [Cargo manifest](polyfish-rs/Cargo.toml)

## Training and evaluation

- Set up the Python environment with `local_setup.sh`, `remote_setup.sh`, or `vast_setup.sh`. The setup scripts select the platform's PyTorch wheel using the common version pin in `requirements.txt`. Recheck the pinned `tch-rs` compatibility before changing that pin.
- `run_training_loop.sh` runs `init_model.py`, Rust `self_play`, and `train.py`. Keep consumed games archived before the arena strength check. A failed strength check aborts the run. Use `--resume [run_id]` to continue or `--new-run` to start a new schedule. `--reset` starts from scratch and deletes the active model, self-play data, and anchor decay marker.
- Keep diagnostic arms in `bisect_arm.sh`. Treat settings exported unconditionally by the training loop as production settings. Check the driver's active environment and use `self_play --print-curriculum` for iteration-specific budgets and search weights.
- Use a fixed `arena --seed` for comparisons. Arena plays both seats for each seed. The scheduled strength check uses a pinned tribe pair and the iteration's curriculum values. Anchor Elo applies to that pair. Keep `tribe_audit` rows out of plateau detection and the Elo fit. Link same-iteration audits to the outgoing anchor when a freeze occurs. [Ladder code](polyfish-rs/ladder.py) · [Ladder tests](polyfish-rs/tests/test_ladder.py)
- `training_log.csv` records run metrics by `run_id`, and `ladder.json` records strength readings. The backup script can snapshot these records and checkpoints. [Backup script](polyfish-rs/scripts/backup_experiment_record.sh)

## Comments and research records

Write comments for rule edge cases, invariants, and tradeoffs. Keep inline comments to one line where possible, parameter docs to two lines each, and function docs to four lines.

Read the engine notes before changing search or evaluation. Check prior experiments and verify their conclusions against the current implementation. [Engine notes](docs/notes/notes.md) · [Heuristics](docs/notes/notes-heuristics.md) · [Observation memory](docs/notes/notes-memory.md) · [Experiment log](docs/experiments/hypothesis_driven_improvements.md) · [Pipeline audit](docs/experiments/expert_pipeline_audit.md)
