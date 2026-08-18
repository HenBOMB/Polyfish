# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Polyfish is an AlphaZero-style AI (MCTS + neural network) that plays *The Battle of Polytopia*. The core is a from-scratch reimplementation of the entire Polytopia game engine — first written in TypeScript, then ported to Rust (`polyfish-rs/`) for performance and training throughput. Everything of consequence lives in `polyfish-rs/`; the root directory is Node-based glue (server launch wrappers, Telegram/Supabase reporting) plus the satellite data-capture projects.

## Repository layout

- `polyfish-rs/` — Rust game engine, AI, web backend, and all training code. This is where ~all work happens.
- `src/public/` — static web UI (JS/HTML/CSS), served by the Rust `polyfish` binary at `http://localhost:3000`.
- `polyfish-ui/` — newer Vite/TypeScript UI. Note it contains a **forked copy** of the static UI under `polyfish-ui/public/simulator/`; the Rust server serves `src/public`, not either copy here. Check which one you're editing.
- `polyfish-mod/` — C# mod (BepInEx/PolyMod) that runs inside the real Steam game to auto-play replays and POST captured game states to the local server.
- `polyfish-scraper/` — TypeScript utilities for gathering game data/assets.
- `polyfish-reader` — the C++ process-memory ripper. **Not checked into this repo**; `scan.sh` compiles and invokes it against a running `Polytopia.exe`.
- `*.py`, `*.sh`, `*.js` at root and in `polyfish-rs/` — training, analysis, and reporting scripts.

## Commands

All `cargo` commands run from `polyfish-rs/`. The root `run-server.sh` and `polyfish-rs/run_training_loop.sh` `cd` there for you.

**Run the web server / simulator** (port 3000, serves `src/public`):
```bash
./run-server.sh                      # from repo root; kills :3000, then `cargo run --bin polyfish`
```
On startup `main.rs` tries to load game state from `live_game.json`, `saved_state.json`, then the newest `replays/mod_replay_*.json`, before falling back to a generated map.

**Build the training binaries** (release is required for any real training):
```bash
cd polyfish-rs && cargo build --release --bin polyfish --bin self_play --bin arena
```

**Tests** (tool binaries in `src/bin/` set `test = false` in `Cargo.toml`; `#[ignore]` marks heavy integration probes — neither runs in CI):
```bash
# CI-equivalent — note --no-default-features, which is what .github/workflows/rust.yml runs
cd polyfish-rs && cargo test --no-default-features --lib --tests --bin self_play

cd polyfish-rs && cargo test --test integration my_test_name       # a single libtest case
cd polyfish-rs && cargo test -- --ignored test_min_capital_distance_1v1   # heavy mapgen probe
cd polyfish-rs && cargo run --bin stats -- --games 50              # manual diagnostic tool
```

**Python training setup** (creates `polyfish-rs/.venv` from `requirements.txt`):
```bash
cd polyfish-rs && ./local_setup.sh           # or remote_setup.sh / vast_setup.sh on a GPU box
```

**Full self-play + train loop** (the main training driver):
```bash
cd polyfish-rs && ./run_training_loop.sh [flags]
```
Short flags (getopts `fbcri:g:n:a:e:l:k:`): `-f` force-train, `-b` boost-threads, `-c` chill, `-r` reward-shaping, `-i` iterations, `-g` games-per-iter, `-n` mcts-iters, `-a` actors, `-e` eval-servers, `-l` league/gauge interval (default 10), `-k` gumbel-k. Long flags: `--resume [run_id]`, `--reset`.

The loop is: `init_model.py` → `self_play` (Rust, generates `games_*.safetensors`) → `train.py` (Python/PyTorch, updates `model.safetensors`) → log a CSV row → checkpoint every 50 iters into `checkpoints/` → archive consumed games, plus a strength-gauge match every `-l` iterations. CUDA is opt-in via the `cuda`/`cudnn` Cargo features. `--reset` deletes `model.safetensors` and all self-play game data (`games_*.safetensors` in root and `archive/`) before starting; it forces a new run (overrides `--resume`) and leaves `checkpoints/`, `training_log.csv`, and `moves_by_turn.json` untouched. Each run is a **new run** by default; `--resume` continues the latest.

**Head-to-head evaluation:**
```bash
cd polyfish-rs && cargo run --release --bin arena -- --model1 a.safetensors --model2 b.safetensors --games 32 --mcts 64
```

## Architecture

### Game engine (`polyfish-rs/src/`)
- `states.rs` / `types.rs` — the `GameState` data model and all enums (tribes, units, tech, moves, etc.). `GameState` (de)serializes to/from the JSON produced by the mod and reader.
- `game.rs` — the `Game` controller: load state, run `post_load()` (recompute tile indices/visibility), apply moves, manage turns. The engine is intended to be "perfect": legal-move generation and `execute` should never panic on valid input — panics are treated as bugs to surface, not suppress.
- `moves/` — the `Move` trait and `generate_legal_moves(state)`. Moves split into **economy moves** (city/tech/structure — mostly keyed by `target_index`) and **army moves** (step/attack/capture/abilities — mostly keyed by `src_index`). Unit abilities live in `moves/abilities/`.
- `actions/` — lower-level reusable state mutations (gain stars, exploration, effects) that moves compose, with undo-callback support for MCTS rollouts.
- `settings/` — static game data tables: `units.rs`, `technology.rs`, `structures.rs`, `resources.rs`, `tasks.rs`.
- `mapgen.rs`, `coords.rs`, `fow.rs`, `memory.rs`, `score.rs`, `hash.rs` — map generation, coordinate/index math, fog-of-war, observation memory, scoring, state hashing. **Training runs with FOW enabled** (deliberate, to avoid learning cheating/FOW-less strategies).
- `replay/` — the replay subsystem (`schema.rs`, `loader.rs`, `executor.rs`, `playback.rs`, `validator.rs`, `recorder.rs`, `training.rs`). Distinct from the top-level `recorder.rs`, which records human/mod steps for imitation data. `version_sync.rs` tracks which Polytopia version (`GameVersion`) the rules target.

### AI (`polyfish-rs/src/ai/`)
- `mcts_zero.rs` — the AlphaZero-style MCTS (`ZeroMctsAgent`). `gumbel_mcts.rs` is the Gumbel variant and the one training actually uses; `heuristic_mcts.rs` is a network-free MCTS for fast UI analysis and the interactive trainer. `mcts_common.rs` holds the shared backup/descent logic; `mcts.rs` and `original_mcts_zero.rs` are older implementations — check whether a change needs to land in more than one.
- `brain.rs` — top-level agent wiring. `Brain::with_backend(...)` plus the `with_prior_heuristic_weight` / `with_policy_target_q_weight` / `with_tree_q_weight` builders decide what the agent actually is. **These knobs are set in `self_play.rs` and left at their defaults in `arena.rs`** — if you change search behavior, check both call sites or training and evaluation will silently diverge.
- `network.rs` — `PolyZeroNet`, the candle network: player-state embedding + ResBlocks + cross-attention + a **decomposed policy** and a value head.
- `features.rs` — encodes `GameState` into the input tensor. Key constants: `MAP_SIZE = 11`, `NUM_CHANNELS`, `RawFeatures::PLAYER_STATE_DIM`. Maps are 11×11.
- `mapper.rs` — `DecomposedMapper` / `DecomposedTargets`: the policy is decomposed into four heads — `action_type`, `source_spatial` (H·W), `target_spatial` (H·W), and a unified `move_option` (192, with offset blocks for structures/units/techs/abilities). This decomposition exists because raw legal-move ordering is non-deterministic across states, so moves are mapped to stable semantic coordinates instead of a flat action index.
- `evaluator/` — heuristic state evaluation split by concern: `economy.rs`, `army.rs`, `research.rs`, `exploration.rs`, `expansion.rs`, `gamestate.rs`, `player.rs`. Used to shape/guide self-play and for non-NN play.
- `reward.rs` — the shared per-move reward used by both TD value labels and reward-aware MCTS backup. Note `reward::REL_W` and `self_play.rs`'s `FINAL_OUTCOME_REL_W` both control relative-vs-absolute weighting and are currently set inconsistently; read both comment blocks before touching either.
- `book.rs` — opening-move library; `ordering.rs` — move ordering; `policy_composer.rs` — assembles head outputs into a move distribution; `decision_trace.rs` — search introspection.

### Inference backends
Four implementations read the same `model.safetensors`, selected by Cargo feature:
- `network.rs` (candle) — default, and the only one on non-Apple hardware.
- `tch_network.rs` (`tch-eval`) — libtorch/MPS. Requires PyTorch 2.12.x plus `LIBTORCH_USE_PYTORCH=1`, `LIBTORCH_BYPASS_VERSION_CHECK=1`, and `.venv/bin` on `PATH`; see the comments in `Cargo.toml`.
- `metal_network.rs` (`metal-eval`) — hand-composed MPSGraph, bypassing libtorch's serial MPS dispatch queue. Fastest on Apple silicon.
- `eval_backend.rs` / `eval_server.rs` — the batching layer that fans leaf evaluations across actors.

`examples/tch_parity.rs` and `examples/metal_parity.rs` exist to check backends against each other — run them after any architecture change.

### ⚠️ The multi-implementation sync constraint
The network architecture is implemented in **Rust (candle) and Python (PyTorch)** and must stay byte-compatible because they read/write the same `model.safetensors`:
- Rust: `polyfish-rs/src/ai/network.rs` — used by `self_play`, `arena`, the server, and the Rust `train` binary.
- Python: `polyfish-rs/train.py` — the primary trainer used by `run_training_loop.sh`; `init_model.py` creates the initial weights from this definition.

If you change layer shapes, channel counts, or head sizes in one, you must mirror it in the other (**and** in `tch_network.rs` / `metal_network.rs`, and in `features.rs` / `mapper.rs` constants). Current values: spatial channels **142** (`features.rs` `NUM_CHANNELS`, `train.py` `SPATIAL_CHANNELS`; = 136 + 6 fog-memory channels), player-state dim **16** (`features.rs:216`, `train.py` `PLAYER_STATE_DIM`), map 11×11, 6 ResBlocks on a 64-filter trunk, policy heads = action + source + target + option(192), normalization = GroupNorm(`GN_GROUPS = 8`) — no BatchNorm anywhere; the 1-channel pool convs are fully linear (no norm, no activation, since an unnormed ReLU there dies irreversibly). Mismatches surface as safetensors load errors or silent garbage. Legacy 136-channel training data is zero-padded at load by `train.py` (channels were appended at the end of the layout); BatchNorm-era checkpoints are rejected at load.

**Known trap:** `network.rs` exports `NUM_ACTION_TYPES = 12` (used by the self-play/replay writers to size the `action_type` target) while the `pi_action` layer is built with a hardcoded `11` in both `network.rs` and `train.py`. `mapper.rs` maps `MoveType::Resign → 11`, one past the head. Verify this is reconciled before trusting any freshly written `games_*.safetensors`.

**Exception:** the `aux_*` heads (train.py `AUX_DIMS`: ownership/fog/SPT+5/opp-tech) are training-only and deliberately NOT mirrored in Rust — every Rust backend loads weights by name and ignores the extra keys. Do not add them to `network.rs`, and never save `model.safetensors` from `src/bin/train.rs` (candle `VarMap::save` strips them; it saves to `model_candle.safetensors` instead).

### Training-only environment switches
`train.py` reads several env vars that materially change training and are set by the shell driver, not by any config file. Check these before diagnosing a training result:
- `DETACH_VALUE_TRUNK` — shields the trunk from value-loss gradient (a bisect arm, not a normal setting).
- `VALUE_LOSS_WEIGHT`, `OWNERSHIP_LOSS_WEIGHT` — head weighting.
- `AUGMENT_D4` — D4 symmetry augmentation; implemented, off unless explicitly exported.
- `TRAIN_EPOCHS`, `LEARNING_RATE`, `BATCH_SIZE`.

`bisect_arm.sh` is where diagnostic arms belong; anything exported unconditionally from `run_training_loop.sh` is a production setting.

### Binaries (`polyfish-rs/src/bin/`)
27 binaries; the load-bearing ones:
- `self_play.rs` — generates training games (`--num-games`, `--mcts-iters`, `--tribe1/2`, `--opponent <checkpoint>`, `--anchor-frac`, `--value-trust`, `--reward-shaping`, `--iteration`); emits `METRICS:` JSON lines parsed by the loop script and writes `games_*.safetensors`. Also owns the value-label definition and the curriculum.
- `arena.rs` — battle two configurations head-to-head (`--model1 --model2 --games --mcts --backend1/2`). Plays each seed twice with sides swapped.
- `train.rs` — Rust/candle trainer (alternative to `train.py`).
- `trainer.rs` — interactive CLI to play against the AI and correct its moves.
- Diagnostics: `benchmark.rs`, `actor_ceiling.rs`, `compare_evaluators.rs`, `repro_loop.rs`, `validate_csv.rs`, `stats.rs`, `debug_*.rs`, `verify_*.rs`.
- Replay management: `import_replays.rs`, `upload_replays.rs`, `download_replay.rs`, `delete_all_replays.rs`, `extract_versions.rs`.

Any binary invoked by a shell script forms a **CLI contract with that script**. Nothing in CI checks it — if you rename or remove an argument, grep `run_training_loop.sh` and `auto_train.sh` in the same change.

### Strength measurement
Separate from the training metrics, and the instrument every experiment depends on:
- `arena` plays the matches; `ladder.py` owns `ladder.json` (frozen anchors, gauge readings, freeze/plateau verdicts); `elo.py` computes ratings.
- `run_training_loop.sh` runs a gauge match every `-l` iterations against the ladder's active anchor, records the reading, and can freeze a new anchor (≥80% win rate) or stop the run (plateau).
- `.anchor_state.json` / `.anchor_decay_start` persist anchor-gate state across invocations.
- `arena` seeds from the wall clock and has no seed flag, so readings are not on a common map set — keep that in mind when comparing readings across iterations.

### Data flow
Steam game → `polyfish-mod` (C#) / the C++ reader → JSON game states (`live_game.json`, `replays/`) → loaded by `polyfish` server or the replay subsystem. Separately, `self_play` → `games_*.safetensors` → `train.py` → `model.safetensors` → `checkpoints/`. Training metrics go to `training_log.csv` (canonical store, keyed by `run_id` per training campaign) plus a `moves_by_turn.json` sidecar; `run_training_loop.sh` uses `training_log.py` to parse METRICS and append rows. Live dashboard: `http://localhost:3000/training.html` (Chart.js, reads `/api/runs`, `/api/training-metrics`, `/api/moves-by-turn`, `/api/value-distribution` from the Rust server). `training_metrics_schema.sql` + root `telegram_agent.js`/`run_analysis_now.js` push progress to Supabase/Telegram. `session.log` is a raw debug transcript only.

## Comments

Keep comments strictly minimal. Prefer clear code over commentary — do not narrate what the code obviously does.

Add comments only when they add real value:
- A brief note above a dense or non-obvious block (game-rule edge cases, tricky invariants, performance trade-offs).
- Function docs (what/why, not step-by-step rehash of the body).
- Parameter docs when the name alone is not enough.

Length limits:
- **Inline comments:** one line; two lines is rare and needs a strong reason.
- **Parameter docs:** at most 2 lines each.
- **Function docs:** at most 4 lines total.

Do not add comments for every variable, branch, or trivial operation. Do not restate the code in prose.

## Notes
- `notes.md` and `notes-heuristics.md` document design rationale and the branching-factor analysis (Polytopia has a narrow but very deep per-turn search tree — ~8 plies to complete one game turn — which drives the MCTS depth/iteration choices). Read them before changing search or evaluation behavior. `notes-memory.md` covers the observation-memory channels.
- `hypothesis_driven_improvements.md` is a pre-registered experiment log (EXP 1–11, EXP_ELO_*) with COMMITTED/REJECTED verdicts. Read it before proposing a change — several obvious ideas have already been tried and measured. `expert_review.md` and `expert_boost_throughput.md` hold a prior architecture review and a measured throughput investigation (including a "What NOT to do" section).
- **`expert_pipeline_audit.md` (Aug 2026) is the open-work list — read it first.** It records three shell↔binary contract breaks that stop `run_training_loop.sh` from running at all and that have prevented the strength gauge from ever recording a reading, plus the rest of the audit with per-item status and re-verify commands. Any gauge-derived conclusion in the experiment log predates those breaks being found.
- A verdict recorded in those docs means the experiment ran, not that the code still reflects it. Confirm in the source before relying on it. The reverse also happens: a measured rationale can be lost when a comment is rewritten (see the `AUGMENT_D4` case in the audit) — check `git log -S` on a constant before assuming its current comment is the whole story.
- `main` is the default branch and PRs target it.
