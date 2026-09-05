# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Polyfish is an AlphaZero-style AI (MCTS + neural network) that plays *The Battle of Polytopia*. The core is a from-scratch reimplementation of the entire Polytopia game engine — first written in TypeScript, then ported to Rust (`polyfish-rs/`) for performance and training throughput. Everything of consequence lives in `polyfish-rs/`; the root directory is Node-based glue (server launch wrappers, Telegram/Supabase reporting, dashboard) plus three satellite data-capture projects.

## Repository layout

- `polyfish-rs/` — Rust game engine, AI, web backend, and all training code. This is where ~all work happens.
- `src/public/` — static web UI (JS/HTML/CSS), served by the Rust `polyfish` binary at `http://localhost:3000`.
- `polyfish-mod/` — C# mod (BepInEx/PolyMod) that runs inside the real Steam game to auto-play replays and POST captured game states to the local server.
- `polyfish-scraper/` — TypeScript utilities for gathering game data/assets.
- `polyfish-reader/` (referenced as the C++ ripper; invoked via `scan.sh`) — reads live game state out of the running Polytopia process memory.
- `*.py`, `*.sh`, `*.js` at root and in `polyfish-rs/` — training, analysis, and reporting scripts.

## Commands

All `cargo` commands run from `polyfish-rs/`. The root `run-server.sh` `cd`s there for you; `run_training_loop.sh` does **not** — it must be launched with `polyfish-rs/` as the working directory or its first `cargo build` fails with "could not find Cargo.toml".

**Run the web server / simulator** (port 3000, serves `src/public`):
```bash
./run-server.sh                      # from repo root; kills :3000, then `cargo run --bin polyfish`
```
On startup `main.rs` tries to load game state from `live_game.json`, `saved_state.json`, then the newest `replays/mod_replay_*.json`, before falling back to a generated map.

**Build the training binaries** (release is required for any real training):
```bash
cd polyfish-rs && cargo build --release --bin polyfish --bin self_play
```

**Tests** (tool binaries in `src/bin/` set `test = false` in `Cargo.toml`; `#[ignore]` marks heavy integration probes — neither runs in CI):
```bash
cd polyfish-rs && cargo test --lib --tests --bin self_play   # CI-equivalent
cd polyfish-rs && cargo test --test integration my_test_name   # a single libtest case
cd polyfish-rs && cargo run --bin stats -- --games 50   # run a manual diagnostic tool on demand
cd polyfish-rs && cargo test -- --ignored test_min_capital_distance_1v1   # heavy mapgen probe
```

**Python training setup** (creates `polyfish-rs/.venv` from `requirements.txt`):
```bash
cd polyfish-rs && ./local_setup.sh           # or remote_setup.sh on a GPU box
```

**Full self-play + train loop** (the main training driver):
```bash
cd polyfish-rs && ./run_training_loop.sh [-f force-train] [-b boost-threads] [-c chill] [-r disable-reward-shaping] [-i iterations] [-g games-per-iter] [-n mcts-iters] [--reset]
```
This loops: `init_model.py` → `self_play` (Rust, generates `games_*.safetensors`) → `train.py` (Python/PyTorch, updates `model.safetensors`) → log a CSV row → checkpoint every 50 iters into `checkpoints/` → archive consumed games. It also runs "league" matches against random historical checkpoints ~20% of the time. CUDA is opt-in via the `cuda`/`cudnn` Cargo features. Pass `--reset` to delete `model.safetensors` and all self-play game data (`games_*.safetensors` in root and `archive/`) before starting, seeding a brand-new model from scratch; it forces a new run (overrides `--resume`) and leaves `checkpoints/`, `training_log.csv`, and `moves_by_turn.json` untouched.

## Architecture

### Game engine (`polyfish-rs/src/`)
- `states.rs` / `types.rs` — the `GameState` data model and all enums (tribes, units, tech, moves, etc.). `GameState` (de)serializes to/from the JSON produced by the mod and reader.
- `game.rs` — the `Game` controller: load state, run `post_load()` (recompute tile indices/visibility), apply moves, manage turns. The engine is intended to be "perfect": legal-move generation and `execute` should never panic on valid input — panics are treated as bugs to surface, not suppress.
- `moves/` — the `Move` trait and `generate_legal_moves(state)`. Moves split into **economy moves** (city/tech/structure — mostly keyed by `target_index`) and **army moves** (step/attack/capture/abilities — mostly keyed by `src_index`). Unit abilities live in `moves/abilities/`.
- `actions/` — lower-level reusable state mutations (gain stars, exploration, effects) that moves compose, with undo-callback support for MCTS rollouts.
- `settings/` — static game data tables: `units.rs`, `technology.rs`, `structures.rs`, `resources.rs`, `tasks.rs`.
- `mapgen.rs`, `coords.rs`, `fow.rs`, `score.rs`, `hash.rs` — map generation, coordinate/index math, fog-of-war, scoring, state hashing. **Training runs with FOW enabled** (deliberate, to avoid learning cheating/FOW-less strategies).
- `recorder.rs`, `replayer.rs`, `version_sync.rs` — replay capture/playback and tracking of which Polytopia game version (`GameVersion`) the rules target.

### AI (`polyfish-rs/src/ai/`)
- `mcts_zero.rs` — the AlphaZero-style MCTS (`ZeroMctsAgent`) that uses the network for policy+value. `gumbel_mcts.rs` is the Gumbel variant; `heuristic_mcts.rs` is a lightweight network-free MCTS used for fast UI analysis and the interactive trainer.
- `network.rs` — `PolyZeroNet`, the candle (Rust) network: player-state embedding + ResBlocks + cross-attention + a **decomposed policy** and a value head.
- `features.rs` — encodes `GameState` into the input tensor. Key constants: `MAP_SIZE = 11`, `NUM_CHANNELS` (spatial channels), player-state dim. Maps are 11×11.
- `mapper.rs` — `DecomposedMapper` / `DecomposedTargets`: the policy is decomposed into four heads — `action_type` (11), `source_spatial` (H·W), `target_spatial` (H·W), and a unified `move_option` (192, with offset blocks for structures/units/techs/abilities). This decomposition exists because raw legal-move ordering is non-deterministic across states, so moves are mapped to stable semantic coordinates instead of a flat action index.
- `evaluator/` — heuristic state evaluation split by concern: `economy.rs`, `army.rs`, `research.rs`, `exploration.rs`, `gamestate.rs`. Used to shape/guide self-play and for non-NN play.
- `book.rs` — opening-move library; `ordering.rs` — move ordering; `policy_composer.rs` — assembles head outputs into a move distribution; `brain.rs` — top-level agent wiring.

### ⚠️ The dual-network sync constraint
The network architecture is implemented **twice** and the two must stay byte-compatible because they read/write the same `model.safetensors`:
- Rust: `polyfish-rs/src/ai/network.rs` (candle) — used by `self_play`, `arena`, the server, and the Rust `train` binary.
- Python: `polyfish-rs/train.py` (PyTorch) — the primary trainer used by `run_training_loop.sh`; `init_model.py` creates the initial weights from this definition.

If you change layer shapes, channel counts, or head sizes in one, you must mirror it in the other (and in `features.rs` / `mapper.rs` constants). Current values: spatial channels 169 (incl. observation-memory/ghost + per-tile pursuit channel + EXP_ELO_028 goal channels: 3 order planes — one per `OrderKind`, same-kind targets max-merged into one plane — plus 4 stance planes, the fourth being the v7 SAVE stance, Aug 2026), player-state dim 10, map 11×11, policy heads = action(11) + source + target + option(192), normalization = GroupNorm(GN_GROUPS=8) on the 64-filter trunk (no BatchNorm anywhere; the pool convs are fully linear — no norm and no activation, since an unnormed ReLU there dies irreversibly, Jul 2026; the value pool `v_pool_conv` is 8-channel since Jul 2026 to remove the bottleneck, policy pools stay 1-channel). Mismatches surface as safetensors load errors or silent garbage. ⚠️ **The tch and Metal eval backends stub `progress` to 0.0** (`eval_server.rs`) — only the candle path computes it, so anything reading `EvalResult.1` on the fast path is reading a constant. Any new head threaded through `EvalResult` must be plumbed into those two backends too, or it will silently read zero the same way.

Old narrower training data (154 pre-obs-memory, 161 pre-pursuit, 162 pre-goal-channels) is zero-padded at load by train.py (channels are only ever appended at the end of the layout); BatchNorm-era checkpoints are rejected at load by all backends (quarantined in `checkpoints/bn_era/`, usable only with pre-GN binaries).

**Exception:** the `aux_*` heads (train.py `AUX_DIMS`: ownership/fog/SPT+5/opp-tech) are training-only and deliberately NOT mirrored in Rust — **with three deliberate exceptions, all inference-consumed:** `aux_fog` (since Aug 2026 — per-tile P(enemy under fog), read off the post-cross-attention trunk exactly as train.py does, consumed by the v9 risk term); `pi_macro_stance`/`pi_macro_order` (EXP_ELO_061 — macro-mcts's root-prior PUCT bonus, off `v_latent`/the trunk respectively); and `pi_rollout_value` (EXP_ELO_125 — macro-mcts's depth-gated frozen-edge cheap estimator, off `v_latent`, tanh-bounded like `v_win`). All three follow the same pattern: loaded OPTIONALLY (`vs.contains_tensor`) so pre-existing checkpoints gain no new rejection reason, and the corresponding `ValueOutput`/`RawPolicyOutput` field is `Option` so callers branch instead of reading a silent zero. Every other aux head stays training-only — every Rust backend loads weights by name and ignores the extra keys. Do not add them to `network.rs`, and never save `model.safetensors` from `src/bin/train.rs` (candle `VarMap::save` strips them; it saves to `model_candle.safetensors` instead).

### Binaries (`polyfish-rs/src/bin/`)
- `self_play.rs` — generates training games (`--num-games`, `--mcts-iters`, `--tribe1/2`, `--opponent <checkpoint>`, `--no-reward-shaping`, `--td-w`, `--iteration`); writes `.last_self_play_metrics.json` (read by `training_log.py`) and `games_*.safetensors`. Reward shaping (TD(λ) + final-outcome blended value target, `--td-w` weights the blend) is on by default — EXP_ELO_004 (Jul 13, 2026) showed it clearly beats the flat final-outcome-only fallback at matched budget; `--no-reward-shaping` opts out.
- `train.rs` — Rust/candle trainer (alternative to `train.py`).
- `arena.rs` — battle two model checkpoints head-to-head (`--model1 --model2 --games --mcts`).
- `trainer.rs` — interactive CLI to play against the AI and correct its moves.
- `benchmark.rs`, `compare_evaluators.rs`, `repro_loop.rs`, `validate_csv.rs`, `load_json.rs`, `debug_summon.rs` — diagnostics/repro tools.

### Data flow
Steam game → `polyfish-mod` (C#) / `polyfish-reader` (C++) → JSON game states (`live_game.json`, `replays/`) → loaded by `polyfish` server or replayer. Separately, `self_play` → `games_*.safetensors` → `train.py` → `model.safetensors` → `checkpoints/`. Training metrics go to `training_log.csv` (canonical store, keyed by `run_id` per training campaign) plus `moves_by_turn.json` sidecar; `run_training_loop.sh` uses `training_log.py` to read `.last_self_play_metrics.json` and append rows (the old `METRICS:` stdout line no longer exists). Live dashboard: `http://localhost:3000/training.html` (Chart.js, reads `/api/runs`, `/api/training-metrics`, `/api/moves-by-turn`, `/api/value-distribution` from the Rust server). Default loop behavior: each `./run_training_loop.sh` starts a **new run**; pass `--resume` to continue the latest (or `--resume <run_id>`). `training_metrics_schema.sql` + root `telegram_agent.js`/`run_analysis_now.js` push progress to Supabase/Telegram. `session.log` is a raw debug transcript only.

## File size

**Keep every source file under ~1000 lines.** This is long-standing practice
here — `src/ai/reward/mod.rs`, `src/ai/search/gumbel_mcts/mod.rs` and
`src/ai/reward/goal_potential_tests.rs` all cite it as the reason they were
split — but it was never written down, so `self_play.rs` reached 5.8k before
anyone noticed. Three sanctioned ways to split, in order of preference:

- **Directory module + re-export** (`ai/reward/`, `rules/eco_plan/`): `mod.rs`
  keeps the shared vocabulary and re-exports the parts, so every existing
  `crate::ai::reward::X` call site keeps resolving unchanged.
- **Impl-block split** (`ai/search/gumbel_mcts/`): one `impl Type` block per
  file. Rust merges impl blocks across files, so `agent.method()` call sites
  are untouched and no logic moves.
- **Sibling test file** (`goal_potential.rs` + `goal_potential_tests.rs`):
  `#[cfg(test)] #[path = "x_tests.rs"] mod tests;` as a three-line trailer.
  Thorough tests should never be what pushes a module over the limit.

Binaries split the same way: `src/bin/<name>/main.rs` plus siblings is
auto-discovered by Cargo, needs no `Cargo.toml` change, and preserves the
bin's `test` setting.

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
- **`current_understanding.md` is the single source of current truth** for how the AI plays, where it's weak, and why — **read it first**, and before changing search or evaluation behavior. `notes-heuristics.md` is the evaluator/heuristic-design spec. `notes.md` and `hypothesis_driven_improvements.md` are historical/audit logs (append-only): treat only each experiment's *final* verdict as current, and never cite a claim tagged `(SUPERSEDED)` / `⚠️ SUPERSEDED` as current. The branching-factor analysis (Polytopia has a narrow but very deep per-turn search tree — ~8 plies to complete one game turn — which drives the MCTS depth/iteration choices) lives in `notes.md`.
- The active development branch is `verdi`; PRs target `main`.
- **When analyzing model/policy behavior from decision traces or self-play data, evaluate across a whole turn (every ply, turn-start through EndTurn), never a single ply in isolation.** A single ply's candidate set bundles every unit's Step options with whatever economy actions are live — comparing one action type's prior against ~25 competing Step candidates in the same ply measures a popularity contest it was never meant to win. Group captured plies by (game, turn) and reconstruct the whole-turn sequence before concluding a model "never does X."
