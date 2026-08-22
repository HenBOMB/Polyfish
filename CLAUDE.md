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

**Tests** (tool binaries in `src/bin/` set `test = false` in `Cargo.toml`; `#[ignore]` marks heavy probes and measurement tools, which CI does not run):
```bash
# The main gate — note --no-default-features, which is what .github/workflows/rust.yml runs
cd polyfish-rs && cargo test --no-default-features --lib --tests --bin self_play

cd polyfish-rs && cargo test --test integration my_test_name       # a single libtest case
cd polyfish-rs && cargo test -- --ignored test_min_capital_distance_1v1   # heavy mapgen probe
cd polyfish-rs && cargo run --bin stats -- --games 50              # manual diagnostic tool
```

CI (`.github/workflows/rust.yml`) also runs, and these are worth running locally before touching what they cover:
```bash
cd polyfish-rs && cargo test --no-default-features --test parity_widths   # Rust vs train.py head widths
cd polyfish-rs && python3 scripts/check_cli_contract.py                   # shell -> binary/python-CLI flag contract
cd polyfish-rs && cargo clippy --no-default-features --all-targets        # gated on a correctness subset
cd polyfish-rs && cargo fmt --check                                       # advisory
cd polyfish-rs && ./scripts/run_python_tests.sh                          # ladder.py + train.py (stdlib unittest)
cd polyfish-rs && ./scripts/run_forward_parity.sh                        # candle CPU vs train.py on one checkpoint
```
`.github/workflows/smoke.yml` runs `scripts/smoke_train_loop.sh` nightly (and on demand): a real one-iteration `self_play` → `games_*.safetensors` → `train.py` → `model.safetensors` pass plus an `arena` gauge reading, into a scratch dir under `target/smoke/`. That seam is where all three of the 2026 pipeline blockers hid — run it after changing `run_training_loop.sh`, `train.py`, or either binary's CLI. It also forces the anchor-freeze and audit branches, which no reading it can afford would otherwise reach (`GAUGE_FREEZE_WR=0`, `GAUGE_LINK_GAMES=1`, `GAUGE_AUDIT_EVERY=1`; `SMOKE_FORCE_FREEZE=0` turns that off) — those branches had never executed anywhere, and their first run would have been mid-campaign in a fail-fatal loop (#35).

**Python training setup** (creates `polyfish-rs/.venv` from `requirements.txt`):
```bash
cd polyfish-rs && ./local_setup.sh           # macOS/Apple silicon or plain Linux CPU
cd polyfish-rs && ./remote_setup.sh          # generic CUDA box (also installs Rust, builds release)
cd polyfish-rs && ./vast_setup.sh            # Vast.ai; additionally builds tch-eval
```
`requirements.txt` holds the shared, pinned core and **deliberately does not list torch** — each target needs a different wheel index. All three scripts read the single version pin from the `# POLYFISH_TORCH_VERSION=` line in `requirements.txt` and install the matching wheel (PyPI on macOS, CPU index on Linux, cu128 on a GPU box). Raising that pin means re-checking `tch-rs` first: `tch-eval` links against this exact torch.

**Full self-play + train loop** (the main training driver):
```bash
cd polyfish-rs && ./run_training_loop.sh [flags]
```
Short flags (getopts `fbcri:g:n:a:e:l:k:`): `-f` force-train, `-b` boost-threads, `-c` chill, `-r` reward-shaping, `-i` iterations, `-g` games-per-iter, `-n` mcts-iters, `-a` actors, `-e` eval-servers, `-l` league/gauge interval (default 10), `-k` gumbel-k. Long flags: `--resume [run_id]`, `--new-run`/`-N`, `--reset`, `--no-server`.

The loop is: `init_model.py` → `self_play` (Rust, generates `games_*.safetensors`) → `train.py` (Python/PyTorch, updates `model.safetensors`) → log a CSV row → checkpoint every 50 iters into `checkpoints/` → archive consumed games, plus a strength-gauge match every `-l` iterations. A failed gauge reading is now fatal — the loop aborts rather than continuing blind, which is what left an entire campaign without a single recorded reading. CUDA is opt-in via the `cuda`/`cudnn` Cargo features. `--reset` deletes `model.safetensors`, all self-play game data (`games_*.safetensors` in root and `archive/`) and `.anchor_decay_start` before starting; it forces a new run (overrides `--resume`) and leaves `checkpoints/`, `training_log.csv`, and `moves_by_turn.json` untouched. Each run is a **new run** by default; `--resume` continues the latest.

**Head-to-head evaluation:**
```bash
cd polyfish-rs && cargo run --release --bin arena -- --model1 a.safetensors --model2 b.safetensors --games 32 --mcts 64 --seed 20260811
```
Pass `--seed` for anything you intend to compare: without it `arena` seeds from the wall clock and re-rolls the map set every run.

## Architecture

### Game engine (`polyfish-rs/src/`)
- `states.rs` / `types.rs` — the `GameState` data model and all enums (tribes, units, tech, moves, etc.). `GameState` (de)serializes to/from the JSON produced by the mod and reader.
- `game.rs` — the `Game` controller: load state, run `post_load()` (recompute tile indices/visibility), apply moves, manage turns. The engine is intended to be "perfect": legal-move generation and `execute` should never panic on valid input — panics are treated as bugs to surface, not suppress. `play_move` is the real path; `simulate_move` is the in-tree path (it does not set `_are_you_sure`, so the search never reveals fog — deliberate anti-cheating, not a bug). `simulate_move`'s `EndTurn` branch is the one place the two diverge in game semantics: see the adversarial-search switch below.
- `moves/` — the `Move` trait and `generate_legal_moves(state)`. Moves split into **economy moves** (city/tech/structure — mostly keyed by `target_index`) and **army moves** (step/attack/capture/abilities — mostly keyed by `src_index`). Unit abilities live in `moves/abilities/`.
- `actions/` — lower-level reusable state mutations (gain stars, exploration, effects) that moves compose, with undo-callback support for MCTS rollouts.
- `settings/` — static game data tables: `units.rs`, `technology.rs`, `structures.rs`, `resources.rs`, `tasks.rs`.
- `mapgen.rs`, `coords.rs`, `fow.rs`, `memory.rs`, `score.rs`, `hash.rs` — map generation, coordinate/index math, fog-of-war, observation memory, scoring, state hashing. **Training runs with FOW enabled** (deliberate, to avoid learning cheating/FOW-less strategies).
- `replay/` — the replay subsystem (`schema.rs`, `loader.rs`, `executor.rs`, `playback.rs`, `validator.rs`, `recorder.rs`, `training.rs`). Distinct from the top-level `recorder.rs`, which records human/mod steps for imitation data. `version_sync.rs` tracks which Polytopia version (`GameVersion`) the rules target.

### AI (`polyfish-rs/src/ai/`)
- `mcts_zero.rs` — the AlphaZero-style MCTS (`ZeroMctsAgent`). `gumbel_mcts.rs` is the Gumbel variant and the one training actually uses; `heuristic_mcts.rs` is a network-free MCTS for fast UI analysis and the interactive trainer. `mcts_common.rs` holds the shared backup/descent logic; `mcts.rs` and `original_mcts_zero.rs` are older implementations — check whether a change needs to land in more than one.
- **Two backup conventions, deliberately.** `mcts_common::backpropagate_and_remove_virtual_loss` stores each node's value in **its own player's** perspective (used by `mcts_zero`), while `backpropagate_return_with_rewards` stores the action value of the edge into the node, i.e. the **parent's** perspective (used by `gumbel_mcts`). Both are self-consistent; each agent's child-selection rule has to match its own convention — `mcts_zero` negates a handover child before comparing siblings (`effective_value_for_parent`), Gumbel compares `q_value()` directly. Get this wrong and the parent picks the move that is best for the opponent. `mcts_common::edge_hands_over` is the shared "does this edge change the mover" test.
- `brain.rs` — top-level agent wiring. `Brain::with_backend(...)` plus the `with_prior_heuristic_weight` / `with_policy_target_q_weight` / `with_tree_q_weight` builders decide what the agent actually is. `arena.rs` threads all three; both binaries also search a fog-obscured `clone_for_mcts` view. The schedules that set them live in `ai/curriculum.rs` — self-play applies them, `arena` falls back to `CONVERGED_*` for a hand-run with no iteration in hand, and the gauge is passed the iteration's actual values (see below), so a new knob has to be added to the shared module rather than mirrored. `max_turns_ahead` here is the in-tree turn horizon (`MIN_TURNS_AHEAD` / `MAX_TURNS_AHEAD`).
- `network.rs` — `PolyZeroNet`, the candle network: player-state embedding + ResBlocks + cross-attention + a **decomposed policy** and a value head (two outputs: `v_win` and `v_progress`).
- **`v_progress` is trained but deliberately unread by the search.** It is a real head — `network.rs`, `train.py`'s MSE on the `progress` target, the target written by `self_play` — but nothing in `gumbel_mcts`/`mcts_zero` consumes it. It used to be added into `GumbelNode::q_value()` — which reaches both labels, since the root q_value is the TD bootstrap for the value label and `extract_policy_targets` builds π′ from the children's q_values. Since only candle computes the head (tch and metal stub it to 0), that made training data depend on which backend generated it; it also mis-signed the opponent's progress under adversarial search, and, being `0.0` on unexpanded children, biased π′ toward whatever the search expanded. Do not fold it back into Q without implementing it in all backends *and* negating it across handover edges (EXP_LABEL_002; `tests/test_progress_head_not_in_search.rs` fails if you do).
- `features.rs` — encodes `GameState` into the input tensor. Key constants: `MAP_SIZE = 11`, `NUM_CHANNELS`, `RawFeatures::PLAYER_STATE_DIM`. Maps are 11×11.
- `mapper.rs` — `DecomposedMapper` / `DecomposedTargets`: the policy is decomposed into four heads — `action_type`, `source_spatial` (H·W), `target_spatial` (H·W), and a unified `move_option` (192, with offset blocks for structures/units/techs/abilities). This decomposition exists because raw legal-move ordering is non-deterministic across states, so moves are mapped to stable semantic coordinates instead of a flat action index.
- `evaluator/` — heuristic state evaluation split by concern: `economy.rs`, `army.rs`, `research.rs`, `exploration.rs`, `expansion.rs`, `gamestate.rs`, `player.rs`. Used to shape/guide self-play and for non-NN play.
- `reward.rs` — the shared per-move reward used by both TD value labels and reward-aware MCTS backup. `reward::REL_W` is now the **single** relative-vs-absolute constant, read by both the TD body and self_play's final-outcome tail (`self_play.rs`'s separate `FINAL_OUTCOME_REL_W` is gone). It is 1.0 = pure relative, because the backup negates across every player-turn boundary and that is only valid for an antisymmetric value. Read the comment block on the constant before lowering it; `GOOD_BOT_FINAL_SCORE` is the absolute yardstick it would reintroduce.
- `book.rs` — opening-move library; `ordering.rs` — move ordering; `policy_composer.rs` — assembles head outputs into a move distribution; `decision_trace.rs` — search introspection.

### Inference backends
Four implementations read the same `model.safetensors`, selected by Cargo feature:
- `network.rs` (candle) — default, and the only one on non-Apple hardware.
- `tch_network.rs` (`tch-eval`) — libtorch/MPS. Requires PyTorch 2.12.x plus `LIBTORCH_USE_PYTORCH=1`, `LIBTORCH_BYPASS_VERSION_CHECK=1`, and `.venv/bin` on `PATH`; see the comments in `Cargo.toml`.
- `metal_network.rs` (`metal-eval`) — hand-composed MPSGraph, bypassing libtorch's serial MPS dispatch queue. Fastest on Apple silicon.
- `eval_backend.rs` / `eval_server.rs` — the batching layer that fans leaf evaluations across actors.

`examples/tch_parity.rs` and `examples/metal_parity.rs` exist to check backends against each other — run them after any architecture change. Neither runs off Apple hardware; **`polyfish-rs/scripts/run_forward_parity.sh` is the one that does** (candle CPU vs `train.py`'s PyTorch on the same `model.safetensors`, CI job `forward-parity`). Run it after touching `network.rs`, `train.py`, `features.rs` or `mapper.rs`.

⚠️ **candle and strided tensors.** `candle_nn::Linear` on a non-contiguous 3-D input returns wrong values for every batch row after the first — by position, not contents. `network.rs` hit this on the cross-attention query tokens and every batched evaluation on the default backend was corrupted for all but one row (see audit T1). Call `.contiguous()` on anything that reaches a matmul after a `transpose`. A batch-invariance test does **not** catch it; only an oracle outside candle does.

### ⚠️ The multi-implementation sync constraint
The network architecture is implemented in **Rust (candle) and Python (PyTorch)** and must stay byte-compatible because they read/write the same `model.safetensors`:
- Rust: `polyfish-rs/src/ai/network.rs` — used by `self_play`, `arena`, the server, and the Rust `train` binary.
- Python: `polyfish-rs/train.py` — the primary trainer used by `run_training_loop.sh`; `init_model.py` creates the initial weights from this definition.

If you change layer shapes, channel counts, or head sizes in one, you must mirror it in the other (**and** in `tch_network.rs` / `metal_network.rs`, and in `features.rs` / `mapper.rs` constants). Current values: spatial channels **142** (`features.rs` `NUM_CHANNELS`, `train.py` `SPATIAL_CHANNELS`; = 136 + 6 fog-memory channels), player-state dim **16** (`features.rs:216`, `train.py` `PLAYER_STATE_DIM`), map 11×11, 6 ResBlocks on a 64-filter trunk, policy heads = action + source + target + option(192), normalization = GroupNorm(`GN_GROUPS = 8`) — no BatchNorm anywhere; the 1-channel pool convs are fully linear (no norm, no activation, since an unnormed ReLU there dies irreversibly). Mismatches surface as safetensors load errors or silent garbage. Legacy 136-channel training data is zero-padded at load by `train.py` (`pad_spatial`, channels were appended at the end of the layout), as are pre-widening 11-column `action_type` targets; BatchNorm-era checkpoints are rejected by `migrate_model.py:25-30`.

**Resolved trap, kept as context:** `network.rs` used to export `NUM_ACTION_TYPES = 12` for the self-play/replay writers while building `pi_action` with a hardcoded `11` in both `network.rs` and `train.py` — so every target the writers produced was one column wider than the head, because `mapper.rs` maps `MoveType::Resign → 11`. Both sides now derive from the constant (`network.rs:10`, layer at `:227`; `train.py:96`, layer at `:165`), a const assertion keeps `Resign` inside the head (`network.rs:20`), and `tests/parity_widths.rs` fails the build if the Rust and Python widths ever disagree again. Add the same kind of assertion for any new width you introduce. Note `ResignMove` is still never emitted by `generate_legal_moves`, so slot 11 gets no self-play gradient — widening the head did not make resignation learnable.

**Exception:** the `aux_*` heads (train.py `AUX_DIMS`: ownership/fog/SPT+5/opp-tech) are training-only and deliberately NOT mirrored in Rust — every Rust backend loads weights by name and ignores the extra keys. Do not add them to `network.rs`, and never save `model.safetensors` from `src/bin/train.rs` (candle `VarMap::save` strips them; it saves to `model_candle.safetensors` instead).

### Runtime switch: adversarial in-tree search
`game::adversarial_search()` decides whether an in-tree `EndTurn` hands control to the next player (adversarial) or cycles straight back to the mover, deleting the opponent's turn (the legacy single-player behaviour). **Default off.** Enable with `POLYFISH_ADVERSARIAL_SEARCH=1`, `game::set_adversarial_search(true)`, or `arena --adversarial`; it is process-wide and read on every in-tree `EndTurn`, so tests that touch it must serialize. With it on, `clone_for_mcts` also confines the in-tree opponent to the root player's vision, so the opponent it searches against is a belief-state army, not the real one. Nothing in `run_training_loop.sh` sets it — it is an unmeasured arm, registered as EXP_SEARCH_001.

### Training-only environment switches
`train.py` reads several env vars that materially change training and are set by the shell driver, not by any config file. Check these before diagnosing a training result:
- `DETACH_VALUE_TRUNK` — shields the trunk from value-loss gradient (a bisect arm, not a normal setting).
- `VALUE_LOSS_WEIGHT`, `OWNERSHIP_LOSS_WEIGHT` — head weighting.
- `AUGMENT_D4` — D4 symmetry augmentation; implemented, off unless explicitly exported.
- `TRAIN_EPOCHS`, `LEARNING_RATE`, `BATCH_SIZE`.

`bisect_arm.sh` is where diagnostic arms belong; anything exported unconditionally from `run_training_loop.sh` is a production setting.

### Binaries (`polyfish-rs/src/bin/`)
28 binaries; the load-bearing ones:
- `self_play.rs` — generates training games (`--num-games`, `--mcts-iters`, `--tribe1/2`, `--opponent <checkpoint>`, `--anchor-frac`, `--value-trust`, `--reward-shaping`, `--iteration`, `--decay-last-iter`, `--anchor-decay-start`, `--symmetric` (default true), `--opening-temp-moves` (default 8), `--print-curriculum`); emits `METRICS:` JSON lines parsed by the loop script and writes `games_*.safetensors`. Also owns the value-label definition and the curriculum — `--print-curriculum` is how other tools ask for it instead of mirroring its thresholds.
- `arena.rs` — battle two configurations head-to-head (`--model1 --model2 --games --mcts --backend1/2 --seed --max-turns --dump-stats-dir --symmetric --adversarial --tribe1/2`). Plays each seed twice with sides swapped; `--seed` pins the map set so readings are paired, and `--tribe1/2` (default an Imperius mirror) pins the tribes — the swap pairs the tribe alongside the seat, and arena prints the pair it played on a `Tribes:` line so callers record the match rather than their intent.
- `train.rs` — Rust/candle trainer (alternative to `train.py`).
- `trainer.rs` — interactive CLI to play against the AI and correct its moves.
- Diagnostics: `benchmark.rs`, `actor_ceiling.rs`, `compare_evaluators.rs`, `repro_loop.rs`, `validate_csv.rs`, `stats.rs`, `debug_*.rs`, `verify_*.rs`.
- Replay management: `import_replays.rs`, `upload_replays.rs`, `download_replay.rs`, `delete_all_replays.rs`, `extract_versions.rs`.

Any binary invoked by a shell script forms a **CLI contract with that script**, and so does any python CLI beside it (`ladder.py`, `training_log.py`). `scripts/check_cli_contract.py` checks both in CI now (it builds the binaries and diffs every long flag the shell scripts pass against that target's `--help` — per subcommand for the python ones — and fails closed on anything it cannot resolve), but still grep `run_training_loop.sh` and `auto_train.sh` when you rename or remove an argument — three such breaks once stopped the pipeline running at all.

### Strength measurement
Separate from the training metrics, and the instrument every experiment depends on:
- `arena` plays the matches; `ladder.py` owns `ladder.json` (frozen anchors, gauge readings, freeze/plateau verdicts); `elo.py` fits ratings from those readings against the Elo-0 greedy floor.
- `run_training_loop.sh` runs a gauge match every `-l` iterations against the ladder's active anchor, records the reading, and can freeze a new anchor or stop the run (plateau). A freeze needs the reading's Wilson lower bound to clear 80%. The plateau gate is EXP 11's registered rule — pooled window halves flat-or-down **and** least-squares slope ≤ 0, both directional, so a climb the gauge cannot yet *prove* no longer counts as a plateau (#31; the interval-overlap test it replaced struck on every climb below ~12pp). Its window is scoped to the current `run_id` and to the reading budget, which includes `max_turns`; strikes reset on a run change. A failed reading aborts the run.
- The gauge pins its map set (`--seed`) and asks `self_play --print-curriculum` for the iteration's `max_turns` **and its search knobs** (`prior_heuristic_w`, `policy_target_q_w`), passing them to `arena`. Both ramp over a run — prior 0.5 → 0.1 → 0, σ(Q) 0 → 1 — so a gauge on the converged constants graded a searcher self-play never used for the first ~30–53 iterations (#32). The knobs are recorded on each reading but deliberately **not** in `ladder.py`'s `_budget_key`: they change every iteration by design, so keying on them would leave the plateau window permanently empty.
- **The gauge's tribe pair is pinned (`arena --tribe1/--tribe2`, default an Imperius mirror) while self-play trains on config.json's 5-tribe pool.** The pin is variance control — the tribe block effect rivals a campaign's whole measured improvement — but it means every ladder Elo is a statement about Imperius-vs-Imperius play, not about the distribution training optimizes. That scope limit is recorded in `ladder.json`'s `scope` field, and the pair is read back off `arena`'s own `Tribes:` line rather than assumed, so the record can never again disagree with the match it describes (#34: the ladder was handed the shuffled *training* pair for a match `arena` hardcoded to Imperius). A `--kind tribe_audit` row at the audit cadence re-reads the active anchor on the iteration's training pair; it is a cross-check only — excluded from the plateau window (`_gauge_series`) and from the Elo fit (`elo.py`'s `EXCLUDED_KINDS`), since its games share a node pair with the pinned reading and would fold the block effect straight back in.
- **The freeze and audit branches are reached only from `run_training_loop.sh`**, so they are exercised nowhere else: the smoke forces them (above), `tests/test_ladder.py`'s `ShellCommandLineTest` runs `freeze` / `audit-opponents` / `record --kind audit|tribe_audit` as subprocesses with the flags read back off the loop script, and the CLI contract check covers the flags statically. An audit cadence landing on a freeze iteration plays every cross-check against the *outgoing* anchor, so those rows pass `--opponent` explicitly rather than letting `ladder.py` assume the active one (#35).
- `.anchor_state.json` / `.anchor_decay_start` persist anchor-gate state across invocations.
- Both `self_play` and `arena` default `--symmetric` to true, and `run_gauge_match` passes it explicitly (`GAUGE_SYMMETRIC`), so the ladder reads the same map family training generates. **Known gap:** a ~64-game reading still only resolves to about ±12pp — `ladder.json` stores the interval, so use it rather than the point estimate.

### Data flow
Steam game → `polyfish-mod` (C#) / the C++ reader → JSON game states (`live_game.json`, `replays/`) → loaded by `polyfish` server or the replay subsystem. Separately, `self_play` → `games_*.safetensors` → `train.py` → `model.safetensors` → `checkpoints/`. Training metrics go to `training_log.csv` (canonical store, keyed by `run_id` per training campaign) plus a `moves_by_turn.json` sidecar; `run_training_loop.sh` uses `training_log.py` to parse METRICS and append rows. `training_log.csv` and `ladder.json` are **tracked in git** (they are the experiment record); `checkpoints/` is not — `scripts/backup_experiment_record.sh` snapshots all of it to another disk or a remote. Live dashboard: `http://localhost:3000/training.html` (Chart.js, reads `/api/runs`, `/api/training-metrics`, `/api/moves-by-turn`, `/api/value-distribution`, `/api/elo-ladder` from the Rust server). `/api/training-metrics` is served by a header-driven CSV reader in `main.rs` / `bin/dashboard.rs`, so a column added to the CSV reaches the dashboard without a Rust change. `training_metrics_schema.sql` + root `telegram_agent.js`/`run_analysis_now.js` push progress to Supabase/Telegram. `session.log` is a raw debug transcript only.

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
- **`expert_pipeline_audit.md` (Aug 2026) is the open-work list — read it first.** Its "Status — Aug 18, 2026" block is the index: the three shell↔binary contract breaks and the gauge repairs have landed, and each item carries a status plus a re-verify command. **No gauge reading has been taken on the repaired instrument yet**, so every gauge-derived conclusion in the experiment log — the plateau verdict, EXP_ELO_002's "success bar not met" — is provisional pending a re-baseline, and several landed behaviour changes are registered-but-unmeasured.
- A verdict recorded in those docs means the experiment ran, not that the code still reflects it. Confirm in the source before relying on it. The reverse also happens: a measured rationale can be lost when a comment is rewritten (see the `AUGMENT_D4` case in the audit) — check `git log -S` on a constant before assuming its current comment is the whole story.
- `main` is the default branch and PRs target it.
