# Minor Bugs (Non-Blocking for Experiment 2)

This document catalogs minor bugs, technical debt, and compiler/Clippy warnings discovered during the engine and neural architecture audit. None of these issues block or impact the ongoing **EXP_ARCH_002** (Exp 2) self-play and training pipeline.

---

## 1. Metal Backend Architecture Drift (`EXP_ARCH_002` Value Head Missing)

- **Severity**: Low / Platform-specific
- **Files**:
  - [`polyfish-rs/src/ai/metal_network.rs`](polyfish-rs/src/ai/metal_network.rs#L645-L666)
  - [`polyfish-rs/examples/metal_parity.rs`](polyfish-rs/examples/metal_parity.rs)
  - [`polyfish-rs/tests/test_backend_weight_keys.py`](polyfish-rs/tests/test_backend_weight_keys.py)
- **Description**:
  The Metal backend (`PolyZeroMetalNet`) was not migrated when the dual-path value head architecture (`EXP_ARCH_002`) was introduced. It lacks `v_spatial_conv` (a 1x1 Conv2d with 2 filters) and still expects `v_fc1` to take `2 * filters = 128` inputs (from pooled mean and max) rather than `dual_path_in_dim = 370` inputs (`242` spatial + `128` pooled).
- **Why It Does NOT Impact Exp 2**:
  Exp 2 training and self-play run exclusively on Linux servers using PyTorch/CUDA and the Candle CPU/CUDA backend (`network.rs`). Metal is an Apple Silicon-specific acceleration path and is not invoked during Linux training.
- **Remediation**:
  Port the 1x1 convolution and 370-dimensional concatenation into the Metal MPSGraph build pipeline in `metal_network.rs`, and update `expected_shapes()` in `metal_parity.rs`.

---

## 2. Behavioral Probe Test Assumes Historical Iteration Checkpoint

- **Severity**: Low / Test Harness
- **Files**:
  - [`polyfish-rs/tests/test_neural_behavioral_probe.rs`](polyfish-rs/tests/test_neural_behavioral_probe.rs#L27-L33)
  - Lines: 174–175, 320–321, 387–388
- **Description**:
  The zero-search tactical interpretability probe test directly executes `load_network("model_iter129.safetensors")` and `load_network("model_dagger_foundation.safetensors")` without verifying whether those checkpoint files exist on disk, panicking if they are missing.
- **Why It Does NOT Impact Exp 2**:
  This file is an offline diagnostic behavioral probe for qualitative tactical inspection across historical checkpoints (iter 76 vs iter 129). It is not run during CI/self-play or neural training.
- **Remediation**:
  Annotate the probe tests with `#[ignore]` so `cargo test` skips them by default unless specifically invoked with `-- --ignored`, or add an early-return guard checking `Path::new(path).exists()`.

---

## 3. Unsigned Zero Comparison Clippy Warnings (`move_count < TEMPERATURE_MOVE_THRESHOLD`)

- **Severity**: Low / Dead Code & Clippy Warning
- **Files**:
  - [`polyfish-rs/src/ai/mcts_zero.rs`](polyfish-rs/src/ai/mcts_zero.rs#L289) (`pub const TEMPERATURE_MOVE_THRESHOLD: usize = 0;`)
  - [`polyfish-rs/src/ai/mcts_zero.rs`](polyfish-rs/src/ai/mcts_zero.rs#L388)
  - [`polyfish-rs/src/ai/gumbel_mcts.rs`](polyfish-rs/src/ai/gumbel_mcts.rs#L394)
  - [`polyfish-rs/src/ai/heuristic_mcts.rs`](polyfish-rs/src/ai/heuristic_mcts.rs#L217)
  - [`polyfish-rs/src/ai/gene_agent.rs`](polyfish-rs/src/ai/gene_agent.rs#L162)
- **Description**:
  `TEMPERATURE_MOVE_THRESHOLD` is statically set to `0: usize`. Comparisons of the form `if move_count < TEMPERATURE_MOVE_THRESHOLD` are always false because an unsigned `usize` cannot be strictly less than zero. Clippy warns:
  `comparison involving the minimum or maximum element for this type contains a case that is always true or always false`.
- **Why It Does NOT Impact Exp 2**:
  Temperature-based sampling in early turns was deliberately disabled in favor of Gumbel root noise exploration. Evaluating the condition to false is the intended runtime behavior; only the lint is raised.
- **Remediation**:
  Either remove the check and conditionally compile/configure temperature moves via a runtime config parameter, or adjust the threshold constant if early-turn temperature exploration is reintroduced.

---

## 4. Duplicate Serde Aliases on Domination Genes

- **Severity**: Low / Deserialization Shadowing
- **Files**:
  - [`polyfish-rs/src/ai/genes.rs`](polyfish-rs/src/ai/genes.rs#L162-L169)
- **Description**:
  In `EvaluatorGenes`, lines 152–159 specify:
  ```rust
  #[serde(alias = "early_eco", default)]
  pub early_perf_eco: f32,
  #[serde(alias = "early_mil", default)]
  pub early_perf_mil: f32,
  #[serde(alias = "early_exp", default)]
  pub early_perf_exp: f32,
  #[serde(alias = "early_fow", default)]
  pub early_perf_fow: f32,
  ```
  Immediately followed on lines 162–169 by the identical aliases on Domination fields:
  ```rust
  #[serde(alias = "early_eco", default)]
  pub early_dom_eco: f32,
  ...
  ```
  Rust generates `warning: unreachable pattern` because Serde matches the first field with that alias (`early_perf_*`), making the second alias unreachable during JSON deserialization.
- **Why It Does NOT Impact Exp 2**:
  Exp 2 self-play and training utilize neural guidance (`GumbelMctsAgent` + `PolyZeroNet`). When genetic evaluators are deserialized, canonical field names (e.g., `"early_dom_eco"`) are used, which do not collide.
- **Remediation**:
  Remove the duplicate `early_eco` aliases from the domination fields or change them to distinct aliases like `"early_dom_eco"`.

---

## 5. Non-Looping `loop` in MCTS Node Traversal

- **Severity**: Low / Code Smells
- **Files**:
  - [`polyfish-rs/src/ai/mcts_zero.rs`](polyfish-rs/src/ai/mcts_zero.rs#L504-L547)
- **Description**:
  A `loop { ... }` construct around root child selection unconditionally exits via `break Some(...)` or `break None`. The loop never executes more than one iteration. Clippy warns:
  `this loop could be written as a block or an if`.
- **Why It Does NOT Impact Exp 2**:
  The logic terminates correctly on all paths and performs the exact desired one-step traversal before handing off to index-based traversal. Exp 2 primarily relies on `gumbel_mcts.rs`.
- **Remediation**:
  Refactor the single-pass block into an `if let` / match block without an enclosing `loop`.

---

## 6. Deprecated `rand` 0.8 / 0.9 Methods in Genetic Mutation

- **Severity**: Low / Compiler Deprecation Warnings
- **Files**:
  - [`polyfish-rs/src/ai/genes.rs`](polyfish-rs/src/ai/genes.rs#L750-L757, #L879-L959)
- **Description**:
  Uses `rand::thread_rng()` (renamed to `rng()`), `rng.r#gen()` (renamed to `random()` to avoid keyword collision in Rust 2024), and `rng.gen_range()` (renamed to `random_range()`).
- **Why It Does NOT Impact Exp 2**:
  These are standard deprecation warnings that do not affect runtime execution or numeric outputs.
- **Remediation**:
  Run `cargo fix --lib -p polyfish` or update the calls to the new method names when bumping the MSRV or Rust edition.

---

## 7. Dead Code & Unused Imports in Binary Utilities

- **Severity**: Low / Hygiene
- **Files**:
  - [`polyfish-rs/src/bin/self_play.rs`](polyfish-rs/src/bin/self_play.rs#L9, #L30, #L50-L57, #L99)
  - [`polyfish-rs/src/bin/extract_versions.rs`](polyfish-rs/src/bin/extract_versions.rs#L4)
  - [`polyfish-rs/src/ai/gumbel_mcts.rs`](polyfish-rs/src/ai/gumbel_mcts.rs#L1330) (`unused_mut`)
- **Description**:
  Unused constants (`NEAR_DELTA_TURNS`, `NEAR_DELTA_NORM_FRAC`, `NEAR_DELTA_NORM_FLOOR`, `NEAR_DELTA_W`), unused imports (`BackendSpec`, `IntoEnumIterator`, `PathBuf`), and an unnecessary `mut` qualifier.
- **Why It Does NOT Impact Exp 2**:
  Unused constants and imports do not generate runtime code and have zero effect on execution semantics.
- **Remediation**:
  Remove dead constants or prefix unused function arguments with `_`.
