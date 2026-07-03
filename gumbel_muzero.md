# Gumbel MuZero Search Rewrite — Implementation Plan

## Overview — what and why

**Goal:** Replace the current, partially-broken Gumbel MCTS agent with a correct implementation of [Gumbel MuZero](https://openreview.net/forum?id=bERaNdoegnO) search, so that self-play can reach the same move quality with far fewer simulations per move (target: Gumbel at ~32–64 sims ≈ the existing AlphaZero-style agent at ~200 sims). Fewer sims per move means cheaper self-play, which is the binding constraint on how fast we can generate good training data.

**Why now:** The existing `gumbel_mcts.rs` has real correctness bugs, not just style issues. Two matter most: (1) it selects children with the wrong formula instead of the paper's σ/completed-Q rule, so the search isn't actually Gumbel MuZero; and (2) it physically truncates the root's child list after each halving round, which collapses the exported policy target to ~1–2 actions and feeds degenerate targets to the trainer. Separately, the load-bearing value-backprop sign-flip fix is duplicated byte-for-byte across the two search files and can silently diverge.

**Approach (high level):**

1. Extract the shared, correctness-critical helpers (leaf handling, terminal scoring, and the sign-flip backprop) into a new `mcts_common.rs` so both agents use one copy.
2. Put the Gumbel-specific math (v_mix, completed-Q, σ, the sequential-halving visit schedule) into a pure, unit-testable `gumbel_qtransform.rs`.
3. Rewrite the Gumbel agent around a never-truncated root plus round-robin sequential-halving, keeping NN-call batching.
4. Wire it behind a flag defaulting to the old agent, then validate via arena before changing any defaults.

Everything below is the file-and-line-level detail for executing these four steps.

## 0. Ground truth confirmed this session

- **`PolyZeroNet::forward()`** is literally `forward_t(map, player, false)` (`polyfish-rs/src/ai/network.rs:296-302`) — so bug #7 in the brief (Gumbel using `forward` instead of `forward_t(..., false)`) is cosmetically inconsistent but not a correctness bug. Still worth fixing for uniformity when the file is rewritten anyway.
- **`Brain`** (`polyfish-rs/src/ai/brain.rs`) has no backend field; `think` is hard-typed to return `Option<ZeroMctsAgent<'_>>`; `arena.rs` and `self_play.rs` both construct agents directly/via `Brain::new`, no backend switch exists anywhere.
- **`MoveVisit.visits: f32`** (`polyfish-rs/src/ai/mcts_types.rs`) is consumed only by `DecomposedMapper::move_visit_to_targets` (`polyfish-rs/src/ai/mapper.rs:128-156`), which reads only move-identity fields, never `visits` itself — confirmed fully decoupled, safe to populate with any nonnegative weight.
- **`self_play.rs`'s visit-aggregation loop** (`polyfish-rs/src/bin/self_play.rs` ~178-228) sums `mv.visits` per decomposed target index and normalizes by `total_visits` — confirmed backend-agnostic.
- **`policy_composer.rs`** exposes both:
  - `compute_move_priors(...) -> Vec<f32>` (normalized probs, softmax per head)
  - `compute_move_log_probs(...) -> Vec<f32>` (log-probs, log-softmax per head)

  Gumbel needs raw logits summed across heads pre-softmax for its transform; need to verify/decide whether `compute_move_log_probs` gives combined per-move log-probability (sum of head log-probs) suitable as `logit(a)`, or whether a new `compute_move_logits` is needed that returns unnormalized/pre-softmax scores. This must be checked in Step (a) below — if `compute_move_log_probs` already sums per-head log-softmax outputs (which behave like a valid joint log-probability, itself usable as "logit" in the Gumbel-top-k sense per the paper, since Gumbel-max works with any monotonic transform of the categorical log-probabilities), it can be reused as-is; no need for a separate raw-logit head extraction. This is a scoping question for the implementer, not fully resolved by the read-only exploration I could do.

- **`CLAUDE.md`** confirms the dual-network-sync constraint is about `network.rs` / `train.py` byte-compatibility (layer shapes, channel counts, head sizes) — this Gumbel rewrite is purely a search-time consumer of existing `PolicyOutput`/`ValueOutput`; it does not touch `network.rs`, `features.rs`, `mapper.rs` constants, or `train.py`. Confirmed zero risk to the sync constraint, and this should be stated explicitly in the PR description of every step below so reviewers don't go looking for a Python-side change.

---

## 1. Module structure

Create `polyfish-rs/src/ai/mcts_common.rs`, and add `pub mod mcts_common;` next to the existing `pub mod mcts_zero;` / `pub mod gumbel_mcts;` in `polyfish-rs/src/ai/mod.rs`.

### Move here (shared, byte-identical behavior for both agents)

- **`pub(crate) struct LeafData`** — `{ path_indices, path_players, features, legal_moves, map_size, terminal_value }` — structurally identical in both files today; unify into one definition.
- **`pub(crate) fn compute_terminal_outcome(game: &Game) -> f32`** — extract the score-comparison block duplicated in `mcts_zero.rs:614-634` and `gumbel_mcts.rs:638-658` verbatim.
- **`pub(crate) fn extract_leaf_data(...)`** — the "at the leaf, before undo" extraction block (terminal / needs_expansion / horizon three-way branch), currently duplicated near-verbatim in both `select_and_extract_leaf` implementations.
- **`pub(crate) trait BackpropNode`** + **`pub(crate) fn backpropagate_and_remove_virtual_loss`** — captures the player-aware sign-flip fix exactly once. `ZeroNode` and `GumbelNode` each implement `BackpropNode` with trivial field accessors. This is the single highest-value unification: right now the sign-flip logic (the "recent correctness fix" called out as load-bearing in the brief) is duplicated byte-for-byte in `mcts_zero.rs:725-778` and `gumbel_mcts.rs:724-778` — a future fix to one and not the other is a real risk today.
- **`pub(crate) fn get_node_by_path` / `get_node_by_path_mut`** — generic over any node type with a `children()`/`children_mut()` accessor (via a small `TreeNode` trait, or just duplicate — these are ~6 lines each, low priority to unify vs. the backprop fn).
- **`pub(crate) const VIRTUAL_LOSS: f32 = 1.0`** and **`pub(crate) const DEFAULT_BATCH_SIZE: usize = 24`** — currently magic-numbered separately in each file.

### Do NOT unify (stay Gumbel-specific in the rewritten `gumbel_mcts.rs`)

- The σ/completed-Q transform (`sigma_completed_q`, `compute_v_mix`) — Zero has no analogue.
- The round-robin Sequential-Halving root controller — structurally nothing like Zero's flat PUCT loop.
- Interior (non-root) selection formula (`softmax(logit + sigma(Q))` then `probs(a) - visits(a)/(1+sum visits)`) — different from Zero's PUCT.
- Policy-target extraction over the full legal set via π' — Zero's target is raw visit counts, structurally simpler.
- **`GumbelNode` itself** (extra logit/gumbel fields vs Zero's prior) — do not try to force a shared Node struct.

  Zero's `expand_node_single` / `expand_node_from_network_output` / `expand_node_from_precomputed` are prior-normalization-specific — instead write Gumbel-specific `expand_gumbel_node_from_precomputed(...)` in `gumbel_mcts.rs` that calls `compute_move_log_probs` instead of `compute_move_priors` and skips the sum-normalization step.

### Where the σ/completed-Q code lives

New private submodule inside `gumbel_mcts.rs`, e.g. a `mod qtransform { ... }` block, **or** a sibling file `polyfish-rs/src/ai/gumbel_qtransform.rs` if the implementer wants it independently unit-testable.

**Recommendation:** `polyfish-rs/src/ai/gumbel_qtransform.rs`, `pub(crate)` visibility, `pub mod gumbel_qtransform;` added to `ai/mod.rs`.

---

## 2. Exact signatures

### 2.1 `gumbel_qtransform.rs` (new file, pure functions, no Game/Move/tree dependency)

```rust
/// Constants from mctx qtransform_completed_by_mix_value defaults.
pub const C_VISIT: f32 = 50.0;      // maxvisit_init
pub const C_SCALE: f32 = 0.1;       // value_scale
pub const EPSILON: f32 = 1e-8;

/// v_mix at a node: imputed value used for Q of unvisited children.
/// `raw_value` = this node's own NN value prediction (this node's perspective).
/// `child_visit_counts[i]`, `child_priors[i]`, `child_qvalues[i]` all same length,
/// one entry per child of this node. `child_qvalues[i]` is meaningless/unused when
/// `child_visit_counts[i] == 0.0` (not read in that branch).
pub fn compute_v_mix(
    raw_value: f32,
    child_priors: &[f32],
    child_qvalues: &[f32],
    child_visit_counts: &[f32],
) -> f32 {
    let sum_visits: f32 = child_visit_counts.iter().sum();
    if sum_visits == 0.0 {
        return raw_value;
    }
    let sum_probs_visited: f32 = child_priors.iter().zip(child_visit_counts)
        .filter(|(_, &v)| v > 0.0)
        .map(|(p, _)| p)
        .sum();
    let weighted_q: f32 = if sum_probs_visited > EPSILON {
        child_priors.iter().zip(child_qvalues).zip(child_visit_counts)
            .filter(|((_, _), &v)| v > 0.0)
            .map(|((p, q), _)| p * q / sum_probs_visited)
            .sum()
    } else {
        0.0
    };
    (raw_value + sum_visits * weighted_q) / (sum_visits + 1.0)
}

/// Completed Q per child: real Q if visited, else v_mix.
pub fn compute_completed_qvalues(
    child_qvalues: &[f32],
    child_visit_counts: &[f32],
    v_mix: f32,
) -> Vec<f32> {
    child_qvalues.iter().zip(child_visit_counts)
        .map(|(&q, &v)| if v > 0.0 { q } else { v_mix })
        .collect()
}

/// Optional min-max rescale to [0,1] over the given values.
pub fn rescale_min_max(values: &[f32]) -> Vec<f32> {
    let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if (max - min).abs() < EPSILON {
        return vec![0.0; values.len()];
    }
    values.iter().map(|&v| (v - min) / (max - min)).collect()
}

/// sigma(completed_Q) = (C_VISIT + max_visit) * C_SCALE * completed_Q
pub fn sigma(completed_q: &[f32], max_child_visits: f32) -> Vec<f32> {
    let scale = (C_VISIT + max_child_visits) * C_SCALE;
    completed_q.iter().map(|&q| scale * q).collect()
}

/// Convenience: full pipeline used by both root-recommendation and interior selection.
/// rescale=true reproduces mctx's default `rescale_values=True`.
pub fn sigma_completed_q(
    raw_value: f32,
    child_priors: &[f32],
    child_qvalues: &[f32],
    child_visit_counts: &[f32],
    rescale: bool,
) -> Vec<f32> {
    let v_mix = compute_v_mix(raw_value, child_priors, child_qvalues, child_visit_counts);
    let mut completed = compute_completed_qvalues(child_qvalues, child_visit_counts, v_mix);
    if rescale {
        completed = rescale_min_max(&completed);
    }
    let max_visits = child_visit_counts.iter().cloned().fold(0.0f32, f32::max);
    sigma(&completed, max_visits)
}
```

> **Note:** `raw_value` at the root is the root's own NN value prediction (captured at root expansion time and stored on `GumbelNode`/`GumbelMctsAgent` local state — Zero doesn't currently store this anywhere, Gumbel will need to). At interior nodes, `raw_value` is that node's own NN value prediction from when it was expanded — so `GumbelNode` needs a new field `own_value: f32` populated at expansion time.

### 2.2 `gumbel_mcts.rs` — `GumbelNode`

```rust
struct GumbelNode {
    visits: f32,
    value_sum: f32,
    logit: f32,
    gumbel: f32,        // 0.0 for non-root nodes (unchanged from today)
    own_value: f32,     // NEW: this node's own NN value prediction at expansion time
    children: Vec<GumbelNode>,
    move_to_here: Option<Box<dyn Move>>,
    is_expanded: bool,
    virtual_loss: RefCell<f32>,
}

impl GumbelNode {
    fn new(logit: f32, gumbel: f32, move_to_here: Option<Box<dyn Move>>) -> Self { ... }
    fn q_value(&self) -> f32 { ... } // unchanged: value_sum/visits, 0.0 if unvisited
}
```

### 2.3 `GumbelMctsAgent` constructor — fix param-not-field bug

```rust
pub struct GumbelMctsAgent<'a> {
    pub network: &'a PolyZeroNet,
    pub iterations: usize,
    pub k: usize,
    pub batch_size: usize,
    pub c_visit: f32,   // exposed for tuning/testing, default C_VISIT
    pub c_scale: f32,   // exposed for tuning/testing, default C_SCALE
}

impl<'a> GumbelMctsAgent<'a> {
    pub fn new(network: &'a PolyZeroNet, iterations: usize, k: usize) -> Self {
        Self {
            network,
            iterations,
            k,
            batch_size: mcts_common::DEFAULT_BATCH_SIZE,
            c_visit: gumbel_qtransform::C_VISIT,
            c_scale: gumbel_qtransform::C_SCALE,
        }
    }
}
```

This directly fixes the brief's complaint that `k` is a post-construction field mutation — `new()` now takes it as a real parameter, matching how `ZeroMctsAgent::new` takes `iterations`.

### 2.4 Root sequential-halving controller

Replaces the three near-duplicate `select_move*` bodies.

```rust
/// Runs the full root-level search: initial top-k Gumbel cut, then Sequential
/// Halving rounds with round-robin equal-visit allocation, batched leaf collection
/// per round-step. Returns the fully-populated root (all initially-sampled top-k
/// children present with their final visit counts; eliminated candidates keep
/// their partial visit counts, they are NOT removed from `root.children`).
fn run_search(&self, game: &mut Game, root_value: f32, root: &mut GumbelNode, start_turn: i32) {
    let considered_visits_table = sequence_of_considered_visits(self.k, self.iterations);
    let mut num_considered = self.k.min(root.children.len());

    for (round_idx, round_considered, visits_this_round) in considered_visits_table {
        if round_considered <= 1 { break; }
        self.rerank_root_children(root, round_idx);
        self.run_round_robin_round(game, root, round_considered, visits_this_round, start_turn);
    }
}
```

`sequence_of_considered_visits(k, num_simulations)` implements the mctx table exactly:

```rust
fn sequence_of_considered_visits(max_num_considered: usize, num_simulations: usize) -> Vec<(usize, usize, usize)> {
    if max_num_considered <= 1 {
        return vec![(0, max_num_considered.max(1), num_simulations)];
    }
    let log2max = (max_num_considered as f32).log2().ceil() as usize;
    let mut num_considered = max_num_considered;
    let mut rounds = Vec::new();
    let mut round_idx = 0;
    let mut total_assigned = 0usize;
    while total_assigned < num_simulations && num_considered >= 2 {
        let extra = ((num_simulations / (log2max.max(1) * num_considered)).max(1));
        rounds.push((round_idx, num_considered, extra));
        total_assigned += extra * num_considered;
        num_considered = (num_considered / 2).max(2);
        round_idx += 1;
        if round_idx > 2 * log2max + 4 { break; }
    }
    rounds
}
```

> **Implementer TODO:** Unit-test directly against hand-computed small cases (e.g. `k=4`, `num_simulations=16`). Port mctx's `get_sequence_of_considered_visits` test cases as Rust unit tests.

### 2.5 Interior (non-root) selection formula

Only called for depth ≥ 1 (i.e., current node is NOT the root). No Gumbel term.

```rust
fn select_child_interior(&self, node: &GumbelNode) -> Option<usize> {
    let n = node.children.len();
    if n == 0 { return None; }
    let child_qvalues: Vec<f32> = node.children.iter().map(|c| c.q_value()).collect();
    let child_visits: Vec<f32> = node.children.iter().map(|c| c.effective_visits()).collect();
    let child_priors: Vec<f32> = node.children.iter().map(|c| c.logit).collect();
    let sigma_q = gumbel_qtransform::sigma_completed_q(
        node.own_value, &child_priors, &child_qvalues, &child_visits, true,
    );
    let combined: Vec<f32> = child_priors.iter().zip(&sigma_q).map(|(l, s)| l + s).collect();
    let probs = softmax(&combined);
    let sum_visits: f32 = child_visits.iter().sum();
    (0..n).max_by(|&a, &b| {
        let score_a = probs[a] - child_visits[a] / (1.0 + sum_visits);
        let score_b = probs[b] - child_visits[b] / (1.0 + sum_visits);
        score_a.partial_cmp(&score_b).unwrap()
    })
}
```

This fully replaces `select_child_in_tree`'s `argmax(logit + effective_value)` (bug #1 in the brief).

### 2.6 Root selection during search (per-simulation-step, within a round-robin round)

```rust
fn select_root_child_for_step(
    &self, root: &GumbelNode, round_considered: usize, target_visit_this_step: f32,
) -> Option<usize> {
    let child_qvalues: Vec<f32> = root.children.iter().map(|c| c.q_value()).collect();
    let child_visits: Vec<f32> = root.children.iter().map(|c| c.effective_visits()).collect();
    let child_priors: Vec<f32> = root.children.iter().map(|c| c.logit).collect();
    let sigma_q = gumbel_qtransform::sigma_completed_q(
        root.own_value, &child_priors, &child_qvalues, &child_visits, true,
    );
    (0..round_considered.min(root.children.len()))
        .filter(|&i| (root.children[i].visits - target_visit_this_step).abs() < 0.5)
        .max_by(|&a, &b| {
            let score_a = root.children[a].gumbel + child_priors[a] + sigma_q[a];
            let score_b = root.children[b].gumbel + child_priors[b] + sigma_q[b];
            score_a.partial_cmp(&score_b).unwrap()
        })
}
```

### 2.7 Final move recommendation (after search loop ends)

```rust
fn recommend_final_move(&self, root: &GumbelNode) -> usize {
    let max_visit = root.children.iter().map(|c| c.visits).fold(0.0f32, f32::max);
    let child_qvalues: Vec<f32> = root.children.iter().map(|c| c.q_value()).collect();
    let child_visits: Vec<f32> = root.children.iter().map(|c| c.visits).collect();
    let child_priors: Vec<f32> = root.children.iter().map(|c| c.logit).collect();
    let sigma_q = gumbel_qtransform::sigma_completed_q(
        root.own_value, &child_priors, &child_qvalues, &child_visits, true,
    );
    root.children.iter().enumerate()
        .filter(|(i, c)| (c.visits - max_visit).abs() < 0.5)
        .max_by(|(a, ca), (b, cb)| {
            let sa = ca.gumbel + child_priors[*a] + sigma_q[*a];
            let sb = cb.gumbel + child_priors[*b] + sigma_q[*b];
            sa.partial_cmp(&sb).unwrap()
        })
        .map(|(i, _)| i)
        .unwrap_or(0)
}
```

### 2.8 Public API — matching Zero's signature exactly

```rust
pub fn select_move(&self, game: &mut Game) -> Option<Box<dyn Move>>;

pub fn select_move_with_decomposed_visits(
    &self,
    game: &mut Game,
    move_count: usize,
) -> (Option<Box<dyn Move>>, Vec<crate::ai::mcts_types::MoveVisit>);

pub fn select_move_with_stats(&self, game: &mut Game) -> (Option<Box<dyn Move>>, Vec<f32>);
```

All three share one internal core:

```rust
fn search_and_extract(&self, game: &mut Game) -> GumbelNode { ... }
```

`select_move_with_decomposed_visits` calls `search_and_extract`, then builds `Vec<MoveVisit>` via the π' formula (§5), then picks the final action via `recommend_final_move` (§2.7) — **not** via the π' weights and not via temperature-sampled `WeightedIndex` over visits.

**Decision:** Do NOT apply Zero's `TEMPERATURE_MOVE_THRESHOLD`/`WeightedIndex` temperature-sampling logic to Gumbel.

**Justification:** Gumbel's exploration is already stochastic by construction (fresh Gumbel(0,1) noise every call). Layering `WeightedIndex` sampling over visits would double up exploration mechanisms and frequently pick candidates that Sequential Halving explicitly demoted.

Recommendation: bypass temperature sampling entirely; keep `move_count` parameter for signature compatibility only (`let _ = move_count;` with a comment).

### 2.9 Policy target extraction (π')

See section 5 for the algorithm.

```rust
fn extract_policy_targets(&self, root: &GumbelNode) -> Vec<crate::ai::mcts_types::MoveVisit>;
```

---

## 3. Batching implications — round-robin-aware Phase 1/2/3

Zero's `parallel_search_batch` (`mcts_zero.rs:370-494`) collects `batch_size` (24) leaves via repeated independent tree descents — there's no structural grouping by candidate. This doesn't map cleanly onto Sequential Halving's round-robin constraint.

**Design:** batch within a round, across that round's `round_considered` candidate subtrees, one NN call per "wave."

```rust
fn run_round_robin_round(
    &self, game: &mut Game, root: &mut GumbelNode,
    round_considered: usize, visits_per_candidate: usize, start_turn: i32,
) {
    let total_leaves_needed = round_considered * visits_per_candidate;
    let mut collected_per_candidate = vec![0usize; round_considered];
    let mut total_collected = 0;

    while total_collected < total_leaves_needed {
        let mut leaves: Vec<(usize, mcts_common::LeafData)> = Vec::with_capacity(self.batch_size);

        'wave: loop {
            let mut made_progress = false;
            for cand_idx in 0..round_considered {
                if leaves.len() >= self.batch_size { break 'wave; }
                if collected_per_candidate[cand_idx] >= visits_per_candidate { continue; }
                if let Some(leaf) = self.select_and_extract_leaf_under_candidate(
                    root, cand_idx, game, /* turn_horizon */ ..., &device
                ) {
                    leaves.push((cand_idx, leaf));
                    collected_per_candidate[cand_idx] += 1;
                    made_progress = true;
                }
            }
            if !made_progress { break 'wave; }
        }
        if leaves.is_empty() { break; }
        total_collected += leaves.len();

        // Phase 2: batched NN call (same pattern as Zero)
        let values = self.batched_evaluate_and_expand(root, &leaves);

        // Phase 3: backprop via shared mcts_common helper
        for ((_, leaf), value) in leaves.iter().zip(values.iter()) {
            mcts_common::backpropagate_and_remove_virtual_loss(
                root, &leaf.path_indices, &leaf.path_players,
                mcts_common::VIRTUAL_LOSS, *value,
            );
        }
    }
}
```

`select_and_extract_leaf_under_candidate` descends starting from `root.children[cand_idx]` and below uses `select_child_interior` (§2.5) at every subsequent level.

> **Note on `own_value`:** Each root candidate subtree gets its NN value prediction from the same batched `forward_t` call — store into `own_value` on that node exactly like a normal leaf expansion.

---

## 4. Wiring into Brain / `self_play.rs`

### 4.1 Enum vs trait object — recommend enum

```rust
// polyfish-rs/src/ai/brain.rs
use crate::ai::gumbel_mcts::GumbelMctsAgent;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchBackend { Zero, Gumbel { k: usize } }

impl Default for SearchBackend {
    fn default() -> Self { SearchBackend::Zero }
}

pub struct Brain<'a> {
    pub network: &'a PolyZeroNet,
    pub max_iterations: usize,
    pub backend: SearchBackend,   // NEW field
}

enum SearchAgent<'a> {
    Zero(ZeroMctsAgent<'a>),
    Gumbel(GumbelMctsAgent<'a>),
}

impl<'a> SearchAgent<'a> {
    fn select_move_with_decomposed_visits(&self, game: &mut Game, move_count: usize)
        -> (Option<Box<dyn Move>>, Vec<MoveVisit>) {
        match self {
            SearchAgent::Zero(a) => a.select_move_with_decomposed_visits(game, move_count),
            SearchAgent::Gumbel(a) => a.select_move_with_decomposed_visits(game, move_count),
        }
    }
    fn select_move_with_stats(&self, game: &mut Game) -> (Option<Box<dyn Move>>, Vec<f32>) {
        match self {
            SearchAgent::Zero(a) => a.select_move_with_stats(game),
            SearchAgent::Gumbel(a) => a.select_move_with_stats(game),
        }
    }
}

impl<'a> Brain<'a> {
    pub fn new(network: &'a PolyZeroNet, max_iterations: usize) -> Self {
        Self { network, max_iterations, backend: SearchBackend::default() }
    }
    pub fn with_backend(network: &'a PolyZeroNet, max_iterations: usize, backend: SearchBackend) -> Self {
        Self { network, max_iterations, backend }
    }

    fn think(&'_ self, game: &Game) -> (Option<SearchAgent<'_>>, Vec<Box<dyn Move>>) {
        let moves = generate_legal_moves(&game.state);
        if moves.len() == 1 { return (None, moves); }
        let agent = match self.backend {
            SearchBackend::Zero => SearchAgent::Zero(ZeroMctsAgent::new(self.network, self.max_iterations)),
            SearchBackend::Gumbel { k } => SearchAgent::Gumbel(GumbelMctsAgent::new(self.network, self.max_iterations, k)),
        };
        (Some(agent), moves)
    }

    pub fn think_decomposed(&self, game: &Game, move_count: usize) -> (Option<Box<dyn Move>>, Vec<MoveVisit>) {
        let (agent, mut moves) = self.think(game);
        if agent.is_none() { return (moves.pop(), Vec::new()); }
        agent.unwrap().select_move_with_decomposed_visits(&mut game.clone(), move_count)
    }

    pub fn think_with_stats(&self, game: &Game) -> (Option<Box<dyn Move>>, Vec<f32>) {
        let (agent, mut moves) = self.think(game);
        if agent.is_none() { return (moves.pop(), Vec::new()); }
        agent.unwrap().select_move_with_stats(&mut game.clone())
    }
}
```

This keeps `Brain::new` backward-compatible — existing call sites in `self_play.rs:120-121` keep compiling unchanged, defaulting to `SearchBackend::Zero`.

### 4.2 `self_play.rs` — new CLI flags

```rust
/// Search backend to use for MCTS.
#[arg(long, value_enum, default_value_t = SearchBackendArg::Zero)]
search_backend: SearchBackendArg,

/// Gumbel: number of initial top-k candidates sampled at the root.
#[arg(long, default_value_t = 16)]
gumbel_k: usize,

#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum SearchBackendArg { Zero, Gumbel }
```

Construction site becomes:

```rust
let backend = match args.search_backend {
    SearchBackendArg::Zero => SearchBackend::Zero,
    SearchBackendArg::Gumbel => SearchBackend::Gumbel { k: args.gumbel_k },
};
let agent1 = Brain::with_backend(network1, mcts_iters, backend);
let agent2 = Brain::with_backend(network2, mcts_iters, backend);
```

Add `search_backend: SearchBackend` as one more parameter to `play_single_game` alongside `mcts_iters`.

### 4.3 `arena.rs` — future Zero-vs-Gumbel validation

Add mirrored flags:

```rust
#[arg(long, value_enum, default_value_t = SearchBackendArg::Zero)]
backend1: SearchBackendArg,
#[arg(long, value_enum, default_value_t = SearchBackendArg::Zero)]
backend2: SearchBackendArg,
#[arg(long, default_value_t = 16)]
gumbel_k: usize,
```

Replace hardcoded `ZeroMctsAgent::new(...)` with a local `make_agent(...)` returning `SearchAgent`. Deferred to Step (d) in the sequencing plan (§7).

---

## 5. Policy target extraction (π')

Confirmed: this only needs a single root-level `v_mix` computation once search ends — not per-round. The formula π'(a) ∝ exp(logit(a) + sigma(completed_Q(a))) is evaluated once, over the full legal move set at the root.

### 5.1 Structural change: never truncate `root.children`

The current buggy code (`gumbel_mcts.rs:181-183, 278`) calls `root.children.truncate(kept_count)` after each halving round — this is bug #4 and the direct cause of "policy target support collapses to ~1-2 actions."

**The rewrite must never truncate.** Instead, sequential halving tracks an `active` mask (or uses `round_considered` as a prefix-length index). All k initially top-k-cut candidates remain present in `root.children` at the end of search.

### 5.2 Two categories of un/under-visited actions requiring v_mix imputation

1. **Legal moves outside the initial top-k Gumbel cut** — never became a `GumbelNode` at all. Need to be represented in the π' output but have no visit-count/own-Q.
2. **In-cut candidates eliminated by halving** — have a `GumbelNode` with visits > 0 but stopped receiving new visits. Since visits > 0, they use their real accumulated Q, not v_mix.

### 5.3 Algorithm

```rust
fn extract_policy_targets(
    &self,
    root: &GumbelNode,
    all_legal_moves_at_root: &[Box<dyn Move>],
    root_map_size: usize,
) -> Vec<MoveVisit> {
    let child_qvalues: Vec<f32> = root.children.iter().map(|c| c.q_value()).collect();
    let child_visits: Vec<f32> = root.children.iter().map(|c| c.visits).collect();
    let child_priors: Vec<f32> = root.children.iter().map(|c| c.logit).collect();

    let v_mix = gumbel_qtransform::compute_v_mix(
        root.own_value, &child_priors, &child_qvalues, &child_visits,
    );
    let completed_q_in_cut = gumbel_qtransform::compute_completed_qvalues(
        &child_qvalues, &child_visits, v_mix,
    );

    let mut targets = Vec::with_capacity(all_legal_moves_at_root.len());
    let mut raw_scores = Vec::with_capacity(all_legal_moves_at_root.len());
    for m in all_legal_moves_at_root {
        let (logit, completed_q) = match find_in_cut_match(root, m) {
            Some(idx) => (child_priors[idx], completed_q_in_cut[idx]),
            None => (lookup_full_legal_logit(m), v_mix),
        };
        raw_scores.push(logit + gumbel_qtransform::sigma(
            &[completed_q],
            child_visits.iter().cloned().fold(0.0, f32::max),
        )[0]);
    }
    let probs = softmax(&raw_scores); // pi'(a)

    for (m, &p) in all_legal_moves_at_root.iter().zip(probs.iter()) {
        targets.push(MoveVisit {
            move_type: m.move_type(),
            visits: p, // semantically pi'(a) — safe per mapper.rs analysis
            source_idx: m.source_idx().ok(),
            target_idx: m.target_idx().ok(),
            structure_type: m.structure_type().ok(),
            unit_type: m.unit_type().ok(),
            tech_type: m.tech_type().ok(),
            ability_type: m.ability_type().ok(),
        });
    }
    targets
}
```

**Key implementation requirement:** The initial root expansion must retain logits for the full legal move set, not just the top-k slice.

**Recommended data model:**

- `root.children` holds ALL legal moves' `GumbelNode`s (full legal-move count, no k-truncation of storage)
- Separate `in_cut: Vec<usize>` (length k, sorted by initial logit+gumbel) drives §2.6/§3's root-selection/round-robin logic

This avoids the two-list-reconciliation problem entirely. §5.3 simplifies to: iterate `root.children`, for `i in in_cut` use real Q, for `i not in in_cut` use v_mix.

---

## 6. Testing plan

### 6.1 Pure-function unit tests (no Game, no network)

In `polyfish-rs/src/ai/gumbel_qtransform.rs` (`#[cfg(test)] mod tests`):

- `test_v_mix_no_visits_returns_raw_value`
- `test_v_mix_hand_computed`
- `test_completed_q_uses_real_q_when_visited`
- `test_completed_q_imputes_v_mix_when_unvisited`
- `test_rescale_min_max_degenerate_single_visited_child`
- `test_sigma_matches_formula`

In `gumbel_mcts.rs` or `mcts_common.rs` (or `tests/test_sequential_halving.rs`):

- `test_sequence_of_considered_visits_k4` — port mctx's worked examples
- `test_round_robin_allocation_is_equal_per_candidate`
- `test_sequential_halving_toy_2_action`
- `test_sequential_halving_toy_3_action`
- `test_policy_target_sums_to_one`
- `test_policy_target_covers_full_legal_set` — direct regression test for bug #4

### 6.2 Structural-invariant tests mirroring `test_value_backprop.rs`

New `polyfish-rs/tests/test_gumbel_value_backprop.rs`:

- Undo fully restores `game.state.settings.current_player_turn_id` after a full `select_move_with_decomposed_visits` call
- Terminal-state score→outcome conversion (via unified `compute_terminal_outcome`)
- Player-sequence-tracking preconditions for sign-flip logic
- Numeric-correctness test on backprop sign via `backpropagate_and_remove_virtual_loss` directly (testable once in `mcts_common.rs`)

### 6.3 Smoke tests (replace/extend existing `test_gumbel_mcts.rs`)

- Fix struct-literal construction to use `GumbelMctsAgent::new(&network, iterations, k)`
- Keep `best_move.is_some()` as baseline smoke assertion
- Add: `select_move_with_decomposed_visits` returns `Vec<MoveVisit>` whose length equals legal move count
- Add: 3-4 consecutive `select_move_with_decomposed_visits` calls on a real game loop without panicking

### 6.4 `arena.rs`-based end-to-end validation (go/no-go signal)

Before running any Zero-vs-Gumbel comparison, save a checkpoint anchor:

```bash
cp model.safetensors checkpoints/pre_gumbel_validation_<date>.safetensors
```

Validation run:

```bash
cargo run --release --bin arena -- \
  --model1 <anchor_checkpoint> \
  --model2 <anchor_checkpoint> \
  --backend1 zero \
  --backend2 gumbel \
  --gumbel-k 16 \
  --mcts 200 \
  --games 100
```

**Go/no-go:** Gumbel at 32-64 simulations should roughly match Zero at 200 simulations in win rate (e.g. Gumbel win rate not below ~35-40% against Zero-at-200).

---

## 7. Sequencing (ordered, independently-verifiable PRs)

### (a) Shared σ/completed-Q helpers, isolated, no wiring changes

Add `gumbel_qtransform.rs` with pure functions from §2.1, plus `sequence_of_considered_visits` (§2.4) and unit tests (§6.1). Also extract `mcts_common.rs` shared helpers and refactor `mcts_zero.rs` to use them. Run existing Zero test suite to confirm zero behavior change.

**Highest-leverage/lowest-risk first move** — eliminates sign-flip duplication risk independent of the rest of the Gumbel rewrite.

### (b) Rewrite `GumbelMctsAgent`'s search core

Full rewrite of `gumbel_mcts.rs` per §2-§3 and §5. Update tests per §6.1-§6.3. `brain.rs`, `self_play.rs`, `arena.rs` remain untouched — Gumbel tested standalone via direct construction.

### (c) Wire into Brain/`self_play.rs` behind a flag defaulting to Zero

Add `SearchBackend` enum, `Brain::with_backend`, CLI flags per §4.1-§4.2. Verify with:

```bash
# manual smoke test
cargo run --release --bin self_play -- --search-backend gumbel --num-games 2 --mcts-iters 32
```

### (d) Arena validation

Add arena flags per §4.3. Save checkpoint anchor. Run Zero-at-200-vs-Gumbel-at-32/64 comparison; record win rate, game length, and wall-clock time per move.

### (e) Only after (d) passes, consider changing defaults to Gumbel

Out of scope for this plan — do not fold into any of the above PRs.

### Why this order minimizes risk

Each step is independently buildable, testable, and revertable. None of (a)-(d) touch `network.rs`, `features.rs`, `mapper.rs`'s constants, or `train.py` — the dual-network-sync constraint is inapplicable to this entire body of work.

---

## Critical files for implementation

| File | Role |
|------|------|
| `polyfish-rs/src/ai/gumbel_mcts.rs` | Main Gumbel search rewrite |
| `polyfish-rs/src/ai/mcts_zero.rs` | Shared extraction source + regression baseline |
| `polyfish-rs/src/ai/brain.rs` | Backend enum wiring |
| `polyfish-rs/src/ai/mcts_common.rs` | **New** — shared leaf/backprop helpers |
| `polyfish-rs/src/ai/gumbel_qtransform.rs` | **New** — σ/completed-Q pure functions |
| `polyfish-rs/src/bin/self_play.rs` | CLI flags + Brain construction |
| `polyfish-rs/src/bin/arena.rs` | Zero-vs-Gumbel validation |
| `polyfish-rs/src/ai/policy_composer.rs` | Logit/prior computation |
