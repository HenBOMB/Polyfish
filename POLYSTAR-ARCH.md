# PolyStar: Next-Generation Hierarchical & Autoregressive Architecture for Polytopia

This document specifies the **PolyStar** architecture—a hierarchical, autoregressive, and contrastively ranked neural network designed to overcome the structural failure modes of the legacy PolyZero architecture.

---

## 1. Executive Summary & Root-Cause Diagnosis

The legacy network [`PolyZeroNet`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/network.rs#L145) plateaued between ~600 and 743 Elo in [elo_ratings.json:2-157](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/elo_ratings.json#L2-L157) despite extensive tuning. The root cause is not raw model capacity or training iterations; it is a fundamental architectural mismatch with the game mechanics of *The Battle of Polytopia*.

### Fatal Flaws in the Legacy Architecture

1. **The Frankenstein Trunk (Gradient Interference)**:
   In [network.rs:161-170](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/network.rs#L161-L170) and [train.py:406-409](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/train.py#L406-L409), up to 12 distinct prediction heads (4 decomposed policy heads, 2 value heads, 4 auxiliary task heads, and experimental macro heads) were attached to a single 64-channel ResNet backbone. Conflicting gradient updates from micro-action cross-entropy, auxiliary fog reconstruction, and scalar game outcomes destroyed the trunk's representation.

2. **Open-Loop Micro Policy (Goal Blindness)**:
   While macro heads were prototyped on branch `origin/exp-elo-005-raise-relative-weight-in-value-targets`, the micro move generator remained unconditioned on the macro intent. The micro policy evaluated board states in isolation, while the macro head acted as an auxiliary prediction rather than an operational constraint.

3. **Value Head Calibration vs. Discrimination Breakdown**:
   As documented in commit `27055090` (`EXP_ELO_069`), training value heads with Mean Squared Error (MSE) regression produced aggregate calibration ($R^2 \approx 0.972$ on late-game states) but failed at local ranking discrimination. The value head was unable to reliably rank immediate candidate moves, dropping net win rates by 13 percentage points against hand-written heuristics.

4. **Multi-Ply Temporal Amnesia**:
   In [features.rs:116-128](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/features.rs#L116-L128), state encoding is strictly Markovian, relying only on 6 static decayed channels for fog memory. Between sequential plies in a ~440-move game, the network has no internal latent memory of its ongoing multi-turn plans, causing intention drift across turns.

5. **Single-Pass Action Decomposition Failure**:
   In [network.rs:161-164](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/network.rs#L161-L164), four independent policy heads (`pi_action`, `pi_source`, `pi_target`, `pi_option`) make blind parallel predictions that [policy_composer.rs:1-50](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/policy_composer.rs#L1-L50) multiplies together. The source head must guess which action type will be picked, while the target head predicts destinations without knowing which unit was selected.

---

## 2. The PolyStar Architecture Overview

PolyStar refactors state evaluation into a **two-tier hierarchical pipeline** with **autoregressive micro-decisions** and **contrastive pairwise value ranking**.

```
                ┌────────────────────────────────────────┐
                │        Raw Board Features (142x11x11)  │
                └───────────────────┬────────────────────┘
                                    │
                         ┌──────────▼──────────┐
                         │   Spatial Backbone  │ (ConvNeXt-Tiny / Swin-T)
                         │     128 Channels    │
                         └──────────┬──────────┘
                                    │
         ┌──────────────────────────┴──────────────────────────┐
         │                                                     │
┌────────▼─────────────────────┐             ┌─────────────────▼─────────────────┐
│     Macro Goal Generator     │             │     Contrastive Ranker Head       │
│  (Evaluates ONCE per turn)   │             │   (Trained on pairwise ranking:   │
│  Outputs K Strategic Tokens: │             │     Bradley-Terry Preference)     │
│  [Expand(3,4), Train(0,0)]   │             └───────────────────────────────────┘
└────────┬─────────────────────┘
         │ (K goal tokens: G_T)
         │
         │   ┌──────────────────────────────────────────────┐
         └───►  Cross-Attention: Micro Policy conditioned   │
             │                 on Goal Tokens               │
             └──────────────────────┬───────────────────────┘
                                    │
                      ┌─────────────▼──────────────┐
                      │ Autoregressive Micro Head  │
                      │  Pass 1: P(ActionType | S) │
                      │  Pass 2: P(Unit | Action)  │
                      │  Pass 3: P(Target | Unit)  │
                      └────────────────────────────┘
```

---

## 3. Four Core Architectural Pillars

### Pillar 1: Closed-Loop Goal Conditioning (Macro $\to$ Micro)
* **Turn-Level Intent Generation**: At the beginning of player turn $T$ (triggered by `EndTurn`), the spatial trunk routes features into a dedicated Macro Transformer block that emits $K=4$ **Goal Tokens**:
  $$G_T = \{g_{\text{expansion}}, g_{\text{military}}, g_{\text{economy}}, g_{\text{exploration}}\} \subset \mathbb{R}^{d_{\text{model}}}$$
  Each token encodes both categorical intent and spatial coordinates (e.g., $g_{\text{expansion}}$ binds to tile $(3, 4)$ where an uncaptured village is situated).
* **Tactical Cross-Attention**: During intra-turn plies, the micro policy network conditions its feature maps on $G_T$ via Multi-Head Cross-Attention:
  $$\text{Query} = \text{MicroSpatialTokens}, \quad \text{Key/Value} = G_T$$
  The micro policy cannot select actions without attending to the active turn objectives, preventing units from wandering aimlessly.

### Pillar 2: Autoregressive Action Tokenization
Instead of parallel independent heads, the micro policy decomposes the move probability autoregressively:
$$P(\text{Move} \mid s, G_T) = P(a_{\text{type}} \mid s, G_T) \cdot P(u_{\text{source}} \mid s, G_T, a_{\text{type}}) \cdot P(t_{\text{target}} \mid s, G_T, a_{\text{type}}, u_{\text{source}})$$

1. **Step 1 (`pi_action`)**: Outputs a distribution over 11 action classes (`Step`, `Attack`, `Train`, `Harvest`, etc.).
2. **Step 2 (`pi_source`)**: The selected `ActionType` is embedded and injected into the spatial map to predict the active source unit. Invalid units for the selected action are masked to $-\infty$.
3. **Step 3 (`pi_target`)**: The chosen `(ActionType, SourceUnit)` pair is embedded, and spatial cross-attention predicts valid target coordinates (destination tile, attack target, structure placement).

This eliminates the head-multiplication collapse in [policy_composer.rs:1-50](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/policy_composer.rs#L1-L50) and guarantees valid, coherent action proposals.

### Pillar 3: Contrastive Preference Ranking for Value
To resolve the calibration vs. discrimination breakdown observed in commit `27055090`, the value head discards pure MSE regression on scalar outcomes. It is trained via **Bradley-Terry Pairwise Preference Loss**:

$$\mathcal{L}_{\text{rank}} = -\log \sigma\left( V(s_{\text{winner}}) - V(s_{\text{loser}}) \right)$$

* **Data Generation**: During MCTS rollouts, sibling states expanded from the same root node are paired based on search visit counts and final Q-values.
* **Objective**: The value head is explicitly trained to discriminate which of two local tactical board states is superior, providing sharp, high-frequency gradients for leaf evaluation.

### Pillar 4: Latent Recurrent Core (Temporal Persistence)
To address the POMDP / Fog-of-War amnesia documented in [notes-memory.md:1-15](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/notes-memory.md#L1-L15):
* A 128-dimensional recurrent hidden state $h_t$ is propagated across plies and turns:
  $$h_{t+1} = \text{GRUCell}(h_t, \text{Pooling}(\text{BackboneOut}_t))$$
* This gives the agent a persistent memory of concealed enemy units, ongoing multi-turn expansion paths, and long-range strategic commitments.

---

## 4. Parameter & Channel Budget

| Component | Legacy PolyZero | PolyStar | Rationale |
| :--- | :--- | :--- | :--- |
| **Input Channels** | 142 ([features.rs:128](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/features.rs#L128)) | 142 (Preserves [states.rs](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/states.rs) compatibility) | Keeps engine feature encoding stable |
| **Backbone Filters** | 64 channels (6 ResBlocks) | 128 channels (4 ConvNeXt / Modern ResBlocks) | Prevents trunk representation saturation |
| **Normalization** | GroupNorm (8 groups) | GroupNorm (16 groups) or LayerNorm | Stable batch-independent inference |
| **Macro Mechanism** | Disconnected auxiliary heads | 4 Goal Tokens ($d=128$) via Cross-Attention | Closed-loop strategic steering |
| **Micro Action Policy** | 4 parallel heads multiplied blindly | 3-step autoregressive decoder | Eliminates combinatorial move conflicts |
| **Value Head** | Scalar MSE ($L_2$ regression) | Contrastive Pairwise Bradley-Terry Ranker | Fixes tactical leaf discrimination |
| **Temporal Memory** | None (Static Markovian snapshot) | 128-dim recurrent latent state ($h_t$) | Solves multi-ply intention drift |
| **Total Parameters** | ~320,000 (~1.2 MB safetensors) | ~1,850,000 (~7.4 MB safetensors) | Feasible on RunPod / modern GPUs |

---

## 5. Dual-Stack Implementation Plan

As required by [CLAUDE.md:71-78](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/CLAUDE.md#L71-L78), all layer definitions and parameter shapes must remain byte-compatible between the Rust inference engine and Python training script:

1. **Rust Implementation (`polyfish-rs/src/ai/network.rs`)**:
   * Implement `PolyStarNet` in Candle.
   * Provide `forward_macro(state) -> GoalTokens` and `forward_micro(state, goals, h_t) -> (ActionDist, Value, h_{t+1})`.
   * Preserve the headless evaluation path for high-throughput batching in [expert_boost_throughput.md:28-40](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/expert_boost_throughput.md#L28-L40).

2. **Python Implementation (`polyfish-rs/train.py`)**:
   * Implement PyTorch `PolyStarNet` mirroring the Candle layer names and tensor dimensions.
   * Add the pairwise ranking loss $\mathcal{L}_{\text{rank}}$ to the replay buffer sampling logic.
   * Supervise Macro Goal Tokens using high-level targets (city expansion vectors and military rally points).

---

## 6. Experimental Validation Roadmap (Protocol Compliance)

In accordance with [hypothesis_driven_improvements.md:9-19](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/hypothesis_driven_improvements.md#L9-L19):

* **Fast-Loop Verification (n=32)**:
  * Benchmark PolyStar vs Greedy on fixed Bardur+Imperius seeds.
  * Target Metric: Third-city rate $\ge 0.80$ by turn 13 and Army Value $\ge 25$ by turn 12 (surpassing the bottlenecks identified in [hypothesis_driven_improvements.md:198-205](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/hypothesis_driven_improvements.md#L198-L205)).
* **Slow-Loop Verification (Elo Ladder)**:
  * Deploy on RunPod using [run_training_runpod.sh](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/run_training_runpod.sh).
  * Success Criterion: Break the 742 Elo plateau in [elo_ratings.json:14-25](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/elo_ratings.json#L14-L25), targeting $\ge 1,000$ Elo against the frozen Greedy anchor.

---

## 7. Source Catalog & Evidence Concordance

Every claim, empirical finding, failure mode, and parameter dimension in this document is verified in the repository. The table below provides the ground-truth concordance:

| Finding / Specification | Source Location in Repository | Commit / Context |
| :--- | :--- | :--- |
| **Legacy `PolyZeroNet` Structure & Heads** | [network.rs:145-187](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/network.rs#L145-L187) | 6-block ResNet trunk, 64 channels, 4 parallel policy heads |
| **PyTorch Architecture & Aux Heads** | [train.py:397-409](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/train.py#L397-L409) | `SPATIAL_CHANNELS = 142`, 4 aux supervision heads |
| **Independent Policy Head Product Collapse** | [policy_composer.rs:1-50](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/policy_composer.rs#L1-L50) | Multiplicative combination of uncoordinated logits |
| **Input Spatial Channels (142)** | [features.rs:116-128](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/features.rs#L116-L128) | Dynamic `NUM_CHANNELS = CH_MEM_END` (142 channels) |
| **Fog-of-War Memory Specifications** | [notes-memory.md:1-145](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/notes-memory.md#L1-L145) | 6 decaying observation channels for enemy units in fog |
| **Elo Plateau Ratings (~600–743)** | [elo_ratings.json:2-157](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/elo_ratings.json#L2-L157) | Greedy baseline 571.6, iter156 peak 742.9, iter264 at 700.9 |
| **Search Horizon Limitation (~8 plies/turn)** | [notes.md:74-75](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/notes.md#L74-L75), [notes.md:180-186](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/notes.md#L180-L186) | 64 sims reach 2–3 plies; single-turn horizon blind spot |
| **Search Depth Failure on First Capture** | [hypothesis_driven_improvements.md:43-53](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/hypothesis_driven_improvements.md#L43-L53) | EXP 3: 64 $\to$ 256 sims failed to accelerate village walk |
| **Mid-Game Autopsy (Units, 3rd City, Tech Trap)** | [hypothesis_driven_improvements.md:187-205](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/hypothesis_driven_improvements.md#L187-L205) | EXP_ELO_001: 3rd city rate 39% vs 81%, tech overpurchase |
| **Outcome Label Noise Floor (~50% noise)** | `polyfish-rs/notes-runs-2026-07-14-16.md` | Branch `hen-training-experimental` (commit `e73713f5`) |
| **Anchor Limit-Cycle Fix (`-W 0.70 -P 0.2`)** | `polyfish-rs/notes-runs-2026-07-14-16.md` | Branch `hen-training-experimental` (commit `e73713f5`) |
| **Value Calibration vs. Discrimination Defect** | `hypothesis_driven_improvements.md` (EXP_ELO_069) | Branch `origin/exp-elo-005-...` (commit `27055090`) |
| **Dual-Network Synchronization Rule** | [CLAUDE.md:71-78](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/CLAUDE.md#L71-L78) | Rust Candle $\leftrightarrow$ Python PyTorch byte-compatibility |
| **Batched Evaluator Throughput (578 moves/s)** | [expert_boost_throughput.md:28-40](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/expert_boost_throughput.md#L28-L40) | Pipelined MPSGraph / CUDA worker specifications |
| **RunPod Autonomous CUDA Training Script** | `polyfish-rs/run_training_runpod.sh` | Branch `hen-training-experimental` (commit `4749eb5a`) |
| **Experiment Protocol Standards** | [hypothesis_driven_improvements.md:9-19](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/hypothesis_driven_improvements.md#L9-L19) | Fast-loop $n=32$ vs slow-loop Elo ladder protocols |

