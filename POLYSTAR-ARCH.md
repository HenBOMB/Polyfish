# PolyStar v2: 20M Entity-Transformer & League Reinforcement Learning Architecture

This document specifies the **PolyStar v2** architecture—a 20-million parameter hybrid Entity-Map Transformer and model-free reinforcement learning pipeline designed to succeed the legacy PolyZero architecture. 

It synthesizes the architectural post-mortems in [`BOTTLENECK.md`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/BOTTLENECK.md) and [`FAILURES.md`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/FAILURES.md), resolving the structural bottlenecks that stalled Polyfish at ~743 Elo in [`elo_ratings.json:2-157`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/elo_ratings.json#L2-L157).

---

## 1. Executive Summary & Paradigm Shift

Standard AlphaZero/MCTS fails on *The Battle of Polytopia* due to domain mismatches:
1. **The Intra-Turn Depth Trap**: A single turn requires 8–15 atomic plies ending in `MoveType::EndTurn` ([`polyfish-rs/src/types.rs:714`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/types.rs#L714)). A 64-simulation MCTS path exhausts its budget permuting intra-turn move orders without penetrating into opponent turns ([`BOTTLENECK.md:21-42`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/BOTTLENECK.md#L21-L42)).
2. **Planar Representation Flattening**: Forcing non-spatial Directed Acyclic Graphs (the Tech Tree in [`settings/technology.rs:460-485`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/settings/technology.rs#L460-L485)), global star counters, and discrete unit entities into a 2D convolutional grid destroys relational structure ([`BOTTLENECK.md:47-58`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/BOTTLENECK.md#L47-L58)).
3. **The Multi-Task Frankenstein Trunk**: Adding 12 auxiliary macro heads to a 64-channel ResNet backbone ([`network.rs:161-170`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/network.rs#L161-L170)) induced negative gradient transfer (EXP_ELO_069), dropping win rates by 13 percentage points ([`FAILURES.md:46-50`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/FAILURES.md#L46-L50)).
4. **Policy Decomposition Collapse**: Multiplying unconditioned marginal head distributions in [`policy_composer.rs:16-41`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/policy_composer.rs#L16-L41) causes source and target coordinates to mismatch.
5. **Passivity Collapse in Pure Self-Play**: Tabula rasa self-play with a single network degenerates into mutual non-aggression ([`FAILURES.md:28-31`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/FAILURES.md#L28-L31)).

**PolyStar v2 replaces flat MCTS with a direct reactive 20M Entity-Transformer policy** trained via **Supervised Behavioral Cloning Warmup followed by PPO League Play** (inspired by DeepMind AlphaStar and OpenAI Five).

---

## 2. Architectural Comparison

| Dimension | Legacy PolyZero ([`network.rs`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/network.rs)) | PolyStar v2 (Entity-Transformer PPO) | Rationale |
| :--- | :--- | :--- | :--- |
| **Search Paradigm** | 64-iteration Gumbel MCTS | Model-Free Reactive Policy (Single Pass) | Bypasses intra-turn depth trap ([`BOTTLENECK.md:32`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/BOTTLENECK.md#L32)) |
| **Total Parameters** | ~320,000 (~1.2 MB safetensors) | **~21,500,000 (~86 MB safetensors)** | Overcomes model capacity deficit ([`BOTTLENECK.md:59-71`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/BOTTLENECK.md#L59-L71)) |
| **Backbone Structure** | 6 ResBlocks, 64 channels | 12-layer Transformer ($d=384$, 6 heads, Pre-LN) | ViT-Small class; deep relational attention |
| **Sequence Length** | N/A (Planar 2D Convolutions) | 170 tokens ($121\text{ map} + 32\text{ units} + 16\text{ cities} + 1\text{ global}$) | Unified entity-spatial attention |
| **Unit Tracking** | Anonymous grid cells | Persistent Unit IDs ([`states.rs:256`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/states.rs#L256)) | Preserves unit identity across plies |
| **Move Scoring** | 4 multiplied marginal heads | Single-Pass Bilinear Move Pointer | Fixes multiplication collapse ([`policy_composer.rs:16-41`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/policy_composer.rs#L16-L41)) |
| **Training Engine** | Policy distillation + scalar MSE | Behavioral Cloning Warmup $\to$ PPO + GAE | Smooth RL convergence without tabula rasa collapse |
| **Opponent Pool** | Single-checkpoint self-play | 3-Tier League (Anchors + Pool + Exploiters) | Eliminates cyclical forgetting ([`FAILURES.md:33-35`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/FAILURES.md#L33-L35)) |
| **Move Throughput** | ~578 moves/s ([`expert_boost_throughput.md:36`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/expert_boost_throughput.md#L36)) | **2,500+ moves/s (Batched GPU)** | 64× FLOP dividend from dropping MCTS |

---

## 3. The 20M Hybrid Entity-Map Architecture

```
[11x11 Spatial Map]           [Discrete Entity Tokens]           [Global Context Token]
  (Terrain, Roads,               (Units: up to 32,                 (Tech DAG Bitmask,
   Structures, FOW)                Cities: up to 16)                Stars, Score, Turn)
         │                                │                                  │
         ▼                                ▼                                  ▼
 Patch Conv (Patch 1x1)            Entity Projector                 Global MLP Projector
 [121 tokens x 384]               [48 tokens x 384]                   [1 token x 384]
         │                                │                                  │
         └────────────────────────┬───────┴──────────────────────────────────┘
                                  ▼
             Concatenated Token Sequence [170 tokens x 384]
                                  │
                                  ▼
             12-Layer Pre-LN Transformer Backbone (6 heads, d=384, MLP=1536)
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
      Actor (Move Pointer)              Critic (Value Head)
   Bilinear Legal Move Softmax         Scalar Win [-1, 1] + Progress
```

### 3.1 Tokenization Specification

1. **Spatial Map Tokens ($M \in \mathbb{R}^{121 \times 384}$)**:
   - Input channels: 142 channels preserving [`features.rs:128`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/features.rs#L128) and [`train.py:397`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/train.py#L397) (including 6 fog memory channels from [`notes-memory.md:1-26`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/notes-memory.md#L1-L26)).
   - A $1 \times 1$ pointwise conv maps $142 \to 384$, added with 2D learnable spatial positional embeddings.

2. **Unit Entity Tokens ($U \in \mathbb{R}^{32 \times 384}$)**:
   - For each unit in `tribe.units` ([`states.rs:408`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/states.rs#L408)):
     $$\text{Token}_u = \text{Embed}(\text{owner}) + \text{Embed}(\text{unit\_type}) + \text{Linear}(\text{hp}, \text{veteran}, \text{kills}, \text{moved}, \text{attacked}) + \text{PosEmbed}(\text{coords})$$
   - Unused unit slots (up to 32) are zero-padded and attention-masked.

3. **City Entity Tokens ($C \in \mathbb{R}^{16 \times 384}$)**:
   - For each city in `tribe.cities` ([`states.rs:406`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/states.rs#L406)):
     $$\text{Token}_c = \text{Embed}(\text{owner}) + \text{Linear}(\text{level}, \text{production}, \text{progress}, \text{border\_size}) + \text{PosEmbed}(\text{city.idx})$$
   - Unused city slots (up to 16) are zero-padded and attention-masked.

4. **Global Context Token ($G \in \mathbb{R}^{1 \times 384}$)**:
   - Binary bitmask of 25 researched vanilla technologies ([`settings/technology.rs:460-485`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/settings/technology.rs#L460-L485)).
   - Continuous normalized scalars: current stars ([`states.rs:389`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/states.rs#L389)), current score ([`states.rs:387`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/states.rs#L387)), game turn, and relative score delta.

### 3.2 Transformer Core Specifications
- **Layers**: 12 Pre-LayerNorm Transformer encoder blocks.
- **Hidden Dim ($d_{\text{model}}$)**: 384.
- **Attention Heads**: 6 ($d_{\text{head}} = 64$).
- **Feedforward Hidden Dim**: 1,536 ($4 \times d_{\text{model}}$) with GELU activations.
- **Parameters**:
  - Attention projections: $12 \times (4 \times 384^2) \approx 7.08\text{M}$.
  - MLP projections: $12 \times (2 \times 384 \times 1536) \approx 14.16\text{M}$.
  - Embeddings & heads: $\approx 0.25\text{M}$.
  - **Total**: **~21.5 Million Parameters**.

---

## 4. Single-Pass Bilinear Move Pointer

To eliminate the head multiplication breakdown in [`policy_composer.rs:16-41`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/policy_composer.rs#L16-L41) without adding serial autoregressive GPU stalls:

1. The Rust engine extracts all legal moves $L(s) = [m_1, \dots, m_K]$ via `game.legal_moves()`.
2. Each legal move $m_i$ identifies:
   - Action type $a \in \{0, \dots, 10\}$ (`Step`, `Attack`, `Train`, etc.).
   - Source token latent $\mathbf{h}_{\text{source}} \in \mathbb{R}^{384}$ (from active unit or city token; null vector for non-spatial moves).
   - Target token latent $\mathbf{h}_{\text{target}} \in \mathbb{R}^{384}$ (from target map tile or target enemy unit token).
   - Move option index $o \in \{0, \dots, 191\}$ ([`train.py:148`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/train.py#L148)).
3. The move logit is computed via bilinear interaction:
   $$\text{logit}(m_i) = \mathbf{w}_{\text{action}}[a] + \frac{\langle \mathbf{h}_{\text{source}}(m_i), \, \mathbf{h}_{\text{target}}(m_i) \rangle}{\sqrt{d_{\text{model}}}} + \mathbf{w}_{\text{option}}[o]$$
4. Softmax is evaluated directly over the legal moves:
   $$P(m_i \mid s) = \frac{\exp(\text{logit}(m_i) / \tau)}{\sum_{j=1}^K \exp(\text{logit}(m_j) / \tau)}$$

**Properties**:
- Single GPU forward pass evaluates the full latent state.
- Candidate scoring is evaluated in microseconds in Rust.
- Conditioning destination on source is exact via $\langle \mathbf{h}_{\text{source}}, \mathbf{h}_{\text{target}} \rangle$.

---

## 5. Critic Value Head & Anchoring

The Critic head pools the global context token and mean-pooled entity latents through a 2-layer MLP:
$$V(s) = \mathbf{w}_{\text{val}}^T \text{GELU}(\mathbf{W}_2 \text{GELU}(\mathbf{W}_1 [\mathbf{h}_G; \bar{\mathbf{h}}_E]))$$
- Primary output: $V_{\text{win}}(s) \in [-1, +1]$ (hyperbolic tangent activation).
- Secondary output: $V_{\text{progress}}(s) \in [0, 1]$ (sigmoid activation).

Trained with Generalized Advantage Estimation (GAE):
$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t), \quad \hat{A}_t = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}$$
where $\gamma = 0.99, \lambda = 0.95$.

---

## 6. Training Protocol: Supervised Warmup + PPO League

To prevent the tabula rasa passivity collapse in [`FAILURES.md:28-31`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/FAILURES.md#L28-L31) and anchor curriculum fadeout collapse in [`FAILURES.md:33-35`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/FAILURES.md#L33-L35):

### Phase 1: Supervised Behavioral Cloning Kickstart (The AlphaStar Strategy)
- Train the 20M Transformer for 15–20 epochs on offline game datasets from [`archive/games_*.safetensors`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/train.py#L374) and games played against the Greedy heuristic engine ([`ai/evaluator/`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/evaluator/)).
- Cross-entropy loss over legal move choices:
  $$\mathcal{L}_{\text{BC}} = -\sum_{i} \log P_\theta(m_i^* \mid s_i)$$
- **Outcome**: The 20M network enters RL with an established ~800 Elo tactical baseline, knowing how to conquer villages, buy techs, and attack efficiently.

### Phase 2: PPO League Play
Rollout workers generate games against a 3-tier league matchmaker:
1. **Greedy Heuristic Anchor (25% of matches)**: Prevents economic drift and passive opening degeneration.
2. **Historical Checkpoint Pool (50% of matches)**: Uniform sampling from past iterations, permanently fixing cyclical forgetting ([`FAILURES.md:33-35`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/FAILURES.md#L33-L35)).
3. **Main Exploiters (25% of matches)**: Policies conditioned or tuned for aggressive early combat to punish defensive turtling.

PPO Clipped Objective:
$$\mathcal{L}_{\text{PPO}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( \frac{P_\theta(m_t \mid s_t)}{P_{\theta_{\text{old}}}(m_t \mid s_t)} \hat{A}_t, \, \text{clip}\left(\frac{P_\theta(m_t \mid s_t)}{P_{\theta_{\text{old}}}(m_t \mid s_t)}, 1-\epsilon, 1+\epsilon\right) \hat{A}_t \right) \right]$$
with $\epsilon = 0.20$ and entropy bonus coefficient $c_{\text{ent}} = 0.01$.

---

## 7. Dual-Stack Implementation Roadmap

As required by [`CLAUDE.md:71-78`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/CLAUDE.md#L71-L78), layer definitions and tensor shapes must remain strictly byte-compatible between Rust Candle and Python PyTorch:

1. **Step 1 (`polyfish-rs/src/ai/features.rs`)**:
   Implement `state_to_entity_features` extracting spatial map ($142 \times 11 \times 11$), unit entity matrix ($32 \times D_u$), city entity matrix ($16 \times D_c$), and global context vector.
2. **Step 2 (`polyfish-rs/src/ai/network.rs` & `train_ppo.py`)**:
   Implement `PolyEntityNet` (12-layer, $d=384$ Pre-LN Transformer) in Candle and PyTorch with matching state dict keys:
   `transformer_blocks.{0..11}.attn.*`, `transformer_blocks.{0..11}.mlp.*`, `map_conv.*`, `actor_pointer.*`, `critic.*`.
3. **Step 3 (`polyfish-rs/src/ai/policy_composer.rs`)**:
   Implement `compute_move_pointer_priors` evaluating legal candidate logits via bilinear dot products.
4. **Step 4 (`polyfish-rs/src/bin/self_play.rs`)**:
   Add `--backend ppo-policy` mode to execute rollouts using direct policy sampling without MCTS simulation loops.
5. **Step 5 (`polyfish-rs/train_ppo.py`)**:
   Implement the PyTorch PPO training loop with GAE trajectory buffers, clipped surrogate loss, and league model checkpoints.

---

## 8. Source Catalog & Evidence Concordance

| Specification / Empirical Grounding | Source Location in Repository | Evidence Context |
| :--- | :--- | :--- |
| **Legacy ResNet-6 & Heads** | [`network.rs:145-187`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/network.rs#L145-L187) | 6 blocks, 64 channels, 4 parallel policy heads |
| **Intra-Turn MCTS Depth Trap** | [`BOTTLENECK.md:21-42`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/BOTTLENECK.md#L21-L42) | 8-ply path fails to exit Turn 1; combinatorial space $30^{10}$ |
| **Planar Flattening & Tech DAG** | [`BOTTLENECK.md:47-58`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/BOTTLENECK.md#L47-L58) | 2D convs unable to model DAG prerequisites ([`technology.rs:460-485`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/settings/technology.rs#L460-L485)) |
| **Persistent Unit Identifiers** | [`states.rs:256-258`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/states.rs#L256-L258) | `pub id: u32` minted at spawn, persistent across moves |
| **City State Schema** | [`states.rs:291-311`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/states.rs#L291-L311) | Level, progress, border size, territory indices |
| **Fog Memory Spatial Channels** | [`features.rs:116-128`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/features.rs#L116-L128) | 6 decayed enemy observation channels |
| **Head Multiplication Breakdown** | [`policy_composer.rs:16-41`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/policy_composer.rs#L16-L41) | Independent marginal products cause source/target confusion |
| **192 Action Options Head** | [`train.py:147-148`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/train.py#L147-L148) | Unified structures, units, techs, abilities, rewards |
| **Multi-Head Gradient Interference** | [`FAILURES.md:46-50`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/FAILURES.md#L46-L50) | EXP_ELO_069: 12 macro heads dropped win rate by 13% |
| **Tabula Rasa Passivity Collapse** | [`FAILURES.md:28-31`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/FAILURES.md#L28-L31) | Mutual non-aggression; captures dropped 6.5 to 3.2 |
| **Curriculum Anchor Fadeout Defect** | [`FAILURES.md:33-35`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/FAILURES.md#L33-L35) | Fading anchor to 5% crashed win rate from 81% to 25% |
| **Self-Play Throughput Baseline** | [`expert_boost_throughput.md:36`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/expert_boost_throughput.md#L36) | ~578 moves/s bound by CPU↔GPU sync stalls |
| **Offline Game Replay Archive** | [`train.py:374`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/train.py#L374) | `archive/games_*.safetensors` available for warmup |
