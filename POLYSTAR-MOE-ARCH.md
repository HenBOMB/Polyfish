# PolyStar MoE v2.2: 37.17M Token-Routed Mixture-of-Experts Entity-Transformer

This document specifies the **PolyStar MoE v2.2** architecture—a 37.17-million parameter Token-Routed Mixture-of-Experts (MoE) Entity-Map Transformer and model-free reinforcement learning pipeline designed to succeed legacy PolyZero.

It resolves the structural MCTS limits in [`BOTTLENECK.md:21-71`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/BOTTLENECK.md#L21-L71), the multi-timescale gradient interference in [`FAILURES.md:46-72`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/FAILURES.md#L46-L72), the bilinear pointer blind spot on target-only actions, the alternating two-player advantage sign corruption, padded entity leakage across attention residuals, and null-token index bounds.

---

## 1. Executive Summary & Paradigm Shift

Standard AlphaZero/MCTS fails on *The Battle of Polytopia* due to four fundamental domain mismatches:
1. **The Intra-Turn Depth Trap**: A single turn requires 8–15 atomic plies ending in `MoveType::EndTurn` ([`polyfish-rs/src/types.rs:714`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/types.rs#L714)). A 64-simulation MCTS path exhausts its budget permuting intra-turn move orders without penetrating into opponent turns ([`BOTTLENECK.md:21-42`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/BOTTLENECK.md#L21-L42)).
2. **Planar Representation Flattening**: Forcing non-spatial Directed Acyclic Graphs (the Tech Tree in [`settings/technology.rs:470-479`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/settings/technology.rs#L470-L479)), global star counters ([`states.rs:389`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/states.rs#L389)), and discrete unit entities into a 2D convolutional grid destroys relational structure ([`BOTTLENECK.md:47-58`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/BOTTLENECK.md#L47-L58)).
3. **Timescale Conflict & Negative Gradient Transfer**: Multi-task supervision on a shared dense trunk induced negative gradient transfer (EXP_ELO_069), dropping win rates by 13 percentage points ([`FAILURES.md:46-50`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/FAILURES.md#L46-L50)), because macroeconomic compounding (turns), meso strategy (phases), and micro tactics (plies) conflict in backpropagation ([`FAILURES.md:64-72`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/FAILURES.md#L64-L72)).
4. **Policy Decomposition Collapse**: Multiplying unconditioned marginal head distributions in [`policy_composer.rs:69-105`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/policy_composer.rs#L69-L105) causes source and target coordinates to mismatch.

**PolyStar MoE v2.2 replaces flat MCTS with a direct reactive 37.17M Token-Routed MoE Entity-Transformer policy** trained via **Supervised Behavioral Cloning Warmup followed by Alternating Two-Player PPO League Play**:
- **Global Self-Attention** provides empire-wide communication between terrain, units, cities, and technology.
- **Deterministic Modality-Routed Expert FFNs with Masked Residual Boundaries** provide strict gradient isolation between macroeconomic planning and tactical combat while preventing padded entity leakage.
- **Four-Component Context-Conditioned Bilinear Move Pointer with Appended Null Vector** scores candidate moves dynamically across spatial, entity, and non-spatial actions without autoregressive stalls, target-only blind spots, or tensor out-of-bounds indexing.

---

## 2. Architectural Comparison

| Dimension | Legacy PolyZero ([`network.rs:145-187`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/network.rs#L145-L187)) | PolyStar MoE v2.2 (Token-Routed MoE) | Empirical / Theoretical Grounding |
| :--- | :--- | :--- | :--- |
| **Search Paradigm** | 64-iteration Gumbel MCTS | Model-Free Reactive Policy (Single Pass) | Eliminates intra-turn search horizon depletion ([`BOTTLENECK.md:32`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/BOTTLENECK.md#L32)) |
| **Total Parameters** | ~583,000 (~2.3 MB safetensors) | **~37,173,500 (~148.7 MB safetensors)** | Overcomes 580k capacity deficit ([`BOTTLENECK.md:59-71`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/BOTTLENECK.md#L59-L71)) |
| **Active Parameters** | ~583,000 per forward pass | **~16,570,000 per token** (Sparsely activated) | High capacity with lightweight per-token compute |
| **Backbone Structure** | 6 ResBlocks, 64 channels | 12-layer MoE Transformer ($d=384$, 6 heads, 3 experts) | Relational attention with isolated feedforward specialization |
| **Sequence Length** | N/A (Planar 2D Convolutions) | 170 active tokens ($121\text{ map} + 32\text{ units} + 16\text{ cities} + 1\text{ global}$) + 1 appended null token ($171 \times 384$) | Unified entity-spatial attention with branchless gather support |
| **Unit Tracking** | Anonymous grid cells | Persistent Unit IDs ([`states.rs:257`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/states.rs#L257)) | Tracks individual unit identities and combat history across moves |
| **Move Scoring** | 4 multiplied marginal heads | 4-Component Bilinear Move Pointer ($\mathbf{W}_G$, unbiased $\mathbf{W}_S, \mathbf{W}_T, \mathbf{W}_{ST}$) | Fixes multiplication collapse ([`policy_composer.rs:69-105`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/policy_composer.rs#L69-L105)) and target-only blind spots |
| **Gradient Interference** | Disparate tasks collide in trunk | Specialized Expert MLPs ($\text{MLP}_{\text{spatial}}, \text{MLP}_{\text{tactical}}, \text{MLP}_{\text{macro}}$) | Prevents EXP_ELO_069 negative transfer ([`FAILURES.md:46-50`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/FAILURES.md#L46-L50)) |
| **Opponent Pool** | Single-checkpoint self-play | 3-Tier League (Anchors + Pool + Exploiters) | Eliminates cyclical forgetting ([`FAILURES.md:33-35`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/FAILURES.md#L33-L35)) |
| **Move Throughput** | ~578 moves/s ([`expert_boost_throughput.md:36`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/expert_boost_throughput.md#L36)) | **2,000+ moves/s (Batched GPU)** | 64× FLOP dividend from dropping MCTS loops |

---

## 3. Structural Tokenization Specification

The input game state is tokenized into an ordered sequence of 170 active tokens of dimension $d_{\text{model}} = 384$:
$$\mathbf{X} = [\mathbf{M}_{0..120}; \, \mathbf{U}_{0..31}; \, \mathbf{C}_{0..15}; \, \mathbf{G}_0] \in \mathbb{R}^{170 \times 384}$$

```
[11x11 Spatial Map]           [Visible Unit Entities]           [City Entities]           [Global Context]
  (142 Channels)                (Up to 32 Units)                 (Up to 16 Cities)          (Techs, Stars, Turn)
        │                               │                               │                         │
        ▼                               ▼                               ▼                         ▼
   Pointwise Conv                Unit Projector                  City Projector            Global Projector
 [121 tokens x 384]             [32 tokens x 384]               [16 tokens x 384]          [1 token x 384]
        │                               │                               │                         │
        └───────────────────────────────┴───────────────┬───────────────┴─────────────────────────┘
                                                        ▼
                                   Concatenated Sequence [170 tokens x 384]
```

### 3.1 Spatial Map Tokens ($M \in \mathbb{R}^{121 \times 384}$, Slice `0..121`)
- **Input Channels**: 142 channels preserving [`features.rs:128`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/features.rs#L128) and [`train.py:397`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/train.py#L397), including 6 fog memory channels ([`features.rs:116-125`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/features.rs#L116-L125)).
- **Projection**: $1 \times 1$ pointwise convolution maps $142 \to 384$ (`conv_spatial`).
- **Position Embedding**: Learnable spatial positional embedding table $\mathbf{E}_{\text{pos}} \in \mathbb{R}^{121 \times 384}$ indexed per tile $(x, y) \in [0, 10] \times [0, 10]$ ([`features.rs:17`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/features.rs#L17)). This positional table is shared with unit and city tokens to ground entities directly into map coordinates.

### 3.2 Visible Unit Entity Tokens ($U \in \mathbb{R}^{32 \times 384}$, Slice `121..153`)
Iterates over all units visible to the active player across all tribes in `GameState::tribes` ([`states.rs:605`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/states.rs#L605)):
1. **Selection & Truncation Priority**:
   - Priority 1: Active player units from `tribe.units` ([`states.rs:408`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/states.rs#L408)). If active friendly units exceed 32 (rare on 11x11), units are sorted descending by combat value ($\text{hp} \times \text{attack}$).
   - Priority 2: Spotted enemy units outside fog of war that are within movement or attack range of any friendly unit, or threatening friendly cities.
   - Priority 3: Other spotted enemy units, sorted ascending by Chebyshev distance to the nearest friendly city.
   - *Attack Resolution Invariant*: Because an $11 \times 11$ board rarely contains more than 15–20 total units, guaranteeing priority to combat-adjacent enemies ensures that every enemy targeted by a legal `MoveType::Attack` is tracked in an entity slot, guaranteeing `tile_to_unit_slot` resolves to a unit token and eliminating the need to fall back to raw spatial tile tokens.
2. **Token Construction & Input Zero-Masking**:
   $$\mathbf{t}_{u, i} = m_{u, i} \cdot \left( \text{Embed}_{\text{rel}}(\text{owner\_rel}) + \text{Embed}_{\text{type}}(\text{unit\_type}) + \text{Linear}_{u}(\mathbf{f}_u) + \mathbf{E}_{\text{pos}}[\text{tile\_idx}] \right)$$
   Multiplying by the boolean entity mask $m_{u, i}$ immediately at Layer 0 ($\mathbf{U}_0 = \mathbf{M}_U \odot \mathbf{U}$) guarantees that inactive/padding slots do not carry embedding biases or $\mathbf{E}_{\text{pos}}[0]$ into the Transformer backbone.
3. **Explicit Entity Tensor Schema**:
   To prevent ambiguity between continuous features and categorical embedding lookups, unit state is extracted into five synchronized tensors:
   - `unit_types`: $[B, 32]$ (`int64`, values in $[0, 45]$ indexing all 46 active unit variants via `UNIT_INDEX`; [`features.rs:162-165`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/features.rs#L162-L165))
   - `unit_owners`: $[B, 32]$ (`int64`, $\text{Self}=0, \text{Enemy}=1$)
   - `unit_tiles`: $[B, 32]$ (`int64`, tile coordinate in $[0, 120]$ indexing $\mathbf{E}_{\text{pos}}$)
   - `unit_features`: $[B, 32, 10]$ (`float32`, continuous and boolean unit attributes)
   - `unit_mask`: $[B, 32]$ (`uint8` / `bool`, $1$ for active units, $0$ for padding)
4. **Unit Continuous Feature Normalization ($\mathbf{f}_u \in \mathbb{R}^{10}$)**:
   - $\text{hp} / \text{max\_hp} \in [0.0, 1.0]$
   - $\text{veteran} \in \{0.0, 1.0\}$
   - $\text{kills} / 3.0 \in [0.0, 1.0]$ (clamped at 1.0, reflecting the 3-kill promotion threshold)
   - $\text{moved} \in \{0.0, 1.0\}$
   - $\text{attacked} \in \{0.0, 1.0\}$
   - $\text{frozen} \in \{0.0, 1.0\}$ ([`features.rs:90`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/features.rs#L90))
   - $\text{boosted} \in \{0.0, 1.0\}$ ([`features.rs:88`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/features.rs#L88))
   - $\text{poisoned} \in \{0.0, 1.0\}$ ([`features.rs:87`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/features.rs#L87))
   - $\text{has\_passenger} \in \{0.0, 1.0\}$ ([`features.rs:91`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/features.rs#L91))
   - $\text{converted} \in \{0.0, 1.0\}$

### 3.3 City Entity Tokens ($C \in \mathbb{R}^{16 \times 384}$, Slice `153..169`)
Iterates over all discovered cities and neutral villages:
1. **Selection & Discovery Query**:
   - Priority 1: Active player cities from `tribe.cities` ([`states.rs:406`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/states.rs#L406)).
   - Priority 2: Discovered enemy cities from opponent `tribe.cities`.
   - Priority 3: Discovered neutral villages queried from `state.structures` where `structure_type == StructureType::Village` (neutral villages do not possess a `CityState` record prior to capture; [`scoring.rs:1222`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/scoring.rs#L1222)).
   - Unclaimed neutral villages are populated with synthetic baseline features: $\text{level}=0, \text{progress}=0, \text{production}=0, \text{border\_size}=1, \text{connected}=0, \text{is\_capital}=0, \text{has\_walls}=0, \text{has\_riot}=0$.
2. **Token Construction & Input Zero-Masking**:
   $$\mathbf{t}_{c, j} = m_{c, j} \cdot \left( \text{Embed}_{\text{rel}}(\text{owner\_rel}) + \text{Linear}_{c}(\mathbf{f}_c) + \mathbf{E}_{\text{pos}}[\text{city.idx}] \right)$$
   Multiplying by $m_{c, j}$ at Layer 0 ($\mathbf{C}_0 = \mathbf{M}_C \odot \mathbf{C}$) suppresses embedding biases and positional offsets on padded slots.
3. **Explicit City Tensor Schema**:
   - `city_owners`: $[B, 16]$ (`int64`, $\text{Self}=0, \text{Enemy}=1, \text{Neutral}=2$)
   - `city_tiles`: $[B, 16]$ (`int64`, tile coordinate in $[0, 120]$ indexing $\mathbf{E}_{\text{pos}}$)
   - `city_features`: $[B, 16, 8]$ (`float32`, continuous and boolean city attributes)
   - `city_mask`: $[B, 16]$ (`uint8` / `bool`, $1$ for active cities/villages, $0$ for padding)
4. **City Continuous Feature Normalization ($\mathbf{f}_c \in \mathbb{R}^8$)**:
   - $\text{level} / 8.0 \in [0.0, 1.0]$
   - $\text{progress} / 10.0 \in [0.0, 1.0]$
   - $\text{production} / 20.0 \in [0.0, 1.0]$
   - $\text{border\_size} / 2.0 \in [0.5, 1.0]$ (1 for standard $3 \times 3$, 2 for expanded $5 \times 5$)
   - $\text{connected} \in \{0.0, 1.0\}$ (connected to capital via road/water network)
   - $\text{is\_capital} \in \{0.0, 1.0\}$ ([`features.rs:107`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/features.rs#L107))
   - $\text{has\_walls} \in \{0.0, 1.0\}$ ([`features.rs:109`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/features.rs#L109))
   - $\text{has\_riot} \in \{0.0, 1.0\}$ ([`features.rs:110`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/features.rs#L110))

### 3.4 Global Context Token ($G \in \mathbb{R}^{1 \times 384}$, Slice `169..170`)
- **Technology Vector**: 25-dimensional binary bitmask indicating researched vanilla technologies ([`settings/technology.rs:470-479`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/settings/technology.rs#L470-L479)), directly reading `tech_vanilla[i].discovered`.
- **Economic & Match Scalars**: 5 continuous normalized features:
  - $\text{stars} / 100.0 \in [0.0, 1.0]$ ([`states.rs:389`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/states.rs#L389))
  - $\text{score} / 10000.0 \in [0.0, 1.0]$ ([`states.rs:387`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/states.rs#L387))
  - $\text{turn} / 50.0 \in [0.0, 1.0]$
  - $\text{stars\_per\_turn (SPT)} / 50.0 \in [0.0, 1.0]$
  - $\text{score\_delta} / 5000.0 \in [-1.0, 1.0]$ (active player score minus visible opponent score)
- **Projection**: 2-layer MLP ($30 \to 384 \to 384$) with GELU activation (`proj_global`). The global token is never inactive and carries an entity mask of $1$.

---

## 4. The Modality-Routed Mixture-of-Experts (MoE) Backbone

```
                     ┌───────────────────────────────────────────────┐
                     │ Concatenated Sequence Input [170 tokens x 384]│
                     └───────────────────────┬───────────────────────┘
                                             ▼
                     ┌───────────────────────────────────────────────┐
                     │ Pre-LayerNorm Multi-Head Attention (6 heads)  │
                     │ (Full cross-talk between map, units, cities,  │
                     │  and global economic context)                 │
                     └───────────────────────┬───────────────────────┘
                                             ▼
                     ┌───────────────────────────────────────────────┐
                     │ Residual Addition & Masking (Zero Inactives)  │
                     └───────────────────────┬───────────────────────┘
                                             ▼
                     ┌───────────────── Token Slicer ────────────────┐
                     │                       │                       │
      Slice [0..121] │       Slice [121..153]│       Slice [153..170]│
                     ▼                       ▼                       ▼
            ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
            │ Spatial Expert  │     │ Tactical Expert │     │  Macro Expert   │
            │ MLP (d_ffn=1024)│     │ MLP (d_ffn=1024)│     │ MLP (d_ffn=1024)│
            │                 │     │                 │     │                 │
            │ Terrain, Roads, │     │ Units, Combat,  │     │ Cities, Techs,  │
            │ Fog Exploration │     │ Retaliation HP  │     │ Stars, Economy  │
            └────────┬────────┘     └────────┬────────┘     └────────┬────────┘
                     │                       │                       │
                     └───────────────────────┼───────────────────────┘
                                             ▼
                     ┌───────────────────────────────────────────────┐
                     │ Concatenate Slices & Residual Addition        │
                     │ (Apply Composite Mask M across all tokens)    │
                     └───────────────────────┬───────────────────────┘
                                             ▼
                                     (Repeated x12 Layers)
```

### 4.1 Layer Forward Pass Formulation
Let $\mathbf{M} \in \{0, 1\}^{170 \times 1}$ be the complete composite sequence mask:
$$\mathbf{M} = [\mathbf{1}_{121}; \, \mathbf{M}_U; \, \mathbf{M}_C] \in \{0, 1\}^{170 \times 1}$$
where $\mathbf{1}_{121}$ is an all-ones vector for spatial tiles, $\mathbf{M}_U = [m_{u, 0}, \dots, m_{u, 31}]^T$ is the unit entity mask, and $\mathbf{M}_C = [m_{c, 0}, \dots, m_{c, 15}, 1]^T$ masks city entity slots while permanently keeping the 17th macro token (the global context token) active.

For each block $l \in [1, \dots, 12]$:
$$\mathbf{X}' = \mathbf{M} \odot \left( \mathbf{X}^{(l-1)} + \text{MHA}(\text{PreLN}(\mathbf{X}^{(l-1)})) \right)$$
$$\mathbf{H}_{\text{spatial}} = \text{MLP}_{\text{spatial}}(\text{PreLN}(\mathbf{X}'[0..121]))$$
$$\mathbf{H}_{\text{tactical}} = \mathbf{M}_U \odot \text{MLP}_{\text{tactical}}(\text{PreLN}(\mathbf{X}'[121..153]))$$
$$\mathbf{H}_{\text{macro}} = \mathbf{M}_C \odot \text{MLP}_{\text{macro}}(\text{PreLN}(\mathbf{X}'[153..170]))$$
$$\mathbf{X}^{(l)} = \mathbf{M} \odot \left( \mathbf{X}' + [\mathbf{H}_{\text{spatial}}; \, \mathbf{H}_{\text{tactical}}; \, \mathbf{H}_{\text{macro}}] \right)$$

Applying the composite mask $\mathbf{M}$ after **both** attention and FFN residuals guarantees that inactive query outputs cannot pollute padded entity slots, preventing bias accumulation and phantom activations across the 12 Transformer layers. In self-attention, inactive key slots are masked with $-\infty$ so active tokens never attend to padded entities. Furthermore, multiplying each expert FFN output by $\mathbf{M}_U$ / $\mathbf{M}_C$ ensures that the LayerNorm bias $\beta$ evaluated on padded zero vectors is strictly suppressed before reaching the residual stream.

### 4.2 Expert Specifications
Each expert MLP consists of two linear projections with GELU activation:
$$\text{MLP}_e(\mathbf{z}) = \mathbf{W}_{e, 2} \, \text{GELU}(\mathbf{W}_{e, 1} \, \mathbf{z} + \mathbf{b}_{e, 1}) + \mathbf{b}_{e, 2}$$
where $\mathbf{W}_{e, 1} \in \mathbb{R}^{1024 \times 384}$ and $\mathbf{W}_{e, 2} \in \mathbb{R}^{384 \times 1024}$.

1. **Spatial Expert ($\text{MLP}_{\text{spatial}}$, Slice `0..121`)**:
   Specializes in terrain elevation, movement constraints, road connections, and fog frontier expansion.
2. **Tactical Combat Expert ($\text{MLP}_{\text{tactical}}$, Slice `121..153`)**:
   Specializes in unit HP thresholds, retaliation damage formulas, veteran bonuses, freeze/poison status, and combat lethality.
3. **Macroeconomic Expert ($\text{MLP}_{\text{macro}}$, Slice `153..170`)**:
   Specializes in city growth progress, border expansions, capital defense, tech prerequisite chains ([`settings/technology.rs:470`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/settings/technology.rs#L470)), star compounding, and turn budgeting across both city entities and global context.

### 4.3 Parameter Budget & Compute Efficiency

| Component | Dimensions / Shape | Parameters (Weights + Biases) |
| :--- | :--- | :--- |
| **Attention Projections (12 layers)** | $12 \times 4 \times (384 \times 384 + 384)$ | $7,096,320$ |
| **Expert MLPs (12 layers $\times$ 3 experts)** | $12 \times 3 \times (384 \times 1024 + 1024 + 1024 \times 384 + 384)$ | $28,362,240$ |
| **LayerNorms (12 layers $\times$ 2 + final)** | $25 \times (2 \times 384)$ | $19,200$ |
| **Spatial Pointwise Conv** | $142 \times 384 + 384$ | $54,912$ |
| **Spatial Positional Embeddings $\mathbf{E}_{\text{pos}}$** | $121 \times 384$ | $46,464$ |
| **Unit Entity Projector** | Embeds ($2 \times 384 + 46 \times 384$) + Linear ($10 \times 384 + 384$) | $22,656$ |
| **City Entity Projector** | Embeds ($3 \times 384$) + Linear ($8 \times 384 + 384$) | $4,608$ |
| **Global Context Projector** | Linear 1 ($30 \times 384 + 384$) + Linear 2 ($384 \times 384 + 384$) | $159,744$ |
| **Move Pointer Embeddings** | $\mathbf{E}_{\text{action}}$ ($12 \times 384$) + $\mathbf{E}_{\text{option}}$ ($193 \times 384$) | $78,720$ |
| **Move Pointer Projections** | $\mathbf{W}_G$ ($384 \times 384 + 384$) + $\mathbf{W}_S, \mathbf{W}_T, \mathbf{W}_{ST}$ (unbiased, $3 \times [384 \times 384]$) | $590,208$ |
| **Critic Value Head** | $\mathbf{W}_1$ ($1536 \to 384$) + $\mathbf{W}_2$ ($384 \to 384$) + head ($384 \to 1$) | $738,433$ |
| **Total Model Parameters** | — | **37,173,505 (~148.7 MB safetensors)** |

- **Active Parameters per Token**:
  $$\text{Active FLOPs} \propto 7.12\text{M} + \frac{28.36\text{M}}{3} \approx 16.57\text{M active parameters equivalent}$$
Each token activates only one expert MLP per layer, delivering the full representational capacity of a 37.17M parameter network with the inference throughput of a 16.57M dense model.

---

## 5. Four-Component Bilinear Move Pointer

This mechanism resolves the head multiplication collapse in [`policy_composer.rs:69-105`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/policy_composer.rs#L69-L105) and fixes the state-independent logit defect for target-only moves (`Build`, `Harvest`, `Reward`).

### 5.1 Appended Null Representation for Branchless Gather
To prevent out-of-bounds indexing in tensor evaluation, the output representations of the final Transformer layer $\mathbf{X}^{(12)} \in \mathbb{R}^{B \times 170 \times 384}$ are first normalized by the final Pre-LN layer `norm_f`:
$$\mathbf{X}_{\text{norm}} = \text{norm\_f}(\mathbf{X}^{(12)}) \in \mathbb{R}^{B \times 170 \times 384}$$
and then appended with an explicit all-zero token at index $170$:
$$\mathbf{X}_{\text{eval}} = [\mathbf{X}_{\text{norm}}; \, \mathbf{0}_{B \times 1 \times 384}] \in \mathbb{R}^{B \times 171 \times 384}$$

*Critical Order Invariant*: The null token $\mathbf{0}$ must be appended **after** `norm_f`. If $\mathbf{0}$ were appended before `norm_f`, evaluating $\text{LayerNorm}(\mathbf{0})$ would produce the learnable bias $\beta_{\text{norm\_f}} \ne \mathbf{0}$. This would corrupt the null token into a non-zero vector, causing linear projections $\mathbf{W}_S, \mathbf{W}_T, \mathbf{W}_{ST}$ to emit non-zero activations at index 170 and destroying the branchless zeroing of inactive affordances.

1. The Rust engine queries legal moves $L(s) = [m_1, \dots, m_K]$ via `game.legal_moves()`.
2. Each legal move $m_i$ provides:
   - Action type $a \in [0, 11]$ mapped via `DecomposedMapper::move_type_to_idx` ([`mapper.rs:82-97`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/mapper.rs#L82-L97)), covering `None=0`, `Attack=1`, `Step=2`, `Capture=3`, `Ability=4`, `Summon=5`, `Harvest=6`, `Build=7`, `Research=8`, `Reward=9`, `EndTurn=10`, `Resign=11`. Note that the engine enum discriminant in [`types.rs:705-706`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/types.rs#L705-L706) (`Step=1`, `Attack=2`) is transposed to $a=2$ and $a=1$ by the mapper.
   - Option index $o \in [0, 192]$ ([`train.py:148`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/train.py#L148), [`mapper.rs:38-43`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/mapper.rs#L38-L43)), where $[0, 191]$ embed structures, units, techs, abilities, and rewards, while reserved index $o = 192$ represents `NO_OPTION` for non-option plies, preventing gradient pollution into structure slot 0.
   - Source token index $s_i \in [0, 170]$: points to unit entity slot in $[121, 152]$ (slice `121..153`), city entity slot in $[153, 168]$ (slice `153..169`), or null token index $170$.
   - Target token index $t_i \in [0, 170]$: points to destination tile in $[0, 120]$ (slice `0..121`) (e.g. `Step`, `Build`, `Harvest`, tile-targeted abilities), target enemy unit in $[121, 152]$ (slice `121..153`) for `Attack`, city entity in $[153, 168]$ (slice `153..169`) for `Reward`, or null token index $170$. Under the selection priority in Section 3.2, in-range enemy units are guaranteed entity slots, ensuring $t_i$ for `Attack` strictly resolves to a unit token $[121, 152]$ and preventing semantic manifold conflict with spatial tile tokens.
   - **Engine-to-Token Index Resolution**: Because [`Move::source_idx()`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/mapper.rs#L104-L107) and [`Move::target_idx()`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/mapper.rs#L109-L112) in the Rust engine return raw board tile indices ($0..120$ on an $11 \times 11$ map; e.g. [`moves/reward.rs:196-198`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/moves/reward.rs#L196-L198), [`moves/summon.rs:74-76`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/moves/summon.rs#L74-L76)), the feature extractor builds two reverse-lookup tables: `tile_to_unit_slot: [Option<usize>; 121]` and `tile_to_city_slot: [Option<usize>; 121]`. These map board tile coordinates to unit slots $[121, 152]$ or city slots $[153, 168]$, resolving moves into candidate token indices $s_i, t_i$ branchlessly.
3. The move query representation is computed from learned embedding tables:
   $$\mathbf{q}(m_i) = \mathbf{E}_{\text{action}}[a] + \mathbf{E}_{\text{option}}[o] \in \mathbb{R}^{384}$$
4. The move logit evaluates four contextual interactions, scaled by $\frac{1}{\sqrt{d_{\text{model}}}}$ to ensure stable logit variance and prevent premature softmax saturation:
   $$\text{logit}(m_i) = \frac{\mathbf{q}(m_i)^T \mathbf{W}_G \mathbf{h}_G + \mathbf{q}(m_i)^T \mathbf{W}_S \mathbf{h}_{\text{source}}(m_i) + \mathbf{q}(m_i)^T \mathbf{W}_T \mathbf{h}_{\text{target}}(m_i) + \mathbf{h}_{\text{source}}(m_i)^T \mathbf{W}_{ST} \mathbf{h}_{\text{target}}(m_i)}{\sqrt{d_{\text{model}}}}$$
   where $\mathbf{h}_{\text{source}}(m_i) = \mathbf{X}_{\text{eval}}[s_i]$ and $\mathbf{h}_{\text{target}}(m_i) = \mathbf{X}_{\text{eval}}[t_i]$. Pointer projection matrices $\mathbf{W}_G, \mathbf{W}_S, \mathbf{W}_T$ and embedding tables are initialized with standard deviation $\sigma = 0.02$. Bilinear interaction matrix $\mathbf{W}_{ST}$ is initialized with $\sigma = 0.01$ (half variance) to prevent the four additive terms in `Step` and `Attack` from dominating 1-term (`EndTurn`, `Research`) and 2-term (`Build`, `Summon`) actions during early exploration.
5. Softmax is evaluated directly over legal candidate moves:
   $$P(m_i \mid s) = \frac{\exp(\text{logit}(m_i) / \tau)}{\sum_{j=1}^K \exp(\text{logit}(m_j) / \tau)}$$

### 5.2 Vectorized Batched Evaluation
Rather than projecting candidate representations per move, projections are evaluated once across the full sequence:
- $\mathbf{z}_G = \mathbf{W}_G \mathbf{h}_G + \mathbf{b}_G \in \mathbb{R}^{B \times 384}$ (linear with bias)
- $\mathbf{Z}_S = \mathbf{X}_{\text{eval}} \mathbf{W}_S^T \in \mathbb{R}^{B \times 171 \times 384}$ (strictly linear without bias: `bias=False`)
- $\mathbf{Z}_T = \mathbf{X}_{\text{eval}} \mathbf{W}_T^T \in \mathbb{R}^{B \times 171 \times 384}$ (strictly linear without bias: `bias=False`)
- $\mathbf{Z}_{ST} = \mathbf{X}_{\text{eval}} \mathbf{W}_{ST}^T \in \mathbb{R}^{B \times 171 \times 384}$ (strictly linear without bias: `bias=False`)

Logits for all $K$ candidates are computed via batched gather operations:
$$\text{logit}(m_i) = \frac{\mathbf{q}(m_i) \cdot \mathbf{z}_G + \mathbf{q}(m_i) \cdot \mathbf{Z}_S[s_i] + \mathbf{q}(m_i) \cdot \mathbf{Z}_T[t_i] + \mathbf{X}_{\text{eval}}[s_i] \cdot \mathbf{Z}_{ST}[t_i]}{\sqrt{384}}$$
Because $\mathbf{X}_{\text{eval}}[170] = \mathbf{0}$ and $\mathbf{W}_S, \mathbf{W}_T, \mathbf{W}_{ST}$ contain no bias parameters, gathering index $170$ mathematically yields $\mathbf{Z}_S[170] = \mathbf{0}$, $\mathbf{Z}_T[170] = \mathbf{0}$, and $\mathbf{Z}_{ST}[170] = \mathbf{0}$, zeroing out the corresponding affordance and interaction terms without branch conditions.

### 5.3 Mathematical Behavior Across Action Classes:
- **`MoveType::Build` / `MoveType::Harvest` ([`moves/build.rs:105`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/moves/build.rs#L105), [`moves/harvest.rs:74-76`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/moves/harvest.rs#L74-L76))**:
  $s_i = 170 \implies \mathbf{h}_{\text{source}} = \mathbf{0}, \, t_i \in [0, 120] \implies \mathbf{h}_{\text{target}} = \mathbf{h}_{\text{tile}}$.
  $$\text{logit}(m_i) = \mathbf{q}(m_i)^T \mathbf{W}_G \mathbf{h}_G + \mathbf{q}(m_i)^T \mathbf{W}_T \mathbf{h}_{\text{tile}}$$
  Directly scores target tile value (e.g. building a road toward an enemy vs a dead end) without requiring a dummy source unit. For `Harvest`, $o = 192$.
- **`MoveType::Reward` ([`moves/reward.rs:44`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/moves/reward.rs#L44))**:
  $s_i = 170 \implies \mathbf{h}_{\text{source}} = \mathbf{0}, \, t_i \in [153, 168] \implies \mathbf{h}_{\text{target}} = \mathbf{h}_{\text{city}}$.
  $$\text{logit}(m_i) = \mathbf{q}(m_i)^T \mathbf{W}_G \mathbf{h}_G + \mathbf{q}(m_i)^T \mathbf{W}_T \mathbf{h}_{\text{city}}$$
  Dynamically conditions reward selection (City Wall vs Workshop) on the specific city entity leveling up.
- **`MoveType::Capture` ([`moves/capture.rs:180-182`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/moves/capture.rs#L180-L182))**:
  $s_i \in [121, 152] \implies \mathbf{h}_{\text{source}} = \mathbf{h}_{\text{unit}}, \, t_i = 170 \implies \mathbf{h}_{\text{target}} = \mathbf{0}$.
  $$\text{logit}(m_i) = \mathbf{q}(\text{Capture}, 192)^T \mathbf{W}_G \mathbf{h}_G + \mathbf{q}(\text{Capture}, 192)^T \mathbf{W}_S \mathbf{h}_{\text{unit}}$$
  Evaluates capturing a city/village/ruin using the acting unit's tactical context with reserved option index $192$.
- **`MoveType::Research` ([`types.rs:711`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/types.rs#L711))**:
  $s_i = 170, t_i = 170 \implies \mathbf{h}_{\text{source}} = \mathbf{0}, \mathbf{h}_{\text{target}} = \mathbf{0}$.
  $$\text{logit}(m_i) = \mathbf{q}(\text{Research}, t)^T \mathbf{W}_G \mathbf{h}_G$$
  Evaluates tech value against current stars ([`states.rs:389`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/states.rs#L389)), researched tech DAG, and strategic urgency.
- **`MoveType::EndTurn` ([`types.rs:714`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/types.rs#L714))**:
  $s_i = 170, t_i = 170 \implies \mathbf{h}_{\text{source}} = \mathbf{0}, \mathbf{h}_{\text{target}} = \mathbf{0}$.
  $$\text{logit}(m_i) = \mathbf{q}(\text{EndTurn}, 192)^T \mathbf{W}_G \mathbf{h}_G$$
  Dynamically evaluates whether to end the turn based on whether remaining stars or unmoved units can deliver further advantage, using reserved option index $192$.
- **`MoveType::Summon` / `Train` at City $C$ ([`types.rs:708`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/types.rs#L708))**:
  $s_i \in [153, 168] \implies \mathbf{h}_{\text{source}} = \mathbf{h}_{\text{city}}, \, t_i = 170 \implies \mathbf{h}_{\text{target}} = \mathbf{0}$.
  $$\text{logit}(m_i) = \mathbf{q}(\text{Summon}, u)^T \mathbf{W}_G \mathbf{h}_G + \mathbf{q}(\text{Summon}, u)^T \mathbf{W}_S \mathbf{h}_{\text{city}}$$
  Balances empire budget against local city production and defense needs.
- **`MoveType::Ability` (Promote, Heal, Freeze, Convert, Disband)**:
  $s_i \in [121, 152] \implies \mathbf{h}_{\text{source}} = \mathbf{h}_{\text{unit}}$. If untargeted (Promote, Disband, Recover, Explode), $t_i = 170$; if unit-targeted (Freeze enemy, Heal ally, Convert), $t_i \in [121, 152]$; if tile-targeted (BreakIce, BurnForest, ClearForest), $t_i \in [0, 120]$.
- **`MoveType::Step` / `MoveType::Attack` ([`types.rs:705-706`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/types.rs#L705-L706))**:
  $o = 192$. For `Step`, $t_i \in [0, 120]$ targets destination tile; for `Attack`, $t_i \in [121, 152]$ targets the enemy unit entity. All four terms fire, simultaneously evaluating campaign strategy ($\mathbf{h}_G$), acting unit role ($\mathbf{h}_{\text{source}}$), destination/enemy affordance ($\mathbf{h}_{\text{target}}$), and mutual source-target combat interaction ($\mathbf{h}_{\text{source}}^T \mathbf{W}_{ST} \mathbf{h}_{\text{target}}$).

---

## 6. Critic Value Head & Alternating Two-Player GAE

The Critic pools representations from the normalized sequence $\mathbf{X}_{\text{norm}} = \text{norm\_f}(\mathbf{X}^{(12)}) \in \mathbb{R}^{B \times 170 \times 384}$ (prior to null vector appending):
$$\mathbf{z}_{\text{val}} = \text{GELU}(\mathbf{W}_2 \text{GELU}(\mathbf{W}_1 [\mathbf{h}_G; \, \bar{\mathbf{h}}_M; \, \bar{\mathbf{h}}_U; \, \bar{\mathbf{h}}_C]))$$
$$V(s) = \tanh(\mathbf{w}_{\text{win}}^T \mathbf{z}_{\text{val}}) \in [-1, +1]$$
where:
- $\bar{\mathbf{h}}_M = \frac{1}{121} \sum_{i=0}^{120} \mathbf{h}_{M, i}$ captures aggregate board control, fog discovery, and road networks, honoring the empirical necessity of full-trunk spatial pooling demonstrated in EXP_ARCH_001 ([`train.py:155-160`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/train.py#L155-L160)).
- $\bar{\mathbf{h}}_U = \frac{\sum_{i=1}^{32} m_{u, i} \mathbf{h}_{U, i}}{\sum_{i=1}^{32} m_{u, i} + \epsilon}$ and $\bar{\mathbf{h}}_C = \frac{\sum_{j=1}^{16} m_{c, j} \mathbf{h}_{C, j}}{\sum_{j=1}^{16} m_{c, j} + \epsilon}$ apply masked mean-pooling so inactive zero slots do not contaminate the value prediction.
- Primary output: $V(s) \in [-1, +1]$ represents the acting player's win value. Secondary progress/potential targets are excluded to eliminate multi-task value interference and potential reward distortion, honoring HenBOMB's empirical findings in [`FAILURES.md:37-40`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/FAILURES.md#L37-L40) and [`FAILURES.md:46-50`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/FAILURES.md#L46-L50).

The Critic value loss is evaluated directly against target returns:
$$\mathcal{L}_{\text{value}}(\theta) = \frac{1}{|D|} \sum_{t} \left( V_{\theta}(s_t) - y_t \right)^2$$
where value targets $y_t = \hat{A}_t + V_{\theta_{\text{old}}}(s_t)$.

### Alternating Two-Player Generalized Advantage Estimation (GAE)
Because Polytopia is a zero-sum two-player game where turns alternate after 8–15 atomic plies ([`types.rs:714`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/types.rs#L714)), standard single-agent GAE corrupts advantages across player transitions. PolyStar MoE applies **sign-inverting temporal difference and advantage propagation**:

$$\delta_t = \begin{cases} 
r_t + \gamma V(s_{t+1}) - V(s_t) & \text{if } \text{player}(s_{t+1}) = \text{player}(s_t) \text{ (intra-turn ply)} \\
r_t - \gamma V(s_{t+1}) - V(s_t) & \text{if } \text{player}(s_{t+1}) \ne \text{player}(s_t) \text{ (turn change)}
\end{cases}$$

$$\hat{A}_t = \begin{cases} 
\delta_t + \gamma \lambda \hat{A}_{t+1} & \text{if } \text{player}(s_{t+1}) = \text{player}(s_t) \text{ (intra-turn ply)} \\
\delta_t - \gamma \lambda \hat{A}_{t+1} & \text{if } \text{player}(s_{t+1}) \ne \text{player}(s_t) \text{ (turn change)}
\end{cases}$$
with discount factor $\gamma = 0.99$ and GAE trace parameter $\lambda = 0.95$. Terminal rewards at final step $T-1$ are $r_{T-1} = +1$ (acting player wins), $-1$ (acting player loses), or $0$ (draw); for all intermediate transitions $r_t = 0$. At terminal state $s_T$, $V(s_T) = 0$, giving $\delta_{T-1} = r_{T-1} - V(s_{T-1})$ and $\hat{A}_{T-1} = \delta_{T-1}$.

*Advantage Normalization*: Within each mini-batch, advantages are normalized across active trainee decisions:
$$\hat{A}_t \leftarrow \frac{\hat{A}_t - \mu_{\hat{A}}}{\sigma_{\hat{A}} + 10^{-8}}$$
This stabilizes gradient variance and prevents early policy destabilization.

*Trajectory Masking in Asymmetric League Matches*: In games played against the Greedy Heuristic Anchor or non-policy baselines, $\hat{A}_t$ is computed across the complete alternating trajectory using critic value states $V(s)$, but the PPO surrogate loss $\mathcal{L}_{\text{PPO}}(\theta)$ is strictly masked to only optimize decisions where $\text{player}(s_t) == \text{trainee\_id}$.

---

## 7. Training Protocol: Supervised Warmup + PPO League

```
  [Engine Replay Trajectories]
   (Self-Play & Heuristic Games)
               │
               ▼
   Phase 1: Supervised BC + Critic Warmup ───► Baseline ~800 Elo Policy + Calibrated Critic
                                                                   │
                                                                   ▼
   Phase 2: PPO League Matchmaker ◄────────────────────────────────┘
         ├── 25% Heuristic Anchor (Greedy)
         ├── 50% Historical Checkpoint Pool
         └── 25% Adversarial Main Exploiters
```

### 7.1 Data Schemas & Batched Trajectory Layouts

#### (a) Per-Step Candidate Move Schema (GPU Evaluation)
Because legal candidate move counts $K(s)$ vary from $\approx 4$ to $>60$, candidate sets are represented using a padded rectangular schema ($K_{\max} = 128$):
- `cand_actions`: $[B, 128]$ (`uint8`, values in $[0, 11]$)
- `cand_options`: $[B, 128]$ (`uint8`, values in $[0, 192]$)
- `cand_source_idx`: $[B, 128]$ (`uint8`, token index in $[0, 170]$)
- `cand_target_idx`: $[B, 128]$ (`uint8`, token index in $[0, 170]$)
- `cand_mask`: $[B, 128]$ (`uint8`, `1` for valid legal candidates, `0` for padding; converted to `bool` in PyTorch)
- `chosen_cand_idx`: $[B]$ (`int64`, index of executed move within $[0, K-1]$)

*Safe Overflow Truncation*: If extreme late-game states produce $K(s) > 128$, the executed move $m^*$ must be unconditionally injected into candidate slot 0 before sorting and truncating remaining candidates to 127, guaranteeing that `chosen_cand_idx` is always valid.

Candidate logits are masked with $-10^9$ where $\neg\text{cand\_mask}$ prior to softmax.

#### (b) Complete Trajectory Rollout Buffer Schema (`games_*.safetensors`)
For Phase 2 PPO rollouts and alternating GAE, trajectory buffers record both state observations, candidate sets, and reinforcement learning transition metadata:
- **Map & Entities**: `spatial_maps` ($[B, 142, 11, 11]$ `uint8`), `unit_types` ($[B, 32]$ `int64`), `unit_owners` ($[B, 32]$ `int64`), `unit_tiles` ($[B, 32]$ `int64`), `unit_features` ($[B, 32, 10]$ `float32`), `unit_mask` ($[B, 32]$ `uint8`), `city_owners` ($[B, 16]$ `int64`), `city_tiles` ($[B, 16]$ `int64`), `city_features` ($[B, 16, 8]$ `float32`), `city_mask` ($[B, 16]$ `uint8`), `global_context` ($[B, 30]$ `float32`).
- **Candidate Sets**: `cand_actions`, `cand_options`, `cand_source_idx`, `cand_target_idx`, `cand_mask` (all $[B, 128]$ `uint8`), `chosen_cand_idx` ($[B]$ `int64`).
- **RL Trajectory Metadata**:
  - `old_log_prob`: $[B]$ (`float32`, log-probability of chosen move under rollout policy $\pi_{\theta_{\text{old}}}$)
  - `values`: $[B]$ (`float32`, critic evaluation $V_{\theta_{\text{old}}}(s_t)$ recorded during rollout)
  - `rewards`: $[B]$ (`float32`, terminal $r_{T-1} \in \{-1, +1, 0\}$, intermediate $r_t = 0$)
  - `player_ids`: $[B]$ (`uint8`, acting player id, required for alternating GAE sign flips and trainee masking)
  - `dones`: $[B]$ (`uint8`, `1` at game termination, resetting GAE advantage accumulation)

### 7.2 Phase 1: Joint Supervised Behavioral Cloning & Critic Warmup
- **Storage-Optimized Data Generation**: Storing spatial maps and candidate indices as `uint8` reduces the storage footprint per step from ~76 KB down to ~19 KB. An extraction run of 12,000 games played between heuristic bots (`ai/evaluator/`) and baseline checkpoints generates ~1,000,000 training steps requiring only ~19 GB of disk storage, comfortably fitting within the host machine's 50 GB storage budget.
- **Joint Warmup Objective**: Train for 15 epochs minimizing joint behavioral cloning policy cross-entropy and critic outcome loss:
  $$\mathcal{L}_{\text{Warmup}}(\theta) = \mathcal{L}_{\text{BC}}(\theta) + c_{\text{val}} \frac{1}{|D|} \sum_{t} \left( V_\theta(s_t) - z_t \right)^2$$
  where $z_t \in \{-1, +1\}$ is the final game outcome for the acting player.
- **Outcome**: Bypasses the random tabula rasa passivity collapse ([`FAILURES.md:28-31`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/FAILURES.md#L28-L31)) and initializes the Critic with calibrated baseline evaluations, preventing advantage explosion and policy collapse in early PPO iterations.

### 7.3 Phase 2: PPO League Play
Rollout workers generate games against a 3-tier league matchmaker:
1. **Greedy Heuristic Anchor (25% of matches)**:
   Permanently retained to prevent economic drift and curriculum fadeout ([`FAILURES.md:33-35`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/FAILURES.md#L33-L35)).
2. **Historical Checkpoint Pool (50% of matches)**:
   Uniformly sampled from past checkpoints every 5 iterations, preventing cyclical strategy forgetting.
3. **Main Exploiters (25% of matches)**:
   Dedicated PPO workers trained exclusively against the active main policy under pure terminal zero-sum outcomes ($r_T \in \{-1, +1\}, r_t = 0$), strictly honoring HenBOMB's empirical rejection of intermediate potential shaping ([`FAILURES.md:37-40`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/FAILURES.md#L37-L40)). Their sole objective is to discover and punish tactical blind spots, passivity traps, or degenerate equilibria in the current main checkpoint. If an exploiter's win rate plateaus without discovering exploits across 10 iterations, its weights are re-seeded from the current main policy to probe alternative strategic vectors.

**PPO Clipped Minimization Loss**:
$$\mathcal{L}_{\text{PPO}}(\theta) = -\hat{\mathbb{E}}_t \left[ \min\left( \rho_t(\theta) \hat{A}_t, \, \text{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right] + c_{\text{val}} \mathcal{L}_{\text{value}} - c_{\text{ent}} \mathcal{S}[\pi_\theta]$$
where $\rho_t(\theta) = \frac{P_\theta(m_t \mid s_t)}{P_{\theta_{\text{old}}}(m_t \mid s_t)}$, clipping parameter $\epsilon = 0.20$, value loss coefficient $c_{\text{val}} = 0.5$, and entropy regularizer coefficient $c_{\text{ent}} = 0.01$.

---

## 8. Dual-Stack Implementation Roadmap

As required by [`CLAUDE.md:135-165`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/CLAUDE.md#L135-L165), layer definitions, tensor shapes, and state dict keys must remain strictly byte-compatible between Rust Candle and Python PyTorch.

### Complete State Dict Contract (`model.safetensors`):
- **Spatial Map**: `conv_spatial.weight` $[384, 142, 1, 1]$, `conv_spatial.bias` $[384]$, `pos_embed` $[121, 384]$
- **Unit Projector**: `proj_unit.embed_rel.weight` $[2, 384]$, `proj_unit.embed_type.weight` $[46, 384]$, `proj_unit.linear.weight` $[384, 10]$, `proj_unit.linear.bias` $[384]$
- **City Projector**: `proj_city.embed_rel.weight` $[3, 384]$, `proj_city.linear.weight` $[384, 8]$, `proj_city.linear.bias` $[384]$
- **Global Projector**: `proj_global.fc1.weight` $[384, 30]$, `proj_global.fc1.bias` $[384]$, `proj_global.fc2.weight` $[384, 384]$, `proj_global.fc2.bias` $[384]$
- **Transformer Backbone (`blocks.{0..11}`)**:
  - `blocks.{0..11}.norm1.weight` $[384]$, `blocks.{0..11}.norm1.bias` $[384]$
  - `blocks.{0..11}.attn.in_proj_weight` $[1152, 384]$, `blocks.{0..11}.attn.in_proj_bias` $[1152]$ (matching [`network.rs:70-84`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/network.rs#L70-L84) and [`train.py:95`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/train.py#L95))
  - `blocks.{0..11}.attn.out_proj.weight` $[384, 384]$, `blocks.{0..11}.attn.out_proj.bias` $[384]$
  - `blocks.{0..11}.norm2.weight` $[384]$, `blocks.{0..11}.norm2.bias` $[384]$
  - `blocks.{0..11}.mlp_spatial.fc1.weight` $[1024, 384]$, `...bias` $[1024]$, `...fc2.weight` $[384, 1024]$, `...bias` $[384]$
  - `blocks.{0..11}.mlp_tactical.fc1.weight` $[1024, 384]$, `...bias` $[1024]$, `...fc2.weight` $[384, 1024]$, `...bias` $[384]$
  - `blocks.{0..11}.mlp_macro.fc1.weight` $[1024, 384]$, `...bias` $[1024]$, `...fc2.weight` $[384, 1024]$, `...bias` $[384]$
- **Final Norm**: `norm_f.weight` $[384]$, `norm_f.bias` $[384]$
- **Move Pointer**:
  - `pointer.e_act.weight` $[12, 384]$ (12 actions covering `Resign=11`), `pointer.e_opt.weight` $[193, 384]$
  - `pointer.w_g.weight` $[384, 384]$, `pointer.w_g.bias` $[384]$
  - `pointer.w_s.weight` $[384, 384]$ (unbiased, `bias=False`)
  - `pointer.w_t.weight` $[384, 384]$ (unbiased, `bias=False`)
  - `pointer.w_st.weight` $[384, 384]$ (unbiased, `bias=False`)
- **Critic Value Head**:
  - `critic.fc1.weight` $[384, 1536]$, `critic.fc1.bias` $[384]$
  - `critic.fc2.weight` $[384, 384]$, `critic.fc2.bias` $[384]$
  - `critic.v_win.weight` $[1, 384]$, `critic.v_win.bias` $[1]$

### Sequential Rollout Milestones:
1. **Step 1 (`polyfish-rs/src/ai/features.rs`)**:
   Implement `state_to_moe_features`:
   - Spatial map tensor ($142 \times 11 \times 11$, `uint8`).
   - Unit entity synchronized tensors: continuous features `unit_features` ($32 \times 10$ `float32`), categorical `unit_types` ($32$ `int64`), `unit_owners` ($32$ `int64`), `unit_tiles` ($32$ `int64`), and boolean entity mask `unit_mask` ($32$ `uint8`).
   - City entity synchronized tensors: continuous features `city_features` ($16 \times 8$ `float32`), categorical `city_owners` ($16$ `int64`), `city_tiles` ($16$ `int64`), and boolean entity mask `city_mask` ($16$ `uint8`), querying discovered cities and neutral villages from `state.structures`.
   - Global context vector ($30$ `float32`).
   - Reverse-lookup tables `tile_to_unit_slot: [Option<usize>; 121]` and `tile_to_city_slot: [Option<usize>; 121]` to resolve raw engine move coordinates into entity tokens.
2. **Step 2 (`polyfish-rs/src/ai/network.rs` & `train_moe.py`)**:
   Implement `PolyStarMoeNet` (12-layer, $d=384$ Pre-LN Transformer with 3 sliced expert MLPs, composite mask application, scaled bilinear pointer, pure $V_{\text{win}}$ value head, and branchless null vector padding appended after `norm_f`) in Candle and PyTorch.
3. **Step 3 (`polyfish-rs/src/ai/policy_composer.rs`)**:
   Implement `compute_moe_pointer_priors` evaluating legal candidate logits via vectorized gather and scaled four-component bilinear formulations ($\mathbf{W}_G, \mathbf{W}_S, \mathbf{W}_T, \mathbf{W}_{ST}$).
4. **Step 4 (`polyfish-rs/src/bin/self_play.rs`)**:
   Add `--backend moe-policy` mode to execute rollouts using batched reactive policy evaluation without MCTS simulation loops, serializing candidate tensors (`cand_actions`, `cand_options`, `cand_source_idx`, `cand_target_idx`, `cand_mask` as `uint8`, `chosen_cand_idx` as `int64`) and trajectory metadata (`old_log_prob`, `values`, `rewards` as `float32`, `player_ids`, `dones` as `uint8`) into compressed `games_*.safetensors`.
5. **Step 5 (`polyfish-rs/train_moe.py`)**:
   Implement the PyTorch PPO training loop with alternating-turn GAE trajectory buffers, trainee advantage normalization, pure terminal $V_{\text{win}}$ surrogate minimization loss, and 3-tier league matchmaker.

---

## 9. Source Catalog & Evidence Concordance

| Specification / Empirical Grounding | Source Location in Repository | Evidence Context |
| :--- | :--- | :--- |
| **Legacy ResNet-6 Trunk** | [`network.rs:145-187`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/network.rs#L145-L187) | 6 blocks, 64 channels, ~583k parameters |
| **Intra-Turn MCTS Depth Trap** | [`BOTTLENECK.md:21-42`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/BOTTLENECK.md#L21-L42) | 8-ply path fails to exit Turn 1; combinatorial space $30^{10}$ |
| **Planar Flattening & Tech DAG** | [`BOTTLENECK.md:47-58`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/BOTTLENECK.md#L47-L58) | 2D convs unable to model DAG prerequisites ([`settings/technology.rs:470-479`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/settings/technology.rs#L470-L479)) |
| **Persistent Unit Identifiers** | [`states.rs:257`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/states.rs#L257) | `pub id: u32` minted at creation, persistent across plies |
| **UnitType Variants Count (46)** | [`types.rs:268-333`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/types.rs#L268-L333) | 46 active unit variants mapped sequentially via `UNIT_INDEX` ([`features.rs:162-165`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/features.rs#L162-L165)) |
| **City State Schema** | [`states.rs:291-311`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/states.rs#L291-L311) | Level, progress, border size, production, territory |
| **Neutral Villages on Structures** | [`scoring.rs:1222`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/scoring.rs#L1222) | Neutral villages reside on `structures` as `StructureType::Village` |
| **GameState Tribes IndexMap** | [`states.rs:605`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/states.rs#L605) | Multi-tribe state storing active and enemy units/cities |
| **Fog Memory Spatial Channels** | [`features.rs:116-125`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/features.rs#L116-L125) | 6 decayed enemy observation memory channels |
| **Total Spatial Channels (142)** | [`features.rs:128`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/features.rs#L128) | `pub const NUM_CHANNELS: usize = CH_MEM_END;` |
| **Map Grid Dimension (11x11)** | [`features.rs:17`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/features.rs#L17) | `pub const MAP_SIZE: usize = 11;` |
| **Action Variant Mapping (12)** | [`mapper.rs:82-97`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/mapper.rs#L82-L97) | 12 move variants mapped from [`types.rs:702-716`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/types.rs#L702-L716) |
| **Target-Only Build & Harvest** | [`moves/build.rs:105`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/moves/build.rs#L105), [`moves/harvest.rs:74-76`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/moves/harvest.rs#L74-L76) | Action types with spatial targets but no source entity |
| **Source-Only Capture Move** | [`moves/capture.rs:180-182`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/moves/capture.rs#L180-L182) | Capture move providing source index with no target index |
| **Policy Multiplication Defect** | [`policy_composer.rs:69-105`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/policy_composer.rs#L69-L105) | Independent marginal products cause source/target confusion |
| **192 Action Options + Reserved Slot** | [`train.py:148`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/train.py#L148), [`mapper.rs:38-46`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/mapper.rs#L38-L46) | 192 options plus index 192 reserved for NO_OPTION plies |
| **Dual-Stack Sync Requirement** | [`CLAUDE.md:135-165`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/CLAUDE.md#L135-L165) | Candle and PyTorch byte-compatibility and safetensors contract |
| **Attention Weight Splitting** | [`network.rs:70-84`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/network.rs#L70-L84), [`train.py:95`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/train.py#L95) | Candle in-proj splitting matching PyTorch MultiheadAttention |
| **Negative Gradient Transfer** | [`FAILURES.md:46-50`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/FAILURES.md#L46-L50) | EXP_ELO_069: Shared trunk dropped win rate by 13% |
| **Timescale Conflict Diagnosis** | [`FAILURES.md:64-72`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/FAILURES.md#L64-L72) | Mismatch between macro (turns), meso (phases), and micro (plies) |
| **Tabula Rasa Passivity Collapse** | [`FAILURES.md:28-31`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/FAILURES.md#L28-L31) | Mutual non-aggression; captures dropped from 6.5 to 3.2 |
| **Curriculum Anchor Fadeout Defect** | [`FAILURES.md:33-35`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/FAILURES.md#L33-L35) | Fading anchor to 5% crashed win rate from 81% to 25% |
| **Self-Play Throughput Baseline** | [`expert_boost_throughput.md:36`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/expert_boost_throughput.md#L36) | ~578 moves/s bound by CPU↔GPU sync stalls |
