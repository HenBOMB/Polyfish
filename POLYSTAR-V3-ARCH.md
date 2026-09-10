# PolyStar v3.1: Entity-Transformer Policy with League PPO

**Status:** specification, pre-implementation. v3.1 is the 2026-09-10 code-level review pass over the v3.0 draft (Appendix C).
**Supersedes:** PolyStar MoE v2.2 (git `07bef810`, then named `POLYSTAR-MOE-ARCH.md`) and the v3.0 draft of this file. Renamed to `POLYSTAR-V3-ARCH.md` on 2026-09-10 because the Mixture-of-Experts backbone survives only as a deferred ablation (§4.4).

This document specifies a dense-first Entity-Map Transformer policy (7.21M-parameter **S** baseline, 23.2M-parameter **M** scale-up) and a model-free reinforcement learning pipeline (supervised behavioral-cloning warmup → league PPO) designed to succeed legacy PolyZero.

It resolves the structural MCTS limits in [`BOTTLENECK.md:21-71`](BOTTLENECK.md#L21-L71), the calibration-vs-discrimination breakdown in [`FAILURES.md:46-50`](FAILURES.md#L46-L50), the bilinear pointer blind spot on target-only actions, the alternating two-player advantage sign corruption, padded-entity leakage across attention residuals, and null-token index bounds. Relative to v2.2 it also fixes the per-ply discount, the diplomacy index aliasing, the candidate-set cap, the dataset arithmetic, the throughput rationale, and the dual-implementation requirement (Appendix B lists every change). The v3.1 review pass (Appendix C) fixed the TorchScript return path, a feature plane that saturates under `uint8`, an ability name that does not exist in the engine, the ply-count arithmetic and the meaning of the teacher targets, widened the unit slot budget, and replaced the alternating-perspective GAE with own-ply trajectories.

---

## 0. Decision Record (2026-09-10)

| # | Decision | Choice | Rationale |
| :--- | :--- | :--- | :--- |
| D1 | Search paradigm | Drop MCTS from the training loop; single-pass reactive policy. Shallow search on top of the policy stays available at evaluation time as an ablation (§7.4) | Intra-turn depth trap (§1.1) and the latency structure of 64 sequential forwards per move (§2). Read honestly, EXP 3 ([`hypothesis_driven_improvements.md:43-53`](hypothesis_driven_improvements.md#L43-L53)) shows search *helps*: 64→256 sims raised first-village capture rate 0.81→1.00 and cut conditional capture time 7.9→7.3 turns, and was rejected for 2.3× wall-clock. D1 rests on cost, not on search being useless. |
| D2 | Model size | Start at **S** ($d=256$, 8 layers, 7.21M); move to **M** ($d=384$, 12 layers, 23.2M) when BC held-out accuracy plateaus (§4.3) | The July 2026 "modest capacity steps only" rule was written for CPU self-play under MCTS and for the dual-network blast radius; D1 and D3 remove both premises. S is cheap enough to shake the pipeline down in hours. |
| D3 | Inference stack | One PyTorch definition; TorchScript export; loaded in Rust through `tch` (`tch-eval` feature) | No Candle mirror, no byte-compatibility contract to maintain by hand (§8) |
| D4 | Backbone | Dense Pre-LN FFN + learned modality-type embeddings; MoE experts deferred to an ablation | The v2.2 "gradient isolation" claim was overstated and its evidence misattributed (§1.3, §4.4) |
| D5 | BC data budget | 3,000 teacher games ≈ 1.39M steps ≈ 30 GB unpacked | Fits the 50 GB host budget with ~20 GB for PPO rollouts (≈ 2.5 GB per 256-game iteration, §7.1); index-pack (§7.1) before growing to 12,000 games |
| D6 | League | 25% Greedy anchor floor + 75% PFSP-weighted checkpoint pool; **no exploiters** in v3.1 | Exploiters are a second training stream; the 20% anchor floor already stopped the 81%→25% forgetting crash ([`DISCOVERIES.md:54-60`](DISCOVERIES.md#L54-L60)) |
| D7 | Candidate layout | Ragged (CSR), no cap | Random-agent measurement: max 244 legal moves, 174 by turn 28 (Appendix A.1) |
| D8 | Discount | $\gamma = 1.0$ per ply; optional per-turn $\gamma_{\text{turn}} = 0.995$ applied only at EndTurn transitions | $0.99^{464} \approx 0.009$ erased the outcome signal for all but the last ~10 turns (§6) |
| D9 | Entity slot budget | 64 unit slots, 16 city slots (sequence 202 + null) | Summon allows level + 1 units per city (`summon.rs:130-133`); two sides with four or five mid-level cities approach 40 units, so 32 would have missed Attack targets, and the count is baked into every recorded game (§3.2) |
| D10 | Advantage estimation | Own-ply trajectories: the opponent's turn is part of the environment transition; no perspective flips, no critic on opponent plies | Removes one extra trainee forward per opponent ply in every asymmetric game and keeps the critic on-policy (§6) |
| D11 | Move pointer targets | Capture and unit-centered abilities point at the acting unit's own tile; option-free actions score through a 2-layer MLP of the global token | A null target hid what is being captured; a linear probe of $\mathbf{h}_G$ was the entire decision function for Research and EndTurn (§5) |

---

## 1. Executive Summary & Paradigm Shift

Standard AlphaZero/MCTS fails on *The Battle of Polytopia* due to four fundamental domain mismatches:

1. **The Intra-Turn Depth Trap**: A single turn requires 8–15 atomic plies ending in `MoveType::EndTurn` ([`polyfish-rs/src/types.rs:714`](polyfish-rs/src/types.rs#L714); `training_results/moves_by_turn.json` shows ≈8 plies per turn at turn 10). A 64-simulation MCTS path exhausts its budget permuting intra-turn move orders without penetrating into opponent turns ([`BOTTLENECK.md:21-42`](BOTTLENECK.md#L21-L42), [`notes.md:182-184`](notes.md#L182-L184)).
2. **Planar Representation Flattening**: Forcing non-spatial Directed Acyclic Graphs (the Tech Tree in [`settings/technology.rs:470-479`](polyfish-rs/src/settings/technology.rs#L470-L479)), global star counters ([`states.rs:389`](polyfish-rs/src/states.rs#L389)), and discrete unit entities into a 2D convolutional grid destroys relational structure ([`BOTTLENECK.md:47-58`](BOTTLENECK.md#L47-L58)).
3. **Calibration ≠ Discrimination (EXP_ELO_069)**: The neural value leaf lost **13.1 percentage points** against the heuristic leaf in the arena (44.0% → 30.9%) even after late-turn calibration reached $R^2 = 0.972$ ([`FAILURES.md:46-50`](FAILURES.md#L46-L50), `DISCOVERIES.md` §2.7). Negative transfer from auxiliary macro heads on the shared 64-channel trunk is reported *qualitatively* in the same section, without a number; v2.2 cited the 13.1-point figure as if it measured that interference. **Consequence for v3.1:** the baseline trains exactly one policy loss and one value loss on the trunk and uses no potential shaping (`vlab_*` rejected, `DISCOVERIES.md` §2.6). Auxiliary *prediction* heads (e.g. the end-of-game ownership target `train.py` already carries as `v_ownership`) are not shaping and are permitted as a BC-phase ablation, where held-out top-1 agreement is a clean metric; the v3.0 blanket ban rested on the same misattributed evidence.
4. **Policy Decomposition Collapse**: Multiplying unconditioned marginal head distributions in [`policy_composer.rs:69-105`](polyfish-rs/src/ai/policy_composer.rs#L69-L105) causes source and target coordinates to mismatch.

**PolyStar v3.1 replaces flat MCTS with a direct reactive Entity-Transformer policy** trained via **Supervised Behavioral Cloning Warmup followed by Alternating Two-Player PPO League Play**:
- **Global Self-Attention** provides empire-wide communication between terrain, units, cities, and technology.
- **Learned modality-type embeddings** (map / unit / city / global) give each token class its own inductive bias without splitting the feed-forward path (§3.5).
- **Composite masking after every residual** prevents padded entity slots from leaking into the sequence (§4.1).
- **Four-Component Context-Conditioned Bilinear Move Pointer with Appended Null Vector** scores candidate moves dynamically across spatial, entity, and non-spatial actions without autoregressive stalls, target-only blind spots, or tensor out-of-bounds indexing (§5).
- **Ragged candidate sets** and an **exhaustive move-type resolution table** guarantee every legal move is scored, including the diplomacy moves whose engine indices are not tiles (§5.1, §7.1).

---

## 2. Architectural Comparison

| Dimension | Legacy PolyZero ([`network.rs:145-187`](polyfish-rs/src/ai/network.rs#L145-L187)) | PolyStar v3.1 | Grounding |
| :--- | :--- | :--- | :--- |
| **Search Paradigm** | 64-iteration Gumbel MCTS | Model-free reactive policy (single pass); shallow search optional at eval (§7.4) | Eliminates intra-turn search horizon depletion ([`BOTTLENECK.md:32`](BOTTLENECK.md#L32)) |
| **Total Parameters** | 583,441 (~2.3 MB; recomputed from `train.py`) | **S: 7,210,753 (28.8 MB)** → M: 23,173,249 (92.7 MB) | Size ladder (§4.3); capacity deficit ([`BOTTLENECK.md:59-71`](BOTTLENECK.md#L59-L71)) |
| **Backbone Structure** | 6 ResBlocks, 64 channels | S: 8-layer Pre-LN Transformer ($d=256$, 4 heads, FFN 1024); M: 12 layers ($d=384$, 6 heads, FFN 1536) | Relational attention |
| **Sequence Length** | N/A (planar 2D convolutions) | 202 active tokens ($121$ map $+ 64$ units $+ 16$ cities $+ 1$ global) + 1 appended null token | Unified entity-spatial attention with branchless gather |
| **Unit Tracking** | Anonymous grid cells | Persistent Unit IDs ([`states.rs:257`](polyfish-rs/src/states.rs#L257)) | Tracks individual unit identities across plies |
| **Move Scoring** | 4 multiplied marginal heads | 4-component bilinear move pointer (MLP $g(\mathbf{h}_G)$; unbiased $\mathbf{W}_S, \mathbf{W}_T, \mathbf{W}_{ST}$) | Fixes multiplication collapse and target-only blind spots (§5) |
| **Compute per move** | 0.13 GFLOP/forward × 64 forwards ≈ **8.1 GFLOP** | S: **2.9 GFLOP** (1 forward); M: 9.3 GFLOP | ~2.8× less (S) / ~1.15× more (M) compute per move than the legacy 64-sim search — *not* the 64× claimed in v2.2 (Appendix A.2) |
| **Pipeline shape** | 64 sequential CPU↔GPU round-trips per move; GPU 2.5% busy ([`expert_boost_throughput.md`](expert_boost_throughput.md)) | One batched forward per ply | The real throughput lever is latency structure, not FLOPs |
| **Move Throughput** | ~578 moves/s (Metal, M3 Max; [`expert_boost_throughput.md:36`](expert_boost_throughput.md#L36)) | **Target 2,000+ moves/s — to be measured**; spec value = min(actor ceiling, GPU ceiling) (Appendix A.3) | Not a derived number |
| **Opponent Pool** | Single-checkpoint self-play (+ anchor gate) | 25% Greedy floor + 75% PFSP checkpoint pool | Eliminates cyclical forgetting ([`FAILURES.md:33-35`](FAILURES.md#L33-L35)) |
| **Implementation** | Candle + PyTorch + tch + Metal, kept byte-compatible by hand | One PyTorch definition; TorchScript loaded by `tch` | Removes the dual-network sync constraint (§8) |

---

## 3. Structural Tokenization Specification

The input game state is tokenized into an ordered sequence of 202 active tokens of dimension $d_{\text{model}}$ ($256$ for S, $384$ for M):
$$\mathbf{X} = [\mathbf{M}_{0..120}; \, \mathbf{U}_{0..63}; \, \mathbf{C}_{0..15}; \, \mathbf{G}_0] \in \mathbb{R}^{202 \times d}$$

```
[11x11 Spatial Map]           [Visible Unit Entities]           [City Entities]           [Global Context]
  (145 Channels)                (Up to 64 Units)                 (Up to 16 Cities)        (Techs, Stars, Turn, Tribe)
        │                               │                               │                         │
        ▼                               ▼                               ▼                         ▼
   Pointwise Conv                Unit Projector                  City Projector            Global Projector
 [121 tokens x d]               [64 tokens x d]                 [16 tokens x d]            [1 token x d]
   + E_type[map]                  + E_type[unit]                  + E_type[city]            + E_type[global]
        │                               │                               │                         │
        └───────────────────────────────┴───────────────┬───────────────┴─────────────────────────┘
                                                        ▼
                                Concatenated Sequence [202 tokens x d], masked by M at Layer 0
```

### 3.1 Spatial Map Tokens ($M \in \mathbb{R}^{121 \times d}$, Slice `0..121`)
- **Input Channels: 145** = the 142 legacy planes of [`features.rs:128`](polyfish-rs/src/ai/features.rs#L128) (8 terrain + 10 tile flags + 9 resources + 35 structures + 46 unit types + 16 unit stats + 12 city stats + 6 fog-memory) with each of the three owner planes (`CH_TILE_OWNER`, `CH_UNIT_OWNER`, `CH_CITY_OWNER`; legacy values $\{-1, 0, +1\}$ at [`features.rs:399-405, 488-490, 589-591`](polyfish-rs/src/ai/features.rs#L399-L405)) **split into two binary planes** (`*_OWNER_SELF`, `*_OWNER_ENEMY`). The split happens in the new extractor only; the legacy `state_to_features` and `NUM_CHANNELS = 142` are untouched. Rationale: a $1 \times 1$ conv is linear, so the v2.2 remap $\{1.0, 0.5, 0.0\}$ made "enemy" the midpoint of "self" and "neutral" before the first nonlinearity.
- **Value range and `uint8` quantization**: with the owner planes split and one legacy fix, every plane is in $[0, 1]$ (binary flags; `climate/17` with `TribeType` ∈ 0..17; `hp/max_hp`; `kills/3` clamped; memory `attack/5` with max attack 5.0; decayed memory in $(0, 1]$). **The fix:** the legacy `CH_UNIT_PASSENGER_TYPE` divides the raw `UnitType` discriminant by 46 (`features.rs:562`) and discriminants reach 62 (`types.rs:268-333`), so the legacy value reaches 1.35 and would silently saturate at 255; the new extractor writes `UNIT_INDEX[passenger] / 46` (the sequential index, `features.rs:162-165`). `CH_UNIT_MAX_HP` is hard-coded 0.0 (`features.rs:498`) and is carried as a constant-zero plane so the legacy channel indices stay valid. Rust serializes `(feat * 255.0).round().clamp(0.0, 255.0) as u8`, `debug_assert!`s $0 \le v \le 1$ per value, and counts out-of-range values in a `feat_clip` METRICS counter in release builds, which must read 0; PyTorch dequantizes with `spatial_maps.float() / 255.0`.
- **Projection**: $1 \times 1$ pointwise convolution `conv_spatial` maps $145 \to d$.
- **Position Embedding**: learnable $\mathbf{E}_{\text{pos}} \in \mathbb{R}^{121 \times d}$ indexed per tile ([`features.rs:17`](polyfish-rs/src/ai/features.rs#L17)), shared with unit and city tokens to ground entities in map coordinates.

### 3.2 Visible Unit Entity Tokens ($U \in \mathbb{R}^{64 \times d}$, Slice `121..185`)
Iterates over all units visible to the active player across all tribes in `GameState::tribes` ([`states.rs:605`](polyfish-rs/src/states.rs#L605)):
1. **Selection & Truncation Priority**:
   - Priority 1: Active player units from `tribe.units` ([`states.rs:408`](polyfish-rs/src/states.rs#L408)). If they exceed 64, sort descending by combat value ($\text{hp} \times \text{attack}$).
   - Priority 2: Spotted enemy units that are the target of a legal `Attack` this ply (so no Attack candidate can miss a slot), then those within movement or attack range of any friendly unit or threatening friendly cities.
   - Priority 3: Other spotted enemy units, sorted ascending by Chebyshev distance to the nearest friendly city.
   - *Budget check*: `generate_summon_moves` blocks training once a city's unit count *exceeds* its level (`polyfish-rs/src/moves/summon.rs:130-133`), i.e. a city may hold level + 1 units, so a side's army is bounded by $\sum_c (\text{level}_c + 1)$. Two sides with four or five level-3–5 cities approach 40 units, and the random-agent maximum of 244 legal moves (Appendix A.1) implies more. The v3.0 budget of 32 for both sides would have overflowed exactly in the late-game fights the Attack bilinear term exists for, and the slot count is baked into every recorded game. **v3.1 uses 64 slots** (sequence 202 vs 170, ≈ +21% attention FLOPs, Appendix A.2). Any move whose acting or target unit is still **not** in a slot falls back to the raw tile token and increments the `entity_slot_miss` METRICS counter (§5.1); that counter must stay ≈0 or the budget is raised again *before* the corpus is recorded.
2. **Token Construction & Input Zero-Masking**:
   $$\mathbf{t}_{u, i} = m_{u, i} \cdot \left( \text{Embed}_{\text{rel}}(\text{owner\_rel}) + \text{Embed}_{\text{type}}(\text{unit\_type}) + \text{Embed}_{\text{pass}}(\text{passenger\_type}) + \text{Linear}_{u}(\mathbf{f}_u) + \mathbf{E}_{\text{pos}}[\text{tile\_idx}] + \mathbf{E}_{\text{type}}[\text{unit}] \right)$$
   Multiplying by the boolean entity mask $m_{u, i}$ at Layer 0 guarantees that padding slots carry no embedding bias, no $\mathbf{E}_{\text{pos}}[0]$, and no type embedding into the backbone.
3. **Entity Tensor Schema** (all categorical fields stored as `uint8`, cast with `.long()` in the loader):
   - `unit_types`: $[B, 64]$, values in $[0, 45]$ via `UNIT_INDEX` ([`features.rs:162-165`](polyfish-rs/src/ai/features.rs#L162-L165)). **Index 0 is `UnitType::None` and doubles as the padding type.** (The mapper's `UNIT_MAP`, used for Summon option indices, excludes `None`, so the two index spaces differ by one; they are never mixed.)
   - `unit_passenger_types`: $[B, 64]$, the carried unit's `UNIT_INDEX` (0 = none) through its own 46-row table $\text{Embed}_{\text{pass}}$; this, not the legacy scalar plane, is the model's view of what a boat carries.
   - `unit_owners`: $[B, 64]$ ($\text{Self}=0, \text{Enemy}=1$)
   - `unit_tiles`: $[B, 64]$, tile index in $[0, 120]$ indexing $\mathbf{E}_{\text{pos}}$
   - `unit_features`: $[B, 64, 10]$ `float32`
   - `unit_mask`: $[B, 64]$ (1 active, 0 padding)
4. **Unit Continuous Features ($\mathbf{f}_u \in \mathbb{R}^{10}$)**: $\text{hp}/\text{max\_hp}$; veteran; $\min(\text{kills}/3, 1)$; moved; attacked; frozen ([`features.rs:90`](polyfish-rs/src/ai/features.rs#L90)); boosted ([`features.rs:88`](polyfish-rs/src/ai/features.rs#L88)); poisoned ([`features.rs:87`](polyfish-rs/src/ai/features.rs#L87)); has\_passenger ([`features.rs:91`](polyfish-rs/src/ai/features.rs#L91)); converted.

### 3.3 City Entity Tokens ($C \in \mathbb{R}^{16 \times d}$, Slice `185..201`)
Iterates over all discovered cities and neutral villages:
1. **Selection & Discovery Query**:
   - Priority 1: Active player cities from `tribe.cities` ([`states.rs:406`](polyfish-rs/src/states.rs#L406)).
   - Priority 2: Discovered enemy cities from opponent `tribe.cities`.
   - Priority 3: Discovered neutral villages from `state.structures` where `structure_type == StructureType::Village` (no `CityState` exists before capture; [`scoring.rs:1222`](polyfish-rs/src/ai/scoring.rs#L1222)), with baseline features $\text{level}=0, \text{progress}=0, \text{production}=0, \text{border\_size}=1$ and all flags $0$.
2. **Token Construction & Input Zero-Masking**:
   $$\mathbf{t}_{c, j} = m_{c, j} \cdot \left( \text{Embed}_{\text{rel}}(\text{owner\_rel}) + \text{Linear}_{c}(\mathbf{f}_c) + \mathbf{E}_{\text{pos}}[\text{city.idx}] + \mathbf{E}_{\text{type}}[\text{city}] \right)$$
3. **City Tensor Schema**: `city_owners` $[B, 16]$ `uint8` ($\text{Self}=0, \text{Enemy}=1, \text{Neutral}=2$); `city_tiles` $[B, 16]$ `uint8`; `city_features` $[B, 16, 9]$ `float32`; `city_mask` $[B, 16]$ `uint8`.
4. **City Continuous Features ($\mathbf{f}_c \in \mathbb{R}^9$)**, normalized exactly as the legacy extractor does so the two stay comparable:
   - $\min(\text{level}/10, 1)$ (v2.2 used $/8$; cities exceed level 8)
   - $\min(\text{progress}/(\text{level}+1), 1)$ (the progress needed to level up is $\text{level}+1$)
   - $\min(\text{production}/10, 1)$ via `get_city_production`
   - $\text{border\_size}/2 \in \{0.5, 1.0\}$
   - $\min(\text{units}_c / (\text{level} + 1), 1)$ via `get_city_unit_count`: the summon cap (`summon.rs:130-133`). The policy can infer it from whether a Summon candidate exists; the critic cannot.
   - connected, is\_capital, has\_walls, has\_riot $\in \{0, 1\}$. These are **derived**, not `CityState` fields ([`states.rs:291-311`](polyfish-rs/src/states.rs#L291-L311)): `city.connected_to_capital`, `tile.capital_of == player`, `city.has_walls()`, and the riot flag, exactly as the legacy city block does (`polyfish-rs/src/ai/features.rs:596-640`).

### 3.4 Global Context Token ($G \in \mathbb{R}^{1 \times d}$, Slice `201..202`)
- **Technology Vector**: 25-dimensional binary mask over the vanilla technologies, in `TECH_MAP` order (`TechnologyType` minus `Basic`, `BeyondComprehension`, and the 15 tribe-locked Polaris/Cymanti/Aquarion/Elyrion entries in `settings/technology.rs`), read from `tech_vanilla[i].discovered` ([`states.rs:387-405`](polyfish-rs/src/states.rs#L387-L405)). Sufficient for the configured training tribes (Imperius, Bardur, Oumaji, Kickoo, XinXi — all vanilla; `training_results/config.json`). Widen to the mapper's 40 research slots if a special tribe is ever added.
- **Scalars** (7): $\min(\text{stars}/100, 1)$; $\min(\text{score}/10000, 1)$ (the training log shows `max_score` 10,200); $\text{turn}/50$; $(\text{max\_turns} - \text{turn})/50$; $\text{max\_turns}/50$ (the curriculum varies the cap: 10/15/20/45, `polyfish-rs/src/bin/self_play.rs:201-211`); $\text{SPT}/50$; $\text{score\_delta}/5000 \in [-1, 1]$ (active player score minus visible opponent score).
- **Tribe identity**: `tribe_ids` $[B, 2]$ `uint8` (own tribe, known opponent tribe or 0). Two embedding tables $\mathbf{E}_{\text{tribe}}, \mathbf{E}_{\text{tribe,opp}} \in \mathbb{R}^{18 \times d}$ (`TribeType` 0..17) are summed into the global token. v2.2 carried no tribe identity anywhere; the legacy `player_vec` did (`tribe_type_norm`, `max_turns_norm`).
- **Projection**: 2-layer MLP ($32 \to d \to d$, GELU) `proj_global`, plus $\mathbf{E}_{\text{type}}[\text{global}]$. The global token is never masked.

### 3.5 Modality-Type Embedding
A learned table $\mathbf{E}_{\text{type}} \in \mathbb{R}^{4 \times d}$ (map = 0, unit = 1, city = 2, global = 3) is added to every token at input, before the Layer-0 mask. This is the standard entity-transformer device for giving token classes distinct processing without splitting the feed-forward path; v3.1 uses it in place of v2.2's modality-routed experts (§4.4).

---

## 4. Dense Pre-LN Transformer Backbone

```
                     ┌───────────────────────────────────────────────┐
                     │ Concatenated Sequence Input [202 tokens x d]  │
                     │ (masked by M at Layer 0)                      │
                     └───────────────────────┬───────────────────────┘
                                             ▼
                     ┌───────────────────────────────────────────────┐
                     │ Pre-LN Multi-Head Self-Attention              │
                     │ (S: 4 heads x 64, M: 6 heads x 64)            │
                     │ key_padding_mask = ~M  (padded keys ignored)  │
                     └───────────────────────┬───────────────────────┘
                                             ▼
                     ┌───────────────────────────────────────────────┐
                     │ Residual Addition, then Mask M                │
                     └───────────────────────┬───────────────────────┘
                                             ▼
                     ┌───────────────────────────────────────────────┐
                     │ Pre-LN Dense FFN  d -> d_ffn -> d, GELU       │
                     │ (S: d_ffn = 1024, M: d_ffn = 1536)            │
                     └───────────────────────┬───────────────────────┘
                                             ▼
                     ┌───────────────────────────────────────────────┐
                     │ Residual Addition, then Mask M                │
                     └───────────────────────┬───────────────────────┘
                                             ▼
                              (Repeated x L layers: S = 8, M = 12)
                                             ▼
                     ┌───────────────────────────────────────────────┐
                     │ norm_f, Mask M, append null token -> X_eval   │
                     └───────────────────────────────────────────────┘
```

### 4.1 Layer Forward Pass Formulation
Let $\mathbf{M} \in \{0, 1\}^{202 \times 1}$ be the composite sequence mask:
$$\mathbf{M} = [\mathbf{1}_{121}; \, \mathbf{M}_U; \, \mathbf{M}_C; \, 1]$$
where $\mathbf{1}_{121}$ covers the spatial tiles, $\mathbf{M}_U$ the 64 unit slots, $\mathbf{M}_C$ the 16 city slots, and the trailing $1$ keeps the global token permanently active.

For each block $l \in [1, \dots, L]$:
$$\mathbf{X}' = \mathbf{M} \odot \left( \mathbf{X}^{(l-1)} + \text{MHA}(\text{LN}_1(\mathbf{X}^{(l-1)})) \right)$$
$$\mathbf{X}^{(l)} = \mathbf{M} \odot \left( \mathbf{X}' + \text{FFN}(\text{LN}_2(\mathbf{X}')) \right)$$

Two masks do two different jobs: the key mask (`key_padding_mask = ~composite_mask.bool()`, `True` = ignore) stops active tokens from attending to padding; the composite mask after each residual stops padded rows from accumulating the LayerNorm bias $\beta$ and phantom activations across layers. Both are required.

Implement attention with `F.scaled_dot_product_attention` and an explicit boolean mask rather than `nn.MultiheadAttention`: the module's eval-mode nested-tensor fast path changes numerics under `key_padding_mask` relative to training, which would make the fp32 parity test of §8 step 4 flaky and is invisible inside TorchScript.

Since the network exists only in PyTorch (D3), the v2.2 Candle additive-bias convention is dropped; TorchScript carries the PyTorch semantics into Rust unchanged.

### 4.2 Feed-Forward Block
$$\text{FFN}(\mathbf{z}) = \mathbf{W}_{2} \, \text{GELU}(\mathbf{W}_{1} \, \mathbf{z} + \mathbf{b}_{1}) + \mathbf{b}_{2}, \qquad \mathbf{W}_{1} \in \mathbb{R}^{d_{\text{ffn}} \times d}, \; \mathbf{W}_{2} \in \mathbb{R}^{d \times d_{\text{ffn}}}$$

### 4.3 Size Ladder & Parameter Budget

| Config | $d$ | Layers | Heads | $d_{\text{ffn}}$ | Parameters | fp32 size | GFLOP / forward (seq 202) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **S (v3.1 baseline)** | 256 | 8 | 4 | 1024 | **7,210,753** | 28.8 MB | 2.88 |
| M (scale-up) | 384 | 12 | 6 | 1536 | 23,173,249 | 92.7 MB | 9.35 |
| MoE v2.2 (reference only, seq 170) | 384 | 12 | 6 | 3 × 1024 | 37,173,505 | 148.7 MB | 6.16 |

Per-component budget (weights + biases):

| Component | Shape rule | S | M |
| :--- | :--- | :--- | :--- |
| Attention projections | $L \times 4 \times (d^2 + d)$ | 2,105,344 | 7,096,320 |
| Dense FFN | $L \times (2 d \, d_{\text{ffn}} + d_{\text{ffn}} + d)$ | 4,204,544 | 14,178,816 |
| LayerNorms ($2L + 1$) | $(2L+1) \times 2d$ | 8,704 | 19,200 |
| `conv_spatial` ($145 \to d$) | $145d + d$ | 37,376 | 56,064 |
| $\mathbf{E}_{\text{pos}}$ | $121 d$ | 30,976 | 46,464 |
| $\mathbf{E}_{\text{type}}$ | $4d$ | 1,024 | 1,536 |
| Unit projector | $2d + 46d + 46d + 10d + d$ | 26,880 | 40,320 |
| City projector | $3d + 9d + d$ | 3,328 | 4,992 |
| Global projector + tribe embeddings | $32d + d + d^2 + d + 2 \times 18d$ | 83,456 | 174,336 |
| Move pointer embeddings | $12d + 193d$ | 52,480 | 78,720 |
| Move pointer projections | $2(d^2 + d) + 3d^2$ | 328,192 | 738,048 |
| Critic value head | $4d \cdot d + d + d^2 + d + d + 1$ | 328,449 | 738,433 |
| **Total** | | **7,210,753** | **23,173,249** |

**Promotion rule S → M**: promote when held-out BC top-1 agreement stops improving for 3 consecutive epochs at S, or when PPO gate G2 (§7.3) stalls for 30 iterations while the pool still beats Greedy. M is a config change only; nothing else in this document depends on $d$.

### 4.4 Deferred Ablation: Modality-Routed Expert FFNs (the v2.2 design)
The v2.2 backbone replaced the dense FFN with three FFNs sliced by token range (spatial `0..121`, tactical = the unit slots, macro = the city slots + global; `121..153` / `153..170` in v2.2, `121..185` / `185..202` in the v3.1 layout), each token passing through exactly one. It is kept here as an ablation, with three corrections to how it was justified:
1. Attention and the residual stream remain shared, so the experts isolate FFN *weights* only; gradients still cross modalities in every layer. This is an inductive-bias hypothesis, not "strict gradient isolation".
2. The negative-transfer evidence cited for it (EXP_ELO_069) measured value-leaf discrimination, not head interference (§1.3). With one policy loss and one value loss there is no multi-task conflict to isolate.
3. A 1024-wide macro expert trained on ≤17 tokens per sample is under-constrained. If run, size experts to their token counts — spatial 1024 / tactical 512 / macro 256 hidden at $d = 384$ (16,550,400 FFN parameters versus 14,178,816 for dense-1536) — and compare at equal wall-clock against M.

Run only after PPO gate G2 passes at M.

---

## 5. Four-Component Bilinear Move Pointer

This mechanism resolves the head multiplication collapse in [`policy_composer.rs:69-105`](polyfish-rs/src/ai/policy_composer.rs#L69-L105) and fixes the state-independent logit defect for target-only moves (`Build`, `Harvest`, `Reward`).

### 5.1 Appended Null Representation for Branchless Gather
The output of the final layer $\mathbf{X}^{(L)} \in \mathbb{R}^{B \times 202 \times d}$ is normalized by `norm_f`, masked, and appended with an explicit all-zero token at index $202$:
$$\mathbf{X}_{\text{norm}} = \mathbf{M} \odot \text{norm\_f}(\mathbf{X}^{(L)}), \qquad \mathbf{X}_{\text{eval}} = [\mathbf{X}_{\text{norm}}; \, \mathbf{0}_{B \times 1 \times d}] \in \mathbb{R}^{B \times 203 \times d}$$

*Critical Order Invariant*: the mask is applied **after** `norm_f` and **before** the null token is appended. This suppresses $\beta_{\text{norm\_f}}$ on inactive entity slots $[121, 200]$ and keeps index $202$ exactly $\mathbf{0}$, so gathering either through the bias-free projections $\mathbf{W}_S, \mathbf{W}_T, \mathbf{W}_{ST}$ yields zero without branch conditions.

1. The Rust engine queries legal moves $L(s) = [m_1, \dots, m_K]$ via `game.legal_moves()` (`polyfish-rs/src/game.rs:262`, wrapping `generate_legal_moves`, `polyfish-rs/src/moves/mod.rs:195`). When a city level-up reward is pending, the generator returns **only** the reward moves (`moves/mod.rs:202-205`), so that candidate set is exactly the pending reward moves: two per city with an outstanding reward (`reward.rs:206-235`). One road connection can level two cities at once, so 4+ is possible; the ragged layout carries it.
2. Each legal move $m_i$ provides:
   - Action type $a \in [0, 11]$ via `DecomposedMapper::move_type_to_idx` ([`mapper.rs:82-97`](polyfish-rs/src/ai/mapper.rs#L82-L97)): `None=0`, `Attack=1`, `Step=2`, `Capture=3`, `Ability=4`, `Summon=5`, `Harvest=6`, `Build=7`, `Research=8`, `Reward=9`, `EndTurn=10`, `Resign=11`. The engine discriminants (`Step=1`, `Attack=2` in [`types.rs:705-706`](polyfish-rs/src/types.rs#L705-L706)) are transposed by the mapper. `Resign` is never emitted by `generate_legal_moves`; slot 11 is inert.
   - Option index $o \in [0, 192]$ ([`mapper.rs:38-45`](polyfish-rs/src/ai/mapper.rs#L38-L45)): structures `0..48`, units `48..112`, techs `112..160`, abilities `160..=180` (21 variants in `AbilityType::iter()` order), rewards `181..=188`, **`189 = OPT_PEACE_REJECT` (new)**, `190` unused, `191` legacy reward fallback, **`192 = NO_OPTION`** for option-free plies.
   - Source token index $s_i \in [0, 202]$ and target token index $t_i \in [0, 202]$ from the resolution table below.
3. **Engine-to-Token Resolution (exhaustive table).** `Move::source_idx()` / `target_idx()` default to `Err` in `polyfish-rs/src/moves/mod.rs:67-75`; concrete moves return board tile indices — **except the five diplomacy moves**, which return `opponent_id` (player 1 or 2) as the source and, for `PeaceRequestResponse`, the accept flag as the target (`polyfish-rs/src/moves/abilities/diplomacy.rs:107, 234`). The v2.2 "raw tile index" rule would have aliased these to tiles 0–2. The extractor therefore matches on (`move_type`, `ability_type`) **exhaustively**: an unmatched variant, or a resolved tile outside `0..=120`, panics in self-play (engine rule: panics surface bugs). Unit and city lookups go through `tile_to_unit_slot: [Option<u8>; 121]` and `tile_to_city_slot: [Option<u8>; 121]`.

| Move type | Ability | $s_i$ | $t_i$ | $o$ |
| :--- | :--- | :--- | :--- | :--- |
| `Step` | — | unit slot of `src_index` | tile `target_index` | 192 |
| `Attack` | — | unit slot of `src_index` | unit slot of `target_index` (miss → tile, counted) | 192 |
| `Capture` | — | unit slot of `src_index` | tile `src_index` (the unit's own tile: its village / city / ruin / starfish planes) | 192 |
| `Summon` (train) | — | city slot of `src_index` | null | 48 + unit index |
| `Summon` (naval `UpgradeMove`, `upgrade.rs:29`) | — | unit slot of `src_index` (city lookup misses) | null | 48 + unit index |
| `Harvest` | — | null | tile `target_index` | 192 |
| `Build` | — | null | tile `target_index` | structure index |
| `Research` | — | null | null | 112 + tech index |
| `Reward` | — | null | city slot of `target_index` | 181 + reward index |
| `EndTurn` | — | null | null | 192 |
| `Ability` | Swarm (the boost skill: `BoostMove::ability_type()` returns `AbilityType::Swarm`, `boost.rs:64-65`, generated at `unit_actions.rs:73-82`; there is no `Boost` variant), Recover, Promote, Disband, Explode, HealOthers, FreezeArea | unit slot of `src_index` | tile `src_index` (own tile) | 160 + ability index |
| `Ability` | Decompose, Destroy, EnchantAnimal, GrowForest, BurnForest, ClearForest | null | tile `target_index` | 160 + ability index |
| `Ability` | BreakIce (reports the ice tile through `source_idx`, `break_ice.rs:88`) | null | tile from `source_idx()` | 160 + ability index |
| `Ability` | EstablishEmbassy, DestroyEmbassy | null | city slot of the opponent capital (`target_index`) | 160 + ability index |
| `Ability` | PeaceTreaty | null | city slot of own capital (`target_index`) | 160 + ability index |
| `Ability` | BreakPeace | null | null | 160 + ability index |
| `Ability` | PeaceRequestResponse, accept | null | null | 160 + ability index |
| `Ability` | PeaceRequestResponse, reject | null | null | **189** |
| `Ability` | Convert, Drain | *unreachable* | *unreachable* | — |

   Notes: `Summon` resolves `tile_to_city_slot[src]` first and falls back to `tile_to_unit_slot[src]` for naval upgrades. In 1v1 the diplomacy `opponent_id` carries no information, so dropping it is lossless. `ConvertMove` (`moves/abilities/convert.rs`) is never constructed by the generator and `AbilityType::Drain` has no move struct at all, so no candidate set contains either; their match arms are `unreachable!()`, which is what makes the `match` over the 21 `ABILITY_MAP` variants exhaustive: 19 generated + Convert + Drain. Capture and the unit-centered abilities point at the acting unit's own tile so the bilinear term sees unit × tile (what is being captured, healed around, or promoted on); the tile token is always present, so these never miss. Any entity lookup that misses (slot overflow) falls back to the raw tile token and increments `entity_slot_miss` in the METRICS line.
4. The move query representation comes from learned embedding tables:
   $$\mathbf{q}(m_i) = \mathbf{E}_{\text{action}}[a] + \mathbf{E}_{\text{option}}[o] \in \mathbb{R}^{d}$$
5. The move logit evaluates four contextual interactions, scaled by $1/\sqrt{d}$:
   $$\text{logit}(m_i) = \frac{\mathbf{q}(m_i)^T \mathbf{z}_G + \mathbf{q}(m_i)^T \mathbf{W}_S \mathbf{h}_{\text{source}}(m_i) + \mathbf{q}(m_i)^T \mathbf{W}_T \mathbf{h}_{\text{target}}(m_i) + \mathbf{h}_{\text{source}}(m_i)^T \mathbf{W}_{ST} \mathbf{h}_{\text{target}}(m_i)}{\sqrt{d}}$$
   with $\mathbf{h}_{\text{source}}(m_i) = \mathbf{X}_{\text{eval}}[s_i]$, $\mathbf{h}_{\text{target}}(m_i) = \mathbf{X}_{\text{eval}}[t_i]$, and $\mathbf{z}_G = g(\mathbf{h}_G) = \mathbf{W}_{G,2}\,\text{GELU}(\mathbf{W}_{G,1} \mathbf{h}_G + \mathbf{b}_1) + \mathbf{b}_2 \in \mathbb{R}^d$. The MLP $g$ (v3.0 used one linear map) matters for the option-free actions: for `Research` and `EndTurn` the global term is the *only* term, so a linear probe of $\mathbf{h}_G$ per (action, option) was the entire decision function of the most important stop decision in the game.
   **Initialization.** $\mathbf{W}_{G,\cdot}, \mathbf{W}_S, \mathbf{W}_T$ and the embedding tables use $\sigma = 0.02$; $\mathbf{W}_{ST}$ is **zero-initialized** (equivalently $\sigma \le 0.02/\sqrt{d} \approx 1.2 \times 10^{-3}$). v3.0 prescribed $\sigma_{ST} = 0.01$ "so four-term actions do not dominate early", which is backwards: after `norm_f` the token vectors are unit-scale while the query entries are $\approx 0.03$, so at $\sigma_{ST} = 0.01$ the bilinear term starts at $\approx 0.16$ logits against $\approx 0.009$ for each query term after the $1/\sqrt{d}$ scale, about 18× larger. Zero-init starts all four terms at comparable (near-zero) magnitude and lets `Step`/`Attack` earn the interaction term.
6. Softmax is evaluated over each state's own candidate segment (ragged layout, §7.1):
   $$P(m_i \mid s) = \frac{\exp(\text{logit}(m_i) / \tau)}{\sum_{j=1}^{K(s)} \exp(\text{logit}(m_j) / \tau)}$$

### 5.2 Vectorized Batched Evaluation (ragged)
Projections are evaluated once across the full sequence:
- $\mathbf{z}_G = g(\mathbf{h}_G) \in \mathbb{R}^{B \times d}$ (2-layer MLP, with biases)
- $\mathbf{Z}_S = \mathbf{X}_{\text{eval}} \mathbf{W}_S^T$, $\mathbf{Z}_T = \mathbf{X}_{\text{eval}} \mathbf{W}_T^T$, $\mathbf{Z}_{ST} = \mathbf{X}_{\text{eval}} \mathbf{W}_{ST}^T \in \mathbb{R}^{B \times 203 \times d}$ (`bias=False`)

For the flat candidate arrays of length $N = \sum_b K_b$ with `seg = repeat_interleave(arange(B), counts)`:
$$\text{logit}_k = \frac{\mathbf{q}_k \cdot \mathbf{z}_G[\text{seg}_k] + \mathbf{q}_k \cdot \mathbf{Z}_S[\text{seg}_k, s_k] + \mathbf{q}_k \cdot \mathbf{Z}_T[\text{seg}_k, t_k] + \mathbf{X}_{\text{eval}}[\text{seg}_k, s_k] \cdot \mathbf{Z}_{ST}[\text{seg}_k, t_k]}{\sqrt{d}}$$

```python
seg  = torch.repeat_interleave(torch.arange(B, device=dev), counts)      # [N]
m    = torch.full((B,), -inf, device=dev).scatter_reduce(0, seg, logits, "amax")
ex   = (logits - m[seg]).exp()
Z    = torch.zeros(B, device=dev).scatter_add(0, seg, ex)
logp = logits - m[seg] - Z[seg].log()                                    # per-candidate log-prob
```

Because $\mathbf{X}_{\text{eval}}[202] = \mathbf{0}$ and $\mathbf{W}_S, \mathbf{W}_T, \mathbf{W}_{ST}$ carry no bias, gathering index $202$ yields $\mathbf{Z}_S[202] = \mathbf{Z}_T[202] = \mathbf{Z}_{ST}[202] = \mathbf{0}$, zeroing the corresponding terms without branches.

### 5.3 Mathematical Behavior Across Action Classes
- **`Build` / `Harvest`** ([`moves/build.rs:105`](polyfish-rs/src/moves/build.rs#L105), [`moves/harvest.rs:74-76`](polyfish-rs/src/moves/harvest.rs#L74-L76)): $s_i = 202$, $t_i \in [0, 120]$.
  $$\text{logit}(m_i) = \mathbf{q}(m_i)^T \mathbf{z}_G + \mathbf{q}(m_i)^T \mathbf{W}_T \mathbf{h}_{\text{tile}}$$
  Scores the target tile directly (a road toward an enemy vs a dead end) without a dummy source. For `Harvest`, $o = 192$.
- **`Reward`** ([`moves/reward.rs:196-198`](polyfish-rs/src/moves/reward.rs#L196-L198)): $s_i = 202$, $t_i \in [185, 200]$. Conditions the reward choice (City Wall vs Workshop) on the specific city that leveled up. The candidate set here is exactly the pending reward moves (§5.1 step 1).
- **`Capture`** ([`moves/capture.rs:180-182`](polyfish-rs/src/moves/capture.rs#L180-L182)): $s_i \in [121, 184]$, $t_i \in [0, 120]$ = the unit's own tile, $o = 192$. All four terms fire: the acting unit, the tile it stands on (village, enemy city, ruin, starfish), and their interaction. v3.0 used a null target here, which hid what was being captured.
- **`Research`** ([`types.rs:711`](polyfish-rs/src/types.rs#L711)): $s_i = t_i = 202$.
  $$\text{logit}(m_i) = \mathbf{q}(\text{Research}, t)^T \mathbf{z}_G$$
  Evaluates tech value against stars, the researched DAG (25-bit mask), and turn budget through $\mathbf{z}_G = g(\mathbf{h}_G)$.
- **`EndTurn`** ([`types.rs:714`](polyfish-rs/src/types.rs#L714)): $s_i = t_i = 202$, $o = 192$. Decides whether remaining stars or unmoved units can still deliver advantage.
- **`Summon` / naval `Upgrade`** ([`types.rs:708`](polyfish-rs/src/types.rs#L708), [`moves/upgrade.rs:29, 74`](polyfish-rs/src/moves/upgrade.rs#L29)): $s_i$ is the city slot (train) or unit slot (upgrade), $t_i = 202$.
  $$\text{logit}(m_i) = \mathbf{q}(\text{Summon}, u)^T \mathbf{z}_G + \mathbf{q}(\text{Summon}, u)^T \mathbf{W}_S \mathbf{h}_{\text{source}}$$
- **`Ability`**:
  - *Unit-centered* (Swarm, Recover, Promote, Disband, Explode, HealOthers, FreezeArea): $s_i \in [121, 184]$, $t_i \in [0, 120]$ = the acting unit's own tile. `FreezeAreaMove` ([`freeze_area.rs:9`](polyfish-rs/src/moves/abilities/freeze_area.rs#L9)) and `HealOthersMove` ([`heal_others.rs:9`](polyfish-rs/src/moves/abilities/heal_others.rs#L9)) are AoE effects centered on the acting unit.
  - *Tile-targeted* (Decompose, Destroy, EnchantAnimal, GrowForest, BurnForest, ClearForest, BreakIce): $s_i = 202$, $t_i \in [0, 120]$. BreakIce reports its tile through `source_idx` ([`break_ice.rs:88-90`](polyfish-rs/src/moves/abilities/break_ice.rs#L88-L90)) and is normalized to a target by the table.
  - *Diplomacy* (EstablishEmbassy, DestroyEmbassy, PeaceTreaty): $s_i = 202$, $t_i$ = the capital's city slot; (BreakPeace, PeaceRequestResponse): $s_i = t_i = 202$, with accept and reject distinguished by $o$ (ability slot vs 189). These moves only appear once `Diplomacy` is researched (`diplomacy.rs:509-511`).
- **`Step` / `Attack`** ([`types.rs:705-706`](polyfish-rs/src/types.rs#L705-L706)): $o = 192$. `Step` targets a tile, `Attack` an enemy unit token. All four terms fire: campaign context ($\mathbf{z}_G$), acting unit ($\mathbf{h}_{\text{source}}$), destination or enemy affordance ($\mathbf{h}_{\text{target}}$), and the bilinear source–target combat interaction.

---

## 6. Critic Value Head & Own-Ply GAE

The Critic pools the normalized sequence $\mathbf{X}_{\text{norm}} = \mathbf{M} \odot \text{norm\_f}(\mathbf{X}^{(L)})$ (prior to null-token appending):
$$\mathbf{z}_{\text{val}} = \text{GELU}(\mathbf{W}_2 \, \text{GELU}(\mathbf{W}_1 [\mathbf{h}_G; \, \bar{\mathbf{h}}_M; \, \bar{\mathbf{h}}_U; \, \bar{\mathbf{h}}_C])), \qquad V(s) = \tanh(\mathbf{w}_{\text{win}}^T \mathbf{z}_{\text{val}}) \in [-1, +1]$$
where $\bar{\mathbf{h}}_M$ is the mean over the 121 map tokens (full-trunk spatial pooling, as EXP_ARCH_001 required for the value head), and $\bar{\mathbf{h}}_U$, $\bar{\mathbf{h}}_C$ are masked means over active unit and city slots ($\epsilon$-guarded). $V(s)$ is the acting player's win value. No secondary progress or potential targets (`FAILURES.md:37-40`, `DISCOVERIES.md` §2.6).

Value loss against GAE returns, over trainee plies only (below):
$$\mathcal{L}_{\text{value}}(\theta) = \frac{1}{|D|} \sum_{i} \left( V_{\theta}(s_{t_i}) - y_i \right)^2, \qquad y_i = \hat{A}_i + V_{\theta_{\text{old}}}(s_{t_i})$$
Unclipped by default; the PPO-style clip of $V_\theta$ to $\pm 0.2$ around $V_{\theta_{\text{old}}}$ is an ablation flag (v3.0 had it on; value clipping is not reliably helpful, Engstrom et al. 2020).

### Own-Ply Trajectories ($\gamma = 1$)
Polytopia alternates turns of 8–15 plies. v3.0 bootstrapped across the opponent's plies with a perspective flip $\sigma_t$; v3.1 instead defines the trainee's decision process over its **own plies**, with the opponent's entire turn folded into the environment transition (the standard construction for single-agent PPO in alternating-move games). Let $t_1 < t_2 < \dots < t_N$ be the plies at which the trainee acts (`is_trainee = 1`): consecutive within a turn, separated by the opponent's whole turn at turn boundaries. Then
$$\delta_i = r_i + V(s_{t_{i+1}}) - V(s_{t_i}), \qquad \hat{A}_i = \delta_i + \lambda \, \hat{A}_{i+1}, \qquad V(s_{t_{N+1}}) := 0, \; \hat{A}_N = \delta_N$$
with $\lambda = 0.95$ (sweep, §7.3).

**Rewards**: $r_i = 0$ for every non-terminal own ply. $r_N = +1$ if the trainee wins, $-1$ if it loses, $0$ on an exact tie, attached to the trainee's **last own ply** whether the game ended on that ply (the trainee took the last capital), during the opponent's turn (the trainee's capital fell), or by adjudication at the turn cap (`self_play.rs:931-957`; `decisive_frac` ≈ 0.90 at cap 45, so adjudications are a minority but not rare). The legacy adjudication breaks equal scores arbitrarily with `max_by_key` (`self_play.rs:950-956`); the entity recorder detects the tie itself and writes 0.

**Why own-ply rather than alternating** (D10):
1. No value is ever needed on an opponent ply. The v3.0 formulation required $V_{\theta}$ on every opponent ply, i.e. one extra trainee forward per opponent ply in *every* asymmetric game, not only the Greedy-anchor games it budgeted for (the pool opponent's own critic is a different network and cannot stand in).
2. The critic is trained only on states reached under the trainee's policy. v3.0's "value loss uses all plies" would have regressed $V$ toward Greedy's value function on 25% of games and toward stale checkpoints' on the rest.
3. No sign bookkeeping; `player_ids` identifies the seat and nothing more.

When both seats are the trainee (self-play against the current weights, allowed but not part of the v3.1 matchmaking), each seat is its own trajectory, so no data is lost relative to the alternating form. PPO surrogate, entropy and value loss are all restricted to `is_trainee = 1`; Phase 2 files store trainee plies only. The loader rebuilds each seat's chain from (`game_id`, `seat`) in recorded order, so no extra tensor is needed.

**Why $\gamma = 1$** (D8): games average **464 plies** at cap 45 (mean over the 64 cap-45 rows of run 1784424062, range 353–614; the single last row, which v3.0 quoted, reads 396). The v2.2 setting $\gamma = 0.99$ *per ply* scales the terminal outcome by $0.99^{464} \approx 0.009$ by the opening, while the BC phase trains the critic on undiscounted outcomes; PPO would then have pulled $V$ toward zero for everything before the last ~10 turns. AlphaZero and AlphaStar use undiscounted win/loss.

*Optional per-turn discount (ablation only, off by default)*: $\gamma_i = 0.995$ when own ply $t_{i+1}$ lies in a later turn than $t_i$, $\gamma_i = 1$ otherwise:
$$\delta_i = r_i + \gamma_i V(s_{t_{i+1}}) - V(s_{t_i}), \qquad \hat{A}_i = \delta_i + \gamma_i \lambda \hat{A}_{i+1}$$
so $0.995^{45} = 0.80$ of the outcome survives to turn 0.

*Ablation (the v3.0 alternating form, off by default)*: with $\sigma_t = +1$ if $\text{player}(s_{t+1}) = \text{player}(s_t)$ and $-1$ otherwise, $\delta_t = r_t + \sigma_t V(s_{t+1}) - V(s_t)$ and $\hat{A}_t = \delta_t + \lambda \sigma_t \hat{A}_{t+1}$ over the full alternating trajectory. It is a minimax-style assumption (the opponent is valued as if it played the trainee's policy) and pays the two costs above; run it only to measure whether cross-seat bootstrapping helps in pure self-play.

**Advantage normalization**: per minibatch (every ply in a Phase 2 minibatch is a trainee ply):
$$\hat{A}_i \leftarrow \frac{\hat{A}_i - \mu_{\hat{A}}}{\sigma_{\hat{A}} + 10^{-8}}$$

**Horizon caveat**: with $\gamma = 1$ and $\lambda = 0.95$ the estimator's effective horizon is ~20 own plies, so beyond that the advantage leans on the critic, and the critic's outcome fit was measured at $R^2 \approx 0.45$–$0.52$ (`DISCOVERIES.md` §2.6, the label-noise floor). Sweep $\lambda \in \{0.95, 0.98, 0.99\}$ in the first 30 PPO iterations (§7.3).

---

## 7. Training Protocol: Supervised Warmup + PPO League

```
  [Teacher Trajectories: Greedy + best MCTS checkpoints, recorded in the entity schema]
               │
               ▼
   Phase 1: Supervised BC (π'/one-hot teacher targets) + Critic Warmup ──► Gate G1: ≥ 50% vs Greedy (100 games)
                                                                                   │
                                                                                   ▼
   Phase 2: PPO League ◄───────────────────────────────────────────────────────────┘
         ├── 25% Greedy Heuristic Anchor (permanent floor)
         └── 75% Historical Checkpoint Pool (PFSP-weighted)          Gate G2 every 10 iterations
```

### 7.1 Data Schemas (`games_*.safetensors`, ragged candidate layout)

**(a) Observation tensors** (`uint8` everywhere except the continuous feature blocks):
- `spatial_maps` $[B, 145, 11, 11]$ `uint8`
- `unit_types`, `unit_passenger_types`, `unit_owners`, `unit_tiles`, `unit_mask` $[B, 64]$ `uint8`; `unit_features` $[B, 64, 10]$ `float32`
- `city_owners`, `city_tiles`, `city_mask` $[B, 16]$ `uint8`; `city_features` $[B, 16, 9]$ `float32`
- `global_context` $[B, 32]$ `float32`; `tribe_ids` $[B, 2]$ `uint8`

**(b) Candidate sets, ragged (CSR)**. Legal move counts run from 2 (a single pending reward; two per pending city) to a measured maximum of 244 (Appendix A.1), so no fixed cap is used:
- `cand_offsets` $[B+1]$ `int32` — candidates of state $b$ occupy `cand_offsets[b] .. cand_offsets[b+1]`
- `cand_actions`, `cand_options`, `cand_source_idx`, `cand_target_idx` $[N]$ `uint8` ($N = \sum_b K_b$; token indices in $[0, 202]$, options in $[0, 192]$, so every value fits `uint8`)
- `chosen_cand_idx` $[B]$ `int32` — **flat** index of the executed move, `cand_offsets[b] ≤ idx < cand_offsets[b+1]`
- `cand_target_probs` $[N]$ `float16` — the teacher's policy target over the segment (BC only). For Gumbel teachers this is the **improved policy** $\pi'(a) \propto \exp(\text{logit}(a) + \beta\,\sigma(\text{completed-}Q(a)))$ over the *full* legal set, which is what the agent already exports: `MoveVisit.visits` carries $\pi'$ mass, not a visit count (`gumbel_mcts.rs:1019-1060`). v3.0 called this "visit fractions"; the name was wrong and the data is better (dense, full support). `has_target_probs` $[B]$ `uint8` = 0 for Greedy plies (then the target is one-hot on `chosen_cand_idx`)

There is no truncation and therefore no `cand_mask`. The extractor still emits `cand_max_k` in the METRICS line so growth is visible.

**(c) RL trajectory metadata** (Phase 2 rollouts):
- `old_log_prob` $[B]$ `float32` — log-probability of the executed move under $\pi_{\theta_{\text{old}}}$ at the rollout temperature
- `values` $[B]$ `float32` — $V_{\theta_{\text{old}}}(s_t)$ from the acting player's perspective
- `rewards` $[B]$ `float32` — terminal $\pm 1 / 0$, else $0$
- `player_ids` $[B]$ `uint8` — acting player (1 or 2); identifies the seat (own-ply trajectories, §6, need no sign flips)
- `dones` $[B]$ `uint8` — 1 at game termination
- `is_trainee` $[B]$ `uint8` — 1 where the acting player is the PPO trainee. Phase 2 files store trainee plies only, so this is 1 throughout; it stays in the schema so both-seat self-play games can be added without a format change
- `game_id` $[B]$ `int32`, `seat` $[B]$ `uint8`, `turn` $[B]$ `uint8`

**Loader rules**: `spatial_maps.float() / 255.0`; every categorical or index tensor `.long()` before `nn.Embedding` / `gather`; `cand_target_probs.float()`.

**Storage per step and totals** (464 plies per game, the cap-45 mean of run 1784424062):

| Layout | Bytes / step | 3,000 games (1.39M steps) | 12,000 games (5.57M steps) |
| :--- | :--- | :--- | :--- |
| Unpacked `uint8` planes (as above) | 21,366 | 29.7 GB | 119.0 GB |
| Index-packed categorical planes | 9,992 | 13.9 GB | 55.6 GB |

Index packing stores the four mutually exclusive one-hot groups (terrain 8, resource 9, structure 35, unit type 46 = 98 planes) as four `uint8` index maps $[121]$ and keeps the remaining 47 planes as `uint8`; the loader re-expands them. D5 chooses **3,000 games, unpacked** for the first BC corpus. (v2.2's "12,000 games = 1,000,000 steps = 19 GB" combined three mutually inconsistent numbers; v3.0 sized from the last logged row, 396 plies, instead of the run mean.) The 64 unit slots and the passenger / unit-count fields add ≈ 1.5 KB per step over v3.0. At 12,000 games the unpacked corpus no longer fits the 50 GB host; index-pack first.

### 7.2 Phase 1: Joint Supervised Behavioral Cloning & Critic Warmup
- **Teacher corpus generation** (roadmap Step 2): `self_play --record-schema entity` on the *existing* MCTS / Greedy path. That path already exports, per legal move, the Gumbel improved policy $\pi'$ (`MoveVisit` in `polyfish-rs/src/ai/mcts_types.rs`, built at `gumbel_mcts.rs:1019-1060`, aggregated at `self_play.rs:734-784` for the legacy decomposed targets); the new recorder writes it onto the candidate segment as `cand_target_probs`. Two knobs shape $\pi'$ and must be pinned and logged in the METRICS line: $\beta$ = `policy_target_q_weight` ramps as $\min(1, \text{iteration}/20)$ from the `--iteration` argument (`self_play.rs:82-86`), so record with `--iteration ≥ 31` (which also selects cap 45); and `prior_heuristic_weight` (default 0.0, `gumbel_mcts.rs:214`) must stay 0 so the target is the checkpoint's search policy, not a heuristic blend. **Caveat**: at 128 sims with Gumbel top-k 16, ~200 of a late state's ~240 candidates are never visited; for them $\pi'$ reduces to the legacy network's composed prior, so BC partially distills the decomposed-head artifacts of §1.4. That is acceptable for a warm start (PPO is what corrects it) and is one reason G1 is a floor, not a target. Mix, seat-swapped, cap 45 turns (curriculum ≥ iteration 31):
  - 40% Greedy vs Greedy
  - 30% Greedy vs the best MCTS checkpoints at 128 iterations (iter156 of run 3, `model_checkpoint_iter156_20260716_135346`; iter186 of run 1784266164; if a pod reset wiped them, the strongest surviving checkpoint)
  - 30% checkpoint vs checkpoint at 128 iterations
  Existing `archive/games_*.safetensors` cannot be reused: they store only the decomposed marginal targets (`self_play.rs:2298-2306`).
- **Targets**: cross-entropy against `cand_target_probs` where `has_target_probs = 1` (the Gumbel improved policy $\pi'$, strictly more informative than the executed move), else one-hot on `chosen_cand_idx`. Critic target $z_t \in \{-1, 0, +1\}$ is the final outcome for the acting player.
- **Loss**:
  $$\mathcal{L}_{\text{Warmup}}(\theta) = \mathcal{L}_{\text{BC}}(\theta) + c_{\text{val}} \frac{1}{|D|} \sum_{t} \left( V_\theta(s_t) - z_t \right)^2, \qquad c_{\text{val}} = 0.5$$
- **Optimization**: AdamW, peak lr $1 \times 10^{-4}$, 500-step linear warmup, cosine decay to $1 \times 10^{-5}$, $\beta = (0.9, 0.95)$, weight decay 0.01, batch 1,024 plies, 15 epochs, bf16 autocast, gradient clip 1.0. Hold out 5% of games by `game_id`.
- **Gate G1** (replaces v2.2's unsupported "~800 Elo"): report held-out top-1 agreement with the teacher; require **≥ 50% vs Greedy over 100 symmetric arena games** with argmax play (`arena --backend1 policy --backend2 greedy --symmetric --games 100`; `policy` is a new `SearchBackendArg` variant, `--symmetric` exists at `arena.rs:86-88`). Expectation: parity when cloning Greedy (Greedy rates 511–641 across the per-campaign ledgers), above parity when cloning the 128-iteration checkpoints, which beat Greedy ~70% of the time.

### 7.3 Phase 2: PPO League Play
- **Rollouts**: 128–256 games per iteration (≈ 60k–120k plies at 464 per game), trainee seat alternating, sampling at $\tau = 1.0$. If $\tau \ne 1$ is ever used, `old_log_prob` must be the log-probability under the tempered distribution and $\rho_t$ must use the same $\tau$.
- **Matchmaking** (D6):
  1. **Greedy Heuristic Anchor, 25% of matches, permanent.** No fade-out: fading to 5% caused the 81% → 25% crash; the 20% floor fixed it ([`DISCOVERIES.md:54-60`](DISCOVERIES.md#L54-L60)).
  2. **Historical Checkpoint Pool, 75% of matches.** Snapshot every 5 iterations; sample by prioritized fictitious self-play, $p_i \propto (1 - w_i) + 0.05$, where $w_i$ is the trainee's win rate against checkpoint $i$ over its last 50 games ($0.5$ if unplayed). At most **4 distinct checkpoints** are drawn per iteration so that each opponent's inference server (§8 step 4) sees full batches.
  3. **No exploiters in v3.1.** They need a second training stream; revisit in v3.2 only if G2 stalls.
- **Loss**:
  $$\mathcal{L}_{\text{PPO}}(\theta) = -\hat{\mathbb{E}}_t \left[ \min\left( \rho_t \hat{A}_t, \, \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right] + c_{\text{val}} \mathcal{L}_{\text{value}} - c_{\text{ent}} \mathcal{S}[\pi_\theta] + \beta_{\text{BC}} \, \text{KL}(\pi_\theta \,\|\, \pi_{\text{BC}})$$
  with $\rho_t = P_\theta(m_t \mid s_t) / P_{\theta_{\text{old}}}(m_t \mid s_t)$, $\epsilon = 0.20$, $c_{\text{val}} = 0.5$ (unclipped; the $\pm 0.2$ clip around $V_{\theta_{\text{old}}}$ is an ablation flag, §6), $c_{\text{ent}} = 0.01$, and $\beta_{\text{BC}} = 0.05$ for the first 100 iterations, then decayed linearly to $0$ over iterations 100–200 (v3.0 dropped it to zero at iteration 100, a cliff) — an AlphaStar-style anchor to the frozen BC policy, insurance against the passivity collapse of [`notes.md:174-187`](notes.md#L174-L187). Entropy is computed over each candidate segment.
- **Optimization**: 3 epochs per rollout, minibatch 1,024 plies, AdamW lr $3 \times 10^{-5}$ (500-step warmup at PPO start), $\beta = (0.9, 0.95)$, weight decay 0.01, gradient clip 1.0, early-stop the epoch when mean $\text{KL}(\pi_{\text{old}} \| \pi_\theta) > 0.02$, bf16 autocast. $\lambda$: start at 0.95 and run a $\{0.95, 0.98, 0.99\}$ sweep over the first 30 iterations (§6, horizon caveat).
- **Gate G2**: every 10 iterations, 100 symmetric games vs Greedy and 100 vs the checkpoint from 10 iterations earlier, logged to `elo_ratings.json` / `matches.jsonl`. Continue while the 10-iteration head-to-head stays ≥ 55%; promote S → M (§4.3) if it stalls for 30 iterations while still beating Greedy.

### 7.4 Search at Evaluation Time (kept as an ablation)
The network's interface, candidate logits over the legal set plus a value, maps directly onto what the existing Gumbel MCTS (`gumbel_mcts.rs`) needs as root prior and leaf evaluator; only a thin adapter from candidate logits to child priors is required. Training stays search-free (D1), but nothing in the architecture forecloses search, and a shallow search at evaluation time is both free Elo and the direct test of D1's premise. Add `arena --backend1 policy-gumbel --mcts 16|32|64` once Step 4 lands and report it next to the argmax number at every G2 gate. If 32 sims add ≥ 100 Elo over argmax, the reactive policy is leaving strength on the table that a later version may want to recover during training as well.

---

## 8. Implementation Roadmap (single-definition stack)

**Contract (D3).** The network is defined **once**, in `polyfish-rs/train_polystar.py`. Training checkpoints stay `model.safetensors` (PyTorch state dict). Every checkpoint additionally exports `policy.pt` via `torch.jit.script` with a fixed signature:
- inputs: `spatial_maps` u8 $[B,145,11,11]$; `unit_types`, `unit_passenger_types`, `unit_owners`, `unit_tiles`, `unit_mask` u8 $[B,64]$; `unit_features` f32 $[B,64,10]$; `city_owners`, `city_tiles`, `city_mask` u8 $[B,16]$; `city_features` f32 $[B,16,9]$; `global_context` f32 $[B,32]$; `tribe_ids` u8 $[B,2]$; `cand_offsets` i32 $[B+1]$; `cand_actions`, `cand_options`, `cand_source_idx`, `cand_target_idx` u8 $[N]$
- outputs: `cand_logits` f32 $[N]$, `values` f32 $[B]$

Rust loads it with `tch::CModule::load_on_device` (`src/wrappers/jit.rs:445` at the pinned tch-rs revision `bbf49e6`) and calls **`forward_is`** (`jit.rs:490`), unpacking the returned `IValue::Tuple` into the two tensors. v3.0 named `forward_ts` (`jit.rs:481`), which returns a single `Tensor` and cannot carry two outputs; the alternative is to return one concatenated tensor and split it on the Rust side. The CLAUDE.md "dual-network sync constraint" applies to `PolyZeroNet` only; when Steps 3–4 land, add a note there that PolyStar has one definition and derived artifacts.

**Build.** `cargo build --release --features tch-eval` with `LIBTORCH_USE_PYTORCH=1` against the CUDA pip torch in `.venv` (the mechanism documented in `Cargo.toml`). On this branch `vast_setup.sh:35-37` already exports `LIBTORCH_USE_PYTORCH=1`, `LIBTORCH_BYPASS_VERSION_CHECK=1` and a Linux `LD_LIBRARY_PATH` to `<site-packages>/torch/lib`, while `run_training_loop.sh:44-48` sets only the macOS `DYLD_LIBRARY_PATH`. The RunPod scripts (`runpod_setup.sh`, `run_training_runpod.sh`) exist only inside the unapplied `runpod-training.patch`; whichever loop script runs Phase 2 must export `LD_LIBRARY_PATH` the way `vast_setup.sh` does.

**Steps** (each gated by the previous):
1. **`polyfish-rs/src/ai/features.rs`** — `state_to_entity_features`: 145 planes with the owner split; unit/city/global tensors (§3); `tile_to_unit_slot` / `tile_to_city_slot`; the passenger-plane fix and the $[0,1]$ range assertion (§3.1); the exhaustive candidate encoder of §5.1 with panics on unmatched variants; `entity_slot_miss`, `cand_max_k` and `feat_clip` METRICS counters.
2. **`polyfish-rs/src/bin/self_play.rs`** — `--record-schema entity` on the existing MCTS / Greedy path: writes §7.1(a) + (b) with `cand_target_probs` and outcome values, and pins and logs $\beta$ and the heuristic prior weight (§7.2). No new network is required; this produces the BC corpus.
3. **`polyfish-rs/train_polystar.py`** — model (S/M configs), BC trainer (§7.2), `export_policy()` → `policy.pt`, and a fixture dump (one recorded batch plus its outputs) for the Rust parity test.
4. **`polyfish-rs/src/ai/tch_policy.rs`** (feature `tch-eval`) — CModule loader (`forward_is`); a `PolicyEvalServer` that generalizes the request coalescing of `eval_server.rs` over request/response types (today it is typed to `PolyZeroNet`, `RawFeatures` and `RawPolicyOutput`, `eval_server.rs:27-45`), with **one loaded module per distinct network per iteration** (trainee + ≤ 4 pool checkpoints, §7.3) and coalescing per model; `self_play --backend policy` (rollouts + §7.1(c) metadata, trainee plies only); `arena --backend1/--backend2 policy`; parity test against the fixture (max $|\Delta| \le 10^{-4}$ in fp32).
5. **`train_polystar.py` PPO loop + `run_polystar_loop.sh`** — GAE (§6), league matchmaker (§7.3), Elo ledger integration.
6. **Ablations**: shallow search at eval (§7.4) can run as soon as Step 4 lands; after G2 passes: size M (§4.3), MoE experts (§4.4), per-turn discount and the alternating GAE (§6), value clipping (§6), auxiliary BC heads (§1.3).

**Gate G0, before Step 3**: measure the two throughput ceilings (Appendix A.3) and record them in §2.

---

## 9. Source Catalog & Evidence Concordance

| Specification / Empirical Grounding | Source Location | Evidence Context |
| :--- | :--- | :--- |
| **Legacy ResNet-6 Trunk** | [`network.rs:145-187`](polyfish-rs/src/ai/network.rs#L145-L187), `train.py` | 6 blocks, 64 channels, 583,441 parameters (recomputed) |
| **Intra-Turn MCTS Depth Trap** | [`BOTTLENECK.md:21-42`](BOTTLENECK.md#L21-L42), `training_results/moves_by_turn.json` | 2–3 plies of depth vs ≈8 plies per turn |
| **More search helps but is unaffordable at legacy throughput** | [`hypothesis_driven_improvements.md:43-53`](hypothesis_driven_improvements.md#L43-L53) | EXP 3: 64 → 256 sims, capture rate 0.81 → 1.00, 7.9 → 7.3 turns, 2.3× wall-clock; rejected on cost |
| **Planar Flattening & Tech DAG** | [`BOTTLENECK.md:47-58`](BOTTLENECK.md#L47-L58) | 2D convs cannot model DAG prerequisites |
| **Calibration vs Discrimination (EXP_ELO_069)** | [`FAILURES.md:46-50`](FAILURES.md#L46-L50), `DISCOVERIES.md` §2.7 | Neural value leaf −13.1 pp vs heuristic leaf (44.0 → 30.9) at $R^2 = 0.972$; macro-head negative transfer reported qualitatively only |
| **Persistent Unit Identifiers** | [`states.rs:257`](polyfish-rs/src/states.rs#L257) | `pub id: u32`, stable across plies |
| **UnitType Variants (46 incl. `None`)** | [`types.rs:268-333`](polyfish-rs/src/types.rs#L268-L333), [`features.rs:162-165`](polyfish-rs/src/ai/features.rs#L162-L165) | `UNIT_INDEX` keeps `None` at 0 (padding type); mapper `UNIT_MAP` drops it |
| **City State Schema** | [`states.rs:291-311`](polyfish-rs/src/states.rs#L291-L311), `features.rs:596-640` | Walls / riot / capital are derived, not fields |
| **Neutral Villages on Structures** | [`scoring.rs:1222`](polyfish-rs/src/ai/scoring.rs#L1222) | `StructureType::Village` without a `CityState` |
| **Unit count bound** | `moves/summon.rs:130-133` | Training blocked when city unit count > level, i.e. up to level + 1 units per city (drives D9 and the city unit-count feature) |
| **Fog Memory Spatial Channels** | [`features.rs:116-125`](polyfish-rs/src/ai/features.rs#L116-L125) | 6 decayed observation-memory channels |
| **Total Spatial Channels (142 legacy)** | [`features.rs:128`](polyfish-rs/src/ai/features.rs#L128) | 8+10+9+35+46+16+12+6; owner planes the only signed values; `CH_UNIT_MAX_HP` constant 0 (`features.rs:498`); passenger plane exceeds 1 (`features.rs:562`, discriminants to 62) |
| **Move / option index layout** | [`mapper.rs:38-45, 82-97`](polyfish-rs/src/ai/mapper.rs#L38-L45) | 12 actions; options 0..192 with 189–190 free |
| **Move index defaults** | `moves/mod.rs:67-75` | `source_idx` / `target_idx` default to `Err` (v2.2 cited mapper.rs) |
| **Diplomacy index aliasing** | `moves/abilities/diplomacy.rs:107, 234, 509-511` | Source = `opponent_id`; response target = accept flag; gated on `Diplomacy` tech |
| **Reward-only candidate sets** | `moves/mod.rs:202-205` | Generator returns early when a reward is pending |
| **Target-Only Build & Harvest** | [`moves/build.rs:105`](polyfish-rs/src/moves/build.rs#L105), [`moves/harvest.rs:74-76`](polyfish-rs/src/moves/harvest.rs#L74-L76) | Spatial target, no source entity |
| **Source-Only Capture** | [`moves/capture.rs:180-182`](polyfish-rs/src/moves/capture.rs#L180-L182) | Source index only |
| **Naval upgrade reuses Summon** | [`moves/upgrade.rs:29`](polyfish-rs/src/moves/upgrade.rs#L29) | Unit-sourced `MoveType::Summon` |
| **Policy Multiplication Defect** | [`policy_composer.rs:69-105`](polyfish-rs/src/ai/policy_composer.rs#L69-L105) | Independent marginal products |
| **Gumbel $\pi'$ targets available** | `gumbel_mcts.rs:1019-1060`, `self_play.rs:82-86, 734-784`, `ai/mcts_types.rs` | `MoveVisit.visits` = $\pi'$ mass over the full legal set; $\beta$ ramps with `--iteration` → `cand_target_probs` |
| **Legacy game files lack candidate sets** | `self_play.rs:2298-2306` | Only decomposed marginal targets are stored |
| **Tabula Rasa Passivity Collapse** | [`FAILURES.md:28-31`](FAILURES.md#L28-L31), [`notes.md:174-187`](notes.md#L174-L187) | Captures 6.5 → 3.2 while policy loss fell |
| **Curriculum Anchor Fadeout Defect** | [`FAILURES.md:33-35`](FAILURES.md#L33-L35), [`DISCOVERIES.md:54-60`](DISCOVERIES.md#L54-L60) | 81% → 25% at 5% anchor; 20% floor fixed it |
| **Outcome label noise floor** | `DISCOVERIES.md` §2.6 | ~50% noise; $R^2$ capped ≈ 0.45–0.52; shaping rejected |
| **Self-Play Throughput Baseline** | [`expert_boost_throughput.md:36`](expert_boost_throughput.md#L36) | ~578 moves/s, CPU↔GPU sync bound, GPU 2.5% busy |
| **TorchScript loading in Rust** | tch-rs `bbf49e6`, `src/wrappers/jit.rs:445, 481, 490` | `CModule::load_on_device`; `forward_is` for the two-tensor tuple (`forward_ts` returns one `Tensor`) |
| **Boost skill is `AbilityType::Swarm`** | `moves/abilities/boost.rs:64-65`, `moves/abilities/unit_actions.rs:73-82` | No `Boost` variant exists; `Drain` has no move struct |
| **Reward moves per pending city** | `moves/reward.rs:206-235` | Two moves per city with an outstanding reward; several cities can be pending at once |
| **Plies per game at cap 45** | `training_results/training_log.csv`, run 1784424062 | Mean 464 over 64 rows (353–614); the last row alone reads 396 |
| **Eval server is typed to the legacy net** | `ai/eval_server.rs:27-45` | `PolyZeroNet`, `RawFeatures`, `RawPolicyOutput`; generalize for the policy server |
| **Linux libtorch path already scripted** | `vast_setup.sh:35-37`, `run_training_loop.sh:44-48` | `LD_LIBRARY_PATH` on Linux, `DYLD_LIBRARY_PATH` on macOS; RunPod scripts live only in `runpod-training.patch` |
| **Measured branching factor** | Appendix A.1 | Max 244 legal moves; 174 by turn 28 |

---

## Appendix A — Measurements

### A.1 Legal-move counts (random agent)
`cargo run --bin stats -- --games 100 --max-turns 45` (random agent, Perfection mode, Luxidoor vs Imperius, 2026-09-10; the tool's table ends at turn 30):

| Turn | Avg legal moves | Max |
| :--- | :--- | :--- |
| 0 | 5.5 | 15 |
| 5 | 11.6 | 43 |
| 10 | 16.5 | 64 |
| 15 | 21.3 | 86 |
| 20 | 24.9 | 99 |
| 25 | 27.2 | 126 |
| 28 | 31.0 | 174 |
| 30 | 32.1 | 148 |
| **all steps** | **26.7** | **244** |

Random play is not trained play: trained late games hold more cities and units, so treat these as a floor on the late-game maximum. The July 2026 1,000-game table in `notes.md` reached 123 by turn 19. Both rule out the v2.2 cap of 128.

### A.2 Compute per forward (analytic, sequence length 202)

| Network | GFLOP / forward | Forwards / move | GFLOP / move |
| :--- | :--- | :--- | :--- |
| PolyZeroNet (6 × 64-ch ResBlocks) + 64 Gumbel sims | 0.13 | 64 | 8.1 |
| PolyStar S | 2.88 | 1 | 2.9 |
| PolyStar M | 9.35 | 1 | 9.3 |
| MoE v2.2 (reference, seq 170) | 6.16 | 1 | 6.2 |

Linear-layer FLOPs $= 2 \times 202 \times (\text{attention} + \text{FFN parameters})$; attention scores $= 4 \times 202^2 \times d \times L$. Going from 32 to 64 unit slots (170 → 202 tokens) costs +19% on the linear terms and +41% on the score terms, +21% overall.

### A.3 Throughput measurement plan (Gate G0)
1. **Actor ceiling** (no network): `cargo run --release --bin actor_ceiling -- --mcts-iters 1 --num-games 64 --actors 16`. The dummy evaluator with one root evaluation per ply approximates the reactive-policy actor path (game sim + entity feature encode).
2. **GPU ceiling**: PyTorch micro-benchmark of `policy.pt` at batch 512 in bf16 on the rented GPU; forwards/s × 1 move per forward.
3. Spec throughput (§2) = min of the two, minus the pipeline overhead measured once `self_play --backend policy` exists.

---

## Appendix B — Change Log v2.2 → v3.0 (2026-09-10)

1. §6: discount $\gamma = 0.99$ per ply → $\gamma = 1.0$ (per-turn 0.995 as an off-by-default ablation). $0.99^{396} = 0.019$.
2. §5.1: exhaustive (move type, ability) resolution table with panics; diplomacy moves mapped to null / capital-city tokens; `OPT_PEACE_REJECT = 189`; missing abilities added; `ConvertMove` removed (never generated).
3. §7.1: candidate cap $K_{\max} = 128$ → ragged CSR layout, no cap; measured max 244 (Appendix A.1). Prefix truncation would have dropped econ, econ-ability and diplomacy moves first because the generator appends them last.
4. §7.2: dataset arithmetic corrected (396 plies/game; 3,000 games ≈ 1.19M steps ≈ 24 GB unpacked); index-packing option documented.
5. §2, Appendix A.2: "64× FLOP dividend" replaced with the per-move cost table; throughput becomes a measured gate (G0).
6. §0 D2, §4.3: single 37M design → S/M size ladder with a promotion rule; explicit decision record against the July "modest steps" rule.
7. §8: Candle mirror → single PyTorch definition + TorchScript loaded by `tch`; build and RunPod path notes; CLAUDE.md follow-up.
8. §1.3, §4.4, §9: EXP_ELO_069 re-attributed (value leaf vs heuristic leaf); "strict gradient isolation" withdrawn.
9. §4, §3.5: MoE experts → dense FFN + modality-type embeddings; experts kept as a sized ablation.
10. §3.4: global token gains turns-remaining, `max_turns`, and tribe embeddings; scalars clamped.
11. §3.1: owner planes split into self/enemy binaries (145 channels) in the new extractor only.
12. §7.2, §8 Step 2: teacher-corpus recorder added to the roadmap; BC targets use MCTS visit distributions.
13. §7.2: "~800 Elo after BC" → Gate G1 (≥ 50% vs Greedy over 100 symmetric games).
14. §7.3: full PPO hyperparameter block; KL-to-BC anchor; `is_trainee`, `game_id`, `seat`, `turn` added to the schema; rollout $\tau$ consistency rule.
15. §7.3: exploiters removed; PFSP-weighted pool.
16. §3.2, §3.3, §5.1, §9: citation fixes (`moves/mod.rs:67-75`), `UNIT_INDEX` `None` padding, reward-only candidate sets, city normalization aligned with `features.rs`.
17. Outside this spec: `polyfish-rs/src/dotnet_rng.rs` used checked subtraction where the .NET algorithm wraps; debug builds panicked on some seeds (release builds were unaffected). Fixed with `wrapping_sub`.

---

## Appendix C — Change Log v3.0 → v3.1 (2026-09-10 code-level review)

Verified against the code and left unchanged: the S/M parameter arithmetic and FLOP formulas (recomputed below for the new shapes), the 142-channel layout, the option offsets, every source/target convention in the §5.1 table, the 25 vanilla techs, the 21 `ABILITY_MAP` entries, the 8 reward types, `TribeType::None = 0` as the unknown-opponent row, the tch-rs line numbers, the curriculum, the reward-first generator, the "legacy files lack candidate sets" claim, and the legacy count of 583,441 (65 of which are the `v_ownership` aux head).

1. §8: `forward_ts` → `forward_is`. `forward_ts` returns one `Tensor`; the module returns two.
2. §3.1: the legacy passenger-type plane divides the raw `UnitType` discriminant (max 62) by 46 and reaches 1.35; the new extractor uses `UNIT_INDEX`, asserts the $[0,1]$ range and counts clips (`feat_clip`). `CH_UNIT_MAX_HP` documented as constant zero. §3.2: `unit_passenger_types` + $\text{Embed}_{\text{pass}}$ added to the unit token.
3. §5.1, §5.3: "Boost" → `Swarm` (`BoostMove::ability_type()`); explicit unreachable arms for `Convert` and `Drain`; the 21-variant reconciliation (19 generated + 2).
4. D5, D8, §6, §7.1, §7.3: plies per game 396 → 464 (run mean, not the last row); $0.99^{464} \approx 0.009$; corpus 1.39M steps ≈ 29.7 GB unpacked; rollouts 60k–120k plies.
5. §7.1, §7.2, §9: `cand_visit_probs` → `cand_target_probs`; the target is the Gumbel improved policy $\pi'$, $\beta$ ramps with `--iteration`, heuristic prior weight pinned to 0; the unvisited-move caveat recorded.
6. §5.1, §7.1: a pending-reward candidate set is two moves *per pending city*, not always two.
7. D1, §9: EXP 3 re-read: search raised capture rate 0.81 → 1.00 and was rejected on cost; D1 rests on latency structure, not on search being useless. §7.4 keeps shallow search at eval time as an ablation.
8. §5.1: $\mathbf{W}_{ST}$ zero-initialized; the v3.0 $\sigma_{ST} = 0.01$ argument was inverted (the bilinear term started ≈ 18× the query terms).
9. §8, §7.2, §7.3: `vast_setup.sh` already handles the Linux library path; RunPod scripts live only in `runpod-training.patch`; arena flags are `--backend1/--backend2`.
10. D9, §2, §3.2, §4.1, §4.3, §5, A.2: unit slots 32 → 64 (sequence 202 + null = 203; slices `121..185`, `185..201`, `201..202`; null index 202); enemy units that are legal Attack targets are prioritized into slots.
11. D10, §6, §7.1(c): alternating-perspective GAE → own-ply trajectories (the opponent's turn folded into the transition); value / PPO / entropy on trainee plies only; no critic on opponent plies; the alternating form kept as an ablation.
12. D11, §5.1, §5.3: Capture and unit-centered abilities point at the acting unit's own tile; option-free actions score through a 2-layer MLP $g(\mathbf{h}_G)$.
13. §3.3: city feature $\min(\text{units}_c/(\text{level}+1), 1)$ (the summon cap); 9 city features.
14. §4.1: attention via `F.scaled_dot_product_attention` with an explicit mask (the `nn.MultiheadAttention` fast path changes numerics under padding masks).
15. §6, §7.3: value clipping off by default (ablation flag); $\beta_{\text{BC}}$ decays over iterations 100–200 instead of a cliff; $\lambda$ sweep {0.95, 0.98, 0.99}; explicit tie handling in the recorder.
16. §1.3, §8 step 6: auxiliary BC heads permitted as an ablation (potential shaping remains rejected).
17. §7.3, §8 step 4: ≤ 4 distinct pool checkpoints per iteration; one loaded module per network; the `EvalServer` generalization named as work.
18. §2, §4.3, A.2: parameters S 7,132,929 → 7,210,753 (28.8 MB); M 23,007,361 → 23,173,249 (92.7 MB). GFLOP/forward at sequence 202: S 2.88, M 9.35.
19. Links made repo-relative; file renamed `POLYSTAR-MOE-ARCH.md` → `POLYSTAR-V3-ARCH.md`.

**Not applied**: a "plies elapsed this turn" scalar for the global token. `GameState` carries no per-turn ply counter, so it would have to be threaded through every extractor call site (server, arena, recorder); the unit `moved` / `attacked` flags and the star count already carry most of that signal. Revisit if EndTurn timing is a visible failure mode after BC.
