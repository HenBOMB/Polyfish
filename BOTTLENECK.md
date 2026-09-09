# Polyfish Architecture Bottlenecks & Failure Analysis

This document provides a comprehensive post-mortem and scientific analysis of the architectural bottlenecks, search horizon limits, neural representation failures, and design traps that stalled the AlphaZero/MCTS approach in Polyfish.

---

## 1. Executive Summary: Why AlphaZero Stalled on Polytopia

AlphaZero and its derivatives conquered Chess, Shogi, and Go by coupling deep Monte Carlo Tree Search (MCTS) with a single residual convolutional neural network (ResNet). That formulation succeeds because those board games share key structural invariants:
1. **Alternating, Single-Action Plies**: Exactly one discrete move is made per ply ($A \to B \to A \to B$).
2. **Full Observability**: Zero hidden information, zero fog of war, zero stochastic generation.
3. **Fixed Homogeneous Spatial Domain**: An $8 \times 8$ or $19 \times 19$ grid where piece movements represent all game mechanics.
4. **Immediate Tactical Consequence**: Piece captures and board state changes translate directly to value shifts within 2–6 plies.

Polytopia violates every single one of these invariants. Attempting to force Polytopia into an AlphaZero ResNet-6 + Gumbel MCTS pipeline resulted in an empirical ceiling where self-play models struggled to reliably outperform basic greedy heuristics, collapsed into passive mutual non-aggression ([notes.md:174-187](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/notes.md#L174-L187)), or fell victim to value calibration breakdown vs leaf discrimination ([DISCOVERIES.md:68-73](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/DISCOVERIES.md#L68-L73), EXP_ELO_069 on branch `origin/exp-elo-005-raise-relative-weight-in-value-targets` commit `27055090`).

---

## 2. The Turn Structure & Combinatorial Search Horizon Bottleneck

### 2.1 The Intra-Turn Depth Trap
In Chess or Go, search depth directly measures interaction with the opponent: depth 6 means 3 moves by player A and 3 moves by player B. 

In Polytopia, a single turn consists of an ordered sequence of 8–15 atomic moves ending in `MoveType::EndTurn` ([polyfish-rs/src/types.rs:714](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/types.rs#L714)). The engine enum defines 12 move variants ([polyfish-rs/src/types.rs:702-716](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/types.rs#L702-L716)) spanning 11 playable action types (`Step`, `Attack`, `Ability`, `Summon`, `Harvest`, `Build`, `Research`, `Capture`, `Reward`, `EndTurn`, and `Resign`, plus fallback `None`), while the neural policy head (`pi_action`) maps 11 action logits ([polyfish-rs/src/ai/network.rs:178](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/network.rs#L178)) covering actions 0–10.

```
Turn T (Player 1)                                                     Turn T+1 (Player 2)
[Research] -> [Summon] -> [Step U1] -> [Attack U1] -> ... -> [EndTurn] -> [Step E1] -> ...
|----------------------- 8 to 15 plies ----------------------|
```

Because MCTS traverses atomic moves as tree edges, **an entire 8-ply MCTS search path does not even reach the end of the current player's first turn**.

### 2.2 Branching Factor Explosion
At any single ply within a turn, the player may have 20–60 legal moves (moving multiple units, researching multiple techs, upgrading multiple cities). 
- If a turn requires 10 atomic decisions with an average legal action branching factor $b \approx 30$:
  $$\text{Combinatorial turn action space} \approx 30^{10} \approx 5.9 \times 10^{14}$$
- Gumbel MCTS operates with a budget of 32–64 iterations ([polyfish-rs/src/ai/gumbel_mcts.rs:49](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/gumbel_mcts.rs#L49)) and samples top-$k$ candidates (typically $k=4$ or $k=8$).
- As proven in EXP 3 ([hypothesis_driven_improvements.md:43-53](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/hypothesis_driven_improvements.md#L43-L53)), quadrupling search budget from 64 to 256 simulations failed to accelerate first-village capture times (7.9 turns $\to$ 7.3 turns) while reducing throughput by 2.3×. 

MCTS exhausts its computational budget permuting intra-turn move orders (e.g. moving unit A before unit B) while remaining **completely blind to opponent reactions and multi-turn strategic compounding**.

---

## 3. The Single-Pass ResNet Representation Bottleneck

### 3.1 Heterogeneous State Flattening
The Polyfish network `PolyZeroNet` ([polyfish-rs/src/ai/network.rs:145-170](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/network.rs#L145-L170)) relies on 2D convolutions over spatial feature planes ([polyfish-rs/src/ai/features.rs:42-65](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/features.rs#L42-L65)):
- 8 terrain channels ([polyfish-rs/src/ai/features.rs:26, 42-43](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/features.rs#L26))
- 10 tile flag channels ([polyfish-rs/src/ai/features.rs:46-60](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/features.rs#L46-L60))
- Resource and structure channels
- A 16-dimensional scalar player vector injected via cross-attention ([polyfish-rs/src/ai/features.rs:216](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/features.rs#L216), [polyfish-rs/src/ai/network.rs:177, 191-198](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/network.rs#L177))

This planar spatial encoding creates severe representational friction:
1. **The Technology Tree is a Directed Acyclic Graph (DAG)**: Techs have prerequisite chains ([polyfish-rs/src/settings/technology.rs:466-479](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/settings/technology.rs#L466-L479)). Forcing tech states into a single scalar feature (`tech_norm`) destroys graph topology.
2. **Global Economy is Non-Spatial**: Star income, city level caps, and tech purchase inflation depend on global game counters, not localized $3 \times 3$ convolutional receptive fields.
3. **Discrete Unit Entities are Blended**: Stacking unit types, health values, and veteran statuses into 2D grid cells prevents the model from tracking specific unit identities across plies.

### 3.2 Model Capacity Deficit
`PolyZeroNet` uses a 6-block ResNet with 64 filters ([polyfish-rs/src/ai/network.rs:174-175](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/network.rs#L174-L175)):
- `PolyZeroNet` contains ~583,000 total parameters (443k in the 6 residual blocks alone).
- In Chess, a network of this scale needs only to evaluate static board piece relations.
- In Polytopia, this tiny capacity was tasked with simultaneously modeling:
  - Fog of War frontier exploration
  - Diagonal Chebyshev pathfinding geometry ([hypothesis_driven_improvements.md:65-74](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/hypothesis_driven_improvements.md#L65-L74))
  - Combat damage formulas and retaliation sequences ([polyfish-rs/src/actions/units.rs:1324-1365](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/actions/units.rs#L1324-L1365), [polyfish-rs/src/states.rs:578-585](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/states.rs#L578-L585))
  - Multi-tier tech prerequisite planning ([polyfish-rs/src/settings/technology.rs:460-492](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/settings/technology.rs#L460-L492))
  - Long-term city population leveling economics

A single forward pass through a 6-block convolution cannot perform the multi-scale reasoning required to balance macro strategy against micro tactics.

---

## 4. Multi-Head Gradient Interference & Leaf Discrimination (EXP_ELO_069)

In an attempt to address macro planning, Verdi's research branch (`origin/exp-elo-005-raise-relative-weight-in-value-targets`) experimented with auxiliary macro heads (`pi_macro_stance` and `pi_macro_order`) alongside micro actions on the shared ResNet trunk:
- `macro_stance_probs` (Grow, Arm, Defend, Save)
- `macro_order_maps` (spatial target distributions)

### The Negative Transfer & Discrimination Trap
When auxiliary macro objectives and calibrated value targets were trained:
1. **Gradient Conflict**: Backpropagating disparate macro objectives through the compact 64-channel convolutional trunk ([polyfish-rs/src/ai/network.rs:174-175](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/network.rs#L174-L175)) conflicted with spatial representations predicting micro move coordinates ([polyfish-rs/src/ai/network.rs:162-163](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/network.rs#L162-L163)).
2. **Discrimination Degradation**: In EXP_ELO_069 ([DISCOVERIES.md:68-73](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/DISCOVERIES.md#L68-L73), commit `27055090` on branch `origin/exp-elo-005-raise-relative-weight-in-value-targets`), evaluating the neural value leaf (`MacroLeaf::NetAsym`) in the arena resulted in a **13.1 percentage point win rate drop** (44.0% down to 30.9%) compared to the heuristic leaf, even while macro score calibration reached $R^2 = 0.972$ on late turns in EXP_ELO_067.
3. The network learned smooth global correlations (who is ahead on points) but lost the fine-grained discrimination required to pick the winning tactical move among candidate children at an MCTS node.

---

## 5. The Handcrafted Reward Shaping Trap (The Potential Paradox)

Faced with sparse terminal game outcomes, Verdi introduced extensive potential-based reward shaping on branch `origin/exp-elo-005-raise-relative-weight-in-value-targets` (split into `ai/reward/goal_potential.rs` in commit `0e37ed2c`), defining over 20 heuristic weights:
- `SHAPE_GOAL_COMPLETION = 75.0`
- `SHAPE_GOAL_RETAKE_W = 0.75`
- `SHAPE_CITY_TRAIN_BLOCKED = 200.0`
- `SHAPE_GOAL_RUIN_W = 0.35`
- `SHAPE_GOAL_DEFEND_COVER = 600.0`

### Why Potential Shaping Failed
1. **Reward Hacking & Local Optima**: As documented in `polyfish-rs/notes-runs-2026-07-14-16.md` (commit `e73713f5`), hand-tuned potential landscapes create artificial valleys and peaks. Units discovered behavioral loops that maximized local potential delta without advancing the game state toward victory, prompting HenBOMB to formally reject intermediate potential reward shaping (`vlab_*`).
2. **Fragile Hyperparameter Coupling**: Changing one constant broke unrelated components. As logged in commit `f9f82ca` and `current_understanding.md` (on branch `origin/exp-elo-005-raise-relative-weight-in-value-targets`), exempting threatened cities from train-blocking penalties broke `city_risk_is_priced_without_any_defend_order`, because a besieged city suddenly scored higher than a safe one.
3. **Detour Pathologies**: Prior to the 0.35 discount fix ([polyfish-rs/src/ai/scoring.rs:507-512](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/scoring.rs#L507-L512)), move ordering heuristics scored ruins and villages identically, causing units to detour to one-time ruins rather than securing compounding second cities.

---

## 6. Imperfect Information & Fog of War (FOW) Inadequacy

### 6.1 Single Determinization Vulnerability
Standard MCTS requires deterministic state transitions. In Polyfish, Fog of War was handled by determinization via `GameState::obscure_fog` ([polyfish-rs/src/states.rs:675-736](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/states.rs#L675-L736)), which fills unexplored tiles with predicted terrain or default field tiles:
- **Strategy-Stealing & Hallucination**: An MCTS tree operating on a single determinized state assumes perfect certainty about what lies beneath the fog in that sample.
- **Blindness to Information Value**: Standard MCTS cannot value an action simply because it *reveals information*. A warrior stepping onto a mountain to scout fog receives zero MCTS reward unless hardcoded heuristic bonuses (`regional_openness`, resource discovery) are injected into move ordering ([polyfish-rs/src/ai/scoring.rs:531-570](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/scoring.rs#L531-L570)).

---

## 7. Credit Assignment Noise Floor

In chess, an average game lasts ~40 moves per player, with deterministic piece captures.
In Polytopia:
- A game lasts 30–45 turns, comprising **300–600 atomic actions per player**.
- Map generation randomness (isolated island spawns vs. resource-dense spawns) introduces massive baseline variance into game outcomes.
- Backpropagating a single win/loss scalar (+1 / -1) from turn 45 across 450 atomic decisions produces a credit assignment signal that is ~50% noise ([DISCOVERIES.md:62-67](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/DISCOVERIES.md#L62-L67)).
- In mirror matches, identical models expanded at the same speed, generating empty relative labels ($\Delta \text{score} \approx 0$) while `sigma_completed_q` amplified interior noise by 5–6 logits ([notes.md:250-329](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/notes.md#L250-L329)).

---

---

## 8. Synthesis of Structural Bottlenecks in the Current Stack

Reviewing the codebase and empirical training records highlights six interconnected structural bottlenecks that enforce an upper bound on the current architecture:

1. **Planar Representation Bottleneck**: Forcing heterogeneous game dynamics (tech trees, global economy counters, discrete unit attributes) into spatial 2D feature grids ([polyfish-rs/src/ai/features.rs:42-128](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/features.rs#L42-L128)) strips topological and relational structure from the state representation.
2. **Search Horizon Depletion**: Because turns require 8–15 atomic decisions ending in `MoveType::EndTurn` ([polyfish-rs/src/types.rs:714](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/types.rs#L714)), MCTS simulation budgets ([polyfish-rs/src/ai/gumbel_mcts.rs:49](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/gumbel_mcts.rs#L49), [polyfish-rs/src/bin/self_play.rs:1176](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/bin/self_play.rs#L1176)) are consumed permuting intra-turn move sequences rather than evaluating multi-turn strategic outcomes ([notes.md:182-184](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/notes.md#L182-L184)).
3. **Model Capacity Ceiling**: A 64-channel 6-block ResNet ([polyfish-rs/src/ai/network.rs:174-175](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/network.rs#L174-L175)) does not possess sufficient parameter capacity to simultaneously capture spatial combat, non-spatial tech prerequisites, economic compounding, and exploration under uncertainty.
4. **Calibration vs Leaf Discrimination Breakdown**: High macro score calibration on long horizons fails to transfer to fine-grained tactical discrimination at search leaves, causing neural leaf evaluation to plunge in playing strength against heuristic baselines ([DISCOVERIES.md:68-73](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/DISCOVERIES.md#L68-L73), EXP_ELO_069 on branch `origin/exp-elo-005-raise-relative-weight-in-value-targets` commit `27055090`).
5. **Heuristic Reward Hacking**: Replacing sparse outcomes with large sets of handcrafted potential polynomials creates local optima and behavioral looping, divorcing policy learning from actual game victory.
6. **Information Value Blindness**: Single-sample determinization under Fog of War ([polyfish-rs/src/states.rs:675-736](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/states.rs#L675-L736)) prevents standard MCTS from valuing scouting and exploration without hardcoded heuristic overrides ([polyfish-rs/src/ai/scoring.rs:531-570](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/scoring.rs#L531-L570)).

---

## 9. Conclusion: The AlphaZero Paradigm Ceiling in 4X Domains

The empirical plateaus, passivity collapses, and tuning instabilities observed throughout Polyfish demonstrate that standard AlphaZero MCTS does not transfer gracefully to 4X strategy games.

In domains characterized by sequential multi-action turns, imperfect information, directed tech trees, and long economic compounding horizons, single-pass convolutional networks combined with flat move-by-move tree search fail to capture the multi-scale hierarchy of the decision space. The current architecture hits a ceiling not because of implementation bugs, but because its core representational and search assumptions are incompatible with the structural nature of Polytopia.
