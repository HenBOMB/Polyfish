# Architectural Failures & Paradigm Dead-Ends in Polyfish

This document provides a post-mortem of the foundational **architectural hypotheses, design paradigms, and structural dead-ends** attempted by **HenBOMB (Henry)** and **Verdi Kapuku** during the evolution of Polyfish. 

Unlike a bug ledger, this document analyzes the **high-level machine learning and algorithmic concepts** that were designed, tested, and ultimately failed due to fundamental domain mismatches with Polytopia.

---

## 1. Executive Overview: The Paradigm Crisis

The core ambition of Polyfish was to scale AlphaZero from perfect-information board games (Chess, Go) to a 4X turn-based strategy game. Both researchers approached this challenge with distinct architectural philosophies:

1. **HenBOMB's Paradigm**: Pure reinforcement learning, tabula rasa self-play, and strict algorithmic fidelity to the DeepMind AlphaZero / Gumbel MCTS stack.
2. **Verdi's Paradigm**: Multi-head auxiliary supervision, dense potential-based reward shaping, and hybrid macro/micro heuristic steering.

Both paradigms reached empirical plateaus. HenBOMB's pure RL approach collapsed into unanchored passivity and catastrophic forgetting, while Verdi's multi-head and potential-shaping approach collapsed under negative gradient transfer and reward hacking.

---

## 2. HenBOMB's Architectural Dead-Ends

### 2.1 The Flat AlphaZero Transposition Fallacy
* **The Hypothesis**: Polytopia is played on a discrete 2D grid with turn-based rules; therefore, the AlphaZero formulation (a convolutional ResNet coupled with MCTS tree search over legal actions) will generalize directly as it did from Go to Chess.
* **Architectural Failure**: In Chess, a move transitions the turn to the opponent ($A \to B \to A \to B$). In Polytopia, a single turn requires an ordered sequence of 8–15 atomic plies ending in `MoveType::EndTurn` ([polyfish-rs/src/types.rs:714](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/types.rs#L714)).
* **Why it Failed**: Flat MCTS operates at the granularity of atomic actions. Given an average branching factor $b \approx 30$, a standard search budget of 32–64 iterations ([polyfish-rs/src/ai/gumbel_mcts.rs:49](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/gumbel_mcts.rs#L49), [polyfish-rs/src/bin/self_play.rs:1176](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/bin/self_play.rs#L1176)) exhausts all depth permuting intra-turn move orders (e.g. unit A before unit B). As diagnosed in [notes.md:182-184](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/notes.md#L182-L184), 64 simulations with ~15 visits to the winning child yields only 2–3 plies of depth vs ~8 plies per turn, so the tree search **never penetrates beyond the acting player's own turn**, rendering the search completely blind to opponent responses and long-term economic returns.

### 2.2 Tabula Rasa & Pure Outcome Reinforcement Collapse
* **The Hypothesis**: Like AlphaGo Zero, the agent should learn from scratch with zero human heuristics, optimizing solely via self-play against terminal game outcomes (+1 / -1 win/loss).
* **Architectural Failure**: HenBOMB advocated the tabula rasa / pure-outcome RL paradigm, but the empirical collapse occurred during Verdi's run `1783285900` ([notes.md:174-187](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/notes.md#L174-L187)) when the heuristic crutch was removed. In that run, the agent suffered complete passivity collapse: captures fell from 6.5 to 3.2, attacks dropped from 31 to 15, and average moves collapsed from 592 to 460—while policy loss deceptively dropped from 3.12 to 2.70.
* **Why it Failed**: In early random games, neither player knows how to coordinate an army, conquer cities, or close out a game. Because early games were capped at 30 turns ([notes.md:174](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/notes.md#L174)), passive games resulted in draws or arbitrary score adjudications. The network discovered that avoiding conflict prolonged survival, learning mutual non-aggression. Pure self-play reinforced its own degenerate passivity because the terminal reward signal contained zero gradient toward aggressive expansion.

### 2.3 The "Training Wheels" Anchor Curriculum Fadeout
* **The Hypothesis**: Handcrafted heuristics are merely bootstrap "training wheels." Once the self-play model achieves a winning record against the heuristic (e.g. >55% win rate), anchor games should be phased down from 50% to 5% so the policy can achieve superhuman transcendence.
* **Architectural Failure**: In run `1784074147` ([DISCOVERIES.md:54, 56-60](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/DISCOVERIES.md#L54-L60)), after checkpoint `iter84` reached an 81% win rate (13-3) against Greedy, reducing anchor games to 5% caused an immediate crash to a 25% win rate (4-12) within 6 iterations.
* **Why it Failed**: The architecture lacked an episodic memory consolidation or experience replay buffer. The policy network immediately drifted into degenerate self-play equilibria, rapidly forgetting early-game opening principles and tactical fundamentals. The concept of "fading out" the teacher failed because the self-play environment lacked the multi-turn depth to generate its own corrective pressures.

### 2.4 Intermediate Potential Reward Shaping (`vlab_*`)
* **The Hypothesis**: Sparse terminal outcomes make learning too slow; adding intermediate potential-based reward shaping terms (`vlab_*`) could give the value head a continuous gradient.
* **Architectural Failure & Rejection**: HenBOMB formally rejected training on intermediate potential reward shaping (`vlab_*`) before training models on it, diagnosing that arbitrary potential terms alter the optimization target and invite reward hacking ([DISCOVERIES.md:66](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/DISCOVERIES.md#L66), git commit `e73713f5:polyfish-rs/notes-runs-2026-07-14-16.md`). The `vlab_*` CSV columns were kept as zeroed scaffolding and never trained on.
* **Why it Failed**: Polytopia's non-linear economic compounding means local score or tile gains often conflict with optimal long-term strategy (e.g. sacrificing a unit to delay an enemy capture, or deliberately starving city growth to rush a high-tier tech). Artificial potential terms warp the optimal policy landscape, causing models to optimize proxy heuristics rather than the true win condition.

---

## 3. Verdi Kapuku's Architectural Dead-Ends

### 3.1 Value Calibration vs. Discrimination Breakdown (EXP_ELO_069)
* **The Hypothesis**: Higher value calibration across longer game horizons will directly improve MCTS leaf evaluations and overall playing strength against heuristic baselines.
* **Architectural Failure**: In EXP_ELO_069 ([DISCOVERIES.md:68-73](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/DISCOVERIES.md#L68-L73), commit `27055090` on branch `origin/exp-elo-005-raise-relative-weight-in-value-targets`), testing the neural leaf evaluator (`MacroLeaf::NetAsym`) in the arena resulted in a win rate drop of **13.1 percentage points** (44.0% down to 30.9%) compared to the heuristic leaf, even after value head calibration on late turns [30, 40) achieved $R^2 = 0.972$ in EXP_ELO_067.
* **Why it Failed**: Calibration measures aggregate score correlation across full games, whereas MCTS leaf evaluation requires per-decision discrimination between candidate children. Value heads trained via mean squared error learn macro scoreboard correlations but fail to distinguish fine-grained tactical advantages at search tree leaves. Furthermore, experimental branches exploring macro heads alongside micro actions on a shared 64-channel ResNet trunk ([polyfish-rs/src/ai/network.rs:174-175](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/network.rs#L174-L175)) suffered from negative gradient transfer.

### 3.2 The Handcrafted Potential Polynomial Stack (`ai/reward/goal_potential.rs`)
* **The Hypothesis**: If neural value heads cannot predict long-term compounding, we can hand-craft a comprehensive potential-based reward function $\Phi(s)$ covering all strategic objectives (Expand, Defend, Grow, Arm).
* **Architectural Failure**: On experimental branch `origin/exp-elo-005-raise-relative-weight-in-value-targets` (split in commit `0e37ed2c`), Verdi engineered an extensive potential polynomial with over 20 hand-tuned weights (`SHAPE_GOAL_COMPLETION`, `SHAPE_CITY_TRAIN_BLOCKED`, `SHAPE_GOAL_RETAKE_W`, `SHAPE_GOAL_RUIN_W`, `SHAPE_GOAL_DEFEND_COVER`).
* **Why it Failed (The Potential Trap)**:
  1. **Hyperparameter Fragility**: The terms were tightly coupled. As recorded in commit `f9f82ca` and `current_understanding.md` (on branch `origin/exp-elo-005-raise-relative-weight-in-value-targets`), altering `SHAPE_CITY_TRAIN_BLOCKED` to exempt threatened cities broke `city_risk_is_priced_without_any_defend_order` by causing besieged cities to score higher than peaceful ones.
  2. **Sub-Goal Detours**: In move ordering heuristics, setting identical approach weights for ruins and villages originally caused units to detour for one-time ruin rewards rather than securing permanent compounding second cities, requiring a manual 0.35 discount fix ([polyfish-rs/src/ai/scoring.rs:507-512](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/scoring.rs#L507-L512)).
  3. **Engineering Dead-End**: Replacing reinforcement learning with hand-tuned potential polynomials transformed the AI into an unmaintainable rule-based expert system that was brittle to map variations.

### 3.3 The Stochastic Rollout Teacher
* **The Hypothesis**: MCTS rollouts simulate deeper game trajectories than greedy 1-ply heuristics; therefore, using rollout MCTS for the anchor teacher seat will produce superior training targets.
* **Architectural Failure**: In EXP 7 ([hypothesis_driven_improvements.md:87-96](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/hypothesis_driven_improvements.md#L87-L96)), the anchor teacher seat using rollout MCTS produced noisy, tactical blunders that masked basic opening principles (anchor village capture turn was 8.9). Replacing rollout MCTS in the anchor seat with a pure greedy heuristic evaluator immediately improved the teacher's demonstration capture speed to 6.47 turns (100% capture rate).
* **Why it Failed**: Rollout simulations in Polytopia introduce extreme stochastic variance. Because random or pseudo-random rollouts make tactical blunders deep in the tree, backpropagated values diluted the sharp tactical directives of basic opening play.

### 3.4 Single-Pass Feedforward Compression
* **The Hypothesis**: A single forward pass through a convolutional network can simultaneously resolve macro-economic investment, tech-tree research, and unit tactical combat.
* **Architectural Failure**: The model repeatedly failed to balance tech purchases against army recruitment. It either hoarded stars for useless late-game technologies or neglected economy entirely to spam basic warriors.
* **Why it Failed**: Decisions in 4X games operate on fundamentally mismatched timescales:
  - **Macro Economy (Turns)**: Researching a tech or leveling a city takes multiple turns to amortize star cost.
  - **Meso Strategy (Turn/Phase)**: Directing armies toward a specific front or village.
  - **Micro Tactics (Plies)**: Attacking in the exact sequence to minimize retaliation damage ([polyfish-rs/src/actions/units.rs:1324-1365](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/actions/units.rs#L1324-L1365), [polyfish-rs/src/states.rs:578-585](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/states.rs#L578-L585)).
  Compressing all three timescales into a single 6-block feedforward convolutional pass without temporal or hierarchical decoupling proved computationally impossible.

---

## 4. Shared & Foundational Architectural Dead-Ends

### 4.1 Planar Grid Flattening of Non-Spatial Domains
Both researchers relied on the AlphaZero paradigm of projecting the spatial board state into a 2D spatial tensor ([polyfish-rs/src/ai/features.rs:42-65](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/features.rs#L42-L65)):
- **The Defect**: Polytopia is not purely spatial. The Technology Tree is a collection of prerequisite chains ([polyfish-rs/src/settings/technology.rs:466-479](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/settings/technology.rs#L466-L479)), and global economic metrics (stars, SPT, score) are non-spatial scalars.
- **The Consequence**: While global scalars were separated into `player_vec` ([polyfish-rs/src/ai/features.rs:704-722](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/features.rs#L704-L722)) and injected via cross-attention ([polyfish-rs/src/ai/network.rs:151-155](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/network.rs#L151-L155), [267-275](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/network.rs#L267-L275)), technology knowledge was reduced to a single normalized scalar (`tech_norm`), forcing the network to reason about multi-step prerequisite graphs without structural relational priors.

### 4.2 The Anonymous Entity Deficit (Stateless Units)
Until commit `94373d70`, `UnitState` had no persistent identifier ([polyfish-rs/src/states.rs:250-257](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/states.rs#L250-L257)):
- **The Defect**: Units were tracked purely by their grid coordinates (`coords.idx`) or their position in `tribe.units`. 
- **The Consequence**: Coordinates change every move, and array indices shift whenever any unit dies. The architecture had no persistent identity to bind goals to specific units across plies, preventing the development of any attention-based entity tracking or multi-turn unit coordination until stable unit IDs were introduced.

### 4.3 Single-Determinization Fog of War MCTS
To handle imperfect information, MCTS was run on a single determinized state generated by `GameState::obscure_fog` ([polyfish-rs/src/states.rs:675-736](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/states.rs#L675-L736)):
- **The Defect**: A single determinized sample replaces hidden fog with a fixed guess of terrain and enemy positions.
- **The Consequence**: Standard MCTS assumes perfect certainty about what lies beneath the fog in that sample. The search tree suffered from "strategy-stealing" and could never value scouting or information-gathering actions, requiring artificial heuristic beacons to force units to explore ([polyfish-rs/src/ai/scoring.rs:531-570](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/scoring.rs#L531-L570)).

---

## 5. Architectural Summary Table

| Architectural Hypothesis | Originator | Intended Benefit | Why It Failed in Practice |
| :--- | :--- | :--- | :--- |
| **Flat AlphaZero MCTS** | HenBOMB | Direct reuse of proven Go/Chess tree search | 8–15 plies per turn trapped search within current turn; blind to opponent ([notes.md:182-184](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/notes.md#L182-L184)) |
| **Tabula Rasa Self-Play** | HenBOMB | Unbiased superhuman discovery from zero heuristics | Passivity collapse; early random games lacked signal to discover conquest ([notes.md:174-178](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/notes.md#L174-L178)) |
| **Curriculum Anchor Fadeout** | HenBOMB | Wean model off heuristics down to 5% | Catastrophic forgetting; policy immediately unlearned opening principles ([DISCOVERIES.md:54-60](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/DISCOVERIES.md#L54-L60)) |
| **Intermediate Reward Shaping (`vlab_*`)** | HenBOMB | Dense gradient from potential terms | Rejected as scaffolding to prevent reward hacking; pure outcome remained noisy ([DISCOVERIES.md:66](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/DISCOVERIES.md#L66)) |
| **Calibration-Driven Value Heads** | Verdi | Output calibrated value over long horizons | Calibration $\ne$ discrimination; high $R^2$ failed to rank immediate leaf moves ([DISCOVERIES.md:68-73](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/DISCOVERIES.md#L68-L73)) |
| **Potential Polynomial Stack** | Verdi | Guide MCTS toward strategic goals with dense reward | The Potential Trap; brittle coupling and local optima on experimental branch |
| **MCTS Rollout Teacher** | Verdi | Deeper teacher signal than 1-ply heuristics | Rollout variance diluted sharp opening principles in the teacher seat ([hypothesis_driven_improvements.md:87-96](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/hypothesis_driven_improvements.md#L87-L96)) |
| **Planar Conv2D Flattening** | Joint | Spatial image-like processing of game state | Non-spatial tech chains and global economics lack relational graph priors |
| **Anonymous Unit Entities** | Joint | Simple coordinate-based unit state | Impossible to bind multi-turn goals to units until stable unit IDs added ([states.rs:250-257](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/states.rs#L250-L257)) |
| **Single-Determinization FOW** | Joint | Enable standard MCTS on imperfect information | Hallucinated certainty; MCTS blind to scouting without heuristic beacons ([scoring.rs:531-570](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/scoring.rs#L531-L570)) |
