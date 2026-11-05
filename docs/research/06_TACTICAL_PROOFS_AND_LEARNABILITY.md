# Tactical Proofs and Adversarial Learnability Specification

## 1. Overview & Epistemic Framework

When designing Reinforcement Learning (RL) and Monte Carlo Tree Search (MCTS) agents for complex strategy games like *The Battle of Polytopia*, a central question is:
> **Which tactical behaviors can be naturally discovered and learned by the neural network through adversarial lookahead, and which require structural pruning or heuristic priors?**

This document catalogues the formal **Proof Suite** implemented in `polyfish-rs`. It distinguishes between:
1. **Physical Engine Determinism:** What the game tree physically allows and evaluates (verified via deterministic unit tests).
2. **Statistical Convergence in Practice:** The real-world constraints (search visit budgets, credit assignment, and prior initialization) that dictate whether an RL agent will actually converge to optimal play.

---

## 2. Catalog of Formal Proofs

All proofs listed below are covered by deterministic integration tests in the repository and run as part of CI (`cargo test`).

### Proof #1: Instant Fatal Retaliation
* **Test:** [`polyfish-rs/tests/adversarial_search.rs::proof_adversarial_tactics_retaliation_and_bad_trades`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/tests/adversarial_search.rs#L318)
* **Mechanic:** In Polytopia, combat damage is calculated simultaneously. The defender retaliates immediately during the attacker's own turn using its current health and defense multiplier.
* **Deterministic Verification:**
  - Setup: A 3.0 HP Warrior attacks a 15.0 HP Defender.
  - Formula: $\text{Defense Damage} = \frac{\text{Def Force}}{\text{Total Force}} \times \text{Def Stat} \times 4.5 = \frac{3.0}{3.6} \times 3.0 \times 4.5 = 11.25$.
  - Result: Warrior takes $11.25$ damage, drops below 0 HP, and is removed from the board **on the attacker's own turn**.
* **Learnability Verdict:** **Trivial to learn.** Zero lookahead required. The state value drop $\Delta V$ occurs at depth 1. Even a pure policy gradient with 0-ply search penalizes this action immediately.

---

### Proof #2: In-Tree Adversarial Punishment (Unfavorable Trades)
* **Test:** [`polyfish-rs/tests/adversarial_search.rs::proof_adversarial_tactics_retaliation_and_bad_trades`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/tests/adversarial_search.rs#L368)
* **Mechanic:** Attacking a high-defense unit with a full-health unit often leaves the attacker crippled. The fatal blow does not happen on turn 1, but on the opponent's counter-attack on turn 2.
* **Deterministic Verification:**
  - Setup: 10.0 HP Warrior attacks a 15.0 HP Defender.
  - Turn 1: Attacker deals $3.6$ damage, but suffers $8.1$ retaliatory damage, ending at **1.9 HP**.
  - Turn Boundary: MCTS simulates `EndTurn`.
    - Under *legacy single-player search*, `EndTurn` cycles back to Player 1. Opponent never acts. The unit appears alive and safe.
    - Under *adversarial search* (`set_adversarial_search(true)`), `EndTurn` transfers control to Player 2.
  - Turn 2: Player 2 finds `AttackMove` on the 1.9 HP Warrior, executes it, and **eliminates the Warrior**.
* **Learnability Verdict:** **Provable via adversarial lookahead.** The value head $V(s)$ receives a negative payoff due to unit loss. 
* **Operational Failure Mode:** Requires the search tree to cross the `EndTurn` boundary and sample the opponent's attack. If visit budget is $< 32$ or branching factor on Player 1's turn is large, visit starvation can prevent MCTS from reaching Player 2's turn.

---

### Proof #3: Rider Hit-and-Run (Escape Mechanic)
* **Test:** [`polyfish-rs/tests/adversarial_search.rs::proof_adversarial_tactics_rider_hit_and_run_escape`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/tests/adversarial_search.rs#L468)
* **Mechanic:** Units with `SkillType::Escape` (e.g. Riders) have `moved` reset to `false` after attacking ([`actions/units.rs`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/actions/units.rs#L1470)). This permits a subsequent `StepMove` on the same turn.
* **Deterministic Verification:**
  - Setup: P1 Rider at $(x, y)$, P2 Warrior at $(x+1, y)$ (distance 1), safe retreat tile at $(x-1, y)$ (distance 2 from Warrior).
  - **Branch A (Attack and Stay):**
    - Rider attacks Warrior, takes retaliation, but does not move away.
    - P1 plays `EndTurn`. Turn passes to P2.
    - P2 Warrior has an adjacent `AttackMove`. P2 attacks and **eliminates the Rider**.
  - **Branch B (Attack and Escape):**
    - Rider attacks Warrior, takes retaliation, then plays `StepMove` to retreat tile.
    - Rider is now at Chebyshev distance 2 from Warrior.
    - P1 plays `EndTurn`. Turn passes to P2.
    - P2 Warrior (movement 1, range 1) has **0 legal attack moves** on the Rider.
    - Rider survives unharmed.
* **Learnability Verdict:** **Provable.** Minimax backpropagation directly produces $Q(\text{Branch B}) \gg Q(\text{Branch A})$.
* **Operational Failure Mode:** If the neural net prior $\pi_\theta(a|s)$ strongly favors `EndTurn` immediately after attacking, the tree may prune the escape step before evaluating it.

---

### Proof #4: City/Capital Infiltration Threat
* **Test:** [`polyfish-rs/tests/adversarial_search.rs::in_tree_opponent_can_threaten_an_undefended_city`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/tests/adversarial_search.rs#L226)
* **Mechanic:** Walking a defender off a city when an enemy unit is nearby allows the enemy to step onto and capture/siege the city on the very next turn.
* **Deterministic Verification:**
  - In adversarial mode, moving off the capital and playing `EndTurn` immediately exposes legal enemy moves (`Step`, `Attack`, `Capture`) targeting the capital tile.
  - Under non-adversarial mode, this threat is 100% invisible because Player 2 never takes a turn in-tree.
* **Learnability Verdict:** **Provable.** Adversarial lookahead is strictly required to prevent the model from walking defenders away from frontline cities.

---

### Proof #5: Non-Viable Technology Pruning
* **Test:** [`polyfish-rs/tests/test_tech_pruning.rs`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/tests/test_tech_pruning.rs)
* **Mechanic:** Physical impossibilities (e.g. Sailing on waterless maps, Farming with 0 crops) and unactivatable Tier 2 investments (Trade without 5 stars liquid capital) waste search budget.
* **Deterministic Verification:**
  - Waterless maps mask Sailing/Aquatism to $-f32::\text{INFINITY}$, removing them from MCTS prior candidates entirely.
  - Tier 2 techs without capital are masked or penalized below `EndTurn` ($0.0$), ensuring MCTS does not squander top-$K$ exploration slots on dead investments.
* **Learnability Verdict:** **Not suitable for pure RL; requires hard pruning.** Even with millions of games, exploring impossible tech wastes exponential compute during MCTS candidate generation. Structural pruning solves this at zero runtime cost.

---

### Proof #6: Fog-of-War Horizon Barrier (POMDP Observation Erasure)
* **Test:** [`polyfish-rs/tests/test_category_2_proofs.rs::proof_category_2_item_1_fow_horizon_barrier`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/tests/test_category_2_proofs.rs#L34)
* **Mechanic:** In POMDPs, the agent cannot learn from what it cannot observe. `clone_for_mcts` completely blanks out unexplored terrain, structures, and units under fog.
* **Deterministic Verification:**
  - Real game state contains hidden villages and ruins.
  - In `clone_for_mcts`, `obscure_fog` erases all structures and units under fog (`structures.get(idx) == None`).
  - Stepping into the fog within MCTS simulation yields 0 in-tree village captures, 0 city income, and 0 structure rewards.
* **Learnability Verdict:** **Mathematically unlearnable without inductive bias.** Stepping into fog incurs movement cost and defensive exposure without in-tree reward. Without synthetic discovery reward or vision heuristics, MCTS strictly favors defensive corner-huddling.

---

### Proof #7: Top-K Gumbel Candidate Displacement
* **Test:** [`polyfish-rs/tests/test_category_2_proofs.rs::proof_category_2_item_2_top_k_gumbel_displacement`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/tests/test_category_2_proofs.rs#L85)
* **Mechanic:** Gumbel noise $g \sim \text{Gumbel}(0, 1)$ exhibits heavy right-tail extremes across large legal move sets ($N=50$). When dead moves are unpruned, extreme tail draws displace high-value moves from the Top-$K$ ($K=4$) candidate cut.
* **Deterministic Verification:**
  - Simulated 1,000 trials with 49 unpruned junk moves ($\ell = 0.0$) and 1 critical move ($\ell = 1.5$).
  - **Empirical Result:** In **68.7% of trials**, junk moves displace the critical move completely from the Top-4 cut.
  - When displaced, the critical move receives **exactly 0 simulation visits**.
  - When junk moves are pruned to $-f32::\text{INFINITY}$, displacement drops to **0.0%**.
* **Learnability Verdict:** **Physical search failure.** The network cannot learn the value of a move that is systematically starved of MCTS visits by unpruned competitors.

---

### Proof #8: Teacher Poisoning Under Supervised DAgger Distillation
* **Test:** [`polyfish-rs/tests/test_category_2_proofs.rs::proof_category_2_item_3_teacher_poisoning_dagger`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/tests/test_category_2_proofs.rs#L182)
* **Mechanic:** When training under DAgger imitation learning, the student is supervised by the heuristic teacher's target distribution $\pi^*_{\text{teacher}}$. If the teacher has an inverted heuristic (preferring PopGrowth over BorderGrowth for city territory $\ge 10$), the student is punished for selecting the superior move.
* **Deterministic Verification:**
  - Heuristic teacher scores PopGrowth at $208.0$ and BorderGrowth at $205.0$ ($\Delta = +3.0$), yielding $\pi^*_{\text{teacher}}(\text{PopGrowth}) = 0.953$ and $\pi^*_{\text{teacher}}(\text{BorderGrowth}) = 0.047$.
  - A student network that predicts $\pi_{\text{student}}(\text{BorderGrowth}) = 0.80$ produces a strictly positive cross-entropy gradient:
    $$\frac{\partial \mathcal{L}}{\partial z_{\text{BorderGrowth}}} = 0.80 - 0.047 = +0.753$$
  - Under gradient descent, $z \leftarrow z - \alpha \nabla$, the student's probability for BorderGrowth is **actively decreased**.
* **Learnability Verdict:** **Mathematically impossible to unlearn under pure DAgger.** Supervised distillation forces the student to converge to the teacher's policy ceiling, actively penalizing student innovations.

---

## 3. The Reality Matrix: RL vs Structural Engineering

| Tactical Behavior | Can Pure RL Learn It? | Engine Proof Status | Recommended Approach |
| :--- | :---: | :---: | :--- |
| **Instant Retaliation Deaths** | **Yes** (1-ply) | Verified (`adversarial_search.rs`) | Pure RL / Value Head |
| **Hit-and-Run / Kiting** | **Yes** (with budget) | Verified (`adversarial_search.rs`) | Adversarial MCTS + Move Ordering prior |
| **City Defense Garrisoning** | **Yes** (with budget) | Verified (`adversarial_search.rs`) | Adversarial MCTS |
| **Waterless Sailing / Dead Tech** | **No** (Wastes compute) | Verified (`test_tech_pruning.rs`) | **Hard Structural Pruning** (`-inf`) |
| **Tier 2 Activation Capital** | **Difficult** (Delayed payoff) | Verified (`test_tech_pruning.rs`) | **Hybrid Activation Pruning** |
| **Fog-of-War Bluffing** | **No** (Imperfect Info) | Verified via Belief State | Believed-state MCTS / Determinization |

---

## 4. Summary & Best Practices

1. **Do not use heuristics where adversarial search can see the outcome:** If an action results in an in-tree death or loss within 2 turns, let MCTS and the value network evaluate it naturally.
2. **Do use hard pruning where physics make the action useless:** Physical impossibilities (no water, no resources, insufficient stars for Tier 2 payoffs) should be masked out in the scoring/prior stage so MCTS spends 100% of its budget on meaningful branches.
3. **Guard against visit starvation:** When running fast self-play with $< 32$ visits per move, prioritize high-threat candidate generation (e.g., candidate move sorting) so that `EndTurn` and enemy retaliations are actually expanded within the visit budget.
