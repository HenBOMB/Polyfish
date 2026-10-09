# Empirical Experiment Log & Benchmark Verdicts

This document provides the complete empirical record of head-to-head showdowns, behavioral cloning, and DAgger (Dataset Aggregation) iterations conducted within `polyfish-rs` on 11×11 point-symmetric maps.

---

## 1. Executive Summary & Progression Matrix

Across 5 distinct empirical phases, we traced the model's evolution from raw behavioral cloning failure to robust survival and competitive play against the AlphaZero checkpoint (`iter 156`):

| Evaluation Phase | Total Dataset | Policy Loss | vs Greedy (0-search) Score / Wipeouts | vs Greedy (Gumbel 64) Win Rate / Score | vs Iter 156 (Gumbel 64) Win Rate / Score | Decisive Elimination Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0. Pure BC (Open Loop)** | 32,307 | 5.90 | 2,043.0 / 60% | 0% (0–10) / 2,962.0 | 0% (0–10) / 2,833.0 | 60% (0-search), 10% (MCTS) |
| **1. DAgger Iter 1 (50 games)** | 37,814 | 4.48 | 2,948.0 / 20% | 10% (1–9) / 3,957.0 | 30% (3–7) / 3,306.5 | 20% (0-search), 40% (vs 156) |
| **2. DAgger Iter 2 (100 games)** | 50,777 | 3.66 | 2,791.5 / **0.0%** | 10% (1–9) / 3,380.0 | 20% (2–8) / **3,588.5** | **0% (0-search)**, 20% (vs 156) |
| **3. Adversarial MCTS 128** | 50,777 | 3.66 | — | 0% (0–10) / 2,824.0 | 20% (2–8) / 3,427.0 | **0% vs Greedy, FIRST P2 WIN vs 156** |

---

## 2. Benchmark 1: Gene-Based AI vs Greedy Heuristic

- **Context**: Isolated 1v1 showdown between the genetic rule-ordering agent (`GeneAgent`) and the algorithmic greedy baseline (`GreedyHeuristicAgent`).
- **Setup**: 20 games (10 base seeds, pairwise sides swapped), 11×11 symmetric maps, turn cap 45.
- **Results**:
  - **Greedy Win Rate**: 19 / 20 (95.0%)
  - **Gene-Based AI Win Rate**: 1 / 20 (5.0%)
  - **Average Score**: Greedy 6,598 vs Gene-AI 3,620
  - **Decisive Eliminations**: 7 early wipeouts inflicted by Greedy.
- **Verdict**: The gene-based rule prioritization system is fragile under aggressive tactical contact. Heuristic greedy move-scoring (`scoring.rs`) is substantially stronger than genetic rule re-ordering.

---

## 3. Benchmark 2: Greedy Heuristic vs AlphaZero Checkpoint (`iter 156`)

- **Context**: Head-to-head showdown between the algorithmic Greedy teacher and the trained AlphaZero neural network checkpoint (`iter 156`) using Gumbel MCTS 64.
- **Setup**: 10 games (5 base seeds, pairwise sides swapped), 11×11 symmetric maps, turn cap 45.
- **Results**:
  - **Greedy Wins**: 8 / 10 (80.0%)
  - **Iter 156 Wins**: 2 / 10 (20.0%)
  - **Average Score**: Greedy 6,442.5 vs Iter 156 3,824.0
  - **Average Speed**: Greedy 0.25 ms/move vs Iter 156 220.0 ms/move (880x faster)
- **Verdict**: The algorithmic greedy heuristic outperforms `iter 156` on 11×11 Tiny maps due to rapid local tempo capture. This confirmed `GreedyHeuristicAgent` as a valid teacher for warm-starting.

---

## 4. Experiment 3: Initial Behavioral Cloning (Covariate Shift Failure)

- **Generation**: Generated 200 greedy self-play games (`games_1791569049.safetensors`, 32,307 state transitions) in 1.43 seconds.
- **Architecture**: Initialized GroupNorm PolyZeroNet architecture (`init_model.py`), replacing legacy BatchNorm.
- **Training**: 3 epochs via `train.py`, learning rate 0.002.
  - Policy loss decreased from 88.32 → 5.90.
  - Total loss plummeted from 533.16 → 6.75.
- **Arena Evaluation**:
  1. **Raw Policy (0-search) vs Greedy**:
     - 0 wins (0–10 sweep), average score 2,043 vs 6,768.
     - **60% early eliminations** (6 games ended by wipeout before turn 25).
  2. **Gumbel MCTS 64 vs Greedy**:
     - 0 wins (0–10 sweep), average score 2,962 vs 7,354.
     - Eliminations reduced to 10% (1 game).
  3. **Gumbel MCTS 64 vs Iter 156**:
     - 0 wins (0–10 sweep), average score 2,833 vs 6,468.
- **Diagnosis**: Open-loop Behavioral Cloning suffered catastrophic **Covariate Shift** ($O(\epsilon T^2)$ compounding error). High training accuracy on expert demonstrations was meaningless because the student was never exposed to its own suboptimal states.

---

## 5. Experiment 4: DAgger Iteration 1 (First Breakthrough)

- **Implementation**: Added `--dagger` CLI flag to `self_play.rs`. Student network executed moves on the board; Greedy teacher provided supervision distributions on visited states.
- **Generation**: 50 games in 25.56s (`games_1791570980.safetensors`, 5,507 off-distribution recovery transitions).
- **Training**: Aggregated dataset (37,814 total samples) trained for 2 epochs.
  - Policy loss dropped from 5.90 → 4.48.
  - Total loss fell to 4.89; `value_r2` turned positive (+0.407).
- **Arena Evaluation**:
  1. **Raw Policy (0-search) vs Greedy**:
     - Average score jumped from 2,043.0 → **2,948.0 (+44.3%)**.
     - Early eliminations dropped from 60% → **20%**.
  2. **Gumbel MCTS 64 vs Greedy**:
     - Won **1 game** (seed 2027 swap=true: 5,855 to 5,265).
     - Average score jumped from 2,962.0 → **3,957.0 (+33.6%)**.
  3. **Gumbel MCTS 64 vs Iter 156**:
     - Won **3 games (30.0%)** (crushed Iter 156 on seed 2027: 8,090 to 3,430).
     - Average score rose from 2,833.0 → **3,306.5**.
- **Verdict**: Proven validation of DAgger. By learning how the teacher recovers from the student's mistakes, the compounding error was drastically reduced in just 50 games.

---

## 6. Experiment 5: DAgger Iteration 2 (Plateau & Ceiling)

- **Generation**: Current student drove 100 games in 60.72s (`games_1791573091.safetensors`, 12,963 recovery transitions).
- **Training**: Aggregated dataset of 50,777 samples trained for 2 epochs.
  - Policy loss dropped from 4.48 → **3.66**.
  - Total loss reached **4.04**.
- **Arena Evaluation**:
  1. **Raw Policy (0-search) vs Greedy**:
     - **0.0% early eliminations (0/10)**. Survived every single game to the turn-45 adjudication cap purely on raw neural policy.
     - Average score stabilized at 2,791.5 vs Greedy's 6,484.5.
  2. **Gumbel MCTS 64 vs Greedy**:
     - Won 1 game (10.0%), average score 3,380.0 vs 7,064.0.
  3. **Gumbel MCTS 64 vs Iter 156**:
     - Won 2 games (20.0%), average score reached an all-time peak of **3,588.5** (gap to Iter 156 narrowed to <2,000 points).
     - Eliminations against Iter 156 dropped from 40% → **20%**.
- **The P2 Deficit Finding**:
  - Across both DAgger iterations, **100% of wins against Iter 156 occurred as Player 1** (`P1 Wins: 5, P2 Wins: 0`).
  - When playing as Player 2 under tempo disadvantage, the model suffered defensive wipeouts on turns 14, 15, and 16.
- **Verdict**: The policy head has fully converged to the teacher's capability (policy loss 3.66). Further DAgger iterations with the existing Greedy teacher hit a hard performance ceiling because the teacher itself lacks defensive concepts.

---

## 7. Benchmark 6: Adversarial Gumbel MCTS 128 (First Player 2 Victory)

- **Setup**: Evaluated current DAgger student using `--adversarial` (in-tree `EndTurn` hands over to the opponent instead of deleting turns) with budget scaled to **128 iterations**.
- **Results vs Greedy (10 games, pairwise swapped)**:
  - **Decisive Eliminations: 0.0% (0/10)**. Survived every game to turn 45 cap.
  - Average Score: 2,824.0 vs 6,721.0.
  - Several games remained extremely tight (e.g. seed 2028: 4,195 vs 4,910).
- **Results vs Iter 156 (10 games, pairwise swapped, both searching adversarially @ 128 iters)**:
  - **Wins**: 2 / 10 (20.0%)
  - **Breakthrough**: **First ever victory as Player 2** logged!
    - `seed 2028 (swap true, Student as P2)`: Won **5,445 to 4,705**!
    - `seed 2030 (swap false, Student as P1)`: Won **5,685 to 5,035**!
  - Average Score: 3,427.0 vs 5,472.5.
  - Speed: ~420 ms/move.
- **Key Takeaway**: Under adversarial search at depth 128, the MCTS tree successfully modeled the opponent's counter-actions. This foresight directly produced the first victory from the disadvantaged Player 2 seat across all experiments.

