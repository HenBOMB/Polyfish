# Polyfish Project Discoveries & Research Chronicle

This document provides a comprehensive, ground-truth record of the scientific breakthroughs, engine bugs, empirical lessons, and architectural discoveries made throughout the development of Polyfish by **Verdi Kapuku** and **HenBOMB (Henry)**.

Every discovery documented below carries exact citations to the repository files, commits, or branch ledgers that establish it.

---

## 1. Search & MCTS Tree Dynamics

### 1.1 The Value Backpropagation Sign Error
* **Discovered by**: Diagnosed by Claude Fable ([expert_review.md:11-13](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/expert_review.md#L11-L13)); Verified & Fixed by Verdi Kapuku ([notes.md:79-87](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/notes.md#L79-L87))
* **Citations**: [expert_review.md:11-13, 72-99](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/expert_review.md#L11-L13), [notes.md:79-87](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/notes.md#L79-L87)
* **Finding**: In standard two-player games (like Chess), acting players alternate every move. In Polytopia, a player makes ~8–15 sequential moves per turn before calling `EndTurn`. MCTS backpropagation was negating the value sign on every tree edge (`value = -value`). As a consequence, during selection parents frequently preferred children whose evaluations were destructive to the acting player.
* **Fix**: Value polarity was anchored to player ID transitions across `EndTurn` boundaries rather than edge depth parity.

### 1.2 The Arena Double-Apply Disaster
* **Discovered by**: HenBOMB
* **Citation**: `polyfish-rs/notes-run-1783898182.md` on branch `hen-training-experimental` (commit `e73713f5`)
* **Finding**: `gumbel_mcts.rs::next_root_hash_for` applied the chosen move directly to the game object it was passed. While `self_play` used clones, the evaluation `arena` passed the live game object. Consequently, **every move in the arena was applied twice**. When a bot played `EndTurn`, the second execution immediately skipped the opponent's entire turn. This generated false "100% win rate" readings, caused exact 50-50 model splits (first mover locked the opponent out), and invalidated the early Elo ledger.
* **Fix**: Internal cloning inside `next_root_hash_for` was enforced, and `matches.jsonl` was deleted to re-rate all historical models from clean matches.

### 1.3 Search Depth Limits on Turn Horizons (~8 Plies per Turn)
* **Discovered by**: Verdi Kapuku
* **Citations**: [notes.md:74-75](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/notes.md#L74-L75), [notes.md:180-186](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/notes.md#L180-L186), [hypothesis_driven_improvements.md:43-53](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/hypothesis_driven_improvements.md#L43-L53) (EXP 3)
* **Finding**: Branching factor analysis revealed that Polytopia has a narrow but very deep search tree within each turn (~8 plies to complete one turn). At 64 MCTS simulations, the search tree only reaches 2–3 plies deep. In EXP 3, quadrupling the budget from 64 to 256 simulations failed to speed up first-village capture times (7.9 $\to$ 7.3 turns) while reducing throughput by 2.3×.
* **Lesson**: Reaching a village is a multi-turn walk beyond single-turn search horizons. Deepening single-ply search cannot replace directional priors and multi-turn value guidance.

---

## 2. Learning Signals, Self-Play & Value Head Dynamics

### 2.1 Unanchored Self-Play Passivity Collapse
* **Discovered by**: Verdi Kapuku
* **Citation**: [notes.md:174-187](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/notes.md#L174-L187)
* **Finding**: In run `1783285900`, pure outcome-driven self-play without heuristics suffered a catastrophic passivity collapse: captures plummeted from 6.5 to 3.2, attacks halved from 31 to 15, and average moves per game dropped from 592 to 460—even while policy cross-entropy loss *fell* from 3.12 to 2.70.
* **Lesson**: The network learned to play mutual non-aggression and passivity because neither player knew how to close games. Unanchored self-play confidently reinforced its own degraded habits.

### 2.2 Behavior Cloning (BC) Bootstrap Unlock
* **Discovered by**: Verdi Kapuku
* **Citation**: [notes.md:188-205](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/notes.md#L188-L205)
* **Finding**: To escape the passivity collapse, 1,024 games were generated using the network-free heuristic backend and pre-trained directly into fresh weights via supervised cross-entropy.
* **Outcome**: Examination of the resulting model showed captures jumped to 7.72 per game and first village capture speed dropped to 6.9 turns. Behavior Cloning baked opening competence directly into the weights at full gradient strength.

### 2.3 The "Empty Label" & Min-Max Q-Noise Amplification
* **Discovered by**: Verdi Kapuku
* **Citation**: [notes.md:250-329](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/notes.md#L250-L329)
* **Finding**: Per-decision trace logging revealed that relative score delta labels in mirror matches produced near-zero signal ($\approx 0$) because both identical copies of the model expanded at the same speed. Simultaneously, `sigma_completed_q` min-max normalized interior Q-values into $[0, 1]$, which multiplied residual noise by 5–6 logits during Sequential Halving. Search was actively destroying correct prior predictions.
* **Fix**: Added anchor games against heuristic opponents ([notes.md:338-343](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/notes.md#L338-L343)) and introduced the in-tree trust gate $\beta_{\text{tree}}$ to scale Q-values.

### 2.4 The Breakthrough Run (First Legitimate Beat of Greedy)
* **Discovered by**: HenBOMB
* **Citation**: `polyfish-rs/notes-runs-2026-07-14-16.md` on branch `hen-training-experimental` (commit `e73713f5`)
* **Finding**: In run `1784074147` (with the double-apply bug patched, 128 MCTS iterations, and 50% anchor games), checkpoint `iter84` reached Elo 846 vs Greedy's 641, winning 13 out of 16 matches (81% win rate). It was the first model in project history to legitimately and consistently beat the greedy heuristic.

### 2.5 Anchor Gate Limit Cycles & Catastrophic Forgetting
* **Discovered by**: HenBOMB
* **Citation**: `polyfish-rs/notes-runs-2026-07-14-16.md` on branch `hen-training-experimental` (commit `e73713f5`)
* **Finding**: When the training loop automatically reduced anchor games from 50% down to 5% after reaching a 55% win rate, the model suffered immediate catastrophic forgetting, crashing from 13-3 down to 4-12 against Greedy in 6 iterations.
* **Fix**: Tuned the anchor schedule parameters to `-W 0.70 -P 0.2` (70% win rate threshold, 20% permanent probe floor), which stabilized win rates in a healthy 44–72% band.

### 2.6 The Outcome Label Noise Floor
* **Discovered by**: HenBOMB
* **Citation**: `polyfish-rs/notes-runs-2026-07-14-16.md` on branch `hen-training-experimental` (commit `e73713f5`)
* **Finding**: Diagnosed that assigning credit to ~440 micro-moves per game from a single scalar outcome at turn 45 is ~50% noise due to initial map spawn luck and score adjudication caps, establishing an $R^2$ ceiling around 0.45–0.52.
* **Principle Formulated**: Formally approved *variance reduction* (mirror-relative labels) while rejecting arbitrary potential *reward shaping* (`vlab_*`) to avoid policy reward hacking.

### 2.7 Value Calibration vs. Discrimination Breakdown
* **Discovered by**: Verdi Kapuku
* **Citation**: `hypothesis_driven_improvements.md` (EXP_ELO_069) on branch `origin/exp-elo-005-raise-relative-weight-in-value-targets` (commit `27055090`)
* **Finding**: While tuning value heads on long-horizon games, aggregate calibration improved substantially ($R^2 = 0.972$ on late turns in EXP_ELO_067), yet net win rate dropped by 13 percentage points against hand-written heuristics.
* **Lesson**: Value heads trained via MSE regression learn macro score correlations (calibration) but fail to distinguish fine-grained tactical advantages between immediate candidate states (discrimination).

---

## 3. Game Mechanics, Heuristics & Movement Cues

### 3.1 Doorstep Flight (Chebyshev vs. Manhattan Bug)
* **Discovered by**: Verdi Kapuku
* **Citation**: [hypothesis_driven_improvements.md:65-74](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/hypothesis_driven_improvements.md#L65-L74) (EXP 5)
* **Finding**: Units located two tiles away from a neutral village walked away 50% of the time. Investigation revealed distance math used Manhattan geometry ($|x_1 - x_2| + |y_1 - y_2|$) while Polytopia units move diagonally on a Chebyshev metric ($\max(|x_1 - x_2|, |y_1 - y_2|)$), causing distance calculations to misguide unit priorities.

### 3.2 Capture Outranking Attack
* **Discovered by**: Verdi Kapuku
* **Citation**: [hypothesis_driven_improvements.md:76-85](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/hypothesis_driven_improvements.md#L76-L85) (EXP 6)
* **Finding**: Best attack moves were scored at 110.0 while village captures were scored at 99.8 in move ordering. Units standing directly on villages frequently attacked passing enemies rather than securing the city. Village capture scores were elevated above all attack scores, raising village conversion to 100%.

### 3.3 Frontier Resource Beacon
* **Discovered by**: Verdi Kapuku
* **Citation**: [hypothesis_driven_improvements.md:98-108](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/hypothesis_driven_improvements.md#L98-L108) (EXP 8)
* **Finding**: In Polytopia, natural resources only spawn adjacent to villages. A resource on the edge of the fog indicates an undiscovered village. Adding a heuristic pull toward resources touching fog accelerated blind exploration, lowering conditional capture turn to 5.97.

### 3.4 Greedy Anchor Replacement
* **Discovered by**: Verdi Kapuku
* **Citation**: [hypothesis_driven_improvements.md:87-96](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/hypothesis_driven_improvements.md#L87-L96) (EXP 7)
* **Finding**: The anchor teacher seat originally ran rollout MCTS, which added noise and obscured tuned heuristics. Replacing the rollout teacher with pure greedy move scoring yielded the largest single jump in capture speed (8.9 $\to$ 6.47 turns).

### 3.5 Negative Population Income Penalty
* **Discovered by**: HenBOMB
* **Citation**: Commit `03342047`, PR #22 (`fix/negative-population-income-penalty`)
* **Finding**: Engine bug where cities losing population retained income levels. Corrected engine rules to drop city level and apply income penalties when population becomes negative.

---

## 4. Neural Network Architecture & Representation

### 4.1 Fog-of-War Memory Channels
* **Discovered by**: HenBOMB
* **Citations**: [notes-memory.md:1-145](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/notes-memory.md#L1-L145), [features.rs:116-128](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/features.rs#L116-L128), commit `88dcb24c`
* **Finding**: In standard AlphaZero representations, units stepping into unexplored fog vanish instantly from the input tensor, creating complete tactical amnesia.
* **Implementation**: Added 6 observation channels (`CH_MEM_ENEMY_SEEN`, `CH_MEM_ENEMY_HP`, `CH_MEM_ENEMY_ATTACK`, `CH_MEM_ENEMY_RANGED`, `CH_MEM_ENEMY_NAVAL`, `CH_MEM_ATTACKED_HERE`) with exponential decay ($0.85^{\Delta t}$) computed over an 8-turn horizon.

### 4.2 BatchNorm to GroupNorm Migration
* **Discovered by**: Verdi Kapuku
* **Citations**: [CLAUDE.md:150-153](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/CLAUDE.md#L150-L153), commit `d8e45d12`
* **Finding**: BatchNorm created severe evaluation-time discrepancies due to mismatched running statistics between single-leaf inference and large training batches. Replaced with GroupNorm (`GN_GROUPS = 8`), unifying forward pass behavior across training and self-play.

### 4.3 Linear Pooling Paths (Dying ReLU Prevention)
* **Discovered by**: HenBOMB
* **Citation**: [CLAUDE.md:150-153](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/CLAUDE.md#L150-L153), commit `7c994745`
* **Finding**: The 1-channel spatial pooling convolutions (`p_pool_conv`) originally used un-normalized ReLU activations, which caused irreversible dead-neuron collapse. Kept the pooling convs fully linear (no norm, no activation).

---

## 5. Strategic Macro Bottlenecks & Elo Autopsies

### 5.1 The Strength Blind Spot
* **Discovered by**: Verdi Kapuku
* **Citation**: [hypothesis_driven_improvements.md:136-170](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/hypothesis_driven_improvements.md#L136-L170) (EXP 10)
* **Finding**: While behavioral metrics (first capture speed, SPT) steadily improved, the frozen-anchor Elo ladder revealed the network still lost ~2:1 against the greedy heuristic (25–34% win rate). Opening speed optimization had masked mid-game tactical incompetence.

### 5.2 Mid-Game Loss Autopsy (The Tech Trap)
* **Discovered by**: Verdi Kapuku
* **Citation**: [hypothesis_driven_improvements.md:187-205](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/hypothesis_driven_improvements.md#L187-L205) (EXP_ELO_001)
* **Finding**: Detailed telemetry identified the exact causal chain behind losses against Greedy:
  1. *Unit Production Deficit*: Model produced 1.8 units by turn 4 vs. Greedy's 3.0.
  2. *Expansion Stagnation*: Model grabbed one village and stopped, reaching a 3rd city in only 39% of games vs. Greedy's 81%.
  3. *The Tech Trap*: The model purchased 17.3 techs by turn 24 vs. Greedy's 12.1, spending stars on immediate scoreboard points that failed to compound into military or economic assets.

### 5.3 Macro-MCTS & Persistent Unit Goals
* **Discovered by**: Verdi Kapuku
* **Citation**: Commits `1882e553` through `4ffd88a9` on branch `origin/exp-elo-005-raise-relative-weight-in-value-targets`
* **Finding**: Recognizing that single micro-moves overwhelm MCTS, introduced `UnitGoalStore` with persistent unit IDs to assign multi-turn targets, and tested two-tier Macro-MCTS (`macro_stance` and `macro_order`).

---

## 6. Performance, Tooling & Infrastructure

### 6.1 Apple Metal Pipelined Inference Scaling (31 $\to$ 578 moves/sec)
* **Discovered by**: Verdi Kapuku
* **Citations**: [notes.md:128-142](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/notes.md#L128-L142), [expert_boost_throughput.md:28-40](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/expert_boost_throughput.md#L28-L40)
* **Finding**: Profiled Candle's Metal performance and identified severe CPU-GPU synchronization stalls from repeated `.to_device(Cpu)` readbacks. Replaced with MPSGraph, cached executables, and pipelined workers, accelerating self-play throughput from 31 to ~578–650 moves/sec.

### 6.2 Autonomous RunPod All-Local Pipeline
* **Discovered by**: HenBOMB
* **Citation**: `polyfish-rs/run_training_runpod.sh` and `polyfish-rs/runpod_setup.sh` on branch `hen-training-experimental` (commit `4749eb5a`)
* **Finding**: Eliminated external dependencies by building a single-box CUDA workflow on RunPod where self-play and `train.py` share the same GPU, including automatic CUDA priority detection and dependency scripts.

### 6.3 Live Steam Game Injection
* **Discovered by**: HenBOMB
* **Citation**: Commit `b0da6111` on branch `test-bot-execution` (`polyfish-mod/src/PolyfishBot.cs:1-50`)
* **Finding**: Implemented `SendCommand` integration inside the C# BepInEx/PolyMod suite, proving the trained engine can play the live commercial Steam game in real time.
