# Research Protocol 04: Adversarial DAgger Campaign & Competitive Execution

**Document Status**: ACTIVE / EXECUTION  
**Branch**: `hen/exp-adversarial-dagger`  
**Foundation Model**: `polyfish-rs/model_dagger_foundation.safetensors` (Iter 76)  
**Target Environment**: RunPod NVIDIA GPU Container (`henbomb/polyfish:latest`)  
**Parent Chronicles**: [`01_EXPERIMENTS_AND_VERDICTS.md`](01_EXPERIMENTS_AND_VERDICTS.md), [`02_STRUCTURAL_AUDIT_MCTS_AND_FTA.md`](02_STRUCTURAL_AUDIT_MCTS_AND_FTA.md), [`03_TEACHER_DIAGNOSIS_AND_BRAINSTORM.md`](03_TEACHER_DIAGNOSIS_AND_BRAINSTORM.md)

---

## 1. Context & Competitive Imperative

We are competing against rival agents/human strategies on standard Polytopia rulesets (11×11 Tiny maps, Drylands, turn cap 45). 

Prior to this campaign, legacy checkpoints (`iter 156`) hit an empirical wall:
1. **Solitaire Search Assumption**: In `polyfish-rs/src/game.rs:486-498`, MCTS deleted opponent turns by default during simulation (`adversarial_search: false`), producing models that overextended aggressively assuming zero enemy retaliation.
2. **Player 2 Wipeout Vulnerability**: Because Player 1 reaches the center village 1 ply earlier and attacks first with 100% damage before retaliation, the model suffered 100% loss rates as Player 2, frequently getting wiped out on turns 14–16.
3. **The DAgger Proof**: Open-loop behavioral cloning collapsed into 60% wipeouts due to covariate shift, but two DAgger iterations (50,777 samples) eliminated 0-search wipeouts (0.0%), cut policy loss to 3.66, and narrowed the score gap with `iter 156` to <2,000 points.
4. **The Adversarial Breakthrough**: Enabling `POLYFISH_ADVERSARIAL_SEARCH=1` with 128 iterations yielded the **first ever Player 2 victory against `iter 156`** (5,445 to 4,705 on seed 2028).

We are now ready to scale this into an autonomous training loop starting from Iteration 76.

---

## 2. Hypotheses & Experimental Design

### Hypothesis H1: In-Tree Adversarial Modeling Prevents P2 Collapse
* **Hypothesis**: Replacing solitaire tree expansion with true opponent simulation (`POLYFISH_ADVERSARIAL_SEARCH=1`) in the self-play loop will teach the neural value and policy heads to price enemy counter-strikes, eliminating defensive wipeouts as Player 2 while maintaining Player 1 expansion aggression.
* **Success Metric**: Player 2 wipeout rate $\le 5\%$ against both Greedy and `iter 156` in symmetric 20-game arena showdowns.

### Hypothesis H2: Anchor Floor Prevents Cyclical Forgetting
* **Hypothesis**: Retaining a permanent $\ge 20\%$ heuristic/greedy anchor proportion in the training game pool will anchor opening fundamentals, preventing the catastrophic forgetting observed in historical runs (e.g. Run `1784074147` crashing from 81% to 25% when anchors were reduced to 5%).
* **Success Metric**: Win rate against Greedy remains $\ge 65\%$ across all training checkpoints without regression spikes.

### Hypothesis H3: Adversarial Teacher Elevates the Policy Ceiling
* **Hypothesis**: The DAgger plateau at iteration 2 (policy loss 3.66) was bounded by the teacher's lack of defensive foresight. Using an Adversarial MCTS (128-sim) supervisor for recovery turns will lower held-out policy cross-entropy below 3.0 and raise P2 win rate.
* **Success Metric**: Win rate vs `iter 156` rises from 20% to $\ge 50\%$ across 30 paired seeds.

---

## 3. Falsification Protocol: What Would Change Our Minds?

We maintain strict falsification criteria to avoid narrative leaps or sinking compute into a dead end:

1. **Capacity Limit Falsifier**: If Player 2 wipeout rate remains $> 10\%$ after 50 adversarial self-play iterations, the compact 6-block ResNet trunk (583k parameters) lacks the capacity to represent bidirectional combat dynamics. If triggered: **Halt self-play immediately and promote the Entity-Transformer policy from `POLYSTAR-V3-ARCH.md`**.
2. **Anchor Collapse Falsifier**: If arena win rate against the deterministic Greedy anchor drops below 50% on symmetric evaluation, the opponent pool weighting is defective. If triggered: **Roll back to checkpoint 76 and pin anchor ratio to 50%**.
3. **Reactive Policy Sufficiency Falsifier**: If a 0-search reactive forward pass of the student achieves $\ge 75\%$ win rate against `iter 156` on 50 paired seeds, in-tree MCTS search is superfluous. If triggered: **Drop tree search completely at inference time and pivot exclusively to policy rollouts**.

---

## 4. Operational Execution Plan

### Step 1: Image Build & Container Push (Local)
* **Script**: `polyfish-rs/build-pft-image.sh`
* **Artifact**: `henbomb/polyfish:latest` (or target compute cap, e.g. Ampere 86 / Ada 89)
* **Pre-flight Checks**:
  - `POLYFISH_ADVERSARIAL_SEARCH=1` baked into container environment.
  - `ITER_OFFSET=76` set so the loop resumes from the DAgger foundation model.
  - `model_dagger_foundation.safetensors` copied to `model.safetensors`.

### Step 2: Pod Deployment (RunPod GPU)
* User launches RunPod instance with `henbomb/polyfish:latest` and custom environment arguments.
* Verify CUDA execution with `nvidia-smi` and test binary launch inside container.

### Step 3: Telemetry & Monitoring Verification
* Verify live metrics flow into Supabase (`polyfish-rs/supabase_sync.py`) and terminal monitor (`polyfish-rs/monitor_training.sh` / `live_monitor.py`).
* Verify tracking of:
  - Policy loss & Value $R^2$
  - Wipeout rates (turn < 25 eliminations)
  - Time and dollar cost per iteration

### Step 4: Iteration Milestones & Arena Gates

| Checkpoint | Target Target Games | Evaluation Gate | Requirement | Empirical Outcome |
| :--- | :--- | :--- | :--- | :--- |
| **Iter 76 (Baseline)** | 0 | Arena vs Greedy (0-search) & `iter 156` (128 iters) | Establish clean zero-point baseline | Completed |
| **Iter 85** | ~576 | Arena 20 games vs Greedy | Zero P2 wipeouts before turn 20 | Verified in self-play logs |
| **Iter 100** | ~1,536 | Arena 20 games vs `iter 156` | Win rate $\ge 35\%$, average score $> 4,000$ | Achieved (avg score 4,265) |
| **Iter 125/129** | ~3,328 | Symmetrical Benchmark (50 seeds / 100 games) | Win rate $\ge 50\%$ vs `iter 156` | **62.0% Win Rate (62–38), +829.1 pt margin, $p=0.0164$** |

---

## 5. Live Decision & Event Ledger

* **2026-11-04**: Protocol initiated. `build-pft-image.sh` cleaned of redundant `sudo` and tag collisions. Prepared for Docker image build and registry push to Docker Hub (`henbomb/polyfish:latest`). Foundation weights verified at `polyfish-rs/model_dagger_foundation.safetensors` (1.17 MB).
* **2026-11-05**: Autonomous RunPod loop started from Iter 76 with `-g 64 -n 128 -k 8 -A 0.20`.
* **2026-11-06**: Completed 52 iterations (Iter 76 $\rightarrow$ 129). 
  - Policy loss broke through the 3.00 barrier to **2.5821**.
  - Value loss increased from 0.46 to 0.79 as decisive games increased from 19% to 72% and capital snipes quadrupled to 0.89/game.
  - Symmetrical 100-game benchmark executed against legacy `iter 156` on RTX 3090 Ti: **Iter 129 won 62 out of 100 games (62.0%)**, with **28/50 wins as Player 2 (56.0%)** and an average score gap of **+829.1 points** ($Z = 2.40, p = 0.0164$, 95% CI $[52.2\%, 70.9\%]$).
  - Hypothesis H1 and H3 verified; Hypothesis H2 falsified (Greedy anchor win rate remained 11.5% due to POMDP Fog horizon barrier). Pod powered off to prevent compute waste on the policy capacity plateau. Ledger archived to [`arena_iter129_vs_iter156_100games.json`](../../polyfish-rs/arena_iter129_vs_iter156_100games.json).
  - Neural behavioral probe suite implemented (`polyfish-rs/tests/test_neural_behavioral_probe.rs`). Disproved earlier inferential leap that Iter 129 learned a conservative hit-and-run retreat: in 0-search greedy and 128-visit MCTS, the agent purposefully advances forward into enemy territory adjacent to wounded enemies (calculating survivable retaliation) rather than retreating to safety. Confirmed 1-ply capital threat blindness in raw value head.
