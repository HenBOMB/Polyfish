# PolyAI Project-Level Agent Guidelines & Experimentation Protocol

This file defines the project-scoped operating rules, scientific standards, and training protocols for the **PolyAI** codebase. These rules apply strictly to this repository.

---

## 1. Scientific Experimentation Loop

- **Isolate One Degree of Freedom**: Change only one architectural mechanism, loss term, or search parameter per experiment (e.g., Action-Level Bilinear Compatibility in `EXP_ARCH_003` before Turn-Level EBM in `EXP_ARCH_004`). Never conflate multiple structural modifications in a single experiment.
- **Pre-Register Falsifiable Predictions**: Before launching a training run on compute pods, record in `hypothesis_driven_improvements.md`:
  1. The core theoretical hypothesis.
  2. Concrete numerical predictions (e.g., Probe $\Delta \ge 0.35$, Arena Win Rate $\ge 60\%$ vs. predecessor baseline).
  3. Strict falsification criteria (the exact metric floor or plateau that proves the hypothesis false).
- **Prior-Entrenchment Audit Before Resuming**: Before choosing between warm-starting from a late checkpoint (`--resume-from N`) and restarting from the DAgger foundation (`--resume-from 76`):
  - Empirically measure the predecessor's policy priors using neural behavioral probes.
  - If the predecessor severely suppressed valid tactical options due to a structural defect (e.g., Iter 130 suppressing `CityWall` by 35,000x due to factored policy blindness), resume from the uncorrupted foundation (`--resume-from 76`) so the new head trains in tandem with the trunk rather than spending compute fighting an entrenched negative prior.

---

## 2. Behavioral Probes & Mathematical Invariants

- **0-Search Neural Behavioral Probes**: Validate structural and tactical capabilities at raw policy level (0-ply inference) before evaluating through MCTS. If the neural trunk cannot express spatial or tactical distinctions at 0-search, MCTS is merely compensating for a blind prior.
- **Log-Space Policy Invariant**: Always compute policy comparisons across board locations in log-space ($\Delta = \log P(a_1) - \log P(a_2)$). Never rely on clamped linear probability products ($p < 10^{-10}$), which introduce artificial ratio distortions on low-probability spatial tiles.
- **Dual-Implementation Forward Parity Gate**: Before burning pod compute, verify numerical parity between Candle (Rust inference) and PyTorch (training) using `scripts/run_forward_parity.sh`. Raw logit delta must remain $< 10^{-3}$ ($< 10^{-6}$ for projection heads).

---

## 3. Experiment Documentation & Tracking

- **Maintain `hypothesis_driven_improvements.md`**: Every experimental architecture or search change must have an entry detailing hypothesis, architecture diffs, pre-registered metrics, and empirical verdicts (**COMMITTED**, **REJECTED**, or **WATCH**).
- **Maintain `notes.md`**: Record deep technical context, empirical numbers, post-mortems, and engineering discoveries.
- **Automated Checkpoint Migration**: All training and self-play entrypoints must programmatically invoke `migrate_model.py` on `model.safetensors` prior to launching self-play or server processes, ensuring missing heads are initialized before inference occurs.

---

## 4. Non-Destructive Pod & Training Telemetry

- **Telemetry Inspection Before Intervention**: Never abort, restart, or assume a long-running build or training process is stuck without inspecting live system telemetry:
  1. Check network socket throughput (`/proc/net/dev`) for active package or base image downloads.
  2. Check thread wait channels and CPU accumulation (`top`, `ps`) to distinguish CPU-bound compilation from I/O wait.
  3. Compare git commit timestamps against process start times (`ps -o lstart`) before assuming a build lacks recent changes.
- **Real-Time Telemetry Tracking**: Gauge pod training health through objective telemetry: moves/second throughput, eval-server queue depth, actor saturation, and loss component trajectories.
