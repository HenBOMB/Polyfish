# RunPod all-local training

Train Polyfish entirely on a single rented NVIDIA GPU: self-play **and** the
gradient step both run on the GPU. No Kaggle. Two new files, nothing existing
is modified:

- `runpod_setup.sh` — one-time (compile-once) setup, idempotent.
- `run_training_runpod.sh` — the loop. Same UX/flags as `run_training_loop.sh`.

## Why the persistent volume matters (this is the "don't waste hours" part)

The slow part is the **first Rust build** (fat-LTO, single codegen unit → 20–40
min). You do NOT want to pay GPU rates for that on every pod boot.

Put the repo on a **RunPod persistent network volume** (mounted at
`/workspace`). Then `target/`, `.venv`, and the cargo registry survive pod
stop/start, so you compile **once** and every later boot starts training in
seconds. Volume storage is ~$0.10/GB/month while stopped — far cheaper than a
rebuild on GPU time.

## Pod template

Pick an image with the CUDA **toolkit** (`nvcc`), not runtime-only — candle
compiles CUDA kernels. RunPod "PyTorch 2.x" / CUDA 12.x **devel** templates
work. A used **RTX 3090/4090** (24 GB) is the value pick; anything Ampere+ is
fine. Verify on the box:

```bash
nvidia-smi        # driver + GPU
nvcc --version    # CUDA toolkit present
```

## First run

```bash
cd /workspace/Polyfish/polyfish-rs        # your volume path
./runpod_setup.sh                          # slow ONCE; or: FAST_BUILD=1 ./runpod_setup.sh
tmux new -s train                          # survive SSH disconnects
./run_training_runpod.sh --reset           # fresh model from scratch
```

## Later boots (volume already set up)

```bash
tmux new -s train
SKIP_BUILD=1 ./run_training_runpod.sh --resume
```

`SKIP_BUILD=1` skips the cargo rebuild when the binaries already exist —
near-instant startup.

## Flags (identical to `run_training_loop.sh`)

```
--reset          wipe model + games, seed a fresh model (forces a new run)
--resume [id]    continue latest run (or a specific run_id)
-g <games>       games per iteration (default 64; drives all schedules)
-n <mcts>        MCTS iterations (default 128)
-i <iters>       loop iterations
-a <actors>      self-play actors (default 128 on GPU); -b doubles, -c pins 8
-r               reward shaping
-l <n>           league match every n iterations (0 = off)
-E 0             disable background Elo rating (skips a first-run arena compile)
```

Build/train env: `SKIP_BUILD=1`, `FAST_BUILD=1`, `TRAIN_EPOCHS=N` (default 2).

## What differs from the main loop

| | `run_training_loop.sh` | `run_training_runpod.sh` |
|---|---|---|
| Build | CPU (tch) — CUDA branch is disabled | CUDA (candle) |
| Gradient step | Kaggle kernel (`kaggle_manager.py all`) | local `train.py` on the GPU |
| Deps | Kaggle CLI + auth | none beyond the GPU |

The training **regime** (schedules, curriculum, league, anchor gate, value-trust
ramp, checkpoint/prune cadence, Elo) is byte-for-byte the same.

## Monitoring

The loop starts the backend server, so the live dashboard is at
`http://localhost:3000/training.html`. Expose port 3000 via RunPod's HTTP proxy
to watch it in a browser. Logs stream to `session.log`; Elo to `elo.log`.

## Getting the model back

`model.safetensors` and `checkpoints/` live in the working dir. Pull the trained
model down with `runpodctl send` / `scp`, or keep it on the persistent volume.
