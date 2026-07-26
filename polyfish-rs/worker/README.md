# Self-play worker (second Mac)

Adds a second machine that generates self-play games against the current model
and ships them back. `run_training_loop.sh` is **unmodified** — remote games
land as ordinary `games_*.safetensors`, which `train.py`'s glob and the loop's
archive `mv` already pick up.

```
MAIN (this box)                          WORKER (second M3 Max)
run_training_loop.sh                     worker_loop.sh
  self_play → train.py → model             self_play → outbox/
      ▲                                        ▲   │
      │  games                    model+config │   │ games
      └──────────── sync_worker.sh ────────────┘◄──┘
```

Only the **worker** needs Remote Login; the main box always initiates SSH.

## Setup

On the worker — clone/copy the repo, then:

```bash
cd polyfish-rs && ./worker/setup_worker_mac.sh
```

Builds `self_play` with `metal,accelerate,metal-eval` and **no `tch-eval`**: a
generation-only worker never touches libtorch, so this skips PyTorch, the 2 GB
libtorch download, the 2.12.1 version pin, and `DYLD_LIBRARY_PATH`. MPSGraph is
the fast path regardless.

Enable Remote Login on the worker (Settings → General → Sharing), then from
main: `ssh-copy-id verdi@<worker>.local`.

## Running

Worker (idles until main publishes a model):

```bash
GAMES=32 ./worker/worker_loop.sh
```

Main, alongside the training loop:

```bash
WORKER_HOST=verdi@<worker>.local ./worker/sync_worker.sh
```

## How config stays in sync

`publish_manifest.py` reads the **running self_play process's argv** and
forwards it verbatim minus hardware flags. This matters because `--iteration`
is not cosmetic — it drives the `value_trust` β-ramp on σ(Q) in the policy
targets (`self_play.rs:1206`) and the anchor-decay offset. Re-deriving it would
mean re-implementing the loop's games-scaled schedule and `ITER_OFFSET`;
reading the live argv cannot drift from it.

Dropped and set locally: `--num-games`, `--actors`, `--eval-servers`,
`--eval-backend`, `--max-batch`, `--coalesce-timeout-us`, `--leaf-batch`, and
`--opponent` (league matches are the main box's evaluation duty).

On a cold start the manifest appears only once the daemon catches a self_play
window — during `train.py` it prints `push=skip` and waits.

## Correctness guarantees

| Risk | Guard |
|---|---|
| `train.py` globs a half-written game file | rsync → `inbox/`, then `mv` (same-fs rename) |
| Worker rsyncs a game it's still writing | worker writes to cwd, then `mv` into `outbox/` |
| Main ships a half-written model (`save_file` isn't atomic) | snapshot + safetensors header/length check before push |
| Filename collision (1-second timestamp, two machines) | `_<hosttag>` suffix; still matches `games_*` |
| Worker feature width ≠ model `conv1` | `spatial_maps.shape[1]/121` checked pre-ship; mismatch → `worker/rejected/` |
| Worker source drift | `GIT_SHA` compared each batch, warns |

## Known side effects

- **Replay window shortens in iterations.** `ARCHIVE_KEEP` prunes by *file
  count*, so two producers keep the buffer ~constant in games but halve its
  reach in iterations. Raise `-b`/`REPLAY_BUFFER_FILES` to restore the span.
- **`training_log.csv` undercounts.** `training_log.py parse-self-play` reads
  only the local self_play stdout, so logged `num_games` covers the main box
  only. Actual training data per iteration is higher.
- **Staleness.** Worker games are generated against the previous iteration's
  model — ordinary off-policy lag, already absorbed by the replay buffer.
