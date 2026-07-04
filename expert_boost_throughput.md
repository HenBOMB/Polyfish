# Self-Play Throughput: Findings & Staircase to the Ceiling

*Jul 4, 2026 — written after the tch/MPS eval swap, Metal trace analysis (v5), and the
actor-ceiling measurement. Companion docs: `profiling_report.md` (the original candle
investigation), `notes.md` (candle attention bug + backend benchmark table).*

## Goal

Maximize self-play moves/sec at the locked-in search budget (64 MCTS iters) until the
binding constraint is physical hardware (14 CPU cores / CPU↔GPU round-trip latency),
not software waste.

## Measured timeline

| Milestone | moves/sec |
|---|---|
| candle Metal eval server (baseline) | ~31 |
| tch/libtorch MPS backend swap | 161 |
| + cache-hash offload to actor threads | 195 |
| **Actor ceiling** (dummy evaluator, no GPU — the hard cap) | **~1,500** |

Target: **1,000+**, i.e. ~5x from 195, with ~7.7x of actor headroom above us.

## Root cause (Metal System Trace, v5, pid 22923, 13.3s run) — MEASURED

The bottleneck is **CPU↔GPU synchronization latency, not GPU compute and not CPU
compute**:

- **GPU execution: 0.32s total = 2.5% of wall.** The net is tiny for an M3 Max
  (~23µs median per command buffer). GPU capacity is a non-issue at ~40x current load.
- **Eval thread: 8.46s = 66% of wall inside `CommandBufferSubmission`** — not working,
  *blocked*. Distribution is bimodal: 55% of CBs take <100µs (0.07s total); the slowest
  23% (>500µs, up to 15ms) hold 7.9s = 93% of the time.
- ~5,877 CBs / ~735 forwards ≈ 8 CBs per forward; ~1.9 of them stall per forward.
  A stall = a synchronous device→CPU readback: each `.to_device(Cpu)` on MPS forces
  commit + waitUntilCompleted.
- Culprit in code: `tch_network.rs::forward_batch` did **5 separate readbacks** per
  forward (value + 4 policy heads).

This retro-explains earlier anomalies: raising `--max-batch` 256→512 did nothing
(stall is per-forward, not per-row), and the hash offload helped (removed serial CPU
work between stalls).

Historical note: the "one thread only may touch the device" invariant came from a
**candle** Metal bug (tensor corruption under multi-threaded encoding — see
`eval_server.rs` header / `bug_handoff.md`). It does **not** apply to the tch/libtorch
backend, which has its own internal MPS stream locking. Multi-threaded tch use should
still get a quick stress test before being trusted at scale.

## The staircase

Estimates marked (est.); everything else measured. Budget math: 43.6 leaf evals/move,
~37 GPU rows/move after the ~15% cache → **1,000 moves/s needs ~37K GPU rows/s**
(vs ~8.7K at the 195 baseline).

| # | Stair | Status | Expected landing (est.) |
|---|---|---|---|
| 1 | **Single readback** — concat value + 4 heads on-device into one `[B, 446]` row, one `.to_device(Cpu)`, split by offset on CPU | **DONE** (`tch_network.rs`) | ~450–550 |
| 2 | **Actor-ceiling benchmark** — dummy evaluator, measures the hard cap | **DONE: ~1,500** | (calibrates everything below) |
| 3 | **Sharded eval servers** — N servers, requests routed `hash % N` | **DONE** (`ShardedEvalHandle`, `eval_server.rs`) | with #1: ~800–1,100 |
| 4 | **f16 upload** — cast features f32→f16 on CPU, upload half the bytes, cast back on-device; net stays f32 | not started | +10–20% on top |
| 5 | **Batch/actor tuning** — refill `avg_batch` per shard, sweep actors 32→64 | ongoing | multiplies with the above |
| 6 | **Actor-side engine cost** — movegen/apply/tree allocations | later | raises the 1.5K ceiling itself |

### Why sharded servers ≈ the double-buffer/pipeline win

Per-forward cost is dominated by *waiting* (~8–12ms sync) not compute (~0.5ms).
Waiting parallelizes perfectly: while server A is parked on its round-trip, server B
tensorizes and submits. Evidence demand exists: avg_batch=193 with the server 96% busy
means a standing ~200-leaf backlog at all times — a second server starts draining it
immediately. GPU at 2.5% won't notice the contention.

**Routing must be by hash, not round-robin.** Each shard owns its own LRU cache;
round-robin would halve the effective hit rate. `hash % N` sends a repeated position
to the same shard every time — cache locality preserved for free (the hash is already
computed on the actor thread).

### f16 upload details (stair 4)

Only the *input features* are quantized, on the wire: f32→f16 on CPU, upload, f16→f32
on-device, forward runs in f32 — no train/inference parity risk. Features are
overwhelmingly 0/1 flags and small counts, exactly representable in f16. Run one
parity check on any normalized fractional channels. Skip the u8 variant: 4x instead of
2x but needs a mixed-format scheme for fractional channels — not worth it while upload
is ~0.5–1ms of the per-forward cost.

## What NOT to do

- **CoreML / ANE / MPSGraph compiled forward ("stair 7").** Can't help: the actor
  ceiling (1.5K) binds before eval capacity would. Infinite-speed eval still lands at
  1,500. Only revisit if stair 6 raises the ceiling dramatically.
- **Any candle inference tuning.** The backend is retired for self-play; kernels are
  2–8% of hardware capability at these shapes (see `profiling_report.md`), and its
  cross-attention loaded untrained weights (see `notes.md` bug entry).
- **Raising `--max-batch` further.** Proven no-op; the cost is per-forward, not per-row.
- **Micro-optimizing `RawPolicyOutput` allocations.** ~1% of eval-thread time; tidy-up
  at best.

## Measurement checklist after each stair

1. Same benchmark every time: `--num-games 32 --mcts-iters 64 --actors 32` (vary
   actors only in stair-5 sweeps).
2. Read `EVAL_SERVER_STATS`: `busy_frac` per shard (saturated vs starved),
   `avg_batch` (fill health), `cache_hit_rate` (should hold ~15%; drops mean routing
   or capacity regressed).
3. The load-bearing number is **eval-thread ms/forward** (`busy_s / forwards`).
   After #1 expect ~7–9ms; if it's >12ms, one sync costs more than estimated and
   stair-4/5 priorities flip.
4. Past ~1K, watch for the mixed bottleneck: more actors + hotter eval threads compete
   for the same 14 cores. That's the handoff point to stair 6.

## Beyond 1.5K (different project)

Cheaper per-move engine work, fewer leaf evals per move (better tree reuse — note
cache hits already prove transposition locality), or more machines. The 64-iter search
budget is a quality decision, not a throughput lever.
