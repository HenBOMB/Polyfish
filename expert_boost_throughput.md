# Self-Play Throughput: Findings & Staircase to the Ceiling

*Jul 4, 2026 — written after the tch/MPS eval swap, Metal trace analysis (v5), and the
actor-ceiling measurement. Companion docs: `profiling_report.md` (the original candle
investigation), `notes.md` (candle attention bug + backend benchmark table).*

*Jul 5, 2026 update — the MPSGraph bypass (metal backend) landed and rewrote the
bottom half of this doc. Stairs 3–4 as originally written are superseded; see
"The metal stair (Jul 5)" below for what was measured vs. estimated. Headline:
**~578 moves/s** at defaults (`--eval-backend metal`, 128 actors, 2 servers × 2
pipelined GPU workers), vs 195 when this doc was written.*

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
| tch, tuned (single readback, 32 actors) | 245 → later re-measured 157–242* |
| metal (MPSGraph) backend, cached executables, 32 actors / 1 server | 242* |
| metal + actor scaling (96 actors / 3 sharded servers) | 435* |
| **metal + pipelined workers (128 actors / 2 servers × 2 workers)** | **~578*** |
| **Actor ceiling** (dummy evaluator, no GPU — the hard cap) | **~1,500** |

*\* Jul 5 numbers are on a newer `model.safetensors` than the Jul 4 rows — games
differ, so cross-day comparisons are approximate; same-day A/Bs are exact.*

Target: **1,000+**, with ~2.6x of actor headroom above the current 578.

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
| 3 | **Sharded eval servers** — N servers, requests routed `hash % N` | **DONE — FALSIFIED on tch** (see below) | ~~with #1: ~800–1,100~~ measured: 2 tch shards *halve* throughput (157→83) |
| 4 | **f16 upload** — cast features f32→f16 on CPU, upload half the bytes, cast back on-device; net stays f32 | dropped | upload measured ≈2µs/forward on metal — nothing to save |
| 5 | **Batch/actor tuning** — refill `avg_batch` per shard, sweep actors 32→64 | **DONE** (folded into the metal stair) | actors 32→128 was worth ~2.4x *with* metal |
| 6 | **Actor-side engine cost** — movegen/apply/tree allocations | next | raises the 1.5K ceiling itself |

### ~~Why sharded servers ≈ the double-buffer/pipeline win~~ (falsified Jul 5)

The original reasoning assumed waiting parallelizes across libtorch threads. It
doesn't: **libtorch's MPS backend funnels every op through one serial C++
`dispatch_queue`**, one layer below tch-rs. Measured: 1 tch shard 157 moves/s,
2 tch shards 83 moves/s — a *47% regression*, both shards ~90% "busy" (i.e. parked
in the same serial queue). Hash-routing for cache locality remains correct and is
kept in `ShardedEvalHandle`.

## The metal stair (Jul 5) — MPSGraph bypass, measured

The libtorch serial queue can't be patched around, so inference moved off libtorch
entirely: `metal_network.rs` composes PolyZeroNet's forward as an **MPSGraph** driven
from Rust (`--eval-backend metal`, `metal-eval` feature) — same tuned Apple kernels
libtorch lowers onto, but on command queues *we* own. Parity vs tch-CPU: ~1e-6 on
softmaxed heads (same class as libtorch's own CPU↔MPS agreement).

What the microbenchmarks established (`examples/metal_bench.rs`, M3 Max 30-core):

- **Cost model per synchronous forward: ~1.3ms fixed + ~36µs/row.** Compiled
  executables are cached per batch size (first call per size pays graph
  compile; a fresh-graph-per-call implementation ran 5x slower end-to-end).
- **One blocking command queue tops out ~22K rows/s** (batch 256). The GPU is
  not the limit — **2/3/4 independent queues reach 32K/37K/40K rows/s**
  (1.45x/1.67x/1.79x). Sync dispatch latency is the single-queue wall.
- The 1K-moves/s budget (~37K rows/s) therefore *requires* multiple concurrent
  queues; no single-threaded eval loop can get there.

How that turned into the shipped design (`eval_server.rs::run_metal_pipelined_loop`):
each eval server splits into a **coalescer** (owns the LRU cache, batches
requests) + **N pipelined GPU workers** (own weights copy + own `MTLCommandQueue`
each; the binding exposes no completion callback, so per-queue blocking *is* the
completion signal). Unlike sharding, the batch stream and cache stay unified.

Measured landscape (128 actors unless noted, newer model, same-day A/Bs):

| config | moves/s | note |
|---|---|---|
| 96 actors, 3 sharded servers (no workers) | 425–435 | sharding starves batches (avg 47) |
| 96 actors, 1 server × 2 workers | 423 | single coalescer serializes |
| 96 actors, 1 server × 3 workers | 345 | deeper pipeline = worse actor latency |
| **128–160 actors, 2 servers × 2 workers** | **576–587** | shipped default |
| 128 actors, 3 servers × 2 workers | 564 | 6 queues, avg_batch collapses to 13 |

Knobs probed and settled:
- `--coalesce-timeout-us`: 1000 is near-optimal (2000 → 343 moves/s; 500 → 550).
  **The system is actor-latency-bound** — anything that delays replies loses more
  than batch efficiency gains.
- `--leaf-batch` 4→6: fatter batches (avg 47→60) but a consistent ~10% *loss*
  (more evals/move, cache hit rate 0.19→0.17). Kept at 4.
- f16 upload (old stair 4): pointless on metal — upload is ~2µs of a ~4–6ms forward.

### Toward 1K from 578

Eval workers run ~90%+ busy at ~23K rows/s in-loop (vs 32–40K in isolation); the
gap is CPU oversubscription (128 actors + 2 coalescers + 4 GPU workers on 14
cores) and per-request latency, not GPU capacity. The remaining levers, in
expected order:
1. **Stair 6 — actor-side engine cost** (movegen/apply/undo/tree allocations):
   cuts CPU contention *and* raises the 1.5K ceiling. Now the binding constraint.
2. **Per-row CPU cost in the eval path** — `RawPolicyOutput` full-row clones for
   cache insert + reply (~1.8KB × 3 per row); `Arc`-ing rows would cut most of it.
3. **`reduced_precision_fast_math` on the MPSGraph compile** — untried; needs a
   parity re-check before trusting.

## What NOT to do

- ~~**CoreML / ANE / MPSGraph compiled forward ("stair 7"). Can't help.**~~
  **Falsified Jul 5** — the "actor ceiling binds first" reasoning was wrong because
  eval capacity *did* bind (libtorch's serial queue capped it far below the actor
  ceiling). The MPSGraph forward is now the main path and worth 2.4x+.
- **Any candle inference tuning.** Still true: retired for self-play; kernels are
  2–8% of hardware capability at these shapes (see `profiling_report.md`), and its
  cross-attention loaded untrained weights (see `notes.md` bug entry).
- **Raising `--max-batch` further.** Still a no-op, now measured twice (coalesced
  batches never approach 256 anyway — actor latency-sensitivity caps them first).
- **Raising `--coalesce-timeout-us` or `--leaf-batch` for fatter batches.** Both
  measured net-negative (Jul 5); the loop is actor-latency-bound.
- ~~**Micro-optimizing `RawPolicyOutput` allocations.**~~ Re-opened: at 578 moves/s
  the per-row clones are no longer ~1% — see "Toward 1K", item 2.

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
