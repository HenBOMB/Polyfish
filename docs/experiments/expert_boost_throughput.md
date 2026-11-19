# Self-Play Throughput: Findings & Staircase to the Ceiling

_Jul 4, 2026 — written after the tch/MPS eval swap, Metal trace analysis (v5),
and the actor-ceiling measurement. Companion doc: `profiling_report.md` (the
original candle investigation), `notes.md` (candle attention bug + backend
benchmark table)._

_Jul 5, 2026 — the MPSGraph bypass (metal backend) landed and rewrote the bottom
half of this doc. Headline: **~578 moves/s** at defaults
(`--eval-backend metal`, 128 actors, 2 servers × 2 pipelined GPU workers), vs
195 when this doc was written._

_Jul 5, 2026 — merged in the actor-ceiling sweep and the MPSGraph-recompile
investigation. **This is now the single source of truth for self-play throughput
work** (the old `actor_ceiling_sweep.md` was folded in here and deleted). It
also corrects the earlier "Stair 6 is now the binding constraint" claim: the
actor-ceiling sweep + dummy-eval evidence show the **eval path**, not actor
engine cost, is what binds at 578 — see "Where the 578→925 gap actually is"._

## Goal

Maximize self-play moves/sec at the locked-in search budget (64 MCTS iters)
until the binding constraint is physical hardware (14 CPU cores / CPU↔GPU
round-trip latency), not software waste.

## Measured timeline

| Milestone                                                          | moves/sec                        |
| ------------------------------------------------------------------ | -------------------------------- |
| candle Metal eval server (baseline)                                | ~31                              |
| tch/libtorch MPS backend swap                                      | 161                              |
| + cache-hash offload to actor threads                              | 195                              |
| tch, tuned (single readback, 32 actors)                            | 245 → later re-measured 157–242* |
| metal (MPSGraph) backend, cached executables, 32 actors / 1 server | 242*                             |
| metal + actor scaling (96 actors / 3 sharded servers)              | 435*                             |
| **metal + pipelined workers (128 actors / 2 servers × 2 workers)** | **~578***                        |
| Arc'd rows + async GPU submit, depth-2 worker pipeline (best: 4×2) | ~593 (wash)                      |
| **+ per-worker MTLBuffer pooling (best: 3 servers × 2 workers)**   | **~610–650**                     |
| **Actor ceiling** (dummy evaluator, no GPU — the hard cap)         | **~1,650**                       |

_\* Jul 5 numbers are on a newer `model.safetensors` than the Jul 4 rows — games
differ, so cross-day comparisons are approximate; same-day A/Bs are exact._

Target: **1,000+**, with ~2.85x of actor headroom above the current 578.

## Root cause (Metal System Trace, v5, pid 22923, 13.3s run) — MEASURED

The bottleneck is **CPU↔GPU synchronization latency, not GPU compute and not CPU
compute**:

- **GPU execution: 0.32s total = 2.5% of wall.** The net is tiny for an M3 Max
  (~23µs median per command buffer). GPU capacity is a non-issue at ~40x current
  load.
- **Eval thread: 8.46s = 66% of wall inside `CommandBufferSubmission`** — not
  working, _blocked_. Distribution is bimodal: 55% of CBs take <100µs (0.07s
  total); the slowest 23% (>500µs, up to 15ms) hold 7.9s = 93% of the time.
- ~5,877 CBs / ~735 forwards ≈ 8 CBs per forward; ~1.9 of them stall per
  forward. A stall = a synchronous device→CPU readback: each `.to_device(Cpu)`
  on MPS forces commit + waitUntilCompleted.
- Culprit in code: `tch_network.rs::forward_batch` did **5 separate readbacks**
  per forward (value + 4 policy heads). (Fixed in stair 1; the metal path does
  one dispatch.)

This retro-explains earlier anomalies: raising `--max-batch` 256→512 did nothing
(stall is per-forward, not per-row), and the hash offload helped (removed serial
CPU work between stalls).

Historical note: the "one thread only may touch the device" invariant came from
a **candle** Metal bug (tensor corruption under multi-threaded encoding — see
`eval_server.rs` header / `bug_handoff.md`). It does **not** apply to the
tch/libtorch backend, which has its own internal MPS stream locking.

## The staircase

Estimates marked (est.); everything else measured. Budget math: 43.6 leaf
evals/move, ~37 GPU rows/move after the ~15% cache → **1,000 moves/s needs ~37K
GPU rows/s** (vs ~8.7K at the 195 baseline).

| # | Stair                                                                                                                         | Status                                  | Expected landing (est.)                                                            |
| - | ----------------------------------------------------------------------------------------------------------------------------- | --------------------------------------- | ---------------------------------------------------------------------------------- |
| 1 | **Single readback** — concat value + 4 heads on-device into one `[B, 446]` row, one `.to_device(Cpu)`, split by offset on CPU | **DONE** (`tch_network.rs`)             | ~450–550                                                                           |
| 2 | **Actor-ceiling benchmark** — dummy evaluator, measures the hard cap                                                          | **DONE: ~1,650**                        | (calibrates everything below)                                                      |
| 3 | **Sharded eval servers** — N servers, requests routed `hash % N`                                                              | **DONE — FALSIFIED on tch** (see below) | ~~with #1: ~800–1,100~~ measured: 2 tch shards _halve_ throughput (157→83)         |
| 4 | **f16 upload** — cast features f32→f16 on CPU, upload half the bytes, cast back on-device                                     | dropped                                 | upload measured ≈2µs/forward on metal — nothing to save                            |
| 5 | **Batch/actor tuning** — refill `avg_batch` per shard, sweep actors 32→128                                                    | **DONE** (folded into the metal stair)  | actors 32→128 was worth ~2.4x _with_ metal                                         |
| 6 | **Actor-side engine cost** — movegen/apply/tree allocations                                                                   | later                                   | raises the ~1,650 ceiling itself — _not_ the binding constraint at 578 (see below) |

### ~~Why sharded servers ≈ the double-buffer/pipeline win~~ (falsified Jul 5)

The original reasoning assumed waiting parallelizes across libtorch threads. It
doesn't: **libtorch's MPS backend funnels every op through one serial C++
`dispatch_queue`**, one layer below tch-rs. Measured: 1 tch shard 157 moves/s, 2
tch shards 83 moves/s — a _47% regression_, both shards ~90% "busy" (i.e. parked
in the same serial queue). Hash-routing for cache locality remains correct and
is kept in `ShardedEvalHandle`.

## The metal stair (Jul 5) — MPSGraph bypass, measured

The libtorch serial queue can't be patched around, so inference moved off
libtorch entirely: `metal_network.rs` composes PolyZeroNet's forward as an
**MPSGraph** driven from Rust (`--eval-backend metal`, `metal-eval` feature) —
same tuned Apple kernels libtorch lowers onto, but on command queues _we_ own.
Parity vs tch-CPU: ~1e-6 on softmaxed heads.

What the microbenchmarks established (`examples/metal_bench.rs`, M3 Max
30-core):

- **Cost model per synchronous forward: ~1.3ms fixed + ~36µs/row.** Compiled
  executables are cached per batch size (first call per size pays graph compile;
  a fresh-graph-per-call implementation ran 5x slower end-to-end).
- **One blocking command queue tops out ~22K rows/s** (batch 256). The GPU is
  not the limit — **2/3/4 independent queues reach 32K/37K/40K rows/s**
  (1.45x/1.67x/1.79x). Sync dispatch latency is the single-queue wall.
- The 1K-moves/s budget (~37K rows/s) therefore _requires_ multiple concurrent
  queues; no single-threaded eval loop can get there.

How that turned into the shipped design
(`eval_server.rs::run_metal_pipelined_loop`): each eval server splits into a
**coalescer** (owns the LRU cache, batches requests) + **N pipelined GPU
workers** (own weights copy + own `MTLCommandQueue` each; the binding exposes no
completion callback, so per-queue blocking _is_ the completion signal). Unlike
sharding, the batch stream and cache stay unified.

Measured landscape (128 actors unless noted, newer model, same-day A/Bs):

| config                                    | moves/s     | note                                  |
| ----------------------------------------- | ----------- | ------------------------------------- |
| 96 actors, 3 sharded servers (no workers) | 425–435     | sharding starves batches (avg 47)     |
| 96 actors, 1 server × 2 workers           | 423         | single coalescer serializes           |
| 96 actors, 1 server × 3 workers           | 345         | deeper pipeline = worse actor latency |
| **128–160 actors, 2 servers × 2 workers** | **576–587** | shipped default                       |
| 128 actors, 3 servers × 2 workers         | 564         | 6 queues, avg_batch collapses to 13   |

Knobs probed and settled:

- `--coalesce-timeout-us`: 1000 is near-optimal (2000 → 343 moves/s; 500 → 550).
  **The system is actor-latency-bound** — anything that delays replies loses
  more than batch efficiency gains.
- `--leaf-batch` 4→6: fatter batches (avg 47→60) but a consistent ~10% _loss_
  (more evals/move, cache hit rate 0.19→0.17). Kept at 4.
- f16 upload: pointless on metal — upload is ~2µs of a ~4–6ms forward.

## Actor-ceiling sweep (Jul 5, 2:24pm) — where the ceiling actually is

Benchmark: `actor_ceiling` **dummy evaluator**, 64 mcts-iters, 64 games (partial
grid), M3 Max / 14 cores. The sim models eval as a fixed latency (µs/batch)
across N concurrent lanes — it does _not_ model cold compiles, batch starvation,
or eval-worker core contention, so it is an idealized upper bound for a given
eval latency.

**Pure actor ceiling (no eval sim):** 128 actors → **1,652 moves/s.** Hard CPU
cap; real self-play with GPU eval sits below this.

**Latency × lanes @ 128 actors** (partial — killed at 3000/2):

| latency | 1 lane | 2   | 4       | 6    | 8       |
| ------- | ------ | --- | ------- | ---- | ------- |
| 1000µs  | 240    | 465 | **925** | 1194 | 1469    |
| 1500µs  | 160    | 323 | 624     | 897  | 1053    |
| 2000µs  | 120    | 243 | 461     | 695  | 894     |
| 2500µs  | 97     | 193 | 380     | 538  | 732     |
| 3000µs  | 81     | 164 | **317** | —    | **592** |

_317 and 592 measured @ 32 games in focused follow-up. Scaling is smooth —
1500/2500 and lane=6 are interpolatable._ Rule of thumb from the grid:
`moves/s ≈ lanes × (1_000_000 / latency_us) × 0.24`.

**Actors @ 1000µs / 4 lanes (32 games, metal-shaped):**

| actors  | 64  | 96  | 128     | 160 |
| ------- | --- | --- | ------- | --- |
| moves/s | 868 | 847 | **882** | 822 |

128 actors is the sweet spot; 160 oversubscribes past 14 cores and regresses
(~7%).

**Deployment mapping:**

| Real config                    | Sim equivalent   | Grid (idealized) | Actual self-play |
| ------------------------------ | ---------------- | ---------------- | ---------------- |
| metal, 4 workers, ~1.3ms/batch | 1000µs / 4 lanes | **925**          | ~578             |
| 3ms, 4 lanes                   | 3000µs / 4 lanes | **317**          | —                |
| 3ms, 8 lanes (upper bound)     | 3000µs / 8 lanes | **592**          | —                |

Takeaways:

1. **More lanes always wins** in the _sim_ until the ~1,650 actor ceiling — but
   real extra lanes regressed (core contention + batch starvation, see the metal
   landscape). The sim overstates lane benefit because it charges nothing for the
   CPU/cores used by each lane.
2. **At 1ms + 4 lanes** (metal-shaped) the idealized ceiling is ~925 — headroom
   exists.
3. **128 actors is optimal** for 1000µs/4-lanes.

## Where the 578→925 gap actually is (corrects "Stair 6 is binding")

Three numbers frame it:

|                               | moves/s   | what it is                                            |
| ----------------------------- | --------- | ----------------------------------------------------- |
| Pure actor ceiling @128       | **1,652** | hard CPU cap, actors only, no eval                    |
| Metal-shaped sim (1ms/4-lane) | **925**   | eval modeled as a _warm, constant_ 1ms/batch blackbox |
| Real self-play                | **578**   | actual                                                |

Because a **dummy evaluator reaches 1.5–3K moves/s** (same actor engine work, no
GPU eval), actor-side engine cost is **not** the binding constraint at 578 — the
whole fight is the **578→925 eval-path gap** (~40%). This corrects the earlier
"Stair 6 — actor-side engine cost is now the binding constraint" claim: Stair 6
raises the ~1,650 ceiling and only matters once the eval path is fixed.

The 925→578 gap has three contributors: **coalesce overhead**, **eval-worker CPU
contending with 128 actors on 14 cores**, and **per-forward synchronous dispatch
latency** (the sim assumes a flat 1ms; reality is jittery). Eval workers run
~90%+ busy at ~23K rows/s in-loop vs 32–40K in isolation — the deficit is CPU
oversubscription + per-request latency, _not_ GPU capacity.

### Toward 1K — remaining levers, in order

1. **Per-row CPU cost in the eval path** — `RawPolicyOutput` full-row clones for
   cache insert + reply (~1.8 KB × 3 per row); `Arc`-ing rows would cut most of
   it. **This is the binding lever at 578** — squarely in the 578→925 gap, and
   the dummy-eval evidence confirms the eval path (not actors) is what caps us.
2. **`reduced_precision_fast_math` on the MPSGraph compile** — untried; needs a
   parity re-check before trusting. GPU isn't the limit, so expected small.
3. **Stair 6 — actor-side engine cost** (movegen/apply/undo/tree allocations):
   raises the ~1,650 ceiling itself. Relevant _after_ the eval path is unblocked
   toward ~925, not before.

### MPSGraph executable-recompile hypothesis — investigated & FALSIFIED (Jul 5)

An outside reviewer attributed the 925→578 gap to **MPSGraph executable
recompiles**: `metal_network.rs` compiles one `Executable` per distinct batch
size, and because production coalesced sizes vary (1–259) rather than being
fixed like the microbenchmark (256), the claim was that a large fraction of
forwards pay a graph compile. Proposed fix: pad every forward to a bucketed size
(multiple of 32) so only ~9 executables ever exist.

**Code review already weakened it.** The executable cache is a
`HashMap<usize, Executable>` with **no eviction**
(`entry(batch).or_insert_with(...)`). So compiles are strictly a **one-time
warmup cost per distinct size per worker**, bounded at ≤~260; in steady state
the cache is 100% hits. The metal path also already does **one** sync dispatch
per forward (`executable.run`), not the multi-readback stall the tch path had.

**Instrumented to measure it** (aggregate counters kept in tree; the per-compile
stderr line + `analyze_compiles.sh` were scaffolding, since removed):

- `metal_network.rs::build_and_compile` times the **full** build (graph
  tracing + compile, not just `compile()`) into per-net counters.
- `EvalServerStats` gained `compiles` / `compile_us`, folded per-worker into the
  `EVAL_SERVER_STATS_AGG` line as `compiles`, `compile_s`, `compile_frac_wall`,
  `compile_frac_busy` — the durable signal for re-checking after model/arch
  changes.

**Measured**
(`--num-games 32 --mcts-iters 64 --actors 128 --eval-backend metal`):

| timer scope                | compiles            | total compile time | distinct sizes (aggregate) |
| -------------------------- | ------------------- | ------------------ | -------------------------- |
| `compile()` only           | 256 (64/worker × 4) | 0.45 s             | 68                         |
| full build (trace+compile) | 258                 | **1.47 s**         | 69                         |

- Each worker compiles a size **exactly once** when encountered (averaging ~64
  compiles per worker out of the 68–69 globally distinct sizes). Graph tracing
  is ~2× the compile step, hence 0.45 s → 1.47 s.
- The 1.47 s is **summed across 4 workers warming concurrently** (~365 ms each).
  Wall-clock warmup window ≈ heaviest worker ≈ **0.38 s** — a low-single-digit %
  of a tens-of-seconds run, during which only actors routed to a still-compiling
  worker stall.
- Bucketing 69→~9 sizes would cut warmup ~1.47 s→~0.2 s (save ~0.3 s wall)
  **while adding a padding tax to every steady-state forward** (avg batch 47 →
  64 ≈ 36% more GPU rows + readback). Net-negative on a CPU-contended loop.

**Conclusion: recompiles are not the gap.** They are a ~0.4 s wall one-time
warmup, not the ~40% sim→real deficit. Dropped bucketing/prewarm; the lever is
`RawPolicyOutput` Arc-ing (item 1 above).

_Incidental:_ batch sizes are **not** "clustered heavily" — they smear across 69
distinct values (5–9 _and_ 47–61); the cache converges to hits via 69 one-time
compiles, not "a small set of common sizes." The stale `metal_network.rs`
module-doc comment claiming otherwise has been corrected.

## Arc'd rows + async GPU pipeline (Jul 5, later) — MEASURED: a wash (~593), but the instrumentation found the real cost

Levers 1 and "async completion" landed together. Net throughput effect ≈ zero —
but the `prep/wait/post` split they added pinpoints where the next win actually
is (see below). An early "742 moves/s" reading was a **phantom**: that run had
deadlocked (below) and the number never survived re-measurement.

1. **`EvalResult` rows are `Arc<RawPolicyOutput>`** (`eval_server.rs`). The ~3
   full-row clones per row (cache insert, reply, cache hit) are now refcount
   bumps. Call sites were untouched (deref coercion); only
   `examples/tch_parity.rs` needed a type fix.
2. **Async GPU submit + depth-2 worker pipeline**
   (`metal_network.rs::submit_batch`, `eval_server.rs` metal worker loop).
   Forwards go down via `runAsyncWithMTLCommandQueue`; completion is a
   per-forward **queue-order barrier** (empty command buffer committed
   immediately after the forward — it completes exactly when that forward does,
   later submissions notwithstanding). Each worker keeps one forward in flight
   while prepping/submitting the next and replying for the previous, so worker
   CPU (flatten, encode, readback, scatter) overlaps its own GPU execution.
   - Binding gotcha: `run_async_with_descriptor` with `results: None` reaches
     MPSGraph as an _empty_ (not nil) results array and crashes in the Swift
     shim — result `TensorData` must be caller-preallocated.
   - Parity: `async_submit_matches_sync_forward` (in-tree test) proves the async
     path is byte-identical to sync `forward_batch` with two forwards in flight.
   - **Deadlock (found via thread sample, fixed):** a worker holding undelivered
     replies must **never block on the jobs mutex** — an idle sibling parks
     inside `recv` _holding that lock_, and only wakes when actors submit work,
     possibly the very actors the in-flight forward must reply to. Fires at
     end-of-run drain when jobs go sparse. The poll branch uses `try_lock`;
     contention = "no job", settle the in-flight forward instead.
3. **`busy_s` now splits into `prep_s` / `wait_s` / `post_s`** in
   `EVAL_SERVER_STATS_AGG` — this is the instrument that adjudicates CPU-cost vs
   GPU-wait arguments from data instead of estimates.

**Measured** (`bench_eval_sweep.sh`, 128 games / 128 actors / 64 iters, iter-40
model, same-day A/B against the pre-change sweep):

| config  | before | after   | prep/wait/post (s) |
| ------- | ------ | ------- | ------------------ |
| 1×3     | 314    | 308     | 67/87/3            |
| 2×2     | 467    | 450     | 88/63/6            |
| 2×3     | —      | 459     | 94/84/6            |
| 3×2     | 522    | 563     | 92/91/8            |
| 3×3     | —      | 584     | 105/147/11         |
| **4×2** | 575    | **593** | 103/151/10         |
| 5×3     | 590    | 538     | 210/312/28         |

**Why a wash:** the split shows `post` collapsed (Arc-ing worked:
~0.1ms/forward) but `prep` ballooned to ~1–1.3ms/forward — the async entry
point's required **pre-allocated result buffers add 5 `TensorData` (MTLBuffer)
creations per forward** on top of the 2 inputs, and Metal buffer creation is
expensive. The pipelining win and the added prep cost roughly cancel; low-shard
configs (which coalesce fatter batches → fewer, bigger forwards) net slightly
negative, high-shard slightly positive. 5×3's regression is 20 workers × heavy
prep = pure CPU blowup (550s busy on 14 cores).

### Buffer pooling (same day) — MEASURED: 590 → ~610–650, new best at 3×2

The prep fix: pool `MTLBuffer` sets per worker (`metal_network.rs::BufferSet`,
pow2-bucketed capacity, min 64 rows; acquired per forward, returned on
readback). Inputs are written straight into the pooled buffer via `contents()`
(`submit_features` — no intermediate flatten `Vec`), outputs are wrapped in
exact-shape `TensorData::from_buffer` views (cheap) and never zero-filled (the
executable overwrites every element). Parity test extended to recycled-buffer
rounds

- both entry points.

| config  | pre-change | +async (wash) | +pooling            |
| ------- | ---------- | ------------- | ------------------- |
| 1×3     | 314        | 308           | 545                 |
| 2×2     | 467        | 450           | 571                 |
| 2×3     | —          | 459           | 507                 |
| **3×2** | 522        | 563           | **649, repeat 612** |
| 3×3     | —          | 584           | 533                 |
| 4×2     | 575        | 593           | 604                 |
| 5×3     | 590        | 538           | 593                 |

Reading the splits: `prep` at 3×2 fell ~92s → 50s and `wait` now dominates every
config — the eval path is finally **GPU-queue-bound**, not CPU-bound. The
optimum moved from many-thin-queues (4×2/5×3, which papered over CPU-heavy
workers) to **3 servers × 2 workers**; extra queues past ~6 now just add GPU
contention (`wait` at 5×3: 416s). Single-server (1×3) went 308 → 545 — fat
batches are cheap when prep is cheap, which reopens the unified-cache design as
a contender.

Remaining `prep` (~0.9ms/forward at 3×2) is row memcpys (~150µs), 7
`TensorData::from_buffer` wrappers, the barrier CB, and the compile-delta fold —
diminishing returns. The next real lever is `wait`: per-forward GPU cost
(`reduced_precision_fast_math`, fp16 inference) or fewer rows (cache, tree
reuse).

## What NOT to do

- ~~**CoreML / ANE / MPSGraph compiled forward ("stair 7"). Can't help.**~~
  **Falsified Jul 5** — the "actor ceiling binds first" reasoning was wrong
  because eval capacity _did_ bind (libtorch's serial queue capped it far below
  the actor ceiling). The MPSGraph forward is now the main path and worth 2.4x+.
- **Any candle inference tuning.** Retired for self-play; kernels are 2–8% of
  hardware capability at these shapes (see `profiling_report.md`), and its
  cross-attention loaded untrained weights (see `notes.md` bug entry).
- **Raising `--max-batch` further.** A no-op, measured twice (coalesced batches
  never approach 256 anyway — actor latency-sensitivity caps them first).
- **Raising `--coalesce-timeout-us` or `--leaf-batch` for fatter batches.** Both
  measured net-negative (Jul 5); the loop is actor-latency-bound.
- **Bucketing/padding batch sizes to cut MPSGraph recompiles.** Falsified —
  recompiles are a ~0.4 s one-time warmup, and padding taxes every steady-state
  forward (see above).
- **Adding more eval lanes/queues in reality.** The sim says lanes always win,
  but real extra queues starve batches and oversubscribe cores (3 servers × 2
  workers → 564, avg batch 13).

## Measurement checklist after each change

1. Same benchmark every time:
   `--num-games 32 --mcts-iters 64 --actors 128
   --eval-backend metal` (vary
   actors only in scaling sweeps).
2. Read `EVAL_SERVER_STATS_AGG`: `busy_frac` per shard (saturated vs starved),
   `avg_batch` (fill health), `cache_hit_rate` (should hold ~15%; drops mean
   routing or capacity regressed), and `compile_frac_busy` (warmup tax — expect
   low single digits).
3. The load-bearing number is **eval-thread ms/forward** (`busy_s / forwards`).
4. Past ~1K, watch the mixed bottleneck: more actors + hotter eval threads
   compete for the same 14 cores — the handoff point to stair 6 (raising the
   ~1,650 ceiling).

## Re-run commands

- Actor-ceiling sweep: `./bench_actor_ceiling.sh` (~2 min, 32 games); full grid
  (slow): `./bench_actor_ceiling.sh --full`.
- Self-play:
  `./target/release/self_play --num-games 32 --mcts-iters 64 --actors 128
  --eval-backend metal`;
  read `compiles` / `compile_frac_busy` off the `EVAL_SERVER_STATS_AGG` line for
  the warmup tax.

## Beyond 1.5K (different project)

Cheaper per-move engine work, fewer leaf evals per move (better tree reuse —
cache hits already prove transposition locality), or more machines. The 64-iter
search budget is a quality decision, not a throughput lever.
