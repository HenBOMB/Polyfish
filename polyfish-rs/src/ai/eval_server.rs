//! Batched inference for MCTS leaf evaluation.
//!
//! Two implementations of the same interface:
//! - [`EvalServer`] / [`EvalHandle`]: a dedicated OS thread owns the network
//!   and device; many actor threads send [`RawFeatures`] batches and block
//!   on a reply. The server coalesces pending requests across actors (up to
//!   `max_batch` items or `coalesce_timeout`, whichever comes first) into one
//!   `forward_t` call. A blocked actor thread parks and consumes no CPU,
//!   exactly like an awaited task would — so plain OS threads give the same
//!   coalescing behavior as an async runtime here without the async coloring
//!   spreading through the rest of MCTS. Self-play's actor count is RAM-bound
//!   (a `Game` clone + MCTS tree per actor) to the same tens-to-low-hundreds
//!   range either way, so there's no lightweight-task advantage to give up.
//! - [`InlineEvalHandle`]: evaluates immediately on the caller's own
//!   thread/device, no channel involved. Used by callers that don't need
//!   (or can't easily use) cross-actor batching: arena, UI analysis, tests.
//!
//! ## The Metal thread-safety invariant
//!
//! candle's Metal backend corrupts tensors if more than one thread encodes
//! ops against the same `Device` (see `bug_handoff.md`). `EvalServer` is the
//! *only* place that may hold device `Tensor`s for its network: requests in
//! (`RawFeatures`) and replies out (`f32` values, [`RawPolicyOutput`] rows)
//! must stay plain CPU data. Do not add a field or return type here that
//! carries a `Tensor` across the channel boundary.

use crate::ai::features::RawFeatures;
use crate::ai::network::{PolyZeroNet, RawPolicyOutput};
use candle_core::{Device, Tensor};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc as std_mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// One evaluated leaf: NN value estimate + decomposed policy logits, all as
/// owned CPU floats. The policy row is `Arc`d because every row is shared
/// between the LRU cache and one or more replies — at ~1.8 KB × tens of
/// thousands of rows/s, cloning full rows for each insert/reply was a
/// measurable allocator tax on the eval threads.
pub type EvalResult = (f32, f32, Arc<RawPolicyOutput>);

struct EvalRequest {
    features: Vec<RawFeatures>,
    /// Cache keys for `features`, one per row so w're not doing it on the single eval thread.
    // but across all actors.
    hashes: Vec<u64>,
    respond_to: std_mpsc::Sender<Vec<EvalResult>>,
}

/// Handle to a running [`EvalServer`]. Cheap to clone; each clone shares the
/// same request channel.
#[derive(Clone)]
pub struct EvalHandle {
    sender: std_mpsc::Sender<EvalRequest>,
}

impl EvalHandle {
    /// Submit a batch and return the reply receiver without blocking. The
    /// server will reply on the returned channel once it has coalesced this
    /// request with others (if any), run one `forward_t`, and sliced out this
    /// caller's rows. Order-preserving: the reply's `result[i]` corresponds to
    /// `batch[i]`.
    ///
    /// Split out from [`evaluate`](Self::evaluate) so a sharded handle can
    /// submit to N servers and then collect all replies — letting the N
    /// servers run their forwards concurrently instead of serializing on
    /// blocking `evaluate` calls.
    fn submit(&self, batch: Vec<RawFeatures>) -> std_mpsc::Receiver<Vec<EvalResult>> {
        let (tx, rx) = std_mpsc::channel();
        // Hash on the caller (actor) thread, not the eval thread
        let hashes: Vec<u64> = batch.iter().map(|f| f.hash()).collect();
        let request = EvalRequest {
            features: batch,
            hashes,
            respond_to: tx,
        };
        if self.sender.send(request).is_err() {
            panic!("BUG: EvalServer thread has shut down while a handle is still in use");
        }
        rx
    }

    /// Block until the server replies for a previously-submitted batch.
    fn collect(rx: std_mpsc::Receiver<Vec<EvalResult>>) -> Vec<EvalResult> {
        rx.recv()
            .expect("BUG: EvalServer dropped the response channel without replying")
    }

    /// Evaluate a batch of leaves. Blocks the calling thread until the
    /// server has coalesced this request with others (if any), run one
    /// `forward_t`, and sliced out this caller's rows. Order-preserving:
    /// `result[i]` corresponds to `batch[i]`. Convenience wrapper over
    /// [`submit`](Self::submit) + [`collect`](Self::collect).
    pub fn evaluate(&self, batch: Vec<RawFeatures>) -> Vec<EvalResult> {
        Self::collect(self.submit(batch))
    }
}

/// Owns the network + device on a dedicated thread; never touched from any
/// other thread. Dropping this stops the server (the request channel closes,
/// so outstanding/future `EvalHandle::evaluate` calls would panic — callers
/// must keep the server alive for as long as any handle is in use).
pub struct EvalServer {
    _thread: std::thread::JoinHandle<()>,
    stats: Arc<EvalServerStats>,
}

/// Live counters for the server's coalescing behavior, readable from any
/// thread while the server runs. The key tuning signal is
/// `rows / forwards` (average coalesced batch size): if it sits near the
/// per-request leaf batch, actors are not overlapping and the coalesce
/// timeout / actor count need adjusting.
#[derive(Default)]
pub struct EvalServerStats {
    /// Number of `forward_t` calls issued.
    pub forwards: AtomicU64,
    /// Total leaf rows evaluated across all forwards.
    pub rows: AtomicU64,
    /// Largest single coalesced batch seen.
    pub max_batch: AtomicU64,
    /// Wall time spent inside tensorize + forward + readback, in microseconds.
    /// Everything else is the server thread sitting idle waiting for work.
    /// On the metal pipelined path this is `prep + wait + post`.
    pub busy_us: AtomicU64,
    /// Metal only — `busy_us` component: CPU time flattening features,
    /// creating input tensors, and async-submitting to the GPU queue.
    pub prep_us: AtomicU64,
    /// Metal only — `busy_us` component: time blocked on the per-forward
    /// completion barrier. High `wait` with low `prep`+`post` means the GPU
    /// queue is the limit, not eval-thread CPU (and vice versa).
    pub wait_us: AtomicU64,
    /// Metal only — `busy_us` component: CPU time reading outputs back,
    /// building rows, scattering replies, and shipping cache inserts.
    pub post_us: AtomicU64,
    /// Leaf rows served from the eval cache without hitting the GPU.
    pub cache_hits: AtomicU64,
    /// Leaf rows that missed the cache and required a GPU `forward_t` row.
    pub cache_misses: AtomicU64,
    /// Metal only: number of MPSGraph executables built (one per distinct
    /// batch size, per worker) — a one-time warmup cost, not steady-state.
    pub compiles: AtomicU64,
    /// Metal only: total wall time spent tracing + compiling those graphs,
    /// in microseconds. Compare against `busy_us` to size the warmup tax.
    pub compile_us: AtomicU64,
}

/// Tuning knobs for request coalescing.
#[derive(Clone, Copy, Debug)]
pub struct EvalServerConfig {
    /// Maximum number of leaves combined into one `forward_t` call.
    pub max_batch: usize,
    /// How long to wait for more requests to coalesce with the first one
    /// pending, before flushing whatever has arrived.
    pub coalesce_timeout: Duration,
    /// Capacity of the eval cache (keyed by a 64-bit hash of the
    /// `RawFeatures` bytes). `Some(n)` enables an LRU of `n` entries; `None`
    /// disables caching. Every hit skips the GPU entirely. The cache lives on
    /// the eval-server thread, so it needs no locks.
    pub cache_capacity: Option<usize>,
    /// Metal (MPSGraph) backend only; ignored by candle/tch. Number of
    /// pipelined GPU worker threads behind the coalescer. 
    pub pipeline_workers: usize,
}

/// Default cache capacity: 512K entries. At ~1.8 KB per entry (8 B key + f32
/// value + ~1.78 KB `RawPolicyOutput` row) this is roughly 900 MB of resident
/// state — sized to fit transposition + same-turn re-search locality in
/// Polytopia without blowing past a typical self-play RAM budget.
pub const DEFAULT_CACHE_CAPACITY: usize = 512 * 1024;

impl Default for EvalServerConfig {
    fn default() -> Self {
        Self {
            max_batch: 256,
            coalesce_timeout: Duration::from_micros(1000),
            cache_capacity: Some(DEFAULT_CACHE_CAPACITY),
            pipeline_workers: 2,
        }
    }
}

/// Inference backend recipe, chosen by the caller and moved onto the
/// eval-server thread (must be `Send`). The actual device model is built from
/// this *inside* the thread so no device handle crosses a thread boundary.
pub enum BackendSpec {
    /// candle network (Metal/CUDA/CPU), already loaded by the caller.
    Candle(Arc<PolyZeroNet>),
    /// libtorch/MPS network, loaded from a state_dict on the eval thread.
    /// ~19x faster than candle Metal for PolyZeroNet (see `tch_network.rs`).
    #[cfg(feature = "tch-eval")]
    Tch {
        model_path: String,
        device: tch::Device,
    },
    /// MPSGraph network, bypassing libtorch's serial MPS dispatch queue
    /// entirely (see `metal_network.rs`). Loaded from a state_dict on the
    /// eval thread, same as `Tch`.
    #[cfg(feature = "metal-eval")]
    MetalMps { model_path: String },
}

impl BackendSpec {
    /// Build the concrete backend. Runs on the eval-server thread.
    fn build(self) -> InferenceBackend {
        match self {
            BackendSpec::Candle(network) => {
                let device = network.device();
                InferenceBackend::Candle { network, device }
            }
            #[cfg(feature = "tch-eval")]
            BackendSpec::Tch { model_path, device } => {
                let net = crate::ai::tch_network::TchPolyZeroNet::load(&model_path, device)
                    .expect("BUG: failed to load tch model for eval server");
                InferenceBackend::Tch(net)
            }
            #[cfg(feature = "metal-eval")]
            BackendSpec::MetalMps { model_path } => {
                let net = crate::ai::metal_network::MetalPolyZeroNet::load(&model_path)
                    .expect("BUG: failed to load metal model for eval server");
                InferenceBackend::Metal(net)
            }
        }
    }
}

/// The built inference backend, owned by (and only ever touched on) the
/// eval-server thread.
enum InferenceBackend {
    Candle {
        network: Arc<PolyZeroNet>,
        device: Device,
    },
    #[cfg(feature = "tch-eval")]
    Tch(crate::ai::tch_network::TchPolyZeroNet),
    #[cfg(feature = "metal-eval")]
    Metal(crate::ai::metal_network::MetalPolyZeroNet),
}

impl InferenceBackend {
    /// Run one batched forward over `feats`, returning per-row value + policy
    /// as plain CPU floats. This is the only place a device tensor exists.
    fn forward(&self, feats: &[RawFeatures]) -> (Vec<f32>, Vec<f32>, Vec<RawPolicyOutput>) {
        match self {
            InferenceBackend::Candle { network, device } => {
                let batch_size = feats.len();
                let mut spatial_flat = Vec::with_capacity(batch_size * RawFeatures::spatial_len());
                let mut player_flat = Vec::with_capacity(batch_size * RawFeatures::player_len());
                for feat in feats {
                    spatial_flat.extend_from_slice(&feat.spatial);
                    player_flat.extend_from_slice(&feat.player);
                }

                let spatial_tensor = Tensor::from_vec(
                    spatial_flat,
                    (
                        batch_size,
                        crate::ai::features::NUM_CHANNELS,
                        crate::ai::features::MAP_SIZE,
                        crate::ai::features::MAP_SIZE,
                    ),
                    device,
                )
                .expect("BUG: failed to tensorize eval-server spatial batch");
                let player_tensor =
                    Tensor::from_vec(player_flat, (batch_size, RawFeatures::player_len()), device)
                        .expect("BUG: failed to tensorize eval-server player batch");

                let (policy_out, value_out) = network
                    .forward_t(&spatial_tensor, &player_tensor, false)
                    .expect("BUG: eval-server forward_t failed");

                let values = value_out
                    .win_value
                    .flatten_all()
                    .expect("BUG: flatten win_value")
                    .to_vec1::<f32>()
                    .expect("BUG: win_value to_vec1");
                let progress = value_out
                    .progress_value
                    .flatten_all()
                    .expect("BUG: flatten progress_value")
                    .to_vec1::<f32>()
                    .expect("BUG: progress_value to_vec1");
                let policy_rows = policy_out
                    .to_raw_rows()
                    .expect("BUG: failed to read policy batch to CPU");
                (values, progress, policy_rows)
            }
            #[cfg(feature = "tch-eval")]
            InferenceBackend::Tch(net) => {
                let batch_size = feats.len();
                let mut spatial_flat = Vec::with_capacity(batch_size * RawFeatures::spatial_len());
                let mut player_flat = Vec::with_capacity(batch_size * RawFeatures::player_len());
                for feat in feats {
                    spatial_flat.extend_from_slice(&feat.spatial);
                    player_flat.extend_from_slice(&feat.player);
                }
                let (vals, pols) = net.forward_batch(&spatial_flat, &player_flat, batch_size);
                (vals, vec![0.0; batch_size], pols)
            }
            #[cfg(feature = "metal-eval")]
            InferenceBackend::Metal(net) => {
                let batch_size = feats.len();
                let mut spatial_flat = Vec::with_capacity(batch_size * RawFeatures::spatial_len());
                let mut player_flat = Vec::with_capacity(batch_size * RawFeatures::player_len());
                for feat in feats {
                    spatial_flat.extend_from_slice(&feat.spatial);
                    player_flat.extend_from_slice(&feat.player);
                }
                let (vals, pols) = net.forward_batch(&spatial_flat, &player_flat, batch_size);
                (vals, vec![0.0; batch_size], pols)
            }
        }
    }
}

impl EvalServer {
    /// Spawn the dedicated inference thread. The backend is built from `spec`
    /// on that thread and never touched elsewhere.
    pub fn start(spec: BackendSpec, config: EvalServerConfig) -> (Self, EvalHandle) {
        let (sender, receiver) = std_mpsc::channel::<EvalRequest>();
        let stats = Arc::new(EvalServerStats::default());
        let thread_stats = stats.clone();

        let thread = std::thread::Builder::new()
            .name("eval-server".to_string())
            .spawn(move || run_eval_loop(spec, receiver, config, thread_stats))
            .expect("BUG: failed to spawn eval-server thread");

        (
            Self {
                _thread: thread,
                stats,
            },
            EvalHandle { sender },
        )
    }

    pub fn stats(&self) -> &EvalServerStats {
        &self.stats
    }

    /// Deterministically stop the server thread and wait for it to finish.
    ///
    /// Call this *after* every [`EvalHandle`] has been dropped (which closes
    /// the request channel and makes the loop's `recv` error out). Joining
    /// here guarantees the inference backend — and any device tensors it owns
    /// — is fully dropped while the runtime is still healthy, *before* the
    /// process begins static/atexit teardown.
    ///
    /// This matters for the libtorch/MPS backend: libtorch destroys its global
    /// mutexes during atexit, and if this thread is still freeing MPS tensors
    /// at that moment it locks an already-destroyed `recursive_mutex` and the
    /// process aborts (`recursive_mutex lock failed: Invalid argument`).
    pub fn shutdown(self) {
        let _ = self._thread.join();
    }
}

fn run_eval_loop(
    spec: BackendSpec,
    receiver: std_mpsc::Receiver<EvalRequest>,
    config: EvalServerConfig,
    stats: Arc<EvalServerStats>,
) {
    // The metal backend gets the pipelined coalescer/worker split
    #[cfg(feature = "metal-eval")]
    if let BackendSpec::MetalMps { model_path } = &spec {
        if config.pipeline_workers >= 1 {
            return run_metal_pipelined_loop(model_path.clone(), receiver, config, stats);
        }
    }
    let backend = spec.build();
    let mut cache = config
        .cache_capacity
        .map(|cap| lru::LruCache::<u64, EvalResult>::new(NonZeroUsize::new(cap).expect("BUG: cache capacity must be > 0")));

    loop {
        // Block for the first request; once we have one, coalesce more
        // arrivals up to max_batch or until coalesce_timeout has elapsed
        // since this first request.
        let first = match receiver.recv() {
            Ok(req) => req,
            Err(_) => return, // all EvalHandles dropped; shut down
        };

        let mut requests = vec![first];
        let mut total_items: usize = requests[0].features.len();
        let deadline = Instant::now() + config.coalesce_timeout;

        while total_items < config.max_batch {
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            match receiver.recv_timeout(deadline - now) {
                Ok(req) => {
                    total_items += req.features.len();
                    requests.push(req);
                }
                Err(std_mpsc::RecvTimeoutError::Timeout) => break,
                Err(std_mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }

        let busy_start = Instant::now();
        evaluate_batch(&backend, requests, cache.as_mut(), &stats);
        stats.forwards.fetch_add(1, Ordering::Relaxed);
        stats.rows.fetch_add(total_items as u64, Ordering::Relaxed);
        stats
            .max_batch
            .fetch_max(total_items as u64, Ordering::Relaxed);
        stats
            .busy_us
            .fetch_add(busy_start.elapsed().as_micros() as u64, Ordering::Relaxed);
    }
}

/// Partition the coalesced batch by eval-cache membership: rows whose
/// `RawFeatures` hash to a cached entry are replied to immediately without
/// touching the GPU; the remaining misses are tensorized into one
/// `forward_t`, read back, inserted into the cache, and scattered back to
/// each request in original order.
///
/// Cache hit/miss counters are bumped on `stats` BEFORE replies are sent, so
/// a caller that reads stats right after its reply sees them up to date.
fn evaluate_batch(
    backend: &InferenceBackend,
    requests: Vec<EvalRequest>,
    mut cache: Option<&mut lru::LruCache<u64, EvalResult>>,
    stats: &EvalServerStats,
) {
    let per_request_len: Vec<usize> = requests.iter().map(|r| r.features.len()).collect();
    let total_rows: usize = per_request_len.iter().sum();

    if total_rows == 0 {
        for req in requests {
            let _ = req.respond_to.send(Vec::new());
        }
        return;
    }

    // Resolve every row against the cache first. `row_results` holds the
    // final answer for each row (hit now, miss filled in after the GPU call),
    // indexed as a flat slab in request order. `miss_slots` records which flat
    // positions still need a GPU result, in the order they'll appear in the
    // miss batch.
    let mut row_results: Vec<Option<EvalResult>> = Vec::with_capacity(total_rows);
    let mut miss_slots: Vec<usize> = Vec::with_capacity(total_rows);
    let mut miss_features: Vec<RawFeatures> = Vec::with_capacity(total_rows);
    let mut miss_hashes: Vec<u64> = Vec::with_capacity(total_rows);
    let mut served_from_cache: u64 = 0;

    for req in &requests {
        for (feat, &hash) in req.features.iter().zip(req.hashes.iter()) {
            let hit = cache
                .as_mut()
                .and_then(|c| c.get(&hash).cloned());
            match hit {
                Some(result) => {
                    served_from_cache += 1;
                    row_results.push(Some(result));
                }
                None => {
                    let flat_idx = row_results.len();
                    row_results.push(None);
                    miss_slots.push(flat_idx);
                    miss_features.push(RawFeatures {
                        spatial: feat.spatial.clone(),
                        player: feat.player.clone(),
                    });
                    miss_hashes.push(hash);
                }
            }
        }
    }

    let misses = miss_slots.len() as u64;

    if !miss_slots.is_empty() {
        let batch_size = miss_slots.len();
        let (values, progress, policy_rows) = backend.forward(&miss_features);

        debug_assert_eq!(values.len(), batch_size);
        debug_assert_eq!(progress.len(), batch_size);
        debug_assert_eq!(policy_rows.len(), batch_size);

        for ((&flat_idx, row), i) in miss_slots.iter().zip(policy_rows).zip(0..) {
            let result: EvalResult = (values[i], progress[i], Arc::new(row));
            if let Some(c) = cache.as_mut() {
                c.put(miss_hashes[i], result.clone());
            }
            row_results[flat_idx] = Some(result);
        }
    }

    // Count before replying: a caller may read stats as soon as its reply
    // arrives, and the counters must already reflect this batch.
    stats.cache_hits.fetch_add(served_from_cache, Ordering::Relaxed);
    stats.cache_misses.fetch_add(misses, Ordering::Relaxed);

    scatter_replies(requests, per_request_len, row_results);
}

/// Scatter contiguous slices of `row_results` back to each request in
/// original order and reply on its channel.
fn scatter_replies(
    requests: Vec<EvalRequest>,
    per_request_len: Vec<usize>,
    mut row_results: Vec<Option<EvalResult>>,
) {
    let mut offset = 0;
    for (req, len) in requests.into_iter().zip(per_request_len.into_iter()) {
        let results: Vec<EvalResult> = (offset..offset + len)
            .map(|i| row_results[i].take().expect("BUG: every row must be resolved"))
            .collect();
        offset += len;
        // Ignore send errors: the requesting thread may have given up.
        let _ = req.respond_to.send(results);
    }
}

/// One coalesced batch, cache-resolved, ready for a GPU worker: the rows
/// that missed the cache plus everything needed to complete + scatter the
/// replies on the worker thread.
#[cfg(feature = "metal-eval")]
struct MetalJob {
    miss_features: Vec<RawFeatures>,
    miss_hashes: Vec<u64>,
    /// Flat row index (request order) for each miss row.
    miss_slots: Vec<usize>,
    /// Per-row final results; cache hits pre-filled, misses `None`.
    row_results: Vec<Option<EvalResult>>,
    requests: Vec<EvalRequest>,
    per_request_len: Vec<usize>,
}

/// A [`MetalJob`] whose forward has been async-submitted to the GPU but not
/// yet completed: the in-flight handle plus everything needed to finish the
/// replies once it lands.
#[cfg(feature = "metal-eval")]
struct SubmittedJob {
    fwd: crate::ai::metal_network::InFlightForward,
    miss_hashes: Vec<u64>,
    miss_slots: Vec<usize>,
    row_results: Vec<Option<EvalResult>>,
    requests: Vec<EvalRequest>,
    per_request_len: Vec<usize>,
}

/// Stage 2 of the MPSGraph bypass plan: pipelined eval for the metal
/// backend.
///
/// Measured on M3 Max: one blocking queue tops out ~22K rows/s while 2–4
/// independent queues reach ~32–40K rows/s aggregate — the GPU has headroom
/// a single blocked thread can't reach.
///
/// So the server splits into:
/// - the **coalescer** (this thread): coalesces requests, resolves the LRU
///   cache, and hands each cache-missing batch to a worker channel;
/// - `pipeline_workers` **GPU workers**: each owns its own
///   `MetalPolyZeroNet` (own weights copy, own `MTLCommandQueue`, own
///   compiled-executable cache) and runs a depth-2 software pipeline:
///   async-submit forward N+1 (`MetalPolyZeroNet::submit_batch`), then
///   barrier-wait + reply for forward N — so each worker's CPU work
///   (flatten, encode, readback, scatter) overlaps its own GPU execution
///   instead of serializing with it.
///
/// Unlike hash-sharding across N servers, the batch stream and cache stay
/// unified — batches keep their full coalesced size instead of being split
/// N ways (sharding starved per-shard batches down to ~30 rows where fixed
/// per-forward overhead dominates).
///
/// Cache writes flow back to the coalescer over a channel and are drained
/// non-blockingly each cycle; two in-flight jobs can therefore evaluate the
/// same position twice (both miss before either insert lands) — harmless,
/// and already true across shards today.
#[cfg(feature = "metal-eval")]
fn run_metal_pipelined_loop(
    model_path: String,
    receiver: std_mpsc::Receiver<EvalRequest>,
    config: EvalServerConfig,
    stats: Arc<EvalServerStats>,
) {
    let (jobs_tx, jobs_rx) = std_mpsc::channel::<MetalJob>();
    let jobs_rx = Arc::new(std::sync::Mutex::new(jobs_rx));
    let (cache_tx, cache_rx) = std_mpsc::channel::<Vec<(u64, EvalResult)>>();

    let workers: Vec<std::thread::JoinHandle<()>> = (0..config.pipeline_workers)
        .map(|i| {
            let jobs_rx = jobs_rx.clone();
            let cache_tx = cache_tx.clone();
            let stats = stats.clone();
            let model_path = model_path.clone();
            std::thread::Builder::new()
                .name(format!("metal-eval-worker-{i}"))
                .spawn(move || {
                    let net = crate::ai::metal_network::MetalPolyZeroNet::load(&model_path)
                        .expect("BUG: failed to load metal model for eval worker");
                    let (mut last_compiles, mut last_compile_us) = (0u64, 0u64);

                    // Flatten + async-submit one job; CPU cost lands in `prep_us`.
                    let mut submit = |job: MetalJob| -> SubmittedJob {
                        let t0 = Instant::now();
                        let MetalJob {
                            miss_features,
                            miss_hashes,
                            miss_slots,
                            row_results,
                            requests,
                            per_request_len,
                        } = job;
                        let fwd = net.submit_features(&miss_features);
                        let prep = t0.elapsed().as_micros() as u64;
                        stats.forwards.fetch_add(1, Ordering::Relaxed);
                        stats.prep_us.fetch_add(prep, Ordering::Relaxed);
                        stats.busy_us.fetch_add(prep, Ordering::Relaxed);

                        // Fold this net's cumulative compile counters (grown
                        // inside submit_batch on a cold size) into the shared
                        // stats as a delta, so the AGG line sums across workers.
                        let (cc, cu) = (net.compiles(), net.compile_us());
                        if cc != last_compiles {
                            stats.compiles.fetch_add(cc - last_compiles, Ordering::Relaxed);
                            stats.compile_us.fetch_add(cu - last_compile_us, Ordering::Relaxed);
                            last_compiles = cc;
                            last_compile_us = cu;
                        }
                        SubmittedJob {
                            fwd,
                            miss_hashes,
                            miss_slots,
                            row_results,
                            requests,
                            per_request_len,
                        }
                    };

                    // Barrier-wait + readback + reply; time splits into
                    // `wait_us` (GPU) and `post_us` (CPU).
                    let complete = |sub: SubmittedJob| {
                        let SubmittedJob {
                            fwd,
                            miss_hashes,
                            miss_slots,
                            mut row_results,
                            requests,
                            per_request_len,
                        } = sub;
                        let t0 = Instant::now();
                        fwd.wait();
                        let wait = t0.elapsed().as_micros() as u64;

                        let t1 = Instant::now();
                        let (values, policy_rows) = fwd.readback();
                        // MPSGraph doesn't compute the progress head; stub
                        // zeros exactly like the sync Metal and Tch paths.
                        let progress = vec![0.0f32; values.len()];
                        debug_assert_eq!(values.len(), miss_slots.len());

                        let mut cache_inserts = Vec::with_capacity(miss_slots.len());
                        for ((&flat_idx, row), i) in miss_slots.iter().zip(policy_rows).zip(0..) {
                            let result: EvalResult = (values[i], progress[i], Arc::new(row));
                            cache_inserts.push((miss_hashes[i], result.clone()));
                            row_results[flat_idx] = Some(result);
                        }
                        scatter_replies(requests, per_request_len, row_results);
                        // Coalescer may have shut down while we ran; then the
                        // cache no longer matters.
                        let _ = cache_tx.send(cache_inserts);
                        let post = t1.elapsed().as_micros() as u64;
                        stats.wait_us.fetch_add(wait, Ordering::Relaxed);
                        stats.post_us.fetch_add(post, Ordering::Relaxed);
                        stats.busy_us.fetch_add(wait + post, Ordering::Relaxed);
                    };

                    // Depth-2 software pipeline: keep one forward in flight on
                    // the GPU while this thread preps/submits the next and
                    // replies for the previous. Block on the jobs channel only
                    // when nothing is in flight, so an idle worker parks.
                    let mut inflight: Option<SubmittedJob> = None;
                    loop {
                        let job = if inflight.is_some() {
                            // Never block on the jobs mutex while holding
                            // undelivered replies: an idle sibling parks
                            // inside `recv` *holding the lock* and only wakes
                            // when actors submit more work — possibly the very
                            // actors our in-flight forward must reply to.
                            // Blocking here deadlocks (measured, end-of-run
                            // drain); on contention just settle the in-flight
                            // forward instead.
                            match jobs_rx.try_lock() {
                                Ok(rx) => match rx.try_recv() {
                                    Ok(job) => Some(job),
                                    Err(std_mpsc::TryRecvError::Empty) => None,
                                    Err(std_mpsc::TryRecvError::Disconnected) => break,
                                },
                                Err(std::sync::TryLockError::WouldBlock) => None,
                                Err(std::sync::TryLockError::Poisoned(_)) => {
                                    panic!("BUG: jobs mutex poisoned")
                                }
                            }
                        } else {
                            match jobs_rx.lock().expect("BUG: jobs mutex poisoned").recv() {
                                Ok(job) => Some(job),
                                Err(_) => break, // coalescer gone; shut down
                            }
                        };
                        match job {
                            Some(job) => {
                                let submitted = submit(job);
                                if let Some(prev) = inflight.take() {
                                    complete(prev);
                                }
                                inflight = Some(submitted);
                            }
                            // Queue momentarily empty: settle the in-flight
                            // forward so its actors unblock; the next
                            // iteration parks on recv.
                            None => {
                                if let Some(prev) = inflight.take() {
                                    complete(prev);
                                }
                            }
                        }
                    }
                    if let Some(prev) = inflight.take() {
                        complete(prev);
                    }
                })
                .expect("BUG: failed to spawn metal eval worker")
        })
        .collect();
    drop(cache_tx); // coalescer keeps only the receiving side

    let mut cache = config.cache_capacity.map(|cap| {
        lru::LruCache::<u64, EvalResult>::new(
            NonZeroUsize::new(cap).expect("BUG: cache capacity must be > 0"),
        )
    });

    loop {
        let first = match receiver.recv() {
            Ok(req) => req,
            Err(_) => break, // all EvalHandles dropped; shut down
        };

        // Land completed cache inserts from the workers before resolving
        // this cycle's lookups.
        if let Some(c) = cache.as_mut() {
            while let Ok(inserts) = cache_rx.try_recv() {
                for (hash, result) in inserts {
                    c.put(hash, result);
                }
            }
        }

        let mut requests = vec![first];
        let mut total_items: usize = requests[0].features.len();
        let deadline = Instant::now() + config.coalesce_timeout;

        while total_items < config.max_batch {
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            match receiver.recv_timeout(deadline - now) {
                Ok(req) => {
                    total_items += req.features.len();
                    requests.push(req);
                }
                Err(std_mpsc::RecvTimeoutError::Timeout) => break,
                Err(std_mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }

        // Resolve the cache on the coalescer (single-threaded owner), then
        // ship misses to a worker. All-hit batches reply right here.
        let per_request_len: Vec<usize> = requests.iter().map(|r| r.features.len()).collect();
        let mut row_results: Vec<Option<EvalResult>> = Vec::with_capacity(total_items);
        let mut miss_slots: Vec<usize> = Vec::new();
        let mut miss_features: Vec<RawFeatures> = Vec::new();
        let mut miss_hashes: Vec<u64> = Vec::new();
        let mut served_from_cache: u64 = 0;

        for req in &requests {
            for (feat, &hash) in req.features.iter().zip(req.hashes.iter()) {
                let hit = cache.as_mut().and_then(|c| c.get(&hash).cloned());
                match hit {
                    Some(result) => {
                        served_from_cache += 1;
                        row_results.push(Some(result));
                    }
                    None => {
                        let flat_idx = row_results.len();
                        row_results.push(None);
                        miss_slots.push(flat_idx);
                        miss_features.push(RawFeatures {
                            spatial: feat.spatial.clone(),
                            player: feat.player.clone(),
                        });
                        miss_hashes.push(hash);
                    }
                }
            }
        }

        stats.rows.fetch_add(total_items as u64, Ordering::Relaxed);
        stats
            .max_batch
            .fetch_max(total_items as u64, Ordering::Relaxed);
        stats
            .cache_hits
            .fetch_add(served_from_cache, Ordering::Relaxed);
        stats
            .cache_misses
            .fetch_add(miss_slots.len() as u64, Ordering::Relaxed);

        if miss_slots.is_empty() {
            scatter_replies(requests, per_request_len, row_results);
            continue;
        }

        jobs_tx
            .send(MetalJob {
                miss_features,
                miss_hashes,
                miss_slots,
                row_results,
                requests,
                per_request_len,
            })
            .expect("BUG: all metal eval workers have died");
    }

    // Close the jobs channel so workers drain and exit, then join them so
    // the server's GPU state is fully dropped before process teardown.
    drop(jobs_tx);
    for worker in workers {
        let _ = worker.join();
    }
}

/// Synchronous, non-batched evaluator with the same interface as
/// [`EvalHandle`]. Runs `forward_t` immediately on the caller's own
/// thread/device — correct as long as that thread is the sole owner of the
/// device (true for arena/UI/test callers, which do not share a device
/// across threads the way self-play's actor pool does).
#[derive(Clone)]
pub struct InlineEvalHandle {
    network: Arc<PolyZeroNet>,
}

impl InlineEvalHandle {
    pub fn new(network: Arc<PolyZeroNet>) -> Self {
        Self { network }
    }

    pub fn evaluate(&self, batch: Vec<RawFeatures>) -> Vec<EvalResult> {
        if batch.is_empty() {
            return Vec::new();
        }
        let device = self.network.device();
        let batch_size = batch.len();

        let mut spatial_flat = Vec::with_capacity(batch_size * RawFeatures::spatial_len());
        let mut player_flat = Vec::with_capacity(batch_size * RawFeatures::player_len());
        for feat in &batch {
            spatial_flat.extend_from_slice(&feat.spatial);
            player_flat.extend_from_slice(&feat.player);
        }

        let spatial_tensor = Tensor::from_vec(
            spatial_flat,
            (
                batch_size,
                crate::ai::features::NUM_CHANNELS,
                crate::ai::features::MAP_SIZE,
                crate::ai::features::MAP_SIZE,
            ),
            &device,
        )
        .expect("BUG: failed to tensorize inline eval batch");
        let player_tensor = Tensor::from_vec(
            player_flat,
            (batch_size, RawFeatures::player_len()),
            &device,
        )
        .expect("BUG: failed to tensorize inline eval player batch");

        let (policy_out, value_out) = self
            .network
            .forward_t(&spatial_tensor, &player_tensor, false)
            .expect("BUG: inline eval forward_t failed");

        let values = value_out
            .win_value
            .flatten_all()
            .expect("BUG: flatten win_value")
            .to_vec1::<f32>()
            .expect("BUG: win_value to_vec1");
        let progress = value_out
            .progress_value
            .flatten_all()
            .expect("BUG: flatten progress_value")
            .to_vec1::<f32>()
            .expect("BUG: progress_value to_vec1");
        let policy_rows = policy_out
            .to_raw_rows()
            .expect("BUG: failed to read policy batch to CPU");

        values
            .into_iter()
            .zip(progress.into_iter())
            .zip(policy_rows.into_iter().map(Arc::new))
            .map(|((v, p), pol)| (v, p, pol))
            .collect()
    }
}

/// No-op evaluator used for the `actor_ceiling` benchmark. Returns a uniform
/// policy (all-zero logits, so softmax is uniform) and zero value per leaf,
/// with no model or threading—`evaluate` runs synchronously in O(rows).
///
/// Optionally simulates latency and limited concurrency by sleeping on each call,
/// and restricting the number of concurrent batches ("lanes") to emulate GPU contention.
#[derive(Clone)]
pub struct DummyEvalHandle {
    uniform: Arc<RawPolicyOutput>,
    stats: Arc<DummyStats>,
    /// Simulated per-batch processing time. `None` = return immediately
    /// (the pure actor-side ceiling).
    latency: Option<Duration>,
    /// Simulated concurrent-batch capacity. `None` = unlimited.
    lanes: Option<Arc<SimLanes>>,
}

/// Counting semaphore (std has none) bounding simulated in-flight batches.
struct SimLanes {
    free: Mutex<usize>,
    cv: Condvar,
}

impl SimLanes {
    fn acquire(&self) {
        let mut free = self.free.lock().unwrap();
        while *free == 0 {
            free = self.cv.wait(free).unwrap();
        }
        *free -= 1;
    }

    fn release(&self) {
        let mut free = self.free.lock().unwrap();
        *free += 1;
        self.cv.notify_one();
    }
}

/// Counters for the dummy evaluator, readable from any actor thread while a
/// benchmark runs. Lets `actor_ceiling` report leaves/s and leaves/move
/// alongside moves/s so the sim cost per move is visible.
#[derive(Default)]
pub struct DummyStats {
    pub evals: AtomicU64,
    pub leaves: AtomicU64,
}

impl DummyEvalHandle {
    pub fn new() -> Self {
        let spatial = crate::ai::features::MAP_SIZE * crate::ai::features::MAP_SIZE;
        Self {
            uniform: Arc::new(RawPolicyOutput {
                action_type: vec![0.0; 11],
                source_spatial: vec![0.0; spatial],
                target_spatial: vec![0.0; spatial],
                move_option: vec![0.0; 192],
            }),
            stats: Arc::new(DummyStats::default()),
            latency: None,
            lanes: None,
        }
    }

    /// Simulate an eval server: every batch takes `latency` to "process",
    /// and at most `lanes` batches are in flight at once across all actors
    /// (`0` = unlimited, i.e. latency with no contention).
    pub fn with_sim(mut self, latency: Duration, lanes: usize) -> Self {
        self.latency = (!latency.is_zero()).then_some(latency);
        self.lanes = (lanes > 0).then(|| {
            Arc::new(SimLanes {
                free: Mutex::new(lanes),
                cv: Condvar::new(),
            })
        });
        self
    }

    pub fn stats(&self) -> &DummyStats {
        &self.stats
    }

    pub fn evaluate(&self, batch: Vec<RawFeatures>) -> Vec<EvalResult> {
        self.stats.evals.fetch_add(1, Ordering::Relaxed);
        self.stats
            .leaves
            .fetch_add(batch.len() as u64, Ordering::Relaxed);
        if let Some(latency) = self.latency {
            match &self.lanes {
                Some(lanes) => {
                    lanes.acquire();
                    std::thread::sleep(latency);
                    lanes.release();
                }
                None => std::thread::sleep(latency),
            }
        }
        batch.iter().map(|_| (0.0, 0.0, self.uniform.clone())).collect()
    }
}

impl Default for DummyEvalHandle {
    fn default() -> Self {
        Self::new()
    }
}

/// N independent [`EvalServer`]s, each with its own weights copy + LRU cache.
/// Leaves are routed by `hash % N` so a given position always lands on the
/// same shard — preserving cache locality at zero extra cost (the hash is
/// already computed on the actor thread in [`EvalHandle::submit`]).
///
/// `evaluate` submits to all non-empty shards (non-blocking channel sends),
/// then collects all replies. This is the whole point: the N servers run
/// their forwards **concurrently**. In the latency-bound regime (per-forward
/// cost dominated by fixed MPS sync wait, not compute), while one server is
/// parked in `waitUntilCompleted` another can tensorize + submit — so N
/// shards ≈ N× eval capacity. Calling [`EvalHandle::evaluate`] in a loop
/// instead would serialize (each call blocks until its server replies before
/// the next is even submitted), which is why this handle exists.
///
/// Order-preserving: `result[i]` corresponds to `batch[i]`.
#[derive(Clone)]
pub struct ShardedEvalHandle {
    handles: Vec<EvalHandle>,
}

impl ShardedEvalHandle {
    pub fn new(handles: Vec<EvalHandle>) -> Self {
        assert!(!handles.is_empty(), "BUG: ShardedEvalHandle requires >= 1 handle");
        Self { handles }
    }

    pub fn len(&self) -> usize {
        self.handles.len()
    }

    pub fn evaluate(&self, batch: Vec<RawFeatures>) -> Vec<EvalResult> {
        let n = self.handles.len();
        if n == 1 {
            // Zero-overhead fast path: skip the partition + reassembly.
            return self.handles[0].evaluate(batch);
        }

        let total = batch.len();
        // RawFeatures isn't Clone, so build the bucket vecs without vec![..; n]
        // (which requires Vec<RawFeatures>: Clone).
        let mut buckets: Vec<Vec<RawFeatures>> = (0..n).map(|_| Vec::new()).collect();
        let mut orig_idx: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (i, f) in batch.into_iter().enumerate() {
            let s = (f.hash() % n as u64) as usize;
            orig_idx[s].push(i);
            buckets[s].push(f);
        }

        // Submit all non-empty shards (non-blocking), then collect all so the
        // shards run concurrently.
        let mut pending: Vec<(std_mpsc::Receiver<Vec<EvalResult>>, Vec<usize>)> =
            Vec::with_capacity(n);
        for (s, b) in buckets.into_iter().enumerate() {
            if b.is_empty() {
                continue;
            }
            let idxs = std::mem::take(&mut orig_idx[s]);
            pending.push((self.handles[s].submit(b), idxs));
        }

        let mut out: Vec<Option<EvalResult>> = (0..total).map(|_| None).collect();
        for (rx, idxs) in pending {
            let rows = EvalHandle::collect(rx);
            debug_assert_eq!(rows.len(), idxs.len());
            for (row, &orig) in rows.into_iter().zip(idxs.iter()) {
                out[orig] = Some(row);
            }
        }
        out.into_iter()
            .map(|o| o.expect("BUG: sharded evaluate left a row unfilled"))
            .collect()
    }
}

/// Either backend an MCTS agent can evaluate leaves through, unified behind
/// one call so `ZeroMctsAgent`/`GumbelMctsAgent` don't need to know which
/// they hold. `Server` is used by self-play (cross-game batching via
/// [`EvalServer`]); `Sharded` is the multi-server path (see
/// [`ShardedEvalHandle`]); `Inline` is used everywhere else (arena, UI
/// analysis, tests) where each caller owns its network/device outright;
/// `Dummy` is the no-inference benchmark path (see [`DummyEvalHandle`]).
#[derive(Clone)]
pub enum Evaluator {
    Server(EvalHandle),
    Sharded(ShardedEvalHandle),
    Inline(InlineEvalHandle),
    Dummy(DummyEvalHandle),
}

impl Evaluator {
    pub fn evaluate(&self, batch: Vec<RawFeatures>) -> Vec<EvalResult> {
        match self {
            Evaluator::Server(h) => h.evaluate(batch),
            Evaluator::Sharded(h) => h.evaluate(batch),
            Evaluator::Inline(h) => h.evaluate(batch),
            Evaluator::Dummy(h) => h.evaluate(batch),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::features::state_to_cpu_features;
    use crate::game::Game;

    fn test_network(device: &Device) -> PolyZeroNet {
        let varmap = candle_nn::VarMap::new();
        PolyZeroNet::new(candle_nn::VarBuilder::from_varmap(
            &varmap,
            candle_core::DType::F32,
            device,
        ))
        .unwrap()
    }

    #[test]
    fn eval_server_matches_inline_handle() {
        let device = Device::Cpu;
        let network = Arc::new(test_network(&device));

        let game = Game::default();
        let feat1 = state_to_cpu_features(&game.state, 1).unwrap();
        let feat2 = state_to_cpu_features(&game.state, 2).unwrap();

        let inline = InlineEvalHandle::new(network.clone());
        let inline_results = inline.evaluate(vec![
            RawFeatures {
                spatial: feat1.spatial.clone(),
                player: feat1.player.clone(),
            },
            RawFeatures {
                spatial: feat2.spatial.clone(),
                player: feat2.player.clone(),
            },
        ]);

        let (_server, handle) = EvalServer::start(BackendSpec::Candle(network), EvalServerConfig::default());
        let server_results = handle.evaluate(vec![
            RawFeatures {
                spatial: feat1.spatial,
                player: feat1.player,
            },
            RawFeatures {
                spatial: feat2.spatial,
                player: feat2.player,
            },
        ]);

        assert_eq!(inline_results.len(), 2);
        assert_eq!(server_results.len(), 2);
        for i in 0..2 {
            assert!((inline_results[i].0 - server_results[i].0).abs() < 1e-6);
            assert_eq!(
                inline_results[i].2.action_type,
                server_results[i].2.action_type
            );
            assert_eq!(
                inline_results[i].2.source_spatial,
                server_results[i].2.source_spatial
            );
            assert_eq!(
                inline_results[i].2.target_spatial,
                server_results[i].2.target_spatial
            );
            assert_eq!(
                inline_results[i].2.move_option,
                server_results[i].2.move_option
            );
        }
    }

    #[test]
    fn eval_server_coalesces_concurrent_requests() {
        let device = Device::Cpu;
        let network = Arc::new(test_network(&device));
        let game = Game::default();
        let feat = state_to_cpu_features(&game.state, 1).unwrap();

        let (_server, handle) = EvalServer::start(
            BackendSpec::Candle(network),
            EvalServerConfig {
                max_batch: 256,
                coalesce_timeout: Duration::from_millis(20),
                cache_capacity: None,
                pipeline_workers: 0,
            },
        );

        let threads: Vec<_> = (0..8)
            .map(|_| {
                let handle = handle.clone();
                let raw = RawFeatures {
                    spatial: feat.spatial.clone(),
                    player: feat.player.clone(),
                };
                std::thread::spawn(move || handle.evaluate(vec![raw]))
            })
            .collect();

        for t in threads {
            let result = t.join().unwrap();
            assert_eq!(result.len(), 1);
        }
    }

    #[test]
    fn eval_cache_serves_repeated_rows_without_gpu() {
        let device = Device::Cpu;
        let network = Arc::new(test_network(&device));
        let game = Game::default();
        let feat = state_to_cpu_features(&game.state, 1).unwrap();

        let (server, handle) = EvalServer::start(BackendSpec::Candle(network), EvalServerConfig::default());

        let raw = |f: &RawFeatures| RawFeatures {
            spatial: f.spatial.clone(),
            player: f.player.clone(),
        };

        // First call misses the cache and populates it.
        let first = handle.evaluate(vec![raw(&feat)]);
        assert_eq!(first.len(), 1);

        // Second identical call must be served from the cache: stats.cache_hits
        // increments and the returned row is byte-identical to the first.
        let second = handle.evaluate(vec![raw(&feat)]);
        assert_eq!(second.len(), 1);
        assert!((first[0].0 - second[0].0).abs() < 1e-6);
        assert_eq!(first[0].2.action_type, second[0].2.action_type);
        assert_eq!(first[0].2.source_spatial, second[0].2.source_spatial);
        assert_eq!(first[0].2.target_spatial, second[0].2.target_spatial);
        assert_eq!(first[0].2.move_option, second[0].2.move_option);

        let stats = server.stats();
        let hits = stats.cache_hits.load(Ordering::Relaxed);
        let misses = stats.cache_misses.load(Ordering::Relaxed);
        assert_eq!(misses, 1, "first call must be a cache miss");
        assert_eq!(hits, 1, "second call must be a cache hit");
    }
}
