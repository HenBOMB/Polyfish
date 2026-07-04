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
use std::sync::Arc;
use std::sync::mpsc as std_mpsc;
use std::time::{Duration, Instant};

/// One evaluated leaf: NN value estimate + decomposed policy logits, all as
/// owned CPU floats.
pub type EvalResult = (f32, RawPolicyOutput);

struct EvalRequest {
    features: Vec<RawFeatures>,
    respond_to: std_mpsc::Sender<Vec<EvalResult>>,
}

/// Handle to a running [`EvalServer`]. Cheap to clone; each clone shares the
/// same request channel.
#[derive(Clone)]
pub struct EvalHandle {
    sender: std_mpsc::Sender<EvalRequest>,
}

impl EvalHandle {
    /// Evaluate a batch of leaves. Blocks the calling thread until the
    /// server has coalesced this request with others (if any), run one
    /// `forward_t`, and sliced out this caller's rows. Order-preserving:
    /// `result[i]` corresponds to `batch[i]`.
    pub fn evaluate(&self, batch: Vec<RawFeatures>) -> Vec<EvalResult> {
        let (tx, rx) = std_mpsc::channel();
        let request = EvalRequest {
            features: batch,
            respond_to: tx,
        };
        if self.sender.send(request).is_err() {
            panic!("BUG: EvalServer thread has shut down while a handle is still in use");
        }
        rx.recv()
            .expect("BUG: EvalServer dropped the response channel without replying")
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
    pub busy_us: AtomicU64,
    /// Leaf rows served from the eval cache without hitting the GPU.
    pub cache_hits: AtomicU64,
    /// Leaf rows that missed the cache and required a GPU `forward_t` row.
    pub cache_misses: AtomicU64,
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
}

impl InferenceBackend {
    /// Run one batched forward over `feats`, returning per-row value + policy
    /// as plain CPU floats. This is the only place a device tensor exists.
    fn forward(&self, feats: &[RawFeatures]) -> (Vec<f32>, Vec<RawPolicyOutput>) {
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
                let policy_rows = policy_out
                    .to_raw_rows()
                    .expect("BUG: failed to read policy batch to CPU");
                (values, policy_rows)
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
                net.forward_batch(&spatial_flat, &player_flat, batch_size)
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
}

fn run_eval_loop(
    spec: BackendSpec,
    receiver: std_mpsc::Receiver<EvalRequest>,
    config: EvalServerConfig,
    stats: Arc<EvalServerStats>,
) {
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
        let (served, misses) = evaluate_batch(&backend, requests, cache.as_mut());
        stats.forwards.fetch_add(1, Ordering::Relaxed);
        stats.rows.fetch_add(total_items as u64, Ordering::Relaxed);
        stats
            .max_batch
            .fetch_max(total_items as u64, Ordering::Relaxed);
        stats
            .busy_us
            .fetch_add(busy_start.elapsed().as_micros() as u64, Ordering::Relaxed);
        stats.cache_hits.fetch_add(served.served_from_cache, Ordering::Relaxed);
        stats
            .cache_misses
            .fetch_add(misses, Ordering::Relaxed);
    }
}

/// Per-call cache counters returned by `evaluate_batch`.
struct CacheTally {
    served_from_cache: u64,
}

/// Partition the coalesced batch by eval-cache membership: rows whose
/// `RawFeatures` hash to a cached entry are replied to immediately without
/// touching the GPU; the remaining misses are tensorized into one
/// `forward_t`, read back, inserted into the cache, and scattered back to
/// each request in original order.
///
/// Returns `(CacheTally, misses)` where `misses` is the number of rows that
/// required a GPU row (used for stats), and replies are sent on each
/// request's `respond_to` channel.
fn evaluate_batch(
    backend: &InferenceBackend,
    requests: Vec<EvalRequest>,
    mut cache: Option<&mut lru::LruCache<u64, EvalResult>>,
) -> (CacheTally, u64) {
    let per_request_len: Vec<usize> = requests.iter().map(|r| r.features.len()).collect();
    let total_rows: usize = per_request_len.iter().sum();

    if total_rows == 0 {
        for req in requests {
            let _ = req.respond_to.send(Vec::new());
        }
        return (CacheTally { served_from_cache: 0 }, 0);
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
        for feat in &req.features {
            let hash = feat.hash();
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
        let (values, policy_rows) = backend.forward(&miss_features);

        debug_assert_eq!(values.len(), batch_size);
        debug_assert_eq!(policy_rows.len(), batch_size);

        for (i, &flat_idx) in miss_slots.iter().enumerate() {
            let result: EvalResult = (values[i], policy_rows[i].clone());
            if let Some(c) = cache.as_mut() {
                c.put(miss_hashes[i], result.clone());
            }
            row_results[flat_idx] = Some(result);
        }
    }

    // Scatter contiguous slices back to each request in original order.
    let mut offset = 0;
    for (req, len) in requests.into_iter().zip(per_request_len.into_iter()) {
        let results: Vec<EvalResult> = (offset..offset + len)
            .map(|i| row_results[i].take().expect("BUG: every row must be resolved"))
            .collect();
        offset += len;
        // Ignore send errors: the requesting thread may have given up.
        let _ = req.respond_to.send(results);
    }

    (CacheTally { served_from_cache }, misses)
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
        let policy_rows = policy_out
            .to_raw_rows()
            .expect("BUG: failed to read policy batch to CPU");

        values.into_iter().zip(policy_rows).collect()
    }
}

/// Either backend an MCTS agent can evaluate leaves through, unified behind
/// one call so `ZeroMctsAgent`/`GumbelMctsAgent` don't need to know which
/// they hold. `Server` is used by self-play (cross-game batching via
/// [`EvalServer`]); `Inline` is used everywhere else (arena, UI analysis,
/// tests) where each caller owns its network/device outright.
#[derive(Clone)]
pub enum Evaluator {
    Server(EvalHandle),
    Inline(InlineEvalHandle),
}

impl Evaluator {
    pub fn evaluate(&self, batch: Vec<RawFeatures>) -> Vec<EvalResult> {
        match self {
            Evaluator::Server(h) => h.evaluate(batch),
            Evaluator::Inline(h) => h.evaluate(batch),
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
                inline_results[i].1.action_type,
                server_results[i].1.action_type
            );
            assert_eq!(
                inline_results[i].1.source_spatial,
                server_results[i].1.source_spatial
            );
            assert_eq!(
                inline_results[i].1.target_spatial,
                server_results[i].1.target_spatial
            );
            assert_eq!(
                inline_results[i].1.move_option,
                server_results[i].1.move_option
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
        assert_eq!(first[0].1.action_type, second[0].1.action_type);
        assert_eq!(first[0].1.source_spatial, second[0].1.source_spatial);
        assert_eq!(first[0].1.target_spatial, second[0].1.target_spatial);
        assert_eq!(first[0].1.move_option, second[0].1.move_option);

        let stats = server.stats();
        let hits = stats.cache_hits.load(Ordering::Relaxed);
        let misses = stats.cache_misses.load(Ordering::Relaxed);
        assert_eq!(misses, 1, "first call must be a cache miss");
        assert_eq!(hits, 1, "second call must be a cache hit");
    }
}
