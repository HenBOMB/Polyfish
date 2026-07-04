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
}

/// Tuning knobs for request coalescing.
#[derive(Clone, Copy, Debug)]
pub struct EvalServerConfig {
    /// Maximum number of leaves combined into one `forward_t` call.
    pub max_batch: usize,
    /// How long to wait for more requests to coalesce with the first one
    /// pending, before flushing whatever has arrived.
    pub coalesce_timeout: Duration,
}

impl Default for EvalServerConfig {
    fn default() -> Self {
        Self {
            max_batch: 256,
            coalesce_timeout: Duration::from_micros(1000),
        }
    }
}

impl EvalServer {
    /// Spawn the dedicated inference thread. `network` is moved onto that
    /// thread and never touched elsewhere.
    pub fn start(network: Arc<PolyZeroNet>, config: EvalServerConfig) -> (Self, EvalHandle) {
        let (sender, receiver) = std_mpsc::channel::<EvalRequest>();
        let stats = Arc::new(EvalServerStats::default());
        let thread_stats = stats.clone();

        let thread = std::thread::Builder::new()
            .name("eval-server".to_string())
            .spawn(move || run_eval_loop(network, receiver, config, thread_stats))
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
    network: Arc<PolyZeroNet>,
    receiver: std_mpsc::Receiver<EvalRequest>,
    config: EvalServerConfig,
    stats: Arc<EvalServerStats>,
) {
    let device = network.device();
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
        evaluate_batch(&network, &device, requests);
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

/// Tensorize every request's `RawFeatures` into one batch, run a single
/// `forward_t`, slice the results back to CPU floats, and reply to each
/// request's own reply channel with its slice in original order.
fn evaluate_batch(network: &PolyZeroNet, device: &Device, requests: Vec<EvalRequest>) {
    let per_request_len: Vec<usize> = requests.iter().map(|r| r.features.len()).collect();
    let batch_size: usize = per_request_len.iter().sum();

    if batch_size == 0 {
        for req in requests {
            let _ = req.respond_to.send(Vec::new());
        }
        return;
    }

    let mut spatial_flat = Vec::with_capacity(batch_size * RawFeatures::spatial_len());
    let mut player_flat = Vec::with_capacity(batch_size * RawFeatures::player_len());
    for req in &requests {
        for feat in &req.features {
            spatial_flat.extend_from_slice(&feat.spatial);
            player_flat.extend_from_slice(&feat.player);
        }
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
    let player_tensor = Tensor::from_vec(
        player_flat,
        (batch_size, RawFeatures::player_len()),
        device,
    )
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

    debug_assert_eq!(values.len(), batch_size);
    debug_assert_eq!(policy_rows.len(), batch_size);

    // Scatter contiguous slices back to each request in original order.
    let mut offset = 0;
    for (req, len) in requests.into_iter().zip(per_request_len.into_iter()) {
        let results: Vec<EvalResult> = (offset..offset + len)
            .map(|i| (values[i], policy_rows[i].clone()))
            .collect();
        offset += len;
        // Ignore send errors: the requesting thread may have given up.
        let _ = req.respond_to.send(results);
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

        let (_server, handle) = EvalServer::start(network, EvalServerConfig::default());
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
            network,
            EvalServerConfig {
                max_batch: 256,
                coalesce_timeout: Duration::from_millis(20),
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
}
