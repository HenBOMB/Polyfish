//! Shared NN inference-backend selection and shard spawning, used by both
//! `self_play` and `arena` so there is exactly one implementation of "pick a
//! backend, build N `BackendSpec`s, spawn N `EvalServer`s, wrap them as a
//! `ShardedEvalHandle`".
//!
//! ## Candle device-isolation contract
//! candle's Metal backend corrupts if more than one thread encodes ops
//! against the *same* `Device` (see `eval_server.rs`'s module doc). Under
//! [`EvalBackendKind::Candle`], [`build_two_player_evaluators`] spawns one
//! `EvalServer` thread per shard, each holding the `candle_net` passed in for
//! that player. When the two players use **different** model files (so two
//! independent shard sets run concurrently), the caller MUST load each
//! player's candle net on its own freshly-obtained `Device`
//! (`Device::metal_if_available`/`cuda_if_available` allocate a fresh
//! command queue on every call — same physical GPU, distinct device
//! handles). When both players share one model, share one `Device` too: only
//! one shard set's threads ever touch it. This module cannot enforce that
//! itself — it only sees pre-loaded `Arc<PolyZeroNet>`s — so it's on each
//! `main()` call site.

use std::sync::Arc;
use anyhow::{bail, Context, Result};
use candle_core::{utils, Device};
use crate::ai::eval_server::{BackendSpec, EvalHandle, EvalServer, EvalServerConfig, Evaluator, ShardedEvalHandle,};
use crate::ai::network::PolyZeroNet;

/// Select the compute device: Metal (macOS) > CUDA (NVIDIA) > CPU, unless
/// overridden via `POLYFISH_DEVICE` (`cpu`|`metal`|`cuda`). Each call
/// allocates a fresh device handle (own command queue), even for the same
/// physical GPU — callers that need two isolated devices just call this
/// twice.
pub fn select_device() -> Result<Device> {
    let requested = std::env::var("POLYFISH_DEVICE")
        .unwrap_or_else(|_| "auto".to_owned())
        .trim()
        .to_ascii_lowercase();

    match requested.as_str() {
        "" | "auto" => select_device_automatically(),

        "cpu" => Ok(Device::Cpu),

        "cuda" => select_cuda().context(
            "POLYFISH_DEVICE=cuda was requested, but CUDA is unavailable",
        ),

        "metal" => select_metal().context(
            "POLYFISH_DEVICE=metal was requested, but Metal is unavailable",
        ),

        other => bail!(
            "invalid POLYFISH_DEVICE={other:?}; expected one of: auto, cpu, cuda, metal"
        ),
    }
}

fn select_device_automatically() -> Result<Device> {
    // Prefer CUDA whenever this binary includes CUDA support and a CUDA
    // device is available.
    #[cfg(feature = "cuda")]
    if utils::cuda_is_available() {
        return Device::new_cuda(0).context("CUDA is available, but device 0 could not be initialized");
    }

    // The `metal` Cargo feature should enable Candle's Metal backend.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    if utils::metal_is_available() {
        return Device::new_metal(0).context("Metal is available, but device 0 could not be initialized");
    }

    Ok(Device::Cpu)
}

fn select_cuda() -> Result<Device> {
    #[cfg(feature = "cuda")]
    {
        if !utils::cuda_is_available() {
            bail!("CUDA was requested, but no usable CUDA device is available");
        }

        Device::new_cuda(0).context("failed to initialize CUDA device 0")
    }

    #[cfg(not(feature = "cuda"))]
    {
        bail!("CUDA was requested, but this binary was built without the `cuda` feature");
    }
}

fn select_metal() -> Result<Device> {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        if !utils::metal_is_available() {
            bail!("Metal was requested, but no usable Metal device is available");
        }

        Device::new_metal(0).context("failed to initialize Metal device 0")
    }

    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    {
        bail!("Metal was requested, but this binary was built without Metal support");
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EvalBackendKind {
    Candle,
    Tch,
    Metal,
}

/// Resolve `--eval-backend` (`"candle"`/`"tch"`/`"metal"`/`""` for auto).
/// Auto picks metal if `metal-eval` is compiled in, else tch if `tch-eval`
/// is, else candle.
pub fn resolve_eval_backend_kind(arg: &str) -> anyhow::Result<EvalBackendKind> {
    match arg {
        "tch" => {
            if !cfg!(feature = "tch-eval") {
                anyhow::bail!("--eval-backend tch requires building with --features tch-eval");
            }
            Ok(EvalBackendKind::Tch)
        }
        "metal" => {
            if !cfg!(feature = "metal-eval") {
                anyhow::bail!("--eval-backend metal requires building with --features metal-eval");
            }
            Ok(EvalBackendKind::Metal)
        }
        "candle" => Ok(EvalBackendKind::Candle),
        "" => {
            if cfg!(feature = "metal-eval") {
                Ok(EvalBackendKind::Metal)
            } else if cfg!(feature = "tch-eval") {
                Ok(EvalBackendKind::Tch)
            } else {
                Ok(EvalBackendKind::Candle)
            }
        }
        other => anyhow::bail!("unknown --eval-backend {other:?} (want candle|tch|metal)"),
    }
}

/// Resolve shard count from `--eval-servers` (`0` = auto: 3 on metal, 1
/// otherwise), validating candle's hard >1-shard prohibition (Metal
/// corrupts when >1 thread encodes on the same device).
pub fn resolve_eval_servers(kind: EvalBackendKind, requested: usize) -> anyhow::Result<usize> {
    match requested {
        0 => Ok(if kind == EvalBackendKind::Metal { 3 } else { 1 }),
        n => {
            if n > 1 && kind == EvalBackendKind::Candle {
                anyhow::bail!(
                    "--eval-servers > 1 requires --eval-backend tch or metal \
                     (candle Metal corrupts when >1 thread encodes on the same device). \
                     Resolved backend: candle; compiled features: metal-eval={}, tch-eval={}. \
                     If both are false, this binary was built without `--features apple` \
                     (e.g. replaced by a concurrent `cargo build`/`cargo test`) — rebuild it.",
                    cfg!(feature = "metal-eval"),
                    cfg!(feature = "tch-eval")
                );
            }
            Ok(n)
        }
    }
}

/// Divide `total_cache` across `shard_count` so aggregate resident cache
/// stays ~constant regardless of shard count (`0` disables caching).
pub fn split_cache_capacity(total_cache: usize, shard_count: usize) -> Option<usize> {
    if total_cache == 0 {
        None
    } else {
        Some(total_cache / shard_count.max(1))
    }
}

/// Build `n` [`BackendSpec`]s for one player. For tch: every shard gets its
/// own `BackendSpec::Tch` on the shared MPS device; each shard's thread
/// loads its own `TchPolyZeroNet` (duplicated weights, a few MB). For metal:
/// every shard gets its own `BackendSpec::MetalMps`, each shard's thread
/// loads its own `MetalPolyZeroNet` and owns its own `MTLCommandQueue`. For
/// candle: `n` clones of `candle_net` — see the module doc for the device-
/// isolation contract this relies on.
pub fn build_backend_specs(
    kind: EvalBackendKind,
    n: usize,
    model_path: &str,
    candle_net: &Arc<PolyZeroNet>,
) -> Vec<BackendSpec> {
    // Only read by the tch/metal branches below; in builds with neither
    // feature it's unused. Touch it so the fn compiles cleanly either way.
    let _ = model_path;
    match kind {
        EvalBackendKind::Tch => {
            #[cfg(feature = "tch-eval")]
            {
                let dev = if tch::utils::has_mps() {
                    tch::Device::Mps
                } else {
                    tch::Device::Cpu
                };
                return (0..n)
                    .map(|_| BackendSpec::Tch {
                        model_path: model_path.to_string(),
                        device: dev,
                    })
                    .collect();
            }
            #[cfg(not(feature = "tch-eval"))]
            unreachable!("EvalBackendKind::Tch guarded by cfg above");
        }
        EvalBackendKind::Metal => {
            #[cfg(feature = "metal-eval")]
            {
                return (0..n)
                    .map(|_| BackendSpec::MetalMps {
                        model_path: model_path.to_string(),
                    })
                    .collect();
            }
            #[cfg(not(feature = "metal-eval"))]
            unreachable!("EvalBackendKind::Metal guarded by cfg above");
        }
        EvalBackendKind::Candle => (0..n).map(|_| BackendSpec::Candle(candle_net.clone())).collect(),
    }
}

/// Spawn one [`EvalServer`] per spec. Returns the servers (caller owns
/// shutdown — see the module doc in `eval_server.rs`) and the handles (to
/// wrap in a [`ShardedEvalHandle`]).
pub fn spawn_eval_shards(specs: Vec<BackendSpec>, config: EvalServerConfig) -> (Vec<EvalServer>, Vec<EvalHandle>) {
    let mut servers = Vec::with_capacity(specs.len());
    let mut handles = Vec::with_capacity(specs.len());
    for spec in specs {
        let (srv, h) = EvalServer::start(spec, config);
        servers.push(srv);
        handles.push(h);
    }
    (servers, handles)
}

/// One player's backend inputs: the model file to load shards from, plus an
/// already-loaded candle net (only read when `kind == Candle`; ignored by
/// tch/metal, which load their own weights from `model_path` on the eval-
/// server thread).
pub struct PlayerBackend<'a> {
    pub model_path: &'a str,
    pub candle_net: &'a Arc<PolyZeroNet>,
}

/// Build the full eval-serving setup for a two-player run. If `p2` is
/// `None`, player 2 shares player 1's shard set (same weights, one set of
/// `EvalServer` threads) — mirrors self_play's no-opponent path and arena's
/// same-model optimization. If `Some`, an independent shard set is spawned
/// for player 2.
///
/// Returns `(p1_servers, p2_servers_or_none, eval1, eval2)` — the caller
/// owns shutdown ordering: drop `eval1`/`eval2` (and any clones/borrows of
/// them) first, then call `.shutdown()` on every server. See
/// `eval_server.rs`'s `EvalServer::shutdown` doc for why order matters on
/// the tch/libtorch backend.
pub fn build_two_player_evaluators(
    kind: EvalBackendKind,
    shard_count: usize,
    config: EvalServerConfig,
    p1: PlayerBackend,
    p2: Option<PlayerBackend>,
) -> (Vec<EvalServer>, Option<Vec<EvalServer>>, Evaluator, Evaluator) {
    let p1_specs = build_backend_specs(kind, shard_count, p1.model_path, p1.candle_net);
    let (p1_servers, p1_handles) = spawn_eval_shards(p1_specs, config);

    let (p2_servers, p2_handles) = match p2 {
        Some(p2) => {
            let p2_specs = build_backend_specs(kind, shard_count, p2.model_path, p2.candle_net);
            let (servers, handles) = spawn_eval_shards(p2_specs, config);
            (Some(servers), handles)
        }
        None => (None, p1_handles.clone()),
    };

    let eval1 = Evaluator::Sharded(ShardedEvalHandle::new(p1_handles));
    let eval2 = Evaluator::Sharded(ShardedEvalHandle::new(p2_handles));
    (p1_servers, p2_servers, eval1, eval2)
}
