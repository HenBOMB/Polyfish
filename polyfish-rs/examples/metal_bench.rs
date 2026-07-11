//! Throughput microbenchmark for the MPSGraph eval path (and tch for
//! reference). Isolates the eval-server's forward cost from the actor side
//! so the fixed-vs-per-row cost structure is measurable directly:
//!
//!   cost(batch) = fixed_overhead + per_row * batch
//!
//! Run (with the tch build env):
//!   cargo run --release --features "tch-eval metal-eval" --example metal_bench
//!
//! Phases measured separately per batch size, steady-state (post-compile):
//!   - upload:  TensorData::from_f32_slice for spatial+player
//!   - total:   full forward_batch (upload + run + readback + slicing)
//! The `run+read` column is total - upload. GPU compute for this net is
//! known-tiny (~23us/CB from the Metal trace), so run+read is dominated by
//! encode/sync/output-alloc overhead — the Stage 2 target.

use polyfish::ai::features::{state_to_cpu_features, RawFeatures, MAP_SIZE, NUM_CHANNELS};
use polyfish::ai::metal_network::MetalPolyZeroNet;
use polyfish::ai::tch_network::TchPolyZeroNet;
use polyfish::game::Game;
use std::time::Instant;

const MODEL: &str = "model.safetensors";
const WARMUP: usize = 8;
const ITERS: usize = 60;

fn real_features() -> Vec<RawFeatures> {
    let mut feats = Vec::new();
    for seed in 1i64..=8 {
        let mut game = Game::new();
        game.state = polyfish::mapgen::generate(polyfish::mapgen::MapGenSettings {
            size: polyfish::types::MapSize::Tiny,
            map_type: polyfish::types::MapType::Drylands,
            tribes: vec![
                polyfish::types::TribeType::Imperius,
                polyfish::types::TribeType::Bardur,
            ],
            seed,
            ..Default::default()
        });
        game.post_load();
        for pov in [1, 2] {
            feats.push(state_to_cpu_features(&game.state, pov).unwrap());
        }
    }
    feats
}

/// Tile the base features up to `batch` rows, flattened.
fn flat_batch(base: &[RawFeatures], batch: usize) -> (Vec<f32>, Vec<f32>) {
    let mut spatial = Vec::with_capacity(batch * NUM_CHANNELS * MAP_SIZE * MAP_SIZE);
    let mut player = Vec::with_capacity(batch * 16);
    for i in 0..batch {
        let f = &base[i % base.len()];
        spatial.extend_from_slice(&f.spatial);
        player.extend_from_slice(&f.player);
    }
    (spatial, player)
}

fn main() {
    let base = real_features();
    let batches = [8usize, 16, 32, 64, 128, 256];

    println!("== metal (MPSGraph, cached executable) ==");
    let net = MetalPolyZeroNet::load(MODEL).unwrap();
    println!(
        "{:>6} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "batch", "total ms", "upload ms", "run+read ms", "us/row", "rows/s"
    );
    for &b in &batches {
        let (spatial, player) = flat_batch(&base, b);

        // Warmup (first call compiles the executable for this batch size).
        for _ in 0..WARMUP {
            let _ = net.forward_batch(&spatial, &player, b);
        }

        // Upload phase alone: create the input TensorDatas and drop them.
        let up_start = Instant::now();
        for _ in 0..ITERS {
            let _sd = net.make_tensor_data(&spatial, &[b, NUM_CHANNELS, MAP_SIZE, MAP_SIZE]);
            let _pd = net.make_tensor_data(&player, &[b, 16]);
        }
        let upload_ms = up_start.elapsed().as_secs_f64() * 1000.0 / ITERS as f64;

        let start = Instant::now();
        for _ in 0..ITERS {
            let _ = net.forward_batch(&spatial, &player, b);
        }
        let total_ms = start.elapsed().as_secs_f64() * 1000.0 / ITERS as f64;

        println!(
            "{:>6} {:>12.3} {:>12.3} {:>12.3} {:>12.1} {:>12.0}",
            b,
            total_ms,
            upload_ms,
            total_ms - upload_ms,
            total_ms * 1000.0 / b as f64,
            b as f64 / (total_ms / 1000.0),
        );
    }

    // GPU-capacity probe: N independent nets (own MTLCommandQueue each) on N
    // threads, hammering the same batch size concurrently. If aggregate
    // rows/s scales ~Nx over single-thread, the GPU has headroom and the
    // single-thread wall is synchronous dispatch; if it stays ~1x, the GPU
    // itself is saturated and no amount of pipelining/sharding helps.
    println!("\n== metal concurrency probe (independent nets+queues, batch 128) ==");
    let single_rows_s = {
        let (spatial, player) = flat_batch(&base, 128);
        for _ in 0..WARMUP {
            let _ = net.forward_batch(&spatial, &player, 128);
        }
        let start = Instant::now();
        for _ in 0..ITERS {
            let _ = net.forward_batch(&spatial, &player, 128);
        }
        (128 * ITERS) as f64 / start.elapsed().as_secs_f64()
    };
    println!("{:>8} {:>12} {:>10}", "threads", "agg rows/s", "scaling");
    println!("{:>8} {:>12.0} {:>10.2}", 1, single_rows_s, 1.0);
    for n in [2usize, 3, 4] {
        let start = Instant::now();
        let handles: Vec<_> = (0..n)
            .map(|_| {
                std::thread::spawn(move || {
                    let base = real_features();
                    let net = MetalPolyZeroNet::load(MODEL).unwrap();
                    let (spatial, player) = flat_batch(&base, 128);
                    for _ in 0..WARMUP {
                        let _ = net.forward_batch(&spatial, &player, 128);
                    }
                    // Synchronize roughly: everyone warms up, then measures.
                    let t = Instant::now();
                    for _ in 0..ITERS {
                        let _ = net.forward_batch(&spatial, &player, 128);
                    }
                    (128 * ITERS) as f64 / t.elapsed().as_secs_f64()
                })
            })
            .collect();
        let per_thread: Vec<f64> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        let agg: f64 = per_thread.iter().sum();
        let _total = start.elapsed();
        println!(
            "{:>8} {:>12.0} {:>10.2}   (per-thread: {})",
            n,
            agg,
            agg / single_rows_s,
            per_thread
                .iter()
                .map(|r| format!("{r:.0}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    println!("\n== tch (libtorch/MPS) ==");
    let dev = if tch::utils::has_mps() {
        tch::Device::Mps
    } else {
        tch::Device::Cpu
    };
    let tch_net = TchPolyZeroNet::load(MODEL, dev).unwrap();
    println!(
        "{:>6} {:>12} {:>12} {:>12}",
        "batch", "total ms", "us/row", "rows/s"
    );
    for &b in &batches {
        let (spatial, player) = flat_batch(&base, b);
        for _ in 0..WARMUP {
            let _ = tch_net.forward_batch(&spatial, &player, b);
        }
        let start = Instant::now();
        for _ in 0..ITERS {
            let _ = tch_net.forward_batch(&spatial, &player, b);
        }
        let total_ms = start.elapsed().as_secs_f64() * 1000.0 / ITERS as f64;
        println!(
            "{:>6} {:>12.3} {:>12.1} {:>12.0}",
            b,
            total_ms,
            total_ms * 1000.0 / b as f64,
            b as f64 / (total_ms / 1000.0),
        );
    }
}
