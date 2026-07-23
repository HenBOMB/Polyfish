//! Minimal isolated benchmark of the eval-server hot path: batched
//! `forward_t` + full CPU readback, on a single thread with no MCTS, no
//! actors, no channels. Used to decide whether the ~50ms/forward observed in
//! self_play is inherent to candle+Metal at this network shape or an
//! emergent property of the self_play environment.
//!
//! Run: cargo run --release --example bench_forward -- [batch] [iters]

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use polyfish::ai::features::{MAP_SIZE, NUM_CHANNELS};
use polyfish::ai::network::PolyZeroNet;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let batch: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(128);
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(60);

    // BENCH_DEVICE=cpu forces the CPU backend (Accelerate BLAS when built with
    // --features accelerate); =metal forces Metal; unset auto-selects Metal.
    let device = match std::env::var("BENCH_DEVICE").as_deref() {
        Ok("cpu") => Device::Cpu,
        Ok("metal") => Device::new_metal(0)?,
        _ => Device::metal_if_available(0).unwrap_or(Device::Cpu),
    };
    println!(
        "device={:?} batch={} iters={} compute_per_buffer={:?}",
        device,
        batch,
        iters,
        std::env::var("CANDLE_METAL_COMPUTE_PER_BUFFER").ok()
    );

    let network = PolyZeroNet::new(VarBuilder::zeros(DType::F32, &device))?;

    let spatial_len = NUM_CHANNELS * MAP_SIZE * MAP_SIZE;
    let player_len = 10;

    let mut times_ms: Vec<f64> = Vec::with_capacity(iters);
    for i in 0..iters {
        // Fresh host data every iter (mirrors eval_server tensorization).
        let spatial_flat: Vec<f32> = (0..batch * spatial_len)
            .map(|j| ((i + j) % 97) as f32 / 97.0)
            .collect();
        let player_flat: Vec<f32> = (0..batch * player_len)
            .map(|j| ((i * 31 + j) % 89) as f32 / 89.0)
            .collect();

        let t0 = Instant::now();
        let spatial =
            Tensor::from_vec(spatial_flat, (batch, NUM_CHANNELS, MAP_SIZE, MAP_SIZE), &device)?;
        let player = Tensor::from_vec(player_flat, (batch, player_len), &device)?;
        let t_tensorize = t0.elapsed();

        let (policy_out, value_out) = network.forward_t(&spatial, &player, false)?;
        let t_encode = t0.elapsed();

        // Full readback, same calls as eval_server::evaluate_batch. The first
        // to_vec1 is the sync point that waits for the GPU.
        let values = value_out.win_value.flatten_all()?.to_vec1::<f32>()?;
        let t_sync = t0.elapsed();
        let rows = policy_out.to_raw_rows()?;
        let dt = t0.elapsed().as_secs_f64() * 1e3;

        assert_eq!(values.len(), batch);
        assert_eq!(rows.len(), batch);
        if i >= 10 && i < 15 {
            println!(
                "  iter {i}: tensorize={:.2}ms encode={:.2}ms sync(value)={:.2}ms policy_read={:.2}ms total={:.2}ms",
                t_tensorize.as_secs_f64() * 1e3,
                (t_encode - t_tensorize).as_secs_f64() * 1e3,
                (t_sync - t_encode).as_secs_f64() * 1e3,
                dt - t_sync.as_secs_f64() * 1e3,
                dt
            );
        }
        times_ms.push(dt);
    }

    // Skip warmup iters (shader compilation etc.), report the steady state.
    let steady = &times_ms[times_ms.len().min(10)..];
    let mut sorted: Vec<f64> = steady.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let sum: f64 = steady.iter().sum();
    println!(
        "warmup first 3: {:.1} {:.1} {:.1} ms",
        times_ms[0], times_ms[1], times_ms[2]
    );
    println!(
        "steady ({} iters): mean={:.2}ms median={:.2}ms min={:.2}ms p90={:.2}ms max={:.2}ms",
        steady.len(),
        sum / steady.len() as f64,
        sorted[sorted.len() / 2],
        sorted[0],
        sorted[sorted.len() * 9 / 10],
        sorted[sorted.len() - 1]
    );
    println!(
        "throughput: {:.1} forwards/s, {:.0} rows/s",
        1e3 * steady.len() as f64 / sum,
        1e3 * (steady.len() * batch) as f64 / sum
    );
    Ok(())
}
