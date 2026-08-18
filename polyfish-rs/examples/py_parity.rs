//! Rust↔Python forward parity: emit the candle-CPU forward as JSON so
//! `scripts/py_parity.py` can compare it against train.py's PyTorch
//! definition on the same weights and the same input.
//!
//! This is the check audit T1 calls the highest-value missing test in the repo.
//! The Rust and Python networks read and write the same `model.safetensors`, so
//! they must stay byte-compatible, and until now nothing loaded a
//! Python-produced checkpoint into the Rust network and compared outputs.
//! `tch_parity.rs` and `metal_parity.rs` cover candle-vs-tch and tch-vs-MPS,
//! both of which need libtorch and (for the latter) Apple hardware — neither
//! covers the split that actually matters, and neither runs on a plain CI box.
//! This one needs only the default candle-CPU build and a CPU torch wheel.
//!
//! Run:
//!   cargo run --no-default-features --example py_parity -- model.safetensors > /tmp/rust.json
//!   .venv/bin/python3 scripts/py_parity.py /tmp/rust.json model.safetensors
//!
//! The input is generated from a closed form rather than read from a file, so
//! both languages construct bit-identical inputs without a shared artifact that
//! could drift or go missing.

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use polyfish::ai::features::{MAP_SIZE, NUM_CHANNELS, RawFeatures};
use polyfish::ai::network::PolyZeroNet;

const BATCH: usize = 4;

/// Deterministic stand-in for a real encoded position. Mirrored exactly in
/// `scripts/py_parity.py`; keep the two in step or the comparison is
/// meaningless rather than merely failing.
///
/// The index is wrapped before scaling so the argument to sin/cos stays small.
/// Unwrapped, the last of ~69k elements would be sin(1168), where one ulp of
/// difference in the product becomes ~1e-4 of difference in the result — the
/// harness would then be measuring its own input, not the two networks.
fn spatial_value(i: usize) -> f32 {
    (((i % 1009) as f32) * 0.017).sin()
}

fn player_value(i: usize) -> f32 {
    (((i % 251) as f32) * 0.31).cos()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::args().nth(1).unwrap_or_else(|| "model.safetensors".into());
    let device = Device::Cpu;

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model.as_str()], DType::F32, &device)? };
    let net = PolyZeroNet::new(vb)?;

    let hw = MAP_SIZE * MAP_SIZE;
    let spatial: Vec<f32> = (0..BATCH * NUM_CHANNELS * hw).map(spatial_value).collect();
    let player: Vec<f32> = (0..BATCH * RawFeatures::PLAYER_STATE_DIM)
        .map(player_value)
        .collect();

    let map_input = Tensor::from_vec(spatial, (BATCH, NUM_CHANNELS, MAP_SIZE, MAP_SIZE), &device)?;
    let player_input =
        Tensor::from_vec(player, (BATCH, RawFeatures::PLAYER_STATE_DIM), &device)?;

    let (policy, value) = net.forward(&map_input, &player_input)?;

    // Checksums so the Python side can prove it built the identical input
    // before it blames the networks for a difference.
    let spatial_check: f64 = (0..BATCH * NUM_CHANNELS * hw)
        .map(|i| spatial_value(i) as f64)
        .sum();
    let player_check: f64 = (0..BATCH * RawFeatures::PLAYER_STATE_DIM)
        .map(|i| player_value(i) as f64)
        .sum();

    let out = serde_json::json!({
        "batch": BATCH,
        "spatial_check": spatial_check,
        "player_check": player_check,
        "spatial_channels": NUM_CHANNELS,
        "player_state_dim": RawFeatures::PLAYER_STATE_DIM,
        "map_size": MAP_SIZE,
        "win": value.win_value.flatten_all()?.to_vec1::<f32>()?,
        "progress": value.progress_value.flatten_all()?.to_vec1::<f32>()?,
        "action_type": policy.action_type.flatten_all()?.to_vec1::<f32>()?,
        "source_spatial": policy.source_spatial.flatten_all()?.to_vec1::<f32>()?,
        "target_spatial": policy.target_spatial.flatten_all()?.to_vec1::<f32>()?,
        "move_option": policy.move_option.flatten_all()?.to_vec1::<f32>()?,
    });
    println!("{out}");
    Ok(())
}
