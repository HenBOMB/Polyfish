//! Rust half of the candle<->PyTorch network parity test.
//!
//! Loads model.safetensors, runs a deterministic synthetic input through the
//! candle PolyZeroNet, and writes inputs + all head outputs to a safetensors
//! file. `parity_check.py` replays the same input through the PyTorch net and
//! compares. Run both after any architecture change.

use candle_core::{DType, Device, Tensor};
use polyfish::ai::features::{MAP_SIZE, NUM_CHANNELS, RawFeatures};
use polyfish::ai::network::PolyZeroNet;
use std::collections::HashMap;

/// Deterministic pseudo-random values in [0, 1), identical across platforms.
fn synth(len: usize, salt: u64) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = (i as u64).wrapping_add(salt).wrapping_mul(2654435761);
            ((x >> 8) % 10_000) as f32 / 10_000.0
        })
        .collect()
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).map(String::as_str).unwrap_or("model.safetensors");
    let out_path = args.get(2).map(String::as_str).unwrap_or("parity_dump.safetensors");

    let device = Device::Cpu;
    let vs = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)?
    };
    let net = PolyZeroNet::new(vs)?;

    let spatial = synth(NUM_CHANNELS * MAP_SIZE * MAP_SIZE, 1);
    let player = synth(RawFeatures::PLAYER_STATE_DIM, 2);

    let map_t = Tensor::from_vec(
        spatial.clone(),
        (1, NUM_CHANNELS, MAP_SIZE, MAP_SIZE),
        &device,
    )?;
    let player_t = Tensor::from_vec(player.clone(), (1, RawFeatures::PLAYER_STATE_DIM), &device)?;

    let (policy, value) = net.forward(&map_t, &player_t)?;

    let mut tensors: HashMap<String, Tensor> = HashMap::from([
        ("input_spatial".to_string(), map_t),
        ("input_player".to_string(), player_t),
        ("action_type".to_string(), policy.action_type),
        ("source_spatial".to_string(), policy.source_spatial),
        ("target_spatial".to_string(), policy.target_spatial),
        ("move_option".to_string(), policy.move_option),
        ("win_value".to_string(), value.win_value),
        ("progress_value".to_string(), value.progress_value),
    ]);
    if let Some(own) = value.ownership_value {
        tensors.insert("ownership_value".to_string(), own);
    }

    candle_core::safetensors::save(&tensors, out_path)?;
    println!(
        "Dumped parity tensors for {} -> {} (channels={}, player_dim={}, map={})",
        model_path,
        out_path,
        NUM_CHANNELS,
        RawFeatures::PLAYER_STATE_DIM,
        MAP_SIZE
    );
    Ok(())
}
