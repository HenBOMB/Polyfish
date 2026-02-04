use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Optimizer, VarBuilder, VarMap};
use polyfish::ai::network::PolyZeroNet;
use std::fs;
use std::path::Path;

const BATCH_SIZE: usize = 64;
const LEARNING_RATE: f64 = 0.001;
const EPOCHS: usize = 10;

fn main() -> Result<()> {
    // 1. Setup Device (CUDA if available)
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("Training on device: {:?}", device);

    // 2. Load Model
    let mut varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = PolyZeroNet::new(vs.clone())?;

    // Load weights if they exist
    let model_path = Path::new("model.safetensors");
    if model_path.exists() {
        println!("Loading existing weights from {:?}", model_path);
        varmap.load(model_path)?;
    } else {
        println!("Initializing new random weights");
    }

    // 3. Setup Optimizer
    let mut adam = candle_nn::AdamW::new_lr(varmap.all_vars(), LEARNING_RATE)?;

    // 4. Load Data (Concatenate all games_*.safetensors)
    let mut all_states = Vec::new();
    let mut all_policies = Vec::new();
    let mut all_values = Vec::new();

    let entries = fs::read_dir(".")?;
    let mut found_data = false;

    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
            if path
                .file_name()
                .and_then(|s| s.to_str())
                .map(|s| s.starts_with("games_"))
                .unwrap_or(false)
            {
                println!("Loading data from {:?}", path);
                // Use the high-level load function which returns HashMap<String, Tensor>
                let tensors = candle_core::safetensors::load(&path, &device)?;

                if let (Some(s), Some(p), Some(v)) = (
                    tensors.get("states"),
                    tensors.get("policies"),
                    tensors.get("values"),
                ) {
                    all_states.push(s.clone());
                    all_policies.push(p.clone());
                    all_values.push(v.clone());
                    found_data = true;
                }
            }
        }
    }

    if !found_data {
        println!("No training data found (games_*.safetensors). Run self_play first.");
        return Ok(());
    }

    let states = Tensor::cat(&all_states, 0)?;
    let target_policies = Tensor::cat(&all_policies, 0)?;
    let target_values = Tensor::cat(&all_values, 0)?;

    let n_samples = states.dim(0)?;
    println!("Loaded {} total samples", n_samples);

    // Reshape states to BCHW
    let c = polyfish::ai::features::NUM_CHANNELS;
    let h = polyfish::ai::features::MAP_HEIGHT;
    let w = polyfish::ai::features::MAP_WIDTH;
    let states = states.reshape((n_samples, c, h, w))?;

    // 5. Training Loop
    for epoch in 1..=EPOCHS {
        let mut total_loss = 0.0;
        let mut batches = 0;

        for i in (0..n_samples).step_by(BATCH_SIZE) {
            let end = (i + BATCH_SIZE).min(n_samples);
            let batch_size = end - i;

            let batch_states = states.narrow(0, i, batch_size)?;
            let batch_target_p = target_policies.narrow(0, i, batch_size)?;
            let batch_target_v = target_values.narrow(0, i, batch_size)?;

            // Forward Pass
            let (pred_p_logits, pred_v) = model.forward_t(&batch_states, true)?;

            // Loss Calculation
            // Policy Loss: Cross Entropy
            // We use: -sum(target * log_softmax(pred))
            let log_probs = candle_nn::ops::log_softmax(&pred_p_logits, 1)?;
            // Removed trailing ? because division of Tensor by f64 returns Tensor (not Result)
            let p_loss = (batch_target_p * log_probs)?.sum_all()? / (-(batch_size as f64));

            // Value Loss: MSE
            let v_loss = candle_nn::loss::mse(&pred_v, &batch_target_v)?;

            let loss = (p_loss + v_loss)?;

            // Backward & Step
            adam.backward_step(&loss)?;

            total_loss += loss.to_vec0::<f32>()? as f64;
            batches += 1;
        }

        if batches > 0 {
            println!(
                "Epoch {} | Avg Loss: {:.4}",
                epoch,
                total_loss / batches as f64
            );
        }
    }

    // 6. Save Model
    println!("Saving updated model to model.safetensors");
    varmap.save("model.safetensors")?;

    Ok(())
}
