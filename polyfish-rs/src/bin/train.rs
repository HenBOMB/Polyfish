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
    // 1. Setup Device
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("Training on device: {:?}", device);

    // 2. Load Model
    let mut varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = PolyZeroNet::new(vs.clone())?;

    let model_path = Path::new("model.safetensors");
    if model_path.exists() {
        println!("Loading existing weights from {:?}", model_path);
        varmap.load(model_path)?;
    } else {
        println!("Initializing new random weights");
    }

    // 3. Setup Optimizer
    let mut adam = candle_nn::AdamW::new_lr(varmap.all_vars(), LEARNING_RATE)?;
    // Decaying LR schedule is manual in loop

    // 4. Load Data
    let mut all_spatial_maps = Vec::new();
    let mut all_player_states = Vec::new();

    // Policy Targets
    let mut t_action = Vec::new();
    let mut t_source = Vec::new();
    let mut t_target = Vec::new();
    let mut t_option = Vec::new();

    // Value Targets
    let mut t_val_win = Vec::new();
    let mut t_val_eco = Vec::new();
    let mut t_val_mil = Vec::new();

    let entries = fs::read_dir(".")?;
    let mut found_data = false;

    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
            if path
                .file_name()
                .and_then(|s| s.to_str())
                .map(|s| {
                    s.starts_with("games_")
                        || s.starts_with("human_games_")
                        || s.starts_with("games_human_")
                })
                .unwrap_or(false)
            {
                println!("Loading data from {:?}", path);
                let tensors = candle_core::safetensors::load(&path, &device)?;

                // Helper to get and push
                macro_rules! append_tensor {
                    ($key:expr, $dest:expr) => {
                        if let Some(t) = tensors.get($key) {
                            $dest.push(t.clone());
                        }
                    };
                }

                append_tensor!("spatial_maps", all_spatial_maps);
                append_tensor!("player_states", all_player_states);
                append_tensor!("action_type", t_action);
                append_tensor!("source_spatial", t_source);
                append_tensor!("target_spatial", t_target);
                append_tensor!("move_option", t_option);
                append_tensor!("values", t_val_win);
                append_tensor!("eco_targets", t_val_eco);
                append_tensor!("mil_targets", t_val_mil);

                found_data = true;
            }
        }
    }

    if !found_data {
        println!("No training data found (games_*.safetensors). Run self_play first.");
        return Ok(());
    }

    // Cat all tensors
    println!("Concatenating tensors...");
    let spatial_maps = Tensor::cat(&all_spatial_maps, 0)?;
    let player_states = Tensor::cat(&all_player_states, 0)?;

    let target_action = Tensor::cat(&t_action, 0)?;
    let target_source = Tensor::cat(&t_source, 0)?;
    let target_target = Tensor::cat(&t_target, 0)?;
    let target_option = Tensor::cat(&t_option, 0)?;

    let target_win = Tensor::cat(&t_val_win, 0)?;
    let target_eco = if !t_val_eco.is_empty() {
        Some(Tensor::cat(&t_val_eco, 0)?)
    } else {
        None
    };
    let target_mil = if !t_val_mil.is_empty() {
        Some(Tensor::cat(&t_val_mil, 0)?)
    } else {
        None
    };

    let n_samples = spatial_maps.dim(0)?;
    println!("Loaded {} total samples", n_samples);

    // Reshape spatial to BCHW
    let c = polyfish::ai::features::NUM_CHANNELS;
    let h = polyfish::ai::features::MAP_SIZE;
    let w = polyfish::ai::features::MAP_SIZE;
    let spatial_maps = spatial_maps.reshape((n_samples, c, h, w))?;

    // Loss weights
    let w_action = 1.0;
    let w_spatial = 0.5;
    let w_option = 0.2;
    let w_value = 1.0;

    // 5. Training Loop
    for epoch in 1..=EPOCHS {
        let mut total_loss = 0.0;
        let mut total_p_loss = 0.0;
        let mut total_v_loss = 0.0;
        let mut batches = 0;

        // Shuffle indices (approximate shuffle by chunks or full index shuffle)
        // For simplicity in this example, we iterate sequentially, but ideally should shuffle.
        // TODO: Implement shuffle

        for i in (0..n_samples).step_by(BATCH_SIZE) {
            let end = (i + BATCH_SIZE).min(n_samples);
            let batch_size = end - i;

            // Batch data
            let b_spatial = spatial_maps.narrow(0, i, batch_size)?;
            let b_player = player_states.narrow(0, i, batch_size)?;

            let b_t_action = target_action.narrow(0, i, batch_size)?;
            let b_t_source = target_source.narrow(0, i, batch_size)?;
            let b_t_target = target_target.narrow(0, i, batch_size)?;
            let b_t_option = target_option.narrow(0, i, batch_size)?;

            let b_t_win = target_win.narrow(0, i, batch_size)?;
            let b_t_eco = target_eco
                .as_ref()
                .map(|t| t.narrow(0, i, batch_size))
                .transpose()?;
            let b_t_mil = target_mil
                .as_ref()
                .map(|t| t.narrow(0, i, batch_size))
                .transpose()?;

            // Forward Pass
            let (policy_out, value_out) = model.forward_t(&b_spatial, &b_player, true)?;

            // Soft Cross Entropy Loss Helper
            // Loss = -sum(target * log_softmax(pred)) / batch_size
            fn soft_ce(pred: &Tensor, target: &Tensor) -> Result<Tensor> {
                let log_probs = candle_nn::ops::log_softmax(pred, 1)?;
                let temp = (target * log_probs)?; // Element-wise
                let sum = temp.sum_all()?;
                let batch_size = target.dim(0)? as f64;
                Ok((sum / -batch_size)?)
            }

            // Policy Losses
            let l_action = soft_ce(&policy_out.action_type, &b_t_action)?;
            let l_source = soft_ce(&policy_out.source_spatial, &b_t_source)?;
            let l_target = soft_ce(&policy_out.target_spatial, &b_t_target)?;
            let l_option = soft_ce(&policy_out.move_option, &b_t_option)?;

            let p_loss = (l_action * w_action)?;
            let p_loss = (p_loss + (l_source * w_spatial)?)?;
            let p_loss = (p_loss + (l_target * w_spatial)?)?;
            let p_loss = (p_loss + (l_option * w_option)?)?;

            // Value Losses (MSE)
            let l_win = candle_nn::loss::mse(&value_out.win_value, &b_t_win)?;

            let l_eco = if let Some(t) = &b_t_eco {
                candle_nn::loss::mse(&value_out.eco_value, t)?
            } else {
                Tensor::new(0.0f32, &device)?
            };

            let l_mil = if let Some(t) = &b_t_mil {
                candle_nn::loss::mse(&value_out.mil_value, t)?
            } else {
                Tensor::new(0.0f32, &device)?
            };

            let v_loss = ((l_win + l_eco)? + l_mil)?;

            // Total loss
            let loss = (p_loss.clone() + (v_loss.clone() * w_value)?)?;

            // Backward
            adam.backward_step(&loss)?;

            total_loss += loss.to_vec0::<f32>()? as f64;
            total_p_loss += p_loss.to_vec0::<f32>()? as f64;
            total_v_loss += v_loss.to_vec0::<f32>()? as f64;
            batches += 1;
        }

        if batches > 0 {
            println!(
                "Epoch {} | Loss: {:.4} (Pol: {:.4}, Val: {:.4})",
                epoch,
                total_loss / batches as f64,
                total_p_loss / batches as f64,
                total_v_loss / batches as f64
            );
        }
    }

    // 6. Save Model
    println!("Saving updated model to model.safetensors");
    varmap.save("model.safetensors")?;

    Ok(())
}
