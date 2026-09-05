//! Writing one training shard: the per-step buffers the game loop fills
//! get stacked into named tensors, cast to f16, and saved as safetensors.
//!
//! Shard naming is a contract -- worker_loop.sh globs `games_*` and
//! train.py consumes it; `--trace-villages` runs quarantine themselves
//! under `trace_games_*` so diagnostics never enter the training set.

use candle_core::Tensor;
use polyfish::ai::features;
use std::collections::HashMap;

// Sharded output (Jul 28): flush every SHARD_GAMES games. A single cat of
// a -g 512 run needs one ~19GB Metal buffer — over the device allocation
// limit — while shards stay at the ~2.4GB scale 64-game runs proved.
// Constant games-per-FILE also keeps the loop's file-counted replay
// window exact in games at any -g.
pub(crate) const SHARD_GAMES: usize = 64;
#[allow(clippy::too_many_arguments)]
pub(crate) fn flush_shard(
    collected_spatial_maps: Vec<Tensor>,
    collected_player_states: Vec<Tensor>,
    collected_action_type: Vec<Vec<f32>>,
    collected_source_spatial: Vec<Vec<f32>>,
    collected_target_spatial: Vec<Vec<f32>>,
    collected_option: Vec<Vec<f32>>,
    collected_values: Vec<f32>,
    collected_progress: Vec<f32>,
    collected_aux_own: Vec<Vec<f32>>,
    collected_aux_fog: Vec<Vec<f32>>,
    collected_aux_spt: Vec<f32>,
    collected_aux_territory5: Vec<f32>,
    collected_aux_territory_now: Vec<f32>,
    collected_aux_territory1: Vec<f32>,
    collected_aux_eco_ceiling: Vec<Vec<f32>>,
    collected_eco_ceiling_mask: Vec<f32>,
    collected_rollout_value: Vec<f32>,
    collected_rollout_value_mask: Vec<f32>,
    collected_aux_pressure: Vec<f32>,
    collected_aux_army5: Vec<f32>,
    collected_aux_pursuit: Vec<f32>,
    collected_aux_city_spt: Vec<Vec<f32>>,
    collected_aux_tech: Vec<Vec<f32>>,
    num_techs: usize,
    collected_macro_stance: Vec<Vec<f32>>,
    collected_macro_order: Vec<Vec<f32>>,
    collected_macro_mask: Vec<f32>,
    device: &candle_core::Device,
    path: &str,
) -> anyhow::Result<()> {
    let total_steps = collected_spatial_maps.len();
    let spatial_dim = features::NUM_CHANNELS * features::MAP_SIZE * features::MAP_SIZE;
    let player_dim = features::RawFeatures::PLAYER_STATE_DIM;

    let spatial_maps_tensor = Tensor::cat(&collected_spatial_maps, 0)?;
    let spatial_maps_tensor = spatial_maps_tensor.reshape((total_steps, spatial_dim))?;
    let player_states_tensor = Tensor::cat(&collected_player_states, 0)?;
    let player_states_tensor = player_states_tensor.reshape((total_steps, player_dim))?;

    fn flatten_vec(v: Vec<Vec<f32>>) -> Vec<f32> {
        v.into_iter().flatten().collect()
    }

    let action_tensor = Tensor::from_vec(
        flatten_vec(collected_action_type),
        (total_steps, 11),
        device,
    )?;
    let spatial_logit_dim = features::MAP_SIZE * features::MAP_SIZE;
    let source_tensor = Tensor::from_vec(
        flatten_vec(collected_source_spatial),
        (total_steps, spatial_logit_dim),
        device,
    )?;
    let target_tensor = Tensor::from_vec(
        flatten_vec(collected_target_spatial),
        (total_steps, spatial_logit_dim),
        device,
    )?;
    let option_tensor =
        Tensor::from_vec(flatten_vec(collected_option), (total_steps, 192), device)?;
    let values_tensor = Tensor::from_vec(collected_values, (total_steps, 1), device)?;
    let progress_tensor = Tensor::from_vec(collected_progress, (total_steps, 1), device)?;

    // Aux-head targets — always emitted together (train.py's per-file
    // presence mask treats them as all-or-nothing).
    let aux_own_tensor = Tensor::from_vec(
        flatten_vec(collected_aux_own),
        (total_steps, spatial_logit_dim),
        device,
    )?;
    let aux_fog_tensor = Tensor::from_vec(
        flatten_vec(collected_aux_fog),
        (total_steps, spatial_logit_dim),
        device,
    )?;
    let aux_spt_tensor = Tensor::from_vec(collected_aux_spt, (total_steps, 2), device)?;
    let aux_territory5_tensor =
        Tensor::from_vec(collected_aux_territory5, (total_steps, 2), device)?;
    // Phase-2 spike (EXP_ELO_120): current + turn+1 territory, the pair a
    // chainable transition target needs. Same per-file AUX_DIMS convention
    // as territory5 — no row mask needed, every row has both.
    let aux_territory_now_tensor =
        Tensor::from_vec(collected_aux_territory_now, (total_steps, 2), device)?;
    let aux_territory1_tensor =
        Tensor::from_vec(collected_aux_territory1, (total_steps, 2), device)?;
    let aux_eco_ceiling_tensor = Tensor::from_vec(
        flatten_vec(collected_aux_eco_ceiling),
        (total_steps, 4),
        device,
    )?;
    let eco_ceiling_mask_tensor =
        Tensor::from_vec(collected_eco_ceiling_mask, (total_steps, 1), device)?;
    let rollout_value_tensor =
        Tensor::from_vec(collected_rollout_value, (total_steps, 1), device)?;
    let rollout_value_mask_tensor =
        Tensor::from_vec(collected_rollout_value_mask, (total_steps, 1), device)?;
    let aux_pressure_tensor =
        Tensor::from_vec(collected_aux_pressure, (total_steps, 1), device)?;
    let aux_army5_tensor = Tensor::from_vec(collected_aux_army5, (total_steps, 2), device)?;
    let aux_pursuit_tensor =
        Tensor::from_vec(collected_aux_pursuit, (total_steps, 1), device)?;
    let aux_city_spt_tensor = Tensor::from_vec(
        flatten_vec(collected_aux_city_spt),
        (total_steps, spatial_logit_dim),
        device,
    )?;
    let aux_tech_tensor = Tensor::from_vec(
        flatten_vec(collected_aux_tech),
        (total_steps, num_techs),
        device,
    )?;

    // EXP_ELO_061 (Stage 3b): macro policy targets. Per-row mask (not
    // the aux heads' per-file convention) since even a macro-mcts-heavy
    // run has unsupervised steps (opponent seat, anchor games) — see
    // the collection site's comment.
    let macro_stance_tensor =
        Tensor::from_vec(flatten_vec(collected_macro_stance), (total_steps, 4), device)?;
    let macro_order_tensor = Tensor::from_vec(
        flatten_vec(collected_macro_order),
        (total_steps, 3 * spatial_logit_dim),
        device,
    )?;
    let macro_mask_tensor =
        Tensor::from_vec(collected_macro_mask, (total_steps, 1), device)?;

    let mut tensors = HashMap::new();
    tensors.insert("spatial_maps".to_string(), spatial_maps_tensor);
    tensors.insert("player_states".to_string(), player_states_tensor);
    tensors.insert("action_type".to_string(), action_tensor);
    tensors.insert("source_spatial".to_string(), source_tensor);
    tensors.insert("target_spatial".to_string(), target_tensor);
    tensors.insert("move_option".to_string(), option_tensor);
    tensors.insert("values".to_string(), values_tensor);
    tensors.insert("progress".to_string(), progress_tensor);
    tensors.insert("aux_ownership".to_string(), aux_own_tensor);
    tensors.insert("aux_fog_units".to_string(), aux_fog_tensor);
    tensors.insert("aux_spt".to_string(), aux_spt_tensor);
    tensors.insert("aux_territory5".to_string(), aux_territory5_tensor);
    tensors.insert("aux_territory_now".to_string(), aux_territory_now_tensor);
    tensors.insert("aux_territory1".to_string(), aux_territory1_tensor);
    tensors.insert("aux_eco_ceiling".to_string(), aux_eco_ceiling_tensor);
    tensors.insert("aux_eco_ceiling_mask".to_string(), eco_ceiling_mask_tensor);
    tensors.insert("aux_rollout_value".to_string(), rollout_value_tensor);
    tensors.insert("aux_rollout_value_mask".to_string(), rollout_value_mask_tensor);
    tensors.insert("aux_pressure".to_string(), aux_pressure_tensor);
    tensors.insert("aux_army5".to_string(), aux_army5_tensor);
    tensors.insert("aux_opp_tech".to_string(), aux_tech_tensor);
    tensors.insert("aux_pursuit".to_string(), aux_pursuit_tensor);
    tensors.insert("aux_city_spt".to_string(), aux_city_spt_tensor);
    tensors.insert("macro_stance".to_string(), macro_stance_tensor);
    tensors.insert("macro_order".to_string(), macro_order_tensor);
    tensors.insert("macro_mask".to_string(), macro_mask_tensor);
    // f16 on disk (Jul 28): halves file size. Every stored tensor is
    // bounded ([-1,1] targets, probabilities, normalized features), so
    // f16's ~3 significant digits lose nothing that matters.
    for t in tensors.values_mut() {
        *t = t.to_dtype(candle_core::DType::F16)?;
    }
    candle_core::safetensors::save(&tensors, path)?;
    println!("💾 Shard saved: {path} ({total_steps} steps, f16)");
    Ok(())
}
