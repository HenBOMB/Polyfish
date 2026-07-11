// Enhanced PolyZero Network with Decomposed Policy Heads
// Based on the successful Python architecture

use candle_core::{Module, ModuleT, Result, Tensor};
use candle_nn::{BatchNorm, Conv2d, LayerNorm, Linear, VarBuilder};

fn conv(
    in_c: usize,
    out_c: usize,
    k: usize,
    stride: usize,
    padding: usize,
    vs: VarBuilder,
) -> Result<Conv2d> {
    let config = candle_nn::Conv2dConfig {
        padding,
        stride,
        ..Default::default()
    };
    candle_nn::conv2d(in_c, out_c, k, config, vs)
}

fn batch_norm(c: usize, vs: VarBuilder) -> Result<BatchNorm> {
    candle_nn::batch_norm(c, 1e-5, vs)
}

struct ResBlock {
    c1: Conv2d,
    bn1: BatchNorm,
    c2: Conv2d,
    bn2: BatchNorm,
}

impl ResBlock {
    fn new(c: usize, vs: VarBuilder) -> Result<Self> {
        let c1 = conv(c, c, 3, 1, 1, vs.pp("c1"))?;
        let bn1 = batch_norm(c, vs.pp("bn1"))?;
        let c2 = conv(c, c, 3, 1, 1, vs.pp("c2"))?;
        let bn2 = batch_norm(c, vs.pp("bn2"))?;
        Ok(Self { c1, bn1, c2, bn2 })
    }
}

impl ModuleT for ResBlock {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let ys = self.c1.forward(xs)?;
        let ys = self.bn1.forward_t(&ys, train)?;
        let ys = ys.relu()?;
        let ys = self.c2.forward(&ys)?;
        let ys = self.bn2.forward_t(&ys, train)?;
        (xs.add(&ys))?.relu()
    }
}

// Cross-Attention: allow Spatial (Q) to attend to Player (K, V)
struct CrossAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    norm: LayerNorm,
    nhead: usize,
    d_model: usize,
}

impl CrossAttention {
    fn new(d_model: usize, nhead: usize, vs: VarBuilder) -> Result<Self> {
        // train.py's CrossAttention wraps nn.MultiheadAttention, which packs
        // [Wq; Wk; Wv] into `attn.in_proj_weight` [3*D, D] / `in_proj_bias`
        // [3*D] rather than separate q/k/v_proj weights. Split here to match
        // the layout actually saved in model.safetensors (see tch_network.rs
        // for the mirrored PyTorch-side split).
        let attn_vs = vs.pp("attn");
        let in_w = attn_vs.get((3 * d_model, d_model), "in_proj_weight")?;
        let in_b = attn_vs.get(3 * d_model, "in_proj_bias")?;
        let wq = in_w.narrow(0, 0, d_model)?.contiguous()?;
        let wk = in_w.narrow(0, d_model, d_model)?.contiguous()?;
        let wv = in_w.narrow(0, 2 * d_model, d_model)?.contiguous()?;
        let bq = in_b.narrow(0, 0, d_model)?.contiguous()?;
        let bk = in_b.narrow(0, d_model, d_model)?.contiguous()?;
        let bv = in_b.narrow(0, 2 * d_model, d_model)?.contiguous()?;

        let q_proj = Linear::new(wq, Some(bq));
        let k_proj = Linear::new(wk, Some(bk));
        let v_proj = Linear::new(wv, Some(bv));
        let o_proj = candle_nn::linear(d_model, d_model, attn_vs.pp("out_proj"))?;
        let norm = candle_nn::layer_norm(d_model, 1e-5, vs.pp("norm"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            norm,
            nhead,
            d_model,
        })
    }

    fn forward(&self, query: &Tensor, context: &Tensor) -> Result<Tensor> {
        let (batch_sz, q_len, _) = query.dims3()?;
        let (_, c_len, _) = context.dims3()?;
        let head_dim = self.d_model / self.nhead;

        // Projections
        let q = self.q_proj.forward(query)?;
        let k = self.k_proj.forward(context)?;
        let v = self.v_proj.forward(context)?;

        // Multi-head split: [B, L, D] -> [B, Head, L, HeadDim]
        // Note: .contiguous() is required after transpose for Metal backend matmul compatibility
        let q = q
            .reshape((batch_sz, q_len, self.nhead, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((batch_sz, c_len, self.nhead, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((batch_sz, c_len, self.nhead, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Scaled dot-product attention
        let scale = 1.0 / (head_dim as f64).sqrt();
        let k_t = k.transpose(2, 3)?.contiguous()?;  // Make contiguous for Metal matmul
        let scores = (q.contiguous()?.matmul(&k_t)? * scale)?;
        let attn = candle_nn::ops::softmax(&scores, candle_core::D::Minus1)?;
        // Both operands must be contiguous for Metal matmul
        let out = attn.contiguous()?.matmul(&v.contiguous()?)?;

        // Merge heads: [B, Head, L, HeadDim] -> [B, L, D]
        let out = out
            .transpose(1, 2)?
            .reshape((batch_sz, q_len, self.d_model))?;
        let out = self.o_proj.forward(&out)?;

        // Residual + Norm
        self.norm.forward(&(query + out)?)
    }
}

pub struct PolyZeroNet {
    // Backbone
    conv1: Conv2d,
    bn1: BatchNorm,
    res_blocks: Vec<ResBlock>,

    // Cross-Attention integration
    player_feature_embeddings: Tensor, // [10, 64]
    player_pos_embeddings: Tensor,     // [10, 64]
    player_fc: Linear,
    cross_attention: CrossAttention,

    // Decomposed Policy Heads
    p_pool_conv: Conv2d,
    p_pool_bn: BatchNorm,
    p_fc_shared: Linear,

    pi_action: Linear, // Action type (12)
    pi_source: Conv2d, // Spatial source
    pi_target: Conv2d, // Spatial target
    pi_option: Linear, // Unified 192 options head

    v_pool_conv: Conv2d,
    v_pool_bn: BatchNorm,
    v_fc_shared: Linear,
    v_win: Linear,
    v_progress: Linear,
    // Aux ownership head (train.py's v_ownership): per-tile end-of-game
    // ownership. Training-only signal — search never reads it. Optional so
    // checkpoints that predate the head (league/arena opponents) still load.
    v_ownership: Option<Conv2d>,
}

impl PolyZeroNet {
    pub fn new(vs: VarBuilder) -> Result<Self> {
        let filters = 64;
        let blocks = 6;
        let input_channels = crate::ai::features::NUM_CHANNELS;
        let player_state_dim = crate::ai::features::RawFeatures::PLAYER_STATE_DIM;
        let num_action_types = 11;
        let num_options = 192;

        let conv1 = conv(input_channels, filters, 3, 1, 1, vs.pp("conv1"))?;
        let bn1 = batch_norm(filters, vs.pp("bn1"))?;

        let mut res_blocks = Vec::new();
        for i in 0..blocks {
            res_blocks.push(ResBlock::new(filters, vs.pp(format!("res_blocks.{}", i)))?);
        }

        // Player state tokenization
        // Use vs.get to load the learnable embeddings [10, 64]
        let player_feature_embeddings = vs.get(
            (player_state_dim as usize, filters),
            "player_feature_embeddings",
        )?;
        let player_pos_embeddings = vs.get(
            (player_state_dim as usize, filters),
            "player_pos_embeddings",
        )?;
        let player_fc = candle_nn::linear(filters, filters, vs.pp("player_fc"))?;

        // Cross-Attention layer
        let cross_attention = CrossAttention::new(filters, 4, vs.pp("cross_attention"))?;

        // Shared policy processing
        let p_pool_conv = conv(filters, 1, 1, 1, 0, vs.pp("p_pool_conv"))?;
        let p_pool_bn = batch_norm(1, vs.pp("p_pool_bn"))?;
        let p_fc_shared = candle_nn::linear(
            1 * crate::ai::features::MAP_SIZE * crate::ai::features::MAP_SIZE,
            filters,
            vs.pp("p_fc_shared"),
        )?;

        // Policy heads
        let pi_action = candle_nn::linear(filters, num_action_types, vs.pp("pi_action"))?;
        let pi_source = conv(filters, 1, 1, 1, 0, vs.pp("pi_source"))?;
        let pi_target = conv(filters, 1, 1, 1, 0, vs.pp("pi_target"))?;
        let pi_option = candle_nn::linear(filters, num_options, vs.pp("pi_option"))?;

        // Value processing
        let v_pool_conv = conv(filters, 1, 1, 1, 0, vs.pp("v_pool_conv"))?;
        let v_pool_bn = batch_norm(1, vs.pp("v_pool_bn"))?;
        let v_fc_shared = candle_nn::linear(
            1 * crate::ai::features::MAP_SIZE * crate::ai::features::MAP_SIZE,
            filters,
            vs.pp("v_fc_shared"),
        )?;
        let v_win = candle_nn::linear(filters, 1, vs.pp("v_win"))?;
        let v_progress = candle_nn::linear(filters, 1, vs.pp("v_progress"))?;
        // NB: a VarMap-backed builder (bin/train.rs) reports false here, so
        // the candle trainer neither trains nor re-saves these weights —
        // train.py re-inits the head after a candle-trainer save.
        let v_ownership = if vs.contains_tensor("v_ownership.weight") {
            Some(conv(filters, 1, 1, 1, 0, vs.pp("v_ownership"))?)
        } else {
            None
        };

        Ok(Self {
            conv1,
            bn1,
            res_blocks,
            player_feature_embeddings,
            player_pos_embeddings,
            player_fc,
            cross_attention,
            p_pool_conv,
            p_pool_bn,
            p_fc_shared,
            pi_action,
            pi_source,
            pi_target,
            pi_option,
            v_pool_conv,
            v_pool_bn,
            v_fc_shared,
            v_win,
            v_progress,
            v_ownership,
        })
    }

    pub fn forward_t(
        &self,
        map_input: &Tensor,
        player_input: &Tensor,
        train: bool,
    ) -> Result<(PolicyOutput, ValueOutput)> {
        let (batch_size, _in_channels, h, w) = map_input.dims4()?;
        let filters = self.conv1.weight().dims()[0];

        // 1. Process map through backbone
        let mut x = self.conv1.forward(map_input)?;
        x = self.bn1.forward_t(&x, train)?;
        x = x.relu()?;

        for block in &self.res_blocks {
            x = block.forward_t(&x, train)?;
        }

        // 2. Tokenize inputs for Cross-Attention
        // Spatial tokens: [B, H*W, Filters]
        let spatial_tokens = x.flatten_from(2)?.transpose(1, 2)?;

        // Player tokens: [B, 10, Filters]
        // player_tokens = player_input[B, 10, 1] * embeddings[1, 10, 64]
        let p_tokens = player_input
            .unsqueeze(2)?
            .broadcast_mul(&self.player_feature_embeddings.unsqueeze(0)?)?;
        let p_tokens = p_tokens.broadcast_add(&self.player_pos_embeddings.unsqueeze(0)?)?;
        let p_tokens = self.player_fc.forward(&p_tokens)?.relu()?;

        // 3. Apply Cross-Attention
        let x_attended = self.cross_attention.forward(&spatial_tokens, &p_tokens)?;

        // Reshape back to [B, Filters, H, W]
        let shared = x_attended
            .transpose(1, 2)?
            .reshape((batch_size, filters, h, w))?;

        // 4. Policy Heads
        let p_pooled = self.p_pool_conv.forward(&shared)?;
        let p_pooled = self.p_pool_bn.forward_t(&p_pooled, train)?;
        let p_pooled = p_pooled.relu()?;
        let p_pooled = p_pooled.flatten_from(1)?;
        let p_latent = self.p_fc_shared.forward(&p_pooled)?.relu()?;

        let pi_action = self.pi_action.forward(&p_latent)?;
        let pi_option = self.pi_option.forward(&p_latent)?;
        let pi_source = self.pi_source.forward(&shared)?.flatten_from(1)?;
        let pi_target = self.pi_target.forward(&shared)?.flatten_from(1)?;

        let policy_output = PolicyOutput {
            action_type: pi_action,
            source_spatial: pi_source,
            target_spatial: pi_target,
            move_option: pi_option,
        };

        // 5. Value Heads
        let v_pooled = self.v_pool_conv.forward(&shared)?;
        let v_pooled = self.v_pool_bn.forward_t(&v_pooled, train)?;
        let v_pooled = v_pooled.relu()?;
        let v_pooled = v_pooled.flatten_from(1)?;
        let v_latent = self.v_fc_shared.forward(&v_pooled)?.relu()?;
        let v_win = self.v_win.forward(&v_latent)?;
        let v_progress = self.v_progress.forward(&v_latent)?;
        let v_ownership = match &self.v_ownership {
            Some(head) => Some(head.forward(&shared)?.flatten_from(1)?),
            None => None,
        };

        Ok((policy_output, ValueOutput {
            win_value: v_win,
            progress_value: v_progress,
            ownership_value: v_ownership,
        }))
    }

    pub fn forward(
        &self,
        map_input: &Tensor,
        player_input: &Tensor,
    ) -> Result<(PolicyOutput, ValueOutput)> {
        self.forward_t(map_input, player_input, false)
    }

    /// Get the device this network is on
    pub fn device(&self) -> candle_core::Device {
        self.conv1.weight().device().clone()
    }
}

// Output structures
#[derive(Debug)]
pub struct PolicyOutput {
    pub action_type: Tensor,    // [B, num_action_types]
    pub source_spatial: Tensor, // [B, H*W]
    pub target_spatial: Tensor, // [B, H*W]
    pub move_option: Tensor,    // [B, 192]
}

#[derive(Debug)]
pub struct ValueOutput {
    pub win_value: Tensor, // [B, 1]
    pub progress_value: Tensor, // [B, 1]
    /// Aux ownership head output, [B, H*W]. None on checkpoints that predate
    /// the head. Diagnostics only — search must never read this.
    pub ownership_value: Option<Tensor>,
}

/// Device-free policy output for a single leaf: one row of each decomposed
/// head, as plain `Vec<f32>`. Safe to send across threads (unlike
/// [`PolicyOutput`], which holds device `Tensor`s).
///
/// Produced by reading a whole batched [`PolicyOutput`] to CPU once (see
/// [`PolicyOutput::to_raw_rows`]) and slicing by row — never by per-leaf
/// `Tensor::get`, which is a device op and must stay off actor threads.
#[derive(Debug, Clone)]
pub struct RawPolicyOutput {
    pub action_type: Vec<f32>,
    pub source_spatial: Vec<f32>,
    pub target_spatial: Vec<f32>,
    pub move_option: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarMap;

    fn forward_dummy(net: &PolyZeroNet, device: &Device) -> (PolicyOutput, ValueOutput) {
        let map = Tensor::zeros(
            (2, crate::ai::features::NUM_CHANNELS, crate::ai::features::MAP_SIZE, crate::ai::features::MAP_SIZE),
            candle_core::DType::F32,
            device,
        )
        .unwrap();
        let player = Tensor::zeros(
            (2, crate::ai::features::RawFeatures::PLAYER_STATE_DIM),
            candle_core::DType::F32,
            device,
        )
        .unwrap();
        net.forward(&map, &player).unwrap()
    }

    /// Both checkpoint generations must load: files predating the aux
    /// ownership head leave it None; files carrying v_ownership.* populate it.
    #[test]
    fn test_optional_ownership_head_load() {
        let device = Device::Cpu;
        let dir = std::env::temp_dir();
        let old_path = dir.join("polyfish_test_net_pre_ownership.safetensors");
        let new_path = dir.join("polyfish_test_net_with_ownership.safetensors");

        // A fresh VarMap-backed net registers no ownership head, so its save
        // is byte-equivalent to a pre-ownership checkpoint.
        let varmap = VarMap::new();
        let vs = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
        let _ = PolyZeroNet::new(vs).unwrap();
        varmap.save(&old_path).unwrap();

        let vs_old = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                &[old_path.clone()],
                candle_core::DType::F32,
                &device,
            )
            .unwrap()
        };
        let net_old = PolyZeroNet::new(vs_old).unwrap();
        assert!(net_old.v_ownership.is_none());
        let (_, values) = forward_dummy(&net_old, &device);
        assert!(values.ownership_value.is_none());

        // Same checkpoint plus v_ownership.* (train.py layout) -> head loads.
        let mut tensors = candle_core::safetensors::load(&old_path, &device).unwrap();
        tensors.insert(
            "v_ownership.weight".to_string(),
            Tensor::zeros((1, 64, 1, 1), candle_core::DType::F32, &device).unwrap(),
        );
        tensors.insert(
            "v_ownership.bias".to_string(),
            Tensor::zeros(1, candle_core::DType::F32, &device).unwrap(),
        );
        candle_core::safetensors::save(&tensors, &new_path).unwrap();

        let vs_new = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                &[new_path.clone()],
                candle_core::DType::F32,
                &device,
            )
            .unwrap()
        };
        let net_new = PolyZeroNet::new(vs_new).unwrap();
        assert!(net_new.v_ownership.is_some());
        let (_, values) = forward_dummy(&net_new, &device);
        let ownership = values.ownership_value.unwrap();
        assert_eq!(
            ownership.dims(),
            &[2, crate::ai::features::MAP_SIZE * crate::ai::features::MAP_SIZE]
        );

        let _ = std::fs::remove_file(old_path);
        let _ = std::fs::remove_file(new_path);
    }
}

impl PolicyOutput {
    /// Read this batched policy output to CPU and split it into one
    /// [`RawPolicyOutput`] per row (leaf). Call once per batch — the
    /// `to_vec1`/`to_vec2` reads are the only device ops involved.
    pub fn to_raw_rows(&self) -> Result<Vec<RawPolicyOutput>> {
        let action_type = self.action_type.to_vec2::<f32>()?;
        let source_spatial = self.source_spatial.to_vec2::<f32>()?;
        let target_spatial = self.target_spatial.to_vec2::<f32>()?;
        let move_option = self.move_option.to_vec2::<f32>()?;

        let batch = action_type.len();
        let mut rows = Vec::with_capacity(batch);
        for i in 0..batch {
            rows.push(RawPolicyOutput {
                action_type: action_type[i].clone(),
                source_spatial: source_spatial[i].clone(),
                target_spatial: target_spatial[i].clone(),
                move_option: move_option[i].clone(),
            });
        }
        Ok(rows)
    }
}
