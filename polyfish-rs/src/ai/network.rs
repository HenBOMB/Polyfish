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
        let q_proj = candle_nn::linear(d_model, d_model, vs.pp("q_proj"))?;
        let k_proj = candle_nn::linear(d_model, d_model, vs.pp("k_proj"))?;
        let v_proj = candle_nn::linear(d_model, d_model, vs.pp("v_proj"))?;
        let o_proj = candle_nn::linear(d_model, d_model, vs.pp("o_proj"))?;
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
}

impl PolyZeroNet {
    pub fn new(vs: VarBuilder) -> Result<Self> {
        let filters = 64;
        let blocks = 6;
        let input_channels = crate::ai::features::NUM_CHANNELS;
        let player_state_dim = 10;
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

        Ok(Self {
            conv1,
            bn1,
            res_blocks,
            player_feature_embeddings,
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
        let v_win = self.v_win.forward(&v_latent)?.tanh()?;

        Ok((policy_output, ValueOutput { win_value: v_win }))
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
}
