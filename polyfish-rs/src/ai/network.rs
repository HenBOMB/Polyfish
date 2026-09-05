// Enhanced PolyZero Network with Decomposed Policy Heads
// Based on the successful Python architecture

use candle_core::{Module, ModuleT, Result, Tensor};
use candle_nn::{Conv2d, GroupNorm, LayerNorm, Linear, VarBuilder};

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

/// Mirror of train.py's GN_GROUPS. GroupNorm has no train/eval duality —
/// identical function in both modes, no running stats.
const GN_GROUPS: usize = 8;

fn group_norm(c: usize, vs: VarBuilder) -> Result<GroupNorm> {
    candle_nn::group_norm(GN_GROUPS, c, 1e-5, vs)
}

struct ResBlock {
    c1: Conv2d,
    bn1: GroupNorm,
    c2: Conv2d,
    bn2: GroupNorm,
}

impl ResBlock {
    fn new(c: usize, vs: VarBuilder) -> Result<Self> {
        let c1 = conv(c, c, 3, 1, 1, vs.pp("c1"))?;
        let bn1 = group_norm(c, vs.pp("bn1"))?;
        let c2 = conv(c, c, 3, 1, 1, vs.pp("c2"))?;
        let bn2 = group_norm(c, vs.pp("bn2"))?;
        Ok(Self { c1, bn1, c2, bn2 })
    }
}

impl ModuleT for ResBlock {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Result<Tensor> {
        let ys = self.c1.forward(xs)?;
        let ys = self.bn1.forward(&ys)?;
        let ys = ys.relu()?;
        let ys = self.c2.forward(&ys)?;
        let ys = self.bn2.forward(&ys)?;
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
    bn1: GroupNorm,
    res_blocks: Vec<ResBlock>,

    // Cross-Attention integration
    player_feature_embeddings: Tensor, // [10, 64]
    player_fc: Linear,
    cross_attention: CrossAttention,

    // Decomposed Policy Heads
    p_pool_conv: Conv2d,
    p_fc_shared: Linear,

    pi_action: Linear, // Action type (12)
    pi_source: Conv2d, // Spatial source
    pi_target: Conv2d, // Spatial target
    pi_option: Linear, // Unified 192 options head

    v_pool_conv: Conv2d,
    v_fc_shared: Linear,
    v_win: Linear,
    v_progress: Linear,

    /// v9: the ONE aux head mirrored into Rust — per-tile P(enemy unit under
    /// fog), the fog-encounter signal the risk term needs. Every other `aux_*`
    /// head stays training-only (see CLAUDE.md). Optional because one legacy
    /// checkpoint (`model_gn_v1.safetensors`) predates the aux heads entirely
    /// and the opponent loader must not start rejecting it.
    aux_fog: Option<Conv2d>,

    /// EXP_ELO_061 (Stage 3b): macro-mcts root prior, mirrored into Rust for
    /// the same reason as `aux_fog` — it's consumed at inference (macro
    /// root candidate scoring), not just during training. Softmax over
    /// `Stance` (4-way). Optional: no checkpoint carries this yet.
    pi_macro_stance: Option<Linear>,
    /// Per-tile P(order of this kind targets this tile), one channel per
    /// `OrderKind` (3) — a sigmoid intensity map, NOT a spatial softmax,
    /// because a goal's orders are non-exclusive across kinds and across
    /// same-kind targets (see EXP_ELO_061's overnight ballot analysis).
    pi_macro_order: Option<Conv2d>,

    /// EXP_ELO_125 (piece 4): cheap rollout-value estimator, mirrored into
    /// Rust for the same reason as `aux_fog`/`pi_macro_stance` — consumed at
    /// inference (macro-mcts's depth-gated frozen-edge shortcut), not just
    /// during training. Off the value trunk (`v_latent`), tanh-bounded like
    /// `v_win`. Optional: no checkpoint carries this until piece 4's
    /// training run lands.
    pi_rollout_value: Option<Linear>,
}

impl PolyZeroNet {
    pub fn new(vs: VarBuilder) -> Result<Self> {
        let filters = 64;
        let blocks = 6;
        let input_channels = crate::ai::features::NUM_CHANNELS;
        let player_state_dim = 10;
        let num_action_types = 11;
        let num_options = 192;

        // BN-era checkpoints carry bn1.weight/bias too, so they'd load into
        // GroupNorm code silently and play garbage — refuse them loudly.
        // Synthetic backends (e.g. VarBuilder::zeros) claim to contain every
        // name, including this probe — they aren't checkpoints, skip them.
        let synthetic_backend = vs.contains_tensor("__polyfish_bn_era_probe__");
        if !synthetic_backend && vs.contains_tensor("bn1.running_mean") {
            candle_core::bail!(
                "model file is a BatchNorm-era checkpoint (has bn1.running_mean); \
                 this build uses GroupNorm — regenerate the model (init_model.py + retrain)"
            );
        }

        let conv1 = conv(input_channels, filters, 3, 1, 1, vs.pp("conv1"))?;
        let bn1 = group_norm(filters, vs.pp("bn1"))?;

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

        // Shared policy processing (no norm on the 1-channel pool: a
        // per-sample norm would erase the map's overall level)
        let p_pool_conv = conv(filters, 1, 1, 1, 0, vs.pp("p_pool_conv"))?;
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

        // Value processing. v_pool_conv widened 1->8 channels (Jul 2026) to
        // remove the 1-channel value bottleneck; v_fc_shared in-features track it.
        let v_pool_conv = conv(filters, 8, 1, 1, 0, vs.pp("v_pool_conv"))?;
        let v_fc_shared = candle_nn::linear(
            8 * crate::ai::features::MAP_SIZE * crate::ai::features::MAP_SIZE,
            filters,
            vs.pp("v_fc_shared"),
        )?;
        let v_win = candle_nn::linear(filters, 1, vs.pp("v_win"))?;
        let v_progress = candle_nn::linear(filters, 1, vs.pp("v_progress"))?;

        // train.py: nn.Conv2d(filters, 1, 1) -> [B, 1, H, W] logits.
        let aux_fog = if vs.contains_tensor("aux_fog.weight") {
            Some(conv(filters, 1, 1, 1, 0, vs.pp("aux_fog"))?)
        } else {
            None
        };

        // EXP_ELO_061: macro policy head, optional (no checkpoint carries it
        // yet). Both tensors are always trained/saved together; gating each
        // independently on its own tensor name matches the aux_fog pattern
        // and stays robust to a partially-written checkpoint either way.
        let pi_macro_stance = if vs.contains_tensor("pi_macro_stance.weight") {
            Some(candle_nn::linear(filters, 4, vs.pp("pi_macro_stance"))?)
        } else {
            None
        };
        let pi_macro_order = if vs.contains_tensor("pi_macro_order.weight") {
            Some(conv(filters, 3, 1, 1, 0, vs.pp("pi_macro_order"))?)
        } else {
            None
        };

        // EXP_ELO_125 (piece 4): off the value trunk, single scalar output.
        let pi_rollout_value = if vs.contains_tensor("pi_rollout_value.weight") {
            Some(candle_nn::linear(filters, 1, vs.pp("pi_rollout_value"))?)
        } else {
            None
        };

        Ok(Self {
            conv1,
            bn1,
            res_blocks,
            player_feature_embeddings,
            player_fc,
            cross_attention,
            p_pool_conv,
            p_fc_shared,
            pi_action,
            pi_source,
            pi_target,
            pi_option,
            v_pool_conv,
            v_fc_shared,
            v_win,
            v_progress,
            aux_fog,
            pi_macro_stance,
            pi_macro_order,
            pi_rollout_value,
        })
    }

    /// True when this checkpoint carries the mirrored fog head. Callers must
    /// branch on it rather than reading a silent zero — see the `progress`
    /// precedent, where the tch and Metal backends stub the value to 0.0 and
    /// every consumer has been reading a constant.
    pub fn has_fog_head(&self) -> bool {
        self.aux_fog.is_some()
    }

    /// True when this checkpoint carries the mirrored macro policy head
    /// (EXP_ELO_061). Same silent-zero caveat as `has_fog_head`: callers on
    /// the tch/Metal eval backends must branch on this once this head is
    /// threaded through `EvalResult`, not yet done as of this head's landing.
    pub fn has_macro_policy_head(&self) -> bool {
        self.pi_macro_stance.is_some() && self.pi_macro_order.is_some()
    }

    /// True when this checkpoint carries the mirrored rollout-value head
    /// (EXP_ELO_125). Same silent-zero caveat as `has_fog_head`.
    pub fn has_rollout_value_head(&self) -> bool {
        self.pi_rollout_value.is_some()
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
        x = self.bn1.forward(&x)?;
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
        // Pool convs are linear: no norm/activation (unnormed ReLU here goes dead).
        let p_pooled = self.p_pool_conv.forward(&shared)?;
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
        let v_pooled = v_pooled.flatten_from(1)?;
        let v_latent = self.v_fc_shared.forward(&v_pooled)?.relu()?;
        let v_win = self.v_win.forward(&v_latent)?.tanh()?;
        let v_progress = self.v_progress.forward(&v_latent)?;

        // v9: mirrors train.py — the aux heads read the post-cross-attention
        // trunk (`shared` here, `x` there), not the pre-attention conv stack.
        let fog_probs = match &self.aux_fog {
            Some(head) => Some(candle_nn::ops::sigmoid(
                &head.forward(&shared)?.flatten_from(1)?,
            )?),
            None => None,
        };

        // EXP_ELO_061: macro-mcts root prior. Stance is a 4-way softmax
        // (mutually exclusive, off `v_latent` — reuses the value trunk's
        // pooling rather than adding a third pool conv). Order is a per-tile
        // sigmoid intensity, one channel per OrderKind (like aux_fog, not
        // like the softmaxed ply-level policy heads), because a goal's
        // orders are non-exclusive across kinds and across same-kind
        // targets — see the overnight ballot analysis in the ledger.
        let macro_stance_probs = match &self.pi_macro_stance {
            Some(head) => Some(candle_nn::ops::softmax(&head.forward(&v_latent)?, 1)?),
            None => None,
        };
        let macro_order_maps = match &self.pi_macro_order {
            Some(head) => Some(candle_nn::ops::sigmoid(
                &head.forward(&shared)?.flatten_from(1)?,
            )?),
            None => None,
        };

        // EXP_ELO_125 (piece 4): off the value trunk, like pi_macro_stance.
        let rollout_value = match &self.pi_rollout_value {
            Some(head) => Some(head.forward(&v_latent)?.tanh()?),
            None => None,
        };

        Ok((
            policy_output,
            ValueOutput {
                win_value: v_win,
                progress_value: v_progress,
                fog_probs,
                macro_stance_probs,
                macro_order_maps,
                rollout_value,
            },
        ))
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
    /// v9: per-tile P(enemy unit under fog), [B, H*W], already through the
    /// sigmoid. `None` when the checkpoint predates the aux heads — callers
    /// must branch rather than read a zero.
    pub fog_probs: Option<Tensor>,
    /// EXP_ELO_061: macro-mcts root prior, stance half. [B, 4], already
    /// through softmax (`Stance` order: Grow, Arm, Unlock, Save). `None`
    /// until a checkpoint carries this head.
    pub macro_stance_probs: Option<Tensor>,
    /// EXP_ELO_061: macro-mcts root prior, order half. [B, 3*H*W] (one
    /// H*W-tile plane per `OrderKind`, row-major within each plane),
    /// already through sigmoid — independent per-tile intensities, not a
    /// spatial softmax (orders are non-exclusive). `None` until a
    /// checkpoint carries this head.
    pub macro_order_maps: Option<Tensor>,
    /// EXP_ELO_125 (piece 4): cheap rollout-value estimator, [B, 1],
    /// already tanh-bounded. `None` until a checkpoint carries this head.
    pub rollout_value: Option<Tensor>,
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
    /// v9: per-tile P(enemy unit under fog), H*W long, already sigmoided.
    /// `None` when the backend or checkpoint cannot produce it — callers MUST
    /// branch instead of reading a zero (the `progress` value is stubbed to
    /// 0.0 on the tch and Metal paths; that trap is not repeated here).
    pub fog: Option<Vec<f32>>,
    /// EXP_ELO_061: macro-mcts root prior, stance half. 4 long, already
    /// softmaxed. Mirrored on all three backends (candle, Metal, tch —
    /// `tch_network.rs`'s `has_macro_policy`/`pi_macro_stance`). Only
    /// `progress` is candle-only project-wide.
    pub macro_stance: Option<Vec<f32>>,
    /// EXP_ELO_061: macro-mcts root prior, order half. 3*H*W long, already
    /// sigmoided. Same backend coverage as `macro_stance`.
    pub macro_order: Option<Vec<f32>>,
    /// EXP_ELO_125 (piece 4): cheap rollout-value estimator, a single
    /// tanh-bounded scalar, already through the head. `None` when the
    /// checkpoint or backend cannot produce it -- callers must fall back to
    /// full `execute_turn` simulation, same convention as `fog`/`macro_*`.
    pub rollout_value: Option<f32>,
}

impl PolicyOutput {
    /// Read this batched policy output to CPU and split it into one
    /// [`RawPolicyOutput`] per row (leaf). Call once per batch — the
    /// `to_vec1`/`to_vec2` reads are the only device ops involved.
    pub fn to_raw_rows(
        &self,
        fog: Option<&Tensor>,
        macro_stance: Option<&Tensor>,
        macro_order: Option<&Tensor>,
        rollout_value: Option<&Tensor>,
    ) -> Result<Vec<RawPolicyOutput>> {
        let action_type = self.action_type.to_vec2::<f32>()?;
        let source_spatial = self.source_spatial.to_vec2::<f32>()?;
        let target_spatial = self.target_spatial.to_vec2::<f32>()?;
        let move_option = self.move_option.to_vec2::<f32>()?;

        let fog_rows = match fog {
            Some(t) => Some(t.to_vec2::<f32>()?),
            None => None,
        };
        let stance_rows = match macro_stance {
            Some(t) => Some(t.to_vec2::<f32>()?),
            None => None,
        };
        let order_rows = match macro_order {
            Some(t) => Some(t.to_vec2::<f32>()?),
            None => None,
        };
        let rollout_value_rows = match rollout_value {
            Some(t) => Some(t.flatten_all()?.to_vec1::<f32>()?),
            None => None,
        };

        let batch = action_type.len();
        let mut rows = Vec::with_capacity(batch);
        for i in 0..batch {
            rows.push(RawPolicyOutput {
                action_type: action_type[i].clone(),
                source_spatial: source_spatial[i].clone(),
                target_spatial: target_spatial[i].clone(),
                move_option: move_option[i].clone(),
                fog: fog_rows.as_ref().map(|f| f[i].clone()),
                macro_stance: stance_rows.as_ref().map(|f| f[i].clone()),
                macro_order: order_rows.as_ref().map(|f| f[i].clone()),
                rollout_value: rollout_value_rows.as_ref().map(|f| f[i]),
            });
        }
        Ok(rows)
    }
}

#[cfg(test)]
mod fog_head_tests {
    use super::*;
    use candle_core::{DType, Device};

    /// The fog head must be OPTIONAL: `model_gn_v1.safetensors` predates the
    /// aux heads entirely, and the opponent loader is strict — a hard
    /// requirement here would crash the first league iteration that drew it.
    #[test]
    fn fog_head_is_optional_and_absent_on_pre_aux_checkpoints() {
        let path = std::path::Path::new("checkpoints/model_gn_v1.safetensors");
        if !path.exists() {
            return; // checkpoint set not present in this working copy
        }
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                &[path],
                DType::F32,
                &Device::Cpu,
            )
        };
        let Ok(vb) = vb else { return };
        match PolyZeroNet::new(vb) {
            // If it loads, the head must simply be absent — never a hard error.
            Ok(net) => assert!(!net.has_fog_head()),
            // It does NOT currently load, but for a pre-existing reason: it
            // predates the 8-channel value pool (Jul 2026). The fog head must
            // not become a NEW rejection reason, so pin that the failure is
            // still the old shape mismatch.
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    msg.contains("v_pool_conv"),
                    "legacy checkpoint must fail on its pre-existing arch gap, not on aux_fog: {msg}"
                );
            }
        }
    }

    /// …and present on a real trained checkpoint, with train.py's shape
    /// (filters -> 1). Deliberately NOT `model.safetensors`: that file is
    /// rewritten by every training iteration and by any weight rollback, so
    /// pinning ground truth to it makes the test fail for reasons that have
    /// nothing to do with the mirror being wrong.
    #[test]
    fn fog_head_loads_from_a_frozen_checkpoint_and_produces_per_tile_probs() {
        let path = std::path::Path::new("checkpoints/gauge_1785601511_iter5.safetensors");
        if !path.exists() {
            return;
        }
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &Device::Cpu)
        };
        let Ok(vb) = vb else { return };
        let Ok(net) = PolyZeroNet::new(vb) else { return };
        assert!(net.has_fog_head(), "a train.py checkpoint carries aux_fog");

        let hw = crate::ai::features::MAP_SIZE;
        let map = Tensor::zeros(
            (1, crate::ai::features::NUM_CHANNELS, hw, hw),
            DType::F32,
            &Device::Cpu,
        )
        .unwrap();
        let player = Tensor::zeros((1, 10), DType::F32, &Device::Cpu).unwrap();
        let (_, value) = net.forward(&map, &player).unwrap();
        let fog = value.fog_probs.expect("fog head present -> Some");
        assert_eq!(fog.dims(), &[1, hw * hw], "one probability per tile");
        let v = fog.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(
            v.iter().all(|p| (0.0..=1.0).contains(p)),
            "sigmoid output must be a probability"
        );

        // GROUND TRUTH: train.py's own PolyZeroNet, same weights, all-zero
        // input (scratchpad/fog_vs_pytorch.py). Mirroring a head is only worth
        // anything if the mirror is faithful — this pins the Rust side to
        // PyTorch itself, not to another Rust implementation.
        let pytorch = [0.004229_f32, 0.000252, 0.002072, 0.000861, 0.000959, 0.000989];
        for (i, expect) in pytorch.iter().enumerate() {
            assert!((v[i] - expect).abs() < 2e-5, "tile {i}: rust {} vs pytorch {expect}", v[i]);
        }
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        assert!((mean - 0.002729).abs() < 2e-5, "mean {mean} vs pytorch 0.002729");
    }
}

#[cfg(test)]
mod macro_policy_head_tests {
    use super::*;
    use candle_core::{DType, Device};

    /// EXP_ELO_061: absent on every checkpoint that exists today — no
    /// checkpoint has been trained with this head yet. Same optional-load
    /// contract as aux_fog: loading must succeed, the head must just be None.
    #[test]
    fn macro_policy_head_is_optional_and_absent_on_current_checkpoints() {
        let path = std::path::Path::new("checkpoints/gauge_1785601511_iter5.safetensors");
        if !path.exists() {
            return;
        }
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &Device::Cpu)
        };
        let Ok(vb) = vb else { return };
        let Ok(net) = PolyZeroNet::new(vb) else { return };
        assert!(!net.has_macro_policy_head());
    }

    /// Shape/activation correctness in isolation from any trained weights:
    /// `VarBuilder::zeros` satisfies `contains_tensor` for every name (a
    /// synthetic backend, not a checkpoint — see the `synthetic_backend`
    /// guard in `PolyZeroNet::new`), so this exercises the real forward path
    /// end to end without needing train.py to have shipped the head first.
    #[test]
    fn macro_policy_head_shapes_and_activations_are_correct() {
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &Device::Cpu);
        let net = PolyZeroNet::new(vb).expect("zeros backend must satisfy every tensor request");
        assert!(net.has_macro_policy_head());

        let hw = crate::ai::features::MAP_SIZE;
        let map = Tensor::zeros(
            (1, crate::ai::features::NUM_CHANNELS, hw, hw),
            DType::F32,
            &Device::Cpu,
        )
        .unwrap();
        let player = Tensor::zeros((1, 10), DType::F32, &Device::Cpu).unwrap();
        let (_, value) = net.forward(&map, &player).unwrap();

        let stance = value.macro_stance_probs.expect("stance head present -> Some");
        assert_eq!(stance.dims(), &[1, 4], "one probability per Stance variant");
        let sv = stance.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let sum: f32 = sv.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax must sum to 1, got {sum}");
        assert!(sv.iter().all(|p| (0.0..=1.0).contains(p)));

        let order = value.macro_order_maps.expect("order head present -> Some");
        assert_eq!(order.dims(), &[1, 3 * hw * hw], "3 OrderKind planes, flattened");
        let ov = order.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(
            ov.iter().all(|p| (0.0..=1.0).contains(p)),
            "sigmoid output must be a probability, independent per tile/kind"
        );
    }

    /// GROUND TRUTH: train.py's own PolyZeroNet (seed 1787, real non-trivial
    /// weights — not zeros, which would trivially agree regardless of which
    /// tensor either side actually reads), constant 0.1 input, saved via
    /// `scratchpad/gen_macro_head_ref.py` to
    /// `checkpoints/macro_policy_head_ref.safetensors` (gitignored like every
    /// other checkpoint fixture — regenerate locally if absent). Mirroring a
    /// head is only worth anything if the mirror is faithful: same trunk
    /// tensor, same weights, same activation, checked against PyTorch itself
    /// rather than against another Rust implementation.
    #[test]
    fn macro_policy_head_matches_pytorch_reference() {
        let path = std::path::Path::new("checkpoints/macro_policy_head_ref.safetensors");
        if !path.exists() {
            return;
        }
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &Device::Cpu)
        };
        let Ok(vb) = vb else { return };
        let Ok(net) = PolyZeroNet::new(vb) else { return };
        assert!(net.has_macro_policy_head());

        let hw = crate::ai::features::MAP_SIZE;
        let map = (Tensor::ones(
            (1, crate::ai::features::NUM_CHANNELS, hw, hw),
            DType::F32,
            &Device::Cpu,
        )
        .unwrap()
            * 0.1)
            .unwrap();
        let player = (Tensor::ones((1, 10), DType::F32, &Device::Cpu).unwrap() * 0.1).unwrap();
        let (_, value) = net.forward(&map, &player).unwrap();

        let stance = value.macro_stance_probs.expect("stance head present -> Some");
        let sv = stance.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let pytorch_stance = [0.1972564309835434_f32, 0.3269766569137573, 0.23765535652637482, 0.23811152577400208];
        for (i, expect) in pytorch_stance.iter().enumerate() {
            assert!((sv[i] - expect).abs() < 2e-5, "stance[{i}]: rust {} vs pytorch {expect}", sv[i]);
        }

        let order = value.macro_order_maps.expect("order head present -> Some");
        let ov = order.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(ov.len(), 3 * hw * hw);
        let pytorch_order_first6 = [0.6025543808937073_f32, 0.5542975068092346, 0.5799309015274048, 0.5303328633308411, 0.5658227205276489, 0.5697625279426575];
        for (i, expect) in pytorch_order_first6.iter().enumerate() {
            assert!((ov[i] - expect).abs() < 2e-5, "order[{i}]: rust {} vs pytorch {expect}", ov[i]);
        }
        let mean = ov.iter().sum::<f32>() / ov.len() as f32;
        assert!((mean - 0.6129000892159696).abs() < 2e-5, "order mean {mean} vs pytorch 0.6129000892159696");
    }
}
