// Enhanced PolyZero Network with Decomposed Policy Heads
// Based on the successful Python architecture

use candle_core::{Module, ModuleT, Result, Tensor};
use candle_nn::{BatchNorm, Conv2d, Linear, VarBuilder};

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

// Player state MLP for global features
struct PlayerEmbedding {
    fc1: Linear,
    fc2: Linear,
}

impl PlayerEmbedding {
    fn new(input_dim: usize, hidden_dim: usize, output_dim: usize, vs: VarBuilder) -> Result<Self> {
        let fc1 = candle_nn::linear(input_dim, hidden_dim, vs.pp("fc1"))?;
        let fc2 = candle_nn::linear(hidden_dim, output_dim, vs.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(xs)?;
        let x = x.relu()?;
        self.fc2.forward(&x)
    }
}

pub struct PolyZeroNet {
    // Backbone
    conv1: Conv2d,
    bn1: BatchNorm,
    res_blocks: Vec<ResBlock>,

    // Player state embedding
    player_embed: PlayerEmbedding,

    // Fusion layer
    fusion_conv: Conv2d,
    fusion_bn: BatchNorm,

    // Post-fusion processing
    post_fusion_block: ResBlock,

    // Decomposed Policy Heads
    p_pool_conv: Conv2d, // Pooling for non-spatial policies
    p_pool_bn: BatchNorm,
    p_fc_shared: Linear, // Shared FC for categorical policies

    // Unified Policy Head
    pi_action_type: Linear, // Action type (Attack/Build/Move/etc)
    pi_source_conv: Conv2d, // Spatial: source tile
    pi_target_conv: Conv2d, // Spatial: target tile
    pi_option: Linear,      // Unified head for structure/unit/tech/ability/reward (size 192)

    // Value Heads (3: win, eco, mil)
    v_pool_conv: Conv2d,
    v_pool_bn: BatchNorm,
    v_fc_shared: Linear,
    v_win: Linear, // Win probability (score differential)
    v_eco: Linear, // Economic strength (SPT prediction)
    v_mil: Linear, // Military strength (unit count)
}

impl PolyZeroNet {
    pub fn new(vs: VarBuilder) -> Result<Self> {
        let filters = 64;
        let blocks = 12; // Increased from 8 to match Python
        let input_channels = crate::ai::features::NUM_CHANNELS;
        let player_state_dim = 10; // Global features (stars, spt, tech, etc)

        // Action dimensions
        let num_action_types = 11; // Attack, Build, Move, Research, EndTurn, etc
        let num_options = 192; // Unified options size

        // Backbone
        let conv1 = conv(input_channels, filters, 3, 1, 1, vs.pp("conv1"))?;
        let bn1 = batch_norm(filters, vs.pp("bn1"))?;

        let mut res_blocks = Vec::new();
        for i in 0..blocks {
            res_blocks.push(ResBlock::new(filters, vs.pp(format!("res{}", i)))?);
        }

        // Player state embedding
        let player_embed = PlayerEmbedding::new(
            player_state_dim,
            64,      // hidden
            filters, // output matches conv channels
            vs.pp("player_embed"),
        )?;

        // Fusion layer (combines spatial + player)
        let fusion_conv = conv(filters * 2, filters, 1, 1, 0, vs.pp("fusion_conv"))?;
        let fusion_bn = batch_norm(filters, vs.pp("fusion_bn"))?;

        // Post-fusion processing
        let post_fusion_block = ResBlock::new(filters, vs.pp("post_fusion"))?;

        // Policy heads - shared processing
        let p_pool_conv = conv(filters, 1, 1, 1, 0, vs.pp("p_pool_conv"))?;
        let p_pool_bn = batch_norm(1, vs.pp("p_pool_bn"))?;
        let p_fc_shared = candle_nn::linear(1 * 30 * 30, filters, vs.pp("p_fc_shared"))?;

        // Policy heads
        let pi_action_type = candle_nn::linear(filters, num_action_types, vs.pp("pi_action"))?;
        let pi_source_conv = conv(filters, 1, 1, 1, 0, vs.pp("pi_source"))?;
        let pi_target_conv = conv(filters, 1, 1, 1, 0, vs.pp("pi_target"))?;
        let pi_option = candle_nn::linear(filters, num_options, vs.pp("pi_option"))?;

        // Value heads - shared processing
        let v_pool_conv = conv(filters, 1, 1, 1, 0, vs.pp("v_pool_conv"))?;
        let v_pool_bn = batch_norm(1, vs.pp("v_pool_bn"))?;
        let v_fc_shared = candle_nn::linear(1 * 30 * 30, filters, vs.pp("v_fc_shared"))?;

        // 3 value heads
        let v_win = candle_nn::linear(filters, 1, vs.pp("v_win"))?;
        let v_eco = candle_nn::linear(filters, 1, vs.pp("v_eco"))?;
        let v_mil = candle_nn::linear(filters, 1, vs.pp("v_mil"))?;

        Ok(Self {
            conv1,
            bn1,
            res_blocks,
            player_embed,
            fusion_conv,
            fusion_bn,
            post_fusion_block,
            p_pool_conv,
            p_pool_bn,
            p_fc_shared,
            pi_action_type,
            pi_source_conv,
            pi_target_conv,
            pi_option,
            v_pool_conv,
            v_pool_bn,
            v_fc_shared,
            v_win,
            v_eco,
            v_mil,
        })
    }

    pub fn forward_t(
        &self,
        map_input: &Tensor,
        player_input: &Tensor,
        train: bool,
    ) -> Result<(PolicyOutput, ValueOutput)> {
        let (batch_size, _, h, w) = map_input.dims4()?;

        // 1. Process map through backbone
        let mut x = self.conv1.forward(map_input)?;
        x = self.bn1.forward_t(&x, train)?;
        x = x.relu()?;

        for block in &self.res_blocks {
            x = block.forward_t(&x, train)?;
        }

        // 2. Process player state
        let player_emb = self.player_embed.forward(player_input)?; // [B, 64]

        // 3. Broadcast and fuse
        let player_broadcast = player_emb
            .unsqueeze(2)? // [B, 64, 1]
            .unsqueeze(3)? // [B, 64, 1, 1]
            .broadcast_as((batch_size, 64, h, w))?;

        let fused = Tensor::cat(&[&x, &player_broadcast], 1)?; // [B, 128, H, W]
        let mut fused = self.fusion_conv.forward(&fused)?;
        fused = self.fusion_bn.forward_t(&fused, train)?;
        fused = fused.relu()?;

        // 4. Post-fusion processing
        let shared = self.post_fusion_block.forward_t(&fused, train)?;

        // 5. Policy Heads
        // Non-spatial policies (pooled)
        let p_pooled = self.p_pool_conv.forward(&shared)?;
        let p_pooled = self.p_pool_bn.forward_t(&p_pooled, train)?;
        let p_pooled = p_pooled.relu()?;
        let p_pooled = p_pooled.flatten_from(1)?;
        let p_latent = self.p_fc_shared.forward(&p_pooled)?;
        let p_latent = p_latent.relu()?;

        // Categorical policy heads
        let pi_action = self.pi_action_type.forward(&p_latent)?;
        let pi_option = self.pi_option.forward(&p_latent)?;

        // Spatial policy heads
        let pi_source = self.pi_source_conv.forward(&shared)?.flatten_from(1)?;
        let pi_target = self.pi_target_conv.forward(&shared)?.flatten_from(1)?;

        let policy_output = PolicyOutput {
            action_type: pi_action,
            source_spatial: pi_source,
            target_spatial: pi_target,
            move_option: pi_option,
        };

        // 6. Value Heads
        let v_pooled = self.v_pool_conv.forward(&shared)?;
        let v_pooled = self.v_pool_bn.forward_t(&v_pooled, train)?;
        let v_pooled = v_pooled.relu()?;
        let v_pooled = v_pooled.flatten_from(1)?;
        let v_latent = self.v_fc_shared.forward(&v_pooled)?;
        let v_latent = v_latent.relu()?;

        let v_win = self.v_win.forward(&v_latent)?.tanh()?; // [-1, 1]
        let v_eco = self.v_eco.forward(&v_latent)?.tanh()?; // Normalized SPT
        let v_mil = self.v_mil.forward(&v_latent)?.tanh()?; // Normalized units

        let value_output = ValueOutput {
            win_value: v_win,
            eco_value: v_eco,
            mil_value: v_mil,
        };

        Ok((policy_output, value_output))
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
    pub eco_value: Tensor, // [B, 1]
    pub mil_value: Tensor, // [B, 1]
}
