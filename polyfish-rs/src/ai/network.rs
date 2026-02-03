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

pub struct PolyZeroNet {
    conv1: Conv2d,
    bn1: BatchNorm,
    res_blocks: Vec<ResBlock>,
    // Policy Head - Fully Convolutional
    // Input: 32 channels. Output: 64 channels (ActionMapper::TOTAL_CHANNELS)
    p_conv1: Conv2d, // 64 -> 32 (intermediate)
    p_bn1: BatchNorm,
    p_conv2: Conv2d, // 32 -> 64 (final logic)
    // Value Head
    v_conv: Conv2d,
    v_bn: BatchNorm,
    v_fc1: Linear,
    v_fc2: Linear,
}

impl PolyZeroNet {
    pub fn new(vs: VarBuilder) -> Result<Self> {
        let filters = 64;
        let blocks = 4;
        let input_channels = crate::ai::features::NUM_CHANNELS;
        let policy_channels = crate::ai::mapper::ActionMapper::TOTAL_CHANNELS; // 64

        let conv1 = conv(input_channels, filters, 3, 1, 1, vs.pp("conv1"))?;
        let bn1 = batch_norm(filters, vs.pp("bn1"))?;

        let mut res_blocks = Vec::new();
        for i in 0..blocks {
            res_blocks.push(ResBlock::new(filters, vs.pp(format!("res{}", i)))?);
        }

        // Policy Head
        // Reduce to 32 channels with 1x1
        let p_conv1 = conv(filters, 32, 1, 1, 0, vs.pp("p_conv1"))?;
        let p_bn1 = batch_norm(32, vs.pp("p_bn1"))?;
        // Map to 64 channels with 1x1 (Logits map map-wise)
        let p_conv2 = conv(32, policy_channels, 1, 1, 0, vs.pp("p_conv2"))?;

        // Value Head
        let v_conv = conv(filters, 1, 1, 1, 0, vs.pp("v_conv"))?;
        let v_bn = batch_norm(1, vs.pp("v_bn"))?;
        let v_fc1 = candle_nn::linear(1 * 30 * 30, 64, vs.pp("v_fc1"))?;
        let v_fc2 = candle_nn::linear(64, 1, vs.pp("v_fc2"))?;

        Ok(Self {
            conv1,
            bn1,
            res_blocks,
            p_conv1,
            p_bn1,
            p_conv2,
            v_conv,
            v_bn,
            v_fc1,
            v_fc2,
        })
    }

    pub fn forward_t(&self, xs: &Tensor, train: bool) -> Result<(Tensor, Tensor)> {
        let mut x = self.conv1.forward(xs)?;
        x = self.bn1.forward_t(&x, train)?;
        x = x.relu()?;

        for block in &self.res_blocks {
            x = block.forward_t(&x, train)?;
        }

        // Policy
        let p = self.p_conv1.forward(&x)?;
        let p = self.p_bn1.forward_t(&p, train)?;
        let p = p.relu()?;
        let p = self.p_conv2.forward(&p)?;
        // Output: (B, 64, H, W).
        // When flattened, it is (B, 64*H*W).
        // Order: C=0 (entire map), C=1 (entire map)...
        // ActionMapper order: channel * (30*30) + pixel. Matches!

        let policy = p.flatten_from(1)?; // Flatten (C, H, W) -> Vector

        // Value
        let v = self.v_conv.forward(&x)?;
        let v = self.v_bn.forward_t(&v, train)?;
        let v = v.relu()?;
        let v = v.flatten_from(1)?;
        let v = self.v_fc1.forward(&v)?;
        let v = v.relu()?;
        let value = self.v_fc2.forward(&v)?;
        let value = value.tanh()?;

        Ok((policy, value))
    }

    /// Get the device this network is on
    pub fn device(&self) -> candle_core::Device {
        // Get device from first conv layer's weight
        self.conv1.weight().device().clone()
    }
}
