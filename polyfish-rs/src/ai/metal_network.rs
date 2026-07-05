//! MPSGraph inference path for PolyZeroNet (behind the `metal-eval` feature).
//!
//! libtorch's MPS backend funnels every op through one serial C++
//! `dispatch_queue`, so two eval-server shards on `tch-eval` HALVE
//! throughput instead of doubling it (measured: 156.6 moves/s @ 1 shard vs
//! 83.3 moves/s @ 2 shards). This module bypasses libtorch for inference
//! entirely by composing the forward pass directly as an `MPSGraph` — the
//! same tuned, MLIR-fused graph executor libtorch-MPS itself lowers onto —
//! driven straight from Rust so we own the command queue instead of going
//! through libtorch's serial dispatch.
//!
//! This is a byte-for-byte op mirror of `tch_network.rs`'s
//! `TchPolyZeroNet::forward_batch` (same op order, same `nn.MultiheadAttention`
//! semantics), so weights stay compatible with `model.safetensors`.
//!
//! `forward_batch` compiles the graph once per distinct batch size and
//! caches the resulting `Executable`, replaying it on every subsequent call
//! at that batch size (see [`MetalPolyZeroNet::build_and_compile`]). A
//! single graph can't cover every batch size here: this crate's `reshape`
//! takes a literal `&[usize]` target shape with no `-1`/inferred-dim
//! sentinel, and the batch size is baked into several reshapes (token
//! flattening, attention head splitting, etc.), so each batch size needs
//! its own compiled `Executable`. Coalesced batch sizes cluster heavily in
//! practice (steady-state load produces a small set of common sizes), so
//! the cache converges to mostly-hits after a brief warmup.

use crate::ai::features::{MAP_SIZE, NUM_CHANNELS};
use crate::ai::network::RawPolicyOutput;
use apple_metal::{CommandQueue, MetalDevice};
use apple_mpsgraph::{
    data_type, padding_style, tensor_named_data_layout, Convolution2DDescriptor,
    Convolution2DDescriptorInfo, Executable, FeedDescription, Graph, Tensor, TensorData,
    UnaryArithmeticOp,
};
use safetensors::tensor::{Dtype, SafeTensors};
use std::cell::RefCell;
use std::collections::HashMap;

const FILTERS: usize = 64;
const NHEAD: usize = 4;
const HEAD_DIM: usize = FILTERS / NHEAD; // 16
const PLAYER_DIM: usize = 10;
const SPATIAL: usize = MAP_SIZE * MAP_SIZE; // 121
const BN_EPS: f64 = 1e-5;
const LN_EPS: f64 = 1e-5;
const NUM_RES_BLOCKS: usize = 6;

/// Decode an IEEE 754 binary16 (half-precision) bit pattern to f32. Standard
/// bit-manipulation conversion (e.g. Fabian Giesen's `half_to_float`); no
/// external `half` crate dependency needed for this one-shot load-time use.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exp = u32::from(bits >> 10) & 0x1f;
    let mut mantissa = u32::from(bits) & 0x3ff;

    let bits32 = if exp == 0 {
        if mantissa == 0 {
            sign
        } else {
            // Subnormal half -> normalize into a single-precision exponent.
            let mut out_exp: i32 = 127 - 15 + 1;
            while mantissa & 0x400 == 0 {
                mantissa <<= 1;
                out_exp -= 1;
            }
            mantissa &= 0x3ff;
            sign | ((out_exp as u32) << 23) | (mantissa << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (mantissa << 13) // Inf (mantissa=0) or NaN
    } else {
        let out_exp = exp + (127 - 15);
        sign | (out_exp << 23) | (mantissa << 13)
    };

    f32::from_bits(bits32)
}

#[cfg(test)]
mod tests {
    use super::f16_to_f32;

    #[test]
    fn f16_to_f32_known_values() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x8000), -0.0);
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xbc00), -1.0);
        assert_eq!(f16_to_f32(0x4000), 2.0);
        assert_eq!(f16_to_f32(0x3555), 0.33325195);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24)); // smallest subnormal
        assert!(f16_to_f32(0x7c00).is_infinite());
        assert!(f16_to_f32(0x7e00).is_nan());
    }
}

/// PolyZeroNet inference via a hand-composed MPSGraph. Holds the raw weight
/// bytes (loaded once from `model.safetensors`), the Metal device + a single
/// reused `CommandQueue`, and a cache of compiled `Executable`s keyed by
/// batch size (see module docs). Not `Send`/`Sync`-safe to *use*
/// concurrently, same invariant as the tch and candle nets: lives on a
/// single eval-server thread (the `RefCell` cache assumes single-threaded
/// access).
pub struct MetalPolyZeroNet {
    weights: HashMap<String, (Vec<usize>, Vec<f32>)>,
    device: MetalDevice,
    queue: CommandQueue,
    /// Compiled forward graphs, one per batch size seen so far. Built
    /// lazily on first use of a given size, then replayed.
    executables: RefCell<HashMap<usize, Executable>>,
}

impl MetalPolyZeroNet {
    /// Load `model.safetensors` (a PyTorch state_dict) into host memory.
    ///
    /// The checkpoint stores every tensor as F16 (`model.half()` at save
    /// time; even `num_batches_tracked`, which is unused here). Decode to
    /// f32 on load — mirrors `TchPolyZeroNet::load`'s
    /// `.to_kind(Kind::Float)`, just done on the CPU host side instead of by
    /// libtorch.
    pub fn load(path: &str) -> anyhow::Result<Self> {
        if !std::path::Path::new(path).exists() {
            anyhow::bail!("Model file {path} not found (run init_model.py first)");
        }
        let bytes = std::fs::read(path)?;
        let tensors = SafeTensors::deserialize(&bytes)?;
        let mut weights = HashMap::with_capacity(tensors.len());
        for (name, view) in tensors.tensors() {
            let shape = view.shape().to_vec();
            let data = view.data();
            let floats: Vec<f32> = match view.dtype() {
                Dtype::F32 => {
                    anyhow::ensure!(
                        data.len() % 4 == 0,
                        "weight {name} byte length {} not a multiple of 4",
                        data.len()
                    );
                    data.chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect()
                }
                Dtype::F16 => {
                    anyhow::ensure!(
                        data.len() % 2 == 0,
                        "weight {name} byte length {} not a multiple of 2",
                        data.len()
                    );
                    data.chunks_exact(2)
                        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                        .collect()
                }
                other => anyhow::bail!("unexpected dtype for {name}: {other:?} (expected F32 or F16)"),
            };
            weights.insert(name, (shape, floats));
        }
        let device = MetalDevice::system_default()
            .ok_or_else(|| anyhow::anyhow!("no Metal device available"))?;
        let queue = device
            .new_command_queue()
            .ok_or_else(|| anyhow::anyhow!("failed to create Metal command queue"))?;
        Ok(Self {
            weights,
            device,
            queue,
            executables: RefCell::new(HashMap::new()),
        })
    }

    fn get(&self, name: &str) -> &(Vec<usize>, Vec<f32>) {
        self.weights
            .get(name)
            .unwrap_or_else(|| panic!("BUG: missing weight {name} in model.safetensors"))
    }

    /// Create a graph constant with the tensor's natural (stored) shape.
    fn const_natural(&self, graph: &Graph, name: &str) -> Tensor {
        let (shape, data) = self.get(name);
        graph
            .constant_f32_slice(data, shape)
            .unwrap_or_else(|| panic!("failed to create constant {name}"))
    }

    /// Create a graph constant reshaped to `shape` (must have the same
    /// element count as the tensor's natural stored shape).
    fn const_shape(&self, graph: &Graph, name: &str, shape: &[usize]) -> Tensor {
        let (_, data) = self.get(name);
        graph
            .constant_f32_slice(data, shape)
            .unwrap_or_else(|| panic!("failed to create constant {name} with shape {shape:?}"))
    }

    /// PyTorch `nn.Linear`: `y = x @ W^T + b`, W is `[out, in]`. Broadcasts
    /// over any leading batch dims of `x`.
    fn linear(&self, graph: &Graph, x: &Tensor, prefix: &str) -> Tensor {
        let (wshape, _) = self.get(&format!("{prefix}.weight"));
        let out_dim = wshape[0];
        let w = self.const_natural(graph, &format!("{prefix}.weight"));
        let wt = graph
            .transpose(&w, &[1, 0], None)
            .expect("linear: transpose weight");
        let b = self.const_shape(graph, &format!("{prefix}.bias"), &[out_dim]);
        let mm = graph
            .matrix_multiplication(x, &wt, None)
            .expect("linear: matmul");
        graph.addition(&mm, &b, None).expect("linear: bias add")
    }

    fn conv2d(&self, graph: &Graph, x: &Tensor, prefix: &str, padding: usize) -> Tensor {
        let w = self.const_natural(graph, &format!("{prefix}.weight")); // [out, in, kh, kw]
        let (bshape, _) = self.get(&format!("{prefix}.bias"));
        let c = bshape[0];
        let b = self.const_shape(graph, &format!("{prefix}.bias"), &[1, c, 1, 1]);
        let descriptor = Convolution2DDescriptor::new(Convolution2DDescriptorInfo {
            stride_in_x: 1,
            stride_in_y: 1,
            dilation_rate_in_x: 1,
            dilation_rate_in_y: 1,
            groups: 1,
            padding_left: padding,
            padding_right: padding,
            padding_top: padding,
            padding_bottom: padding,
            padding_style: padding_style::EXPLICIT,
            data_layout: tensor_named_data_layout::NCHW,
            weights_layout: tensor_named_data_layout::OIHW,
        })
        .expect("conv2d: descriptor");
        let conv = graph
            .convolution2d(x, &w, &descriptor, None)
            .expect("conv2d: op");
        graph.addition(&conv, &b, None).expect("conv2d: bias add")
    }

    /// BatchNorm2d in eval mode: `scale = weight / sqrt(running_var + eps)`,
    /// `shift = bias - running_mean * scale`,
    /// `y = x * scale.reshape(1,C,1,1) + shift.reshape(1,C,1,1)` — composed
    /// from elementary broadcast ops (mirrors `tch_network.rs::batch_norm`).
    fn batch_norm(&self, graph: &Graph, x: &Tensor, prefix: &str) -> Tensor {
        let (wshape, _) = self.get(&format!("{prefix}.weight"));
        let c = wshape[0];
        let weight = self.const_natural(graph, &format!("{prefix}.weight"));
        let bias = self.const_natural(graph, &format!("{prefix}.bias"));
        let rm = self.const_natural(graph, &format!("{prefix}.running_mean"));
        let rv = self.const_natural(graph, &format!("{prefix}.running_var"));

        let eps = graph
            .constant_scalar(BN_EPS, data_type::FLOAT32)
            .expect("bn: eps constant");
        let var_eps = graph.addition(&rv, &eps, None).expect("bn: var+eps");
        let std = graph
            .unary_arithmetic(UnaryArithmeticOp::SquareRoot, &var_eps, None)
            .expect("bn: sqrt");
        let scale = graph.division(&weight, &std, None).expect("bn: scale");
        let rm_scale = graph
            .multiplication(&rm, &scale, None)
            .expect("bn: mean*scale");
        let shift = graph
            .subtraction(&bias, &rm_scale, None)
            .expect("bn: shift");

        let scale_r = graph
            .reshape(&scale, &[1, c, 1, 1], None)
            .expect("bn: reshape scale");
        let shift_r = graph
            .reshape(&shift, &[1, c, 1, 1], None)
            .expect("bn: reshape shift");
        let scaled = graph.multiplication(x, &scale_r, None).expect("bn: x*scale");
        graph
            .addition(&scaled, &shift_r, None)
            .expect("bn: +shift")
    }

    fn res_block(&self, graph: &Graph, x: &Tensor, i: usize) -> Tensor {
        let p = format!("res_blocks.{i}");
        let out = self.conv2d(graph, x, &format!("{p}.c1"), 1);
        let out_bn = self.batch_norm(graph, &out, &format!("{p}.bn1"));
        let out = graph.relu(&out_bn, None).expect("res_block: relu1");
        let out = self.conv2d(graph, &out, &format!("{p}.c2"), 1);
        let out = self.batch_norm(graph, &out, &format!("{p}.bn2"));
        let summed = graph.addition(&out, x, None).expect("res_block: residual add");
        graph.relu(&summed, None).expect("res_block: relu2")
    }

    /// Mirrors `tch_network.rs::cross_attention`: `nn.MultiheadAttention`
    /// (batch_first=True) — q attends to kv, then `LayerNorm(q + attn_out)`.
    /// `q`: [B, Lq, D], `kv`: [B, Lkv, D].
    fn cross_attention(
        &self,
        graph: &Graph,
        q: &Tensor,
        kv: &Tensor,
        b: usize,
        lq: usize,
        lkv: usize,
    ) -> Tensor {
        // nn.MultiheadAttention packs [Wq; Wk; Wv] into in_proj_weight [3D, D].
        let in_w = self.const_natural(graph, "cross_attention.attn.in_proj_weight");
        let in_b = self.const_natural(graph, "cross_attention.attn.in_proj_bias");
        let f = FILTERS as isize;

        let wq = graph.slice(&in_w, 0, 0, f, None).expect("attn: slice wq");
        let wk = graph.slice(&in_w, 0, f, f, None).expect("attn: slice wk");
        let wv = graph.slice(&in_w, 0, 2 * f, f, None).expect("attn: slice wv");
        let bq = graph.slice(&in_b, 0, 0, f, None).expect("attn: slice bq");
        let bk = graph.slice(&in_b, 0, f, f, None).expect("attn: slice bk");
        let bv = graph.slice(&in_b, 0, 2 * f, f, None).expect("attn: slice bv");

        let wqt = graph.transpose(&wq, &[1, 0], None).expect("attn: wq^T");
        let wkt = graph.transpose(&wk, &[1, 0], None).expect("attn: wk^T");
        let wvt = graph.transpose(&wv, &[1, 0], None).expect("attn: wv^T");

        let qp = {
            let mm = graph.matrix_multiplication(q, &wqt, None).expect("attn: q proj mm");
            graph.addition(&mm, &bq, None).expect("attn: q proj +b")
        };
        let kp = {
            let mm = graph.matrix_multiplication(kv, &wkt, None).expect("attn: k proj mm");
            graph.addition(&mm, &bk, None).expect("attn: k proj +b")
        };
        let vp = {
            let mm = graph.matrix_multiplication(kv, &wvt, None).expect("attn: v proj mm");
            graph.addition(&mm, &bv, None).expect("attn: v proj +b")
        };

        // [B, L, D] -> [B, nhead, L, head_dim]
        let to_heads = |t: &Tensor, l: usize| -> Tensor {
            let r = graph
                .reshape(t, &[b, l, NHEAD, HEAD_DIM], None)
                .expect("attn: reshape to heads");
            graph
                .transpose(&r, &[0, 2, 1, 3], None)
                .expect("attn: transpose to heads")
        };
        let qh = to_heads(&qp, lq);
        let kh = to_heads(&kp, lkv);
        let vh = to_heads(&vp, lkv);

        let kht = graph
            .transpose(&kh, &[0, 1, 3, 2], None)
            .expect("attn: k^T");
        let scale = 1.0_f64 / (HEAD_DIM as f64).sqrt();
        let scale_t = graph
            .constant_scalar(scale, data_type::FLOAT32)
            .expect("attn: scale constant");
        let raw_scores = graph
            .matrix_multiplication(&qh, &kht, None)
            .expect("attn: qk^T");
        let scores = graph
            .multiplication(&raw_scores, &scale_t, None)
            .expect("attn: scale scores");
        let attn = graph.softmax(&scores, -1, None).expect("attn: softmax");
        let ctx = graph
            .matrix_multiplication(&attn, &vh, None)
            .expect("attn: attn@v"); // [B, nhead, Lq, head_dim]

        // merge heads -> [B, Lq, D]
        let ctx = graph
            .transpose(&ctx, &[0, 2, 1, 3], None)
            .expect("attn: merge heads transpose");
        let ctx = graph
            .reshape(&ctx, &[b, lq, FILTERS], None)
            .expect("attn: merge heads reshape");
        let attn_out = self.linear(graph, &ctx, "cross_attention.attn.out_proj");

        // CrossAttention: LayerNorm(q + attn_out), eps=1e-5, over last axis D.
        let summed = graph.addition(q, &attn_out, None).expect("attn: q+attn_out");

        let mean = graph
            .mean(&summed, &[2], None)
            .expect("attn: layernorm mean");
        let diff = graph
            .subtraction(&summed, &mean, None)
            .expect("attn: layernorm diff");
        let sq = graph
            .multiplication(&diff, &diff, None)
            .expect("attn: layernorm sq");
        let var = graph.mean(&sq, &[2], None).expect("attn: layernorm var");
        let eps = graph
            .constant_scalar(LN_EPS, data_type::FLOAT32)
            .expect("attn: layernorm eps");
        let var_eps = graph.addition(&var, &eps, None).expect("attn: var+eps");
        let std = graph
            .unary_arithmetic(UnaryArithmeticOp::SquareRoot, &var_eps, None)
            .expect("attn: layernorm sqrt");
        let normed = graph
            .division(&diff, &std, None)
            .expect("attn: layernorm normed");

        let gamma = self.const_shape(graph, "cross_attention.norm.weight", &[1, 1, FILTERS]);
        let beta = self.const_shape(graph, "cross_attention.norm.bias", &[1, 1, FILTERS]);
        let scaled = graph
            .multiplication(&normed, &gamma, None)
            .expect("attn: layernorm *gamma");
        graph
            .addition(&scaled, &beta, None)
            .expect("attn: layernorm +beta")
    }

    /// Create an input `TensorData` on this net's device. Public only so
    /// benchmarks (`examples/metal_bench.rs`) can measure the upload phase
    /// in isolation; not part of the inference API.
    pub fn make_tensor_data(&self, data: &[f32], shape: &[usize]) -> TensorData {
        TensorData::from_f32_slice(&self.device, data, shape)
            .expect("make_tensor_data: TensorData::from_f32_slice")
    }

    /// Trace the full forward pass for a fixed `batch` size and compile it to
    /// an `Executable`. Called once per distinct batch size (see module
    /// docs); the `Graph` and its intermediate `Tensor`s are dropped after
    /// `compile` returns — the compiled `Executable` is a self-contained
    /// artifact and doesn't need the tracing graph to stay alive.
    fn build_and_compile(&self, batch: usize) -> Executable {
        let b = batch;
        let h = MAP_SIZE;
        let w = MAP_SIZE;
        let graph = Graph::new().expect("mpsgraph: Graph::new");

        let spatial_ph = graph
            .placeholder(Some(&[b, NUM_CHANNELS, h, w]), data_type::FLOAT32, Some("spatial"))
            .expect("mpsgraph: spatial placeholder");
        let player_ph = graph
            .placeholder(Some(&[b, PLAYER_DIM]), data_type::FLOAT32, Some("player"))
            .expect("mpsgraph: player placeholder");

        // 1. Spatial backbone
        let mut x = self.conv2d(&graph, &spatial_ph, "conv1", 1);
        let x_bn = self.batch_norm(&graph, &x, "bn1");
        x = graph.relu(&x_bn, None).expect("forward: relu bn1");
        for i in 0..NUM_RES_BLOCKS {
            x = self.res_block(&graph, &x, i);
        }

        // 2. Cross-attention inputs
        // spatial tokens: [B, D, H, W] -> [B, D, H*W] -> [B, H*W, D]
        let spatial_dhw = graph
            .reshape(&x, &[b, FILTERS, h * w], None)
            .expect("forward: reshape spatial tokens");
        let spatial_tokens = graph
            .transpose(&spatial_dhw, &[0, 2, 1], None)
            .expect("forward: transpose spatial tokens");

        // player tokens: player[B,10,1] * embeddings[1,10,D] -> [B,10,D]
        let emb = self.const_shape(
            &graph,
            "player_feature_embeddings",
            &[1, PLAYER_DIM, FILTERS],
        );
        let player_r = graph
            .reshape(&player_ph, &[b, PLAYER_DIM, 1], None)
            .expect("forward: reshape player");
        let player_tokens_raw = graph
            .multiplication(&player_r, &emb, None)
            .expect("forward: player * embeddings");
        let player_tokens_lin = self.linear(&graph, &player_tokens_raw, "player_fc");
        let player_tokens = graph
            .relu(&player_tokens_lin, None)
            .expect("forward: relu player_fc");

        // 3. Cross-attention, back to [B, D, H, W]
        let attended = self.cross_attention(&graph, &spatial_tokens, &player_tokens, b, h * w, PLAYER_DIM);
        let attended_dhw = graph
            .transpose(&attended, &[0, 2, 1], None)
            .expect("forward: transpose attended");
        let x2 = graph
            .reshape(&attended_dhw, &[b, FILTERS, h, w], None)
            .expect("forward: reshape attended to spatial");

        // Policy heads
        let p_pooled_conv = self.conv2d(&graph, &x2, "p_pool_conv", 0);
        let p_pooled_bn = self.batch_norm(&graph, &p_pooled_conv, "p_pool_bn");
        let p_pooled_relu = graph.relu(&p_pooled_bn, None).expect("forward: relu p_pool_bn");
        let (p_pool_w_shape, _) = self.get("p_pool_conv.weight");
        let cp = p_pool_w_shape[0];
        let p_pooled = graph
            .reshape(&p_pooled_relu, &[b, cp * h * w], None)
            .expect("forward: flatten p_pooled");
        let p_latent_lin = self.linear(&graph, &p_pooled, "p_fc_shared");
        let p_latent = graph.relu(&p_latent_lin, None).expect("forward: relu p_fc_shared");
        let action_type = self.linear(&graph, &p_latent, "pi_action"); // [B, 11]
        let move_option = self.linear(&graph, &p_latent, "pi_option"); // [B, 192]

        let source_conv = self.conv2d(&graph, &x2, "pi_source", 0);
        let (source_w_shape, _) = self.get("pi_source.weight");
        let source_c = source_w_shape[0];
        let source_spatial = graph
            .reshape(&source_conv, &[b, source_c * h * w], None)
            .expect("forward: flatten pi_source"); // [B, 121]

        let target_conv = self.conv2d(&graph, &x2, "pi_target", 0);
        let (target_w_shape, _) = self.get("pi_target.weight");
        let target_c = target_w_shape[0];
        let target_spatial = graph
            .reshape(&target_conv, &[b, target_c * h * w], None)
            .expect("forward: flatten pi_target"); // [B, 121]

        // Value head
        let v_pooled_conv = self.conv2d(&graph, &x2, "v_pool_conv", 0);
        let v_pooled_bn = self.batch_norm(&graph, &v_pooled_conv, "v_pool_bn");
        let v_pooled_relu = graph.relu(&v_pooled_bn, None).expect("forward: relu v_pool_bn");
        let (v_pool_w_shape, _) = self.get("v_pool_conv.weight");
        let vp = v_pool_w_shape[0];
        let v_pooled = graph
            .reshape(&v_pooled_relu, &[b, vp * h * w], None)
            .expect("forward: flatten v_pooled");
        let v_latent_lin = self.linear(&graph, &v_pooled, "v_fc_shared");
        let v_latent = graph.relu(&v_latent_lin, None).expect("forward: relu v_fc_shared");
        let win_raw = self.linear(&graph, &v_latent, "v_win");
        let win = graph
            .unary_arithmetic(UnaryArithmeticOp::Tanh, &win_raw, None)
            .expect("forward: tanh win"); // [B, 1]

        // Compile for this fixed batch size. The `Executable` is
        // self-contained; `graph` and its intermediate `Tensor`s (including
        // the placeholders) are dropped when this function returns.
        let feed_descs = [
            FeedDescription {
                tensor: &spatial_ph,
                shape: &[b, NUM_CHANNELS, h, w],
                data_type: data_type::FLOAT32,
            },
            FeedDescription {
                tensor: &player_ph,
                shape: &[b, PLAYER_DIM],
                data_type: data_type::FLOAT32,
            },
        ];
        let targets: [&Tensor; 5] = [&win, &action_type, &source_spatial, &target_spatial, &move_option];
        graph
            .compile(&self.device, &feed_descs, &targets)
            .expect("mpsgraph: graph.compile failed")
    }

    /// Forward a coalesced batch. `spatial_flat` is
    /// `batch * NUM_CHANNELS * MAP_SIZE * MAP_SIZE` row-major, `player_flat` is
    /// `batch * PLAYER_DIM`. Returns per-row `(value, policy)` as CPU floats.
    ///
    /// Compiles once per distinct `batch` size (cached in `self.executables`)
    /// and replays the compiled `Executable` on every call — see module docs.
    pub fn forward_batch(
        &self,
        spatial_flat: &[f32],
        player_flat: &[f32],
        batch: usize,
    ) -> (Vec<f32>, Vec<RawPolicyOutput>) {
        let h = MAP_SIZE;
        let w = MAP_SIZE;

        let spatial_data =
            TensorData::from_f32_slice(&self.device, spatial_flat, &[batch, NUM_CHANNELS, h, w])
                .expect("forward: spatial tensor data");
        let player_data = TensorData::from_f32_slice(&self.device, player_flat, &[batch, PLAYER_DIM])
            .expect("forward: player tensor data");

        let mut executables = self.executables.borrow_mut();
        let executable = executables
            .entry(batch)
            .or_insert_with(|| self.build_and_compile(batch));
        let results = executable
            .run(&self.queue, &[&spatial_data, &player_data])
            .expect("mpsgraph: executable.run failed");

        let win_v = results[0].read_f32().expect("forward: read win");
        let action_v = results[1].read_f32().expect("forward: read action");
        let source_v = results[2].read_f32().expect("forward: read source");
        let target_v = results[3].read_f32().expect("forward: read target");
        let option_v = results[4].read_f32().expect("forward: read option");

        const AT: usize = 11;
        const SS: usize = SPATIAL; // 121
        const TS: usize = SPATIAL; // 121
        const MO: usize = 192;

        let mut values = Vec::with_capacity(batch);
        let mut policy = Vec::with_capacity(batch);
        for i in 0..batch {
            values.push(win_v[i]);
            policy.push(RawPolicyOutput {
                action_type: action_v[i * AT..(i + 1) * AT].to_vec(),
                source_spatial: source_v[i * SS..(i + 1) * SS].to_vec(),
                target_spatial: target_v[i * TS..(i + 1) * TS].to_vec(),
                move_option: option_v[i * MO..(i + 1) * MO].to_vec(),
            });
        }
        (values, policy)
    }
}
