//! libtorch/MPS inference path for PolyZeroNet (behind the `tch-eval` feature).
//!
//! Candle's Metal kernels run this network ~19x slower than Apple's MPS
//! kernels (measured: ~94ms vs ~5ms per batch-128 forward). This module
//! routes the self-play eval-server's inference through libtorch (via tch-rs)
//! so it dispatches Apple's MPS kernels instead.
//!
//! It does NOT use tch's `nn` module builders. It loads the raw PyTorch
//! state_dict from `model.safetensors` into a name->tensor map and runs an
//! explicit functional forward that mirrors `train.py`'s `PolyZeroNet.forward`
//! exactly (same op order, same `nn.MultiheadAttention` semantics). This keeps
//! the weights byte-identical to what training writes and avoids depending on
//! tch exposing every high-level module.
//!
//! The produced `(f32 value, RawPolicyOutput)` rows are plain CPU data, so the
//! eval-server's thread-ownership invariant (only the eval thread touches the
//! device) is preserved exactly as with the candle path.

use crate::ai::features::{MAP_SIZE, NUM_CHANNELS};
use crate::ai::network::RawPolicyOutput;
use std::collections::HashMap;
use tch::{Device, Kind, Tensor};

const FILTERS: i64 = 64;
const NHEAD: i64 = 4;
const HEAD_DIM: i64 = FILTERS / NHEAD; // 16
const PLAYER_DIM: i64 = 16;
const SPATIAL: i64 = (MAP_SIZE * MAP_SIZE) as i64; // 121
const BN_EPS: f64 = 1e-5;
const LN_EPS: f64 = 1e-5;
const NUM_RES_BLOCKS: usize = 6;

/// PolyZeroNet inference on a libtorch device (MPS on Mac). Owns the weight
/// tensors on that device; not `Send`/`Sync`-safe to *use* across threads, so
/// it lives on the single eval-server thread just like the candle net.
pub struct TchPolyZeroNet {
    w: HashMap<String, Tensor>,
    device: Device,
}

impl TchPolyZeroNet {
    /// Load `model.safetensors` (a PyTorch state_dict) onto `device`.
    pub fn load(path: &str, device: Device) -> anyhow::Result<Self> {
        if !std::path::Path::new(path).exists() {
            anyhow::bail!("Model file {path} not found (run init_model.py first)");
        }
        let named = Tensor::read_safetensors(path)?;
        
        // Reject older checkpoints that used BatchNorm to avoid subtle eval bugs
        if named.keys().any(|k| k.contains("running_mean")) {
            anyhow::bail!("Rejecting BatchNorm-era checkpoint (found running_mean)");
        }

        let mut w = HashMap::with_capacity(named.len());
        for (name, tensor) in named {
            // Move each parameter onto the inference device once, up front.
            w.insert(name, tensor.to_device(device).to_kind(Kind::Float));
        }
        Ok(Self { w, device })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    fn get(&self, name: &str) -> &Tensor {
        self.w
            .get(name)
            .unwrap_or_else(|| panic!("BUG: missing weight {name} in model.safetensors"))
    }

    /// PyTorch `nn.Linear`: `y = x @ W^T + b`, W is `[out, in]`. Broadcasts
    /// over any leading batch dims of `x`.
    fn linear(&self, x: &Tensor, prefix: &str) -> Tensor {
        let wt = self.get(&format!("{prefix}.weight")).transpose(0, 1);
        let b = self.get(&format!("{prefix}.bias"));
        x.matmul(&wt) + b
    }

    fn conv2d(&self, x: &Tensor, prefix: &str, padding: i64) -> Tensor {
        let w = self.get(&format!("{prefix}.weight"));
        let b = self.get(&format!("{prefix}.bias"));
        x.conv2d(w, Some(b), [1, 1], [padding, padding], [1, 1], 1)
    }

    /// BatchNorm2d in eval mode: uses stored running stats (no batch stats).
    fn batch_norm(&self, x: &Tensor, prefix: &str) -> Tensor {
        let weight = self.get(&format!("{prefix}.weight"));
        let bias = self.get(&format!("{prefix}.bias"));
        let rm = self.get(&format!("{prefix}.running_mean"));
        let rv = self.get(&format!("{prefix}.running_var"));
        let c = weight.size()[0];
        let shape = [1, c, 1, 1];
        let scale = (weight / (rv + BN_EPS).sqrt()).view(shape);
        let shift = (bias - rm * &scale.view([c])).view(shape);
        x * scale + shift
    }

    fn group_norm(&self, x: &Tensor, prefix: &str) -> Tensor {
        let weight = self.get(&format!("{prefix}.weight"));
        let bias = self.get(&format!("{prefix}.bias"));
        x.group_norm(8, Some(weight), Some(bias), 1e-5, true)
    }

    fn res_block(&self, x: &Tensor, i: usize) -> Tensor {
        let p = format!("res_blocks.{i}");
        let residual = x.shallow_clone();
        let out = self.conv2d(x, &format!("{p}.c1"), 1);
        let out = self.group_norm(&out, &format!("{p}.gn1")).relu();
        let out = self.conv2d(&out, &format!("{p}.c2"), 1);
        let out = self.group_norm(&out, &format!("{p}.gn2"));
        (out + residual).relu()
    }

    /// Mirrors `train.py`'s `CrossAttention` wrapping `nn.MultiheadAttention`
    /// (batch_first=True): q attends to kv, then `LayerNorm(q + attn_out)`.
    /// `q`: [B, Lq, D], `kv`: [B, Lkv, D].
    fn cross_attention(&self, q: &Tensor, kv: &Tensor) -> Tensor {
        let (b, lq) = (q.size()[0], q.size()[1]);
        let lkv = kv.size()[1];

        // nn.MultiheadAttention packs [Wq; Wk; Wv] into in_proj_weight [3D, D].
        let in_w = self.get("cross_attention.attn.in_proj_weight");
        let in_b = self.get("cross_attention.attn.in_proj_bias");
        let wq = in_w.narrow(0, 0, FILTERS);
        let wk = in_w.narrow(0, FILTERS, FILTERS);
        let wv = in_w.narrow(0, 2 * FILTERS, FILTERS);
        let bq = in_b.narrow(0, 0, FILTERS);
        let bk = in_b.narrow(0, FILTERS, FILTERS);
        let bv = in_b.narrow(0, 2 * FILTERS, FILTERS);

        let qp = q.matmul(&wq.transpose(0, 1)) + bq; // [B, Lq, D]
        let kp = kv.matmul(&wk.transpose(0, 1)) + bk; // [B, Lkv, D]
        let vp = kv.matmul(&wv.transpose(0, 1)) + bv; // [B, Lkv, D]

        // [B, L, D] -> [B, nhead, L, head_dim]
        let to_heads = |t: &Tensor, l: i64| {
            t.reshape([b, l, NHEAD, HEAD_DIM])
                .transpose(1, 2)
                .contiguous()
        };
        let qh = to_heads(&qp, lq);
        let kh = to_heads(&kp, lkv);
        let vh = to_heads(&vp, lkv);

        let scale = 1.0 / (HEAD_DIM as f64).sqrt();
        let scores = qh.matmul(&kh.transpose(-2, -1)) * scale; // [B, nhead, Lq, Lkv]
        let attn = scores.softmax(-1, Kind::Float);
        let ctx = attn.matmul(&vh); // [B, nhead, Lq, head_dim]

        // merge heads -> [B, Lq, D]
        let ctx = ctx.transpose(1, 2).contiguous().reshape([b, lq, FILTERS]);
        let attn_out = self.linear(&ctx, "cross_attention.attn.out_proj");

        // CrossAttention: LayerNorm(q + attn_out)
        let summed = q + attn_out;
        let nw = self.get("cross_attention.norm.weight");
        let nb = self.get("cross_attention.norm.bias");
        summed.layer_norm([FILTERS], Some(nw), Some(nb), LN_EPS, false)
    }

    /// Forward a coalesced batch. `spatial_flat` is
    /// `batch * NUM_CHANNELS * MAP_SIZE * MAP_SIZE` row-major, `player_flat` is
    /// `batch * PLAYER_DIM`. Returns per-row `(value, policy)` as CPU floats.
    pub fn forward_batch(
        &self,
        spatial_flat: &[f32],
        player_flat: &[f32],
        batch: usize,
    ) -> (Vec<f32>, Vec<RawPolicyOutput>) {
        let b = batch as i64;
        let _no_grad = tch::no_grad_guard();

        let spatial = Tensor::from_slice(spatial_flat)
            .reshape([b, NUM_CHANNELS as i64, MAP_SIZE as i64, MAP_SIZE as i64])
            .to_device(self.device);
        let player = Tensor::from_slice(player_flat)
            .reshape([b, PLAYER_DIM])
            .to_device(self.device);

        // 1. Spatial backbone
        let mut x = self.conv2d(&spatial, "conv1", 1);
        x = self.group_norm(&x, "gn1").relu();
        for i in 0..NUM_RES_BLOCKS {
            x = self.res_block(&x, i);
        }

        // 2. Cross-attention inputs
        // spatial tokens: [B, D, H, W] -> [B, H*W, D]
        let spatial_tokens = x.flatten(2, 3).transpose(1, 2);
        // player tokens: player[B,10,1] * embeddings[1,10,D] -> [B,10,D]
        let emb = self.get("player_feature_embeddings").unsqueeze(0);
        let pos_emb = self.get("player_pos_embeddings").unsqueeze(0);
        let player_tokens = player.unsqueeze(-1) * emb + pos_emb;
        let player_tokens = self.linear(&player_tokens, "player_fc").relu();

        // 3. Cross-attention, back to [B, D, H, W]
        let attended = self.cross_attention(&spatial_tokens, &player_tokens);
        let x = attended.transpose(1, 2).contiguous().view([
            b,
            FILTERS,
            MAP_SIZE as i64,
            MAP_SIZE as i64,
        ]);

        // Policy heads
        let p_pooled = self.conv2d(&x, "p_pool_conv", 0);
        let p_pooled = self.batch_norm(&p_pooled, "p_pool_bn").relu().flatten(1, 3);
        let p_latent = self.linear(&p_pooled, "p_fc_shared").relu();
        let action_type = self.linear(&p_latent, "pi_action"); // [B, 11]
        let move_option = self.linear(&p_latent, "pi_option"); // [B, 192]
        let source_spatial = self.conv2d(&x, "pi_source", 0).flatten(1, 3); // [B, 121]
        let target_spatial = self.conv2d(&x, "pi_target", 0).flatten(1, 3); // [B, 121]

        // Value head (EXP_ARCH_001): global mean+max pool over the full trunk
        // -> MLP. Mirrors network.rs / train.py.
        let v_mean = x.mean_dim([2i64, 3].as_slice(), false, Kind::Float); // [B, FILTERS]
        let v_max = x.amax([2i64, 3].as_slice(), false); // [B, FILTERS]
        let v_feat = Tensor::cat(&[&v_mean, &v_max], 1); // [B, 2*FILTERS]
        let v_latent = self.linear(&v_feat, "v_fc1").relu();
        let v_latent = self.linear(&v_latent, "v_fc2").relu();
        let win = self.linear(&v_latent, "v_win"); // [B, 1]

        // Read value + 4 policy heads back to CPU in a SINGLE device->CPU
        // copy. Each .to_device(Cpu) on MPS forces a commit +
        // waitUntilCompleted, so per-head readbacks stall the stream ~5x per
        // forward. Concat on-device into one [B, 446] row, one readback, then
        // split by static offset on the CPU. Bit-identical to the per-head
        // path (cat/slice copy bytes, no op changes).
        const V: usize = 1;
        const AT: usize = 11;
        const SS: usize = SPATIAL as usize; // 121
        const TS: usize = SPATIAL as usize; // 121
        const MO: usize = 192;
        const ROW: usize = V + AT + SS + TS + MO; // 446

        let row = Tensor::cat(
            &[
                &win,
                &action_type,
                &source_spatial,
                &target_spatial,
                &move_option,
            ],
            1,
        ); // [B, 446], on-device
        let flat: Vec<f32> = row.flatten(0, 1).to_device(Device::Cpu).try_into().unwrap(); // single sync point

        let mut values = Vec::with_capacity(batch);
        let mut policy = Vec::with_capacity(batch);
        for i in 0..batch {
            let base = i * ROW;
            values.push(flat[base]); // win, len 1
            let at_off = base + V;
            let action_type = flat[at_off..at_off + AT].to_vec();
            let ss_off = at_off + AT;
            let source_spatial = flat[ss_off..ss_off + SS].to_vec();
            let ts_off = ss_off + SS;
            let target_spatial = flat[ts_off..ts_off + TS].to_vec();
            let mo_off = ts_off + TS;
            let move_option = flat[mo_off..mo_off + MO].to_vec();
            policy.push(RawPolicyOutput {
                action_type,
                source_spatial,
                target_spatial,
                move_option,
            });
        }
        (values, policy)
    }
}
