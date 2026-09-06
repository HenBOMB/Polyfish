//! EXP_ELO_131 Phase 1: a standalone candle port of `train_ply_ranker.py`'s
//! `TrunkNet` — a decomposed-head ply ranker trained to reproduce
//! `rank_plies`'s `score_move + lambda*dphi_full` ranking without the
//! per-candidate `simulate_move`/`goal_potential_with_belief`/undo pass.
//!
//! Deliberately NOT a new head on [`crate::ai::network::PolyZeroNet`] — two
//! rungs were tried and rejected first (see `hypothesis_driven_improvements.md`,
//! EXP_ELO_131 "Rung B"/"Rung A2"): the existing production policy heads
//! under-perform this dedicated net by ~2x and regress on Attack, and
//! fine-tuning them is capped by `p_pool_conv`'s 1-channel pool, an
//! architectural ceiling, not a training-recipe fix. This net's own
//! `action_head`/`option_head` pool the FULL 64-dim trunk (global average
//! pool), matching `TrunkNet`'s own shapes exactly.
//!
//! Own weights file (`ply_ranker.safetensors`), own tiny forward pass, called
//! directly on the actor thread — `rank_plies` is synchronous CPU code
//! called up to ~8*(1+k) times per real turn, not routed through the batched
//! `Evaluator`/`eval_server` built for the ~20x-larger main net.

use candle_core::{Device, Result, Tensor};
use candle_nn::{Conv2d, GroupNorm, Linear, Module, VarBuilder};

use crate::ai::features::RawFeatures;
use crate::ai::network::RawPolicyOutput;

const FILTERS: usize = 64;
const NUM_BLOCKS: usize = 4;
const NUM_ACTION_TYPES: usize = 11;
const NUM_OPTIONS: usize = 192;

fn conv3x3(in_c: usize, out_c: usize, vs: VarBuilder) -> Result<Conv2d> {
    let config = candle_nn::Conv2dConfig {
        padding: 1,
        ..Default::default()
    };
    candle_nn::conv2d(in_c, out_c, 3, config, vs)
}

fn conv1x1(in_c: usize, out_c: usize, vs: VarBuilder) -> Result<Conv2d> {
    candle_nn::conv2d(in_c, out_c, 1, Default::default(), vs)
}

fn group_norm(c: usize, vs: VarBuilder) -> Result<GroupNorm> {
    candle_nn::group_norm(8, c, 1e-5, vs)
}

/// `train_ply_ranker.py`'s per-block `Sequential(Conv-GN-ReLU-Conv-GN)`,
/// residual-added and ReLU'd by the caller — same split as
/// `network::ResBlock`, duplicated here rather than shared since this is a
/// deliberately separate, standalone net (see module doc).
struct RankerBlock {
    c1: Conv2d,
    gn1: GroupNorm,
    c2: Conv2d,
    gn2: GroupNorm,
}

impl RankerBlock {
    fn new(vs: VarBuilder) -> Result<Self> {
        Ok(Self {
            c1: conv3x3(FILTERS, FILTERS, vs.pp("0"))?,
            gn1: group_norm(FILTERS, vs.pp("1"))?,
            c2: conv3x3(FILTERS, FILTERS, vs.pp("3"))?,
            gn2: group_norm(FILTERS, vs.pp("4"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let ys = self.c1.forward(xs)?;
        let ys = self.gn1.forward(&ys)?;
        let ys = ys.relu()?;
        let ys = self.c2.forward(&ys)?;
        self.gn2.forward(&ys)
    }
}

/// Standalone ply-ranker: mirrors `train_ply_ranker.py::TrunkNet` exactly
/// (stem conv -> GN -> ReLU -> +player_proj broadcast -> 4x residual block
/// -> global-average-pool -> action/option heads off the pool, source/target
/// heads off the pre-pool trunk).
pub struct PlyRanker {
    stem: Conv2d,
    stem_gn: GroupNorm,
    player_proj: Linear,
    blocks: Vec<RankerBlock>,
    action_head: Linear,
    option_head: Linear,
    source_head: Conv2d,
    target_head: Conv2d,
    device: Device,
}

impl PlyRanker {
    pub fn new(vs: VarBuilder, n_channels: usize, player_dim: usize) -> Result<Self> {
        let stem = conv3x3(n_channels, FILTERS, vs.pp("stem"))?;
        let stem_gn = group_norm(FILTERS, vs.pp("stem_gn"))?;
        let player_proj = candle_nn::linear(player_dim, FILTERS, vs.pp("player_proj"))?;
        let mut blocks = Vec::with_capacity(NUM_BLOCKS);
        for i in 0..NUM_BLOCKS {
            blocks.push(RankerBlock::new(vs.pp(format!("blocks.{i}")))?);
        }
        let action_head = candle_nn::linear(FILTERS, NUM_ACTION_TYPES, vs.pp("action_head"))?;
        let option_head = candle_nn::linear(FILTERS, NUM_OPTIONS, vs.pp("option_head"))?;
        let source_head = conv1x1(FILTERS, 1, vs.pp("source_head"))?;
        let target_head = conv1x1(FILTERS, 1, vs.pp("target_head"))?;
        Ok(Self {
            stem,
            stem_gn,
            player_proj,
            blocks,
            action_head,
            option_head,
            source_head,
            target_head,
            device: vs.device().clone(),
        })
    }

    /// Load from a standalone `ply_ranker.safetensors` file. Returns `Ok(None)`
    /// (not an error) when the file doesn't exist -- a missing ranker is a
    /// clean fallback to the CPU `rank_plies` path, not a startup failure.
    pub fn load_optional(path: &str, n_channels: usize, player_dim: usize) -> Result<Option<Self>> {
        if !std::path::Path::new(path).exists() {
            return Ok(None);
        }
        let vs = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                &[path],
                candle_core::DType::F32,
                &Device::Cpu,
            )?
        };
        Ok(Some(Self::new(vs, n_channels, player_dim)?))
    }

    /// One forward pass, batch size 1 -- `rank_plies` is a per-ply synchronous
    /// call, not a batched actor path (see module doc).
    pub fn forward_raw(&self, features: &RawFeatures) -> Result<RawPolicyOutput> {
        let n_channels = features.spatial.len() / (crate::ai::features::MAP_SIZE * crate::ai::features::MAP_SIZE);
        let spatial = Tensor::from_slice(
            &features.spatial,
            (1, n_channels, crate::ai::features::MAP_SIZE, crate::ai::features::MAP_SIZE),
            &self.device,
        )?;
        let player = Tensor::from_slice(&features.player, (1, features.player.len()), &self.device)?;

        let mut x = self.stem.forward(&spatial)?;
        x = self.stem_gn.forward(&x)?;
        x = x.relu()?;

        // TrunkNet applies no activation to player_proj before the
        // broadcast-add (unlike PolyZeroNet's player_fc, which does).
        let p = self.player_proj.forward(&player)?;
        let p = p.reshape((1, FILTERS, 1, 1))?;
        x = x.broadcast_add(&p)?;

        for block in &self.blocks {
            let ys = block.forward(&x)?;
            x = (x + ys)?.relu()?;
        }

        let pooled = x.mean(3)?.mean(2)?; // [1, C, H, W] -> [1, C]

        let action_type = self.action_head.forward(&pooled)?.flatten_all()?.to_vec1::<f32>()?;
        let move_option = self.option_head.forward(&pooled)?.flatten_all()?.to_vec1::<f32>()?;
        let source_spatial = self.source_head.forward(&x)?.flatten_all()?.to_vec1::<f32>()?;
        let target_spatial = self.target_head.forward(&x)?.flatten_all()?.to_vec1::<f32>()?;

        Ok(RawPolicyOutput {
            action_type,
            source_spatial,
            target_spatial,
            move_option,
            fog: None,
            macro_stance: None,
            macro_order: None,
            rollout_value: None,
        })
    }
}

/// Default path checked when `POLYFISH_PLY_RANKER` is unset -- mirrors
/// `model.safetensors`'s own implicit-load-from-cwd convention
/// (`self_play/main.rs`'s `p1_path`/`p2_path` defaults). A missing file at
/// this path is silently `None` (see `load_optional`), so this is safe to
/// check unconditionally: no checkpoint present means no behavior change.
const DEFAULT_PLY_RANKER_PATH: &str = "ply_ranker.safetensors";

/// Process-wide optional ranker. `POLYFISH_PLY_RANKER=<path>` picks an
/// explicit checkpoint; `POLYFISH_PLY_RANKER=0` disables the ranker
/// entirely (matches `POLYFISH_ENDTURN_HARD_GATE`'s convention) even if
/// `ply_ranker.safetensors` exists in cwd; unset checks for
/// `ply_ranker.safetensors` in cwd by default. Same env-var-gated-
/// `OnceLock` idiom as `micro_mcts_params`/`dphi_probe_path`, chosen over
/// threading an `Option<&PlyRanker>` through `rank_plies`/`rank_view`/
/// `execute_turn`/every one of their callers: it keeps every existing call
/// site (including every test) byte-identical whenever no checkpoint is
/// present. `None` on a missing file OR a load error (logged once) -- never
/// a hard failure, since a missing ranker must fall back to the CPU path.
pub fn ply_ranker() -> Option<&'static PlyRanker> {
    static RANKER: std::sync::OnceLock<Option<PlyRanker>> = std::sync::OnceLock::new();
    RANKER
        .get_or_init(|| {
            let path = match std::env::var("POLYFISH_PLY_RANKER") {
                Ok(v) if v == "0" => return None,
                Ok(v) => v,
                Err(_) => DEFAULT_PLY_RANKER_PATH.to_string(),
            };
            match PlyRanker::load_optional(
                &path,
                crate::ai::features::NUM_CHANNELS,
                RawFeatures::PLAYER_STATE_DIM,
            ) {
                Ok(Some(ranker)) => {
                    eprintln!("[ply_ranker] loaded {path}");
                    Some(ranker)
                }
                Ok(None) => None,
                Err(e) => {
                    eprintln!("[ply_ranker] failed to load {path}: {e}, falling back to CPU rank_plies");
                    None
                }
            }
        })
        .as_ref()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// EXP_ELO_131 Phase 1: cross-checks this Rust port against the exact
    /// logits `train_ply_ranker.py`'s TrunkNet produced from the SAME
    /// checkpoint on the SAME inputs (see `golden_npz_to_bin.py`) -- the
    /// only thing that catches a silent shape/transposition bug before it
    /// ever reaches self_play. Skipped (not failed) when the env-provided
    /// weights/golden files aren't present, since both are session-local
    /// offline artifacts, not checked-in fixtures.
    #[test]
    fn matches_python_trunknet_golden_vectors() {
        let Ok(weights) = std::env::var("POLYFISH_PLY_RANKER_TEST_WEIGHTS") else {
            eprintln!("skipping: POLYFISH_PLY_RANKER_TEST_WEIGHTS not set");
            return;
        };
        let Ok(golden) = std::env::var("POLYFISH_PLY_RANKER_TEST_GOLDEN") else {
            eprintln!("skipping: POLYFISH_PLY_RANKER_TEST_GOLDEN not set");
            return;
        };
        let ranker = PlyRanker::load_optional(&weights, crate::ai::features::NUM_CHANNELS, RawFeatures::PLAYER_STATE_DIM)
            .expect("load ply_ranker.safetensors")
            .expect("weights file must exist when this test is enabled");

        let bytes = std::fs::read(&golden).expect("read golden vectors file");
        let mut off = 0usize;
        let read_u32 = |bytes: &[u8], off: &mut usize| -> u32 {
            let v = u32::from_le_bytes(bytes[*off..*off + 4].try_into().unwrap());
            *off += 4;
            v
        };
        let read_f32_vec = |bytes: &[u8], off: &mut usize, n: usize| -> Vec<f32> {
            let v = bytes[*off..*off + 4 * n]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            *off += 4 * n;
            v
        };

        let mut n_examples = 0;
        let mut max_abs_diff = 0.0f32;
        while off < bytes.len() {
            let n_channels = read_u32(&bytes, &mut off) as usize;
            let spatial = read_f32_vec(&bytes, &mut off, n_channels * 121);
            let player = read_f32_vec(&bytes, &mut off, 10);
            let want_action = read_f32_vec(&bytes, &mut off, 11);
            let want_source = read_f32_vec(&bytes, &mut off, 121);
            let want_target = read_f32_vec(&bytes, &mut off, 121);
            let want_option = read_f32_vec(&bytes, &mut off, 192);

            let raw = ranker
                .forward_raw(&RawFeatures { spatial, player })
                .expect("forward_raw");

            for (name, got, want) in [
                ("action", &raw.action_type, &want_action),
                ("source", &raw.source_spatial, &want_source),
                ("target", &raw.target_spatial, &want_target),
                ("option", &raw.move_option, &want_option),
            ] {
                assert_eq!(got.len(), want.len(), "{name}: length mismatch");
                for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
                    let diff = (g - w).abs();
                    max_abs_diff = max_abs_diff.max(diff);
                    assert!(
                        diff < 1e-2,
                        "{name}[{i}] example {n_examples}: rust={g} python={w} diff={diff}"
                    );
                }
            }
            n_examples += 1;
        }
        assert!(n_examples > 0, "golden vectors file was empty");
        eprintln!("[ply_ranker] {n_examples} golden examples matched, max_abs_diff={max_abs_diff:.6}");
    }
}
