//! Parity check: the MPSGraph (metal-eval) PolyZeroNet forward must match
//! the tch/libtorch-CPU forward (already validated against train.py) on
//! identical weights + input. This is the Stage 0 gate from the
//! bypass-libtorch plan — nothing else starts until parity holds.
//!
//! Run (with the tch build env, since ground truth is tch-CPU):
//!   export PATH="$PWD/.venv/bin:$PATH"
//!   export LIBTORCH_USE_PYTORCH=1
//!   export LIBTORCH_BYPASS_VERSION_CHECK=1
//!   export DYLD_LIBRARY_PATH="$PWD/.venv/lib/python3.13/site-packages/torch/lib"
//!   cargo run --release --features "tch-eval metal-eval" --example metal_parity

use polyfish::ai::features::{state_to_cpu_features, RawFeatures};
use polyfish::ai::metal_network::MetalPolyZeroNet;
use polyfish::ai::network::RawPolicyOutput;
use polyfish::ai::tch_network::TchPolyZeroNet;
use polyfish::game::Game;

const MODEL: &str = "model.safetensors";

fn softmax(v: &[f32]) -> Vec<f32> {
    let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let ex: Vec<f32> = v.iter().map(|x| (x - m).exp()).collect();
    let s: f32 = ex.iter().sum();
    ex.iter().map(|x| x / s).collect()
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

/// Compare a tch (ground truth) result against a metal result over value +
/// softmaxed policy heads.
fn report(tag: &str, t: &(Vec<f32>, Vec<RawPolicyOutput>), m: &(Vec<f32>, Vec<RawPolicyOutput>)) {
    let mut vmax = 0f32;
    let (mut a, mut s, mut tt, mut o) = (0f32, 0f32, 0f32, 0f32);
    for i in 0..t.0.len() {
        vmax = vmax.max((t.0[i] - m.0[i]).abs());
        let tp = &t.1[i];
        let mp = &m.1[i];
        a = a.max(max_abs(&softmax(&tp.action_type), &softmax(&mp.action_type)));
        s = s.max(max_abs(&softmax(&tp.source_spatial), &softmax(&mp.source_spatial)));
        tt = tt.max(max_abs(&softmax(&tp.target_spatial), &softmax(&mp.target_spatial)));
        o = o.max(max_abs(&softmax(&tp.move_option), &softmax(&mp.move_option)));
    }
    println!(
        "{tag}: max|Δvalue|={vmax:.2e}  max|Δsoftmax| action={a:.2e} source={s:.2e} target={tt:.2e} option={o:.2e}"
    );
}

fn tch_forward(device: tch::Device, feats: &[RawFeatures]) -> (Vec<f32>, Vec<RawPolicyOutput>) {
    let net = TchPolyZeroNet::load(MODEL, device).unwrap();
    let mut spatial = Vec::new();
    let mut player = Vec::new();
    for f in feats {
        spatial.extend_from_slice(&f.spatial);
        player.extend_from_slice(&f.player);
    }
    net.forward_batch(&spatial, &player, feats.len())
}

fn metal_forward(feats: &[RawFeatures]) -> (Vec<f32>, Vec<RawPolicyOutput>) {
    let net = MetalPolyZeroNet::load(MODEL).unwrap();
    let mut spatial = Vec::new();
    let mut player = Vec::new();
    for f in feats {
        spatial.extend_from_slice(&f.spatial);
        player.extend_from_slice(&f.player);
    }
    net.forward_batch(&spatial, &player, feats.len())
}

fn main() {
    // A few distinct leaf states for a real (not all-zero) input distribution.
    let mut feats = Vec::new();
    for seed in [1i64, 2, 3, 4] {
        let mut game = Game::new();
        game.state = polyfish::mapgen::generate(polyfish::mapgen::MapGenSettings {
            size: polyfish::types::MapSize::Tiny,
            map_type: polyfish::types::MapType::Drylands,
            tribes: vec![polyfish::types::TribeType::Imperius, polyfish::types::TribeType::Bardur],
            seed,
            ..Default::default()
        });
        game.post_load();
        for pov in [1, 2] {
            feats.push(state_to_cpu_features(&game.state, pov).unwrap());
        }
    }
    println!("batch size: {}", feats.len());

    let tch_cpu = tch_forward(tch::Device::Cpu, &feats);
    let metal = metal_forward(&feats);
    report("tch-CPU vs metal-MPSGraph", &tch_cpu, &metal);
}
