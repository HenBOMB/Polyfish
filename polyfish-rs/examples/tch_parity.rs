//! Parity check: the tch/libtorch PolyZeroNet forward must match the candle
//! forward (already validated against train.py) on identical weights + input.
//!
//! Run (with the tch build env):
//!   cargo run --release --features tch-eval --example tch_parity
//!
//! Checks:
//!   1. candle-CPU vs tch-CPU  — tight (proves the architecture port is exact)
//!   2. tch-MPS   vs tch-CPU   — looser (proves MPS produces the same result)

use polyfish::ai::eval_server::{Evaluator, InlineEvalHandle};
use polyfish::ai::features::{state_to_cpu_features, RawFeatures};
use polyfish::ai::network::{PolyZeroNet, RawPolicyOutput};
use polyfish::ai::tch_network::TchPolyZeroNet;
use polyfish::game::Game;
use std::sync::Arc;

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

/// Compare a candle result against a tch result over value + softmaxed heads.
fn report(tag: &str, c: &[(f32, RawPolicyOutput)], t: &(Vec<f32>, Vec<RawPolicyOutput>)) {
    let mut vmax = 0f32;
    let (mut a, mut s, mut tt, mut o) = (0f32, 0f32, 0f32, 0f32);
    for (i, (cv, cp)) in c.iter().enumerate() {
        vmax = vmax.max((cv - t.0[i]).abs());
        let tp = &t.1[i];
        a = a.max(max_abs(&softmax(&cp.action_type), &softmax(&tp.action_type)));
        s = s.max(max_abs(&softmax(&cp.source_spatial), &softmax(&tp.source_spatial)));
        tt = tt.max(max_abs(&softmax(&cp.target_spatial), &softmax(&tp.target_spatial)));
        o = o.max(max_abs(&softmax(&cp.move_option), &softmax(&tp.move_option)));
    }
    println!(
        "{tag}: max|Δvalue|={vmax:.2e}  max|Δsoftmax| action={a:.2e} source={s:.2e} target={tt:.2e} option={o:.2e}"
    );
}

fn candle_forward(device: &candle_core::Device, feats: &[RawFeatures]) -> Vec<(f32, RawPolicyOutput)> {
    let mut vm = candle_nn::VarMap::new();
    vm.load(MODEL).unwrap();
    let net = Arc::new(
        PolyZeroNet::new(candle_nn::VarBuilder::from_varmap(
            &vm,
            candle_core::DType::F32,
            device,
        ))
        .unwrap(),
    );
    let handle = InlineEvalHandle::new(net);
    let batch: Vec<RawFeatures> = feats
        .iter()
        .map(|f| RawFeatures {
            spatial: f.spatial.clone(),
            player: f.player.clone(),
        })
        .collect();
    match Evaluator::Inline(handle) {
        Evaluator::Inline(h) => h.evaluate(batch),
        _ => unreachable!(),
    }
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

    let candle_cpu = candle_forward(&candle_core::Device::Cpu, &feats);
    let tch_cpu = tch_forward(tch::Device::Cpu, &feats);
    report("candle-CPU vs tch-CPU", &candle_cpu, &tch_cpu);

    // Dump inputs + tch outputs so a Python script can compare against
    // train.py's PolyZeroNet (the ground-truth interpretation of the weights).
    let mut spatial = Vec::new();
    let mut player = Vec::new();
    for f in &feats {
        spatial.extend_from_slice(&f.spatial);
        player.extend_from_slice(&f.player);
    }
    let dump = serde_json::json!({
        "batch": feats.len(),
        "spatial": spatial,
        "player": player,
        "tch_value": tch_cpu.0,
        "tch_action": tch_cpu.1.iter().map(|p| p.action_type.clone()).collect::<Vec<_>>(),
        "tch_source": tch_cpu.1.iter().map(|p| p.source_spatial.clone()).collect::<Vec<_>>(),
        "tch_target": tch_cpu.1.iter().map(|p| p.target_spatial.clone()).collect::<Vec<_>>(),
        "tch_option": tch_cpu.1.iter().map(|p| p.move_option.clone()).collect::<Vec<_>>(),
    });
    std::fs::write("/tmp/parity_rust.json", serde_json::to_string(&dump).unwrap()).unwrap();
    println!("wrote /tmp/parity_rust.json for PyTorch comparison");

    if tch::utils::has_mps() {
        let tch_mps = tch_forward(tch::Device::Mps, &feats);
        // reuse report by wrapping tch_cpu as the "candle" side
        let cpu_as_ref: Vec<(f32, RawPolicyOutput)> =
            tch_cpu.0.iter().cloned().zip(tch_cpu.1.iter().cloned()).collect();
        report("tch-CPU   vs tch-MPS", &cpu_as_ref, &tch_mps);
    } else {
        println!("(MPS not available — skipped MPS parity)");
    }
}
