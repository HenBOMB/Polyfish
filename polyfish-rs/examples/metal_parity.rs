//! Parity check: the MPSGraph (metal-eval) PolyZeroNet forward must match
//! the tch/libtorch-CPU forward (already validated against train.py) on
//! identical weights + input. This is the Stage 0 gate from the
//! bypass-libtorch plan — nothing else starts until parity holds.
//!
//! Exits non-zero on any mismatch, so it can gate an architecture change.
//!
//! Run (with the tch build env, since ground truth is tch-CPU):
//!   export PATH="$PWD/.venv/bin:$PATH"
//!   export LIBTORCH_USE_PYTORCH=1
//!   export LIBTORCH_BYPASS_VERSION_CHECK=1
//!   export DYLD_LIBRARY_PATH="$PWD/.venv/lib/python3.13/site-packages/torch/lib"
//!   cargo run --release --features "tch-eval metal-eval" --example metal_parity
//!
//! Checks:
//!   1. model.safetensors carries every tensor the Rust backends load, at the
//!      shapes the current constants imply (train.py's extra training-only
//!      `aux_*` heads are ignored by design)
//!   2. every backend returns the head widths the mapper expects, all finite
//!   3. tch-CPU vs metal-MPSGraph at several batch sizes — each size compiles
//!      its own executable, so they are separately fallible

use polyfish::ai::features::{state_to_cpu_features, RawFeatures, MAP_SIZE, NUM_CHANNELS};
use polyfish::ai::mapper::NUM_MOVE_OPTIONS;
use polyfish::ai::metal_network::MetalPolyZeroNet;
use polyfish::ai::network::{RawPolicyOutput, NUM_ACTION_TYPES};
use polyfish::ai::tch_network::TchPolyZeroNet;
use polyfish::game::Game;
use safetensors::SafeTensors;

const MODEL: &str = "model.safetensors";
const FILTERS: usize = 64;
const RES_BLOCKS: usize = 6;

/// Max |Δ| tolerated between the CPU reference and the GPU graph, on the value
/// and on every raw policy logit (raw, not softmaxed: softmax is
/// shift-invariant and would hide a constant logit offset).
const TOL_DEVICE: f32 = 5e-3;

type Batch = (Vec<f32>, Vec<RawPolicyOutput>);

/// Every tensor the Rust backends load, at the shape the current constants
/// imply. Anything else in the checkpoint is extra and ignored.
fn expected_shapes() -> Vec<(String, Vec<usize>)> {
    let hw = MAP_SIZE * MAP_SIZE;
    let p = RawFeatures::PLAYER_STATE_DIM;
    let mut want: Vec<(String, Vec<usize>)> = vec![
        ("conv1.weight".into(), vec![FILTERS, NUM_CHANNELS, 3, 3]),
        ("conv1.bias".into(), vec![FILTERS]),
        ("gn1.weight".into(), vec![FILTERS]),
        ("gn1.bias".into(), vec![FILTERS]),
        (
            "cross_attention.attn.in_proj_weight".into(),
            vec![3 * FILTERS, FILTERS],
        ),
        ("cross_attention.attn.in_proj_bias".into(), vec![3 * FILTERS]),
        (
            "cross_attention.attn.out_proj.weight".into(),
            vec![FILTERS, FILTERS],
        ),
        ("cross_attention.attn.out_proj.bias".into(), vec![FILTERS]),
        ("cross_attention.norm.weight".into(), vec![FILTERS]),
        ("cross_attention.norm.bias".into(), vec![FILTERS]),
        ("player_feature_embeddings".into(), vec![p, FILTERS]),
        ("player_pos_embeddings".into(), vec![p, FILTERS]),
        ("player_fc.weight".into(), vec![FILTERS, FILTERS]),
        ("player_fc.bias".into(), vec![FILTERS]),
        ("p_pool_conv.weight".into(), vec![1, FILTERS, 1, 1]),
        ("p_pool_conv.bias".into(), vec![1]),
        ("p_fc_shared.weight".into(), vec![FILTERS, hw]),
        ("p_fc_shared.bias".into(), vec![FILTERS]),
        ("pi_action.weight".into(), vec![NUM_ACTION_TYPES, FILTERS]),
        ("pi_action.bias".into(), vec![NUM_ACTION_TYPES]),
        ("pi_option.weight".into(), vec![NUM_MOVE_OPTIONS, FILTERS]),
        ("pi_option.bias".into(), vec![NUM_MOVE_OPTIONS]),
        ("pi_source.weight".into(), vec![1, FILTERS, 1, 1]),
        ("pi_source.bias".into(), vec![1]),
        ("pi_target.weight".into(), vec![1, FILTERS, 1, 1]),
        ("pi_target.bias".into(), vec![1]),
        ("v_fc1.weight".into(), vec![FILTERS, 2 * FILTERS]),
        ("v_fc1.bias".into(), vec![FILTERS]),
        ("v_fc2.weight".into(), vec![FILTERS, FILTERS]),
        ("v_fc2.bias".into(), vec![FILTERS]),
        ("v_win.weight".into(), vec![1, FILTERS]),
        ("v_win.bias".into(), vec![1]),
    ];
    for i in 0..RES_BLOCKS {
        for (leaf, shape) in [
            ("c1.weight", vec![FILTERS, FILTERS, 3, 3]),
            ("c1.bias", vec![FILTERS]),
            ("c2.weight", vec![FILTERS, FILTERS, 3, 3]),
            ("c2.bias", vec![FILTERS]),
            ("gn1.weight", vec![FILTERS]),
            ("gn1.bias", vec![FILTERS]),
            ("gn2.weight", vec![FILTERS]),
            ("gn2.bias", vec![FILTERS]),
        ] {
            want.push((format!("res_blocks.{i}.{leaf}"), shape));
        }
    }
    want
}

fn check_weights(failures: &mut Vec<String>) {
    let bytes = match std::fs::read(MODEL) {
        Ok(b) => b,
        Err(e) => {
            failures.push(format!("cannot read {MODEL}: {e}"));
            return;
        }
    };
    let st = match SafeTensors::deserialize(&bytes) {
        Ok(st) => st,
        Err(e) => {
            failures.push(format!("{MODEL} is not valid safetensors: {e}"));
            return;
        }
    };
    let want = expected_shapes();
    let mut found = 0usize;
    for (name, shape) in &want {
        match st.tensor(name) {
            Ok(view) => {
                found += 1;
                let got = view.shape().to_vec();
                if &got != shape {
                    failures.push(format!("{name}: shape {got:?}, expected {shape:?}"));
                }
            }
            Err(_) => failures.push(format!("{name}: missing from {MODEL}")),
        }
    }
    println!(
        "{MODEL}: {found}/{} required tensors present, {} extra (training-only) ignored",
        want.len(),
        st.len().saturating_sub(found)
    );
}

/// Head widths + finiteness: the subset that catches a checkpoint/backend
/// width mismatch or a garbage readback.
fn check_outputs(tag: &str, batch: usize, out: &Batch, failures: &mut Vec<String>) {
    if out.0.len() != batch || out.1.len() != batch {
        failures.push(format!(
            "{tag}: returned {} values / {} policy rows for batch {batch}",
            out.0.len(),
            out.1.len()
        ));
        return;
    }
    let hw = MAP_SIZE * MAP_SIZE;
    for (i, p) in out.1.iter().enumerate() {
        for (head, got, want) in [
            ("action_type", p.action_type.len(), NUM_ACTION_TYPES),
            ("source_spatial", p.source_spatial.len(), hw),
            ("target_spatial", p.target_spatial.len(), hw),
            ("move_option", p.move_option.len(), NUM_MOVE_OPTIONS),
        ] {
            if got != want {
                failures.push(format!(
                    "{tag}: row {i} head {head} is {got} wide, expected {want}"
                ));
            }
        }
        let finite = out.0[i].is_finite()
            && p.action_type
                .iter()
                .chain(&p.source_spatial)
                .chain(&p.target_spatial)
                .chain(&p.move_option)
                .all(|x| x.is_finite());
        if !finite {
            failures.push(format!("{tag}: row {i} has a non-finite output"));
        }
    }
}

/// Largest absolute elementwise difference; NaN anywhere returns NaN so it
/// can't be silently folded away.
fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    let mut worst = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        let d = (x - y).abs();
        if d.is_nan() {
            return f32::NAN;
        }
        if d > worst {
            worst = d;
        }
    }
    worst
}

fn compare(tag: &str, a: &Batch, b: &Batch, tol: f32, failures: &mut Vec<String>) {
    if a.0.len() != b.0.len() || a.1.len() != b.1.len() {
        failures.push(format!("{tag}: batch sizes differ, cannot compare"));
        return;
    }
    let mut worst = [0f32; 5];
    for i in 0..a.0.len() {
        let (x, y) = (&a.1[i], &b.1[i]);
        let row = [
            (a.0[i] - b.0[i]).abs(),
            max_abs(&x.action_type, &y.action_type),
            max_abs(&x.source_spatial, &y.source_spatial),
            max_abs(&x.target_spatial, &y.target_spatial),
            max_abs(&x.move_option, &y.move_option),
        ];
        for (w, d) in worst.iter_mut().zip(row) {
            if w.is_nan() {
                continue;
            }
            if d.is_nan() || d > *w {
                *w = d;
            }
        }
    }
    println!(
        "{tag}: max|Δ| value={:.2e} action={:.2e} source={:.2e} target={:.2e} option={:.2e} (tol {tol:.0e})",
        worst[0], worst[1], worst[2], worst[3], worst[4]
    );
    for (head, d) in ["value", "action", "source", "target", "option"]
        .iter()
        .zip(worst)
    {
        if !(d <= tol) {
            failures.push(format!("{tag}: {head} max|Δ|={d:.3e} exceeds tol {tol:.0e}"));
        }
    }
}

fn flatten(feats: &[RawFeatures]) -> (Vec<f32>, Vec<f32>) {
    let mut spatial = Vec::new();
    let mut player = Vec::new();
    for f in feats {
        spatial.extend_from_slice(&f.spatial);
        player.extend_from_slice(&f.player);
    }
    (spatial, player)
}

/// A few distinct leaf states, for a real (not all-zero) input distribution.
fn sample_features() -> Vec<RawFeatures> {
    let mut feats = Vec::new();
    for seed in [1i64, 2, 3, 4] {
        let mut game = Game::new();
        game.state = polyfish::mapgen::generate(polyfish::mapgen::MapGenSettings {
            size: polyfish::types::MapSize::Tiny,
            map_type: polyfish::types::MapType::Drylands,
            tribes: vec![
                polyfish::types::TribeType::Imperius,
                polyfish::types::TribeType::Bardur,
            ],
            seed,
            ..Default::default()
        });
        game.post_load();
        for pov in [1, 2] {
            feats.push(state_to_cpu_features(&game.state, pov).unwrap());
        }
    }
    feats
}

fn main() {
    let mut failures: Vec<String> = Vec::new();
    check_weights(&mut failures);

    let feats = sample_features();
    let tch_net = TchPolyZeroNet::load(MODEL, tch::Device::Cpu).unwrap();
    let metal_net = MetalPolyZeroNet::load(MODEL).unwrap();

    for batch in [1usize, 3, feats.len()] {
        let (spatial, player) = flatten(&feats[..batch]);
        let tch_out = tch_net.forward_batch(&spatial, &player, batch);
        let metal_out = metal_net.forward_batch(&spatial, &player, batch);
        check_outputs(&format!("batch {batch} tch-CPU"), batch, &tch_out, &mut failures);
        check_outputs(
            &format!("batch {batch} metal-MPSGraph"),
            batch,
            &metal_out,
            &mut failures,
        );
        compare(
            &format!("batch {batch}: tch-CPU vs metal-MPSGraph"),
            &tch_out,
            &metal_out,
            TOL_DEVICE,
            &mut failures,
        );
    }

    if failures.is_empty() {
        println!("PARITY OK");
    } else {
        for f in &failures {
            eprintln!("FAIL {f}");
        }
        eprintln!("{} parity check(s) failed", failures.len());
        std::process::exit(1);
    }
}
