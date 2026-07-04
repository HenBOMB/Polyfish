//! Micro-benchmark of the individual GPU ops used in PolyZeroNet's forward
//! pass, each timed with a full device sync, to find which op is responsible
//! for the ~0.75ms/row wall cost measured in bench_forward.
//!
//! Run: cargo run --release --example bench_ops -- [batch]

use candle_core::{DType, Device, Tensor};
use candle_nn::{BatchNorm, Module, VarBuilder, VarMap};
use std::time::Instant;

fn sync(t: &Tensor) -> anyhow::Result<()> {
    // Force execution + wait: cheap scalar readback.
    let _ = t.sum_all()?.to_vec0::<f32>()?;
    Ok(())
}

fn time_op<F: FnMut() -> anyhow::Result<Tensor>>(
    name: &str,
    iters: usize,
    mut f: F,
) -> anyhow::Result<()> {
    // warmup
    for _ in 0..3 {
        sync(&f()?)?;
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        sync(&f()?)?;
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
    println!("{name:<44} {ms:8.2}ms/iter");
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let batch: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(128);
    let device = Device::metal_if_available(0)?;
    println!("device=Metal batch={batch}");

    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let x_in = Tensor::randn(0f32, 1f32, (batch, 154, 11, 11), &device)?;
    let x64 = Tensor::randn(0f32, 1f32, (batch, 64, 11, 11), &device)?;

    // A. stem conv 154->64
    let conv_cfg = candle_nn::Conv2dConfig {
        padding: 1,
        ..Default::default()
    };
    let stem = candle_nn::conv2d(154, 64, 3, conv_cfg, vs.pp("stem"))?;
    time_op("conv2d 154->64 3x3", 10, || Ok(stem.forward(&x_in)?))?;

    // B. one res-conv 64->64
    let c64 = candle_nn::conv2d(64, 64, 3, conv_cfg, vs.pp("c64"))?;
    time_op("conv2d 64->64 3x3", 10, || Ok(c64.forward(&x64)?))?;

    // C. batch_norm eval mode
    let bn: BatchNorm = candle_nn::batch_norm(64, 1e-5, vs.pp("bn"))?;
    time_op("batch_norm(64) eval", 10, || {
        Ok(candle_nn::ModuleT::forward_t(&bn, &x64, false)?)
    })?;

    // D. relu
    time_op("relu", 10, || Ok(x64.relu()?))?;

    // E. attention shapes: q[b,4,121,16] @ k_t[b,4,16,121]
    let q = Tensor::randn(0f32, 1f32, (batch, 4, 121, 16), &device)?;
    let kt = Tensor::randn(0f32, 1f32, (batch, 4, 16, 121), &device)?;
    time_op("bmm q@k_t -> [b,4,121,121]", 10, || Ok(q.matmul(&kt)?))?;

    let scores = q.matmul(&kt)?;
    time_op("softmax [b,4,121,121]", 10, || {
        Ok(candle_nn::ops::softmax(&scores, candle_core::D::Minus1)?)
    })?;

    let v = Tensor::randn(0f32, 1f32, (batch, 4, 121, 16), &device)?;
    let attn = candle_nn::ops::softmax(&scores, candle_core::D::Minus1)?;
    time_op("bmm attn@v -> [b,4,121,16]", 10, || Ok(attn.matmul(&v)?))?;

    // F. transpose + contiguous (used around attention)
    let t = Tensor::randn(0f32, 1f32, (batch, 121, 4, 16), &device)?;
    time_op("transpose(1,2)+contiguous", 10, || {
        Ok(t.transpose(1, 2)?.contiguous()?)
    })?;

    // G. linear 64->64 on [b,121,64] tokens
    let tok = Tensor::randn(0f32, 1f32, (batch, 121, 64), &device)?;
    let lin = candle_nn::linear(64, 64, vs.pp("lin"))?;
    time_op("linear 64->64 on [b,121,64]", 10, || Ok(lin.forward(&tok)?))?;

    // H. layer_norm on tokens
    let ln = candle_nn::layer_norm(64, 1e-5, vs.pp("ln"))?;
    time_op("layer_norm(64) on [b,121,64]", 10, || Ok(ln.forward(&tok)?))?;

    Ok(())
}
