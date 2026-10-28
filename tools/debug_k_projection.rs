// Debug K projection all-zeros issue
// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2025 Qwen3-0.6B K projection debug

use infer_train::inference::InferenceContext;
use infer_train::model::DataType;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        return;
    }

    let model_path = &args[1];
    let ctx = InferenceContext::from_gguf(
        model_path,
        infer_train::inference::Device::Cpu,
    )
    .expect("Failed to load model");

    println!("Model architecture: {:?}", ctx.model.config.architecture);
    println!("Hidden size: {}", ctx.model.config.hidden_size);
    println!();

    let layer = &ctx.model.layers[0];
    let hidden_size = ctx.model.config.hidden_size;

    // Create test input
    let test_hidden: Vec<f32> =
        (0..hidden_size).map(|i| (i as f32 % 100.0) / 100.0).collect();
    println!("Testing with synthetic input [0..5]: {:?}", &test_hidden[0..5]);
    println!();

    // Test each projection separately
    if let (Some(wq), Some(wk), Some(wv)) = (&layer.wq, &layer.wk, &layer.wv) {
        let q_out_size = wq.shape[1];
        let k_out_size = wk.shape[1];
        let v_out_size = wv.shape[1];

        println!(
            "Weight dtypes: wq={:?}, wk={:?}, wv={:?}",
            wq.dtype, wk.dtype, wv.dtype
        );
        println!();

        // Test gemv_into
        println!("=== Testing gemv_into ===");
        let mut q_result = vec![0.0f32; q_out_size];
        ctx.model.gemv_into(&test_hidden, wq, &mut q_result);
        println!("  Q[0..5]: {:?}", &q_result[0..5]);
        let q_nonzero = q_result.iter().filter(|&&x| x != 0.0).count();
        println!("  Q non-zero count: {}/{}", q_nonzero, q_result.len());

        let mut k_result = vec![0.0f32; k_out_size];
        ctx.model.gemv_into(&test_hidden, wk, &mut k_result);
        println!("  K[0..5]: {:?}", &k_result[0..5]);
        let k_nonzero = k_result.iter().filter(|&&x| x != 0.0).count();
        println!("  K non-zero count: {}/{}", k_nonzero, k_result.len());

        let mut v_result = vec![0.0f32; v_out_size];
        ctx.model.gemv_into(&test_hidden, wv, &mut v_result);
        println!("  V[0..5]: {:?}", &v_result[0..5]);
        let v_nonzero = v_result.iter().filter(|&&x| x != 0.0).count();
        println!("  V non-zero count: {}/{}", v_nonzero, v_result.len());
        println!();

        // Test gemv_impl
        println!("=== Testing gemv_impl ===");
        if let Some(q) = ctx.model.gemv_impl(&test_hidden, wq, q_out_size) {
            println!("  Q[0..5]: {:?}", &q[0..5]);
            let nonzero = q.iter().filter(|&&x| x != 0.0).count();
            println!("  Q non-zero count: {}/{}", nonzero, q.len());
        }

        if let Some(k) = ctx.model.gemv_impl(&test_hidden, wk, k_out_size) {
            println!("  K[0..5]: {:?}", &k[0..5]);
            let nonzero = k.iter().filter(|&&x| x != 0.0).count();
            println!("  K non-zero count: {}/{}", nonzero, k.len());
        }

        if let Some(v) = ctx.model.gemv_impl(&test_hidden, wv, v_out_size) {
            println!("  V[0..5]: {:?}", &v[0..5]);
            let nonzero = v.iter().filter(|&&x| x != 0.0).count();
            println!("  V non-zero count: {}/{}", nonzero, v.len());
        }
    }
}
