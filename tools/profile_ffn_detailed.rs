// Profile FFN with detailed per-operation timing
// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright 2026 Jia Liu

use infer_train::cpu::CpuWorkBuffers;
use infer_train::inference::{
    Device, InferenceContext, KVCache, KVCacheBase, KVCacheFormat,
};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <model.gguf> <batch_size>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    let batch_size: usize = args[2].parse().expect("Invalid batch size");

    println!("=== FFN Detailed Profile ===\n");

    let ctx = InferenceContext::from_gguf(model_path, Device::Cpu)
        .expect("Failed to load model");

    let hidden_size = ctx.config.hidden_size;
    let intermediate_size = ctx.config.intermediate_size;
    let n_layers = ctx.model.layers.len();

    println!(
        "Model: {} layers, hidden={}, intermediate={}",
        n_layers, hidden_size, intermediate_size
    );
    let n_heads = ctx.config.n_heads;
    let n_kv_heads = ctx.config.n_kv_heads;
    let head_dim = ctx.config.head_dim;
    let n_layers = ctx.model.layers.len();
    let vocab_size = ctx.config.vocab_size;

    let qkv_out_size =
        ctx.model.layers[0].attn_qkv.as_ref().map(|t| t.shape[1]).unwrap_or(0);

    let mut buffers = CpuWorkBuffers::new(
        hidden_size,
        intermediate_size,
        n_heads,
        n_kv_heads,
        head_dim,
        ctx.config.key_length,
        ctx.config.value_length,
        vocab_size,
        qkv_out_size,
        2048,
        n_layers,
    );
    buffers.init_norm_weights(&ctx.model);

    let mut kv_cache = KVCache::new(&ctx.config, 2048, KVCacheFormat::F32);

    // Generate tokens
    let tokens: Vec<u32> = (1..=batch_size as u32).collect();

    // Warmup
    let _ = ctx.model.forward(&tokens, &mut kv_cache);
    kv_cache.clear();

    // Detailed FFN timing for one layer
    println!("=== Single Layer FFN Breakdown ===");

    let layer = &ctx.model.layers[0];

    // Prepare inputs (simulate after attention)
    let mut hidden = vec![0.1f32; hidden_size * batch_size];
    for i in 0..hidden.len() {
        hidden[i] = (i % 100) as f32 / 100.0;
    }

    // FFN norm
    let t_norm = Instant::now();
    ctx.model.apply_rms_norm_predequant(
        &hidden[..hidden_size],
        &buffers.norm_weights[..hidden_size],
        &mut buffers.hidden_alt,
    );
    println!("FFN norm: {:.3} ms", t_norm.elapsed().as_secs_f64() * 1000.0);

    // Prepare batch inputs
    let ffn_inputs: Vec<&[f32]> =
        (0..batch_size).map(|i| &buffers.hidden_alt[..hidden_size]).collect();

    let ffn_output_len = intermediate_size * batch_size;

    // Gate projection (w1)
    if let Some(ref w1) = layer.w1 {
        let mut timing =
            (std::time::Duration::new(0, 0), std::time::Duration::new(0, 0));

        let t_gate = Instant::now();
        ctx.model.gemm_batch_prefill(
            &ffn_inputs,
            w1,
            &mut buffers.ffn_gate_batch[..ffn_output_len],
            &mut buffers.q8k_input,
            &mut buffers.x_padded,
            &mut buffers.x_padded_threads,
            Some(&mut timing),
        );
        let gate_total = t_gate.elapsed();

        // Calculate thread count used
        let intermediate_size = w1.shape[0];
        let hidden_size = w1.shape[1];
        let scheduler = infer_train::cpu::get_scheduler();
        let n_threads = scheduler.optimal_threads(
            intermediate_size,
            hidden_size,
            batch_size,
        );

        println!(
            "Gate (w1) total: {:.3} ms (quant: {:.3} ms, GEMM: {:.3} ms) [threads: {}, m={}, k={}, n={}, dtype={:?}]",
            gate_total.as_secs_f64() * 1000.0,
            timing.0.as_secs_f64() * 1000.0,
            timing.1.as_secs_f64() * 1000.0,
            n_threads,
            intermediate_size,
            hidden_size,
            batch_size,
            w1.dtype
        );
    }

    // Up projection (w3)
    if let Some(ref w3) = layer.w3 {
        let mut timing =
            (std::time::Duration::new(0, 0), std::time::Duration::new(0, 0));

        let t_up = Instant::now();
        ctx.model.gemm_batch_prefill(
            &ffn_inputs,
            w3,
            &mut buffers.ffn_up_batch[..ffn_output_len],
            &mut buffers.q8k_input,
            &mut buffers.x_padded,
            &mut buffers.x_padded_threads,
            Some(&mut timing),
        );
        let up_total = t_up.elapsed();

        println!(
            "Up (w3) total: {:.3} ms (quant: {:.3} ms, GEMM: {:.3} ms)",
            up_total.as_secs_f64() * 1000.0,
            timing.0.as_secs_f64() * 1000.0,
            timing.1.as_secs_f64() * 1000.0
        );
    }

    // SiLU activation (in-place)
    let t_silu = Instant::now();
    unsafe {
        // Convert to raw pointers to allow overlapping borrows (gate and dst are the same buffer)
        let gate_ptr = buffers.ffn_gate_batch.as_ptr();
        let up_ptr = buffers.ffn_up_batch.as_ptr();
        let dst_ptr = buffers.ffn_gate_batch.as_mut_ptr();

        let gate_slice = std::slice::from_raw_parts(gate_ptr, ffn_output_len);
        let up_slice = std::slice::from_raw_parts(up_ptr, ffn_output_len);
        let dst_slice = std::slice::from_raw_parts_mut(dst_ptr, ffn_output_len);

        infer_train::ops::arm::neon::swiglu_f32_neon(
            gate_slice, up_slice, dst_slice,
        );
    }
    println!("SwiGLU: {:.3} ms", t_silu.elapsed().as_secs_f64() * 1000.0);

    // Down projection (w2)
    let down_inputs: Vec<&[f32]> = (0..batch_size)
        .map(|i| {
            &buffers.ffn_gate_batch
                [i * intermediate_size..(i + 1) * intermediate_size]
        })
        .collect();

    if let Some(ref w2) = layer.w2 {
        let mut timing =
            (std::time::Duration::new(0, 0), std::time::Duration::new(0, 0));

        let t_down = Instant::now();
        ctx.model.gemm_batch_prefill(
            &down_inputs,
            w2,
            &mut hidden[..hidden_size * batch_size],
            &mut buffers.q8k_input,
            &mut buffers.x_padded,
            &mut buffers.x_padded_threads,
            Some(&mut timing),
        );
        let down_total = t_down.elapsed();

        println!(
            "Down (w2) total: {:.3} ms (quant: {:.3} ms, GEMM: {:.3} ms)",
            down_total.as_secs_f64() * 1000.0,
            timing.0.as_secs_f64() * 1000.0,
            timing.1.as_secs_f64() * 1000.0
        );
    }

    println!("\n=== All Layers Average ===");

    // Profile all layers
    let mut total_gate_quant = std::time::Duration::new(0, 0);
    let mut total_gate_gemm = std::time::Duration::new(0, 0);
    let mut total_up_quant = std::time::Duration::new(0, 0);
    let mut total_up_gemm = std::time::Duration::new(0, 0);
    let mut total_down_quant = std::time::Duration::new(0, 0);
    let mut total_down_gemm = std::time::Duration::new(0, 0);
    let mut total_silu = std::time::Duration::new(0, 0);

    let mut batch_hidden = vec![0.0f32; hidden_size * batch_size];

    for layer_idx in 0..n_layers {
        let layer = &ctx.model.layers[layer_idx];

        // Gate
        if let Some(ref w1) = layer.w1 {
            let mut timing = (
                std::time::Duration::new(0, 0),
                std::time::Duration::new(0, 0),
            );
            ctx.model.gemm_batch_prefill(
                &ffn_inputs,
                w1,
                &mut buffers.ffn_gate_batch[..ffn_output_len],
                &mut buffers.q8k_input,
                &mut buffers.x_padded,
                &mut buffers.x_padded_threads,
                Some(&mut timing),
            );
            total_gate_quant += timing.0;
            total_gate_gemm += timing.1;
        }

        // Up
        if let Some(ref w3) = layer.w3 {
            let mut timing = (
                std::time::Duration::new(0, 0),
                std::time::Duration::new(0, 0),
            );
            ctx.model.gemm_batch_prefill(
                &ffn_inputs,
                w3,
                &mut buffers.ffn_up_batch[..ffn_output_len],
                &mut buffers.q8k_input,
                &mut buffers.x_padded,
                &mut buffers.x_padded_threads,
                Some(&mut timing),
            );
            total_up_quant += timing.0;
            total_up_gemm += timing.1;
        }

        // SwiGLU (in-place)
        let t = Instant::now();
        unsafe {
            // Convert to raw pointers to allow overlapping borrows
            let gate_ptr = buffers.ffn_gate_batch.as_ptr();
            let up_ptr = buffers.ffn_up_batch.as_ptr();
            let dst_ptr = buffers.ffn_gate_batch.as_mut_ptr();

            let gate_slice =
                std::slice::from_raw_parts(gate_ptr, ffn_output_len);
            let up_slice = std::slice::from_raw_parts(up_ptr, ffn_output_len);
            let dst_slice =
                std::slice::from_raw_parts_mut(dst_ptr, ffn_output_len);

            infer_train::ops::arm::neon::swiglu_f32_neon(
                gate_slice, up_slice, dst_slice,
            );
        }
        total_silu += t.elapsed();

        // Down - create inputs fresh each iteration to avoid borrow conflicts
        if let Some(ref w2) = layer.w2 {
            let down_inputs: Vec<&[f32]> = (0..batch_size)
                .map(|i| {
                    &buffers.ffn_gate_batch
                        [i * intermediate_size..(i + 1) * intermediate_size]
                })
                .collect();

            let mut timing = (
                std::time::Duration::new(0, 0),
                std::time::Duration::new(0, 0),
            );
            ctx.model.gemm_batch_prefill(
                &down_inputs,
                w2,
                &mut batch_hidden,
                &mut buffers.q8k_input,
                &mut buffers.x_padded,
                &mut buffers.x_padded_threads,
                Some(&mut timing),
            );
            total_down_quant += timing.0;
            total_down_gemm += timing.1;
        }
    }

    println!(
        "Gate quant: {:.3} ms, Gate GEMM: {:.3} ms",
        total_gate_quant.as_secs_f64() * 1000.0,
        total_gate_gemm.as_secs_f64() * 1000.0
    );
    println!(
        "Up quant: {:.3} ms, Up GEMM: {:.3} ms",
        total_up_quant.as_secs_f64() * 1000.0,
        total_up_gemm.as_secs_f64() * 1000.0
    );
    println!(
        "Down quant: {:.3} ms, Down GEMM: {:.3} ms",
        total_down_quant.as_secs_f64() * 1000.0,
        total_down_gemm.as_secs_f64() * 1000.0
    );
    println!("SwiGLU: {:.3} ms", total_silu.as_secs_f64() * 1000.0);

    let total_ffn = total_gate_quant
        + total_gate_gemm
        + total_up_quant
        + total_up_gemm
        + total_down_quant
        + total_down_gemm
        + total_silu;
    println!("Total FFN: {:.3} ms", total_ffn.as_secs_f64() * 1000.0);
}
