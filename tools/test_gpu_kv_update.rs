// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test KV cache update kernel

use infer_train::inference::{Device, InferenceContext};
use infer_train::model::GgufFile;
use infer_train::tokenize::BpeTokenizer;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use infer_train::inference::KVCacheGPU;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];

    let gguf = GgufFile::open(model_path).expect("Failed to open GGUF");
    let tokenizer =
        BpeTokenizer::from_gguf(&gguf).expect("Failed to load tokenizer");

    println!("Loading model: {}", model_path);
    let mut ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
        .expect("Failed to load model");

    let prompt = "Hello";
    let tokens = tokenizer.encode(prompt);
    let token = tokens[0];
    println!("\nToken: {}", token);

    let hidden =
        ctx.model.embedding_lookup_impl(token).expect("Embedding failed");
    let layer0 = &ctx.model.layers[0];
    let attn_norm = layer0.attention_norm.as_ref().expect("No attention norm");
    let wk = layer0.wk.as_ref().expect("No wk");
    let wv = layer0.wv.as_ref().expect("No wv");

    // CPU version
    let cpu_normed = ctx
        .model
        .apply_rms_norm(&hidden, attn_norm, hidden.len())
        .expect("RMS norm failed");
    let cpu_k =
        ctx.model.gemv(&cpu_normed, wk, wk.shape[1]).expect("GEMV failed");
    let cpu_v =
        ctx.model.gemv(&cpu_normed, wv, wv.shape[1]).expect("GEMV failed");

    println!(
        "CPU K: len={}, min={:.4}, max={:.4}",
        cpu_k.len(),
        cpu_k.iter().cloned().fold(f32::INFINITY, f32::min),
        cpu_k.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );
    println!(
        "CPU V: len={}, min={:.4}, max={:.4}",
        cpu_v.len(),
        cpu_v.iter().cloned().fold(f32::INFINITY, f32::min),
        cpu_v.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );

    // GPU KV cache update
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        if let (Some(ref gpu_ctx), Some(ref mut kv_cache)) =
            (ctx.gpu_ctx.as_ref(), ctx.gpu_kv_cache.as_mut())
        {
            use metal::*;

            // Clear KV cache first
            kv_cache.clear();

            // Verify cache is zero after clear
            let k_cache = kv_cache.k_buffer(0);
            let v_cache = kv_cache.v_buffer(0);
            let head_dim = ctx.model.config.head_dim;
            let n_kv_heads = cpu_k.len() / head_dim;
            let max_seq = 2048; // Actual max_seq used in InferenceContext

            println!("\n=== KV Cache Parameters ===");
            println!("K cache buffer address: {:?}", k_cache.contents());
            println!("V cache buffer address: {:?}", v_cache.contents());
            println!(
                "Buffers are same: {}",
                k_cache.contents() == v_cache.contents()
            );
            println!("K cache buffer length: {} bytes", k_cache.length());
            println!("V cache buffer length: {} bytes", v_cache.length());
            println!(
                "Expected buffer size: {} bytes",
                n_kv_heads * max_seq * head_dim * 2
            );
            println!("Model config head_dim: {}", ctx.model.config.head_dim);
            println!(
                "Model config n_kv_heads: {}",
                ctx.model.config.n_kv_heads
            );
            println!("CPU K len: {}", cpu_k.len());
            println!(
                "Computed n_kv_heads: {} (cpu_k.len() / head_dim)",
                n_kv_heads
            );
            println!("Max seq: {}", max_seq);

            let total_f16_elements = max_seq * n_kv_heads * head_dim;
            let k_cache_f16_after_clear: Vec<u16> = unsafe {
                std::slice::from_raw_parts(
                    k_cache.contents() as *const u16,
                    total_f16_elements,
                )
                .to_vec()
            };
            let k_cache_f32_after_clear: Vec<f32> = k_cache_f16_after_clear
                .iter()
                .map(|&h| half::f16::from_bits(h).to_f32())
                .collect();
            let nonzero_after_clear =
                k_cache_f32_after_clear.iter().filter(|&&v| v != 0.0).count();
            println!(
                "After clear, nonzero count in K cache: {}",
                nonzero_after_clear
            );

            let device = gpu_ctx.device();
            let queue = gpu_ctx.queue();

            // Upload K and V
            let k_buffer = std::rc::Rc::new(device.new_buffer_with_data(
                cpu_k.as_ptr() as *const std::ffi::c_void,
                (cpu_k.len() * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            ));
            let v_buffer = std::rc::Rc::new(device.new_buffer_with_data(
                cpu_v.as_ptr() as *const std::ffi::c_void,
                (cpu_v.len() * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            ));

            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            let head_dim = ctx.model.config.head_dim;
            let n_kv_heads = cpu_k.len() / head_dim;

            // KV update
            let k_cache = kv_cache.k_buffer(0);
            let v_cache = kv_cache.v_buffer(0);

            gpu_ctx.kv_update_k_encode(
                encoder, &k_buffer, k_cache, 0, // pos
                max_seq, n_kv_heads, head_dim,
            );

            gpu_ctx.kv_update_v_encode(
                encoder, &v_buffer, v_cache, 0, // pos
                max_seq, n_kv_heads, head_dim,
            );

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            // Read back KV cache (F16 format)
            let total_f16_elements = max_seq * n_kv_heads * head_dim;
            let k_cache_f16: Vec<u16> = unsafe {
                std::slice::from_raw_parts(
                    k_cache.contents() as *const u16,
                    total_f16_elements,
                )
                .to_vec()
            };

            let v_cache_f16: Vec<u16> = unsafe {
                std::slice::from_raw_parts(
                    v_cache.contents() as *const u16,
                    total_f16_elements,
                )
                .to_vec()
            };

            // Convert F16 to F32
            let k_cache_data: Vec<f32> = k_cache_f16
                .iter()
                .map(|&h| half::f16::from_bits(h).to_f32())
                .collect();
            let v_cache_data: Vec<f32> = v_cache_f16
                .iter()
                .map(|&h| half::f16::from_bits(h).to_f32())
                .collect();

            // Check position 0 for all heads
            // CPU data layout: [n_kv_heads, head_dim]
            // GPU cache layout: [n_kv_heads, max_seq, head_dim]
            let mut max_k_diff = 0.0f32;
            let mut max_v_diff = 0.0f32;

            for h in 0..n_kv_heads {
                for d in 0..head_dim {
                    let cpu_idx = h * head_dim + d;
                    let gpu_idx = h * max_seq * head_dim + d;

                    let k_diff = (cpu_k[cpu_idx] - k_cache_data[gpu_idx]).abs();
                    let v_diff = (cpu_v[cpu_idx] - v_cache_data[gpu_idx]).abs();
                    max_k_diff = max_k_diff.max(k_diff);
                    max_v_diff = max_v_diff.max(v_diff);
                }
            }

            // Check that unwritten positions are zero
            // Layout: [n_kv_heads, max_seq, head_dim]
            // For each head, only position 0 is written, positions 1..max_seq-1 should be zero
            let mut found_nonzero_k = false;
            let mut found_nonzero_v = false;

            for h in 0..n_kv_heads {
                for pos in 1..max_seq {
                    // Check this position in both K and V caches
                    for d in 0..head_dim {
                        let idx = h * max_seq * head_dim + pos * head_dim + d;

                        if k_cache_data[idx] != 0.0 && !found_nonzero_k {
                            found_nonzero_k = true;
                            println!(
                                "Unexpected K data at h={}, pos={}, d={}, index={}, value={:.4}",
                                h, pos, d, idx, k_cache_data[idx]
                            );
                        }
                        if v_cache_data[idx] != 0.0 && !found_nonzero_v {
                            found_nonzero_v = true;
                            println!(
                                "Unexpected V data at h={}, pos={}, d={}, index={}, value={:.4}",
                                h, pos, d, idx, v_cache_data[idx]
                            );
                        }
                    }
                }
            }

            println!("GPU K cache (pos 0): max diff={:.6}", max_k_diff);
            println!("GPU V cache (pos 0): max diff={:.6}", max_v_diff);
            println!(
                "Unwritten positions have nonzero: K={}, V={}",
                found_nonzero_k, found_nonzero_v
            );
            println!(
                "K cache nan count: {}",
                k_cache_data.iter().filter(|&&v| v.is_nan()).count()
            );
            println!(
                "V cache nan count: {}",
                v_cache_data.iter().filter(|&&v| v.is_nan()).count()
            );

            // Print first 10 values from head 0
            println!("\nHead 0, first 10 K values:");
            for d in 0..10 {
                let cpu_idx = d;
                let gpu_idx = d;
                println!(
                    "  CPU K[{}]={:.4}, GPU K cache[{}]={:.4}",
                    cpu_idx, cpu_k[cpu_idx], gpu_idx, k_cache_data[gpu_idx]
                );
            }

            println!("\nHead 0, first 10 V values:");
            for d in 0..10 {
                let cpu_idx = d;
                let gpu_idx = d;
                println!(
                    "  CPU V[{}]={:.4}, GPU V cache[{}]={:.4}",
                    cpu_idx, cpu_v[cpu_idx], gpu_idx, v_cache_data[gpu_idx]
                );
            }

            // Print head 1's first value to verify it's in the right place
            println!("\nHead 1, first K value:");
            let cpu_idx = head_dim; // Head 1, dimension 0
            let gpu_idx = max_seq * head_dim; // Head 1, pos 0, dimension 0
            println!(
                "  CPU K[{}]={:.4}, GPU K cache[{}]={:.4}",
                cpu_idx, cpu_k[cpu_idx], gpu_idx, k_cache_data[gpu_idx]
            );
        }
    }
}
