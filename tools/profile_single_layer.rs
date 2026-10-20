// Profile individual kernels in a single layer
// SPDX-License-Identifier: Apache-2.0-or-later
// SPDX-FileCopyrightText: 2026 Jia Liu

use infer_train::inference::{InferenceContext, KVCacheGPU};
use infer_train::buffer::ForwardBufferPool;
use std::rc::Rc;
use std::cell::RefCell;
use metal::Buffer;

fn main() {
    println!("=== Single Layer Kernel Profile ===\n");

    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "/Users/jia/Desktop/infer_train/DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf".to_string());

    let ctx = InferenceContext::from_gguf(
        &model_path,
        infer_train::inference::Device::Gpu,
    ).expect("Failed to load model");

    let gpu_ctx = ctx.gpu_ctx.as_ref().expect("No GPU context");
    let device = gpu_ctx.device().clone();
    let reuse_pool = Rc::new(RefCell::new(Vec::new()));

    let batch = 256;
    let hidden_size = ctx.model.config.hidden_size;
    let head_dim = ctx.model.config.head_dim;
    let n_heads = ctx.model.config.n_heads;
    let n_kv_heads = ctx.model.layers[0].wk.as_ref().map(|w| w.shape[1] / head_dim).unwrap_or(n_heads);

    println!("Hidden size: {}", hidden_size);
    println!("Batch: {}", batch);
    println!("n_heads: {}, n_kv_heads: {}", n_heads, n_kv_heads);

    // Get layer 0 weights
    let layer = &ctx.model.layers[0];
    let attention_norm = layer.attention_norm.as_ref().expect("No attention_norm");
    let wq = layer.wq.as_ref().expect("No wq");
    let wk = layer.wk.as_ref().expect("No wk");
    let wv = layer.wv.as_ref().expect("No wv");
    let wo = layer.wo.as_ref().expect("No wo");
    let q_out_size = wq.shape[1];
    let k_out_size = wk.shape[1];
    let v_out_size = wv.shape[1];
    let intermediate_size = layer.w1.as_ref().map(|w| w.shape[1]).unwrap_or(0);

    println!("q_out_size: {}, k_out_size: {}, v_out_size: {}", q_out_size, k_out_size, v_out_size);
    println!("intermediate_size: {}", intermediate_size);

    // Warmup run
    {
        let mut kv_cache = KVCacheGPU::new(&device, &ctx.model.config, 512);
        let mut forward_pool = ForwardBufferPool::new(device.clone(), reuse_pool.clone());
        let command_buffer = gpu_ctx.queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        // Just create some buffers and run a simple operation
        let normed_buffer = forward_pool.get(batch * hidden_size * 4);
        let norm_weight_buffer = gpu_ctx.get_or_create_buffer(attention_norm);
        gpu_ctx.rms_norm_encode(&encoder, &forward_pool.get(hidden_size * 4), &norm_weight_buffer, &normed_buffer, batch, hidden_size, 1e-5);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    let mut kv_cache = KVCacheGPU::new(&device, &ctx.model.config, 512);

    // Test each kernel type individually
    println!("\n=== Individual Kernel Timings ===");

    // Test RMSNorm
    {
        let mut forward_pool = ForwardBufferPool::new(device.clone(), reuse_pool.clone());
        let input_buffer = forward_pool.get(batch * hidden_size * 4);
        let normed_buffer = forward_pool.get(batch * hidden_size * 4);
        let norm_weight_buffer = gpu_ctx.get_or_create_buffer(attention_norm);

        let start = std::time::Instant::now();
        let command_buffer = gpu_ctx.queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        gpu_ctx.rms_norm_encode(&encoder, &input_buffer, &norm_weight_buffer, &normed_buffer, batch, hidden_size, 1e-5);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        println!("RMSNorm: {:.2} ms", start.elapsed().as_secs_f64() * 1000.0);
    }

    // Test F32->F16 conversion
    {
        let mut forward_pool = ForwardBufferPool::new(device.clone(), reuse_pool.clone());
        let input_buffer = forward_pool.get(batch * hidden_size * 4);
        let output_buffer = forward_pool.get(batch * hidden_size * 2);

        let start = std::time::Instant::now();
        let command_buffer = gpu_ctx.queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        gpu_ctx.f32_to_f16_encode(&encoder, &input_buffer, &output_buffer, batch * hidden_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        println!("F32->F16 ({} elements): {:.2} ms", batch * hidden_size, start.elapsed().as_secs_f64() * 1000.0);
    }

    // Test RoPE
    {
        let mut forward_pool = ForwardBufferPool::new(device.clone(), reuse_pool.clone());
        let input_buffer = forward_pool.get(batch * q_out_size * 4);
        let output_buffer = forward_pool.get(batch * q_out_size * 4);
        let positions: Vec<i32> = (0..batch as i32).collect();

        let start = std::time::Instant::now();
        let command_buffer = gpu_ctx.queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        gpu_ctx.rope_neox_encode(&encoder, &input_buffer, &positions, batch, n_heads, head_dim, 10000.0, 1.0, &output_buffer);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        println!("RoPE ({} x {} heads): {:.2} ms", batch, n_heads, start.elapsed().as_secs_f64() * 1000.0);
    }

    // Test KV cache update
    {
        let mut forward_pool = ForwardBufferPool::new(device.clone(), reuse_pool.clone());
        let k_cache = kv_cache.k_buffer(0);
        let k_input = forward_pool.get(batch * k_out_size * 2); // F16

        let start = std::time::Instant::now();
        let command_buffer = gpu_ctx.queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        gpu_ctx.kv_update_batched_f16_encode(&encoder, &k_input, k_cache, 0, 512, n_kv_heads, head_dim, batch);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        println!("KV update ({} x {} kv_heads): {:.2} ms", batch, n_kv_heads, start.elapsed().as_secs_f64() * 1000.0);
    }

    // Test Flash Attention
    {
        let mut forward_pool = ForwardBufferPool::new(device.clone(), reuse_pool.clone());
        let q_buffer = forward_pool.get(batch * q_out_size * 4);
        let k_cache = kv_cache.k_buffer(0);
        let v_cache = kv_cache.v_buffer(0);
        let output_buffer = forward_pool.get(batch * q_out_size * 4);

        let seq_len = batch;
        let q_stride_b = n_heads * head_dim;
        let q_stride_h = head_dim;
        let q_stride_s = 1;
        let kv_stride_h = 512 * head_dim;
        let kv_stride_s = head_dim;

        let start = std::time::Instant::now();
        let command_buffer = gpu_ctx.queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        gpu_ctx.flash_prefill_batch_encode(
            &encoder, batch, n_heads, batch, seq_len, head_dim,
            &q_buffer, k_cache, v_cache, &output_buffer,
            q_stride_b, q_stride_h, q_stride_s,
            0, kv_stride_h, kv_stride_s,
            0, kv_stride_h, kv_stride_s,
            q_stride_b, q_stride_h, q_stride_s,
            0,
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        println!("Flash Attention: {:.2} ms", start.elapsed().as_secs_f64() * 1000.0);
    }

    // Test GEMM
    {
        let mut forward_pool = ForwardBufferPool::new(device.clone(), reuse_pool.clone());
        let wq_buffer = gpu_ctx.get_or_create_buffer(wq);
        let input_buffer = forward_pool.get(batch * hidden_size * 2); // F16
        let output_buffer = forward_pool.get(batch * q_out_size * 4); // F32

        let start = std::time::Instant::now();
        let command_buffer = gpu_ctx.queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        gpu_ctx.mul_mm_q5_k_encode(&encoder, &wq_buffer, &input_buffer, &output_buffer, q_out_size, batch, hidden_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        println!("GEMM (Q projection): {:.2} ms", start.elapsed().as_secs_f64() * 1000.0);
    }

    // Now test full layer
    println!("\n=== Full Layer ===");
    {
        let mut forward_pool = ForwardBufferPool::new(device.clone(), reuse_pool.clone());
        let input_buffer = forward_pool.get(batch * hidden_size * 4);

        let start = std::time::Instant::now();
        let command_buffer = gpu_ctx.queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        if let Some(_output) = ctx.model.forward_layer_prefill_batched(
            layer, 0, &input_buffer, batch,
            &mut kv_cache, gpu_ctx, &mut forward_pool, &encoder,
        ) {
            // Layer completed
        }

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        println!("Full layer: {:.2} ms", start.elapsed().as_secs_f64() * 1000.0);
    }

    println!("\n=== Test Complete ===");
}
