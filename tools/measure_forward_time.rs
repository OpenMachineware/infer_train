#!/usr/bin/env cargo
//! 测量完整forward pass绝对时间（不依赖模型文件）
//! 模拟真实推理流程

use std::time::{Duration, Instant};

fn main() {
    // Qwen3-0.6B配置
    let n_tokens = 64;
    let n_layers = 28;
    let n_heads = 16;
    let n_kv_heads = 8;
    let head_dim = 128;
    let hidden_size = 896;
    let intermediate_size = 4864;

    let kv_size = n_kv_heads * head_dim;
    let q_size = n_heads * head_dim;

    println!("=== Forward Pass 绝对时间测量 ===");
    println!("配置: {} layers, {} tokens, hidden={}\n",
             n_layers, n_tokens, hidden_size);

    let iterations = 10;

    // 测试数据
    let input: Vec<f32> = (0..hidden_size * n_tokens).map(|i| (i as f32 * 0.001).sin()).collect();

    // 测量QKV projection
    println!("=== 1. QKV Projection ===");
    let qkv_weight: Vec<f32> = (0..hidden_size * q_size).map(|i| (i as f32 * 0.001).cos()).collect();
    let mut qkv_output = vec![0.0f32; q_size * n_tokens];

    let mut qkv_total = Duration::new(0, 0);
    for _ in 0..iterations {
        let start = Instant::now();
        // 简化：只测一次GEMM
        for t in 0..n_tokens {
            for i in 0..q_size {
                let mut sum = 0.0f32;
                for j in 0..hidden_size {
                    sum += input[t * hidden_size + j] * qkv_weight[j * q_size + i];
                }
                qkv_output[t * q_size + i] = sum;
            }
        }
        qkv_total += start.elapsed();
    }

    let qkv_ops = hidden_size * q_size * n_tokens;
    let qkv_time_us = qkv_total.as_micros() as f64 / iterations as f64;
    println!("绝对时间: {:.1} µs", qkv_time_us);
    println!("操作数: {:.0} M ops", qkv_ops as f64 / 1e6);

    // 测量我们的simd_gemm
    println!("\n=== 2. 使用我们的simd_gemm ===");
    let mut gemm_output = vec![0.0f32; q_size * n_tokens];

    let mut gemm_total = Duration::new(0, 0);
    for _ in 0..iterations {
        let start = Instant::now();
        unsafe {
            infer_train::cpu::simd_gemm::simd_gemm(
                gemm_output.as_mut_ptr(),
                input.as_ptr(),
                qkv_weight.as_ptr(),
                n_tokens,
                hidden_size,
                q_size,
            );
        }
        gemm_total += start.elapsed();
        std::hint::black_box(&gemm_output);
    }

    let gemm_time_us = gemm_total.as_micros() as f64 / iterations as f64;
    println!("绝对时间: {:.1} µs", gemm_time_us);
    println!("vs 标量: {:.1}x 加速", qkv_time_us / gemm_time_us);

    // 测量FFN
    println!("\n=== 3. FFN (gate + up + down) ===");
    let gate_weight: Vec<f32> = (0..hidden_size * intermediate_size).map(|i| (i as f32 * 0.001).cos()).collect();
    let up_weight: Vec<f32> = (0..hidden_size * intermediate_size).map(|i| (i as f32 * 0.002).sin()).collect();
    let down_weight: Vec<f32> = (0..intermediate_size * hidden_size).map(|i| (i as f32 * 0.001).cos()).collect();

    let mut ffn_total = Duration::new(0, 0);
    for _ in 0..iterations {
        let start = Instant::now();

        // Gate: input @ gate_weight
        let mut gate_out = vec![0.0f32; intermediate_size * n_tokens];
        unsafe {
            infer_train::cpu::simd_gemm::simd_gemm(
                gate_out.as_mut_ptr(),
                input.as_ptr(),
                gate_weight.as_ptr(),
                n_tokens,
                hidden_size,
                intermediate_size,
            );
        }

        // Up: input @ up_weight
        let mut up_out = vec![0.0f32; intermediate_size * n_tokens];
        unsafe {
            infer_train::cpu::simd_gemm::simd_gemm(
                up_out.as_mut_ptr(),
                input.as_ptr(),
                up_weight.as_ptr(),
                n_tokens,
                hidden_size,
                intermediate_size,
            );
        }

        // Down: (gate * up) @ down_weight
        let mut ffn_out = vec![0.0f32; hidden_size * n_tokens];
        for i in 0..intermediate_size * n_tokens {
            gate_out[i] *= up_out[i];
        }
        unsafe {
            infer_train::cpu::simd_gemm::simd_gemm(
                ffn_out.as_mut_ptr(),
                gate_out.as_ptr(),
                down_weight.as_ptr(),
                n_tokens,
                intermediate_size,
                hidden_size,
            );
        }

        ffn_total += start.elapsed();
        std::hint::black_box(&ffn_out);
    }

    let ffn_time_us = ffn_total.as_micros() as f64 / iterations as f64;
    println!("绝对时间: {:.1} µs", ffn_time_us);

    // 计算28层总时间
    println!("\n=== 4. 28层理论时间 ===");

    // QKV: 3个投影 (q + k + v)
    let qkv_layer_time = gemm_time_us * 3.0;  // 使用simd_gemm
    println!("QKV (3×): {:.1} µs", qkv_layer_time);

    // FFN: 3个GEMM
    let ffn_layer_time = ffn_time_us;
    println!("FFN: {:.1} µs", ffn_layer_time);

    // Attention (从之前测量): 41.779 µs per head × 16 heads = 668.5 µs
    let attention_layer_time = 41.779 * n_heads as f64;
    println!("Attention: {:.1} µs", attention_layer_time);

    let layer_total = qkv_layer_time + ffn_layer_time + attention_layer_time;
    println!("单层总时间: {:.1} µs", layer_total);

    let total_28_layers_ms = layer_total * n_layers as f64 / 1000.0;
    println!("28层总时间: {:.1} ms", total_28_layers_ms);

    let tokens_per_sec = n_tokens as f64 / total_28_layers_ms * 1000.0;
    println!("理论吞吐量: {:.1} t/s", tokens_per_sec);

    // 对比llama.cpp
    println!("\n=== 5. 对比llama.cpp ===");
    println!("llama.cpp实测: 163.3 t/s");
    println!("我们理论: {:.1} t/s", tokens_per_sec);
    println!("差距: {:.2}x", 163.3 / tokens_per_sec);

    // 绝对时间对比
    let llama_time_ms = n_tokens as f64 / 163.3 * 1000.0;
    println!("\nllama.cpp 64 tokens: {:.1} ms", llama_time_ms);
    println!("我们 64 tokens: {:.1} ms", total_28_layers_ms);
    println!("绝对时间差距: {:.1} ms", total_28_layers_ms - llama_time_ms);
}
