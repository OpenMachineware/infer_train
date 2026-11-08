#!/usr/bin/env cargo
//! 测量CPU算子绝对性能
//! 不依赖模型文件，直接测试计算核心

use std::arch::aarch64::*;
use std::time::{Duration, Instant};

fn main() {
    // Qwen3-0.6B 配置
    let n_tokens = 64;
    let n_heads = 16;
    let n_kv_heads = 8;
    let head_dim = 128;
    let hidden_size = 896;
    let intermediate_size = 4864;

    println!("=== CPU算子绝对性能测量 ===");
    println!("配置: {} tokens, {} heads, hidden={}\n",
             n_tokens, n_heads, hidden_size);

    // 测试数据
    let input: Vec<f32> = (0..hidden_size * n_tokens).map(|i| (i as f32 * 0.001).sin()).collect();
    let weight: Vec<f32> = (0..hidden_size * hidden_size).map(|i| (i as f32 * 0.001).cos()).collect();

    let iterations = 100;

    // 1. 测试GEMV (hidden_size × hidden_size)
    let mut gemv_time = Duration::new(0, 0);
    let mut output = vec![0.0f32; hidden_size];

    for _ in 0..iterations {
        let start = Instant::now();
        for i in 0..hidden_size {
            let mut sum = 0.0f32;
            for j in 0..hidden_size {
                sum += input[j] * weight[i * hidden_size + j];
            }
            output[i] = sum;
        }
        gemv_time += start.elapsed();
    }

    let gemv_ops = 2 * hidden_size * hidden_size;
    let gemv_gflops = gemv_ops as f64 * iterations as f64 / gemv_time.as_secs_f64() / 1e9;
    println!("GEMV ({}×{}):", hidden_size, hidden_size);
    println!("  时间: {:?}", gemv_time / iterations);
    println!("  GFLOPS: {:.1}", gemv_gflops);

    // 2. 测试GEMM (n_tokens × hidden_size × hidden_size)
    let mut gemm_time = Duration::new(0, 0);
    let mut output_batch = vec![0.0f32; hidden_size * n_tokens];

    for _ in 0..iterations {
        let start = Instant::now();
        for t in 0..n_tokens {
            for i in 0..hidden_size {
                let mut sum = 0.0f32;
                for j in 0..hidden_size {
                    sum += input[t * hidden_size + j] * weight[i * hidden_size + j];
                }
                output_batch[t * hidden_size + i] = sum;
            }
        }
        gemm_time += start.elapsed();
    }

    let gemm_ops = 2 * n_tokens * hidden_size * hidden_size;
    let gemm_gflops = gemm_ops as f64 * iterations as f64 / gemm_time.as_secs_f64() / 1e9;
    println!("\nGEMM ({}×{}×{}):", n_tokens, hidden_size, hidden_size);
    println!("  时间: {:?}", gemm_time / iterations);
    println!("  GFLOPS: {:.1}", gemm_gflops);

    // 3. 测试我们的simd_gemm
    let mut simd_gemm_time = Duration::new(0, 0);
    let a: Vec<f32> = (0..n_tokens * hidden_size).map(|i| (i as f32 * 0.001).sin()).collect();
    let b: Vec<f32> = (0..hidden_size * hidden_size).map(|i| (i as f32 * 0.001).cos()).collect();
    let mut c: Vec<f32> = vec![0.0; n_tokens * hidden_size];

    for _ in 0..iterations {
        let start = Instant::now();
        unsafe {
            infer_train::cpu::simd_gemm::simd_gemm(
                c.as_mut_ptr(),
                a.as_ptr(),
                b.as_ptr(),
                n_tokens,
                hidden_size,
                hidden_size,
            );
        }
        simd_gemm_time += start.elapsed();
    }

    let simd_gemm_gflops = gemm_ops as f64 * iterations as f64 / simd_gemm_time.as_secs_f64() / 1e9;
    println!("\n我们的simd_gemm:");
    println!("  时间: {:?}", simd_gemm_time / iterations);
    println!("  GFLOPS: {:.1}", simd_gemm_gflops);
    println!("  vs 标量GEMM: {:.2}x", gemm_gflops / simd_gemm_gflops);

    // 4. 计算理论forward时间
    println!("\n=== 理论forward时间计算 ===");

    // QKV: 3 × hidden × (n_heads × head_dim + 2 × n_kv_heads × head_dim)
    let qkv_size = n_heads * head_dim + 2 * n_kv_heads * head_dim;
    let qkv_ops = 3 * hidden_size * qkv_size * n_tokens;
    let qkv_time_us = qkv_ops as f64 / simd_gemm_gflops / 1e9 * 1e6;
    println!("QKV: {:.0} GFLOPs, 时间 {:.1} µs", qkv_ops as f64 / 1e9, qkv_time_us);

    // FFN: gate + up + down = 3 × hidden × intermediate_size
    let ffn_ops = 3 * hidden_size * intermediate_size * n_tokens;
    let ffn_time_us = ffn_ops as f64 / simd_gemm_gflops / 1e9 * 1e6;
    println!("FFN: {:.0} GFLOPs, 时间 {:.1} µs", ffn_ops as f64 / 1e9, ffn_time_us);

    // Attention (from our previous measurement)
    let attention_time_us = 41.779 * n_heads as f64; // per head × n_heads
    println!("Attention: {:.1} µs (实测)", attention_time_us);

    // Total per layer
    let layer_time_us = qkv_time_us + ffn_time_us + attention_time_us;
    println!("\n单层时间: {:.1} µs", layer_time_us);

    // All layers
    let n_layers = 28;
    let total_time_ms = layer_time_us * n_layers as f64 / 1000.0;
    println!("28层总时间: {:.1} ms", total_time_ms);

    let tokens_per_sec = n_tokens as f64 / total_time_ms * 1000.0;
    println!("理论吞吐量: {:.1} t/s", tokens_per_sec);

    println!("\n=== 对比llama.cpp ===");
    println!("llama.cpp吞吐量: 163.3 t/s");
    println!("差距: {:.2}x", tokens_per_sec / 163.3);
}
