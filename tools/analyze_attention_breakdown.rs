#!/usr/bin/env cargo
//! Detailed attention breakdown comparison with llama.cpp
//! Measure each step to find the actual bottleneck

use infer_train::*;
use infer_train::inference::KVCacheBase;
use std::time::{Duration, Instant};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        return;
    }

    let model_path = &args[1];

    // Load model
    let mut ctx = inference::InferenceContext::from_gguf(model_path, inference::Device::Cpu)
        .expect("Failed to load model");

    let n_heads = ctx.config.n_heads;
    let n_kv_heads = ctx.config.n_kv_heads;
    let head_dim = ctx.config.head_dim;

    println!("Model config: {} heads, {} kv_heads, head_dim {}", n_heads, n_kv_heads, head_dim);

    // Test configuration
    let n_tokens = 64;
    let seq_len = 64;

    println!("\nTest: {} tokens, seq_len {}", n_tokens, seq_len);

    // Create dummy Q, K, V
    let q_size = n_heads * head_dim * n_tokens;
    let kv_size = n_kv_heads * head_dim * n_tokens;
    let q: Vec<f32> = (0..q_size).map(|i| (i as f32 * 0.001).sin()).collect();
    let k: Vec<f32> = (0..kv_size).map(|i| (i as f32 * 0.001).cos()).collect();
    let v: Vec<f32> = (0..kv_size).map(|i| (i as f32 * 0.002).sin()).collect();

    let scale = 1.0f32 / (head_dim as f32).sqrt();

    println!("\n=== Step-by-step timing ===");

    // Create scratch
    let mut scratch = crate::cpu::FlashAttnPrefillScratch::new(head_dim);
    let mut out: Vec<f32> = vec![0.0; n_heads * head_dim * n_tokens];

    const Q_TILE_SZ: usize = 64;
    const KV_TILE_SZ: usize = 64;

    let mut time_load_k = Duration::new(0, 0);
    let mut time_load_v = Duration::new(0, 0);
    let mut time_load_q = Duration::new(0, 0);
    let mut time_gemm_qk = Duration::new(0, 0);
    let mut time_softmax = Duration::new(0, 0);
    let mut time_gemm_vkq = Duration::new(0, 0);
    let mut time_normalize = Duration::new(0, 0);

    // Measure one layer's attention
    let n_iterations: usize = 10;

    for _ in 0..n_iterations {
        // Clear output
        out.fill(0.0);

        // Simulate tiled processing for one head
        let h = 0;
        let kv_head = 0;
        let tile_rows = n_tokens.min(Q_TILE_SZ);
        let kv_tiles = (seq_len + KV_TILE_SZ - 1) / KV_TILE_SZ;

        for kv_tile_idx in 0..kv_tiles {
            let kv_tile_start = kv_tile_idx * KV_TILE_SZ;
            let kv_tile_end = (kv_tile_start + KV_TILE_SZ).min(seq_len);
            let kv_tile_size = kv_tile_end - kv_tile_start;

            // Load K tile transposed
            let start = Instant::now();
            for tk in 0..kv_tile_size {
                let kv_pos = kv_tile_start + tk;
                for d in 0..head_dim {
                    let k_offset = kv_pos * kv_size / n_tokens + kv_head * head_dim + d;
                    scratch.k_tile[d * KV_TILE_SZ + tk] = k[k_offset];
                }
            }
            time_load_k += start.elapsed();

            // Load V tile
            let start = Instant::now();
            for tk in 0..kv_tile_size {
                let kv_pos = kv_tile_start + tk;
                for d in 0..head_dim {
                    let v_offset = kv_pos * kv_size / n_tokens + kv_head * head_dim + d;
                    scratch.v_tile[tk * head_dim + d] = v[v_offset];
                }
            }
            time_load_v += start.elapsed();

            // Load Q tile
            let start = Instant::now();
            for tq in 0..tile_rows {
                let q_offset = tq * q_size / n_tokens + h * head_dim;
                scratch.q_tile[tq * head_dim..(tq + 1) * head_dim]
                    .copy_from_slice(&q[q_offset..q_offset + head_dim]);
            }
            time_load_q += start.elapsed();

            // GEMM: KQ = Q @ K^T
            let start = Instant::now();
            unsafe {
                crate::cpu::simd_gemm::simd_gemm(
                    scratch.kq.as_mut_ptr(),
                    scratch.q_tile.as_ptr(),
                    scratch.k_tile.as_ptr(),
                    tile_rows,
                    head_dim,
                    kv_tile_size,
                );
            }
            time_gemm_qk += start.elapsed();

            // Apply scale and softmax
            let start = Instant::now();
            for i in 0..tile_rows {
                // Apply scale
                for j in 0..kv_tile_size {
                    scratch.kq[i * KV_TILE_SZ + j] *= scale;
                }

                // Find max
                let mut max_score = f32::NEG_INFINITY;
                for j in 0..kv_tile_size {
                    max_score = max_score.max(scratch.kq[i * KV_TILE_SZ + j]);
                }

                // Exp and sum
                let mut sum_exp = 0.0f32;
                for j in 0..kv_tile_size {
                    scratch.kq[i * KV_TILE_SZ + j] = (scratch.kq[i * KV_TILE_SZ + j] - max_score).exp();
                    sum_exp += scratch.kq[i * KV_TILE_SZ + j];
                }

                // Normalize weights
                for j in 0..kv_tile_size {
                    scratch.kq[i * KV_TILE_SZ + j] /= sum_exp;
                }
            }
            time_softmax += start.elapsed();

            // GEMM: VKQ = KQ @ V
            let start = Instant::now();
            unsafe {
                crate::cpu::simd_gemm::simd_gemm(
                    scratch.vkq.as_mut_ptr(),
                    scratch.kq.as_ptr(),
                    scratch.v_tile.as_ptr(),
                    tile_rows,
                    kv_tile_size,
                    head_dim,
                );
            }
            time_gemm_vkq += start.elapsed();
        }
    }

    println!("\nPer-layer timing (average over {} iterations):", n_iterations);
    println!("Load K tile:     {:?}", time_load_k / n_iterations as u32);
    println!("Load V tile:     {:?}", time_load_v / n_iterations as u32);
    println!("Load Q tile:     {:?}", time_load_q / n_iterations as u32);
    println!("GEMM Q·K:        {:?}", time_gemm_qk / n_iterations as u32);
    println!("Softmax:         {:?}", time_softmax / n_iterations as u32);
    println!("GEMM VKQ:        {:?}", time_gemm_vkq / n_iterations as u32);
    println!("Total:           {:?}", (time_load_k + time_load_v + time_load_q + time_gemm_qk + time_softmax + time_gemm_vkq) / n_iterations as u32);

    let total_gemm_time = time_gemm_qk + time_gemm_vkq;
    let total_load_time = time_load_k + time_load_v + time_load_q;

    println!("\n=== Breakdown ===");
    println!("GEMM total:      {:?} ({:.1}%)", total_gemm_time / n_iterations as u32,
             total_gemm_time.as_secs_f64() / (total_gemm_time + total_load_time + time_softmax).as_secs_f64() * 100.0);
    println!("Load total:      {:?} ({:.1}%)", total_load_time / n_iterations as u32,
             total_load_time.as_secs_f64() / (total_gemm_time + total_load_time + time_softmax).as_secs_f64() * 100.0);
    println!("Softmax:         {:?} ({:.1}%)", time_softmax / n_iterations as u32,
             time_softmax.as_secs_f64() / (total_gemm_time + total_load_time + time_softmax).as_secs_f64() * 100.0);

    // Calculate FLOPs
    let avg_tile_rows = n_tokens;
    let avg_kv_tile_size = KV_TILE_SZ;
    let avg_kv_tiles = (seq_len + KV_TILE_SZ - 1) / KV_TILE_SZ;

    let gemm_qk_flops = 2 * avg_tile_rows * head_dim * avg_kv_tile_size * avg_kv_tiles * n_iterations;
    let gemm_vkq_flops = 2 * avg_tile_rows * avg_kv_tile_size * head_dim * avg_kv_tiles * n_iterations;
    let total_gemm_flops = gemm_qk_flops + gemm_vkq_flops;

    println!("\n=== GEMM Performance ===");
    println!("GEMM FLOPs:      {:.3}M", total_gemm_flops as f64 / 1e6);
    println!("GEMM time:       {:?}", total_gemm_time);
    println!("GEMM GFLOPS:     {:.1}", total_gemm_flops as f64 / total_gemm_time.as_secs_f64() / 1e9);
}
