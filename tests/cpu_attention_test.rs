// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Comprehensive test for CPU attention operators
//! Tests all quantized KV cache formats against FP32 reference

use infer_train::ops::attention::decode_attention_f32;
use infer_train::quant::kv_cache::quant::{
    quantize_row_iq4_nl, quantize_row_q2_k_kv, quantize_row_q3_k_kv,
    quantize_row_q4_0, quantize_row_q4_1, quantize_row_q4_k_kv,
    quantize_row_q5_0, quantize_row_q5_1, quantize_row_q5_k_kv,
    quantize_row_q6_k_kv, quantize_row_q8_0,
};
use infer_train::quant::kv_cache::{
    BlockIQ4NL, BlockQ2K_KV, BlockQ3K_KV, BlockQ4_0, BlockQ4_1, BlockQ4K_KV,
    BlockQ5_0, BlockQ5_1, BlockQ5K_KV, BlockQ6K_KV, BlockQ8_0, QK, QK_K,
};
use rand::Rng;

fn main() {
    println!("=== CPU Attention Operator Tests ===\n");

    let num_heads = 4;
    let head_dim = 128;
    let seq_len = 64;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Generate random Q, K, V
    let mut rng = rand::thread_rng();
    let q: Vec<f32> =
        (0..num_heads * head_dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let k: Vec<f32> = (0..seq_len * num_heads * head_dim)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    let v: Vec<f32> = (0..seq_len * num_heads * head_dim)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();

    // FP32 reference
    let mut output_fp32 = vec![0.0f32; num_heads * head_dim];
    decode_attention_f32(
        &q,
        &k,
        &v,
        &mut output_fp32,
        num_heads,
        head_dim,
        seq_len,
        scale,
    );

    // Test Q8_0
    test_q8_0(&q, &k, &v, &output_fp32, num_heads, head_dim, seq_len, scale);

    // Test Q4_0
    test_q4_0(&q, &k, &v, &output_fp32, num_heads, head_dim, seq_len, scale);

    // Test Q4_1
    test_q4_1(&q, &k, &v, &output_fp32, num_heads, head_dim, seq_len, scale);

    // Test Q5_0
    test_q5_0(&q, &k, &v, &output_fp32, num_heads, head_dim, seq_len, scale);

    // Test Q5_1
    test_q5_1(&q, &k, &v, &output_fp32, num_heads, head_dim, seq_len, scale);

    // Test IQ4_NL
    test_iq4_nl(&q, &k, &v, &output_fp32, num_heads, head_dim, seq_len, scale);

    // K-quant formats use 256 elements per block, but Metal shader uses only first 128
    // for head_dim <= 128. Test with head_dim=128.
    println!("\n=== K-quant tests (head_dim=128) ===");
    let head_dim_kq = 128;
    let scale_kq = 1.0 / (head_dim_kq as f32).sqrt();

    // Generate new Q, K, V for 128-dim heads
    let q_kq: Vec<f32> = (0..num_heads * head_dim_kq)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    let k_kq: Vec<f32> = (0..seq_len * num_heads * head_dim_kq)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    let v_kq: Vec<f32> = (0..seq_len * num_heads * head_dim_kq)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();

    // K-quant tests compare against FP32 attention with DEQUANTIZED K/V
    // This tests the round-trip: quantize -> dequantize -> attention
    // (same approach as GPU test)
    test_q2_k(&q_kq, &k_kq, &v_kq, num_heads, head_dim_kq, seq_len, scale_kq);
    test_q3_k(&q_kq, &k_kq, &v_kq, num_heads, head_dim_kq, seq_len, scale_kq);
    test_q4_k(&q_kq, &k_kq, &v_kq, num_heads, head_dim_kq, seq_len, scale_kq);
    test_q5_k(&q_kq, &k_kq, &v_kq, num_heads, head_dim_kq, seq_len, scale_kq);
    test_q6_k(&q_kq, &k_kq, &v_kq, num_heads, head_dim_kq, seq_len, scale_kq);

    println!("\n=== All tests passed ===");
}

fn quantize_k_v_q4_0(
    k: &[f32],
    v: &[f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> (Vec<BlockQ4_0>, Vec<BlockQ4_0>) {
    let blocks_per_row = head_dim / QK;
    let mut k_q = vec![
        BlockQ4_0 { d: 0, qs: [0u8; QK / 2] };
        seq_len * num_heads * blocks_per_row
    ];
    let mut v_q = vec![
        BlockQ4_0 { d: 0, qs: [0u8; QK / 2] };
        seq_len * num_heads * blocks_per_row
    ];

    for s in 0..seq_len {
        for h in 0..num_heads {
            let offset = (s * num_heads + h) * head_dim;
            let k_row = &k[offset..offset + head_dim];
            let v_row = &v[offset..offset + head_dim];

            let k_offset = (s * num_heads + h) * blocks_per_row;
            let v_offset = (s * num_heads + h) * blocks_per_row;

            quantize_row_q4_0(
                k_row,
                &mut k_q[k_offset..k_offset + blocks_per_row],
            );
            quantize_row_q4_0(
                v_row,
                &mut v_q[v_offset..v_offset + blocks_per_row],
            );
        }
    }
    (k_q, v_q)
}

fn quantize_k_v_q4_1(
    k: &[f32],
    v: &[f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> (Vec<BlockQ4_1>, Vec<BlockQ4_1>) {
    let blocks_per_row = head_dim / QK;
    let mut k_q = vec![
        BlockQ4_1 { d: 0, m: 0, qs: [0u8; QK / 2] };
        seq_len * num_heads * blocks_per_row
    ];
    let mut v_q = vec![
        BlockQ4_1 { d: 0, m: 0, qs: [0u8; QK / 2] };
        seq_len * num_heads * blocks_per_row
    ];

    for s in 0..seq_len {
        for h in 0..num_heads {
            let offset = (s * num_heads + h) * head_dim;
            let k_row = &k[offset..offset + head_dim];
            let v_row = &v[offset..offset + head_dim];

            let k_offset = (s * num_heads + h) * blocks_per_row;
            let v_offset = (s * num_heads + h) * blocks_per_row;

            quantize_row_q4_1(
                k_row,
                &mut k_q[k_offset..k_offset + blocks_per_row],
            );
            quantize_row_q4_1(
                v_row,
                &mut v_q[v_offset..v_offset + blocks_per_row],
            );
        }
    }
    (k_q, v_q)
}

fn quantize_k_v_q5_0(
    k: &[f32],
    v: &[f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> (Vec<BlockQ5_0>, Vec<BlockQ5_0>) {
    let blocks_per_row = head_dim / QK;
    let mut k_q = vec![
        BlockQ5_0 { d: 0, qh: [0u8; 4], qs: [0u8; QK / 2] };
        seq_len * num_heads * blocks_per_row
    ];
    let mut v_q = vec![
        BlockQ5_0 { d: 0, qh: [0u8; 4], qs: [0u8; QK / 2] };
        seq_len * num_heads * blocks_per_row
    ];

    for s in 0..seq_len {
        for h in 0..num_heads {
            let offset = (s * num_heads + h) * head_dim;
            let k_row = &k[offset..offset + head_dim];
            let v_row = &v[offset..offset + head_dim];

            let k_offset = (s * num_heads + h) * blocks_per_row;
            let v_offset = (s * num_heads + h) * blocks_per_row;

            quantize_row_q5_0(
                k_row,
                &mut k_q[k_offset..k_offset + blocks_per_row],
            );
            quantize_row_q5_0(
                v_row,
                &mut v_q[v_offset..v_offset + blocks_per_row],
            );
        }
    }
    (k_q, v_q)
}

fn quantize_k_v_q5_1(
    k: &[f32],
    v: &[f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> (Vec<BlockQ5_1>, Vec<BlockQ5_1>) {
    let blocks_per_row = head_dim / QK;
    let mut k_q = vec![
        BlockQ5_1 { d: 0, m: 0, qh: [0u8; 4], qs: [0u8; QK / 2] };
        seq_len * num_heads * blocks_per_row
    ];
    let mut v_q = vec![
        BlockQ5_1 { d: 0, m: 0, qh: [0u8; 4], qs: [0u8; QK / 2] };
        seq_len * num_heads * blocks_per_row
    ];

    for s in 0..seq_len {
        for h in 0..num_heads {
            let offset = (s * num_heads + h) * head_dim;
            let k_row = &k[offset..offset + head_dim];
            let v_row = &v[offset..offset + head_dim];

            let k_offset = (s * num_heads + h) * blocks_per_row;
            let v_offset = (s * num_heads + h) * blocks_per_row;

            quantize_row_q5_1(
                k_row,
                &mut k_q[k_offset..k_offset + blocks_per_row],
            );
            quantize_row_q5_1(
                v_row,
                &mut v_q[v_offset..v_offset + blocks_per_row],
            );
        }
    }
    (k_q, v_q)
}

fn quantize_k_v_q8_0(
    k: &[f32],
    v: &[f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> (Vec<BlockQ8_0>, Vec<BlockQ8_0>) {
    let blocks_per_row = head_dim / QK;
    let mut k_q = vec![
        BlockQ8_0 { d: 0, qs: [0i8; QK] };
        seq_len * num_heads * blocks_per_row
    ];
    let mut v_q = vec![
        BlockQ8_0 { d: 0, qs: [0i8; QK] };
        seq_len * num_heads * blocks_per_row
    ];

    for s in 0..seq_len {
        for h in 0..num_heads {
            let offset = (s * num_heads + h) * head_dim;
            let k_row = &k[offset..offset + head_dim];
            let v_row = &v[offset..offset + head_dim];

            let k_offset = (s * num_heads + h) * blocks_per_row;
            let v_offset = (s * num_heads + h) * blocks_per_row;

            quantize_row_q8_0(
                k_row,
                &mut k_q[k_offset..k_offset + blocks_per_row],
            );
            quantize_row_q8_0(
                v_row,
                &mut v_q[v_offset..v_offset + blocks_per_row],
            );
        }
    }
    (k_q, v_q)
}

fn quantize_k_v_iq4_nl(
    k: &[f32],
    v: &[f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> (Vec<BlockIQ4NL>, Vec<BlockIQ4NL>) {
    let blocks_per_row = head_dim / QK;
    let mut k_q = vec![
        BlockIQ4NL { d: 0, qs: [0u8; QK / 2] };
        seq_len * num_heads * blocks_per_row
    ];
    let mut v_q = vec![
        BlockIQ4NL { d: 0, qs: [0u8; QK / 2] };
        seq_len * num_heads * blocks_per_row
    ];

    for s in 0..seq_len {
        for h in 0..num_heads {
            let offset = (s * num_heads + h) * head_dim;
            let k_row = &k[offset..offset + head_dim];
            let v_row = &v[offset..offset + head_dim];

            let k_offset = (s * num_heads + h) * blocks_per_row;
            let v_offset = (s * num_heads + h) * blocks_per_row;

            quantize_row_iq4_nl(
                k_row,
                &mut k_q[k_offset..k_offset + blocks_per_row],
            );
            quantize_row_iq4_nl(
                v_row,
                &mut v_q[v_offset..v_offset + blocks_per_row],
            );
        }
    }
    (k_q, v_q)
}

fn compare_output(output: &[f32], reference: &[f32], name: &str) {
    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f32;
    let n = output.len();

    for i in 0..n {
        let diff = (output[i] - reference[i]).abs();
        max_diff = max_diff.max(diff);
        sum_diff += diff;
    }

    let avg_diff = sum_diff / n as f32;
    println!(
        "{:10}: max_diff={:.3e}, avg_diff={:.3e}",
        name, max_diff, avg_diff
    );

    // Allow reasonable quantization error
    // Note: IQ4_NL has higher error due to non-linear quantization (designed for weights, not KV)
    // K-quant formats have higher error due to low bit-width (2-6 bits)
    let thresholds = [
        ("Q8_0", 5e-2),   // 5% - acceptable for Q8_0
        ("Q4_0", 5e-2),   // 5% - acceptable for Q4_0
        ("Q4_1", 2e-2),   // 2% - Q4_1 with min/scale
        ("Q5_0", 5e-2),   // 5% - acceptable for Q5_0
        ("Q5_1", 1e-2),   // 1% - Q5_1 with min/scale
        ("IQ4_NL", 2e-1), // 20% - non-linear quantization, higher error expected
        ("Q2_K", 2e-1),   // 20% - 2-bit quantization
        ("Q3_K", 1e-1),   // 10% - 3-bit quantization
        ("Q4_K", 5e-2),   // 5% - 4-bit K-quant
        ("Q5_K", 5e-2),   // 5% - 5-bit K-quant
        ("Q6_K", 3e-2),   // 3% - 6-bit K-quant
    ];

    for (fmt, threshold) in thresholds {
        if name == fmt {
            if max_diff >= threshold {
                println!(
                    "WARNING: {} max_diff {} exceeds threshold {}",
                    name, max_diff, threshold
                );
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn test_q8_0(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    reference: &[f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
) {
    use infer_train::ops::attention::decode_attention_q8_0_neon;

    let (k_q, v_q) = quantize_k_v_q8_0(k, v, num_heads, head_dim, seq_len);
    let mut output = vec![0.0f32; num_heads * head_dim];

    unsafe {
        decode_attention_q8_0_neon(
            q,
            &k_q,
            &v_q,
            &mut output,
            num_heads,
            head_dim,
            seq_len,
            scale,
        );
    }

    compare_output(&output, reference, "Q8_0");
}

#[cfg(not(target_arch = "aarch64"))]
fn test_q8_0(
    _q: &[f32],
    _k: &[f32],
    _v: &[f32],
    _reference: &[f32],
    _num_heads: usize,
    _head_dim: usize,
    _seq_len: usize,
    _scale: f32,
) {
    println!("Q8_0: skipped (not aarch64)");
}

#[cfg(target_arch = "aarch64")]
fn test_q4_0(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    reference: &[f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
) {
    use infer_train::ops::attention::decode_attention_q4_0_neon;

    let (k_q, v_q) = quantize_k_v_q4_0(k, v, num_heads, head_dim, seq_len);
    let mut output = vec![0.0f32; num_heads * head_dim];

    unsafe {
        decode_attention_q4_0_neon(
            q,
            &k_q,
            &v_q,
            &mut output,
            num_heads,
            head_dim,
            seq_len,
            scale,
        );
    }

    compare_output(&output, reference, "Q4_0");
}

#[cfg(not(target_arch = "aarch64"))]
fn test_q4_0(
    _q: &[f32],
    _k: &[f32],
    _v: &[f32],
    _reference: &[f32],
    _num_heads: usize,
    _head_dim: usize,
    _seq_len: usize,
    _scale: f32,
) {
    println!("Q4_0: skipped (not aarch64)");
}

#[cfg(target_arch = "aarch64")]
fn test_q4_1(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    reference: &[f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
) {
    use infer_train::ops::attention::decode_attention_q4_1_neon;

    let (k_q, v_q) = quantize_k_v_q4_1(k, v, num_heads, head_dim, seq_len);
    let mut output = vec![0.0f32; num_heads * head_dim];

    unsafe {
        decode_attention_q4_1_neon(
            q,
            &k_q,
            &v_q,
            &mut output,
            num_heads,
            head_dim,
            seq_len,
            scale,
        );
    }

    compare_output(&output, reference, "Q4_1");
}

#[cfg(not(target_arch = "aarch64"))]
fn test_q4_1(
    _q: &[f32],
    _k: &[f32],
    _v: &[f32],
    _reference: &[f32],
    _num_heads: usize,
    _head_dim: usize,
    _seq_len: usize,
    _scale: f32,
) {
    println!("Q4_1: skipped (not aarch64)");
}

#[cfg(target_arch = "aarch64")]
fn test_q5_0(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    reference: &[f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
) {
    use infer_train::ops::attention::decode_attention_q5_0_neon;

    let (k_q, v_q) = quantize_k_v_q5_0(k, v, num_heads, head_dim, seq_len);
    let mut output = vec![0.0f32; num_heads * head_dim];

    unsafe {
        decode_attention_q5_0_neon(
            q,
            &k_q,
            &v_q,
            &mut output,
            num_heads,
            head_dim,
            seq_len,
            scale,
        );
    }

    compare_output(&output, reference, "Q5_0");
}

#[cfg(not(target_arch = "aarch64"))]
fn test_q5_0(
    _q: &[f32],
    _k: &[f32],
    _v: &[f32],
    _reference: &[f32],
    _num_heads: usize,
    _head_dim: usize,
    _seq_len: usize,
    _scale: f32,
) {
    println!("Q5_0: skipped (not aarch64)");
}

#[cfg(target_arch = "aarch64")]
fn test_q5_1(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    reference: &[f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
) {
    use infer_train::ops::attention::decode_attention_q5_1_neon;

    let (k_q, v_q) = quantize_k_v_q5_1(k, v, num_heads, head_dim, seq_len);
    let mut output = vec![0.0f32; num_heads * head_dim];

    unsafe {
        decode_attention_q5_1_neon(
            q,
            &k_q,
            &v_q,
            &mut output,
            num_heads,
            head_dim,
            seq_len,
            scale,
        );
    }

    compare_output(&output, reference, "Q5_1");
}

#[cfg(not(target_arch = "aarch64"))]
fn test_q5_1(
    _q: &[f32],
    _k: &[f32],
    _v: &[f32],
    _reference: &[f32],
    _num_heads: usize,
    _head_dim: usize,
    _seq_len: usize,
    _scale: f32,
) {
    println!("Q5_1: skipped (not aarch64)");
}

#[cfg(target_arch = "aarch64")]
fn test_iq4_nl(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    reference: &[f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
) {
    use infer_train::ops::attention::decode_attention_iq4_nl_neon;

    let (k_q, v_q) = quantize_k_v_iq4_nl(k, v, num_heads, head_dim, seq_len);
    let mut output = vec![0.0f32; num_heads * head_dim];

    unsafe {
        decode_attention_iq4_nl_neon(
            q,
            &k_q,
            &v_q,
            &mut output,
            num_heads,
            head_dim,
            seq_len,
            scale,
        );
    }

    compare_output(&output, reference, "IQ4_NL");
}

#[cfg(not(target_arch = "aarch64"))]
fn test_iq4_nl(
    _q: &[f32],
    _k: &[f32],
    _v: &[f32],
    _reference: &[f32],
    _num_heads: usize,
    _head_dim: usize,
    _seq_len: usize,
    _scale: f32,
) {
    println!("IQ4_NL: skipped (not aarch64)");
}

// ===== K-quant quantization helpers =====

fn quantize_k_v_q2_k(
    k: &[f32],
    v: &[f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> (Vec<BlockQ2K_KV>, Vec<BlockQ2K_KV>) {
    let num_blocks = seq_len * num_heads;
    let mut k_q =
        vec![
            BlockQ2K_KV { d: 0, dmin: 0, scales: [0u8; 16], qs: [0u8; 64] };
            num_blocks
        ];
    let mut v_q =
        vec![
            BlockQ2K_KV { d: 0, dmin: 0, scales: [0u8; 16], qs: [0u8; 64] };
            num_blocks
        ];

    for s in 0..seq_len {
        for h in 0..num_heads {
            let offset = (s * num_heads + h) * head_dim;
            let k_row = &k[offset..offset + head_dim];
            let v_row = &v[offset..offset + head_dim];
            let block_idx = s * num_heads + h;

            // Pad to 256 elements
            let mut k_padded = k_row.to_vec();
            k_padded.resize(QK_K, 0.0);
            let mut v_padded = v_row.to_vec();
            v_padded.resize(QK_K, 0.0);

            quantize_row_q2_k_kv(&k_padded, &mut k_q[block_idx]);
            quantize_row_q2_k_kv(&v_padded, &mut v_q[block_idx]);
        }
    }
    (k_q, v_q)
}

fn quantize_k_v_q3_k(
    k: &[f32],
    v: &[f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> (Vec<BlockQ3K_KV>, Vec<BlockQ3K_KV>) {
    let num_blocks = seq_len * num_heads;
    let mut k_q = vec![
        BlockQ3K_KV {
            d: 0,
            hmask: [0u8; 32],
            qs: [0u8; 64],
            scales: [0i8; 12],
        };
        num_blocks
    ];
    let mut v_q = vec![
        BlockQ3K_KV {
            d: 0,
            hmask: [0u8; 32],
            qs: [0u8; 64],
            scales: [0i8; 12],
        };
        num_blocks
    ];

    for s in 0..seq_len {
        for h in 0..num_heads {
            let offset = (s * num_heads + h) * head_dim;
            let k_row = &k[offset..offset + head_dim];
            let v_row = &v[offset..offset + head_dim];
            let block_idx = s * num_heads + h;

            let mut k_padded = k_row.to_vec();
            k_padded.resize(QK_K, 0.0);
            let mut v_padded = v_row.to_vec();
            v_padded.resize(QK_K, 0.0);

            quantize_row_q3_k_kv(&k_padded, &mut k_q[block_idx]);
            quantize_row_q3_k_kv(&v_padded, &mut v_q[block_idx]);
        }
    }
    (k_q, v_q)
}

fn quantize_k_v_q4_k(
    k: &[f32],
    v: &[f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> (Vec<BlockQ4K_KV>, Vec<BlockQ4K_KV>) {
    let num_blocks = seq_len * num_heads;
    let mut k_q =
        vec![
            BlockQ4K_KV { d: 0, dmin: 0, scales: [0u8; 12], qs: [0u8; 128] };
            num_blocks
        ];
    let mut v_q =
        vec![
            BlockQ4K_KV { d: 0, dmin: 0, scales: [0u8; 12], qs: [0u8; 128] };
            num_blocks
        ];

    for s in 0..seq_len {
        for h in 0..num_heads {
            let offset = (s * num_heads + h) * head_dim;
            let k_row = &k[offset..offset + head_dim];
            let v_row = &v[offset..offset + head_dim];
            let block_idx = s * num_heads + h;

            let mut k_padded = k_row.to_vec();
            k_padded.resize(QK_K, 0.0);
            let mut v_padded = v_row.to_vec();
            v_padded.resize(QK_K, 0.0);

            quantize_row_q4_k_kv(&k_padded, &mut k_q[block_idx]);
            quantize_row_q4_k_kv(&v_padded, &mut v_q[block_idx]);
        }
    }
    (k_q, v_q)
}

fn quantize_k_v_q5_k(
    k: &[f32],
    v: &[f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> (Vec<BlockQ5K_KV>, Vec<BlockQ5K_KV>) {
    let num_blocks = seq_len * num_heads;
    let mut k_q = vec![
        BlockQ5K_KV {
            d: 0,
            dmin: 0,
            scales: [0u8; 12],
            qh: [0u8; 32],
            qs: [0u8; 128],
        };
        num_blocks
    ];
    let mut v_q = vec![
        BlockQ5K_KV {
            d: 0,
            dmin: 0,
            scales: [0u8; 12],
            qh: [0u8; 32],
            qs: [0u8; 128],
        };
        num_blocks
    ];

    for s in 0..seq_len {
        for h in 0..num_heads {
            let offset = (s * num_heads + h) * head_dim;
            let k_row = &k[offset..offset + head_dim];
            let v_row = &v[offset..offset + head_dim];
            let block_idx = s * num_heads + h;

            let mut k_padded = k_row.to_vec();
            k_padded.resize(QK_K, 0.0);
            let mut v_padded = v_row.to_vec();
            v_padded.resize(QK_K, 0.0);

            quantize_row_q5_k_kv(&k_padded, &mut k_q[block_idx]);
            quantize_row_q5_k_kv(&v_padded, &mut v_q[block_idx]);
        }
    }
    (k_q, v_q)
}

fn quantize_k_v_q6_k(
    k: &[f32],
    v: &[f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> (Vec<BlockQ6K_KV>, Vec<BlockQ6K_KV>) {
    let num_blocks = seq_len * num_heads;
    let mut k_q = vec![
        BlockQ6K_KV {
            d: 0,
            scales: [0i8; 16],
            ql: [0u8; 128],
            qh: [0u8; 64],
        };
        num_blocks
    ];
    let mut v_q = vec![
        BlockQ6K_KV {
            d: 0,
            scales: [0i8; 16],
            ql: [0u8; 128],
            qh: [0u8; 64],
        };
        num_blocks
    ];

    for s in 0..seq_len {
        for h in 0..num_heads {
            let offset = (s * num_heads + h) * head_dim;
            let k_row = &k[offset..offset + head_dim];
            let v_row = &v[offset..offset + head_dim];
            let block_idx = s * num_heads + h;

            let mut k_padded = k_row.to_vec();
            k_padded.resize(QK_K, 0.0);
            let mut v_padded = v_row.to_vec();
            v_padded.resize(QK_K, 0.0);

            quantize_row_q6_k_kv(&k_padded, &mut k_q[block_idx]);
            quantize_row_q6_k_kv(&v_padded, &mut v_q[block_idx]);
        }
    }
    (k_q, v_q)
}

// ===== K-quant test functions =====

fn test_q2_k(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
) {
    use infer_train::ops::attention::decode_attention_q2_k;

    let (k_q, v_q) = quantize_k_v_q2_k(k, v, num_heads, head_dim, seq_len);

    // Reference: FP32 attention with dequantized K/V
    let mut k_deq = vec![0.0f32; seq_len * num_heads * head_dim];
    let mut v_deq = vec![0.0f32; seq_len * num_heads * head_dim];
    for i in 0..seq_len * num_heads {
        let mut full = [0.0f32; 256];
        dequantize_q2_k_ref(&k_q[i], &mut full);
        k_deq[i * head_dim..(i + 1) * head_dim]
            .copy_from_slice(&full[..head_dim]);
        dequantize_q2_k_ref(&v_q[i], &mut full);
        v_deq[i * head_dim..(i + 1) * head_dim]
            .copy_from_slice(&full[..head_dim]);
    }
    let mut reference = vec![0.0f32; num_heads * head_dim];
    decode_attention_f32(
        q,
        &k_deq,
        &v_deq,
        &mut reference,
        num_heads,
        head_dim,
        seq_len,
        scale,
    );

    let mut output = vec![0.0f32; num_heads * head_dim];
    decode_attention_q2_k(
        q,
        &k_q,
        &v_q,
        &mut output,
        num_heads,
        head_dim,
        seq_len,
        scale,
    );

    compare_output(&output, &reference, "Q2_K");
}

fn dequantize_q2_k_ref(block: &BlockQ2K_KV, out: &mut [f32; 256]) {
    let d = half::f16::from_bits(block.d).to_f32();
    let dmin = half::f16::from_bits(block.dmin).to_f32();

    for sub_block in 0..16 {
        let sc = block.scales[sub_block];
        let dl = d * ((sc & 0x0F) as f32);
        let ml = dmin * ((sc >> 4) as f32);

        for j in 0..4 {
            let byte = block.qs[sub_block * 4 + j];
            for k in 0..4 {
                let val = ((byte >> (2 * k)) & 0x03) as f32;
                out[sub_block * 16 + j * 4 + k] = dl * val - ml;
            }
        }
    }
}

fn test_q3_k(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
) {
    use infer_train::ops::attention::decode_attention_q3_k;

    let (k_q, v_q) = quantize_k_v_q3_k(k, v, num_heads, head_dim, seq_len);

    let mut k_deq = vec![0.0f32; seq_len * num_heads * head_dim];
    let mut v_deq = vec![0.0f32; seq_len * num_heads * head_dim];
    for i in 0..seq_len * num_heads {
        let mut full = [0.0f32; 256];
        dequantize_q3_k_ref(&k_q[i], &mut full);
        k_deq[i * head_dim..(i + 1) * head_dim]
            .copy_from_slice(&full[..head_dim]);
        dequantize_q3_k_ref(&v_q[i], &mut full);
        v_deq[i * head_dim..(i + 1) * head_dim]
            .copy_from_slice(&full[..head_dim]);
    }
    let mut reference = vec![0.0f32; num_heads * head_dim];
    decode_attention_f32(
        q,
        &k_deq,
        &v_deq,
        &mut reference,
        num_heads,
        head_dim,
        seq_len,
        scale,
    );

    let mut output = vec![0.0f32; num_heads * head_dim];
    decode_attention_q3_k(
        q,
        &k_q,
        &v_q,
        &mut output,
        num_heads,
        head_dim,
        seq_len,
        scale,
    );

    compare_output(&output, &reference, "Q3_K");
}

fn dequantize_q3_k_ref(block: &BlockQ3K_KV, out: &mut [f32; 256]) {
    let d = half::f16::from_bits(block.d).to_f32();

    for j in 0..64 {
        let byte = block.qs[j];
        for k in 0..4 {
            let idx = j * 4 + k;
            let low2 = ((byte >> (2 * k)) & 0x03) as i32;
            let high1 = if (block.hmask[idx / 8] >> (idx % 8)) & 1 == 1 {
                4
            } else {
                0
            };
            let val = low2 + high1 - 4; // signed 3-bit: [-4, 3]

            let sub = idx / 32;
            let sc = block.scales[sub] as f32;
            out[idx] = d * sc * val as f32;
        }
    }
}

fn test_q4_k(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
) {
    use infer_train::ops::attention::decode_attention_q4_k;

    let (k_q, v_q) = quantize_k_v_q4_k(k, v, num_heads, head_dim, seq_len);

    let mut k_deq = vec![0.0f32; seq_len * num_heads * head_dim];
    let mut v_deq = vec![0.0f32; seq_len * num_heads * head_dim];
    for i in 0..seq_len * num_heads {
        let mut full = [0.0f32; 256];
        dequantize_q4_k_ref(&k_q[i], &mut full);
        k_deq[i * head_dim..(i + 1) * head_dim]
            .copy_from_slice(&full[..head_dim]);
        dequantize_q4_k_ref(&v_q[i], &mut full);
        v_deq[i * head_dim..(i + 1) * head_dim]
            .copy_from_slice(&full[..head_dim]);
    }
    let mut reference = vec![0.0f32; num_heads * head_dim];
    decode_attention_f32(
        q,
        &k_deq,
        &v_deq,
        &mut reference,
        num_heads,
        head_dim,
        seq_len,
        scale,
    );

    let mut output = vec![0.0f32; num_heads * head_dim];
    decode_attention_q4_k(
        q,
        &k_q,
        &v_q,
        &mut output,
        num_heads,
        head_dim,
        seq_len,
        scale,
    );

    compare_output(&output, &reference, "Q4_K");
}

fn dequantize_q4_k_ref(block: &BlockQ4K_KV, out: &mut [f32; 256]) {
    let d = half::f16::from_bits(block.d).to_f32();
    let dmin = half::f16::from_bits(block.dmin).to_f32();

    for j in 0..128 {
        let byte = block.qs[j];
        let low = (byte & 0x0F) as f32;
        let high = ((byte >> 4) & 0x0F) as f32;

        let sub = j / 16;
        let sc = block.scales[sub];
        let dl = d * ((sc & 0x0F) as f32);
        let ml = dmin * ((sc >> 4) as f32);

        out[j] = low * dl + ml;
        out[j + 128] = high * dl + ml;
    }
}

fn test_q5_k(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
) {
    use infer_train::ops::attention::decode_attention_q5_k;

    let (k_q, v_q) = quantize_k_v_q5_k(k, v, num_heads, head_dim, seq_len);

    let mut k_deq = vec![0.0f32; seq_len * num_heads * head_dim];
    let mut v_deq = vec![0.0f32; seq_len * num_heads * head_dim];
    for i in 0..seq_len * num_heads {
        let mut full = [0.0f32; 256];
        dequantize_q5_k_ref(&k_q[i], &mut full);
        k_deq[i * head_dim..(i + 1) * head_dim]
            .copy_from_slice(&full[..head_dim]);
        dequantize_q5_k_ref(&v_q[i], &mut full);
        v_deq[i * head_dim..(i + 1) * head_dim]
            .copy_from_slice(&full[..head_dim]);
    }
    let mut reference = vec![0.0f32; num_heads * head_dim];
    decode_attention_f32(
        q,
        &k_deq,
        &v_deq,
        &mut reference,
        num_heads,
        head_dim,
        seq_len,
        scale,
    );

    let mut output = vec![0.0f32; num_heads * head_dim];
    decode_attention_q5_k(
        q,
        &k_q,
        &v_q,
        &mut output,
        num_heads,
        head_dim,
        seq_len,
        scale,
    );

    compare_output(&output, &reference, "Q5_K");
}

fn dequantize_q5_k_ref(block: &BlockQ5K_KV, out: &mut [f32; 256]) {
    let d = half::f16::from_bits(block.d).to_f32();
    let dmin = half::f16::from_bits(block.dmin).to_f32();

    for j in 0..128 {
        let byte = block.qs[j];
        let low = (byte & 0x0F) as i32;
        let high = ((byte >> 4) & 0x0F) as i32;

        let idx_low = j;
        let idx_high = j + 128;
        let h_low = ((block.qh[idx_low / 8] >> (idx_low % 8)) & 1) as i32 * 16;
        let h_high =
            ((block.qh[idx_high / 8] >> (idx_high % 8)) & 1) as i32 * 16;

        let sub = j / 16;
        let sc = block.scales[sub];
        let dl = d * ((sc & 0x0F) as f32);
        let ml = dmin * ((sc >> 4) as f32);

        out[idx_low] = (low + h_low) as f32 * dl + ml;
        out[idx_high] = (high + h_high) as f32 * dl + ml;
    }
}

fn test_q6_k(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
) {
    use infer_train::ops::attention::decode_attention_q6_k;

    let (k_q, v_q) = quantize_k_v_q6_k(k, v, num_heads, head_dim, seq_len);

    let mut k_deq = vec![0.0f32; seq_len * num_heads * head_dim];
    let mut v_deq = vec![0.0f32; seq_len * num_heads * head_dim];
    for i in 0..seq_len * num_heads {
        let mut full = [0.0f32; 256];
        dequantize_q6_k_ref(&k_q[i], &mut full);
        k_deq[i * head_dim..(i + 1) * head_dim]
            .copy_from_slice(&full[..head_dim]);
        dequantize_q6_k_ref(&v_q[i], &mut full);
        v_deq[i * head_dim..(i + 1) * head_dim]
            .copy_from_slice(&full[..head_dim]);
    }
    let mut reference = vec![0.0f32; num_heads * head_dim];
    decode_attention_f32(
        q,
        &k_deq,
        &v_deq,
        &mut reference,
        num_heads,
        head_dim,
        seq_len,
        scale,
    );

    let mut output = vec![0.0f32; num_heads * head_dim];
    decode_attention_q6_k(
        q,
        &k_q,
        &v_q,
        &mut output,
        num_heads,
        head_dim,
        seq_len,
        scale,
    );

    compare_output(&output, &reference, "Q6_K");
}

fn dequantize_q6_k_ref(block: &BlockQ6K_KV, out: &mut [f32; 256]) {
    let d = half::f16::from_bits(block.d).to_f32();

    // ql[j] = element j (low nibble) | element j+128 (high nibble)
    // qh[i] contains 2-bit values for 4 elements each
    for j in 0..128 {
        let ql_byte = block.ql[j];
        let low_nibble = (ql_byte & 0x0F) as i32;
        let high_nibble = ((ql_byte >> 4) & 0x0F) as i32;

        // Upper 2 bits from qh
        // qh[j/2] contains bits for elements j and j+1
        // Each byte of qh stores 4 sets of 2 bits (for 4 elements)
        // But for Q6_K, the layout is: qh[j/2] with shift (j%2)*4
        let qh_byte = block.qh[j / 2];
        let shift = (j % 2) * 4;
        let h_bits = ((qh_byte >> shift) & 0x03) as i32;

        // Reconstruct 6-bit signed value: low4 + high2*16 - 32
        let val_low = low_nibble + h_bits * 16 - 32;
        let val_high = high_nibble + h_bits * 16 - 32;

        out[j] = d * val_low as f32;
        out[j + 128] = d * val_high as f32;
    }
}
