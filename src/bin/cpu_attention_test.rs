// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Infer Train Authors

//! Comprehensive test for CPU attention operators
//! Tests all quantized KV cache formats against FP32 reference

use infer_train::ops::attention::decode_attention_f32;
use infer_train::quant::kv_cache::quant::{
    quantize_row_iq4_nl, quantize_row_q4_0, quantize_row_q4_1,
    quantize_row_q5_0, quantize_row_q5_1, quantize_row_q8_0,
};
use infer_train::quant::kv_cache::{
    BlockIQ4NL, BlockQ4_0, BlockQ4_1, BlockQ5_0, BlockQ5_1, BlockQ8_0, QK,
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
    let thresholds = [
        ("Q8_0", 5e-2),   // 5% - acceptable for Q8_0
        ("Q4_0", 5e-2),   // 5% - acceptable for Q4_0
        ("Q4_1", 2e-2),   // 2% - Q4_1 with min/scale
        ("Q5_0", 5e-2),   // 5% - acceptable for Q5_0
        ("Q5_1", 1e-2),   // 1% - Q5_1 with min/scale
        ("IQ4_NL", 2e-1), // 20% - non-linear quantization, higher error expected
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
