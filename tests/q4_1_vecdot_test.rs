// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Verify Q4_1 quantization round-trip
use infer_train::quant::kv_cache::quant::{
    quantize_row_q4_1, quantize_row_q8_1,
};
use infer_train::quant::kv_cache::{
    BlockQ4_1 as BlockQ4_1KV, BlockQ8_1 as BlockQ8_1KV, QK,
};
use infer_train::quant::types::{BlockQ4_1, BlockQ8_1};
use infer_train::quant::vec_dot::arm::q_0_1::vec_dot_q4_1_q8_1_neon;
use rand::Rng;

fn dequantize_q4_1(block: &BlockQ4_1, n: usize) -> Vec<f32> {
    let scale = half::f16::from_bits(block.d).to_f32();
    let min = half::f16::from_bits(block.m).to_f32();
    let mut result = vec![0.0f32; n];

    for j in 0..(n / 2) {
        let packed = block.qs[j];
        let lo = (packed & 0x0F) as f32;
        let hi = ((packed >> 4) & 0x0F) as f32;
        result[j] = scale * lo + min;
        result[j + n / 2] = scale * hi + min;
    }
    result
}

fn dequantize_q8_1(block: &BlockQ8_1, n: usize) -> Vec<f32> {
    let scale = half::f16::from_bits(block.d).to_f32();
    let mut result = vec![0.0f32; n];

    for i in 0..n {
        result[i] = (block.qs[i] as f32) * scale;
    }
    result
}

fn main() {
    println!("=== Q4_1 Round-Trip Test ===\n");

    let n = QK;
    let mut rng = rand::thread_rng();

    // Test 1: Simple values
    let k_vals: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
    let q_vals: Vec<f32> = (0..n).map(|i| (i as f32) * 0.05).collect();

    test_q4_1(&k_vals, &q_vals, "Simple ascending values");

    // Test 2: Random values
    let k_vals: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let q_vals: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();

    test_q4_1(&k_vals, &q_vals, "Random values [-1, 1]");

    // Test 3: Larger values
    let k_vals: Vec<f32> = (0..n).map(|_| rng.gen_range(-10.0..10.0)).collect();
    let q_vals: Vec<f32> = (0..n).map(|_| rng.gen_range(-10.0..10.0)).collect();

    test_q4_1(&k_vals, &q_vals, "Random values [-10, 10]");

    // Test 4: All positive values
    let k_vals: Vec<f32> = (0..n).map(|_| rng.gen_range(0.0..1.0)).collect();
    let q_vals: Vec<f32> = (0..n).map(|_| rng.gen_range(0.0..1.0)).collect();

    test_q4_1(&k_vals, &q_vals, "Positive values [0, 1]");
}

fn test_q4_1(k_vals: &[f32], q_vals: &[f32], name: &str) {
    let n = k_vals.len();

    // Quantize
    let mut k_q4_1_kv = vec![BlockQ4_1KV { d: 0, m: 0, qs: [0u8; QK / 2] }; 1];
    let mut q_q8_1_kv = vec![BlockQ8_1KV { d: 0, s: 0, qs: [0i8; QK] }; 1];
    quantize_row_q4_1(k_vals, &mut k_q4_1_kv);
    quantize_row_q8_1(q_vals, &mut q_q8_1_kv);

    // Transmute to types
    let k_q4_1: &[BlockQ4_1] = unsafe {
        std::slice::from_raw_parts(k_q4_1_kv.as_ptr() as *const BlockQ4_1, 1)
    };
    let q_q8_1: &[BlockQ8_1] = unsafe {
        std::slice::from_raw_parts(q_q8_1_kv.as_ptr() as *const BlockQ8_1, 1)
    };

    // Dequantize
    let k_deq = dequantize_q4_1(&k_q4_1[0], n);
    let q_deq = dequantize_q8_1(&q_q8_1[0], n);

    // Compute reference dot product
    let ref_dot: f32 =
        k_vals.iter().zip(q_vals.iter()).map(|(k, q)| k * q).sum();

    // Compute dequantized dot product
    let deq_dot: f32 = k_deq.iter().zip(q_deq.iter()).map(|(k, q)| k * q).sum();

    // Compute NEON vec_dot
    let neon_dot = unsafe { vec_dot_q4_1_q8_1_neon(n, k_q4_1, q_q8_1) };

    // Compute error
    let deq_err = (deq_dot - ref_dot).abs() / ref_dot.abs().max(1e-6);
    let neon_err = (neon_dot - ref_dot).abs() / ref_dot.abs().max(1e-6);

    println!("{}", name);
    println!("  Reference:     {:.6}", ref_dot);
    println!(
        "  Dequantized:   {:.6} (error: {:.2}%)",
        deq_dot,
        deq_err * 100.0
    );
    println!(
        "  NEON vec_dot:  {:.6} (error: {:.2}%)",
        neon_dot,
        neon_err * 100.0
    );

    // Check K dequantization error
    let k_deq_err: f32 = k_vals
        .iter()
        .zip(k_deq.iter())
        .map(|(o, d)| (o - d).abs())
        .sum::<f32>()
        / n as f32;
    println!("  K dequant avg error: {:.6}", k_deq_err);

    // Check Q dequantization error
    let q_deq_err: f32 = q_vals
        .iter()
        .zip(q_deq.iter())
        .map(|(o, d)| (o - d).abs())
        .sum::<f32>()
        / n as f32;
    println!("  Q dequant avg error: {:.6}", q_deq_err);
    println!();
}
