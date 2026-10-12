// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Quick correctness test for attention
#[cfg(target_arch = "aarch64")]
use infer_train::ops::attention::decode_attention_f16_neon;
use infer_train::ops::attention::decode_attention_f32;

fn main() {
    let num_heads = 2;
    let head_dim = 8;
    let seq_len = 4;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Simple test data
    let q: Vec<f32> = vec![1.0; num_heads * head_dim];
    let k: Vec<f32> = vec![0.5; seq_len * num_heads * head_dim];
    let v: Vec<f32> = vec![0.3; seq_len * num_heads * head_dim];

    let mut output_f32 = vec![0.0f32; num_heads * head_dim];
    decode_attention_f32(
        &q,
        &k,
        &v,
        &mut output_f32,
        num_heads,
        head_dim,
        seq_len,
        scale,
    );

    println!("FP32 output: {:?}", &output_f32[..4]);

    // Expected: with softmax, output should be a weighted average of V values
    // With Q=1.0, K=0.5, scores are all equal, so softmax gives uniform distribution
    // V=0.3 for all, so output should be close to 0.3 for each element

    #[cfg(target_arch = "aarch64")]
    {
        // Test FP16
        let q_f16: Vec<u16> =
            q.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect();
        let k_f16: Vec<u16> =
            k.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect();
        let v_f16: Vec<u16> =
            v.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect();

        let mut output_f16 = vec![0.0f32; num_heads * head_dim];
        unsafe {
            decode_attention_f16_neon(
                &q_f16,
                &k_f16,
                &v_f16,
                &mut output_f16,
                num_heads,
                head_dim,
                seq_len,
                scale,
            );
        }

        println!("FP16 output: {:?}", &output_f16[..4]);

        // Check difference
        let max_diff: f32 = output_f32
            .iter()
            .zip(output_f16.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        println!("Max difference FP32 vs FP16: {}", max_diff);

        if max_diff < 0.01 {
            println!("✅ FP16 attention matches FP32 within tolerance");
        } else {
            println!("❌ FP16 attention differs from FP32 significantly");
        }
    }
}
