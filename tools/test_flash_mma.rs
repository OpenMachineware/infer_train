// Test Flash Attention MMA kernel correctness
// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use infer_train::quant::vec_dot::gpu::metal::MetalContext;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
fn main() {
    let ctx = MetalContext::new().expect("Failed to create MetalContext");
    let device = ctx.device().clone();

    // Test parameters
    let n_heads = 16;
    let n_kv_heads = 4;
    let head_dim = 128;
    let seq_len = 10;
    let kv_stride = 256;

    println!("=== Flash Attention MMA Correctness Test ===");
    println!(
        "n_heads={}, n_kv_heads={}, head_dim={}, seq_len={}",
        n_heads, n_kv_heads, head_dim, seq_len
    );

    // Create test data
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Q: [n_heads, head_dim]
    let q_data: Vec<half::f16> = (0..n_heads * head_dim)
        .map(|i| half::f16::from_f32((i as f32 * 0.01).sin()))
        .collect();

    // K: [n_kv_heads, kv_stride, head_dim]
    let k_data: Vec<half::f16> = (0..n_kv_heads * kv_stride * head_dim)
        .map(|i| half::f16::from_f32((i as f32 * 0.02).cos()))
        .collect();

    // Debug: print first few Q values
    println!("\nQ[0] first 8 elements (as half):");
    for i in 0..8 {
        print!("{:.6} ", q_data[i].to_f32());
    }
    println!();

    // V: [n_kv_heads, kv_stride, head_dim]
    let v_data: Vec<half::f16> = (0..n_kv_heads * kv_stride * head_dim)
        .map(|i| {
            half::f16::from_f32((i as f32 * 0.03).tan().max(-1.0).min(1.0))
        })
        .collect();

    // Create buffers
    let q_buffer = device.new_buffer_with_data(
        q_data.as_ptr() as *const std::ffi::c_void,
        (n_heads * head_dim * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let k_buffer = device.new_buffer_with_data(
        k_data.as_ptr() as *const std::ffi::c_void,
        (n_kv_heads * kv_stride * head_dim * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let v_buffer = device.new_buffer_with_data(
        v_data.as_ptr() as *const std::ffi::c_void,
        (n_kv_heads * kv_stride * head_dim * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let output_buffer = device.new_buffer(
        (n_heads * head_dim * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Run MMA kernel
    let queue = ctx.queue();
    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    // Call MMA kernel
    ctx.flash_attn_mma_encode(
        encoder,
        n_heads,
        n_kv_heads,
        seq_len,
        kv_stride,
        &q_buffer,
        &k_buffer,
        &v_buffer,
        &output_buffer,
    );

    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Reference: compute expected output on CPU
    let mut expected = vec![0.0f32; n_heads * head_dim];

    for h in 0..n_heads {
        let kv_head = h / (n_heads / n_kv_heads);

        // Compute attention scores
        let mut scores: Vec<f32> = Vec::with_capacity(seq_len);
        for pos in 0..seq_len {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                let q_val = q_data[h * head_dim + d].to_f32();
                let k_val = k_data
                    [kv_head * kv_stride * head_dim + pos * head_dim + d]
                    .to_f32();
                dot += q_val * k_val;
            }
            scores.push(dot * scale);
        }

        // Softmax
        let max_score =
            scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f32> =
            scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum_exp: f32 = exp_scores.iter().sum();
        let weights: Vec<f32> =
            exp_scores.iter().map(|&e| e / sum_exp).collect();

        // Weighted sum of V
        for d in 0..head_dim {
            let mut out = 0.0f32;
            for pos in 0..seq_len {
                let v_val = v_data
                    [kv_head * kv_stride * head_dim + pos * head_dim + d]
                    .to_f32();
                out += weights[pos] * v_val;
            }
            expected[h * head_dim + d] = out;
        }
    }

    println!("\nExpected output (first head, first 8 elements):");
    for i in 0..8 {
        print!("{:.6} ", expected[i]);
    }
    println!();

    // Read GPU output
    let output_ptr = output_buffer.contents() as *const f32;
    let gpu_output: Vec<f32> = (0..n_heads * head_dim)
        .map(|i| unsafe { *output_ptr.add(i) })
        .collect();

    println!("\nGPU output (first head, first 8 elements):");
    for i in 0..8 {
        print!("{:.6} ", gpu_output[i]);
    }
    println!();

    // Compare
    let mut max_rel_err = 0.0f32;
    let mut max_abs_err = 0.0f32;
    for i in 0..n_heads * head_dim {
        let abs_err = (gpu_output[i] - expected[i]).abs();
        let rel_err = if expected[i].abs() > 1e-6 {
            abs_err / expected[i].abs()
        } else {
            abs_err
        };
        max_abs_err = max_abs_err.max(abs_err);
        max_rel_err = max_rel_err.max(rel_err);
    }

    println!("\nMax absolute error: {:.6}", max_abs_err);
    println!("Max relative error: {:.6}%", max_rel_err * 100.0);

    if max_rel_err < 0.001 {
        println!("\n✅ PASSED: Relative error < 0.1%");
    } else {
        println!("\n❌ FAILED: Relative error >= 0.1%");
    }
}

#[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
fn main() {
    println!("This test requires Metal support (macOS with metal feature)");
}
