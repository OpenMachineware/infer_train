// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Strict correctness test for all K-quant GEMV kernels
//! Compare GPU vs CPU element-by-element at larger scale

use bytemuck::Zeroable;
use half::f16;
use infer_train::inference::dequant::{
    dequant_q2_k, dequant_q3_k, dequant_q4_k, dequant_q5_k, dequant_q6_k,
};
use infer_train::quant::types::{
    BlockQ2K, BlockQ3K, BlockQ4K, BlockQ5K, BlockQ6K, QK_K,
};
use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use metal::{MTLResourceOptions, NSRange};

#[test]
fn test_q4k_gemv_large_scale() {
    test_q4k_gemv(256, 4096); // Realistic size: 256 rows, 4096 cols
}

#[test]
fn test_q2k_gemv_large_scale() {
    test_q2k_gemv(256, 4096);
}

#[test]
fn test_q3k_gemv_large_scale() {
    test_q3k_gemv(256, 4096);
}

#[test]
fn test_q5k_gemv_large_scale() {
    test_q5k_gemv(256, 4096);
}

#[test]
fn test_q6k_gemv_large_scale() {
    test_q6k_gemv(256, 4096);
}

fn test_q4k_gemv(m: usize, k: usize) {
    let ctx = MetalContext::new().expect("Failed to create Metal context");
    let n_blocks = k / QK_K;
    let mut weights = vec![BlockQ4K::zeroed(); m * n_blocks];

    // Fill with deterministic pattern
    for (row_idx, row_blocks) in weights.chunks_mut(n_blocks).enumerate() {
        for (block_idx, block) in row_blocks.iter_mut().enumerate() {
            let idx = row_idx * n_blocks + block_idx;
            block.d =
                f16::to_bits(f16::from_f32(0.5 + 0.001 * (idx % 100) as f32));
            block.dmin = f16::to_bits(f16::from_f32(0.1 * ((idx % 10) as f32)));
            for j in 0..12 {
                block.scales[j] = ((idx * 7 + j * 11 + 1) % 64) as u8;
            }
            for j in 0..128 {
                block.qs[j] = ((idx * 13 + j * 17 + 8) % 256) as u8;
            }
        }
    }

    let input: Vec<f32> =
        (0..k).map(|i| 0.1 * ((i % 10) as f32 + 1.0)).collect();

    // Print first block info for debugging
    if m == 1 && n_blocks == 2 {
        println!("=== Block 0 ===");
        println!("d = {:.4}", f16::from_bits(weights[0].d).to_f32());
        println!("scales[0..4] = {:?}", &weights[0].scales[..4]);
        println!("qs[0..8] = {:?}", &weights[0].qs[..8]);

        println!("\n=== Block 1 ===");
        println!("d = {:.4}", f16::from_bits(weights[1].d).to_f32());
        println!("scales[0..4] = {:?}", &weights[1].scales[..4]);
        println!("qs[0..8] = {:?}", &weights[1].qs[..8]);

        // Dequant each block and compute partial sums
        let mut block_sums_cpu = [0.0f32; 2];
        for block_idx in 0..2 {
            let block = &weights[block_idx];
            let block_bytes = unsafe {
                std::slice::from_raw_parts(
                    block as *const BlockQ4K as *const u8,
                    std::mem::size_of::<BlockQ4K>(),
                )
            };
            let mut dequant_values = [0.0f32; QK_K];
            dequant_q4_k(block_bytes, &mut dequant_values, QK_K);

            let sum: f32 = (0..QK_K)
                .map(|i| dequant_values[i] * input[block_idx * QK_K + i])
                .sum();
            block_sums_cpu[block_idx] = sum;
            println!("\nCPU block {} sum = {:.6}", block_idx, sum);
        }
        println!("CPU total = {:.6}", block_sums_cpu[0] + block_sums_cpu[1]);
    }

    let weights_buffer = ctx.device().new_buffer(
        (weights.len() * std::mem::size_of::<BlockQ4K>()) as u64,
        MTLResourceOptions::StorageModeManaged,
    );
    unsafe {
        std::ptr::copy_nonoverlapping(
            weights.as_ptr() as *const u8,
            weights_buffer.contents() as *mut u8,
            weights.len() * std::mem::size_of::<BlockQ4K>(),
        );
    }
    weights_buffer.did_modify_range(NSRange {
        location: 0,
        length: weights_buffer.length(),
    });

    let input_buffer = ctx.device().new_buffer(
        (input.len() * 4) as u64,
        MTLResourceOptions::StorageModeManaged,
    );
    unsafe {
        std::ptr::copy_nonoverlapping(
            input.as_ptr() as *const u8,
            input_buffer.contents() as *mut u8,
            input.len() * 4,
        );
    }
    input_buffer.did_modify_range(NSRange {
        location: 0,
        length: input_buffer.length(),
    });

    let output_buffer = ctx
        .mv_q4_k_f32_gpu_weights(m, k, &weights_buffer, &input_buffer)
        .expect("Q4_K MV failed");

    let output_ptr = output_buffer.contents() as *const f32;
    let gpu_output = unsafe { std::slice::from_raw_parts(output_ptr, m) };

    // CPU reference
    let mut cpu_output = vec![0.0f32; m];
    for row in 0..m {
        let mut sum = 0.0f32;
        for block_idx in 0..n_blocks {
            let block = &weights[row * n_blocks + block_idx];
            let block_bytes = unsafe {
                std::slice::from_raw_parts(
                    block as *const BlockQ4K as *const u8,
                    std::mem::size_of::<BlockQ4K>(),
                )
            };
            let mut dequant_values = [0.0f32; QK_K];
            dequant_q4_k(block_bytes, &mut dequant_values, QK_K);
            for i in 0..QK_K {
                sum += dequant_values[i] * input[block_idx * QK_K + i];
            }
        }
        cpu_output[row] = sum;
    }

    if m == 1 && n_blocks == 2 {
        println!("GPU total = {:.10}", gpu_output[0]);
        println!("CPU total = {:.10}", cpu_output[0]);
        println!("Diff = {:.10}", (gpu_output[0] - cpu_output[0]).abs());
    }

    // Check all rows
    let mut max_diff = 0.0f32;
    let mut mismatch_count = 0;
    for row in 0..m {
        let diff = (gpu_output[row] - cpu_output[row]).abs();
        // Use relative error for large values
        let rel_err = if cpu_output[row].abs() > 1.0 {
            diff / cpu_output[row].abs()
        } else {
            diff
        };
        // Acceptable error: 0.1 absolute or 0.1% relative
        if diff > 0.1 && rel_err > 0.001 {
            mismatch_count += 1;
        }
        if diff > max_diff {
            max_diff = diff;
        }
    }

    // Debug output for single row cases
    if m == 1 && n_blocks <= 16 {
        println!(
            "GPU = {:.10}, CPU = {:.10}, Diff = {:.10}",
            gpu_output[0], cpu_output[0], max_diff
        );
    }

    println!(
        "Q4_K: {} rows, max_diff={:.6}, mismatches={}/{}",
        m, max_diff, mismatch_count, m
    );
    assert!(mismatch_count == 0, "Q4_K GPU kernel mismatch");
}

fn test_q2k_gemv(m: usize, k: usize) {
    let ctx = MetalContext::new().expect("Failed to create Metal context");
    let n_blocks = k / QK_K;
    let mut weights = vec![BlockQ2K::zeroed(); m * n_blocks];

    for (row_idx, row_blocks) in weights.chunks_mut(n_blocks).enumerate() {
        for (block_idx, block) in row_blocks.iter_mut().enumerate() {
            let idx = row_idx * n_blocks + block_idx;
            block.d =
                f16::to_bits(f16::from_f32(0.5 + 0.001 * (idx % 100) as f32));
            block.dmin = f16::to_bits(f16::from_f32(0.1 * ((idx % 10) as f32)));
            for j in 0..16 {
                block.scales[j] = ((idx * 7 + j * 11 + 1) % 256) as u8;
            }
            for j in 0..64 {
                block.qs[j] = ((idx * 13 + j * 17 + 8) % 256) as u8;
            }
        }
    }

    let input: Vec<f32> =
        (0..k).map(|i| 0.1 * ((i % 10) as f32 + 1.0)).collect();

    let weights_buffer = ctx.device().new_buffer(
        (weights.len() * std::mem::size_of::<BlockQ2K>()) as u64,
        MTLResourceOptions::StorageModeManaged,
    );
    unsafe {
        std::ptr::copy_nonoverlapping(
            weights.as_ptr() as *const u8,
            weights_buffer.contents() as *mut u8,
            weights.len() * std::mem::size_of::<BlockQ2K>(),
        );
    }
    weights_buffer.did_modify_range(NSRange {
        location: 0,
        length: weights_buffer.length(),
    });

    let input_buffer = ctx.device().new_buffer(
        (input.len() * 4) as u64,
        MTLResourceOptions::StorageModeManaged,
    );
    unsafe {
        std::ptr::copy_nonoverlapping(
            input.as_ptr() as *const u8,
            input_buffer.contents() as *mut u8,
            input.len() * 4,
        );
    }
    input_buffer.did_modify_range(NSRange {
        location: 0,
        length: input_buffer.length(),
    });

    let output_buffer = ctx
        .mv_q2_k_f32_gpu_weights(m, k, &weights_buffer, &input_buffer)
        .expect("Q2_K MV failed");

    let output_ptr = output_buffer.contents() as *const f32;
    let gpu_output = unsafe { std::slice::from_raw_parts(output_ptr, m) };

    let mut cpu_output = vec![0.0f32; m];
    for row in 0..m {
        let mut sum = 0.0f32;
        for block_idx in 0..n_blocks {
            let block = &weights[row * n_blocks + block_idx];
            let block_bytes = unsafe {
                std::slice::from_raw_parts(
                    block as *const BlockQ2K as *const u8,
                    std::mem::size_of::<BlockQ2K>(),
                )
            };
            let mut dequant_values = [0.0f32; QK_K];
            dequant_q2_k(block_bytes, &mut dequant_values, QK_K);
            for i in 0..QK_K {
                sum += dequant_values[i] * input[block_idx * QK_K + i];
            }
        }
        cpu_output[row] = sum;
    }

    let mut max_diff = 0.0f32;
    let mut mismatch_count = 0;
    for row in 0..m {
        let diff = (gpu_output[row] - cpu_output[row]).abs();
        if diff >= 1e-3 {
            mismatch_count += 1;
        }
        if diff > max_diff {
            max_diff = diff;
        }
    }

    println!(
        "Q2_K: {} rows, max_diff={:.6}, mismatches={}/{}",
        m, max_diff, mismatch_count, m
    );
    assert!(mismatch_count == 0, "Q2_K GPU kernel mismatch");
}

fn test_q3k_gemv(m: usize, k: usize) {
    let ctx = MetalContext::new().expect("Failed to create Metal context");
    let n_blocks = k / QK_K;
    let mut weights = vec![BlockQ3K::zeroed(); m * n_blocks];

    for (row_idx, row_blocks) in weights.chunks_mut(n_blocks).enumerate() {
        for (block_idx, block) in row_blocks.iter_mut().enumerate() {
            let idx = row_idx * n_blocks + block_idx;
            block.d =
                f16::to_bits(f16::from_f32(0.5 + 0.001 * (idx % 100) as f32));
            for j in 0..12 {
                block.scales[j] = ((idx * 7 + j * 11 + 1) % 256) as u8;
            }
            for j in 0..32 {
                block.hmask[j] = ((idx * 13 + j * 17) % 256) as u8;
            }
            for j in 0..64 {
                block.qs[j] = ((idx * 13 + j * 17 + 8) % 256) as u8;
            }
        }
    }

    let input: Vec<f32> =
        (0..k).map(|i| 0.1 * ((i % 10) as f32 + 1.0)).collect();

    let weights_buffer = ctx.device().new_buffer(
        (weights.len() * std::mem::size_of::<BlockQ3K>()) as u64,
        MTLResourceOptions::StorageModeManaged,
    );
    unsafe {
        std::ptr::copy_nonoverlapping(
            weights.as_ptr() as *const u8,
            weights_buffer.contents() as *mut u8,
            weights.len() * std::mem::size_of::<BlockQ3K>(),
        );
    }
    weights_buffer.did_modify_range(NSRange {
        location: 0,
        length: weights_buffer.length(),
    });

    let input_buffer = ctx.device().new_buffer(
        (input.len() * 4) as u64,
        MTLResourceOptions::StorageModeManaged,
    );
    unsafe {
        std::ptr::copy_nonoverlapping(
            input.as_ptr() as *const u8,
            input_buffer.contents() as *mut u8,
            input.len() * 4,
        );
    }
    input_buffer.did_modify_range(NSRange {
        location: 0,
        length: input_buffer.length(),
    });

    let output_buffer = ctx
        .mv_q3_k_f32_gpu_weights(m, k, &weights_buffer, &input_buffer)
        .expect("Q3_K MV failed");

    let output_ptr = output_buffer.contents() as *const f32;
    let gpu_output = unsafe { std::slice::from_raw_parts(output_ptr, m) };

    let mut cpu_output = vec![0.0f32; m];
    for row in 0..m {
        let mut sum = 0.0f32;
        for block_idx in 0..n_blocks {
            let block = &weights[row * n_blocks + block_idx];
            let block_bytes = unsafe {
                std::slice::from_raw_parts(
                    block as *const BlockQ3K as *const u8,
                    std::mem::size_of::<BlockQ3K>(),
                )
            };
            let mut dequant_values = [0.0f32; QK_K];
            dequant_q3_k(block_bytes, &mut dequant_values, QK_K);
            for i in 0..QK_K {
                sum += dequant_values[i] * input[block_idx * QK_K + i];
            }
        }
        cpu_output[row] = sum;
    }

    let mut max_diff = 0.0f32;
    let mut mismatch_count = 0;
    for row in 0..m {
        let diff = (gpu_output[row] - cpu_output[row]).abs();
        if diff >= 1e-3 {
            mismatch_count += 1;
        }
        if diff > max_diff {
            max_diff = diff;
        }
    }

    println!(
        "Q3_K: {} rows, max_diff={:.6}, mismatches={}/{}",
        m, max_diff, mismatch_count, m
    );
    assert!(mismatch_count == 0, "Q3_K GPU kernel mismatch");
}

fn test_q5k_gemv(m: usize, k: usize) {
    let ctx = MetalContext::new().expect("Failed to create Metal context");
    let n_blocks = k / QK_K;
    let mut weights = vec![BlockQ5K::zeroed(); m * n_blocks];

    for (row_idx, row_blocks) in weights.chunks_mut(n_blocks).enumerate() {
        for (block_idx, block) in row_blocks.iter_mut().enumerate() {
            let idx = row_idx * n_blocks + block_idx;
            block.d =
                f16::to_bits(f16::from_f32(0.5 + 0.001 * (idx % 100) as f32));
            block.dmin = f16::to_bits(f16::from_f32(0.1 * ((idx % 10) as f32)));
            for j in 0..12 {
                block.scales[j] = ((idx * 7 + j * 11 + 1) % 64) as u8;
            }
            for j in 0..32 {
                block.qh[j] = ((idx * 13 + j * 17) % 256) as u8;
            }
            for j in 0..128 {
                block.qs[j] = ((idx * 13 + j * 17 + 8) % 256) as u8;
            }
        }
    }

    let input: Vec<f32> =
        (0..k).map(|i| 0.1 * ((i % 10) as f32 + 1.0)).collect();

    let weights_buffer = ctx.device().new_buffer(
        (weights.len() * std::mem::size_of::<BlockQ5K>()) as u64,
        MTLResourceOptions::StorageModeManaged,
    );
    unsafe {
        std::ptr::copy_nonoverlapping(
            weights.as_ptr() as *const u8,
            weights_buffer.contents() as *mut u8,
            weights.len() * std::mem::size_of::<BlockQ5K>(),
        );
    }
    weights_buffer.did_modify_range(NSRange {
        location: 0,
        length: weights_buffer.length(),
    });

    let input_buffer = ctx.device().new_buffer(
        (input.len() * 4) as u64,
        MTLResourceOptions::StorageModeManaged,
    );
    unsafe {
        std::ptr::copy_nonoverlapping(
            input.as_ptr() as *const u8,
            input_buffer.contents() as *mut u8,
            input.len() * 4,
        );
    }
    input_buffer.did_modify_range(NSRange {
        location: 0,
        length: input_buffer.length(),
    });

    let output_buffer = ctx
        .mv_q5_k_f32_gpu_weights(m, k, &weights_buffer, &input_buffer)
        .expect("Q5_K MV failed");

    let output_ptr = output_buffer.contents() as *const f32;
    let gpu_output = unsafe { std::slice::from_raw_parts(output_ptr, m) };

    let mut cpu_output = vec![0.0f32; m];
    for row in 0..m {
        let mut sum = 0.0f32;
        for block_idx in 0..n_blocks {
            let block = &weights[row * n_blocks + block_idx];
            let block_bytes = unsafe {
                std::slice::from_raw_parts(
                    block as *const BlockQ5K as *const u8,
                    std::mem::size_of::<BlockQ5K>(),
                )
            };
            let mut dequant_values = [0.0f32; QK_K];
            dequant_q5_k(block_bytes, &mut dequant_values, QK_K);
            for i in 0..QK_K {
                sum += dequant_values[i] * input[block_idx * QK_K + i];
            }
        }
        cpu_output[row] = sum;
    }

    let mut max_diff = 0.0f32;
    let mut mismatch_count = 0;
    for row in 0..m {
        let diff = (gpu_output[row] - cpu_output[row]).abs();
        // Use relative error for large values
        let rel_err = if cpu_output[row].abs() > 1.0 {
            diff / cpu_output[row].abs()
        } else {
            diff
        };
        // Acceptable error: 0.1 absolute or 0.1% relative
        if diff > 0.1 && rel_err > 0.001 {
            mismatch_count += 1;
        }
        if diff > max_diff {
            max_diff = diff;
        }
    }

    // Debug output for small tests
    if m == 1 && n_blocks <= 2 {
        println!(
            "Q5_K: GPU = {:.10}, CPU = {:.10}, Diff = {:.10}",
            gpu_output[0], cpu_output[0], max_diff
        );
    }

    println!(
        "Q5_K: {} rows, max_diff={:.6}, mismatches={}/{}",
        m, max_diff, mismatch_count, m
    );
    assert!(mismatch_count == 0, "Q5_K GPU kernel mismatch");
}

fn test_q6k_gemv(m: usize, k: usize) {
    let ctx = MetalContext::new().expect("Failed to create Metal context");
    let n_blocks = k / QK_K;
    let mut weights = vec![BlockQ6K::zeroed(); m * n_blocks];

    for (row_idx, row_blocks) in weights.chunks_mut(n_blocks).enumerate() {
        for (block_idx, block) in row_blocks.iter_mut().enumerate() {
            let idx = row_idx * n_blocks + block_idx;
            block.d =
                f16::to_bits(f16::from_f32(0.5 + 0.001 * (idx % 100) as f32));
            for j in 0..16 {
                block.scales[j] = ((idx * 7 + j * 11 + 1) % 64) as i8;
            }
            for j in 0..128 {
                block.ql[j] = ((idx * 13 + j * 17 + 8) % 256) as u8;
            }
            for j in 0..64 {
                block.qh[j] = ((idx * 13 + j * 17) % 256) as u8;
            }
        }
    }

    let input: Vec<f32> =
        (0..k).map(|i| 0.1 * ((i % 10) as f32 + 1.0)).collect();

    let weights_buffer = ctx.device().new_buffer(
        (weights.len() * std::mem::size_of::<BlockQ6K>()) as u64,
        MTLResourceOptions::StorageModeManaged,
    );
    unsafe {
        std::ptr::copy_nonoverlapping(
            weights.as_ptr() as *const u8,
            weights_buffer.contents() as *mut u8,
            weights.len() * std::mem::size_of::<BlockQ6K>(),
        );
    }
    weights_buffer.did_modify_range(NSRange {
        location: 0,
        length: weights_buffer.length(),
    });

    let input_buffer = ctx.device().new_buffer(
        (input.len() * 4) as u64,
        MTLResourceOptions::StorageModeManaged,
    );
    unsafe {
        std::ptr::copy_nonoverlapping(
            input.as_ptr() as *const u8,
            input_buffer.contents() as *mut u8,
            input.len() * 4,
        );
    }
    input_buffer.did_modify_range(NSRange {
        location: 0,
        length: input_buffer.length(),
    });

    let output_buffer = ctx
        .mv_q6_k_f32_gpu_weights(m, k, &weights_buffer, &input_buffer)
        .expect("Q6_K MV failed");

    let output_ptr = output_buffer.contents() as *const f32;
    let gpu_output = unsafe { std::slice::from_raw_parts(output_ptr, m) };

    let mut cpu_output = vec![0.0f32; m];
    for row in 0..m {
        let mut sum = 0.0f32;
        for block_idx in 0..n_blocks {
            let block = &weights[row * n_blocks + block_idx];
            let block_bytes = unsafe {
                std::slice::from_raw_parts(
                    block as *const BlockQ6K as *const u8,
                    std::mem::size_of::<BlockQ6K>(),
                )
            };
            let mut dequant_values = [0.0f32; QK_K];
            dequant_q6_k(block_bytes, &mut dequant_values, QK_K);
            for i in 0..QK_K {
                sum += dequant_values[i] * input[block_idx * QK_K + i];
            }
        }
        cpu_output[row] = sum;
    }

    let mut max_diff = 0.0f32;
    let mut mismatch_count = 0;
    for row in 0..m {
        let diff = (gpu_output[row] - cpu_output[row]).abs();
        // Use relative error for large values
        let rel_err = if cpu_output[row].abs() > 1.0 {
            diff / cpu_output[row].abs()
        } else {
            diff
        };
        // Acceptable error: 0.1 absolute or 0.1% relative
        if diff > 0.1 && rel_err > 0.001 {
            mismatch_count += 1;
        }
        if diff > max_diff {
            max_diff = diff;
        }
    }

    // Debug output for small tests
    if m == 1 && n_blocks == 1 {
        println!(
            "Q6_K: GPU = {:.10}, CPU = {:.10}, Diff = {:.10}",
            gpu_output[0], cpu_output[0], max_diff
        );
    }

    println!(
        "Q6_K: {} rows, max_diff={:.6}, mismatches={}/{}",
        m, max_diff, mismatch_count, m
    );
    assert!(mismatch_count == 0, "Q6_K GPU kernel mismatch");
}
