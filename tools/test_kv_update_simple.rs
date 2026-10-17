// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Simple test for KV cache update kernel correctness

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
fn main() {
    use infer_train::quant::vec_dot::gpu::metal::MetalContext;
    use metal::*;
    use std::rc::Rc;

    // Test parameters
    let n_kv_heads = 2;
    let max_seq = 2048;
    let head_dim = 128;

    println!("=== KV Cache Update Kernel Test ===");
    println!("n_kv_heads: {}", n_kv_heads);
    println!("max_seq: {}", max_seq);
    println!("head_dim: {}", head_dim);
    println!("Total elements per cache: {}", n_kv_heads * max_seq * head_dim);

    // Create GPU context
    let gpu_ctx = MetalContext::new().expect("Failed to create Metal context");
    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    // Create synthetic K and V data (CPU)
    let cpu_k: Vec<f32> = (0..n_kv_heads * head_dim)
        .map(|i| (i as f32 * 0.01) - 1.0) // Range: -1.0 to ~0.27
        .collect();
    let cpu_v: Vec<f32> = (0..n_kv_heads * head_dim)
        .map(|i| (i as f32 * 0.02) - 2.0) // Range: -2.0 to ~0.54
        .collect();

    println!("\n=== CPU K/V Values ===");
    println!(
        "K: len={}, min={:.4}, max={:.4}",
        cpu_k.len(),
        cpu_k.iter().cloned().fold(f32::INFINITY, f32::min),
        cpu_k.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );
    println!(
        "V: len={}, min={:.4}, max={:.4}",
        cpu_v.len(),
        cpu_v.iter().cloned().fold(f32::INFINITY, f32::min),
        cpu_v.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );

    // Create GPU buffers for K and V
    let k_buffer = Rc::new(device.new_buffer_with_data(
        cpu_k.as_ptr() as *const std::ffi::c_void,
        (cpu_k.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    ));
    let v_buffer = Rc::new(device.new_buffer_with_data(
        cpu_v.as_ptr() as *const std::ffi::c_void,
        (cpu_v.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    ));

    // Create KV cache buffers (F16 format)
    let total_f16_elements = max_seq * n_kv_heads * head_dim;
    let k_cache = Rc::new(device.new_buffer(
        (total_f16_elements * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    ));
    let v_cache = Rc::new(device.new_buffer(
        (total_f16_elements * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    ));

    // Clear KV cache
    unsafe {
        std::ptr::write_bytes(
            k_cache.contents() as *mut u8,
            0,
            total_f16_elements * 2,
        );
        std::ptr::write_bytes(
            v_cache.contents() as *mut u8,
            0,
            total_f16_elements * 2,
        );
    }

    // Upload and update KV cache
    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    gpu_ctx.kv_update_k_encode(
        encoder, &k_buffer, &k_cache, 0, // pos
        max_seq, n_kv_heads, head_dim,
    );

    gpu_ctx.kv_update_v_encode(
        encoder, &v_buffer, &v_cache, 0, // pos
        max_seq, n_kv_heads, head_dim,
    );

    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Read back KV cache (F16 format)
    let k_cache_f16: Vec<u16> = unsafe {
        std::slice::from_raw_parts(
            k_cache.contents() as *const u16,
            total_f16_elements,
        )
        .to_vec()
    };

    let v_cache_f16: Vec<u16> = unsafe {
        std::slice::from_raw_parts(
            v_cache.contents() as *const u16,
            total_f16_elements,
        )
        .to_vec()
    };

    // Convert F16 to F32
    let k_cache_data: Vec<f32> =
        k_cache_f16.iter().map(|&h| half::f16::from_bits(h).to_f32()).collect();
    let v_cache_data: Vec<f32> =
        v_cache_f16.iter().map(|&h| half::f16::from_bits(h).to_f32()).collect();

    // Check position 0 for all heads
    // CPU data layout: [n_kv_heads, head_dim]
    // GPU cache layout: [n_kv_heads, max_seq, head_dim]
    let mut max_k_diff = 0.0f32;
    let mut max_v_diff = 0.0f32;

    for h in 0..n_kv_heads {
        for d in 0..head_dim {
            let cpu_idx = h * head_dim + d;
            let gpu_idx = h * max_seq * head_dim + d;

            let k_diff = (cpu_k[cpu_idx] - k_cache_data[gpu_idx]).abs();
            let v_diff = (cpu_v[cpu_idx] - v_cache_data[gpu_idx]).abs();
            max_k_diff = max_k_diff.max(k_diff);
            max_v_diff = max_v_diff.max(v_diff);
        }
    }

    // Check that unwritten positions are zero
    // Layout: [n_kv_heads, max_seq, head_dim]
    // For each head, only position 0 is written, positions 1..max_seq-1 should be zero
    let mut found_nonzero_k = false;
    let mut found_nonzero_v = false;

    for h in 0..n_kv_heads {
        for pos in 1..max_seq {
            // Check this position in both K and V caches
            for d in 0..head_dim {
                let idx = h * max_seq * head_dim + pos * head_dim + d;

                if k_cache_data[idx] != 0.0 && !found_nonzero_k {
                    found_nonzero_k = true;
                    println!(
                        "Unexpected K data at h={}, pos={}, d={}, index={}, value={:.4}",
                        h, pos, d, idx, k_cache_data[idx]
                    );
                }
                if v_cache_data[idx] != 0.0 && !found_nonzero_v {
                    found_nonzero_v = true;
                    println!(
                        "Unexpected V data at h={}, pos={}, d={}, index={}, value={:.4}",
                        h, pos, d, idx, v_cache_data[idx]
                    );
                }
            }
        }
    }

    println!("\n=== Results ===");
    println!("GPU K cache (pos 0): max diff={:.6}", max_k_diff);
    println!("GPU V cache (pos 0): max diff={:.6}", max_v_diff);
    println!(
        "Unwritten positions have nonzero: K={}, V={}",
        found_nonzero_k, found_nonzero_v
    );
    println!(
        "K cache nan count: {}",
        k_cache_data.iter().filter(|&&v| v.is_nan()).count()
    );
    println!(
        "V cache nan count: {}",
        v_cache_data.iter().filter(|&&v| v.is_nan()).count()
    );

    // Print head 0, first 10 values
    println!("\nHead 0, first 10 K values:");
    for d in 0..10 {
        let cpu_idx = d;
        let gpu_idx = d;
        println!(
            "  CPU K[{}]={:.4}, GPU K cache[{}]={:.4}",
            cpu_idx, cpu_k[cpu_idx], gpu_idx, k_cache_data[gpu_idx]
        );
    }

    println!("\nHead 0, first 10 V values:");
    for d in 0..10 {
        let cpu_idx = d;
        let gpu_idx = d;
        println!(
            "  CPU V[{}]={:.4}, GPU V cache[{}]={:.4}",
            cpu_idx, cpu_v[cpu_idx], gpu_idx, v_cache_data[gpu_idx]
        );
    }

    // Print head 1's first value to verify it's in the right place
    println!("\nHead 1, first K value:");
    let cpu_idx = head_dim; // Head 1, dimension 0
    let gpu_idx = max_seq * head_dim; // Head 1, pos 0, dimension 0
    println!(
        "  CPU K[{}]={:.4}, GPU K cache[{}]={:.4}",
        cpu_idx, cpu_k[cpu_idx], gpu_idx, k_cache_data[gpu_idx]
    );

    // Final verdict
    println!("\n=== Test Result ===");
    if max_k_diff < 0.001
        && max_v_diff < 0.001
        && !found_nonzero_k
        && !found_nonzero_v
    {
        println!("✓ PASS: KV cache update kernel is correct");
    } else {
        println!("✗ FAIL: KV cache update kernel has errors");
        if max_k_diff >= 0.001 {
            println!("  - K cache max diff {:.6} >= 0.001", max_k_diff);
        }
        if max_v_diff >= 0.001 {
            println!("  - V cache max diff {:.6} >= 0.001", max_v_diff);
        }
        if found_nonzero_k {
            println!("  - Unwritten K positions have nonzero values");
        }
        if found_nonzero_v {
            println!("  - Unwritten V positions have nonzero values");
        }
    }
}

#[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
fn main() {
    println!("This test requires Metal support (aarch64 + metal feature)");
}
