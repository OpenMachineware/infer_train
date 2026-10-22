// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Benchmark Q4_K GEMV kernel with different NSG configurations
//! Compare NSG=2 (original) vs NSG=8/16/32 (MMA-style)
//!
//! Usage: cargo run --release --bin bench-nsg-comparison

use metal::*;
use std::time::Instant;

fn create_q4_k_test_data(
    device: &DeviceRef,
    m: usize,
    k: usize,
) -> (Buffer, Buffer, Buffer) {
    // Create random Q4_K weight matrix
    let nb = k / 256; // Number of blocks per row
    let weights_size = m * nb * 144; // 144 bytes per QK_K block
    let mut weights_data = vec![0u8; weights_size];

    // Fill with random data (simplified - just ensure no NaN)
    for i in 0..weights_data.len() {
        weights_data[i] = (i % 256) as u8;
    }

    // Set reasonable scales (avoid NaN)
    for row in 0..m {
        for block in 0..nb {
            let offset = (row * nb + block) * 144;
            // Set d and dmin to reasonable values
            let d_bytes = 0x3C00u16.to_le_bytes(); // 1.0 in FP16
            weights_data[offset] = d_bytes[0];
            weights_data[offset + 1] = d_bytes[1];
            weights_data[offset + 2] = d_bytes[0];
            weights_data[offset + 3] = d_bytes[1];
        }
    }

    let weights_buffer = device.new_buffer_with_data(
        weights_data.as_ptr() as *const std::ffi::c_void,
        weights_size as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Create random input vector
    let mut input_data = vec![0.0f32; k];
    for i in 0..k {
        input_data[i] = ((i as f32) % 10.0) / 10.0;
    }

    let input_buffer = device.new_buffer_with_data(
        input_data.as_ptr() as *const std::ffi::c_void,
        (k * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Create output buffer
    let output_buffer = device
        .new_buffer((m * 4) as u64, MTLResourceOptions::StorageModeShared);

    (weights_buffer, input_buffer, output_buffer)
}

fn get_adaptive_nsg(num_rows: usize) -> usize {
    if num_rows <= 64 {
        32 // Decode scenario
    } else if num_rows <= 6144 {
        16
    } else {
        8
    }
}

fn main() {
    let device = Device::system_default().expect("No Metal device found");
    let queue = device.new_command_queue();
    let library = device
        .new_library_with_source(
            include_str!("../src/ops/metal/gemv/kquant_mv_mma.metal"),
            &CompileOptions::new(),
        )
        .expect("Failed to compile Metal library");

    // Test dimensions matching decode scenario
    let test_cases = vec![
        ("Q projection", 2048, 1024), // Typical Q projection
        ("K projection", 1024, 1024), // Typical K projection
        ("Gate projection", 3072, 1024), // Typical gate projection
    ];

    println!("Q4_K GEMV NSG Benchmark");
    println!("========================\n");

    for (name, m, k) in test_cases {
        println!("Test: {} (M={}, K={})", name, m, k);
        println!("----------------------------------------");

        let (weights, input, output) = create_q4_k_test_data(&device, m, k);
        let recommended_nsg = get_adaptive_nsg(m);
        println!("Recommended NSG: {}", recommended_nsg);

        // Test original kernel (NSG=2)
        let pipeline_nsg2 =
            library.get_function("kernel_mul_mv_q4_K_f32", None).unwrap();
        let pipeline_nsg2 = device
            .new_compute_pipeline_state_with_function(&pipeline_nsg2)
            .unwrap();

        // Warm up
        for _ in 0..10 {
            let cmd_buf = queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline_nsg2);
            encoder.set_buffer(0, Some(&weights), 0);
            encoder.set_buffer(1, Some(&input), 0);
            encoder.set_buffer(2, Some(&output), 0);

            let args = kquant_mv_mma_args {
                ne00: k as u32,
                ne01: m as u32,
                nb01: ((k / 256) * 144) as u64,
            };
            encoder.set_bytes(
                3,
                std::mem::size_of::<kquant_mv_mma_args>() as u64,
                &args as *const _ as *const std::ffi::c_void,
            );

            let thread_groups =
                MTLSize { width: ((m + 3) / 4) as u64, height: 1, depth: 1 };
            let thread_group_size = MTLSize { width: 64, height: 1, depth: 1 };
            encoder.dispatch_thread_groups(thread_groups, thread_group_size);
            encoder.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();
        }

        // Benchmark NSG=2
        let start = Instant::now();
        let iterations = 100;
        for _ in 0..iterations {
            let cmd_buf = queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline_nsg2);
            encoder.set_buffer(0, Some(&weights), 0);
            encoder.set_buffer(1, Some(&input), 0);
            encoder.set_buffer(2, Some(&output), 0);

            let args = kquant_mv_mma_args {
                ne00: k as u32,
                ne01: m as u32,
                nb01: ((k / 256) * 144) as u64,
            };
            encoder.set_bytes(
                3,
                std::mem::size_of::<kquant_mv_mma_args>() as u64,
                &args as *const _ as *const std::ffi::c_void,
            );

            let thread_groups =
                MTLSize { width: ((m + 3) / 4) as u64, height: 1, depth: 1 };
            let thread_group_size = MTLSize { width: 64, height: 1, depth: 1 };
            encoder.dispatch_thread_groups(thread_groups, thread_group_size);
            encoder.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();
        }
        let time_nsg2 = start.elapsed().as_micros() as f64 / iterations as f64;

        // Test MMA kernels for each NSG
        for nsg in [8, 16, 32] {
            let kernel_name = format!("kernel_mul_mv_q4_K_f32_mma_nsg{}", nsg);
            let pipeline = match library.get_function(&kernel_name, None) {
                Ok(f) => {
                    device.new_compute_pipeline_state_with_function(&f).unwrap()
                }
                Err(_) => {
                    println!("  NSG={}: Kernel not found, skipping", nsg);
                    continue;
                }
            };

            // Warm up
            for _ in 0..10 {
                let cmd_buf = queue.new_command_buffer();
                let encoder = cmd_buf.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&pipeline);
                encoder.set_buffer(0, Some(&weights), 0);
                encoder.set_buffer(1, Some(&input), 0);
                encoder.set_buffer(2, Some(&output), 0);

                let args = kquant_mv_mma_args {
                    ne00: k as u32,
                    ne01: m as u32,
                    nb01: ((k / 256) * 144) as u64,
                };
                encoder.set_bytes(
                    3,
                    std::mem::size_of::<kquant_mv_mma_args>() as u64,
                    &args as *const _ as *const std::ffi::c_void,
                );

                let thread_groups = MTLSize {
                    width: ((m + nsg - 1) / nsg) as u64,
                    height: 1,
                    depth: 1,
                };
                let thread_group_size =
                    MTLSize { width: (nsg * 32) as u64, height: 1, depth: 1 };
                encoder
                    .dispatch_thread_groups(thread_groups, thread_group_size);
                encoder.end_encoding();
                cmd_buf.commit();
                cmd_buf.wait_until_completed();
            }

            // Benchmark
            let start = Instant::now();
            for _ in 0..iterations {
                let cmd_buf = queue.new_command_buffer();
                let encoder = cmd_buf.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&pipeline);
                encoder.set_buffer(0, Some(&weights), 0);
                encoder.set_buffer(1, Some(&input), 0);
                encoder.set_buffer(2, Some(&output), 0);

                let args = kquant_mv_mma_args {
                    ne00: k as u32,
                    ne01: m as u32,
                    nb01: ((k / 256) * 144) as u64,
                };
                encoder.set_bytes(
                    3,
                    std::mem::size_of::<kquant_mv_mma_args>() as u64,
                    &args as *const _ as *const std::ffi::c_void,
                );

                let thread_groups = MTLSize {
                    width: ((m + nsg - 1) / nsg) as u64,
                    height: 1,
                    depth: 1,
                };
                let thread_group_size =
                    MTLSize { width: (nsg * 32) as u64, height: 1, depth: 1 };
                encoder
                    .dispatch_thread_groups(thread_groups, thread_group_size);
                encoder.end_encoding();
                cmd_buf.commit();
                cmd_buf.wait_until_completed();
            }
            let time_nsg =
                start.elapsed().as_micros() as f64 / iterations as f64;

            let speedup = time_nsg2 / time_nsg;
            let data_gb = (m as f64 * k as f64 * 0.5625
                + m as f64 * 4.0
                + k as f64 * 4.0)
                / 1e9; // Q4_K = 0.5625 bytes per element
            let bandwidth = data_gb / (time_nsg / 1e6);
            let theoretical_bw = 100.0; // GB/s (M3 Max)
            let utilization = bandwidth / theoretical_bw * 100.0;

            println!(
                "  NSG={:2}: {:7.2} μs, {:5.2}x speedup, {:5.2} GB/s, {:5.2} pct util",
                nsg, time_nsg, speedup, bandwidth, utilization
            );
        }

        println!();
    }
}

#[repr(C)]
struct kquant_mv_mma_args {
    ne00: u32,
    ne01: u32,
    nb01: u64,
}
