// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Benchmark Q4_K GEMV kernel with different K dimensions
//! Test if NSG helps when K is large enough

use metal::*;
use std::time::Instant;

fn create_q4_k_test_data(
    device: &DeviceRef,
    m: usize,
    k: usize,
) -> (Buffer, Buffer, Buffer) {
    let nb = k / 256;
    let weights_size = m * nb * 144;
    let mut weights_data = vec![0u8; weights_size];

    for row in 0..m {
        for block in 0..nb {
            let offset = (row * nb + block) * 144;
            let d_bytes = 0x3C00u16.to_le_bytes();
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

    let mut input_data = vec![1.0f32; k];
    let input_buffer = device.new_buffer_with_data(
        input_data.as_ptr() as *const std::ffi::c_void,
        (k * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let output_buffer = device
        .new_buffer((m * 4) as u64, MTLResourceOptions::StorageModeShared);

    (weights_buffer, input_buffer, output_buffer)
}

fn main() {
    let device = Device::system_default().expect("No Metal device found");

    let ggml_common = include_str!("../src/ops/metal/include/blocks.h");
    let ggml_common_metal =
        format!("#define GGML_COMMON_DECL_METAL\n{}", ggml_common);

    let kernel_source =
        include_str!("../src/ops/metal/gemv/kquant_mv_mma.metal");
    let kernel_source =
        kernel_source.replace("#include \"../include/blocks.h\"", "");

    let combined = format!("{}\n{}", ggml_common_metal, kernel_source);

    let library = device
        .new_library_with_source(&combined, &CompileOptions::new())
        .expect("Failed to compile Metal library");

    // Test different K dimensions
    let test_cases = vec![
        ("K=256  (1 block)", 1, 256),
        ("K=1024 (4 blocks)", 1, 1024),
        ("K=4096 (16 blocks)", 1, 4096),
        ("K=8192 (32 blocks)", 1, 8192),
    ];

    println!("Q4_K GEMV K-Dimension Test (M=1)");
    println!("==================================\n");

    for (name, m, k) in test_cases {
        let nb = k / 256;
        println!("Test: {} (M={}, K={}, nb={})", name, m, k, nb);
        println!("----------------------------------------");

        let (weights, input, output) = create_q4_k_test_data(&device, m, k);

        for nsg in [8, 16, 32] {
            let kernel_name = format!("kernel_mul_mv_q4_K_f32_mma_nsg{}", nsg);
            let function = library.get_function(&kernel_name, None).unwrap();
            let pipeline = device
                .new_compute_pipeline_state_with_function(&function)
                .unwrap();

            let queue = device.new_command_queue();

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
                    nb01: (nb * 144) as u64,
                };
                encoder.set_bytes(
                    3,
                    std::mem::size_of::<kquant_mv_mma_args>() as u64,
                    &args as *const _ as *const std::ffi::c_void,
                );

                let thread_groups =
                    MTLSize { width: m as u64, height: 1, depth: 1 };
                let thread_group_size =
                    MTLSize { width: (nsg * 32) as u64, height: 1, depth: 1 };
                encoder
                    .dispatch_thread_groups(thread_groups, thread_group_size);
                encoder.end_encoding();
                cmd_buf.commit();
                cmd_buf.wait_until_completed();
            }

            // Benchmark
            let iterations = 100;
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
                    nb01: (nb * 144) as u64,
                };
                encoder.set_bytes(
                    3,
                    std::mem::size_of::<kquant_mv_mma_args>() as u64,
                    &args as *const _ as *const std::ffi::c_void,
                );

                let thread_groups =
                    MTLSize { width: m as u64, height: 1, depth: 1 };
                let thread_group_size =
                    MTLSize { width: (nsg * 32) as u64, height: 1, depth: 1 };
                encoder
                    .dispatch_thread_groups(thread_groups, thread_group_size);
                encoder.end_encoding();
                cmd_buf.commit();
                cmd_buf.wait_until_completed();
            }
            let time_us =
                start.elapsed().as_micros() as f64 / iterations as f64;

            let data_gb = (m as f64 * k as f64 * 0.5625
                + m as f64 * 4.0
                + k as f64 * 4.0)
                / 1e9;
            let bandwidth = data_gb / (time_us / 1e6);
            let utilization = bandwidth / 100.0 * 100.0;

            let active_sgs = std::cmp::min(nb, nsg as usize);
            let blocks_per_sg = (nb + nsg as usize - 1) / nsg as usize;

            println!(
                "  NSG={:2}: {:7.2} μs, {:5.2} GB/s, {:5.2} pct util, {} active SGs, {} blocks/SG",
                nsg, time_us, bandwidth, utilization, active_sgs, blocks_per_sg
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
